#!/usr/bin/env python3
"""
Karan ML Tool for MCP Server
=============================

MCP tool that integrates Karan ML service for advanced stock predictions.
Provides access to 4 ML models (RandomForest, LightGBM, XGBoost, DQN) with
real-time Fyers data and historical Yahoo Finance data.

Option C: MCP Integration
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests

# Import MCP server components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mcp_server.mcp_trading_server import MCPToolResult, MCPToolStatus

logger = logging.getLogger(__name__)

@dataclass
class KaranPrediction:
    """Karan ML prediction result"""
    symbol: str
    action: str  # LONG, SHORT, HOLD
    confidence: float
    predicted_return: float
    current_price: float
    predicted_price: float
    horizon: str
    model_predictions: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    timestamp: str = None


class KaranMLTool:
    """
    Karan ML Tool for MCP Server
    
    Features:
    - Advanced ML predictions using 4 models (RF, LGB, XGB, DQN)
    - Real-time Fyers data integration
    - Historical Yahoo Finance data
    - 50+ technical indicators
    - Multi-horizon analysis (intraday, short, long)
    - Risk-aware position sizing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "karan_ml_tool")
        
        # Karan ML service configuration
        self.karan_url = config.get("karan_url", os.getenv("KARAN_ML_URL", "http://localhost:5000"))
        self.timeout = config.get("timeout", 120)
        self.scan_timeout = config.get("scan_timeout", 240)
        self.train_timeout = config.get("train_timeout", 360)
        
        # Feature flags
        self.auto_train = config.get("auto_train", True)
        self.cache_predictions = config.get("cache_predictions", True)
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = config.get("cache_ttl", 30)  # seconds
        
        # Health check
        self._check_service_health()
        
        logger.info(f"Karan ML Tool {self.tool_id} initialized with service at {self.karan_url}")
    
    def _check_service_health(self):
        """Check if Karan ML service is available"""
        try:
            response = requests.get(f"{self.karan_url}/tools/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Karan ML service is healthy: {health_data.get('status')}")
                return True
            else:
                logger.warning(f"Karan ML service returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Karan ML service health check failed: {e}")
            return False
    
    async def predict(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Generate ML predictions using Karan service
        
        Args:
            arguments: {
                "symbols": ["RELIANCE.NS", "TCS.NS", ...],
                "horizon": "intraday" | "short" | "long",
                "risk_profile": "low" | "moderate" | "high" (optional),
                "stop_loss_pct": float (optional),
                "capital_risk_pct": float (optional)
            }
        
        Returns:
            MCPToolResult with predictions
        """
        try:
            symbols = arguments.get("symbols", [])
            horizon = arguments.get("horizon", "intraday")
            risk_profile = arguments.get("risk_profile")
            stop_loss_pct = arguments.get("stop_loss_pct")
            capital_risk_pct = arguments.get("capital_risk_pct")
            
            if not symbols:
                return MCPToolResult(
                    status=MCPToolStatus.VALIDATION_ERROR,
                    error="No symbols provided"
                )
            
            # Check cache
            cache_key = f"{hash(str(symbols))}_{horizon}_{risk_profile}"
            if self.cache_predictions and cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                cache_age = (datetime.now() - datetime.fromisoformat(cached_result["timestamp"])).seconds
                if cache_age < self.cache_ttl:
                    logger.info(f"Returning cached predictions (age: {cache_age}s)")
                    return MCPToolResult(
                        status=MCPToolStatus.SUCCESS,
                        data=cached_result,
                        reasoning=f"Cached predictions (age: {cache_age}s)",
                        confidence=0.9
                    )
            
            # Call Karan ML service
            payload = {
                "symbols": symbols,
                "horizon": horizon
            }
            if risk_profile:
                payload["risk_profile"] = risk_profile
            if stop_loss_pct is not None:
                payload["stop_loss_pct"] = stop_loss_pct
            if capital_risk_pct is not None:
                payload["capital_risk_pct"] = capital_risk_pct
            
            logger.info(f"Calling Karan ML predict: {symbols}, horizon={horizon}")
            response = requests.post(
                f"{self.karan_url}/tools/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            predictions = result.get("predictions", [])
            
            # Convert to Karan prediction format
            karan_predictions = []
            for pred in predictions:
                if "error" not in pred:
                    karan_predictions.append(KaranPrediction(
                        symbol=pred.get("symbol"),
                        action=pred.get("action"),
                        confidence=pred.get("confidence", 0),
                        predicted_return=pred.get("predicted_return", 0),
                        current_price=pred.get("current_price", 0),
                        predicted_price=pred.get("predicted_price", 0),
                        horizon=pred.get("horizon"),
                        model_predictions=pred.get("model_predictions"),
                        risk_metrics=pred.get("risk_metrics"),
                        timestamp=datetime.now().isoformat()
                    ))
            
            # Prepare response
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "horizon": horizon,
                "total_predictions": len(karan_predictions),
                "predictions": [asdict(pred) for pred in karan_predictions],
                "metadata": result.get("metadata", {})
            }
            
            # Cache result
            if self.cache_predictions:
                self.prediction_cache[cache_key] = response_data
            
            # Calculate average confidence
            avg_confidence = sum(p.confidence for p in karan_predictions) / len(karan_predictions) if karan_predictions else 0
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Generated {len(karan_predictions)} predictions using Karan ML (4 models: RF+LGB+XGB+DQN)",
                confidence=avg_confidence
            )
            
        except requests.exceptions.Timeout:
            logger.error(f"Karan ML service timeout after {self.timeout}s")
            return MCPToolResult(
                status=MCPToolStatus.TIMEOUT,
                error=f"ML service timeout - predictions may take longer for new symbols (auto-training enabled)"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Karan ML service error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=f"ML service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in Karan ML predict: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def scan_all(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Scan multiple symbols and return ranked shortlist
        
        Args:
            arguments: {
                "symbols": ["RELIANCE.NS", "TCS.NS", ...],
                "horizon": "intraday" | "short" | "long",
                "min_confidence": float (0.0-1.0)
            }
        
        Returns:
            MCPToolResult with ranked shortlist
        """
        try:
            symbols = arguments.get("symbols", [])
            horizon = arguments.get("horizon", "intraday")
            min_confidence = arguments.get("min_confidence", 0.5)
            
            if not symbols:
                return MCPToolResult(
                    status=MCPToolStatus.VALIDATION_ERROR,
                    error="No symbols provided"
                )
            
            # Call Karan ML service
            payload = {
                "symbols": symbols,
                "horizon": horizon,
                "min_confidence": min_confidence
            }
            
            logger.info(f"Calling Karan ML scan_all: {len(symbols)} symbols, horizon={horizon}")
            response = requests.post(
                f"{self.karan_url}/tools/scan_all",
                json=payload,
                timeout=self.scan_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            shortlist = result.get("shortlist", [])
            
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "horizon": horizon,
                "min_confidence": min_confidence,
                "shortlist_count": len(shortlist),
                "shortlist": shortlist,
                "metadata": result.get("metadata", {})
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Scanned {len(symbols)} symbols, {len(shortlist)} passed confidence threshold",
                confidence=0.85
            )
            
        except requests.exceptions.Timeout:
            logger.error(f"Karan ML scan timeout after {self.scan_timeout}s")
            return MCPToolResult(
                status=MCPToolStatus.TIMEOUT,
                error=f"ML scan timeout - try with fewer symbols"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Karan ML scan error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=f"ML service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in Karan ML scan: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def analyze(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Deep analysis of single symbol across multiple horizons
        
        Args:
            arguments: {
                "symbol": "RELIANCE.NS",
                "horizons": ["intraday", "short", "long"],
                "stop_loss_pct": float,
                "capital_risk_pct": float
            }
        
        Returns:
            MCPToolResult with multi-horizon analysis
        """
        try:
            symbol = arguments.get("symbol")
            horizons = arguments.get("horizons", ["intraday", "short", "long"])
            stop_loss_pct = arguments.get("stop_loss_pct", 2.0)
            capital_risk_pct = arguments.get("capital_risk_pct", 1.0)
            drawdown_limit_pct = arguments.get("drawdown_limit_pct", 5.0)
            
            if not symbol:
                return MCPToolResult(
                    status=MCPToolStatus.VALIDATION_ERROR,
                    error="No symbol provided"
                )
            
            # Call Karan ML service
            payload = {
                "symbol": symbol,
                "horizons": horizons,
                "stop_loss_pct": stop_loss_pct,
                "capital_risk_pct": capital_risk_pct,
                "drawdown_limit_pct": drawdown_limit_pct
            }
            
            logger.info(f"Calling Karan ML analyze: {symbol}, horizons={horizons}")
            response = requests.post(
                f"{self.karan_url}/tools/analyze",
                json=payload,
                timeout=self.timeout * 1.5
            )
            response.raise_for_status()
            
            result = response.json()
            
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "horizons": horizons,
                "predictions": result.get("predictions", []),
                "metadata": result.get("metadata", {})
            }
            
            consensus = result.get("metadata", {}).get("consensus", "Unknown")
            avg_confidence = result.get("metadata", {}).get("average_confidence", 0)
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Multi-horizon analysis for {symbol}: {consensus}",
                confidence=avg_confidence
            )
            
        except requests.exceptions.Timeout:
            logger.error(f"Karan ML analyze timeout")
            return MCPToolResult(
                status=MCPToolStatus.TIMEOUT,
                error=f"ML analyze timeout - try with fewer horizons"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Karan ML analyze error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=f"ML service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in Karan ML analyze: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def train_models(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Train or retrain ML models for a symbol
        
        Args:
            arguments: {
                "symbol": "RELIANCE.NS",
                "horizon": "intraday" | "short" | "long",
                "n_episodes": int (10-100),
                "force_retrain": bool
            }
        
        Returns:
            MCPToolResult with training status
        """
        try:
            symbol = arguments.get("symbol")
            horizon = arguments.get("horizon", "intraday")
            n_episodes = arguments.get("n_episodes", 10)
            force_retrain = arguments.get("force_retrain", False)
            
            if not symbol:
                return MCPToolResult(
                    status=MCPToolStatus.VALIDATION_ERROR,
                    error="No symbol provided"
                )
            
            # Call Karan ML service
            payload = {
                "symbol": symbol,
                "horizon": horizon,
                "n_episodes": n_episodes,
                "force_retrain": force_retrain
            }
            
            logger.info(f"Calling Karan ML train: {symbol}, horizon={horizon}, episodes={n_episodes}")
            response = requests.post(
                f"{self.karan_url}/tools/train_rl",
                json=payload,
                timeout=self.train_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=result,
                reasoning=f"Training completed for {symbol} ({horizon})",
                confidence=0.9
            )
            
        except requests.exceptions.Timeout:
            logger.error(f"Karan ML train timeout after {self.train_timeout}s")
            return MCPToolResult(
                status=MCPToolStatus.TIMEOUT,
                error=f"ML training timeout - this is a long operation (check logs)"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Karan ML train error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=f"ML service unavailable: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in Karan ML train: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get Karan ML tool status"""
        service_healthy = self._check_service_health()
        
        return {
            "tool_id": self.tool_id,
            "karan_url": self.karan_url,
            "service_healthy": service_healthy,
            "auto_train": self.auto_train,
            "cache_enabled": self.cache_predictions,
            "cache_size": len(self.prediction_cache),
            "timeouts": {
                "predict": self.timeout,
                "scan": self.scan_timeout,
                "train": self.train_timeout
            },
            "status": "active" if service_healthy else "degraded"
        }

