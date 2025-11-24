#!/usr/bin/env python3
"""
MCP Analyze Tool
===============

LLaMA reasoning tool for the Model Context Protocol server
with standardized JSON responses and advanced market analysis.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult
# Import stock analysis components
from utils.ml_components.stock_analysis_complete import (
    EnhancedDataIngester,
    FeatureEngineer,
    predict_stock_price,
    train_ml_models,
    DATA_CACHE_DIR,
    FEATURE_CACHE_DIR,
    MODEL_DIR,
    LOGS_DIR
)
import json
import time
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Request/Response logging
MCP_LOG_DIR = LOGS_DIR / "mcp_requests"
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class AnalyzeTool:
    """
    MCP Analyze Tool
    Provides deep analysis of stock symbols using Groq API reasoning
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "analyze_tool")
        self.groq_enabled = config.get("groq_enabled", False)
        self.groq_api_key = config.get("groq_api_key", "")
        self.groq_base_url = config.get(
            "groq_base_url", "https://api.groq.com/openai/v1")
        self.groq_model = config.get("groq_model", "llama-3.1-8b-instant")
        self.langgraph_enabled = config.get("langgraph_enabled", False)

        # Initialize stock analysis components
        self.ingester = EnhancedDataIngester()
        self.engineer = FeatureEngineer()
        self.request_counter = 0

        logger.info(f"Analyze Tool {self.tool_id} initialized")

    def _log_request(self, tool_name: str, request_data: Dict) -> str:
        """Log incoming request"""
        self.request_counter += 1
        request_id = f"{tool_name}_{int(time.time())}_{self.request_counter}"

        log_entry = {
            "request_id": request_id,
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "request": request_data
        }

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_requests.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(f"MCP Request [{request_id}]: {tool_name}")
        return request_id

    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types and other non-JSON-serializable types to Python native types"""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return obj

    def _log_response(self, request_id: str, response_data: Dict, duration_ms: float):
        """Log outgoing response"""
        # Convert numpy types to Python native types for JSON serialization
        sanitized_response = self._convert_to_json_serializable(response_data)

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "response": sanitized_response
        }

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_responses.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except TypeError as e:
            logger.error(f"Failed to log response: {e}")
            logger.error(f"Problematic data: {log_entry}")
            # Try to log with default=str as fallback
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry, default=str) + '\n')
            except Exception as e2:
                logger.error(
                    f"Failed to log response even with default=str: {e2}")

        logger.info(f"MCP Response [{request_id}]: {duration_ms:.2f}ms")

    def _analyze_single_symbol(self, symbol: str, horizons: List[str], request_id: str) -> Dict[str, Any]:
        """Analyze a single symbol across multiple horizons"""
        try:
            logger.info(
                f"[{request_id}] Analyzing {symbol} across {len(horizons)} horizons...")

            predictions = []

            # First ensure data and features exist (only once, not per horizon)
            json_path = DATA_CACHE_DIR / f"{symbol}_all_data.json"

            if not json_path.exists():
                logger.info(
                    f"[{request_id}] Data not found for {symbol}. Fetching...")
                try:
                    self.ingester.fetch_all_data(symbol, period="2y")
                    logger.info(f"[{request_id}] Data fetched for {symbol}")
                except Exception as e:
                    logger.error(f"[{request_id}] Failed to fetch data: {e}")
                    return {
                        "symbol": symbol,
                        "error": f"Data fetch failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }

            # Ensure features are calculated
            features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
            if not features_path.exists():
                logger.info(
                    f"[{request_id}] Features not found. Calculating...")
                all_data = self.ingester.load_all_data(symbol)
                if all_data:
                    df = all_data.get('price_history')
                    if df is not None and not df.empty:
                        features_df = self.engineer.calculate_all_features(
                            df, symbol)
                        self.engineer.save_features(features_df, symbol)
                        logger.info(
                            f"[{request_id}] Features calculated for {symbol}")

            # Now process each horizon
            for horizon in horizons:
                try:
                    # Check if models exist for this horizon
                    model_files = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*"))

                    if not model_files:
                        logger.info(
                            f"[{request_id}] Models not found for {symbol} ({horizon}). Training...")
                        try:
                            training_result = train_ml_models(
                                symbol, horizon, verbose=True)

                            # Handle both dict and bool return formats
                            success = training_result.get('success', False) if isinstance(
                                training_result, dict) else training_result

                            if not success:
                                logger.error(
                                    f"[{request_id}] Training failed for {horizon}")
                                predictions.append({
                                    "symbol": symbol,
                                    "horizon": horizon,
                                    "error": "Model training failed"
                                })
                                continue
                            logger.info(
                                f"[{request_id}] Models trained for {horizon}")
                        except Exception as e:
                            logger.error(
                                f"[{request_id}] Training error for {horizon}: {e}", exc_info=True)
                            predictions.append({
                                "symbol": symbol,
                                "horizon": horizon,
                                "error": f"Training failed: {str(e)}"
                            })
                            continue

                    # Generate prediction
                    prediction = predict_stock_price(
                        symbol, horizon=horizon, verbose=True)

                    if prediction:
                        predictions.append(prediction)
                        logger.info(f"[{request_id}] [OK] {horizon}: {prediction['action']} "
                                    f"(conf: {prediction['confidence']:.4f})")
                    else:
                        logger.warning(
                            f"[{request_id}] [FAIL] {horizon}: No prediction")
                        predictions.append({
                            "symbol": symbol,
                            "horizon": horizon,
                            "error": "Prediction failed"
                        })

                except Exception as e:
                    logger.error(f"[{request_id}] Error on {horizon}: {e}")
                    predictions.append({
                        "symbol": symbol,
                        "horizon": horizon,
                        "error": str(e)
                    })

            # Calculate consensus across horizons
            actions = [p.get('action') for p in predictions if 'action' in p]
            avg_confidence = sum(p.get('confidence', 0)
                                 for p in predictions if 'confidence' in p) / len(predictions) if predictions else 0

            consensus = None
            if len(set(actions)) == 1:
                consensus = f"Strong {actions[0]} - All horizons agree"
            elif actions.count('LONG') > actions.count('SHORT'):
                consensus = "Bullish - Majority LONG signals"
            elif actions.count('SHORT') > actions.count('LONG'):
                consensus = "Bearish - Majority SHORT signals"
            else:
                consensus = "Mixed signals - Exercise caution"

            result = {
                "metadata": {
                    "symbol": symbol,
                    "horizons": horizons,
                    "count": len(predictions),
                    "average_confidence": round(avg_confidence, 4),
                    "consensus": consensus,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "predictions": predictions
            }

            return result

        except Exception as e:
            logger.error(f"[{request_id}] Analysis error: {e}", exc_info=True)
            return {
                "metadata": {
                    "symbol": symbol,
                    "error": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                "predictions": []
            }

    async def analyze(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Analyze stock symbols with deep reasoning

        Args:
            arguments: Tool arguments containing predictions and analysis parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with analysis
        """
        start_time = time.time()

        try:
            # Extract parameters
            predictions = arguments.get("predictions", [])
            analysis_depth = arguments.get("analysis_depth", "detailed")
            include_risk_assessment = arguments.get(
                "include_risk_assessment", True)
            horizons = arguments.get("horizons", ["intraday", "short", "long"])

            if not predictions:
                return MCPToolResult(
                    status="ERROR",
                    error="No predictions provided for analysis",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Extract symbols from predictions
            symbols = list(set(pred.get("symbol")
                           for pred in predictions if "symbol" in pred))

            if not symbols:
                return MCPToolResult(
                    status="ERROR",
                    error="No valid symbols found in predictions",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            request_data = {
                "symbols": symbols,
                "horizons": horizons,
                "analysis_depth": analysis_depth
            }
            request_id = self._log_request("analyze", request_data)

            # Analyze each symbol
            analysis_results = []
            total_confidence = 0.0

            for symbol in symbols:
                try:
                    result = self._analyze_single_symbol(
                        symbol, horizons, request_id)
                    analysis_results.append(result)

                    # Calculate confidence from predictions
                    symbol_predictions = result.get("predictions", [])
                    if symbol_predictions:
                        valid_predictions = [
                            p for p in symbol_predictions if "confidence" in p]
                        if valid_predictions:
                            symbol_confidence = sum(
                                p["confidence"] for p in valid_predictions) / len(valid_predictions)
                            total_confidence += symbol_confidence

                except Exception as e:
                    logger.warning(f"Analysis failed for {symbol}: {e}")
                    analysis_results.append({
                        "symbol": symbol,
                        "error": str(e)
                    })

            # Calculate average confidence
            confidence = total_confidence / len(symbols) if symbols else 0.0

            # Convert numpy types to Python native types for JSON serialization
            sanitized_analysis_results = self._convert_to_json_serializable(
                analysis_results)
            sanitized_confidence = self._convert_to_json_serializable(
                confidence)

            duration_ms = (time.time() - start_time) * 1000
            self._log_response(
                request_id, {"analysis_results": sanitized_analysis_results}, duration_ms)

            execution_time = time.time() - start_time

            return MCPToolResult(
                status="SUCCESS",
                data={
                    "analysis_results": sanitized_analysis_results,
                    "symbols_analyzed": len(symbols),
                    "average_confidence": sanitized_confidence,
                    "analysis_depth": analysis_depth
                },
                confidence=sanitized_confidence,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "symbols_count": len(symbols),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error in analyze tool: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )


# Tool availability flag
ANALYZE_TOOL_AVAILABLE = True
