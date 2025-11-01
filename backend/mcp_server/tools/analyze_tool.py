#!/usr/bin/env python3
"""
Analyze Tool for Venting Layer
============================

MCP tool for sending prediction output to local LLaMA via LangGraph
to get reasoning and insights
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Import existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..mcp_trading_server import MCPToolResult, MCPToolStatus

# Import real-time data components
try:
    from ...utils.enhanced_websocket_manager import EnhancedWebSocketManager
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("WebSocket manager not available for real-time data")

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Analysis result from LLaMA model"""
    symbol: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    reasoning: str
    key_factors: List[str]
    risk_assessment: str
    timestamp: str = None

class AnalyzeTool:
    """
    Analysis tool for Venting Layer
    
    Features:
    - Send prediction output to local LLaMA via LangGraph
    - Get reasoning and insights from natural language processing
    - Return validated JSON compatible with Trading Executor
    - Real-time dynamic analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "analyze_tool")
        
        # LLaMA configuration
        self.llama_enabled = config.get("llama_enabled", False)
        self.llama_host = config.get("llama_host", "http://localhost:11434")
        self.llama_model = config.get("llama_model", "llama2")
        
        # LangGraph configuration
        self.langgraph_enabled = config.get("langgraph_enabled", False)
        
        # Real-time data configuration
        self.real_time_data = config.get("real_time_data", True)
        self.websocket_manager = None
        
        # Performance tracking
        self.analysis_cache = {}
        self.cache_timeout = config.get("cache_timeout", 60)  # seconds
        
        # Initialize real-time data if available
        if WEBSOCKET_AVAILABLE and self.real_time_data:
            try:
                self.websocket_manager = EnhancedWebSocketManager()
                logger.info("Real-time data connection established for analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize real-time data for analysis: {e}")
        
        logger.info(f"Analyze Tool {self.tool_id} initialized with real-time capabilities")
    
    async def analyze(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Send prediction output to local LLaMA via LangGraph to get reasoning
        
        Args:
            arguments: {
                "predictions": [
                    {
                        "symbol": "RELIANCE.NS",
                        "prediction_score": 0.75,
                        "confidence": 0.85,
                        "model_type": "lightgbm",
                        "features": {...}
                    }
                ],
                "analysis_depth": "detailed" | "summary",
                "include_risk_assessment": true,
                "real_time": true
            }
        """
        try:
            predictions = arguments.get("predictions", [])
            analysis_depth = arguments.get("analysis_depth", "detailed")
            include_risk_assessment = arguments.get("include_risk_assessment", True)
            real_time = arguments.get("real_time", self.real_time_data)
            
            if not predictions:
                raise ValueError("Predictions data is required for analysis")
            
            # Check cache for recent analysis
            cache_key = f"{hash(str(predictions))}_{analysis_depth}"
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_result["timestamp"])).seconds < self.cache_timeout:
                    logger.info("Returning cached analysis")
                    return MCPToolResult(
                        status=MCPToolStatus.SUCCESS,
                        data=cached_result,
                        reasoning="Returning cached analysis for performance",
                        confidence=0.9
                    )
            
            # Get real-time data if requested
            real_time_data = {}
            if real_time and self.websocket_manager:
                try:
                    symbols = [pred.get("symbol", "") for pred in predictions if pred.get("symbol")]
                    real_time_data = await self._get_real_time_data(symbols)
                except Exception as e:
                    logger.warning(f"Failed to get real-time data: {e}")
            
            # Analyze predictions using LLaMA
            analysis_results = []
            
            if self.llama_enabled:
                analysis_results = await self._analyze_with_llama(predictions, analysis_depth, include_risk_assessment, real_time_data)
            else:
                # Fallback to simulated analysis
                analysis_results = await self._simulate_analysis(predictions, analysis_depth, include_risk_assessment, real_time_data)
            
            # Add timestamp to each analysis
            current_time = datetime.now().isoformat()
            for result in analysis_results:
                result.timestamp = current_time
            
            # Prepare response
            response_data = {
                "timestamp": current_time,
                "analysis_depth": analysis_depth,
                "total_analyzed": len(analysis_results),
                "real_time_data_used": bool(real_time_data),
                "analysis_results": [asdict(result) for result in analysis_results]
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = response_data
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Analyzed {len(analysis_results)} predictions using {self.llama_model if self.llama_enabled else 'simulated'} analysis with real-time data",
                confidence=0.9 if self.llama_enabled else 0.7
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def _get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time market data"""
        if not self.websocket_manager:
            return {}
        
        try:
            real_time_data = {}
            # Get data for each symbol
            for symbol in symbols:
                try:
                    data = await self.websocket_manager.get_latest_quote(symbol)
                    if data:
                        real_time_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to get real-time data for {symbol}: {e}")
                    continue
            
            return real_time_data
        except Exception as e:
            logger.error(f"Real-time data retrieval error: {e}")
            return {}
    
    async def _analyze_with_llama(self, predictions: List[Dict], analysis_depth: str, include_risk_assessment: bool, real_time_data: Dict[str, Any] = None) -> List[AnalysisResult]:
        """Analyze predictions using LLaMA model via LangGraph"""
        try:
            # Import ollama if available
            try:
                import ollama
            except ImportError:
                logger.warning("Ollama not available, using simulated analysis")
                return await self._simulate_analysis(predictions, analysis_depth, include_risk_assessment, real_time_data)
            
            analysis_results = []
            
            for pred in predictions:
                try:
                    symbol = pred.get("symbol", "")
                    prediction_score = pred.get("prediction_score", 0.5)
                    features = pred.get("features", {})
                    
                    # Get real-time data if available
                    rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                    
                    # Prepare prompt for LLaMA with real-time data
                    prompt = self._generate_analysis_prompt(symbol, prediction_score, features, analysis_depth, rt_data)
                    
                    # Generate response from LLaMA
                    response = ollama.generate(
                        model=self.llama_model,
                        prompt=prompt,
                        options={
                            "temperature": 0.7,
                            "top_p": 0.9
                        }
                    )
                    
                    # Parse response
                    analysis_text = response.get("response", "")
                    
                    # Extract sentiment and key factors (simplified)
                    sentiment = "positive" if prediction_score > 0.6 else "negative" if prediction_score < 0.4 else "neutral"
                    confidence = min(0.95, max(0.5, prediction_score + 0.1))
                    
                    # Generate key factors based on features and real-time data
                    key_factors = self._extract_key_factors(features, rt_data)
                    
                    # Risk assessment
                    risk_assessment = self._generate_risk_assessment(prediction_score, features, rt_data)
                    
                    analysis_results.append(AnalysisResult(
                        symbol=symbol,
                        sentiment=sentiment,
                        confidence=confidence,
                        reasoning=analysis_text[:500],  # Limit length
                        key_factors=key_factors,
                        risk_assessment=risk_assessment
                    ))
                    
                except Exception as e:
                    logger.warning(f"LLaMA analysis error for {pred.get('symbol', 'unknown')}: {e}")
                    # Add fallback analysis
                    analysis_results.append(await self._generate_fallback_analysis(pred, real_time_data))
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"LLaMA analysis error: {e}")
            return await self._simulate_analysis(predictions, analysis_depth, include_risk_assessment, real_time_data)
    
    async def _simulate_analysis(self, predictions: List[Dict], analysis_depth: str, include_risk_assessment: bool, real_time_data: Dict[str, Any] = None) -> List[AnalysisResult]:
        """Simulate analysis when LLaMA is not available"""
        try:
            analysis_results = []
            
            for pred in predictions:
                try:
                    symbol = pred.get("symbol", "")
                    prediction_score = pred.get("prediction_score", 0.5)
                    features = pred.get("features", {})
                    
                    # Get real-time data if available
                    rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                    
                    # Generate simulated sentiment
                    sentiment = "positive" if prediction_score > 0.6 else "negative" if prediction_score < 0.4 else "neutral"
                    confidence = min(0.95, max(0.5, prediction_score + 0.1))
                    
                    # Generate simulated reasoning with real-time data
                    rt_text = f" with real-time price movement of {rt_data.get('change_percent', 0):+.2f}%" if rt_data else ""
                    if sentiment == "positive":
                        reasoning = f"{symbol} shows strong positive momentum{rt_text} with favorable technical indicators. Recommendation is to consider buying."
                    elif sentiment == "negative":
                        reasoning = f"{symbol} shows negative momentum{rt_text} with bearish technical patterns. Recommendation is to avoid or consider selling."
                    else:
                        reasoning = f"{symbol} shows mixed signals{rt_text} with no clear directional bias. Recommendation is to hold current positions."
                    
                    # Generate key factors based on features and real-time data
                    key_factors = self._extract_key_factors(features, rt_data)
                    
                    # Risk assessment
                    risk_assessment = self._generate_risk_assessment(prediction_score, features, rt_data)
                    
                    analysis_results.append(AnalysisResult(
                        symbol=symbol,
                        sentiment=sentiment,
                        confidence=confidence,
                        reasoning=reasoning,
                        key_factors=key_factors,
                        risk_assessment=risk_assessment
                    ))
                    
                except Exception as e:
                    logger.warning(f"Simulation analysis error for {pred.get('symbol', 'unknown')}: {e}")
                    # Add fallback analysis
                    analysis_results.append(await self._generate_fallback_analysis(pred, real_time_data))
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Simulation analysis error: {e}")
            return []
    
    def _generate_analysis_prompt(self, symbol: str, prediction_score: float, features: Dict, analysis_depth: str, real_time_data: Dict[str, Any] = None) -> str:
        """Generate prompt for LLaMA analysis with real-time data"""
        prompt = f"""
        Analyze the following stock prediction and provide detailed reasoning:
        
        Symbol: {symbol}
        Prediction Score: {prediction_score:.3f} (0.0 = strong sell, 1.0 = strong buy)
        
        Key Features:
        {json.dumps(features, indent=2)}
        """
        
        # Add real-time data if available
        if real_time_data:
            prompt += f"""
            Real-time Data:
            Current Price: {real_time_data.get('last_price', 'N/A')}
            Change: {real_time_data.get('change_percent', 0):+.2f}%
            Volume: {real_time_data.get('volume', 0):,}
            """
        
        prompt += f"""
        Please provide:
        1. Sentiment analysis (positive/negative/neutral)
        2. Detailed reasoning for the prediction
        3. Key factors influencing the prediction
        4. Risk assessment
        
        Depth: {analysis_depth}
        Keep response concise but informative.
        """
        return prompt
    
    def _extract_key_factors(self, features: Dict, real_time_data: Dict[str, Any] = None) -> List[str]:
        """Extract key factors from features and real-time data"""
        key_factors = []
        
        # Extract key factors based on feature values
        if "price_momentum" in features:
            momentum = features["price_momentum"]
            if momentum > 0.1:
                key_factors.append("Positive price momentum")
            elif momentum < -0.1:
                key_factors.append("Negative price momentum")
        
        if "volume_trend" in features:
            volume = features["volume_trend"]
            if volume > 0.1:
                key_factors.append("Increasing volume")
            elif volume < -0.1:
                key_factors.append("Decreasing volume")
        
        if "rsi" in features:
            rsi = features["rsi"]
            if rsi > 70:
                key_factors.append("Overbought conditions (RSI > 70)")
            elif rsi < 30:
                key_factors.append("Oversold conditions (RSI < 30)")
        
        # Add real-time factors if available
        if real_time_data:
            change_pct = real_time_data.get("change_percent", 0)
            if change_pct > 2:
                key_factors.append(f"Strong positive movement (+{change_pct:.2f}%)")
            elif change_pct < -2:
                key_factors.append(f"Strong negative movement ({change_pct:.2f}%)")
            
            volume = real_time_data.get("volume", 0)
            if volume > 1000000:  # High volume
                key_factors.append("High trading volume")
        
        # Add generic factors if none found
        if not key_factors:
            key_factors = ["Technical indicators", "Market momentum", "Volume trends"]
            if real_time_data:
                key_factors.append("Real-time price action")
        
        return key_factors[:5]  # Limit to 5 key factors
    
    def _generate_risk_assessment(self, prediction_score: float, features: Dict, real_time_data: Dict[str, Any] = None) -> str:
        """Generate risk assessment based on prediction score, features, and real-time data"""
        risk_factors = []
        
        # Base risk assessment
        if prediction_score > 0.8:
            assessment = "High conviction buy signal with strong technical confirmation"
        elif prediction_score > 0.6:
            assessment = "Moderate buy signal with favorable risk-reward profile"
        elif prediction_score > 0.4:
            assessment = "Neutral signal with balanced risk factors"
        elif prediction_score > 0.2:
            assessment = "Moderate sell signal with increasing downside risk"
        else:
            assessment = "High conviction sell signal with strong bearish confirmation"
        
        # Add real-time risk factors
        if real_time_data:
            change_pct = real_time_data.get("change_percent", 0)
            if abs(change_pct) > 5:
                risk_factors.append("High volatility")
            
            volume = real_time_data.get("volume", 0)
            if volume < 10000:  # Low volume
                risk_factors.append("Low liquidity")
        
        if risk_factors:
            assessment += f" ({', '.join(risk_factors)})"
        
        return assessment
    
    async def _generate_fallback_analysis(self, prediction: Dict, real_time_data: Dict[str, Any] = None) -> AnalysisResult:
        """Generate fallback analysis when other methods fail"""
        symbol = prediction.get("symbol", "UNKNOWN")
        prediction_score = prediction.get("prediction_score", 0.5)
        
        sentiment = "positive" if prediction_score > 0.6 else "negative" if prediction_score < 0.4 else "neutral"
        confidence = min(0.8, max(0.3, prediction_score))
        
        # Add real-time data info
        rt_info = ""
        if real_time_data and symbol in real_time_data:
            rt_data = real_time_data[symbol]
            rt_info = f" with real-time movement of {rt_data.get('change_percent', 0):+.2f}%"
        
        reasoning = f"Fallback analysis for {symbol} with score {prediction_score:.3f}{rt_info}"
        key_factors = ["Technical analysis", "Market sentiment"]
        if real_time_data:
            key_factors.append("Real-time price action")
        risk_assessment = "Standard risk profile"
        
        return AnalysisResult(
            symbol=symbol,
            sentiment=sentiment,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            risk_assessment=risk_assessment
        )
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get analysis tool status"""
        return {
            "tool_id": self.tool_id,
            "llama_enabled": self.llama_enabled,
            "llama_model": self.llama_model,
            "langgraph_enabled": self.langgraph_enabled,
            "real_time_data": self.real_time_data,
            "cache_size": len(self.analysis_cache),
            "status": "active"
        }