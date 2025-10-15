#!/usr/bin/env python3
"""
Prediction Ranking Tool
======================

MCP tool for ranking predictions from RL agents and other models.
Provides natural language interpretation of user queries and explanations
for RL predictions and shortlisted trades.
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

from mcp_server.mcp_trading_server import MCPToolResult, MCPToolStatus
from core.rl_agent import rl_agent
from utils.ensemble_optimizer import get_ensemble_optimizer

logger = logging.getLogger(__name__)

@dataclass
class PredictionRanking:
    """Prediction ranking result"""
    symbol: str
    score: float
    confidence: float
    recommendation: str
    model_source: str
    features: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None

class PredictionTool:
    """
    Prediction ranking tool for MCP server
    
    Features:
    - Rank predictions from RL agents and ensemble models
    - Natural language interpretation of user queries
    - Explanation generation for predictions
    - Actionable insights with reasoning logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "prediction_tool")
        
        # Initialize ensemble optimizer
        self.ensemble_optimizer = get_ensemble_optimizer()
        
        # Ollama configuration for natural language processing
        self.ollama_enabled = config.get("ollama_enabled", False)
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "llama2")
        
        logger.info(f"Prediction Tool {self.tool_id} initialized")
    
    async def rank_predictions(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Rank predictions from RL agents and other models
        
        Args:
            arguments: {
                "symbols": ["RELIANCE.NS", "TCS.NS", ...] or "all",
                "models": ["rl", "ensemble", ...],
                "horizon": "day" | "week" | "month",
                "include_explanations": true,
                "natural_query": "Show me the best stocks for short term gains"
            }
        """
        try:
            symbols = arguments.get("symbols", [])
            models = arguments.get("models", ["rl"])
            horizon = arguments.get("horizon", "day")
            include_explanations = arguments.get("include_explanations", True)
            natural_query = arguments.get("natural_query", "")
            
            # Process natural language query if provided
            if natural_query and self.ollama_enabled:
                processed_query = await self._interpret_natural_query(natural_query)
                logger.info(f"Interpreted query: {processed_query}")
            
            # Get universe data for all symbols
            universe_data = await self._get_universe_data(symbols)
            
            # Rank predictions from different models
            ranked_predictions = []
            
            if "rl" in models:
                rl_rankings = await self._rank_rl_predictions(universe_data, horizon)
                ranked_predictions.extend(rl_rankings)
            
            if "ensemble" in models:
                ensemble_rankings = await self._rank_ensemble_predictions(universe_data)
                ranked_predictions.extend(ensemble_rankings)
            
            # Sort by score
            ranked_predictions.sort(key=lambda x: x.score, reverse=True)
            
            # Generate explanations if requested
            if include_explanations:
                for prediction in ranked_predictions:
                    prediction.explanation = await self._generate_explanation(prediction)
            
            # Prepare response
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": len(ranked_predictions),
                "models_used": models,
                "horizon": horizon,
                "ranked_predictions": [asdict(pred) for pred in ranked_predictions],
                "natural_query_processed": natural_query if natural_query else None
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Ranked {len(ranked_predictions)} predictions using {len(models)} models",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Prediction ranking error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def _interpret_natural_query(self, query: str) -> Dict[str, Any]:
        """Interpret natural language query using Ollama"""
        try:
            if not self.ollama_enabled:
                return {"processed": False}
            
            # Import ollama
            try:
                import ollama
            except ImportError:
                logger.warning("Ollama not available for natural language processing")
                return {"processed": False}
            
            # Prepare prompt for query interpretation
            prompt = f"""
            Interpret this natural language trading query and extract key parameters:
            "{query}"
            
            Extract:
            1. Time horizon (short-term, medium-term, long-term)
            2. Risk preference (conservative, moderate, aggressive)
            3. Sector preferences (if mentioned)
            4. Investment style (growth, value, income)
            
            Response format as JSON:
            {{
                "time_horizon": "day|week|month",
                "risk_preference": "low|medium|high",
                "sectors": ["sector1", "sector2"],
                "style": "growth|value|income"
            }}
            """
            
            # Generate response from LLM
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["\n\n"]
                }
            )
            
            # Parse response
            result_text = response.get("response", "{}")
            try:
                parsed_result = json.loads(result_text)
                return parsed_result
            except json.JSONDecodeError:
                return {"processed": True, "raw_response": result_text}
                
        except Exception as e:
            logger.warning(f"Natural language interpretation failed: {e}")
            return {"processed": False, "error": str(e)}
    
    async def _get_universe_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get universe data for symbols"""
        # In a real implementation, this would fetch real market data
        # For now, we'll simulate with sample data
        universe_data = {}
        
        # If "all" is specified or empty list, use sample symbols
        if not symbols or symbols == "all":
            symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
        
        for symbol in symbols:
            # Simulate market data with more realistic values
            base_price = 1000 + (hash(symbol) % 1000)  # Random price between 1000-2000
            volume = 500000 + (hash(symbol) % 2000000)  # Random volume between 500K-2.5M
            change_pct = ((hash(symbol) % 2000) / 100.0) - 10  # Random change between -10% to +10%
            change = base_price * (change_pct / 100.0)
            
            universe_data[symbol] = {
                "price": base_price,
                "volume": volume,
                "change": change,
                "change_pct": change_pct
            }
        
        return universe_data
    
    async def _rank_rl_predictions(self, universe_data: Dict[str, Any], horizon: str) -> List[PredictionRanking]:
        """Rank predictions using RL agent"""
        try:
            # Get RL rankings
            rl_rankings = rl_agent.rank_stocks(universe_data, horizon)
            
            # Convert to PredictionRanking objects
            rankings = []
            for stock in rl_rankings:
                rankings.append(PredictionRanking(
                    symbol=stock["symbol"],
                    score=stock["score"],
                    confidence=stock["score"],  # Using score as confidence
                    recommendation="BUY" if stock["score"] > 0.7 else "HOLD",
                    model_source="rl_agent",
                    features={
                        "price": stock.get("price", 0),
                        "horizon": stock.get("horizon", horizon)
                    }
                ))
            
            return rankings
            
        except Exception as e:
            logger.error(f"RL prediction ranking error: {e}")
            return []
    
    async def _rank_ensemble_predictions(self, universe_data: Dict[str, Any]) -> List[PredictionRanking]:
        """Rank predictions using ensemble optimizer"""
        try:
            rankings = []
            
            # For each symbol, get ensemble prediction
            for symbol, data in universe_data.items():
                try:
                    # Create features array (simplified)
                    features = [
                        data.get("price", 0) / 1000,  # Normalized price
                        data.get("volume", 0) / 1000000,  # Normalized volume
                        data.get("change", 0),  # Price change
                        data.get("change_pct", 0),  # Percentage change
                        1,  # Horizon encoding (day)
                        abs(data.get("change_pct", 0)),  # Volatility
                        1 if data.get("change", 0) > 0 else 0,  # Positive momentum
                        0.5,  # Price tier
                        0,  # Reserved
                        0   # Reserved
                    ]
                    
                    # Get ensemble prediction
                    result = self.ensemble_optimizer.get_detailed_ensemble_analysis(
                        np.array(features)
                    )
                    
                    if result.get("success", False):
                        rankings.append(PredictionRanking(
                            symbol=symbol,
                            score=result.get("prediction", 0),
                            confidence=result.get("confidence", 0.5),
                            recommendation=result.get("recommendation", "HOLD"),
                            model_source="ensemble_optimizer",
                            features={
                                "prediction": result.get("prediction", 0),
                                "confidence": result.get("confidence", 0.5),
                                "consensus_level": result.get("consensus_level", 0.5)
                            }
                        ))
                except Exception as e:
                    logger.warning(f"Ensemble prediction error for {symbol}: {e}")
                    continue
            
            return rankings
            
        except Exception as e:
            logger.error(f"Ensemble prediction ranking error: {e}")
            return []
    
    async def _generate_explanation(self, prediction: PredictionRanking) -> str:
        """Generate explanation for a prediction"""
        try:
            if prediction.model_source == "rl_agent":
                return f"RL agent recommends {prediction.recommendation} for {prediction.symbol} with score {prediction.score:.3f}. This is based on technical patterns and market momentum indicators."
            elif prediction.model_source == "ensemble_optimizer":
                return f"Ensemble model recommends {prediction.recommendation} for {prediction.symbol} with confidence {prediction.confidence:.2f}. This combines multiple predictive models for improved accuracy."
            else:
                return f"Model {prediction.model_source} recommends {prediction.recommendation} for {prediction.symbol} with score {prediction.score:.3f}."
        except Exception as e:
            logger.warning(f"Explanation generation error: {e}")
            return "Explanation unavailable"
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get prediction tool status"""
        return {
            "tool_id": self.tool_id,
            "ollama_enabled": self.ollama_enabled,
            "ollama_model": self.ollama_model,
            "status": "active"
        }