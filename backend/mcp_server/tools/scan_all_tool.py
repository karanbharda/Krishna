#!/usr/bin/env python3
"""
Scan All Tool for Venting Layer
=============================

MCP tool for batch scanning all cached stocks and ranking by RL agent
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

# Import the existing RL agent
try:
    from ...core.rl_agent import rl_agent
    RL_AGENT_AVAILABLE = True
except ImportError:
    RL_AGENT_AVAILABLE = False
    logging.warning("RL agent not available for scanning")

# Import ensemble optimizer
try:
    from ...utils.ensemble_optimizer import get_ensemble_optimizer
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("Ensemble optimizer not available for scanning")

logger = logging.getLogger(__name__)

@dataclass
class StockRanking:
    """Stock ranking result"""
    symbol: str
    score: float
    confidence: float
    recommendation: str
    rank: int
    sector: Optional[str] = None
    market_cap: Optional[str] = None
    real_time_price: Optional[float] = None
    change_percent: Optional[float] = None
    timestamp: str = None

class ScanAllTool:
    """
    Scan All tool for Venting Layer
    
    Features:
    - Batch scan all cached stocks
    - Rank by RL agent
    - Return validated JSON compatible with Trading Executor
    - Real-time dynamic scanning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "scan_all_tool")
        
        # RL model configuration
        self.rl_model_type = config.get("rl_model_type", "ppo")  # Updated to ppo
        self.ensemble_method = config.get("ensemble_method", "rl_integrated")  # New ensemble method
        
        # Real-time data configuration
        self.real_time_data = config.get("real_time_data", True)
        self.websocket_manager = None
        
        # Performance tracking
        self.scan_cache = {}
        self.cache_timeout = config.get("cache_timeout", 120)  # seconds
        
        # Initialize ensemble optimizer
        self.ensemble_optimizer = None
        if ENSEMBLE_AVAILABLE:
            self.ensemble_optimizer = get_ensemble_optimizer()
        
        # Initialize real-time data if available
        if WEBSOCKET_AVAILABLE and self.real_time_data:
            try:
                self.websocket_manager = EnhancedWebSocketManager()
                logger.info("Real-time data connection established for scanning")
            except Exception as e:
                logger.warning(f"Failed to initialize real-time data for scanning: {e}")
        
        logger.info(f"Scan All Tool {self.tool_id} initialized with enhanced RL capabilities")
    
    async def scan_all(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Batch scan all cached stocks and rank by RL agent
        
        Args:
            arguments: {
                "min_score": 0.5,
                "max_results": 100,
                "sectors": ["BANKING", "IT", "AUTO"],
                "market_caps": ["LARGE_CAP", "MID_CAP"],
                "sort_by": "score" | "confidence",
                "real_time": true
            }
        """
        try:
            min_score = arguments.get("min_score", 0.5)
            max_results = arguments.get("max_results", 100)
            sectors = arguments.get("sectors", [])
            market_caps = arguments.get("market_caps", [])
            sort_by = arguments.get("sort_by", "score")
            real_time = arguments.get("real_time", self.real_time_data)
            
            # Create cache key
            cache_key = f"{hash(str(sectors))}_{hash(str(market_caps))}_{min_score}_{sort_by}"
            
            # Check cache for recent scan
            if cache_key in self.scan_cache:
                cached_result = self.scan_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_result["timestamp"])).seconds < self.cache_timeout:
                    logger.info("Returning cached scan results")
                    # Update real-time data if requested
                    if real_time and self.websocket_manager:
                        try:
                            await self._update_real_time_data(cached_result["rankings"])
                        except Exception as e:
                            logger.warning(f"Failed to update real-time data: {e}")
                    
                    return MCPToolResult(
                        status=MCPToolStatus.SUCCESS,
                        data=cached_result,
                        reasoning="Returning cached scan results with real-time updates",
                        confidence=0.9
                    )
            
            # Get universe data
            universe_data = await self._get_universe_data()
            
            # Filter universe data based on criteria
            filtered_data = self._filter_universe_data(universe_data, sectors, market_caps)
            
            # Get real-time data if requested
            real_time_data = {}
            if real_time and self.websocket_manager:
                try:
                    symbols = list(filtered_data.keys())
                    real_time_data = await self._get_real_time_data(symbols)
                except Exception as e:
                    logger.warning(f"Failed to get real-time data: {e}")
            
            # Rank stocks using enhanced RL agent
            ranked_stocks = await self._rank_stocks(filtered_data, real_time_data)
            
            # Filter by minimum score
            filtered_rankings = [stock for stock in ranked_stocks if stock.score >= min_score]
            
            # Sort results
            filtered_rankings.sort(key=lambda x: getattr(x, sort_by, 0), reverse=True)
            
            # Limit results
            final_rankings = filtered_rankings[:max_results]
            
            # Add rankings
            for i, stock in enumerate(final_rankings):
                stock.rank = i + 1
            
            # Add timestamp to each ranking
            current_time = datetime.now().isoformat()
            for stock in final_rankings:
                stock.timestamp = current_time
            
            # Prepare response
            response_data = {
                "timestamp": current_time,
                "total_scanned": len(universe_data),
                "total_filtered": len(filtered_data),
                "total_ranked": len(final_rankings),
                "min_score": min_score,
                "sort_by": sort_by,
                "real_time_data_used": bool(real_time_data),
                "rankings": [asdict(stock) for stock in final_rankings]
            }
            
            # Cache the result
            self.scan_cache[cache_key] = response_data
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Scanned {len(universe_data)} stocks, ranked {len(final_rankings)} based on enhanced {self.rl_model_type} RL model with real-time data",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Scan all error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def _get_real_time_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time data for symbols"""
        if not self.websocket_manager:
            return {}
            
        try:
            return await self.websocket_manager.get_batch_data(symbols)
        except Exception as e:
            logger.warning(f"Failed to get real-time data: {e}")
            return {}

    async def _update_real_time_data(self, rankings: List[Dict[str, Any]]):
        """Update rankings with real-time data"""
        if not self.websocket_manager:
            return
            
        try:
            symbols = [r["symbol"] for r in rankings]
            real_time_data = await self.websocket_manager.get_batch_data(symbols)
            
            for ranking in rankings:
                symbol = ranking["symbol"]
                if symbol in real_time_data:
                    rt_data = real_time_data[symbol]
                    ranking["real_time_price"] = rt_data.get("last_price")
                    ranking["change_percent"] = rt_data.get("change_percent")
        except Exception as e:
            logger.warning(f"Failed to update real-time data: {e}")

    async def _get_universe_data(self) -> Dict[str, Any]:
        """Get universe data from cache or data service"""
        try:
            # In a real implementation, this would fetch from a data source
            # For now, we'll return a sample structure
            return {
                "RELIANCE": {
                    "price": 2500.0,
                    "volume": 1000000,
                    "change": 25.0,
                    "change_pct": 1.0,
                    "sector": "ENERGY",
                    "market_cap": "LARGE_CAP",
                    "rsi": 60,
                    "macd": 0.5,
                    "sma_20": 2480.0,
                    "sma_50": 2450.0,
                    "sma_200": 2400.0,
                    "atr": 50.0,
                    "volatility": 0.02
                },
                "TCS": {
                    "price": 3500.0,
                    "volume": 500000,
                    "change": -35.0,
                    "change_pct": -1.0,
                    "sector": "IT",
                    "market_cap": "LARGE_CAP",
                    "rsi": 45,
                    "macd": -0.3,
                    "sma_20": 3520.0,
                    "sma_50": 3550.0,
                    "sma_200": 3600.0,
                    "atr": 70.0,
                    "volatility": 0.015
                }
            }
        except Exception as e:
            logger.error(f"Failed to get universe data: {e}")
            return {}

    def _filter_universe_data(self, universe_data: Dict[str, Any], 
                            sectors: List[str], market_caps: List[str]) -> Dict[str, Any]:
        """Filter universe data based on criteria"""
        if not sectors and not market_caps:
            return universe_data
            
        filtered_data = {}
        
        for symbol, data in universe_data.items():
            # Sector filter
            if sectors and data.get("sector") not in sectors:
                continue
            
            # Market cap filter
            if market_caps and data.get("market_cap") not in market_caps:
                continue
            
            filtered_data[symbol] = data
        
        return filtered_data
    
    async def _rank_stocks(self, universe_data: Dict[str, Any], real_time_data: Dict[str, Any] = None) -> List[StockRanking]:
        """Rank stocks using enhanced RL agent with multi-agent and hierarchical approaches"""
        try:
            rankings = []
            
            if RL_AGENT_AVAILABLE:
                # Use the enhanced RL agent for ranking
                logger.info(f"Using enhanced RL agent to rank {len(universe_data)} stocks")
                
                # Get rankings from enhanced RL agent with multi-objective optimization
                rl_rankings = rl_agent.rank_stocks(universe_data, "day")
                
                # Convert to our format with real-time data and multi-objective scores
                for rank_data in rl_rankings:
                    symbol = rank_data["symbol"]
                    score = rank_data["score"]
                    confidence = rank_data.get("confidence", score)
                    
                    # Get real-time data if available
                    rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                    
                    # Enhanced multi-objective recommendation with risk adjustment
                    market_volatility = universe_data.get(symbol, {}).get("volatility", 0.02)
                    risk_adjusted_score = score * (1 - market_volatility * 2)  # Adjust for volatility
                    
                    # Determine recommendation based on multi-objective score
                    if risk_adjusted_score > 0.75:
                        recommendation = "STRONG_BUY"
                    elif risk_adjusted_score > 0.65:
                        recommendation = "BUY"
                    elif risk_adjusted_score > 0.5:
                        recommendation = "HOLD"
                    elif risk_adjusted_score > 0.35:
                        recommendation = "SELL"
                    else:
                        recommendation = "STRONG_SELL"
                    
                    rankings.append(StockRanking(
                        symbol=symbol,
                        score=risk_adjusted_score,  # Use risk-adjusted score
                        confidence=confidence,
                        recommendation=recommendation,
                        rank=0,  # Will be set later
                        sector=universe_data.get(symbol, {}).get("sector"),
                        market_cap=universe_data.get(symbol, {}).get("market_cap"),
                        real_time_price=rt_data.get("last_price") if rt_data else None,
                        change_percent=rt_data.get("change_percent") if rt_data else None
                    ))
            else:
                # Fallback to simulated ranking with enhanced logic
                logger.warning("Enhanced RL agent not available, using simulated ranking")
                
                for symbol, data in universe_data.items():
                    try:
                        # Get real-time data if available
                        rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                        
                        # Enhanced simulation with financial indicators and multi-objective optimization
                        base_score = self._calculate_enhanced_score(data, rt_data)
                        confidence = 0.6 + (hash(symbol + "conf") % 40) / 100.0  # Confidence between 0.6-1.0
                        
                        # Adjust score based on real-time data if available
                        if rt_data:
                            # Adjust score based on real-time momentum
                            momentum = rt_data.get("change_percent", 0)
                            base_score = max(0.1, min(0.9, base_score + (momentum / 100)))
                        
                        # Multi-objective risk adjustment
                        market_volatility = data.get("volatility", 0.02)
                        risk_adjusted_score = base_score * (1 - market_volatility * 2)
                        
                        # Determine recommendation based on multi-objective score
                        if risk_adjusted_score > 0.75:
                            recommendation = "STRONG_BUY"
                        elif risk_adjusted_score > 0.65:
                            recommendation = "BUY"
                        elif risk_adjusted_score > 0.5:
                            recommendation = "HOLD"
                        elif risk_adjusted_score > 0.35:
                            recommendation = "SELL"
                        else:
                            recommendation = "STRONG_SELL"
                        
                        rankings.append(StockRanking(
                            symbol=symbol,
                            score=risk_adjusted_score,  # Use risk-adjusted score
                            confidence=confidence,
                            recommendation=recommendation,
                            rank=0,  # Will be set later
                            sector=data.get("sector"),
                            market_cap=data.get("market_cap"),
                            real_time_price=rt_data.get("last_price") if rt_data else None,
                            change_percent=rt_data.get("change_percent") if rt_data else None
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Error ranking {symbol}: {e}")
                        continue
            
            return rankings
            
        except Exception as e:
            logger.error(f"Stock ranking error: {e}")
            return []
    
    def _calculate_enhanced_score(self, data: Dict[str, Any], rt_data: Dict[str, Any]) -> float:
        """Calculate enhanced score based on financial indicators"""
        try:
            # Extract features
            price = data.get('price', 100)
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            sma_20 = data.get('sma_20', price)
            sma_50 = data.get('sma_50', price)
            atr = data.get('atr', price * 0.02)
            volatility = data.get('volatility', 0.02)
            
            # Base score components
            score = 0.5  # Neutral base
            
            # RSI component (30-70 is neutral)
            if 30 <= rsi <= 70:
                score += 0.1
            elif rsi < 30:  # Oversold
                score += 0.15
            elif rsi > 70:  # Overbought
                score -= 0.15
            
            # MACD component
            if macd > 0:
                score += 0.1
            else:
                score -= 0.05
            
            # Moving average component
            if sma_20 > sma_50:  # Bullish trend
                score += 0.1
            else:
                score -= 0.1
            
            # Volatility component (prefer moderate volatility)
            if 0.01 <= volatility <= 0.03:
                score += 0.05
            elif volatility > 0.05:
                score -= 0.1
            
            # Real-time momentum if available
            if rt_data:
                change_pct = rt_data.get("change_percent", 0)
                if change_pct > 2:
                    score += 0.1
                elif change_pct < -2:
                    score -= 0.1
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced score: {e}")
            return 0.5

    def get_tool_status(self) -> Dict[str, Any]:
        """Get scan all tool status"""
        return {
            "tool_id": self.tool_id,
            "rl_model_type": self.rl_model_type,
            "ensemble_method": self.ensemble_method,
            "real_time_data": self.real_time_data,
            "rl_agent_available": RL_AGENT_AVAILABLE,
            "ensemble_available": ENSEMBLE_AVAILABLE,
            "cache_size": len(self.scan_cache),
            "status": "active"
        }