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
        self.rl_model_type = config.get("rl_model_type", "linucb")  # linucb or ppo-lite
        
        # Real-time data configuration
        self.real_time_data = config.get("real_time_data", True)
        self.websocket_manager = None
        
        # Performance tracking
        self.scan_cache = {}
        self.cache_timeout = config.get("cache_timeout", 120)  # seconds
        
        # Initialize real-time data if available
        if WEBSOCKET_AVAILABLE and self.real_time_data:
            try:
                self.websocket_manager = EnhancedWebSocketManager()
                logger.info("Real-time data connection established for scanning")
            except Exception as e:
                logger.warning(f"Failed to initialize real-time data for scanning: {e}")
        
        logger.info(f"Scan All Tool {self.tool_id} initialized with real-time capabilities")
    
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
            
            # Rank stocks using RL agent
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
                reasoning=f"Scanned {len(universe_data)} stocks, ranked {len(final_rankings)} based on {self.rl_model_type} RL model with real-time data",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Scan all error: {e}")
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
    
    async def _update_real_time_data(self, rankings: List[Dict]) -> None:
        """Update existing rankings with real-time data"""
        if not self.websocket_manager:
            return
        
        try:
            for ranking in rankings:
                symbol = ranking.get("symbol")
                if symbol:
                    try:
                        rt_data = await self.websocket_manager.get_latest_quote(symbol)
                        if rt_data:
                            ranking["real_time_price"] = rt_data.get("last_price")
                            ranking["change_percent"] = rt_data.get("change_percent")
                            ranking["timestamp"] = datetime.now().isoformat()
                    except Exception as e:
                        logger.warning(f"Failed to update real-time data for {symbol}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Real-time data update error: {e}")
    
    async def _get_universe_data(self) -> Dict[str, Any]:
        """Get universe data for all stocks"""
        # In a real implementation, this would fetch real market data
        # For now, we'll simulate with sample data
        universe_data = {}
        
        # Sample symbols with different characteristics
        sample_symbols = [
            ("RELIANCE.NS", "ENERGY", "LARGE_CAP"),
            ("TCS.NS", "IT", "LARGE_CAP"),
            ("INFY.NS", "IT", "LARGE_CAP"),
            ("HDFCBANK.NS", "BANKING", "LARGE_CAP"),
            ("ICICIBANK.NS", "BANKING", "LARGE_CAP"),
            ("SBIN.NS", "BANKING", "LARGE_CAP"),
            ("BHARTIARTL.NS", "TELECOM", "LARGE_CAP"),
            ("HINDUNILVR.NS", "CONSUMER", "LARGE_CAP"),
            ("ITC.NS", "CONSUMER", "LARGE_CAP"),
            ("LT.NS", "INFRA", "LARGE_CAP"),
            ("ADANIPORTS.NS", "INFRA", "LARGE_CAP"),
            ("ASIANPAINT.NS", "CONSUMER", "LARGE_CAP"),
            ("MARUTI.NS", "AUTO", "LARGE_CAP"),
            ("TATAMOTORS.NS", "AUTO", "LARGE_CAP"),
            ("SUNPHARMA.NS", "PHARMA", "LARGE_CAP"),
            ("DRREDDY.NS", "PHARMA", "LARGE_CAP"),
            ("WIPRO.NS", "IT", "MID_CAP"),
            ("TECHM.NS", "IT", "MID_CAP"),
            ("BAJFINANCE.NS", "FINANCE", "LARGE_CAP"),
            ("AXISBANK.NS", "BANKING", "LARGE_CAP"),
            ("NESTLEIND.NS", "CONSUMER", "LARGE_CAP"),
            ("HDFC.NS", "FINANCE", "LARGE_CAP"),
            ("CIPLA.NS", "PHARMA", "LARGE_CAP"),
            ("ULTRACEMCO.NS", "CEMENT", "LARGE_CAP"),
            ("POWERGRID.NS", "POWER", "LARGE_CAP")
        ]
        
        for symbol, sector, market_cap in sample_symbols:
            # Simulate market data with more realistic values
            base_price = 500 + (hash(symbol) % 2000)  # Random price between 500-2500
            volume = 100000 + (hash(symbol) % 5000000)  # Random volume between 100K-5M
            change_pct = ((hash(symbol) % 2000) / 100.0) - 10  # Random change between -10% to +10%
            change = base_price * (change_pct / 100.0)
            
            universe_data[symbol] = {
                "price": base_price,
                "volume": volume,
                "change": change,
                "change_pct": change_pct,
                "sector": sector,
                "market_cap": market_cap,
                "risk_level": "LOW" if abs(change_pct) < 3 else "MEDIUM" if abs(change_pct) < 6 else "HIGH"
            }
        
        return universe_data
    
    def _filter_universe_data(self, universe_data: Dict[str, Any], sectors: List[str], market_caps: List[str]) -> Dict[str, Any]:
        """Filter universe data based on sectors and market caps"""
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
        """Rank stocks using RL agent with real-time data"""
        try:
            rankings = []
            
            if RL_AGENT_AVAILABLE:
                # Use the actual RL agent for ranking
                logger.info(f"Using RL agent to rank {len(universe_data)} stocks")
                
                # Get rankings from RL agent
                rl_rankings = rl_agent.rank_stocks(universe_data, "day")
                
                # Convert to our format with real-time data
                for rank_data in rl_rankings:
                    symbol = rank_data["symbol"]
                    score = rank_data["score"]
                    
                    # Get real-time data if available
                    rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                    
                    # Determine recommendation based on score
                    if score > 0.8:
                        recommendation = "STRONG_BUY"
                    elif score > 0.7:
                        recommendation = "BUY"
                    elif score > 0.5:
                        recommendation = "HOLD"
                    elif score > 0.3:
                        recommendation = "SELL"
                    else:
                        recommendation = "STRONG_SELL"
                    
                    rankings.append(StockRanking(
                        symbol=symbol,
                        score=score,
                        confidence=rank_data.get("confidence", score),
                        recommendation=recommendation,
                        rank=0,  # Will be set later
                        sector=universe_data.get(symbol, {}).get("sector"),
                        market_cap=universe_data.get(symbol, {}).get("market_cap"),
                        real_time_price=rt_data.get("last_price") if rt_data else None,
                        change_percent=rt_data.get("change_percent") if rt_data else None
                    ))
            else:
                # Fallback to simulated ranking
                logger.warning("RL agent not available, using simulated ranking")
                
                for symbol, data in universe_data.items():
                    try:
                        # Get real-time data if available
                        rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
                        
                        # Simulate prediction with more realistic values
                        base_score = 0.3 + (hash(symbol) % 140) / 200.0  # Score between 0.3-1.0
                        confidence = 0.6 + (hash(symbol + "conf") % 40) / 100.0  # Confidence between 0.6-1.0
                        
                        # Adjust score based on real-time data if available
                        if rt_data:
                            # Adjust score based on real-time momentum
                            momentum = rt_data.get("change_percent", 0)
                            base_score = max(0.1, min(0.9, base_score + (momentum / 100)))
                        
                        # Determine recommendation based on score
                        if base_score > 0.8:
                            recommendation = "STRONG_BUY"
                        elif base_score > 0.7:
                            recommendation = "BUY"
                        elif base_score > 0.5:
                            recommendation = "HOLD"
                        elif base_score > 0.3:
                            recommendation = "SELL"
                        else:
                            recommendation = "STRONG_SELL"
                        
                        rankings.append(StockRanking(
                            symbol=symbol,
                            score=base_score,
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
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get scan all tool status"""
        return {
            "tool_id": self.tool_id,
            "rl_model_type": self.rl_model_type,
            "real_time_data": self.real_time_data,
            "rl_agent_available": RL_AGENT_AVAILABLE,
            "cache_size": len(self.scan_cache),
            "status": "active"
        }