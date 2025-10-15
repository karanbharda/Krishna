#!/usr/bin/env python3
"""
Scan All Tool
=============

MCP tool for scanning and filtering stock shortlists.
Provides filtered shortlists based on user criteria and market conditions.
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
class ShortlistedStock:
    """Shortlisted stock with filtering criteria"""
    symbol: str
    score: float
    confidence: float
    price: float
    volume: float
    change_pct: float
    recommendation: str
    filters_matched: List[str]
    risk_level: str
    sector: Optional[str] = None
    market_cap: Optional[str] = None

class ScanTool:
    """
    Scan all tool for MCP server
    
    Features:
    - Generate filtered shortlists based on criteria
    - Natural language interpretation of user queries
    - Explanation generation for shortlisted trades
    - Actionable insights with reasoning logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "scan_tool")
        
        # Initialize ensemble optimizer
        self.ensemble_optimizer = get_ensemble_optimizer()
        
        # Ollama configuration for natural language processing
        self.ollama_enabled = config.get("ollama_enabled", False)
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "llama2")
        
        logger.info(f"Scan Tool {self.tool_id} initialized")
    
    async def scan_all(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Scan and filter stock shortlists
        
        Args:
            arguments: {
                "filters": {
                    "min_price": 50,
                    "max_price": 5000,
                    "min_volume": 10000,
                    "min_score": 0.6,
                    "sectors": ["BANKING", "IT", "AUTO"],
                    "market_caps": ["LARGE_CAP", "MID_CAP"],
                    "risk_levels": ["LOW", "MEDIUM"]
                },
                "sort_by": "score" | "confidence" | "volume" | "change_pct",
                "limit": 50,
                "natural_query": "Find mid-cap IT stocks with strong momentum"
            }
        """
        try:
            filters = arguments.get("filters", {})
            sort_by = arguments.get("sort_by", "score")
            limit = arguments.get("limit", 50)
            natural_query = arguments.get("natural_query", "")
            
            # Process natural language query if provided
            if natural_query and self.ollama_enabled:
                processed_query = await self._interpret_natural_query(natural_query)
                # Merge with filters
                if processed_query.get("processed", False):
                    filters = self._merge_filters_with_query(filters, processed_query)
                logger.info(f"Interpreted query: {processed_query}")
            
            # Get universe data
            universe_data = await self._get_universe_data()
            
            # Apply filters and generate shortlist
            shortlisted_stocks = await self._generate_shortlist(universe_data, filters)
            
            # Sort results
            shortlisted_stocks.sort(key=lambda x: getattr(x, sort_by, 0), reverse=True)
            
            # Limit results
            shortlisted_stocks = shortlisted_stocks[:limit]
            
            # Generate explanations for top stocks
            for i, stock in enumerate(shortlisted_stocks[:10]):  # Explain top 10
                stock.explanation = await self._generate_explanation(stock)
            
            # Prepare response
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "total_scanned": len(universe_data),
                "total_shortlisted": len(shortlisted_stocks),
                "filters_applied": filters,
                "sort_by": sort_by,
                "limit": limit,
                "shortlisted_stocks": [asdict(stock) for stock in shortlisted_stocks],
                "natural_query_processed": natural_query if natural_query else None
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Scanned {len(universe_data)} stocks and shortlisted {len(shortlisted_stocks)} based on filters",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Scan all error: {e}")
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
            Interpret this natural language stock scanning query and extract filtering criteria:
            "{query}"
            
            Extract:
            1. Sector preferences (BANKING, IT, AUTO, PHARMA, etc.)
            2. Market capitalization (LARGE_CAP, MID_CAP, SMALL_CAP)
            3. Price range (min, max)
            4. Risk level (LOW, MEDIUM, HIGH)
            5. Investment style (GROWTH, VALUE, INCOME)
            6. Momentum preferences (STRONG, MODERATE, ANY)
            
            Response format as JSON:
            {{
                "sectors": ["IT", "BANKING"],
                "market_caps": ["MID_CAP", "LARGE_CAP"],
                "price_range": {{"min": 100, "max": 2000}},
                "risk_levels": ["LOW", "MEDIUM"],
                "style": "GROWTH",
                "momentum": "STRONG"
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
    
    def _merge_filters_with_query(self, filters: Dict, query_result: Dict) -> Dict:
        """Merge filters with natural language query results"""
        merged = filters.copy()
        
        # Merge sectors
        if "sectors" in query_result:
            existing_sectors = merged.get("sectors", [])
            query_sectors = query_result["sectors"]
            merged["sectors"] = list(set(existing_sectors + query_sectors))
        
        # Merge market caps
        if "market_caps" in query_result:
            existing_caps = merged.get("market_caps", [])
            query_caps = query_result["market_caps"]
            merged["market_caps"] = list(set(existing_caps + query_caps))
        
        # Merge price range
        if "price_range" in query_result:
            price_range = query_result["price_range"]
            if "min" in price_range:
                merged["min_price"] = max(merged.get("min_price", 0), price_range["min"])
            if "max" in price_range:
                merged["max_price"] = min(merged.get("max_price", float('inf')), price_range["max"])
        
        # Merge risk levels
        if "risk_levels" in query_result:
            existing_risk = merged.get("risk_levels", [])
            query_risk = query_result["risk_levels"]
            merged["risk_levels"] = list(set(existing_risk + query_risk))
        
        return merged
    
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
            ("AXISBANK.NS", "BANKING", "LARGE_CAP")
        ]
        
        for symbol, sector, market_cap in sample_symbols:
            # Simulate market data with more realistic values
            base_price = 1000 + (hash(symbol) % 1000)  # Random price between 1000-2000
            volume = 500000 + (hash(symbol) % 2000000)  # Random volume between 500K-2.5M
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
    
    async def _generate_shortlist(self, universe_data: Dict[str, Any], filters: Dict[str, Any]) -> List[ShortlistedStock]:
        """Generate filtered shortlist"""
        try:
            shortlisted = []
            
            # Extract filter criteria
            min_price = filters.get("min_price", 0)
            max_price = filters.get("max_price", float('inf'))
            min_volume = filters.get("min_volume", 0)
            min_score = filters.get("min_score", 0.5)
            sectors = filters.get("sectors", [])
            market_caps = filters.get("market_caps", [])
            risk_levels = filters.get("risk_levels", [])
            
            # Get predictions for all stocks
            rl_predictions = rl_agent.rank_stocks(universe_data, "day")
            
            # Convert RL predictions to a dict for easy lookup
            rl_scores = {stock["symbol"]: stock for stock in rl_predictions}
            
            # Apply filters
            for symbol, data in universe_data.items():
                try:
                    # Price filter
                    price = data.get("price", 0)
                    if not (min_price <= price <= max_price):
                        continue
                    
                    # Volume filter
                    volume = data.get("volume", 0)
                    if volume < min_volume:
                        continue
                    
                    # Sector filter
                    if sectors and data.get("sector") not in sectors:
                        continue
                    
                    # Market cap filter
                    if market_caps and data.get("market_cap") not in market_caps:
                        continue
                    
                    # Risk level filter
                    if risk_levels and data.get("risk_level") not in risk_levels:
                        continue
                    
                    # Get prediction score
                    rl_score = rl_scores.get(symbol, {"score": 0.5, "confidence": 0.5})
                    score = rl_score.get("score", 0.5)
                    
                    # Score filter
                    if score < min_score:
                        continue
                    
                    # Determine recommendation
                    if score > 0.8:
                        recommendation = "STRONG_BUY"
                    elif score > 0.7:
                        recommendation = "BUY"
                    elif score > 0.5:
                        recommendation = "HOLD"
                    else:
                        recommendation = "SELL"
                    
                    # Track matched filters
                    filters_matched = []
                    if min_price <= price <= max_price:
                        filters_matched.append("price")
                    if volume >= min_volume:
                        filters_matched.append("volume")
                    if sectors and data.get("sector") in sectors:
                        filters_matched.append("sector")
                    if market_caps and data.get("market_cap") in market_caps:
                        filters_matched.append("market_cap")
                    if risk_levels and data.get("risk_level") in risk_levels:
                        filters_matched.append("risk_level")
                    if score >= min_score:
                        filters_matched.append("score")
                    
                    # Create shortlisted stock
                    shortlisted_stock = ShortlistedStock(
                        symbol=symbol,
                        score=score,
                        confidence=rl_score.get("confidence", 0.5),
                        price=price,
                        volume=volume,
                        change_pct=data.get("change_pct", 0),
                        recommendation=recommendation,
                        filters_matched=filters_matched,
                        risk_level=data.get("risk_level", "MEDIUM"),
                        sector=data.get("sector"),
                        market_cap=data.get("market_cap")
                    )
                    
                    shortlisted.append(shortlisted_stock)
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            return shortlisted
            
        except Exception as e:
            logger.error(f"Shortlist generation error: {e}")
            return []
    
    async def _generate_explanation(self, stock: ShortlistedStock) -> str:
        """Generate explanation for a shortlisted stock"""
        try:
            explanation = f"{stock.symbol} is shortlisted with a {stock.recommendation} recommendation "
            explanation += f"(score: {stock.score:.3f}). "
            
            # Add reasoning based on filters
            if "price" in stock.filters_matched:
                explanation += f"Price ({stock.price:.2f}) meets criteria. "
            
            if "volume" in stock.filters_matched:
                explanation += f"Volume ({stock.volume:,.0f}) is strong. "
            
            if "sector" in stock.filters_matched and stock.sector:
                explanation += f"{stock.sector} sector aligns with filters. "
            
            if stock.change_pct > 2:
                explanation += f"Positive momentum ({stock.change_pct:+.2f}%). "
            elif stock.change_pct < -2:
                explanation += f"Negative momentum ({stock.change_pct:+.2f}%). "
            
            return explanation.strip()
            
        except Exception as e:
            logger.warning(f"Explanation generation error: {e}")
            return "Explanation unavailable"
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get scan tool status"""
        return {
            "tool_id": self.tool_id,
            "ollama_enabled": self.ollama_enabled,
            "ollama_model": self.ollama_model,
            "status": "active"
        }