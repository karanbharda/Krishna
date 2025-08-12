"""
Professional Sell Logic Integration
Integrates the professional sell logic with existing trading modules
"""

import logging
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .professional_sell_logic import (
    ProfessionalSellLogic, PositionMetrics, SellDecision, SellReason
)
from .market_context_analyzer import MarketContextAnalyzer

logger = logging.getLogger(__name__)

class ProfessionalSellIntegration:
    """
    Integration layer for professional sell logic
    Bridges the gap between existing trading modules and new professional logic
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize professional components
        self.sell_logic = ProfessionalSellLogic(config)
        self.market_analyzer = MarketContextAnalyzer(config)
        
        # Integration settings
        self.enable_professional_logic = config.get("enable_professional_sell_logic", True)
        self.fallback_to_legacy = config.get("fallback_to_legacy_sell", False)
        
        logger.info("Professional Sell Integration initialized")
    
    def evaluate_professional_sell(
        self,
        ticker: str,
        current_price: float,
        portfolio_holdings: Dict,
        analysis_data: Dict,
        price_history: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Main integration point for professional sell evaluation
        Returns a decision compatible with existing trading modules
        """
        
        if not self.enable_professional_logic:
            logger.info("Professional sell logic disabled, using legacy logic")
            return self._legacy_sell_decision(ticker, current_price, portfolio_holdings, analysis_data)
        
        try:
            # Check if we have a position
            if ticker not in portfolio_holdings:
                return self._no_position_decision()
            
            # Build position metrics
            position_metrics = self._build_position_metrics(
                ticker, current_price, portfolio_holdings, price_history
            )
            
            # Build market context
            market_context = self._build_market_context(analysis_data, price_history)
            
            # Extract analysis components
            technical_analysis = analysis_data.get("technical_indicators", {})
            sentiment_analysis = analysis_data.get("sentiment", {})
            ml_analysis = analysis_data.get("ml_analysis", {})
            
            # Get professional sell decision
            sell_decision = self.sell_logic.evaluate_sell_decision(
                ticker=ticker,
                position_metrics=position_metrics,
                market_context=market_context,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                ml_analysis=ml_analysis
            )
            
            # Convert to legacy format
            return self._convert_to_legacy_format(sell_decision, position_metrics)
            
        except Exception as e:
            logger.error(f"Error in professional sell evaluation: {e}")
            if self.fallback_to_legacy:
                logger.info("Falling back to legacy sell logic")
                return self._legacy_sell_decision(ticker, current_price, portfolio_holdings, analysis_data)
            else:
                return self._error_decision(str(e))
    
    def _build_position_metrics(
        self,
        ticker: str,
        current_price: float,
        portfolio_holdings: Dict,
        price_history: Optional[pd.DataFrame]
    ) -> PositionMetrics:
        """Build position metrics from portfolio data"""
        
        holding = portfolio_holdings[ticker]
        entry_price = holding.get("avg_price", current_price)
        quantity = holding.get("qty", 0)
        
        # Calculate P&L
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Calculate days held (estimate if not available)
        days_held = holding.get("days_held", 1)
        
        # Calculate highest/lowest prices since entry
        highest_price = current_price
        lowest_price = current_price
        volatility = 0.02  # Default volatility
        
        if price_history is not None and len(price_history) > 1:
            # Use actual price history if available
            recent_prices = price_history['Close'].tail(30)  # Last 30 days
            highest_price = recent_prices.max()
            lowest_price = recent_prices.min()
            
            # Calculate volatility
            returns = recent_prices.pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std() * (252 ** 0.5)  # Annualized
        
        return PositionMetrics(
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            days_held=days_held,
            highest_price_since_entry=highest_price,
            lowest_price_since_entry=lowest_price,
            volatility=volatility
        )
    
    def _build_market_context(self, analysis_data: Dict, price_history: Optional[pd.DataFrame]):
        """Build market context from analysis data"""
        
        if price_history is not None and len(price_history) > 50:
            # Use professional market analysis
            return self.market_analyzer.analyze_market_context(price_history)
        else:
            # Fallback to basic context from analysis data
            from .professional_sell_logic import MarketTrend, MarketContext
            
            # Extract basic trend info
            technical = analysis_data.get("technical_indicators", {})
            sma_20 = technical.get("sma_20", 0)
            sma_50 = technical.get("sma_50", 0)
            current_price = technical.get("current_price", 0)
            
            # Simple trend detection
            if current_price > sma_20 > sma_50:
                trend = MarketTrend.UPTREND
                trend_strength = 0.6
            elif current_price < sma_20 < sma_50:
                trend = MarketTrend.DOWNTREND
                trend_strength = 0.6
            else:
                trend = MarketTrend.SIDEWAYS
                trend_strength = 0.3
            
            return MarketContext(
                trend=trend,
                trend_strength=trend_strength,
                volatility_regime="normal",
                market_stress=0.3,
                sector_performance=0.0,
                volume_profile=0.5
            )
    
    def _convert_to_legacy_format(self, sell_decision: SellDecision, position_metrics: PositionMetrics) -> Dict:
        """Convert professional sell decision to legacy format"""
        
        if not sell_decision.should_sell:
            return {
                "action": "hold",
                "ticker": "",
                "qty": 0,
                "price": position_metrics.current_price,
                "stop_loss": sell_decision.stop_loss_price,
                "take_profit": sell_decision.take_profit_price,
                "success": True,
                "confidence_score": sell_decision.confidence,
                "signals": len(sell_decision.signals_triggered),
                "reason": "professional_hold",
                "professional_reasoning": sell_decision.reasoning
            }
        
        return {
            "action": "sell",
            "ticker": "",  # Will be set by calling module
            "qty": sell_decision.sell_quantity,
            "price": position_metrics.current_price,
            "stop_loss": sell_decision.stop_loss_price,
            "take_profit": sell_decision.take_profit_price,
            "success": True,  # Will be set by execution
            "confidence_score": sell_decision.confidence,
            "signals": len(sell_decision.signals_triggered),
            "reason": sell_decision.reason.value,
            "professional_reasoning": sell_decision.reasoning,
            "sell_percentage": sell_decision.sell_percentage,
            "urgency": sell_decision.urgency
        }
    
    def _no_position_decision(self) -> Dict:
        """Return decision when no position exists"""
        return {
            "action": "hold",
            "ticker": "",
            "qty": 0,
            "price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": True,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "no_position",
            "professional_reasoning": "No position to sell"
        }
    
    def _legacy_sell_decision(self, ticker: str, current_price: float, portfolio_holdings: Dict, analysis_data: Dict) -> Dict:
        """Fallback to legacy sell logic"""
        return {
            "action": "hold",
            "ticker": ticker,
            "qty": 0,
            "price": current_price,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": True,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "legacy_fallback",
            "professional_reasoning": "Using legacy sell logic"
        }
    
    def _error_decision(self, error_msg: str) -> Dict:
        """Return decision when error occurs"""
        return {
            "action": "hold",
            "ticker": "",
            "qty": 0,
            "price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "success": False,
            "confidence_score": 0.0,
            "signals": 0,
            "reason": "error",
            "professional_reasoning": f"Error: {error_msg}"
        }
    
    def get_professional_config(self) -> Dict:
        """Get professional sell logic configuration"""
        return {
            "min_sell_signals": 2,
            "min_sell_confidence": 0.65,
            "min_weighted_sell_score": 0.40,
            "stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.03,
            "profit_protection_threshold": 0.05,
            "partial_exit_threshold": 0.50,
            "full_exit_threshold": 0.75,
            "uptrend_sell_multiplier": 1.5,
            "downtrend_sell_multiplier": 0.8,
            "enable_professional_sell_logic": True,
            "fallback_to_legacy_sell": False
        }
