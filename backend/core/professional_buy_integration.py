"""
Professional Buy Logic Integration
Integrates the professional buy logic with existing trading modules
"""

import logging
import os
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .professional_buy_logic import (
    ProfessionalBuyLogic, StockMetrics, BuyDecision, BuyReason
)
# Import MarketTrend and MarketContext from professional_sell_logic since they're already defined there
from .professional_sell_logic import MarketTrend, MarketContext
from .market_context_analyzer import MarketContextAnalyzer

logger = logging.getLogger(__name__)

class ProfessionalBuyIntegration:
    """
    Integration layer for professional buy logic
    Bridges the gap between existing trading modules and new professional logic
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize professional components
        self.buy_logic = ProfessionalBuyLogic(config)
        self.market_analyzer = MarketContextAnalyzer(config)
        
        # Integration settings
        self.enable_professional_logic = config.get("enable_professional_buy_logic", True)
        self.fallback_to_legacy = config.get("fallback_to_legacy_buy", True)  # Changed to True for compatibility
        
        logger.info("Professional Buy Integration initialized")
    
    def evaluate_professional_buy(
        self,
        ticker: str,
        current_price: float,
        analysis_data: Dict,
        price_history: Optional[pd.DataFrame] = None,
        portfolio_context: Dict = None
    ) -> Dict:
        """
        Main integration point for professional buy evaluation
        Returns a decision compatible with existing trading modules
        """
        try:
            # Build stock metrics
            stock_metrics = self._build_stock_metrics(
                ticker, current_price, analysis_data, price_history
            )
            
            # Build market context
            market_context = self._build_market_context(analysis_data, price_history)
            
            # Extract analysis components
            technical_analysis = analysis_data.get("technical_indicators", {})
            sentiment_analysis = analysis_data.get("sentiment", {})
            ml_analysis = analysis_data.get("ml_analysis", {})
            
            # Use empty portfolio context if not provided
            if portfolio_context is None:
                portfolio_context = {}
            
            # Get professional buy decision
            buy_decision = self.buy_logic.evaluate_buy_decision(
                ticker=ticker,
                stock_metrics=stock_metrics,
                market_context=market_context,
                technical_analysis=technical_analysis,
                sentiment_analysis=sentiment_analysis,
                ml_analysis=ml_analysis,
                portfolio_context=portfolio_context
            )
            
            # Convert to legacy format
            result = self._convert_to_legacy_format(buy_decision, stock_metrics)
            
            # Add ticker information
            result["ticker"] = ticker
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional buy evaluation: {e}")
            return self._error_decision(str(e))
    
    def _build_stock_metrics(
        self,
        ticker: str,
        current_price: float,
        analysis_data: Dict,
        price_history: Optional[pd.DataFrame]
    ) -> StockMetrics:
        """Build stock metrics from analysis data"""
        
        # Extract technical indicators
        technical = analysis_data.get("technical_indicators", {})
        
        # Calculate volatility from price history if available
        volatility = 0.02  # Default volatility
        atr = 0.0
        
        if price_history is not None and len(price_history) > 1:
            # Calculate volatility
            returns = price_history['Close'].pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std() * (252 ** 0.5)  # Annualized
            
            # Calculate ATR if available
            if 'ATR' in price_history.columns:
                atr = price_history['ATR'].iloc[-1] if not pd.isna(price_history['ATR'].iloc[-1]) else 0.0
        
        # Extract key technical indicators with defaults
        rsi = technical.get("rsi", 50)
        macd = technical.get("macd", 0)
        macd_signal = technical.get("macd_signal", 0)
        sma_20 = technical.get("sma_20", current_price)
        sma_50 = technical.get("sma_50", current_price)
        sma_200 = technical.get("sma_200", current_price)
        support_level = technical.get("support_level", current_price * 0.95)
        resistance_level = technical.get("resistance_level", current_price * 1.05)
        volume_ratio = technical.get("volume_ratio", 1.0)
        
        # Extract fundamental metrics with defaults
        price_to_book = analysis_data.get("fundamental_metrics", {}).get("price_to_book", 2.0)
        price_to_earnings = analysis_data.get("fundamental_metrics", {}).get("price_to_earnings", 15.0)
        
        return StockMetrics(
            current_price=current_price,
            entry_price=current_price,
            quantity=0,  # Not relevant for buy decision
            volatility=volatility,
            atr=atr,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            support_level=support_level,
            resistance_level=resistance_level,
            volume_ratio=volume_ratio,
            price_to_book=price_to_book,
            price_to_earnings=price_to_earnings
        )
    
    def _build_market_context(self, analysis_data: Dict, price_history: Optional[pd.DataFrame]):
        """Build market context from analysis data"""
        
        if price_history is not None and len(price_history) > 50:
            # Use professional market analysis
            return self.market_analyzer.analyze_market_context(price_history)
        else:
            # Fallback to basic context from analysis data
            from .professional_buy_logic import MarketTrend, MarketContext
            
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
    
    def _convert_to_legacy_format(self, buy_decision: BuyDecision, stock_metrics: StockMetrics) -> Dict:
        """Convert professional buy decision to legacy format"""
        
        if not buy_decision.should_buy:
            return {
                "action": "hold",
                "ticker": "",
                "qty": 0,
                "price": stock_metrics.current_price,
                "stop_loss": buy_decision.stop_loss_price,
                "take_profit": buy_decision.take_profit_price,
                "success": True,
                "confidence_score": buy_decision.confidence,
                "signals": len(buy_decision.signals_triggered),
                "reason": "professional_hold",
                "professional_reasoning": buy_decision.reasoning
            }
        
        # Calculate quantity based on portfolio context if available
        qty = buy_decision.buy_quantity
        if qty <= 0:
            # Instead of defaulting to a small position, return hold decision
            return {
                "action": "hold",
                "ticker": "",
                "qty": 0,
                "price": stock_metrics.current_price,
                "stop_loss": buy_decision.stop_loss_price,
                "take_profit": buy_decision.take_profit_price,
                "success": True,
                "confidence_score": buy_decision.confidence,
                "signals": len(buy_decision.signals_triggered),
                "reason": "no_valid_quantity",
                "professional_reasoning": f"Professional buy logic recommends buying but calculated quantity is {qty} which is invalid. No buy action taken."
            }
        
        return {
            "action": "buy",
            "ticker": "",  # Will be set by calling module
            "qty": qty,
            "price": stock_metrics.current_price,
            "stop_loss": buy_decision.stop_loss_price,
            "take_profit": buy_decision.take_profit_price,
            "success": True,  # Will be set by execution
            "confidence_score": buy_decision.confidence,
            "signals": len(buy_decision.signals_triggered),
            "reason": buy_decision.reason.value if hasattr(buy_decision.reason, 'value') else str(buy_decision.reason),
            "professional_reasoning": buy_decision.reasoning,
            "buy_percentage": buy_decision.buy_percentage,
            "urgency": buy_decision.urgency
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
        """Get professional buy logic configuration"""
        return {
            "min_buy_signals": 2,
            "min_buy_confidence": 0.65,
            "min_weighted_buy_score": 0.40,
            "entry_buffer_pct": 0.01,
            "buy_stop_loss_pct": 0.05,
            "take_profit_ratio": 2.0,
            "partial_entry_threshold": 0.50,
            "full_entry_threshold": 0.75,
            "downtrend_buy_multiplier": 0.7,
            "uptrend_buy_multiplier": 1.2,
            "enable_professional_buy_logic": True,
            "fallback_to_legacy_buy": False
        }