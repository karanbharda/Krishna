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
    """
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize professional components
        self.buy_logic = ProfessionalBuyLogic(config)
        self.market_analyzer = MarketContextAnalyzer(config)
        
        # Integration settings
        self.enable_professional_logic = config.get("enable_professional_buy_logic", True)
        self.fallback_to_legacy = config.get("fallback_to_legacy_buy", False)  # Changed to False for consistency
        
        logger.info("Professional Buy Integration initialized")
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json"""
        self.buy_logic.refresh_dynamic_config()
    
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
        # Check if buy is disabled by configuration
        enable_buy = str(os.getenv("ENABLE_BUY", "true")).lower() not in ("false", "0", "no", "off")
        if not enable_buy:
            logger.info(f"❌ BUY DISABLED: Buy signals globally disabled by configuration (ENABLE_BUY=false) for {ticker}")
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
                "reason": "buy_disabled",
                "professional_reasoning": "Buy functionality disabled by configuration"
            }
        
        try:
            logger.info(f"=== STARTING PROFESSIONAL BUY EVALUATION FOR {ticker} ===")
            
            # Build stock metrics
            stock_metrics = self._build_stock_metrics(
                ticker, current_price, analysis_data, price_history
            )
            
            # Build market context
            market_context = self._build_market_context(analysis_data, price_history)
            
            # Extract analysis components
            technical_analysis = analysis_data.get("technical_indicators", {})
            sentiment_analysis = analysis_data.get("sentiment_analysis", {})
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
            
            # Log detailed decision information
            logger.info(f"Professional Buy Decision for {ticker}:")
            logger.info(f"  Should Buy: {buy_decision.should_buy}")
            logger.info(f"  Confidence: {buy_decision.confidence:.3f}")
            logger.info(f"  Buy Percentage: {buy_decision.buy_percentage:.3f}")
            logger.info(f"  Reason: {buy_decision.reason}")
            logger.info(f"  Signals Triggered: {len(buy_decision.signals_triggered)}")
            logger.info(f"  Reasoning: {buy_decision.reasoning}")
            
            # Convert to legacy format
            result = self._convert_to_legacy_format(buy_decision, stock_metrics, portfolio_context)
            
            # Add ticker information
            result["ticker"] = ticker
            
            # Log the final result
            if result['action'] == 'buy':
                logger.info(f"🟢 BUY DECISION FINALIZED: {result['qty']} shares of {ticker}")
                logger.info(f"   Price: ₹{result['price']:.2f}")
                logger.info(f"   Stop Loss: ₹{result['stop_loss']:.2f}")
                logger.info(f"   Take Profit: ₹{result['take_profit']:.2f}")
                logger.info(f"   Confidence: {result['confidence_score']:.3f}")
                logger.info(f"   Reason: {result['reason']}")
                logger.info(f"   Professional Reasoning: {result['professional_reasoning']}")
            elif result['action'] == 'hold':
                logger.info(f"🟡 HOLD DECISION: {ticker}")
                logger.info(f"   Reason: {result['reason']}")
                logger.info(f"   Professional Reasoning: {result['professional_reasoning']}")
                logger.info(f"   Confidence: {result['confidence_score']:.3f}")
            else:
                logger.info(f"🔴 UNEXPECTED DECISION: {result['action']} for {ticker}")
            
            logger.info(f"=== END PROFESSIONAL BUY EVALUATION FOR {ticker} ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional buy evaluation for {ticker}: {e}")
            logger.exception("Full traceback:")
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
                atr_value = price_history['ATR'].iloc[-1] if not pd.isna(price_history['ATR'].iloc[-1]) else 0.0
                atr = safe_float(atr_value, 0.0)
        
        # Extract key technical indicators with defaults and type conversion
        # Helper function to safely convert values to float
        def safe_float(value, default=0.0):
            if isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return default
        
        # Extract key technical indicators with defaults and type conversion
        rsi = safe_float(technical.get("rsi", 50), 50)
        macd = safe_float(technical.get("macd", 0), 0)
        macd_signal = safe_float(technical.get("macd_signal", 0), 0)
        sma_20 = safe_float(technical.get("sma_20", current_price), current_price)
        sma_50 = safe_float(technical.get("sma_50", current_price), current_price)
        sma_200 = safe_float(technical.get("sma_200", current_price), current_price)
        support_level = safe_float(technical.get("support_level", current_price * 0.95), current_price * 0.95)
        resistance_level = safe_float(technical.get("resistance_level", current_price * 1.05), current_price * 1.05)
        volume_ratio = safe_float(technical.get("volume_ratio", 1.0), 1.0)
        
        # Extract fundamental metrics with defaults and type conversion
        fundamental_data = analysis_data.get("fundamental_analysis", {}) or analysis_data.get("fundamental_metrics", {})
        price_to_book = safe_float(fundamental_data.get("price_to_book", 2.0), 2.0)
        price_to_earnings = safe_float(fundamental_data.get("price_to_earnings", 15.0), 15.0)
        earnings_growth = safe_float(fundamental_data.get("earnings_growth", 0.05), 0.05)  # Default 5% growth
        return_on_equity = safe_float(fundamental_data.get("return_on_equity", 0.10), 0.10)  # Default 10% ROE
        free_cash_flow_yield = safe_float(fundamental_data.get("free_cash_flow_yield", 0.05), 0.05)  # Default 5% FCF yield
        debt_to_equity = safe_float(fundamental_data.get("debt_to_equity", 0.5), 0.5)  # Default 0.5 debt-to-equity
        dividend_yield = safe_float(fundamental_data.get("dividend_yield", 0.0), 0.0)  # Default 0% dividend yield
        payout_ratio = safe_float(fundamental_data.get("payout_ratio", 0.0), 0.0)  # Default 0% payout ratio
        earnings_quality = safe_float(fundamental_data.get("earnings_quality", 0.5), 0.5)  # Default 50% earnings quality
        insider_ownership = safe_float(fundamental_data.get("insider_ownership", 0.0), 0.0)  # Default 0% insider ownership
        sector_pe = safe_float(fundamental_data.get("sector_pe", 20.0), 20.0)  # Default sector P/E of 20
        
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
            price_to_earnings=price_to_earnings,
            earnings_growth=earnings_growth,
            return_on_equity=return_on_equity,
            free_cash_flow_yield=free_cash_flow_yield,
            debt_to_equity=debt_to_equity,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            earnings_quality=earnings_quality,
            insider_ownership=insider_ownership,
            sector_pe=sector_pe
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
    
    def _convert_to_legacy_format(self, buy_decision: BuyDecision, stock_metrics: StockMetrics, portfolio_context: Dict = None) -> Dict:
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
        qty = 1  # Default to 1 share
        if portfolio_context and "available_cash" in portfolio_context and "total_value" in portfolio_context:
            available_cash = portfolio_context.get("available_cash", 0)
            total_value = portfolio_context.get("total_value", 0)
            current_price = stock_metrics.current_price
            
            # Calculate target position value based on position scale
            position_scale = buy_decision.buy_percentage
            target_position_value = total_value * position_scale * 0.1  # Limit to 10% of portfolio for safety
            
            # Ensure we don't exceed available cash
            target_position_value = min(target_position_value, available_cash)
            
            # Convert to quantity
            if current_price > 0:
                calculated_qty = int(target_position_value / current_price)
                qty = max(1, calculated_qty)  # Minimum 1 share
        
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