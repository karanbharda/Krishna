"""
Professional Sell Logic Integration
Integrates the professional sell logic with existing trading modules
"""

import logging
import os
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from .professional_sell_logic import (
    ProfessionalSellLogic, PositionMetrics, SellDecision, SellReason
)
from .market_context_analyzer import MarketContextAnalyzer

# PRODUCTION ENHANCEMENT: Import signal tracker
from utils.signal_tracker import get_signal_tracker

logger = logging.getLogger(__name__)

class ProfessionalSellIntegration:
    """
    Integration layer for professional sell logic
    """
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize professional components
        self.sell_logic = ProfessionalSellLogic(config)
        
        # PRODUCTION ENHANCEMENT: Initialize signal tracker
        self.signal_tracker = get_signal_tracker()
        
        # Integration settings
        self.enable_professional_logic = config.get("enable_professional_sell_logic", True)
        self.fallback_to_legacy = config.get("fallback_to_legacy_sell", False)
        
        logger.info("Professional Sell Integration initialized")
    
    def refresh_dynamic_config(self):
        """Refresh dynamic configuration from live_config.json"""
        self.sell_logic.refresh_dynamic_config()
    
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
        # Check if sell is disabled by configuration
        enable_sell = str(os.getenv("ENABLE_SELL", "true")).lower() not in ("false", "0", "no", "off")
        if not enable_sell:
            logger.info(f"❌ SELL DISABLED: Sell signals globally disabled by configuration (ENABLE_SELL=false) for {ticker}")
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
                "reason": "sell_disabled",
                "professional_reasoning": "Sell functionality disabled by configuration"
            }
        
        try:
            logger.info(f"=== STARTING PROFESSIONAL SELL EVALUATION FOR {ticker} ===")
            
            # Check if we have a position
            if ticker not in portfolio_holdings:
                logger.info(f"No position found for {ticker}, returning no position decision")
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
            
            # Log detailed decision information
            logger.info(f"Professional Sell Decision for {ticker}:")
            logger.info(f"  Should Sell: {sell_decision.should_sell}")
            logger.info(f"  Confidence: {sell_decision.confidence:.3f}")
            logger.info(f"  Sell Percentage: {sell_decision.sell_percentage:.3f}")
            logger.info(f"  Reason: {sell_decision.reason}")
            logger.info(f"  Signals Triggered: {len(sell_decision.signals_triggered)}")
            logger.info(f"  Reasoning: {sell_decision.reasoning}")
            
            # Convert to legacy format
            result = self._convert_to_legacy_format(sell_decision, position_metrics)
            
            # Log the final result
            if result['action'] == 'sell':
                logger.info(f"🔴 SELL DECISION FINALIZED: {result['qty']} shares of {ticker}")
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
            
            logger.info(f"=== END PROFESSIONAL SELL EVALUATION FOR {ticker} ===")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in professional sell evaluation for {ticker}: {e}")
            logger.exception("Full traceback:")
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
        
        holding = portfolio_holdings[ticker]
        entry_price = safe_float(holding.get("avg_price", current_price), current_price)
        quantity = safe_float(holding.get("qty", 0), 0)
        
        # Calculate P&L
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Calculate days held (estimate if not available)
        days_held = safe_float(holding.get("days_held", 1), 1)
        
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
        
        # Fetch stop-loss and target price from database
        db_stop_loss, db_target_price = self._fetch_db_stop_loss_target(ticker)
        
        logger.info(f"Position Metrics for {ticker}:")
        logger.info(f"  Entry Price: {entry_price:.2f}")
        logger.info(f"  Current Price: {current_price:.2f}")
        logger.info(f"  DB Stop-Loss: {db_stop_loss if db_stop_loss else 'Not set'}")
        logger.info(f"  DB Target Price: {db_target_price if db_target_price else 'Not set'}")
        
        return PositionMetrics(
            entry_price=entry_price,
            current_price=current_price,
            quantity=quantity,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            days_held=days_held,
            highest_price_since_entry=highest_price,
            lowest_price_since_entry=lowest_price,
            volatility=volatility,
            db_stop_loss=db_stop_loss,
            db_target_price=db_target_price
        )
    
    def _fetch_db_stop_loss_target(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch stop-loss and target price from database for the given ticker"""
        try:
            from ..db.database import DatabaseManager
            
            db_manager = DatabaseManager()
            session = db_manager.Session()
            
            try:
                # Query the most recent buy trade for this ticker
                from ..db.database import Trade
                latest_buy = session.query(Trade).filter(
                    Trade.ticker == ticker,
                    Trade.action == 'buy'
                ).order_by(Trade.timestamp.desc()).first()
                
                # Helper function to safely convert values to float
                def safe_float(value, default=None):
                    if isinstance(value, str):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default
                    elif isinstance(value, (int, float)):
                        return float(value) if value > 0 else default
                    else:
                        return default
                
                if latest_buy:
                    stop_loss = safe_float(latest_buy.stop_loss)
                    target_price = safe_float(latest_buy.take_profit)
                    
                    if stop_loss or target_price:
                        logger.info(f"📊 Database values retrieved for {ticker}: Stop-Loss={stop_loss}, Target={target_price}")
                    
                    return stop_loss, target_price
                else:
                    logger.debug(f"No buy trade found in database for {ticker}")
                    return None, None
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.warning(f"Failed to fetch database stop-loss/target for {ticker}: {e}")
            return None, None
    
    def _fetch_dhan_portfolio_data(self) -> Tuple[float, float, float]:
        """Fetch dynamic portfolio data from Dhan API"""
        try:
            # Import Dhan client if available
            try:
                from ..dhan_client import DhanAPIClient
                from ..config.environment_manager import EnvironmentManager
                
                # Get Dhan credentials
                env_manager = EnvironmentManager()
                dhan_config = env_manager.get_dhan_config()
                
                if dhan_config and dhan_config.get('client_id') and dhan_config.get('access_token'):
                    dhan_client = DhanAPIClient(
                        client_id=dhan_config['client_id'],
                        access_token=dhan_config['access_token']
                    )
                    
                    # Get funds (available cash)
                    funds_data = dhan_client.get_funds()
                    available_cash = funds_data.get('availabelBalance', 0.0)  # Note: typo in Dhan API
                    
                    # Get holdings for P&L calculation
                    holdings = dhan_client.get_holdings()
                    
                    # Calculate unrealized P&L from holdings
                    unrealized_pnl = 0.0
                    for holding in holdings:
                        current_price = holding.get('lastPrice', 0.0)
                        avg_price = holding.get('avgPrice', 0.0)
                        quantity = holding.get('quantity', 0)
                        
                        if current_price > 0 and avg_price > 0:
                            pnl = (current_price - avg_price) * quantity
                            unrealized_pnl += pnl
                    
                    # For realized P&L, we'd need to get it from order history or positions
                    # For now, we'll use 0 as Dhan doesn't provide realized P&L directly in funds
                    realized_pnl = 0.0
                    
                    logger.info(f"📊 Dhan Portfolio Data - Cash: ₹{available_cash:.2f}, Unrealized P&L: ₹{unrealized_pnl:.2f}")
                    
                    return available_cash, realized_pnl, unrealized_pnl
                else:
                    logger.warning("Dhan credentials not available for dynamic portfolio data")
                    return 0.0, 0.0, 0.0
                    
            except ImportError as e:
                logger.warning(f"Dhan client not available: {e}")
                return 0.0, 0.0, 0.0
            except Exception as e:
                logger.error(f"Error fetching Dhan portfolio data: {e}")
                return 0.0, 0.0, 0.0
                
        except Exception as e:
            logger.error(f"Failed to fetch Dhan portfolio data: {e}")
            return 0.0, 0.0, 0.0
    
    def _build_market_context(self, analysis_data: Dict, price_history: Optional[pd.DataFrame]):
        """Build market context from analysis data"""
        
        if price_history is not None and len(price_history) > 50:
            # Use professional market analysis
            return self.market_analyzer.analyze_market_context(price_history)
        else:
            # Fallback to basic context from analysis data
            from .professional_sell_logic import MarketTrend, MarketContext
            
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
            
            # Extract basic trend info
            technical = analysis_data.get("technical_indicators", {})
            sma_20 = safe_float(technical.get("sma_20", 0), 0)
            sma_50 = safe_float(technical.get("sma_50", 0), 0)
            current_price = safe_float(technical.get("current_price", 0), 0)
            
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
        
        # PRODUCTION ENHANCEMENT: Track signals for continuous learning
        self._track_signals(sell_decision)
        
        if not sell_decision.should_sell:
            # Use stored database values if available, otherwise use calculated values
            stop_loss = position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else sell_decision.stop_loss_price
            take_profit = position_metrics.db_target_price if position_metrics.db_target_price is not None else sell_decision.take_profit_price
            
            return {
                "action": "hold",
                "ticker": "",
                "qty": 0,
                "price": position_metrics.current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": True,
                "confidence_score": sell_decision.confidence,
                "signals": len(sell_decision.signals_triggered),
                "reason": "professional_hold",
                "professional_reasoning": sell_decision.reasoning
            }
        
        # Use stored database values if available, otherwise use calculated values
        stop_loss = position_metrics.db_stop_loss if position_metrics.db_stop_loss is not None else sell_decision.stop_loss_price
        take_profit = position_metrics.db_target_price if position_metrics.db_target_price is not None else sell_decision.take_profit_price
        
        return {
            "action": "sell",
            "ticker": "",  # Will be set by calling module
            "qty": sell_decision.sell_quantity,
            "price": position_metrics.current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
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
    
    def _track_signals(self, sell_decision: SellDecision):
        """
        PRODUCTION ENHANCEMENT: Track signals for continuous learning
        """
        try:
            # Record each triggered signal
            for signal in sell_decision.signals_triggered:
                if signal.triggered:
                    signal_data = {
                        'symbol': '',  # Will be filled in by calling function
                        'signal_type': signal.category.lower() if signal.category else 'unknown',
                        'signal_name': signal.name,
                        'signal_strength': signal.strength,
                        'signal_confidence': signal.confidence,
                        'market_regime': 'unknown',  # Would be filled with actual regime
                        'liquidity_score': 0.5,  # Default score
                        'volatility_regime': 'normal',  # Would be filled with actual regime
                        'additional_metrics': {
                            'reasoning': signal.reasoning,
                            'weight': signal.weight
                        }
                    }
                    
                    # Record the signal
                    self.signal_tracker.record_signal(signal_data)
            
            logger.debug(f"Tracked {len(sell_decision.signals_triggered)} signals for continuous learning")
            
        except Exception as e:
            logger.warning(f"Error tracking signals: {e}")
    
    def get_dynamic_portfolio_data(self) -> Dict:
        """Get dynamic portfolio data from Dhan API for live trading"""
        try:
            # Fetch dynamic data from Dhan
            available_cash, realized_pnl, unrealized_pnl = self._fetch_dhan_portfolio_data()
            
            return {
                "available_cash": available_cash,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_value": available_cash + unrealized_pnl,  # Cash + unrealized positions
                "is_dynamic": True
            }
        except Exception as e:
            logger.error(f"Failed to get dynamic portfolio data: {e}")
            # Fallback to database values if Dhan fails
            return {
                "available_cash": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_value": 0.0,
                "is_dynamic": False
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
            "professional_reasoning": f"Error: {error_msg}",
            "error_msg": error_msg
        }