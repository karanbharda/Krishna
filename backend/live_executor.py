#!/usr/bin/env python3
"""
Live Trading Executor for Dhan API Integration
Handles real order execution, portfolio management, and risk controls
Integrates with SQLite database for consistent portfolio calculations
"""

import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dhan_client import DhanAPIClient

# Import database portfolio manager
try:
    from portfolio_manager import DualPortfolioManager
    from db.database import Portfolio, Holding, Trade
except ImportError:
    from .portfolio_manager import DualPortfolioManager
    from .db.database import Portfolio, Holding, Trade

logger = logging.getLogger(__name__)

class LiveTradingExecutor:
    """Execute live trades through Dhan API with database integration and risk management"""
    
    def __init__(self, portfolio_manager: DualPortfolioManager, config: Dict):
        self.portfolio_manager = portfolio_manager
        self.config = config
        self.stop_loss_pct = config.get("stop_loss_pct", 0.05)
        self.max_capital_per_trade = config.get("max_capital_per_trade", 0.25)
        self.max_trade_limit = config.get("max_trade_limit", 150)
        
        # Initialize all attributes before any method calls
        # Trading state
        self.pending_orders = {}
        self.executed_orders = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Risk management
        self.max_daily_loss = config.get("max_daily_loss", 0.05)  # 5% of portfolio
        self.daily_pnl = 0.0

        # Rate limiting for sync operations
        self.last_sync_time = 0
        self.min_sync_interval = 60  # Minimum 60 seconds between syncs

        # Global sell enable flag (env or config)
        self.enable_sell = str(self.config.get("enable_sell", os.getenv("ENABLE_SELL", "true"))).lower() not in ("false", "0", "no", "off")
        
        # Ensure we're in live mode
        if self.portfolio_manager.current_mode != "live":
            self.portfolio_manager.switch_mode("live")
        
        # Initialize Dhan client
        self.dhan_client = DhanAPIClient(
            client_id=config.get("dhan_client_id"),
            access_token=config.get("dhan_access_token")
        )
        
        # Sync portfolio with Dhan account on initialization
        try:
            if not self.sync_portfolio_with_dhan():
                logger.error("Failed to sync portfolio with Dhan account during initialization")
            else:
                logger.info("Successfully synced portfolio with Dhan account")
        except Exception as e:
            logger.error(f"Failed to sync portfolio with Dhan: {e}")

        if not self.enable_sell:
            logger.warning("Sell operations are DISABLED by configuration (ENABLE_SELL=false)")

        logger.info("Live Trading Executor initialized")
    
    def validate_connection(self) -> bool:
        """Validate Dhan API connection"""
        try:
            return self.dhan_client.validate_connection()
        except Exception as e:
            logger.error(f"Failed to validate Dhan connection: {e}")
            return False
    
    def sync_portfolio_with_dhan(self) -> bool:
        """Sync local portfolio with actual Dhan account using database"""
        try:
            # Rate limiting - avoid excessive sync calls
            current_time = time.time()
            if current_time - self.last_sync_time < self.min_sync_interval:
                logger.debug(f"Sync rate limited - last sync {current_time - self.last_sync_time:.1f}s ago")
                return True  # Return success to avoid errors, but skip actual sync

            logger.debug("Syncing portfolio with Dhan account...")
            
            # Get account funds
            funds = self.dhan_client.get_funds()
            # Normalize across possible Dhan fund keys
            def _get_cash(d: Dict[str, Any]) -> float:
                for key in ("availablecash", "availabelBalance", "availableBalance", "netAvailableMargin", "netAvailableCash"):
                    if key in d and d.get(key) is not None:
                        try:
                            return float(d.get(key))
                        except Exception:
                            continue
                # Fallback: if there is a numeric field resembling available funds
                for k, v in d.items():
                    if isinstance(v, (int, float)) and "avail" in k.lower() and "cash" in k.lower():
                        return float(v)
                return 0.0

            available_cash = _get_cash(funds)
            logger.debug(f"Dhan Account Balance: Rs.{available_cash:.2f}")
            
            # Get current holdings from Dhan
            holdings = self.dhan_client.get_holdings()
            logger.debug(f"Dhan Holdings: {len(holdings)} positions")
            
            # Update database portfolio
            session = None
            try:
                session = self.portfolio_manager.db.Session()
                portfolio = session.query(Portfolio).filter_by(mode='live').first()
                if not portfolio:
                    # Create live portfolio if it doesn't exist
                    portfolio = Portfolio(
                        mode='live',
                        cash=available_cash,
                        starting_balance=available_cash,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        last_updated=datetime.now()
                    )
                    session.add(portfolio)
                    session.flush()
                else:
                    # Update existing portfolio cash
                    portfolio.cash = available_cash
                    portfolio.last_updated = datetime.now()
                
                # Clear existing holdings and add current ones from Dhan
                session.query(Holding).filter_by(portfolio_id=portfolio.id).delete()
                
                total_holdings_value = 0
                for holding in holdings:
                    symbol = holding.get("tradingSymbol", "")
                    quantity = int(holding.get("totalQty", 0))
                    avg_price = float(holding.get("avgCostPrice", 0))
                    current_price = float(holding.get("ltp", avg_price))  # Use LTP if available
                    
                    if quantity > 0 and symbol:
                        db_holding = Holding(
                            portfolio_id=portfolio.id,
                            ticker=symbol,
                            quantity=quantity,
                            avg_price=avg_price,
                            last_price=current_price
                        )
                        session.add(db_holding)
                        total_holdings_value += quantity * current_price
                
                # Update portfolio metrics
                portfolio.unrealized_pnl = total_holdings_value - sum(
                    h.quantity * h.avg_price for h in 
                    session.query(Holding).filter_by(portfolio_id=portfolio.id).all()
                )
                
                session.commit()
                
                # Update current portfolio reference
                self.portfolio_manager.current_portfolio = portfolio
                
                logger.debug(f"Portfolio synced - Cash: Rs.{available_cash:.2f}, Holdings: Rs.{total_holdings_value:.2f}")
                self.last_sync_time = time.time()
                return True
                
            except Exception as e:
                if session:
                    session.rollback()
                raise e
            finally:
                if session:
                    session.close()
            
        except Exception as e:
            logger.error(f"Failed to sync portfolio with Dhan: {e}")
            return False
    
    def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price using Fyers API for reliable data"""
        try:
            # Use Fyers for real-time price data (more reliable than Dhan quotes)
            try:
                # Import Fyers utilities
                from testindia import get_stock_data_fyers_or_yf
                
                # Get recent price data from Fyers
                hist = get_stock_data_fyers_or_yf(symbol, period="1d")
                
                if hist is not None and not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    if current_price > 0:
                        logger.debug(f"Got price from Fyers for {symbol}: Rs.{current_price:.2f}")
                        return current_price
                    else:
                        logger.warning(f"Invalid price from Fyers for {symbol}: {current_price}")
                else:
                    logger.warning(f"No price data from Fyers for {symbol}")
                    
            except Exception as fyers_error:
                logger.warning(f"Fyers price fetch failed for {symbol}: {fyers_error}")
            
            # Secondary fallback: Try Dhan API if Fyers fails
            try:
                quote = self.dhan_client.get_quote(symbol)
                price = float(quote.get("ltp", 0))
                
                if price > 0:
                    logger.debug(f"Got fallback price from Dhan API for {symbol}: Rs.{price:.2f}")
                    return price
                    
            except Exception as dhan_error:
                logger.warning(f"Dhan API fallback failed for {symbol}: {dhan_error}")
            
            logger.error(f"All price sources failed for {symbol}")
            return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              signal_strength: float) -> int:
        """Calculate position size based on risk management using database portfolio"""
        session = None
        try:
            # Get portfolio summary from database with fresh session
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if not portfolio:
                logger.warning(f"No live portfolio found for position sizing")
                return 0
                
            available_cash = portfolio.cash
            total_value = available_cash  # Simple calculation for now
            
            if available_cash <= 0:
                logger.warning(f"No available cash for {symbol}: Rs.{available_cash:.2f}")
                return 0
            
            # Maximum capital per trade
            max_trade_amount = available_cash * self.max_capital_per_trade
            
            # Adjust based on signal strength (0.0 to 1.0)
            adjusted_amount = max_trade_amount * min(signal_strength, 1.0)
            
            # Calculate quantity
            quantity = int(adjusted_amount / current_price)
            
            # Minimum quantity check
            if quantity < 1:
                logger.warning(f"Calculated quantity too small for {symbol}: {quantity}")
                return 0
            
            # Maximum position size check (don't exceed 50% of available cash)
            max_quantity = int((available_cash * 0.5) / current_price)
            quantity = min(quantity, max_quantity)
            
            logger.info(f"Position size for {symbol}: {quantity} shares (Rs.{quantity * current_price:.2f})")
            return quantity
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return 0
        finally:
            if session:
                session.close()
    
    def execute_buy_order(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute a buy order through Dhan API with database recording"""
        try:
            # Check daily trade limit
            if self._check_daily_limits():
                return {"success": False, "message": "Daily trade limit exceeded"}
            
            # Check market status
            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}
            
            # Get current price using Fyers (reliable) with fallback
            current_price = self.get_real_time_price(symbol)
            if current_price <= 0:
                # Use the price from trading signal as final fallback
                current_price = signal_data.get("current_price", 0)
                if current_price <= 0:
                    return {"success": False, "message": "Unable to get current price from any source"}
                logger.info(f"Using trading signal price for {symbol}: Rs.{current_price:.2f}")
            
            # Use provided quantity if available, otherwise calculate
            quantity = signal_data.get("quantity", 0)
            if quantity <= 0:
                # Calculate position size if no quantity provided
                signal_strength = signal_data.get("confidence", 0.5)
                quantity = self.calculate_position_size(symbol, current_price, signal_strength)
            
            if quantity <= 0:
                return {"success": False, "message": "Insufficient funds or invalid quantity"}
            
            # Place market buy order
            order_response = self.dhan_client.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type="MARKET",
                side="BUY"
            )
            
            order_id = order_response.get("orderId")
            if order_id:
                # Track pending order
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "side": "BUY",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": signal_data
                }
                
                # Record trade in database immediately (market orders usually execute quickly)
                try:
                    self.portfolio_manager.record_trade(
                        ticker=symbol,
                        action="buy",
                        quantity=quantity,
                        price=current_price,
                        stop_loss=signal_data.get("stop_loss"),
                        take_profit=signal_data.get("take_profit")
                    )
                    logger.info(f"Trade recorded in database: BUY {quantity} {symbol} at Rs.{current_price:.2f}")
                except Exception as db_error:
                    logger.error(f"Failed to record trade in database: {db_error}")
                
                # Update trade count
                self._update_daily_trade_count()
                
                logger.info(f"Buy order placed: {quantity} {symbol} at Rs.{current_price:.2f}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": current_price,
                    "message": f"Buy order placed for {quantity} shares of {symbol}"
                }
            else:
                return {"success": False, "message": "Order placement failed"}
                
        except Exception as e:
            logger.error(f"Failed to execute buy order for {symbol}: {e}")
            return {"success": False, "message": f"Buy order failed: {str(e)}"}

    
    def execute_sell_order(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute a sell order through Dhan API with database integration"""
        try:
            # Enforce global sell disable flag
            if not self.enable_sell:
                logger.info("Sell disabled by configuration (ENABLE_SELL=false). Skipping sell for %s", symbol)
                return {"success": False, "message": "Sell disabled by configuration"}

            # Get current price
            current_price = self.get_real_time_price(symbol)
            if not current_price:
                return {"success": False, "message": "Failed to get current price"}
            
            # Get current holding from database
            session = None
            try:
                session = self.portfolio_manager.db.Session()
                portfolio = session.query(Portfolio).filter_by(mode='live').first()
                if not portfolio:
                    return {"success": False, "message": "No live portfolio found"}
                
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id, 
                    ticker=symbol
                ).first()
                
                if not holding or holding.quantity <= 0:
                    return {"success": False, "message": "No position to sell"}
                    
                # Calculate profit/loss
                pnl = (current_price - holding.avg_price) * holding.quantity
                
                # Place order through Dhan API
                order_response = self.dhan_client.place_order(
                    symbol=symbol,
                    quantity=holding.quantity,
                    order_type="MARKET",
                    side="SELL"
                )
                
                order_id = order_response.get("orderId")
                if order_id:
                    # Record trade in database
                    try:
                        self.portfolio_manager.record_trade(
                            ticker=symbol,
                            action="sell",
                            quantity=holding.quantity,
                            price=current_price,
                            pnl=pnl,
                            stop_loss=signal_data.get("stop_loss"),
                            take_profit=signal_data.get("take_profit")
                        )
                        logger.info(f"Sell trade recorded in database: {holding.quantity} {symbol} at Rs.{current_price:.2f}")
                    except Exception as db_error:
                        logger.error(f"Failed to record sell trade in database: {db_error}")
                    
                    return {
                        "success": True,
                        "order_id": order_id,
                        "quantity": holding.quantity,
                        "price": current_price,
                        "pnl": pnl,
                        "message": f"Sell order placed for {holding.quantity} shares of {symbol}"
                    }
                else:
                    return {"success": False, "message": "Sell order placement failed"}
                    
            finally:
                if session:
                    session.close()

        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            return {"success": False, "message": str(e)}
    
    def check_and_update_orders(self) -> List[Dict]:
        """Check status of pending orders and update portfolio"""
        updated_orders = []
        
        try:
            # Get all orders from Dhan
            all_orders = self.dhan_client.get_orders()
            
            for order_id, pending_order in list(self.pending_orders.items()):
                # Find matching order in Dhan response
                dhan_order = next((o for o in all_orders if o.get("orderId") == order_id), None)
                
                if dhan_order:
                    order_status = dhan_order.get("orderStatus", "").upper()
                    
                    if order_status == "TRADED":
                        # Order executed
                        executed_price = float(dhan_order.get("price", pending_order["price"]))
                        executed_qty = int(dhan_order.get("quantity", pending_order["quantity"]))
                        
                        # Update portfolio
                        self._update_portfolio_after_execution(
                            pending_order["symbol"],
                            pending_order["side"],
                            executed_qty,
                            executed_price
                        )
                        
                        # Move to executed orders
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "executed_price": executed_price,
                            "executed_quantity": executed_qty,
                            "execution_time": datetime.now().isoformat(),
                            "status": "EXECUTED"
                        }
                        
                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]
                        
                        logger.info(f"Order executed: {pending_order['side']} {executed_qty} {pending_order['symbol']} at Rs.{executed_price:.2f}")
                    
                    elif order_status in ["CANCELLED", "REJECTED"]:
                        # Order failed
                        self.executed_orders[order_id] = {
                            **pending_order,
                            "status": order_status,
                            "failure_time": datetime.now().isoformat()
                        }
                        
                        updated_orders.append(self.executed_orders[order_id])
                        del self.pending_orders[order_id]
                        
                        logger.warning(f"Order {order_status.lower()}: {pending_order['side']} {pending_order['symbol']}")
            
            return updated_orders
            
        except Exception as e:
            logger.error(f"Failed to check order status: {e}")
            return []
    
    def _update_portfolio_after_execution(self, symbol: str, side: str, 
                                        quantity: int, price: float):
        """Update portfolio after order execution using database"""
        session = None
        try:
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if not portfolio:
                logger.error("No live portfolio found for execution update")
                return
            
            if side == "BUY":
                # Update or create holding
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id, 
                    ticker=symbol
                ).first()
                
                if holding:
                    # Average down the price
                    total_qty = holding.quantity + quantity
                    total_cost = (holding.quantity * holding.avg_price) + (quantity * price)
                    avg_price = total_cost / total_qty
                    
                    holding.quantity = total_qty
                    holding.avg_price = avg_price
                    holding.last_price = price
                else:
                    # New holding
                    holding = Holding(
                        portfolio_id=portfolio.id,
                        ticker=symbol,
                        quantity=quantity,
                        avg_price=price,
                        last_price=price
                    )
                    session.add(holding)
                
                # Reduce cash
                portfolio.cash -= quantity * price
                
            elif side == "SELL":
                # Update holding
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id, 
                    ticker=symbol
                ).first()
                
                if holding and holding.quantity >= quantity:
                    # Calculate P&L
                    pnl = quantity * (price - holding.avg_price)
                    self.daily_pnl += pnl
                    
                    # Update holding
                    remaining_qty = holding.quantity - quantity
                    if remaining_qty > 0:
                        holding.quantity = remaining_qty
                        holding.last_price = price
                    else:
                        # Remove holding completely
                        session.delete(holding)
                    
                    # Add cash
                    portfolio.cash += quantity * price
                    
                    # Update realized P&L
                    portfolio.realized_pnl += pnl
                    
                    logger.info(f"Trade P&L: Rs.{pnl:.2f} for {quantity} {symbol}")
            
            # Update portfolio timestamp
            portfolio.last_updated = datetime.now()
            session.commit()
            
            # Sync portfolio manager current portfolio
            self.portfolio_manager.current_portfolio = portfolio
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Failed to update portfolio after execution: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded using database"""
        current_date = datetime.now().date()
        
        # Reset daily counters if new day
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
        
        # Check trade count limit
        if self.daily_trade_count >= self.max_trade_limit:
            logger.warning(f"Daily trade limit reached: {self.daily_trade_count}")
            return True
        
        # Check daily loss limit using fresh database session
        session = None
        try:
            session = self.portfolio_manager.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode='live').first()
            if portfolio:
                total_value = portfolio.cash  # Simple calculation
                max_loss = total_value * self.max_daily_loss
                
                if self.daily_pnl < -max_loss:
                    logger.warning(f"Daily loss limit reached: Rs.{self.daily_pnl:.2f}")
                    return True
        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
        finally:
            if session:
                session.close()
        
        return False
    
    def _update_daily_trade_count(self):
        """Update daily trade counter"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 1
            self.last_trade_date = current_date
        else:
            self.daily_trade_count += 1
