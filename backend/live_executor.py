#!/usr/bin/env python3
"""
Live Trading Executor for Dhan API Integration
Handles real order execution, portfolio management, and risk controls
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dhan_client import DhanAPIClient

logger = logging.getLogger(__name__)

class LiveTradingExecutor:
    """Execute live trades through Dhan API with risk management"""
    
    def __init__(self, portfolio, config: Dict):
        self.portfolio = portfolio
        self.config = config
        self.stop_loss_pct = config.get("stop_loss_pct", 0.05)
        self.max_capital_per_trade = config.get("max_capital_per_trade", 0.25)
        self.max_trade_limit = config.get("max_trade_limit", 10)
        
        # Initialize Dhan client
        self.dhan_client = DhanAPIClient(
            client_id=config.get("dhan_client_id"),
            access_token=config.get("dhan_access_token")
        )
        
        # Trading state
        self.pending_orders = {}
        self.executed_orders = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Risk management
        self.max_daily_loss = config.get("max_daily_loss", 0.05)  # 5% of portfolio
        self.daily_pnl = 0.0
        
        logger.info("Live Trading Executor initialized")
    
    def validate_connection(self) -> bool:
        """Validate Dhan API connection"""
        try:
            return self.dhan_client.validate_connection()
        except Exception as e:
            logger.error(f"Failed to validate Dhan connection: {e}")
            return False
    
    def sync_portfolio_with_dhan(self) -> bool:
        """Sync local portfolio with actual Dhan account"""
        try:
            logger.info("Syncing portfolio with Dhan account...")
            
            # Get account funds
            funds = self.dhan_client.get_funds()
            available_cash = float(funds.get("availabelBalance", 0))
            
            # Get current holdings
            holdings = self.dhan_client.get_holdings()
            
            # Update portfolio cash
            self.portfolio.cash = available_cash
            
            # Update holdings
            self.portfolio.holdings = {}
            total_holdings_value = 0
            
            for holding in holdings:
                symbol = holding.get("tradingSymbol", "")
                quantity = int(holding.get("totalQty", 0))
                avg_price = float(holding.get("avgCostPrice", 0))
                current_price = avg_price  # We'll need to get current price separately
                
                if quantity > 0:
                    self.portfolio.holdings[symbol] = {
                        "quantity": quantity,
                        "avg_price": avg_price,
                        "current_price": current_price,
                        "total_value": quantity * current_price,
                        "pnl": quantity * (current_price - avg_price)
                    }
                    total_holdings_value += quantity * current_price
            
            # Update portfolio value
            self.portfolio.total_value = available_cash + total_holdings_value
            
            logger.info(f"Portfolio synced - Cash: Rs.{available_cash:.2f}, Holdings: Rs.{total_holdings_value:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync portfolio with Dhan: {e}")
            return False
    
    def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price from Dhan API"""
        try:
            quote = self.dhan_client.get_quote(symbol)
            price = float(quote.get("ltp", 0))
            
            if price > 0:
                return price
            else:
                logger.warning(f"Invalid price received for {symbol}: {price}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              signal_strength: float) -> int:
        """Calculate position size based on risk management"""
        try:
            # Get available cash
            available_cash = self.portfolio.cash
            
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
    
    def execute_buy_order(self, symbol: str, signal_data: Dict) -> Dict:
        """Execute a buy order through Dhan API"""
        try:
            # Check daily trade limit
            if self._check_daily_limits():
                return {"success": False, "message": "Daily trade limit exceeded"}
            
            # Check market status
            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}
            
            # Get current price
            current_price = self.get_real_time_price(symbol)
            if current_price <= 0:
                return {"success": False, "message": "Unable to get current price"}
            
            # Calculate position size
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
        """Execute a sell order through Dhan API"""
        try:
            # Check if we have holdings
            if symbol not in self.portfolio.holdings:
                return {"success": False, "message": f"No holdings found for {symbol}"}
            
            holding = self.portfolio.holdings[symbol]
            available_quantity = holding["quantity"]
            
            if available_quantity <= 0:
                return {"success": False, "message": f"No shares available to sell for {symbol}"}
            
            # Check market status
            if not self.dhan_client.is_market_open():
                return {"success": False, "message": "Market is closed"}
            
            # Get current price
            current_price = self.get_real_time_price(symbol)
            if current_price <= 0:
                return {"success": False, "message": "Unable to get current price"}
            
            # Determine sell quantity (sell all for now)
            sell_quantity = available_quantity
            
            # Place market sell order
            order_response = self.dhan_client.place_order(
                symbol=symbol,
                quantity=sell_quantity,
                order_type="MARKET",
                side="SELL"
            )
            
            order_id = order_response.get("orderId")
            if order_id:
                # Track pending order
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "quantity": sell_quantity,
                    "side": "SELL",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                    "signal_data": signal_data
                }
                
                # Update trade count
                self._update_daily_trade_count()
                
                logger.info(f"Sell order placed: {sell_quantity} {symbol} at Rs.{current_price:.2f}")
                
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "quantity": sell_quantity,
                    "price": current_price,
                    "message": f"Sell order placed for {sell_quantity} shares of {symbol}"
                }
            else:
                return {"success": False, "message": "Order placement failed"}
                
        except Exception as e:
            logger.error(f"Failed to execute sell order for {symbol}: {e}")
            return {"success": False, "message": f"Sell order failed: {str(e)}"}
    
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
        """Update portfolio after order execution"""
        try:
            if side == "BUY":
                # Add to holdings
                if symbol in self.portfolio.holdings:
                    # Average down the price
                    existing = self.portfolio.holdings[symbol]
                    total_qty = existing["quantity"] + quantity
                    total_cost = (existing["quantity"] * existing["avg_price"]) + (quantity * price)
                    avg_price = total_cost / total_qty
                    
                    self.portfolio.holdings[symbol] = {
                        "quantity": total_qty,
                        "avg_price": avg_price,
                        "current_price": price,
                        "total_value": total_qty * price,
                        "pnl": total_qty * (price - avg_price)
                    }
                else:
                    # New holding
                    self.portfolio.holdings[symbol] = {
                        "quantity": quantity,
                        "avg_price": price,
                        "current_price": price,
                        "total_value": quantity * price,
                        "pnl": 0.0
                    }
                
                # Reduce cash
                self.portfolio.cash -= quantity * price
                
            elif side == "SELL":
                # Remove from holdings
                if symbol in self.portfolio.holdings:
                    existing = self.portfolio.holdings[symbol]
                    
                    if existing["quantity"] >= quantity:
                        # Calculate P&L
                        pnl = quantity * (price - existing["avg_price"])
                        self.daily_pnl += pnl
                        
                        # Update holding
                        remaining_qty = existing["quantity"] - quantity
                        if remaining_qty > 0:
                            self.portfolio.holdings[symbol]["quantity"] = remaining_qty
                            self.portfolio.holdings[symbol]["total_value"] = remaining_qty * price
                            self.portfolio.holdings[symbol]["pnl"] = remaining_qty * (price - existing["avg_price"])
                        else:
                            # Remove holding completely
                            del self.portfolio.holdings[symbol]
                        
                        # Add cash
                        self.portfolio.cash += quantity * price
                        
                        logger.info(f"Trade P&L: Rs.{pnl:.2f} for {quantity} {symbol}")
            
            # Update total portfolio value
            holdings_value = sum(h["total_value"] for h in self.portfolio.holdings.values())
            self.portfolio.total_value = self.portfolio.cash + holdings_value
            
        except Exception as e:
            logger.error(f"Failed to update portfolio after execution: {e}")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded"""
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
        
        # Check daily loss limit
        max_loss = self.portfolio.total_value * self.max_daily_loss
        if self.daily_pnl < -max_loss:
            logger.warning(f"Daily loss limit reached: Rs.{self.daily_pnl:.2f}")
            return True
        
        return False
    
    def _update_daily_trade_count(self):
        """Update daily trade counter"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 1
            self.last_trade_date = current_date
        else:
            self.daily_trade_count += 1
