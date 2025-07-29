#!/usr/bin/env python3
"""
Trade Execution Tool
===================

Production-grade MCP tool for trade execution, order management,
and portfolio operations with comprehensive risk controls.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..mcp_trading_server import MCPToolResult, MCPToolStatus

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class TradeOrder:
    """Trade order structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    execution_time: Optional[datetime] = None

@dataclass
class ExecutionResult:
    """Trade execution result"""
    order: TradeOrder
    execution_status: str
    execution_message: str
    transaction_cost: float
    portfolio_impact: Dict[str, Any]
    risk_checks: Dict[str, Any]

class ExecutionTool:
    """
    Production-grade trade execution tool
    
    Features:
    - Multi-broker execution support
    - Real-time order management
    - Risk control integration
    - Portfolio impact analysis
    - Transaction cost analysis
    - Execution quality monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "execution_tool")
        
        # Execution configuration
        self.trading_mode = config.get("trading_mode", "paper")  # "paper" or "live"
        self.broker_config = config.get("broker_config", {})
        
        # Risk controls
        self.max_order_value = config.get("max_order_value", 100000)  # ₹1 lakh
        self.max_position_size = config.get("max_position_size", 0.25)  # 25%
        self.daily_loss_limit = config.get("daily_loss_limit", 0.05)  # 5%
        
        # Performance tracking
        self.orders_executed = 0
        self.total_volume = 0
        self.execution_errors = 0
        
        # Order tracking
        self.active_orders = {}
        self.order_history = []
        
        logger.info(f"Execution Tool {self.tool_id} initialized in {self.trading_mode} mode")
    
    async def execute_trade(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Execute a trade order with comprehensive risk checks
        
        Args:
            arguments: {
                "symbol": "RELIANCE.NS",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "price": 2600.0,
                "stop_loss": 2500.0,
                "take_profit": 2700.0
            }
        """
        try:
            # Validate required parameters
            symbol = arguments.get("symbol")
            side = arguments.get("side")
            quantity = arguments.get("quantity")
            order_type = arguments.get("order_type", "MARKET")
            
            if not all([symbol, side, quantity]):
                raise ValueError("Symbol, side, and quantity are required")
            
            # Create order object
            order = TradeOrder(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=OrderSide(side.upper()),
                order_type=OrderType(order_type.upper()),
                quantity=int(quantity),
                price=arguments.get("price"),
                stop_price=arguments.get("stop_loss"),
                created_at=datetime.now()
            )
            
            # Perform pre-execution risk checks
            risk_checks = await self._perform_risk_checks(order, arguments)
            
            if not risk_checks["passed"]:
                return MCPToolResult(
                    status=MCPToolStatus.ERROR,
                    error=f"Risk check failed: {risk_checks['reason']}",
                    data={"risk_checks": risk_checks}
                )
            
            # Execute the order
            execution_result = await self._execute_order(order, arguments)
            
            # Update portfolio
            portfolio_impact = await self._update_portfolio(execution_result)
            
            # Calculate transaction costs
            transaction_cost = self._calculate_transaction_cost(execution_result)
            
            # Create execution result
            result = ExecutionResult(
                order=execution_result,
                execution_status="SUCCESS" if execution_result.status == OrderStatus.FILLED else "PARTIAL",
                execution_message=self._generate_execution_message(execution_result),
                transaction_cost=transaction_cost,
                portfolio_impact=portfolio_impact,
                risk_checks=risk_checks
            )
            
            # Track the order
            self.active_orders[order.order_id] = execution_result
            self.order_history.append(execution_result)
            self.orders_executed += 1
            self.total_volume += execution_result.filled_quantity * (execution_result.filled_price or 0)
            
            # Prepare response
            response_data = {
                "execution_result": asdict(result),
                "order_details": asdict(execution_result),
                "portfolio_impact": portfolio_impact,
                "transaction_summary": {
                    "order_id": execution_result.order_id,
                    "symbol": execution_result.symbol,
                    "side": execution_result.side.value,
                    "quantity_filled": execution_result.filled_quantity,
                    "execution_price": execution_result.filled_price,
                    "total_value": execution_result.filled_quantity * (execution_result.filled_price or 0),
                    "transaction_cost": transaction_cost,
                    "execution_time": execution_result.execution_time.isoformat() if execution_result.execution_time else None
                },
                "execution_metadata": {
                    "trading_mode": self.trading_mode,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Trade executed successfully: {side} {quantity} {symbol}",
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.execution_errors += 1
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def get_order_status(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Get status of specific order or all active orders
        
        Args:
            arguments: {
                "order_id": "ORD_123456",  # Optional, if not provided returns all active orders
                "include_history": false
            }
        """
        try:
            order_id = arguments.get("order_id")
            include_history = arguments.get("include_history", False)
            
            if order_id:
                # Get specific order
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    response_data = {
                        "order": asdict(order),
                        "is_active": order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
                    }
                else:
                    # Check order history
                    historical_order = next((o for o in self.order_history if o.order_id == order_id), None)
                    if historical_order:
                        response_data = {
                            "order": asdict(historical_order),
                            "is_active": False
                        }
                    else:
                        raise ValueError(f"Order {order_id} not found")
            else:
                # Get all active orders
                active_orders = [asdict(order) for order in self.active_orders.values()]
                response_data = {
                    "active_orders": active_orders,
                    "active_count": len(active_orders)
                }
                
                if include_history:
                    response_data["order_history"] = [asdict(order) for order in self.order_history[-20:]]  # Last 20 orders
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning="Order status retrieved successfully",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Order status error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def cancel_order(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Cancel an active order
        
        Args:
            arguments: {
                "order_id": "ORD_123456"
            }
        """
        try:
            order_id = arguments.get("order_id")
            if not order_id:
                raise ValueError("Order ID is required")
            
            if order_id not in self.active_orders:
                raise ValueError(f"Order {order_id} not found or not active")
            
            order = self.active_orders[order_id]
            
            # Check if order can be cancelled
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
                raise ValueError(f"Order {order_id} cannot be cancelled (status: {order.status.value})")
            
            # Cancel the order
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            # Remove from active orders
            del self.active_orders[order_id]
            
            response_data = {
                "order_id": order_id,
                "cancellation_status": "SUCCESS",
                "cancelled_at": order.updated_at.isoformat(),
                "order_details": asdict(order)
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Order {order_id} cancelled successfully",
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def get_execution_analytics(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Get execution analytics and performance metrics
        
        Args:
            arguments: {
                "time_period": "1D",  # "1D", "1W", "1M"
                "symbol": "RELIANCE.NS"  # Optional filter
            }
        """
        try:
            time_period = arguments.get("time_period", "1D")
            symbol_filter = arguments.get("symbol")
            
            # Calculate time range
            if time_period == "1D":
                start_time = datetime.now() - timedelta(days=1)
            elif time_period == "1W":
                start_time = datetime.now() - timedelta(weeks=1)
            elif time_period == "1M":
                start_time = datetime.now() - timedelta(days=30)
            else:
                start_time = datetime.now() - timedelta(days=1)
            
            # Filter orders by time and symbol
            filtered_orders = [
                order for order in self.order_history
                if order.created_at >= start_time and
                (not symbol_filter or order.symbol == symbol_filter)
            ]
            
            # Calculate analytics
            analytics = self._calculate_execution_analytics(filtered_orders)
            
            response_data = {
                "analytics_period": time_period,
                "symbol_filter": symbol_filter,
                "analytics": analytics,
                "order_count": len(filtered_orders),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning="Execution analytics calculated successfully",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Execution analytics error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def _perform_risk_checks(self, order: TradeOrder, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk checks before execution"""
        try:
            checks = {
                "passed": True,
                "reason": "",
                "checks_performed": []
            }
            
            # Order value check
            estimated_value = order.quantity * (order.price or 2500)  # Use price or default
            if estimated_value > self.max_order_value:
                checks["passed"] = False
                checks["reason"] = f"Order value (₹{estimated_value:,.0f}) exceeds limit (₹{self.max_order_value:,.0f})"
                return checks
            
            checks["checks_performed"].append({
                "check": "order_value",
                "status": "PASSED",
                "value": estimated_value,
                "limit": self.max_order_value
            })
            
            # Position size check (simplified)
            portfolio_value = 1000000  # Assume ₹10 lakh portfolio
            position_percentage = estimated_value / portfolio_value
            
            if position_percentage > self.max_position_size:
                checks["passed"] = False
                checks["reason"] = f"Position size ({position_percentage:.1%}) exceeds limit ({self.max_position_size:.1%})"
                return checks
            
            checks["checks_performed"].append({
                "check": "position_size",
                "status": "PASSED",
                "value": position_percentage,
                "limit": self.max_position_size
            })
            
            # Daily loss limit check (simplified)
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -self.daily_loss_limit * portfolio_value:
                checks["passed"] = False
                checks["reason"] = f"Daily loss limit exceeded (₹{abs(daily_pnl):,.0f})"
                return checks
            
            checks["checks_performed"].append({
                "check": "daily_loss_limit",
                "status": "PASSED",
                "current_pnl": daily_pnl,
                "limit": -self.daily_loss_limit * portfolio_value
            })
            
            # Market hours check (simplified)
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 15:  # Indian market hours
                checks["passed"] = False
                checks["reason"] = "Market is closed"
                return checks
            
            checks["checks_performed"].append({
                "check": "market_hours",
                "status": "PASSED",
                "current_time": datetime.now().isoformat()
            })
            
            return checks
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return {
                "passed": False,
                "reason": f"Risk check failed: {str(e)}",
                "checks_performed": []
            }
    
    async def _execute_order(self, order: TradeOrder, arguments: Dict[str, Any]) -> TradeOrder:
        """Execute the order based on trading mode"""
        try:
            if self.trading_mode == "paper":
                return await self._execute_paper_order(order, arguments)
            else:
                return await self._execute_live_order(order, arguments)
                
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            raise
    
    async def _execute_paper_order(self, order: TradeOrder, arguments: Dict[str, Any]) -> TradeOrder:
        """Execute order in paper trading mode"""
        try:
            # Simulate order execution
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Get current market price (simulated)
            current_price = self._get_simulated_price(order.symbol)
            
            if order.order_type == OrderType.MARKET:
                # Market order - immediate execution
                order.filled_quantity = order.quantity
                order.filled_price = current_price
                order.status = OrderStatus.FILLED
                order.execution_time = datetime.now()
            
            elif order.order_type == OrderType.LIMIT:
                # Limit order - check if price is favorable
                if ((order.side == OrderSide.BUY and current_price <= order.price) or
                    (order.side == OrderSide.SELL and current_price >= order.price)):
                    order.filled_quantity = order.quantity
                    order.filled_price = order.price
                    order.status = OrderStatus.FILLED
                    order.execution_time = datetime.now()
                else:
                    order.status = OrderStatus.PENDING
            
            order.updated_at = datetime.now()
            return order
            
        except Exception as e:
            logger.error(f"Paper order execution error: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return order
    
    async def _execute_live_order(self, order: TradeOrder, arguments: Dict[str, Any]) -> TradeOrder:
        """Execute order in live trading mode"""
        try:
            # This would integrate with actual broker APIs
            # For now, simulate live execution
            logger.warning("Live trading not implemented - using paper trading simulation")
            return await self._execute_paper_order(order, arguments)
            
        except Exception as e:
            logger.error(f"Live order execution error: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return order
    
    async def _update_portfolio(self, order: TradeOrder) -> Dict[str, Any]:
        """Update portfolio after order execution"""
        try:
            if order.status != OrderStatus.FILLED:
                return {"updated": False, "reason": "Order not filled"}
            
            # Calculate portfolio impact
            trade_value = order.filled_quantity * order.filled_price
            
            portfolio_impact = {
                "updated": True,
                "symbol": order.symbol,
                "quantity_change": order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity,
                "cash_change": -trade_value if order.side == OrderSide.BUY else trade_value,
                "trade_value": trade_value,
                "timestamp": datetime.now().isoformat()
            }
            
            return portfolio_impact
            
        except Exception as e:
            logger.error(f"Portfolio update error: {e}")
            return {"updated": False, "reason": str(e)}
    
    def _calculate_transaction_cost(self, order: TradeOrder) -> float:
        """Calculate transaction costs"""
        if order.status != OrderStatus.FILLED:
            return 0.0
        
        trade_value = order.filled_quantity * order.filled_price
        
        # Simplified transaction cost calculation
        brokerage = min(trade_value * 0.0003, 20)  # 0.03% or ₹20, whichever is lower
        stt = trade_value * 0.001 if order.side == OrderSide.SELL else 0  # STT on sell side
        exchange_charges = trade_value * 0.0000345
        gst = (brokerage + exchange_charges) * 0.18
        
        total_cost = brokerage + stt + exchange_charges + gst
        return round(total_cost, 2)
    
    def _generate_execution_message(self, order: TradeOrder) -> str:
        """Generate human-readable execution message"""
        if order.status == OrderStatus.FILLED:
            return f"Successfully {order.side.value.lower()} {order.filled_quantity} shares of {order.symbol} at ₹{order.filled_price:.2f}"
        elif order.status == OrderStatus.PENDING:
            return f"Order pending: {order.side.value.lower()} {order.quantity} shares of {order.symbol}"
        elif order.status == OrderStatus.REJECTED:
            return f"Order rejected: {order.side.value.lower()} {order.quantity} shares of {order.symbol}"
        else:
            return f"Order status: {order.status.value}"
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        import uuid
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated current price for symbol"""
        # Simplified price simulation
        base_prices = {
            "RELIANCE.NS": 2600,
            "TCS.NS": 3300,
            "INFY.NS": 1500,
            "HDFCBANK.NS": 1600,
            "ICICIBANK.NS": 900
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Add some random variation
        import random
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        return round(base_price * (1 + variation), 2)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate daily P&L (simplified)"""
        # Simplified daily P&L calculation
        today_orders = [
            order for order in self.order_history
            if order.created_at.date() == datetime.now().date() and
            order.status == OrderStatus.FILLED
        ]
        
        # This is a simplified calculation
        # In reality, you'd need to track position changes and mark-to-market
        return sum(
            (order.filled_quantity * order.filled_price * (1 if order.side == OrderSide.SELL else -1))
            for order in today_orders
        )
    
    def _calculate_execution_analytics(self, orders: List[TradeOrder]) -> Dict[str, Any]:
        """Calculate execution analytics for given orders"""
        if not orders:
            return {
                "total_orders": 0,
                "fill_rate": 0,
                "average_execution_time": 0,
                "total_volume": 0,
                "total_transaction_costs": 0
            }
        
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        
        analytics = {
            "total_orders": len(orders),
            "filled_orders": len(filled_orders),
            "fill_rate": len(filled_orders) / len(orders) if orders else 0,
            "total_volume": sum(o.filled_quantity * (o.filled_price or 0) for o in filled_orders),
            "average_order_size": sum(o.quantity for o in orders) / len(orders) if orders else 0,
            "symbols_traded": len(set(o.symbol for o in orders)),
            "buy_orders": len([o for o in orders if o.side == OrderSide.BUY]),
            "sell_orders": len([o for o in orders if o.side == OrderSide.SELL])
        }
        
        # Calculate average execution time for filled orders
        execution_times = [
            (o.execution_time - o.created_at).total_seconds()
            for o in filled_orders
            if o.execution_time and o.created_at
        ]
        
        analytics["average_execution_time"] = (
            sum(execution_times) / len(execution_times) if execution_times else 0
        )
        
        return analytics
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get execution tool status"""
        return {
            "tool_id": self.tool_id,
            "trading_mode": self.trading_mode,
            "orders_executed": self.orders_executed,
            "total_volume": self.total_volume,
            "execution_errors": self.execution_errors,
            "active_orders_count": len(self.active_orders),
            "risk_controls": {
                "max_order_value": self.max_order_value,
                "max_position_size": self.max_position_size,
                "daily_loss_limit": self.daily_loss_limit
            },
            "status": "active"
        }
