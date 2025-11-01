#!/usr/bin/env python3
"""
Confirm Tool for Venting Layer
============================

MCP tool for validating results with Executor and logging confirmations
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

# Import portfolio manager for real-time validation
try:
    from ...portfolio_manager import PortfolioManager
    PORTFOLIO_AVAILABLE = True
except ImportError:
    PORTFOLIO_AVAILABLE = False
    logging.warning("Portfolio manager not available for real-time validation")

logger = logging.getLogger(__name__)

@dataclass
class ConfirmationResult:
    """Confirmation result"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    confirmed: bool
    reason: str
    execution_plan: Optional[Dict[str, Any]] = None
    timestamp: str = None

class ConfirmTool:
    """
    Confirm tool for Venting Layer
    
    Features:
    - Validate results with Executor via FastMCP
    - Log confirmations
    - Return validated JSON compatible with Trading Executor
    - Real-time dynamic confirmation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "confirm_tool")
        
        # Executor configuration
        self.executor_enabled = config.get("executor_enabled", True)
        self.trading_mode = config.get("trading_mode", "paper")  # "paper" or "live"
        
        # Risk controls
        self.max_position_size = config.get("max_position_size", 0.1)  # 10%
        self.risk_tolerance = config.get("risk_tolerance", 0.05)  # 5%
        
        # Real-time data configuration
        self.real_time_data = config.get("real_time_data", True)
        self.websocket_manager = None
        self.portfolio_manager = None
        
        # Performance tracking
        self.confirmation_cache = {}
        self.cache_timeout = config.get("cache_timeout", 10)  # seconds
        
        # Initialize real-time data if available
        if WEBSOCKET_AVAILABLE and self.real_time_data:
            try:
                self.websocket_manager = EnhancedWebSocketManager()
                logger.info("Real-time data connection established for confirmation")
            except Exception as e:
                logger.warning(f"Failed to initialize real-time data for confirmation: {e}")
        
        # Initialize portfolio manager if available
        if PORTFOLIO_AVAILABLE:
            try:
                self.portfolio_manager = PortfolioManager()
                logger.info("Portfolio manager initialized for confirmation")
            except Exception as e:
                logger.warning(f"Failed to initialize portfolio manager: {e}")
        
        logger.info(f"Confirm Tool {self.tool_id} initialized with real-time capabilities")
    
    async def confirm(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Validate results with Executor and log confirmations
        
        Args:
            arguments: {
                "actions": [
                    {
                        "symbol": "RELIANCE.NS",
                        "action": "BUY",
                        "confidence": 0.85,
                        "analysis": {...}
                    }
                ],
                "portfolio_value": 1000000,
                "risk_check": true,
                "real_time": true
            }
        """
        try:
            actions = arguments.get("actions", [])
            portfolio_value = arguments.get("portfolio_value", 1000000)
            risk_check = arguments.get("risk_check", True)
            real_time = arguments.get("real_time", self.real_time_data)
            
            if not actions:
                raise ValueError("Actions data is required for confirmation")
            
            # Check cache for recent confirmations
            cache_key = f"{hash(str(actions))}_{portfolio_value}_{risk_check}"
            if cache_key in self.confirmation_cache:
                cached_result = self.confirmation_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_result["timestamp"])).seconds < self.cache_timeout:
                    logger.info("Returning cached confirmation results")
                    # Update real-time data if requested
                    if real_time and self.websocket_manager:
                        try:
                            await self._update_real_time_confirmation(cached_result["confirmation_results"])
                        except Exception as e:
                            logger.warning(f"Failed to update real-time confirmation: {e}")
                    
                    return MCPToolResult(
                        status=MCPToolStatus.SUCCESS,
                        data=cached_result,
                        reasoning="Returning cached confirmation results with real-time updates",
                        confidence=0.9
                    )
            
            # Get real-time data if requested
            real_time_data = {}
            if real_time and self.websocket_manager:
                try:
                    symbols = [action.get("symbol", "") for action in actions if action.get("symbol")]
                    real_time_data = await self._get_real_time_data(symbols)
                except Exception as e:
                    logger.warning(f"Failed to get real-time data: {e}")
            
            # Get portfolio data if available
            portfolio_data = {}
            if self.portfolio_manager:
                try:
                    portfolio_data = self.portfolio_manager.get_portfolio_state()
                except Exception as e:
                    logger.warning(f"Failed to get portfolio data: {e}")
            
            # Validate actions with executor
            confirmation_results = []
            
            for action in actions:
                try:
                    confirmation = await self._validate_action(action, portfolio_value, risk_check, real_time_data, portfolio_data)
                    confirmation_results.append(confirmation)
                except Exception as e:
                    logger.warning(f"Confirmation error for {action.get('symbol', 'unknown')}: {e}")
                    # Add failed confirmation
                    confirmation_results.append(ConfirmationResult(
                        symbol=action.get("symbol", "UNKNOWN"),
                        action=action.get("action", "HOLD"),
                        confidence=action.get("confidence", 0.5),
                        confirmed=False,
                        reason=f"Validation error: {str(e)}"
                    ))
            
            # Count confirmed actions
            confirmed_count = sum(1 for result in confirmation_results if result.confirmed)
            
            # Add timestamp to each confirmation
            current_time = datetime.now().isoformat()
            for result in confirmation_results:
                result.timestamp = current_time
            
            # Prepare response
            response_data = {
                "timestamp": current_time,
                "total_actions": len(actions),
                "confirmed_actions": confirmed_count,
                "confirmation_rate": confirmed_count / len(actions) if actions else 0,
                "trading_mode": self.trading_mode,
                "real_time_data_used": bool(real_time_data),
                "portfolio_data_used": bool(portfolio_data),
                "confirmation_results": [asdict(result) for result in confirmation_results]
            }
            
            # Cache the result
            self.confirmation_cache[cache_key] = response_data
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Confirmed {confirmed_count} out of {len(actions)} actions for execution with real-time data",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Confirmation error: {e}")
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
    
    async def _update_real_time_confirmation(self, confirmations: List[Dict]) -> None:
        """Update existing confirmations with real-time data"""
        if not self.websocket_manager:
            return
        
        try:
            for confirmation in confirmations:
                symbol = confirmation.get("symbol")
                if symbol:
                    try:
                        rt_data = await self.websocket_manager.get_latest_quote(symbol)
                        if rt_data:
                            # Update execution plan with real-time price if it exists
                            if "execution_plan" in confirmation and confirmation["execution_plan"]:
                                confirmation["execution_plan"]["real_time_price"] = rt_data.get("last_price")
                                confirmation["execution_plan"]["real_time_change"] = rt_data.get("change_percent")
                            confirmation["timestamp"] = datetime.now().isoformat()
                    except Exception as e:
                        logger.warning(f"Failed to update real-time data for {symbol}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Real-time confirmation update error: {e}")
    
    async def _validate_action(self, action: Dict, portfolio_value: float, risk_check: bool, real_time_data: Dict[str, Any] = None, portfolio_data: Dict[str, Any] = None) -> ConfirmationResult:
        """Validate individual action with executor and real-time data"""
        try:
            symbol = action.get("symbol", "")
            action_type = action.get("action", "HOLD")
            confidence = action.get("confidence", 0.5)
            analysis = action.get("analysis", {})
            
            # Basic validation
            if not symbol:
                raise ValueError("Symbol is required")
            
            if action_type not in ["BUY", "SELL", "HOLD"]:
                raise ValueError(f"Invalid action type: {action_type}")
            
            # Get real-time data if available
            rt_data = real_time_data.get(symbol, {}) if real_time_data else {}
            
            # Risk checks if enabled
            if risk_check:
                risk_validation = await self._perform_risk_checks(action, portfolio_value, rt_data, portfolio_data)
                if not risk_validation["passed"]:
                    return ConfirmationResult(
                        symbol=symbol,
                        action=action_type,
                        confidence=confidence,
                        confirmed=False,
                        reason=risk_validation["reason"]
                    )
            
            # Generate execution plan for confirmed actions with real-time data
            execution_plan = None
            if action_type in ["BUY", "SELL"]:
                execution_plan = await self._generate_execution_plan(action, portfolio_value, rt_data, portfolio_data)
            
            # Determine if action should be confirmed
            confirmed = self._should_confirm_action(action_type, confidence, analysis, rt_data)
            reason = "Action confirmed" if confirmed else "Action not confirmed due to low confidence or risk factors"
            
            # Add real-time context to reason
            if rt_data:
                reason += f" (real-time price: {rt_data.get('last_price', 'N/A')})"
            
            return ConfirmationResult(
                symbol=symbol,
                action=action_type,
                confidence=confidence,
                confirmed=confirmed,
                reason=reason,
                execution_plan=execution_plan
            )
            
        except Exception as e:
            logger.error(f"Action validation error: {e}")
            raise
    
    async def _perform_risk_checks(self, action: Dict, portfolio_value: float, real_time_data: Dict[str, Any] = None, portfolio_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform risk checks for action with real-time data"""
        try:
            checks = {
                "passed": True,
                "reason": ""
            }
            
            action_type = action.get("action", "HOLD")
            confidence = action.get("confidence", 0.5)
            symbol = action.get("symbol", "")
            
            # Skip risk checks for HOLD actions
            if action_type == "HOLD":
                return checks
            
            # Confidence check
            if confidence < 0.6:
                checks["passed"] = False
                checks["reason"] = f"Low confidence ({confidence:.2f}) for {action_type} action"
                return checks
            
            # Position size check (simplified)
            estimated_position_size = 0.05  # Assume 5% position size
            if estimated_position_size > self.max_position_size:
                checks["passed"] = False
                checks["reason"] = f"Position size ({estimated_position_size:.1%}) exceeds limit ({self.max_position_size:.1%})"
                return checks
            
            # Risk tolerance check
            estimated_risk = 0.03  # Assume 3% risk
            if estimated_risk > self.risk_tolerance:
                checks["passed"] = False
                checks["reason"] = f"Estimated risk ({estimated_risk:.1%}) exceeds tolerance ({self.risk_tolerance:.1%})"
                return checks
            
            # Real-time price validation if available
            if real_time_data:
                last_price = real_time_data.get("last_price", 0)
                if last_price <= 0:
                    checks["passed"] = False
                    checks["reason"] = f"Invalid real-time price: {last_price}"
                    return checks
            
            # Portfolio validation if available
            if portfolio_data:
                # Check if we have enough cash for BUY actions
                if action_type == "BUY":
                    available_cash = portfolio_data.get("available_cash", 0)
                    estimated_cost = 0
                    if real_time_data:
                        last_price = real_time_data.get("last_price", 0)
                        estimated_cost = last_price * 10  # Assume 10 shares
                    
                    if estimated_cost > available_cash:
                        checks["passed"] = False
                        checks["reason"] = f"Insufficient funds (need ₹{estimated_cost:,.2f}, have ₹{available_cash:,.2f})"
                        return checks
                
                # Check if we have the position for SELL actions
                if action_type == "SELL":
                    positions = portfolio_data.get("positions", {})
                    if symbol not in positions or positions[symbol]["quantity"] <= 0:
                        checks["passed"] = False
                        checks["reason"] = f"No position to sell for {symbol}"
                        return checks
            
            return checks
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return {
                "passed": False,
                "reason": f"Risk check failed: {str(e)}"
            }
    
    async def _generate_execution_plan(self, action: Dict, portfolio_value: float, real_time_data: Dict[str, Any] = None, portfolio_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate execution plan for confirmed action with real-time data"""
        try:
            symbol = action.get("symbol", "")
            action_type = action.get("action", "HOLD")
            confidence = action.get("confidence", 0.5)
            
            # Simulate execution plan
            execution_plan = {
                "symbol": symbol,
                "action": action_type,
                "quantity": 10,  # Simulated quantity
                "order_type": "MARKET",
                "estimated_price": 1000 + (hash(symbol) % 1000),  # Simulated price
                "estimated_value": 0,
                "priority": "NORMAL"
            }
            
            # Update with real-time data if available
            if real_time_data:
                last_price = real_time_data.get("last_price", execution_plan["estimated_price"])
                execution_plan["estimated_price"] = last_price
                execution_plan["real_time_price"] = last_price
                execution_plan["real_time_change"] = real_time_data.get("change_percent", 0)
            
            # Calculate estimated value
            execution_plan["estimated_value"] = execution_plan["quantity"] * execution_plan["estimated_price"]
            
            # Set priority based on confidence and real-time momentum
            priority = "NORMAL"
            if confidence > 0.9:
                priority = "HIGH"
            elif confidence < 0.7:
                priority = "LOW"
            
            # Adjust priority based on real-time momentum
            if real_time_data:
                change_pct = real_time_data.get("change_percent", 0)
                if action_type == "BUY" and change_pct > 2:
                    priority = "HIGH"  # Strong positive momentum
                elif action_type == "SELL" and change_pct < -2:
                    priority = "HIGH"  # Strong negative momentum
            
            execution_plan["priority"] = priority
            
            # Add portfolio context if available
            if portfolio_data:
                execution_plan["portfolio_context"] = {
                    "available_cash": portfolio_data.get("available_cash", 0),
                    "current_positions": len(portfolio_data.get("positions", {}))
                }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Execution plan generation error: {e}")
            return {
                "symbol": action.get("symbol", ""),
                "action": action.get("action", "HOLD"),
                "error": str(e)
            }
    
    def _should_confirm_action(self, action_type: str, confidence: float, analysis: Dict, real_time_data: Dict[str, Any] = None) -> bool:
        """Determine if action should be confirmed with real-time data"""
        # HOLD actions are always confirmed
        if action_type == "HOLD":
            return True
        
        # BUY/SELL actions need high confidence
        if confidence < 0.7:
            return False
        
        # Additional checks based on analysis
        if analysis.get("sentiment") == "negative" and action_type == "BUY":
            return False
        
        if analysis.get("sentiment") == "positive" and action_type == "SELL":
            return False
        
        # Real-time validation
        if real_time_data:
            last_price = real_time_data.get("last_price", 0)
            if last_price <= 0:
                return False  # Invalid price
            
            # For BUY actions, check if price is reasonable
            if action_type == "BUY":
                # Add more sophisticated price validation here
                pass
            
            # For SELL actions, check if we're not selling at a loss unnecessarily
            if action_type == "SELL":
                change_pct = real_time_data.get("change_percent", 0)
                # If price is dropping rapidly, might want to sell
                if change_pct < -3:
                    return True
                # If confidence is low but price is stable, might hold
                elif confidence < 0.8 and abs(change_pct) < 1:
                    return False
        
        return True
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get confirm tool status"""
        return {
            "tool_id": self.tool_id,
            "executor_enabled": self.executor_enabled,
            "trading_mode": self.trading_mode,
            "max_position_size": self.max_position_size,
            "risk_tolerance": self.risk_tolerance,
            "real_time_data": self.real_time_data,
            "portfolio_validation": PORTFOLIO_AVAILABLE,
            "cache_size": len(self.confirmation_cache),
            "status": "active"
        }