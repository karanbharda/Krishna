#!/usr/bin/env python3
"""
MCP Confirm Tool
===============

Trade confirmation tool for the Model Context Protocol server
with executor validation and standardized JSON responses.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult
import json
import time
from datetime import datetime
import logging
import numpy as np
from pathlib import Path

# Import stock analysis components
from utils.ml_components.stock_analysis_complete import LOGS_DIR

logger = logging.getLogger(__name__)

# Request/Response logging
MCP_LOG_DIR = LOGS_DIR / "mcp_requests"
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class ConfirmTool:
    """
    MCP Confirm Tool
    Validates results with Executor via FastMCP and logs confirmations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "confirm_tool")
        self.executor_enabled = config.get("executor_enabled", True)
        self.trading_mode = config.get(
            "trading_mode", "paper")  # paper or live
        self.max_position_size = config.get("max_position_size", 0.1)
        self.risk_tolerance = config.get("risk_tolerance", 0.05)

        self.request_counter = 0

        logger.info(f"Confirm Tool {self.tool_id} initialized")

    def _log_request(self, tool_name: str, request_data: Dict) -> str:
        """Log incoming request"""
        self.request_counter += 1
        request_id = f"{tool_name}_{int(time.time())}_{self.request_counter}"

        log_entry = {
            "request_id": request_id,
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "request": request_data
        }

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_requests.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(f"MCP Request [{request_id}]: {tool_name}")
        return request_id

    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types and other non-JSON-serializable types to Python native types"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return obj

    def _log_response(self, request_id: str, response_data: Dict, duration_ms: float):
        """Log outgoing response"""
        # Convert numpy types to Python native types for JSON serialization
        sanitized_response = self._convert_to_json_serializable(response_data)

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "response": sanitized_response
        }

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_responses.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except TypeError as e:
            logger.error(f"Failed to log response: {e}")
            logger.error(f"Problematic data: {log_entry}")
            # Try to log with default=str as fallback
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry, default=str) + '\n')
            except Exception as e2:
                logger.error(
                    f"Failed to log response even with default=str: {e2}")

        logger.info(f"MCP Response [{request_id}]: {duration_ms:.2f}ms")

    def _log_confirmation(self, symbol: str, decision: str, session_id: str, reason: str):
        """Log trade confirmation to main predictions file"""
        try:
            confirmation_data = {
                "symbol": symbol,
                "decision": decision,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "confirmed" if decision == "approve" else "rejected",
                "reason": reason
            }
            log_file = LOGS_DIR / "mcp_predictions.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(confirmation_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log confirmation: {e}")

    async def confirm(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Confirm trading actions with validation

        Args:
            arguments: Tool arguments containing actions to confirm
            session_id: Session identifier

        Returns:
            MCPToolResult with confirmation results
        """
        start_time = time.time()

        try:
            # Extract parameters
            actions = arguments.get("actions", [])
            portfolio_value = arguments.get("portfolio_value", 0.0)
            risk_check = arguments.get("risk_check", True)

            if not actions:
                return MCPToolResult(
                    status="ERROR",
                    error="No actions provided for confirmation",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            request_data = {
                "actions": actions,
                "portfolio_value": portfolio_value,
                "risk_check": risk_check
            }
            request_id = self._log_request("confirm", request_data)

            # Process each action
            confirmed_actions = []
            rejected_actions = []
            total_confidence = 0.0

            for action in actions:
                try:
                    symbol = action.get("symbol")
                    decision = action.get("action")
                    confidence = action.get("confidence", 0.0)
                    analysis = action.get("analysis", {})

                    if not symbol or not decision:
                        continue

                    # Perform risk checks if enabled
                    if risk_check and confidence < 0.5:
                        # Reject low confidence actions
                        rejected_actions.append({
                            "symbol": symbol,
                            "action": decision,
                            "confidence": confidence,
                            "reason": "Low confidence threshold not met",
                            "status": "REJECTED"
                        })
                        continue

                    # Check position size limits
                    if "position_size" in analysis:
                        position_size = analysis["position_size"]
                        if position_size > self.max_position_size:
                            rejected_actions.append({
                                "symbol": symbol,
                                "action": decision,
                                "confidence": confidence,
                                "reason": f"Position size {position_size:.2%} exceeds limit {self.max_position_size:.2%}",
                                "status": "REJECTED"
                            })
                            continue

                    # Confirm the action
                    reason = f"Confirmed {decision} action with {confidence:.1%} confidence"
                    self._log_confirmation(
                        symbol=symbol,
                        decision="approve" if decision.upper(
                        ) in ["BUY", "SELL"] else "reject",
                        session_id=session_id,
                        reason=reason
                    )

                    result = {
                        "symbol": symbol,
                        "decision": decision,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": "confirmed" if decision.upper() in ["BUY", "SELL"] else "rejected",
                        "reason": reason
                    }

                    confirmed_actions.append({
                        "symbol": symbol,
                        "action": decision,
                        "confidence": confidence,
                        "result": result,
                        "status": "CONFIRMED"
                    })

                    total_confidence += confidence

                except Exception as e:
                    logger.warning(f"Confirmation failed for action: {e}")
                    rejected_actions.append({
                        "symbol": action.get("symbol", "UNKNOWN"),
                        "action": action.get("action", "UNKNOWN"),
                        "error": str(e),
                        "status": "ERROR"
                    })

            # Calculate average confidence
            total_actions = len(confirmed_actions) + len(rejected_actions)
            confidence = total_confidence / \
                len(confirmed_actions) if confirmed_actions else 0.0

            result_data = {
                "confirmed_actions": confirmed_actions,
                "rejected_actions": rejected_actions,
                "total_actions": total_actions,
                "confirmed_count": len(confirmed_actions),
                "rejected_count": len(rejected_actions)
            }

            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, result_data, duration_ms)

            execution_time = time.time() - start_time

            return MCPToolResult(
                status="SUCCESS",
                data=result_data,
                confidence=confidence,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "trading_mode": self.trading_mode,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error in confirm tool: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )


# Tool availability flag
CONFIRM_TOOL_AVAILABLE = True
