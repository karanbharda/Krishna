#!/usr/bin/env python3
"""
MCP Risk Management Tool
========================

Professional risk management tool for the Model Context Protocol server
that provides advanced risk assessment and position sizing recommendations.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult

logger = logging.getLogger(__name__)


class RiskManagementTool:
    """
    MCP Risk Management Tool
    Provides professional-grade risk assessment and management recommendations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "risk_management_tool")
        self.default_risk_tolerance = config.get(
            "default_risk_tolerance", 0.02)  # 2% default
        self.max_position_size = config.get(
            "max_position_size", 0.1)  # 10% max
        self.max_portfolio_risk = config.get(
            "max_portfolio_risk", 0.05)  # 5% max portfolio risk

        # Tool interconnections
        self.predict_tool = None
        self.analyze_tool = None

        logger.info(f"Risk Management Tool {self.tool_id} initialized")

    def connect_tools(self, tool_registry: Dict[str, Any]):
        """Connect to other tools for interconnection"""
        if "predict" in tool_registry:
            self.predict_tool = tool_registry["predict"]
        if "analyze" in tool_registry:
            self.analyze_tool = tool_registry["analyze"]
        logger.info(
            f"Risk Management Tool {self.tool_id} connected to other tools")

    async def assess_position_risk(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Assess risk for a specific position

        Args:
            arguments: Tool arguments containing position parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with risk assessment
        """
        start_time = time.time()

        try:
            # Extract parameters
            symbol = arguments.get("symbol")
            position_size = arguments.get("position_size", 0.0)
            entry_price = arguments.get("entry_price", 0.0)
            stop_loss = arguments.get("stop_loss", 0.0)
            portfolio_value = arguments.get("portfolio_value", 100000.0)
            volatility = arguments.get(
                "volatility", 0.2)  # Annualized volatility
            # Confidence level for VaR
            confidence = arguments.get("confidence", 0.95)
            positions = arguments.get("positions", [])

            # If we don't have a specific symbol but have positions, analyze the portfolio
            if not symbol and positions:
                return await self.assess_portfolio_risk(arguments, session_id)

            if not symbol:
                return MCPToolResult(
                    status="ERROR",
                    error="No symbol provided",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Calculate risk metrics
            position_value = position_size * portfolio_value
            max_loss = position_value * \
                (entry_price - stop_loss) / \
                entry_price if entry_price > 0 and stop_loss > 0 else 0.0

            # Calculate Value at Risk (VaR)
            var = self._calculate_var(position_value, volatility, confidence)

            # Calculate position sizing recommendation
            recommended_size = self._calculate_position_size(
                entry_price, stop_loss, portfolio_value, volatility)

            # Risk rating
            risk_rating = self._assess_risk_rating(
                position_size, recommended_size, max_loss, portfolio_value)

            # Risk mitigation suggestions
            mitigation_suggestions = self._generate_mitigation_suggestions(
                position_size, recommended_size, stop_loss, entry_price, volatility)

            # If we have positions, also provide portfolio-level risk assessment
            portfolio_risk = None
            if positions:
                try:
                    portfolio_result = await self.assess_portfolio_risk(
                        {"positions": positions, "portfolio_value": portfolio_value,
                         "confidence": confidence}, session_id)
                    if portfolio_result.status == "SUCCESS":
                        portfolio_risk = portfolio_result.data
                except Exception as portfolio_error:
                    logger.warning(
                        f"Failed to assess portfolio risk: {portfolio_error}")

            execution_time = time.time() - start_time

            result_data = {
                "symbol": symbol,
                "position_size": position_size,
                "position_value": position_value,
                "max_loss": max_loss,
                "value_at_risk": var,
                "recommended_size": recommended_size,
                "risk_rating": risk_rating,
                "mitigation_suggestions": mitigation_suggestions,
                "confidence": confidence
            }

            # Add portfolio risk if available
            if portfolio_risk:
                result_data["portfolio_risk"] = portfolio_risk

            return MCPToolResult(
                status="SUCCESS",
                data=result_data,
                confidence=0.9,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(
                f"Error in position risk assessment: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

    async def assess_portfolio_risk(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Assess overall portfolio risk

        Args:
            arguments: Tool arguments containing portfolio parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with portfolio risk assessment
        """
        start_time = time.time()

        try:
            # Extract parameters
            positions = arguments.get("positions", [])
            portfolio_value = arguments.get("portfolio_value", 100000.0)
            risk_free_rate = arguments.get(
                "risk_free_rate", 0.02)  # 2% default
            confidence = arguments.get("confidence", 0.95)
            time_horizon = arguments.get("time_horizon", 1)  # Days

            if not positions:
                return MCPToolResult(
                    status="ERROR",
                    error="No positions provided",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            # Calculate portfolio metrics
            total_value = sum(pos.get("value", 0.0) for pos in positions)
            weights = [pos.get(
                "value", 0.0) / total_value if total_value > 0 else 0.0 for pos in positions]

            # Calculate portfolio risk metrics
            portfolio_volatility = self._calculate_portfolio_volatility(
                positions, weights)
            portfolio_var = self._calculate_portfolio_var(
                total_value, portfolio_volatility, confidence, time_horizon)
            sharpe_ratio = self._calculate_sharpe_ratio(
                portfolio_volatility, risk_free_rate)
            max_drawdown = self._estimate_max_drawdown(portfolio_volatility)

            # Risk concentration analysis
            concentration_risk = self._analyze_concentration_risk(positions)

            # Portfolio risk rating
            risk_rating = self._assess_portfolio_risk_rating(
                portfolio_volatility, max_drawdown)

            execution_time = time.time() - start_time

            return MCPToolResult(
                status="SUCCESS",
                data={
                    "total_value": total_value,
                    "portfolio_volatility": portfolio_volatility,
                    "value_at_risk": portfolio_var,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "concentration_risk": concentration_risk,
                    "risk_rating": risk_rating,
                    "confidence": confidence,
                    "time_horizon": time_horizon
                },
                confidence=0.95,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "positions_count": len(positions),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(
                f"Error in portfolio risk assessment: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

    def _calculate_var(self, position_value: float, volatility: float, confidence: float) -> float:
        """Calculate Value at Risk for a position"""
        # Simplified VaR calculation using normal distribution
        # In practice, this would use more sophisticated methods
        import scipy.stats as stats
        z_score = stats.norm.ppf(confidence)
        daily_volatility = volatility / (252 ** 0.5)  # Convert annual to daily
        var = position_value * z_score * daily_volatility
        return max(0.0, var)

    def _calculate_position_size(self, entry_price: float, stop_loss: float, portfolio_value: float, volatility: float) -> float:
        """Calculate recommended position size based on risk management principles"""
        if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
            return 0.0

        # Calculate risk per share
        risk_per_share = entry_price - stop_loss

        # Calculate position size based on risk tolerance (default 2% of portfolio)
        max_risk_amount = portfolio_value * self.default_risk_tolerance
        max_shares = max_risk_amount / risk_per_share if risk_per_share > 0 else 0

        # Also consider volatility-adjusted sizing
        # Reduce size for high volatility
        volatility_adjustment = max(0.1, min(1.0, 0.3 / volatility))
        recommended_value = max_shares * entry_price * volatility_adjustment

        # Ensure it doesn't exceed max position size
        recommended_value = min(
            recommended_value, portfolio_value * self.max_position_size)

        return recommended_value / portfolio_value if portfolio_value > 0 else 0.0

    def _assess_risk_rating(self, current_size: float, recommended_size: float, max_loss: float, portfolio_value: float) -> str:
        """Assess risk rating for a position"""
        if current_size <= 0:
            return "NONE"

        # Compare with recommended size
        size_ratio = current_size / \
            recommended_size if recommended_size > 0 else float('inf')

        if size_ratio > 2.0:
            return "VERY_HIGH"
        elif size_ratio > 1.5:
            return "HIGH"
        elif size_ratio > 1.0:
            return "MODERATE"
        elif size_ratio > 0.5:
            return "LOW"
        else:
            return "VERY_LOW"

    def _generate_mitigation_suggestions(
        self,
        current_size: float,
        recommended_size: float,
        stop_loss: float,
        entry_price: float,
        volatility: float
    ) -> List[str]:
        """Generate risk mitigation suggestions"""
        suggestions = []

        if current_size <= 0:
            return suggestions

        size_ratio = current_size / \
            recommended_size if recommended_size > 0 else float('inf')

        if size_ratio > 2.0:
            suggestions.append(
                "Position size is significantly above recommended level. Consider reducing position.")
        elif size_ratio > 1.5:
            suggestions.append(
                "Position size is above recommended level. Consider partial reduction.")

        if stop_loss <= 0:
            suggestions.append(
                "No stop-loss set. Consider setting a stop-loss to limit downside risk.")
        elif entry_price > 0 and (entry_price - stop_loss) / entry_price > 0.1:
            suggestions.append(
                "Stop-loss is relatively wide. Consider tightening to reduce maximum loss.")

        if volatility > 0.3:
            suggestions.append(
                "High volatility stock. Consider reducing position size or using options for protection.")

        return suggestions

    def _calculate_portfolio_volatility(self, positions: List[Dict], weights: List[float]) -> float:
        """Calculate portfolio volatility (simplified)"""
        # In a real implementation, this would use covariance matrix
        # For now, we'll use a weighted average of individual volatilities
        weighted_vol = 0.0
        total_weight = 0.0

        for pos, weight in zip(positions, weights):
            vol = pos.get("volatility", 0.2)  # Default 20% annualized
            weighted_vol += weight * vol
            total_weight += weight

        return weighted_vol / total_weight if total_weight > 0 else 0.2

    def _calculate_portfolio_var(self, portfolio_value: float, volatility: float, confidence: float, time_horizon: int) -> float:
        """Calculate portfolio Value at Risk"""
        import scipy.stats as stats
        z_score = stats.norm.ppf(confidence)
        horiz_volatility = volatility * \
            (time_horizon / 252) ** 0.5  # Scale volatility
        var = portfolio_value * z_score * horiz_volatility
        return max(0.0, var)

    def _calculate_sharpe_ratio(self, volatility: float, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio (simplified)"""
        expected_return = volatility * 1.5  # Simplified assumption
        return (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0

    def _estimate_max_drawdown(self, volatility: float) -> float:
        """Estimate maximum drawdown (simplified)"""
        # Very rough estimation - in practice would use historical data or Monte Carlo
        return volatility * 2.5  # Rough estimate

    def _analyze_concentration_risk(self, positions: List[Dict]) -> Dict[str, Any]:
        """Analyze concentration risk in portfolio"""
        if not positions:
            return {"score": 0.0, "issues": []}

        # Calculate sector concentration
        sector_exposure = {}
        for pos in positions:
            sector = pos.get("sector", "Unknown")
            value = pos.get("value", 0.0)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + value

        total_value = sum(sector_exposure.values())
        if total_value <= 0:
            return {"score": 0.0, "issues": []}

        # Find highest concentration
        max_sector = max(sector_exposure, key=sector_exposure.get)
        max_concentration = sector_exposure[max_sector] / total_value

        issues = []
        if max_concentration > 0.5:
            issues.append(
                f"High concentration in {max_sector} sector ({max_concentration:.1%})")

        # Concentration score (0-1, lower is better)
        # Normalize to 30% threshold
        concentration_score = min(1.0, max_concentration / 0.3)

        return {
            "score": concentration_score,
            "max_concentration": max_concentration,
            "max_sector": max_sector,
            "issues": issues
        }

    def _assess_portfolio_risk_rating(self, volatility: float, max_drawdown: float) -> str:
        """Assess overall portfolio risk rating"""
        # Combined assessment based on volatility and drawdown
        if volatility > 0.25 or max_drawdown > 0.3:
            return "VERY_HIGH"
        elif volatility > 0.20 or max_drawdown > 0.25:
            return "HIGH"
        elif volatility > 0.15 or max_drawdown > 0.20:
            return "MODERATE"
        elif volatility > 0.10 or max_drawdown > 0.15:
            return "LOW"
        else:
            return "VERY_LOW"


# Tool availability flag
RISK_MANAGEMENT_TOOL_AVAILABLE = True
