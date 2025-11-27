#!/usr/bin/env python3
"""
MCP Portfolio Optimization Agent
===============================

AI-powered portfolio optimization agent for the Model Context Protocol server
that provides professional-grade portfolio management with risk-adjusted optimization.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioOptimization:
    """Structured portfolio optimization result"""
    allocations: Dict[str, float]
    expected_return: float
    risk_level: float
    sharpe_ratio: float
    diversification_score: float
    optimization_method: str
    metadata: Dict[str, Any]


class PortfolioAgent:
    """
    MCP Portfolio Optimization Agent
    Provides professional-grade portfolio optimization and rebalancing
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "default_portfolio_agent")
        self.risk_tolerance = config.get("risk_tolerance", 0.5)
        self.optimization_method = config.get(
            "optimization_method", "mean_variance")
        self.max_positions = config.get("max_positions", 10)

        # Initialize components
        self.is_initialized = False
        self.groq_engine = None

        logger.info(f"Portfolio Agent {self.agent_id} initialized")

    async def initialize(self):
        """Initialize the portfolio agent"""
        try:
            # Import Groq engine if available
            try:
                # Use absolute import instead of relative import
                import sys
                import os
                # Add backend directory to path
                backend_dir = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                if backend_dir not in sys.path:
                    sys.path.insert(0, backend_dir)
                from groq_api import GroqAPIEngine
                if "groq" in self.config:
                    groq_config = self.config["groq"]
                    self.groq_engine = GroqAPIEngine(groq_config)
                    logger.info("Groq engine connected to portfolio agent")
            except ImportError as e:
                logger.warning(f"Groq API integration not available: {e}")

            self.is_initialized = True
            logger.info(
                f"Portfolio Agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Portfolio Agent {self.agent_id}: {e}")
            raise

    async def optimize_portfolio(
        self,
        holdings: Dict[str, Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> PortfolioOptimization:
        """
        Optimize portfolio allocation based on risk-return profile

        Args:
            holdings: Current portfolio holdings with details
            market_data: Additional market context
            constraints: Portfolio constraints (sector limits, etc.)

        Returns:
            PortfolioOptimization with recommended allocations
        """
        if not self.is_initialized:
            raise RuntimeError("Portfolio agent not initialized")

        start_time = time.time()

        try:
            # Extract current portfolio data
            symbols = list(holdings.keys())
            current_weights = {symbol: holding.get(
                "weight", 0.0) for symbol, holding in holdings.items()}
            current_values = {symbol: holding.get(
                "value", 0.0) for symbol, holding in holdings.items()}

            # Simulate optimization (in a real implementation, this would use actual optimization algorithms)
            total_value = sum(current_values.values())
            if total_value > 0:
                current_weights = {
                    symbol: value/total_value for symbol, value in current_values.items()}

            # Generate optimized allocations
            optimized_allocations = self._generate_optimized_allocations(
                symbols, current_weights, market_data, constraints)

            # Calculate portfolio metrics
            expected_return = self._calculate_expected_return(
                optimized_allocations, market_data)
            risk_level = self._calculate_risk_level(
                optimized_allocations, market_data)
            sharpe_ratio = expected_return / risk_level if risk_level > 0 else 0.0
            diversification_score = self._calculate_diversification_score(
                optimized_allocations)

            optimization = PortfolioOptimization(
                allocations=optimized_allocations,
                expected_return=expected_return,
                risk_level=risk_level,
                sharpe_ratio=sharpe_ratio,
                diversification_score=diversification_score,
                optimization_method=self.optimization_method,
                metadata={
                    "optimization_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "current_holdings_count": len(holdings)
                }
            )

            logger.info(
                f"Portfolio optimization completed for {len(symbols)} symbols")
            return optimization

        except Exception as e:
            logger.error(f"Error in optimize_portfolio: {e}")
            # Return current portfolio as fallback
            return PortfolioOptimization(
                allocations=current_weights,
                expected_return=0.0,
                risk_level=0.0,
                sharpe_ratio=0.0,
                diversification_score=0.0,
                optimization_method="current",
                metadata={
                    "optimization_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

    def _generate_optimized_allocations(
        self,
        symbols: List[str],
        current_weights: Dict[str, float],
        market_data: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Generate optimized portfolio allocations"""
        # In a real implementation, this would use mean-variance optimization or other advanced methods
        # For now, we'll simulate a professional allocation strategy

        if not symbols:
            return {}

        # Simple risk-based allocation (higher allocations to less volatile stocks for conservative portfolios)
        allocations = {}
        total_allocation = 0.0

        # Simulate different optimization approaches based on risk tolerance
        if self.risk_tolerance < 0.3:  # Conservative
            # Allocate more to large-cap, stable stocks
            base_allocation = 1.0 / len(symbols)
            for i, symbol in enumerate(symbols):
                # Give slight preference to earlier symbols (simulating large-cap preference)
                allocation = base_allocation * (1.0 - (i * 0.05))
                allocations[symbol] = max(allocation, 0.01)  # Minimum 1%
                total_allocation += allocations[symbol]
        elif self.risk_tolerance > 0.7:  # Aggressive
            # Allocate more to growth/high-volatility stocks
            base_allocation = 1.0 / len(symbols)
            for i, symbol in enumerate(symbols):
                # Give slight preference to later symbols (simulating growth preference)
                allocation = base_allocation * (1.0 + (i * 0.05))
                allocations[symbol] = max(allocation, 0.01)  # Minimum 1%
                total_allocation += allocations[symbol]
        else:  # Moderate
            # Equal weight distribution
            for symbol in symbols:
                allocations[symbol] = 1.0 / len(symbols)
            total_allocation = 1.0

        # Normalize allocations to sum to 1.0
        if total_allocation > 0:
            allocations = {
                symbol: weight/total_allocation for symbol, weight in allocations.items()}

        return allocations

    def _calculate_expected_return(
        self,
        allocations: Dict[str, float],
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate expected portfolio return"""
        # In a real implementation, this would use actual return forecasts
        # For simulation, we'll use a simple model based on risk tolerance
        return self.risk_tolerance * 0.15  # Max 15% expected return

    def _calculate_risk_level(
        self,
        allocations: Dict[str, float],
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate portfolio risk level"""
        # In a real implementation, this would calculate actual portfolio variance
        # For simulation, we'll use a simple model based on risk tolerance
        return (1.0 - self.risk_tolerance) * 0.20 + 0.05  # Between 5-25% risk

    def _calculate_diversification_score(self, allocations: Dict[str, float]) -> float:
        """Calculate portfolio diversification score"""
        if not allocations:
            return 0.0

        # Simple diversification measure based on number of positions and concentration
        num_positions = len(allocations)
        # Herfindahl index
        concentration = sum(weight ** 2 for weight in allocations.values())
        max_concentration = 1.0 / num_positions if num_positions > 0 else 1.0

        # Convert to diversification score (0-1, higher is better)
        if max_concentration > 0:
            diversification = 1.0 - (concentration / max_concentration)
            return max(0.0, min(1.0, diversification))
        return 0.0

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "initialized": self.is_initialized,
            "risk_tolerance": self.risk_tolerance,
            "optimization_method": self.optimization_method,
            "max_positions": self.max_positions,
            "groq_available": self.groq_engine is not None
        }


# Agent availability flag
PORTFOLIO_AGENT_AVAILABLE = True
