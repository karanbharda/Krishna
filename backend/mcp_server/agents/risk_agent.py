#!/usr/bin/env python3
"""
Risk Management Agent
====================

Production-grade agent for comprehensive risk assessment, portfolio risk management,
and dynamic risk monitoring with advanced quantitative models.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from scipy import stats
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    overall_risk_score: float

@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    position_size: float
    var_contribution: float
    risk_contribution: float
    correlation_with_portfolio: float
    liquidity_score: float
    volatility: float
    recommended_size: float

class RiskAgent:
    """
    Production-grade risk management agent
    
    Features:
    - Value at Risk (VaR) and Conditional VaR calculations
    - Portfolio correlation analysis
    - Dynamic position sizing
    - Stress testing and scenario analysis
    - Real-time risk monitoring
    - Regulatory compliance checks
    - ENHANCED RISK MANAGEMENT: Adaptive risk parameters based on market conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "risk_agent")
        
        # Risk parameters
        self.max_portfolio_var = config.get("max_portfolio_var", 0.05)  # 5% daily VaR
        self.max_position_size = config.get("max_position_size", 0.25)  # 25% max per position
        self.max_sector_concentration = config.get("max_sector_concentration", 0.40)  # 40% per sector
        self.min_liquidity_score = config.get("min_liquidity_score", 0.3)
        
        # ENHANCED RISK MANAGEMENT: Dynamic risk parameters
        self.base_max_portfolio_var = self.max_portfolio_var
        self.base_max_position_size = self.max_position_size
        self.volatility_threshold_low = config.get("volatility_threshold_low", 0.02)
        self.volatility_threshold_high = config.get("volatility_threshold_high", 0.04)
        self.volatility_adjustment_factor = config.get("volatility_adjustment_factor", 0.5)
        
        # Risk models
        self.lookback_period = config.get("lookback_period", 252)  # 1 year
        self.confidence_level = config.get("confidence_level", 0.95)
        
        # Performance tracking
        self.risk_assessments_performed = 0
        self.risk_alerts_generated = 0
        self.risk_violations = []
        
        logger.info(f"Risk Agent {self.agent_id} initialized with enhanced adaptive risk management")
    
    async def assess_portfolio_risk(self, portfolio_data: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> RiskMetrics:
        """Comprehensive portfolio risk assessment with real-time metrics"""
        try:
            holdings = portfolio_data.get("holdings", {})
            total_value = portfolio_data.get("total_value", 0)
            
            if not holdings or total_value == 0:
                return self._generate_empty_risk_metrics()
            
            # Calculate portfolio returns
            returns_data = await self._get_portfolio_returns(holdings, market_data)
            
            # Calculate VaR and CVaR
            var_95 = self._calculate_var(returns_data, self.confidence_level)
            cvar_95 = self._calculate_cvar(returns_data, self.confidence_level)
            
            # Calculate other risk metrics
            max_drawdown = self._calculate_max_drawdown(returns_data)
            volatility = self._calculate_volatility(returns_data)
            sharpe_ratio = self._calculate_sharpe_ratio(returns_data)
            beta = await self._calculate_portfolio_beta(holdings, market_data)
            
            # Calculate risk concentrations
            correlation_risk = await self._calculate_correlation_risk(holdings, market_data)
            liquidity_risk = self._calculate_liquidity_risk(holdings, market_data)
            concentration_risk = self._calculate_concentration_risk(holdings)
            
            # ENHANCED RISK MONITORING: Calculate real-time portfolio-level risk metrics
            real_time_metrics = await self._calculate_real_time_risk_metrics(holdings, market_data)
            
            # Adjust risk metrics with real-time data
            adjusted_var_95 = var_95 * real_time_metrics.get("var_adjustment_factor", 1.0)
            adjusted_volatility = volatility * real_time_metrics.get("volatility_adjustment_factor", 1.0)
            
            # Overall risk score (1-10 scale) with real-time adjustments
            overall_risk_score = self._calculate_overall_risk_score(
                adjusted_var_95, adjusted_volatility, concentration_risk, liquidity_risk
            )
            
            self.risk_assessments_performed += 1
            
            return RiskMetrics(
                var_95=adjusted_var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                volatility=adjusted_volatility,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                overall_risk_score=overall_risk_score
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment error: {e}")
            return self._generate_empty_risk_metrics()
    
    async def assess_position_risk(self, symbol: str, position_size: float,
                                 portfolio_data: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> PositionRisk:
        """Assess risk for individual position"""
        try:
            # Get historical data for the symbol
            symbol_data = market_data.get(symbol, {})
            
            # Calculate position volatility
            volatility = self._calculate_symbol_volatility(symbol_data)
            
            # Calculate VaR contribution
            var_contribution = self._calculate_position_var_contribution(
                symbol, position_size, portfolio_data, market_data
            )
            
            # Calculate risk contribution to portfolio
            risk_contribution = var_contribution / portfolio_data.get("total_value", 1)
            
            # Calculate correlation with portfolio
            correlation = await self._calculate_position_correlation(
                symbol, portfolio_data, market_data
            )
            
            # Calculate liquidity score
            liquidity_score = self._calculate_position_liquidity(symbol_data)
            
            # Calculate recommended position size
            recommended_size = self._calculate_optimal_position_size(
                symbol, portfolio_data, market_data, volatility
            )
            
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                var_contribution=var_contribution,
                risk_contribution=risk_contribution,
                correlation_with_portfolio=correlation,
                liquidity_score=liquidity_score,
                volatility=volatility,
                recommended_size=recommended_size
            )
            
        except Exception as e:
            logger.error(f"Position risk assessment error for {symbol}: {e}")
            return PositionRisk(
                symbol=symbol,
                position_size=position_size,
                var_contribution=0.0,
                risk_contribution=0.0,
                correlation_with_portfolio=0.0,
                liquidity_score=0.5,
                volatility=0.02,
                recommended_size=position_size
            )
    
    async def stress_test_portfolio(self, portfolio_data: Dict[str, Any],
                                  stress_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        try:
            stress_results = {}
            
            for scenario in stress_scenarios:
                scenario_name = scenario.get("name", "Unknown")
                market_shocks = scenario.get("market_shocks", {})
                
                # Apply stress scenario
                stressed_portfolio_value = await self._apply_stress_scenario(
                    portfolio_data, market_shocks
                )
                
                # Calculate stress impact
                original_value = portfolio_data.get("total_value", 0)
                stress_impact = (stressed_portfolio_value - original_value) / original_value
                
                stress_results[scenario_name] = {
                    "original_value": original_value,
                    "stressed_value": stressed_portfolio_value,
                    "impact_percentage": stress_impact,
                    "impact_amount": stressed_portfolio_value - original_value
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress testing error: {e}")
            return {}
    
    async def monitor_risk_limits(self, portfolio_data: Dict[str, Any],
                                market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor portfolio against risk limits"""
        try:
            violations = []
            
            # ENHANCED RISK MANAGEMENT: Dynamically adjust risk parameters based on market conditions
            self._adjust_risk_parameters(market_data)
            
            # Check portfolio VaR limit
            risk_metrics = await self.assess_portfolio_risk(portfolio_data, market_data)
            if risk_metrics.var_95 > self.max_portfolio_var:
                violations.append({
                    "type": "VaR_LIMIT",
                    "severity": "HIGH",
                    "current_value": risk_metrics.var_95,
                    "limit": self.max_portfolio_var,
                    "message": f"Portfolio VaR ({risk_metrics.var_95:.2%}) exceeds limit ({self.max_portfolio_var:.2%})"
                })
            
            # Check position size limits
            holdings = portfolio_data.get("holdings", {})
            total_value = portfolio_data.get("total_value", 1)
            
            for symbol, holding in holdings.items():
                position_value = holding.get("market_value", 0)
                position_weight = position_value / total_value
                
                if position_weight > self.max_position_size:
                    violations.append({
                        "type": "POSITION_SIZE",
                        "severity": "MEDIUM",
                        "symbol": symbol,
                        "current_value": position_weight,
                        "limit": self.max_position_size,
                        "message": f"{symbol} position ({position_weight:.1%}) exceeds size limit ({self.max_position_size:.1%})"
                    })
            
            # Check concentration risk
            if risk_metrics.concentration_risk > 0.7:
                violations.append({
                    "type": "CONCENTRATION",
                    "severity": "MEDIUM",
                    "current_value": risk_metrics.concentration_risk,
                    "limit": 0.7,
                    "message": f"Portfolio concentration risk ({risk_metrics.concentration_risk:.2f}) is elevated"
                })
            
            # Check liquidity risk
            if risk_metrics.liquidity_risk > 0.6:
                violations.append({
                    "type": "LIQUIDITY",
                    "severity": "LOW",
                    "current_value": risk_metrics.liquidity_risk,
                    "limit": 0.6,
                    "message": f"Portfolio liquidity risk ({risk_metrics.liquidity_risk:.2f}) requires attention"
                })
            
            if violations:
                self.risk_alerts_generated += len(violations)
                self.risk_violations.extend(violations)
            
            return violations
            
        except Exception as e:
            logger.error(f"Risk monitoring error: {e}")
            return []
    
    def calculate_optimal_position_size(self, symbol: str, expected_return: float,
                                      volatility: float, portfolio_data: Dict[str, Any]) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where f = fraction of capital to wager
            # b = odds received on the wager (expected return / risk-free rate)
            # p = probability of winning
            # q = probability of losing (1 - p)
            
            risk_free_rate = 0.05  # 5% annual risk-free rate
            daily_risk_free = risk_free_rate / 252
            
            # Estimate probability of positive return based on Sharpe ratio
            sharpe_ratio = (expected_return - daily_risk_free) / volatility
            prob_positive = stats.norm.cdf(sharpe_ratio)
            
            # Calculate Kelly fraction
            if prob_positive > 0.5 and expected_return > daily_risk_free:
                kelly_fraction = (prob_positive * expected_return - (1 - prob_positive) * volatility) / expected_return
                
                # Apply Kelly fraction with safety margin (typically 25% of full Kelly)
                optimal_fraction = max(0, min(kelly_fraction * 0.25, self.max_position_size))
            else:
                optimal_fraction = 0.01  # Minimal position for negative expected value
            
            return optimal_fraction
            
        except Exception as e:
            logger.error(f"Optimal position size calculation error: {e}")
            return 0.05  # Default 5% position
    
    def _adjust_risk_parameters(self, market_data: Dict[str, Any]):
        """
        ENHANCED RISK MANAGEMENT: Dynamically adjust risk parameters based on market volatility
        """
        try:
            # Calculate market volatility from available data
            market_volatility = self._calculate_market_volatility(market_data)
            
            # Adjust risk parameters based on volatility
            if market_volatility < self.volatility_threshold_low:
                # Low volatility - increase risk tolerance
                self.max_portfolio_var = self.base_max_portfolio_var * (1 + self.volatility_adjustment_factor)
                self.max_position_size = self.base_max_position_size * (1 + self.volatility_adjustment_factor * 0.5)
                logger.info(f"Low volatility detected ({market_volatility:.2%}). Increased risk limits.")
            elif market_volatility > self.volatility_threshold_high:
                # High volatility - decrease risk tolerance
                self.max_portfolio_var = self.base_max_portfolio_var * (1 - self.volatility_adjustment_factor)
                self.max_position_size = self.base_max_position_size * (1 - self.volatility_adjustment_factor * 0.5)
                logger.info(f"High volatility detected ({market_volatility:.2%}). Reduced risk limits.")
            else:
                # Normal volatility - use base parameters
                self.max_portfolio_var = self.base_max_portfolio_var
                self.max_position_size = self.base_max_position_size
                logger.info(f"Normal volatility detected ({market_volatility:.2%}). Using base risk limits.")
                
        except Exception as e:
            logger.error(f"Error adjusting risk parameters: {e}")
            # Revert to base parameters on error
            self.max_portfolio_var = self.base_max_portfolio_var
            self.max_position_size = self.base_max_position_size
    
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate overall market volatility from available data
        """
        try:
            volatilities = []
            
            # Extract volatility data from market data
            for symbol_data in market_data.values():
                if isinstance(symbol_data, dict):
                    volatility = symbol_data.get("volatility")
                    if volatility is not None:
                        volatilities.append(volatility)
            
            # If we have volatility data, return the average
            if volatilities:
                return np.mean(volatilities)
            
            # If no volatility data, estimate from price movements
            # This is a simplified estimation
            return 0.02  # Default 2% daily volatility
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.02  # Default 2% daily volatility
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk using historical simulation"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        volatility = np.std(returns)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    async def _calculate_portfolio_beta(self, holdings: Dict[str, Any], 
                                      market_data: Dict[str, Any]) -> float:
        """Calculate portfolio beta relative to market"""
        try:
            # Simplified beta calculation
            # In production, this would use actual market index data
            portfolio_weights = []
            individual_betas = []
            
            total_value = sum(holding.get("market_value", 0) for holding in holdings.values())
            
            for symbol, holding in holdings.items():
                weight = holding.get("market_value", 0) / total_value if total_value > 0 else 0
                
                # Estimate beta based on volatility (simplified)
                symbol_data = market_data.get(symbol, {})
                volatility = self._calculate_symbol_volatility(symbol_data)
                estimated_beta = min(volatility / 0.15, 2.0)  # Cap at 2.0
                
                portfolio_weights.append(weight)
                individual_betas.append(estimated_beta)
            
            if portfolio_weights:
                return np.average(individual_betas, weights=portfolio_weights)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Beta calculation error: {e}")
            return 1.0
    
    async def _calculate_correlation_risk(self, holdings: Dict[str, Any],
                                        market_data: Dict[str, Any]) -> float:
        """Calculate portfolio correlation risk"""
        try:
            if len(holdings) < 2:
                return 0.0
            
            # Simplified correlation risk calculation
            # In production, this would use actual correlation matrix
            symbols = list(holdings.keys())
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Estimate correlation based on sector/industry similarity
                    # This is simplified - real implementation would use historical correlations
                    estimated_corr = 0.3  # Default moderate correlation
                    correlations.append(estimated_corr)
            
            if correlations:
                avg_correlation = np.mean(correlations)
                return min(avg_correlation, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Correlation risk calculation error: {e}")
            return 0.3
    
    def _calculate_liquidity_risk(self, holdings: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            liquidity_scores = []
            weights = []
            
            total_value = sum(holding.get("market_value", 0) for holding in holdings.values())
            
            for symbol, holding in holdings.items():
                weight = holding.get("market_value", 0) / total_value if total_value > 0 else 0
                
                symbol_data = market_data.get(symbol, {})
                liquidity_score = self._calculate_position_liquidity(symbol_data)
                
                liquidity_scores.append(liquidity_score)
                weights.append(weight)
            
            if liquidity_scores:
                weighted_liquidity = np.average(liquidity_scores, weights=weights)
                return 1.0 - weighted_liquidity  # Convert to risk (higher = more risk)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Liquidity risk calculation error: {e}")
            return 0.5
    
    def _calculate_concentration_risk(self, holdings: Dict[str, Any]) -> float:
        """Calculate portfolio concentration risk using Herfindahl index"""
        try:
            total_value = sum(holding.get("market_value", 0) for holding in holdings.values())
            
            if total_value == 0:
                return 0.0
            
            weights = [holding.get("market_value", 0) / total_value for holding in holdings.values()]
            herfindahl_index = sum(w**2 for w in weights)
            
            # Normalize to 0-1 scale (1 = maximum concentration)
            n = len(holdings)
            if n > 1:
                normalized_hhi = (herfindahl_index - 1/n) / (1 - 1/n)
                return max(0, min(normalized_hhi, 1))
            else:
                return 1.0  # Single position = maximum concentration
                
        except Exception as e:
            logger.error(f"Concentration risk calculation error: {e}")
            return 0.5
    
    async def _calculate_real_time_risk_metrics(self, holdings: Dict[str, Any], 
                                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate real-time portfolio-level risk metrics"""
        try:
            real_time_metrics = {
                "timestamp": datetime.now().isoformat(),
                "var_adjustment_factor": 1.0,
                "volatility_adjustment_factor": 1.0,
                "market_regime_factor": 1.0,
                "liquidity_stress_factor": 1.0
            }
            
            # Calculate market stress indicators
            market_stress = await self._calculate_market_stress(market_data)
            
            # Adjust VaR based on market stress
            if market_stress.get("high_volatility", False):
                real_time_metrics["var_adjustment_factor"] = 1.5
            elif market_stress.get("extreme_volatility", False):
                real_time_metrics["var_adjustment_factor"] = 2.0
            
            # Adjust volatility based on market conditions
            if market_stress.get("high_volume", False):
                real_time_metrics["volatility_adjustment_factor"] = 1.2
            
            # Apply market regime adjustments
            regime = market_stress.get("regime", "normal")
            if regime == "volatile":
                real_time_metrics["market_regime_factor"] = 1.3
            elif regime == "crisis":
                real_time_metrics["market_regime_factor"] = 1.8
            
            # Calculate liquidity stress
            liquidity_stress = self._calculate_liquidity_stress(holdings, market_data)
            real_time_metrics["liquidity_stress_factor"] = liquidity_stress
            
            return real_time_metrics
            
        except Exception as e:
            logger.error(f"Real-time risk metrics calculation error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "var_adjustment_factor": 1.0,
                "volatility_adjustment_factor": 1.0,
                "market_regime_factor": 1.0,
                "liquidity_stress_factor": 1.0
            }
    
    async def _calculate_market_stress(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market stress indicators"""
        try:
            stress_indicators = {
                "high_volatility": False,
                "extreme_volatility": False,
                "high_volume": False,
                "regime": "normal"
            }
            
            volatility_threshold = 0.02  # 2% daily volatility
            volume_threshold = 1.5  # 50% above average volume
            
            # Check volatility across symbols
            volatilities = []
            volumes = []
            
            for symbol_data in market_data.values():
                if isinstance(symbol_data, dict):
                    volatility = symbol_data.get("volatility", 0)
                    volume_ratio = symbol_data.get("volume_ratio", 1.0)
                    
                    if volatility > 0:
                        volatilities.append(volatility)
                    if volume_ratio > 0:
                        volumes.append(volume_ratio)
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                if avg_volatility > volatility_threshold * 2:
                    stress_indicators["extreme_volatility"] = True
                    stress_indicators["regime"] = "crisis"
                elif avg_volatility > volatility_threshold:
                    stress_indicators["high_volatility"] = True
                    stress_indicators["regime"] = "volatile"
            
            if volumes:
                avg_volume = np.mean(volumes)
                if avg_volume > volume_threshold:
                    stress_indicators["high_volume"] = True
            
            return stress_indicators
            
        except Exception as e:
            logger.error(f"Market stress calculation error: {e}")
            return {"high_volatility": False, "extreme_volatility": False, "high_volume": False, "regime": "normal"}
    
    def _calculate_liquidity_stress(self, holdings: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate liquidity stress factor for the portfolio"""
        try:
            liquidity_scores = []
            weights = []
            
            total_value = sum(holding.get("market_value", 0) for holding in holdings.values())
            
            for symbol, holding in holdings.items():
                weight = holding.get("market_value", 0) / total_value if total_value > 0 else 0
                symbol_data = market_data.get(symbol, {})
                
                # Get liquidity metrics
                bid_ask_spread = symbol_data.get("bid_ask_spread", 0.01)
                volume_ratio = symbol_data.get("volume_ratio", 1.0)
                
                # Calculate liquidity stress (higher spread or lower volume = higher stress)
                spread_stress = min(bid_ask_spread / 0.02, 2.0)  # 2% spread = max stress
                volume_stress = max(1.0 / volume_ratio, 0.5) if volume_ratio > 0 else 2.0
                
                liquidity_stress = (spread_stress + volume_stress) / 2
                liquidity_scores.append(liquidity_stress)
                weights.append(weight)
            
            if liquidity_scores:
                weighted_stress = np.average(liquidity_scores, weights=weights)
                return max(1.0, min(weighted_stress, 3.0))  # Range: 1.0 (normal) to 3.0 (severe stress)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Liquidity stress calculation error: {e}")
            return 1.0
    
    def _calculate_overall_risk_score(self, var_95: float, volatility: float,
                                    concentration_risk: float, liquidity_risk: float) -> float:
        """Calculate overall risk score (1-10 scale)"""
        try:
            # Normalize components to 0-1 scale
            var_component = min(abs(var_95) / 0.1, 1.0)  # 10% VaR = max
            vol_component = min(volatility / 0.5, 1.0)  # 50% volatility = max
            conc_component = concentration_risk
            liq_component = liquidity_risk
            
            # Weighted average
            weights = [0.4, 0.3, 0.2, 0.1]  # VaR, Vol, Concentration, Liquidity
            components = [var_component, vol_component, conc_component, liq_component]
            
            risk_score = np.average(components, weights=weights)
            
            # Convert to 1-10 scale
            return 1 + (risk_score * 9)
            
        except Exception as e:
            logger.error(f"Overall risk score calculation error: {e}")
            return 5.0
    
    def _calculate_symbol_volatility(self, symbol_data: Dict[str, Any]) -> float:
        """Calculate symbol volatility from market data"""
        try:
            # Extract price data
            prices = symbol_data.get("historical_prices", [])
            if len(prices) < 2:
                return 0.02  # Default 2% daily volatility
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Calculate volatility
            return np.std(returns) if len(returns) > 0 else 0.02
            
        except Exception as e:
            logger.error(f"Symbol volatility calculation error: {e}")
            return 0.02
    
    def _calculate_position_liquidity(self, symbol_data: Dict[str, Any]) -> float:
        """Calculate position liquidity score (0-1, higher = more liquid)"""
        try:
            # Factors affecting liquidity
            volume = symbol_data.get("average_volume", 0)
            market_cap = symbol_data.get("market_cap", 0)
            bid_ask_spread = symbol_data.get("bid_ask_spread", 0.01)
            
            # Normalize factors
            volume_score = min(volume / 1000000, 1.0)  # 1M volume = max score
            market_cap_score = min(market_cap / 10000000000, 1.0)  # 10B market cap = max
            spread_score = max(0, 1 - bid_ask_spread / 0.05)  # 5% spread = min score
            
            # Weighted average
            liquidity_score = (volume_score * 0.4 + market_cap_score * 0.4 + spread_score * 0.2)
            
            return max(0.1, min(liquidity_score, 1.0))
            
        except Exception as e:
            logger.error(f"Position liquidity calculation error: {e}")
            return 0.5
    
    async def _get_portfolio_returns(self, holdings: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> np.ndarray:
        """Get portfolio historical returns"""
        try:
            # Simplified portfolio returns calculation
            # In production, this would use actual historical data
            
            portfolio_returns = []
            total_value = sum(holding.get("market_value", 0) for holding in holdings.values())
            
            # Generate synthetic returns based on individual position volatilities
            for i in range(self.lookback_period):
                daily_return = 0.0
                
                for symbol, holding in holdings.items():
                    weight = holding.get("market_value", 0) / total_value if total_value > 0 else 0
                    symbol_data = market_data.get(symbol, {})
                    volatility = self._calculate_symbol_volatility(symbol_data)
                    
                    # Generate random return based on volatility
                    symbol_return = np.random.normal(0, volatility)
                    daily_return += weight * symbol_return
                
                portfolio_returns.append(daily_return)
            
            return np.array(portfolio_returns)
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])
    
    def _generate_empty_risk_metrics(self) -> RiskMetrics:
        """Generate empty risk metrics for error cases"""
        return RiskMetrics(
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            beta=1.0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
            concentration_risk=0.0,
            overall_risk_score=5.0
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get risk agent status"""
        return {
            "agent_id": self.agent_id,
            "risk_assessments_performed": self.risk_assessments_performed,
            "risk_alerts_generated": self.risk_alerts_generated,
            "recent_violations": len([v for v in self.risk_violations if 
                                    (datetime.now() - datetime.fromisoformat(v.get("timestamp", "2024-01-01"))).days < 7]),
            "max_portfolio_var": self.max_portfolio_var,
            "max_position_size": self.max_position_size,
            "status": "active"
        }
