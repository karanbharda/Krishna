"""
Performance Attribution System
Provides detailed analysis of what drives trading returns and performance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceAttribution:
    """Detailed performance attribution analysis"""
    total_return: float
    timestamp: datetime
    components: Dict[str, float]
    weights: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any]

class PerformanceAttributionEngine:
    """Engine for analyzing and attributing trading performance"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.attribution_history = []
        self.factor_models = {}
        self.performance_data = []
        
        # Performance attribution components
        self.components = [
            "market_returns",
            "sector_returns",
            "style_factors",
            "alpha_generation",
            "risk_management",
            "transaction_costs",
            "timing_effects",
            "sentiment_impact",
            "regime_adjustment"
        ]
        
        logger.info("Performance attribution engine initialized")
    
    def analyze_performance(self, portfolio_data: Dict[str, Any], 
                          market_data: Dict[str, Any],
                          period: str = "daily") -> PerformanceAttribution:
        """
        Analyze portfolio performance and attribute to various factors
        
        Args:
            portfolio_data: Portfolio holdings and performance data
            market_data: Market data and benchmarks
            period: Analysis period (daily, weekly, monthly)
            
        Returns:
            PerformanceAttribution object with detailed analysis
        """
        try:
            # Calculate total portfolio return
            total_return = self._calculate_total_return(portfolio_data, period)
            
            # Calculate component contributions
            components = self._calculate_component_contributions(
                portfolio_data, market_data, period
            )
            
            # Calculate component weights
            weights = self._calculate_component_weights(components, total_return)
            
            # Calculate confidence score
            confidence = self._calculate_attribution_confidence(components)
            
            # Create attribution object
            attribution = PerformanceAttribution(
                total_return=total_return,
                timestamp=datetime.now(),
                components=components,
                weights=weights,
                confidence=confidence,
                metadata={
                    "period": period,
                    "analysis_time": datetime.now().isoformat(),
                    "portfolio_value": portfolio_data.get("total_value", 0)
                }
            )
            
            # Store attribution
            self.attribution_history.append(attribution)
            
            # Keep only recent history
            if len(self.attribution_history) > 1000:
                self.attribution_history = self.attribution_history[-1000:]
            
            return attribution
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return PerformanceAttribution(
                total_return=0.0,
                timestamp=datetime.now(),
                components={},
                weights={},
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def _calculate_total_return(self, portfolio_data: Dict[str, Any], period: str) -> float:
        """Calculate total portfolio return for the period"""
        try:
            # This would use actual portfolio performance data
            # For now, we'll simulate based on holdings
            holdings = portfolio_data.get("holdings", {})
            total_value = portfolio_data.get("total_value", 0)
            
            if not holdings or total_value == 0:
                return 0.0
            
            # Simulate return based on weighted average of holding returns
            weighted_return = 0.0
            total_weight = 0.0
            
            for symbol, holding in holdings.items():
                weight = holding.get("market_value", 0) / total_value
                # Simulate return (in real implementation, this would come from actual data)
                return_value = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                
                weighted_return += weight * return_value
                total_weight += weight
            
            return weighted_return if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Total return calculation error: {e}")
            return 0.0
    
    def _calculate_component_contributions(self, portfolio_data: Dict[str, Any], 
                                         market_data: Dict[str, Any], 
                                         period: str) -> Dict[str, float]:
        """Calculate contributions from different performance components"""
        try:
            components = {}
            
            # Market returns (beta * market return)
            market_return = self._calculate_market_contribution(portfolio_data, market_data)
            components["market_returns"] = market_return
            
            # Sector returns
            sector_return = self._calculate_sector_contribution(portfolio_data, market_data)
            components["sector_returns"] = sector_return
            
            # Style factors (value, momentum, quality, etc.)
            style_return = self._calculate_style_contribution(portfolio_data, market_data)
            components["style_factors"] = style_return
            
            # Alpha generation (active return from stock selection)
            alpha_return = self._calculate_alpha_contribution(portfolio_data, market_data)
            components["alpha_generation"] = alpha_return
            
            # Risk management impact
            risk_return = self._calculate_risk_management_contribution(portfolio_data)
            components["risk_management"] = risk_return
            
            # Transaction costs impact
            cost_return = self._calculate_transaction_cost_impact(portfolio_data)
            components["transaction_costs"] = cost_return
            
            # Timing effects
            timing_return = self._calculate_timing_contribution(portfolio_data)
            components["timing_effects"] = timing_return
            
            # Sentiment impact
            sentiment_return = self._calculate_sentiment_impact(portfolio_data, market_data)
            components["sentiment_impact"] = sentiment_return
            
            # Regime adjustment
            regime_return = self._calculate_regime_adjustment(portfolio_data, market_data)
            components["regime_adjustment"] = regime_return
            
            return components
            
        except Exception as e:
            logger.error(f"Component contribution calculation error: {e}")
            return {component: 0.0 for component in self.components}
    
    def _calculate_market_contribution(self, portfolio_data: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> float:
        """Calculate contribution from market returns"""
        try:
            # Simplified calculation: portfolio beta * market return
            portfolio_beta = portfolio_data.get("portfolio_beta", 1.0)
            market_return = market_data.get("market_return", 0.001)  # 0.1% default
            
            return portfolio_beta * market_return
        except Exception as e:
            logger.error(f"Market contribution calculation error: {e}")
            return 0.0
    
    def _calculate_sector_contribution(self, portfolio_data: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> float:
        """Calculate contribution from sector returns"""
        try:
            # Simplified calculation based on sector allocations
            holdings = portfolio_data.get("holdings", {})
            sector_returns = market_data.get("sector_returns", {})
            
            total_contribution = 0.0
            total_value = portfolio_data.get("total_value", 1)
            
            for symbol, holding in holdings.items():
                sector = holding.get("sector", "unknown")
                weight = holding.get("market_value", 0) / total_value
                sector_return = sector_returns.get(sector, 0.0)
                
                total_contribution += weight * sector_return
            
            return total_contribution
        except Exception as e:
            logger.error(f"Sector contribution calculation error: {e}")
            return 0.0
    
    def _calculate_style_contribution(self, portfolio_data: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> float:
        """Calculate contribution from style factors"""
        try:
            # Simplified calculation for style factors
            # In practice, this would use factor models like Fama-French
            style_factors = market_data.get("style_factors", {
                "value": 0.0005,
                "momentum": 0.0003,
                "quality": 0.0002,
                "size": -0.0001
            })
            
            # Simulate portfolio exposure to style factors
            exposures = {
                "value": 0.8,
                "momentum": 0.6,
                "quality": 0.7,
                "size": -0.2
            }
            
            total_contribution = 0.0
            for factor, exposure in exposures.items():
                factor_return = style_factors.get(factor, 0.0)
                total_contribution += exposure * factor_return
            
            return total_contribution
        except Exception as e:
            logger.error(f"Style contribution calculation error: {e}")
            return 0.0
    
    def _calculate_alpha_contribution(self, portfolio_data: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> float:
        """Calculate contribution from alpha generation"""
        try:
            # Alpha = Total return - (Market return + Sector return + Style return)
            total_return = self._calculate_total_return(portfolio_data, "daily")
            market_contribution = self._calculate_market_contribution(portfolio_data, market_data)
            sector_contribution = self._calculate_sector_contribution(portfolio_data, market_data)
            style_contribution = self._calculate_style_contribution(portfolio_data, market_data)
            
            alpha = total_return - (market_contribution + sector_contribution + style_contribution)
            return alpha
        except Exception as e:
            logger.error(f"Alpha contribution calculation error: {e}")
            return 0.0
    
    def _calculate_risk_management_contribution(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate contribution from risk management"""
        try:
            # This would measure the value added by risk management
            # For now, simulate a small positive contribution
            return np.random.normal(0.0002, 0.0001)  # 0.02% mean, 0.01% std
        except Exception as e:
            logger.error(f"Risk management contribution calculation error: {e}")
            return 0.0
    
    def _calculate_transaction_cost_impact(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate negative impact from transaction costs"""
        try:
            # Transaction costs reduce returns
            total_turnover = portfolio_data.get("total_turnover", 0.1)  # 10% turnover
            avg_transaction_cost = 0.001  # 0.1% average cost
            
            return -abs(total_turnover * avg_transaction_cost)
        except Exception as e:
            logger.error(f"Transaction cost calculation error: {e}")
            return 0.0
    
    def _calculate_timing_contribution(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate contribution from market timing"""
        try:
            # This would measure value added from timing decisions
            # For now, simulate a small contribution
            return np.random.normal(0.0001, 0.00005)  # 0.01% mean, 0.005% std
        except Exception as e:
            logger.error(f"Timing contribution calculation error: {e}")
            return 0.0
    
    def _calculate_sentiment_impact(self, portfolio_data: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate contribution from sentiment analysis"""
        try:
            # This would measure impact of sentiment-driven decisions
            # For now, simulate based on sentiment data
            sentiment_score = market_data.get("aggregate_sentiment", 0.0)
            sentiment_impact = 0.0005 * sentiment_score  # 0.05% impact per unit sentiment
            
            return sentiment_impact
        except Exception as e:
            logger.error(f"Sentiment impact calculation error: {e}")
            return 0.0
    
    def _calculate_regime_adjustment(self, portfolio_data: Dict[str, Any], 
                                   market_data: Dict[str, Any]) -> float:
        """Calculate contribution from market regime adjustments"""
        try:
            # This would measure value from regime-based strategy adjustments
            # For now, simulate a small positive contribution
            return np.random.normal(0.00015, 0.00008)  # 0.015% mean, 0.008% std
        except Exception as e:
            logger.error(f"Regime adjustment calculation error: {e}")
            return 0.0
    
    def _calculate_component_weights(self, components: Dict[str, float], 
                                   total_return: float) -> Dict[str, float]:
        """Calculate weights of each component in total return"""
        try:
            if total_return == 0:
                return {component: 0.0 for component in components.keys()}
            
            weights = {}
            for component, contribution in components.items():
                # Calculate percentage contribution (can be negative)
                weight = contribution / abs(total_return) if total_return != 0 else 0.0
                weights[component] = weight
            
            return weights
        except Exception as e:
            logger.error(f"Component weight calculation error: {e}")
            return {component: 0.0 for component in components.keys()}
    
    def _calculate_attribution_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence score for attribution analysis"""
        try:
            # Confidence based on number of components and their magnitudes
            num_components = len([c for c in components.values() if c != 0])
            total_magnitude = sum(abs(c) for c in components.values())
            
            # Higher confidence with more components and larger magnitudes
            confidence = min(1.0, (num_components / len(self.components)) * 
                           min(1.0, total_magnitude * 100))
            
            return confidence
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def get_attribution_report(self, period: str = "daily") -> Dict[str, Any]:
        """Get comprehensive performance attribution report"""
        try:
            if not self.attribution_history:
                return {"status": "no_data", "message": "No attribution data available"}
            
            # Get recent attributions
            recent_attributions = [a for a in self.attribution_history 
                                 if (datetime.now() - a.timestamp).days <= 30]
            
            if not recent_attributions:
                return {"status": "no_recent_data", "message": "No recent attribution data"}
            
            # Calculate aggregate statistics
            component_stats = defaultdict(list)
            total_returns = []
            
            for attribution in recent_attributions:
                total_returns.append(attribution.total_return)
                for component, contribution in attribution.components.items():
                    component_stats[component].append(contribution)
            
            # Calculate averages
            avg_total_return = np.mean(total_returns) if total_returns else 0.0
            component_averages = {}
            component_std_devs = {}
            
            for component, contributions in component_stats.items():
                component_averages[component] = np.mean(contributions) if contributions else 0.0
                component_std_devs[component] = np.std(contributions) if contributions else 0.0
            
            # Calculate component importance (based on contribution magnitude)
            total_abs_contributions = sum(abs(avg) for avg in component_averages.values())
            component_importance = {}
            
            if total_abs_contributions > 0:
                for component, avg_contribution in component_averages.items():
                    component_importance[component] = abs(avg_contribution) / total_abs_contributions
            
            return {
                "status": "success",
                "period": period,
                "analysis_count": len(recent_attributions),
                "avg_total_return": avg_total_return,
                "total_return_std": np.std(total_returns) if total_returns else 0.0,
                "component_averages": component_averages,
                "component_std_devs": component_std_devs,
                "component_importance": component_importance,
                "last_analysis": recent_attributions[-1].timestamp.isoformat() if recent_attributions else None
            }
            
        except Exception as e:
            logger.error(f"Attribution report generation error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_top_drivers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performance drivers"""
        try:
            if not self.attribution_history:
                return []
            
            # Get most recent attribution
            latest_attribution = self.attribution_history[-1]
            
            # Sort components by absolute contribution
            sorted_components = sorted(
                latest_attribution.components.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Return top drivers
            top_drivers = []
            for component, contribution in sorted_components[:limit]:
                top_drivers.append({
                    "component": component,
                    "contribution": contribution,
                    "weight": latest_attribution.weights.get(component, 0.0),
                    "impact": "positive" if contribution > 0 else "negative"
                })
            
            return top_drivers
            
        except Exception as e:
            logger.error(f"Top drivers calculation error: {e}")
            return []

# Global performance attribution engine
_performance_attribution_engine = None

def get_performance_attribution_engine() -> PerformanceAttributionEngine:
    """Get global performance attribution engine instance"""
    global _performance_attribution_engine
    if _performance_attribution_engine is None:
        _performance_attribution_engine = PerformanceAttributionEngine()
    return _performance_attribution_engine

def analyze_performance(portfolio_data: Dict[str, Any], 
                       market_data: Dict[str, Any],
                       period: str = "daily") -> PerformanceAttribution:
    """Convenience function to analyze performance"""
    engine = get_performance_attribution_engine()
    return engine.analyze_performance(portfolio_data, market_data, period)

def get_attribution_report(period: str = "daily") -> Dict[str, Any]:
    """Convenience function to get attribution report"""
    engine = get_performance_attribution_engine()
    return engine.get_attribution_report(period)

def get_top_performance_drivers(limit: int = 5) -> List[Dict[str, Any]]:
    """Convenience function to get top performance drivers"""
    engine = get_performance_attribution_engine()
    return engine.get_top_drivers(limit)