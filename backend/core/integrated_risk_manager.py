"""
Production-Level Integrated Risk Manager
Real-time risk assessment and position sizing
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Individual risk metric"""
    name: str
    value: float  # 0-1 scale
    level: RiskLevel
    description: str
    recommendation: str

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    composite_risk_score: float  # 0-1 scale
    risk_level: RiskLevel
    individual_risks: Dict[str, RiskMetrics]
    recommended_action: str  # APPROVE, REDUCE_SIZE, REJECT
    recommended_position_size: float  # 0-1 multiplier
    stop_loss_adjustment: float  # multiplier for stop loss
    reasoning: str
    risk_breakdown: Dict[str, float]

class IntegratedRiskManager:
    """Production-level integrated risk management"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_risk: float = 0.05):
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk per trade
        self.max_position_risk = max_position_risk    # 5% max position risk
        
        # Risk calculation weights
        self.risk_weights = {
            'portfolio_risk': 0.25,
            'position_risk': 0.20,
            'correlation_risk': 0.15,
            'volatility_risk': 0.15,
            'liquidity_risk': 0.10,
            'market_risk': 0.10,
            'concentration_risk': 0.05
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        
        # Performance tracking
        self.risk_performance_history = []
        
    async def assess_trade_risk(self, symbol: str, action: str, proposed_quantity: int, 
                               current_portfolio: Dict[str, Any], market_context: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive risk assessment for proposed trade"""
        
        logger.info(f"Assessing risk for {action} {proposed_quantity} shares of {symbol}")
        
        # Calculate individual risk metrics in parallel
        risk_tasks = {
            'portfolio_risk': self._calculate_portfolio_risk(symbol, action, proposed_quantity, current_portfolio),
            'position_risk': self._calculate_position_risk(symbol, proposed_quantity, current_portfolio),
            'correlation_risk': self._calculate_correlation_risk(symbol, current_portfolio),
            'volatility_risk': self._calculate_volatility_risk(symbol, market_context),
            'liquidity_risk': self._calculate_liquidity_risk(symbol, proposed_quantity),
            'market_risk': self._calculate_market_risk(market_context),
            'concentration_risk': self._calculate_concentration_risk(symbol, current_portfolio)
        }
        
        # Execute risk calculations in parallel
        risk_results = {}
        try:
            completed_tasks = await asyncio.gather(*risk_tasks.values(), return_exceptions=True)
            
            for risk_name, result in zip(risk_tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    logger.warning(f"Risk calculation {risk_name} failed: {result}")
                    risk_results[risk_name] = RiskMetrics(
                        name=risk_name,
                        value=0.5,  # Default medium risk
                        level=RiskLevel.MEDIUM,
                        description=f"Risk calculation failed: {result}",
                        recommendation="Use caution"
                    )
                else:
                    risk_results[risk_name] = result
                    
        except Exception as e:
            logger.error(f"Critical error in risk assessment: {e}")
            # Return high risk assessment as safety measure
            return self._create_high_risk_assessment(f"Risk assessment failed: {e}")
        
        # Calculate composite risk score
        composite_risk = self._calculate_composite_risk(risk_results)
        
        # Determine risk level
        risk_level = self._determine_risk_level(composite_risk)
        
        # Generate recommendations
        recommendation = self._generate_risk_recommendation(composite_risk, risk_results, action)
        
        # Calculate recommended position size
        recommended_size = self._calculate_recommended_position_size(composite_risk, risk_results, proposed_quantity)
        
        # Calculate stop loss adjustment
        stop_loss_adj = self._calculate_stop_loss_adjustment(composite_risk, risk_results)
        
        # Generate reasoning
        reasoning = self._generate_risk_reasoning(composite_risk, risk_results, recommendation)
        
        # Create risk breakdown
        risk_breakdown = {name: metrics.value for name, metrics in risk_results.items()}
        
        assessment = RiskAssessment(
            composite_risk_score=composite_risk,
            risk_level=risk_level,
            individual_risks=risk_results,
            recommended_action=recommendation['action'],
            recommended_position_size=recommended_size,
            stop_loss_adjustment=stop_loss_adj,
            reasoning=reasoning,
            risk_breakdown=risk_breakdown
        )
        
        # Store performance history
        self.risk_performance_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'composite_risk': composite_risk,
            'recommendation': recommendation['action'],
            'risk_breakdown': risk_breakdown
        })
        
        # Keep only last 200 assessments
        if len(self.risk_performance_history) > 200:
            self.risk_performance_history = self.risk_performance_history[-200:]
        
        logger.info(f"Risk assessment complete: {risk_level.value} risk ({composite_risk:.3f}), "
                   f"recommendation: {recommendation['action']}")
        
        return assessment
    
    async def _calculate_portfolio_risk(self, symbol: str, action: str, quantity: int, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Calculate portfolio-level risk"""
        try:
            current_value = portfolio.get('totalValue', 100000)
            cash = portfolio.get('cash', current_value)
            
            # Estimate trade value (using approximate price)
            estimated_price = await self._get_estimated_price(symbol)
            trade_value = quantity * estimated_price
            
            # Calculate portfolio risk as percentage of total value
            portfolio_risk_pct = trade_value / current_value if current_value > 0 else 1.0
            
            # Normalize to 0-1 scale (2% = 0.5, 4% = 1.0)
            risk_score = min(1.0, portfolio_risk_pct / (self.max_portfolio_risk * 2))
            
            if risk_score > 0.8:
                level = RiskLevel.CRITICAL
                recommendation = "Reject trade - exceeds portfolio risk limits"
            elif risk_score > 0.6:
                level = RiskLevel.HIGH
                recommendation = "Reduce position size significantly"
            elif risk_score > 0.3:
                level = RiskLevel.MEDIUM
                recommendation = "Consider reducing position size"
            else:
                level = RiskLevel.LOW
                recommendation = "Portfolio risk acceptable"
            
            return RiskMetrics(
                name="portfolio_risk",
                value=risk_score,
                level=level,
                description=f"Trade represents {portfolio_risk_pct*100:.1f}% of portfolio value",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return RiskMetrics("portfolio_risk", 0.5, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_position_risk(self, symbol: str, quantity: int, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Calculate position-specific risk"""
        try:
            holdings = portfolio.get('holdings', {})
            current_position = holdings.get(symbol, {}).get('qty', 0)
            
            # Calculate position concentration
            total_positions = len(holdings)
            max_recommended_positions = 20  # Diversification target
            
            concentration_risk = 1.0 / max_recommended_positions if total_positions < max_recommended_positions else 1.0 / total_positions
            
            # Calculate position size risk
            estimated_price = await self._get_estimated_price(symbol)
            position_value = (current_position + quantity) * estimated_price
            total_portfolio_value = portfolio.get('totalValue', 100000)
            
            position_pct = position_value / total_portfolio_value if total_portfolio_value > 0 else 1.0
            
            # Combine concentration and size risks
            risk_score = min(1.0, (position_pct / self.max_position_risk + concentration_risk) / 2)
            
            if risk_score > 0.8:
                level = RiskLevel.CRITICAL
                recommendation = "Position too large - reject or significantly reduce"
            elif risk_score > 0.6:
                level = RiskLevel.HIGH
                recommendation = "Large position - consider reducing size"
            elif risk_score > 0.3:
                level = RiskLevel.MEDIUM
                recommendation = "Moderate position size"
            else:
                level = RiskLevel.LOW
                recommendation = "Position size acceptable"
            
            return RiskMetrics(
                name="position_risk",
                value=risk_score,
                level=level,
                description=f"Position will be {position_pct*100:.1f}% of portfolio",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Position risk calculation failed: {e}")
            return RiskMetrics("position_risk", 0.5, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_correlation_risk(self, symbol: str, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Calculate correlation risk with existing positions"""
        try:
            holdings = portfolio.get('holdings', {})
            
            if not holdings:
                return RiskMetrics("correlation_risk", 0.1, RiskLevel.LOW, "No existing positions", "Low correlation risk")
            
            # Simplified correlation calculation based on sector/industry
            # In production, this would use actual price correlations
            symbol_sector = await self._get_symbol_sector(symbol)
            
            sector_exposure = 0
            total_exposure = 0
            
            for holding_symbol, holding_data in holdings.items():
                holding_value = holding_data.get('qty', 0) * holding_data.get('avg_price', 0)
                total_exposure += holding_value
                
                holding_sector = await self._get_symbol_sector(holding_symbol)
                if holding_sector == symbol_sector:
                    sector_exposure += holding_value
            
            # Calculate sector concentration risk
            sector_concentration = sector_exposure / total_exposure if total_exposure > 0 else 0
            
            # Risk increases with sector concentration
            risk_score = min(1.0, sector_concentration * 2)  # 50% sector concentration = 1.0 risk
            
            if risk_score > 0.8:
                level = RiskLevel.HIGH
                recommendation = "High sector concentration - diversify"
            elif risk_score > 0.5:
                level = RiskLevel.MEDIUM
                recommendation = "Moderate sector concentration"
            else:
                level = RiskLevel.LOW
                recommendation = "Good diversification"
            
            return RiskMetrics(
                name="correlation_risk",
                value=risk_score,
                level=level,
                description=f"{sector_concentration*100:.1f}% exposure to {symbol_sector} sector",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return RiskMetrics("correlation_risk", 0.3, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_volatility_risk(self, symbol: str, market_context: Dict[str, Any]) -> RiskMetrics:
        """Calculate volatility-based risk"""
        try:
            # Get volatility from market context or calculate
            volatility = market_context.get('volatility', 0.02)  # Default 2% daily volatility
            
            # Normalize volatility to risk score (4% daily vol = 1.0 risk)
            risk_score = min(1.0, volatility / 0.04)
            
            if risk_score > 0.8:
                level = RiskLevel.HIGH
                recommendation = "Very high volatility - reduce position size"
            elif risk_score > 0.5:
                level = RiskLevel.MEDIUM
                recommendation = "Elevated volatility - use caution"
            else:
                level = RiskLevel.LOW
                recommendation = "Normal volatility levels"
            
            return RiskMetrics(
                name="volatility_risk",
                value=risk_score,
                level=level,
                description=f"Daily volatility: {volatility*100:.1f}%",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Volatility risk calculation failed: {e}")
            return RiskMetrics("volatility_risk", 0.4, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_liquidity_risk(self, symbol: str, quantity: int) -> RiskMetrics:
        """Calculate liquidity risk"""
        try:
            # Simplified liquidity assessment
            # In production, this would use actual volume data
            avg_daily_volume = await self._get_average_volume(symbol)
            
            if avg_daily_volume == 0:
                risk_score = 1.0
                level = RiskLevel.CRITICAL
                recommendation = "No volume data - very high liquidity risk"
            else:
                # Risk increases if trade size is large relative to average volume
                volume_impact = quantity / avg_daily_volume
                risk_score = min(1.0, volume_impact * 10)  # 10% of daily volume = 1.0 risk
                
                if risk_score > 0.8:
                    level = RiskLevel.HIGH
                    recommendation = "Large trade relative to volume - expect slippage"
                elif risk_score > 0.4:
                    level = RiskLevel.MEDIUM
                    recommendation = "Moderate liquidity impact"
                else:
                    level = RiskLevel.LOW
                    recommendation = "Good liquidity"
            
            return RiskMetrics(
                name="liquidity_risk",
                value=risk_score,
                level=level,
                description=f"Trade size: {quantity}, Avg volume: {avg_daily_volume:,.0f}",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation failed: {e}")
            return RiskMetrics("liquidity_risk", 0.3, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_market_risk(self, market_context: Dict[str, Any]) -> RiskMetrics:
        """Calculate overall market risk"""
        try:
            market_stress = market_context.get('stress_level', 0.3)
            volatility = market_context.get('volatility', 0.02)
            
            # Combine market stress and volatility
            risk_score = min(1.0, (market_stress + volatility / 0.04) / 2)
            
            if risk_score > 0.8:
                level = RiskLevel.HIGH
                recommendation = "High market stress - reduce trading"
            elif risk_score > 0.5:
                level = RiskLevel.MEDIUM
                recommendation = "Elevated market risk"
            else:
                level = RiskLevel.LOW
                recommendation = "Normal market conditions"
            
            return RiskMetrics(
                name="market_risk",
                value=risk_score,
                level=level,
                description=f"Market stress: {market_stress:.2f}, Volatility: {volatility*100:.1f}%",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Market risk calculation failed: {e}")
            return RiskMetrics("market_risk", 0.4, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    async def _calculate_concentration_risk(self, symbol: str, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Calculate concentration risk"""
        try:
            holdings = portfolio.get('holdings', {})
            total_positions = len(holdings)
            
            # Risk decreases with more positions (diversification)
            if total_positions == 0:
                risk_score = 0.1  # Low risk for first position
            elif total_positions < 5:
                risk_score = 0.8  # High concentration risk
            elif total_positions < 10:
                risk_score = 0.5  # Medium concentration risk
            else:
                risk_score = 0.2  # Low concentration risk
            
            if risk_score > 0.7:
                level = RiskLevel.HIGH
                recommendation = "Portfolio not well diversified"
            elif risk_score > 0.4:
                level = RiskLevel.MEDIUM
                recommendation = "Consider more diversification"
            else:
                level = RiskLevel.LOW
                recommendation = "Good diversification"
            
            return RiskMetrics(
                name="concentration_risk",
                value=risk_score,
                level=level,
                description=f"Portfolio has {total_positions} positions",
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return RiskMetrics("concentration_risk", 0.4, RiskLevel.MEDIUM, f"Calculation failed: {e}", "Use caution")
    
    def _calculate_composite_risk(self, risk_metrics: Dict[str, RiskMetrics]) -> float:
        """Calculate weighted composite risk score"""
        total_weighted_risk = 0
        total_weight = 0
        
        for risk_name, metrics in risk_metrics.items():
            weight = self.risk_weights.get(risk_name, 0.1)
            total_weighted_risk += metrics.value * weight
            total_weight += weight
        
        return total_weighted_risk / total_weight if total_weight > 0 else 0.5
    
    def _determine_risk_level(self, composite_risk: float) -> RiskLevel:
        """Determine risk level from composite score"""
        if composite_risk >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL
        elif composite_risk >= self.risk_thresholds['high']:
            return RiskLevel.HIGH
        elif composite_risk >= self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_risk_recommendation(self, composite_risk: float, risk_metrics: Dict[str, RiskMetrics], action: str) -> Dict[str, Any]:
        """Generate risk-based recommendation"""
        if composite_risk >= 0.8:
            return {
                'action': 'REJECT',
                'reason': 'Composite risk too high',
                'confidence': 0.9
            }
        elif composite_risk >= 0.6:
            return {
                'action': 'REDUCE_SIZE',
                'reason': 'High risk - reduce position size',
                'confidence': 0.7
            }
        else:
            return {
                'action': 'APPROVE',
                'reason': 'Risk within acceptable limits',
                'confidence': 0.8
            }
    
    def _calculate_recommended_position_size(self, composite_risk: float, risk_metrics: Dict[str, RiskMetrics], proposed_quantity: int) -> float:
        """Calculate recommended position size multiplier"""
        if composite_risk >= 0.8:
            return 0.0  # Reject trade
        elif composite_risk >= 0.6:
            return 0.5  # Half size
        elif composite_risk >= 0.4:
            return 0.75  # Reduce by 25%
        else:
            return 1.0  # Full size
    
    def _calculate_stop_loss_adjustment(self, composite_risk: float, risk_metrics: Dict[str, RiskMetrics]) -> float:
        """Calculate stop loss adjustment multiplier"""
        volatility_risk = risk_metrics.get('volatility_risk', RiskMetrics('', 0.3, RiskLevel.MEDIUM, '', ''))
        
        # Tighter stops for higher risk
        if composite_risk >= 0.7:
            return 0.8  # 20% tighter stop loss
        elif volatility_risk.value >= 0.6:
            return 0.9  # 10% tighter stop loss
        else:
            return 1.0  # Normal stop loss
    
    def _generate_risk_reasoning(self, composite_risk: float, risk_metrics: Dict[str, RiskMetrics], recommendation: Dict[str, Any]) -> str:
        """Generate human-readable risk reasoning"""
        high_risk_factors = [name for name, metrics in risk_metrics.items() if metrics.value > 0.6]
        
        reasoning_parts = [
            f"Composite risk score: {composite_risk:.3f}",
            f"Recommendation: {recommendation['action']} ({recommendation['reason']})"
        ]
        
        if high_risk_factors:
            reasoning_parts.append(f"High risk factors: {', '.join(high_risk_factors)}")
        
        return " | ".join(reasoning_parts)
    
    def _create_high_risk_assessment(self, error_msg: str) -> RiskAssessment:
        """Create high-risk assessment for error cases"""
        return RiskAssessment(
            composite_risk_score=0.9,
            risk_level=RiskLevel.HIGH,
            individual_risks={},
            recommended_action='REJECT',
            recommended_position_size=0.0,
            stop_loss_adjustment=0.8,
            reasoning=f"High risk due to error: {error_msg}",
            risk_breakdown={}
        )
    
    # Helper methods for data retrieval (to be implemented based on your data sources)
    async def _get_estimated_price(self, symbol: str) -> float:
        """Get estimated current price for symbol"""
        # Placeholder - implement with your price data source
        return 100.0
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        # Placeholder - implement with your sector data
        return "Technology"
    
    async def _get_average_volume(self, symbol: str) -> float:
        """Get average daily volume for symbol"""
        # Placeholder - implement with your volume data
        return 100000.0
    
    def get_risk_performance_metrics(self) -> Dict[str, Any]:
        """Get risk management performance metrics"""
        if not self.risk_performance_history:
            return {}
        
        recent_assessments = self.risk_performance_history[-100:]
        
        risk_distribution = {}
        for assessment in recent_assessments:
            risk_score = assessment['composite_risk']
            if risk_score >= 0.8:
                risk_level = 'critical'
            elif risk_score >= 0.6:
                risk_level = 'high'
            elif risk_score >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            'total_assessments': len(self.risk_performance_history),
            'risk_distribution': risk_distribution,
            'avg_composite_risk': sum(a['composite_risk'] for a in recent_assessments) / len(recent_assessments),
            'risk_weights': self.risk_weights,
            'risk_thresholds': self.risk_thresholds
        }
