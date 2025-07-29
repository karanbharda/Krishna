#!/usr/bin/env python3
"""
Explanation Agent
================

Production-grade agent for generating detailed explanations of trading decisions,
market analysis, and portfolio recommendations with natural language processing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ExplanationRequest:
    """Request for explanation generation"""
    context_type: str  # "trade", "analysis", "portfolio", "risk"
    data: Dict[str, Any]
    user_level: str = "intermediate"  # "beginner", "intermediate", "expert"
    explanation_style: str = "detailed"  # "brief", "detailed", "technical"

@dataclass
class ExplanationResponse:
    """Structured explanation response"""
    summary: str
    detailed_explanation: str
    key_factors: List[str]
    reasoning_steps: List[str]
    confidence: float
    recommendations: List[str]
    risks_warnings: List[str]
    educational_notes: Optional[List[str]] = None

class ExplanationAgent:
    """
    Production-grade explanation agent for trading decisions
    
    Features:
    - Multi-level explanations (beginner to expert)
    - Context-aware reasoning
    - Educational content generation
    - Risk communication
    - Performance metrics explanation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "explanation_agent")
        
        # Explanation templates
        self.templates = self._load_explanation_templates()
        
        # Performance tracking
        self.explanations_generated = 0
        self.user_feedback_scores = []
        
        logger.info(f"Explanation Agent {self.agent_id} initialized")
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different contexts"""
        return {
            "trade_decision": """
**Trade Decision Explanation**

**Summary**: {summary}

**Why This Trade?**
{reasoning}

**Key Factors Considered:**
{key_factors}

**Risk Assessment:**
{risk_assessment}

**Expected Outcome:**
{expected_outcome}

**What to Watch:**
{monitoring_points}
""",
            
            "market_analysis": """
**Market Analysis Explanation**

**Current Market View**: {market_view}

**Technical Analysis:**
{technical_analysis}

**Fundamental Factors:**
{fundamental_factors}

**Sentiment Analysis:**
{sentiment_analysis}

**Key Levels to Watch:**
{key_levels}

**Market Outlook:**
{outlook}
""",
            
            "portfolio_recommendation": """
**Portfolio Recommendation Explanation**

**Current Portfolio Health**: {portfolio_health}

**Recommended Changes:**
{recommendations}

**Rationale:**
{rationale}

**Risk Impact:**
{risk_impact}

**Expected Benefits:**
{benefits}

**Implementation Timeline:**
{timeline}
""",
            
            "risk_assessment": """
**Risk Assessment Explanation**

**Overall Risk Level**: {risk_level}

**Primary Risk Factors:**
{primary_risks}

**Risk Mitigation Strategies:**
{mitigation_strategies}

**Position Sizing Rationale:**
{position_sizing}

**Stop Loss Strategy:**
{stop_loss_strategy}

**Monitoring Requirements:**
{monitoring}
"""
        }
    
    async def explain_trade_decision(self, trade_data: Dict[str, Any], 
                                   user_level: str = "intermediate") -> ExplanationResponse:
        """Generate comprehensive explanation for a trade decision"""
        try:
            # Extract trade information
            symbol = trade_data.get("symbol", "Unknown")
            action = trade_data.get("action", "Unknown")
            confidence = trade_data.get("confidence", 0.0)
            reasoning = trade_data.get("reasoning", "No reasoning provided")
            technical_signals = trade_data.get("technical_signals", {})
            risk_score = trade_data.get("risk_score", 5.0)
            
            # Generate summary
            summary = self._generate_trade_summary(symbol, action, confidence)
            
            # Generate detailed explanation
            detailed_explanation = self._generate_trade_explanation(
                trade_data, user_level
            )
            
            # Extract key factors
            key_factors = self._extract_key_factors(technical_signals, trade_data)
            
            # Generate reasoning steps
            reasoning_steps = self._generate_reasoning_steps(trade_data)
            
            # Generate recommendations
            recommendations = self._generate_trade_recommendations(trade_data)
            
            # Generate risk warnings
            risk_warnings = self._generate_risk_warnings(risk_score, trade_data)
            
            # Add educational notes for beginners
            educational_notes = None
            if user_level == "beginner":
                educational_notes = self._generate_educational_notes(trade_data)
            
            self.explanations_generated += 1
            
            return ExplanationResponse(
                summary=summary,
                detailed_explanation=detailed_explanation,
                key_factors=key_factors,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                recommendations=recommendations,
                risks_warnings=risk_warnings,
                educational_notes=educational_notes
            )
            
        except Exception as e:
            logger.error(f"Trade explanation error: {e}")
            return self._generate_error_explanation(str(e))
    
    async def explain_market_analysis(self, analysis_data: Dict[str, Any],
                                    user_level: str = "intermediate") -> ExplanationResponse:
        """Generate explanation for market analysis results"""
        try:
            symbol = analysis_data.get("symbol", "Market")
            technical_signals = analysis_data.get("technical_signals", {})
            recommendation = analysis_data.get("recommendation", "HOLD")
            confidence = analysis_data.get("confidence", 0.0)
            
            # Generate market view summary
            summary = f"Market analysis for {symbol}: {recommendation} (Confidence: {confidence:.1%})"
            
            # Generate detailed explanation
            detailed_explanation = self._generate_market_explanation(
                analysis_data, user_level
            )
            
            # Extract key technical factors
            key_factors = self._extract_technical_factors(technical_signals)
            
            # Generate analysis steps
            reasoning_steps = [
                "1. Technical indicator analysis",
                "2. Trend and momentum assessment", 
                "3. Support/resistance level identification",
                "4. Volume and volatility analysis",
                "5. Overall signal synthesis"
            ]
            
            # Generate market recommendations
            recommendations = self._generate_market_recommendations(analysis_data)
            
            # Generate market warnings
            risk_warnings = self._generate_market_warnings(analysis_data)
            
            return ExplanationResponse(
                summary=summary,
                detailed_explanation=detailed_explanation,
                key_factors=key_factors,
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                recommendations=recommendations,
                risks_warnings=risk_warnings
            )
            
        except Exception as e:
            logger.error(f"Market analysis explanation error: {e}")
            return self._generate_error_explanation(str(e))
    
    async def explain_portfolio_recommendation(self, portfolio_data: Dict[str, Any],
                                             user_level: str = "intermediate") -> ExplanationResponse:
        """Generate explanation for portfolio recommendations"""
        try:
            current_holdings = portfolio_data.get("holdings", {})
            recommendations = portfolio_data.get("recommendations", [])
            risk_profile = portfolio_data.get("risk_profile", "MEDIUM")
            
            # Generate portfolio summary
            summary = f"Portfolio optimization for {risk_profile} risk profile with {len(current_holdings)} holdings"
            
            # Generate detailed explanation
            detailed_explanation = self._generate_portfolio_explanation(
                portfolio_data, user_level
            )
            
            # Extract key portfolio factors
            key_factors = self._extract_portfolio_factors(portfolio_data)
            
            # Generate optimization steps
            reasoning_steps = [
                "1. Current portfolio analysis",
                "2. Risk-return assessment",
                "3. Diversification evaluation",
                "4. Correlation analysis",
                "5. Optimization recommendations"
            ]
            
            # Generate portfolio recommendations
            portfolio_recommendations = self._generate_portfolio_recommendations(portfolio_data)
            
            # Generate portfolio warnings
            risk_warnings = self._generate_portfolio_warnings(portfolio_data)
            
            return ExplanationResponse(
                summary=summary,
                detailed_explanation=detailed_explanation,
                key_factors=key_factors,
                reasoning_steps=reasoning_steps,
                confidence=0.8,  # Portfolio recommendations typically high confidence
                recommendations=portfolio_recommendations,
                risks_warnings=risk_warnings
            )
            
        except Exception as e:
            logger.error(f"Portfolio explanation error: {e}")
            return self._generate_error_explanation(str(e))
    
    def _generate_trade_summary(self, symbol: str, action: str, confidence: float) -> str:
        """Generate concise trade summary"""
        confidence_text = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        return f"{action} recommendation for {symbol} with {confidence_text} confidence ({confidence:.1%})"
    
    def _generate_trade_explanation(self, trade_data: Dict[str, Any], user_level: str) -> str:
        """Generate detailed trade explanation based on user level"""
        symbol = trade_data.get("symbol", "Unknown")
        action = trade_data.get("action", "Unknown")
        reasoning = trade_data.get("reasoning", "")
        
        if user_level == "beginner":
            return f"""
This recommendation suggests to {action.lower()} {symbol} based on our analysis.

**What this means:**
- {action} means we should {'purchase' if action == 'BUY' else 'sell' if action == 'SELL' else 'maintain'} this stock
- Our analysis looked at price patterns, market trends, and trading volumes
- The recommendation is based on multiple factors working together

**Why we made this decision:**
{reasoning}

**What happens next:**
- Monitor the stock price movement
- Watch for any changes in market conditions
- Be prepared to adjust if new information emerges
"""
        elif user_level == "expert":
            technical_signals = trade_data.get("technical_signals", {})
            return f"""
**Technical Analysis Summary:**
{self._format_technical_signals(technical_signals)}

**Quantitative Metrics:**
- Signal Strength: {trade_data.get('confidence', 0):.3f}
- Risk Score: {trade_data.get('risk_score', 0):.2f}
- Expected Return: {trade_data.get('expected_return', 0):.2%}

**Decision Logic:**
{reasoning}

**Risk Management:**
- Position Size: {trade_data.get('position_size', 0):.1%} of portfolio
- Stop Loss: {trade_data.get('stop_loss', 'Not set')}
- Target Price: {trade_data.get('target_price', 'Not set')}
"""
        else:  # intermediate
            return f"""
**Trade Recommendation: {action} {symbol}**

**Analysis Summary:**
{reasoning}

**Key Indicators:**
{self._format_key_indicators(trade_data.get('technical_signals', {}))}

**Risk Considerations:**
- Risk Level: {self._get_risk_level_text(trade_data.get('risk_score', 5))}
- Recommended Position Size: {trade_data.get('position_size', 0):.1%} of portfolio

**Expected Outcome:**
- Target Return: {trade_data.get('expected_return', 0):.1%}
- Time Horizon: {trade_data.get('time_horizon', 'Medium-term')}
"""
    
    def _extract_key_factors(self, technical_signals: Dict[str, Any], trade_data: Dict[str, Any]) -> List[str]:
        """Extract key factors influencing the decision"""
        factors = []
        
        # Technical factors
        if technical_signals.get("trend", {}).get("composite_score", 0) > 0.3:
            factors.append("Strong upward trend detected")
        elif technical_signals.get("trend", {}).get("composite_score", 0) < -0.3:
            factors.append("Strong downward trend detected")
        
        # Momentum factors
        momentum_score = technical_signals.get("momentum", {}).get("composite_score", 0)
        if momentum_score > 0.2:
            factors.append("Positive momentum indicators")
        elif momentum_score < -0.2:
            factors.append("Negative momentum indicators")
        
        # Volume factors
        volume_score = technical_signals.get("volume", {}).get("composite_score", 0)
        if volume_score > 0.2:
            factors.append("Strong volume confirmation")
        
        # Risk factors
        risk_score = trade_data.get("risk_score", 5)
        if risk_score < 3:
            factors.append("Low risk environment")
        elif risk_score > 7:
            factors.append("High risk environment")
        
        return factors if factors else ["Standard market conditions"]
    
    def _generate_reasoning_steps(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate step-by-step reasoning process"""
        return [
            "1. Analyzed current market data and price trends",
            "2. Evaluated technical indicators and momentum signals",
            "3. Assessed volume patterns and market sentiment",
            "4. Calculated risk-reward ratio and position sizing",
            "5. Generated final recommendation with confidence score"
        ]
    
    def _generate_trade_recommendations(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate actionable trade recommendations"""
        recommendations = []
        
        action = trade_data.get("action", "HOLD")
        if action in ["BUY", "SELL"]:
            recommendations.append(f"Execute {action} order with recommended position size")
            recommendations.append("Set stop-loss order to manage downside risk")
            recommendations.append("Monitor price action for entry/exit timing")
        
        recommendations.append("Review position regularly based on market conditions")
        recommendations.append("Maintain proper risk management discipline")
        
        return recommendations
    
    def _generate_risk_warnings(self, risk_score: float, trade_data: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []
        
        if risk_score > 7:
            warnings.append("⚠️ High risk trade - consider reduced position size")
        if risk_score > 8:
            warnings.append("⚠️ Very high volatility expected")
        
        confidence = trade_data.get("confidence", 0)
        if confidence < 0.6:
            warnings.append("⚠️ Low confidence signal - proceed with caution")
        
        if not trade_data.get("stop_loss"):
            warnings.append("⚠️ No stop-loss defined - ensure risk management")
        
        return warnings if warnings else ["Standard market risks apply"]
    
    def _generate_educational_notes(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate educational content for beginners"""
        return [
            "💡 Technical Analysis: Uses price charts and indicators to predict future movements",
            "💡 Position Sizing: The amount of money allocated to a single trade",
            "💡 Stop Loss: An order to sell if price drops to limit losses",
            "💡 Confidence Score: How certain our analysis is about the recommendation",
            "💡 Risk Score: Measures how risky this trade is (1=low, 10=high)"
        ]
    
    def _generate_error_explanation(self, error_message: str) -> ExplanationResponse:
        """Generate explanation for error cases"""
        return ExplanationResponse(
            summary="Error generating explanation",
            detailed_explanation=f"An error occurred while generating the explanation: {error_message}",
            key_factors=["System error"],
            reasoning_steps=["Error in analysis pipeline"],
            confidence=0.0,
            recommendations=["Please try again or contact support"],
            risks_warnings=["Unable to assess risks due to error"]
        )
    
    def _format_technical_signals(self, signals: Dict[str, Any]) -> str:
        """Format technical signals for expert users"""
        if not signals:
            return "No technical signals available"
        
        formatted = []
        for category, data in signals.items():
            if isinstance(data, dict) and "composite_score" in data:
                score = data["composite_score"]
                formatted.append(f"- {category.title()}: {score:.3f}")
        
        return "\n".join(formatted) if formatted else "Technical signals processing"
    
    def _format_key_indicators(self, signals: Dict[str, Any]) -> str:
        """Format key indicators for intermediate users"""
        if not signals:
            return "Standard technical analysis applied"
        
        indicators = []
        trend_score = signals.get("trend", {}).get("composite_score", 0)
        if abs(trend_score) > 0.2:
            direction = "Upward" if trend_score > 0 else "Downward"
            indicators.append(f"- {direction} trend strength: {abs(trend_score):.1f}")
        
        momentum_score = signals.get("momentum", {}).get("composite_score", 0)
        if abs(momentum_score) > 0.2:
            direction = "Positive" if momentum_score > 0 else "Negative"
            indicators.append(f"- {direction} momentum: {abs(momentum_score):.1f}")
        
        return "\n".join(indicators) if indicators else "- Neutral technical conditions"
    
    def _get_risk_level_text(self, risk_score: float) -> str:
        """Convert risk score to text description"""
        if risk_score < 3:
            return "Low"
        elif risk_score < 7:
            return "Medium"
        else:
            return "High"
    
    def _extract_technical_factors(self, technical_signals: Dict[str, Any]) -> List[str]:
        """Extract technical factors for market analysis"""
        factors = []
        
        for category, data in technical_signals.items():
            if isinstance(data, dict) and "composite_score" in data:
                score = data["composite_score"]
                if abs(score) > 0.2:
                    strength = "Strong" if abs(score) > 0.5 else "Moderate"
                    direction = "positive" if score > 0 else "negative"
                    factors.append(f"{strength} {direction} {category} signals")
        
        return factors if factors else ["Neutral technical conditions"]
    
    def _generate_market_explanation(self, analysis_data: Dict[str, Any], user_level: str) -> str:
        """Generate market analysis explanation"""
        symbol = analysis_data.get("symbol", "Market")
        recommendation = analysis_data.get("recommendation", "HOLD")
        
        return f"""
**Market Analysis for {symbol}**

**Current Recommendation: {recommendation}**

**Technical Overview:**
Our analysis examined multiple technical indicators including trend strength, 
momentum oscillators, volume patterns, and volatility measures.

**Key Findings:**
{analysis_data.get('reasoning', 'Standard market analysis completed')}

**Market Conditions:**
The current market environment shows {self._assess_market_conditions(analysis_data)}
"""
    
    def _assess_market_conditions(self, analysis_data: Dict[str, Any]) -> str:
        """Assess overall market conditions"""
        confidence = analysis_data.get("confidence", 0.5)
        risk_score = analysis_data.get("risk_score", 5)
        
        if confidence > 0.7 and risk_score < 4:
            return "favorable conditions with clear directional signals"
        elif confidence < 0.4 or risk_score > 7:
            return "uncertain conditions requiring careful monitoring"
        else:
            return "mixed signals requiring selective approach"
    
    def _generate_market_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate market-specific recommendations"""
        recommendation = analysis_data.get("recommendation", "HOLD")
        
        recommendations = [
            f"Primary recommendation: {recommendation}",
            "Monitor key technical levels for confirmation",
            "Watch for volume confirmation on breakouts"
        ]
        
        if recommendation == "BUY":
            recommendations.append("Look for pullbacks for better entry points")
        elif recommendation == "SELL":
            recommendations.append("Consider scaling out of positions")
        
        return recommendations
    
    def _generate_market_warnings(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Generate market-specific warnings"""
        warnings = []
        
        confidence = analysis_data.get("confidence", 0.5)
        if confidence < 0.6:
            warnings.append("⚠️ Mixed signals - avoid large positions")
        
        risk_score = analysis_data.get("risk_score", 5)
        if risk_score > 6:
            warnings.append("⚠️ Elevated market volatility expected")
        
        return warnings if warnings else ["Standard market risks apply"]
    
    def _extract_portfolio_factors(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Extract key portfolio factors"""
        factors = []
        
        holdings_count = len(portfolio_data.get("holdings", {}))
        if holdings_count < 5:
            factors.append("Limited diversification - concentrated portfolio")
        elif holdings_count > 15:
            factors.append("High diversification - may dilute returns")
        
        risk_profile = portfolio_data.get("risk_profile", "MEDIUM")
        factors.append(f"Risk profile: {risk_profile}")
        
        return factors
    
    def _generate_portfolio_explanation(self, portfolio_data: Dict[str, Any], user_level: str) -> str:
        """Generate portfolio explanation"""
        holdings_count = len(portfolio_data.get("holdings", {}))
        risk_profile = portfolio_data.get("risk_profile", "MEDIUM")
        
        return f"""
**Portfolio Analysis Summary**

**Current Portfolio:**
- Number of holdings: {holdings_count}
- Risk profile: {risk_profile}
- Diversification level: {self._assess_diversification(holdings_count)}

**Optimization Opportunities:**
{self._identify_optimization_opportunities(portfolio_data)}

**Risk Assessment:**
{self._assess_portfolio_risk(portfolio_data)}
"""
    
    def _assess_diversification(self, holdings_count: int) -> str:
        """Assess portfolio diversification level"""
        if holdings_count < 5:
            return "Low - consider adding more positions"
        elif holdings_count > 15:
            return "High - may be over-diversified"
        else:
            return "Adequate - well-balanced approach"
    
    def _identify_optimization_opportunities(self, portfolio_data: Dict[str, Any]) -> str:
        """Identify portfolio optimization opportunities"""
        return "Analysis of sector allocation, correlation, and risk-adjusted returns suggests potential improvements in position sizing and asset selection."
    
    def _assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> str:
        """Assess overall portfolio risk"""
        risk_profile = portfolio_data.get("risk_profile", "MEDIUM")
        return f"Portfolio risk level aligned with {risk_profile} risk tolerance. Regular rebalancing recommended."
    
    def _generate_portfolio_recommendations(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate portfolio-specific recommendations"""
        return [
            "Review and rebalance portfolio quarterly",
            "Monitor correlation between holdings",
            "Adjust position sizes based on conviction levels",
            "Consider tax implications of any changes"
        ]
    
    def _generate_portfolio_warnings(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Generate portfolio-specific warnings"""
        warnings = []
        
        holdings_count = len(portfolio_data.get("holdings", {}))
        if holdings_count < 3:
            warnings.append("⚠️ Very concentrated portfolio - high specific risk")
        
        return warnings if warnings else ["Standard portfolio risks apply"]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get explanation agent status"""
        avg_feedback = sum(self.user_feedback_scores) / len(self.user_feedback_scores) if self.user_feedback_scores else 0
        
        return {
            "agent_id": self.agent_id,
            "explanations_generated": self.explanations_generated,
            "average_user_feedback": avg_feedback,
            "templates_loaded": len(self.templates),
            "status": "active"
        }
