#!/usr/bin/env python3
"""
Insight Agent
=============

Production-grade agent for generating actionable trading insights with reasoning logging.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class InsightRequest:
    """Request for insight generation"""
    context_type: str  # "prediction", "shortlist", "market", "portfolio"
    data: Dict[str, Any]
    user_preferences: Dict[str, Any]
    risk_profile: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH"

@dataclass
class InsightResponse:
    """Structured insight response"""
    summary: str
    detailed_insight: str
    actionable_recommendations: List[str]
    risk_considerations: List[str]
    confidence: float
    reasoning_log: List[str]
    timestamp: str

class InsightAgent:
    """
    Production-grade insight agent for trading
    
    Features:
    - Actionable insights generation
    - Reasoning logging for transparency
    - Risk-aware recommendations
    - Context-specific insights
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "insight_agent")
        
        # Ollama configuration for enhanced insights
        self.ollama_enabled = config.get("ollama_enabled", False)
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "llama2")
        
        # Performance tracking
        self.insights_generated = 0
        self.reasoning_logs = []
        
        logger.info(f"Insight Agent {self.agent_id} initialized")
    
    async def generate_prediction_insights(self, prediction_data: Dict[str, Any], 
                                         user_preferences: Dict[str, Any]) -> InsightResponse:
        """Generate insights for prediction results"""
        try:
            reasoning_log = []
            reasoning_log.append("Starting prediction insight generation")
            
            # Extract key information
            predictions = prediction_data.get("ranked_predictions", [])
            models_used = prediction_data.get("models_used", [])
            horizon = prediction_data.get("horizon", "day")
            
            reasoning_log.append(f"Analyzing {len(predictions)} predictions using {len(models_used)} models")
            
            # Identify top opportunities
            top_predictions = [p for p in predictions if p.get("score", 0) > 0.7][:5]
            reasoning_log.append(f"Identified {len(top_predictions)} high-confidence predictions")
            
            # Generate summary
            summary = f"Top {len(top_predictions)} trading opportunities identified for {horizon} horizon"
            
            # Generate detailed insight
            detailed_insight = self._generate_prediction_insight_detail(
                top_predictions, models_used, horizon, reasoning_log
            )
            
            # Generate actionable recommendations
            recommendations = self._generate_prediction_recommendations(
                top_predictions, user_preferences, reasoning_log
            )
            
            # Generate risk considerations
            risk_considerations = self._generate_prediction_risks(
                top_predictions, reasoning_log
            )
            
            # Enhance with LLM if available
            if self.ollama_enabled:
                enhanced_insight = await self._enhance_with_llm(
                    "prediction", 
                    {
                        "predictions": top_predictions,
                        "models": models_used,
                        "horizon": horizon
                    },
                    reasoning_log
                )
                if enhanced_insight:
                    detailed_insight += f"\n\nEnhanced insight: {enhanced_insight}"
            
            self.insights_generated += 1
            
            return InsightResponse(
                summary=summary,
                detailed_insight=detailed_insight,
                actionable_recommendations=recommendations,
                risk_considerations=risk_considerations,
                confidence=0.85,
                reasoning_log=reasoning_log,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Prediction insight error: {e}")
            return self._generate_error_insight(str(e))
    
    async def generate_shortlist_insights(self, shortlist_data: Dict[str, Any], 
                                        user_preferences: Dict[str, Any]) -> InsightResponse:
        """Generate insights for shortlisted stocks"""
        try:
            reasoning_log = []
            reasoning_log.append("Starting shortlist insight generation")
            
            # Extract key information
            shortlisted_stocks = shortlist_data.get("shortlisted_stocks", [])
            filters_applied = shortlist_data.get("filters_applied", {})
            total_scanned = shortlist_data.get("total_scanned", 0)
            
            reasoning_log.append(f"Analyzing {len(shortlisted_stocks)} shortlisted stocks from {total_scanned} scanned")
            
            # Categorize stocks
            strong_buys = [s for s in shortlisted_stocks if s.get("recommendation") == "STRONG_BUY"]
            buys = [s for s in shortlisted_stocks if s.get("recommendation") == "BUY"]
            
            reasoning_log.append(f"Categorized: {len(strong_buys)} strong buys, {len(buys)} buys")
            
            # Generate summary
            summary = f"Shortlist analysis: {len(strong_buys)} strong buys, {len(buys)} buys identified"
            
            # Generate detailed insight
            detailed_insight = self._generate_shortlist_insight_detail(
                shortlisted_stocks, filters_applied, reasoning_log
            )
            
            # Generate actionable recommendations
            recommendations = self._generate_shortlist_recommendations(
                shortlisted_stocks, user_preferences, reasoning_log
            )
            
            # Generate risk considerations
            risk_considerations = self._generate_shortlist_risks(
                shortlisted_stocks, reasoning_log
            )
            
            # Enhance with LLM if available
            if self.ollama_enabled:
                enhanced_insight = await self._enhance_with_llm(
                    "shortlist", 
                    {
                        "stocks": shortlisted_stocks[:10],  # Top 10 for context
                        "filters": filters_applied
                    },
                    reasoning_log
                )
                if enhanced_insight:
                    detailed_insight += f"\n\nEnhanced insight: {enhanced_insight}"
            
            self.insights_generated += 1
            
            return InsightResponse(
                summary=summary,
                detailed_insight=detailed_insight,
                actionable_recommendations=recommendations,
                risk_considerations=risk_considerations,
                confidence=0.8,
                reasoning_log=reasoning_log,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Shortlist insight error: {e}")
            return self._generate_error_insight(str(e))
    
    def _generate_prediction_insight_detail(self, predictions: List[Dict], 
                                          models: List[str], horizon: str, 
                                          reasoning_log: List[str]) -> str:
        """Generate detailed prediction insight"""
        reasoning_log.append("Generating detailed prediction insight")
        
        if not predictions:
            return "No high-confidence predictions identified at this time."
        
        detail = f"Analysis of top {len(predictions)} predictions using {', '.join(models)} models:\n\n"
        
        for i, pred in enumerate(predictions[:3]):  # Top 3 detailed
            symbol = pred.get("symbol", "Unknown")
            score = pred.get("score", 0)
            confidence = pred.get("confidence", 0)
            recommendation = pred.get("recommendation", "HOLD")
            
            detail += f"{i+1}. {symbol}: {recommendation} (Score: {score:.3f}, Confidence: {confidence:.2f})\n"
        
        if len(predictions) > 3:
            detail += f"\n... and {len(predictions) - 3} more opportunities."
        
        return detail
    
    def _generate_shortlist_insight_detail(self, stocks: List[Dict], 
                                         filters: Dict[str, Any], 
                                         reasoning_log: List[str]) -> str:
        """Generate detailed shortlist insight"""
        reasoning_log.append("Generating detailed shortlist insight")
        
        if not stocks:
            return "No stocks match the current filtering criteria."
        
        detail = f"Analysis of {len(stocks)} shortlisted stocks with filters: {filters}\n\n"
        
        # Group by sector
        sector_groups = {}
        for stock in stocks:
            sector = stock.get("sector", "Unknown")
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(stock)
        
        for sector, sector_stocks in list(sector_groups.items())[:3]:  # Top 3 sectors
            detail += f"{sector} sector: {len(sector_stocks)} stocks\n"
            for stock in sector_stocks[:2]:  # Top 2 stocks per sector
                symbol = stock.get("symbol", "Unknown")
                score = stock.get("score", 0)
                detail += f"  - {symbol} (Score: {score:.3f})\n"
        
        if len(sector_groups) > 3:
            detail += f"\n... and {len(sector_groups) - 3} more sectors."
        
        return detail
    
    def _generate_prediction_recommendations(self, predictions: List[Dict], 
                                          user_preferences: Dict[str, Any], 
                                          reasoning_log: List[str]) -> List[str]:
        """Generate prediction recommendations"""
        reasoning_log.append("Generating prediction recommendations")
        
        recommendations = []
        
        if not predictions:
            recommendations.append("No high-confidence trading opportunities at this time.")
            return recommendations
        
        # Risk-based recommendations
        risk_profile = user_preferences.get("risk_profile", "MEDIUM")
        
        if risk_profile == "LOW":
            recommendations.append("Focus on top 1-2 predictions with highest confidence scores")
            recommendations.append("Consider smaller position sizes for higher-risk opportunities")
        elif risk_profile == "HIGH":
            recommendations.append("Opportunity to deploy capital across multiple high-score predictions")
            recommendations.append("Consider leveraging strong momentum signals")
        else:  # MEDIUM
            recommendations.append("Balanced approach: 2-3 top predictions with moderate position sizing")
            recommendations.append("Diversify across different sectors when possible")
        
        # Time horizon recommendations
        horizon = user_preferences.get("horizon", "day")
        if horizon == "day":
            recommendations.append("Monitor intraday price action for optimal entry points")
        elif horizon == "week":
            recommendations.append("Weekly monitoring with adjustments based on technical signals")
        else:
            recommendations.append("Longer-term holding strategy with periodic reviews")
        
        return recommendations
    
    def _generate_shortlist_recommendations(self, stocks: List[Dict], 
                                          user_preferences: Dict[str, Any], 
                                          reasoning_log: List[str]) -> List[str]:
        """Generate shortlist recommendations"""
        reasoning_log.append("Generating shortlist recommendations")
        
        recommendations = []
        
        if not stocks:
            recommendations.append("Adjust filtering criteria to expand the shortlist.")
            return recommendations
        
        # Prioritization recommendations
        strong_buys = [s for s in stocks if s.get("recommendation") == "STRONG_BUY"]
        buys = [s for s in stocks if s.get("recommendation") == "BUY"]
        
        if strong_buys:
            recommendations.append(f"Prioritize {len(strong_buys)} strong buy candidates for immediate attention")
        
        if buys:
            recommendations.append(f"Review {len(buys)} buy candidates for potential opportunities")
        
        # Diversification recommendations
        sectors = list(set(s.get("sector") for s in stocks if s.get("sector")))
        if len(sectors) < 3:
            recommendations.append("Consider diversifying across more sectors")
        else:
            recommendations.append(f"Well-diversified across {len(sectors)} sectors")
        
        return recommendations
    
    def _generate_prediction_risks(self, predictions: List[Dict], 
                                 reasoning_log: List[str]) -> List[str]:
        """Generate prediction risk considerations"""
        reasoning_log.append("Generating prediction risk considerations")
        
        risks = []
        
        if not predictions:
            risks.append("No predictions available - market may be in consolidation phase")
            return risks
        
        # Model risk
        risks.append("Model risk: Predictions based on historical patterns may not reflect current market conditions")
        
        # Concentration risk
        if len(predictions) < 5:
            risks.append("Concentration risk: Limited number of opportunities identified")
        
        # Market risk
        risks.append("Market risk: External events can impact all predictions")
        
        return risks
    
    def _generate_shortlist_risks(self, stocks: List[Dict], 
                                reasoning_log: List[str]) -> List[str]:
        """Generate shortlist risk considerations"""
        reasoning_log.append("Generating shortlist risk considerations")
        
        risks = []
        
        if not stocks:
            risks.append("Filtering criteria may be too restrictive")
            return risks
        
        # Filtering risk
        risks.append("Filtering risk: May have excluded viable opportunities outside current criteria")
        
        # Overfitting risk
        risks.append("Overfitting risk: Current filters may not generalize to future market conditions")
        
        # Concentration risk
        sectors = list(set(s.get("sector") for s in stocks if s.get("sector")))
        if len(sectors) < 3:
            risks.append(f"Concentration risk: Overexposure to {', '.join(sectors[:2])} sectors")
        
        return risks
    
    async def _enhance_with_llm(self, context: str, data: Dict[str, Any], 
                              reasoning_log: List[str]) -> Optional[str]:
        """Enhance insights with local LLM (Ollama/Llama)"""
        try:
            if not self.ollama_enabled:
                return None
            
            # Import ollama
            try:
                import ollama
            except ImportError:
                logger.warning("Ollama not available for insight enhancement")
                return None
            
            # Prepare prompt for insight enhancement
            prompt = f"""
            Enhance these trading {context} insights with additional actionable advice:
            
            Data: {json.dumps(data, indent=2)}
            
            Provide concise, actionable insights that complement the existing analysis.
            Focus on risk management, timing, and portfolio positioning.
            Keep response under 100 words.
            """
            
            # Generate response from LLM
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "stop": ["\n\n"]
                }
            )
            
            # Extract enhanced insight
            enhanced_insight = response.get("response", "").strip()
            if enhanced_insight:
                reasoning_log.append("Successfully enhanced insight with LLM")
                return enhanced_insight
            else:
                return None
                
        except Exception as e:
            logger.warning(f"LLM insight enhancement failed: {e}")
            reasoning_log.append(f"LLM enhancement failed: {str(e)}")
            return None
    
    def _generate_error_insight(self, error_message: str) -> InsightResponse:
        """Generate insight for error cases"""
        return InsightResponse(
            summary="Error generating insight",
            detailed_insight=f"An error occurred while generating insights: {error_message}",
            actionable_recommendations=["Please try again or contact support"],
            risk_considerations=["Unable to assess risks due to error"],
            confidence=0.0,
            reasoning_log=[f"Error: {error_message}"],
            timestamp=datetime.now().isoformat()
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get insight agent status"""
        return {
            "agent_id": self.agent_id,
            "insights_generated": self.insights_generated,
            "ollama_enabled": self.ollama_enabled,
            "ollama_model": self.ollama_model,
            "status": "active"
        }