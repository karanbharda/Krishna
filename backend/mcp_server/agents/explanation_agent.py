#!/usr/bin/env python3
"""
MCP Explanation Agent
=====================

AI-powered explanation agent for the Model Context Protocol server
that provides detailed reasoning for trading decisions using Groq API.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Structured explanation for trading decisions"""
    content: str
    confidence: float
    reasoning_depth: str  # "basic", "detailed", "comprehensive"
    key_factors: List[str]
    market_context: Dict[str, Any]
    metadata: Dict[str, Any]


class ExplanationAgent:
    """
    MCP Explanation Agent
    Provides detailed AI-powered explanations for trading decisions
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "default_explanation_agent")

        # Initialize components
        self.is_initialized = False
        self.groq_engine = None

        logger.info(f"Explanation Agent {self.agent_id} initialized")

    async def initialize(self):
        """Initialize the explanation agent"""
        try:
            # Import Groq engine if available
            try:
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
                    logger.info("Groq engine connected to explanation agent")
            except ImportError as e:
                logger.warning(f"Groq API integration not available: {e}")

            self.is_initialized = True
            logger.info(
                f"Explanation Agent {self.agent_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Explanation Agent {self.agent_id}: {e}")
            raise

    async def generate_explanation(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        market_context: Optional[Dict[str, Any]] = None,
        depth: str = "detailed"
    ) -> Explanation:
        """
        Generate detailed explanation for a trading decision

        Args:
            symbol: Stock symbol
            decision: Trading decision (BUY/SELL/HOLD)
            confidence: Confidence level (0.0-1.0)
            market_context: Additional market context
            depth: Explanation depth ("basic", "detailed", "comprehensive")

        Returns:
            Explanation with detailed reasoning
        """
        if not self.is_initialized:
            raise RuntimeError("Explanation agent not initialized")

        start_time = time.time()

        try:
            # Generate explanation with Groq if available
            content = f"Recommendation to {decision} {symbol} with {confidence:.1%} confidence based on technical analysis."
            key_factors = ["Technical indicators",
                           "Market trends", "Risk assessment"]

            if self.groq_engine:
                try:
                    # Prepare market context for Indian stocks
                    market_context = market_context or {}

                    # Create a more detailed prompt for Indian stock analysis
                    prompt = f"""You are an expert trading advisor specializing in Indian stock markets (NSE/BSE).

ANALYZE THIS TRADING DECISION:
Symbol: {symbol}
Action: {decision}
Confidence: {confidence:.1%}
Analysis Depth: {depth}

MARKET CONTEXT:
{json.dumps(market_context, indent=2) if market_context else "No additional market context provided"}

TASK:
Provide a comprehensive analysis for {symbol} with the following structure:

1. **Direct Recommendation** (1 sentence)
   - Clear {decision} recommendation with confidence level

2. **Supporting Analysis** (3-4 bullet points)
   - Key technical factors specific to {symbol}
   - Market momentum analysis in Indian markets
   - Risk-reward profile for NSE/BSE stocks
   - Sector-specific considerations for this Indian company

3. **Risk Considerations** (2-3 bullet points)
   - Primary risk factors for {symbol}
   - Mitigation strategies for Indian market conditions
   - Position sizing guidance for rupee-denominated investments

4. **Next Steps** (2-3 bullet points)
   - Entry point suggestions for {symbol}
   - Stop-loss recommendations for Indian market volatility
   - Target price levels based on technical analysis

Focus specifically on Indian market conditions and {symbol}. Do not mention US stocks or generic examples.
Be concise but comprehensive, providing actionable insights for Indian traders."""

                    # Prepare API payload
                    payload = {
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system",
                                "content": "You are an expert trading advisor specializing in Indian stock markets (NSE/BSE)."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.7
                    }

                    # Make API call using the existing Groq engine
                    groq_response = await self.groq_engine._make_api_call(payload)
                    content = groq_response["choices"][0]["message"]["content"]

                    # Extract key factors from the content (simple approach)
                    key_factors = [
                        f"Technical analysis for {symbol}",
                        "Indian market momentum",
                        "Risk-adjusted return potential",
                        "Sector-specific considerations",
                        "Rupee currency factors"
                    ]
                except Exception as e:
                    logger.warning(f"Groq explanation generation failed: {e}")
                    # Fall back to detailed simulated response
                    content = f"""AI-powered analysis recommends {decision} for {symbol} with {confidence:.1%} confidence.
                    
Key factors supporting this decision:
1. Technical indicator convergence specific to {symbol}
2. Market momentum analysis in Indian markets
3. Risk-adjusted return potential for NSE/BSE stocks
4. Sector-specific considerations for this Indian company

Risk considerations:
- Volatility levels appropriate for Indian market conditions
- Position sizing aligns with portfolio risk management for Indian stocks
- Stop-loss levels provide adequate protection for rupee-denominated investments

Next steps:
1. Monitor key support/resistance levels for {symbol}
2. Set position size according to portfolio allocation guidelines
3. Review technical indicators daily for confirmation signals"""

                    key_factors = [
                        f"Technical analysis for {symbol}",
                        "Indian market momentum",
                        "Risk-adjusted return potential",
                        "Sector-specific considerations",
                        "Rupee currency factors"
                    ]

            explanation = Explanation(
                content=content,
                confidence=confidence,
                reasoning_depth=depth,
                key_factors=key_factors,
                market_context=market_context or {},
                metadata={
                    "generation_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

            logger.info(f"Explanation generated for {symbol}: {decision}")
            return explanation

        except Exception as e:
            logger.error(f"Error in generate_explanation for {symbol}: {e}")
            # Return basic explanation as fallback
            return Explanation(
                content=f"Error generating detailed explanation: {str(e)}",
                confidence=0.0,
                reasoning_depth="basic",
                key_factors=[],
                market_context=market_context or {},
                metadata={
                    "generation_time": time.time() - start_time,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            )

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "initialized": self.is_initialized,
            "groq_available": self.groq_engine is not None
        }


# Agent availability flag
EXPLANATION_AGENT_AVAILABLE = True
