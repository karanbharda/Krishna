#!/usr/bin/env python3
"""
Production-Grade Grok API Integration
=====================================

Advanced Grok model integration for trading decision reasoning, explanation generation,
and intelligent market analysis with production-level error handling and optimization.
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# Production monitoring
from prometheus_client import Counter, Histogram, Gauge

# Load environment variables
from dotenv import load_dotenv
# Load .env file from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

logger = logging.getLogger(__name__)

# Metrics
GROK_REQUESTS = Counter('grok_requests_total',
                        'Total Grok API requests', ['model', 'status'])
GROK_RESPONSE_TIME = Histogram(
    'grok_response_time_seconds', 'Grok response time', ['model'])
GROK_TOKEN_COUNT = Counter(
    'grok_tokens_total', 'Total tokens processed', ['type'])


@dataclass
class GrokResponse:
    """Standardized Grok response structure"""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TradingContext:
    """Trading context for Grok analysis"""
    symbol: str
    current_price: float
    technical_signals: Dict[str, Any]
    market_data: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, Any]] = None
    fundamental_data: Optional[Dict[str, Any]] = None


class GrokAPIEngine:
    """
    Production-grade Grok integration for trading intelligence

    Features:
    - Multiple model support (Grok API)
    - Intelligent prompt engineering
    - Context-aware reasoning
    - Performance optimization
    - Comprehensive error handling
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("grok_api_key")
        self.base_url = config.get("grok_base_url", "https://api.x.ai/v1")
        self.model_name = config.get("grok_model", "grok-beta")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)

        # Session for connection pooling
        self.session = None

        # Prompt templates
        self.prompt_templates = self._load_prompt_templates()

        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0

        # Validate API key
        if not self.api_key:
            raise ValueError("Grok API key is required")

        logger.info(
            f"Grok API Engine initialized - Model: {self.model_name}")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check the health status of the Grok API engine"""
        try:
            # Check if API key is configured
            if not self.api_key:
                return {
                    "status": "error",
                    "message": "Grok API key not configured",
                    "model": self.model_name
                }

            # Check if we can make a simple API call
            async with self:
                # Simple model list request to verify connectivity
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                # Use the session if available, otherwise make direct request
                if self.session:
                    async with self.session.get(
                        f"{self.base_url}/models",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            return {
                                "status": "healthy",
                                "message": "Grok API connection successful",
                                "model": self.model_name,
                                "base_url": self.base_url
                            }
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "message": f"Grok API returned status {response.status}: {error_text}",
                                "model": self.model_name
                            }
                else:
                    # Fallback to direct requests if session not available
                    import requests
                    try:
                        response = requests.get(
                            f"{self.base_url}/models",
                            headers=headers,
                        )
                        if response.status_code == 200:
                            return {
                                "status": "healthy",
                                "message": "Grok API connection successful",
                                "model": self.model_name,
                                "base_url": self.base_url
                            }
                        else:
                            return {
                                "status": "error",
                                "message": f"Grok API returned status {response.status_code}: {response.text}",
                                "model": self.model_name
                            }
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Failed to connect to Grok API: {str(e)}",
                            "model": self.model_name
                        }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "model": self.model_name
            }

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load optimized prompt templates for different use cases"""
        return {
            "market_analysis": """
You are an expert quantitative analyst and trader with 20+ years of experience in Indian stock markets.

TRADING CONTEXT:
Symbol: {symbol}
Current Price: ₹{current_price}
Technical Signals: {technical_signals}
Market Data: {market_data}

ANALYSIS REQUIREMENTS:
1. Analyze the technical signals and market data
2. Provide a clear BUY/SELL/HOLD recommendation
3. Explain your reasoning step-by-step
4. Assess risk level (LOW/MEDIUM/HIGH)
5. Suggest position sizing and stop-loss levels
6. Consider market regime and volatility

RESPONSE FORMAT:
Recommendation: [BUY/SELL/HOLD]
Confidence: [0-100%]
Risk Level: [LOW/MEDIUM/HIGH]
Position Size: [% of portfolio]
Stop Loss: [₹ price level]

Reasoning:
[Detailed step-by-step analysis]

Key Factors:
- [Factor 1]
- [Factor 2]
- [Factor 3]

Market Outlook:
[Short-term and medium-term outlook]
""",

            "risk_assessment": """
You are a senior risk management expert specializing in Indian equity markets.

RISK ANALYSIS REQUEST:
Portfolio Context: {portfolio_context}
Proposed Trade: {trade_details}
Market Conditions: {market_conditions}
Risk Parameters: {risk_parameters}

ASSESSMENT REQUIREMENTS:
1. Calculate position-level risk metrics
2. Assess portfolio-level impact
3. Evaluate market risk factors
4. Recommend risk mitigation strategies
5. Provide risk-adjusted position sizing

RESPONSE FORMAT:
Risk Score: [1-10 scale]
Max Position Size: [% of portfolio]
Stop Loss: [₹ price level]
Risk/Reward Ratio: [X:1]

Risk Factors:
- [High impact factors]
- [Medium impact factors]
- [Low impact factors]

Mitigation Strategies:
- [Strategy 1]
- [Strategy 2]
- [Strategy 3]
""",

            "trade_explanation": """
You are an experienced trading mentor explaining decisions to a learning trader.

TRADE DETAILS:
Action: {action}
Symbol: {symbol}
Entry Price: ₹{entry_price}
Quantity: {quantity}
Reasoning: {reasoning}
Market Context: {market_context}

EXPLANATION REQUIREMENTS:
1. Explain why this trade was taken
2. Break down the decision-making process
3. Highlight key technical/fundamental factors
4. Discuss risk management approach
5. Set expectations for the trade

RESPONSE FORMAT:
Trade Summary:
[Clear, concise summary]

Decision Process:
1. [Step 1 of analysis]
2. [Step 2 of analysis]
3. [Step 3 of analysis]

Key Factors:
- Technical: [Technical reasoning]
- Fundamental: [If applicable]
- Risk Management: [Risk approach]

Expected Outcome:
- Target: ₹[price level]
- Timeline: [expected duration]
- Probability: [success probability]
""",

            "portfolio_optimization": """
You are a portfolio manager optimizing allocations for maximum risk-adjusted returns.

PORTFOLIO CONTEXT:
Current Holdings: {current_holdings}
Available Cash: ₹{available_cash}
Risk Profile: {risk_profile}
Market Outlook: {market_outlook}
Performance History: {performance_history}
Total Portfolio Value: ₹{total_value}
Unrealized PnL: ₹{unrealized_pnl}

OPTIMIZATION REQUIREMENTS:
1. Analyze current portfolio composition
2. Identify optimization opportunities
3. Suggest rebalancing actions
4. Consider correlation and diversification
5. Optimize for risk-adjusted returns

RESPONSE FORMAT:
Portfolio Score: [1-10]
Diversification Level: [LOW/MEDIUM/HIGH]

Recommended Actions:
- [Action 1: Increase/Decrease position]
- [Action 2: Add new position]
- [Action 3: Exit position]

Optimization Rationale:
[Detailed explanation of recommendations]

Expected Impact:
- Risk Reduction: [%]
- Return Enhancement: [%]
- Sharpe Ratio Improvement: [value]
""",

            "trade_recommendation": """
You are a professional trading advisor providing specific buy/sell recommendations.

USER QUERY: {user_query}

ANALYSIS CONTEXT:
Symbol: {symbol}
Current Market Conditions: {market_context}
Technical Analysis: {technical_signals}
Fundamental Outlook: {fundamental_data}

RECOMMENDATION REQUIREMENTS:
1. Provide a clear BUY/SELL/HOLD recommendation
2. Specify entry price levels
3. Define target prices and stop-loss levels
4. Explain the rationale behind your recommendation
5. Assess the risk/reward profile
6. Consider the user's risk profile if provided

RESPONSE FORMAT:
Recommendation: [BUY/SELL/HOLD]
Confidence Level: [0-100%]
Entry Point: [₹ price level or range]
Target Price: [₹ price level]
Stop Loss: [₹ price level]
Risk/Reward Ratio: [X:1]

Analysis Summary:
[Concise summary of key factors]

Detailed Reasoning:
[Step-by-step explanation of analysis]

Risk Considerations:
[Key risks and how they're managed]

Time Horizon:
[Expected holding period]
""",

            "general_trading_advice": """
You are a comprehensive trading and finance expert answering diverse questions.

USER QUESTION: {question}

PROVIDE GUIDANCE ON:
1. Trading strategies and techniques
2. Market analysis methods
3. Risk management principles
4. Portfolio construction and management
5. Behavioral finance and psychology
6. Market dynamics and economics

RESPONSE GUIDELINES:
- Be specific and actionable
- Provide concrete examples where relevant
- Consider both beginner and advanced traders
- Balance technical and fundamental approaches
- Address both short-term trading and long-term investing

RESPONSE FORMAT:
Main Answer:
[Direct, clear response to the question]

Key Points:
- [Point 1]
- [Point 2]
- [Point 3]

Practical Application:
[How to implement the advice]

Additional Considerations:
[Any relevant caveats or related topics]
"""
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_grok_request(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Make request to Grok API with retry logic"""
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Use async context manager.")

        model = model or self.model_name
        start_time = time.time()

        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }

            # Make request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:

                if response.status != 200:
                    error_text = await response.text()

                    # Handle specific error cases with clearer messages
                    if response.status == 403:
                        if "credits" in error_text.lower():
                            logger.error(
                                "Grok API error: Insufficient credits. Please purchase credits at https://console.x.ai/")
                            raise Exception(
                                "Grok API error: Insufficient credits. Please purchase credits at https://console.x.ai/")
                        elif "permission" in error_text.lower():
                            logger.error(
                                "Grok API error: Permission denied. Check your API key and account status.")
                            raise Exception(
                                "Grok API error: Permission denied. Check your API key and account status.")
                    elif response.status == 401:
                        logger.error(
                            "Grok API error: Invalid API key. Please check your GROK_API_KEY in the .env file.")
                        raise Exception(
                            "Grok API error: Invalid API key. Please check your GROK_API_KEY in the .env file.")
                    elif response.status == 400:
                        logger.error(
                            f"Grok API error {response.status}: Bad request. Check your request format. Details: {error_text}")
                        raise Exception(
                            f"Grok API error {response.status}: Bad request. Check your request format. Details: {error_text}")

                    raise Exception(
                        f"Grok API error {response.status}: {error_text}")

                result = await response.json()

                # Record metrics
                execution_time = time.time() - start_time
                GROK_RESPONSE_TIME.labels(model=model).observe(execution_time)
                GROK_REQUESTS.labels(model=model, status="success").inc()

                # Track tokens if available
                usage = result.get("usage", {})
                if "completion_tokens" in usage:
                    GROK_TOKEN_COUNT.labels(
                        type="output").inc(usage["completion_tokens"])
                if "prompt_tokens" in usage:
                    GROK_TOKEN_COUNT.labels(type="input").inc(
                        usage["prompt_tokens"])

                return result

        except Exception as e:
            GROK_REQUESTS.labels(model=model, status="error").inc()
            logger.error(f"Grok API request failed: {e}")
            raise

    async def analyze_market_decision(self, context: TradingContext) -> GrokResponse:
        """Generate comprehensive market analysis and trading decision"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["market_analysis"].format(
                symbol=context.symbol,
                current_price=context.current_price,
                technical_signals=json.dumps(
                    context.technical_signals, indent=2),
                market_data=json.dumps(context.market_data, indent=2)
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise

    async def assess_risk(self, portfolio_context: Dict[str, Any], trade_details: Dict[str, Any],
                          market_conditions: Dict[str, Any], risk_parameters: Dict[str, Any]) -> GrokResponse:
        """Assess risk for a proposed trade"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["risk_assessment"].format(
                portfolio_context=json.dumps(portfolio_context, indent=2),
                trade_details=json.dumps(trade_details, indent=2),
                market_conditions=json.dumps(market_conditions, indent=2),
                risk_parameters=json.dumps(risk_parameters, indent=2)
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            raise

    async def explain_trade(self, action: str, symbol: str, entry_price: float,
                            quantity: int, reasoning: str, market_context: Dict[str, Any]) -> GrokResponse:
        """Generate explanation for a trade decision"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["trade_explanation"].format(
                action=action,
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                reasoning=reasoning,
                market_context=json.dumps(market_context, indent=2)
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"Trade explanation failed: {e}")
            raise

    async def optimize_portfolio(self, current_holdings: List[Dict[str, Any]], available_cash: float,
                                 risk_profile: str, market_outlook: str,
                                 performance_history: Dict[str, Any], total_value: float,
                                 unrealized_pnl: float) -> GrokResponse:
        """Generate portfolio optimization recommendations"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["portfolio_optimization"].format(
                current_holdings=json.dumps(current_holdings, indent=2),
                available_cash=available_cash,
                risk_profile=risk_profile,
                market_outlook=market_outlook,
                performance_history=json.dumps(performance_history, indent=2),
                total_value=total_value,
                unrealized_pnl=unrealized_pnl
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise

    async def get_trade_recommendation(self, user_query: str, symbol: str,
                                       market_context: Dict[str, Any], technical_signals: Dict[str, Any],
                                       fundamental_data: Optional[Dict[str, Any]] = None) -> GrokResponse:
        """Generate specific trade recommendation"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["trade_recommendation"].format(
                user_query=user_query,
                symbol=symbol,
                market_context=json.dumps(market_context, indent=2),
                technical_signals=json.dumps(technical_signals, indent=2),
                fundamental_data=json.dumps(fundamental_data or {}, indent=2)
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"Trade recommendation failed: {e}")
            raise

    async def get_general_trading_advice(self, question: str) -> GrokResponse:
        """Answer general trading questions"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["general_trading_advice"].format(
                question=question
            )

            # Get Grok response
            start_time = time.time()
            response = await self._make_grok_request(prompt)
            execution_time = time.time() - start_time

            # Extract content from response
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Extract token usage
            tokens_used = None
            if "usage" in response:
                usage = response["usage"]
                tokens_used = usage.get("total_tokens", 0)

            return GrokResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata=response
            )

        except Exception as e:
            logger.error(f"General trading advice failed: {e}")
            raise
