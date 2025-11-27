#!/usr/bin/env python3
"""
Production-Grade Groq API Integration
=====================================

Advanced Groq model integration for trading decision reasoning, explanation generation,
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
GROQ_REQUESTS = Counter('groq_requests_total',
                        'Total Groq API requests', ['model', 'status'])
GROQ_RESPONSE_TIME = Histogram(
    'groq_response_time_seconds', 'Groq response time', ['model'])
GROQ_TOKEN_COUNT = Counter(
    'groq_tokens_total', 'Total tokens processed', ['type'])


@dataclass
class GroqResponse:
    """Standardized Groq response structure"""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TradingContext:
    """Trading context for Groq analysis"""
    symbol: str
    current_price: float
    technical_signals: Dict[str, Any]
    market_data: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, Any]] = None
    fundamental_data: Optional[Dict[str, Any]] = None


class GroqAPIEngine:
    """
    Production-grade Groq integration for trading intelligence

    Features:
    - Multiple model support (Groq API)
    - Intelligent prompt engineering
    - Context-aware reasoning
    - Performance optimization
    - Comprehensive error handling
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("groq_api_key")
        self.base_url = config.get(
            "groq_base_url", "https://api.groq.com/openai/v1")
        self.model_name = config.get("groq_model", "llama-3.1-8b-instant")
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
            raise ValueError("Groq API key is required")

        logger.info(
            f"Groq API Engine initialized - Model: {self.model_name}")

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
        """Check the health status of the Groq API engine"""
        try:
            # Check if API key is configured
            if not self.api_key:
                return {
                    "status": "error",
                    "message": "Groq API key not configured",
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
                                "message": "Groq API connection successful",
                                "model": self.model_name,
                                "base_url": self.base_url
                            }
                        else:
                            error_text = await response.text()
                            return {
                                "status": "error",
                                "message": f"Groq API returned status {response.status}: {error_text}",
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
                                "message": "Groq API connection successful",
                                "model": self.model_name,
                                "base_url": self.base_url
                            }
                        else:
                            return {
                                "status": "error",
                                "message": f"Groq API returned status {response.status_code}: {response.text}",
                                "model": self.model_name
                            }
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Failed to connect to Groq API: {str(e)}",
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
You are an expert quantitative analyst and trader with 20+ years of experience in Indian stock markets (NSE/BSE).

TRADING CONTEXT:
Symbol: {symbol}
Current Price: ₹{current_price}
Technical Signals: {technical_signals}
Market Data: {market_data}

PORTFOLIO CONTEXT:
{portfolio_context}

RISK PARAMETERS:
{risk_parameters}

TASK:
Provide a comprehensive market analysis for {symbol} with the following structure:

1. **Immediate Assessment** (2-3 sentences)
   - Current market positioning in Indian markets
   - Key technical signal interpretation
   - Portfolio impact analysis

2. **Risk-Adjusted Recommendation** (1-2 sentences)
   - Clear BUY/HOLD/SELL recommendation
   - Confidence level (0-100%)
   - Position sizing suggestion

3. **Supporting Logic** (3-4 bullet points)
   - Technical analysis insights specific to Indian markets
   - Market context factors (sector trends, economic indicators)
   - Risk mitigation considerations
   - Portfolio optimization opportunities

4. **Execution Strategy** (2-3 bullet points)
   - Optimal entry/exit points
   - Stop-loss and take-profit levels
   - Time horizon recommendation

Format your response in clear sections with code-style headers. Be concise but comprehensive.
""",
            "risk_assessment": """
You are a senior risk management specialist for algorithmic trading systems.

TRADE DETAILS:
{trade_details}

PORTFOLIO CONTEXT:
{portfolio_context}

HISTORICAL PERFORMANCE:
{performance_history}

TASK:
Conduct a comprehensive risk assessment with the following structure:

1. **Risk Profile Summary**
   - Overall risk level (LOW/MODERATE/HIGH/CRITICAL)
   - Key risk factors identified
   - Portfolio concentration impact

2. **Quantitative Risk Metrics**
   - Value at Risk (VaR) estimation
   - Maximum drawdown potential
   - Correlation risk with existing positions

3. **Mitigation Strategies**
   - Specific risk controls to implement
   - Position sizing adjustments
   - Hedging opportunities

4. **Final Risk Rating**
   - Numerical risk score (0-100)
   - Go/No-Go recommendation
   - Monitoring requirements

Be precise and data-driven. Focus on actionable insights.
""",
            "portfolio_optimization": """
You are a portfolio optimization expert specializing in algorithmic trading strategies.

CURRENT HOLDINGS:
{current_holdings}

AVAILABLE CASH: ₹{available_cash}
RISK PROFILE: {risk_profile}
MARKET OUTLOOK: {market_outlook}
PORTFOLIO VALUE: ₹{total_value}
UNREALIZED P&L: ₹{unrealized_pnl}

TASK:
Provide a comprehensive portfolio optimization strategy with the following structure:

1. **Portfolio Health Assessment**
   - Overall diversification score
   - Sector/industry concentration analysis
   - Risk-adjusted return metrics

2. **Optimization Recommendations**
   - Holdings to increase/reduce/eliminate
   - New position opportunities
   - Cash allocation strategy

3. **Risk Management**
   - Portfolio-level risk controls
   - Correlation-based rebalancing
   - Stop-loss adjustments

4. **Implementation Plan**
   - Priority-ranked actions
   - Execution timing guidance
   - Performance monitoring metrics

Focus on practical, implementable strategies that align with the {risk_profile} risk profile.
""",
            "general_trading_advice": """
You are an expert trading advisor with deep knowledge of Indian financial markets (NSE/BSE), technical analysis, and risk management.

USER QUERY:
{query}

TASK:
Provide expert trading advice that is:
- Actionable and specific for Indian stock markets
- Risk-aware and balanced
- Based on sound trading principles
- Concise and clear

When recommending stocks, focus specifically on:
- NSE/BSE listed companies
- Indian market conditions and regulations
- Rupee-denominated investments
- Sector-specific considerations for Indian economy

Structure your response with:
1. **Direct Answer** - Address the core question first
2. **Supporting Analysis** - Provide relevant context and reasoning
3. **Risk Considerations** - Highlight key risks and mitigations
4. **Next Steps** - Suggest concrete actions

Keep responses focused on trading and investment topics in Indian markets. For non-trading queries, politely redirect to trading-related subjects.
""",
            "trade_recommendation": """
You are an expert trading advisor analyzing a specific trade opportunity.

USER QUERY: {user_query}

SYMBOL: {symbol}
CURRENT PRICE: ₹{current_price}
TECHNICAL SIGNALS: {technical_signals}
MARKET CONTEXT: {market_context}
FUNDAMENTAL DATA: {fundamental_data}

TASK:
Provide a comprehensive trade recommendation with the following structure:

1. **Trade Recommendation** (1 sentence)
   - Clear BUY/HOLD/SELL recommendation with confidence level (0-100%)

2. **Key Rationale** (2-3 bullet points)
   - Primary technical factors supporting the recommendation
   - Market context considerations
   - Risk-reward profile assessment

3. **Execution Parameters** (2-3 bullet points)
   - Optimal entry price range
   - Stop-loss level
   - Target price levels

4. **Risk Management** (1-2 bullet points)
   - Position sizing recommendation
   - Key risk factors to monitor

Be concise but thorough. Focus on actionable insights for the specific trade.
""",
            "mcp_explanation": """
You are an expert trading advisor and communicator acting as an intermediary between the Model Context Protocol (MCP) system and the user.
Your role is to translate complex MCP-generated data into clear, actionable insights for traders while maintaining the integrity of the underlying data.

USER QUERY: {user_query}

MCP DATA: {mcp_data}

EXPLANATION TYPE: {explanation_type}

TASK:
Explain the MCP data in a user-friendly way with the following structure:

1. **Direct Response** (1-2 sentences)
   - Provide a clear, direct answer to the user's query based on the MCP data
   - State the key findings and overall confidence level

2. **Key Insights** (3-4 bullet points)
   - Break down the most important insights from the MCP data
   - Explain what the data means for the user's specific query
   - Highlight critical factors and their implications for Indian markets
   - Include relevant numerical values, scores, and confidence levels

3. **Actionable Guidance** (2-3 bullet points)
   - Provide practical steps the user can take based on the MCP data
   - Offer specific trading recommendations with clear rationale
   - Address risk considerations for Indian stock market investments
   - Suggest time horizons and exit strategies where applicable

4. **Important Context** (1-2 sentences)
   - Explain any limitations or caveats of the data
   - Mention key assumptions made in the analysis
   - Note when additional data or monitoring would be beneficial

When communicating:
- Focus exclusively on NSE/BSE listed companies and rupee-denominated investments
- Keep all recommendations specific to Indian market conditions and regulations
- Reference sector-specific considerations for the Indian economy
- Present data with clear numerical values, confidence scores, and timeframes
- Do not generate new analysis beyond explaining the provided MCP data
- Avoid mentioning US stocks or generic international examples

Remember: You are a communicator, not a data generator. Your job is to faithfully explain what the MCP system has produced.
"""
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call with retry logic"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                connector=aiohttp.TCPConnector(limit=10)
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        model = payload.get("model", self.model_name)

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                execution_time = time.time() - start_time
                GROQ_RESPONSE_TIME.labels(model=model).observe(execution_time)

                if response.status == 200:
                    GROQ_REQUESTS.labels(model=model, status="success").inc()
                    result = await response.json()
                    return result
                else:
                    GROQ_REQUESTS.labels(model=model, status="error").inc()
                    error_text = await response.text()

                    # Handle specific error cases
                    if response.status == 401:
                        raise Exception(
                            "Groq API error: Invalid API key. Please check your GROQ_API_KEY in the .env file.")
                    elif response.status == 429:
                        raise Exception(
                            "Groq API error: Rate limit exceeded. Please wait before making more requests.")
                    elif response.status == 400:
                        raise Exception(
                            f"Groq API error: Bad request. Check your request format. Details: {error_text}")
                    else:
                        raise Exception(
                            f"Groq API error {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            GROQ_REQUESTS.labels(model=model, status="error").inc()
            raise Exception(f"Groq API request failed: {str(e)}")
        except Exception as e:
            GROQ_REQUESTS.labels(model=model, status="error").inc()
            raise e

    async def get_general_trading_advice(self, query: str, context: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """Get general trading advice for any query"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["general_trading_advice"]
            prompt = prompt_template.format(query=query)

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert trading advisor."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            return GroqResponse(
                content=content,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"General trading advice failed: {e}")
            raise Exception(f"General trading advice failed: {str(e)}")

    async def analyze_market_conditions(self, context: TradingContext) -> GroqResponse:
        """Analyze market conditions for a specific symbol"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["market_analysis"]

            # Format portfolio context
            portfolio_context = "No portfolio context provided"
            if context.portfolio_context:
                portfolio_context = json.dumps(
                    context.portfolio_context, indent=2)

            # Format risk parameters
            risk_parameters = "No risk parameters provided"
            if context.risk_parameters:
                risk_parameters = json.dumps(context.risk_parameters, indent=2)

            prompt = prompt_template.format(
                symbol=context.symbol,
                current_price=context.current_price,
                technical_signals=json.dumps(
                    context.technical_signals, indent=2),
                market_data=json.dumps(context.market_data, indent=2),
                portfolio_context=portfolio_context,
                risk_parameters=risk_parameters
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            return GroqResponse(
                content=content,
                reasoning="Market analysis based on technical signals and market data",
                confidence=0.85,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "symbol": context.symbol,
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise Exception(f"Market analysis failed: {str(e)}")

    async def analyze_market_decision(self, context: TradingContext) -> GroqResponse:
        """Analyze market decision based on trading context"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["market_analysis"]

            # Format portfolio context
            portfolio_context = "No portfolio context provided"
            if context.portfolio_context:
                portfolio_context = json.dumps(
                    context.portfolio_context, indent=2)

            # Format risk parameters
            risk_parameters = "No risk parameters provided"
            if context.risk_parameters:
                risk_parameters = json.dumps(context.risk_parameters, indent=2)

            prompt = prompt_template.format(
                symbol=context.symbol,
                current_price=context.current_price,
                technical_signals=json.dumps(
                    context.technical_signals, indent=2),
                market_data=json.dumps(context.market_data, indent=2),
                portfolio_context=portfolio_context,
                risk_parameters=risk_parameters
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert trading analyst providing market insights."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            return GroqResponse(
                content=content,
                reasoning="Market analysis based on technical signals and market data",
                confidence=0.85,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "symbol": context.symbol,
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"Market decision analysis failed: {e}")
            raise Exception(f"Market decision analysis failed: {str(e)}")

    async def assess_risk(self, trade_details: Dict[str, Any],
                          portfolio_context: Optional[Dict[str, Any]] = None,
                          performance_history: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """Assess risk for a specific trade"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["risk_assessment"]

            prompt = prompt_template.format(
                trade_details=json.dumps(trade_details, indent=2),
                portfolio_context=json.dumps(
                    portfolio_context or {}, indent=2),
                performance_history=json.dumps(
                    performance_history or {}, indent=2)
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system",
                        "content": "You are a senior risk management specialist."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.3  # Lower temperature for more consistent risk assessments
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            return GroqResponse(
                content=content,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "trade_type": trade_details.get("type", "unknown"),
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            raise Exception(f"Risk assessment failed: {str(e)}")

    async def optimize_portfolio(self, current_holdings: List[Dict[str, Any]],
                                 available_cash: float,
                                 risk_profile: str,
                                 market_outlook: str,
                                 performance_history: Optional[Dict[str, Any]] = None,
                                 total_value: Optional[float] = None,
                                 unrealized_pnl: Optional[float] = None) -> GroqResponse:
        """Optimize portfolio allocation"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["portfolio_optimization"]

            prompt = prompt_template.format(
                current_holdings=json.dumps(current_holdings, indent=2),
                available_cash=available_cash,
                risk_profile=risk_profile,
                market_outlook=market_outlook,
                total_value=total_value or available_cash,
                unrealized_pnl=unrealized_pnl or 0,
                performance_history=json.dumps(
                    performance_history or {}, indent=2)
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system",
                        "content": "You are a portfolio optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.5  # Moderate temperature for balanced optimization
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            return GroqResponse(
                content=content,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "holdings_count": len(current_holdings),
                    "available_cash": available_cash,
                    "risk_profile": risk_profile,
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise Exception(f"Portfolio optimization failed: {str(e)}")

    async def get_trade_recommendation(self, user_query: str, symbol: str,
                                       market_context: Dict[str, Any],
                                       technical_signals: Dict[str, Any],
                                       fundamental_data: Dict[str, Any],
                                       current_price: float = 0.0) -> GroqResponse:
        """Get specific trade recommendation for a symbol"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["trade_recommendation"]

            prompt = prompt_template.format(
                user_query=user_query,
                symbol=symbol,
                current_price=current_price,
                technical_signals=json.dumps(technical_signals, indent=2),
                market_context=json.dumps(market_context, indent=2),
                fundamental_data=json.dumps(fundamental_data, indent=2)
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert trading advisor."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            # Extract confidence from content (simple approach)
            confidence = 0.8  # Default confidence

            return GroqResponse(
                content=content,
                reasoning="Generated trade recommendation based on technical signals and market context",
                confidence=confidence,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "symbol": symbol,
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"Trade recommendation failed: {e}")
            raise Exception(f"Trade recommendation failed: {str(e)}")

    async def explain_mcp_data(self, user_query: str, mcp_data: Dict[str, Any], explanation_type: str) -> GroqResponse:
        """Explain MCP data in a user-friendly way"""
        try:
            # Prepare prompt
            prompt_template = self.prompt_templates["mcp_explanation"]

            # Sanitize MCP data to remove empty or invalid entries
            sanitized_mcp_data = self._sanitize_mcp_data(mcp_data)

            prompt = prompt_template.format(
                user_query=user_query,
                mcp_data=json.dumps(sanitized_mcp_data, indent=2),
                explanation_type=explanation_type
            )

            # Prepare API payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system",
                        "content": "You are an expert trading advisor and communicator explaining MCP data."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.7
            }

            # Make API call
            start_time = time.time()
            response = await self._make_api_call(payload)
            execution_time = time.time() - start_time

            # Extract content
            content = response["choices"][0]["message"]["content"]

            # Count tokens (approximate)
            # Simple word count approximation
            input_tokens = len(prompt.split())
            output_tokens = len(content.split())
            total_tokens = input_tokens + output_tokens
            GROQ_TOKEN_COUNT.labels(type="input").inc(input_tokens)
            GROQ_TOKEN_COUNT.labels(type="output").inc(output_tokens)

            # Extract confidence from content (simple approach)
            confidence = 0.9  # High confidence for data explanation

            return GroqResponse(
                content=content,
                reasoning=f"Explanation of MCP data for {explanation_type}",
                confidence=confidence,
                tokens_used=total_tokens,
                model_used=self.model_name,
                execution_time=execution_time,
                metadata={
                    "explanation_type": explanation_type,
                    "prompt_length": len(prompt),
                    "response_length": len(content)
                }
            )

        except Exception as e:
            logger.error(f"MCP data explanation failed: {e}")
            raise Exception(f"MCP data explanation failed: {str(e)}")

    def _sanitize_mcp_data(self, mcp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize MCP data to remove empty or invalid entries"""
        if not mcp_data:
            return {}

        sanitized = {}
        for key, value in mcp_data.items():
            # Skip empty data
            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                continue
            # Skip error entries
            if isinstance(value, dict) and 'error' in str(value).lower():
                continue
            # Skip entries with only error information
            if isinstance(value, list):
                filtered_list = [item for item in value if item and not (
                    isinstance(item, dict) and 'error' in str(item).lower())]
                if filtered_list:
                    sanitized[key] = filtered_list
            else:
                sanitized[key] = value

        return sanitized

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Groq API engine cleaned up")
