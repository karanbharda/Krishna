#!/usr/bin/env python3
"""
Production-Grade Llama AI Integration
====================================

Advanced Llama model integration for trading decision reasoning, explanation generation,
and intelligent market analysis with production-level error handling and optimization.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# Production monitoring
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
LLAMA_REQUESTS = Counter('llama_requests_total', 'Total Llama API requests', ['model', 'status'])
LLAMA_RESPONSE_TIME = Histogram('llama_response_time_seconds', 'Llama response time', ['model'])
LLAMA_TOKEN_COUNT = Counter('llama_tokens_total', 'Total tokens processed', ['type'])

@dataclass
class LlamaResponse:
    """Standardized Llama response structure"""
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TradingContext:
    """Trading context for Llama analysis"""
    symbol: str
    current_price: float
    technical_signals: Dict[str, Any]
    market_data: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_parameters: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, Any]] = None

class LlamaReasoningEngine:
    """
    Production-grade Llama integration for trading intelligence
    
    Features:
    - Multiple model support (local Ollama, cloud APIs)
    - Intelligent prompt engineering
    - Context-aware reasoning
    - Performance optimization
    - Comprehensive error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("llama_base_url", "http://localhost:11434")
        self.model_name = config.get("llama_model", "llama3.1:8b")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        
        # Session for connection pooling
        self.session = None
        
        # Prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens = 0
        
        logger.info(f"Llama Reasoning Engine initialized - Model: {self.model_name}")
    
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
"""
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_llama_request(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Make request to Llama API with retry logic"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        model = model or self.model_name
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # Make request
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Llama API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Record metrics
                execution_time = time.time() - start_time
                LLAMA_RESPONSE_TIME.labels(model=model).observe(execution_time)
                LLAMA_REQUESTS.labels(model=model, status="success").inc()
                
                # Track tokens if available
                if "eval_count" in result:
                    LLAMA_TOKEN_COUNT.labels(type="output").inc(result["eval_count"])
                if "prompt_eval_count" in result:
                    LLAMA_TOKEN_COUNT.labels(type="input").inc(result["prompt_eval_count"])
                
                return result
                
        except Exception as e:
            LLAMA_REQUESTS.labels(model=model, status="error").inc()
            logger.error(f"Llama API request failed: {e}")
            raise
    
    async def analyze_market_decision(self, context: TradingContext) -> LlamaResponse:
        """Generate comprehensive market analysis and trading decision"""
        try:
            # Prepare prompt
            prompt = self.prompt_templates["market_analysis"].format(
                symbol=context.symbol,
                current_price=context.current_price,
                technical_signals=json.dumps(context.technical_signals, indent=2),
                market_data=json.dumps(context.market_data, indent=2)
            )
            
            # Get Llama response
            result = await self._make_llama_request(prompt)
            
            # Parse response
            content = result.get("response", "")
            
            # Extract structured information
            recommendation, confidence, reasoning = self._parse_market_analysis(content)
            
            return LlamaResponse(
                content=content,
                reasoning=reasoning,
                confidence=confidence,
                tokens_used=result.get("eval_count", 0),
                model_used=self.model_name,
                execution_time=result.get("total_duration", 0) / 1e9,  # Convert to seconds
                metadata={
                    "recommendation": recommendation,
                    "context": "market_analysis"
                }
            )
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return LlamaResponse(
                content=f"Error in market analysis: {str(e)}",
                confidence=0.0
            )
    
    async def assess_trade_risk(self, trade_details: Dict[str, Any], portfolio_context: Dict[str, Any]) -> LlamaResponse:
        """Assess risk for a proposed trade"""
        try:
            prompt = self.prompt_templates["risk_assessment"].format(
                portfolio_context=json.dumps(portfolio_context, indent=2),
                trade_details=json.dumps(trade_details, indent=2),
                market_conditions=json.dumps(trade_details.get("market_conditions", {}), indent=2),
                risk_parameters=json.dumps(trade_details.get("risk_parameters", {}), indent=2)
            )
            
            result = await self._make_llama_request(prompt)
            content = result.get("response", "")
            
            # Parse risk assessment
            risk_score, risk_factors = self._parse_risk_assessment(content)
            
            return LlamaResponse(
                content=content,
                confidence=risk_score / 10.0,  # Convert to 0-1 scale
                tokens_used=result.get("eval_count", 0),
                model_used=self.model_name,
                metadata={
                    "risk_score": risk_score,
                    "risk_factors": risk_factors,
                    "context": "risk_assessment"
                }
            )
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return LlamaResponse(
                content=f"Error in risk assessment: {str(e)}",
                confidence=0.0
            )
    
    async def explain_trade_decision(self, trade_action: str, context: TradingContext) -> LlamaResponse:
        """Generate detailed explanation for a trade decision"""
        try:
            prompt = self.prompt_templates["trade_explanation"].format(
                action=trade_action,
                symbol=context.symbol,
                entry_price=context.current_price,
                quantity="TBD",  # Will be determined by position sizing
                reasoning=json.dumps(context.technical_signals, indent=2),
                market_context=json.dumps(context.market_data, indent=2)
            )
            
            result = await self._make_llama_request(prompt)
            content = result.get("response", "")
            
            return LlamaResponse(
                content=content,
                reasoning=content,  # Full content is the reasoning
                confidence=0.8,  # High confidence for explanations
                tokens_used=result.get("eval_count", 0),
                model_used=self.model_name,
                metadata={"context": "trade_explanation"}
            )
            
        except Exception as e:
            logger.error(f"Trade explanation error: {e}")
            return LlamaResponse(
                content=f"Error generating explanation: {str(e)}",
                confidence=0.0
            )
    
    async def optimize_portfolio(self, portfolio_data: Dict[str, Any]) -> LlamaResponse:
        """Generate portfolio optimization recommendations"""
        try:
            prompt = self.prompt_templates["portfolio_optimization"].format(
                current_holdings=json.dumps(portfolio_data.get("holdings", {}), indent=2),
                available_cash=portfolio_data.get("cash", 0),
                risk_profile=portfolio_data.get("risk_profile", "MEDIUM"),
                market_outlook=json.dumps(portfolio_data.get("market_outlook", {}), indent=2),
                performance_history=json.dumps(portfolio_data.get("performance", {}), indent=2)
            )
            
            result = await self._make_llama_request(prompt)
            content = result.get("response", "")
            
            # Parse optimization recommendations
            actions = self._parse_portfolio_optimization(content)
            
            return LlamaResponse(
                content=content,
                reasoning=content,
                confidence=0.75,
                tokens_used=result.get("eval_count", 0),
                model_used=self.model_name,
                metadata={
                    "optimization_actions": actions,
                    "context": "portfolio_optimization"
                }
            )
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return LlamaResponse(
                content=f"Error in portfolio optimization: {str(e)}",
                confidence=0.0
            )
    
    def _parse_market_analysis(self, content: str) -> tuple[str, float, str]:
        """Parse market analysis response to extract structured data"""
        try:
            lines = content.split('\n')
            recommendation = "HOLD"
            confidence = 0.5
            reasoning = content
            
            for line in lines:
                if "Recommendation:" in line:
                    recommendation = line.split(":")[-1].strip()
                elif "Confidence:" in line:
                    conf_str = line.split(":")[-1].strip().replace("%", "")
                    try:
                        confidence = float(conf_str) / 100.0
                    except ValueError:
                        confidence = 0.5
            
            return recommendation, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error parsing market analysis: {e}")
            return "HOLD", 0.5, content
    
    def _parse_risk_assessment(self, content: str) -> tuple[float, List[str]]:
        """Parse risk assessment response"""
        try:
            lines = content.split('\n')
            risk_score = 5.0
            risk_factors = []
            
            for line in lines:
                if "Risk Score:" in line:
                    try:
                        risk_score = float(line.split(":")[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        risk_score = 5.0
                elif line.strip().startswith("- "):
                    risk_factors.append(line.strip()[2:])
            
            return risk_score, risk_factors
            
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {e}")
            return 5.0, []
    
    def _parse_portfolio_optimization(self, content: str) -> List[Dict[str, str]]:
        """Parse portfolio optimization recommendations"""
        try:
            lines = content.split('\n')
            actions = []
            
            in_actions_section = False
            for line in lines:
                if "Recommended Actions:" in line:
                    in_actions_section = True
                    continue
                elif in_actions_section and line.strip().startswith("- "):
                    action_text = line.strip()[2:]
                    actions.append({"action": action_text, "type": "optimization"})
                elif in_actions_section and not line.strip().startswith("- ") and line.strip():
                    in_actions_section = False
            
            return actions
            
        except Exception as e:
            logger.error(f"Error parsing portfolio optimization: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Llama service health"""
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/tags") as response:
                        if response.status == 200:
                            models = await response.json()
                            return {
                                "status": "healthy",
                                "available_models": [m.get("name") for m in models.get("models", [])],
                                "current_model": self.model_name
                            }
            else:
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        return {
                            "status": "healthy",
                            "available_models": [m.get("name") for m in models.get("models", [])],
                            "current_model": self.model_name
                        }
            
            return {"status": "unhealthy", "error": "Service not responding"}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
