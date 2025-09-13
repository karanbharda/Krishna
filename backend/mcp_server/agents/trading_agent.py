#!/usr/bin/env python3
"""
Intelligent Trading Agent
=========================

Advanced AI trading agent with multi-step reasoning, adaptive strategies,
and autonomous decision-making capabilities for production trading.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Critical Fix: Lazy imports to prevent circular dependencies
import sys
import os
from typing import TYPE_CHECKING

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Critical Fix: Use TYPE_CHECKING for type hints only
if TYPE_CHECKING:
    from llama_integration import LlamaReasoningEngine, TradingContext, LlamaResponse
    from fyers_client import FyersAPIClient
    from mcp_server.tools.market_analysis_tool import MarketAnalysisTool

# Runtime imports will be done lazily in methods

logger = logging.getLogger(__name__)

class TradingDecision(Enum):
    """Trading decision types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    ERROR = "error"

@dataclass
class TradingSignal:
    """Comprehensive trading signal"""
    symbol: str
    decision: TradingDecision
    confidence: float
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float  # Percentage of portfolio
    reasoning: str
    risk_score: float
    expected_return: float
    time_horizon: str
    metadata: Dict[str, Any]

@dataclass
class AgentMemory:
    """Agent memory for learning and adaptation"""
    successful_trades: List[Dict[str, Any]]
    failed_trades: List[Dict[str, Any]]
    market_patterns: Dict[str, Any]
    performance_metrics: Dict[str, float]
    learned_strategies: List[Dict[str, Any]]
    last_updated: datetime

class TradingAgent:
    """
    Intelligent Trading Agent with Advanced AI Capabilities
    
    Features:
    - Multi-step reasoning with Llama AI
    - Adaptive strategy selection
    - Continuous learning from trades
    - Risk-aware position sizing
    - Market regime detection
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get("agent_id", "trading_agent_001")
        self.state = AgentState.IDLE
        
        # Critical Fix: Initialize components with lazy loading
        self._llama_engine = None
        self._fyers_client = None
        self._market_analyzer = None
        
        # Agent memory and learning
        self.memory = AgentMemory(
            successful_trades=[],
            failed_trades=[],
            market_patterns={},
            performance_metrics={},
            learned_strategies=[],
            last_updated=datetime.now()
        )
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Strategy parameters
        self.risk_tolerance = config.get("risk_tolerance", 0.02)  # 2% max risk per trade
        self.max_positions = config.get("max_positions", 5)
        self.min_confidence = config.get("min_confidence", 0.7)
        
        # ENHANCED ML INTEGRATION: New parameters for better ML utilization
        self.ml_weight_multiplier = config.get("ml_weight_multiplier", 1.5)  # Boost ML signal importance
        self.ml_confidence_threshold = config.get("ml_confidence_threshold", 0.6)  # Minimum ML confidence
        self.ensemble_model_count = config.get("ensemble_model_count", 3)  # Number of ML models to ensemble
        
        logger.info(f"Trading Agent {self.agent_id} initialized with enhanced ML integration")

    def _get_llama_engine(self):
        """Critical Fix: Lazy import to prevent circular dependencies"""
        if self._llama_engine is None:
            try:
                from llama_integration import LlamaReasoningEngine
                self._llama_engine = LlamaReasoningEngine(self.config.get("llama", {}))
            except ImportError as e:
                logger.error(f"Failed to import LlamaReasoningEngine: {e}")
                self._llama_engine = False
        return self._llama_engine if self._llama_engine is not False else None

    def _get_fyers_client(self):
        """Critical Fix: Lazy import to prevent circular dependencies"""
        if self._fyers_client is None:
            try:
                from fyers_client import FyersAPIClient
                self._fyers_client = FyersAPIClient(self.config.get("fyers", {}))
            except ImportError as e:
                logger.error(f"Failed to import FyersAPIClient: {e}")
                self._fyers_client = False
        return self._fyers_client if self._fyers_client is not False else None

    def _get_market_analyzer(self):
        """Critical Fix: Lazy import to prevent circular dependencies"""
        if self._market_analyzer is None:
            try:
                from mcp_server.tools.market_analysis_tool import MarketAnalysisTool
                self._market_analyzer = MarketAnalysisTool(self.config)
            except ImportError as e:
                logger.error(f"Failed to import MarketAnalysisTool: {e}")
                self._market_analyzer = False
        return self._market_analyzer if self._market_analyzer is not False else None

    async def initialize(self):
        """Initialize agent components"""
        try:
            # Initialize Llama reasoning engine
            llama_config = self.config.get("llama", {})
            self.llama_engine = LlamaReasoningEngine(llama_config)
            
            # Initialize Fyers client
            fyers_config = self.config.get("fyers", {})
            self.fyers_client = FyersAPIClient(fyers_config)
            
            # Initialize market analyzer
            self.market_analyzer = MarketAnalysisTool(self.fyers_client)
            
            logger.info(f"Agent {self.agent_id} components initialized")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self.state = AgentState.ERROR
            raise
    
    async def analyze_and_decide(self, symbol: str, market_context: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """
        Perform comprehensive analysis and generate trading decision
        
        This is the main decision-making pipeline:
        1. Market analysis with technical indicators
        2. AI-powered reasoning with Llama
        3. Risk assessment and position sizing
        4. Final decision synthesis
        """
        try:
            self.state = AgentState.ANALYZING
            
            # Step 1: Technical Analysis
            technical_analysis = await self._perform_technical_analysis(symbol)
            
            # Step 2: Market Context Analysis
            market_data = await self._gather_market_context(symbol, market_context)
            
            # Step 3: AI Reasoning
            self.state = AgentState.DECIDING

            # Runtime import to avoid circular dependency
            try:
                from llama_integration import TradingContext
            except ImportError:
                logger.error("TradingContext not available - using fallback")
                # Create a simple dict as fallback
                trading_context = {
                    "symbol": symbol,
                    "current_price": market_data["current_price"],
                    "technical_signals": technical_analysis,
                    "market_data": market_data,
                    "portfolio_context": await self._get_portfolio_context(),
                    "risk_parameters": self._get_risk_parameters()
                }
            else:
                trading_context = TradingContext(
                    symbol=symbol,
                    current_price=market_data["current_price"],
                    technical_signals=technical_analysis,
                    market_data=market_data,
                    portfolio_context=await self._get_portfolio_context(),
                    risk_parameters=self._get_risk_parameters()
                )
            
            # Get AI decision
            async with self.llama_engine:
                ai_decision = await self.llama_engine.analyze_market_decision(trading_context)
            
            # Step 4: ENHANCED ML Integration - Get ensemble predictions
            ensemble_predictions = await self._get_ensemble_ml_predictions(trading_context)
            
            # Step 5: Risk Assessment
            risk_assessment = await self._assess_trade_risk(trading_context, ai_decision, ensemble_predictions)
            
            # Step 6: Position Sizing
            position_size = self._calculate_position_size(risk_assessment, trading_context, ensemble_predictions)
            
            # Step 7: Generate Final Signal
            signal = self._synthesize_trading_signal(
                symbol, trading_context, ai_decision, risk_assessment, position_size, ensemble_predictions
            )
            
            # Step 8: Learn and Adapt
            await self._update_agent_memory(signal, trading_context)
            
            self.state = AgentState.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Analysis and decision error for {symbol}: {e}")
            self.state = AgentState.ERROR
            
            # Return safe default signal
            return TradingSignal(
                symbol=symbol,
                decision=TradingDecision.WAIT,
                confidence=0.0,
                entry_price=0.0,
                target_price=None,
                stop_loss=None,
                position_size=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                risk_score=10.0,  # Maximum risk
                expected_return=0.0,
                time_horizon="N/A",
                metadata={"error": str(e)}
            )
    
    async def _perform_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Use the market analysis tool
            analysis_args = {
                "symbol": symbol,
                "timeframe": "1D",
                "lookback_days": 100,
                "analysis_type": "comprehensive"
            }
            
            result = await self.market_analyzer.analyze_market(analysis_args, f"session_{self.agent_id}")
            
            if result.status.value == "success":
                return result.data["technical_signals"]
            else:
                logger.warning(f"Technical analysis failed for {symbol}: {result.error}")
                return {"error": result.error}
                
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _gather_market_context(self, symbol: str, external_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather comprehensive market context"""
        try:
            # Get current market data
            quotes = await self.fyers_client.get_quotes([symbol])
            current_data = quotes.get(symbol)
            
            if not current_data:
                raise ValueError(f"No market data available for {symbol}")
            
            # Get market depth
            depth_data = await self.fyers_client.get_market_depth(symbol)
            
            # Compile market context
            context = {
                "current_price": current_data.ltp,
                "bid_ask_spread": abs(depth_data.asks[0]["price"] - depth_data.bids[0]["price"]) if depth_data.asks and depth_data.bids else 0,
                "volume": current_data.volume,
                "volatility": abs(current_data.high_price - current_data.low_price) / current_data.ltp,
                "market_depth": {
                    "bid_levels": len(depth_data.bids),
                    "ask_levels": len(depth_data.asks),
                    "total_bid_size": sum(bid["size"] for bid in depth_data.bids),
                    "total_ask_size": sum(ask["size"] for ask in depth_data.asks)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add external context if provided
            if external_context:
                context.update(external_context)
            
            return context
            
        except Exception as e:
            logger.error(f"Market context gathering error for {symbol}: {e}")
            return {"error": str(e), "current_price": 0.0}
    
    async def _get_portfolio_context(self) -> Dict[str, Any]:
        """Get current portfolio context"""
        try:
            # Get current positions and funds
            positions = await self.fyers_client.get_positions()
            funds = await self.fyers_client.get_funds()
            
            return {
                "total_positions": len(positions),
                "available_margin": funds.get("availableMargin", 0),
                "used_margin": funds.get("usedMargin", 0),
                "total_capital": funds.get("totalBalance", 0),
                "current_exposure": sum(pos.get("netQty", 0) * pos.get("ltp", 0) for pos in positions)
            }
            
        except Exception as e:
            logger.error(f"Portfolio context error: {e}")
            return {"error": str(e)}
    
    def _get_risk_parameters(self) -> Dict[str, Any]:
        """Get current risk parameters"""
        return {
            "max_risk_per_trade": self.risk_tolerance,
            "max_positions": self.max_positions,
            "min_confidence": self.min_confidence,
            "max_drawdown_limit": 0.15,  # 15% max drawdown
            "position_correlation_limit": 0.7
        }
    
    async def _get_ensemble_ml_predictions(self, context: "TradingContext") -> Dict[str, Any]:
        """
        ENHANCED ML INTEGRATION: Get predictions from multiple ML models and ensemble them
        """
        try:
            # Get predictions from multiple ML models
            predictions = []
            
            # Get LSTM prediction
            lstm_pred = await self._get_lstm_prediction(context)
            if lstm_pred:
                predictions.append(lstm_pred)
            
            # Get Transformer prediction
            transformer_pred = await self._get_transformer_prediction(context)
            if transformer_pred:
                predictions.append(transformer_pred)
            
            # Get RL prediction
            rl_pred = await self._get_rl_prediction(context)
            if rl_pred:
                predictions.append(rl_pred)
            
            # Ensemble the predictions
            if predictions:
                # Weighted average based on confidence
                total_confidence = sum(pred.get("confidence", 0.5) for pred in predictions)
                if total_confidence > 0:
                    ensemble_direction = sum(
                        pred.get("direction", 0) * pred.get("confidence", 0.5) 
                        for pred in predictions
                    ) / total_confidence
                    
                    ensemble_confidence = sum(
                        pred.get("confidence", 0.5) 
                        for pred in predictions
                    ) / len(predictions)
                else:
                    ensemble_direction = 0
                    ensemble_confidence = 0
                
                return {
                    "ensemble_direction": ensemble_direction,
                    "ensemble_confidence": ensemble_confidence,
                    "individual_predictions": predictions,
                    "model_count": len(predictions)
                }
            else:
                return {
                    "ensemble_direction": 0,
                    "ensemble_confidence": 0,
                    "individual_predictions": [],
                    "model_count": 0
                }
                
        except Exception as e:
            logger.error(f"Ensemble ML prediction error: {e}")
            return {
                "ensemble_direction": 0,
                "ensemble_confidence": 0,
                "individual_predictions": [],
                "model_count": 0,
                "error": str(e)
            }
    
    async def _get_lstm_prediction(self, context: "TradingContext") -> Optional[Dict[str, Any]]:
        """Get LSTM model prediction"""
        try:
            # This would typically call an LSTM model service
            # For now, we'll simulate a prediction based on context
            technical_signals = context.technical_signals or {}
            
            # Simple LSTM simulation based on RSI and MACD
            rsi = technical_signals.get("rsi", 50)
            macd = technical_signals.get("macd", 0)
            macd_signal = technical_signals.get("macd_signal", 0)
            
            # LSTM logic: bullish if RSI < 70 and MACD > MACD signal
            direction = 1 if (rsi < 70 and macd > macd_signal) else (-1 if (rsi > 30 and macd < macd_signal) else 0)
            
            # Confidence based on signal strength
            confidence = min(abs(rsi - 50) / 50, 1.0) if direction != 0 else 0.3
            
            return {
                "model": "LSTM",
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
    
    async def _get_transformer_prediction(self, context: "TradingContext") -> Optional[Dict[str, Any]]:
        """Get Transformer model prediction"""
        try:
            # This would typically call a Transformer model service
            # For now, we'll simulate a prediction based on context
            market_data = context.market_data or {}
            
            # Simple Transformer simulation based on volatility and volume
            volatility = market_data.get("volatility", 0.02)
            volume = market_data.get("volume", 1000)
            
            # Transformer logic: bullish in low volatility with high volume
            direction = 1 if (volatility < 0.03 and volume > 1000) else (-1 if (volatility > 0.05) else 0)
            
            # Confidence based on volatility level
            confidence = 1.0 - min(volatility / 0.1, 1.0)
            
            return {
                "model": "Transformer",
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return None
    
    async def _get_rl_prediction(self, context: "TradingContext") -> Optional[Dict[str, Any]]:
        """Get Reinforcement Learning model prediction"""
        try:
            # This would typically call an RL model service
            # For now, we'll simulate a prediction based on context
            portfolio_context = context.portfolio_context or {}
            
            # Simple RL simulation based on portfolio conditions
            total_positions = portfolio_context.get("total_positions", 0)
            available_margin = portfolio_context.get("available_margin", 10000)
            current_exposure = portfolio_context.get("current_exposure", 0)
            
            # RL logic: bullish if we have room for more positions and sufficient margin
            if total_positions < self.max_positions and available_margin > current_exposure * 0.2:
                direction = 1
                confidence = min(available_margin / 100000, 1.0)
            elif total_positions > self.max_positions * 0.8:
                direction = -1
                confidence = min(total_positions / self.max_positions, 1.0)
            else:
                direction = 0
                confidence = 0.5
            
            return {
                "model": "ReinforcementLearning",
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            return None
    
    async def _assess_trade_risk(self, context: "TradingContext", ai_decision: "LlamaResponse", ensemble_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess comprehensive trade risk"""
        try:
            # Prepare risk assessment data
            trade_details = {
                "symbol": context.symbol,
                "proposed_action": ai_decision.metadata.get("recommendation", "HOLD"),
                "confidence": ai_decision.confidence,
                "market_conditions": context.market_data,
                "risk_parameters": context.risk_parameters,
                "ml_predictions": ensemble_predictions
            }
            
            # Get AI risk assessment
            async with self.llama_engine:
                risk_response = await self.llama_engine.assess_trade_risk(
                    trade_details, context.portfolio_context
                )
            
            # Calculate quantitative risk metrics
            volatility = context.market_data.get("volatility", 0.02)
            risk_score = min(volatility * 10, 10)  # Scale to 1-10
            
            # ENHANCED RISK ASSESSMENT: Incorporate ML predictions
            ml_confidence = ensemble_predictions.get("ensemble_confidence", 0.5)
            ml_direction = ensemble_predictions.get("ensemble_direction", 0)
            
            # Adjust risk score based on ML confidence and agreement with AI decision
            ai_recommendation = ai_decision.metadata.get("recommendation", "HOLD")
            ai_direction = 1 if ai_recommendation == "BUY" else (-1 if ai_recommendation == "SELL" else 0)
            
            # If ML and AI agree, reduce risk score; if they disagree, increase it
            agreement_factor = 1.0 if (ml_direction * ai_direction > 0) else 1.2 if (ml_direction * ai_direction < 0) else 1.1
            
            adjusted_risk_score = risk_score * agreement_factor
            
            return {
                "ai_risk_assessment": risk_response.content,
                "risk_score": adjusted_risk_score,
                "original_risk_score": risk_score,
                "risk_factors": risk_response.metadata.get("risk_factors", []),
                "volatility": volatility,
                "liquidity_risk": self._assess_liquidity_risk(context.market_data),
                "correlation_risk": await self._assess_correlation_risk(context.symbol),
                "ml_confidence": ml_confidence,
                "ml_agreement": agreement_factor
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {"error": str(e), "risk_score": 10.0}
    
    def _assess_liquidity_risk(self, market_data: Dict[str, Any]) -> float:
        """Assess liquidity risk based on market depth"""
        try:
            depth = market_data.get("market_depth", {})
            bid_size = depth.get("total_bid_size", 0)
            ask_size = depth.get("total_ask_size", 0)
            
            if bid_size == 0 or ask_size == 0:
                return 1.0  # High liquidity risk
            
            # Lower total size = higher liquidity risk
            total_size = bid_size + ask_size
            liquidity_score = min(total_size / 10000, 1.0)  # Normalize
            
            return 1.0 - liquidity_score  # Convert to risk (higher = more risk)
            
        except Exception:
            return 0.5  # Medium liquidity risk as default
    
    async def _assess_correlation_risk(self, symbol: str) -> float:
        """Assess correlation risk with existing positions"""
        try:
            positions = await self.fyers_client.get_positions()
            
            if not positions:
                return 0.0  # No correlation risk if no positions
            
            # Simplified correlation assessment
            # In production, this would use actual correlation calculations
            sector_symbols = [pos.get("tradingSymbol", "") for pos in positions]
            
            # Check if symbol is in same sector (simplified)
            if any(symbol.split(":")[1].split("-")[0][:3] == 
                   existing.split(":")[1].split("-")[0][:3] for existing in sector_symbols):
                return 0.7  # High correlation risk
            
            return 0.3  # Low correlation risk
            
        except Exception:
            return 0.5  # Medium correlation risk as default
    
    def _calculate_position_size(self, risk_assessment: Dict[str, Any], context: "TradingContext", ensemble_predictions: Dict[str, Any]) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        try:
            # Get risk metrics
            risk_score = risk_assessment.get("risk_score", 5.0)
            volatility = risk_assessment.get("volatility", 0.02)
            
            # Base position size on risk tolerance
            max_risk_amount = context.portfolio_context.get("total_capital", 100000) * self.risk_tolerance
            
            # Adjust for volatility
            volatility_adjustment = max(0.1, 1.0 - volatility * 5)
            
            # Adjust for risk score (1-10 scale)
            risk_adjustment = max(0.1, (11 - risk_score) / 10)
            
            # ENHANCED POSITION SIZING: Incorporate ML predictions
            ml_confidence = ensemble_predictions.get("ensemble_confidence", 0.5)
            ml_direction = ensemble_predictions.get("ensemble_direction", 0)
            
            # Boost position size for high ML confidence
            ml_adjustment = 1.0 + (ml_confidence - 0.5) * 0.5 if ml_direction != 0 else 1.0
            
            # Calculate position size as percentage of portfolio
            base_size = 0.1  # 10% base allocation
            adjusted_size = base_size * volatility_adjustment * risk_adjustment * ml_adjustment
            
            # Ensure within limits
            max_size = 0.25  # Maximum 25% per position
            min_size = 0.01  # Minimum 1% per position
            
            position_size = max(min_size, min(adjusted_size, max_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.05  # Conservative 5% default
    
    def _synthesize_trading_signal(self, symbol: str, context: "TradingContext",
                                 ai_decision: "LlamaResponse", risk_assessment: Dict[str, Any],
                                 position_size: float, ensemble_predictions: Dict[str, Any]) -> TradingSignal:
        """Synthesize final trading signal from all analysis"""
        try:
            # Extract decision from AI response
            recommendation = ai_decision.metadata.get("recommendation", "HOLD")
            decision = TradingDecision(recommendation) if recommendation in [d.value for d in TradingDecision] else TradingDecision.HOLD
            
            # Calculate target and stop loss
            current_price = context.current_price
            volatility = risk_assessment.get("volatility", 0.02)
            
            if decision == TradingDecision.BUY:
                target_price = current_price * (1 + volatility * 3)  # 3x volatility target
                stop_loss = current_price * (1 - volatility * 1.5)   # 1.5x volatility stop
            elif decision == TradingDecision.SELL:
                target_price = current_price * (1 - volatility * 3)
                stop_loss = current_price * (1 + volatility * 1.5)
            else:
                target_price = None
                stop_loss = None
            
            # Calculate expected return
            if target_price and decision in [TradingDecision.BUY, TradingDecision.SELL]:
                expected_return = abs(target_price - current_price) / current_price
            else:
                expected_return = 0.0
            
            # ENHANCED SIGNAL SYNTHESIS: Incorporate ML predictions into confidence
            ai_confidence = ai_decision.confidence or 0.5
            ml_confidence = ensemble_predictions.get("ensemble_confidence", 0.5)
            ml_direction = ensemble_predictions.get("ensemble_direction", 0)
            
            # Combined confidence - weighted average
            combined_confidence = (ai_confidence * 0.7 + ml_confidence * 0.3) if ml_direction != 0 else ai_confidence
            
            return TradingSignal(
                symbol=symbol,
                decision=decision,
                confidence=combined_confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                reasoning=ai_decision.reasoning or "AI-generated decision",
                risk_score=risk_assessment.get("risk_score", 5.0),
                expected_return=expected_return,
                time_horizon="1-5 days",  # Default time horizon
                metadata={
                    "technical_signals": context.technical_signals,
                    "market_data": context.market_data,
                    "risk_assessment": risk_assessment,
                    "ai_confidence": ai_confidence,
                    "ml_predictions": ensemble_predictions,
                    "combined_confidence": combined_confidence,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Signal synthesis error: {e}")
            return TradingSignal(
                symbol=symbol,
                decision=TradingDecision.WAIT,
                confidence=0.0,
                entry_price=context.current_price,
                target_price=None,
                stop_loss=None,
                position_size=0.0,
                reasoning=f"Error in signal synthesis: {str(e)}",
                risk_score=10.0,
                expected_return=0.0,
                time_horizon="N/A",
                metadata={"error": str(e)}
            )
    
    async def _update_agent_memory(self, signal: TradingSignal, context: "TradingContext"):
        """Update agent memory for continuous learning"""
        try:
            # Store decision pattern
            decision_pattern = {
                "symbol": signal.symbol,
                "decision": signal.decision.value,
                "confidence": signal.confidence,
                "technical_signals": context.technical_signals,
                "market_conditions": context.market_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to memory (implement circular buffer for efficiency)
            if not hasattr(self.memory, 'decision_patterns'):
                self.memory.decision_patterns = []
            
            self.memory.decision_patterns.append(decision_pattern)
            
            # Keep only last 1000 patterns
            if len(self.memory.decision_patterns) > 1000:
                self.memory.decision_patterns = self.memory.decision_patterns[-1000:]
            
            self.memory.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Memory update error: {e}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and performance"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "memory_size": len(getattr(self.memory, 'decision_patterns', [])),
            "last_updated": self.memory.last_updated.isoformat(),
            "risk_tolerance": self.risk_tolerance,
            "max_positions": self.max_positions,
            "ml_weight_multiplier": self.ml_weight_multiplier,
            "ml_confidence_threshold": self.ml_confidence_threshold
        }