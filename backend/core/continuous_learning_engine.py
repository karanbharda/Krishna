"""
Production-Level Continuous Learning Engine with RL
Self-improving trading system using reinforcement learning
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import json
from pathlib import Path
import pickle

# RL imports
try:
    import gymnasium as gym
    from gymnasium import spaces
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """Learning performance metrics"""
    total_episodes: int
    avg_reward: float
    win_rate: float
    learning_rate: float
    exploration_rate: float
    model_version: int
    last_update: datetime

@dataclass
class PatternInsight:
    """Discovered trading pattern"""
    pattern_id: str
    pattern_type: str  # 'signal', 'market', 'risk'
    conditions: Dict[str, Any]
    success_rate: float
    avg_return: float
    confidence: float
    sample_size: int
    discovered_at: datetime

class TradingEnvironment(gym.Env):
    """RL Environment for trading decisions"""
    
    def __init__(self, historical_data: pd.DataFrame, initial_balance: float = 100000):
        super().__init__()
        
        if not RL_AVAILABLE:
            raise ImportError("RL dependencies not available")
        
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.max_steps = len(historical_data) - 1
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price_features, technical_indicators, portfolio_state]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_reward = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, True, {}
        
        # Get current and next prices
        current_price = self.historical_data.iloc[self.current_step]['close']
        next_price = self.historical_data.iloc[self.current_step + 1]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price, next_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _execute_action(self, action: int, current_price: float, next_price: float) -> float:
        """Execute trading action and calculate reward"""
        reward = 0
        
        if action == 1:  # Buy
            if self.position == 0 and self.balance >= current_price:
                shares_to_buy = int(self.balance * 0.95 / current_price)  # Use 95% of balance
                if shares_to_buy > 0:
                    self.position = shares_to_buy
                    self.balance -= shares_to_buy * current_price
                    # Reward based on next price movement
                    reward = (next_price - current_price) / current_price * shares_to_buy
        
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance += self.position * current_price
                # Reward based on profit from position
                reward = (current_price - self.historical_data.iloc[self.current_step - 1]['close']) / self.historical_data.iloc[self.current_step - 1]['close'] * self.position
                self.position = 0
        
        # Action == 0 (Hold) gets no immediate reward
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= len(self.historical_data):
            return np.zeros(20, dtype=np.float32)
        
        row = self.historical_data.iloc[self.current_step]
        
        # Price features (normalized)
        price_features = [
            row['close'] / row['open'] - 1,  # Daily return
            row['high'] / row['close'] - 1,  # High vs close
            row['low'] / row['close'] - 1,   # Low vs close
            row['volume'] / row.get('avg_volume', row['volume']),  # Volume ratio
        ]
        
        # Technical indicators (if available)
        technical_features = [
            row.get('rsi', 50) / 100 - 0.5,  # RSI normalized
            row.get('macd', 0),
            row.get('bb_position', 0.5) - 0.5,  # Bollinger band position
            row.get('sma_ratio', 1) - 1,  # Price vs SMA ratio
        ]
        
        # Portfolio state
        portfolio_features = [
            self.balance / self.initial_balance - 1,  # Balance change
            self.position / 1000,  # Position size (normalized)
            (self.balance + self.position * row['close']) / self.initial_balance - 1,  # Total value change
        ]
        
        # Market context (if available)
        market_features = [
            row.get('market_volatility', 0.02),
            row.get('market_trend', 0),
            row.get('market_stress', 0.3),
        ]
        
        # Combine all features
        observation = np.array(
            price_features + technical_features + portfolio_features + market_features + [0] * 6,  # Pad to 20
            dtype=np.float32
        )[:20]  # Ensure exactly 20 features
        
        return observation

class DQNAgent:
    """Deep Q-Network agent for trading decisions"""
    
    def __init__(self, state_size: int = 20, action_size: int = 3, learning_rate: float = 0.001):
        if not RL_AVAILABLE:
            raise ImportError("RL dependencies not available")
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.memory_size = 10000
        
        # Neural network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def _build_network(self) -> nn.Module:
        """Build neural network for Q-learning"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def act(self, state) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch])
        actions = torch.LongTensor([self.memory[i][1] for i in batch])
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch])
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch])
        dones = torch.BoolTensor([self.memory[i][4] for i in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class ContinuousLearningEngine:
    """Production-level continuous learning system"""

    def __init__(self, storage_path: str = "data/learning"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Learning components
        self.rl_agent = None
        self.trading_env = None
        self.pattern_analyzer = PatternAnalyzer()
        self.performance_tracker = PerformanceTracker()

        # Learning state
        self.learning_metrics = LearningMetrics(
            total_episodes=0,
            avg_reward=0.0,
            win_rate=0.0,
            learning_rate=0.001,
            exploration_rate=1.0,
            model_version=1,
            last_update=datetime.now()
        )

        # Pattern insights
        self.discovered_patterns = []
        self.signal_weights = {
            'technical': 0.40,
            'sentiment': 0.25,
            'ml_prediction': 0.25,
            'risk_metrics': 0.10
        }

        # Learning configuration
        self.learning_config = {
            'episodes_per_update': 100,
            'min_experiences': 1000,
            'target_update_frequency': 10,
            'model_save_frequency': 50,
            'pattern_analysis_frequency': 20
        }

        # Initialize RL components if available
        if RL_AVAILABLE:
            self._initialize_rl_components()
        else:
            logger.warning("RL components not available - using pattern-based learning only")

    def record_decision(self, decision_data: Dict[str, Any]) -> None:
        """Record a trading decision for later learning (compatibility method for web backend)"""
        try:
            # Store decision data for later outcome matching
            # In a real implementation, this would store the decision for later learning
            # when the outcome is known
            logger.debug(f"Recorded decision: {decision_data.get('symbol', 'unknown')} {decision_data.get('action', 'HOLD')}")
        except Exception as e:
            logger.error(f"Error recording decision: {e}")

    def _initialize_rl_components(self):
        """Initialize RL agent and environment"""
        try:
            # Create dummy historical data for initialization
            dummy_data = pd.DataFrame({
                'open': np.random.randn(1000) * 0.02 + 100,
                'high': np.random.randn(1000) * 0.02 + 102,
                'low': np.random.randn(1000) * 0.02 + 98,
                'close': np.random.randn(1000) * 0.02 + 100,
                'volume': np.random.randint(10000, 100000, 1000)
            })

            self.trading_env = TradingEnvironment(dummy_data)
            self.rl_agent = DQNAgent()

            # Try to load existing model
            model_path = self.storage_path / "rl_model.pth"
            if model_path.exists():
                self.rl_agent.load_model(str(model_path))
                logger.info("Loaded existing RL model")

        except Exception as e:
            logger.error(f"Error initializing RL components: {e}")
            self.rl_agent = None
            self.trading_env = None

    async def learn_from_decision_outcome(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]):
        """Learn from a trading decision outcome"""

        try:
            # Update performance tracking
            await self.performance_tracker.record_outcome(decision_data, outcome_data)

            # Pattern analysis
            await self._analyze_decision_patterns(decision_data, outcome_data)

            # RL learning (if available)
            if self.rl_agent and RL_AVAILABLE:
                await self._rl_learning_step(decision_data, outcome_data)

            # Update signal weights based on performance
            await self._update_signal_weights()

            # Periodic model updates
            if self.learning_metrics.total_episodes % self.learning_config['episodes_per_update'] == 0:
                await self._periodic_learning_update()

            logger.debug(f"Learning step completed for decision: {decision_data.get('decision_id', 'unknown')}")

        except Exception as e:
            logger.error(f"Error in learning from decision outcome: {e}")

    async def _analyze_decision_patterns(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]):
        """Analyze patterns in decision outcomes"""

        try:
            # Extract key features from decision
            features = {
                'signal_strength': decision_data.get('signal_consensus', 0.0),
                'confidence': decision_data.get('confidence', 0.0),
                'risk_score': decision_data.get('risk_assessment', {}).get('composite_risk_score', 0.5),
                'market_volatility': decision_data.get('market_context', {}).get('volatility', 0.02),
                'action': decision_data.get('action', 'HOLD'),
                'symbol': decision_data.get('symbol', ''),
                'time_of_day': decision_data.get('market_context', {}).get('time_of_day', 'mid_day')
            }

            # Outcome metrics
            profit_loss = outcome_data.get('profit_loss_pct', 0.0)
            is_profitable = profit_loss > 0.01  # > 1% profit

            # Pattern discovery
            await self._discover_patterns(features, is_profitable, profit_loss)

        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {e}")

    async def _discover_patterns(self, features: Dict[str, Any], is_profitable: bool, profit_loss: float):
        """Discover profitable trading patterns"""

        # Pattern: High confidence + Low risk = Good outcomes
        if features['confidence'] > 0.8 and features['risk_score'] < 0.3:
            pattern_id = "high_confidence_low_risk"
            await self._update_pattern_insight(
                pattern_id,
                "signal",
                {"confidence": ">0.8", "risk_score": "<0.3"},
                is_profitable,
                profit_loss
            )

        # Pattern: Strong signal + Normal volatility = Good outcomes
        if abs(features['signal_strength']) > 0.7 and features['market_volatility'] < 0.03:
            pattern_id = "strong_signal_normal_vol"
            await self._update_pattern_insight(
                pattern_id,
                "market",
                {"signal_strength": ">0.7", "market_volatility": "<0.03"},
                is_profitable,
                profit_loss
            )

        # Pattern: Avoid trading during high volatility
        if features['market_volatility'] > 0.05:
            pattern_id = "high_volatility_avoid"
            await self._update_pattern_insight(
                pattern_id,
                "risk",
                {"market_volatility": ">0.05"},
                is_profitable,
                profit_loss
            )

    async def _update_pattern_insight(self, pattern_id: str, pattern_type: str,
                                    conditions: Dict[str, Any], is_profitable: bool, profit_loss: float):
        """Update or create pattern insight"""

        # Find existing pattern
        existing_pattern = None
        for pattern in self.discovered_patterns:
            if pattern.pattern_id == pattern_id:
                existing_pattern = pattern
                break

        if existing_pattern:
            # Update existing pattern
            total_return = existing_pattern.avg_return * existing_pattern.sample_size + profit_loss
            new_sample_size = existing_pattern.sample_size + 1
            new_avg_return = total_return / new_sample_size

            if is_profitable:
                new_success_rate = (existing_pattern.success_rate * existing_pattern.sample_size + 1) / new_sample_size
            else:
                new_success_rate = (existing_pattern.success_rate * existing_pattern.sample_size) / new_sample_size

            existing_pattern.avg_return = new_avg_return
            existing_pattern.success_rate = new_success_rate
            existing_pattern.sample_size = new_sample_size
            existing_pattern.confidence = min(1.0, new_sample_size / 50)  # Confidence increases with sample size

        else:
            # Create new pattern
            new_pattern = PatternInsight(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                conditions=conditions,
                success_rate=1.0 if is_profitable else 0.0,
                avg_return=profit_loss,
                confidence=0.1,  # Low confidence initially
                sample_size=1,
                discovered_at=datetime.now()
            )
            self.discovered_patterns.append(new_pattern)

    async def _rl_learning_step(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]):
        """Perform RL learning step"""

        if not self.rl_agent or not RL_AVAILABLE:
            return

        try:
            # Convert decision to RL format
            state = self._decision_to_state(decision_data)
            action = self._action_to_int(decision_data.get('action', 'HOLD'))
            reward = outcome_data.get('profit_loss_pct', 0.0) * 100  # Scale reward
            next_state = state  # Simplified - in practice, this would be the next market state
            done = True  # Each decision is treated as a complete episode

            # Store experience
            self.rl_agent.remember(state, action, reward, next_state, done)

            # Train if enough experiences
            if len(self.rl_agent.memory) >= self.learning_config['min_experiences']:
                self.rl_agent.replay()

            # Update target network periodically
            if self.learning_metrics.total_episodes % self.learning_config['target_update_frequency'] == 0:
                self.rl_agent.update_target_network()

            # Update learning metrics
            self.learning_metrics.total_episodes += 1
            self.learning_metrics.exploration_rate = self.rl_agent.epsilon

        except Exception as e:
            logger.error(f"Error in RL learning step: {e}")

    def _decision_to_state(self, decision_data: Dict[str, Any]) -> np.ndarray:
        """Convert decision data to RL state representation"""

        # Extract relevant features
        signal_consensus = decision_data.get('signal_consensus', 0.0)
        confidence = decision_data.get('confidence', 0.0)
        risk_score = decision_data.get('risk_assessment', {}).get('composite_risk_score', 0.5)
        market_context = decision_data.get('market_context', {})

        # Create state vector (20 features to match environment)
        state = np.array([
            signal_consensus,
            confidence,
            risk_score,
            market_context.get('volatility', 0.02),
            market_context.get('trend_strength', 0.0),
            market_context.get('stress_level', 0.3),
            # Add more features as needed, pad to 20
        ] + [0.0] * 14, dtype=np.float32)[:20]

        return state

    def _action_to_int(self, action: str) -> int:
        """Convert action string to integer"""
        action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
        return action_map.get(action, 0)

    async def _update_signal_weights(self):
        """Update signal weights based on recent performance"""

        try:
            # Get recent performance by signal type
            signal_performance = await self.performance_tracker.get_signal_performance()

            if not signal_performance:
                return

            # Adjust weights based on performance
            total_adjustment = 0
            for signal_type, performance in signal_performance.items():
                if signal_type in self.signal_weights:
                    # Increase weight for better performing signals
                    if performance['avg_return'] > 0.02:  # > 2% average return
                        adjustment = 0.05
                    elif performance['avg_return'] > 0:
                        adjustment = 0.02
                    elif performance['avg_return'] < -0.02:  # < -2% average return
                        adjustment = -0.05
                    else:
                        adjustment = -0.02

                    self.signal_weights[signal_type] += adjustment
                    total_adjustment += adjustment

            # Normalize weights to sum to 1.0
            total_weight = sum(self.signal_weights.values())
            if total_weight > 0:
                for signal_type in self.signal_weights:
                    self.signal_weights[signal_type] /= total_weight

            logger.info(f"Updated signal weights: {self.signal_weights}")

        except Exception as e:
            logger.error(f"Error updating signal weights: {e}")

    async def _periodic_learning_update(self):
        """Perform periodic learning updates"""

        try:
            # Save RL model
            if self.rl_agent and self.learning_metrics.total_episodes % self.learning_config['model_save_frequency'] == 0:
                model_path = self.storage_path / f"rl_model_v{self.learning_metrics.model_version}.pth"
                self.rl_agent.save_model(str(model_path))
                self.learning_metrics.model_version += 1

            # Update learning metrics
            recent_performance = await self.performance_tracker.get_recent_performance(days=7)
            if recent_performance:
                self.learning_metrics.avg_reward = recent_performance.get('avg_return', 0.0)
                self.learning_metrics.win_rate = recent_performance.get('win_rate', 0.0)

            self.learning_metrics.last_update = datetime.now()

            # Save learning state
            await self._save_learning_state()

            logger.info(f"Periodic learning update completed - Episode {self.learning_metrics.total_episodes}")

        except Exception as e:
            logger.error(f"Error in periodic learning update: {e}")

    async def _save_learning_state(self):
        """Save learning state to disk"""

        try:
            learning_state = {
                'learning_metrics': {
                    'total_episodes': self.learning_metrics.total_episodes,
                    'avg_reward': self.learning_metrics.avg_reward,
                    'win_rate': self.learning_metrics.win_rate,
                    'learning_rate': self.learning_metrics.learning_rate,
                    'exploration_rate': self.learning_metrics.exploration_rate,
                    'model_version': self.learning_metrics.model_version,
                    'last_update': self.learning_metrics.last_update.isoformat()
                },
                'signal_weights': self.signal_weights,
                'discovered_patterns': [
                    {
                        'pattern_id': p.pattern_id,
                        'pattern_type': p.pattern_type,
                        'conditions': p.conditions,
                        'success_rate': p.success_rate,
                        'avg_return': p.avg_return,
                        'confidence': p.confidence,
                        'sample_size': p.sample_size,
                        'discovered_at': p.discovered_at.isoformat()
                    }
                    for p in self.discovered_patterns
                ]
            }

            state_file = self.storage_path / "learning_state.json"
            with open(state_file, 'w') as f:
                json.dump(learning_state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights and recommendations"""

        # Get top patterns
        top_patterns = sorted(
            [p for p in self.discovered_patterns if p.sample_size >= 5],
            key=lambda x: x.success_rate * x.confidence,
            reverse=True
        )[:10]

        # Get signal weight evolution
        signal_weight_insights = {
            'current_weights': self.signal_weights,
            'recommendations': []
        }

        # Generate recommendations based on patterns
        recommendations = []
        for pattern in top_patterns:
            if pattern.success_rate > 0.7 and pattern.confidence > 0.5:
                recommendations.append(f"Pattern '{pattern.pattern_id}' shows {pattern.success_rate:.1%} success rate - consider emphasizing these conditions")

        return {
            'learning_metrics': {
                'total_episodes': self.learning_metrics.total_episodes,
                'avg_reward': self.learning_metrics.avg_reward,
                'win_rate': self.learning_metrics.win_rate,
                'exploration_rate': self.learning_metrics.exploration_rate,
                'model_version': self.learning_metrics.model_version
            },
            'top_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'success_rate': p.success_rate,
                    'avg_return': p.avg_return,
                    'confidence': p.confidence,
                    'sample_size': p.sample_size
                }
                for p in top_patterns
            ],
            'signal_weights': self.signal_weights,
            'recommendations': recommendations,
            'rl_available': RL_AVAILABLE and self.rl_agent is not None
        }

class PatternAnalyzer:
    """Analyze trading patterns for insights"""

    def __init__(self):
        self.patterns = {}

    async def analyze_patterns(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in trading decisions"""
        # Implementation for pattern analysis
        return {}

class PerformanceTracker:
    """Track performance of different components"""

    def __init__(self):
        self.performance_history = []

    async def record_outcome(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]):
        """Record decision outcome"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'decision': decision_data,
            'outcome': outcome_data
        })

        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    async def get_signal_performance(self) -> Dict[str, Any]:
        """Get performance by signal type"""
        # Implementation for signal performance analysis
        return {}

    async def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance metrics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [r for r in self.performance_history if r['timestamp'] >= cutoff_date]

        if not recent_records:
            return {}

        total_return = sum(r['outcome'].get('profit_loss_pct', 0) for r in recent_records)
        profitable_trades = sum(1 for r in recent_records if r['outcome'].get('profit_loss', 0) > 0)

        return {
            'avg_return': total_return / len(recent_records),
            'win_rate': profitable_trades / len(recent_records),
            'total_trades': len(recent_records)
        }

    def add_experience(self, state, action, reward, next_state):
        """PRODUCTION FIX: Add trading experience for learning"""
        try:
            experience = {
                'timestamp': datetime.now(),
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            }

            # Add to experience buffer
            if not hasattr(self, 'experience_buffer'):
                self.experience_buffer = []

            self.experience_buffer.append(experience)

            # Keep buffer size manageable
            if len(self.experience_buffer) > 1000:
                self.experience_buffer = self.experience_buffer[-1000:]

            logger.debug(f"Added experience: action={action}, reward={reward}")

        except Exception as e:
            logger.error(f"Error adding experience: {e}")
