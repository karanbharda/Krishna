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

# Meta-Learning Model for Fast Adaptation
class MetaLearningModel(nn.Module):
    """Meta-learning model for fast adaptation to new market conditions"""
    
    def __init__(self, input_size=25, hidden_size=128, output_size=3):
        super(MetaLearningModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Base model for fast adaptation
        self.base_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Meta-learner for parameter adaptation
        self.meta_learner = nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)  # Output adaptation parameters
        )
        
        # Enhanced meta-learning with attention mechanism
        self.meta_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.meta_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, adaptation_context=None):
        # Base prediction
        base_output = self.base_model(x)
        
        # If adaptation context provided, apply meta-learning
        if adaptation_context is not None:
            # Combine input with adaptation context
            meta_input = torch.cat([x, adaptation_context], dim=-1)
            adaptation_params = self.meta_learner(meta_input)
            
            # Apply attention-based meta-learning
            query = adaptation_params.unsqueeze(1)
            key = adaptation_params.unsqueeze(1)
            value = adaptation_params.unsqueeze(1)
            
            attended_params, _ = self.meta_attention(query, key, value)
            attended_params = attended_params.squeeze(1)
            attended_params = self.meta_norm(attended_params)
            
            # Apply adaptation (simplified version)
            # In practice, this would modify the base model parameters
            adapted_output = base_output + attended_params[:, :base_output.size(-1)]
            return adapted_output
        
        return base_output

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
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
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
        """Get current observation state with enhanced features"""
        if self.current_step >= len(self.historical_data):
            return np.zeros(25, dtype=np.float32)
        
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
            row.get('atr', 0) / row['close'],  # Normalized ATR
            row.get('volatility', 0.02) / 0.02,  # Volatility regime
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
            row.get('sector_performance', 0),
            row.get('volume_profile', 0.5),
        ]
        
        # Combine all features
        observation = np.array(
            price_features + technical_features + portfolio_features + market_features + [0] * 4,  # Pad to 25
            dtype=np.float32
        )[:25]  # Ensure exactly 25 features
        
        return observation

# PPO Agent Implementation
class PPOAgent:
    """Proximal Policy Optimization agent for trading decisions"""
    
    def __init__(self, state_size: int = 25, action_size: int = 3, learning_rate: float = 3e-4):
        if not RL_AVAILABLE:
            raise ImportError("RL dependencies not available")
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = 0.99  # Discount factor
        self.eps_clip = 0.2  # Clipping parameter
        self.K_epochs = 4  # Number of epochs
        self.update_timestep = 2000  # Update policy every n timesteps
        
        # Neural networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': learning_rate},
            {'params': self.critic.parameters(), 'lr': learning_rate}
        ])
        
        # Memory for experience replay
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'returns': [],
            'advantages': []
        }
        
        self.timestep = 0
        
    def _build_actor(self) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic(self) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def act(self, state) -> Tuple[int, float]:
        """Choose action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        action_probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def remember(self, state, action, log_prob, reward, done, value):
        """Store experience in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['values'].append(value)
        
        self.timestep += 1
    
    def compute_returns_and_advantages(self):
        """Compute returns and advantages for PPO update"""
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        # Compute advantages
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values
        
        self.memory['returns'] = returns
        self.memory['advantages'] = advantages
    
    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.memory['states']) < 32:
            return
        
        # Compute returns and advantages
        self.compute_returns_and_advantages()
        
        # Convert to tensors
        states = torch.FloatTensor(self.memory['states'])
        actions = torch.LongTensor(self.memory['actions'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])
        returns = self.memory['returns']
        advantages = self.memory['advantages']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            values = self.critic(states).squeeze()
            
            # Ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'returns': [],
            'advantages': []
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class MultiAgentSystem:
    """Multi-agent system with specialized agents for different market conditions"""
    
    def __init__(self, state_size: int = 25, action_size: int = 3):
        self.agents = {
            'trending': PPOAgent(state_size, action_size),
            'volatile': PPOAgent(state_size, action_size),
            'sideways': PPOAgent(state_size, action_size),
            'general': PPOAgent(state_size, action_size),
            'bullish': PPOAgent(state_size, action_size),
            'bearish': PPOAgent(state_size, action_size)
        }
        self.current_agent = 'general'
        self.market_condition_detector = self._build_condition_detector(state_size)
        
    def _build_condition_detector(self, state_size: int) -> nn.Module:
        """Build neural network for market condition detection"""
        return nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.agents)),
            nn.Softmax(dim=-1)
        )
    
    def detect_market_condition(self, state: np.ndarray) -> str:
        """Detect current market condition using neural network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            condition_probs = self.market_condition_detector(state_tensor)
            condition_idx = torch.argmax(condition_probs, dim=-1).item()
            conditions = list(self.agents.keys())
            return conditions[condition_idx]
    
    def select_agent(self, market_condition: str) -> str:
        """Select appropriate agent based on market condition"""
        if market_condition in self.agents:
            self.current_agent = market_condition
        else:
            self.current_agent = 'general'
        return self.current_agent
    
    def act(self, state, market_condition: str = 'general'):
        """Choose action using selected agent"""
        # Auto-detect market condition if not provided
        if market_condition == 'general':
            market_condition = self.detect_market_condition(state)
        
        agent_key = self.select_agent(market_condition)
        return self.agents[agent_key].act(state)
    
    def remember(self, state, action, log_prob, reward, done, value, market_condition: str = 'general'):
        """Store experience in selected agent's memory"""
        # Auto-detect market condition if not provided
        if market_condition == 'general':
            market_condition = self.detect_market_condition(state)
            
        agent_key = self.select_agent(market_condition)
        self.agents[agent_key].remember(state, action, log_prob, reward, done, value)
    
    def update(self):
        """Update all agents"""
        for agent in self.agents.values():
            agent.update()
    
    def save_models(self, base_path: str):
        """Save all agent models"""
        for name, agent in self.agents.items():
            agent.save_model(f"{base_path}_{name}.pth")
    
    def load_models(self, base_path: str):
        """Load all agent models"""
        for name, agent in self.agents.items():
            try:
                agent.load_model(f"{base_path}_{name}.pth")
            except FileNotFoundError:
                logger.warning(f"Model file not found for {name} agent")

class HierarchicalRLAgent:
    """Hierarchical RL agent for complex trading decisions"""
    
    def __init__(self, state_size: int = 25, action_size: int = 3):
        # High-level policy for strategic decisions
        self.high_level_agent = PPOAgent(state_size, 6)  # 6 strategies: conservative, moderate, aggressive, momentum, mean_reversion, breakout
        
        # Low-level policies for tactical execution
        self.low_level_agents = {
            'conservative': PPOAgent(state_size, action_size),
            'moderate': PPOAgent(state_size, action_size),
            'aggressive': PPOAgent(state_size, action_size),
            'momentum': PPOAgent(state_size, action_size),
            'mean_reversion': PPOAgent(state_size, action_size),
            'breakout': PPOAgent(state_size, action_size)
        }
        
        # Strategy selector network
        self.strategy_selector = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1)
        )
        
    def act(self, state, high_level_action: Optional[int] = None):
        """Hierarchical action selection"""
        if high_level_action is None:
            # Select high-level strategy using neural network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                strategy_probs = self.strategy_selector(state_tensor)
                high_level_action = torch.multinomial(strategy_probs, 1).item()
        
        # Select low-level action based on strategy
        strategy_map = {0: 'conservative', 1: 'moderate', 2: 'aggressive', 
                       3: 'momentum', 4: 'mean_reversion', 5: 'breakout'}
        strategy = strategy_map.get(high_level_action, 'moderate')
        return self.low_level_agents[strategy].act(state)
    
    def remember(self, state, action, log_prob, reward, done, value, high_level_action: Optional[int] = None):
        """Store experience for both levels"""
        if high_level_action is None:
            # Select high-level strategy using neural network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                strategy_probs = self.strategy_selector(state_tensor)
                high_level_action = torch.multinomial(strategy_probs, 1).item()
        
        # Store experience for high-level agent
        self.high_level_agent.remember(state, high_level_action, log_prob, reward, done, value)
        
        # Store experience for low-level agent
        strategy_map = {0: 'conservative', 1: 'moderate', 2: 'aggressive',
                       3: 'momentum', 4: 'mean_reversion', 5: 'breakout'}
        strategy = strategy_map.get(high_level_action, 'moderate')
        self.low_level_agents[strategy].remember(state, action, log_prob, reward, done, value)
    
    def update(self):
        """Update both high-level and low-level agents"""
        self.high_level_agent.update()
        for agent in self.low_level_agents.values():
            agent.update()

class ContinuousLearningEngine:
    """Production-level continuous learning system"""

    def __init__(self, storage_path: str = None):
        # FIXED: Use project root data directory
        if storage_path is None:
            backend_dir = Path(__file__).resolve().parents[1]
            project_root = backend_dir.parent
            storage_path = str(project_root / 'data' / 'learning')
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Learning components
        self.rl_agent = None
        self.multi_agent_system = None
        self.hierarchical_agent = None
        self.meta_learning_model = None  # NEW: Meta-learning model
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

        # ENHANCED REWARD SHAPING: Multi-objective reward function
        self.reward_components = {
            'pnl_reward': 0.4,          # Profit/loss component
            'risk_adjusted_reward': 0.3, # Risk-adjusted returns
            'consistency_reward': 0.2,   # Consistency of performance
            'signal_quality_reward': 0.1 # Quality of signals used
        }
        
        # Multi-objective optimization parameters
        self.objective_weights = {
            'profitability': 0.4,
            'risk_adjusted_return': 0.3,
            'drawdown_control': 0.2,
            'consistency': 0.1
        }
        
        # Advanced learning parameters
        self.reward_shaping_enabled = True
        self.risk_penalty_multiplier = 2.0
        self.consistency_bonus_multiplier = 1.5
        self.signal_quality_weight = 0.3

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
                'volume': np.random.randint(10000, 100000, 1000),
                'rsi': np.random.randint(30, 70, 1000),
                'macd': np.random.randn(1000) * 0.5,
                'sma_ratio': np.random.randn(1000) * 0.05 + 1,
                'atr': np.random.randn(1000) * 2 + 5,
                'volatility': np.random.randn(1000) * 0.01 + 0.02
            })

            self.trading_env = TradingEnvironment(dummy_data)
            self.rl_agent = PPOAgent()
            self.multi_agent_system = MultiAgentSystem()
            self.hierarchical_agent = HierarchicalRLAgent()
            self.meta_learning_model = MetaLearningModel()  # NEW: Initialize meta-learning model

            # Try to load existing models
            model_path = self.storage_path / "rl_model.pth"
            if model_path.exists():
                self.rl_agent.load_model(str(model_path))
                logger.info("Loaded existing RL model")
            
            multi_agent_path = self.storage_path / "multi_agent"
            if multi_agent_path.exists():
                self.multi_agent_system.load_models(str(multi_agent_path))
                logger.info("Loaded existing multi-agent models")
                
        except Exception as e:
            logger.error(f"Error initializing RL components: {e}")
            self.rl_agent = None
            self.multi_agent_system = None
            self.hierarchical_agent = None
            self.meta_learning_model = None  # NEW: Set to None on error
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

    async def _rl_learning_step(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]):
        """Perform ENHANCED RL learning step with advanced reward shaping"""
        
        if not self.rl_agent or not RL_AVAILABLE:
            return
        
        try:
            # Convert decision to RL format
            state = self._decision_to_state(decision_data)
            action = self._action_to_int(decision_data.get('action', 'HOLD'))
            
            # ENHANCED REWARD CALCULATION
            reward = self._calculate_enhanced_reward(decision_data, outcome_data)
            
            next_state = state  # Simplified - in practice, this would be the next market state
            done = True  # Each decision is treated as a complete episode
            
            # Get log probability and value for PPO
            _, log_prob, value = self.rl_agent.act(state)
            
            # Store experience with enhanced reward
            self.rl_agent.remember(state, action, log_prob, reward, done, value)
            
            # Update multi-agent system if available
            if self.multi_agent_system:
                market_condition = decision_data.get('market_context', {}).get('volatility_regime', 'general')
                _, log_prob_ma, value_ma = self.multi_agent_system.act(state, market_condition)
                self.multi_agent_system.remember(state, action, log_prob_ma, reward, done, value_ma, market_condition)
            
            # Update hierarchical agent if available
            if self.hierarchical_agent:
                _, log_prob_ha, value_ha = self.hierarchical_agent.act(state)
                self.hierarchical_agent.remember(state, action, log_prob_ha, reward, done, value_ha)
            
            # Train if enough experiences
            if len(self.rl_agent.memory['states']) >= self.learning_config['min_experiences']:
                self.rl_agent.update()
                if self.multi_agent_system:
                    self.multi_agent_system.update()
                if self.hierarchical_agent:
                    self.hierarchical_agent.update()
            
            # Update learning metrics
            self.learning_metrics.total_episodes += 1
            
        except Exception as e:
            logger.error(f"Error in enhanced RL learning step: {e}")

    def _calculate_enhanced_reward(self, decision_data: Dict[str, Any], outcome_data: Dict[str, Any]) -> float:
        """Calculate sophisticated multi-objective reward"""
        
        base_reward = outcome_data.get('profit_loss_pct', 0.0) * 100  # Scale reward
        
        if not self.reward_shaping_enabled:
            return base_reward
        
        # Component 1: P&L Reward (40% weight)
        pnl_reward = base_reward * self.reward_components['pnl_reward']
        
        # Component 2: Risk-Adjusted Reward (30% weight)
        risk_score = decision_data.get('risk_assessment', {}).get('composite_risk_score', 0.5)
        risk_adjusted_reward = pnl_reward * (1 - risk_score * self.risk_penalty_multiplier)
        risk_adjusted_reward *= self.reward_components['risk_adjusted_reward']
        
        # Component 3: Consistency Reward (20% weight)
        # Reward for consistent performance over time
        recent_performance = self._calculate_recent_consistency()
        consistency_reward = recent_performance * self.consistency_bonus_multiplier
        consistency_reward *= self.reward_components['consistency_reward']
        
        # Component 4: Signal Quality Reward (10% weight)
        signal_quality = self._calculate_signal_quality(decision_data)
        signal_quality_reward = signal_quality * self.signal_quality_weight
        signal_quality_reward *= self.reward_components['signal_quality_reward']
        
        # Combine all components
        total_reward = pnl_reward + risk_adjusted_reward + consistency_reward + signal_quality_reward
        
        # Apply bounds to prevent extreme rewards
        total_reward = max(min(total_reward, 100.0), -100.0)
        
        logger.debug(f"Enhanced reward: {total_reward:.2f} "
                    f"(PNL: {pnl_reward:.2f}, Risk: {risk_adjusted_reward:.2f}, "
                    f"Consistency: {consistency_reward:.2f}, Quality: {signal_quality_reward:.2f})")
        
        return total_reward
    
    def _adjust_objective_weights(self, market_conditions: Dict[str, Any]):
        """Dynamically adjust objective weights based on market conditions"""
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        volume_profile = market_conditions.get('volume_profile', 0.5)
        
        # In high volatility markets, prioritize risk control and consistency
        if volatility > 0.04:
            self.objective_weights['profitability'] = 0.2
            self.objective_weights['risk_adjusted_return'] = 0.4
            self.objective_weights['drawdown_control'] = 0.3
            self.objective_weights['consistency'] = 0.1
        # In trending markets, prioritize profitability
        elif abs(trend_strength) > 0.3:
            self.objective_weights['profitability'] = 0.5
            self.objective_weights['risk_adjusted_return'] = 0.3
            self.objective_weights['drawdown_control'] = 0.1
            self.objective_weights['consistency'] = 0.1
        # In normal markets, balanced approach
        else:
            self.objective_weights['profitability'] = 0.4
            self.objective_weights['risk_adjusted_return'] = 0.3
            self.objective_weights['drawdown_control'] = 0.2
            self.objective_weights['consistency'] = 0.1
            
        logger.debug(f"Adjusted objective weights: {self.objective_weights}")
    
    def _calculate_recent_consistency(self) -> float:
        """Calculate recent performance consistency"""
        # Look at last 10 trades for consistency
        recent_trades = getattr(self.performance_tracker, 'performance_history', [])[-10:]
        
        if len(recent_trades) < 3:
            return 0.0
        
        # Calculate win rate consistency
        recent_pnls = [trade.get('outcome', {}).get('profit_loss', 0) for trade in recent_trades]
        winning_trades = sum(1 for pnl in recent_pnls if pnl > 0)
        win_rate = winning_trades / len(recent_pnls)
        
        # Calculate volatility of returns
        returns_std = np.std(recent_pnls) if recent_pnls else 1.0
        volatility_penalty = 1 / (1 + returns_std)
        
        consistency_score = win_rate * volatility_penalty
        return consistency_score
    
    def _calculate_signal_quality(self, decision_data: Dict[str, Any]) -> float:
        """Calculate quality of signals used in decision"""
        
        signal_consensus = decision_data.get('signal_consensus', 0.0)
        confidence = decision_data.get('confidence', 0.0)
        
        # Factor in signal diversity and strength
        signals_triggered = decision_data.get('signals_triggered', [])
        signal_count = len(signals_triggered)
        avg_signal_strength = np.mean([s.get('strength', 0) for s in signals_triggered]) if signals_triggered else 0.0
        signal_diversity = len(set(s.get('category', '') for s in signals_triggered))
        
        # Quality score based on multiple factors
        quality_score = (signal_consensus * 0.4 + confidence * 0.3 + 
                        (signal_count/10) * 0.2 + (signal_diversity/5) * 0.1)
        
        return quality_score

    def _decision_to_state(self, decision_data: Dict[str, Any]) -> np.ndarray:
        """ENHANCED state representation with more features"""
        
        # Extract enhanced features
        signal_consensus = decision_data.get('signal_consensus', 0.0)
        confidence = decision_data.get('confidence', 0.0)
        risk_score = decision_data.get('risk_assessment', {}).get('composite_risk_score', 0.5)
        market_context = decision_data.get('market_context', {})
        
        # Add more sophisticated features
        signals_triggered = decision_data.get('signals_triggered', [])
        signal_count = len(signals_triggered)
        avg_signal_strength = np.mean([s.get('strength', 0) for s in signals_triggered]) if signals_triggered else 0.0
        signal_diversity = len(set(s.get('category', '') for s in signals_triggered))
        
        # Time-based features
        current_time = datetime.now()
        market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        time_factor = (current_time - market_open).total_seconds() / (6.5 * 3600)  # 6.5 hour trading day
        
        # Technical indicators from decision data
        technical_data = decision_data.get('technical_analysis', {})
        rsi = technical_data.get('rsi', 50) / 100  # Normalize RSI
        macd = technical_data.get('macd', 0)
        sma_ratio = technical_data.get('sma_ratio', 1.0)
        atr = technical_data.get('atr', 0)
        volatility = technical_data.get('volatility', 0.02)
        
        # Create enhanced state vector (25 features)
        state = np.array([
            signal_consensus,                           # 1. Signal consensus
            confidence,                                 # 2. Confidence level
            risk_score,                                 # 3. Risk score
            market_context.get('volatility', 0.02),     # 4. Market volatility
            market_context.get('trend_strength', 0.0),  # 5. Trend strength
            market_context.get('stress_level', 0.3),    # 6. Market stress
            signal_count / 10.0,                        # 7. Signal count (normalized)
            avg_signal_strength,                        # 8. Average signal strength
            signal_diversity / 5.0,                     # 9. Signal diversity (normalized)
            time_factor,                                # 10. Time factor
            market_context.get('volume_profile', 0.5),  # 11. Volume profile
            market_context.get('sector_performance', 0.0), # 12. Sector performance
            rsi,                                        # 13. RSI (normalized)
            macd,                                       # 14. MACD
            sma_ratio,                                  # 15. SMA ratio
            atr,                                        # 16. ATR
            volatility,                                 # 17. Volatility
            market_context.get('market_trend', 0),      # 18. Market trend
            market_context.get('support_level', 0),     # 19. Support level
            market_context.get('resistance_level', 0),  # 20. Resistance level
            market_context.get('liquidity_score', 0.5), # 21. Liquidity score
            market_context.get('momentum', 0),          # 22. Momentum
            market_context.get('volume_ratio', 1.0),    # 23. Volume ratio
            0.0,                                        # 24. Reserved
            0.0                                         # 25. Reserved
        ], dtype=np.float32)
        
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

    async def _periodic_learning_update(self):
        """Perform periodic learning updates"""

        try:
            # Save RL model
            if self.rl_agent and self.learning_metrics.total_episodes % self.learning_config['model_save_frequency'] == 0:
                model_path = self.storage_path / f"rl_model_v{self.learning_metrics.model_version}.pth"
                self.rl_agent.save_model(str(model_path))
                
                # Save multi-agent models if available
                if self.multi_agent_system:
                    multi_agent_path = self.storage_path / f"multi_agent_v{self.learning_metrics.model_version}"
                    self.multi_agent_system.save_models(str(multi_agent_path))
                
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
            'rl_available': RL_AVAILABLE and self.rl_agent is not None,
            'multi_agent_available': self.multi_agent_system is not None,
            'hierarchical_agent_available': self.hierarchical_agent is not None,
            'meta_learning_available': self.meta_learning_model is not None  # NEW: Meta-learning availability
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