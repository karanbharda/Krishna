"""
Complete Stock Analysis Tool - All-in-One
Fetches and views financial data from Yahoo Finance
Integrated with 50+ Technical Indicators
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import io
import zipfile
from enum import Enum
from collections import deque, namedtuple
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas_ta as ta
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union, Tuple
from pathlib import Path
import logging
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import time
import sys
import os

# Force unbuffered output for immediate console display
sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(
    sys.stderr, 'reconfigure') else None


# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories - Updated to work with backend structure
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_CACHE_DIR = DATA_DIR / "cache"
FEATURE_CACHE_DIR = DATA_DIR / "features"
MODEL_DIR = BASE_DIR / "utils" / "ml_components"
LOGS_DIR = DATA_DIR / "logs"  # Unified with config.py - all logs in data/logs
NSE_BHAV_CACHE_DIR = DATA_CACHE_DIR / "nse_bhav"  # NSE Bhav Copy cache

# Create directories
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
NSE_BHAV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Historical data period
HISTORICAL_PERIOD = "2y"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prediction log file
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.json"


def get_symbol_cache_path(symbol: str) -> Path:
    """Get cache path for a symbol"""
    return DATA_CACHE_DIR / f"{symbol}_all_data.json"


# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# DQN COMPONENTS - REINFORCEMENT LEARNING
# ============================================================================

class Action(Enum):
    """Trading actions"""
    LONG = 0    # Buy/Long position
    SHORT = 1   # Sell/Short position
    HOLD = 2    # Hold position


# Experience tuple for replay buffer
Experience = namedtuple(
    'Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer"""
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        logger.info(f"ReplayBuffer initialized with capacity: {capacity}")

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        """Get current size of buffer"""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Deep Q-Network for estimating Q-values"""

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        """Initialize Q-Network"""
        super(QNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Build network layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(
            f"QNetwork initialized: {state_size} -> {hidden_sizes} -> {action_size}")

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(state)


class DQNTradingAgent:
    """DQN Trading Agent using Deep Reinforcement Learning"""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_sizes: List[int] = [256, 128, 64]
    ):
        """Initialize DQN Agent"""
        self.n_features = n_features
        self.n_actions = len(Action)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Q-Networks
        self.q_network = QNetwork(
            n_features, self.n_actions, hidden_sizes).to(self.device)
        self.target_network = QNetwork(
            n_features, self.n_actions, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training metrics
        self.total_episodes = 0
        self.total_steps = 0
        self.cumulative_reward = 0
        self.rewards_history = []

        logger.info(
            f"DQNTradingAgent initialized with {n_features} features, {self.n_actions} actions")

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        # Convert to tensor and move to device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            action = np.random.randint(self.n_actions)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()

        return action

    def update_epsilon(self):
        """Decay epsilon for less exploration over time"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(
            [e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor(
            [e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor(
            [e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(
            [e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)

        # Current Q values
        current_q_values = self.q_network(
            states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, symbol: str, horizon: str):
        """Save trained model"""
        model_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_features': self.n_features,
            'n_actions': self.n_actions,
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'cumulative_reward': self.cumulative_reward
        }, model_path)
        logger.info(f"DQN model saved to {model_path}")

    def load(self, symbol: str, horizon: str):
        """Load trained model"""
        model_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(
            checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Update agent properties if they exist in checkpoint
        if 'n_features' in checkpoint:
            self.n_features = checkpoint['n_features']
        if 'n_actions' in checkpoint:
            self.n_actions = checkpoint['n_actions']
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        if 'total_episodes' in checkpoint:
            self.total_episodes = checkpoint['total_episodes']
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        if 'cumulative_reward' in checkpoint:
            self.cumulative_reward = checkpoint['cumulative_reward']

        logger.info(f"DQN model loaded from {model_path}")

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics for logging"""
        if not self.rewards_history:
            return {}

        avg_reward = np.mean(self.rewards_history[-100:]) if len(
            self.rewards_history) >= 100 else np.mean(self.rewards_history)
        std_reward = np.std(self.rewards_history[-100:]) if len(
            self.rewards_history) >= 100 else np.std(self.rewards_history)

        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = avg_reward / std_reward if std_reward > 0 else 0

        # Calculate win rate (positive rewards)
        win_rate = sum(1 for r in self.rewards_history if r >
                       0) / len(self.rewards_history)

        return {
            'total_episodes': self.total_episodes,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': avg_reward,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer)
        }


# ============================================================================
# DATA INGESTION COMPONENTS
# ============================================================================

class EnhancedDataIngester:
    """
    Enhanced data ingester that fetches ALL available yfinance data
    Including: OHLCV, fundamentals, analyst data, ownership, earnings, options, etc.
    """

    def __init__(self, cache_dir: Path = DATA_CACHE_DIR):
        """Initialize Enhanced Data Ingester"""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] EnhancedDataIngester initialized with cache dir: {cache_dir}")

    def fetch_live_data(
        self,
        symbols: Union[str, List[str]],
        period: str = HISTORICAL_PERIOD,
        interval: str = "1d",
        retry_count: int = 3,
        backoff_factor: float = 2.0
    ) -> pd.DataFrame:
        """
        Fetch live stock data from Yahoo Finance with exponential backoff
        Falls back to NSE Bhav Copy for Indian stocks if yfinance fails

        Args:
            symbols: Stock symbol(s) to fetch
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with stock data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if len(symbols) > 1:
            raise ValueError("fetch_live_data only supports single symbol")

        symbol = symbols[0]

        for attempt in range(retry_count):
            try:
                print(
                    f"[INFO] Attempt {attempt + 1}/{retry_count} to fetch {symbol} from Yahoo Finance...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    raise ValueError("Empty DataFrame returned from yfinance")
                else:
                    # Clean data
                    df = self._clean_data(df)
                    print(
                        f"[INFO] Successfully fetched {len(df)} rows for {symbol} from yfinance")
                    return df

            except Exception as e:
                wait_time = backoff_factor ** attempt
                print(
                    f"[WARNING] Attempt {attempt + 1}/{retry_count} failed for {symbol}: {e}")

                if attempt < retry_count - 1:
                    print(f"[INFO] Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

        # Yfinance failed - try NSE Bhav Copy fallback for Indian stocks
        print(
            f"[INFO] yfinance failed for {symbol} after {retry_count} attempts")

        # Check if this is an Indian stock (.NS or .BO)
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            # Only use NSE fallback for daily data
            if interval == "1d":
                print(
                    f"[INFO] Attempting NSE Bhav Copy fallback for {symbol}...")

                try:
                    # Parse period to date range
                    end_date = datetime.now()

                    # Convert period string to days
                    if period.endswith('y'):
                        years = int(period[:-1])
                        start_date = end_date - timedelta(days=years * 365)
                    elif period.endswith('mo'):
                        months = int(period[:-2])
                        start_date = end_date - timedelta(days=months * 30)
                    elif period.endswith('d'):
                        days = int(period[:-1])
                        start_date = end_date - timedelta(days=days)
                    else:
                        # Default to 2 years
                        start_date = end_date - timedelta(days=730)

                    # Fetch from NSE Bhav Copy
                    df = self.fetch_nse_bhav_historical(
                        symbol, start_date, end_date)

                    if not df.empty:
                        # Clean data
                        df = self._clean_data(df)
                        print(
                            f"[OK] Successfully fetched {len(df)} rows from NSE Bhav Copy fallback")
                        return df
                    else:
                        print(
                            f"[ERROR] NSE Bhav Copy fallback also failed for {symbol}")

                except Exception as e:
                    print(f"[ERROR] NSE Bhav Copy fallback error: {e}")
                    logger.error(f"NSE fallback failed for {symbol}: {e}")
            else:
                print(
                    f"[INFO] NSE Bhav Copy only supports daily data (interval=1d), not {interval}")
                print(f"[INFO] Skipping NSE fallback")
        else:
            print(
                f"[INFO] {symbol} is not an Indian stock (.NS/.BO), NSE fallback not applicable")

        print(f"[ERROR] All data sources failed for {symbol}")
        return pd.DataFrame()

    def _download_nse_bhav_for_date(self, date: datetime, session: Optional[requests.Session] = None, retry_count: int = 2) -> Optional[pd.DataFrame]:
        """
        Download NSE Bhav Copy for a specific date
        Downloads CSV directly without caching the full file (saves space)

        Args:
            date: The date to download data for
            session: Optional requests.Session for connection reuse (faster)
            retry_count: Number of retries for 403 errors (with session refresh)

        Returns:
            DataFrame with NSE data for that date, or None if failed
        """
        # Format date as DDMMYYYY
        date_str = date.strftime("%d%m%Y")

        # NSE Bhav Copy URL
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"

        # Download from NSE (no full file caching - extract symbol-specific data only)
        for attempt in range(retry_count + 1):
            try:
                # Use session if provided (faster connection reuse), otherwise create new request
                if session:
                    response = session.get(url, timeout=15)
                else:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://www.nseindia.com/',
                    }
                    response = requests.get(url, headers=headers, timeout=15)

                response.raise_for_status()

                # Parse CSV from response directly (don't save full file)
                # Use low_memory=False for faster parsing when we know the data size
                df = pd.read_csv(io.StringIO(response.text), low_memory=False)

                logger.debug(f"Downloaded NSE Bhav Copy: {date_str}")

                return df

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # No data for this date (weekend/holiday)
                    logger.debug(
                        f"No NSE data for {date_str} (likely weekend/holiday)")
                    return None
                elif e.response.status_code == 403:
                    # 403 Forbidden - NSE is blocking the request
                    # Try refreshing session if we have retries left
                    if attempt < retry_count and session:
                        try:
                            session.get('https://www.nseindia.com', timeout=10)
                            time.sleep(0.3)  # Small delay
                            logger.debug(
                                f"Refreshed NSE session, retrying {date_str}")
                            continue  # Retry the request
                        except:
                            pass
                    # If no retries left or no session, give up
                    logger.debug(
                        f"NSE 403 Forbidden for {date_str} (after {attempt + 1} attempts)")
                    return None
                else:
                    logger.debug(
                        f"HTTP error downloading NSE data for {date_str}: {e.response.status_code}")
                    return None
            except Exception as e:
                if attempt < retry_count:
                    time.sleep(0.2)  # Small delay before retry
                    continue
                logger.debug(
                    f"Error downloading NSE Bhav Copy for {date_str}: {e}")
                return None

        return None

    def _parse_nse_bhav_csv(self, df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """
        Parse NSE Bhav Copy CSV and extract data for specific symbol

        Args:
            df: Raw NSE Bhav Copy DataFrame
            symbol: Stock symbol to extract (e.g., 'RELIANCE', 'TCS')

        Returns:
            DataFrame with OHLCV data for the symbol
        """
        if df is None or df.empty:
            return None

        try:
            # NSE Bhav Copy column names have specific format with spaces
            # Columns: SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
            #          LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, etc.

            # Normalize column names (strip spaces and convert to uppercase)
            df.columns = df.columns.str.strip().str.upper()

            # Also strip values in SYMBOL and SERIES columns
            df['SYMBOL'] = df['SYMBOL'].str.strip().str.upper()

            # Filter for the specific symbol and EQ (equity) series
            symbol_data = df[(df['SYMBOL'] == symbol.upper())
                             & (df['SERIES'] == 'EQ')]

            if symbol_data.empty:
                logger.debug(f"No data found for {symbol} in NSE Bhav Copy")
                return None

            # Extract OHLCV data and rename columns to match yfinance format
            result_df = pd.DataFrame({
                'Open': symbol_data['OPEN_PRICE'].astype(float),
                'High': symbol_data['HIGH_PRICE'].astype(float),
                'Low': symbol_data['LOW_PRICE'].astype(float),
                'Close': symbol_data['CLOSE_PRICE'].astype(float),
                'Volume': symbol_data['TTL_TRD_QNTY'].astype(int),
                # Add yfinance-compatible columns (NSE Bhav doesn't have these, set to 0)
                'Dividends': 0.0,
                'Stock Splits': 0.0,
            })

            # Parse date from DATE1 column (format: DD-MMM-YYYY like 07-Nov-2025)
            # Strip whitespace from date values before parsing
            if 'DATE1' in symbol_data.columns:
                dates = symbol_data['DATE1'].str.strip()
                result_df.index = pd.to_datetime(
                    dates, format='%d-%b-%Y', errors='coerce')
            elif 'TIMESTAMP' in symbol_data.columns:
                dates = symbol_data['TIMESTAMP'].str.strip()
                result_df.index = pd.to_datetime(
                    dates, format='%d-%b-%Y', errors='coerce')
            else:
                # Fallback: use current date
                result_df.index = pd.DatetimeIndex([datetime.now()])

            # Remove rows with invalid dates (NaT)
            result_df = result_df[result_df.index.notna()]

            return result_df

        except Exception as e:
            logger.error(f"Error parsing NSE Bhav CSV for {symbol}: {e}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return None

    def fetch_nse_bhav_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical data from NSE Bhav Copy for a date range
        Only fetches and caches data for the requested symbol (like yfinance)
        Creates ONE JSON file per symbol with all historical data

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS', 'TCS.NS')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data (yfinance format)
        """
        # Check if symbol-specific cache exists
        clean_symbol_for_file = symbol.replace(
            '.NS', '_NS').replace('.BO', '_BO')
        symbol_cache_file = NSE_BHAV_CACHE_DIR / \
            f"{clean_symbol_for_file}_historical.json"

        # Check cache first
        if symbol_cache_file.exists():
            try:
                with open(symbol_cache_file, 'r') as f:
                    cached_data = json.load(f)

                # Check if cache covers the requested date range
                cached_start = datetime.fromisoformat(
                    cached_data['start_date'])
                cached_end = datetime.fromisoformat(cached_data['end_date'])

                if cached_start <= start_date and cached_end >= end_date:
                    # Cache is valid, load it
                    df = pd.DataFrame(cached_data['data'])
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)

                    # Filter to requested date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]

                    print(
                        f"[INFO] Loaded {symbol} from NSE cache: {len(df)} rows")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load NSE cache for {symbol}: {e}")

        # Cache miss or invalid - fetch fresh data
        print(f"[INFO] Fetching NSE Bhav Copy historical data for {symbol}")
        print(f"  -> Date range: {start_date.date()} to {end_date.date()}")

        # Remove .NS or .BO suffix for NSE lookup
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()

        # Generate list of trading days (skip weekends) - collect all dates first
        trading_dates = []
        temp_date = start_date
        while temp_date <= end_date:
            if temp_date.weekday() < 5:  # Monday=0, Friday=4
                trading_dates.append(temp_date)
            temp_date += timedelta(days=1)

        dates_tried = len(trading_dates)
        print(
            f"  -> Will fetch {dates_tried} trading days (using parallel downloads)")

        # Download in parallel for faster fetching
        # Use ThreadPoolExecutor to download multiple days concurrently
        # Reduced to 5 workers to avoid NSE rate limiting (403 errors)
        # Reduced from 20 to 5 to avoid 403 Forbidden errors
        max_workers = min(5, dates_tried)

        # Create a session for connection reuse (faster than creating new connections)
        session = requests.Session()

        # NSE requires proper headers and cookies - visit homepage first to establish session
        try:
            # Visit NSE homepage first to get cookies (required for accessing archives)
            session.get('https://www.nseindia.com', timeout=10)
            time.sleep(0.5)  # Small delay after initial request
        except Exception as e:
            logger.debug(f"Could not establish NSE session: {e}")

        # Set proper headers that NSE expects
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
            'Upgrade-Insecure-Requests': '1',
        })

        # Increase connection pool size to avoid "pool is full" warnings
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=20)
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        def download_and_parse(date):
            """Download and parse data for a single date"""
            # Add small random delay to avoid rate limiting (0.1-0.3 seconds)
            time.sleep(0.1 + (hash(str(date)) % 20) / 100.0)

            bhav_df = self._download_nse_bhav_for_date(date, session=session)
            if bhav_df is not None:
                symbol_df = self._parse_nse_bhav_csv(bhav_df, clean_symbol)
                if symbol_df is not None and not symbol_df.empty:
                    return symbol_df
            return None

        # Use ThreadPoolExecutor for parallel downloads
        all_data = []
        dates_found = 0
        last_session_refresh = datetime.now()
        # Refresh session every 5 minutes (300 seconds)
        session_refresh_interval = 300

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_date = {executor.submit(
                    download_and_parse, date): date for date in trading_dates}

                # Process completed downloads with progress bar
                for future in tqdm(as_completed(future_to_date), total=len(trading_dates),
                                   desc="Downloading NSE data", unit="day"):
                    try:
                        result = future.result()
                        if result is not None:
                            all_data.append(result)
                            dates_found += 1

                        # Periodically refresh session to avoid 403 errors during long downloads
                        if (datetime.now() - last_session_refresh).total_seconds() > session_refresh_interval:
                            try:
                                session.get(
                                    'https://www.nseindia.com', timeout=10)
                                last_session_refresh = datetime.now()
                                logger.debug(
                                    "Refreshed NSE session during long download")
                            except:
                                pass
                    except Exception as e:
                        date = future_to_date[future]
                        logger.debug(f"Error downloading {date}: {e}")
        finally:
            # Close session to free resources
            session.close()

        print(
            f"  -> Tried {dates_tried} trading days, found data for {dates_found} days")

        if not all_data:
            print(f"  -> [ERROR] No NSE data found for {symbol}")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()

        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

        # Ensure NSE Bhav data matches yfinance format exactly
        # Add missing columns if not present (should already be added in _parse_nse_bhav_csv)
        if 'Dividends' not in combined_df.columns:
            combined_df['Dividends'] = 0.0
        if 'Stock Splits' not in combined_df.columns:
            combined_df['Stock Splits'] = 0.0

        # Ensure column order matches yfinance: Open, High, Low, Close, Volume, Dividends, Stock Splits
        expected_cols = ['Open', 'High', 'Low', 'Close',
                         'Volume', 'Dividends', 'Stock Splits']
        combined_df = combined_df[[
            col for col in expected_cols if col in combined_df.columns]]

        # Clean data (same as yfinance) - remove timezone, handle duplicates, etc.
        combined_df = self._clean_data(combined_df)

        print(
            f"  -> [OK] Fetched {len(combined_df)} rows from NSE Bhav Copy (formatted like yfinance)")

        # Cache symbol-specific data in yfinance-like format
        try:
            # Format cache data to match yfinance cache structure
            cache_data = {
                'symbol': symbol,
                'source': 'NSE Bhav Copy',
                'fetch_timestamp': datetime.now().isoformat(),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'rows': len(combined_df),
                'data': combined_df.reset_index().to_dict('records')
            }

            with open(symbol_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)

            print(f"  -> Cached {symbol} data to {symbol_cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cache symbol data: {e}")

        return combined_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        # Remove timezone info from index if present
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)

        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Forward fill missing values (limited to 5 days)
        df = df.ffill(limit=5)

        # Drop remaining NaN values
        df = df.dropna()

        # Ensure positive prices and volumes
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df = df[df[col] > 0]

        # Sort by date
        df = df.sort_index()

        return df

    def fetch_all_data(
        self,
        symbol: str,
        period: str = HISTORICAL_PERIOD,
        include_fundamentals: bool = True,
        include_analyst: bool = True,
        include_ownership: bool = True,
        include_earnings: bool = True,
        include_options: bool = False,
        include_news: bool = True
    ) -> Dict:
        """
        Fetch ALL available data for a symbol

        Returns:
            Dictionary containing all available data
        """
        print(f"[INFO] Fetching ALL data for {symbol}...")

        ticker = yf.Ticker(symbol)
        all_data = {}

        # 1. OHLCV Price Data
        data_source = 'yfinance'  # Track data source for metadata
        try:
            print(f"  [1/10] Fetching price history from Yahoo Finance...")
            all_data['price_history'] = ticker.history(period=period)
            all_data['price_history_metadata'] = {
                'rows': len(all_data['price_history']),
                'start_date': str(all_data['price_history'].index[0]) if not all_data['price_history'].empty else None,
                'end_date': str(all_data['price_history'].index[-1]) if not all_data['price_history'].empty else None,
                'data_source': 'yfinance'
            }
            if not all_data['price_history'].empty:
                print(
                    f"    [OK] Price history: {len(all_data['price_history'])} rows from Yahoo Finance")
            else:
                raise ValueError("Empty DataFrame returned from yfinance")
        except Exception as e:
            print(
                f"    [ERROR] Error fetching price history from Yahoo Finance: {e}")
            logger.warning(
                f"yfinance failed for {symbol}, attempting NSE Bhav Copy fallback...")

            # FALLBACK: Try NSE Bhav Copy for Indian stocks
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                try:
                    print(
                        f"    [FALLBACK] Attempting NSE Bhav Copy for {symbol}...")

                    # Parse period to date range
                    end_date = datetime.now()
                    if period.endswith('y'):
                        years = int(period[:-1])
                        start_date = end_date - timedelta(days=years * 365)
                    elif period.endswith('mo'):
                        months = int(period[:-2])
                        start_date = end_date - timedelta(days=months * 30)
                    elif period.endswith('d'):
                        days = int(period[:-1])
                        start_date = end_date - timedelta(days=days)
                    else:
                        start_date = end_date - \
                            timedelta(days=730)  # Default 2 years

                    # Fetch from NSE Bhav Copy
                    df_bhav = self.fetch_nse_bhav_historical(
                        symbol, start_date, end_date)

                    if not df_bhav.empty:
                        # Ensure NSE Bhav data matches yfinance format exactly
                        # Add missing columns if not present (should already be added in _parse_nse_bhav_csv)
                        if 'Dividends' not in df_bhav.columns:
                            df_bhav['Dividends'] = 0.0
                        if 'Stock Splits' not in df_bhav.columns:
                            df_bhav['Stock Splits'] = 0.0

                        # Ensure column order matches yfinance: Open, High, Low, Close, Volume, Dividends, Stock Splits
                        expected_cols = [
                            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                        df_bhav = df_bhav[[
                            col for col in expected_cols if col in df_bhav.columns]]

                        # Clean data (same as yfinance)
                        df_bhav = self._clean_data(df_bhav)

                        all_data['price_history'] = df_bhav
                        all_data['price_history_metadata'] = {
                            'rows': len(df_bhav),
                            'start_date': str(df_bhav.index[0]) if not df_bhav.empty else None,
                            'end_date': str(df_bhav.index[-1]) if not df_bhav.empty else None,
                            'data_source': 'nse_bhav'
                        }
                        data_source = 'nse_bhav'
                        print(
                            f"    [OK] Price history: {len(df_bhav)} rows from NSE Bhav Copy (fallback, formatted like yfinance)")
                        logger.info(
                            f"Successfully used NSE Bhav Copy fallback for {symbol}")
                    else:
                        print(
                            f"    [ERROR] NSE Bhav Copy also returned empty data")
                        all_data['price_history'] = pd.DataFrame()
                        all_data['price_history_metadata'] = {
                            'data_source': 'none', 'rows': 0}
                except Exception as e2:
                    print(
                        f"    [ERROR] NSE Bhav Copy fallback also failed: {e2}")
                    logger.error(
                        f"NSE Bhav Copy fallback failed for {symbol}: {e2}")
                    all_data['price_history'] = pd.DataFrame()
                    all_data['price_history_metadata'] = {
                        'data_source': 'none', 'rows': 0}
            else:
                print(
                    f"    [INFO] {symbol} is not an Indian stock (.NS/.BO), NSE fallback not applicable")
                all_data['price_history'] = pd.DataFrame()
                all_data['price_history_metadata'] = {
                    'data_source': 'none', 'rows': 0}

        # 2. Company Information
        try:
            print(f"  [2/10] Fetching company info...")
            all_data['info'] = ticker.info
            print(f"    [OK] Company info: {len(all_data['info'])} fields")
            all_data['key_metrics'] = self._extract_key_metrics(
                all_data['info'])
        except Exception as e:
            print(f"    [ERROR] Error fetching company info: {e}")
            all_data['info'] = {}
            all_data['key_metrics'] = {}

        # 3. Fundamental Data
        if include_fundamentals:
            print(f"  [3/10] Fetching fundamental data...")
            try:
                all_data['financials'] = ticker.financials
                all_data['quarterly_financials'] = ticker.quarterly_financials
                all_data['balance_sheet'] = ticker.balance_sheet
                all_data['quarterly_balance_sheet'] = ticker.quarterly_balance_sheet
                all_data['cashflow'] = ticker.cashflow
                all_data['quarterly_cashflow'] = ticker.quarterly_cashflow
                print(f"    [OK] Fundamentals fetched")
            except Exception as e:
                logger.error(
                    f"Error fetching fundamentals for {symbol}: {e}", exc_info=True)
                print(f"    [ERROR] Error fetching fundamentals: {e}")
                all_data['financials'] = pd.DataFrame()
        else:
            print(f"  [3/10] Skipping fundamentals...")

        # 4. Analyst Data
        if include_analyst:
            print(f"  [4/10] Fetching analyst data...")
            try:
                all_data['recommendations'] = ticker.recommendations
                all_data['analyst_price_targets'] = ticker.analyst_price_targets
                all_data['earnings_estimate'] = ticker.earnings_estimate
                all_data['revenue_estimate'] = ticker.revenue_estimate
                print(f"    [OK] Analyst data fetched")
            except Exception as e:
                logger.error(
                    f"Error fetching analyst data for {symbol}: {e}", exc_info=True)
                print(f"    [ERROR] Error fetching analyst data: {e}")
                all_data['recommendations'] = pd.DataFrame()
        else:
            print(f"  [4/10] Skipping analyst data...")

        # 5. Ownership Data
        if include_ownership:
            print(f"  [5/10] Fetching ownership data...")
            try:
                all_data['major_holders'] = ticker.major_holders
                all_data['institutional_holders'] = ticker.institutional_holders
                all_data['mutualfund_holders'] = ticker.mutualfund_holders
                all_data['insider_transactions'] = ticker.insider_transactions
                all_data['insider_roster_holders'] = ticker.insider_roster_holders
                print(f"    [OK] Ownership data fetched")
            except Exception as e:
                print(f"    [ERROR] Error fetching ownership: {e}")
                all_data['major_holders'] = pd.DataFrame()
        else:
            print(f"  [5/10] Skipping ownership data...")

        # 6. Earnings Data
        if include_earnings:
            print(f"  [6/10] Fetching earnings data...")
            try:
                all_data['earnings'] = ticker.earnings
                all_data['quarterly_earnings'] = ticker.quarterly_earnings
                all_data['earnings_dates'] = ticker.earnings_dates
                all_data['earnings_history'] = ticker.earnings_history
                print(f"    [OK] Earnings data fetched")
            except Exception as e:
                print(f"    [ERROR] Error fetching earnings: {e}")
                all_data['earnings'] = pd.DataFrame()
        else:
            print(f"  [6/10] Skipping earnings data...")

        # 7. Corporate Actions
        try:
            print(f"  [7/10] Fetching corporate actions...")
            all_data['splits'] = ticker.splits
            all_data['dividends'] = ticker.dividends
            print(f"    [OK] Corporate actions fetched")
        except Exception as e:
            print(f"    [ERROR] Error fetching corporate actions: {e}")
            all_data['splits'] = pd.Series()
            all_data['dividends'] = pd.Series()

        # 8. Options Data (if requested)
        if include_options:
            print(f"  [8/10] Fetching options data...")
            try:
                all_data['options'] = ticker.options
                all_data['option_chain'] = {}
                # Only fetch current expiry for performance
                if all_data['options']:
                    current_expiry = all_data['options'][0]
                    all_data['option_chain'][current_expiry] = ticker.option_chain(
                        current_expiry)
                print(f"    [OK] Options data fetched")
            except Exception as e:
                print(f"    [ERROR] Error fetching options: {e}")
                all_data['options'] = ()
                all_data['option_chain'] = {}
        else:
            print(f"  [8/10] Skipping options data...")

        # 9. News Data (if requested)
        if include_news:
            print(f"  [9/10] Fetching news...")
            try:
                all_data['news'] = ticker.news
                print(
                    f"    [OK] News fetched: {len(all_data.get('news', []))} articles")
            except Exception as e:
                print(f"    [ERROR] Error fetching news: {e}")
                all_data['news'] = []
        else:
            print(f"  [9/10] Skipping news...")

        # 10. Metadata
        print(f"  [10/10] Adding metadata...")
        all_data['metadata'] = {
            'symbol': symbol,
            'fetch_timestamp': datetime.now().isoformat(),
            'period': period,
            'data_source': data_source,
            'has_news': include_news and bool(all_data.get('news'))
        }

        # Cache all data
        json_path = get_symbol_cache_path(symbol)
        try:
            with open(json_path, 'w') as f:
                # Convert DataFrame and Series to JSON-serializable format
                serializable_data = {}
                for key, value in all_data.items():
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        serializable_data[key] = value.to_dict()
                    elif isinstance(value, np.ndarray):
                        serializable_data[key] = value.tolist()
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        serializable_data[key] = value.isoformat()
                    else:
                        serializable_data[key] = value

                json.dump(serializable_data, f, indent=2, default=str)
            print(f"  [OK] All data cached to {json_path}")
        except Exception as e:
            print(f"  [ERROR] Failed to cache data: {e}")
            logger.error(f"Failed to cache data for {symbol}: {e}")

        print(f"[INFO] Completed fetching ALL data for {symbol}")
        return all_data

    def load_all_data(self, symbol: str) -> Optional[Dict]:
        """Load all cached data for a symbol"""
        json_path = get_symbol_cache_path(symbol)

        if not json_path.exists():
            print(f"[INFO] No cached data found for {symbol}")
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convert dict back to DataFrame/Series where needed
            for key, value in data.items():
                if key == 'price_history' and isinstance(value, list):
                    # Convert price history back to DataFrame
                    if value:  # Check if not empty
                        df = pd.DataFrame(value)
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        data[key] = df

            print(f"[INFO] Loaded cached data for {symbol}")
            return data

        except Exception as e:
            print(f"[ERROR] Failed to load cached data for {symbol}: {e}")
            logger.error(f"Failed to load cached data for {symbol}: {e}")
            return None

    def _extract_key_metrics(self, info: Dict) -> Dict:
        """Extract key financial metrics from company info"""
        key_metrics = {}

        # Market metrics
        key_metrics['market_cap'] = info.get('marketCap')
        key_metrics['enterprise_value'] = info.get('enterpriseValue')
        key_metrics['trailing_pe'] = info.get('trailingPE')
        key_metrics['forward_pe'] = info.get('forwardPE')
        key_metrics['peg_ratio'] = info.get('pegRatio')

        # Valuation metrics
        key_metrics['price_to_sales'] = info.get(
            'priceToSalesTrailing12Months')
        key_metrics['price_to_book'] = info.get('priceToBook')
        key_metrics['enterprise_to_revenue'] = info.get('enterpriseToRevenue')
        key_metrics['enterprise_to_ebitda'] = info.get('enterpriseToEbitda')

        # Profitability metrics
        key_metrics['profit_margins'] = info.get('profitMargins')
        key_metrics['ebitda_margins'] = info.get('ebitdaMargins')
        key_metrics['gross_margins'] = info.get('grossMargins')
        key_metrics['operating_margins'] = info.get('operatingMargins')

        # Financial health
        key_metrics['total_cash'] = info.get('totalCash')
        key_metrics['total_debt'] = info.get('totalDebt')
        key_metrics['debt_to_equity'] = info.get('debtToEquity')
        key_metrics['current_ratio'] = info.get('currentRatio')
        key_metrics['quick_ratio'] = info.get('quickRatio')

        # Returns
        key_metrics['return_on_assets'] = info.get('returnOnAssets')
        key_metrics['return_on_equity'] = info.get('returnOnEquity')

        # Cash flow
        key_metrics['free_cashflow'] = info.get('freeCashflow')
        key_metrics['operating_cashflow'] = info.get('operatingCashflow')

        # Growth
        key_metrics['revenue_growth'] = info.get('revenueGrowth')
        key_metrics['earnings_growth'] = info.get('earningsGrowth')

        # Dividends
        key_metrics['dividend_rate'] = info.get('dividendRate')
        key_metrics['dividend_yield'] = info.get('dividendYield')
        key_metrics['payout_ratio'] = info.get('payoutRatio')

        # Valuation
        key_metrics['book_value'] = info.get('bookValue')
        key_metrics['price_to_book'] = info.get('priceToBook')

        return key_metrics


# ============================================================================
# FEATURE ENGINEERING COMPONENTS
# ============================================================================

class FeatureEngineer:
    """Feature engineering for stock prediction"""

    def __init__(self, feature_cache_dir: Path = FEATURE_CACHE_DIR):
        """Initialize Feature Engineer"""
        self.feature_cache_dir = feature_cache_dir
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"FeatureEngineer initialized with cache dir: {feature_cache_dir}")

    def calculate_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate all technical indicators and features"""
        if df.empty:
            logger.error(
                f"Cannot calculate features for {symbol}: DataFrame is empty")
            return df

        print(
            f"[FEATURES] Calculating 50+ technical indicators for {symbol}...")

        # Ensure we have enough data
        if len(df) < 50:
            logger.warning(
                f"Insufficient data for {symbol} ({len(df)} rows), some indicators may be NaN")

        # Make a copy to avoid modifying original DataFrame
        df_features = df.copy()

        # Calculate all feature categories
        df_features = self._calculate_price_features(df_features)
        df_features = self._calculate_trend_indicators(df_features)
        df_features = self._calculate_volatility_indicators(df_features)
        df_features = self._calculate_volume_indicators(df_features)
        df_features = self._calculate_momentum_indicators(df_features)
        df_features = self._calculate_support_resistance(df_features)
        df_features = self._calculate_pattern_features(df_features)
        df_features = self._calculate_advanced_analytics(df_features)

        # Remove any rows with all NaN values
        df_features = df_features.dropna(how='all')

        print(
            f"[FEATURES] [OK] Calculated {len(df_features.columns)} features for {symbol}")
        logger.info(
            f"Calculated {len(df_features.columns)} features for {symbol}")

        return df_features

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price-based features (8+ features)"""
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['Close'] - df['Close'].shift(1)
        df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open']
        df['real_body'] = abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (
            df['High'] - np.maximum(df['Close'], df['Open'])) / df['Close']
        df['lower_shadow'] = (np.minimum(
            df['Close'], df['Open']) - df['Low']) / df['Close']
        df['daily_range'] = (df['High'] - df['Low']) / df['Low']
        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators (11+ indicators)"""
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)

        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DMP'] = adx['DMP_14']
            df['DMN'] = adx['DMN_14']

        psar = ta.psar(df['High'], df['Low'], df['Close'])
        if psar is not None:
            psar_long = psar.iloc[:, 0] if isinstance(
                psar, pd.DataFrame) else psar
            psar_short = psar.iloc[:, 1] if isinstance(
                psar, pd.DataFrame) and psar.shape[1] > 1 else None
            df['PSAR'] = psar_long.fillna(
                psar_short) if psar_short is not None else psar_long

        kc = ta.kc(df['High'], df['Low'], df['Close'], length=20)
        if kc is not None:
            df['KC_upper'] = kc['KCUe_20_2']
            df['KC_middle'] = kc['KCBe_20_2']
            df['KC_lower'] = kc['KCLe_20_2']

        dc = ta.donchian(df['High'], df['Low'],
                         lower_length=20, upper_length=20)
        if dc is not None:
            df['DC_upper'] = dc['DCU_20_20']
            df['DC_middle'] = dc['DCM_20_20']
            df['DC_lower'] = dc['DCL_20_20']

        df['MA_alignment'] = (
            (df['SMA_10'] > df['SMA_20']).astype(int) +
            (df['SMA_20'] > df['SMA_50']).astype(int) +
            (df['SMA_50'] > df['SMA_200']).astype(int)
        ) / 3.0

        df['trend_direction'] = np.where(
            df['Close'] > df['SMA_50'], 1,
            np.where(df['Close'] < df['SMA_50'], -1, 0)
        )
        return df

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators (5+ indicators)"""
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None:
            cols = bbands.columns.tolist()
            if len(cols) >= 3:
                df['BB_lower'] = bbands.iloc[:, 0]
                df['BB_middle'] = bbands.iloc[:, 1]
                df['BB_upper'] = bbands.iloc[:, 2]
                if len(cols) >= 4:
                    df['BB_width'] = bbands.iloc[:, 3]
                if len(cols) >= 5:
                    df['BB_pct'] = bbands.iloc[:, 4]

        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['STD_20'] = df['Close'].rolling(window=20).std()
        returns = df['Close'].pct_change()
        df['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators (7+ indicators)"""
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        df['CMF'] = ta.cmf(df['High'], df['Low'],
                           df['Close'], df['Volume'], length=20)
        df['VROC'] = ta.roc(df['Volume'], length=12)
        df['EMV'] = ta.eom(df['High'], df['Low'],
                           df['Close'], df['Volume'], length=14)
        df['volume_ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)
        df['volume_trend'] = np.where(
            df['Volume'] > df['Volume_SMA_20'], 1, -1)
        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators (10+ indicators)"""
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_7'] = ta.rsi(df['Close'], length=7)
        df['RSI_21'] = ta.rsi(df['Close'], length=21)

        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['STOCH_K'] = stoch.iloc[:, 0] if isinstance(
                stoch, pd.DataFrame) else stoch
            df['STOCH_D'] = stoch.iloc[:, 1] if isinstance(
                stoch, pd.DataFrame) and stoch.shape[1] > 1 else None

        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            cols = macd.columns.tolist()
            if len(cols) >= 2:
                df['MACD'] = macd.iloc[:, 0]
                df['MACD_signal'] = macd.iloc[:, 1]
                if len(cols) >= 3:
                    df['MACD_hist'] = macd.iloc[:, 2]

        # CCI
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)

        # Williams %R
        df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)

        # Rate of Change
        df['ROC_10'] = ta.roc(df['Close'], length=10)
        df['ROC_30'] = ta.roc(df['Close'], length=30)

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support/resistance levels (5+ indicators)"""
        df['pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['Low']
        df['pivot_s1'] = 2 * df['pivot'] - df['High']
        df['pivot_r2'] = df['pivot'] + (df['High'] - df['Low'])
        df['pivot_s2'] = df['pivot'] - (df['High'] - df['Low'])

        window = 50
        rolling_high = df['High'].rolling(window=window).max()
        rolling_low = df['Low'].rolling(window=window).min()
        diff = rolling_high - rolling_low

        df['fib_0.236'] = rolling_high - 0.236 * diff
        df['fib_0.382'] = rolling_high - 0.382 * diff
        df['fib_0.500'] = rolling_high - 0.500 * diff
        df['fib_0.618'] = rolling_high - 0.618 * diff
        df['price_position'] = (df['Close'] - rolling_low) / (diff + 1e-10)
        return df

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition features (5+ features)"""
        doji = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])
        if doji is not None:
            df['CDL_DOJI'] = doji.iloc[:, 0].fillna(0).astype(int) if isinstance(
                doji, pd.DataFrame) else doji.fillna(0).astype(int)
        else:
            df['CDL_DOJI'] = 0

        hammer = ta.cdl_pattern(
            df['Open'], df['High'], df['Low'], df['Close'], name='hammer')
        if hammer is not None:
            df['CDL_HAMMER'] = hammer.iloc[:, 0].fillna(0).astype(int) if isinstance(
                hammer, pd.DataFrame) else hammer.fillna(0).astype(int)
        else:
            df['CDL_HAMMER'] = 0

        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        df['gap_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['High'] < df['Low'].shift(1)).astype(int)
        df['bullish_engulf'] = (
            (df['Close'] > df['Open']) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['Close'] > df['Open'].shift(1))
        ).astype(int)
        df['bearish_engulf'] = (
            (df['Close'] < df['Open']) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1))
        ).astype(int)
        return df

    def _calculate_advanced_analytics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced analytics (8+ features)"""
        df['price_to_sma_10'] = df['Close'] / (df['SMA_10'] + 1e-10)
        df['price_to_sma_50'] = df['Close'] / (df['SMA_50'] + 1e-10)
        df['price_to_sma_200'] = df['Close'] / (df['SMA_200'] + 1e-10)

        price_trend = (df['Close'] - df['Close'].shift(5)) > 0
        rsi_trend = (df['RSI_14'] - df['RSI_14'].shift(5)) > 0
        df['rsi_divergence'] = (price_trend != rsi_trend).astype(int)
        df['macd_hist_increasing'] = (
            df['MACD_hist'] > df['MACD_hist'].shift(1)).astype(int)
        df['bb_position'] = (df['Close'] - df['BB_lower']) / \
            (df['BB_upper'] - df['BB_lower'] + 1e-10)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['daily_return'] = df['Close'].pct_change()
        df['daily_return_ma_5'] = df['daily_return'].rolling(window=5).mean()
        returns = df['daily_return']
        df['sharpe_20'] = (
            returns.rolling(window=20).mean() /
            (returns.rolling(window=20).std() + 1e-10)
        ) * np.sqrt(252)
        df['returns_autocorr'] = returns.rolling(window=20).apply(
            lambda x: x.autocorr(), raw=False
        )
        return df

    def save_features(self, df: pd.DataFrame, symbol: str):
        """Save LIVE/CURRENT features only (latest values)"""
        if df.empty:
            logger.error(
                f"Cannot save features for {symbol}: DataFrame is empty")
            return

        latest_features = df.iloc[-1].to_dict()
        latest_date = str(df.index[-1])

        latest_timestamp = pd.to_datetime(df.index[-1])
        if hasattr(latest_timestamp, 'tz') and latest_timestamp.tz is not None:
            latest_timestamp = latest_timestamp.tz_localize(None)

        days_old = (pd.Timestamp.now() - latest_timestamp).days

        live_data = {
            'symbol': symbol,
            'fetch_time': str(pd.Timestamp.now()),
            'latest_date': latest_date,
            'data_freshness_days': days_old,
            'current_price': float(latest_features.get('Close', 0)),
            'current_features': latest_features,
            'total_features': len(df.columns),
            'feature_calculation_periods': {
                'RSI_14': 'Last 14 days',
                'MACD': 'Last 12-26 days',
                'SMA_10': 'Last 10 days',
                'SMA_20': 'Last 20 days',
                'SMA_50': 'Last 50 days',
                'SMA_200': 'Last 200 days',
                'BB_bands': 'Last 20 days',
                'ATR_14': 'Last 14 days',
                'OBV': 'Cumulative (all historical data)',
                'pivot_points': 'Current day High/Low/Close',
                'fibonacci': 'Last 50 days High/Low range'
            }
        }

        json_path = self.feature_cache_dir / f"{symbol}_features.json"

        try:
            with open(json_path, 'w') as f:
                json.dump(live_data, f, indent=2, default=str)

            logger.info(f"Saved LIVE features for {symbol} to {json_path}")

            # Print current indicators to console
            print(f"\n{'='*80}")
            print(f"LIVE TECHNICAL INDICATORS for {symbol}")
            print(f"{'='*80}")
            print(f"Date: {latest_date}")
            print(f"Current Price: ₹{latest_features.get('Close', 0):.2f}")
            print(f"Data Freshness: {days_old} days old")
            print(f"Total Features: {len(df.columns)}")
            print("")
            print("TREND INDICATORS:")
            print(f"  SMA_10: ₹{latest_features.get('SMA_10', 0):.2f}")
            print(f"  SMA_50: ₹{latest_features.get('SMA_50', 0):.2f}")
            print(f"  SMA_200: ₹{latest_features.get('SMA_200', 0):.2f}")
            print(f"  EMA_12: ₹{latest_features.get('EMA_12', 0):.2f}")
            print(f"  EMA_26: ₹{latest_features.get('EMA_26', 0):.2f}")
            print(f"  ADX: {latest_features.get('ADX', 0):.2f}")
            print(
                f"  Trend Direction: {latest_features.get('trend_direction', 0):.0f}")
            print("")
            print("MOMENTUM INDICATORS:")
            print(f"  RSI_14: {latest_features.get('RSI_14', 0):.2f}")
            print(f"  MACD: {latest_features.get('MACD', 0):.4f}")
            print(
                f"  MACD Signal: {latest_features.get('MACD_signal', 0):.4f}")
            print(
                f"  MACD Histogram: {latest_features.get('MACD_hist', 0):.4f}")
            print(f"  CCI: {latest_features.get('CCI', 0):.2f}")
            print("")
            print("VOLATILITY INDICATORS:")
            print(f"  ATR: ₹{latest_features.get('ATR', 0):.2f}")
            print(f"  BB Upper: ₹{latest_features.get('BB_upper', 0):.2f}")
            print(f"  BB Middle: ₹{latest_features.get('BB_middle', 0):.2f}")
            print(f"  BB Lower: ₹{latest_features.get('BB_lower', 0):.2f}")
            print(
                f"  Volatility (20-day): {latest_features.get('volatility_20', 0):.4f}")
            print("")
            print("VOLUME INDICATORS:")
            print(f"  Volume: {latest_features.get('Volume', 0):,.0f}")
            print(f"  OBV: {latest_features.get('OBV', 0):,.0f}")
            print(
                f"  Volume Ratio: {latest_features.get('volume_ratio', 0):.2f}")
            print(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"Failed to save features for {symbol}: {e}")


# ============================================================================
# MACHINE LEARNING COMPONENTS
# ============================================================================

def prepare_features_for_ml(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and target for ML training"""
    # Select feature columns
    X = df[feature_columns].copy()

    # Create target (next day's price change)
    y = df['Close'].shift(-1) / df['Close'] - 1  # Percentage change

    # Remove last row (no target) and any rows with NaN
    X = X[:-1]
    y = y[:-1]
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    # Convert to numpy arrays
    X = X.values
    y = y.values

    return X, y


def train_ml_models(symbol: str, horizon: str = "intraday", verbose: bool = True) -> Union[bool, Dict]:
    """
    Train all ML models for a symbol

    Args:
        symbol: Stock symbol to train on
        horizon: Time horizon ("intraday", "short", "long")
        verbose: Whether to print detailed output

    Returns:
        Dict with training results and metrics, or False if failed
    """
    try:
        if verbose:
            print(f"\n[ML] Training models for {symbol} ({horizon})...")
            print(f"  -> Loading data and features...")

        # Load data
        ingester = EnhancedDataIngester()
        all_data = ingester.load_all_data(symbol)

        if not all_data:
            if verbose:
                print(f"  -> [ERROR] No data found for {symbol}")
            return False

        df = all_data.get('price_history')
        # Fix: Check if df is a DataFrame and not empty, or if it's a dict check if it has data
        if df is None:
            if verbose:
                print(f"  -> [ERROR] No price history for {symbol}")
            return False
        # Handle both DataFrame and dict cases
        elif isinstance(df, pd.DataFrame) and df.empty:
            if verbose:
                print(f"  -> [ERROR] No price history for {symbol}")
            return False
        elif isinstance(df, dict) and len(df) == 0:
            if verbose:
                print(f"  -> [ERROR] No price history for {symbol}")
            return False

        # Convert dict to DataFrame if needed
        if isinstance(df, dict):
            try:
                df = pd.DataFrame(df)
                if df.empty:
                    if verbose:
                        print(f"  -> [ERROR] No price history for {symbol}")
                    return False
            except Exception as e:
                if verbose:
                    print(
                        f"  -> [ERROR] Could not convert dict to DataFrame for {symbol}: {e}")
                return False

        # Load features
        feature_info_path = FEATURE_CACHE_DIR / f"{symbol}_feature_info.json"
        historical_features_path = FEATURE_CACHE_DIR / \
            f"{symbol}_historical_features.pkl"

        if not feature_info_path.exists() or not historical_features_path.exists():
            if verbose:
                print(
                    f"  -> [ERROR] No historical features found for {symbol}")
            return False

        # Load feature info
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)

        feature_columns = feature_info['feature_columns']

        # Load historical features DataFrame
        features_df = pd.read_pickle(historical_features_path)

        # Use features_df instead of df for training
        df = features_df

        if verbose:
            print(f"  -> Found {len(feature_columns)} features")
            print(f"  -> Preparing data for training...")

        # Prepare features and target
        X, y = prepare_features_for_ml(df, feature_columns)

        if len(X) == 0:
            if verbose:
                print(
                    f"  -> [ERROR] No valid training data after preprocessing")
            return False

        if verbose:
            print(f"  -> Training data shape: {X.shape}")
            print(f"  -> Target data shape: {y.shape}")
            print(f"  -> Splitting data...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # 1. Random Forest
        if verbose:
            print(f"  -> Training Random Forest...")

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)

        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        # Save Random Forest model
        rf_path = MODEL_DIR / f"{symbol}_{horizon}_rf_model.pkl"
        joblib.dump({
            'model': rf_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metrics': {
                'mse': rf_mse,
                'mae': rf_mae,
                'r2': rf_r2
            }
        }, rf_path)

        results['rf'] = {
            'mse': rf_mse,
            'mae': rf_mae,
            'r2': rf_r2,
            'model_path': str(rf_path)
        }

        if verbose:
            print(
                f"    [OK] Random Forest - MSE: {rf_mse:.6f}, MAE: {rf_mae:.6f}, R²: {rf_r2:.4f}")

        # 2. LightGBM
        if verbose:
            print(f"  -> Training LightGBM...")

        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)

        # Evaluate LightGBM
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_mse = mean_squared_error(y_test, lgb_pred)
        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        lgb_r2 = r2_score(y_test, lgb_pred)

        # Save LightGBM model
        lgb_path = MODEL_DIR / f"{symbol}_{horizon}_lgb_model.pkl"
        joblib.dump({
            'model': lgb_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metrics': {
                'mse': lgb_mse,
                'mae': lgb_mae,
                'r2': lgb_r2
            }
        }, lgb_path)

        results['lgb'] = {
            'mse': lgb_mse,
            'mae': lgb_mae,
            'r2': lgb_r2,
            'model_path': str(lgb_path)
        }

        if verbose:
            print(
                f"    [OK] LightGBM - MSE: {lgb_mse:.6f}, MAE: {lgb_mae:.6f}, R²: {lgb_r2:.4f}")

        # 3. XGBoost
        if verbose:
            print(f"  -> Training XGBoost...")

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        )
        xgb_model.fit(X_train_scaled, y_train)

        # Evaluate XGBoost
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)

        # Save XGBoost model
        xgb_path = MODEL_DIR / f"{symbol}_{horizon}_xgb_model.pkl"
        joblib.dump({
            'model': xgb_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metrics': {
                'mse': xgb_mse,
                'mae': xgb_mae,
                'r2': xgb_r2
            }
        }, xgb_path)

        results['xgb'] = {
            'mse': xgb_mse,
            'mae': xgb_mae,
            'r2': xgb_r2,
            'model_path': str(xgb_path)
        }

        if verbose:
            print(
                f"    [OK] XGBoost - MSE: {xgb_mse:.6f}, MAE: {xgb_mae:.6f}, R²: {xgb_r2:.4f}")

        # 4. DQN (Reinforcement Learning)
        if verbose:
            print(f"  -> Training DQN (this will take 60-90 seconds)...")

        # Initialize DQN agent
        dqn_agent = DQNTradingAgent(n_features=X_train.shape[1])

        # Training loop
        episodes = 50  # Reduced for faster training
        total_reward = 0
        rewards_history = []

        # Convert to tensors for DQN training
        states = torch.FloatTensor(X_train_scaled).to(dqn_agent.device)
        next_states = torch.FloatTensor(X_test_scaled).to(dqn_agent.device)

        for episode in range(episodes):
            episode_reward = 0

            # Simple training loop - in practice, you'd have a more complex environment
            for i in range(min(len(states), 1000)):  # Limit to 1000 steps per episode
                # Get state
                state = states[i].cpu().numpy()

                # Get action
                action = dqn_agent.get_action(state, training=True)

                # Calculate reward (simplified - in practice, this would come from environment)
                # Here we use the actual return as reward
                if i < len(y_train):
                    reward = y_train[i] * 100  # Scale reward
                else:
                    reward = 0

                # Get next state
                if i + 1 < len(states):
                    next_state = states[i + 1].cpu().numpy()
                    done = False
                else:
                    next_state = np.zeros_like(state)
                    done = True

                # Store experience
                dqn_agent.remember(state, action, reward, next_state, done)

                # Train step
                loss = dqn_agent.train_step()

                # Update target network periodically
                if dqn_agent.total_steps % dqn_agent.target_update_freq == 0:
                    dqn_agent.update_target_network()

                # Update epsilon
                dqn_agent.update_epsilon()

                # Update metrics
                episode_reward += reward
                dqn_agent.total_steps += 1

                if done:
                    break

            dqn_agent.total_episodes += 1
            dqn_agent.cumulative_reward += episode_reward
            dqn_agent.rewards_history.append(episode_reward)
            rewards_history.append(episode_reward)
            total_reward += episode_reward

            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(
                    rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
                print(
                    f"    Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.4f}, Epsilon: {dqn_agent.epsilon:.4f}")

        # Save DQN model
        dqn_agent.save(symbol, horizon)

        # Calculate DQN metrics
        dqn_metrics = dqn_agent._calculate_performance_metrics()

        results['dqn'] = {
            'metrics': dqn_metrics,
            'model_path': str(MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt")
        }

        if verbose:
            print(f"    [OK] DQN - Total Episodes: {dqn_metrics.get('total_episodes', 0)}, "
                  f"Avg Reward: {dqn_metrics.get('average_reward', 0):.6f}")

        # Summary
        if verbose:
            print(f"\n[ML] Training complete for {symbol} ({horizon})")
            print(
                f"  -> Random Forest: MSE={rf_mse:.6f}, MAE={rf_mae:.6f}, R²={rf_r2:.4f}")
            print(
                f"  -> LightGBM: MSE={lgb_mse:.6f}, MAE={lgb_mae:.6f}, R²={lgb_r2:.4f}")
            print(
                f"  -> XGBoost: MSE={xgb_mse:.6f}, MAE={xgb_mae:.6f}, R²={xgb_r2:.4f}")
            print(f"  -> DQN: Avg Reward={dqn_metrics.get('average_reward', 0):.6f}, "
                  f"Sharpe={dqn_metrics.get('sharpe_ratio', 0):.4f}")

        return {
            'success': True,
            'symbol': symbol,
            'horizon': horizon,
            'results': results,
            'dqn_metrics': dqn_metrics
        }

    except Exception as e:
        logger.error(
            f"Error training ML models for {symbol}: {e}", exc_info=True)
        if verbose:
            print(f"  -> [ERROR] Failed to train models: {e}")
        return False


def predict_stock_price(symbol: str, horizon: str = "intraday", verbose: bool = True) -> Optional[Dict]:
    """
    Generate stock price prediction using ensemble of trained models

    Args:
        symbol: Stock symbol to predict
        horizon: Time horizon ("intraday", "short", "long")
        verbose: Whether to print detailed output

    Returns:
        Dict with prediction results
    """
    try:
        if verbose:
            print(
                f"\n[PREDICT] Generating prediction for {symbol} ({horizon})...")

        # Load latest features
        feature_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
        if not feature_path.exists():
            if verbose:
                print(f"  -> [ERROR] No features found for {symbol}")
            return None

        with open(feature_path, 'r') as f:
            feature_data = json.load(f)

        current_features = feature_data['current_features']
        current_price = feature_data['current_price']

        # Get feature columns from one of the models
        model_paths = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*_model.pkl"))
        if not model_paths:
            # Try DQN model
            dqn_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
            if not dqn_path.exists():
                if verbose:
                    print(
                        f"  -> [ERROR] No trained models found for {symbol} ({horizon})")
                return None
            else:
                # Load DQN model
                # Will be updated when loading
                dqn_agent = DQNTradingAgent(n_features=1)
                dqn_agent.load(symbol, horizon)

                # Prepare state
                feature_columns = list(current_features.keys())
                exclude_columns = ['Date', 'Open', 'High', 'Low',
                                   'Close', 'Volume', 'Dividends', 'Stock Splits']
                feature_columns = [
                    col for col in feature_columns if col not in exclude_columns]

                state_values = [current_features.get(
                    col, 0) for col in feature_columns]
                state = np.array(state_values, dtype=np.float32)

                # Get action from DQN
                action = dqn_agent.get_action(state, training=False)

                # Map action to prediction
                action_map = {0: 'LONG', 1: 'SHORT', 2: 'HOLD'}
                predicted_action = action_map.get(action, 'HOLD')

                # Simple return prediction based on action
                if predicted_action == 'LONG':
                    predicted_return = 0.02  # 2% expected return
                    confidence = 0.8
                elif predicted_action == 'SHORT':
                    predicted_return = -0.01  # -1% expected return
                    confidence = 0.7
                else:  # HOLD
                    predicted_return = 0.005  # 0.5% expected return
                    confidence = 0.6

                predicted_price = current_price * (1 + predicted_return)

                prediction = {
                    'symbol': symbol,
                    'horizon': horizon,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predicted_return': predicted_return * 100,  # Convert to percentage
                    'action': predicted_action,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'model_predictions': {
                        'dqn_action': predicted_action,
                        'dqn_confidence': confidence
                    },
                    'horizon_details': {
                        'intraday': {'days': 1, 'description': 'Same day / Next day'},
                        'short': {'days': 5, 'description': '1 week (Swing trading)'},
                        'long': {'days': 30, 'description': '1 month (Position trading)'}
                    }.get(horizon, {'days': 1, 'description': 'Unknown'}),
                    'risk_metrics': {
                        'volatility_20': current_features.get('volatility_20', 0),
                        'sharpe_20': current_features.get('sharpe_20', 0),
                        'atr': current_features.get('ATR', 0)
                    }
                }

                if verbose:
                    print(
                        f"  -> [OK] DQN Prediction: {predicted_action} (Confidence: {confidence:.4f})")
                    print(f"  -> Current Price: ₹{current_price:.2f}")
                    print(f"  -> Predicted Price: ₹{predicted_price:.2f}")
                    print(
                        f"  -> Expected Return: {predicted_return * 100:+.2f}%")

                return prediction

        # Load one model to get feature columns
        model_data = joblib.load(model_paths[0])
        feature_columns = model_data['feature_columns']

        # Prepare features for prediction
        feature_values = [current_features.get(
            col, 0) for col in feature_columns]
        X = np.array(feature_values).reshape(1, -1)

        # Scale features
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X)

        # Load all models and get predictions
        predictions = {}

        # Random Forest
        try:
            rf_path = MODEL_DIR / f"{symbol}_{horizon}_rf_model.pkl"
            if rf_path.exists():
                rf_data = joblib.load(rf_path)
                rf_model = rf_data['model']
                rf_pred = rf_model.predict(X_scaled)[0]
                predictions['rf'] = rf_pred
        except Exception as e:
            logger.warning(f"Failed to load RF model for {symbol}: {e}")

        # LightGBM
        try:
            lgb_path = MODEL_DIR / f"{symbol}_{horizon}_lgb_model.pkl"
            if lgb_path.exists():
                lgb_data = joblib.load(lgb_path)
                lgb_model = lgb_data['model']
                lgb_pred = lgb_model.predict(X_scaled)[0]
                predictions['lgb'] = lgb_pred
        except Exception as e:
            logger.warning(f"Failed to load LGB model for {symbol}: {e}")

        # XGBoost
        try:
            xgb_path = MODEL_DIR / f"{symbol}_{horizon}_xgb_model.pkl"
            if xgb_path.exists():
                xgb_data = joblib.load(xgb_path)
                xgb_model = xgb_data['model']
                xgb_pred = xgb_model.predict(X_scaled)[0]
                predictions['xgb'] = xgb_pred
        except Exception as e:
            logger.warning(f"Failed to load XGB model for {symbol}: {e}")

        # DQN
        dqn_confidence = 0.5
        try:
            dqn_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
            if dqn_path.exists():
                dqn_agent = DQNTradingAgent(n_features=X_scaled.shape[1])
                dqn_agent.load(symbol, horizon)

                # Get action from DQN
                action = dqn_agent.get_action(X_scaled[0], training=False)

                # Map action to return
                action_returns = {0: 0.02, 1: -0.01,
                                  2: 0.005}  # LONG, SHORT, HOLD
                dqn_pred = action_returns.get(action, 0.005)
                predictions['dqn'] = dqn_pred

                # Confidence based on recent performance
                if hasattr(dqn_agent, 'rewards_history') and dqn_agent.rewards_history:
                    recent_rewards = dqn_agent.rewards_history[-20:] if len(
                        dqn_agent.rewards_history) >= 20 else dqn_agent.rewards_history
                    avg_reward = np.mean(recent_rewards)
                    std_reward = np.std(recent_rewards)
                    if std_reward > 0:
                        sharpe = avg_reward / std_reward
                        # Convert Sharpe ratio to confidence (0-1 scale)
                        dqn_confidence = min(1.0, max(0.0, (sharpe + 2) / 4))
        except Exception as e:
            logger.warning(f"Failed to load DQN model for {symbol}: {e}")

        # Ensemble prediction (average of all model predictions)
        if predictions:
            # Weight DQN prediction more heavily (30% weight) since it's reinforcement learning
            weights = {
                'rf': 0.25,
                'lgb': 0.25,
                'xgb': 0.25,
                'dqn': 0.25
            }

            # Only use weights for models that have predictions
            available_models = list(predictions.keys())
            total_weight = sum(weights[model] for model in available_models)

            if total_weight > 0:
                # Normalize weights
                normalized_weights = {
                    model: weights[model] / total_weight for model in available_models}

                # Calculate weighted average
                ensemble_pred = sum(
                    predictions[model] * normalized_weights[model] for model in available_models)
            else:
                ensemble_pred = 0
        else:
            ensemble_pred = 0

        # Determine action based on prediction
        if ensemble_pred > 0.01:  # 1% threshold for LONG
            action = "LONG"
            # Scale confidence based on return
            confidence = min(0.95, 0.7 + (ensemble_pred * 10))
        elif ensemble_pred < -0.01:  # -1% threshold for SHORT
            action = "SHORT"
            confidence = min(0.95, 0.7 + (abs(ensemble_pred) * 10))
        else:
            action = "HOLD"
            confidence = min(0.95, 0.5 + (abs(ensemble_pred) * 50))

        # Apply DQN confidence adjustment
        confidence = confidence * 0.7 + dqn_confidence * 0.3

        # Calculate predicted price
        predicted_return = ensemble_pred
        predicted_price = current_price * (1 + predicted_return)

        # Risk metrics
        risk_metrics = {
            'volatility_20': current_features.get('volatility_20', 0),
            'sharpe_20': current_features.get('sharpe_20', 0),
            'atr': current_features.get('ATR', 0),
            'volume_ratio': current_features.get('volume_ratio', 0)
        }

        prediction = {
            'symbol': symbol,
            'horizon': horizon,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': predicted_return * 100,  # Convert to percentage
            'action': action,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_predictions': predictions,
            'horizon_details': {
                'intraday': {'days': 1, 'description': 'Same day / Next day'},
                'short': {'days': 5, 'description': '1 week (Swing trading)'},
                'long': {'days': 30, 'description': '1 month (Position trading)'}
            }.get(horizon, {'days': 1, 'description': 'Unknown'}),
            'risk_metrics': risk_metrics
        }

        if verbose:
            print(
                f"  -> [OK] Ensemble Prediction: {action} (Confidence: {confidence:.4f})")
            print(f"  -> Current Price: ₹{current_price:.2f}")
            print(f"  -> Predicted Price: ₹{predicted_price:.2f}")
            print(f"  -> Expected Return: {predicted_return * 100:+.2f}%")
            print(f"  -> Models Used: {', '.join(predictions.keys())}")

        return prediction

    except Exception as e:
        logger.error(
            f"Error predicting stock price for {symbol}: {e}", exc_info=True)
        if verbose:
            print(f"  -> [ERROR] Failed to generate prediction: {e}")
        return None


# For testing purposes
if __name__ == "__main__":
    print("Stock Analysis Complete - ML Engine")
    print("=" * 50)

    # Example usage:
    # ingester = EnhancedDataIngester()
    # data = ingester.fetch_all_data("RELIANCE.NS")
    #
    # engineer = FeatureEngineer()
    # if data and not data['price_history'].empty:
    #     features = engineer.calculate_all_features(data['price_history'], "RELIANCE.NS")
    #     engineer.save_features(features, "RELIANCE.NS")
    #
    # result = train_ml_models("RELIANCE.NS", "intraday")
    # if result:
    #     prediction = predict_stock_price("RELIANCE.NS", "intraday")
    #     print(prediction)


def predict_stock_price(symbol: str, horizon: str = "intraday", verbose: bool = True) -> Optional[Dict]:
    """
    Generate stock price prediction using ensemble of trained models

    Args:
        symbol: Stock symbol to predict
        horizon: Time horizon ("intraday", "short", "long")
        verbose: Whether to print detailed output

    Returns:
        Dict with prediction results
    """
    try:
        if verbose:
            print(
                f"\n[PREDICT] Generating prediction for {symbol} ({horizon})...")

        # Load latest features
        feature_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
        if not feature_path.exists():
            if verbose:
                print(
                    f"  -> [INFO] No features found for {symbol}. Attempting to initialize data...")

            # Dynamically initialize data for the symbol
            success = _initialize_symbol_data(symbol, verbose)
            if not success:
                if verbose:
                    print(
                        f"  -> [ERROR] Failed to initialize data for {symbol}")
                return None

            # Check again if features now exist
            if not feature_path.exists():
                if verbose:
                    print(
                        f"  -> [ERROR] Data initialization did not create features for {symbol}")
                return None

        with open(feature_path, 'r') as f:
            feature_data = json.load(f)

        current_features = feature_data['current_features']
        current_price = feature_data['current_price']

        # Validate that we have meaningful features
        if not current_features or current_price <= 0:
            if verbose:
                print(f"  -> [ERROR] Invalid or empty features for {symbol}")
                print(f"  -> [DIAGNOSTIC] Current price: {current_price}")
                print(
                    f"  -> [DIAGNOSTIC] Features count: {len(current_features) if current_features else 0}")
            return None

        # Get feature columns from one of the models
        model_paths = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*_model.pkl"))
        if not model_paths:
            # Try DQN model
            dqn_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
            if not dqn_path.exists():
                if verbose:
                    print(
                        f"  -> [INFO] No trained models found for {symbol} ({horizon}). Attempting to train models...")

                # Dynamically train models for the symbol
                train_success = _train_symbol_models(symbol, horizon, verbose)
                if not train_success:
                    if verbose:
                        print(
                            f"  -> [ERROR] Failed to train models for {symbol} ({horizon})")
                    return None

                # Check again if models now exist
                model_paths = list(MODEL_DIR.glob(
                    f"{symbol}_{horizon}_*_model.pkl"))
                dqn_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
                if not model_paths and not dqn_path.exists():
                    if verbose:
                        print(
                            f"  -> [ERROR] Model training did not create models for {symbol} ({horizon})")
                    return None
            else:
                # Load DQN model
                # Will be updated when loading
                dqn_agent = DQNTradingAgent(n_features=1)
                dqn_agent.load(symbol, horizon)

                # Prepare state
                feature_columns = list(current_features.keys())
                exclude_columns = ['Date', 'Open', 'High', 'Low',
                                   'Close', 'Volume', 'Dividends', 'Stock Splits']
                feature_columns = [
                    col for col in feature_columns if col not in exclude_columns]

                # Check if we have features to work with
                if not feature_columns:
                    if verbose:
                        print(
                            f"  -> [ERROR] No valid features found for {symbol}")
                    return None

                state_values = [current_features.get(
                    col, 0) for col in feature_columns]
                state = np.array(state_values, dtype=np.float32)

                # Validate state
                if np.all(state == 0) or len(state) == 0:
                    if verbose:
                        print(
                            f"  -> [ERROR] All features are zero or empty for {symbol}")
                    return None

                # Get action from DQN
                action = dqn_agent.get_action(state, training=False)

                # Map action to prediction
                action_map = {0: 'LONG', 1: 'SHORT', 2: 'HOLD'}
                predicted_action = action_map.get(action, 'HOLD')

                # Simple return prediction based on action
                if predicted_action == 'LONG':
                    predicted_return = 0.02  # 2% expected return
                    confidence = 0.8
                elif predicted_action == 'SHORT':
                    predicted_return = -0.01  # -1% expected return
                    confidence = 0.7
                else:  # HOLD
                    predicted_return = 0.005  # 0.5% expected return
                    confidence = 0.6

                predicted_price = current_price * (1 + predicted_return)

                prediction = {
                    'symbol': symbol,
                    'horizon': horizon,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'predicted_return': predicted_return * 100,  # Convert to percentage
                    'action': predicted_action,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'model_predictions': {
                        'dqn_action': predicted_action,
                        'dqn_confidence': confidence
                    },
                    'horizon_details': {
                        'intraday': {'days': 1, 'description': 'Same day / Next day'},
                        'short': {'days': 5, 'description': '1 week (Swing trading)'},
                        'long': {'days': 30, 'description': '1 month (Position trading)'}
                    }.get(horizon, {'days': 1, 'description': 'Unknown'}),
                    'risk_metrics': {
                        'volatility_20': current_features.get('volatility_20', 0),
                        'sharpe_20': current_features.get('sharpe_20', 0),
                        'atr': current_features.get('ATR', 0)
                    }
                }

                if verbose:
                    print(
                        f"  -> [OK] DQN Prediction: {predicted_action} (Confidence: {confidence:.4f})")
                    print(f"  -> Current Price: ₹{current_price:.2f}")
                    print(f"  -> Predicted Price: ₹{predicted_price:.2f}")
                    print(
                        f"  -> Expected Return: {predicted_return * 100:+.2f}%")

                return prediction

        # Load one model to get feature columns
        model_data = joblib.load(model_paths[0])
        feature_columns = model_data['feature_columns']

        # Prepare features for prediction
        feature_values = [current_features.get(
            col, 0) for col in feature_columns]
        X = np.array(feature_values).reshape(1, -1)

        # Validate features
        if np.all(X == 0) or X.shape[1] == 0:
            if verbose:
                print(
                    f"  -> [ERROR] All features are zero or empty for {symbol}")
            return None

        # Scale features
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X)

        # Load all models and get predictions
        predictions = {}

        # Random Forest
        try:
            rf_path = MODEL_DIR / f"{symbol}_{horizon}_rf_model.pkl"
            if rf_path.exists():
                rf_data = joblib.load(rf_path)
                rf_model = rf_data['model']
                rf_pred = rf_model.predict(X_scaled)[0]
                predictions['rf'] = rf_pred
        except Exception as e:
            logger.warning(f"Failed to load RF model for {symbol}: {e}")

        # LightGBM
        try:
            lgb_path = MODEL_DIR / f"{symbol}_{horizon}_lgb_model.pkl"
            if lgb_path.exists():
                lgb_data = joblib.load(lgb_path)
                lgb_model = lgb_data['model']
                lgb_pred = lgb_model.predict(X_scaled)[0]
                predictions['lgb'] = lgb_pred
        except Exception as e:
            logger.warning(f"Failed to load LGB model for {symbol}: {e}")

        # XGBoost
        try:
            xgb_path = MODEL_DIR / f"{symbol}_{horizon}_xgb_model.pkl"
            if xgb_path.exists():
                xgb_data = joblib.load(xgb_path)
                xgb_model = xgb_data['model']
                xgb_pred = xgb_model.predict(X_scaled)[0]
                predictions['xgb'] = xgb_pred
        except Exception as e:
            logger.warning(f"Failed to load XGB model for {symbol}: {e}")

        # DQN
        dqn_confidence = 0.5
        try:
            dqn_path = MODEL_DIR / f"{symbol}_{horizon}_dqn_agent.pt"
            if dqn_path.exists():
                dqn_agent = DQNTradingAgent(n_features=X_scaled.shape[1])
                dqn_agent.load(symbol, horizon)

                # Get action from DQN
                action = dqn_agent.get_action(X_scaled[0], training=False)

                # Map action to return
                action_returns = {0: 0.02, 1: -0.01,
                                  2: 0.005}  # LONG, SHORT, HOLD
                dqn_pred = action_returns.get(action, 0.005)
                predictions['dqn'] = dqn_pred

                # Confidence based on recent performance
                if hasattr(dqn_agent, 'rewards_history') and dqn_agent.rewards_history:
                    recent_rewards = dqn_agent.rewards_history[-20:] if len(
                        dqn_agent.rewards_history) >= 20 else dqn_agent.rewards_history
                    avg_reward = np.mean(recent_rewards)
                    std_reward = np.std(recent_rewards)
                    if std_reward > 0:
                        sharpe = avg_reward / std_reward
                        # Convert Sharpe ratio to confidence (0-1 scale)
                        dqn_confidence = min(1.0, max(0.0, (sharpe + 2) / 4))
        except Exception as e:
            logger.warning(f"Failed to load DQN model for {symbol}: {e}")

        # Ensemble prediction (average of all model predictions)
        if predictions:
            # Weight DQN prediction more heavily (30% weight) since it's reinforcement learning
            weights = {
                'rf': 0.25,
                'lgb': 0.25,
                'xgb': 0.25,
                'dqn': 0.25
            }

            # Only use weights for models that have predictions
            available_models = list(predictions.keys())
            total_weight = sum(weights[model] for model in available_models)

            if total_weight > 0:
                # Normalize weights
                normalized_weights = {
                    model: weights[model] / total_weight for model in available_models}

                # Calculate weighted average
                ensemble_pred = sum(
                    predictions[model] * normalized_weights[model] for model in available_models)
            else:
                ensemble_pred = 0
        else:
            ensemble_pred = 0

        # Determine action based on prediction
        if ensemble_pred > 0.01:  # 1% threshold for LONG
            action = "LONG"
            # Scale confidence based on return
            confidence = min(0.95, 0.7 + (ensemble_pred * 10))
        elif ensemble_pred < -0.01:  # -1% threshold for SHORT
            action = "SHORT"
            confidence = min(0.95, 0.7 + (abs(ensemble_pred) * 10))
        else:
            action = "HOLD"
            confidence = min(0.95, 0.5 + (abs(ensemble_pred) * 50))

        # Apply DQN confidence adjustment
        confidence = confidence * 0.7 + dqn_confidence * 0.3

        # Calculate predicted price
        predicted_return = ensemble_pred
        predicted_price = current_price * (1 + predicted_return)

        # Risk metrics
        risk_metrics = {
            'volatility_20': current_features.get('volatility_20', 0),
            'sharpe_20': current_features.get('sharpe_20', 0),
            'atr': current_features.get('ATR', 0),
            'volume_ratio': current_features.get('volume_ratio', 0)
        }

        prediction = {
            'symbol': symbol,
            'horizon': horizon,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'predicted_return': predicted_return * 100,  # Convert to percentage
            'action': action,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_predictions': predictions,
            'horizon_details': {
                'intraday': {'days': 1, 'description': 'Same day / Next day'},
                'short': {'days': 5, 'description': '1 week (Swing trading)'},
                'long': {'days': 30, 'description': '1 month (Position trading)'}
            }.get(horizon, {'days': 1, 'description': 'Unknown'}),
            'risk_metrics': risk_metrics
        }

        if verbose:
            print(
                f"  -> [OK] Ensemble Prediction: {action} (Confidence: {confidence:.4f})")
            print(f"  -> Current Price: ₹{current_price:.2f}")
            print(f"  -> Predicted Price: ₹{predicted_price:.2f}")
            print(f"  -> Expected Return: {predicted_return * 100:+.2f}%")
            print(f"  -> Models Used: {', '.join(predictions.keys())}")

        return prediction

    except Exception as e:
        logger.error(
            f"Error predicting stock price for {symbol}: {e}", exc_info=True)
        if verbose:
            print(f"  -> [ERROR] Failed to generate prediction: {e}")
            import traceback
            print(f"  -> [TRACEBACK] {traceback.format_exc()}")
        return None


def _initialize_symbol_data(symbol: str, verbose: bool = True) -> bool:
    """
    Dynamically initialize data for a symbol that doesn't have cached data

    Args:
        symbol: Stock symbol to initialize
        verbose: Whether to print detailed output

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"  -> [INIT] Initializing data for {symbol}...")

        # Initialize components
        ingester = EnhancedDataIngester()
        engineer = FeatureEngineer()

        # Fetch data
        data = ingester.fetch_all_data(symbol, period="2y")
        if not data:
            if verbose:
                print(f"  -> [ERROR] Failed to fetch data for {symbol}")
            return False

        if verbose:
            print(f"  -> [OK] Data fetched for {symbol}")

        # Save data
        ingester.save_all_data(data, symbol)
        if verbose:
            print(f"  -> [OK] Data saved for {symbol}")

        # Calculate features if we have price history
        if data and 'price_history' in data and data['price_history'] is not None:
            if verbose:
                print(f"  -> [INFO] Calculating features for {symbol}...")
            features_df = engineer.calculate_all_features(
                data['price_history'], symbol)
            if features_df is not None and not features_df.empty:
                engineer.save_features(features_df, symbol)
                if verbose:
                    print(
                        f"  -> [OK] Features calculated and saved for {symbol}")
                    print(
                        f"  -> [INFO] Feature count: {len(features_df.columns)}")
                    print(f"  -> [INFO] Data points: {len(features_df)}")
                return True
            else:
                if verbose:
                    print(
                        f"  -> [ERROR] Failed to calculate features for {symbol}")
                return False
        else:
            if verbose:
                print(
                    f"  -> [ERROR] No price history data available for {symbol}")
            return False

    except Exception as e:
        logger.error(
            f"Error initializing data for {symbol}: {e}", exc_info=True)
        if verbose:
            print(
                f"  -> [ERROR] Exception during data initialization for {symbol}: {e}")
        return False


def _train_symbol_models(symbol: str, horizon: str, verbose: bool = True) -> bool:
    """
    Dynamically train models for a symbol that doesn't have trained models

    Args:
        symbol: Stock symbol to train models for
        horizon: Time horizon ("intraday", "short", "long")
        verbose: Whether to print detailed output

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"  -> [TRAIN] Training {horizon} models for {symbol}...")

        # Train models
        result = train_ml_models(symbol, horizon, verbose=verbose)

        if result:
            if verbose:
                print(
                    f"  -> [OK] {horizon} models trained successfully for {symbol}")
            return True
        else:
            if verbose:
                print(
                    f"  -> [ERROR] Failed to train {horizon} models for {symbol}")
            return False

    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}", exc_info=True)
        if verbose:
            print(
                f"  -> [ERROR] Exception during model training for {symbol}: {e}")
        return False


# For testing purposes
if __name__ == "__main__":
    print("Stock Analysis Complete - ML Engine")
    print("=" * 50)

    # Example usage:
    # ingester = EnhancedDataIngester()
    # data = ingester.fetch_all_data("RELIANCE.NS")
    #
    # engineer = FeatureEngineer()
    # if data and not data['price_history'].empty:
    #     features = engineer.calculate_all_features(data['price_history'], "RELIANCE.NS")
    #     engineer.save_features(features, "RELIANCE.NS")
    #
    # result = train_ml_models("RELIANCE.NS", "intraday")
    # if result:
    #     prediction = predict_stock_price("RELIANCE.NS", "intraday")
    #     print(prediction)
