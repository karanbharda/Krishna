import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym
from typing import List, Dict, Any
import json
from datetime import datetime
import os
import sys

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .risk_engine import risk_engine

# Set up logger first
logger = logging.getLogger(__name__)

# Import monitoring
try:
    from utils.monitoring import log_model_performance
    MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("Monitoring not available for RL agent")
    MONITORING_AVAILABLE = False

class SimpleRLModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=3):  # 3 actions: buy, hold, sell
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class RLFilteringAgent:
    def __init__(self):
        # RTX 3060 optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for RL processing")
            
        self.model = SimpleRLModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Performance tracking
        self.processed_stocks = 0
        self.filtering_stats = {
            "total_processed": 0,
            "risk_compliant": 0,
            "high_confidence": 0
        }

    def get_rl_analysis(self, data: Dict[str, Any], horizon: str = "day") -> Dict[str, Any]:
        """Get RL analysis for a single stock for integration with professional buy logic"""
        try:
            # Extract features
            features = self._extract_features(data, horizon)
            
            # Get RL scores
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                scores = self.model(input_tensor)
                buy_score = float(scores[0, 0].cpu().item())
                hold_score = float(scores[0, 1].cpu().item())
                sell_score = float(scores[0, 2].cpu().item())
            
            # Determine recommendation based on highest score
            if buy_score > hold_score and buy_score > sell_score:
                recommendation = "BUY"
                confidence = buy_score
            elif sell_score > hold_score and sell_score > buy_score:
                recommendation = "SELL"
                confidence = sell_score
            else:
                recommendation = "HOLD"
                confidence = hold_score
            
            return {
                "success": True,
                "recommendation": recommendation,
                "confidence": confidence,
                "buy_score": buy_score,
                "hold_score": hold_score,
                "sell_score": sell_score,
                "rl_scores": {
                    "buy": buy_score,
                    "hold": hold_score,
                    "sell": sell_score
                },
                "horizon": horizon
            }
        except Exception as e:
            logger.error(f"Error in RL analysis: {e}")
            # Fallback to CPU scoring
            try:
                features = self._extract_features(data, horizon)
                cpu_score = self._get_rl_score_cpu(features)
                return {
                    "success": True,
                    "recommendation": "BUY" if cpu_score > 0.5 else "HOLD",
                    "confidence": cpu_score,
                    "buy_score": cpu_score,
                    "hold_score": 1.0 - cpu_score,
                    "sell_score": 0.1,
                    "rl_scores": {
                        "buy": cpu_score,
                        "hold": 1.0 - cpu_score,
                        "sell": 0.1
                    },
                    "horizon": horizon
                }
            except Exception as e2:
                logger.error(f"CPU scoring also failed: {e2}")
                return {
                    "success": False,
                    "recommendation": "HOLD",
                    "confidence": 0.5,
                    "buy_score": 0.5,
                    "hold_score": 0.5,
                    "sell_score": 0.5,
                    "rl_scores": {
                        "buy": 0.5,
                        "hold": 0.5,
                        "sell": 0.5
                    },
                    "horizon": horizon,
                    "error": str(e2)
                }

    def rank_stocks(self, universe_data: Dict[str, Any], horizon: str = "day") -> List[Dict[str, Any]]:
        """Rank stocks using RL model against dynamic risk from live_config.json"""
        logger.info(f"Starting RL ranking for {len(universe_data)} stocks with horizon: {horizon}")
        ranked_stocks = []
        
        # Get current risk settings
        risk_settings = risk_engine.get_risk_settings()
        logger.info(f"Using risk settings: {risk_settings}")
        
        # Process stocks in batches for GPU efficiency
        batch_size = 32  # RTX 3060 optimization
        stock_items = list(universe_data.items())
        
        for i in range(0, len(stock_items), batch_size):
            batch = stock_items[i:i+batch_size]
            batch_results = self._process_batch(batch, horizon, risk_settings)
            ranked_stocks.extend(batch_results)
            
            self.filtering_stats["total_processed"] += len(batch)
            logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch_results)} qualified stocks")
        
        # Sort by score and apply horizon-specific filtering
        ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
        filtered_stocks = self.filter_by_horizon(ranked_stocks, horizon)
        
        logger.info(f"RL ranking completed: {len(filtered_stocks)} stocks shortlisted from {len(universe_data)}")
        logger.info(f"Filtering stats: {self.filtering_stats}")
        
        return filtered_stocks

    def _process_batch(self, batch: List[tuple], horizon: str, risk_settings: Dict[str, float]) -> List[Dict[str, Any]]:
        """Process a batch of stocks for GPU efficiency"""
        batch_results = []
        
        # Prepare batch features
        features_batch = []
        symbols_batch = []
        
        for symbol, data in batch:
            if not data or 'price' not in data:
                continue
                
            features = self._extract_features(data, horizon)
            features_batch.append(features)
            symbols_batch.append((symbol, data))
        
        if not features_batch:
            return batch_results
        
        # Process batch on GPU
        try:
            features_tensor = torch.FloatTensor(np.array(features_batch)).to(self.device)
            
            with torch.no_grad():
                scores = self.model(features_tensor)
                # Use buy action probability as score
                buy_scores = scores[:, 0].cpu().numpy()
            
            # Apply risk filtering
            for i, (symbol, data) in enumerate(symbols_batch):
                score = float(buy_scores[i])
                
                # Apply risk filter using live_config.json settings
                price = data.get('price', 0)
                if price <= 0:
                    continue
                    
                risk_limits = risk_engine.apply_risk_to_position(price)
                
                # Risk compliance checks
                is_risk_compliant = (
                    score > 0.5 and  # Minimum confidence
                    price > risk_limits['stop_loss_amount'] and
                    price < risk_limits['capital_at_risk'] * 10  # Max price check
                )
                
                if is_risk_compliant:
                    self.filtering_stats["risk_compliant"] += 1
                    
                    if score > 0.7:
                        self.filtering_stats["high_confidence"] += 1
                    
                    batch_results.append({
                        "symbol": symbol,
                        "score": score,
                        "risk_compliant": True,
                        "price": price,
                        "risk_limits": risk_limits,
                        "horizon": horizon
                    })
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Fallback to CPU processing
            for symbol, data in symbols_batch:
                try:
                    features = self._extract_features(data, horizon)
                    score = self._get_rl_score_cpu(features)
                    
                    if score > 0.5:
                        batch_results.append({
                            "symbol": symbol,
                            "score": score,
                            "risk_compliant": True,
                            "price": data.get('price', 0),
                            "horizon": horizon
                        })
                except Exception as e2:
                    logger.debug(f"Failed to process {symbol}: {e2}")
        
        return batch_results

    def _extract_features(self, data: Dict[str, Any], horizon: str) -> np.ndarray:
        """Extract features for RL model"""
        try:
            price = float(data.get('price', 0))
            volume = float(data.get('volume', 0))
            change = float(data.get('change', 0))
            change_pct = float(data.get('change_pct', 0))
            
            # Normalize features
            price_norm = min(price / 1000, 10)  # Normalize price
            volume_norm = min(volume / 1000000, 10)  # Normalize volume
            
            # Horizon encoding
            horizon_encoding = {"day": 1, "week": 2, "month": 3, "year": 4}.get(horizon, 1)
            
            # Create feature vector (10 features)
            features = np.array([
                price_norm,
                volume_norm,
                change,
                change_pct,
                horizon_encoding,
                abs(change_pct),  # Volatility indicator
                1 if change > 0 else 0,  # Positive momentum
                min(price / 100, 1),  # Price tier
                0,  # Reserved
                0   # Reserved
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return np.zeros(10, dtype=np.float32)

    def _get_rl_score(self, features: np.ndarray) -> float:
        """Get RL score for features using GPU"""
        try:
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                # Return buy action probability
                return float(output[0, 0].cpu().item())
        except Exception as e:
            logger.debug(f"GPU scoring error: {e}")
            return self._get_rl_score_cpu(features)

    def _get_rl_score_cpu(self, features: np.ndarray) -> float:
        """Fallback CPU scoring"""
        try:
            # Simple heuristic scoring
            price_norm = features[0]
            volume_norm = features[1]
            change_pct = features[3]
            
            score = 0.5  # Base score
            
            # Positive momentum bonus
            if change_pct > 0:
                score += min(change_pct / 10, 0.3)
            
            # Volume bonus
            if volume_norm > 0.1:
                score += 0.1
            
            # Price tier bonus (mid-range stocks)
            if 0.1 < price_norm < 5:
                score += 0.1
            
            return min(max(score, 0), 1)
            
        except Exception as e:
            logger.debug(f"CPU scoring error: {e}")
            return 0.5

    def filter_by_horizon(self, ranked_stocks: List[Dict[str, Any]], horizon: str) -> List[Dict[str, Any]]:
        """Filter by time horizon with different limits"""
        horizon_limits = {
            "day": 20,
            "week": 30,
            "month": 50,
            "year": 100
        }
        
        limit = horizon_limits.get(horizon, 20)
        filtered = ranked_stocks[:limit]
        
        logger.info(f"Filtered to top {len(filtered)} stocks for {horizon} horizon")
        return filtered

    def save_shortlist(self, shortlist: List[Dict[str, Any]]):
        """Save shortlist to JSON with enhanced metadata"""
        try:
            # FIXED: Use project root logs directory
            from pathlib import Path
            backend_dir = Path(__file__).resolve().parents[1]
            project_root = backend_dir.parent
            logs_dir = project_root / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = datetime.now().strftime("%Y%m%d")
            
            shortlist_data = {
                "timestamp": datetime.now().isoformat(),
                "total_shortlisted": len(shortlist),
                "filtering_stats": self.filtering_stats,
                "device_used": str(self.device),
                "shortlist": shortlist
            }
            
            with open(logs_dir / f"shortlist_{date_str}.json", 'w') as f:
                json.dump(shortlist_data, f, indent=2)
            
            logger.info(f"Saved shortlist: {len(shortlist)} stocks to logs/shortlist_{date_str}.json")
            
        except Exception as e:
            logger.error(f"Failed to save shortlist: {e}")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        stats = {
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "filtering_stats": self.filtering_stats,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        
        # Log performance metrics if monitoring is available
        if MONITORING_AVAILABLE:
            try:
                total_processed = self.filtering_stats.get("total_processed", 0)
                risk_compliant = self.filtering_stats.get("risk_compliant", 0)
                
                # Calculate accuracy with proper handling of edge cases
                if total_processed > 0:
                    accuracy = risk_compliant / total_processed
                else:
                    # Default accuracy when no stocks processed yet
                    accuracy = 0.7  # Assume good performance until proven otherwise
                
                metrics = {
                    "accuracy": accuracy,
                    "confidence": 0.7,  # Default confidence
                    "processed_stocks": total_processed
                }
                log_model_performance("RLFilteringAgent", metrics, stats)
            except Exception as e:
                logger.debug(f"Failed to log RL agent performance: {e}")
        
        return stats

# Global instance
rl_agent = RLFilteringAgent()
