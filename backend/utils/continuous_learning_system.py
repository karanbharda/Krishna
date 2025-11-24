"""
Phase 2: Continuous Learning System
Implements online model updating and performance feedback for adaptive trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    prediction_count: int
    last_updated: str


@dataclass
class TradeOutcome:
    """Track trade outcomes for learning"""
    symbol: str
    prediction: float
    actual_outcome: float
    features: List[float]
    timestamp: str
    profit_loss: float
    hold_duration: int


class ContinuousLearningSystem:
    """
    Implements online learning with performance feedback and model adaptation
    """
    
    def __init__(self, model_save_path: str = "models/"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.trade_outcomes = deque(maxlen=1000)  # Keep last 1000 trades
        
        # Learning parameters
        self.min_samples_for_update = 10
        self.performance_window = 50
        self.model_retrain_threshold = 0.1  # Retrain if performance drops 10%
        self.learning_rate = 0.01
        
        # Model ensemble
        self.ensemble_weights = {}
        self.model_types = ['rf', 'gb', 'sgd']  # Random Forest, Gradient Boosting, SGD
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.adaptation_history = []
        
        self._initialize_models()
        logger.info("✅ Continuous Learning System initialized")
    
    def _initialize_models(self):
        """Initialize the ensemble of models"""
        try:
            # Random Forest - good for feature importance and non-linear patterns
            self.models['rf'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting - good for sequential learning
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # SGD Regressor - good for online learning
            self.models['sgd'] = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
            
            # Initialize scalers
            for model_type in self.model_types:
                self.scalers[model_type] = StandardScaler()
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)
            
            logger.info(f"Initialized {len(self.models)} models in ensemble")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def train_initial_models(self, training_data: pd.DataFrame, target_column: str):
        """Train initial models with historical data"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return False
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column].fillna(0)
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_type].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_type].transform(X_val)
                    
                    # Train model
                    self.models[model_type].fit(X_train_scaled, y_train)
                    
                    # Evaluate on validation set
                    y_pred = self.models[model_type].predict(X_val_scaled)
                    performance = self._calculate_performance_metrics(y_val, y_pred)
                    
                    self.baseline_performance[model_type] = performance
                    self.current_performance[model_type] = performance
                    
                    logger.info(f"Model {model_type} trained - R²: {performance.r2_score:.3f}, MSE: {performance.mse:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue
            
            # Save initial models
            self._save_models()
            
            logger.info("✅ Initial model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in initial training: {e}")
            return False
    
    def predict_ensemble(self, features: List[float]) -> Dict:
        """Make prediction using ensemble of models"""
        try:
            features_array = np.array(features).reshape(1, -1)
            predictions = {}
            weights = []
            valid_predictions = []
            
            # Get predictions from each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    features_scaled = self.scalers[model_type].transform(features_array)
                    
                    # Make prediction
                    pred = self.models[model_type].predict(features_scaled)[0]
                    predictions[model_type] = pred
                    
                    # Weight by current performance
                    weight = self.ensemble_weights[model_type]
                    weights.append(weight)
                    valid_predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Error predicting with {model_type}: {e}")
                    continue
            
            if not valid_predictions:
                logger.error("No valid predictions from ensemble")
                return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
            
            # Calculate weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(valid_predictions, weights=weights)
            
            # Calculate confidence based on agreement between models
            pred_std = np.std(valid_predictions)
            confidence = max(0.0, 1.0 - pred_std)  # Higher agreement = higher confidence
            
            return {
                'ensemble_prediction': float(ensemble_pred),
                'model_predictions': predictions,
                'confidence': float(confidence),
                'prediction_count': len(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
    
    def update_with_outcome(self, trade_outcome: TradeOutcome):
        """Update models with actual trade outcome"""
        try:
            # Store trade outcome
            self.trade_outcomes.append(trade_outcome)
            
            # Check if we have enough samples for updating
            if len(self.trade_outcomes) < self.min_samples_for_update:
                return
            
            # Prepare recent training data
            recent_outcomes = list(self.trade_outcomes)[-self.performance_window:]
            
            X_new = np.array([outcome.features for outcome in recent_outcomes])
            y_new = np.array([outcome.actual_outcome for outcome in recent_outcomes])
            
            # Online learning update for SGD model
            try:\n                X_new_scaled = self.scalers['sgd'].transform(X_new)\n                self.models['sgd'].partial_fit(X_new_scaled, y_new)\n                \n                # Update performance\n                y_pred = self.models['sgd'].predict(X_new_scaled)\n                performance = self._calculate_performance_metrics(y_new, y_pred)\n                self.current_performance['sgd'] = performance\n                \n            except Exception as e:\n                logger.warning(f\"Error updating SGD model: {e}\")\n            \n            # Check if models need retraining\n            self._check_model_performance()\n            \n            # Update ensemble weights based on recent performance\n            self._update_ensemble_weights()\n            \n            logger.debug(f\"Updated models with trade outcome: {trade_outcome.symbol}\")\n            \n        except Exception as e:\n            logger.error(f\"Error updating with outcome: {e}\")\n    \n    def _calculate_performance_metrics(self, y_true, y_pred) -> ModelPerformance:\n        \"\"\"Calculate comprehensive performance metrics\"\"\"\n        try:\n            mse = mean_squared_error(y_true, y_pred)\n            mae = mean_absolute_error(y_true, y_pred)\n            r2 = r2_score(y_true, y_pred)\n            \n            # For classification metrics, convert to binary\n            y_true_binary = (y_true > 0).astype(int)\n            y_pred_binary = (y_pred > 0).astype(int)\n            \n            # Calculate precision, recall, f1\n            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))\n            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))\n            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))\n            \n            precision = tp / (tp + fp) if (tp + fp) > 0 else 0\n            recall = tp / (tp + fn) if (tp + fn) > 0 else 0\n            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0\n            accuracy = np.mean(y_true_binary == y_pred_binary)\n            \n            return ModelPerformance(\n                accuracy=accuracy,\n                precision=precision,\n                recall=recall,\n                f1_score=f1,\n                mse=mse,\n                mae=mae,\n                r2_score=r2,\n                prediction_count=len(y_true),\n                last_updated=datetime.now().isoformat()\n            )\n            \n        except Exception as e:\n            logger.error(f\"Error calculating performance metrics: {e}\")\n            return ModelPerformance(0, 0, 0, 0, float('inf'), float('inf'), -1, 0, datetime.now().isoformat())\n    \n    def _check_model_performance(self):\n        \"\"\"Check if models need retraining due to performance degradation\"\"\"\n        try:\n            for model_type in self.model_types:\n                if model_type not in self.baseline_performance or model_type not in self.current_performance:\n                    continue\n                \n                baseline_r2 = self.baseline_performance[model_type].r2_score\n                current_r2 = self.current_performance[model_type].r2_score\n                \n                # Check for significant performance drop\n                if baseline_r2 > 0 and (baseline_r2 - current_r2) > self.model_retrain_threshold:\n                    logger.warning(f\"Model {model_type} performance degraded: {baseline_r2:.3f} → {current_r2:.3f}\")\n                    self._schedule_model_retrain(model_type)\n                    \n        except Exception as e:\n            logger.error(f\"Error checking model performance: {e}\")\n    \n    def _schedule_model_retrain(self, model_type: str):\n        \"\"\"Schedule model retraining (simplified implementation)\"\"\"\n        try:\n            # In a full implementation, this would trigger async retraining\n            logger.info(f\"Scheduling retrain for {model_type} model\")\n            \n            # For now, just reset the baseline to current performance\n            # to avoid repeated retraining notifications\n            if model_type in self.current_performance:\n                self.baseline_performance[model_type] = self.current_performance[model_type]\n                \n        except Exception as e:\n            logger.error(f\"Error scheduling retrain: {e}\")\n    \n    def _update_ensemble_weights(self):\n        \"\"\"Update ensemble weights based on recent performance\"\"\"\n        try:\n            total_performance = 0\n            model_scores = {}\n            \n            # Calculate performance scores (higher is better)\n            for model_type in self.model_types:\n                if model_type in self.current_performance:\n                    # Use combination of metrics for scoring\n                    perf = self.current_performance[model_type]\n                    score = (perf.accuracy + perf.f1_score + max(0, perf.r2_score)) / 3\n                    model_scores[model_type] = max(0.01, score)  # Minimum weight\n                    total_performance += model_scores[model_type]\n                else:\n                    model_scores[model_type] = 0.01\n            \n            # Normalize weights\n            if total_performance > 0:\n                for model_type in self.model_types:\n                    self.ensemble_weights[model_type] = model_scores[model_type] / total_performance\n            \n            logger.debug(f\"Updated ensemble weights: {self.ensemble_weights}\")\n            \n        except Exception as e:\n            logger.error(f\"Error updating ensemble weights: {e}\")\n    \n    def get_learning_insights(self) -> Dict:\n        \"\"\"Get insights about the learning system's performance\"\"\"\n        try:\n            insights = {\n                'total_trades_learned': len(self.trade_outcomes),\n                'models_performance': {},\n                'ensemble_weights': self.ensemble_weights.copy(),\n                'recent_accuracy': self._calculate_recent_accuracy(),\n                'learning_trend': self._analyze_learning_trend(),\n                'feature_importance': self._get_feature_importance()\n            }\n            \n            # Add performance metrics for each model\n            for model_type in self.model_types:\n                if model_type in self.current_performance:\n                    insights['models_performance'][model_type] = asdict(self.current_performance[model_type])\n            \n            return insights\n            \n        except Exception as e:\n            logger.error(f\"Error getting learning insights: {e}\")\n            return {}\n    \n    def _calculate_recent_accuracy(self) -> float:\n        \"\"\"Calculate accuracy on recent predictions\"\"\"\n        try:\n            if len(self.trade_outcomes) < 10:\n                return 0.0\n            \n            recent_outcomes = list(self.trade_outcomes)[-20:]\n            correct_predictions = 0\n            \n            for outcome in recent_outcomes:\n                predicted_direction = 1 if outcome.prediction > 0 else -1\n                actual_direction = 1 if outcome.actual_outcome > 0 else -1\n                \n                if predicted_direction == actual_direction:\n                    correct_predictions += 1\n            \n            return correct_predictions / len(recent_outcomes)\n            \n        except Exception as e:\n            logger.error(f\"Error calculating recent accuracy: {e}\")\n            return 0.0\n    \n    def _analyze_learning_trend(self) -> str:\n        \"\"\"Analyze if the model is improving over time\"\"\"\n        try:\n            if len(self.trade_outcomes) < 20:\n                return \"Insufficient data\"\n            \n            # Compare first half vs second half accuracy\n            all_outcomes = list(self.trade_outcomes)\n            mid_point = len(all_outcomes) // 2\n            \n            first_half = all_outcomes[:mid_point]\n            second_half = all_outcomes[mid_point:]\n            \n            first_accuracy = self._calculate_accuracy_for_outcomes(first_half)\n            second_accuracy = self._calculate_accuracy_for_outcomes(second_half)\n            \n            if second_accuracy > first_accuracy + 0.05:\n                return \"Improving\"\n            elif second_accuracy < first_accuracy - 0.05:\n                return \"Declining\"\n            else:\n                return \"Stable\"\n                \n        except Exception as e:\n            logger.error(f\"Error analyzing learning trend: {e}\")\n            return \"Unknown\"\n    \n    def _calculate_accuracy_for_outcomes(self, outcomes: List[TradeOutcome]) -> float:\n        \"\"\"Calculate accuracy for a list of trade outcomes\"\"\"\n        if not outcomes:\n            return 0.0\n        \n        correct = 0\n        for outcome in outcomes:\n            predicted_direction = 1 if outcome.prediction > 0 else -1\n            actual_direction = 1 if outcome.actual_outcome > 0 else -1\n            if predicted_direction == actual_direction:\n                correct += 1\n        \n        return correct / len(outcomes)\n    \n    def _get_feature_importance(self) -> Dict:\n        \"\"\"Get feature importance from tree-based models\"\"\"\n        try:\n            importance_dict = {}\n            \n            # Get importance from Random Forest\n            if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):\n                rf_importance = self.models['rf'].feature_importances_\n                importance_dict['random_forest'] = rf_importance.tolist()\n            \n            # Get importance from Gradient Boosting\n            if 'gb' in self.models and hasattr(self.models['gb'], 'feature_importances_'):\n                gb_importance = self.models['gb'].feature_importances_\n                importance_dict['gradient_boosting'] = gb_importance.tolist()\n            \n            return importance_dict\n            \n        except Exception as e:\n            logger.error(f\"Error getting feature importance: {e}\")\n            return {}\n    \n    def _save_models(self):\n        \"\"\"Save models and scalers to disk\"\"\"\n        try:\n            import os\n            os.makedirs(self.model_save_path, exist_ok=True)\n            \n            # Save models\n            for model_type, model in self.models.items():\n                model_path = f\"{self.model_save_path}/{model_type}_model.joblib\"\n                joblib.dump(model, model_path)\n            \n            # Save scalers\n            for scaler_type, scaler in self.scalers.items():\n                scaler_path = f\"{self.model_save_path}/{scaler_type}_scaler.joblib\"\n                joblib.dump(scaler, scaler_path)\n            \n            # Save performance history\n            performance_data = {\n                'baseline_performance': {k: asdict(v) for k, v in self.baseline_performance.items()},\n                'current_performance': {k: asdict(v) for k, v in self.current_performance.items()},\n                'ensemble_weights': self.ensemble_weights\n            }\n            \n            with open(f\"{self.model_save_path}/performance_history.json\", 'w') as f:\n                json.dump(performance_data, f, indent=2)\n            \n            logger.info(\"Models and performance data saved\")\n            \n        except Exception as e:\n            logger.error(f\"Error saving models: {e}\")\n    \n    def load_models(self):\n        \"\"\"Load models and scalers from disk\"\"\"\n        try:\n            import os\n            \n            # Load models\n            for model_type in self.model_types:\n                model_path = f\"{self.model_save_path}/{model_type}_model.joblib\"\n                if os.path.exists(model_path):\n                    self.models[model_type] = joblib.load(model_path)\n            \n            # Load scalers\n            for scaler_type in self.model_types:\n                scaler_path = f\"{self.model_save_path}/{scaler_type}_scaler.joblib\"\n                if os.path.exists(scaler_path):\n                    self.scalers[scaler_type] = joblib.load(scaler_path)\n            \n            # Load performance history\n            performance_path = f\"{self.model_save_path}/performance_history.json\"\n            if os.path.exists(performance_path):\n                with open(performance_path, 'r') as f:\n                    performance_data = json.load(f)\n                \n                # Reconstruct performance objects\n                for model_type, perf_dict in performance_data.get('baseline_performance', {}).items():\n                    self.baseline_performance[model_type] = ModelPerformance(**perf_dict)\n                \n                for model_type, perf_dict in performance_data.get('current_performance', {}).items():\n                    self.current_performance[model_type] = ModelPerformance(**perf_dict)\n                \n                self.ensemble_weights = performance_data.get('ensemble_weights', self.ensemble_weights)\n            \n            logger.info(\"Models and performance data loaded\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error loading models: {e}\")\n            return False\n    \n    def reset_learning(self):\n        \"\"\"Reset the learning system\"\"\"\n        try:\n            self.trade_outcomes.clear()\n            self.current_performance.clear()\n            self.baseline_performance.clear()\n            \n            # Reset ensemble weights to equal\n            for model_type in self.model_types:\n                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)\n            \n            logger.info(\"Learning system reset\")\n            \n        except Exception as e:\n            logger.error(f\"Error resetting learning system: {e}\")\n\n\n# Global instance\n_learning_system = None\n\ndef get_learning_system() -> ContinuousLearningSystem:\n    \"\"\"Get the global continuous learning system instance\"\"\"\n    global _learning_system\n    if _learning_system is None:\n        _learning_system = ContinuousLearningSystem()\n    return _learning_system"