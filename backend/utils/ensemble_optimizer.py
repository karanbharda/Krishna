"""
Phase 3: Ensemble Model Optimization System
Implements advanced model combination strategies for improved prediction accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Set up logger first
logger = logging.getLogger(__name__)

# Import monitoring
try:
    from utils.monitoring import log_model_performance
    MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("Monitoring not available for Ensemble Optimizer")
    MONITORING_AVAILABLE = False

class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"
    ADAPTIVE = "adaptive"
    CONFIDENCE_BASED = "confidence_based"


class ModelPerformanceTracker:
    """Track individual model performance"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def add_prediction(self, prediction: float, actual: float, timestamp: str):
        """Add prediction result"""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)
        
        # Keep only recent results
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.actuals = self.actuals[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.predictions) < 5:
            return {'mse': float('inf'), 'r2': -1, 'accuracy': 0.0, 'count': 0}
        
        mse = mean_squared_error(self.actuals, self.predictions)
        r2 = r2_score(self.actuals, self.predictions)
        
        # Direction accuracy
        actual_directions = [1 if a > 0 else -1 for a in self.actuals]
        pred_directions = [1 if p > 0 else -1 for p in self.predictions]
        accuracy = sum(a == p for a, p in zip(actual_directions, pred_directions)) / len(actual_directions)
        
        return {
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'count': len(self.predictions)
        }


class EnsembleOptimizer:
    """
    Optimizes ensemble model combinations for maximum prediction accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_trackers = {}
        self.ensemble_method = EnsembleMethod.ADAPTIVE
        
        # Performance tracking
        self.ensemble_history = []
        self.optimization_history = []
        
        # Optimization parameters
        self.min_performance_samples = 20
        self.weight_adjustment_rate = 0.1
        self.performance_decay = 0.95  # Decay factor for older performance
        
        logger.info("Ensemble Optimizer initialized")
    
    def register_model(self, model_name: str, model: Any, initial_weight: float = 1.0):
        """Register a model in the ensemble"""
        try:
            self.models[model_name] = model
            self.model_weights[model_name] = initial_weight
            self.performance_trackers[model_name] = ModelPerformanceTracker()
            
            logger.info(f"Model '{model_name}' registered with weight {initial_weight}")
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
    
    def predict_ensemble(self, features: np.ndarray, method: Optional[EnsembleMethod] = None) -> Dict:
        """Make ensemble prediction using specified method"""
        try:
            if not self.models:
                return {'prediction': 0.0, 'confidence': 0.0, 'model_predictions': {}}
            
            method = method or self.ensemble_method
            
            # Get predictions from all models
            model_predictions = {}
            valid_predictions = []
            valid_weights = []
            
            for model_name, model in self.models.items():
                try:
                    prediction = model.predict(features.reshape(1, -1))[0]
                    model_predictions[model_name] = prediction
                    
                    weight = self.model_weights.get(model_name, 1.0)
                    valid_predictions.append(prediction)
                    valid_weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if not valid_predictions:
                return {'prediction': 0.0, 'confidence': 0.0, 'model_predictions': {}}
            
            # Combine predictions based on method
            ensemble_pred, confidence = self._combine_predictions(
                valid_predictions, valid_weights, method
            )
            
            return {
                'prediction': ensemble_pred,
                'confidence': confidence,
                'model_predictions': model_predictions,
                'method_used': method.value,
                'num_models': len(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'model_predictions': {}}
    
    def _combine_predictions(self, predictions: List[float], 
                           weights: List[float], 
                           method: EnsembleMethod) -> Tuple[float, float]:
        """Combine predictions using specified method"""
        try:
            predictions = np.array(predictions)
            weights = np.array(weights)
            
            if method == EnsembleMethod.WEIGHTED_AVERAGE:
                # Normalize weights
                weights = weights / weights.sum()
                ensemble_pred = np.average(predictions, weights=weights)
                
                # Confidence based on weight concentration
                weight_entropy = -np.sum(weights * np.log(weights + 1e-10))
                confidence = 1.0 / (1.0 + weight_entropy)
                
            elif method == EnsembleMethod.VOTING:
                # Simple majority voting for direction
                directions = np.sign(predictions)
                ensemble_direction = np.sign(np.sum(directions))
                ensemble_pred = ensemble_direction * np.mean(np.abs(predictions))
                
                # Confidence based on consensus
                consensus = np.abs(np.sum(directions)) / len(directions)
                confidence = consensus
                
            elif method == EnsembleMethod.CONFIDENCE_BASED:
                # Weight by model confidence (based on recent performance)
                performance_weights = self._get_performance_weights()
                adjusted_weights = []
                
                for i, model_name in enumerate(self.models.keys()):
                    if i < len(weights):
                        perf_weight = performance_weights.get(model_name, 1.0)
                        adjusted_weights.append(weights[i] * perf_weight)
                
                if adjusted_weights:
                    adjusted_weights = np.array(adjusted_weights)
                    adjusted_weights = adjusted_weights / adjusted_weights.sum()
                    ensemble_pred = np.average(predictions, weights=adjusted_weights)
                    
                    # Confidence based on performance consistency
                    confidence = np.mean(list(performance_weights.values()))
                else:
                    ensemble_pred = np.mean(predictions)
                    confidence = 0.5
                    
            elif method == EnsembleMethod.ADAPTIVE:
                # Adaptive combination based on recent performance and market conditions
                ensemble_pred, confidence = self._adaptive_combination(predictions, weights)
                
            elif method == EnsembleMethod.STACKING:
                # Stacking with meta-learner
                ensemble_pred, confidence = self._stacking_combination(predictions, weights)
                
            else:  # Default to simple average
                ensemble_pred = np.mean(predictions)
                confidence = 1.0 / (1.0 + np.std(predictions))
            
            return float(ensemble_pred), float(confidence)
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return float(np.mean(predictions)), 0.5
    
    def _get_performance_weights(self) -> Dict[str, float]:
        """Get performance-based weights for models"""
        try:
            performance_weights = {}
            
            for model_name, tracker in self.performance_trackers.items():
                metrics = tracker.get_performance_metrics()
                
                # Combine multiple performance metrics
                accuracy = metrics.get('accuracy', 0)
                r2 = max(0, metrics.get('r2', 0))  # Ensure non-negative
                mse = metrics.get('mse', float('inf'))
                
                # Calculate composite performance score
                if mse == float('inf') or mse == 0:
                    performance_score = accuracy
                else:
                    # Normalize MSE component
                    mse_score = 1.0 / (1.0 + mse)
                    performance_score = 0.4 * accuracy + 0.3 * r2 + 0.3 * mse_score
                
                performance_weights[model_name] = max(0.1, performance_score)  # Minimum weight
            
            return performance_weights
            
        except Exception as e:
            logger.error(f"Error calculating performance weights: {e}")
            return {name: 1.0 for name in self.models.keys()}
    
    def _adaptive_combination(self, predictions: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """Adaptive prediction combination"""
        try:
            # Get recent performance metrics
            performance_weights = self._get_performance_weights()
            
            # Analyze prediction agreement
            prediction_std = np.std(predictions)
            agreement_factor = 1.0 / (1.0 + prediction_std)
            
            # Adjust weights based on agreement and performance
            adjusted_weights = []
            model_names = list(self.models.keys())
            
            for i, weight in enumerate(weights):
                if i < len(model_names):
                    model_name = model_names[i]
                    perf_weight = performance_weights.get(model_name, 1.0)
                    
                    # Boost weight for high-performing models when there's disagreement
                    if agreement_factor < 0.7:  # Low agreement
                        adjusted_weight = weight * perf_weight * 1.5
                    else:  # High agreement
                        adjusted_weight = weight * perf_weight
                    
                    adjusted_weights.append(adjusted_weight)
            
            # Normalize weights
            adjusted_weights = np.array(adjusted_weights)
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            
            # Calculate ensemble prediction
            ensemble_pred = np.average(predictions, weights=adjusted_weights)
            
            # Calculate confidence
            avg_performance = np.mean(list(performance_weights.values()))
            confidence = agreement_factor * avg_performance
            
            return float(ensemble_pred), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in adaptive combination: {e}")
            return float(np.mean(predictions)), 0.5

    def _stacking_combination(self, predictions: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """Stacking combination with meta-learner"""
        try:
            # Get recent performance for meta-learning
            performance_weights = self._get_performance_weights()
            
            # Create features for meta-learner (predictions from base models)
            X_meta = np.array(predictions).reshape(1, -1)
            
            # Simple linear combination as meta-learner
            # In a more advanced implementation, this would be a trained model
            meta_weights = np.array(list(performance_weights.values()))
            if len(meta_weights) == len(predictions):
                # Normalize meta-weights
                meta_weights = meta_weights / meta_weights.sum()
                ensemble_pred = np.dot(predictions, meta_weights)
                
                # Confidence based on meta-weight concentration
                weight_entropy = -np.sum(meta_weights * np.log(meta_weights + 1e-10))
                confidence = 1.0 / (1.0 + weight_entropy)
            else:
                # Fallback to weighted average
                weights = weights / weights.sum()
                ensemble_pred = np.average(predictions, weights=weights)
                confidence = 1.0 / (1.0 + np.std(predictions))
            
            return float(ensemble_pred), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in stacking combination: {e}")
            return float(np.mean(predictions)), 0.5
    
    def update_with_outcome(self, model_predictions: Dict[str, float], actual_outcome: float):
        """Update model performance with actual outcome"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Update individual model trackers
            for model_name, prediction in model_predictions.items():
                if model_name in self.performance_trackers:
                    self.performance_trackers[model_name].add_prediction(
                        prediction, actual_outcome, timestamp
                    )
            
            # Optimize weights based on recent performance
            self._optimize_weights()
            
            # Add to ensemble history
            self.ensemble_history.append({
                'timestamp': timestamp,
                'predictions': model_predictions.copy(),
                'actual': actual_outcome,
                'weights': self.model_weights.copy()
            })
            
            # Keep only recent history
            if len(self.ensemble_history) > 200:
                self.ensemble_history = self.ensemble_history[-200:]
                
        except Exception as e:
            logger.error(f"Error updating with outcome: {e}")
    
    def _optimize_weights(self):
        """Optimize model weights based on recent performance"""
        try:
            if len(self.ensemble_history) < self.min_performance_samples:
                return
            
            # Calculate optimal weights using recent performance
            performance_metrics = {}
            total_performance = 0
            
            for model_name in self.models.keys():
                metrics = self.performance_trackers[model_name].get_performance_metrics()
                
                # Composite performance score
                accuracy = metrics.get('accuracy', 0)
                r2 = max(0, metrics.get('r2', 0))
                
                performance_score = 0.6 * accuracy + 0.4 * r2
                performance_metrics[model_name] = max(0.05, performance_score)  # Minimum 5%
                total_performance += performance_metrics[model_name]
            
            # Update weights with smoothing
            if total_performance > 0:
                for model_name in self.models.keys():
                    optimal_weight = performance_metrics[model_name] / total_performance
                    current_weight = self.model_weights[model_name]
                    
                    # Smooth weight adjustment
                    new_weight = (1 - self.weight_adjustment_rate) * current_weight + \
                                self.weight_adjustment_rate * optimal_weight
                    
                    self.model_weights[model_name] = new_weight
            
            # Record optimization
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'old_weights': {k: v for k, v in self.model_weights.items()},
                'performance_metrics': performance_metrics.copy()
            })
            
            logger.debug(f"Weights optimized: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
    
    def get_ensemble_insights(self) -> Dict:
        """Get insights about ensemble performance"""
        try:
            insights = {
                'total_models': len(self.models),
                'current_weights': self.model_weights.copy(),
                'model_performance': {},
                'ensemble_method': self.ensemble_method.value,
                'optimization_count': len(self.optimization_history)
            }
            
            # Add individual model performance
            for model_name, tracker in self.performance_trackers.items():
                metrics = tracker.get_performance_metrics()
                insights['model_performance'][model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'r2_score': metrics.get('r2', 0),
                    'mse': metrics.get('mse', 0),
                    'prediction_count': metrics.get('count', 0),
                    'weight': self.model_weights.get(model_name, 0)
                }
            
            # Calculate ensemble performance if we have history
            if len(self.ensemble_history) >= 10:
                insights['ensemble_performance'] = self._calculate_ensemble_performance()
            
            # Log performance metrics if monitoring is available
            if MONITORING_AVAILABLE:
                try:
                    # Get ensemble performance metrics
                    ensemble_perf = insights.get('ensemble_performance', {})
                    if ensemble_perf:
                        log_model_performance("EnsembleOptimizer", ensemble_perf, insights)
                except Exception as e:
                    logger.debug(f"Failed to log ensemble performance: {e}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting ensemble insights: {e}")
            return {}

    def get_detailed_ensemble_analysis(self, features: np.ndarray) -> Dict:
        """Get detailed ensemble analysis for integration with professional buy logic"""
        try:
            # Get ensemble prediction
            result = self.predict_ensemble(features)
            
            # Get model performances
            model_performances = {}
            for model_name, tracker in self.performance_trackers.items():
                metrics = tracker.get_performance_metrics()
                model_performances[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'r2_score': metrics.get('r2', 0),
                    'mse': metrics.get('mse', 0)
                }
            
            # Calculate consensus level
            predictions = list(result.get('model_predictions', {}).values())
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                consensus_level = 1.0 / (1.0 + prediction_std)  # Higher consensus = lower std
            else:
                consensus_level = 0.5
            
            # Determine recommendation based on prediction direction and confidence
            prediction = result.get('prediction', 0)
            confidence = result.get('confidence', 0.5)
            
            if prediction > 0.02 and confidence > 0.6:  # Positive prediction with good confidence
                recommendation = "BUY"
            elif prediction < -0.02 and confidence > 0.6:  # Negative prediction with good confidence
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return {
                'success': True,
                'recommendation': recommendation,
                'prediction': prediction,
                'confidence': confidence,
                'consensus_level': consensus_level,
                'model_predictions': result.get('model_predictions', {}),
                'model_performances': model_performances,
                'ensemble_method': result.get('method_used', 'unknown'),
                'num_models': result.get('num_models', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in detailed ensemble analysis: {e}")
            return {
                'success': False,
                'recommendation': "HOLD",
                'prediction': 0.0,
                'confidence': 0.5,
                'consensus_level': 0.5,
                'model_predictions': {},
                'model_performances': {},
                'ensemble_method': 'unknown',
                'num_models': 0,
                'error': str(e)
            }
    
    def _calculate_ensemble_performance(self) -> Dict:
        """Calculate overall ensemble performance"""
        try:
            if not self.ensemble_history:
                return {}
            
            # Get recent ensemble predictions and actuals
            recent_history = self.ensemble_history[-50:]  # Last 50 predictions
            
            ensemble_predictions = []
            actuals = []
            
            for record in recent_history:
                # Recreate ensemble prediction from historical data
                predictions = list(record['predictions'].values())
                weights = list(record['weights'].values())
                
                if predictions and weights:
                    weights_array = np.array(weights)
                    weights_array = weights_array / weights_array.sum()
                    ensemble_pred = np.average(predictions, weights=weights_array)
                    
                    ensemble_predictions.append(ensemble_pred)
                    actuals.append(record['actual'])
            
            if len(ensemble_predictions) < 5:
                return {}
            
            # Calculate metrics
            mse = mean_squared_error(actuals, ensemble_predictions)
            r2 = r2_score(actuals, ensemble_predictions)
            
            # Direction accuracy
            actual_directions = [1 if a > 0 else -1 for a in actuals]
            pred_directions = [1 if p > 0 else -1 for p in ensemble_predictions]
            accuracy = sum(a == p for a, p in zip(actual_directions, pred_directions)) / len(actual_directions)
            
            return {
                'accuracy': accuracy,
                'r2_score': r2,
                'mse': mse,
                'sample_count': len(ensemble_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble performance: {e}")
            return {}
    
    def set_ensemble_method(self, method: EnsembleMethod):
        """Set the ensemble combination method"""
        self.ensemble_method = method
        logger.info(f"Ensemble method set to: {method.value}")
    
    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get models ranked by performance"""
        try:
            rankings = []
            
            for model_name, tracker in self.performance_trackers.items():
                metrics = tracker.get_performance_metrics()
                
                # Composite score
                accuracy = metrics.get('accuracy', 0)
                r2 = max(0, metrics.get('r2', 0))
                count = metrics.get('count', 0)
                
                if count > 5:  # Only rank models with sufficient data
                    score = 0.6 * accuracy + 0.4 * r2
                    rankings.append((model_name, score))
            
            # Sort by score descending
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting model rankings: {e}")
            return []
    
    def remove_underperforming_models(self, min_performance: float = 0.3):
        """Remove models that consistently underperform"""
        try:
            models_to_remove = []
            
            for model_name, tracker in self.performance_trackers.items():
                metrics = tracker.get_performance_metrics()
                
                if metrics.get('count', 0) >= 20:  # Sufficient data
                    accuracy = metrics.get('accuracy', 0)
                    if accuracy < min_performance:
                        models_to_remove.append(model_name)
            
            for model_name in models_to_remove:
                logger.info(f"Removing underperforming model: {model_name}")
                del self.models[model_name]
                del self.model_weights[model_name]
                del self.performance_trackers[model_name]
            
            # Renormalize remaining weights
            if self.model_weights:
                total_weight = sum(self.model_weights.values())
                for name in self.model_weights:
                    self.model_weights[name] /= total_weight
                    
        except Exception as e:
            logger.error(f"Error removing underperforming models: {e}")


# Global instance
_ensemble_optimizer = None

def get_ensemble_optimizer() -> EnsembleOptimizer:
    """Get global ensemble optimizer instance"""
    global _ensemble_optimizer
    if _ensemble_optimizer is None:
        _ensemble_optimizer = EnsembleOptimizer()
    return _ensemble_optimizer