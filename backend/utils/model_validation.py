"""
Model Validation and Performance Monitoring
Validates ML model performance and triggers retraining when performance degrades
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

# Import ML components for validation
from core.rl_agent import rl_agent
from utils.ensemble_optimizer import get_ensemble_optimizer
from utils.advanced_feature_engineer import get_feature_engineer

# Import monitoring
try:
    from utils.monitoring import get_ml_monitor, log_model_performance
    MONITORING_AVAILABLE = True
except ImportError:
    logging.warning("Monitoring not available for Model Validation")
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validates ML model performance and manages retraining triggers
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validation_history = []
        self.retraining_triggers = []
        
        # Validation thresholds
        self.validation_thresholds = {
            'accuracy_degradation': 0.1,  # 10% degradation threshold
            'r2_degradation': 0.1,        # 10% R2 degradation threshold
            'mse_increase': 0.5,          # 50% MSE increase threshold
            'confidence_drop': 0.2,       # 20% confidence drop threshold
            'min_samples': 10,            # Minimum samples for validation (lower for RL agent)
            'sharpe_ratio_degradation': 0.1,  # 10% Sharpe ratio degradation
            'prediction_accuracy_degradation': 0.15  # 15% prediction accuracy degradation
        }
        
        # Retraining configuration
        self.retraining_config = {
            'enable_auto_retraining': self.config.get('enable_auto_retraining', False),
            'retraining_window': self.config.get('retraining_window', 30),  # Days
            'min_performance_drop': self.config.get('min_performance_drop', 0.15)  # 15% drop
        }
        
        logger.info("Model Validator initialized")
    
    def validate_model_performance(self, model_name: str, current_metrics: Dict[str, Any], 
                                 baseline_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate model performance against baseline or thresholds
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics (optional)
            
        Returns:
            Dictionary with validation results
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Create validation record
            validation_record = {
                'timestamp': timestamp,
                'model_name': model_name,
                'current_metrics': current_metrics,
                'baseline_metrics': baseline_metrics,
                'validation_results': {},
                'recommendations': []
            }
            
            # Perform validation checks
            validation_results = self._perform_validation_checks(current_metrics, baseline_metrics)
            validation_record['validation_results'] = validation_results
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, model_name)
            validation_record['recommendations'] = recommendations
            
            # Add to validation history
            self.validation_history.append(validation_record)
            
            # Keep only recent records (last 500)
            if len(self.validation_history) > 500:
                self.validation_history = self.validation_history[-500:]
            
            # Log validation results
            logger.info(f"Model {model_name} validation completed: {len(recommendations)} recommendations")
            
            # Check for retraining triggers
            if self._should_trigger_retraining(validation_results):
                self._trigger_retraining(model_name, validation_results)
            
            # Save validation data
            self._save_validation_data()
            
            return {
                'success': True,
                'validation_record': validation_record,
                'needs_attention': len(recommendations) > 0
            }
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _perform_validation_checks(self, current_metrics: Dict[str, Any], 
                                 baseline_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform validation checks on model metrics
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics (optional)
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        try:
            # If we have baseline metrics, compare against them
            if baseline_metrics:
                # Accuracy validation
                if 'accuracy' in current_metrics and 'accuracy' in baseline_metrics:
                    current_acc = current_metrics['accuracy']
                    baseline_acc = baseline_metrics['accuracy']
                    accuracy_change = current_acc - baseline_acc
                    validation_results['accuracy'] = {
                        'current': current_acc,
                        'baseline': baseline_acc,
                        'change': accuracy_change,
                        'degraded': accuracy_change < -self.validation_thresholds['accuracy_degradation'],
                        'threshold': self.validation_thresholds['accuracy_degradation']
                    }
                
                # R2 score validation
                if 'r2_score' in current_metrics and 'r2_score' in baseline_metrics:
                    current_r2 = current_metrics['r2_score']
                    baseline_r2 = baseline_metrics['r2_score']
                    r2_change = current_r2 - baseline_r2
                    validation_results['r2_score'] = {
                        'current': current_r2,
                        'baseline': baseline_r2,
                        'change': r2_change,
                        'degraded': r2_change < -self.validation_thresholds['r2_degradation'],
                        'threshold': self.validation_thresholds['r2_degradation']
                    }
                
                # MSE validation
                if 'mse' in current_metrics and 'mse' in baseline_metrics:
                    current_mse = current_metrics['mse']
                    baseline_mse = baseline_metrics['mse']
                    mse_change = (current_mse - baseline_mse) / max(baseline_mse, 1e-8)
                    validation_results['mse'] = {
                        'current': current_mse,
                        'baseline': baseline_mse,
                        'change': mse_change,
                        'increased': mse_change > self.validation_thresholds['mse_increase'],
                        'threshold': self.validation_thresholds['mse_increase']
                    }
                
                # Confidence validation
                if 'confidence' in current_metrics and 'confidence' in baseline_metrics:
                    current_conf = current_metrics['confidence']
                    baseline_conf = baseline_metrics['confidence']
                    conf_change = current_conf - baseline_conf
                    validation_results['confidence'] = {
                        'current': current_conf,
                        'baseline': baseline_conf,
                        'change': conf_change,
                        'dropped': conf_change < -self.validation_thresholds['confidence_drop'],
                        'threshold': self.validation_thresholds['confidence_drop']
                    }
                
                # Sharpe ratio validation (for RL models)
                if 'sharpe_ratio' in current_metrics and 'sharpe_ratio' in baseline_metrics:
                    current_sharpe = current_metrics['sharpe_ratio']
                    baseline_sharpe = baseline_metrics['sharpe_ratio']
                    sharpe_change = current_sharpe - baseline_sharpe
                    validation_results['sharpe_ratio'] = {
                        'current': current_sharpe,
                        'baseline': baseline_sharpe,
                        'change': sharpe_change,
                        'degraded': sharpe_change < -self.validation_thresholds['sharpe_ratio_degradation'],
                        'threshold': self.validation_thresholds['sharpe_ratio_degradation']
                    }
                
                # Prediction accuracy validation (for ensemble models)
                if 'prediction_accuracy' in current_metrics and 'prediction_accuracy' in baseline_metrics:
                    current_pred_acc = current_metrics['prediction_accuracy']
                    baseline_pred_acc = baseline_metrics['prediction_accuracy']
                    pred_acc_change = current_pred_acc - baseline_pred_acc
                    validation_results['prediction_accuracy'] = {
                        'current': current_pred_acc,
                        'baseline': baseline_pred_acc,
                        'change': pred_acc_change,
                        'degraded': pred_acc_change < -self.validation_thresholds['prediction_accuracy_degradation'],
                        'threshold': self.validation_thresholds['prediction_accuracy_degradation']
                    }
            else:
                # Validate against absolute thresholds
                if 'accuracy' in current_metrics:
                    accuracy = current_metrics['accuracy']
                    validation_results['accuracy'] = {
                        'current': accuracy,
                        'threshold_check': accuracy >= 0.7,  # Minimum 70% accuracy
                        'min_threshold': 0.7
                    }
                
                if 'r2_score' in current_metrics:
                    r2_score = current_metrics['r2_score']
                    validation_results['r2_score'] = {
                        'current': r2_score,
                        'threshold_check': r2_score >= 0.6,  # Minimum 60% R2
                        'min_threshold': 0.6
                    }
                
                if 'mse' in current_metrics:
                    mse = current_metrics['mse']
                    validation_results['mse'] = {
                        'current': mse,
                        'threshold_check': mse <= 100.0,  # Maximum MSE of 100
                        'max_threshold': 100.0
                    }
                
                if 'confidence' in current_metrics:
                    confidence = current_metrics['confidence']
                    validation_results['confidence'] = {
                        'current': confidence,
                        'threshold_check': confidence >= 0.6,  # Minimum 60% confidence
                        'min_threshold': 0.6
                    }
                
                # Sharpe ratio validation (for RL models)
                if 'sharpe_ratio' in current_metrics:
                    sharpe_ratio = current_metrics['sharpe_ratio']
                    validation_results['sharpe_ratio'] = {
                        'current': sharpe_ratio,
                        'threshold_check': sharpe_ratio >= 0.5,  # Minimum Sharpe ratio of 0.5
                        'min_threshold': 0.5
                    }
                
                # Prediction accuracy validation (for ensemble models)
                if 'prediction_accuracy' in current_metrics:
                    prediction_accuracy = current_metrics['prediction_accuracy']
                    validation_results['prediction_accuracy'] = {
                        'current': prediction_accuracy,
                        'threshold_check': prediction_accuracy >= 0.7,  # Minimum 70% prediction accuracy
                        'min_threshold': 0.7
                    }
            
            # Sample count validation
            if 'sample_count' in current_metrics:
                sample_count = current_metrics['sample_count']
                validation_results['sample_count'] = {
                    'current': sample_count,
                    'sufficient': sample_count >= self.validation_thresholds['min_samples'],
                    'min_required': self.validation_thresholds['min_samples']
                }
            
        except Exception as e:
            logger.error(f"Error performing validation checks: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], model_name: str) -> List[str]:
        """
        Generate recommendations based on validation results
        
        Args:
            validation_results: Validation results dictionary
            model_name: Name of the model
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Check for degraded metrics
            for metric, result in validation_results.items():
                if isinstance(result, dict):
                    if result.get('degraded', False):
                        change_pct = result.get('change', 0) * 100
                        recommendations.append(f"Model {model_name} {metric} degraded by {change_pct:.2f}%")
                    elif result.get('increased', False):
                        change_pct = result.get('change', 0) * 100
                        recommendations.append(f"Model {model_name} {metric} increased by {change_pct:.2f}%")
                    elif result.get('dropped', False):
                        change_pct = result.get('change', 0) * 100
                        recommendations.append(f"Model {model_name} {metric} dropped by {change_pct:.2f}%")
                    elif 'threshold_check' in result and not result['threshold_check']:
                        current_val = result.get('current', 0)
                        threshold = result.get('min_threshold') or result.get('max_threshold', 0)
                        recommendations.append(f"Model {model_name} {metric} ({current_val:.3f}) below threshold ({threshold})")
            
            # Check for insufficient samples
            if 'sample_count' in validation_results:
                sample_result = validation_results['sample_count']
                if not sample_result.get('sufficient', True):
                    current_samples = sample_result.get('current', 0)
                    required_samples = sample_result.get('min_required', 0)
                    recommendations.append(f"Insufficient samples for {model_name}: {current_samples}/{required_samples}")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _should_trigger_retraining(self, validation_results: Dict[str, Any]) -> bool:
        """
        Determine if retraining should be triggered based on validation results
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Boolean indicating if retraining should be triggered
        """
        try:
            if not self.retraining_config['enable_auto_retraining']:
                return False
            
            # Check for significant performance degradation
            significant_degradation = False
            
            for metric, result in validation_results.items():
                if isinstance(result, dict):
                    if result.get('degraded', False):
                        change_abs = abs(result.get('change', 0))
                        if change_abs >= self.retraining_config['min_performance_drop']:
                            significant_degradation = True
                            break
                    elif result.get('increased', False):
                        change_abs = abs(result.get('change', 0))
                        if change_abs >= self.retraining_config['min_performance_drop']:
                            significant_degradation = True
                            break
                    elif result.get('dropped', False):
                        change_abs = abs(result.get('change', 0))
                        if change_abs >= self.retraining_config['min_performance_drop']:
                            significant_degradation = True
                            break
            
            return significant_degradation
            
        except Exception as e:
            logger.error(f"Error checking retraining trigger: {e}")
            return False
    
    def _trigger_retraining(self, model_name: str, validation_results: Dict[str, Any]):
        """
        Trigger model retraining
        
        Args:
            model_name: Name of the model
            validation_results: Validation results dictionary
        """
        try:
            timestamp = datetime.now().isoformat()
            
            retraining_trigger = {
                'timestamp': timestamp,
                'model_name': model_name,
                'validation_results': validation_results,
                'triggered': True
            }
            
            self.retraining_triggers.append(retraining_trigger)
            
            # Keep only recent triggers (last 100)
            if len(self.retraining_triggers) > 100:
                self.retraining_triggers = self.retraining_triggers[-100:]
            
            # Log retraining trigger
            logger.warning(f"Retraining triggered for model {model_name}")
            
            # In a real implementation, this would trigger actual retraining
            # For now, we'll just log it
            
        except Exception as e:
            logger.error(f"Error triggering retraining for {model_name}: {e}")
    
    def _save_validation_data(self):
        """
        Save validation data to file
        """
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Save validation history
            validation_file = log_dir / f"model_validation_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(validation_file, 'w') as f:
                json.dump(self.validation_history, f, indent=2, default=str)
                
            # Save retraining triggers
            if self.retraining_triggers:
                trigger_file = log_dir / f"retraining_triggers_{datetime.now().strftime('%Y%m%d')}.json"
                
                with open(trigger_file, 'w') as f:
                    json.dump(self.retraining_triggers, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"Error saving validation data: {e}")
    
    def get_validation_report(self, model_name: str = None, days: int = 30) -> Dict[str, Any]:
        """
        Get validation report for models
        
        Args:
            model_name: Specific model name (optional)
            days: Number of days to include in report
            
        Returns:
            Dictionary with validation report
        """
        try:
            # Filter data by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_validations = []
            
            for record in self.validation_history:
                try:
                    record_date = datetime.fromisoformat(record['timestamp'])
                    if record_date >= cutoff_date:
                        if model_name is None or record['model_name'] == model_name:
                            recent_validations.append(record)
                except Exception:
                    # Skip records with invalid timestamps
                    continue
            
            if not recent_validations:
                return {
                    "status": "no_data",
                    "message": f"No validation data found for the last {days} days"
                }
            
            # Generate summary statistics
            model_summaries = {}
            for record in recent_validations:
                model = record['model_name']
                if model not in model_summaries:
                    model_summaries[model] = {
                        'total_validations': 0,
                        'issues_found': 0,
                        'recommendations': 0,
                        'latest_validation': None
                    }
                
                model_summaries[model]['total_validations'] += 1
                model_summaries[model]['issues_found'] += len([r for r in record.get('validation_results', {}).values() 
                                                              if isinstance(r, dict) and 
                                                              (r.get('degraded') or r.get('increased') or 
                                                               r.get('dropped') or not r.get('threshold_check', True))])
                model_summaries[model]['recommendations'] += len(record.get('recommendations', []))
                model_summaries[model]['latest_validation'] = record['timestamp']
            
            return {
                "status": "success",
                "report_timestamp": datetime.now().isoformat(),
                "period_days": days,
                "total_validations": len(recent_validations),
                "models": model_summaries,
                "recent_validations": recent_validations[-10:]  # Last 10 validations
            }
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def set_validation_thresholds(self, thresholds: Dict[str, float]):
        """
        Set validation thresholds
        
        Args:
            thresholds: Dictionary of threshold values
        """
        try:
            self.validation_thresholds.update(thresholds)
            logger.info(f"Validation thresholds updated: {thresholds}")
        except Exception as e:
            logger.error(f"Error updating validation thresholds: {e}")
    
    def set_retraining_config(self, config: Dict[str, Any]):
        """
        Set retraining configuration
        
        Args:
            config: Dictionary of retraining configuration
        """
        try:
            self.retraining_config.update(config)
            logger.info(f"Retraining configuration updated: {config}")
        except Exception as e:
            logger.error(f"Error updating retraining configuration: {e}")
    
    def validate_all_ml_models(self) -> Dict[str, Any]:
        """
        Validate all ML models in the system and return a comprehensive report
        
        Returns:
            Dictionary with validation results for all models
        """
        try:
            validation_results = {}
            
            # Validate RL Agent
            rl_metrics = self._get_rl_agent_metrics()
            rl_validation = self.validate_model_performance(
                "RLAgent", rl_metrics
            )
            validation_results["rl_agent"] = rl_validation
            
            # Validate Ensemble Optimizer
            ensemble_metrics = self._get_ensemble_optimizer_metrics()
            ensemble_validation = self.validate_model_performance(
                "EnsembleOptimizer", ensemble_metrics
            )
            validation_results["ensemble_optimizer"] = ensemble_validation
            
            # Validate Feature Engineer
            feature_metrics = self._get_feature_engineer_metrics()
            feature_validation = self.validate_model_performance(
                "FeatureEngineer", feature_metrics
            )
            validation_results["feature_engineer"] = feature_validation
            
            # Overall system health
            overall_health = self._calculate_overall_health(validation_results)
            validation_results["overall_health"] = overall_health
            
            return {
                "success": True,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating all ML models: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_rl_agent_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for RL Agent validation
        
        Returns:
            Dictionary with RL agent metrics
        """
        try:
            # Get RL agent stats
            rl_stats = rl_agent.get_model_stats()
            
            # Calculate performance metrics
            total_processed = rl_stats.get("filtering_stats", {}).get("total_processed", 0)
            risk_compliant = rl_stats.get("filtering_stats", {}).get("risk_compliant", 0)
            
            # Calculate accuracy with proper handling of edge cases (same as RL agent)
            if total_processed > 0:
                accuracy = risk_compliant / total_processed
            else:
                # Default accuracy when no stocks processed yet
                accuracy = 0.7  # Assume good performance until proven otherwise
            
            return {
                "accuracy": accuracy,
                "sample_count": total_processed,
                "sharpe_ratio": rl_stats.get("sharpe_ratio", 0.5),
                "confidence": rl_stats.get("avg_confidence", 0.7)
            }
        except Exception as e:
            logger.error(f"Error getting RL agent metrics: {e}")
            return {
                "accuracy": 0.5,
                "sample_count": 0,
                "sharpe_ratio": 0.0,
                "confidence": 0.5
            }
    
    def _get_ensemble_optimizer_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Ensemble Optimizer validation
        
        Returns:
            Dictionary with ensemble optimizer metrics
        """
        try:
            # Get ensemble optimizer insights
            ensemble_insights = get_ensemble_optimizer().get_ensemble_insights()
            
            # Calculate performance metrics
            model_performance = ensemble_insights.get("model_performance", {})
            
            # Average accuracy across all models
            accuracies = [perf.get("accuracy", 0) for perf in model_performance.values()]
            avg_accuracy = sum(accuracies) / max(len(accuracies), 1) if accuracies else 0.5
            
            # Average R2 score across all models
            r2_scores = [perf.get("r2_score", 0) for perf in model_performance.values()]
            avg_r2 = sum(r2_scores) / max(len(r2_scores), 1) if r2_scores else 0.5
            
            # Prediction accuracy (directional accuracy)
            prediction_accuracy = avg_accuracy
            
            return {
                "accuracy": avg_accuracy,
                "r2_score": avg_r2,
                "prediction_accuracy": prediction_accuracy,
                "sample_count": len(model_performance),
                "confidence": ensemble_insights.get("ensemble_performance", {}).get("accuracy", 0.7)
            }
        except Exception as e:
            logger.error(f"Error getting ensemble optimizer metrics: {e}")
            return {
                "accuracy": 0.5,
                "r2_score": 0.5,
                "prediction_accuracy": 0.5,
                "sample_count": 0,
                "confidence": 0.5
            }
    
    def _get_feature_engineer_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Feature Engineer validation
        
        Returns:
            Dictionary with feature engineer metrics
        """
        try:
            # Get feature engineer summary
            feature_summary = get_feature_engineer().get_feature_summary()
            
            # Calculate performance metrics
            total_features = feature_summary.get("total_features_engineered", 0)
            selected_features = feature_summary.get("selected_features_count", 0)
            
            # Feature selection ratio
            selection_ratio = selected_features / max(total_features, 1)
            
            # Feature quality score (based on engineered features)
            quality_score = min(total_features / 100.0, 1.0)  # Normalize to 0-1
            
            return {
                "accuracy": quality_score,
                "sample_count": total_features,
                "confidence": selection_ratio,
                "feature_quality": quality_score
            }
        except Exception as e:
            logger.error(f"Error getting feature engineer metrics: {e}")
            return {
                "accuracy": 0.5,
                "sample_count": 0,
                "confidence": 0.5,
                "feature_quality": 0.5
            }
    
    def _calculate_overall_health(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall system health based on individual model validations
        
        Args:
            validation_results: Dictionary with individual model validation results
            
        Returns:
            Dictionary with overall health metrics
        """
        try:
            health_scores = []
            
            # Collect health scores from each model
            for model_name, result in validation_results.items():
                if isinstance(result, dict) and "validation_record" in result:
                    # Extract metrics from validation record
                    metrics = result["validation_record"].get("current_metrics", {})
                    
                    # Calculate health score based on key metrics
                    accuracy = metrics.get("accuracy", 0.5)
                    confidence = metrics.get("confidence", 0.5)
                    sample_count = metrics.get("sample_count", 0)
                    
                    # Weighted health score
                    health_score = (accuracy * 0.4 + confidence * 0.4 + 
                                  min(sample_count / 1000, 1.0) * 0.2)
                    health_scores.append(health_score)
            
            # Average health score
            avg_health = sum(health_scores) / max(len(health_scores), 1)
            
            # System status
            if avg_health >= 0.8:
                status = "healthy"
            elif avg_health >= 0.6:
                status = "degraded"
            else:
                status = "critical"
            
            return {
                "health_score": avg_health,
                "status": status,
                "models_validated": len(health_scores)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return {
                "health_score": 0.5,
                "status": "unknown",
                "models_validated": 0
            }

# Global instance
_model_validator = None

def get_model_validator(config: Dict[str, Any] = None) -> ModelValidator:
    """
    Get the global model validator instance
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        ModelValidator instance
    """
    global _model_validator
    if _model_validator is None:
        _model_validator = ModelValidator(config)
    return _model_validator

def validate_model_performance(model_name: str, current_metrics: Dict[str, Any], 
                             baseline_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate model performance (convenience function)
    
    Args:
        model_name: Name of the model
        current_metrics: Current performance metrics
        baseline_metrics: Baseline performance metrics (optional)
        
    Returns:
        Dictionary with validation results
    """
    validator = get_model_validator()
    return validator.validate_model_performance(model_name, current_metrics, baseline_metrics)

def get_validation_report(model_name: str = None, days: int = 30) -> Dict[str, Any]:
    """
    Get validation report (convenience function)
    
    Args:
        model_name: Specific model name (optional)
        days: Number of days to include in report
        
    Returns:
        Dictionary with validation report
    """
    validator = get_model_validator()
    return validator.get_validation_report(model_name, days)