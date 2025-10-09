"""
Unified ML Interface
Provides a consistent interface for accessing all ML models in the trading system
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Import all ML components
from core.rl_agent import rl_agent
from utils.ensemble_optimizer import get_ensemble_optimizer
from utils.advanced_feature_engineer import get_feature_engineer

# Import model validation
try:
    from utils.model_validation import get_model_validator
    VALIDATION_AVAILABLE = True
except ImportError:
    logger.warning("Model validation not available for ML Interface")
    VALIDATION_AVAILABLE = False

# Import monitoring
try:
    from utils.monitoring import log_model_performance, get_model_health_report
    MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("Monitoring not available for ML Interface")
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLInterface:
    """
    Unified interface for all ML models in the trading system
    Provides consistent access to RL agent, ensemble optimizer, and feature engineer
    """
    
    def __init__(self):
        self.rl_agent = rl_agent
        self.ensemble_optimizer = get_ensemble_optimizer()
        self.feature_engineer = get_feature_engineer()
        
        # Initialize model validator
        self.model_validator = get_model_validator() if VALIDATION_AVAILABLE else None
        
        logger.info("ML Interface initialized with all components")
    
    def get_rl_analysis(self, data: Dict[str, Any], horizon: str = "day") -> Dict[str, Any]:
        """
        Get RL analysis for a stock
        
        Args:
            data: Dictionary containing stock data (price, volume, change, change_pct)
            horizon: Time horizon for analysis (day, week, month, year)
            
        Returns:
            Dictionary with RL analysis results
        """
        try:
            return self.rl_agent.get_rl_analysis(data, horizon)
        except Exception as e:
            logger.error(f"Error in RL analysis: {e}")
            return {
                "success": False,
                "recommendation": "HOLD",
                "confidence": 0.5,
                "error": str(e)
            }
    
    def get_ensemble_analysis(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get ensemble analysis for a stock
        
        Args:
            features: Feature vector for the stock
            
        Returns:
            Dictionary with ensemble analysis results
        """
        try:
            return self.ensemble_optimizer.get_detailed_ensemble_analysis(features)
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {e}")
            return {
                "success": False,
                "recommendation": "HOLD",
                "prediction": 0.0,
                "confidence": 0.5,
                "error": str(e)
            }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw stock data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            return self.feature_engineer.engineer_all_features(data)
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return data  # Return original data if engineering fails
    
    def select_best_features(self, features_df: pd.DataFrame, target: pd.Series, k: int = 50) -> List[str]:
        """
        Select best features from engineered features
        
        Args:
            features_df: DataFrame with engineered features
            target: Target series for feature selection
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            return self.feature_engineer.select_best_features(features_df, target, k)
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            # Return first k columns as fallback
            return list(features_df.columns)[:min(k, len(features_df.columns))]
    
    def get_feature_importance(self) -> List[tuple]:
        """
        Get feature importance rankings
        
        Returns:
            List of (feature_name, importance_score) tuples sorted by importance
        """
        try:
            return self.feature_engineer.get_feature_importance_ranking()
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return []
    
    def update_ensemble_model(self, model_name: str, model: Any, initial_weight: float = 1.0):
        """
        Register or update a model in the ensemble optimizer
        
        Args:
            model_name: Name of the model
            model: Model object
            initial_weight: Initial weight for the model
        """
        try:
            self.ensemble_optimizer.register_model(model_name, model, initial_weight)
            logger.info(f"Model '{model_name}' registered in ensemble optimizer")
        except Exception as e:
            logger.error(f"Error registering model '{model_name}': {e}")
    
    def update_ensemble_performance(self, model_predictions: Dict[str, float], actual_outcome: float):
        """
        Update ensemble model performance with actual outcomes
        
        Args:
            model_predictions: Dictionary of model predictions
            actual_outcome: Actual outcome value
        """
        try:
            self.ensemble_optimizer.update_with_outcome(model_predictions, actual_outcome)
            logger.debug("Ensemble optimizer performance updated")
        except Exception as e:
            logger.error(f"Error updating ensemble performance: {e}")
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all models
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            return self.ensemble_optimizer.get_ensemble_insights()
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all ML components
        
        Returns:
            Dictionary with health status
        """
        try:
            health_status = {
                "rl_agent": {
                    "available": True,
                    "device": str(self.rl_agent.device),
                    "gpu_available": self.rl_agent.get_model_stats().get("gpu_available", False),
                    "processed_stocks": self.rl_agent.filtering_stats.get("total_processed", 0)
                },
                "ensemble_optimizer": {
                    "available": True,
                    "models_registered": len(self.ensemble_optimizer.models),
                    "ensemble_method": self.ensemble_optimizer.ensemble_method.value
                },
                "feature_engineer": {
                    "available": True,
                    "features_engineered": len(self.feature_engineer.feature_importance) if self.feature_engineer.feature_importance else 0
                }
            }
            
            # Add monitoring health report if available
            if MONITORING_AVAILABLE:
                try:
                    health_status["monitoring"] = get_model_health_report()
                except Exception as e:
                    logger.debug(f"Failed to get monitoring health report: {e}")
                    health_status["monitoring"] = {"available": False, "error": str(e)}
            
            return health_status
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "rl_agent": {"available": False, "error": str(e)},
                "ensemble_optimizer": {"available": False, "error": str(e)},
                "feature_engineer": {"available": False, "error": str(e)}
            }
    
    def validate_all_models(self) -> Dict[str, Any]:
        """
        Validate all ML models and return comprehensive report
        
        Returns:
            Dictionary with validation results
        """
        try:
            if self.model_validator is None:
                return {"success": False, "error": "Model validation not available"}
            
            return self.model_validator.validate_all_ml_models()
        except Exception as e:
            logger.error(f"Error validating all models: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_model_performance(self, model_name: str, current_metrics: Dict[str, Any], 
                                 baseline_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate performance of a specific model
        
        Args:
            model_name: Name of the model to validate
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics (optional)
            
        Returns:
            Dictionary with validation results
        """
        try:
            if self.model_validator is None:
                return {"success": False, "error": "Model validation not available"}
            
            return self.model_validator.validate_model_performance(
                model_name, current_metrics, baseline_metrics
            )
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get health status of all ML components
        
        Returns:
            Dictionary with health status
        """
        try:
            health_status = {
                "rl_agent": {
                    "available": True,
                    "device": str(self.rl_agent.device),
                    "gpu_available": self.rl_agent.get_model_stats().get("gpu_available", False),
                    "processed_stocks": self.rl_agent.filtering_stats.get("total_processed", 0)
                },
                "ensemble_optimizer": {
                    "available": True,
                    "models_registered": len(self.ensemble_optimizer.models),
                    "ensemble_method": self.ensemble_optimizer.ensemble_method.value
                },
                "feature_engineer": {
                    "available": True,
                    "features_engineered": len(self.feature_engineer.feature_importance) if self.feature_engineer.feature_importance else 0
                }
            }
            return health_status
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "rl_agent": {"available": False, "error": str(e)},
                "ensemble_optimizer": {"available": False, "error": str(e)},
                "feature_engineer": {"available": False, "error": str(e)}
            }

# Global instance
_ml_interface = None

def get_ml_interface() -> MLInterface:
    """
    Get the global ML interface instance
    
    Returns:
        MLInterface instance
    """
    global _ml_interface
    if _ml_interface is None:
        _ml_interface = MLInterface()
    return _ml_interface

def get_unified_ml_analysis(data: Dict[str, Any], features: Optional[np.ndarray] = None, 
                          ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get unified ML analysis from all models
    
    Args:
        data: Dictionary containing stock data for RL agent
        features: Feature vector for ensemble optimizer (optional)
        ohlcv_data: OHLCV data for feature engineering (optional)
        
    Returns:
        Dictionary with unified ML analysis from all models
    """
    try:
        ml_interface = get_ml_interface()
        
        # Get RL analysis
        rl_analysis = ml_interface.get_rl_analysis(data)
        
        # Get ensemble analysis if features provided
        ensemble_analysis = {}
        if features is not None:
            ensemble_analysis = ml_interface.get_ensemble_analysis(features)
        
        # Engineer features if OHLCV data provided
        engineered_features = None
        if ohlcv_data is not None:
            engineered_features = ml_interface.engineer_features(ohlcv_data)
        
        return {
            "success": True,
            "rl_analysis": rl_analysis,
            "ensemble_analysis": ensemble_analysis,
            "engineered_features": engineered_features,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in unified ML analysis: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }