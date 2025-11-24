"""
ML Model Monitoring
Provides monitoring and alerting capabilities for ML models
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class MLModelMonitor:
    """
    Monitor ML model performance and health
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_data = []
        
        # Performance thresholds
        self.performance_thresholds = {
            'accuracy': 0.7,
            'r2_score': 0.6,
            'mse': 100.0,
            'confidence': 0.6
        }
        
        logger.info("ML Model Monitor initialized")
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, Any], 
                            features: Dict[str, Any] = None) -> bool:
        """
        Log model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics dictionary
            features: Feature information (optional)
            
        Returns:
            bool: True if performance is acceptable, False otherwise
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Create performance record
            record = {
                'timestamp': timestamp,
                'model_name': model_name,
                'metrics': metrics,
                'features': features or {},
                'alerts': []
            }
            
            # Check performance against thresholds
            alerts = self._check_performance_thresholds(metrics)
            record['alerts'] = alerts
            
            # Add to monitoring data
            self.monitoring_data.append(record)
            
            # Keep only recent records (last 1000)
            if len(self.monitoring_data) > 1000:
                self.monitoring_data = self.monitoring_data[-1000:]
            
            # Log performance
            logger.info(f"Model {model_name} performance logged: {metrics}")
            
            # Log alerts if any
            if alerts:
                for alert in alerts:
                    logger.warning(f"ALERT for {model_name}: {alert}")
            
            # Save to file
            self._save_monitoring_data()
            
            return len(alerts) == 0
            
        except Exception as e:
            logger.error(f"Error logging model performance for {model_name}: {e}")
            return False
    
    def _check_performance_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Check performance metrics against thresholds
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        try:
            # Check accuracy
            if 'accuracy' in metrics and metrics['accuracy'] < self.performance_thresholds['accuracy']:
                alerts.append(f"Accuracy {metrics['accuracy']:.3f} below threshold {self.performance_thresholds['accuracy']}")
            
            # Check R2 score
            if 'r2_score' in metrics and metrics['r2_score'] < self.performance_thresholds['r2_score']:
                alerts.append(f"R2 score {metrics['r2_score']:.3f} below threshold {self.performance_thresholds['r2_score']}")
            
            # Check MSE
            if 'mse' in metrics and metrics['mse'] > self.performance_thresholds['mse']:
                alerts.append(f"MSE {metrics['mse']:.3f} above threshold {self.performance_thresholds['mse']}")
            
            # Check confidence
            if 'confidence' in metrics and metrics['confidence'] < self.performance_thresholds['confidence']:
                alerts.append(f"Confidence {metrics['confidence']:.3f} below threshold {self.performance_thresholds['confidence']}")
                
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
            alerts.append(f"Error in threshold checking: {e}")
        
        return alerts
    
    def _save_monitoring_data(self):
        """
        Save monitoring data to file
        """
        try:
            # Save to JSON file
            monitoring_file = self.log_dir / f"ml_monitoring_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(monitoring_file, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def get_model_health_report(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get health report for models
        
        Args:
            model_name: Specific model name (optional, if None returns all)
            
        Returns:
            Dictionary with health report
        """
        try:
            # Filter data by model name if specified
            if model_name:
                model_data = [record for record in self.monitoring_data if record['model_name'] == model_name]
            else:
                model_data = self.monitoring_data
            
            if not model_data:
                return {"status": "no_data", "message": "No monitoring data available"}
            
            # Calculate statistics
            recent_data = model_data[-100:]  # Last 100 records
            
            # Group by model
            model_stats = {}
            for record in recent_data:
                model = record['model_name']
                if model not in model_stats:
                    model_stats[model] = {
                        'metrics': [],
                        'alerts': 0,
                        'total_records': 0
                    }
                
                model_stats[model]['metrics'].append(record['metrics'])
                model_stats[model]['alerts'] += len(record['alerts'])
                model_stats[model]['total_records'] += 1
            
            # Calculate averages
            for model, stats in model_stats.items():
                metrics_list = stats['metrics']
                if metrics_list:
                    # Calculate average metrics
                    avg_metrics = {}
                    metric_keys = set()
                    for metrics in metrics_list:
                        metric_keys.update(metrics.keys())
                    
                    for key in metric_keys:
                        values = [m.get(key, 0) for m in metrics_list if key in m]
                        if values:
                            avg_metrics[key] = float(np.mean(values))
                    
                    model_stats[model]['avg_metrics'] = avg_metrics
                    model_stats[model]['health_score'] = self._calculate_health_score(avg_metrics, stats['alerts'], stats['total_records'])
            
            return {
                "status": "success",
                "report_timestamp": datetime.now().isoformat(),
                "models": model_stats,
                "total_records": len(model_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _calculate_health_score(self, avg_metrics: Dict[str, float], alerts: int, total_records: int) -> float:
        """
        Calculate health score for a model
        
        Args:
            avg_metrics: Average metrics
            alerts: Number of alerts
            total_records: Total number of records
            
        Returns:
            Health score (0.0 to 1.0)
        """
        try:
            # Base score from metrics
            score = 0.0
            
            # Accuracy contribution (0.3 weight)
            if 'accuracy' in avg_metrics:
                score += min(avg_metrics['accuracy'], 1.0) * 0.3
            
            # R2 score contribution (0.3 weight)
            if 'r2_score' in avg_metrics:
                score += min(max(avg_metrics['r2_score'], 0.0), 1.0) * 0.3
            
            # Confidence contribution (0.2 weight)
            if 'confidence' in avg_metrics:
                score += min(avg_metrics['confidence'], 1.0) * 0.2
            
            # Alert penalty (0.2 weight)
            if total_records > 0:
                alert_rate = alerts / total_records
                score -= min(alert_rate, 1.0) * 0.2
            
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0.5
    
    def send_alert(self, model_name: str, alert_message: str, severity: str = "warning"):
        """
        Send alert for model issues
        
        Args:
            model_name: Name of the model
            alert_message: Alert message
            severity: Severity level (info, warning, error, critical)
        """
        try:
            timestamp = datetime.now().isoformat()
            alert = {
                'timestamp': timestamp,
                'model_name': model_name,
                'message': alert_message,
                'severity': severity
            }
            
            # Log alert
            if severity == "critical":
                logger.critical(f"CRITICAL ALERT for {model_name}: {alert_message}")
            elif severity == "error":
                logger.error(f"ERROR ALERT for {model_name}: {alert_message}")
            elif severity == "warning":
                logger.warning(f"WARNING for {model_name}: {alert_message}")
            else:
                logger.info(f"INFO for {model_name}: {alert_message}")
            
            # Save alert to file
            alert_file = self.log_dir / f"ml_alerts_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Read existing alerts
            alerts = []
            if alert_file.exists():
                try:
                    with open(alert_file, 'r') as f:
                        alerts = json.load(f)
                except:
                    pass
            
            # Add new alert
            alerts.append(alert)
            
            # Keep only recent alerts (last 1000)
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            # Save alerts
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

# Global instance
_ml_monitor = None

def get_ml_monitor() -> MLModelMonitor:
    """
    Get the global ML monitor instance
    
    Returns:
        MLModelMonitor instance
    """
    global _ml_monitor
    if _ml_monitor is None:
        _ml_monitor = MLModelMonitor()
    return _ml_monitor

def log_model_performance(model_name: str, metrics: Dict[str, Any], 
                         features: Dict[str, Any] = None) -> bool:
    """
    Log model performance (convenience function)
    
    Args:
        model_name: Name of the model
        metrics: Performance metrics
        features: Feature information (optional)
        
    Returns:
        bool: True if performance is acceptable
    """
    monitor = get_ml_monitor()
    return monitor.log_model_performance(model_name, metrics, features)

def get_model_health_report(model_name: str = None) -> Dict[str, Any]:
    """
    Get model health report (convenience function)
    
    Args:
        model_name: Specific model name (optional)
        
    Returns:
        Dictionary with health report
    """
    monitor = get_ml_monitor()
    return monitor.get_model_health_report(model_name)