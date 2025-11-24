"""
Alerting System
Provides comprehensive alerting and monitoring for system performance and anomalies
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)

class Alert:
    """Represents a system alert"""
    
    def __init__(self, alert_type: str, severity: str, message: str, 
                 source: str, data: Optional[Dict[str, Any]] = None):
        self.alert_type = alert_type
        self.severity = severity  # INFO, WARNING, ERROR, CRITICAL
        self.message = message
        self.source = source
        self.data = data or {}
        self.timestamp = datetime.now()
        self.id = f"{alert_type}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }

class AlertingSystem:
    """Comprehensive alerting system for monitoring system performance and anomalies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.performance_metrics = {}
        self.anomaly_detectors = {}
        self.alert_handlers = []
        self.is_running = False
        self.monitoring_thread = None
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5.0,  # seconds
            "error_rate": 0.05,  # 5%
            "ml_model_accuracy": 0.7,
            "trading_performance": -0.02  # 2% drawdown
        }
        
        # Update thresholds from config
        if "thresholds" in self.config:
            self.thresholds.update(self.config["thresholds"])
        
        logger.info("Alerting system initialized")
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Alerting system monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Alerting system monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                # Check system performance
                self._check_system_performance()
                
                # Check trading performance
                self._check_trading_performance()
                
                # Check ML model performance
                self._check_ml_performance()
                
                # Sleep for monitoring interval
                import time
                time.sleep(self.config.get("monitoring_interval", 60))  # Default 1 minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Wait 30 seconds on error
    
    def _check_system_performance(self):
        """Check system performance metrics"""
        try:
            # This would integrate with system monitoring tools
            # For now, we'll simulate some checks
            pass
        except Exception as e:
            logger.error(f"System performance check error: {e}")
    
    def _check_trading_performance(self):
        """Check trading performance metrics"""
        try:
            # This would integrate with trading system metrics
            # For now, we'll simulate some checks
            pass
        except Exception as e:
            logger.error(f"Trading performance check error: {e}")
    
    def _check_ml_performance(self):
        """Check ML model performance metrics"""
        try:
            # This would integrate with ML monitoring system
            # For now, we'll simulate some checks
            pass
        except Exception as e:
            logger.error(f"ML performance check error: {e}")
    
    def add_alert_handler(self, handler):
        """Add alert handler for custom alert processing"""
        self.alert_handlers.append(handler)
        logger.info("Alert handler added")
    
    def remove_alert_handler(self, handler):
        """Remove alert handler"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            logger.info("Alert handler removed")
    
    def send_alert(self, alert: Alert):
        """Send alert through all registered handlers"""
        try:
            # Store alert
            self.alerts.append(alert)
            
            # Log alert
            log_level = getattr(logging, alert.severity, logging.INFO)
            logger.log(log_level, f"ALERT [{alert.alert_type}]: {alert.message}")
            
            # Send through handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
            
            return alert.id
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return None
    
    def create_alert(self, alert_type: str, severity: str, message: str, 
                     source: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create and send alert"""
        alert = Alert(alert_type, severity, message, source, data)
        return self.send_alert(alert)
    
    def check_performance_threshold(self, metric_name: str, value: float, 
                                  context: Optional[Dict[str, Any]] = None):
        """Check if performance metric exceeds threshold and send alert if needed"""
        try:
            threshold = self.thresholds.get(metric_name)
            if threshold is not None and value > threshold:
                self.create_alert(
                    alert_type=f"PERFORMANCE_THRESHOLD_EXCEEDED",
                    severity="WARNING",
                    message=f"{metric_name} exceeded threshold: {value:.2f} > {threshold:.2f}",
                    source="performance_monitor",
                    data={
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "context": context or {}
                    }
                )
        except Exception as e:
            logger.error(f"Performance threshold check error: {e}")
    
    def check_anomaly(self, metric_name: str, value: float, 
                     historical_data: List[float], context: Optional[Dict[str, Any]] = None):
        """Check if value is anomalous compared to historical data"""
        try:
            if len(historical_data) < 5:
                return  # Not enough data
            
            # Calculate statistics
            import numpy as np
            mean = np.mean(historical_data)
            std = np.std(historical_data)
            
            # Check if value is more than 2 standard deviations from mean
            if abs(value - mean) > 2 * std:
                self.create_alert(
                    alert_type=f"ANOMALY_DETECTED",
                    severity="WARNING",
                    message=f"Anomaly detected in {metric_name}: {value:.2f} (mean: {mean:.2f}, std: {std:.2f})",
                    source="anomaly_detector",
                    data={
                        "metric": metric_name,
                        "value": value,
                        "mean": mean,
                        "std": std,
                        "context": context or {}
                    }
                )
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
    
    def get_recent_alerts(self, hours: int = 24, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = []
            
            for alert in reversed(self.alerts):
                if alert.timestamp >= cutoff_time:
                    if severity is None or alert.severity == severity:
                        recent_alerts.append(alert.to_dict())
                else:
                    break  # Alerts are in chronological order
            
            return recent_alerts
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            severity_counts = {}
            alert_types = {}
            
            for alert in self.alerts:
                # Count by severity
                severity = alert.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # Count by type
                alert_type = alert.alert_type
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            return {
                "total_alerts": len(self.alerts),
                "severity_counts": severity_counts,
                "alert_types": alert_types,
                "is_monitoring": self.is_running
            }
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {}

# Global alerting system instance
_alerting_system = None

def get_alerting_system() -> AlertingSystem:
    """Get global alerting system instance"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
        _alerting_system.start_monitoring()
    return _alerting_system

def send_alert(alert_type: str, severity: str, message: str, 
               source: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to send alert"""
    alerting_system = get_alerting_system()
    return alerting_system.create_alert(alert_type, severity, message, source, data)

def check_performance_threshold(metric_name: str, value: float, 
                              context: Optional[Dict[str, Any]] = None):
    """Convenience function to check performance threshold"""
    alerting_system = get_alerting_system()
    alerting_system.check_performance_threshold(metric_name, value, context)

def check_anomaly(metric_name: str, value: float, 
                 historical_data: List[float], context: Optional[Dict[str, Any]] = None):
    """Convenience function to check for anomalies"""
    alerting_system = get_alerting_system()
    alerting_system.check_anomaly(metric_name, value, historical_data, context)