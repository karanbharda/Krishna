import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import threading
import os
import sys

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .risk_engine import risk_engine
from data_service_client import get_data_client

logger = logging.getLogger(__name__)

class TrackerAgent:
    def __init__(self):
        self.client = get_data_client()
        self.tracking = True
        self.monitoring_interval = 60  # Check every minute
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        # Tracking statistics
        self.stats = {
            "alerts_generated": 0,
            "stocks_monitored": 0,
            "last_check": None,
            "errors": 0
        }
        
        logger.info("TrackerAgent initialized and monitoring started")

    def _monitor_loop(self):
        """Background monitoring loop for shortlisted stocks"""
        logger.info("Starting background monitoring loop")
        
        while self.tracking:
            try:
                shortlist = self._load_shortlist()
                if not shortlist:
                    logger.debug("No shortlist found, waiting...")
                    time.sleep(self.monitoring_interval)
                    continue
                
                self.stats["stocks_monitored"] = len(shortlist)
                alerts = []
                
                for stock in shortlist:
                    try:
                        symbol = stock.get('symbol')
                        if not symbol:
                            continue
                            
                        # Get current market data
                        data = self.client.get_symbol_data(symbol)
                        if not data or 'price' not in data:
                            continue
                        
                        current_price = data['price']
                        
                        # Check thresholds using live_config.json settings
                        risk_limits = risk_engine.apply_risk_to_position(current_price)
                        
                        # Stop loss check
                        if current_price < risk_limits['stop_loss_amount']:
                            alerts.append({
                                "symbol": symbol,
                                "alert_type": "stop_loss_hit",
                                "time": datetime.now().isoformat(),
                                "current_price": current_price,
                                "stop_loss_threshold": risk_limits['stop_loss_amount'],
                                "severity": "HIGH"
                            })
                        
                        # Drawdown check
                        original_score = stock.get('score', 0)
                        if original_score > 0.8 and current_price < risk_limits['max_drawdown']:
                            alerts.append({
                                "symbol": symbol,
                                "alert_type": "drawdown_exceeded",
                                "time": datetime.now().isoformat(),
                                "current_price": current_price,
                                "max_drawdown_threshold": risk_limits['max_drawdown'],
                                "original_score": original_score,
                                "severity": "MEDIUM"
                            })
                        
                        # High volatility check
                        price_change_pct = data.get('change_pct', 0)
                        if abs(price_change_pct) > 5:  # 5% change
                            alerts.append({
                                "symbol": symbol,
                                "alert_type": "high_volatility",
                                "time": datetime.now().isoformat(),
                                "current_price": current_price,
                                "change_pct": price_change_pct,
                                "severity": "LOW"
                            })
                    
                    except Exception as e:
                        logger.debug(f"Error monitoring {symbol}: {e}")
                        continue
                
                # Save alerts if any
                if alerts:
                    self._save_alerts(alerts)
                    self.stats["alerts_generated"] += len(alerts)
                    logger.info(f"Generated {len(alerts)} alerts")
                
                self.stats["last_check"] = datetime.now().isoformat()
                logger.debug(f"Monitoring cycle completed: {len(shortlist)} stocks checked")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self.stats["errors"] += 1
                time.sleep(300)  # Wait 5 min on error

    def _load_shortlist(self) -> List[Dict[str, Any]]:
        """Load current shortlist from logs"""
        date_str = datetime.now().strftime("%Y%m%d")
        shortlist_file = f"logs/shortlist_{date_str}.json"
        
        try:
            if os.path.exists(shortlist_file):
                with open(shortlist_file, 'r') as f:
                    data = json.load(f)
                    return data.get('shortlist', [])
            else:
                logger.debug(f"Shortlist file not found: {shortlist_file}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading shortlist: {e}")
            return []

    def _save_alerts(self, alerts: List[Dict[str, Any]]):
        """Save alerts to tracked_DATE.json"""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            os.makedirs("logs", exist_ok=True)
            
            # Append to daily tracking file
            tracking_file = f"logs/tracked_{date_str}.json"
            
            # Load existing alerts if file exists
            existing_alerts = []
            if os.path.exists(tracking_file):
                try:
                    with open(tracking_file, 'r') as f:
                        existing_data = json.load(f)
                        existing_alerts = existing_data.get('alerts', [])
                except:
                    pass
            
            # Combine with new alerts
            all_alerts = existing_alerts + alerts
            
            # Save updated alerts with metadata
            tracking_data = {
                "date": date_str,
                "last_updated": datetime.now().isoformat(),
                "total_alerts": len(all_alerts),
                "alert_summary": self._get_alert_summary(all_alerts),
                "alerts": all_alerts
            }
            
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            
            logger.info(f"Saved {len(alerts)} new alerts to {tracking_file}")
            
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    def _get_alert_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get summary of alert types"""
        summary = {}
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            summary[alert_type] = summary.get(alert_type, 0) + 1
        return summary

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            **self.stats,
            "monitoring_active": self.tracking,
            "monitoring_interval_seconds": self.monitoring_interval,
            "thread_alive": self.thread.is_alive()
        }

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.tracking = False
        logger.info("Monitoring stopped")

    def start_monitoring(self):
        """Start monitoring if stopped"""
        if not self.tracking:
            self.tracking = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("Monitoring restarted")

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts from the last N hours"""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            tracking_file = f"logs/tracked_{date_str}.json"
            
            if not os.path.exists(tracking_file):
                return []
            
            with open(tracking_file, 'r') as f:
                data = json.load(f)
                alerts = data.get('alerts', [])
            
            # Filter by time (simplified - could be enhanced)
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            recent_alerts = []
            
            for alert in alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['time']).timestamp()
                    if alert_time > cutoff_time:
                        recent_alerts.append(alert)
                except:
                    continue
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []

# Global instance
tracker_agent = TrackerAgent()
