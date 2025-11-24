"""
Signal Tracker for Continuous Learning
Tracks signal performance to reduce false signal rates and improve adaptive trading
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import sessionmaker

# Fix import issues by adding the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from db.database import SignalPerformance, init_db

logger = logging.getLogger(__name__)


class SignalTracker:
    """
    Tracks signal performance for continuous learning and false signal rate reduction
    """
    
    def __init__(self, db_path: str = None):
        self.engine = init_db(db_path)
        self.Session = sessionmaker(bind=self.engine)
        self.performance_cache = {}  # Cache for performance metrics
        self.cache_expiry = timedelta(hours=1)  # Cache expiry time
        
        logger.info("Signal Tracker initialized")
    
    def record_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Record a signal for performance tracking
        
        Args:
            signal_data: Dictionary containing signal information
                Required keys: symbol, signal_type, signal_name
                Optional keys: signal_strength, signal_confidence, market_regime, 
                              liquidity_score, volatility_regime, additional_metrics
        """
        try:
            session = self.Session()
            
            # Create signal performance record
            signal_record = SignalPerformance(
                symbol=signal_data.get('symbol', ''),
                signal_type=signal_data.get('signal_type', ''),
                signal_name=signal_data.get('signal_name', ''),
                signal_strength=signal_data.get('signal_strength', 0.0),
                signal_confidence=signal_data.get('signal_confidence', 0.0),
                market_regime=signal_data.get('market_regime', 'unknown'),
                liquidity_score=signal_data.get('liquidity_score', 0.5),
                volatility_regime=signal_data.get('volatility_regime', 'normal'),
                additional_metrics=signal_data.get('additional_metrics', {})
            )
            
            session.add(signal_record)
            session.commit()
            
            # Update cache
            cache_key = f"{signal_data.get('signal_type', '')}_{signal_data.get('signal_name', '')}"
            if cache_key in self.performance_cache:
                del self.performance_cache[cache_key]
            
            logger.debug(f"Recorded signal: {signal_data.get('signal_name', '')} for {signal_data.get('symbol', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
            if session:
                session.rollback()
            return False
        finally:
            if session:
                session.close()
    
    def update_signal_outcome(self, signal_id: int, actual_outcome: float, 
                             outcome_duration: int, is_correct: bool, 
                             false_signal_reason: str = None) -> bool:
        """
        Update a signal with its actual outcome for learning purposes
        
        Args:
            signal_id: ID of the signal record
            actual_outcome: Actual price movement (positive = correct buy signal, negative = correct sell signal)
            outcome_duration: Days until outcome measurement
            is_correct: Whether the signal was correct
            false_signal_reason: Reason for false signal if applicable
        """
        try:
            session = self.Session()
            
            # Find the signal record
            signal_record = session.query(SignalPerformance).filter_by(id=signal_id).first()
            if not signal_record:
                logger.warning(f"Signal record not found: {signal_id}")
                return False
            
            # Update with outcome
            signal_record.actual_outcome = actual_outcome
            signal_record.outcome_duration = outcome_duration
            signal_record.is_correct = is_correct
            if false_signal_reason:
                signal_record.false_signal_reason = false_signal_reason
            
            session.commit()
            
            # Update cache
            cache_key = f"{signal_record.signal_type}_{signal_record.signal_name}"
            if cache_key in self.performance_cache:
                del self.performance_cache[cache_key]
            
            logger.debug(f"Updated signal outcome for ID {signal_id}: {is_correct}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {e}")
            if session:
                session.rollback()
            return False
        finally:
            if session:
                session.close()
    
    def get_signal_performance(self, signal_type: str, signal_name: str, 
                              lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for a specific signal
        
        Args:
            signal_type: Type of signal (e.g., 'technical', 'ml', 'sentiment')
            signal_name: Name of signal (e.g., 'rsi_oversold', 'ml_bullish')
            lookback_days: Number of days to look back for performance data
            
        Returns:
            Dictionary with performance metrics
        """
        cache_key = f"{signal_type}_{signal_name}"
        cache_entry = self.performance_cache.get(cache_key)
        
        # Check cache
        if cache_entry and datetime.now() - cache_entry['timestamp'] < self.cache_expiry:
            return cache_entry['data']
        
        try:
            session = self.Session()
            
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=lookback_days)
            
            # Query signal performance records
            signals = session.query(SignalPerformance).filter(
                SignalPerformance.signal_type == signal_type,
                SignalPerformance.signal_name == signal_name,
                SignalPerformance.timestamp >= date_threshold
            ).all()
            
            if not signals:
                return {
                    'total_signals': 0,
                    'correct_signals': 0,
                    'accuracy': 0.0,
                    'false_signal_rate': 0.0,
                    'avg_strength': 0.0,
                    'avg_confidence': 0.0,
                    'avg_outcome': 0.0,
                    'common_false_reasons': []
                }
            
            # Calculate metrics
            total_signals = len(signals)
            correct_signals = sum(1 for s in signals if s.is_correct)
            accuracy = correct_signals / total_signals if total_signals > 0 else 0.0
            false_signal_rate = 1.0 - accuracy
            
            avg_strength = np.mean([s.signal_strength for s in signals if s.signal_strength is not None])
            avg_confidence = np.mean([s.signal_confidence for s in signals if s.signal_confidence is not None])
            avg_outcome = np.mean([s.actual_outcome for s in signals if s.actual_outcome is not None])
            
            # Get common false signal reasons
            false_reasons = [s.false_signal_reason for s in signals 
                           if not s.is_correct and s.false_signal_reason]
            common_false_reasons = list(set(false_reasons)) if false_reasons else []
            
            performance_data = {
                'total_signals': total_signals,
                'correct_signals': correct_signals,
                'accuracy': accuracy,
                'false_signal_rate': false_signal_rate,
                'avg_strength': avg_strength,
                'avg_confidence': avg_confidence,
                'avg_outcome': avg_outcome,
                'common_false_reasons': common_false_reasons
            }
            
            # Update cache
            self.performance_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': performance_data
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {
                'total_signals': 0,
                'correct_signals': 0,
                'accuracy': 0.0,
                'false_signal_rate': 0.0,
                'avg_strength': 0.0,
                'avg_confidence': 0.0,
                'avg_outcome': 0.0,
                'common_false_reasons': []
            }
        finally:
            if session:
                session.close()
    
    def get_false_signal_analysis(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze false signals across all signal types to identify patterns
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with false signal analysis
        """
        try:
            session = self.Session()
            
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=lookback_days)
            
            # Query false signals
            false_signals = session.query(SignalPerformance).filter(
                SignalPerformance.is_correct == False,
                SignalPerformance.timestamp >= date_threshold
            ).all()
            
            if not false_signals:
                return {
                    'total_false_signals': 0,
                    'false_signal_rate': 0.0,
                    'by_signal_type': {},
                    'by_market_regime': {},
                    'by_volatility_regime': {},
                    'common_reasons': []
                }
            
            total_signals = session.query(SignalPerformance).filter(
                SignalPerformance.timestamp >= date_threshold
            ).count()
            
            false_signal_rate = len(false_signals) / total_signals if total_signals > 0 else 0.0
            
            # Group by signal type
            by_signal_type = {}
            for signal in false_signals:
                signal_type = signal.signal_type
                if signal_type not in by_signal_type:
                    by_signal_type[signal_type] = 0
                by_signal_type[signal_type] += 1
            
            # Group by market regime
            by_market_regime = {}
            for signal in false_signals:
                regime = signal.market_regime
                if regime not in by_market_regime:
                    by_market_regime[regime] = 0
                by_market_regime[regime] += 1
            
            # Group by volatility regime
            by_volatility_regime = {}
            for signal in false_signals:
                vol_regime = signal.volatility_regime
                if vol_regime not in by_volatility_regime:
                    by_volatility_regime[vol_regime] = 0
                by_volatility_regime[vol_regime] += 1
            
            # Get common reasons
            reasons = [s.false_signal_reason for s in false_signals if s.false_signal_reason]
            common_reasons = list(set(reasons)) if reasons else []
            
            return {
                'total_false_signals': len(false_signals),
                'false_signal_rate': false_signal_rate,
                'by_signal_type': by_signal_type,
                'by_market_regime': by_market_regime,
                'by_volatility_regime': by_volatility_regime,
                'common_reasons': common_reasons
            }
            
        except Exception as e:
            logger.error(f"Error in false signal analysis: {e}")
            return {
                'total_false_signals': 0,
                'false_signal_rate': 0.0,
                'by_signal_type': {},
                'by_market_regime': {},
                'by_volatility_regime': {},
                'common_reasons': []
            }
        finally:
            if session:
                session.close()
    
    def get_adaptive_thresholds(self, signal_type: str, signal_name: str) -> Dict[str, float]:
        """
        Get adaptive thresholds based on signal performance to reduce false signals
        
        Args:
            signal_type: Type of signal
            signal_name: Name of signal
            
        Returns:
            Dictionary with adaptive thresholds
        """
        performance = self.get_signal_performance(signal_type, signal_name)
        
        # Adjust thresholds based on performance
        base_strength_threshold = 0.5
        base_confidence_threshold = 0.7
        
        # If false signal rate is high, increase thresholds
        if performance['false_signal_rate'] > 0.3:
            base_strength_threshold *= 1.5
            base_confidence_threshold *= 1.2
        elif performance['false_signal_rate'] < 0.1:
            # If performance is good, we can be more lenient
            base_strength_threshold *= 0.8
            base_confidence_threshold *= 0.9
        
        # Cap thresholds to reasonable values
        base_strength_threshold = min(base_strength_threshold, 0.9)
        base_confidence_threshold = min(base_confidence_threshold, 0.95)
        
        return {
            'strength_threshold': base_strength_threshold,
            'confidence_threshold': base_confidence_threshold
        }


# Global instance
_signal_tracker = None

def get_signal_tracker(db_path: str = None) -> SignalTracker:
    """Get the global signal tracker instance"""
    global _signal_tracker
    if _signal_tracker is None:
        _signal_tracker = SignalTracker(db_path)
    return _signal_tracker