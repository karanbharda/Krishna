"""
Production-Level Decision Audit Trail
Complete decision tracking and analysis
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DecisionRecord:
    """Complete decision record for audit trail"""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    confidence: float
    
    # Input data
    market_context: Dict[str, Any]
    signal_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    
    # Decision process
    signal_consensus: float
    adaptive_thresholds: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    # Final decision
    final_reasoning: str
    execution_details: Dict[str, Any]
    
    # Performance tracking
    performance_tracking_id: str
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[Dict[str, Any]] = None
    performance_score: Optional[float] = None

@dataclass
class PerformanceOutcome:
    """Performance outcome for decision tracking"""
    decision_id: str
    tracking_id: str
    outcome_timestamp: datetime
    profit_loss: float
    profit_loss_pct: float
    holding_period_hours: float
    market_movement: float
    execution_quality: float
    outcome_classification: str  # 'profitable', 'loss', 'breakeven'

class DecisionAuditTrail:
    """Production-level decision audit trail system"""
    
    def __init__(self, storage_path: str = "data/audit_trail"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.decisions_file = self.storage_path / "decisions.jsonl"
        self.performance_file = self.storage_path / "performance.jsonl"
        self.analytics_file = self.storage_path / "analytics.json"
        
        # In-memory cache for recent decisions
        self.recent_decisions = []
        self.max_cache_size = 1000
        
        # Performance tracking
        self.performance_outcomes = []
        self.analytics_cache = {}
        self.last_analytics_update = None

        # Critical Fix: Don't create async tasks in constructor
        self._initialization_complete = False

    async def initialize(self) -> None:
        """Critical Fix: Proper async initialization"""
        if not self._initialization_complete:
            try:
                await self._load_recent_decisions()
                self._initialization_complete = True
                logger.info("Decision audit trail initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing decision audit trail: {e}")
                raise

    async def log_decision(self, decision_data: Dict[str, Any]) -> str:
        """Log a complete trading decision"""
        
        decision_id = str(uuid.uuid4())
        tracking_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create decision record
        decision_record = DecisionRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            symbol=decision_data.get('symbol', ''),
            action=decision_data.get('action', 'HOLD'),
            quantity=decision_data.get('quantity', 0),
            confidence=decision_data.get('confidence', 0.0),
            
            market_context=decision_data.get('market_context', {}),
            signal_data=decision_data.get('signal_data', {}),
            portfolio_state=decision_data.get('portfolio_state', {}),
            
            signal_consensus=decision_data.get('signal_consensus', 0.0),
            adaptive_thresholds=decision_data.get('adaptive_thresholds', {}),
            risk_assessment=decision_data.get('risk_assessment', {}),
            
            final_reasoning=decision_data.get('final_reasoning', ''),
            execution_details=decision_data.get('execution_details', {}),
            
            performance_tracking_id=tracking_id
        )
        
        # Add to cache
        self.recent_decisions.append(decision_record)
        if len(self.recent_decisions) > self.max_cache_size:
            self.recent_decisions = self.recent_decisions[-self.max_cache_size:]
        
        # Write to persistent storage
        await self._write_decision_to_file(decision_record)
        
        logger.info(f"Decision logged: {decision_id} - {decision_record.action} {decision_record.symbol}")
        
        return decision_id
    
    async def log_performance_outcome(self, tracking_id: str, outcome_data: Dict[str, Any]) -> bool:
        """Log performance outcome for a decision"""
        
        try:
            # Find the original decision
            decision_record = await self._find_decision_by_tracking_id(tracking_id)
            if not decision_record:
                logger.warning(f"Decision not found for tracking ID: {tracking_id}")
                return False
            
            # Create performance outcome
            outcome = PerformanceOutcome(
                decision_id=decision_record.decision_id,
                tracking_id=tracking_id,
                outcome_timestamp=datetime.now(),
                profit_loss=outcome_data.get('profit_loss', 0.0),
                profit_loss_pct=outcome_data.get('profit_loss_pct', 0.0),
                holding_period_hours=outcome_data.get('holding_period_hours', 0.0),
                market_movement=outcome_data.get('market_movement', 0.0),
                execution_quality=outcome_data.get('execution_quality', 0.0),
                outcome_classification=self._classify_outcome(outcome_data.get('profit_loss', 0.0))
            )
            
            # Update decision record with outcome
            decision_record.actual_outcome = asdict(outcome)
            decision_record.performance_score = self._calculate_performance_score(outcome, decision_record)
            
            # Add to performance tracking
            self.performance_outcomes.append(outcome)
            
            # Write to persistent storage
            await self._write_performance_to_file(outcome)
            await self._update_decision_with_outcome(decision_record)
            
            logger.info(f"Performance outcome logged: {tracking_id} - {outcome.outcome_classification}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging performance outcome: {e}")
            return False
    
    async def get_decision_history(self, symbol: str = None, days: int = 30, 
                                  action: str = None) -> List[DecisionRecord]:
        """Get decision history with optional filters"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_decisions = []
        for decision in self.recent_decisions:
            # Apply filters
            if decision.timestamp < cutoff_date:
                continue
            if symbol and decision.symbol != symbol:
                continue
            if action and decision.action != action:
                continue
            
            filtered_decisions.append(decision)
        
        # Sort by timestamp (most recent first)
        filtered_decisions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_decisions
    
    async def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        # Check if we need to update analytics cache
        if (self.last_analytics_update is None or 
            datetime.now() - self.last_analytics_update > timedelta(hours=1)):
            await self._update_analytics_cache(days)
        
        return self.analytics_cache.get(f'analytics_{days}', {})
    
    async def get_decision_patterns(self, min_occurrences: int = 5) -> Dict[str, Any]:
        """Identify patterns in decision making"""
        
        patterns = {
            'signal_patterns': {},
            'risk_patterns': {},
            'market_patterns': {},
            'performance_patterns': {}
        }
        
        decisions_with_outcomes = [d for d in self.recent_decisions if d.actual_outcome]
        
        if len(decisions_with_outcomes) < min_occurrences:
            return patterns
        
        # Analyze signal patterns
        for decision in decisions_with_outcomes:
            signal_strength = decision.signal_consensus
            outcome = decision.actual_outcome
            
            # Categorize signal strength
            if signal_strength > 0.8:
                strength_category = 'very_strong'
            elif signal_strength > 0.6:
                strength_category = 'strong'
            elif signal_strength > 0.4:
                strength_category = 'moderate'
            else:
                strength_category = 'weak'
            
            if strength_category not in patterns['signal_patterns']:
                patterns['signal_patterns'][strength_category] = {
                    'count': 0,
                    'profitable': 0,
                    'avg_return': 0.0,
                    'total_return': 0.0
                }
            
            pattern = patterns['signal_patterns'][strength_category]
            pattern['count'] += 1
            pattern['total_return'] += outcome['profit_loss_pct']
            
            if outcome['profit_loss'] > 0:
                pattern['profitable'] += 1
            
            pattern['avg_return'] = pattern['total_return'] / pattern['count']
            pattern['win_rate'] = pattern['profitable'] / pattern['count']
        
        # Analyze risk patterns
        for decision in decisions_with_outcomes:
            risk_score = decision.risk_assessment.get('composite_risk_score', 0.5)
            outcome = decision.actual_outcome
            
            # Categorize risk level
            if risk_score > 0.7:
                risk_category = 'high'
            elif risk_score > 0.4:
                risk_category = 'medium'
            else:
                risk_category = 'low'
            
            if risk_category not in patterns['risk_patterns']:
                patterns['risk_patterns'][risk_category] = {
                    'count': 0,
                    'profitable': 0,
                    'avg_return': 0.0,
                    'total_return': 0.0
                }
            
            pattern = patterns['risk_patterns'][risk_category]
            pattern['count'] += 1
            pattern['total_return'] += outcome['profit_loss_pct']
            
            if outcome['profit_loss'] > 0:
                pattern['profitable'] += 1
            
            pattern['avg_return'] = pattern['total_return'] / pattern['count']
            pattern['win_rate'] = pattern['profitable'] / pattern['count']
        
        return patterns
    
    async def export_audit_data(self, start_date: datetime, end_date: datetime, 
                               format: str = 'json') -> str:
        """Export audit data for external analysis"""
        
        # Filter decisions by date range
        filtered_decisions = []
        for decision in self.recent_decisions:
            if start_date <= decision.timestamp <= end_date:
                filtered_decisions.append(asdict(decision))
        
        # Create export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_decisions': len(filtered_decisions),
            'decisions': filtered_decisions
        }
        
        # Generate filename
        filename = f"audit_export_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.{format}"
        export_path = self.storage_path / filename
        
        # Write export file
        if format == 'json':
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Audit data exported to: {export_path}")
        
        return str(export_path)
    
    async def _write_decision_to_file(self, decision: DecisionRecord):
        """Write decision to persistent storage"""
        try:
            decision_dict = asdict(decision)
            decision_dict['timestamp'] = decision.timestamp.isoformat()
            
            with open(self.decisions_file, 'a') as f:
                f.write(json.dumps(decision_dict, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing decision to file: {e}")
    
    async def _write_performance_to_file(self, outcome: PerformanceOutcome):
        """Write performance outcome to persistent storage"""
        try:
            outcome_dict = asdict(outcome)
            outcome_dict['outcome_timestamp'] = outcome.outcome_timestamp.isoformat()
            
            with open(self.performance_file, 'a') as f:
                f.write(json.dumps(outcome_dict, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing performance to file: {e}")
    
    async def _load_recent_decisions(self):
        """Load recent decisions from persistent storage"""
        try:
            if not self.decisions_file.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with open(self.decisions_file, 'r') as f:
                for line in f:
                    try:
                        decision_dict = json.loads(line.strip())
                        decision_dict['timestamp'] = datetime.fromisoformat(decision_dict['timestamp'])
                        
                        if decision_dict['timestamp'] >= cutoff_date:
                            decision = DecisionRecord(**decision_dict)
                            self.recent_decisions.append(decision)
                            
                    except Exception as e:
                        logger.warning(f"Error loading decision record: {e}")
                        continue
            
            logger.info(f"Loaded {len(self.recent_decisions)} recent decisions")
            
        except Exception as e:
            logger.error(f"Error loading recent decisions: {e}")
    
    async def _find_decision_by_tracking_id(self, tracking_id: str) -> Optional[DecisionRecord]:
        """Find decision by performance tracking ID"""
        for decision in self.recent_decisions:
            if decision.performance_tracking_id == tracking_id:
                return decision
        return None
    
    async def _update_decision_with_outcome(self, decision: DecisionRecord):
        """Update decision record with performance outcome"""
        # Update in cache
        for i, cached_decision in enumerate(self.recent_decisions):
            if cached_decision.decision_id == decision.decision_id:
                self.recent_decisions[i] = decision
                break
    
    def _classify_outcome(self, profit_loss: float) -> str:
        """Classify outcome as profitable, loss, or breakeven"""
        if profit_loss > 0.01:  # > 1% profit
            return 'profitable'
        elif profit_loss < -0.01:  # > 1% loss
            return 'loss'
        else:
            return 'breakeven'
    
    def _calculate_performance_score(self, outcome: PerformanceOutcome, decision: DecisionRecord) -> float:
        """Calculate performance score for decision"""
        # Combine multiple factors into performance score
        return_score = min(1.0, max(-1.0, outcome.profit_loss_pct / 0.1))  # Normalize to ±10%
        confidence_bonus = decision.confidence * 0.2  # Bonus for high confidence correct decisions
        execution_score = outcome.execution_quality * 0.1
        
        return (return_score + confidence_bonus + execution_score) / 1.3
    
    async def _update_analytics_cache(self, days: int):
        """Update analytics cache"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_decisions = [d for d in self.recent_decisions if d.timestamp >= cutoff_date]
            
            if not recent_decisions:
                self.analytics_cache[f'analytics_{days}'] = {}
                return
            
            # Calculate analytics
            total_decisions = len(recent_decisions)
            decisions_with_outcomes = [d for d in recent_decisions if d.actual_outcome]
            
            analytics = {
                'total_decisions': total_decisions,
                'decisions_with_outcomes': len(decisions_with_outcomes),
                'action_distribution': {},
                'confidence_distribution': {},
                'performance_metrics': {}
            }
            
            # Action distribution
            for decision in recent_decisions:
                action = decision.action
                analytics['action_distribution'][action] = analytics['action_distribution'].get(action, 0) + 1
            
            # Confidence distribution
            confidence_ranges = {'low': 0, 'medium': 0, 'high': 0}
            for decision in recent_decisions:
                if decision.confidence < 0.5:
                    confidence_ranges['low'] += 1
                elif decision.confidence < 0.8:
                    confidence_ranges['medium'] += 1
                else:
                    confidence_ranges['high'] += 1
            analytics['confidence_distribution'] = confidence_ranges
            
            # Performance metrics
            if decisions_with_outcomes:
                total_return = sum(d.actual_outcome['profit_loss_pct'] for d in decisions_with_outcomes)
                profitable_decisions = sum(1 for d in decisions_with_outcomes if d.actual_outcome['profit_loss'] > 0)
                
                analytics['performance_metrics'] = {
                    'total_return_pct': total_return,
                    'avg_return_pct': total_return / len(decisions_with_outcomes),
                    'win_rate': profitable_decisions / len(decisions_with_outcomes),
                    'total_trades_with_outcomes': len(decisions_with_outcomes)
                }
            
            self.analytics_cache[f'analytics_{days}'] = analytics
            self.last_analytics_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating analytics cache: {e}")
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit trail system metrics"""
        return {
            'total_decisions_cached': len(self.recent_decisions),
            'total_performance_outcomes': len(self.performance_outcomes),
            'storage_path': str(self.storage_path),
            'files_exist': {
                'decisions': self.decisions_file.exists(),
                'performance': self.performance_file.exists(),
                'analytics': self.analytics_file.exists()
            },
            'last_analytics_update': self.last_analytics_update.isoformat() if self.last_analytics_update else None
        }
