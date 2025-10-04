"""
Production-Level Async Signal Collector
Parallel processing for faster signal collection
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Signal data structure"""
    name: str
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    data: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    quality_score: float = 0.0
    error: Optional[str] = None

@dataclass
class SignalCollectionResult:
    """Result of signal collection"""
    signals: Dict[str, Signal]
    total_time: float
    successful_signals: int
    failed_signals: int
    collection_timestamp: datetime

class AsyncSignalCollector:
    """Production-level parallel signal collector"""
    
    def __init__(self, timeout_per_signal: float = 3.0, max_concurrent_signals: int = 10):
        self.timeout_per_signal = timeout_per_signal
        self.max_concurrent_signals = max_concurrent_signals
        self.signal_sources = {}
        self.performance_history = []
        # Security: Initialize semaphore safely to prevent race conditions
        self.semaphore = None
        self._semaphore_initialized = False
        
    def register_signal_source(self, name: str, collector_func: Callable[[str, Dict[str, Any]], Union[Dict[str, Any], Any]], weight: float = 1.0) -> None:
        """
        Register a signal collection function

        Code Quality: Comprehensive documentation

        Args:
            name: Unique identifier for the signal source
            collector_func: Function that collects signal data (sync or async)
            weight: Relative importance of this signal (0.0 to 1.0)

        Note:
            collector_func should accept (symbol: str, context: Dict) and return
            Dict with keys: strength, confidence, data
        """
        self.signal_sources[name] = {
            'func': collector_func,
            'weight': weight,
            'success_rate': 1.0,
            'avg_time': 1.0
        }
        
    async def _ensure_semaphore_initialized(self):
        """Safely initialize semaphore in async context"""
        if not self._semaphore_initialized:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_signals)
            self._semaphore_initialized = True

    async def collect_signals_parallel(self, symbol: str, context: Dict[str, Any]) -> SignalCollectionResult:
        """Collect all signals in parallel with timeout protection"""
        # Security: Ensure semaphore is properly initialized
        await self._ensure_semaphore_initialized()

        start_time = time.time()
        collection_timestamp = datetime.now()

        logger.info(f"Starting parallel signal collection for {symbol}")
        
        # Create tasks for parallel execution
        tasks = {}
        for source_name, source_info in self.signal_sources.items():
            task = asyncio.create_task(
                self._collect_signal_with_timeout(
                    source_name, 
                    source_info['func'], 
                    symbol, 
                    context
                )
            )
            tasks[source_name] = task
        
        # Wait for all signals with timeout protection
        signals = {}
        successful_signals = 0
        failed_signals = 0
        
        try:
            # Use asyncio.gather with return_exceptions=True to handle failures gracefully
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for source_name, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(f"Signal source {source_name} failed: {result}")
                    signals[source_name] = self._create_neutral_signal(source_name, str(result))
                    failed_signals += 1
                    self._update_source_performance(source_name, False, 0)
                else:
                    signals[source_name] = result
                    successful_signals += 1
                    self._update_source_performance(source_name, True, result.processing_time)
                    
        except Exception as e:
            logger.error(f"Critical error in signal collection: {e}")
            # Create neutral signals for all sources
            for source_name in self.signal_sources.keys():
                signals[source_name] = self._create_neutral_signal(source_name, f"Critical error: {e}")
                failed_signals += 1
        
        total_time = time.time() - start_time
        
        # Calculate quality scores
        self._calculate_signal_quality_scores(signals)
        
        result = SignalCollectionResult(
            signals=signals,
            total_time=total_time,
            successful_signals=successful_signals,
            failed_signals=failed_signals,
            collection_timestamp=collection_timestamp
        )
        
        # Store performance history
        self.performance_history.append({
            'timestamp': collection_timestamp,
            'total_time': total_time,
            'success_rate': successful_signals / len(self.signal_sources) if self.signal_sources else 0,
            'symbol': symbol
        })
        
        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        logger.info(f"Signal collection completed: {successful_signals} successful, {failed_signals} failed, {total_time:.2f}s total")
        
        return result
    
    async def _collect_signal_with_timeout(self, source_name: str, collector_func, symbol: str, context: Dict[str, Any]) -> Signal:
        """Collect individual signal with timeout protection and concurrency control"""
        start_time = time.time()

        # Use semaphore to limit concurrent signal collection
        try:
            async with self.semaphore:
                # Apply timeout to signal collection
                signal_data = await asyncio.wait_for(
                    self._safe_signal_collection(collector_func, symbol, context),
                    timeout=self.timeout_per_signal
                )

                processing_time = time.time() - start_time

                # Create signal object
                signal = Signal(
                    name=source_name,
                    strength=signal_data.get('strength', 0.0),
                    confidence=signal_data.get('confidence', 0.5),
                    data=signal_data.get('data', {}),
                    timestamp=datetime.now(),
                    processing_time=processing_time,
                    quality_score=0.0  # Will be calculated later
                )

                logger.info(f"Signal {source_name} collected successfully in {processing_time:.2f}s")
                return signal

        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.warning(f"Signal {source_name} timed out after {self.timeout_per_signal}s")
            return self._create_neutral_signal(source_name, f"Timeout after {self.timeout_per_signal}s")

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Signal {source_name} failed: {e}")
            logger.debug(f"Signal {source_name} error traceback: {traceback.format_exc()}")
            return self._create_neutral_signal(source_name, str(e))
    
    async def _safe_signal_collection(self, collector_func, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute signal collection function"""
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(collector_func):
                return await collector_func(symbol, context)
            else:
                # Priority 2: Robust event loop handling for sync functions
                try:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, collector_func, symbol, context)
                except RuntimeError:
                    # No running event loop - this should not happen in async context
                    logger.warning(f"No running event loop for sync function {collector_func.__name__}")
                    # Execute synchronously as fallback (not ideal but safe)
                    return collector_func(symbol, context)
        except Exception as e:
            raise e
    
    def _create_neutral_signal(self, source_name: str, error_msg: str) -> Signal:
        """Create neutral signal for failed sources"""
        return Signal(
            name=source_name,
            strength=0.0,  # Neutral
            confidence=0.0,  # No confidence
            data={},
            timestamp=datetime.now(),
            processing_time=0.0,
            quality_score=0.0,
            error=error_msg
        )
    
    def _calculate_signal_quality_scores(self, signals: Dict[str, Signal]):
        """Calculate quality scores for all signals"""
        for signal in signals.values():
            if signal.error:
                signal.quality_score = 0.0
            else:
                # Quality based on confidence, processing time, and source reliability
                source_reliability = self.signal_sources.get(signal.name, {}).get('success_rate', 0.5)
                time_penalty = max(0, 1 - (signal.processing_time / self.timeout_per_signal))
                
                signal.quality_score = (
                    signal.confidence * 0.4 +
                    source_reliability * 0.4 +
                    time_penalty * 0.2
                )
    
    def _update_source_performance(self, source_name: str, success: bool, processing_time: float):
        """Update performance metrics for signal source"""
        if source_name not in self.signal_sources:
            return
            
        source = self.signal_sources[source_name]
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        source['success_rate'] = (1 - alpha) * source['success_rate'] + alpha * (1.0 if success else 0.0)
        
        # Update average processing time
        if success and processing_time > 0:
            source['avg_time'] = (1 - alpha) * source['avg_time'] + alpha * processing_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.performance_history:
            return {}
        
        recent_history = self.performance_history[-20:]  # Last 20 collections
        
        return {
            'avg_collection_time': sum(h['total_time'] for h in recent_history) / len(recent_history),
            'avg_success_rate': sum(h['success_rate'] for h in recent_history) / len(recent_history),
            'source_performance': {
                name: {
                    'success_rate': info['success_rate'],
                    'avg_time': info['avg_time'],
                    'weight': info['weight']
                }
                for name, info in self.signal_sources.items()
            },
            'total_collections': len(self.performance_history)
        }
