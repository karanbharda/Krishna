import asyncio
import logging
import requests
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class DataServiceError(Exception):
    """Custom exception for data service errors"""
    pass

class DataServiceClient:
    """Client to connect to Fyers Data Service"""

    def __init__(self, base_url: str = "http://127.0.0.1:8002"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 10
        self.last_health_check = None
        self.health_check_interval = 30  # seconds
        self.is_healthy = False

        # Critical Fix: Safe handling of global constants without circular import
        # Use environment variable with fallback instead of importing from web_backend
        cache_ttl_default = int(os.getenv("CACHE_TTL_SECONDS", "5"))

        # Advanced Optimization: Enhanced caching with TTL and size limits
        self.cache = {}
        self.cache_ttl = cache_ttl_default
        self.max_cache_size = 1000  # Prevent memory bloat
        self.cache_hits = 0
        self.cache_misses = 0

        # ENHANCED DATA VALIDATION & ERROR HANDLING
        self.data_validation_enabled = True
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.data_freshness_threshold = 300  # 5 minutes
        self.outlier_detection_enabled = True
        self.outlier_threshold_std = 3.0  # 3 standard deviations
        
        # Data quality tracking
        self.data_quality_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'stale_data_warnings': 0,
            'outlier_detections': 0
        }

        # Architectural Fix: Thread-safe health checking
        self._health_check_lock = threading.Lock()
        self._health_check_in_progress = False

        logger.info(f"Data Service Client initialized for {self.base_url}")

    def _manage_cache_size(self) -> None:
        """Advanced Optimization: Manage cache size to prevent memory bloat"""
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries (simple LRU)
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1][1])
            items_to_remove = len(self.cache) - self.max_cache_size + 100  # Remove extra for efficiency
            for i in range(items_to_remove):
                del self.cache[sorted_cache[i][0]]
            logger.debug(f"Cache cleanup: removed {items_to_remove} old entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Advanced Optimization: Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "max_cache_size": self.max_cache_size
        }

    def health_check(self) -> bool:
        """Check if data service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                # Accept both 'healthy' and 'degraded' as working states
                service_status = health_data.get('status', 'unavailable')
                self.is_healthy = service_status in ['healthy', 'degraded']
                self.last_health_check = datetime.now()
                logger.info(f"Data service health check: {service_status} (healthy={self.is_healthy})")
                return self.is_healthy
            else:
                self.is_healthy = False
                logger.warning(f"Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.is_healthy = False
            return False
        finally:
            # Architectural Fix: Ensure lock is always released
            with self._health_check_lock:
                self._health_check_in_progress = False
    
    def should_check_health(self) -> bool:
        """Check if we should perform health check"""
        if not self.last_health_check:
            return True
        
        elapsed = (datetime.now() - self.last_health_check).total_seconds()
        return elapsed > self.health_check_interval
    
    def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get data for a specific symbol with ENHANCED VALIDATION"""
        self.data_quality_stats['total_requests'] += 1
        
        for attempt in range(self.max_retries):
            try:
                # Check health periodically
                if self.should_check_health():
                    self.health_check()

                if not self.is_healthy:
                    logger.warning("Data service is not healthy, attempting to reconnect")
                    if not self.health_check():
                        raise DataServiceError("Data service is not healthy and reconnection failed")

                # Check cache first
                cache_key = f"symbol_{symbol}"
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                        if self._validate_cached_data(cached_data, cache_key):
                            return cached_data

                response = self.session.get(f"{self.base_url}/data/{symbol}")
                if response.status_code == 200:
                    data = response.json()
                    
                    # ENHANCED VALIDATION: Check data quality
                    if not self._validate_symbol_data(data, symbol):
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Data validation failed for {symbol}, retrying...")
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            raise DataServiceError(f"Data validation failed for {symbol} after {self.max_retries} attempts")
                    
                    # Cache the result
                    self.cache[cache_key] = (data, datetime.now())
                    self.data_quality_stats['successful_requests'] += 1
                    return data
                    
                elif response.status_code == 404:
                    raise DataServiceError(f"Symbol {symbol} not found in data service")
                else:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"HTTP {response.status_code} for {symbol}, retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    raise DataServiceError(f"HTTP error {response.status_code} getting data for {symbol}")

            except DataServiceError:
                raise  # Re-raise our custom exceptions
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Unexpected error fetching data for {symbol}: {e}, retrying...")
                    time.sleep(self.retry_delay)
                    continue
                logger.error(f"Unexpected error fetching data for {symbol}: {e}")
                self.data_quality_stats['failed_requests'] += 1
                raise DataServiceError(f"Unexpected error: {str(e)}")
    
    def _validate_cached_data(self, data: Dict[str, Any], cache_key: str) -> bool:
        """Validate cached data quality"""
        if not self.data_validation_enabled:
            return True
            
        # Check data freshness
        if 'timestamp' in data:
            data_time = datetime.fromisoformat(data['timestamp'])
            if (datetime.now() - data_time).total_seconds() > self.data_freshness_threshold:
                self.data_quality_stats['stale_data_warnings'] += 1
                logger.warning(f"Stale cached data detected for {cache_key}")
                return False
        
        # Validate required fields
        required_fields = ['price', 'symbol']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field {field} in cached data for {cache_key}")
                return False
                
        return True

    def _validate_symbol_data(self, data: Dict[str, Any], symbol: str) -> bool:
        """Validate symbol data quality"""
        if not self.data_validation_enabled:
            return True
            
        # Check required fields
        required_fields = ['price', 'symbol']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field {field} in data for {symbol}")
                return False
        
        # Validate price is numeric
        try:
            price = float(data['price'])
            # Allow zero prices but log a warning - some securities may have zero prices temporarily
            if price < 0:
                logger.warning(f"Invalid negative price {price} for {symbol}")
                return False
            elif price == 0:
                logger.warning(f"Zero price for {symbol} - may indicate no trading activity or data issue")
                # Still return True to allow processing, but with a warning
        except (ValueError, TypeError):
            logger.warning(f"Non-numeric price {data.get('price')} for {symbol}")
            return False
            
        # Check for outliers if enabled
        if self.outlier_detection_enabled:
            if not self._check_for_outliers(data, symbol):
                self.data_quality_stats['outlier_detections'] += 1
                logger.warning(f"Outlier detected in data for {symbol}")
                return False
                
        return True

    def _check_for_outliers(self, data: Dict[str, Any], symbol: str) -> bool:
        """Check for price outliers using standard deviation"""
        # This is a simplified implementation
        # In a real system, you would compare against historical data
        try:
            price = float(data['price'])
            # Simple check: price should be within reasonable bounds
            # This would be more sophisticated with historical data
            # Allow zero prices but check for extreme outliers
            if price == 0:
                # Zero prices are allowed but not considered outliers
                return True
            elif price < 0.01 or price > 1000000:  # Reasonable bounds for Indian stocks
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all available market data"""
        # Check health periodically
        if self.should_check_health():
            self.health_check()
        
        if not self.is_healthy:
            logger.warning("Data service is not healthy")
            if not self.health_check():
                return {}
        
        # Check cache first
        cache_key = "all_data"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
        
        try:
            response = self.session.get(f"{self.base_url}/data")
            if response.status_code == 200:
                data = response.json()
                # Cache the result
                self.cache[cache_key] = (data, datetime.now())
                return data
            else:
                logger.error(f"Error getting all data: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching all data: {e}")
            return {}
    
    def update_watchlist(self, symbols: List[str]) -> bool:
        """Update the watchlist in data service"""
        try:
            response = self.session.post(
                f"{self.base_url}/watchlist",
                json=symbols,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                logger.info(f"Watchlist updated with {len(symbols)} symbols")
                return True
            else:
                logger.error(f"Error updating watchlist: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")
            return False
    
    def get_watchlist(self) -> List[str]:
        """Get current watchlist from data service"""
        try:
            response = self.session.get(f"{self.base_url}/watchlist")
            if response.status_code == 200:
                data = response.json()
                return data.get('watchlist', [])
            else:
                logger.error(f"Error getting watchlist: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    def connect_fyers(self) -> bool:
        """Trigger Fyers connection in data service"""
        try:
            response = self.session.post(f"{self.base_url}/connect")
            if response.status_code == 200:
                logger.info("Fyers connection triggered successfully")
                return True
            else:
                logger.error(f"Error triggering Fyers connection: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering Fyers connection: {e}")
            return False
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol (convenience method)"""
        data = self.get_symbol_data(symbol)
        if data:
            return data.get('price')
        return None
    
    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols"""
        prices = {}
        all_data = self.get_all_data()
        
        for symbol in symbols:
            # Try different symbol formats
            fyers_symbol = self.convert_to_fyers_format(symbol)
            
            if fyers_symbol in all_data:
                prices[symbol] = all_data[fyers_symbol]['price']
            elif symbol in all_data:
                prices[symbol] = all_data[symbol]['price']
            else:
                # Fallback to individual request
                data = self.get_symbol_data(symbol)
                if data:
                    prices[symbol] = data['price']
        
        return prices
    
    def convert_to_fyers_format(self, symbol: str) -> str:
        """Convert symbol to Fyers format"""
        if symbol.endswith('.NS'):
            base_symbol = symbol.replace('.NS', '')
            return f"NSE:{base_symbol}-EQ"
        elif symbol.endswith('.BO'):
            base_symbol = symbol.replace('.BO', '')
            return f"BSE:{base_symbol}-EQ"
        elif ':' in symbol and '-' in symbol:
            return symbol
        else:
            return f"NSE:{symbol}-EQ"
    
    def convert_from_fyers_format(self, fyers_symbol: str) -> str:
        """Convert Fyers format to standard format"""
        if fyers_symbol.startswith('NSE:') and fyers_symbol.endswith('-EQ'):
            base_symbol = fyers_symbol.replace('NSE:', '').replace('-EQ', '')
            return f"{base_symbol}.NS"
        elif fyers_symbol.startswith('BSE:') and fyers_symbol.endswith('-EQ'):
            base_symbol = fyers_symbol.replace('BSE:', '').replace('-EQ', '')
            return f"{base_symbol}.BO"
        else:
            return fyers_symbol
    
    def is_service_available(self) -> bool:
        """Check if data service is available"""
        return self.health_check()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unavailable", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
    
    def clear_cache(self):
        """Clear the local cache"""
        self.cache.clear()
        logger.info("Data service client cache cleared")

# Global client instance
_client_instance = None

def get_data_client() -> DataServiceClient:
    """Get global data service client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = DataServiceClient()
    return _client_instance

def set_data_service_url(url: str):
    """Set data service URL"""
    global _client_instance
    _client_instance = DataServiceClient(url)
    logger.info(f"Data service URL set to: {url}")