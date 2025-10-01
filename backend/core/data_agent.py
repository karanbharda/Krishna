import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
import csv

# Fix import paths
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_service_client import get_data_client

logger = logging.getLogger(__name__)

class DataAgent:
    def __init__(self, db_path=None):
        # FIXED: Use project root data directory
        if db_path is None:
            from pathlib import Path
            backend_dir = Path(__file__).resolve().parents[1]
            project_root = backend_dir.parent
            db_path = str(project_root / 'data' / 'market_cache.db')
        
        self.client = get_data_client()
        self.db_path = db_path
        self._init_db()
        self.scheduler = BackgroundScheduler()
        self._schedule_scan()
        logger.info("DataAgent initialized with scheduler")

    def _init_db(self):
        """Initialize SQLite database for persistent caching"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        logger.info("Market data cache database initialized")

    def _schedule_scan(self):
        """Schedule full market scan at 9:15 AM IST"""
        try:
            ist = pytz.timezone('Asia/Kolkata')
            self.scheduler.add_job(
                self.kickoff_scan,
                'cron',
                hour=9,
                minute=15,
                timezone=ist,
                id='market_scan'
            )
            self.scheduler.start()
            logger.info("Market scan scheduled for 9:15 AM IST")
        except Exception as e:
            logger.error(f"Failed to schedule market scan: {e}")

    def fetch_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch data for a batch of symbols in parallel"""
        batch_data = {}
        max_workers = min(10, len(symbols))  # RTX 3060 optimization
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_single_symbol, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data:
                        batch_data[symbol] = data
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
        
        logger.info(f"Fetched data for {len(batch_data)}/{len(symbols)} symbols")
        return batch_data

    def _fetch_single_symbol(self, symbol: str) -> Dict[str, Any]:
        """Fetch data for a single symbol with error handling"""
        try:
            return self.client.get_symbol_data(symbol)
        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    def kickoff_scan(self):
        """Kickoff full market scan for NSE/BSE stocks"""
        logger.info("Starting full market scan...")
        start_time = datetime.now()
        
        try:
            # Load universe from security_id_list.csv
            all_symbols = self._get_universe_symbols()
            logger.info(f"Loaded {len(all_symbols)} symbols for scanning")
            
            universe_data = {}
            batch_size = 50  # RTX 3060 optimization - smaller batches
            total_batches = (len(all_symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i+batch_size]
                batch_data = self.fetch_batch(batch)
                universe_data.update(batch_data)
                
                # Cache to SQLite
                self._cache_batch(batch_data)
                
                batch_num = i // batch_size + 1
                logger.info(f"Processed batch {batch_num}/{total_batches}: {len(batch_data)} symbols")
            
            # Save baseline - FIXED: Use project root logs directory
            from pathlib import Path
            backend_dir = Path(__file__).resolve().parents[1]
            project_root = backend_dir.parent
            logs_dir = project_root / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            date_str = datetime.now().strftime("%Y%m%d")
            
            baseline_data = {
                "timestamp": datetime.now().isoformat(),
                "scan_duration_seconds": (datetime.now() - start_time).total_seconds(),
                "total_symbols_requested": len(all_symbols),
                "successful_fetches": len(universe_data),
                "stocks": universe_data
            }
            
            with open(logs_dir / f"universe_open_{date_str}.json", 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            logger.info(f"Full market scan completed: {len(universe_data)} stocks in {(datetime.now() - start_time).total_seconds():.2f}s")
            return universe_data
            
        except Exception as e:
            logger.error(f"Market scan failed: {e}")
            return {}

    def _get_universe_symbols(self) -> List[str]:
        """Get list of all NSE/BSE symbols from security_id_list.csv"""
        symbols = []
        
        try:
            # Try to read from security_id_list.csv
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "security_id_list.csv")
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header if exists
                    for row in reader:
                        if row and len(row) > 0:
                            symbol = row[0].strip()
                            if symbol and not symbol.startswith('#'):
                                symbols.append(symbol)
                
                # Limit for testing - remove [:100] for production
                symbols = symbols[:100]  # RTX 3060 optimization for testing
                logger.info(f"Loaded {len(symbols)} symbols from security_id_list.csv")
                
        except Exception as e:
            logger.warning(f"Could not read security_id_list.csv: {e}")
        
        # Fallback to hardcoded list if CSV fails
        if not symbols:
            symbols = [
                "RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", 
                "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
                "HDFCBANK", "AXISBANK", "MARUTI", "HINDUNILVR", "WIPRO",
                "SUNPHARMA", "ONGC", "NTPC", "POWERGRID", "COALINDIA"
            ]
            logger.info(f"Using fallback symbol list: {len(symbols)} symbols")
        
        return symbols

    def _cache_batch(self, batch_data: Dict[str, Any]):
        """Cache batch to SQLite"""
        if not batch_data:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                for symbol, data in batch_data.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO market_data (symbol, data, timestamp) VALUES (?, ?, ?)",
                        (symbol, json.dumps(data), datetime.now())
                    )
            logger.debug(f"Cached {len(batch_data)} symbols to database")
        except Exception as e:
            logger.error(f"Failed to cache batch: {e}")

    def get_cached_data(self, symbol: str) -> Dict[str, Any]:
        """Get cached data for symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT data FROM market_data WHERE symbol = ?", (symbol,)).fetchone()
                return json.loads(row[0]) if row else {}
        except Exception as e:
            logger.error(f"Failed to get cached data for {symbol}: {e}")
            return {}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
                latest = conn.execute("SELECT MAX(timestamp) FROM market_data").fetchone()[0]
                return {
                    "cached_symbols": count,
                    "latest_update": latest,
                    "cache_file": self.db_path
                }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

# Global instance
data_agent = DataAgent()
