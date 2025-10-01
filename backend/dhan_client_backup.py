#!/usr/bin/env python3
"""
Dhan API Client for Live Trading Integration
Handles real-time market data, portfolio management, and order execution
"""

import requests
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class DhanAPIClient:
    """Dhan API client for live trading operations"""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.base_url = "https://api.dhan.co"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': access_token
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Validation rate limiting
        self.last_validation_time = 0
        self.validation_cache_duration = 300  # Cache validation for 5 minutes
        
        # Security ID cache for dynamic lookups
        self.security_id_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # Cache for 1 hour
        
        # Remove all hardcoded security ID mappings - will fetch dynamically from API
        # Manual security ID mappings - takes precedence over dynamic search
        # self.manual_security_id_mapping = {}  # Removed hardcoded mappings
        
        # Instrument data cache
        self.instrument_cache = {}
        self.instrument_cache_expiry = {}
        self.instrument_cache_duration = 86400  # Cache instruments for 24 hours
        
        # Daily instrument master (Dhan's recommended approach)
        self.daily_instruments_df = None
        self.daily_instruments_last_updated = None
        self.daily_instruments_cache_duration = 3600  # Refresh every hour
        self.daily_refresh_hour = 8  # Refresh at 8 AM daily
        
        # Dhan instrument master URLs (no auth required)
        self.compact_url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        self.detailed_url = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"
        
        # Data directory for caching - FIXED: Use project root data directory
        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent
        self.data_dir = project_root / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for corrected security IDs
        self._init_security_id_db()
        
        logger.info("Dhan API client initialized with daily instrument master integration")
    
    def _init_security_id_db(self):
        """Initialize database for storing corrected security IDs"""
        try:
            db_path = self.data_dir / "security_ids.db"
            self.security_db = sqlite3.connect(db_path, check_same_thread=False)
            cursor = self.security_db.cursor()
            
            # Create table for corrected security IDs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrected_security_ids (
                    symbol TEXT PRIMARY KEY,
                    security_id TEXT NOT NULL,
                    exchange_segment TEXT DEFAULT 'NSE_EQ',
                    last_validated TIMESTAMP,
                    validation_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create index for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_exchange 
                ON corrected_security_ids (symbol, exchange_segment)
            ''')
            
            self.security_db.commit()
            logger.info("Security ID database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize security ID database: {e}")
            self.security_db = None
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make API request with error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=data)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle specific status codes more gracefully
            if response.status_code == 500 and endpoint == '/v2/holdings':
                # 500 error on holdings usually means empty portfolio
                logger.debug("Holdings endpoint returned 500 - likely empty portfolio")
                return {"data": []}

            # Enhanced error handling for 400 errors
            if response.status_code == 400:
                try:
                    error_detail = response.json()
                    logger.error(f"Dhan API 400 Error - {endpoint}: {error_detail}")
                    if data:
                        logger.error(f"Request data that caused 400 error: {data}")
                    raise Exception(f"Dhan API error (400): {error_detail}")
                except ValueError:
                    # JSON decode error
                    logger.error(f"Dhan API 400 Error - {endpoint}: {response.text}")
                    if data:
                        logger.error(f"Request data that caused 400 error: {data}")
                    raise Exception(f"Dhan API error (400): {response.text}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            # More specific error handling
            if "500" in str(e) and endpoint == '/v2/holdings':
                logger.info("Holdings API returned 500 - empty portfolio (normal for new accounts)")
                return {"data": []}
            else:
                logger.error(f"Dhan API request failed: {e}")
                if data:
                    logger.error(f"Request data: {data}")
                raise Exception(f"Dhan API error: {str(e)}")
    
    def get_profile(self) -> Dict:
        """Get user profile and account information"""
        try:
            return self._make_request('GET', '/v2/profile')
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return {}
    
    def get_funds(self) -> Dict:
        """Get account funds and margin information"""
        try:
            response = self._make_request('GET', '/v2/fundlimit')
            logger.debug(f"Dhan Account Funds Response: {json.dumps(response, indent=2)}")
            return response
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {"availabelBalance": 0, "sodLimit": 0}
    
    def get_holdings(self) -> List[Dict]:
        """Get current stock holdings"""
        try:
            response = self._make_request('GET', '/v2/holdings')
            
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('data', [])
            else:
                logger.warning(f"Unexpected response format from holdings: {type(response)}")
                return []
                
        except Exception as e:
            # More graceful error handling for holdings
            if "500" in str(e):
                logger.info("No holdings found (empty portfolio) - this is normal for new accounts")
            else:
                logger.error(f"Failed to get holdings: {e}")
            return []
    
    def get_positions(self) -> List[Dict]:
        """Get current trading positions"""
        try:
            response = self._make_request('GET', '/v2/positions')
            
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('data', [])
            else:
                logger.warning(f"Unexpected response format from positions: {type(response)}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get order history"""
        try:
            response = self._make_request('GET', '/v2/orders')
            
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict):
                return response.get('data', [])
            else:
                logger.warning(f"Unexpected response format from orders: {type(response)}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_order_by_id(self, order_id: str) -> Dict:
        """Get specific order by ID"""
        try:
            response = self._make_request('GET', f'/v2/orders/{order_id}')
            
            # Handle different response formats
            if isinstance(response, dict):
                return response
            else:
                logger.warning(f"Unexpected response format from order {order_id}: {type(response)}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return {}
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Get real-time quote for a symbol"""
        try:
            # Get numeric security ID instead of symbol
            security_id = self.get_security_id(symbol)
            
            data = {
                "securityId": security_id,
                "exchangeSegment": "NSE_EQ"
            }
            
            response = self._make_request('POST', '/v2/marketdata/quote', data)
            return response.get('data', {})
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, timeframe: str = "1D", 
                          from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Get numeric security ID
            security_id = self.get_security_id(symbol)
            
            # Default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            data = {
                "securityId": security_id,
                "exchangeSegment": "NSE_EQ",
                "timeframe": timeframe,
                "fromDate": from_date,
                "toDate": to_date
            }
            
            response = self._make_request('POST', '/v2/charts/historical', data)
            
            # Convert to DataFrame
            if 'data' in response and response['data']:
                df = pd.DataFrame(response['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_corrected_security_id(self, symbol: str, exchange: str = "NSE_EQ") -> Optional[str]:
        """Get corrected security ID from database if available"""
        try:
            if not self.security_db:
                return None
                
            cursor = self.security_db.cursor()
            cursor.execute('''
                SELECT security_id FROM corrected_security_ids 
                WHERE symbol = ? AND exchange_segment = ? AND validation_count > 0
                ORDER BY last_validated DESC LIMIT 1
            ''', (symbol.upper(), exchange))
            
            result = cursor.fetchone()
            if result:
                logger.info(f"✅ Found corrected security ID in database for {symbol}: {result[0]}")
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Error retrieving corrected security ID for {symbol}: {e}")
            return None
    
    def _store_corrected_security_id(self, symbol: str, security_id: str, exchange: str = "NSE_EQ"):
        """Store corrected security ID in database"""
        try:
            if not self.security_db:
                return
                
            cursor = self.security_db.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO corrected_security_ids 
                (symbol, security_id, exchange_segment, last_validated, validation_count)
                VALUES (?, ?, ?, ?, 
                    CASE 
                        WHEN EXISTS(SELECT 1 FROM corrected_security_ids WHERE symbol = ? AND exchange_segment = ?) 
                        THEN (SELECT validation_count FROM corrected_security_ids WHERE symbol = ? AND exchange_segment = ?) + 1
                        ELSE 1
                    END
                )
            ''', (symbol.upper(), security_id, exchange, datetime.now(), 
                  symbol.upper(), exchange, symbol.upper(), exchange))
            
            self.security_db.commit()
            logger.info(f"✅ Stored corrected security ID in database: {symbol} -> {security_id}")
        except Exception as e:
            logger.error(f"Error storing corrected security ID for {symbol}: {e}")
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached security ID is still valid"""
        if symbol not in self.security_id_cache:
            return False
        
        if symbol not in self.cache_expiry:
            return False
        
        return time.time() < self.cache_expiry[symbol]
    
    def _cache_security_id(self, symbol: str, security_id: str):
        """Cache security ID for future use"""
        self.security_id_cache[symbol] = security_id
        self.cache_expiry[symbol] = time.time() + self.cache_duration
        logger.debug(f"Cached security ID for {symbol}: {security_id}")
    
    def _get_cached_security_id(self, symbol: str) -> Optional[str]:
        """Get security ID from cache if valid"""
        if self._is_cache_valid(symbol):
            logger.debug(f"Using cached security ID for {symbol}: {self.security_id_cache[symbol]}")
            return self.security_id_cache[symbol]
        return None
    
    def _get_security_id_from_nse_database(self, symbol: str) -> Optional[str]:
        """Get security ID from comprehensive NSE symbol database - now returns None as we removed hardcoded mappings"""
        # Removed all hardcoded NSE symbol to security ID mapping
        # This includes most actively traded NSE stocks
        # Now we fetch all data dynamically from https://api.dhan.co/v2/instrument/NSE_EQ
        logger.debug("Using dynamic fetching instead of hardcoded NSE database mappings")
        return None
    
    def _should_refresh_daily_instruments(self) -> bool:
        """Check if we need daily refresh of instrument master"""
        if not self.daily_instruments_last_updated:
            return True
        
        now = datetime.now()
        last_update_date = self.daily_instruments_last_updated.date()
        
        # If it's a new day and past refresh hour, update
        if (now.date() > last_update_date and 
            now.hour >= self.daily_refresh_hour):
            return True
        
        # If cache is expired, update
        if (now - self.daily_instruments_last_updated).total_seconds() > self.daily_instruments_cache_duration:
            return True
        
        return False
    
    def _download_daily_instrument_master(self, use_detailed: bool = True) -> bool:
        """Download latest instrument master from Dhan (no auth required)"""
        try:
            url = self.detailed_url if use_detailed else self.compact_url
            file_name = "dhan_instruments_detailed.csv" if use_detailed else "dhan_instruments_compact.csv"
            file_path = self.data_dir / file_name
            
            logger.info(f"📥 Downloading daily instrument master from Dhan...")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Daily instrument master downloaded: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download daily instrument master: {e}")
            return False
    
    def _load_daily_instrument_master(self, force_refresh: bool = False) -> bool:
        """Load daily instrument master into DataFrame with smart caching"""
        try:
            # Check if refresh is needed
            if not force_refresh and not self._should_refresh_daily_instruments():
                if self.daily_instruments_df is not None:
                    logger.debug("Using cached daily instrument data")
                    return True
            
            # Try to download fresh data
            download_success = self._download_daily_instrument_master(use_detailed=True)
            
            # Load from file (either fresh or existing)
            detailed_file = self.data_dir / "dhan_instruments_detailed.csv"
            compact_file = self.data_dir / "dhan_instruments_compact.csv"
            
            file_to_load = detailed_file if detailed_file.exists() else compact_file
            if not file_to_load.exists():
                logger.error("No daily instrument master file available")
                return False
            
            # Load CSV into DataFrame with proper error handling
            logger.info(f"📊 Loading daily instrument data from {file_to_load}")
            
            # First, try to determine the file structure
            try:
                # Read a few lines to understand the format
                with open(file_to_load, 'r') as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                
                # Check if it has headers by looking for common column names
                has_headers = any(col in first_line.upper() for col in 
                                ['TRADINGSYMBOL', 'SECURITYID', 'EXCHANGE', 'SYMBOL'])
                
                if has_headers:
                    # File has headers
                    self.daily_instruments_df = pd.read_csv(file_to_load, low_memory=False)
                else:
                    # File doesn't have headers - define them based on Dhan format
                    # Based on Dhan CSV format: Exchange,Segment,SecurityId,ISIN,InstrumentType,etc.
                    column_names = [
                        'Exchange', 'Segment', 'SecurityId', 'ISIN', 'InstrumentType',
                        'Unknown1', 'TradingSymbol', 'CompanyName', 'FullName', 'InstrumentName',
                        'Unknown2', 'LotSize', 'ExpiryDate', 'StrikePrice', 'OptionType',
                        'TickSize', 'Unknown3', 'Unknown4', 'Unknown5', 'Unknown6', 'Unknown7',
                        'Unknown8', 'Unknown9', 'Unknown10', 'Unknown11', 'Unknown12', 'Unknown13',
                        'Unknown14', 'Unknown15', 'Unknown16', 'Unknown17', 'Unknown18', 'Unknown19',
                        'Unknown20', 'Unknown21', 'Unknown22', 'Unknown23', 'Unknown24', 'Unknown25',
                        'Unknown26', 'Unknown27'
                    ]
                    
                    self.daily_instruments_df = pd.read_csv(
                        file_to_load, 
                        names=column_names,
                        low_memory=False,
                        dtype=str  # Read all as strings to avoid type issues
                    )
                
                # Ensure we have the required columns
                if 'TradingSymbol' not in self.daily_instruments_df.columns:
                    # Try alternative column names
                    for col in self.daily_instruments_df.columns:
                        if 'symbol' in col.lower() or 'trading' in col.lower():
                            self.daily_instruments_df.rename(columns={col: 'TradingSymbol'}, inplace=True)
                            break
                    else:
                        logger.error(f"Could not find trading symbol column. Available columns: {list(self.daily_instruments_df.columns)}")
                        return False
                
                if 'SecurityId' not in self.daily_instruments_df.columns:
                    # Try alternative column names
                    for col in self.daily_instruments_df.columns:
                        if 'security' in col.lower() and 'id' in col.lower():
                            self.daily_instruments_df.rename(columns={col: 'SecurityId'}, inplace=True)
                            break
                    else:
                        logger.error(f"Could not find security ID column. Available columns: {list(self.daily_instruments_df.columns)}")
                        return False
                
                # Add exchange segment column if missing
                if 'ExchangeSegment' not in self.daily_instruments_df.columns:
                    if 'Exchange' in self.daily_instruments_df.columns and 'Segment' in self.daily_instruments_df.columns:
                        self.daily_instruments_df['ExchangeSegment'] = (
                            self.daily_instruments_df['Exchange'].astype(str) + '_' + 
                            self.daily_instruments_df['Segment'].astype(str)
                        )
                    else:
                        # Default to NSE_EQ for equity instruments
                        self.daily_instruments_df['ExchangeSegment'] = 'NSE_EQ'
                
                # Normalize trading symbols for reliable lookup
                self.daily_instruments_df['TradingSymbol'] = (
                    self.daily_instruments_df['TradingSymbol']
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )
                
                # Filter for equity instruments only (to make processing faster)
                if 'InstrumentType' in self.daily_instruments_df.columns:
                    equity_mask = self.daily_instruments_df['InstrumentType'].isin(['EQ', 'EQUITY', 'EQT'])
                    self.daily_instruments_df = self.daily_instruments_df[equity_mask]
                
                self.daily_instruments_last_updated = datetime.now()
                
                logger.info(f"✅ Loaded {len(self.daily_instruments_df)} instruments from daily master")
                
                # Log available exchanges for debugging
                if 'ExchangeSegment' in self.daily_instruments_df.columns:
                    unique_exchanges = self.daily_instruments_df['ExchangeSegment'].unique()
                    logger.info(f"Available exchanges: {unique_exchanges[:10]}")
                
                return True
                
            except Exception as parse_error:
                logger.error(f"Error parsing CSV file: {parse_error}")
                logger.error(f"First line: {first_line[:100]}...")  # Show first 100 chars
                return False
            
        except Exception as e:
            logger.error(f"Failed to load daily instrument master: {e}")
            return False
    
    def _get_security_id_from_daily_master(self, symbol: str, exchange: str = "NSE_EQ") -> Optional[str]:
        """Get security ID from daily instrument master (Dhan's recommended approach)"""
        try:
            # Check database first for previously corrected IDs
            corrected_id = self._get_corrected_security_id(symbol, exchange)
            if corrected_id:
                return corrected_id
            
            # Special handling for known problematic symbols
            # Remove hardcoded security ID for RAJOOENG - using dynamic fetching instead
            # if symbol.upper() == "RAJOOENG":
            #     # Use the manually corrected ID for RAJOOENG
            #     manual_id = "539297"
            #     logger.info(f"Using manual correction for RAJOOENG: {manual_id}")
            #     # Validate and store if valid
            #     if self._validate_security_id(manual_id, symbol):
            #         self._store_corrected_security_id(symbol, manual_id)
            #         return manual_id
            #     else:
            #         logger.warning(f"Manual correction ID {manual_id} for RAJOOENG is also invalid")
            
            # Ensure we have loaded daily instrument data
            if not self._load_daily_instrument_master():
                logger.error("Failed to load daily instrument master")
                return None
            
            # Clean symbol (remove .NS, .BO suffixes)
            clean_symbol = (symbol.upper()
                          .replace(".NS", "")
                          .replace(".BO", "") 
                          .replace(".BSE", "")
                          .strip())
            
            # Look up in DataFrame - exact match first
            match = self.daily_instruments_df[
                (self.daily_instruments_df['TradingSymbol'] == clean_symbol) &
                (self.daily_instruments_df['ExchangeSegment'] == exchange)
            ]
            
            if not match.empty:
                security_id = str(match.iloc[0]['SecurityId'])
                logger.info(f"✅ Found in daily master: {clean_symbol} -> {security_id}")
                
                # Validate the security ID with the live API
                try:
                    if self._validate_security_id(security_id, clean_symbol):
                        return security_id
                    else:
                        logger.warning(f"❌ Security ID {security_id} from daily master is INVALID for {clean_symbol}")
                        # Try to find a valid security ID using instrument search as fallback
                        search_results = self.search_instruments_by_symbol(clean_symbol, exchange)
                        if search_results:
                            # Find the exact match in search results
                            for instrument in search_results:
                                if instrument.get('tradingSymbol', '').upper() == clean_symbol:
                                    new_security_id = str(instrument.get('securityId', ''))
                                    if new_security_id and new_security_id != security_id:
                                        logger.info(f"✅ Found valid security ID via instrument search: {clean_symbol} -> {new_security_id}")
                                        # Validate the new security ID
                                        if self._validate_security_id(new_security_id, clean_symbol):
                                            # Store the corrected ID
                                            self._store_corrected_security_id(clean_symbol, new_security_id)
                                            return new_security_id
                        # Don't use invalid security IDs - try other methods instead
                        return None
                except Exception as validation_error:
                    logger.warning(f"⚠️ Security ID {security_id} validation failed with error: {validation_error}")
                    # For API errors, we'll still return the ID but with a warning
                    # This allows the system to try using it while logging the issue
                    return security_id
            
            # Try partial matching if exact match fails
            partial_matches = self.daily_instruments_df[
                self.daily_instruments_df['TradingSymbol'].str.contains(clean_symbol, na=False) &
                (self.daily_instruments_df['ExchangeSegment'] == exchange)
            ]
            
            if not partial_matches.empty:
                # Get the best match (exact substring match preferred)
                best_match = partial_matches[partial_matches['TradingSymbol'] == clean_symbol]
                if best_match.empty:
                    best_match = partial_matches.iloc[[0]]  # Take first partial match
                
                security_id = str(best_match.iloc[0]['SecurityId'])
                actual_symbol = best_match.iloc[0]['TradingSymbol']
                logger.info(f"✅ Found partial match in daily master: {clean_symbol} -> {actual_symbol} -> {security_id}")
                
                # Validate the security ID with the live API
                try:
                    if self._validate_security_id(security_id, actual_symbol):
                        return security_id
                    else:
                        logger.warning(f"❌ Security ID {security_id} from daily master is INVALID for {actual_symbol}")
                        # Try to find a valid security ID using instrument search as fallback
                        search_results = self.search_instruments_by_symbol(actual_symbol, exchange)
                        if search_results:
                            # Find the exact match in search results
                            for instrument in search_results:
                                if instrument.get('tradingSymbol', '').upper() == actual_symbol:
                                    new_security_id = str(instrument.get('securityId', ''))
                                    if new_security_id and new_security_id != security_id:
                                        logger.info(f"✅ Found valid security ID via instrument search: {actual_symbol} -> {new_security_id}")
                                        # Validate the new security ID
                                        if self._validate_security_id(new_security_id, actual_symbol):
                                            return new_security_id
                        # Don't use invalid security IDs - try other methods instead
                        return None
                except Exception as validation_error:
                    logger.warning(f"⚠️ Security ID {security_id} validation failed with error: {validation_error}")
                    # For API errors, we'll still return the ID but with a warning
                    return security_id
            
            # Try other exchanges if NSE_EQ fails
            if exchange == "NSE_EQ":
                for alt_exchange in ["BSE_EQ", "NSE_FO", "BSE_FO"]:
                    alt_match = self.daily_instruments_df[
                        (self.daily_instruments_df['TradingSymbol'] == clean_symbol) &
                        (self.daily_instruments_df['ExchangeSegment'] == alt_exchange)
                    ]
                    if not alt_match.empty:
                        security_id = str(alt_match.iloc[0]['SecurityId'])
                        logger.info(f"✅ Found in {alt_exchange} daily master: {clean_symbol} -> {security_id}")
                        
                        # Validate the security ID with the live API
                        try:
                            if self._validate_security_id(security_id, clean_symbol):
                                return security_id
                            else:
                                logger.warning(f"❌ Security ID {security_id} from daily master is INVALID for {clean_symbol}")
                                # Try to find a valid security ID using instrument search as fallback
                                search_results = self.search_instruments_by_symbol(clean_symbol, alt_exchange)
                                if search_results:
                                    # Find the exact match in search results
                                    for instrument in search_results:
                                        if instrument.get('tradingSymbol', '').upper() == clean_symbol:
                                            new_security_id = str(instrument.get('securityId', ''))
                                            if new_security_id and new_security_id != security_id:
                                                logger.info(f"✅ Found valid security ID via instrument search: {clean_symbol} -> {new_security_id}")
                                                # Validate the new security ID
                                                if self._validate_security_id(new_security_id, clean_symbol):
                                                    return new_security_id
                                # Don't use invalid security IDs - try other methods instead
                                return None
                        except Exception as validation_error:
                            logger.warning(f"⚠️ Security ID {security_id} validation failed with error: {validation_error}")
                            # For API errors, we'll still return the ID but with a warning
                            return security_id
            
            logger.warning(f"❌ Symbol {clean_symbol} not found in daily master for {exchange}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting security ID from daily master for {symbol}: {e}")
            return None
    
    def _get_instrument_data_from_dhan(self, exchange_segment: str = "NSE_EQ") -> Dict[str, str]:
        """Fetch real instrument data from Dhan API v2 endpoint with caching"""
        try:
            # Check cache first
            if exchange_segment in self.instrument_cache:
                if time.time() < self.instrument_cache_expiry.get(exchange_segment, 0):
                    logger.debug(f"Using cached instrument data for {exchange_segment}")
                    return self.instrument_cache[exchange_segment]
            
            logger.info(f"Fetching instrument data from Dhan for {exchange_segment}")
            
            # Skip API calls if using test credentials
            if self.client_id in ["TEST_CLIENT", "temp"]:
                logger.debug("Using test credentials - skipping real instrument fetch")
                return {}
            
            # Fetch instruments from Dhan API v2 endpoint: https://api.dhan.co/v2/instrument/NSE_EQ
            response = self._make_request('GET', f'/v2/instrument/{exchange_segment}')
            
            if response and isinstance(response, list):
                instrument_mapping = {}
                for instrument in response:
                    if isinstance(instrument, dict):
                        trading_symbol = instrument.get('tradingSymbol', '').upper()
                        security_id = instrument.get('securityId', '')
                        if trading_symbol and security_id:
                            instrument_mapping[trading_symbol] = str(security_id)
                
                # Cache the result
                self.instrument_cache[exchange_segment] = instrument_mapping
                self.instrument_cache_expiry[exchange_segment] = time.time() + self.instrument_cache_duration
                
                logger.info(f"✅ Fetched and cached {len(instrument_mapping)} instruments from Dhan API v2 endpoint")
                return instrument_mapping
            
            logger.warning("No instrument data received from Dhan API")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch instrument data from Dhan: {e}")
            return {}
    
    def _search_security_id_from_instruments(self, symbol: str) -> Optional[str]:
        """Search for security ID in real Dhan instrument data"""
        try:
            # First try NSE_EQ segment
            instruments = self._get_instrument_data_from_dhan("NSE_EQ")
            
            # Try exact match first
            if symbol.upper() in instruments:
                security_id = instruments[symbol.upper()]
                logger.info(f"✅ Found exact match in Dhan instruments: {symbol} -> {security_id}")
                return security_id
            
            # Try partial matches for common variations
            symbol_variations = [
                symbol.upper(),
                symbol.upper().replace('-', ''),
                symbol.upper().replace('_', ''),
                symbol.upper() + "EQ",
                symbol.upper() + "-EQ"
            ]
            
            for variation in symbol_variations:
                if variation in instruments:
                    security_id = instruments[variation]
                    logger.info(f"✅ Found variation match in Dhan instruments: {symbol} ({variation}) -> {security_id}")
                    return security_id
            
            # Search for partial matches in trading symbols
            for trading_symbol, security_id in instruments.items():
                if symbol.upper() in trading_symbol or trading_symbol.startswith(symbol.upper()):
                    logger.info(f"✅ Found partial match in Dhan instruments: {symbol} -> {trading_symbol} -> {security_id}")
                    return security_id
            
            logger.warning(f"Symbol {symbol} not found in Dhan instrument data")
            return None
            
        except Exception as e:
            logger.error(f"Error searching Dhan instruments for {symbol}: {e}")
            return None

    def search_instruments_by_symbol(self, symbol: str, exchange_segment: str = "NSE_EQ") -> List[Dict]:
        """Search for instruments by symbol using Dhan's search API"""
        try:
            # Skip API calls if using test credentials
            if self.client_id in ["TEST_CLIENT", "temp"]:
                logger.debug("Using test credentials - skipping instrument search")
                return []
            
            logger.info(f"🔍 Searching instruments for symbol: {symbol} in {exchange_segment}")
            
            # Search using Dhan's instrument search endpoint
            search_data = {
                "exchangeSegment": exchange_segment,
                "searchText": symbol.upper()
            }
            
            response = self._make_request('POST', '/v2/instruments/search', search_data)
            
            if response and isinstance(response, dict) and 'data' in response:
                instruments = response['data']
                if isinstance(instruments, list):
                    logger.info(f"✅ Found {len(instruments)} matching instruments for {symbol}")
                    return instruments
            
            logger.warning(f"No instruments found for {symbol} in {exchange_segment}")
            return []
            
        except Exception as e:
            logger.error(f"Error searching instruments for {symbol}: {e}")
            return []

    def _search_security_id_dynamic(self, symbol: str) -> Optional[str]:
        """Final fallback: Try generated IDs with API validation (last resort)"""
        try:
            logger.info(f"Final fallback: Trying generated IDs for {symbol}")
            
            # Skip API calls if using test credentials
            if self.client_id in ["TEST_CLIENT", "temp"]:
                logger.debug("Using test credentials - skipping API validation")
                return None
            
            # Try a few well-known patterns for common stocks
            possible_ids = self._generate_possible_security_ids(symbol)
            
            for test_id in possible_ids:
                try:
                    # Test if this security ID is valid by making a quote request
                    quote_data = {
                        "securityId": str(test_id),
                        "exchangeSegment": "NSE_EQ"
                    }
                    
                    response = self._make_request('POST', '/v2/marketdata/quote', quote_data)
                    
                    # If we get a valid response, this security ID works
                    if response and 'data' in response:
                        data = response['data']
                        if data and isinstance(data, dict):
                            # Verify the symbol matches
                            trading_symbol = data.get('tradingSymbol', '').upper()
                            if trading_symbol == symbol.upper():
                                logger.info(f"✅ Found valid security ID via fallback: {symbol} -> {test_id}")
                                return str(test_id)
                            else:
                                logger.debug(f"Security ID {test_id} valid but for different symbol: {trading_symbol}")
                                
                except Exception as quote_error:
                    # Handle specific API errors
                    if "404" in str(quote_error) or "Invalid" in str(quote_error):
                        logger.debug(f"Security ID {test_id} not valid for {symbol}")
                    else:
                        logger.debug(f"Security ID {test_id} validation failed for {symbol}: {quote_error}")
                    continue
            
            logger.warning(f"Could not find valid security ID for {symbol} via fallback validation")
            return None
            
        except Exception as e:
            logger.error(f"Fallback security ID search failed for {symbol}: {e}")
            return None
    
    def _generate_possible_security_ids(self, symbol: str) -> List[str]:
        """Generate possible security IDs for a symbol with improved patterns"""
        possible_ids = []
        
        # Remove all hardcoded security ID mappings - will fetch dynamically from API
        # Enhanced known corrections for specific symbols
        symbol_alternatives = {
            # All hardcoded mappings removed - using dynamic fetching instead
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in symbol_alternatives:
            possible_ids.extend(symbol_alternatives[symbol_upper])
        
        # Generate pattern-based IDs as fallback (more conservative)
        import hashlib
        hash_val = int(hashlib.md5(symbol.encode()).hexdigest()[:6], 16)
        
        # More conservative pattern generation
        pattern_ids = [
            f"5{(hash_val % 10000):04d}",     # 5xxxx pattern (common for NSE)
            f"53{(hash_val % 1000):03d}",     # 53xxx pattern
            f"11{(hash_val % 1000):03d}",     # 11xxx pattern (seen in docs)
        ]
        possible_ids.extend(pattern_ids)
        
        # Remove duplicates and limit to prevent excessive API calls
        return list(dict.fromkeys(possible_ids))[:6]  # Reduced to 6 for efficiency
    
    def _validate_security_id(self, security_id: str, symbol: str = None) -> bool:
        """Validate a security ID with the live API to ensure it's active and valid"""
        try:
            # Skip validation for test credentials
            if self.client_id in ["TEST_CLIENT", "temp"]:
                logger.debug(f"Using test credentials - skipping security ID validation for {security_id}")
                return True
            
            # Rate limiting for validation calls
            current_time = time.time()
            if current_time - self.last_validation_time < 1.0:  # Minimum 1 second between validations
                time.sleep(1.0 - (current_time - self.last_validation_time))
            self.last_validation_time = time.time()
            
            # Validate by making a quote request
            quote_data = {
                "securityId": str(security_id),
                "exchangeSegment": "NSE_EQ"
            }
            
            response = self._make_request('POST', '/v2/marketdata/quote', quote_data)
            
            # Check if we got a valid response
            if response and 'data' in response:
                data = response['data']
                if data and isinstance(data, dict):
                    # If symbol provided, verify it matches
                    if symbol:
                        trading_symbol = data.get('tradingSymbol', '').upper()
                        if trading_symbol == symbol.upper():
                            logger.debug(f"✅ Security ID {security_id} validated successfully for {symbol}")
                            # Store the validated ID in database
                            self._store_corrected_security_id(symbol, security_id)
                            return True
                        else:
                            logger.debug(f"❌ Security ID {security_id} is valid but for different symbol: {trading_symbol}")
                            return False
                    else:
                        # Just check if the security ID is valid
                        logger.debug(f"✅ Security ID {security_id} validated successfully")
                        return True
            
            logger.debug(f"❌ Security ID {security_id} validation failed - invalid response")
            return False
            
        except Exception as e:
            # Handle specific API errors that indicate invalid security IDs
            error_str = str(e).lower()
            if "404" in error_str or "invalid" in error_str or "not found" in error_str or "dh-905" in error_str:
                logger.debug(f"❌ Security ID {security_id} is INVALID: {e}")
                return False
            else:
                # For network errors or other issues, we don't want to blacklist potentially valid IDs
                logger.warning(f"⚠️ Security ID {security_id} validation encountered error (assuming valid): {e}")
                # Return True to allow the system to try using the ID
                # This prevents false negatives due to temporary API issues
                return True  # Assume valid to avoid false negatives due to API issues
    
    def get_security_id(self, symbol: str) -> str:
        """Get numeric security ID for a symbol using multi-layer approach with dynamic fetching"""
        # Validate and clean symbol format
        original_symbol = symbol
        
        # Handle special cases and invalid formats
        if symbol.startswith('$'):
            logger.warning(f"Invalid symbol format detected: {symbol} - removing '$' prefix")
            symbol = symbol[1:]
        
        # Convert symbol format
        if symbol.endswith('.NS'):
            symbol = symbol[:-3]
        elif symbol.endswith('.BO'):
            symbol = symbol[:-3]
        
        symbol = symbol.upper()
        
        # Check if symbol is empty after cleaning
        if not symbol:
            logger.error(f"Invalid symbol after cleaning: '{original_symbol}'")
            raise ValueError(f"Invalid symbol format: '{original_symbol}'")
        
        # Check cache first
        cached_id = self._get_cached_security_id(symbol)
        if cached_id:
            logger.debug(f"Using cached security ID for {symbol}: {cached_id}")
            return cached_id
        
        # Method 1: Daily instrument master (Dhan's recommended approach - MOST RELIABLE)
        security_id = self._get_security_id_from_daily_master(symbol)
        if security_id:
            self._cache_security_id(symbol, security_id)
            logger.info(f"✅ Found security ID via daily master for {symbol}: {security_id}")
            return security_id
        
        # Method 2: Live API instrument data (real-time fallback) - now the primary method since we removed hardcoded mappings
        security_id = self._search_security_id_from_instruments(symbol)
        if security_id:
            self._cache_security_id(symbol, security_id)
            logger.info(f"✅ Found security ID via API instruments for {symbol}: {security_id}")
            return security_id
        
        # Method 3: Dynamic validation with generated IDs (last resort)
        security_id = self._search_security_id_dynamic(symbol)
        if security_id:
            self._cache_security_id(symbol, security_id)
            logger.info(f"✅ Found security ID via dynamic validation for {symbol}: {security_id}")
            return security_id
        
        logger.error(f"❌ Security ID not found for {symbol} (original: {original_symbol})")
        raise ValueError(f"Security ID not found for {symbol}")
    
    def validate_order_prerequisites(self) -> bool:
        """Validate that all prerequisites for order placement are met"""
        try:
            # Check credentials
            if not self.client_id or not self.access_token:
                logger.error("Dhan credentials not properly configured")
                return False
            
            # Validate API connection
            if not self.validate_connection():
                logger.error("Dhan API connection validation failed")
                return False
            
            # Check market status
            if not self.is_market_open():
                logger.warning("Market is currently closed")
                return False
            
            logger.info("All order prerequisites validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Order prerequisites validation failed: {e}")
            return False
    
    def place_order(self, symbol: str, quantity: int, order_type: str = "MARKET", 
                   side: str = "BUY", price: float = None) -> Dict:
        """Place a trading order with comprehensive validation"""
        try:
            # Validate prerequisites first
            if not self.validate_order_prerequisites():
                raise Exception("Order prerequisites validation failed")
            
            # Get numeric security ID
            security_id = self.get_security_id(symbol)
            
            # Prepare order data according to official Dhan API v2 format
            order_data = {
                "dhanClientId": self.client_id,
                "transactionType": side.upper(),
                "exchangeSegment": "NSE_EQ",
                "productType": "CNC",  # CNC for delivery trading (positions carry forward)
                "orderType": order_type.upper(),
                "validity": "DAY",
                "securityId": str(security_id),  # Must be string
                "quantity": int(quantity),  # Must be integer
                "disclosedQuantity": 0,  # Integer, not string
                "triggerPrice": 0.0,  # Float, always required
                "afterMarketOrder": False,  # Boolean, always required
                "boProfitValue": 0.0,  # Float, always required 
                "boStopLossValue": 0.0  # Float, always required
            }
            
            # Set price field (always required according to API docs)
            if order_type.upper() == "LIMIT" and price:
                order_data["price"] = float(price)
            else:
                # For MARKET orders, price must be 0.0 (float)
                order_data["price"] = 0.0
            
            logger.info(f"Placing order: {side} {quantity} {symbol} (ID: {security_id})")
            logger.debug(f"Order data: {order_data}")
            
            response = self._make_request('POST', '/v2/orders', order_data)
            
            if response and 'orderId' in response:
                logger.info(f"Order placed successfully: {side} {quantity} {symbol} - Order ID: {response.get('orderId')}")
            else:
                logger.warning(f"Unexpected order response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise Exception(f"Order placement failed: {str(e)}")
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            response = self._make_request('DELETE', f'/orders/{order_id}')
            logger.info(f"Order cancelled: {order_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise Exception(f"Order cancellation failed: {str(e)}")
    
    def modify_order(self, order_id: str, quantity: int = None, 
                    price: float = None, order_type: str = None) -> Dict:
        """Modify an existing order"""
        try:
            modify_data = {"orderId": order_id}
            
            if quantity:
                modify_data["quantity"] = quantity
            if price:
                modify_data["price"] = price
            if order_type:
                modify_data["orderType"] = order_type.upper()
            
            response = self._make_request('PUT', f'/orders/{order_id}', modify_data)
            logger.info(f"Order modified: {order_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            raise Exception(f"Order modification failed: {str(e)}")
    
    def _convert_symbol_to_dhan(self, symbol: str) -> str:
        """Convert Yahoo Finance symbol to Dhan symbol format"""
        # Remove .NS suffix if present
        if symbol.endswith('.NS'):
            symbol = symbol[:-3]
        
        # Return the symbol in uppercase format
        # No more hardcoded mappings - using dynamic resolution
        return symbol.upper()
    
    def get_market_status(self) -> Dict:
        """Get current market status using profile endpoint as fallback"""
        try:
            # Use profile endpoint to check API connectivity since marketstatus doesn't exist in v2
            profile = self._make_request('GET', '/v2/profile')
            # If profile works, assume market is accessible
            if profile:
                # Check if it's during market hours (9:15 AM to 3:30 PM IST)
                from datetime import datetime, time
                import pytz

                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)
                market_open = time(9, 15)  # 9:15 AM
                market_close = time(15, 30)  # 3:30 PM

                # Check if it's a weekday and within market hours
                if now.weekday() < 5 and market_open <= now.time() <= market_close:
                    return {"marketStatus": "OPEN"}
                else:
                    return {"marketStatus": "CLOSED"}
            else:
                return {"marketStatus": "UNKNOWN"}
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {"marketStatus": "UNKNOWN"}
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            status = self.get_market_status()
            return status.get("marketStatus", "").upper() == "OPEN"
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    def validate_connection(self) -> bool:
        """Validate API connection and credentials with rate limiting"""
        try:
            # Rate limiting - avoid excessive validation calls
            current_time = time.time()
            if current_time - self.last_validation_time < self.validation_cache_duration:
                logger.debug("Using cached validation result")
                return True  # Assume valid if recently validated

            profile = self.get_profile()
            logger.debug(f"Profile response: {profile}")
            if profile and 'dhanClientId' in profile:
                logger.debug("Dhan API connection validated successfully")
                self.last_validation_time = current_time
                return True
            else:
                logger.error(f"Dhan API validation failed - invalid response: {profile}")
                return False
        except Exception as e:
            logger.error(f"Dhan API validation failed: {e}")
            return False
