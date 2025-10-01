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
from io import StringIO

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
        self.manual_security_id_mapping = {}
        
        # Instrument data cache
        self.instrument_cache = {}
        self.instrument_cache_expiry = {}
        self.instrument_cache_duration = 3600  # Reduce cache duration to 1 hour for fresher data
        
        # Daily instrument master (Dhan's recommended approach)
        self.daily_instruments_df = None
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
        
        logger.info("Dhan API client initialized with direct API instrument fetching")
    
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
    
    def _fetch_instrument_data(self, exchange_segment: str = "NSE_EQ") -> pd.DataFrame:
        """Fetch real-time instrument data directly from Dhan API v2 endpoint"""
        try:
            # Use the specific endpoint as requested: https://api.dhan.co/v2/instrument/NSE_EQ
            url = f"{self.base_url}/v2/instrument/{exchange_segment}"
            logger.info(f"Fetching instrument data from: {url}")
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                # Parse CSV data directly into DataFrame
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"✅ Loaded instruments: {len(df)}")
                
                # Log some sample data for debugging
                if not df.empty:
                    logger.debug(f"Sample instrument data: {df.head(3).to_dict('records')}")
                
                return df
            else:
                logger.error(f"Failed to fetch instruments. Status: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching instrument data: {e}")
            return pd.DataFrame()
    
    def _search_security_id_in_instruments(self, symbol: str, exchange_segment: str = "NSE_EQ") -> Optional[str]:
        """Search for security ID in instrument data fetched from Dhan API v2"""
        try:
            # Fetch instrument data if not already cached or cache expired
            cache_key = f"instruments_{exchange_segment}"
            if (cache_key not in self.instrument_cache or 
                time.time() > self.instrument_cache_expiry.get(cache_key, 0)):
                
                df = self._fetch_instrument_data(exchange_segment)
                if not df.empty:
                    self.instrument_cache[cache_key] = df
                    self.instrument_cache_expiry[cache_key] = time.time() + self.instrument_cache_duration
                else:
                    return None
            else:
                df = self.instrument_cache[cache_key]
            
            # Search for symbol in the DataFrame
            # Handle different column names that might be present
            symbol_columns = ['SYMBOL_NAME', 'TradingSymbol', 'tradingSymbol', 'symbol', 'Symbol']
            security_id_columns = ['SECURITY_ID', 'SecurityId', 'securityId', 'SecurityID']
            
            symbol_col = None
            security_id_col = None
            
            # Find the appropriate columns
            for col in df.columns:
                if col in symbol_columns:
                    symbol_col = col
                elif col in security_id_columns:
                    security_id_col = col
            
            if not symbol_col or not security_id_col:
                logger.error(f"Could not find required columns in instrument data. Available columns: {df.columns.tolist()}")
                return None
            
            # Create a more dynamic and comprehensive search without relying on hardcoded mappings
            search_symbol = symbol.upper()
            
            # Special handling for symbols with numeric suffixes like JAYNECOIND
            # Try to find variations that might match
            variations = [
                search_symbol,
                search_symbol.replace('-', ' '),
                search_symbol.replace('_', ' '),
                search_symbol.replace('&', ' AND '),
                search_symbol + ' LTD',
                search_symbol + ' LIMITED',
                search_symbol + ' EQ',
                search_symbol + '-EQ',
                'THE ' + search_symbol,
                search_symbol.replace('M&M', 'MAHINDRA AND MAHINDRA'),
                search_symbol.replace('M&M', 'MAHINDRA & MAHINDRA'),
                search_symbol.replace('L&T', 'LARSEN AND TOUBRO'),
                search_symbol.replace('L&T', 'LARSEN & TOUBRO')
            ]
            
            # Add specific handling for JAYNECOIND and similar symbols
            if search_symbol == "JAYNECOIND":
                variations.extend([
                    "JAYASWAL NUCLEUS",
                    "JAYASWAL NUCLEUS LTD",
                    "JAYASWAL NUCLEUS LIMITED",
                    "JAYNECOIND EQ",
                    "JAYNECOIND-EQ"
                ])
            
            # Add specific handling for ASHOKLEY
            if search_symbol == "ASHOKLEY":
                variations.extend([
                    "ASHOK LEYLAND",
                    "ASHOK LEYLAND LTD",
                    "ASHOK LEYLAND LIMITED",
                    "ASHOKLEY EQ",
                    "ASHOKLEY-EQ",
                    "ASHOK LEYLAND EQ"
                ])
            
            # Add variations with common suffixes/prefixes
            common_suffixes = [' LTD', ' LIMITED', ' CORPORATION', ' COMPANY', ' INDIA', ' INDUSTRIES', ' SOLUTIONS', ' SERVICES']
            common_prefixes = ['THE ', 'M/S ']
            
            for suffix in common_suffixes:
                if not search_symbol.endswith(suffix.strip()):
                    variations.append(search_symbol + suffix)
            
            for prefix in common_prefixes:
                if not search_symbol.startswith(prefix.strip()):
                    variations.append(prefix + search_symbol)
            
            # Special handling for known abbreviation patterns
            abbreviation_mapping = {
                'RVNL': 'RAIL VIKAS NIGAM',
                'HINDUNILVR': 'HINDUSTAN UNILEVER',
                'BAJFINANCE': 'BAJAJ FINANCE',
                'BAJAJFINSV': 'BAJAJ FINSERV',
                'TATAMOTORS': 'TATA MOTORS',
                'TATASTEEL': 'TATA STEEL',
                'TATAPOWER': 'TATA POWER',
                'TATACONSUM': 'TATA CONSUMER',
                'M&MFIN': 'MAHINDRA FINANCE',
                'TECHM': 'TECH MAHINDRA',
                'HCLTECH': 'HCL TECHNOLOGIES',
                'ULTRACEMCO': 'ULTRATECH CEMENT',
                'DRREDDY': 'DR. REDDY',
                'DIVISLAB': 'DIVI S LABORATORIES',
                'BHARTIARTL': 'BHARTI AIRTEL',
                'ONGC': 'OIL AND NATURAL GAS',
                'POWERGRID': 'POWER GRID',
                'COALINDIA': 'COAL INDIA',
                'SHANTIGOLD': 'Shanti Gold International Ltd',
                'IDBI': 'IDBI Bank Ltd',
                'DBI': 'IDBI Bank Ltd',
                'MOTHERSON': 'Samvardhana Motherson International Ltd',
                'GMRAIRPORT': 'GMR Airports Ltd',
                'GMRI': 'GMR Airports Ltd',
                'ASHOKLEY': 'ASHOK LEYLAND' , # Add specific mapping for ASHOKLEY
                'IEX':' Indian Energy Exchange Ltd'
            }
            
            # Apply abbreviation mapping if available
            if search_symbol in abbreviation_mapping:
                expanded_name = abbreviation_mapping[search_symbol]
                variations.append(expanded_name)
                variations.append(expanded_name + ' LTD')
                variations.append(expanded_name + ' LIMITED')
            
            # Try each variation
            for variation in variations:
                result = df[df[symbol_col].str.upper() == variation]
                if not result.empty:
                    security_id = str(result.iloc[0][security_id_col])
                    logger.info(f"✅ Found security ID for {symbol} (variation match: {variation}): {security_id}")
                    return security_id
            
            # Try partial matches as a last resort
            for variation in variations:
                result = df[df[symbol_col].str.contains(variation, case=False, na=False)]
                if not result.empty:
                    security_id = str(result.iloc[0][security_id_col])
                    logger.info(f"✅ Found security ID for {symbol} (partial match: {variation}): {security_id}")
                    return security_id
            
            # Try reverse partial match (search symbol contains instrument symbol)
            for idx, row in df.iterrows():
                instrument_symbol = str(row[symbol_col]).upper()
                if search_symbol in instrument_symbol or instrument_symbol in search_symbol:
                    security_id = str(row[security_id_col])
                    logger.info(f"✅ Found security ID for {symbol} (reverse partial match: {instrument_symbol}): {security_id}")
                    return security_id
            
            logger.warning(f"❌ Symbol {symbol} not found in instrument data")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for security ID: {e}")
            return None
    
    def get_security_id(self, symbol: str) -> str:
        """Get numeric security ID for a symbol using direct API approach from https://api.dhan.co/v2/instrument/NSE_EQ"""
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
        
        # Special handling for ASHOKLEY -> ASHOKA LEYLAND mapping
        if symbol == "ASHOKLEY":
            logger.info("Special handling for ASHOKLEY -> ASHOK LEYLAND")
            symbol = "ASHOK LEYLAND"
        
        # Check cache first
        cached_id = self._get_cached_security_id(symbol)
        if cached_id:
            logger.debug(f"Using cached security ID for {symbol}: {cached_id}")
            return cached_id
        
        # Check database for previously validated security IDs
        db_id = self._get_corrected_security_id(symbol)
        if db_id:
            self._cache_security_id(symbol, db_id)
            return db_id
        
        # Fetch and search in real-time instrument data from the specific endpoint
        security_id = self._search_security_id_in_instruments(symbol)
        if security_id:
            self._cache_security_id(symbol, security_id)
            # Store in database for future use
            self._store_corrected_security_id(symbol, security_id)
            return security_id
        
        # Fallback: Try to search with a broader approach for problematic symbols
        if symbol == "JAYNECOIND":
            # Try alternative search for Jayaswal Nucleus
            alternative_symbols = ["JAYASWAL NUCLEUS", "JAYASWAL", "JAYA NUCLEUS"]
            for alt_symbol in alternative_symbols:
                alt_security_id = self._search_security_id_in_instruments(alt_symbol)
                if alt_security_id:
                    logger.info(f"✅ Found alternative security ID for {symbol} using {alt_symbol}: {alt_security_id}")
                    self._cache_security_id(symbol, alt_security_id)
                    self._store_corrected_security_id(symbol, alt_security_id)
                    return alt_security_id
        
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
            
            # Only use CNC (delivery) product type for equity trades
            # Prepare order data according to official Dhan API v2 format
            order_data = {
                "dhanClientId": self.client_id,
                "transactionType": side.upper(),
                "exchangeSegment": "NSE_EQ",
                "productType": "CNC",  # Cash and Carry - for delivery-based trading
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
            
            logger.info(f"Placing order: {side} {quantity} {symbol} (ID: {security_id}, Product: CNC)")
            logger.debug(f"Order data: {order_data}")
            
            response = self._make_request('POST', '/v2/orders', order_data)
            
            if response and 'orderId' in response:
                logger.info(f"Order placed successfully: {side} {quantity} {symbol} - Order ID: {response.get('orderId')}")
                return response
            else:
                logger.warning(f"Unexpected order response: {response}")
                raise Exception(f"Order placement failed with unexpected response: {response}")
            
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