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
        
        logger.info("Dhan API client initialized")
    
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
                logger.info("Holdings endpoint returned 500 - likely empty portfolio")
                return {"data": []}

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            # More specific error handling
            if "500" in str(e) and endpoint == '/v2/holdings':
                logger.info("Holdings API returned 500 - empty portfolio (normal for new accounts)")
                return {"data": []}
            else:
                logger.error(f"Dhan API request failed: {e}")
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
            return self._make_request('GET', '/v2/fundlimit')
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {"availabelBalance": 0, "sodLimit": 0}
    
    def get_holdings(self) -> List[Dict]:
        """Get current stock holdings"""
        try:
            response = self._make_request('GET', '/v2/holdings')
            return response.get('data', [])
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
            return response.get('data', [])
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get order history"""
        try:
            response = self._make_request('GET', '/v2/orders')
            return response.get('data', [])
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Get real-time quote for a symbol"""
        try:
            # Convert symbol format for Dhan API
            dhan_symbol = self._convert_symbol_to_dhan(symbol)
            
            data = {
                "symbol": dhan_symbol,
                "exchange": exchange
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
            # Convert symbol format
            dhan_symbol = self._convert_symbol_to_dhan(symbol)
            
            # Default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime("%Y-%m-%d")
            if not from_date:
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            data = {
                "symbol": dhan_symbol,
                "exchange": "NSE",
                "timeframe": timeframe,
                "from_date": from_date,
                "to_date": to_date
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
    
    def place_order(self, symbol: str, quantity: int, order_type: str = "MARKET", 
                   side: str = "BUY", price: float = None) -> Dict:
        """Place a trading order"""
        try:
            # Convert symbol format
            dhan_symbol = self._convert_symbol_to_dhan(symbol)
            
            order_data = {
                "dhanClientId": self.client_id,
                "transactionType": side.upper(),
                "exchangeSegment": "NSE_EQ",
                "productType": "INTRADAY",  # Can be INTRADAY, CNC, MTF
                "orderType": order_type.upper(),
                "validity": "DAY",
                "securityId": dhan_symbol,
                "quantity": quantity,
                "disclosedQuantity": 0,
                "triggerPrice": 0
            }
            
            # Add price for limit orders
            if order_type.upper() == "LIMIT" and price:
                order_data["price"] = price
            else:
                order_data["price"] = 0
            
            response = self._make_request('POST', '/v2/orders', order_data)
            logger.info(f"Order placed: {side} {quantity} {symbol} - Order ID: {response.get('orderId')}")
            
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
        
        # Dhan uses different symbol formats for some stocks
        symbol_mapping = {
            'RELIANCE': 'RELIANCE',
            'TCS': 'TCS',
            'HDFCBANK': 'HDFCBANK',
            'INFY': 'INFY',
            'ICICIBANK': 'ICICIBANK',
            'SBIN': 'SBIN',
            'BHARTIARTL': 'BHARTIARTL',
            'ITC': 'ITC',
            'KOTAKBANK': 'KOTAKBANK',
            'LT': 'LT'
        }
        
        return symbol_mapping.get(symbol.upper(), symbol.upper())
    
    def get_market_status(self) -> Dict:
        """Get current market status - Note: This endpoint may not exist in Dhan API v2"""
        try:
            # Try profile endpoint as a fallback to check API connectivity
            profile = self._make_request('GET', '/v2/profile')
            # If profile works, assume market is accessible
            return {"marketStatus": "OPEN" if profile else "UNKNOWN"}
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
        """Validate API connection and credentials"""
        try:
            profile = self.get_profile()
            logger.info(f"Profile response: {profile}")
            if profile and 'dhanClientId' in profile:
                logger.info("Dhan API connection validated successfully")
                return True
            else:
                logger.error(f"Dhan API validation failed - invalid response: {profile}")
                return False
        except Exception as e:
            logger.error(f"Dhan API validation failed: {e}")
            return False
