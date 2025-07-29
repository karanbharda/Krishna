#!/usr/bin/env python3
"""
Production-Grade Fyers API Client
=================================

Enterprise-level Fyers API integration with WebSocket support for real-time market data.
Provides robust error handling, automatic reconnection, and comprehensive monitoring.
"""

import asyncio
import json
import logging
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
from queue import Queue
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Production monitoring
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
FYERS_API_CALLS = Counter('fyers_api_calls_total', 'Total Fyers API calls', ['endpoint', 'status'])
FYERS_WS_MESSAGES = Counter('fyers_ws_messages_total', 'WebSocket messages', ['type'])
FYERS_CONNECTION_STATUS = Gauge('fyers_connection_status', 'Connection status (1=connected, 0=disconnected)')

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    ltp: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    timestamp: datetime
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

@dataclass
class OrderBookData:
    """Order book depth data"""
    symbol: str
    bids: List[Dict[str, float]]  # [{"price": 100.0, "size": 1000}, ...]
    asks: List[Dict[str, float]]
    timestamp: datetime

class FyersAPIClient:
    """
    Production-grade Fyers API client with advanced features:
    - Automatic retry and error handling
    - Rate limiting compliance
    - Connection pooling
    - Comprehensive logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.access_token = config.get("fyers_access_token")
        self.client_id = config.get("fyers_client_id")
        self.base_url = "https://api-t1.fyers.in/api/v3"
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers
        self.session.headers.update({
            "Authorization": f"{self.client_id}:{self.access_token}",
            "Content-Type": "application/json"
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("Fyers API client initialized")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request with comprehensive error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            # Record success metrics
            FYERS_API_CALLS.labels(endpoint=endpoint, status="success").inc()
            
            return result
            
        except requests.exceptions.RequestException as e:
            FYERS_API_CALLS.labels(endpoint=endpoint, status="error").inc()
            logger.error(f"Fyers API error - {endpoint}: {e}")
            raise
        except Exception as e:
            FYERS_API_CALLS.labels(endpoint=endpoint, status="error").inc()
            logger.error(f"Unexpected error - {endpoint}: {e}")
            raise
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get user profile information"""
        return await self._make_request("GET", "profile")
    
    async def get_funds(self) -> Dict[str, Any]:
        """Get account funds information"""
        return await self._make_request("GET", "funds")
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get current holdings"""
        result = await self._make_request("GET", "holdings")
        return result.get("holdings", [])
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        result = await self._make_request("GET", "positions")
        return result.get("netPositions", [])
    
    async def get_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time quotes for multiple symbols"""
        symbol_string = ",".join(symbols)
        data = {"symbols": symbol_string}
        
        result = await self._make_request("GET", "data/quotes", data)
        quotes = {}
        
        for symbol, quote_data in result.get("d", {}).items():
            quotes[symbol] = MarketData(
                symbol=symbol,
                ltp=float(quote_data.get("v", {}).get("lp", 0)),
                open_price=float(quote_data.get("v", {}).get("o", 0)),
                high_price=float(quote_data.get("v", {}).get("h", 0)),
                low_price=float(quote_data.get("v", {}).get("l", 0)),
                close_price=float(quote_data.get("v", {}).get("prev_close_price", 0)),
                volume=int(quote_data.get("v", {}).get("volume", 0)),
                timestamp=datetime.now()
            )
        
        return quotes
    
    async def get_market_depth(self, symbol: str) -> OrderBookData:
        """Get market depth (order book) for a symbol"""
        data = {"symbol": symbol, "ohlcv_flag": "1"}
        result = await self._make_request("GET", "data/depth", data)
        
        depth_data = result.get("d", {}).get(symbol, {})
        
        bids = []
        asks = []
        
        # Parse bid/ask data
        for i in range(5):  # Top 5 levels
            bid_key = f"bid{i+1}"
            ask_key = f"ask{i+1}"
            
            if bid_key in depth_data:
                bids.append({
                    "price": float(depth_data[bid_key].get("price", 0)),
                    "size": int(depth_data[bid_key].get("size", 0))
                })
            
            if ask_key in depth_data:
                asks.append({
                    "price": float(depth_data[ask_key].get("price", 0)),
                    "size": int(depth_data[ask_key].get("size", 0))
                })
        
        return OrderBookData(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
    
    async def get_historical_data(self, symbol: str, resolution: str, from_date: str, to_date: str) -> List[Dict]:
        """Get historical OHLCV data"""
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }
        
        result = await self._make_request("GET", "data/history", data)
        return result.get("candles", [])
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a trading order"""
        return await self._make_request("POST", "orders", order_data)
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get order history"""
        result = await self._make_request("GET", "orders")
        return result.get("orderBook", [])
    
    async def get_trades(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        result = await self._make_request("GET", "tradebook")
        return result.get("tradeBook", [])

class FyersWebSocketClient:
    """
    Production-grade WebSocket client for real-time market data
    Features:
    - Automatic reconnection
    - Subscription management
    - Data validation
    - Performance monitoring
    """
    
    def __init__(self, access_token: str, client_id: str):
        self.access_token = access_token
        self.client_id = client_id
        self.ws_url = "wss://api-t1.fyers.in/socket/v3/dataSock"
        
        self.websocket = None
        self.is_connected = False
        self.subscriptions = set()
        self.callbacks = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Message queue for handling
        self.message_queue = Queue()
        self.message_handler_thread = None
        
        logger.info("Fyers WebSocket client initialized")
    
    async def connect(self):
        """Connect to Fyers WebSocket"""
        try:
            headers = {
                "Authorization": f"{self.client_id}:{self.access_token}"
            }
            
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            FYERS_CONNECTION_STATUS.set(1)
            
            logger.info("Connected to Fyers WebSocket")
            
            # Start message handler
            if not self.message_handler_thread or not self.message_handler_thread.is_alive():
                self.message_handler_thread = threading.Thread(target=self._handle_messages)
                self.message_handler_thread.daemon = True
                self.message_handler_thread.start()
            
            # Listen for messages
            await self._listen()
            
        except Exception as e:
            self.is_connected = False
            FYERS_CONNECTION_STATUS.set(0)
            logger.error(f"WebSocket connection failed: {e}")
            await self._handle_reconnection()
    
    async def _listen(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.websocket:
                FYERS_WS_MESSAGES.labels(type="received").inc()
                self.message_queue.put(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            FYERS_CONNECTION_STATUS.set(0)
            await self._handle_reconnection()
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")
            self.is_connected = False
            FYERS_CONNECTION_STATUS.set(0)
            await self._handle_reconnection()
    
    def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        while True:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get()
                    self._process_message(message)
                else:
                    time.sleep(0.01)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Message handling error: {e}")
    
    def _process_message(self, message: str):
        """Process individual WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type in self.callbacks:
                for callback in self.callbacks[msg_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        
        logger.info(f"Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(wait_time)
        
        await self.connect()
    
    async def subscribe(self, symbols: List[str], data_type: str = "symbolData"):
        """Subscribe to real-time data for symbols"""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return
        
        subscription_msg = {
            "T": "SUB_L2",
            "L2LIST": symbols,
            "SUB_T": 1
        }
        
        try:
            await self.websocket.send(json.dumps(subscription_msg))
            self.subscriptions.update(symbols)
            FYERS_WS_MESSAGES.labels(type="sent").inc()
            logger.info(f"Subscribed to {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Subscription error: {e}")
    
    def add_callback(self, message_type: str, callback: Callable):
        """Add callback for specific message types"""
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        self.callbacks[message_type].append(callback)
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
        FYERS_CONNECTION_STATUS.set(0)
        logger.info("Disconnected from Fyers WebSocket")
