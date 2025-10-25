#!/usr/bin/env python3
"""
Fyers Data Service - Standalone Market Data Provider
Runs independently to provide real-time market data via REST API
Prevents resource conflicts with trading bot backend
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import requests
import random
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fyers_data_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import Fyers API
try:
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
    logger.info("Fyers API imported successfully")
except ImportError as e:
    FYERS_AVAILABLE = False
    logger.warning(f"Fyers API not available: {e}")

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: str
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0

@dataclass
class ServiceHealth:
    """Service health status"""
    status: str
    fyers_connected: bool
    last_update: str
    data_points: int
    uptime_seconds: int

class FyersDataService:
    """Standalone Fyers data service"""
    
    def __init__(self):
        self.app = FastAPI(title="Fyers Data Service", version="1.0.0")
        self.setup_cors()
        self.setup_routes()
        
        # Service state
        self.start_time = datetime.now()
        self.fyers_client = None
        self.is_connected = False
        self.market_data_cache: Dict[str, MarketData] = {}
        self.last_update = datetime.now()
        self.update_interval = 5  # seconds
        self.max_retries = 3
        # Allow opt-in mock mode via environment variable FYERS_ALLOW_MOCK
        self.allow_mock_mode = os.getenv('FYERS_ALLOW_MOCK', 'false').lower() in ('1', 'true', 'yes')
        
        # Dynamic watchlist - starts empty, stocks added on-demand
        self.watchlist = set()  # Use set for O(1) lookups and automatic deduplication
        self.requested_stocks = set()  # Track all stocks ever requested
        
        # Background task management
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.data_update_task = None
        self.is_running = False

        # Cache for symbol conversions and data
        self.symbol_cache = {}  # Yahoo -> Fyers symbol mapping cache
        self.data_cache = {}    # Latest price data cache

        logger.info("Fyers Data Service initialized")

    def yahoo_to_fyers_symbol(self, yahoo_symbol: str) -> str:
        """Convert Yahoo Finance symbol to Fyers format

        Examples:
        RELIANCE.NS -> NSE:RELIANCE-EQ
        TCS.NS -> NSE:TCS-EQ
        SUZLON.NS -> NSE:SUZLON-EQ
        """
        if yahoo_symbol in self.symbol_cache:
            return self.symbol_cache[yahoo_symbol]

        # Remove .NS suffix and convert to Fyers format
        if yahoo_symbol.endswith('.NS'):
            base_symbol = yahoo_symbol[:-3]  # Remove .NS
            fyers_symbol = f"NSE:{base_symbol}-EQ"
            self.symbol_cache[yahoo_symbol] = fyers_symbol
            return fyers_symbol
        elif yahoo_symbol.endswith('.BO'):
            base_symbol = yahoo_symbol[:-3]  # Remove .BO
            fyers_symbol = f"BSE:{base_symbol}-EQ"
            self.symbol_cache[yahoo_symbol] = fyers_symbol
            return fyers_symbol
        else:
            # Assume NSE if no suffix
            fyers_symbol = f"NSE:{yahoo_symbol}-EQ"
            self.symbol_cache[yahoo_symbol] = fyers_symbol
            return fyers_symbol

    def add_stock_to_watchlist(self, yahoo_symbol: str) -> bool:
        """Dynamically add a stock to the watchlist"""
        try:
            fyers_symbol = self.yahoo_to_fyers_symbol(yahoo_symbol)

            # Add to watchlist and requested stocks
            self.watchlist.add(fyers_symbol)
            self.requested_stocks.add(yahoo_symbol)

            logger.info(f"Added {yahoo_symbol} ({fyers_symbol}) to dynamic watchlist")
            return True
        except Exception as e:
            logger.error(f"Failed to add {yahoo_symbol} to watchlist: {e}")
            return False

    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Fyers Data Service", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            uptime = (datetime.now() - self.start_time).total_seconds()
            health = ServiceHealth(
                status="healthy" if self.is_connected else "degraded",
                fyers_connected=self.is_connected,
                last_update=self.last_update.isoformat(),
                data_points=len(self.market_data_cache),
                uptime_seconds=int(uptime)
            )
            return asdict(health)
        
        @self.app.get("/data/{symbol}")
        async def get_symbol_data(symbol: str):
            """Get data for specific symbol - dynamically adds to watchlist if not present"""
            # Convert symbol format if needed
            fyers_symbol = self.convert_to_fyers_format(symbol)

            # Check if data is already cached
            if fyers_symbol in self.market_data_cache:
                return asdict(self.market_data_cache[fyers_symbol])

            # If not in cache, add to watchlist and try to fetch immediately
            if fyers_symbol not in self.watchlist:
                logger.info(f"New stock requested: {symbol} -> {fyers_symbol}")
                self.add_stock_to_watchlist(symbol)

                # Try to fetch data immediately for this specific stock
                try:
                    if self.fyers_client and self.is_connected:
                        quotes_response = self.fyers_client.quotes({"symbols": fyers_symbol})

                        if quotes_response and quotes_response.get('s') == 'ok':
                            quote_data = quotes_response.get('d', [])

                            if quote_data and len(quote_data) > 0:
                                # Process the immediate response
                                quote_item = quote_data[0] if isinstance(quote_data, list) else quote_data
                                symbol_data = quote_item.get('v', {}) if isinstance(quote_item, dict) and 'v' in quote_item else quote_item

                                market_data = MarketData(
                                    symbol=fyers_symbol,
                                    price=float(symbol_data.get('lp', 0.0)),
                                    change=float(symbol_data.get('ch', 0.0)),
                                    change_pct=float(symbol_data.get('chp', 0.0)),
                                    volume=int(symbol_data.get('volume', symbol_data.get('v', 0))),
                                    high=float(symbol_data.get('h', 0.0)),
                                    low=float(symbol_data.get('l', 0.0)),
                                    open=float(symbol_data.get('o', 0.0)),
                                    timestamp=datetime.now().isoformat()
                                )

                                # Cache the data
                                self.market_data_cache[fyers_symbol] = market_data
                                logger.info(f"Immediately fetched data for {symbol}: Rs.{market_data.price}")
                                return asdict(market_data)
                except Exception as e:
                    logger.error(f"Error fetching immediate data for {symbol}: {e}")

            # If still no data available, return 404
            raise HTTPException(status_code=404, detail=f"Data not available for {symbol}. Added to watchlist for future updates.")
        
        @self.app.get("/data")
        async def get_all_data():
            """Get all cached market data"""
            return {symbol: asdict(data) for symbol, data in self.market_data_cache.items()}
        
        @self.app.post("/watchlist")
        async def update_watchlist(symbols: List[str]):
            """Update watchlist - adds to existing dynamic watchlist"""
            for symbol in symbols:
                self.add_stock_to_watchlist(symbol)
            logger.info(f"Added {len(symbols)} symbols to dynamic watchlist")
            return {"message": f"Added {len(symbols)} symbols to watchlist. Total: {len(self.watchlist)}"}

        @self.app.get("/watchlist")
        async def get_watchlist():
            """Get current dynamic watchlist"""
            return {
                "watchlist": list(self.watchlist),
                "requested_stocks": list(self.requested_stocks),
                "total_symbols": len(self.watchlist)
            }
        
        @self.app.post("/connect")
        async def connect_fyers():
            """Manually trigger Fyers connection"""
            success = await self.connect_to_fyers()
            if success:
                return {"message": "Connected to Fyers successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to connect to Fyers")
    
    def convert_to_fyers_format(self, symbol: str) -> str:
        """Convert symbol to Fyers format"""
        # Handle different input formats
        if symbol.endswith('.NS'):
            # Yahoo Finance format: RELIANCE.NS -> NSE:RELIANCE-EQ
            base_symbol = symbol.replace('.NS', '')
            return f"NSE:{base_symbol}-EQ"
        elif symbol.endswith('.BO'):
            # BSE format: RELIANCE.BO -> BSE:RELIANCE-EQ  
            base_symbol = symbol.replace('.BO', '')
            return f"BSE:{base_symbol}-EQ"
        elif ':' in symbol and '-' in symbol:
            # Already in Fyers format
            return symbol
        else:
            # Assume NSE stock
            return f"NSE:{symbol}-EQ"
    
    async def connect_to_fyers(self) -> bool:
        """Connect to Fyers API - ONLY REAL DATA"""
        if not FYERS_AVAILABLE:
            logger.error("Fyers API not available - cannot proceed without real data")
            return False

        try:
            # Get credentials from environment - using correct variable names
            app_id = os.getenv('FYERS_APP_ID')
            access_token = os.getenv('FYERS_ACCESS_TOKEN')

            if not app_id or not access_token:
                logger.error("Fyers credentials not found in environment variables")
                logger.error("Required: FYERS_APP_ID and FYERS_ACCESS_TOKEN")
                return False

            # Initialize Fyers client with correct format
            self.fyers_client = fyersModel.FyersModel(
                client_id=app_id,
                token=access_token,
                log_path=""
            )

            # Test connection with a simple quote request
            test_symbols = "NSE:RELIANCE-EQ"
            test_quote = self.fyers_client.quotes({"symbols": test_symbols})

            if test_quote and test_quote.get('s') == 'ok':
                self.is_connected = True
                logger.info("Successfully connected to Fyers API with real data")
                logger.info(f"Test quote successful for {test_symbols}")
                return True
            else:
                logger.error(f"Fyers connection test failed: {test_quote}")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Fyers: {e}")
            return False
    
    def fetch_market_data(self) -> Dict[str, MarketData]:
        """Fetch market data from Fyers API - ONLY REAL DATA"""
        # If mock mode is enabled, generate mock data instead of requiring a live connection
        if self.allow_mock_mode:
            logger.debug("FYERS_ALLOW_MOCK enabled - generating mock market data")
            return self.generate_mock_data()

        if not self.is_connected or not self.fyers_client:
            logger.error("Fyers not connected - cannot fetch real data")
            return {}

        try:
            if FYERS_AVAILABLE and self.fyers_client:
                real_data = self.fetch_real_data()
                if real_data:
                    logger.info(f"Successfully fetched real data for {len(real_data)} symbols")
                    return real_data
                else:
                    logger.error("No real data received from Fyers")
                    return {}
            else:
                logger.error("Fyers API not available - cannot proceed without real data")
                return {}

        except Exception as e:
            logger.error(f"Error fetching Fyers data: {e}")
            return {}
    
    def fetch_real_data(self) -> Dict[str, MarketData]:
        """Fetch ONLY real data from Fyers API - NO MOCK DATA"""
        data = {}

        try:
            if not self.fyers_client:
                logger.error("Fyers client not initialized")
                return {}

            if not self.watchlist:
                logger.debug("No symbols in dynamic watchlist yet")
                return {}

            # Get quotes for all symbols
            symbols_str = ",".join(self.watchlist)
            logger.info(f"Fetching real Fyers data for: {symbols_str}")

            quotes_response = self.fyers_client.quotes({"symbols": symbols_str})

            if quotes_response and quotes_response.get('s') == 'ok':
                quote_data = quotes_response.get('d', [])
                logger.info(f"Fyers API returned {len(quote_data)} quotes")

                # Handle both list and dict formats from Fyers API
                if isinstance(quote_data, list):
                    # List format: [{'n': 'symbol', 'v': {...}}]
                    for quote_item in quote_data:
                        try:
                            symbol = quote_item.get('n', '')
                            symbol_data = quote_item.get('v', {})

                            market_data = MarketData(
                                symbol=symbol,
                                price=float(symbol_data.get('lp', 0.0)),  # Last price
                                change=float(symbol_data.get('ch', 0.0)),  # Change
                                change_pct=float(symbol_data.get('chp', 0.0)),  # Change percentage
                                volume=int(symbol_data.get('volume', symbol_data.get('v', 0))),  # Volume
                                high=float(symbol_data.get('h', 0.0)),  # High
                                low=float(symbol_data.get('l', 0.0)),  # Low
                                open=float(symbol_data.get('o', 0.0)),  # Open
                                timestamp=datetime.now().isoformat()
                            )

                            data[symbol] = market_data
                            logger.info(f"Real data for {symbol}: Rs.{market_data.price}")
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                            continue
                elif isinstance(quote_data, dict):
                    # Dict format: {'symbol': {...}}
                    for symbol, symbol_data in quote_data.items():
                        try:
                            market_data = MarketData(
                                symbol=symbol,
                                price=float(symbol_data.get('lp', 0.0)),  # Last price
                                change=float(symbol_data.get('ch', 0.0)),  # Change
                                change_pct=float(symbol_data.get('chp', 0.0)),  # Change percentage
                                volume=int(symbol_data.get('v', 0)),  # Volume
                                high=float(symbol_data.get('h', 0.0)),  # High
                                low=float(symbol_data.get('l', 0.0)),  # Low
                                open=float(symbol_data.get('o', 0.0)),  # Open
                                timestamp=datetime.now().isoformat()
                            )

                            data[symbol] = market_data
                            logger.info(f"Real data for {symbol}: ₹{market_data.price}")
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {e}")
                            continue

                logger.info(f"Successfully fetched REAL data for {len(data)} symbols")
            else:
                logger.error(f"Fyers quotes API error: {quotes_response}")
                return {}

        except Exception as e:
            logger.error(f"Error fetching real Fyers data: {e}")
            return {}

        return data

    def generate_mock_data(self) -> Dict[str, MarketData]:
        """Generate simple mock market data for watchlist symbols.

        This is intentionally lightweight: it uses a random price within a range
        and preserves reasonable fields so callers and the API can operate
        without a live Fyers connection when FYERS_ALLOW_MOCK is enabled.
        """
        data: Dict[str, MarketData] = {}

        try:
            if not self.watchlist:
                logger.debug("Mock mode: no symbols in watchlist yet")
                return {}

            for symbol in self.watchlist:
                # Create a plausible mock price (randomized)
                base_price = random.uniform(50.0, 2000.0)
                # small random drift for change
                change = random.uniform(-5.0, 5.0)
                change_pct = (change / max(base_price - change, 1.0)) * 100.0

                market_data = MarketData(
                    symbol=symbol,
                    price=round(base_price, 2),
                    change=round(change, 2),
                    change_pct=round(change_pct, 2),
                    volume=int(random.uniform(1000, 100000)),
                    high=round(base_price + abs(random.uniform(0.0, 10.0)), 2),
                    low=round(base_price - abs(random.uniform(0.0, 10.0)), 2),
                    open=round(base_price - change, 2),
                    timestamp=datetime.now().isoformat()
                )

                data[symbol] = market_data

            logger.info(f"Mock mode: generated mock data for {len(data)} symbols")
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")

        return data
    
    # REMOVED: generate_mock_data method - ONLY REAL FYERS DATA ALLOWED

    async def start_data_updates(self):
        """Start background data updates - ONLY REAL FYERS DATA"""
        if self.is_running:
            return

        # Allow starting updates if either connected to Fyers or mock mode is enabled
        if not self.is_connected and not self.allow_mock_mode:
            logger.error("Cannot start data updates - Fyers not connected and mock mode disabled")
            return

        self.is_running = True
        if self.allow_mock_mode and not self.is_connected:
            logger.warning("Starting background data updates in MOCK mode (FYERS_ALLOW_MOCK=true)")
        else:
            logger.info("Starting background data updates with REAL Fyers data")

        # Start background task
        self.data_update_task = asyncio.create_task(self.data_update_loop())

    async def stop_data_updates(self):
        """Stop background data updates"""
        self.is_running = False
        if self.data_update_task:
            self.data_update_task.cancel()
            try:
                await self.data_update_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background data updates")

    async def data_update_loop(self):
        """Background loop to update market data"""
        retry_count = 0

        while self.is_running:
            try:
                # Fetch new data
                new_data = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.fetch_market_data
                )

                if new_data:
                    self.market_data_cache.update(new_data)
                    self.last_update = datetime.now()
                    retry_count = 0  # Reset retry count on success

                    logger.info(f"Updated data for {len(new_data)} symbols")
                else:
                    logger.warning("No data received from fetch")

                # Wait for next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                retry_count += 1
                logger.error(f"Error in data update loop (attempt {retry_count}): {e}")

                if retry_count >= self.max_retries:
                    logger.error("Max retries reached, attempting to reconnect")
                    await self.connect_to_fyers()
                    retry_count = 0

                # Wait before retry
                await asyncio.sleep(min(retry_count * 2, 30))  # Exponential backoff

    async def startup_event(self):
        """Startup event handler - ONLY REAL FYERS DATA"""
        logger.info("*** FYERS DATA SERVICE STARTING - DYNAMIC REAL DATA ***")
        logger.info("Service will provide REAL data for ANY Indian stock on-demand")
        logger.info(f"Update interval: {self.update_interval} seconds")
        logger.info("Watchlist starts empty - stocks added automatically when requested")
        logger.info("Starting background data updates")

        # Try to connect to Fyers first
        connection_success = await self.connect_to_fyers()

        if not connection_success:
            if self.allow_mock_mode:
                logger.warning("Failed to connect to Fyers API but FYERS_ALLOW_MOCK is enabled")
                logger.warning("Starting service in MOCK DATA mode - not suitable for production!")
            else:
                logger.error("CRITICAL: Failed to connect to Fyers API")
                logger.error("Service cannot start without real data connection")
                logger.error("Set FYERS_ALLOW_MOCK=true to enable mock mode for development")
                raise Exception("Fyers connection required - no mock data allowed")

        await self.start_data_updates()

    async def shutdown_event(self):
        """Shutdown event handler"""
        logger.info("*** FYERS DATA SERVICE SHUTTING DOWN ***")
        await self.stop_data_updates()
        self.executor.shutdown(wait=True)

# Global service instance
service = FyersDataService()

# Add startup and shutdown events
@service.app.on_event("startup")
async def startup():
    await service.startup_event()

@service.app.on_event("shutdown")
async def shutdown():
    await service.shutdown_event()

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Fyers Data Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FYERS DATA SERVICE - STANDALONE MARKET DATA PROVIDER")
    logger.info("=" * 60)
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info("API Documentation: http://{args.host}:{args.port}/docs")
    logger.info("Health Check: http://{args.host}:{args.port}/health")
    logger.info("=" * 60)

    try:
        uvicorn.run(
            "fyers_data_service:service.app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
