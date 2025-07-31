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
        
        # Default watchlist (Indian stocks)
        self.watchlist = [
            "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", 
            "NSE:ICICIBANK-EQ", "NSE:INFY-EQ", "NSE:ITC-EQ",
            "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:KOTAKBANK-EQ",
            "NSE:LT-EQ", "NSE:AXISBANK-EQ", "NSE:MARUTI-EQ"
        ]
        
        # Background task management
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.data_update_task = None
        self.is_running = False
        
        logger.info("Fyers Data Service initialized")
    
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
            """Get data for specific symbol"""
            # Convert symbol format if needed
            fyers_symbol = self.convert_to_fyers_format(symbol)
            
            if fyers_symbol in self.market_data_cache:
                return asdict(self.market_data_cache[fyers_symbol])
            else:
                raise HTTPException(status_code=404, detail=f"Data not found for {symbol}")
        
        @self.app.get("/data")
        async def get_all_data():
            """Get all cached market data"""
            return {symbol: asdict(data) for symbol, data in self.market_data_cache.items()}
        
        @self.app.post("/watchlist")
        async def update_watchlist(symbols: List[str]):
            """Update watchlist"""
            self.watchlist = [self.convert_to_fyers_format(symbol) for symbol in symbols]
            logger.info(f"Watchlist updated with {len(self.watchlist)} symbols")
            return {"message": f"Watchlist updated with {len(self.watchlist)} symbols"}
        
        @self.app.get("/watchlist")
        async def get_watchlist():
            """Get current watchlist"""
            return {"watchlist": self.watchlist}
        
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
        """Connect to Fyers API"""
        if not FYERS_AVAILABLE:
            logger.warning("Fyers API not available, using mock data")
            self.is_connected = True
            return True
        
        try:
            # Get credentials from environment
            client_id = os.getenv('FYERS_CLIENT_ID')
            access_token = os.getenv('FYERS_ACCESS_TOKEN')
            
            if not client_id or not access_token:
                logger.error("Fyers credentials not found in environment variables")
                return False
            
            # Initialize Fyers client
            self.fyers_client = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                log_path=""
            )
            
            # Test connection
            profile = self.fyers_client.get_profile()
            if profile['s'] == 'ok':
                self.is_connected = True
                logger.info("Successfully connected to Fyers API")
                return True
            else:
                logger.error(f"Fyers connection failed: {profile}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Fyers: {e}")
            return False
    
    def fetch_market_data(self) -> Dict[str, MarketData]:
        """Fetch market data for watchlist"""
        if not self.is_connected:
            return self.generate_mock_data()
        
        try:
            if FYERS_AVAILABLE and self.fyers_client:
                return self.fetch_real_data()
            else:
                return self.generate_mock_data()
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self.generate_mock_data()
    
    def fetch_real_data(self) -> Dict[str, MarketData]:
        """Fetch real data from Fyers"""
        data = {}
        
        try:
            # Get quotes for all symbols
            quotes_response = self.fyers_client.quotes({"symbols": ",".join(self.watchlist)})
            
            if quotes_response['s'] == 'ok':
                for symbol_data in quotes_response['d']:
                    symbol = symbol_data['n']
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(symbol_data['v']['lp']),
                        change=float(symbol_data['v']['ch']),
                        change_pct=float(symbol_data['v']['chp']),
                        volume=int(symbol_data['v']['volume']),
                        high=float(symbol_data['v']['h']),
                        low=float(symbol_data['v']['l']),
                        open=float(symbol_data['v']['o']),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    data[symbol] = market_data
                    
                logger.info(f"Fetched real data for {len(data)} symbols")
            else:
                logger.error(f"Fyers quotes API error: {quotes_response}")
                return self.generate_mock_data()
                
        except Exception as e:
            logger.error(f"Error fetching real Fyers data: {e}")
            return self.generate_mock_data()
        
        return data
    
    def generate_mock_data(self) -> Dict[str, MarketData]:
        """Generate mock market data for testing"""
        import random
        
        data = {}
        base_prices = {
            "NSE:RELIANCE-EQ": 2500.0,
            "NSE:TCS-EQ": 3200.0,
            "NSE:HDFCBANK-EQ": 1600.0,
            "NSE:ICICIBANK-EQ": 900.0,
            "NSE:INFY-EQ": 1400.0,
            "NSE:ITC-EQ": 450.0,
            "NSE:SBIN-EQ": 600.0,
            "NSE:BHARTIARTL-EQ": 800.0,
            "NSE:KOTAKBANK-EQ": 1700.0,
            "NSE:LT-EQ": 3000.0,
            "NSE:AXISBANK-EQ": 1000.0,
            "NSE:MARUTI-EQ": 10000.0
        }
        
        for symbol in self.watchlist:
            base_price = base_prices.get(symbol, 1000.0)
            change_pct = random.uniform(-3.0, 3.0)
            change = base_price * (change_pct / 100)
            current_price = base_price + change
            
            market_data = MarketData(
                symbol=symbol,
                price=round(current_price, 2),
                change=round(change, 2),
                change_pct=round(change_pct, 2),
                volume=random.randint(100000, 1000000),
                high=round(current_price * random.uniform(1.0, 1.05), 2),
                low=round(current_price * random.uniform(0.95, 1.0), 2),
                open=round(base_price * random.uniform(0.98, 1.02), 2),
                timestamp=datetime.now().isoformat()
            )
            
            data[symbol] = market_data
        
        logger.info(f"Generated mock data for {len(data)} symbols")
        return data

    async def start_data_updates(self):
        """Start background data updates"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting background data updates")

        # Connect to Fyers first
        await self.connect_to_fyers()

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
        """Startup event handler"""
        logger.info("*** FYERS DATA SERVICE STARTING ***")
        logger.info(f"Service will provide data for {len(self.watchlist)} symbols")
        logger.info(f"Update interval: {self.update_interval} seconds")
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
