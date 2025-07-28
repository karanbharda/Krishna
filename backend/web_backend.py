#!/usr/bin/env python3
"""
FastAPI backend for the Indian Stock Trading Bot Web Interface
Provides REST API endpoints for the HTML/CSS/JS frontend
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import threading
import time

# Import FastAPI components with fallback handling
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel
    import asyncio
    import json
except ImportError as e:
    print(f"Error importing FastAPI components: {e}")
    print("Please install FastAPI dependencies:")
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Import new components for live trading
try:
    from portfolio_manager import DualPortfolioManager
    from dhan_client import DhanAPIClient
    from live_executor import LiveTradingExecutor
    LIVE_TRADING_AVAILABLE = True
except ImportError as e:
    print(f"Live trading components not available: {e}")
    LIVE_TRADING_AVAILABLE = False

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_trading_bot.log'),
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)

# Import the trading bot components
try:
    # Add the backend directory to the Python path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from testindia import (
        ChatbotCommandHandler, VirtualPortfolio,
        TradingExecutor, DataFeed, Stock, StockTradingBot
    )
except ImportError as e:
    print(f"Error importing trading bot components: {e}")
    print("Make sure testindia.py is in the same directory")
    sys.exit(1)

# Pydantic Models for Request/Response validation
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class WatchlistRequest(BaseModel):
    ticker: str
    action: str  # ADD or REMOVE

class WatchlistResponse(BaseModel):
    message: str
    tickers: List[str]

class BulkWatchlistRequest(BaseModel):
    tickers: List[str]
    action: str = "ADD"  # ADD or REMOVE

class BulkWatchlistResponse(BaseModel):
    message: str
    successful_tickers: List[str]
    failed_tickers: List[str]
    total_processed: int

class SettingsRequest(BaseModel):
    mode: Optional[str] = None
    riskLevel: Optional[str] = None
    stop_loss_pct: Optional[float] = None
    max_capital_per_trade: Optional[float] = None
    max_trade_limit: Optional[int] = None

class PortfolioMetrics(BaseModel):
    total_value: float
    cash: float
    holdings: Dict[str, Any]
    total_return: float
    return_percentage: float
    realized_pnl: float
    unrealized_pnl: float
    total_exposure: float
    active_positions: int

class BotStatus(BaseModel):
    is_running: bool
    last_update: str
    mode: str

class MessageResponse(BaseModel):
    message: str

# Logger already configured above

# Initialize FastAPI app
app = FastAPI(
    title="Indian Stock Trading Bot API",
    description="REST API for the Indian Stock Trading Bot Web Interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
trading_bot = None
bot_thread = None
bot_running = False

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        message_str = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

class WebTradingBot:
    """Wrapper class for the actual trading bot to work with web interface"""

    def __init__(self, config):
        self.config = config

        # Initialize dual portfolio manager
        if LIVE_TRADING_AVAILABLE:
            self.portfolio_manager = DualPortfolioManager()
            self.portfolio_manager.switch_mode(config.get("mode", "paper"))
        else:
            self.portfolio_manager = None

        # Initialize the actual StockTradingBot from testindia.py
        self.trading_bot = StockTradingBot(config)
        self.is_running = False
        self.last_update = datetime.now()
        self.trading_thread = None

        # Initialize live trading components if available
        self.live_executor = None
        self.dhan_client = None

        if LIVE_TRADING_AVAILABLE and config.get("mode") == "live":
            self._initialize_live_trading()

        # Register WebSocket callback for real-time updates
        self.trading_bot.portfolio.add_trade_callback(self._on_trade_executed)

    def _initialize_live_trading(self):
        """Initialize live trading components"""
        try:
            if not self.config.get("dhan_client_id") or not self.config.get("dhan_access_token"):
                logger.error("Dhan credentials not found in config")
                return False

            # Initialize Dhan client
            self.dhan_client = DhanAPIClient(
                client_id=self.config["dhan_client_id"],
                access_token=self.config["dhan_access_token"]
            )

            # Validate connection
            if not self.dhan_client.validate_connection():
                logger.error("Failed to validate Dhan API connection")
                return False

            # Initialize live executor
            self.live_executor = LiveTradingExecutor(
                portfolio=self.trading_bot.portfolio,
                config=self.config
            )

            # Sync portfolio with Dhan account
            if self.live_executor.sync_portfolio_with_dhan():
                logger.info("Live trading initialized successfully")
                return True
            else:
                logger.error("Failed to sync portfolio with Dhan")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize live trading: {e}")
            return False

    def switch_trading_mode(self, new_mode: str) -> bool:
        """Switch between paper and live trading modes"""
        try:
            if new_mode not in ["paper", "live"]:
                logger.error(f"Invalid trading mode: {new_mode}")
                return False

            if new_mode == self.config.get("mode"):
                logger.info(f"Already in {new_mode} mode")
                return True

            # Stop bot if running
            was_running = self.is_running
            if was_running:
                self.stop()
                time.sleep(1)  # Give time to stop

            # Switch portfolio manager mode
            if self.portfolio_manager:
                self.portfolio_manager.switch_mode(new_mode)

            # Update config
            old_mode = self.config.get("mode", "paper")
            self.config["mode"] = new_mode

            # Initialize/deinitialize live trading components
            if new_mode == "live" and LIVE_TRADING_AVAILABLE:
                if not self._initialize_live_trading():
                    logger.error("Failed to initialize live trading, reverting to paper mode")
                    self.config["mode"] = "paper"
                    if self.portfolio_manager:
                        self.portfolio_manager.switch_mode("paper")
                    # Return True because we successfully handled the failure by reverting
                    logger.info("Successfully reverted to paper mode after live trading failure")
                    return True
            else:
                # Clear live trading components for paper mode
                self.live_executor = None
                self.dhan_client = None

            # Update trading bot config
            self.trading_bot.config.update(self.config)

            # Restart bot if it was running
            if was_running:
                time.sleep(1)
                self.start()

            logger.info(f"Successfully switched from {old_mode} to {new_mode} mode")
            return True

        except Exception as e:
            logger.error(f"Failed to switch trading mode: {e}")
            return False

    def start(self):
        """Start the trading bot"""
        if not self.is_running:
            self.is_running = True
            logger.info("Starting Indian Stock Trading Bot...")
            logger.info(f"Trading Mode: {self.config.get('mode', 'paper').upper()}")
            logger.info(f"Starting Balance: Rs.{self.config.get('starting_balance', 1000000):,.2f}")
            logger.info(f"Watchlist: {', '.join(self.config['tickers'])}")
            logger.info("=" * 60)

            # Start the actual trading bot in a separate thread
            self.trading_thread = threading.Thread(target=self.trading_bot.run, daemon=True)
            self.trading_thread.start()
            logger.info("Web Trading Bot started successfully")
        else:
            logger.info("Trading bot is already running")

    def stop(self):
        """Stop the trading bot"""
        if self.is_running:
            self.is_running = False
            # Stop the actual trading bot
            self.trading_bot.bot_running = False
            logger.info("Stopping Trading Bot...")
            if self.trading_thread and self.trading_thread.is_alive():
                logger.info("Waiting for trading thread to finish...")
                # Wait for the thread to finish with a timeout
                self.trading_thread.join(timeout=10.0)
                if self.trading_thread.is_alive():
                    logger.warning("Trading thread did not stop within timeout, forcing stop...")
                else:
                    logger.info("Trading thread stopped successfully")
            logger.info("Web Trading Bot stopped successfully")
        else:
            logger.info("Trading bot is already stopped")



    def get_status(self):
        """Get current bot status"""
        return {
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat(),
            "mode": self.config.get("mode", "paper")
        }

    def get_portfolio_metrics(self):
        """Get portfolio metrics from saved portfolio file"""
        import json
        import os
        import yfinance as yf

        try:
            # Try to read from the actual portfolio file created by the trading bot
            # Use absolute path to data folder and current mode
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            current_mode = self.config.get("mode", "paper")
            portfolio_file = os.path.join(project_root, "data", f"{current_mode}_portfolio.json")
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)

                starting_balance = portfolio_data.get('starting_balance', 10000)
                cash = portfolio_data.get('cash', starting_balance)
                holdings = portfolio_data.get('holdings', {})

                # Get current prices for unrealized P&L calculation
                current_prices = {}
                unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)  # Use saved value as fallback
                price_fetch_success = False

                if holdings:
                    try:
                        for ticker in holdings.keys():
                            stock = yf.Ticker(ticker)
                            hist = stock.history(period="1d")
                            if not hist.empty:
                                current_prices[ticker] = hist['Close'].iloc[-1]
                                price_fetch_success = True
                            else:
                                current_prices[ticker] = holdings[ticker]['avg_price']  # Fallback to avg price
                    except Exception as e:
                        logger.warning(f"Error fetching current prices: {e}")
                        # Fallback: use average prices
                        for ticker, data in holdings.items():
                            current_prices[ticker] = data['avg_price']

                # Calculate unrealized P&L with current prices only if we successfully fetched prices
                if price_fetch_success:
                    unrealized_pnl = 0
                    for ticker, data in holdings.items():
                        current_price = current_prices.get(ticker, data['avg_price'])
                        unrealized_pnl += (current_price - data['avg_price']) * data['qty']

                # Calculate total exposure and total value with current prices
                total_exposure = sum(data['qty'] * data['avg_price'] for data in holdings.values())

                # If we successfully fetched current prices, use them
                if price_fetch_success:
                    current_market_value = sum(data['qty'] * current_prices.get(ticker, data['avg_price'])
                                             for ticker, data in holdings.items())
                else:
                    # If we couldn't fetch current prices, calculate market value using unrealized P&L
                    current_market_value = total_exposure + unrealized_pnl

                total_value = cash + current_market_value

                # Calculate cash invested (starting balance minus current cash)
                cash_invested = starting_balance - cash

                # Calculate total return based on unrealized P&L (more accurate)
                # Total return = unrealized P&L + realized P&L
                realized_pnl = portfolio_data.get('realized_pnl', 0)
                total_return = unrealized_pnl + realized_pnl
                return_pct = (total_return / cash_invested) * 100 if cash_invested > 0 else 0

                # Add current prices to holdings for frontend
                enriched_holdings = {}
                for ticker, data in holdings.items():
                    enriched_holdings[ticker] = {
                        **data,
                        'currentPrice': current_prices.get(ticker, data['avg_price'])
                    }

                # Get trade log
                trade_log = self.get_recent_trades(limit=100)  # Get all trades for portfolio

                return {
                    "total_value": total_value,
                    "cash": cash,
                    "holdings": enriched_holdings,
                    "total_return": total_return,
                    "return_percentage": return_pct,
                    "realized_pnl": portfolio_data.get('realized_pnl', 0),
                    "unrealized_pnl": unrealized_pnl,
                    "total_exposure": total_exposure,
                    "active_positions": len(holdings),
                    "trade_log": trade_log
                }
            else:
                # Fallback to default values if no portfolio file exists
                starting_balance = self.config.get('starting_balance', 10000)
                return {
                    "total_value": starting_balance,
                    "cash": starting_balance,
                    "holdings": {},
                    "total_return": 0,
                    "return_percentage": 0,
                    "realized_pnl": 0,
                    "unrealized_pnl": 0,
                    "total_exposure": 0,
                    "active_positions": 0,
                    "trade_log": []
                }
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            starting_balance = self.config.get('starting_balance', 10000)
            return {
                "total_value": starting_balance,
                "cash": starting_balance,
                "holdings": {},
                "total_return": 0,
                "return_percentage": 0,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "total_exposure": 0,
                "active_positions": 0,
                "trade_log": []
            }

    def get_recent_trades(self, limit=10):
        """Get recent trades from saved trade log file"""
        import json
        import os

        try:
            # Try to read from the actual trade log file created by the trading bot
            # Use absolute path to data folder and current mode
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            current_mode = self.config.get("mode", "paper")
            trade_log_file = os.path.join(project_root, "data", f"{current_mode}_trades.json")
            if os.path.exists(trade_log_file):
                with open(trade_log_file, 'r') as f:
                    trades = json.load(f)

                # Return the most recent trades (reversed order)
                recent_trades = trades[-limit:] if trades else []
                return list(reversed(recent_trades))
            else:
                logger.warning("Trade log file not found")
                return []
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def process_chat_command(self, message):
        """Process chat command"""
        try:
            return self.trading_bot.chatbot.process_command(message)
        except Exception as e:
            logger.error(f"Error processing chat command: {e}")
            return f"Error processing command: {str(e)}"

    def get_complete_bot_data(self):
        """Get complete bot data for React frontend"""
        try:
            portfolio_metrics = self.get_portfolio_metrics()

            return {
                "isRunning": self.is_running,
                "config": {
                    "mode": self.config.get("mode", "paper"),
                    "tickers": self.config.get("tickers", []),
                    "stopLossPct": self.config.get("stop_loss_pct", 0.05),
                    "maxAllocation": self.config.get("max_capital_per_trade", 0.25),
                    "maxTradeLimit": self.config.get("max_trade_limit", 10)
                },
                "portfolio": {
                    "totalValue": portfolio_metrics["total_value"],
                    "cash": portfolio_metrics["cash"],
                    "holdings": portfolio_metrics["holdings"],
                    "startingBalance": self.trading_bot.portfolio.starting_balance,
                    "unrealizedPnL": portfolio_metrics["unrealized_pnl"],
                    "realizedPnL": portfolio_metrics["realized_pnl"],
                    "tradeLog": self.get_recent_trades(50)
                },
                "lastUpdate": self.last_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting complete bot data: {e}")
            return {
                "isRunning": False,
                "config": {
                    "mode": "paper",
                    "tickers": [],
                    "stopLossPct": 0.05,
                    "maxAllocation": 0.25,
                    "maxTradeLimit": 10
                },
                "portfolio": {
                    "totalValue": 10000,
                    "cash": 10000,
                    "holdings": {},
                    "startingBalance": 10000,
                    "unrealizedPnL": 0,
                    "realizedPnL": 0,
                    "tradeLog": []
                },
                "lastUpdate": datetime.now().isoformat()
            }

    async def broadcast_portfolio_update(self):
        """Broadcast portfolio update to all connected WebSocket clients"""
        try:
            portfolio_metrics = self.get_portfolio_metrics()
            update_data = {
                "type": "portfolio_update",
                "data": {
                    "totalValue": portfolio_metrics["total_value"],
                    "cash": portfolio_metrics["cash"],
                    "holdings": portfolio_metrics["holdings"],
                    "unrealizedPnL": portfolio_metrics["unrealized_pnl"],
                    "realizedPnL": portfolio_metrics["realized_pnl"],
                    "tradeLog": self.get_recent_trades(10)
                },
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast(update_data)
            logger.info("Portfolio update broadcasted to WebSocket clients")
        except Exception as e:
            logger.error(f"Error broadcasting portfolio update: {e}")

    async def broadcast_trade_update(self, trade_data):
        """Broadcast trade update to all connected WebSocket clients"""
        try:
            update_data = {
                "type": "trade_update",
                "data": trade_data,
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast(update_data)
            logger.info(f"Trade update broadcasted: {trade_data}")
        except Exception as e:
            logger.error(f"Error broadcasting trade update: {e}")

    def _on_trade_executed(self, trade_data):
        """Callback method called when a trade is executed"""
        try:
            # Schedule the broadcast in the main event loop
            import asyncio
            import threading

            def run_broadcast():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.broadcast_trade_update(trade_data))
                    loop.run_until_complete(self.broadcast_portfolio_update())
                    loop.close()
                except Exception as e:
                    logger.error(f"Error in broadcast thread: {e}")

            # Run broadcast in a separate thread to avoid blocking
            broadcast_thread = threading.Thread(target=run_broadcast, daemon=True)
            broadcast_thread.start()

        except Exception as e:
            logger.error(f"Error in trade callback: {e}")

def apply_risk_level_settings(bot, risk_level, custom_stop_loss=None, custom_allocation=None):
    """Apply risk level settings to the trading bot"""
    try:
        # Define risk level mappings
        risk_mappings = {
            "LOW": {"stop_loss": 0.03, "allocation": 0.15},      # 3% stop-loss, 15% allocation
            "MEDIUM": {"stop_loss": 0.05, "allocation": 0.25},   # 5% stop-loss, 25% allocation
            "HIGH": {"stop_loss": 0.08, "allocation": 0.35}      # 8% stop-loss, 35% allocation
        }

        if risk_level == "CUSTOM":
            # Use custom values if provided
            if custom_stop_loss is not None:
                bot.config['stop_loss_pct'] = custom_stop_loss
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.stop_loss_pct = custom_stop_loss
            if custom_allocation is not None:
                bot.config['max_capital_per_trade'] = custom_allocation
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.max_capital_per_trade = custom_allocation
        elif risk_level in risk_mappings:
            # Apply predefined risk level settings
            settings = risk_mappings[risk_level]
            bot.config['stop_loss_pct'] = settings['stop_loss']
            bot.config['max_capital_per_trade'] = settings['allocation']

            # Update executor if it exists
            if hasattr(bot, 'executor') and bot.executor:
                bot.executor.stop_loss_pct = settings['stop_loss']
                bot.executor.max_capital_per_trade = settings['allocation']

        logger.info(f"Applied {risk_level} risk settings: "
                   f"Stop Loss={bot.config.get('stop_loss_pct')*100}%, "
                   f"Max Allocation={bot.config.get('max_capital_per_trade')*100}%")

    except Exception as e:
        logger.error(f"Error applying risk level settings: {e}")

def initialize_bot():
    """Initialize the trading bot with default configuration"""
    global trading_bot
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Default configuration
        config = {
            "tickers": [],  # Empty by default - users can add tickers manually
            "starting_balance": 10000,  # Rs.10 thousand
            "current_portfolio_value": 10000,
            "current_pnl": 0,
            "mode": "paper",  # Default to paper mode for web interface
            "riskLevel": "MEDIUM",  # Default risk level
            "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
            "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
            "period": "3y",
            "prediction_days": 30,
            "benchmark_tickers": ["^NSEI"],
            "sleep_interval": 30,  # 30 seconds
            # Risk management settings - will be set by risk level
            "stop_loss_pct": 0.05,  # Default 5% (MEDIUM)
            "max_capital_per_trade": 0.25,  # Default 25% (MEDIUM)
            "max_trade_limit": 10
        }
        
        trading_bot = WebTradingBot(config)

        # Apply default risk level settings
        apply_risk_level_settings(trading_bot, config["riskLevel"])

        logger.info("Trading bot initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing trading bot: {e}")
        raise

# Static file serving
app.mount("/static", StaticFiles(directory="."), name="static")

# API Routes

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page"""
    try:
        with open('web_interface.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Web interface HTML file not found")

@app.get("/styles.css")
async def styles():
    """Serve the CSS file"""
    try:
        return FileResponse('styles.css', media_type='text/css')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/app.js")
async def app_js():
    """Serve the JavaScript file"""
    try:
        return FileResponse('app.js', media_type='application/javascript')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/api/status", response_model=BotStatus)
async def get_status():
    """Get bot status"""
    try:
        if trading_bot:
            status = trading_bot.get_status()
            return BotStatus(**status)
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bot-data")
async def get_bot_data():
    """Get complete bot data for React frontend"""
    try:
        if trading_bot:
            return trading_bot.get_complete_bot_data()
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except Exception as e:
        logger.error(f"Error getting bot data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio", response_model=PortfolioMetrics)
async def get_portfolio():
    """Get portfolio metrics"""
    try:
        if trading_bot:
            metrics = trading_bot.get_portfolio_metrics()
            return PortfolioMetrics(**metrics)
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(limit: int = 10):
    """Get recent trades"""
    try:
        if trading_bot:
            trades = trading_bot.get_recent_trades(limit)
            return trades
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/watchlist")
async def get_watchlist():
    """Get current watchlist"""
    try:
        if trading_bot:
            return trading_bot.config["tickers"]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist", response_model=WatchlistResponse)
async def update_watchlist(request: WatchlistRequest):
    """Add or remove ticker from watchlist"""
    try:
        ticker = request.ticker.upper()
        action = request.action.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")

        if trading_bot:
            current_tickers = trading_bot.config["tickers"]

            if action == "ADD":
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    message = f"Added {ticker} to watchlist"
                else:
                    message = f"{ticker} is already in watchlist"
            elif action == "REMOVE":
                if ticker in current_tickers:
                    current_tickers.remove(ticker)
                    message = f"Removed {ticker} from watchlist"
                else:
                    message = f"{ticker} is not in watchlist"
            else:
                raise HTTPException(status_code=400, detail="Invalid action. Use ADD or REMOVE")

            return WatchlistResponse(message=message, tickers=current_tickers)
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/bulk", response_model=BulkWatchlistResponse)
async def bulk_update_watchlist(request: BulkWatchlistRequest):
    """Add or remove multiple tickers from watchlist"""
    try:
        if not trading_bot:
            raise HTTPException(status_code=503, detail="Trading bot not initialized")

        action = request.action.upper()
        if action not in ["ADD", "REMOVE"]:
            raise HTTPException(status_code=400, detail="Action must be ADD or REMOVE")

        successful_tickers = []
        failed_tickers = []

        for ticker in request.tickers:
            try:
                ticker = ticker.strip().upper()

                # Validate ticker format
                if not ticker:
                    failed_tickers.append(f"{ticker}: Empty ticker")
                    continue

                # Add .NS suffix if not present for Indian stocks
                if not ticker.endswith(('.NS', '.BO')):
                    ticker += '.NS'

                # Validate ticker format
                if not ticker.replace('.', '').replace('-', '').replace('&', '').isalnum():
                    failed_tickers.append(f"{ticker}: Invalid format")
                    continue

                if action == "ADD":
                    if ticker in trading_bot.config["tickers"]:
                        failed_tickers.append(f"{ticker}: Already in watchlist")
                        continue

                    # Add ticker to config
                    trading_bot.config["tickers"].append(ticker)
                    successful_tickers.append(ticker)
                    logger.info(f"Added ticker {ticker} to watchlist via bulk upload")

                elif action == "REMOVE":
                    if ticker not in trading_bot.config["tickers"]:
                        failed_tickers.append(f"{ticker}: Not in watchlist")
                        continue

                    # Remove ticker from config
                    trading_bot.config["tickers"].remove(ticker)
                    successful_tickers.append(ticker)
                    logger.info(f"Removed ticker {ticker} from watchlist via bulk upload")

            except Exception as e:
                failed_tickers.append(f"{ticker}: {str(e)}")
                logger.error(f"Error processing ticker {ticker}: {e}")

        # Update data feed with new tickers
        if successful_tickers and action == "ADD":
            try:
                trading_bot.data_feed = DataFeed(trading_bot.config["tickers"])
                logger.info(f"Updated data feed with {len(successful_tickers)} new tickers")
            except Exception as e:
                logger.error(f"Error updating data feed: {e}")

        # Prepare response message
        if successful_tickers and not failed_tickers:
            message = f"Successfully {action.lower()}ed {len(successful_tickers)} ticker(s)"
        elif successful_tickers and failed_tickers:
            message = f"Processed {len(successful_tickers)} ticker(s) successfully, {len(failed_tickers)} failed"
        elif failed_tickers and not successful_tickers:
            message = f"Failed to process all {len(failed_tickers)} ticker(s)"
        else:
            message = "No tickers processed"

        return BulkWatchlistResponse(
            message=message,
            successful_tickers=successful_tickers,
            failed_tickers=failed_tickers,
            total_processed=len(request.tickers)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk watchlist update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message with Advanced Market Agent (LangChain + LangGraph + Fyers)"""
    try:
        message = request.message.strip()

        if not message:
            return ChatResponse(
                response="Please enter a message.",
                timestamp=datetime.now().isoformat()
            )

        # Use the Dynamic Market Expert for ALL queries
        try:
            from dynamic_market_expert import DynamicMarketExpert

            # Initialize the market expert (cached for performance)
            if not hasattr(chat, '_market_expert'):
                chat._market_expert = DynamicMarketExpert()
                logger.info("Dynamic Market Expert initialized for web chat")

            # Process query with timeout protection
            import threading
            import queue

            result_queue = queue.Queue()

            def process_with_expert():
                try:
                    result = chat._market_expert.process_query(message)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            thread = threading.Thread(target=process_with_expert)
            thread.daemon = True
            thread.start()
            thread.join(timeout=15)  # 15 second timeout

            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success" and result:
                    return ChatResponse(
                        response=result,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    logger.error(f"Expert processing error: {result}")
            else:
                logger.warning("Dynamic Market Expert response timed out")

        except ImportError as e:
            logger.error(f"Could not import Dynamic Market Expert: {e}")
        except Exception as e:
            logger.error(f"Error with Dynamic Market Expert: {e}")

        # Fallback to direct professional response with live data
        try:
            # Use existing trading bot components
            if hasattr(trading_bot, 'llm'):
                llm = trading_bot.llm
            else:
                llm = None

            # Use the Dynamic Market Expert instead
            try:
                from dynamic_market_expert import DynamicMarketExpert
                market_expert = DynamicMarketExpert()
                response = market_expert.process_query(message)
                return {"response": response, "timestamp": datetime.now().isoformat()}
            except Exception as expert_error:
                logger.error(f"Dynamic Market Expert error: {expert_error}")

            # Simple fallback response
            if True:  # Always execute fallback
                # Simple fallback response
                pass

        except Exception as e:
            logger.error(f"Error with fallback response: {e}")

        # Handle commands
        if message.startswith('/') and trading_bot:
            try:
                response = trading_bot.process_chat_command(message)
                return ChatResponse(
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                logger.error(f"Error with command: {e}")

        # Final professional fallback
        return ChatResponse(
            response=f"""I'm your professional stock market advisor! 📈

I can help you with:
• **Live Stock Prices** - "What's the price of {', '.join(['Reliance', 'TCS', 'HDFC Bank'])}?"
• **Market Analysis** - "How is the IT sector performing?"
• **Investment Advice** - "Should I buy banking stocks now?"
• **Portfolio Management** - Use /get_pnl, /list_positions

**Current Market Focus:** Indian equities (NSE/BSE)
**Data Source:** Live Fyers API integration

What would you like to analyze today?""",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return ChatResponse(
            response="I apologize for the error. Please try asking about stock prices or portfolio information.",
            timestamp=datetime.now().isoformat()
        )

@app.post("/api/start", response_model=MessageResponse)
async def start_bot():
    """Start the trading bot"""
    try:
        global trading_bot

        # Initialize bot if not already initialized
        if not trading_bot:
            try:
                initialize_bot()
                logger.info("Bot initialized before starting")
            except Exception as init_error:
                logger.error(f"Failed to initialize bot: {init_error}")
                raise HTTPException(status_code=500, detail=f"Failed to initialize bot: {str(init_error)}")

        if trading_bot:
            # Apply current risk level settings before starting
            risk_level = trading_bot.config.get("riskLevel", "MEDIUM")
            apply_risk_level_settings(trading_bot, risk_level)

            trading_bot.start()
            stop_loss_pct = trading_bot.config.get('stop_loss_pct', 0.05) * 100
            max_allocation_pct = trading_bot.config.get('max_capital_per_trade', 0.25) * 100
            logger.info(f"Trading bot started with {risk_level} risk level")
            return MessageResponse(message=f"Bot started successfully with {risk_level} risk level (Stop Loss: {stop_loss_pct}%, Max Allocation: {max_allocation_pct}%)")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/init", response_model=MessageResponse)
async def init_bot():
    """Manually initialize the trading bot"""
    try:
        global trading_bot
        initialize_bot()
        return MessageResponse(message="Trading bot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop", response_model=MessageResponse)
async def stop_bot():
    """Stop the trading bot"""
    try:
        if trading_bot:
            trading_bot.stop()
            return MessageResponse(message="Bot stopped successfully")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def get_settings():
    """Get current settings"""
    try:
        if trading_bot:
            return {
                "mode": trading_bot.config.get("mode", "paper"),
                "riskLevel": trading_bot.config.get("riskLevel", "MEDIUM"),
                "stop_loss_pct": trading_bot.config.get("stop_loss_pct", 0.05),
                "max_capital_per_trade": trading_bot.config.get("max_capital_per_trade", 0.25),
                "max_trade_limit": trading_bot.config.get("max_trade_limit", 10)
            }
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings", response_model=MessageResponse)
async def update_settings(request: SettingsRequest):
    """Update bot settings"""
    try:
        if trading_bot:
            # Update configuration
            if request.mode is not None:
                # Handle mode switching
                old_mode = trading_bot.config.get('mode', 'paper')
                if request.mode != old_mode:
                    if trading_bot.switch_trading_mode(request.mode):
                        # Check if the mode actually changed (could have reverted)
                        actual_mode = trading_bot.config.get('mode', 'paper')
                        if actual_mode != request.mode:
                            logger.warning(f"Requested {request.mode} mode but reverted to {actual_mode} mode")
                        else:
                            logger.info(f"Successfully switched from {old_mode} to {request.mode} mode")
                    else:
                        raise HTTPException(status_code=400, detail=f"Failed to switch to {request.mode} mode")
                else:
                    trading_bot.config['mode'] = request.mode
            if request.riskLevel is not None:
                trading_bot.config['riskLevel'] = request.riskLevel
                # Apply risk level settings dynamically
                # For predefined levels, don't pass custom values so they use the mappings
                if request.riskLevel in ["LOW", "MEDIUM", "HIGH"]:
                    apply_risk_level_settings(trading_bot, request.riskLevel)
                else:
                    # For CUSTOM, use the provided values
                    apply_risk_level_settings(trading_bot, request.riskLevel, request.stop_loss_pct, request.max_capital_per_trade)
            if request.stop_loss_pct is not None:
                trading_bot.config['stop_loss_pct'] = request.stop_loss_pct
                # Update executor if it exists
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.stop_loss_pct = request.stop_loss_pct
            if request.max_capital_per_trade is not None:
                trading_bot.config['max_capital_per_trade'] = request.max_capital_per_trade
                # Update executor if it exists
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.max_capital_per_trade = request.max_capital_per_trade
            if request.max_trade_limit is not None:
                trading_bot.config['max_trade_limit'] = request.max_trade_limit

            logger.info(f"Settings updated: Mode={trading_bot.config.get('mode')}, "
                       f"Risk Level={trading_bot.config.get('riskLevel')}, "
                       f"Stop Loss={trading_bot.config.get('stop_loss_pct')*100}%, "
                       f"Max Allocation={trading_bot.config.get('max_capital_per_trade')*100}%")

            return MessageResponse(message="Settings updated successfully")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/live-status")
async def get_live_trading_status():
    """Get live trading status and connection info"""
    try:
        if not LIVE_TRADING_AVAILABLE:
            return {
                "available": False,
                "message": "Live trading components not installed"
            }

        if trading_bot and trading_bot.config.get("mode") == "live":
            # Check Dhan connection
            dhan_connected = False
            market_status = "UNKNOWN"
            account_info = {}

            if trading_bot.dhan_client:
                try:
                    dhan_connected = trading_bot.dhan_client.validate_connection()
                    if dhan_connected:
                        market_status_data = trading_bot.dhan_client.get_market_status()
                        market_status = market_status_data.get("marketStatus", "UNKNOWN")

                        # Get account info
                        profile = trading_bot.dhan_client.get_profile()
                        funds = trading_bot.dhan_client.get_funds()

                        account_info = {
                            "client_id": profile.get("clientId", ""),
                            "available_cash": funds.get("availablecash", 0),
                            "used_margin": funds.get("sodlimit", 0) - funds.get("availablecash", 0)
                        }
                except Exception as e:
                    logger.error(f"Error getting live trading status: {e}")

            return {
                "available": True,
                "mode": "live",
                "dhan_connected": dhan_connected,
                "market_status": market_status,
                "account_info": account_info,
                "portfolio_synced": trading_bot.live_executor is not None
            }
        else:
            return {
                "available": True,
                "mode": "paper",
                "message": "Currently in paper trading mode"
            }

    except Exception as e:
        logger.error(f"Error getting live trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        # Send initial data when client connects
        if trading_bot:
            initial_data = trading_bot.get_complete_bot_data()
            await manager.send_personal_message(
                json.dumps({
                    "type": "initial_data",
                    "data": initial_data,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )

        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            if data == "ping":
                await manager.send_personal_message("pong", websocket)
            elif data == "get_initial_data":
                if trading_bot:
                    initial_data = trading_bot.get_complete_bot_data()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "initial_data",
                            "data": initial_data,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """Run the FastAPI web server with uvicorn"""
    try:
        # Initialize the trading bot
        initialize_bot()

        logger.info(f"Starting FastAPI web server on http://{host}:{port}")
        logger.info("Web interface will be available at the above URL")
        logger.info("API documentation available at http://{host}:{port}/docs")

        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info" if debug else "warning",
            reload=debug,
            access_log=debug
        )

        # Run the FastAPI app with uvicorn
        server = uvicorn.Server(config)
        server.run()

    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the trading bot on startup"""
    try:
        initialize_bot()
        logger.info("Trading bot initialized on startup")
    except Exception as e:
        logger.error(f"Error initializing bot on startup: {e}")
        # Try to initialize with minimal config as fallback
        try:
            global trading_bot
            if not trading_bot:
                logger.info("Attempting fallback initialization...")
                from dotenv import load_dotenv
                load_dotenv()

                minimal_config = {
                    "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
                    "starting_balance": 10000,
                    "current_portfolio_value": 10000,
                    "current_pnl": 0,
                    "mode": "paper"
                }
                trading_bot = WebTradingBot(minimal_config)
                logger.info("Fallback trading bot initialized")
        except Exception as fallback_error:
            logger.error(f"Fallback initialization also failed: {fallback_error}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global trading_bot
    if trading_bot:
        trading_bot.stop()
        logger.info("Trading bot stopped on shutdown")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indian Stock Trading Bot Web Interface (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    try:
        run_web_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        sys.exit(1)
