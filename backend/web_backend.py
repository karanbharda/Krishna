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

# Import FastAPI components with fallback handling
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from pydantic import BaseModel
except ImportError as e:
    print(f"Error importing FastAPI components: {e}")
    print("Please install FastAPI dependencies:")
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Import the trading bot components
try:
    from backend.testindia import (
        ChatbotCommandHandler, VirtualPortfolio,
        TradingExecutor, DataFeed, Stock
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

class SettingsRequest(BaseModel):
    mode: Optional[str] = None
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class WebTradingBot:
    """Wrapper class for the trading bot to work with web interface"""
    
    def __init__(self, config):
        self.config = config
        self.portfolio = VirtualPortfolio(config)
        self.chatbot = ChatbotCommandHandler(self)
        self.executor = TradingExecutor(self.portfolio, config)
        self.data_feed = DataFeed(config["tickers"])
        self.stock_analyzer = Stock()
        self.is_running = False
        self.last_update = datetime.now()
        
    def start(self):
        """Start the trading bot"""
        self.is_running = True
        logger.info("Web Trading Bot started")
        
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Web Trading Bot stopped")
        
    def get_status(self):
        """Get current bot status"""
        return {
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat(),
            "mode": self.config.get("mode", "paper")
        }
        
    def get_portfolio_metrics(self):
        """Get portfolio metrics"""
        try:
            metrics = self.portfolio.get_metrics()
            starting_balance = self.portfolio.starting_balance
            total_return = metrics['total_value'] - starting_balance
            return_pct = (total_return / starting_balance) * 100 if starting_balance > 0 else 0
            
            return {
                "total_value": metrics['total_value'],
                "cash": metrics['cash'],
                "holdings": metrics['holdings'],
                "total_return": total_return,
                "return_percentage": return_pct,
                "realized_pnl": metrics.get('realized_pnl', 0),
                "unrealized_pnl": metrics.get('unrealized_pnl', 0),
                "total_exposure": metrics.get('total_exposure', 0),
                "active_positions": len(metrics['holdings'])
            }
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            return {
                "total_value": self.portfolio.starting_balance,
                "cash": self.portfolio.starting_balance,
                "holdings": {},
                "total_return": 0,
                "return_percentage": 0,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "total_exposure": 0,
                "active_positions": 0
            }
    
    def get_recent_trades(self, limit=10):
        """Get recent trades"""
        try:
            trades = self.portfolio.trade_log[-limit:] if self.portfolio.trade_log else []
            return list(reversed(trades))
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def process_chat_command(self, message):
        """Process chat command"""
        try:
            return self.chatbot.process_command(message)
        except Exception as e:
            logger.error(f"Error processing chat command: {e}")
            return f"Error processing command: {str(e)}"

def initialize_bot():
    """Initialize the trading bot with default configuration"""
    global trading_bot
    
    try:
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Default configuration
        config = {
            "tickers": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
                "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS", "LT.NS",
                "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "NESTLEIND.NS"
            ],
            "starting_balance": 1000000,  # ₹10 lakh
            "current_portfolio_value": 1000000,
            "current_pnl": 0,
            "mode": "paper",  # Default to paper mode for web interface
            "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
            "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
            "period": "3y",
            "prediction_days": 30,
            "benchmark_tickers": ["^NSEI"],
            "sleep_interval": 300,  # 5 minutes
            # Risk management settings
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
            "max_capital_per_trade": float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.25")),
            "max_trade_limit": int(os.getenv("MAX_TRADE_LIMIT", "10"))
        }
        
        trading_bot = WebTradingBot(config)
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message"""
    try:
        message = request.message

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        if trading_bot:
            response = trading_bot.process_chat_command(message)
            return ChatResponse(
                response=response,
                timestamp=datetime.now().isoformat()
            )
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start", response_model=MessageResponse)
async def start_bot():
    """Start the trading bot"""
    try:
        if trading_bot:
            trading_bot.start()
            return MessageResponse(message="Bot started successfully")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
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
                trading_bot.config['mode'] = request.mode
            if request.stop_loss_pct is not None:
                trading_bot.config['stop_loss_pct'] = request.stop_loss_pct
            if request.max_capital_per_trade is not None:
                trading_bot.config['max_capital_per_trade'] = request.max_capital_per_trade
            if request.max_trade_limit is not None:
                trading_bot.config['max_trade_limit'] = request.max_trade_limit

            return MessageResponse(message="Settings updated successfully")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
