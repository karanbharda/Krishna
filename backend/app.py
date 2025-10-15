#!/usr/bin/env python3
"""
FastAPI Backend for Trading Bot with Dynamic Global News Scraper Integration
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import redis

# Import our trading system components
from mcp_server.tools.sentiment_tool import SentimentTool
from testindia import Stock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for requests
class EvaluateBuyRequest(BaseModel):
    symbol: str
    mode: str = "auto"  # "auto" or "approval"
    session_id: Optional[str] = None

class ConfirmTradeRequest(BaseModel):
    symbol: str
    decision: str  # "approve" or "reject"
    session_id: str

# Initialize FastAPI app
app = FastAPI(title="Trading Bot API", version="1.0.0")

# Initialize Redis for caching (if available)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except:
    redis_client = None
    logger.warning("Redis not available, continuing without caching")

# Initialize trading components
sentiment_tool = SentimentTool({
    "tool_id": "dynamic_news_sentiment_tool",
    "sentiment_sources": ["news", "social", "market", "indian_news"]
})

# Initialize stock analyzer
stock_analyzer = Stock()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Watchlist symbols for background polling
watchlist_symbols = ["PARAS.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Background task for polling news
async def poll_news_for_watchlist():
    """Background task to poll news for watchlist symbols"""
    while True:
        try:
            for symbol in watchlist_symbols:
                # Get sentiment analysis for the symbol
                session_id = str(uuid.uuid4())
                result = await sentiment_tool.analyze_sentiment({
                    "symbol": symbol,
                    "sources": ["news", "social", "market", "indian_news"],
                    "lookback_days": 1,
                    "include_news_items": True
                }, session_id)
                
                # Broadcast to all active WebSocket connections
                if result.status == "SUCCESS":
                    news_data = {
                        "type": "news_update",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        "sentiment": result.data.get("overall_sentiment", {}),
                        "news_items_count": result.data.get("news_items_count", 0)
                    }
                    
                    # Send to all active WebSocket connections
                    for connection in active_connections[:]:  # Create a copy to avoid modification during iteration
                        try:
                            await connection.send_text(json.dumps(news_data))
                        except:
                            active_connections.remove(connection)
                
                # Cache result in Redis if available
                if redis_client and result.status == "SUCCESS":
                    cache_key = f"sentiment:{symbol}"
                    redis_client.setex(cache_key, 300, json.dumps(result.data))  # Cache for 5 minutes
                    
            # Wait for 60 seconds before next poll
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in news polling: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    # Start the news polling task
    asyncio.create_task(poll_news_for_watchlist())
    logger.info("Background news polling task started")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Trading Bot API with Dynamic Global News Scraper"}

@app.post("/evaluate_buy")
async def evaluate_buy(request: EvaluateBuyRequest):
    """Evaluate buy decision based on sentiment analysis"""
    try:
        logger.info(f"Evaluating buy for {request.symbol} in {request.mode} mode")
        
        # Check Redis cache first
        cache_key = f"sentiment:{request.symbol}"
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Using cached sentiment data for {request.symbol}")
                sentiment_data = json.loads(cached_result)
                # Return cached result with boosted confidence
                return {
                    "symbol": request.symbol,
                    "action": "BUY" if sentiment_data.get("overall_sentiment", {}).get("compound", 0) > 0.1 else "HOLD",
                    "confidence": min(sentiment_data.get("overall_sentiment", {}).get("confidence", 0.5) * 1.2, 1.0),
                    "sentiment": sentiment_data.get("overall_sentiment", {}),
                    "sources": sentiment_data.get("analysis_metadata", {}).get("sources_analyzed", []),
                    "cached": True
                }
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get sentiment analysis
        sentiment_result = await sentiment_tool.analyze_sentiment({
            "symbol": request.symbol,
            "sources": ["news", "social", "market", "indian_news"],
            "lookback_days": 7,
            "include_news_items": True
        }, session_id)
        
        if sentiment_result.status != "SUCCESS":
            raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {sentiment_result.error}")
        
        # Extract sentiment data
        sentiment_data = sentiment_result.data
        overall_sentiment = sentiment_data.get("overall_sentiment", {})
        compound_score = overall_sentiment.get("compound", 0)
        confidence = overall_sentiment.get("confidence", 0.5)
        
        # Apply regime multipliers (simplified)
        # In a real implementation, this would come from market regime detection
        regime_multiplier = 1.0
        if compound_score > 0.3:  # Trending market
            regime_multiplier = 1.4
        elif compound_score < -0.3:  # Volatile market
            regime_multiplier = 0.6
        
        # Adjust confidence based on regime
        adjusted_confidence = min(confidence * regime_multiplier, 1.0)
        
        # Determine action based on sentiment
        if compound_score > 0.1 and adjusted_confidence > 0.6:
            action = "BUY"
        elif compound_score < -0.1 and adjusted_confidence > 0.6:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Cache result in Redis if available
        if redis_client:
            redis_client.setex(cache_key, 300, json.dumps(sentiment_data))  # Cache for 5 minutes
        
        return {
            "symbol": request.symbol,
            "action": action,
            "confidence": adjusted_confidence,
            "sentiment": overall_sentiment,
            "sources": sentiment_data.get("analysis_metadata", {}).get("sources_analyzed", []),
            "regime_multiplier": regime_multiplier,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"Error evaluating buy for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/confirm")
async def confirm_trade(request: ConfirmTradeRequest):
    """Confirm or reject a trade (for approval mode)"""
    try:
        logger.info(f"Trade confirmation for {request.symbol}: {request.decision} (session: {request.session_id})")
        
        # In a real implementation, this would interact with the trading system
        # For now, we'll just log the decision
        result = {
            "symbol": request.symbol,
            "decision": request.decision,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "confirmed" if request.decision == "approve" else "rejected"
        }
        
        # Broadcast to WebSocket connections
        confirmation_data = {
            "type": "trade_confirmation",
            "data": result
        }
        
        for connection in active_connections[:]:
            try:
                await connection.send_text(json.dumps(confirmation_data))
            except:
                active_connections.remove(connection)
        
        return result
        
    except Exception as e:
        logger.error(f"Error confirming trade for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/analyze")
async def analyze_symbol(request: EvaluateBuyRequest):
    """Analyze a symbol using the sentiment tool"""
    return await evaluate_buy(request)

@app.post("/tools/predict")
async def predict_symbol(request: EvaluateBuyRequest):
    """Predict symbol performance (placeholder for ML integration)"""
    try:
        # Get stock data and analysis
        analysis = stock_analyzer.analyze_stock(request.symbol)
        
        if not analysis.get("success"):
            raise HTTPException(status_code=500, detail="Failed to analyze stock")
        
        # Extract ML analysis
        ml_analysis = analysis.get("ml_analysis", {})
        sentiment_analysis = analysis.get("sentiment_analysis", {})
        
        # Combine sentiment and ML analysis
        prediction = {
            "symbol": request.symbol,
            "predicted_price": ml_analysis.get("predicted_price", 0),
            "confidence": ml_analysis.get("confidence", 0.5),
            "sentiment_score": sentiment_analysis.get("weighted_aggregated", {}).get("positive", 0) - 
                              sentiment_analysis.get("weighted_aggregated", {}).get("negative", 0),
            "recommendation": "BUY" if ml_analysis.get("prediction_direction", 0) > 0.02 else 
                            "SELL" if ml_analysis.get("prediction_direction", 0) < -0.02 else "HOLD"
        }
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/news")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time news updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"New WebSocket connection established. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep the connection alive
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.info(f"WebSocket connection closed: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket connection removed. Total connections: {len(active_connections)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")