"""
Phase 1: Enhanced WebSocket Manager
Replaces 5-second polling with real-time data streaming and intelligent updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import weakref
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of real-time updates"""
    PORTFOLIO = "portfolio"
    TRADES = "trades"
    MARKET_DATA = "market_data"
    SIGNALS = "signals"
    HEARTBEAT = "heartbeat"
    INITIAL_DATA = "initial_data"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message"""
    type: UpdateType
    data: Any
    timestamp: str
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "client_id": self.client_id
        }


class EnhancedConnectionManager:
    """
    Enhanced WebSocket connection manager with intelligent broadcasting
    and real-time data streaming capabilities
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_subscriptions: Dict[str, Set[UpdateType]] = {}
        self.last_data_sent: Dict[str, Dict[UpdateType, float]] = {}
        self.data_cache: Dict[UpdateType, Any] = {}
        self.cache_timestamps: Dict[UpdateType, float] = {}
        
        # Performance optimization settings
        self.min_update_interval = {
            UpdateType.PORTFOLIO: 1.0,      # Portfolio updates every 1 second max
            UpdateType.TRADES: 0.5,         # Trade updates every 0.5 seconds max
            UpdateType.MARKET_DATA: 2.0,    # Market data every 2 seconds max
            UpdateType.SIGNALS: 1.0,        # Signal updates every 1 second max
            UpdateType.HEARTBEAT: 30.0      # Heartbeat every 30 seconds
        }
        
        self.cache_duration = {
            UpdateType.PORTFOLIO: 0.5,      # Cache portfolio for 0.5 seconds
            UpdateType.MARKET_DATA: 1.0,    # Cache market data for 1 second
            UpdateType.SIGNALS: 0.3         # Cache signals for 0.3 seconds
        }
        
        # Start background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        self._start_background_tasks()
        
        logger.info("✅ Enhanced WebSocket manager initialized")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Start heartbeat task
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start cleanup task
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                
        except Exception as e:
            logger.warning(f"Could not start background tasks: {e}")
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Connect a new WebSocket client"""
        await websocket.accept()
        
        # Generate client ID if not provided
        if client_id is None:
            client_id = f"client_{datetime.now().timestamp():.0f}_{len(self.active_connections)}"
        
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = {
            UpdateType.PORTFOLIO, 
            UpdateType.TRADES, 
            UpdateType.MARKET_DATA,
            UpdateType.HEARTBEAT
        }
        self.last_data_sent[client_id] = {}
        
        logger.info(f"✅ WebSocket client {client_id} connected. Total: {len(self.active_connections)}")
        
        # Send initial heartbeat
        await self._send_heartbeat(client_id)
        
        return client_id
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
        if client_id in self.last_data_sent:
            del self.last_data_sent[client_id]
            
        logger.info(f"❌ WebSocket client {client_id} disconnected. Total: {len(self.active_connections)}")
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client with throttling"""
        if client_id not in self.active_connections:
            return
        
        websocket = self.active_connections[client_id]
        
        try:
            # Check if client is subscribed to this update type
            if message.type not in self.client_subscriptions.get(client_id, set()):
                return
            
            # Throttling check
            now = datetime.now().timestamp()
            last_sent = self.last_data_sent[client_id].get(message.type, 0)
            min_interval = self.min_update_interval.get(message.type, 1.0)
            
            if now - last_sent < min_interval:
                return  # Skip this update due to throttling
            
            # Send message
            message.client_id = client_id
            await websocket.send_text(json.dumps(message.to_dict()))
            self.last_data_sent[client_id][message.type] = now
            
        except WebSocketDisconnect:
            self.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            self.disconnect(client_id)
    
    async def broadcast(self, message: WebSocketMessage, exclude_clients: Optional[List[str]] = None):
        """Broadcast message to all connected clients with intelligent caching"""
        if not self.active_connections:
            return
        
        exclude_clients = exclude_clients or []
        
        # Check cache for this update type
        now = datetime.now().timestamp()
        cache_key = message.type
        cache_duration = self.cache_duration.get(message.type, 0)
        
        # Use cached data if available and recent
        if (cache_key in self.cache_timestamps and 
            now - self.cache_timestamps[cache_key] < cache_duration):
            return  # Skip broadcast, data is still fresh
        
        # Update cache
        self.data_cache[cache_key] = message.data
        self.cache_timestamps[cache_key] = now
        
        # Send to all connected clients
        disconnected_clients = []
        
        for client_id in list(self.active_connections.keys()):
            if client_id in exclude_clients:
                continue
                
            try:
                await self.send_to_client(client_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Broadcast portfolio update with intelligent diffing"""
        message = WebSocketMessage(
            type=UpdateType.PORTFOLIO,
            data=portfolio_data,
            timestamp=datetime.now().isoformat()
        )
        await self.broadcast(message)
    
    async def broadcast_trade_update(self, trade_data: Dict[str, Any]):
        """Broadcast trade execution update"""
        message = WebSocketMessage(
            type=UpdateType.TRADES,
            data=trade_data,
            timestamp=datetime.now().isoformat()
        )
        await self.broadcast(message)
    
    async def broadcast_market_data(self, market_data: List[Dict[str, Any]]):
        """Broadcast market data update"""
        message = WebSocketMessage(
            type=UpdateType.MARKET_DATA,
            data=market_data,
            timestamp=datetime.now().isoformat()
        )
        await self.broadcast(message)
    
    async def broadcast_signal_update(self, signal_data: Dict[str, Any]):
        """Broadcast trading signal update"""
        message = WebSocketMessage(
            type=UpdateType.SIGNALS,
            data=signal_data,
            timestamp=datetime.now().isoformat()
        )
        await self.broadcast(message)
    
    async def send_initial_data(self, client_id: str, bot_data: Dict[str, Any]):
        """Send initial data to newly connected client"""
        message = WebSocketMessage(
            type=UpdateType.INITIAL_DATA,
            data=bot_data,
            timestamp=datetime.now().isoformat()
        )
        await self.send_to_client(client_id, message)
    
    async def _send_heartbeat(self, client_id: str):
        """Send heartbeat to specific client"""
        message = WebSocketMessage(
            type=UpdateType.HEARTBEAT,
            data={"status": "alive", "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now().isoformat()
        )
        await self.send_to_client(client_id, message)
    
    async def _heartbeat_loop(self):
        """Background task to send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                for client_id in list(self.active_connections.keys()):
                    await self._send_heartbeat(client_id)
                    
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections and cache"""
        while True:
            try:
                await asyncio.sleep(60)  # Run cleanup every minute
                
                # Clean up old cache entries
                now = datetime.now().timestamp()
                expired_keys = []
                
                for cache_key, timestamp in self.cache_timestamps.items():
                    if now - timestamp > 300:  # Remove cache older than 5 minutes
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    if key in self.data_cache:
                        del self.data_cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                
                logger.debug(f"Cleanup: {len(expired_keys)} cache entries removed")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    def subscribe_client(self, client_id: str, update_types: List[UpdateType]):
        """Subscribe client to specific update types"""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id] = set(update_types)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "cache_entries": len(self.data_cache),
            "subscriptions": {
                client_id: [t.value for t in subs] 
                for client_id, subs in self.client_subscriptions.items()
            }
        }
    
    def stop_background_tasks(self):
        """Stop all background tasks"""
        try:
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
        except Exception as e:
            logger.warning(f"Error stopping background tasks: {e}")


# Global instance
_enhanced_manager = None

def get_enhanced_websocket_manager() -> EnhancedConnectionManager:
    """Get the global enhanced WebSocket manager instance"""
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedConnectionManager()
    return _enhanced_manager