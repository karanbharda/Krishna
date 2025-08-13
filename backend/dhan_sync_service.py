#!/usr/bin/env python3
"""
Dhan Account Sync Service - Real-time background sync with Dhan account
"""

import os
import json
import sqlite3
import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DhanSyncService:
    """Background service to sync portfolio with Dhan account in real-time"""
    
    def __init__(self, sync_interval: int = 30):
        """
        Initialize Dhan sync service
        
        Args:
            sync_interval: Sync interval in seconds (default: 30 seconds)
        """
        self.sync_interval = sync_interval
        self.client_id = os.getenv("DHAN_CLIENT_ID")
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN")
        self.is_running = False
        self.last_sync_time = None
        self.last_known_balance = 0.0
        
        # Validate credentials
        if not self.client_id or not self.access_token:
            logger.error("Dhan credentials not found in environment variables")
            raise ValueError("Dhan credentials not configured")
        
        logger.info(f"Dhan Sync Service initialized with {sync_interval}s interval")
    
    def get_dhan_funds(self) -> Optional[Dict[str, Any]]:
        """Get current funds from Dhan API"""
        try:
            headers = {
                'access-token': self.access_token,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                "https://api.dhan.co/v2/fundlimit",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Dhan API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching funds from Dhan API: {e}")
            return None
    
    def get_dhan_holdings(self) -> Optional[list]:
        """Get current holdings from Dhan API"""
        try:
            headers = {
                'access-token': self.access_token,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                "https://api.dhan.co/v2/holdings",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Dhan holdings API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching holdings from Dhan API: {e}")
            return []
    
    def update_database(self, cash_amount: float, holdings: list) -> bool:
        """Update the database with new portfolio data"""
        try:
            # Get the correct path to the database
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            db_path = os.path.join(project_root, 'data', 'trading.db')

            if not os.path.exists(db_path):
                logger.error(f"Database not found at {db_path}")
                return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Update live portfolio
            cursor.execute("""
                UPDATE portfolios 
                SET cash = ?, last_updated = ?
                WHERE mode = 'live'
            """, (cash_amount, datetime.now().isoformat()))
            
            if cursor.rowcount == 0:
                # Insert new live portfolio if it doesn't exist
                cursor.execute("""
                    INSERT INTO portfolios (mode, cash, starting_balance, realized_pnl, unrealized_pnl, last_updated)
                    VALUES ('live', ?, ?, 0.0, 0.0, ?)
                """, (cash_amount, cash_amount, datetime.now().isoformat()))
                logger.info("Created new live portfolio in database")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            return False
    
    def update_json_files(self, cash_amount: float, holdings: list) -> bool:
        """Update JSON files with new portfolio data"""
        try:
            # Prepare holdings data
            holdings_dict = {}
            for holding in holdings:
                symbol = holding.get("tradingSymbol", "")
                quantity = int(holding.get("totalQty", 0))
                avg_price = float(holding.get("avgCostPrice", 0))
                
                if quantity > 0:
                    holdings_dict[symbol] = {
                        "qty": quantity,
                        "avg_price": avg_price,
                        "last_price": avg_price
                    }
            
            # Calculate total holdings value
            total_holdings_value = sum(
                h["qty"] * h["avg_price"] for h in holdings_dict.values()
            )
            
            # Get the correct path to data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            data_dir = os.path.join(project_root, 'data')

            # Update portfolio_india_live.json
            portfolio_data = {
                "cash": cash_amount,
                "holdings": holdings_dict,
                "starting_balance": max(cash_amount, self.last_known_balance),
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "last_updated": datetime.now().isoformat()
            }

            portfolio_file = os.path.join(data_dir, 'portfolio_india_live.json')
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=4)

            # Update live_portfolio.json
            live_portfolio_data = {
                "cash": cash_amount,
                "total_value": cash_amount + total_holdings_value,
                "starting_balance": max(cash_amount, self.last_known_balance),
                "holdings": holdings_dict,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "last_updated": datetime.now().isoformat()
            }

            live_portfolio_file = os.path.join(data_dir, 'live_portfolio.json')
            with open(live_portfolio_file, 'w') as f:
                json.dump(live_portfolio_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"JSON update failed: {e}")
            return False
    
    def sync_once(self) -> bool:
        """Perform a single sync operation"""
        try:
            # Get data from Dhan
            funds_data = self.get_dhan_funds()
            holdings_data = self.get_dhan_holdings()
            
            if funds_data is None:
                logger.warning("Failed to fetch funds data from Dhan")
                return False
            
            if holdings_data is None:
                holdings_data = []
            
            # Extract cash balance
            current_balance = funds_data.get('availabelBalance', 0.0)
            
            # Check if balance changed
            balance_changed = abs(current_balance - self.last_known_balance) > 0.01
            
            if balance_changed:
                logger.info(f"💰 Balance changed: ₹{self.last_known_balance} → ₹{current_balance}")
                
                # Update database and JSON files
                db_success = self.update_database(current_balance, holdings_data)
                json_success = self.update_json_files(current_balance, holdings_data)
                
                if db_success and json_success:
                    self.last_known_balance = current_balance
                    self.last_sync_time = datetime.now()
                    logger.info(f"✅ Portfolio synced successfully: ₹{current_balance}")
                    return True
                else:
                    logger.error("Failed to update portfolio data")
                    return False
            else:
                logger.debug(f"No balance change detected: ₹{current_balance}")
                self.last_sync_time = datetime.now()
                return True
                
        except Exception as e:
            logger.error(f"Sync operation failed: {e}")
            return False
    
    async def start_background_sync(self):
        """Start the background sync service"""
        self.is_running = True
        logger.info(f"🚀 Starting Dhan sync service (every {self.sync_interval}s)")
        
        # Initial sync
        self.sync_once()
        
        while self.is_running:
            try:
                await asyncio.sleep(self.sync_interval)
                if self.is_running:
                    self.sync_once()
                    
            except Exception as e:
                logger.error(f"Error in background sync loop: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
    
    def stop(self):
        """Stop the background sync service"""
        self.is_running = False
        logger.info("🛑 Dhan sync service stopped")

# Global sync service instance
_sync_service = None

def get_sync_service() -> Optional[DhanSyncService]:
    """Get the global sync service instance"""
    return _sync_service

def start_sync_service(sync_interval: int = 30) -> DhanSyncService:
    """Start the global sync service"""
    global _sync_service
    
    if _sync_service is None:
        try:
            _sync_service = DhanSyncService(sync_interval)
            # Start the background task
            asyncio.create_task(_sync_service.start_background_sync())
            logger.info("Dhan sync service started successfully")
        except Exception as e:
            logger.error(f"Failed to start sync service: {e}")
            _sync_service = None
    
    return _sync_service

def stop_sync_service():
    """Stop the global sync service"""
    global _sync_service
    
    if _sync_service:
        _sync_service.stop()
        _sync_service = None
        logger.info("Dhan sync service stopped")

if __name__ == "__main__":
    # Test the sync service
    async def test_sync():
        service = DhanSyncService(sync_interval=10)  # 10 seconds for testing
        await service.start_background_sync()
    
    asyncio.run(test_sync())
