#!/usr/bin/env python3
"""
Dual Portfolio Manager for Paper and Live Trading Modes
Manages separate data storage and seamless mode switching
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy

logger = logging.getLogger(__name__)

class DualPortfolioManager:
    """Manages separate portfolios for paper and live trading modes"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.current_mode = "paper"  # Default mode
        
        # File paths for different modes
        self.files = {
            "paper": {
                "portfolio": os.path.join(data_dir, "paper_portfolio.json"),
                "trades": os.path.join(data_dir, "paper_trades.json"),
                "config": os.path.join(data_dir, "paper_config.json")
            },
            "live": {
                "portfolio": os.path.join(data_dir, "live_portfolio.json"),
                "trades": os.path.join(data_dir, "live_trades.json"),
                "config": os.path.join(data_dir, "live_config.json")
            }
        }
        
        # Current portfolio data
        self.portfolio_data = {}
        self.trade_history = []
        self.config_data = {}
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize default data
        self._initialize_default_data()
        
        logger.info(f"Dual Portfolio Manager initialized with data directory: {data_dir}")
    
    def _initialize_default_data(self):
        """Initialize default portfolio data for both modes"""
        default_portfolio = {
            "cash": 10000.0,
            "total_value": 10000.0,
            "starting_balance": 10000.0,
            "holdings": {},
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        default_config = {
            "mode": "paper",
            "riskLevel": "MEDIUM",
            "stop_loss_pct": 0.05,
            "max_capital_per_trade": 0.25,
            "max_trade_limit": 10,
            "created_at": datetime.now().isoformat()
        }
        
        # Initialize files if they don't exist
        for mode in ["paper", "live"]:
            # Portfolio file
            if not os.path.exists(self.files[mode]["portfolio"]):
                portfolio_copy = deepcopy(default_portfolio)
                if mode == "live":
                    portfolio_copy["cash"] = 0.0  # Live mode starts with actual account balance
                    portfolio_copy["total_value"] = 0.0
                    portfolio_copy["starting_balance"] = 0.0
                
                self._save_json(self.files[mode]["portfolio"], portfolio_copy)
            
            # Trades file
            if not os.path.exists(self.files[mode]["trades"]):
                self._save_json(self.files[mode]["trades"], [])
            
            # Config file
            if not os.path.exists(self.files[mode]["config"]):
                config_copy = deepcopy(default_config)
                config_copy["mode"] = mode
                self._save_json(self.files[mode]["config"], config_copy)
    
    def switch_mode(self, new_mode: str) -> bool:
        """Switch between paper and live trading modes"""
        try:
            if new_mode not in ["paper", "live"]:
                logger.error(f"Invalid mode: {new_mode}")
                return False
            
            if new_mode == self.current_mode:
                logger.info(f"Already in {new_mode} mode")
                return True
            
            # Save current data before switching
            self.save_current_data()
            
            # Switch mode
            old_mode = self.current_mode
            self.current_mode = new_mode
            
            # Load data for new mode
            self.load_current_data()
            
            logger.info(f"Switched from {old_mode} mode to {new_mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch mode to {new_mode}: {e}")
            return False
    
    def load_current_data(self) -> bool:
        """Load data for current mode"""
        try:
            # Load portfolio
            self.portfolio_data = self._load_json(
                self.files[self.current_mode]["portfolio"], 
                self._get_default_portfolio()
            )
            
            # Load trades
            self.trade_history = self._load_json(
                self.files[self.current_mode]["trades"], 
                []
            )
            
            # Load config
            self.config_data = self._load_json(
                self.files[self.current_mode]["config"], 
                self._get_default_config()
            )
            
            logger.info(f"Loaded {self.current_mode} mode data - "
                       f"Cash: Rs.{self.portfolio_data.get('cash', 0):.2f}, "
                       f"Holdings: {len(self.portfolio_data.get('holdings', {}))}, "
                       f"Trades: {len(self.trade_history)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.current_mode} mode data: {e}")
            return False
    
    def save_current_data(self) -> bool:
        """Save current data to appropriate files"""
        try:
            # Update timestamp
            self.portfolio_data["last_updated"] = datetime.now().isoformat()
            
            # Save portfolio
            self._save_json(self.files[self.current_mode]["portfolio"], self.portfolio_data)
            
            # Save trades
            self._save_json(self.files[self.current_mode]["trades"], self.trade_history)
            
            # Save config
            self._save_json(self.files[self.current_mode]["config"], self.config_data)
            
            logger.debug(f"Saved {self.current_mode} mode data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {self.current_mode} mode data: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary for current mode"""
        try:
            holdings = self.portfolio_data.get("holdings", {})
            cash = self.portfolio_data.get("cash", 0)
            
            # Calculate holdings value
            holdings_value = 0
            for symbol, holding in holdings.items():
                holdings_value += holding.get("total_value", 0)
            
            total_value = cash + holdings_value
            starting_balance = self.portfolio_data.get("starting_balance", 10000)
            
            # Calculate returns
            total_return = total_value - starting_balance
            return_percentage = (total_return / starting_balance * 100) if starting_balance > 0 else 0
            
            return {
                "mode": self.current_mode,
                "cash": cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
                "starting_balance": starting_balance,
                "total_return": total_return,
                "return_percentage": return_percentage,
                "unrealized_pnl": self.portfolio_data.get("unrealized_pnl", 0),
                "realized_pnl": self.portfolio_data.get("realized_pnl", 0),
                "total_trades": len(self.trade_history),
                "active_positions": len(holdings),
                "last_updated": self.portfolio_data.get("last_updated")
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}
    
    def add_trade(self, trade_data: Dict) -> bool:
        """Add a trade to current mode's history"""
        try:
            trade_entry = {
                "id": len(self.trade_history) + 1,
                "timestamp": datetime.now().isoformat(),
                "mode": self.current_mode,
                **trade_data
            }
            
            self.trade_history.append(trade_entry)
            
            # Update portfolio stats
            self.portfolio_data["total_trades"] = len(self.trade_history)
            
            # Update win/loss stats if trade has P&L
            if "pnl" in trade_data:
                pnl = trade_data["pnl"]
                if pnl > 0:
                    self.portfolio_data["winning_trades"] = self.portfolio_data.get("winning_trades", 0) + 1
                elif pnl < 0:
                    self.portfolio_data["losing_trades"] = self.portfolio_data.get("losing_trades", 0) + 1
                
                # Update realized P&L
                self.portfolio_data["realized_pnl"] = self.portfolio_data.get("realized_pnl", 0) + pnl
            
            logger.info(f"Added trade to {self.current_mode} mode: {trade_data.get('action', 'UNKNOWN')} {trade_data.get('symbol', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add trade: {e}")
            return False
    
    def update_portfolio(self, updates: Dict) -> bool:
        """Update portfolio data for current mode"""
        try:
            self.portfolio_data.update(updates)
            self.portfolio_data["last_updated"] = datetime.now().isoformat()
            
            logger.debug(f"Updated {self.current_mode} portfolio data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")
            return False
    
    def get_trade_history(self, limit: int = None) -> List[Dict]:
        """Get trade history for current mode"""
        try:
            if limit:
                return self.trade_history[-limit:]
            return self.trade_history
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def get_performance_comparison(self) -> Dict:
        """Compare performance between paper and live modes"""
        try:
            comparison = {}
            
            for mode in ["paper", "live"]:
                # Load data for each mode
                portfolio_file = self.files[mode]["portfolio"]
                trades_file = self.files[mode]["trades"]
                
                portfolio_data = self._load_json(portfolio_file, {})
                trades_data = self._load_json(trades_file, [])
                
                if portfolio_data:
                    starting_balance = portfolio_data.get("starting_balance", 10000)
                    total_value = portfolio_data.get("total_value", starting_balance)
                    total_return = total_value - starting_balance
                    return_pct = (total_return / starting_balance * 100) if starting_balance > 0 else 0
                    
                    comparison[mode] = {
                        "starting_balance": starting_balance,
                        "current_value": total_value,
                        "total_return": total_return,
                        "return_percentage": return_pct,
                        "total_trades": len(trades_data),
                        "winning_trades": portfolio_data.get("winning_trades", 0),
                        "losing_trades": portfolio_data.get("losing_trades", 0),
                        "realized_pnl": portfolio_data.get("realized_pnl", 0),
                        "last_updated": portfolio_data.get("last_updated")
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to get performance comparison: {e}")
            return {}
    
    def copy_paper_to_live(self) -> bool:
        """Copy paper trading portfolio to live mode (for testing)"""
        try:
            # Load paper data
            paper_portfolio = self._load_json(self.files["paper"]["portfolio"], {})
            paper_trades = self._load_json(self.files["paper"]["trades"], [])
            
            if not paper_portfolio:
                logger.error("No paper portfolio data to copy")
                return False
            
            # Create live copy
            live_portfolio = deepcopy(paper_portfolio)
            live_trades = deepcopy(paper_trades)
            
            # Update mode references
            for trade in live_trades:
                trade["mode"] = "live"
                trade["copied_from_paper"] = True
            
            # Save to live files
            self._save_json(self.files["live"]["portfolio"], live_portfolio)
            self._save_json(self.files["live"]["trades"], live_trades)
            
            logger.info("Copied paper portfolio to live mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy paper to live: {e}")
            return False
    
    def reset_portfolio(self, mode: str = None) -> bool:
        """Reset portfolio for specified mode (or current mode)"""
        try:
            target_mode = mode or self.current_mode
            
            if target_mode not in ["paper", "live"]:
                logger.error(f"Invalid mode for reset: {target_mode}")
                return False
            
            # Reset to default values
            default_portfolio = self._get_default_portfolio()
            default_trades = []
            
            # Save reset data
            self._save_json(self.files[target_mode]["portfolio"], default_portfolio)
            self._save_json(self.files[target_mode]["trades"], default_trades)
            
            # If resetting current mode, reload data
            if target_mode == self.current_mode:
                self.load_current_data()
            
            logger.info(f"Reset {target_mode} portfolio to default values")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset portfolio: {e}")
            return False
    
    def _load_json(self, filepath: str, default: Any = None) -> Any:
        """Load JSON data from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                return default
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return default
    
    def _save_json(self, filepath: str, data: Any) -> bool:
        """Save data to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    def _get_default_portfolio(self) -> Dict:
        """Get default portfolio structure"""
        return {
            "cash": 10000.0,
            "total_value": 10000.0,
            "starting_balance": 10000.0,
            "holdings": {},
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_default_config(self) -> Dict:
        """Get default config structure"""
        return {
            "mode": self.current_mode,
            "riskLevel": "MEDIUM",
            "stop_loss_pct": 0.05,
            "max_capital_per_trade": 0.25,
            "max_trade_limit": 10,
            "created_at": datetime.now().isoformat()
        }
