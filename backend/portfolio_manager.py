#!/usr/bin/env python3
"""
Dual Portfolio Manager for Paper and Live Trading Modes
Manages portfolios using SQLite database with LangGraph checkpoint system
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy
import json

# Fix import paths permanently
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from db.database import DatabaseManager, Portfolio, Holding, Trade

logger = logging.getLogger(__name__)

class DualPortfolioManager:
    """Manages separate portfolios for paper and live trading using SQLite database"""
    
    def __init__(self, data_dir: str = "data"):
        # Normalize to project-root data directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(project_root, data_dir)
        self.current_mode = "paper"  # Default mode
        self.trade_callbacks = []  # List of callbacks to notify on trade execution
        
        # Initialize database
        db_path = f'sqlite:///{os.path.join(self.data_dir, "trading.db")}'
        self.db = DatabaseManager(db_path)
        
        # Config files (keeping configs in JSON for flexibility)
        self.config_files = {
            "paper": os.path.join(self.data_dir, "paper_config.json"),
            "live": os.path.join(self.data_dir, "live_config.json")
        }
        
        # Current session data
        self.current_portfolio = None
        self.current_holdings_dict = {}  # Separate dict for backward compatibility
        self.config_data = {}
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database and load initial mode
        logger.info("Starting portfolio manager initialization")
        try:
            self._initialize_database()
            logger.info(f"Dual Portfolio Manager initialized with SQLite database in {data_dir}")
        except Exception as e:
            logger.error(f"Error initializing portfolio manager: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_database(self):
        """Initialize database with default data if needed"""
        session = self.db.Session()
        try:
            # Check if we need to migrate from JSON first
            need_migration = False
            for mode in ["paper", "live"]:
                portfolio = session.query(Portfolio).filter_by(mode=mode).first()
                if not portfolio:
                    # Check if we have JSON data to migrate
                    json_file = os.path.join(self.data_dir, f'portfolio_india_{mode}.json')
                    if os.path.exists(json_file):
                        need_migration = True
                        logger.info(f"Found existing JSON data for {mode} mode")
                    else:
                        # For live mode, we'll sync with Dhan later
                        # For paper mode, use default values
                        initial_balance = 50000.0 if mode == "paper" else 0.0
                        portfolio = Portfolio(
                            mode=mode,
                            cash=initial_balance,
                            starting_balance=initial_balance,
                            realized_pnl=0.0,
                            unrealized_pnl=0.0,
                            last_updated=datetime.now()
                        )
                        session.add(portfolio)
                        logger.info(f"Created new {mode} portfolio")

                # Initialize config files if they don't exist
                config_file = self.config_files[mode]
                if not os.path.exists(config_file):
                    default_config = {
                        "risk_per_trade": 0.02,
                        "max_position_size": 0.1,
                        "stop_loss_atr_multiplier": 2.0,
                        "take_profit_atr_multiplier": 4.0
                    }
                    self._save_json(config_file, default_config)
                    
            # If we found JSON data but no database entries, migrate the data
            if need_migration:
                logger.info("Migrating existing JSON data to database")
                try:
                    self.db.migrate_json_to_sqlite(self.data_dir)
                except Exception as e:
                    logger.error(f"Error during JSON migration: {e}")
                    logger.error(f"Error type: {type(e)}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise
                    
            session.commit()
            logger.info("Loading mode after database initialization")
            self.load_mode(self.current_mode)
            logger.info("Mode loaded successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing database: {e}")
            logger.error(f"Error type: {type(e)}")
            # Log the full traceback for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            session.close()
            
    def _save_json(self, filepath: str, data: Dict) -> None:
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_mode(self, mode: str) -> None:
        """Load portfolio and configuration for specified mode"""
        if mode not in ["paper", "live"]:
            raise ValueError(f"Invalid mode: {mode}")
            
        self.current_mode = mode
        session = self.db.Session()
        try:
            # Load portfolio from database
            portfolio = session.query(Portfolio).filter_by(mode=mode).first()
            if not portfolio:
                raise ValueError(f"Portfolio not found for mode: {mode}")
                
            self.current_portfolio = portfolio
            self.current_holdings_dict = {}  # Initialize holdings dict
            
            # Load holdings from database into holdings dict for backward compatibility
            logger.info(f"Loading holdings for {mode} mode")
            logger.info(f"Number of holdings in database: {len(portfolio.holdings)}")
            for holding in portfolio.holdings:
                logger.info(f"Processing holding: {holding.ticker}, quantity: {holding.quantity}")
                if holding.quantity > 0:  # Only include active holdings
                    self.current_holdings_dict[holding.ticker] = {
                        "qty": holding.quantity,
                        "avg_price": holding.avg_price,
                        "last_price": holding.last_price
                    }
            
            # Load config data
            config_file = self.config_files[mode]
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config_data = json.load(f)
            else:
                logger.warning(f"Config file not found for {mode} mode")
                self.config_data = {}
                
        except Exception as e:
            logger.error(f"Error loading {mode} mode: {e}")
            logger.error(f"Error type: {type(e)}")
            # Log the full traceback for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        finally:
            session.close()
            
    def get_current_portfolio(self) -> Portfolio:
        """Get current portfolio data"""
        # Ensure holdings dict is synchronized with database
        if self.current_portfolio:
            # Rebuild holdings dict from database to ensure consistency
            session = self.db.Session()
            try:
                # Load holdings from database into holdings dict for backward compatibility
                self.current_holdings_dict = {}
                for holding in self.current_portfolio.holdings:
                    if holding.quantity > 0:  # Only include active holdings
                        self.current_holdings_dict[holding.ticker] = {
                            "qty": holding.quantity,
                            "avg_price": holding.avg_price,
                            "last_price": holding.last_price
                        }
            except Exception as e:
                logger.error(f"Error synchronizing portfolio holdings: {e}")
            finally:
                session.close()
                
        return self.current_portfolio

    def switch_mode(self, mode: str) -> None:
        """Switch between paper and live trading modes"""
        if mode not in ["paper", "live"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        if mode == self.current_mode:
            logger.info(f"Already in {mode} mode")
            return
        
        logger.info(f"Switching from {self.current_mode} to {mode} mode")
        self.load_mode(mode)
        logger.info(f"Successfully switched to {mode} mode")

    def add_trade_callback(self, callback) -> None:
        """Add a callback function to be called when trades are executed"""
        self.trade_callbacks.append(callback)
        
    def record_trade(self, ticker: str, action: str, quantity: float, price: float, 
                     pnl: float = 0.0, stop_loss: float = None, take_profit: float = None):
        """Record a trade in the database and update portfolio"""
        session = None
        try:
            session = self.db.Session()
            portfolio = session.query(Portfolio).filter_by(mode=self.current_mode).first()
            
            if not portfolio:
                # Create portfolio if it doesn't exist
                portfolio = Portfolio(
                    mode=self.current_mode,
                    cash=self.starting_balance if self.current_mode == "paper" else 100000.0,
                    starting_balance=self.starting_balance,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    last_updated=datetime.now()
                )
                session.add(portfolio)
                session.flush()
            
            # Create trade record
            trade = Trade(
                portfolio_id=portfolio.id,
                ticker=ticker,
                action=action,
                quantity=quantity,
                price=price,
                pnl=pnl,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=datetime.now()
            )
            session.add(trade)
            
            # Update portfolio and holdings
            trade_value = quantity * price
            
            if action == 'buy':
                # Check if enough cash available
                if trade_value > portfolio.cash:
                    raise ValueError(f"Insufficient cash for trade: {trade_value:.2f} needed, {portfolio.cash:.2f} available")
                    
                portfolio.cash -= trade_value
                # Update or create holding
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id, ticker=ticker
                ).first()
                
                if holding:
                    # Update existing holding
                    total_cost = (holding.quantity * holding.avg_price) + trade_value
                    total_quantity = holding.quantity + quantity
                    holding.avg_price = total_cost / total_quantity
                    holding.quantity = total_quantity
                else:
                    # Create new holding
                    holding = Holding(
                        portfolio_id=portfolio.id,
                        ticker=ticker,
                        quantity=quantity,
                        avg_price=price,
                        last_price=price
                    )
                    session.add(holding)
                
                # Update in-memory holdings
                if ticker in self.current_holdings_dict:
                    self.current_holdings_dict[ticker]["qty"] += quantity
                    # Recalculate average price
                    total_qty = self.current_holdings_dict[ticker]["qty"]
                    total_cost = (self.current_holdings_dict[ticker]["qty"] - quantity) * self.current_holdings_dict[ticker]["avg_price"] + trade_value
                    self.current_holdings_dict[ticker]["avg_price"] = total_cost / total_qty
                else:
                    self.current_holdings_dict[ticker] = {
                        "qty": quantity,
                        "avg_price": price,
                        "last_price": price
                    }
            else:  # sell
                # Check if we have enough holdings to sell
                holding = session.query(Holding).filter_by(
                    portfolio_id=portfolio.id, ticker=ticker
                ).first()
                if not holding or holding.quantity < quantity:
                    raise ValueError(f"Insufficient quantity for {ticker}: {quantity} requested, {holding.quantity if holding else 0} available")
                
                portfolio.cash += trade_value
                
                # Calculate realized P&L if not provided
                if pnl == 0.0:
                    pnl = (price - holding.avg_price) * quantity
                
                # Update realized P&L
                portfolio.realized_pnl += pnl
                
                # Update holdings
                holding.quantity -= quantity
                if holding.quantity == 0:
                    session.delete(holding)
                
                # Update in-memory holdings
                if ticker in self.current_holdings_dict:
                    self.current_holdings_dict[ticker]["qty"] -= quantity
                    if self.current_holdings_dict[ticker]["qty"] <= 0:
                        del self.current_holdings_dict[ticker]
            
            portfolio.last_updated = datetime.now()
            session.commit()
            
            # Update current portfolio reference to prevent detached instance issues
            self.current_portfolio = portfolio
            
            # Sync changes to JSON files
            try:
                self._sync_to_json(session, portfolio)
            except Exception as sync_error:
                logger.error(f"Error syncing to JSON files: {sync_error}")
            
            # NEW: Update data service watchlist after trade execution
            self._update_data_service_watchlist_after_trade(session, portfolio)
            
            # Notify callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade)
                except Exception as callback_error:
                    logger.error(f"Error in trade callback: {callback_error}")
                    
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Error recording trade: {e}")
            raise
        finally:
            if session:
                session.close()
                    
    def _update_data_service_watchlist_after_trade(self, session, portfolio):
        """Update data service watchlist after a trade is executed"""
        try:
            # Get current holdings from database
            holdings = session.query(Holding).filter_by(portfolio_id=portfolio.id).all()
            holding_symbols = [holding.ticker for holding in holdings]
            
            # Import data service client
            from data_service_client import get_data_client
            data_client = get_data_client()
            
            # Update watchlist with current holdings
            data_client.update_watchlist(holding_symbols)
            logger.info(f"Updated data service watchlist after trade with {len(holding_symbols)} symbols")
                
        except Exception as e:
            logger.error(f"Failed to update data service watchlist after trade: {e}")
            
    def _sync_to_json(self, session, portfolio):
        """Sync database changes back to JSON files for backward compatibility"""
        try:
            # Build portfolio data
            portfolio_data = {
                "cash": portfolio.cash,
                "starting_balance": portfolio.starting_balance,
                "realized_pnl": portfolio.realized_pnl,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "holdings": {}
            }
            
            # Add holdings
            for holding in portfolio.holdings:
                portfolio_data["holdings"][holding.ticker] = {
                    "qty": holding.quantity,
                    "avg_price": holding.avg_price,
                    "last_price": holding.last_price
                }
            
            # Save to JSON
            json_file = os.path.join(self.data_dir, f'portfolio_india_{portfolio.mode}.json')
            with open(json_file, 'w') as f:
                json.dump(portfolio_data, f, indent=4)
                
            # Save trades
            trades_data = []
            for trade in portfolio.trades:
                trades_data.append({
                    "timestamp": trade.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "asset": trade.ticker,
                    "action": trade.action,
                    "qty": trade.quantity,
                    "price": trade.price,
                    "pnl": trade.pnl
                })
            
            trades_file = os.path.join(self.data_dir, f'trade_log_india_{portfolio.mode}.json')
            with open(trades_file, 'w') as f:
                json.dump(trades_data, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error syncing to JSON: {e}")
            raise
    
    def refresh_holdings_from_database(self):
        """Refresh holdings dict from database to ensure consistency"""
        if not self.current_portfolio:
            return
            
        session = self.db.Session()
        try:
            # Load holdings from database into holdings dict for backward compatibility
            self.current_holdings_dict = {}
            for holding in self.current_portfolio.holdings:
                if holding.quantity > 0:  # Only include active holdings
                    self.current_holdings_dict[holding.ticker] = {
                        "qty": holding.quantity,
                        "avg_price": holding.avg_price,
                        "last_price": holding.last_price
                    }
        except Exception as e:
            logger.error(f"Error refreshing portfolio holdings: {e}")
        finally:
            session.close()
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary for current mode"""
        try:
            session = self.db.Session()
            holdings = session.query(Holding).filter_by(portfolio_id=self.current_portfolio.id).all()
            trades = session.query(Trade).filter_by(portfolio_id=self.current_portfolio.id).all()
            
            # Calculate holdings value using last_price from database
            holdings_value = sum(holding.quantity * holding.last_price for holding in holdings)
            
            # Calculate unrealized P&L
            cost_basis = sum(holding.quantity * holding.avg_price for holding in holdings)
            unrealized_pnl = holdings_value - cost_basis
            
            # Calculate portfolio metrics
            total_value = self.current_portfolio.cash + holdings_value
            total_return = total_value - self.current_portfolio.starting_balance
            return_percentage = (total_return / self.current_portfolio.starting_balance * 100) if self.current_portfolio.starting_balance > 0 else 0
            
            return {
                "mode": self.current_portfolio.mode,
                "cash": self.current_portfolio.cash,
                "holdings_value": holdings_value,
                "total_value": total_value,
                "starting_balance": self.current_portfolio.starting_balance,
                "total_return": total_return,
                "return_percentage": return_percentage,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": self.current_portfolio.realized_pnl,
                "total_trades": len(trades),
                "active_positions": len(holdings),
                "last_updated": self.current_portfolio.last_updated
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            raise
        finally:
            session.close()
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Trade]:
        """Get trade history for current mode"""
        session = self.db.Session()
        try:
            query = session.query(Trade).filter_by(portfolio_id=self.current_portfolio.id)
            if limit:
                query = query.limit(limit)
            return query.all()
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            raise
        finally:
            session.close()
    
    def reset_portfolio(self, mode: Optional[str] = None) -> bool:
        """Reset portfolio to initial state"""
        try:
            target_mode = mode or self.current_mode
            
            if target_mode not in ["paper", "live"]:
                logger.error(f"Invalid mode for reset: {target_mode}")
                return False
            
            session = self.db.Session()
            try:
                # Delete all holdings and trades for the target portfolio
                portfolio = session.query(Portfolio).filter_by(mode=target_mode).first()
                if not portfolio:
                    raise ValueError(f"Portfolio not found for mode: {target_mode}")
                
                # Delete holdings and trades
                session.query(Holding).filter_by(portfolio_id=portfolio.id).delete()
                session.query(Trade).filter_by(portfolio_id=portfolio.id).delete()
                
                # Reset portfolio values
                portfolio.cash = 50000.0
                portfolio.starting_balance = 50000.0
                portfolio.realized_pnl = 0.0
                portfolio.unrealized_pnl = 0.0
                portfolio.last_updated = datetime.now()
                
                session.commit()
                logger.info(f"Reset {target_mode} portfolio to default values")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to reset portfolio: {e}")
                raise
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Failed to reset portfolio: {e}")
            return False
