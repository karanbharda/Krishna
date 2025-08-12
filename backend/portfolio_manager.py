#!/usr/bin/env python3
"""
Dual Portfolio Manager for Paper and Live Trading Modes
Manages portfolios using SQLite database with LangGraph checkpoint system
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from copy import deepcopy
import json
from backend.db.database import DatabaseManager, Portfolio, Holding, Trade

logger = logging.getLogger(__name__)

class DualPortfolioManager:
    """Manages separate portfolios for paper and live trading using SQLite database"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.current_mode = "paper"  # Default mode
        self.trade_callbacks = []  # List of callbacks to notify on trade execution
        
        # Initialize database
        db_path = f'sqlite:///{os.path.join(data_dir, "trading.db")}'
        self.db = DatabaseManager(db_path)
        
        # Config files (keeping configs in JSON for flexibility)
        self.config_files = {
            "paper": os.path.join(data_dir, "paper_config.json"),
            "live": os.path.join(data_dir, "live_config.json")
        }
        
        # Current session data
        self.current_portfolio = None
        self.config_data = {}
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database and load initial mode
        self._initialize_database()
        logger.info(f"Dual Portfolio Manager initialized with SQLite database in {data_dir}")
    
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
                self.db.migrate_json_to_sqlite(self.data_dir)

            session.commit()
            self.load_mode(self.current_mode)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing database: {e}")
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
            raise
        finally:
            session.close()
            
    def get_current_portfolio(self) -> Portfolio:
        """Get current portfolio data"""
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
        
    def record_trade(self, ticker: str, action: str, quantity: int, price: float,
                    pnl: float = 0.0, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> None:
        """Record a trade and update portfolio in the database"""
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("ticker must be a non-empty string")
        if not isinstance(action, str) or action.lower() not in ["buy", "sell"]:
            raise ValueError("Action must be 'buy' or 'sell'")
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            raise ValueError("Quantity must be a positive number")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive number")
            
        action = action.lower()  # Normalize action to lowercase
        session = self.db.Session()
        try:
            # Get current portfolio
            portfolio = session.query(Portfolio).filter_by(mode=self.current_mode).first()
            if not portfolio:
                raise ValueError(f"No portfolio found for mode {self.current_mode}")
                
            # Create new trade record
            trade = Trade(
                portfolio_id=portfolio.id,
                timestamp=datetime.now(),
                ticker=ticker,
                action=action,
                quantity=quantity,
                price=price,
                pnl=pnl,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"source": "auto"}
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
                
                logger.info(f"Realized P&L for {ticker}: {pnl:.2f} (Total: {portfolio.realized_pnl:.2f})")
            
            portfolio.last_updated = datetime.now()
            session.commit()
            
            try:
                self._sync_to_json(session, portfolio)
            except Exception as sync_error:
                logger.error(f"Error syncing to JSON files: {sync_error}")
            
            # Notify callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(trade)
                except Exception as callback_error:
                    logger.error(f"Error in trade callback: {callback_error}")
                    
            # Log P&L calculation if applicable
            if action == 'sell':
                logger.info(f"Realized P&L for {ticker}: {pnl:.2f} (Total: {portfolio.realized_pnl:.2f})")
            
            portfolio.last_updated = datetime.now()
            session.commit()
            
            # Sync changes to JSON files
            try:
                self._sync_to_json(session, portfolio)
            except Exception as sync_error:
                logger.error(f"Error syncing to JSON files: {sync_error}")
                    
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording trade: {e}")
            raise
        finally:
            session.close()
                    
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
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary for current mode"""
        try:
            session = self.db.Session()
            holdings = session.query(Holding).filter_by(portfolio_id=self.current_portfolio.id).all()
            trades = session.query(Trade).filter_by(portfolio_id=self.current_portfolio.id).all()
            
            # Calculate holdings value (need current prices here)
            holdings_value = 0.0  # This should be updated with real-time prices
            
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
                "unrealized_pnl": self.current_portfolio.unrealized_pnl,
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
