"""
Database configuration and initialization for LangGraph SQLite checkpoint system
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path
import json
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)
Base = declarative_base()

class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    mode = Column(String, index=True)  # 'paper' or 'live'
    cash = Column(Float, default=0.0)
    starting_balance = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    holdings = relationship("Holding", back_populates="portfolio")
    trades = relationship("Trade", back_populates="portfolio")

class Holding(Base):
    __tablename__ = 'holdings'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    ticker = Column(String, index=True)
    quantity = Column(Integer)
    avg_price = Column(Float)
    last_price = Column(Float)
    portfolio = relationship("Portfolio", back_populates="holdings")

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    timestamp = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    action = Column(String)  # 'buy' or 'sell'
    quantity = Column(Integer)
    price = Column(Float)
    pnl = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    trade_metadata = Column(JSON)  # For additional trade data
    portfolio = relationship("Portfolio", back_populates="trades")

def _default_db_uri() -> str:
    # Resolve to project root /data/trading.db regardless of working dir
    backend_dir = Path(__file__).resolve().parents[1]
    project_root = backend_dir.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'trading.db').as_posix()}"


def init_db(db_path: str | None = None):
    """Initialize the database and create tables"""
    db_uri = db_path if db_path else _default_db_uri()
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    return engine

def create_session(engine):
    """Create a new database session"""
    Session = sessionmaker(bind=engine)
    return Session()

class DatabaseManager:
    def __init__(self, db_path: str | None = None):
        self.engine = init_db(db_path)
        self.Session = sessionmaker(bind=self.engine)
    
    def migrate_json_to_sqlite(self, data_dir: str):
        """Migrate existing JSON data to SQLite"""
        try:
            session = self.Session()
            
            # Clean up existing data
            session.query(Trade).delete()
            session.query(Holding).delete()
            session.query(Portfolio).delete()
            session.commit()
            
            # Load JSON files
            paper_portfolio = self._load_json(os.path.join(data_dir, 'portfolio_india_paper.json'))
            live_portfolio = self._load_json(os.path.join(data_dir, 'portfolio_india_live.json'))
            paper_trades = self._load_json(os.path.join(data_dir, 'trade_log_india_paper.json'))
            live_trades = self._load_json(os.path.join(data_dir, 'trade_log_india_live.json'))
            
            # Migrate paper portfolio
            if paper_portfolio:
                self._migrate_portfolio(session, paper_portfolio, 'paper', paper_trades)
            
            # Migrate live portfolio
            if live_portfolio:
                self._migrate_portfolio(session, live_portfolio, 'live', live_trades)
            
            session.commit()
            logger.info("Successfully migrated JSON data to SQLite")
            
            # Create backup of JSON files
            self._backup_json_files(data_dir)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error during migration: {e}")
            raise
        finally:
            session.close()
    
    def _load_json(self, filepath: str) -> Optional[Dict]:
        """Load data from JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
        return None
        
    def _backup_json_files(self, data_dir: str):
        """Create backups of JSON files"""
        backup_dir = os.path.join(data_dir, 'backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        
        files_to_backup = [
            'portfolio_india_paper.json',
            'portfolio_india_live.json',
            'trade_log_india_paper.json',
            'trade_log_india_live.json'
        ]
        
        for filename in files_to_backup:
            src = os.path.join(data_dir, filename)
            if os.path.exists(src):
                dest = os.path.join(backup_dir, filename)
                with open(src, 'r') as f_src, open(dest, 'w') as f_dest:
                    f_dest.write(f_src.read())
                    
    def _migrate_portfolio(self, session, portfolio_data: Dict, mode: str, trades_data: Optional[List] = None):
        """Migrate a portfolio and its trades to SQLite"""
        # Clean up any existing data for this mode
        portfolio = session.query(Portfolio).filter_by(mode=mode).first()
        if portfolio:
            session.query(Holding).filter_by(portfolio_id=portfolio.id).delete()
            session.query(Trade).filter_by(portfolio_id=portfolio.id).delete()
            session.delete(portfolio)
            session.commit()
        
        # Create new portfolio
        portfolio = Portfolio(mode=mode)
        portfolio.cash = portfolio_data.get('cash', 0.0)
        portfolio.starting_balance = portfolio_data.get('starting_balance', 0.0)
        portfolio.realized_pnl = portfolio_data.get('realized_pnl', 0.0)
        portfolio.unrealized_pnl = portfolio_data.get('unrealized_pnl', 0.0)
        portfolio.last_updated = datetime.now()
        session.add(portfolio)
        session.flush()  # Get portfolio ID
        
        # Create holdings
        holdings = portfolio_data.get('holdings', {})
        for ticker, data in holdings.items():
            holding = Holding(
                portfolio_id=portfolio.id,
                ticker=ticker,
                quantity=data.get('qty', 0),
                avg_price=data.get('avg_price', 0.0),
                last_price=data.get('last_price', data.get('avg_price', 0.0))
            )
            session.add(holding)
            
        # Create trades
        if trades_data:
            for trade_data in trades_data:
                trade = Trade(
                    portfolio_id=portfolio.id,
                    timestamp=datetime.strptime(trade_data['timestamp'], '%Y-%m-%d %H:%M:%S.%f'),
                    ticker=trade_data['asset'],
                    action=trade_data['action'],
                    quantity=trade_data['qty'],
                    price=trade_data['price'],
                    pnl=trade_data.get('pnl', 0.0),
                    stop_loss=trade_data.get('stop_loss', 0.0),
                    take_profit=trade_data.get('take_profit', 0.0),
                    trade_metadata=trade_data.get('metadata', {})
                )
                session.add(trade)
                
        session.flush()  # Ensure IDs are generated
    
    def _load_json(self, filepath: str) -> Dict:
        """Load JSON file with error handling"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
        return {}
    
    def _backup_json_files(self, data_dir: str):
        """Create backup of original JSON files"""
        backup_dir = os.path.join(data_dir, 'json_backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        
        for filename in ['portfolio_india_paper.json', 'portfolio_india_live.json',
                        'trade_log_india_paper.json', 'trade_log_india_live.json']:
            src = os.path.join(data_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, filename)
                with open(src, 'r') as f_src, open(dst, 'w') as f_dst:
                    f_dst.write(f_src.read())
        
        logger.info(f"Created JSON backups in {backup_dir}")
