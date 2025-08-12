from database import DatabaseManager, Portfolio, Trade, Holding
import os
from datetime import datetime

def fix_database():
    db_path = f'sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "trading.db")}'
    print(f"Fixing database at: {db_path}")
    
    db = DatabaseManager(db_path)
    session = db.Session()
    
    try:
        # Process each portfolio
        for portfolio in session.query(Portfolio).all():
            print(f"\nProcessing {portfolio.mode} portfolio...")
            
            # Fix duplicate holdings
            holdings_seen = {}
            holdings_to_delete = []
            
            for holding in portfolio.holdings:
                key = f"{holding.ticker}"
                if key in holdings_seen:
                    # Merge quantities and delete duplicate
                    holdings_seen[key].quantity += holding.quantity
                    holdings_to_delete.append(holding)
                else:
                    holdings_seen[key] = holding
            
            # Delete duplicate holdings
            for holding in holdings_to_delete:
                session.delete(holding)
            
            # Fix duplicate trades
            trades_seen = {}
            trades_to_delete = []
            
            for trade in portfolio.trades:
                key = f"{trade.timestamp}_{trade.ticker}_{trade.action}_{trade.quantity}"
                if key in trades_seen:
                    trades_to_delete.append(trade)
                else:
                    trades_seen[key] = trade
            
            # Delete duplicate trades
            for trade in trades_to_delete:
                session.delete(trade)
            
            # Recalculate P&L
            portfolio.realized_pnl = 0.0
            portfolio.unrealized_pnl = 0.0
            
            # Process trades in chronological order
            holdings_state = {}  # Keep track of holdings for P&L calculation
            
            for trade in sorted(trades_seen.values(), key=lambda t: t.timestamp):
                if trade.action.lower() == 'buy':
                    if trade.ticker not in holdings_state:
                        holdings_state[trade.ticker] = {'quantity': 0, 'total_cost': 0.0}
                    
                    holdings_state[trade.ticker]['quantity'] += trade.quantity
                    holdings_state[trade.ticker]['total_cost'] += (trade.quantity * trade.price)
                    
                elif trade.action.lower() == 'sell':
                    if trade.ticker in holdings_state:
                        avg_cost = holdings_state[trade.ticker]['total_cost'] / holdings_state[trade.ticker]['quantity']
                        trade.pnl = (trade.price - avg_cost) * trade.quantity
                        portfolio.realized_pnl += trade.pnl
                        
                        # Update holdings state
                        holdings_state[trade.ticker]['quantity'] -= trade.quantity
                        holdings_state[trade.ticker]['total_cost'] -= (avg_cost * trade.quantity)
            
            portfolio.last_updated = datetime.now()
            
            print(f"Removed {len(holdings_to_delete)} duplicate holdings")
            print(f"Removed {len(trades_to_delete)} duplicate trades")
            print(f"Recalculated realized P&L: {portfolio.realized_pnl:.2f}")
        
        session.commit()
        print("\nDatabase cleanup completed successfully")
        
    except Exception as e:
        session.rollback()
        print(f"Error fixing database: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    fix_database()
