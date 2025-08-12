"""
Script to verify migrated data
"""
import os
from pathlib import Path
from backend.db.database import DatabaseManager, Portfolio, Holding, Trade

def main():
    """Verify migrated data"""
    project_dir = Path(__file__).parent.parent.parent
    data_dir = os.path.join(project_dir, 'data')
    db_path = f'sqlite:///{os.path.join(data_dir, "trading.db")}'
    
    print(f"Verifying migration in {db_path}")
    
    db = DatabaseManager(db_path)
    session = db.Session()
    
    try:
        # Check portfolios
        for mode in ['paper', 'live']:
            portfolio = session.query(Portfolio).filter_by(mode=mode).first()
            if portfolio:
                print(f"\n{mode.title()} Portfolio:")
                print(f"Cash: {portfolio.cash:.2f}")
                print(f"Starting Balance: {portfolio.starting_balance:.2f}")
                print(f"Realized PnL: {portfolio.realized_pnl:.2f}")
                print(f"Unrealized PnL: {portfolio.unrealized_pnl:.2f}")
                
                print(f"\n{mode.title()} Holdings ({len(portfolio.holdings)}):")
                holdings_by_ticker = {}
                for holding in portfolio.holdings:
                    if holding.ticker not in holdings_by_ticker:
                        holdings_by_ticker[holding.ticker] = 0
                    holdings_by_ticker[holding.ticker] += holding.quantity
                
                for ticker, quantity in sorted(holdings_by_ticker.items()):
                    holding = next(h for h in portfolio.holdings if h.ticker == ticker)
                    print(f"{ticker}: {quantity} shares @ {holding.avg_price:.2f}")
                
                print(f"\n{mode.title()} Trades ({len(portfolio.trades)}):")
                for trade in sorted(portfolio.trades, key=lambda t: t.timestamp):
                    print(f"{trade.timestamp}: {trade.action} {trade.quantity} {trade.ticker} @ {trade.price:.2f}")
            
    finally:
        session.close()

if __name__ == "__main__":
    main()
