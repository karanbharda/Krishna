from database import DatabaseManager, Portfolio, Trade, Holding
import os

def verify_database():
    db_path = f'sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "trading.db")}'
    print(f"Checking database at: {db_path}")
    
    db = DatabaseManager(db_path)
    session = db.Session()
    
    try:
        # Check portfolios
        print("\nPortfolios:")
        portfolios = session.query(Portfolio).all()
        for p in portfolios:
            print(f"\n{p.mode.upper()} Portfolio:")
            print(f"Cash: {p.cash:.2f}")
            print(f"Starting Balance: {p.starting_balance:.2f}")
            print(f"Realized PnL: {p.realized_pnl:.2f}")
            print(f"Unrealized PnL: {p.unrealized_pnl:.2f}")
            print(f"Last Updated: {p.last_updated}")
            
            print(f"\nHoldings:")
            for h in p.holdings:
                print(f"{h.ticker}: {h.quantity} @ {h.avg_price:.2f}")
            
            print(f"\nTrades:")
            for t in p.trades:
                print(f"{t.timestamp}: {t.action} {t.quantity} {t.ticker} @ {t.price:.2f} (PnL: {t.pnl:.2f})")
                
    finally:
        session.close()

if __name__ == "__main__":
    verify_database()
