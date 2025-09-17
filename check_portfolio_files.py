import os

# Check for portfolio files with correct paths
print('Paper portfolio file exists:', os.path.exists('data/portfolio_india_paper.json'))
print('Live portfolio file exists:', os.path.exists('data/portfolio_india_live.json'))
print('Paper trade log exists:', os.path.exists('data/trade_log_india_paper.json'))
print('Live trade log exists:', os.path.exists('data/trade_log_india_live.json'))

# Show the actual content of these files if they exist
if os.path.exists('data/portfolio_india_paper.json'):
    with open('data/portfolio_india_paper.json', 'r') as f:
        import json
        data = json.load(f)
        print("\nPaper Portfolio Content:")
        print(f"  Cash: {data.get('cash', 'N/A')}")
        print(f"  Holdings: {len(data.get('holdings', {}))} positions")
        print(f"  Starting Balance: {data.get('starting_balance', 'N/A')}")

if os.path.exists('data/portfolio_india_live.json'):
    with open('data/portfolio_india_live.json', 'r') as f:
        import json
        data = json.load(f)
        print("\nLive Portfolio Content:")
        print(f"  Cash: {data.get('cash', 'N/A')}")
        print(f"  Holdings: {len(data.get('holdings', {}))} positions")
        print(f"  Starting Balance: {data.get('starting_balance', 'N/A')}")