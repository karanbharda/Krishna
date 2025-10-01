#!/usr/bin/env python3
"""
Test script to verify dynamic stop-loss configuration from live_config.json
"""

import sys
import os
import json

# Add the backend directory to Python path
sys.path.append('c:/Users/Admin/Desktop/backup/project/backend')

try:
    from core.professional_buy_logic import ProfessionalBuyLogic

    # Test loading dynamic config
    print("=== Testing Dynamic Config Loading ===")

    # Create a test config
    test_config = {
        "min_buy_signals": 2,
        "min_buy_confidence": 0.40,
        "min_weighted_buy_score": 0.04
    }

    # Initialize buy logic
    buy_logic = ProfessionalBuyLogic(test_config)

    # Check if dynamic values were loaded
    print(f"Stop Loss Percentage: {buy_logic.stop_loss_pct}")
    print(f"Take Profit Ratio: {buy_logic.take_profit_ratio}")

    # Check current live_config.json
    config_path = 'c:/Users/Admin/Desktop/backup/project/data/live_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            live_config = json.load(f)
        print(f"Live Config Stop Loss: {live_config.get('stop_loss_pct', 'Not Set')}")
        print(f"Live Config Take Profit: {live_config.get('take_profit_ratio', 'Not Set')}")

        # Test refresh
        print("\n=== Testing Config Refresh ===")
        buy_logic.refresh_dynamic_config()
        print(f"After Refresh - Stop Loss: {buy_logic.stop_loss_pct}")
        print(f"After Refresh - Take Profit: {buy_logic.take_profit_ratio}")

        print("\n✅ Dynamic config loading test completed successfully!")
    else:
        print("❌ live_config.json not found")

except Exception as e:
    print(f"❌ Error testing dynamic config: {e}")
    import traceback
    traceback.print_exc()
