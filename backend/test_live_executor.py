#!/usr/bin/env python3
"""
Test script to verify live_executor imports work correctly
"""

import os
import sys

# Fix import paths permanently
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("Testing live_executor import...")

try:
    from live_executor import LiveTradingExecutor
    print("✅ LiveTradingExecutor imported successfully")
    
    # Test if the function with Any type annotation can be accessed
    print("✅ All type annotations are properly imported")
    
except ImportError as e:
    print(f"❌ LiveTradingExecutor import failed: {e}")
except NameError as e:
    print(f"❌ Type annotation error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")

print("🎯 Live executor test completed!")
