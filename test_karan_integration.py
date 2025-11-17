#!/usr/bin/env python3
"""
Test script to verify Karan folder integration with the trading system
"""

import sys
import os
import json

# Add the backend directory to the path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_mcp_adapter():
    """Test the MCP adapter integration"""
    print("Testing Karan folder integration...")
    print("=" * 50)
    
    try:
        # Import the MCP adapter
        from ml_engine.core.mcp_adapter import MCPAdapter
        print("✓ MCP Adapter imported successfully")
        
        # Create an instance
        adapter = MCPAdapter()
        print("✓ MCP Adapter instance created successfully")
        
        # Check available methods
        methods = [method for method in dir(adapter) if not method.startswith('_') and method in ['predict', 'scan_all', 'analyze', 'confirm']]
        print(f"✓ Available tools: {methods}")
        
        if len(methods) == 4:
            print("✓ All required tools (predict, scan_all, analyze, confirm) are available")
        else:
            print(f"✗ Missing tools. Expected 4, found {len(methods)}")
            return False
            
        # Test method signatures (without actually calling them to avoid data fetching)
        print("\nTesting method signatures...")
        import inspect
        
        # Test predict method
        predict_sig = inspect.signature(adapter.predict)
        print(f"✓ predict method signature: {predict_sig}")
        
        # Test scan_all method
        scan_all_sig = inspect.signature(adapter.scan_all)
        print(f"✓ scan_all method signature: {scan_all_sig}")
        
        # Test analyze method
        analyze_sig = inspect.signature(adapter.analyze)
        print(f"✓ analyze method signature: {analyze_sig}")
        
        # Test confirm method
        confirm_sig = inspect.signature(adapter.confirm)
        print(f"✓ confirm method signature: {confirm_sig}")
        
        print("\n" + "=" * 50)
        print("✓ Karan folder integration test PASSED")
        print("✓ All four tools are properly integrated:")
        print("  1. predict - Generate stock price predictions")
        print("  2. scan_all - Scan multiple symbols and return ranked shortlist")
        print("  3. analyze - Analyze single symbol across multiple horizons")
        print("  4. confirm - Confirm or reject a trade decision")
        return True
        
    except Exception as e:
        print(f"✗ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mcp_adapter()
    sys.exit(0 if success else 1)