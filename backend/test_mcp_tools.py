#!/usr/bin/env python3
"""
Test script for new MCP tools
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from mcp_server.tools.prediction_tool import PredictionTool
from mcp_server.tools.scan_tool import ScanTool

async def test_prediction_tool():
    """Test the prediction tool"""
    print("Testing Prediction Tool...")
    
    # Initialize the tool
    prediction_tool = PredictionTool({
        "tool_id": "test_prediction_tool",
        "ollama_enabled": False  # Disable Ollama for testing
    })
    
    # Test ranking predictions
    arguments = {
        "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "models": ["rl"],
        "horizon": "day",
        "include_explanations": True,
        "natural_query": ""
    }
    
    session_id = "test_session_1"
    result = await prediction_tool.rank_predictions(arguments, session_id)
    
    print(f"Prediction tool result status: {result.status}")
    if result.status.name == "SUCCESS":
        print(f"Ranked {result.data['total_predictions']} predictions")
        print(f"Models used: {result.data['models_used']}")
        print(f"Horizon: {result.data['horizon']}")
        if result.data['ranked_predictions']:
            print("Top prediction:")
            top_pred = result.data['ranked_predictions'][0]
            print(f"  Symbol: {top_pred['symbol']}")
            print(f"  Score: {top_pred['score']:.3f}")
            print(f"  Recommendation: {top_pred['recommendation']}")
            if top_pred.get('explanation'):
                print(f"  Explanation: {top_pred['explanation']}")
        else:
            print("No predictions were generated - all stocks failed risk compliance")
        return result
    else:
        print(f"Error: {result.error}")
        return result
    
    return result

async def test_scan_tool():
    """Test the scan tool"""
    print("\nTesting Scan Tool...")
    
    # Initialize the tool
    scan_tool = ScanTool({
        "tool_id": "test_scan_tool",
        "ollama_enabled": False  # Disable Ollama for testing
    })
    
    # Test scanning with filters
    arguments = {
        "filters": {
            "min_price": 100,
            "max_price": 5000,
            "sectors": ["IT", "BANKING"],
            "risk_levels": ["LOW", "MEDIUM"]
        },
        "sort_by": "score",
        "limit": 10,
        "natural_query": ""
    }
    
    session_id = "test_session_2"
    result = await scan_tool.scan_all(arguments, session_id)
    
    print(f"Scan tool result status: {result.status}")
    if result.status.name == "SUCCESS":
        print(f"Scanned {result.data['total_scanned']} stocks")
        print(f"Shortlisted {result.data['total_shortlisted']} stocks")
        print(f"Filters applied: {result.data['filters_applied']}")
        if result.data['shortlisted_stocks']:
            print("Top shortlisted stock:")
            top_stock = result.data['shortlisted_stocks'][0]
            print(f"  Symbol: {top_stock['symbol']}")
            print(f"  Score: {top_stock['score']:.3f}")
            print(f"  Recommendation: {top_stock['recommendation']}")
            print(f"  Price: {top_stock['price']:.2f}")
            print(f"  Filters matched: {', '.join(top_stock['filters_matched'])}")
            if top_stock.get('explanation'):
                print(f"  Explanation: {top_stock['explanation']}")
        else:
            print("No stocks were shortlisted - all stocks failed filters")
        return result
    else:
        print(f"Error: {result.error}")
        return result
    
    return result

async def main():
    """Main test function"""
    print("Testing MCP Tools Integration")
    print("=" * 50)
    
    try:
        # Test prediction tool
        pred_result = await test_prediction_tool()
        
        # Test scan tool
        scan_result = await test_scan_tool()
        
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"Prediction Tool: {'PASS' if pred_result.status.name == 'SUCCESS' else 'FAIL'}")
        print(f"Scan Tool: {'PASS' if scan_result.status.name == 'SUCCESS' else 'FAIL'}")
        
        if pred_result.status.name == "SUCCESS" and scan_result.status.name == "SUCCESS":
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"Test error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)