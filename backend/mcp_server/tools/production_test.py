#!/usr/bin/env python3
"""
Production Test for Venting Layer
================================

Test script to verify all four tools are working together in a production-level pipeline
"""

import asyncio
import json
import sys
import os

# Add the backend directory to the path
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from mcp_server.tools.predict_tool import PredictTool
from mcp_server.tools.analyze_tool import AnalyzeTool
from mcp_server.tools.scan_all_tool import ScanAllTool
from mcp_server.tools.confirm_tool import ConfirmTool

async def test_venting_layer_pipeline():
    """Test the complete venting layer pipeline"""
    print("=== Production Test: Venting Layer Pipeline ===")
    
    # Initialize all four tools
    predict_tool = PredictTool({
        "tool_id": "predict_tool_test",
        "lightgbm_enabled": True,
        "rl_model_type": "linucb",
        "real_time_data": False  # Disable for testing
    })
    
    analyze_tool = AnalyzeTool({
        "tool_id": "analyze_tool_test",
        "llama_enabled": False,  # Disable LLaMA for testing
        "real_time_data": False
    })
    
    scan_all_tool = ScanAllTool({
        "tool_id": "scan_all_tool_test",
        "rl_model_type": "linucb",
        "real_time_data": False
    })
    
    confirm_tool = ConfirmTool({
        "tool_id": "confirm_tool_test",
        "executor_enabled": True,
        "trading_mode": "paper",
        "real_time_data": False
    })
    
    print("✓ All tools initialized successfully")
    
    # Test 1: Scan all stocks
    print("\n1. Testing Scan All Tool...")
    scan_result = await scan_all_tool.scan_all({
        "min_score": 0.6,
        "max_results": 5,
        "sectors": ["IT", "BANKING"]
    }, "test_session_1")
    
    if scan_result.status.value != "success":
        print(f"✗ Scan failed: {scan_result.error}")
        return False
    
    print(f"✓ Scan completed: {scan_result.data['total_ranked']} stocks ranked")
    
    # Test 2: Get predictions for top stocks
    print("\n2. Testing Predict Tool...")
    top_symbols = [rank["symbol"] for rank in scan_result.data["rankings"][:3]]
    predict_result = await predict_tool.predict({
        "symbols": top_symbols,
        "model_type": "ensemble"
    }, "test_session_2")
    
    if predict_result.status.value != "success":
        print(f"✗ Prediction failed: {predict_result.error}")
        return False
    
    print(f"✓ Predictions generated: {len(predict_result.data['predictions'])} predictions")
    
    # Test 3: Analyze predictions
    print("\n3. Testing Analyze Tool...")
    analyze_result = await analyze_tool.analyze({
        "predictions": predict_result.data["predictions"],
        "analysis_depth": "detailed"
    }, "test_session_3")
    
    if analyze_result.status.value != "success":
        print(f"✗ Analysis failed: {analyze_result.error}")
        return False
    
    print(f"✓ Analysis completed: {len(analyze_result.data['analysis_results'])} analyses")
    
    # Test 4: Create actions based on analysis
    print("\n4. Creating trade actions...")
    actions = []
    for i, (pred, analysis) in enumerate(zip(
        predict_result.data["predictions"], 
        analyze_result.data["analysis_results"]
    )):
        action = "BUY" if pred["prediction_score"] > 0.7 else "HOLD"
        if i == 0:  # Force one sell for demo
            action = "SELL"
        
        actions.append({
            "symbol": pred["symbol"],
            "action": action,
            "confidence": pred["prediction_score"],
            "analysis": {
                "sentiment": analysis["sentiment"],
                "key_factors": analysis["key_factors"]
            }
        })
    
    print(f"✓ Created {len(actions)} actions")
    
    # Test 5: Confirm actions
    print("\n5. Testing Confirm Tool...")
    confirm_result = await confirm_tool.confirm({
        "actions": actions,
        "portfolio_value": 1000000,
        "risk_check": True
    }, "test_session_4")
    
    if confirm_result.status.value != "success":
        print(f"✗ Confirmation failed: {confirm_result.error}")
        return False
    
    confirmed_count = confirm_result.data["confirmed_actions"]
    total_actions = confirm_result.data["total_actions"]
    print(f"✓ Confirmation completed: {confirmed_count}/{total_actions} actions confirmed")
    
    # Print final results
    print("\n=== Final Results ===")
    for result in confirm_result.data["confirmation_results"]:
        status = "✓" if result["confirmed"] else "✗"
        print(f"   {status} {result['symbol']}: {result['action']} ({result['confidence']:.2f}) - {result['reason']}")
    
    print("\n=== Pipeline Test Completed Successfully ===")
    return True

def main():
    """Main function"""
    try:
        result = asyncio.run(test_venting_layer_pipeline())
        if result:
            print("\n🎉 All tests passed! Venting layer is working correctly.")
            return 0
        else:
            print("\n❌ Some tests failed.")
            return 1
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())