#!/usr/bin/env python3
"""
Venting Layer Demo
==================

Demonstration of the four tools working together in the venting layer
"""

import asyncio
import json
from datetime import datetime

# Import the venting layer tools
from .predict_tool import PredictTool
from .analyze_tool import AnalyzeTool
from .scan_all_tool import ScanAllTool
from .confirm_tool import ConfirmTool

class VentingLayerDemo:
    """Demo class to show how the four tools work together"""
    
    def __init__(self):
        # Initialize all four tools
        self.predict_tool = PredictTool({
            "tool_id": "predict_tool_demo",
            "lightgbm_enabled": True,
            "rl_model_type": "linucb",
            "real_time_data": True
        })
        
        self.analyze_tool = AnalyzeTool({
            "tool_id": "analyze_tool_demo",
            "llama_enabled": False,
            "real_time_data": True
        })
        
        self.scan_all_tool = ScanAllTool({
            "tool_id": "scan_all_tool_demo",
            "rl_model_type": "linucb",
            "real_time_data": True
        })
        
        self.confirm_tool = ConfirmTool({
            "tool_id": "confirm_tool_demo",
            "executor_enabled": True,
            "trading_mode": "paper",
            "real_time_data": True
        })
    
    async def run_full_workflow(self):
        """Run the complete venting layer workflow"""
        print("=== Venting Layer Demo ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        # Step 1: Scan all stocks
        print("1. Scanning all stocks...")
        scan_result = await self.scan_all_tool.scan_all({
            "min_score": 0.6,
            "max_results": 5,
            "sectors": ["IT", "BANKING"],
            "real_time": True
        }, "demo_session_1")
        
        if scan_result.status.value != "success":
            print(f"Scan failed: {scan_result.error}")
            return
        
        print(f"   Found {scan_result.data['total_ranked']} stocks")
        print()
        
        # Step 2: Get predictions for top stocks
        print("2. Generating predictions...")
        top_symbols = [rank["symbol"] for rank in scan_result.data["rankings"][:3]]
        predict_result = await self.predict_tool.predict({
            "symbols": top_symbols,
            "model_type": "ensemble",
            "real_time": True
        }, "demo_session_2")
        
        if predict_result.status.value != "success":
            print(f"Prediction failed: {predict_result.error}")
            return
        
        print(f"   Generated {len(predict_result.data['predictions'])} predictions")
        print()
        
        # Step 3: Analyze predictions
        print("3. Analyzing predictions...")
        analyze_result = await self.analyze_tool.analyze({
            "predictions": predict_result.data["predictions"],
            "analysis_depth": "detailed",
            "real_time": True
        }, "demo_session_3")
        
        if analyze_result.status.value != "success":
            print(f"Analysis failed: {analyze_result.error}")
            return
        
        print(f"   Analyzed {len(analyze_result.data['analysis_results'])} predictions")
        print()
        
        # Step 4: Create actions based on analysis
        print("4. Creating trade actions...")
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
        
        print(f"   Created {len(actions)} actions")
        print()
        
        # Step 5: Confirm actions
        print("5. Confirming actions...")
        confirm_result = await self.confirm_tool.confirm({
            "actions": actions,
            "portfolio_value": 1000000,
            "risk_check": True,
            "real_time": True
        }, "demo_session_4")
        
        if confirm_result.status.value != "success":
            print(f"Confirmation failed: {confirm_result.error}")
            return
        
        confirmed_count = confirm_result.data["confirmed_actions"]
        total_actions = confirm_result.data["total_actions"]
        print(f"   Confirmed {confirmed_count}/{total_actions} actions")
        print()
        
        # Print final results
        print("=== Final Results ===")
        for result in confirm_result.data["confirmation_results"]:
            status = "✓" if result["confirmed"] else "✗"
            print(f"   {status} {result['symbol']}: {result['action']} ({result['confidence']:.2f}) - {result['reason']}")
        
        print()
        print("=== Workflow Completed Successfully ===")
        
        return {
            "scan_result": scan_result,
            "predict_result": predict_result,
            "analyze_result": analyze_result,
            "confirm_result": confirm_result
        }

# Demo function
async def main():
    """Main demo function"""
    demo = VentingLayerDemo()
    results = await demo.run_full_workflow()
    return results

if __name__ == "__main__":
    asyncio.run(main())