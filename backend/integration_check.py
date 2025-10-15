import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all major components can be imported"""
    print("Testing component imports...")
    
    components = {
        "FastAPI App": "app",
        "Sentiment Tool": "mcp_server.tools.sentiment_tool",
        "Stock Analyzer": "testindia",
        "Professional Buy Logic": "core.professional_buy_logic",
        "Professional Sell Logic": "core.professional_sell_logic",
        "Market Analysis Tool": "mcp_server.tools.market_analysis_tool",
        "ML Interface": "ml_interface",
        "Web Backend": "web_backend"
    }
    
    results = {}
    
    for name, module in components.items():
        try:
            __import__(module)
            results[name] = "✅ PASS"
            print(f"   {name}: ✅ PASS")
        except Exception as e:
            results[name] = f"❌ FAIL - {str(e)}"
            print(f"   {name}: ❌ FAIL - {str(e)}")
    
    return results

def test_initialization():
    """Test that components can be initialized"""
    print("\nTesting component initialization...")
    
    init_results = {}
    
    # Test Sentiment Tool
    try:
        from mcp_server.tools.sentiment_tool import SentimentTool
        sentiment_tool = SentimentTool({
            "tool_id": "test_sentiment_tool",
            "sentiment_sources": ["news", "social"]
        })
        init_results["Sentiment Tool"] = "✅ PASS"
        print("   Sentiment Tool: ✅ PASS")
    except Exception as e:
        init_results["Sentiment Tool"] = f"❌ FAIL - {str(e)}"
        print(f"   Sentiment Tool: ❌ FAIL - {str(e)}")
    
    # Test Stock Analyzer
    try:
        from testindia import Stock
        stock_analyzer = Stock()
        init_results["Stock Analyzer"] = "✅ PASS"
        print("   Stock Analyzer: ✅ PASS")
    except Exception as e:
        init_results["Stock Analyzer"] = f"❌ FAIL - {str(e)}"
        print(f"   Stock Analyzer: ❌ FAIL - {str(e)}")
    
    # Test Professional Buy Logic
    try:
        from core.professional_buy_logic import ProfessionalBuyLogic
        buy_logic = ProfessionalBuyLogic({
            "min_buy_signals": 3,
            "min_buy_confidence": 0.60
        })
        init_results["Professional Buy Logic"] = "✅ PASS"
        print("   Professional Buy Logic: ✅ PASS")
    except Exception as e:
        init_results["Professional Buy Logic"] = f"❌ FAIL - {str(e)}"
        print(f"   Professional Buy Logic: ❌ FAIL - {str(e)}")
    
    # Test Professional Sell Logic
    try:
        from core.professional_sell_logic import ProfessionalSellLogic
        sell_logic = ProfessionalSellLogic({
            "min_sell_signals": 2,
            "min_sell_confidence": 0.65
        })
        init_results["Professional Sell Logic"] = "✅ PASS"
        print("   Professional Sell Logic: ✅ PASS")
    except Exception as e:
        init_results["Professional Sell Logic"] = f"❌ FAIL - {str(e)}"
        print(f"   Professional Sell Logic: ❌ FAIL - {str(e)}")
    
    # Test Market Analysis Tool
    try:
        from mcp_server.tools.market_analysis_tool import MarketAnalysisTool
        market_tool = MarketAnalysisTool({
            "tool_id": "test_market_tool"
        })
        init_results["Market Analysis Tool"] = "✅ PASS"
        print("   Market Analysis Tool: ✅ PASS")
    except Exception as e:
        init_results["Market Analysis Tool"] = f"❌ FAIL - {str(e)}"
        print(f"   Market Analysis Tool: ❌ FAIL - {str(e)}")
    
    # Test ML Interface
    try:
        from ml_interface import get_ml_interface
        ml_interface = get_ml_interface()
        init_results["ML Interface"] = "✅ PASS"
        print("   ML Interface: ✅ PASS")
    except Exception as e:
        init_results["ML Interface"] = f"❌ FAIL - {str(e)}"
        print(f"   ML Interface: ❌ FAIL - {str(e)}")
    
    return init_results

def main():
    """Main function"""
    print("=" * 60)
    print("TRADING SYSTEM COMPONENT INTEGRATION CHECK")
    print("=" * 60)
    print(f"Check started at: {datetime.now()}")
    print()
    
    # Test imports
    import_results = test_imports()
    
    # Test initialization
    init_results = test_initialization()
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION CHECK SUMMARY")
    print("=" * 60)
    
    print("\nImport Tests:")
    import_passes = 0
    for component, result in import_results.items():
        print(f"   {component}: {result}")
        if "✅ PASS" in result:
            import_passes += 1
    
    print("\nInitialization Tests:")
    init_passes = 0
    for component, result in init_results.items():
        print(f"   {component}: {result}")
        if "✅ PASS" in result:
            init_passes += 1
    
    total_tests = len(import_results) + len(init_results)
    total_passes = import_passes + init_passes
    
    print(f"\nOverall Result: {total_passes}/{total_tests} tests passed")
    if total_passes == total_tests:
        print("🎉 ALL COMPONENTS INTEGRATED SUCCESSFULLY!")
    else:
        print(f"⚠️  {total_tests - total_passes} component(s) failed integration")
    
    print(f"Check completed at: {datetime.now()}")
    print("=" * 60)

if __name__ == "__main__":
    main()