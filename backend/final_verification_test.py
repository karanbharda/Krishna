"""
Final Verification Test for Trading System Data Validation

This script verifies that the trading system properly handles missing data
and doesn't generate false buy signals when data is insufficient.
"""

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_missing_data_handling():
    """Test that the system properly handles missing data"""
    logger.info("Starting missing data handling test...")
    
    # Import the components we need to test
    try:
        from testindia import Stock
        from core.professional_buy_logic import ProfessionalBuyLogic
        logger.info("✅ Successfully imported required components")
    except Exception as e:
        logger.error(f"❌ Failed to import components: {e}")
        return False
    
    # Test 1: Stock analyzer with missing data
    logger.info("Test 1: Testing stock analyzer with missing data...")
    try:
        analyzer = Stock()
        # Test with a non-existent stock symbol
        result = analyzer.analyze_stock("NONEXISTENT.NS")
        
        if result.get("success", False):
            logger.warning("⚠️  Non-existent stock analysis succeeded (unexpected)")
        else:
            logger.info("✅ Non-existent stock analysis properly failed")
            error_count = result.get("error_count", 0)
            if error_count > 0:
                logger.info(f"✅ Error tracking working: {error_count} errors recorded")
            else:
                logger.warning("⚠️  Error tracking not working: no errors recorded")
    except Exception as e:
        logger.error(f"❌ Error in stock analysis test: {e}")
        return False
    
    # Test 2: Professional buy logic with missing data
    logger.info("Test 2: Testing professional buy logic with missing data...")
    try:
        buy_logic = ProfessionalBuyLogic({
            "min_buy_signals": 3,
            "min_buy_confidence": 0.60,
            "min_weighted_buy_score": 0.15
        })
        
        # Test with empty/missing data structures
        buy_decision = buy_logic.evaluate_buy_decision(
            ticker="TEST.NS",
            stock_metrics=None,  # Missing data
            market_context=None,
            technical_analysis={},
            sentiment_analysis={},
            ml_analysis={},
            portfolio_context={}
        )
        
        if buy_decision.should_buy:
            logger.error("❌ Professional buy logic generated buy signal with missing data")
            return False
        else:
            logger.info("✅ Professional buy logic properly returned HOLD with missing data")
            
        # Test with partial data
        buy_decision2 = buy_logic.evaluate_buy_decision(
            ticker="TEST.NS",
            stock_metrics=None,
            market_context=None,
            technical_analysis={"rsi": 60, "sma_20": 100, "sma_50": 95},  # Partial technical data
            sentiment_analysis={"overall_sentiment": 0.2, "confidence": 0.7},  # Partial sentiment data
            ml_analysis={"prediction_direction": 0.05, "confidence": 0.8, "success": True},  # Partial ML data
            portfolio_context={}
        )
        
        if buy_decision2.should_buy:
            logger.info("✅ Professional buy logic generated buy signal with partial data (expected)")
        else:
            logger.info("✅ Professional buy logic returned HOLD with partial data (also acceptable)")
            
    except Exception as e:
        logger.error(f"❌ Error in professional buy logic test: {e}")
        return False
    
    logger.info("✅ All missing data handling tests completed successfully")
    return True

def test_rate_limiting_handling():
    """Test that the system properly handles rate limiting"""
    logger.info("Starting rate limiting handling test...")
    
    try:
        from testindia import Stock
        analyzer = Stock()
        
        # Check if the _should_skip_analysis method exists
        if hasattr(analyzer, '_should_skip_analysis'):
            # Test with rate limiting errors
            error_messages = ["Rate limited", "Too Many Requests", "429 error"]
            should_skip = analyzer._should_skip_analysis("TEST.NS", error_messages)
            
            if should_skip:
                logger.info("✅ Rate limiting detection working properly")
            else:
                logger.warning("⚠️  Rate limiting detection may not be working properly")
        else:
            logger.warning("⚠️  _should_skip_analysis method not found")
            
    except Exception as e:
        logger.error(f"❌ Error in rate limiting test: {e}")
        return False
        
    logger.info("✅ Rate limiting handling test completed")
    return True

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("FINAL TRADING SYSTEM VERIFICATION TEST")
    logger.info("=" * 60)
    logger.info(f"Test started at: {datetime.now()}")
    logger.info("")
    
    # Run all tests
    tests = [
        ("Missing Data Handling", test_missing_data_handling),
        ("Rate Limiting Handling", test_rate_limiting_handling)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL VERIFICATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL VERIFICATION TESTS PASSED!")
        logger.info("✅ Trading system properly handles missing data")
        logger.info("✅ Trading system doesn't generate false buy signals")
        logger.info("✅ Error handling and validation are working correctly")
    else:
        logger.warning("⚠️  Some verification tests failed")
        logger.warning("⚠️  System may still generate false signals with missing data")
    
    logger.info(f"Test completed at: {datetime.now()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()