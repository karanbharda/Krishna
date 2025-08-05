"""
Code Quality: Integration tests for backend components
"""

import unittest
import asyncio
import sys
import os

# Priority 2: Fix test module import paths
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

# Import from utils modules
from utils import ConfigValidator, TradingBotError, ValidationError
from utils.exceptions import DataServiceError
from data_service_client import DataServiceClient

# Priority 1: Make test config self-contained
TEST_CONFIG = {
    "starting_balance": 10000,
    "mode": "paper",
    "tickers": ["RELIANCE.NS", "TCS.NS"],
    "stop_loss_pct": 0.05,
    "max_capital_per_trade": 0.25,
    "sleep_interval": 300
}


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration passes validation"""
        try:
            ConfigValidator.validate_config(TEST_CONFIG.copy())
        except Exception as e:
            self.fail(f"Valid config failed validation: {e}")
    
    def test_invalid_starting_balance(self):
        """Test invalid starting balance raises error"""
        config = TEST_CONFIG.copy()
        config["starting_balance"] = -1000
        
        with self.assertRaises(ValueError):
            ConfigValidator.validate_config(config)
    
    def test_invalid_stop_loss(self):
        """Test invalid stop loss raises error"""
        config = TEST_CONFIG.copy()
        config["stop_loss_pct"] = 1.5  # > 1.0
        
        with self.assertRaises(ValueError):
            ConfigValidator.validate_config(config)


class TestDataServiceClient(unittest.TestCase):
    """Test data service client functionality"""
    
    def setUp(self):
        """Set up test client"""
        self.client = DataServiceClient("http://localhost:8001")
    
    def test_client_initialization(self):
        """Test client initializes correctly"""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.base_url, "http://localhost:8001")
        self.assertFalse(self.client.is_healthy)
    
    def test_cache_stats(self):
        """Test cache statistics"""
        stats = self.client.get_cache_stats()
        self.assertIn("cache_size", stats)
        self.assertIn("hit_rate_percent", stats)


class TestExceptionHierarchy(unittest.TestCase):
    """Test standardized exception hierarchy"""

    def test_exception_inheritance(self):
        """Test exception inheritance chain"""
        self.assertTrue(issubclass(ValidationError, TradingBotError))
        self.assertTrue(issubclass(DataServiceError, TradingBotError))
        self.assertTrue(issubclass(TradingBotError, Exception))


class TestModularIntegration(unittest.TestCase):
    """Priority 4: Test modular integration"""

    def test_utils_import(self):
        """Test utils modules can be imported"""
        try:
            from utils import ConfigValidator, PerformanceMonitor
            self.assertIsNotNone(ConfigValidator)
            self.assertIsNotNone(PerformanceMonitor)
        except ImportError:
            self.fail("Utils modules should be importable")

    def test_performance_monitor_functionality(self):
        """Test performance monitor basic functionality"""
        from utils import PerformanceMonitor
        monitor = PerformanceMonitor()

        # Test recording request
        monitor.record_request(0.1, True)
        stats = monitor.get_stats()

        self.assertIn("total_requests", stats)
        self.assertEqual(stats["total_requests"], 1)

    def test_config_validator_integration(self):
        """Test config validator with real config"""
        from utils import ConfigValidator

        valid_config = {
            "starting_balance": 10000,
            "stop_loss_pct": 0.05,
            "max_capital_per_trade": 0.25,
            "tickers": ["TEST.NS"],
            "sleep_interval": 300
        }

        # Should not raise exception
        result = ConfigValidator.validate_config(valid_config)
        self.assertEqual(result, valid_config)


if __name__ == "__main__":
    unittest.main()
