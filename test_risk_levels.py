#!/usr/bin/env python3
"""
Test script to verify dynamic risk level functionality in the trading bot.
Tests both frontend settings and backend application of risk parameters.
"""

import requests
import json
import time
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# API endpoints
BASE_URL = "http://127.0.0.1:5000/api"
FRONTEND_URL = "http://localhost:3001"

class RiskLevelTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name, success, message):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message
        })
    
    def test_backend_connection(self):
        """Test if backend is running"""
        try:
            response = self.session.get(f"{BASE_URL}/status")
            if response.status_code == 200:
                self.log_test("Backend Connection", True, "Backend is running")
                return True
            else:
                self.log_test("Backend Connection", False, f"Backend returned {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Backend Connection", False, f"Cannot connect to backend: {e}")
            return False
    
    def test_get_default_settings(self):
        """Test getting default settings"""
        try:
            response = self.session.get(f"{BASE_URL}/settings")
            if response.status_code == 200:
                settings = response.json()
                expected_keys = ["mode", "riskLevel", "stop_loss_pct", "max_capital_per_trade", "max_trade_limit"]
                
                missing_keys = [key for key in expected_keys if key not in settings]
                if missing_keys:
                    self.log_test("Default Settings", False, f"Missing keys: {missing_keys}")
                    return False
                
                # Check default values
                if settings["riskLevel"] == "MEDIUM":
                    self.log_test("Default Settings", True, f"Default risk level is MEDIUM: {settings}")
                    return settings
                else:
                    self.log_test("Default Settings", False, f"Expected MEDIUM, got {settings['riskLevel']}")
                    return False
            else:
                self.log_test("Default Settings", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Default Settings", False, f"Error: {e}")
            return False
    
    def test_risk_level_settings(self, risk_level, expected_stop_loss, expected_allocation):
        """Test setting a specific risk level"""
        try:
            # Set the risk level
            settings_data = {
                "riskLevel": risk_level,
                "mode": "paper"
            }
            
            # Add custom values for CUSTOM risk level
            if risk_level == "CUSTOM":
                settings_data["stop_loss_pct"] = expected_stop_loss
                settings_data["max_capital_per_trade"] = expected_allocation
            
            response = self.session.post(f"{BASE_URL}/settings", json=settings_data)
            
            if response.status_code == 200:
                # Verify the settings were applied
                time.sleep(0.5)  # Give backend time to process
                verify_response = self.session.get(f"{BASE_URL}/settings")
                
                if verify_response.status_code == 200:
                    settings = verify_response.json()
                    
                    # Check if risk level was set
                    if settings["riskLevel"] != risk_level:
                        self.log_test(f"Risk Level {risk_level}", False, 
                                    f"Risk level not set correctly: expected {risk_level}, got {settings['riskLevel']}")
                        return False
                    
                    # Check stop loss and allocation
                    actual_stop_loss = settings["stop_loss_pct"]
                    actual_allocation = settings["max_capital_per_trade"]
                    
                    stop_loss_match = abs(actual_stop_loss - expected_stop_loss) < 0.001
                    allocation_match = abs(actual_allocation - expected_allocation) < 0.001
                    
                    if stop_loss_match and allocation_match:
                        self.log_test(f"Risk Level {risk_level}", True, 
                                    f"Stop Loss: {actual_stop_loss*100}%, Allocation: {actual_allocation*100}%")
                        return True
                    else:
                        self.log_test(f"Risk Level {risk_level}", False, 
                                    f"Values mismatch - Expected: SL={expected_stop_loss*100}%, AL={expected_allocation*100}% | "
                                    f"Actual: SL={actual_stop_loss*100}%, AL={actual_allocation*100}%")
                        return False
                else:
                    self.log_test(f"Risk Level {risk_level}", False, "Failed to verify settings")
                    return False
            else:
                self.log_test(f"Risk Level {risk_level}", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test(f"Risk Level {risk_level}", False, f"Error: {e}")
            return False
    
    def test_bot_start_with_risk_level(self, risk_level):
        """Test starting bot with specific risk level"""
        try:
            # First set the risk level
            settings_data = {"riskLevel": risk_level, "mode": "paper"}
            if risk_level == "CUSTOM":
                settings_data["stop_loss_pct"] = 0.07  # 7%
                settings_data["max_capital_per_trade"] = 0.30  # 30%

            self.session.post(f"{BASE_URL}/settings", json=settings_data)
            time.sleep(0.5)

            # Start the bot
            response = self.session.post(f"{BASE_URL}/start")

            if response.status_code == 200:
                result = response.json()
                if risk_level in result.get("message", ""):
                    self.log_test(f"Bot Start {risk_level}", True, result["message"])

                    # Stop the bot
                    time.sleep(1)
                    stop_response = self.session.post(f"{BASE_URL}/stop")
                    return True
                else:
                    self.log_test(f"Bot Start {risk_level}", False, f"Risk level not mentioned in response: {result}")
                    return False
            else:
                self.log_test(f"Bot Start {risk_level}", False, f"HTTP {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self.log_test(f"Bot Start {risk_level}", False, f"Error: {e}")
            return False

    def test_live_trading_mode_switch(self):
        """Test switching between paper and live trading modes"""
        try:
            # Test switching to live mode
            settings_data = {"mode": "live", "riskLevel": "MEDIUM"}
            response = self.session.post(f"{BASE_URL}/settings", json=settings_data)

            if response.status_code == 200:
                # Verify the mode was switched
                time.sleep(0.5)
                verify_response = self.session.get(f"{BASE_URL}/settings")

                if verify_response.status_code == 200:
                    settings = verify_response.json()
                    if settings.get("mode") == "live":
                        self.log_test("Live Mode Switch", True, "Successfully switched to live mode")

                        # Switch back to paper mode
                        settings_data = {"mode": "paper", "riskLevel": "MEDIUM"}
                        response = self.session.post(f"{BASE_URL}/settings", json=settings_data)

                        if response.status_code == 200:
                            self.log_test("Paper Mode Switch", True, "Successfully switched back to paper mode")
                            return True
                        else:
                            self.log_test("Paper Mode Switch", False, f"Failed to switch back: {response.status_code}")
                            return False
                    else:
                        self.log_test("Live Mode Switch", False, f"Mode not switched: {settings.get('mode')}")
                        return False
                else:
                    self.log_test("Live Mode Switch", False, "Failed to verify mode switch")
                    return False
            else:
                self.log_test("Live Mode Switch", False, f"HTTP {response.status_code}: {response.text}")
                return False

        except Exception as e:
            self.log_test("Live Mode Switch", False, f"Error: {e}")
            return False

    def test_live_status_endpoint(self):
        """Test live trading status endpoint"""
        try:
            response = self.session.get(f"{BASE_URL}/live-status")

            if response.status_code == 200:
                status = response.json()
                expected_keys = ["available", "mode"]

                missing_keys = [key for key in expected_keys if key not in status]
                if missing_keys:
                    self.log_test("Live Status Endpoint", False, f"Missing keys: {missing_keys}")
                    return False

                self.log_test("Live Status Endpoint", True, f"Status: {status}")
                return True
            else:
                self.log_test("Live Status Endpoint", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_test("Live Status Endpoint", False, f"Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive risk level tests"""
        print("🚀 Starting Risk Level Tests...")
        print("=" * 60)
        
        # Test 1: Backend connection
        if not self.test_backend_connection():
            print("❌ Cannot proceed without backend connection")
            return False
        
        # Test 2: Default settings
        default_settings = self.test_get_default_settings()
        if not default_settings:
            print("❌ Cannot proceed without default settings")
            return False
        
        # Test 3: Risk level mappings
        risk_mappings = {
            "LOW": {"stop_loss": 0.03, "allocation": 0.15},      # 3%, 15%
            "MEDIUM": {"stop_loss": 0.05, "allocation": 0.25},   # 5%, 25%
            "HIGH": {"stop_loss": 0.08, "allocation": 0.35},     # 8%, 35%
            "CUSTOM": {"stop_loss": 0.12, "allocation": 0.40}    # 12%, 40% (custom values)
        }
        
        print("\n📊 Testing Risk Level Settings...")
        for risk_level, settings in risk_mappings.items():
            self.test_risk_level_settings(risk_level, settings["stop_loss"], settings["allocation"])
        
        # Test 4: Bot start with different risk levels
        print("\n🤖 Testing Bot Start with Risk Levels...")
        for risk_level in ["LOW", "MEDIUM", "HIGH", "CUSTOM"]:
            self.test_bot_start_with_risk_level(risk_level)

        # Test 5: Live trading mode switching
        print("\n🔄 Testing Live Trading Mode Switching...")
        self.test_live_trading_mode_switch()

        # Test 6: Live status endpoint
        print("\n📊 Testing Live Status Endpoint...")
        self.test_live_status_endpoint()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Risk level functionality is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
            return False

def main():
    """Main test function"""
    print("🧪 Trading Bot Risk Level Test Suite")
    print("=" * 60)
    print("This script tests the dynamic risk level functionality.")
    print("Make sure both backend (port 5000) and frontend (port 3001) are running.")
    print()
    
    # Wait for user confirmation
    input("Press Enter to start tests...")
    
    tester = RiskLevelTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("You can now test the frontend by:")
        print("1. Opening http://localhost:3001")
        print("2. Clicking the Settings (⚙️) button")
        print("3. Changing risk levels and observing the parameter updates")
        print("4. Starting the bot and checking the console logs")
    else:
        print("\n❌ Some tests failed. Please check the backend implementation.")
    
    return success

if __name__ == "__main__":
    main()
