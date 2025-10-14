import json
import os
import time
import random
import numpy as np
import pandas as pd
import yfinance as yf
# Fyers API integration
try:
    from fyers_apiv3 import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False
from datetime import datetime, timedelta
import requests
import csv
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
from gnews import GNews
import pandas_market_calendars as mcal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import spaces
import traceback
import warnings
import logging
# PRODUCTION FIX: Add concurrent processing for speed
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pickle

from requests.exceptions import HTTPError
from dotenv import load_dotenv
from dhanhq import dhanhq
import argparse
import sys
import signal
import queue
import logging.handlers

# Fix import paths permanently - ensure backend modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from typing import Dict, List, Optional, Union, Any

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Technical Analysis imports
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
# threading removed - not used in current implementation
import queue
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory

from typing import Optional, List

# Add import for professional buy logic
from core.professional_buy_integration import ProfessionalBuyIntegration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_bot_india.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import unified ML interface
try:
    from ml_interface import get_ml_interface, get_unified_ml_analysis
    ML_INTERFACE_AVAILABLE = True
except ImportError:
    logger.warning("ML interface not available, using individual components")
    ML_INTERFACE_AVAILABLE = False


class FyersTickerMapper:
    """Fyers API-based ticker-to-company name mapping for Indian stock market"""

    def __init__(self, cache_file="data/fyers_ticker_mapping.pkl"):
        self.cache_file = cache_file
        self.ticker_mapping = {}
        self.last_updated = None
        self.update_frequency = timedelta(days=7)  # Update weekly
        self.fyers_client = None

        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Initialize Fyers client
        self._initialize_fyers()

        # Load cached data if available
        self.load_cache()

    def _initialize_fyers(self):
        """Initialize Fyers API client"""
        try:
            if not FYERS_AVAILABLE:
                logger.warning("Fyers API not available - falling back to cached data")
                return

            app_id = os.getenv("FYERS_APP_ID")
            access_token = os.getenv("FYERS_ACCESS_TOKEN")

            if app_id and access_token:
                self.fyers_client = fyersModel.FyersModel(
                    client_id=app_id,
                    token=access_token,
                    log_path=""
                )
                logger.info("Fyers API initialized successfully for ticker mapping")
            else:
                logger.warning("Fyers credentials not found - using cached data only")
        except Exception as e:
            logger.error(f"Error initializing Fyers for ticker mapping: {e}")

    def get_ticker_mapping(self, force_update=False):
        """Get ticker mapping with automatic updates"""
        if self.should_update() or force_update:
            logger.info("Updating ticker mapping from Fyers API...")
            self.update_mapping()
            self.save_cache()

        return self.ticker_mapping

    def should_update(self):
        """Check if mapping needs update"""
        if not self.last_updated:
            return True
        return datetime.now() - self.last_updated > self.update_frequency

    def fetch_fyers_stock_list(self):
        """Fetch all Indian stocks using Fyers API - more reliable than NSE scraping"""
        try:
            stock_mapping = {}

            if not self.fyers_client:
                logger.warning("Fyers client not available - cannot fetch stock list")
                return stock_mapping

            logger.info("Fetching comprehensive stock list from Fyers API...")

            # Get master data for NSE equity instruments
            try:
                # Fyers doesn't have master_data() method - skip this approach
                logger.info("Skipping Fyers master data - method not available")
                master_data = None

                if False:  # Disable this block since master_data() doesn't exist
                    instruments = master_data.get('d', [])
                    logger.info(f"Received {len(instruments)} instruments from Fyers")

                    # Filter for NSE equity instruments
                    nse_equities = [
                        inst for inst in instruments
                        if inst.get('exchange') == 'NSE' and
                        inst.get('segment') == 'EQ' and
                        inst.get('symbol_details', {}).get('symbol_type') == 'EQ'
                    ]

                    logger.info(f"Found {len(nse_equities)} NSE equity instruments")

                    for instrument in nse_equities:
                        try:
                            symbol = instrument.get('symbol_details', {}).get('symbol', '')
                            company_name = instrument.get('symbol_details', {}).get('long_name', '')

                            if symbol and company_name:
                                # Convert to Yahoo Finance format
                                ticker = f"{symbol}.NS"
                                variations = self.create_name_variations(company_name)
                                stock_mapping[ticker] = variations

                        except Exception as e:
                            logger.debug(f"Error processing instrument: {e}")
                            continue

                    logger.info(f"Successfully mapped {len(stock_mapping)} stocks from Fyers")

                else:
                    logger.error(f"Fyers master data request failed: {master_data}")

            except Exception as e:
                logger.error(f"Error fetching Fyers master data: {e}")
                logger.info("Fyers master data not available - using fallback approach")

                # Fallback: Try to get quotes for known major stocks to build partial mapping
                logger.info("Attempting fallback approach with major stock quotes...")
                major_stocks = [
                    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ",
                    "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ", "NSE:BHARTIARTL-EQ", "NSE:ITC-EQ",
                    "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:HCLTECH-EQ", "NSE:ASIANPAINT-EQ",
                    "NSE:MARUTI-EQ", "NSE:AXISBANK-EQ", "NSE:TITAN-EQ", "NSE:SUNPHARMA-EQ",
                    "NSE:ULTRACEMCO-EQ", "NSE:WIPRO-EQ", "NSE:NESTLEIND-EQ", "NSE:POWERGRID-EQ"
                ]

                try:
                    quotes_response = self.fyers_client.quotes({"symbols": ",".join(major_stocks)})
                    if quotes_response and quotes_response.get('s') == 'ok':
                        quotes_data = quotes_response.get('d', [])
                        for quote in quotes_data:
                            symbol_info = quote.get('n', '')
                            if symbol_info:
                                # Extract symbol from Fyers format
                                symbol = symbol_info.split(':')[1].split('-')[0] if ':' in symbol_info else ''
                                if symbol:
                                    ticker = f"{symbol}.NS"
                                    # Use symbol as company name for fallback
                                    variations = self.create_name_variations(symbol)
                                    stock_mapping[ticker] = variations

                        logger.info(f"Fallback approach yielded {len(stock_mapping)} stocks")
                except Exception as fallback_error:
                    logger.error(f"Fallback approach also failed: {fallback_error}")

            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_fyers_stock_list: {e}")
            return {}

    def fetch_yahoo_major_stocks(self):
        """Fetch major Indian stocks using Yahoo Finance as fallback"""
        try:
            stock_mapping = {}

            # Major Indian stocks that are reliably available on Yahoo Finance
            major_tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "HINDUNILVR.NS",
                "BAJFINANCE.NS", "AXISBANK.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJFINSV.NS",
                "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "ASIANPAINT.NS", "MARUTI.NS",
                "TITAN.NS", "NESTLEIND.NS", "BRITANNIA.NS", "LT.NS", "ULTRACEMCO.NS",
                "ONGC.NS", "BPCL.NS", "IOC.NS", "COALINDIA.NS", "NTPC.NS", "POWERGRID.NS",
                "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"
            ]

            logger.info(f"Fetching info for {len(major_tickers)} major stocks from Yahoo Finance...")

            for ticker in major_tickers:
                try:
                    # Get stock info from Yahoo Finance
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    if info and 'longName' in info:
                        company_name = info['longName']
                        variations = self.create_name_variations(company_name)
                        stock_mapping[ticker] = variations
                    else:
                        # Fallback to symbol-based name
                        symbol = ticker.replace('.NS', '')
                        variations = self.create_name_variations(symbol)
                        stock_mapping[ticker] = variations

                    # Small delay to be respectful to Yahoo Finance
                    time.sleep(0.1)

                except Exception as e:
                    logger.debug(f"Error fetching {ticker} from Yahoo Finance: {e}")
                    continue

            logger.info(f"Successfully fetched {len(stock_mapping)} stocks from Yahoo Finance")
            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_yahoo_major_stocks: {e}")
            return {}

    def fetch_bse_stock_list(self):
        """Fetch BSE stocks using multiple alternative methods"""
        try:
            stock_mapping = {}

            # Method 1: Try BSE official API with enhanced headers
            try:
                url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.bseindia.com/',
                    'Origin': 'https://www.bseindia.com',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-site'
                }

                logger.info(f"Trying BSE API: {url}")
                response = requests.get(url, headers=headers, timeout=15)
                logger.info(f"BSE API response status: {response.status_code}")
                logger.info(f"BSE API response headers: {dict(response.headers)}")

                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    logger.info(f"BSE API content type: {content_type}")

                    # Check if response is JSON
                    if 'application/json' in content_type or response.text.strip().startswith('{'):
                        try:
                            data = response.json()
                            logger.info(f"BSE API JSON parsed successfully. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

                            # Try different possible data structures
                            stocks_data = []
                            if isinstance(data, dict):
                                stocks_data = data.get('Table', data.get('data', data.get('stocks', [])))
                            elif isinstance(data, list):
                                stocks_data = data

                            logger.info(f"Found {len(stocks_data)} stocks in BSE response")

                            for stock in stocks_data:
                                if isinstance(stock, dict):
                                    # Try different field names
                                    scrip_cd = stock.get('Scrip_cd') or stock.get('scrip_cd') or stock.get('code') or stock.get('symbol')
                                    scrip_name = (stock.get('Scrip_Name') or stock.get('scrip_name') or
                                                stock.get('name') or stock.get('company_name') or '')

                                    if scrip_cd and scrip_name:
                                        ticker = f"{scrip_cd}.BO"
                                        variations = self.create_name_variations(scrip_name)
                                        stock_mapping[ticker] = variations

                            logger.info(f"Successfully processed {len(stock_mapping)} stocks from BSE API")

                        except ValueError as e:
                            logger.warning(f"BSE API JSON parsing failed: {e}")
                            logger.warning(f"Response text preview: {response.text[:300]}")
                    else:
                        logger.warning(f"BSE API returned non-JSON content: {response.text[:200]}")
                else:
                    logger.warning(f"BSE API failed with status {response.status_code}: {response.text[:200]}")

            except Exception as e:
                logger.warning(f"BSE API method 1 failed: {e}")

            # Method 2: Try alternative BSE endpoint if first method failed
            if len(stock_mapping) == 0:
                try:
                    alt_url = "https://www.bseindia.com/corporates/List_Scrips.html"
                    logger.info(f"Trying alternative BSE approach: {alt_url}")

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }

                    response = requests.get(alt_url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        logger.info("Alternative BSE endpoint accessible, but HTML parsing not implemented")
                        # Note: HTML parsing would require BeautifulSoup, keeping simple for now

                except Exception as e:
                    logger.warning(f"BSE alternative method failed: {e}")

            # Method 3: Use hardcoded BSE stocks if APIs fail
            if len(stock_mapping) == 0:
                logger.info("Using fallback BSE stocks")
                fallback_bse_stocks = {
                    "500325.BO": ["Reliance Industries", "Reliance", "RIL"],
                    "500209.BO": ["Infosys", "Infosys Technologies", "INFY"],
                    "500180.BO": ["HDFC Bank", "HDFC", "Housing Development Finance Corporation"],
                    "500034.BO": ["State Bank of India", "SBI", "State Bank"],
                    "500696.BO": ["Hindustan Unilever", "HUL", "Unilever"],
                    "500875.BO": ["ITC", "Indian Tobacco Company"],
                    "532540.BO": ["Tata Consultancy Services", "TCS", "Tata Consultancy"],
                    "500010.BO": ["HDFC", "Housing Development Finance Corporation"],
                    "500112.BO": ["State Bank of India", "SBI"],
                    "500820.BO": ["Asian Paints", "Asian Paint"]
                }
                stock_mapping.update(fallback_bse_stocks)
                logger.info(f"Added {len(fallback_bse_stocks)} fallback BSE stocks")

            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_bse_stock_list: {e}")
            return {}

    def create_name_variations(self, company_name):
        """Create multiple variations of company name for better search"""
        if not company_name:
            return []

        variations = [company_name.strip()]

        # Remove common suffixes
        suffixes_to_remove = [
            " Limited", " Ltd", " Ltd.", " Private Limited", " Pvt Ltd", " Pvt. Ltd.",
            " Corporation", " Corp", " Company", " Co.", " Inc", " Incorporated",
            " Private", " Pvt", " Public Limited", " Public Ltd"
        ]

        clean_name = company_name.strip()
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                if clean_name and clean_name not in variations:
                    variations.append(clean_name)

        # Generate dynamic abbreviations using intelligent pattern recognition
        name_lower = company_name.lower()

        # Dynamic abbreviation generation
        dynamic_abbreviations = self.generate_dynamic_abbreviations(company_name)
        variations.extend(dynamic_abbreviations)

        

        # Remove duplicates and empty strings
        variations = [v.strip() for v in variations if v.strip()]
        variations = list(dict.fromkeys(variations))  # Remove duplicates while preserving order

        return variations[:5]  # Limit to top 5 variations to avoid overly long queries

    def generate_dynamic_abbreviations(self, company_name):
        """Generate abbreviations dynamically using pattern recognition"""
        if not company_name:
            return []

        abbreviations = []
        name_lower = company_name.lower()
        words = company_name.split()

        # Method 1: First letter of each significant word
        significant_words = []
        skip_words = {'limited', 'ltd', 'private', 'pvt', 'corporation', 'corp',
                     'company', 'co', 'inc', 'incorporated', 'public', 'the', 'and', '&'}

        for word in words:
            if word.lower() not in skip_words and len(word) > 1:
                significant_words.append(word)

        if len(significant_words) >= 2:
            # Create acronym from first letters
            acronym = ''.join([word[0].upper() for word in significant_words])
            if len(acronym) >= 2 and len(acronym) <= 6:
                abbreviations.append(acronym)

        # Method 2: Common business patterns
        if 'bank' in name_lower:
            # For banks, often use first word + "Bank"
            first_word = words[0] if words else ""
            if first_word and first_word.lower() != 'bank':
                abbreviations.append(f"{first_word} Bank")

        if 'technologies' in name_lower or 'technology' in name_lower:
            # For tech companies, often use first word + "Tech"
            first_word = words[0] if words else ""
            if first_word and 'tech' not in first_word.lower():
                abbreviations.append(f"{first_word} Tech")

        if 'industries' in name_lower or 'industry' in name_lower:
            # For industrial companies, often just use first word
            first_word = words[0] if words else ""
            if first_word and first_word.lower() != 'industries':
                abbreviations.append(first_word)

        # Method 3: Group/Parent company patterns
        if 'group' in name_lower:
            # Extract group name
            group_word = None
            for i, word in enumerate(words):
                if word.lower() == 'group' and i > 0:
                    group_word = words[i-1]
                    break
            if group_word:
                abbreviations.append(f"{group_word} Group")

        # Method 4: Sector-specific patterns
        sector_patterns = {
            'energy': ['Energy', 'Power'],
            'power': ['Power', 'Energy'],
            'finance': ['Finance', 'Financial'],
            'consultancy': ['Consultancy', 'Consulting'],
            'services': ['Services'],
            'motors': ['Motors', 'Auto'],
            'automotive': ['Auto', 'Motors'],
            'cement': ['Cement'],
            'steel': ['Steel'],
            'pharma': ['Pharma', 'Pharmaceutical'],
            'pharmaceutical': ['Pharma'],
            'telecom': ['Telecom', 'Communications'],
            'communications': ['Telecom'],
            'oil': ['Oil', 'Petroleum'],
            'gas': ['Gas'],
            'ports': ['Ports', 'Port'],
            'shipping': ['Shipping'],
            'textiles': ['Textiles'],
            'chemicals': ['Chemicals', 'Chem'],
            'metals': ['Metals'],
            'mining': ['Mining'],
            'real estate': ['Realty', 'Real Estate'],
            'realty': ['Real Estate'],
            'insurance': ['Insurance'],
            'mutual fund': ['MF', 'Mutual Fund'],
            'asset management': ['AMC', 'Asset Management']
        }

        for sector, variations in sector_patterns.items():
            if sector in name_lower:
                # Add sector-specific variations
                first_word = words[0] if words else ""
                if first_word:
                    for variation in variations:
                        abbreviations.append(f"{first_word} {variation}")

        # Method 5: Handle special characters and conjunctions
        if '&' in company_name or ' and ' in name_lower:
            # For companies with & or "and", create variations
            parts = company_name.replace('&', 'and').split(' and ')
            if len(parts) == 2:
                # Create "FirstWord & SecondWord" format
                first_part = parts[0].strip().split()[0] if parts[0].strip() else ""
                second_part = parts[1].strip().split()[0] if parts[1].strip() else ""
                if first_part and second_part:
                    abbreviations.append(f"{first_part} & {second_part}")
                    abbreviations.append(f"{first_part[0]}&{second_part[0]}")

        # Method 6: Remove common words and create short forms
        clean_words = []
        for word in words:
            if (word.lower() not in skip_words and
                len(word) > 2 and
                not word.lower().endswith('ing')):
                clean_words.append(word)

        if len(clean_words) >= 1:
            # Just the main company name without suffixes
            main_name = ' '.join(clean_words[:2])  # Take first 2 significant words
            if main_name != company_name:
                abbreviations.append(main_name)

        return abbreviations

    def test_api_connectivity(self):
        """Test connectivity to NSE and BSE APIs"""
        results = {
            'nse_main': False,
            'nse_market_data': False,
            'bse_api': False,
            'errors': []
        }

        # Test NSE main page
        try:
            response = requests.get("https://www.nseindia.com", timeout=10)
            results['nse_main'] = response.status_code == 200
            if not results['nse_main']:
                results['errors'].append(f"NSE main page returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"NSE main page error: {e}")

        # Test NSE market data page
        try:
            response = requests.get("https://www.nseindia.com/market-data", timeout=10)
            results['nse_market_data'] = response.status_code == 200
            if not results['nse_market_data']:
                results['errors'].append(f"NSE market data page returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"NSE market data error: {e}")

        # Test BSE API
        try:
            response = requests.get("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w", timeout=10)
            results['bse_api'] = response.status_code == 200
            if not results['bse_api']:
                results['errors'].append(f"BSE API returned {response.status_code}")
        except Exception as e:
            results['errors'].append(f"BSE API error: {e}")

        logger.info(f"API Connectivity Test Results: {results}")
        return results

    def update_mapping(self):
        """Update mapping using Fyers API as primary source with fallbacks"""
        all_mappings = {}

        # Method 1: Try Fyers API (Primary - Most Reliable)
        try:
            logger.info("Attempting to fetch stocks from Fyers API...")
            fyers_data = self.fetch_fyers_stock_list()
            all_mappings.update(fyers_data)
            logger.info(f"[SUCCESS] Successfully fetched {len(fyers_data)} stocks from Fyers API")
        except Exception as e:
            logger.warning(f"[ERROR] Fyers API failed: {e}")

        # Method 2: Try BSE API (Secondary)
        try:
            logger.info("Attempting BSE API as secondary source...")
            bse_data = self.fetch_bse_stock_list()
            # Merge with existing data
            for ticker, names in bse_data.items():
                if ticker in all_mappings:
                    # Combine name variations
                    all_mappings[ticker].extend(names)
                    all_mappings[ticker] = list(set(all_mappings[ticker]))
                else:
                    all_mappings[ticker] = names
            logger.info(f"[SUCCESS] Successfully added {len(bse_data)} stocks from BSE API")
        except Exception as e:
            logger.warning(f"[ERROR] BSE API failed: {e}")

        # Method 3: Yahoo Finance fallback for major stocks
        if len(all_mappings) < 100:
            logger.info("Adding Yahoo Finance data for major stocks...")
            try:
                yahoo_data = self.fetch_yahoo_major_stocks()
                for ticker, names in yahoo_data.items():
                    if ticker not in all_mappings:
                        all_mappings[ticker] = names
                logger.info(f"[SUCCESS] Added {len(yahoo_data)} stocks from Yahoo Finance")
            except Exception as e:
                logger.warning(f"[ERROR] Yahoo Finance fallback failed: {e}")

        # Method 4: Hardcoded fallback if still insufficient data
        if len(all_mappings) < 50:
            logger.warning(f"Only {len(all_mappings)} stocks fetched from APIs. Using fallback hardcoded mapping.")
            fallback_mapping = self.get_fallback_mapping()
            for ticker, names in fallback_mapping.items():
                if ticker not in all_mappings:
                    all_mappings[ticker] = names
            logger.info(f"[SUCCESS] Added {len(fallback_mapping)} fallback stocks")
        else:
            logger.info(f"[SUCCESS] Sufficient stocks ({len(all_mappings)}) fetched from APIs")

        self.ticker_mapping = all_mappings
        self.last_updated = datetime.now()

        logger.info(f"Final mapping contains {len(self.ticker_mapping)} stocks")

        # Log sample of stocks for verification
        sample_stocks = list(self.ticker_mapping.keys())[:5]
        logger.info(f"[DATA] Sample stocks: {sample_stocks}")

    def get_fallback_mapping(self):
        """Fallback hardcoded mapping for essential stocks - expanded to 50+ stocks"""
        return {
            # Top 10 by market cap
            "RELIANCE.NS": ["Reliance Industries", "RIL", "Reliance"],
            "TCS.NS": ["Tata Consultancy Services", "TCS"],
            "HDFCBANK.NS": ["HDFC Bank", "Housing Development Finance Corporation"],
            "INFY.NS": ["Infosys", "Infosys Limited"],
            "ICICIBANK.NS": ["ICICI Bank", "Industrial Credit and Investment Corporation"],
            "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank"],
            "BHARTIARTL.NS": ["Bharti Airtel", "Airtel"],
            "ITC.NS": ["ITC Limited", "Indian Tobacco Company"],
            "SBIN.NS": ["State Bank of India", "SBI"],
            "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],

            # Banking & Financial Services
            "BAJFINANCE.NS": ["Bajaj Finance", "Bajaj Finserv"],
            "AXISBANK.NS": ["Axis Bank"],
            "HDFCLIFE.NS": ["HDFC Life Insurance", "HDFC Life"],
            "SBILIFE.NS": ["SBI Life Insurance", "SBI Life"],
            "BAJAJFINSV.NS": ["Bajaj Finserv"],
            "INDUSINDBK.NS": ["IndusInd Bank"],
            "BANDHANBNK.NS": ["Bandhan Bank"],
            "IDFCFIRSTB.NS": ["IDFC First Bank"],

            # IT & Technology
            "HCLTECH.NS": ["HCL Technologies", "HCL Tech"],
            "WIPRO.NS": ["Wipro Limited", "Wipro"],
            "TECHM.NS": ["Tech Mahindra"],
            "LTIM.NS": ["LTIMindtree", "L&T Infotech"],
            "MPHASIS.NS": ["Mphasis"],

            # Consumer Goods
            "ASIANPAINT.NS": ["Asian Paints", "Asian Paint"],
            "MARUTI.NS": ["Maruti Suzuki", "Maruti", "Suzuki India"],
            "TITAN.NS": ["Titan Company", "Titan Industries"],
            "NESTLEIND.NS": ["Nestle India", "Nestle"],
            "BRITANNIA.NS": ["Britannia Industries", "Britannia"],
            "DABUR.NS": ["Dabur India", "Dabur"],
            "MARICO.NS": ["Marico"],
            "GODREJCP.NS": ["Godrej Consumer Products", "Godrej"],

            # Infrastructure & Construction
            "LT.NS": ["Larsen & Toubro", "L&T", "Larsen Toubro"],
            "ULTRACEMCO.NS": ["UltraTech Cement", "Ultratech"],
            "GRASIM.NS": ["Grasim Industries", "Grasim"],
            "SHREECEM.NS": ["Shree Cement"],
            "RAMCOCEM.NS": ["Ramco Cements", "Ramco"],

            # Energy & Oil
            "ONGC.NS": ["Oil and Natural Gas Corporation", "ONGC"],
            "BPCL.NS": ["Bharat Petroleum", "BPCL"],
            "IOC.NS": ["Indian Oil Corporation", "IOC"],
            "COALINDIA.NS": ["Coal India", "CIL"],
            "NTPC.NS": ["NTPC"],
            "POWERGRID.NS": ["Power Grid Corporation", "PowerGrid"],

            # Adani Group
            "ADANIENSOL.NS": ["Adani Energy Solutions", "Adani Green Energy", "Adani Power", "Adani Energy"],
            "ADANIPORTS.NS": ["Adani Ports", "Adani Port"],
            "ADANITRANS.NS": ["Adani Transmission"],
            "ATGL.NS": ["Adani Total Gas"],
            "ADANIENT.NS": ["Adani Enterprises"],

            # Pharmaceuticals
            "SUNPHARMA.NS": ["Sun Pharmaceutical", "Sun Pharma"],
            "DRREDDY.NS": ["Dr. Reddy's Laboratories", "Dr Reddy"],
            "CIPLA.NS": ["Cipla"],
            "DIVISLAB.NS": ["Divi's Laboratories", "Divis Lab"],
            "BIOCON.NS": ["Biocon"],
            "LUPIN.NS": ["Lupin"],

            # Metals & Mining
            "TATASTEEL.NS": ["Tata Steel"],
            "HINDALCO.NS": ["Hindalco Industries", "Hindalco"],
            "JSWSTEEL.NS": ["JSW Steel"],
            "SAIL.NS": ["Steel Authority of India", "SAIL"],
            "VEDL.NS": ["Vedanta"],

            # Auto & Auto Components
            "BAJAJ-AUTO.NS": ["Bajaj Auto"],
            "M&M.NS": ["Mahindra & Mahindra", "Mahindra"],
            "TATAMOTORS.NS": ["Tata Motors"],
            "EICHERMOT.NS": ["Eicher Motors"],
            "HEROMOTOCO.NS": ["Hero MotoCorp", "Hero"],

            # Telecom
            "JIOFINANCE.NS": ["Jio Financial Services", "Jio Finance"],

            # Others
            "APOLLOHOSP.NS": ["Apollo Hospitals", "Apollo"],
            "PIDILITIND.NS": ["Pidilite Industries", "Pidilite"],
            "BERGEPAINT.NS": ["Berger Paints", "Berger"],
            "HAVELLS.NS": ["Havells India", "Havells"],
        }

    def save_cache(self):
        """Save mapping to cache file"""
        try:
            cache_data = {
                'mapping': self.ticker_mapping,
                'last_updated': self.last_updated
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved ticker mapping cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def load_cache(self):
        """Load mapping from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.ticker_mapping = cache_data.get('mapping', {})
                self.last_updated = cache_data.get('last_updated')
                logger.info(f"Loaded {len(self.ticker_mapping)} stocks from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.ticker_mapping = {}
            self.last_updated = None

    def get_company_names(self, ticker):
        """Get company names for a specific ticker"""
        mapping = self.get_ticker_mapping()
        return mapping.get(ticker, [ticker.replace(".NS", "").replace(".BO", "")])

    def search_ticker_by_name(self, company_name):
        """Reverse lookup: find ticker by company name"""
        mapping = self.get_ticker_mapping()
        company_name_lower = company_name.lower()

        for ticker, names in mapping.items():
            for name in names:
                if company_name_lower in name.lower() or name.lower() in company_name_lower:
                    return ticker
        return None


# Custom Llama LLM Integration via Ollama
class LlamaLLM(LLM):
    """Custom LLM wrapper for Ollama Llama integration."""

    # Define Pydantic fields properly
    model_name: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"

    def __init__(self, model_name: str = "llama3.2:latest", **kwargs):
        # Initialize with proper Pydantic field handling
        super().__init__(
            model_name=model_name,
            base_url="http://localhost:11434",
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "llama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call Ollama API with the given prompt."""
        try:
            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Sorry, I couldn't generate a response.")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Sorry, I'm having trouble connecting to the local AI service."

        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout")
            return "Sorry, the local AI service is taking too long to respond."
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error: {e}")
            return "Sorry, I'm having trouble connecting to the local AI service. Make sure Ollama is running."
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API call: {e}")
            return "Sorry, I encountered an unexpected error."

# Chat Interaction Logger
class ChatLogger:
    """Handles logging of chat interactions to JSON file."""

    def __init__(self):
        # Use absolute path to project root data directory
        import os
        from pathlib import Path
        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent
        data_dir = project_root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = str(data_dir / 'chat_interactions.json')
        self.interactions = self.load_interactions()
    def load_interactions(self):
        """Load existing chat interactions from file."""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading chat interactions: {e}")
            return []

    def log_interaction(self, user_input: str, bot_response: str, command_type: str = None):
        """Log a chat interaction."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "command_type": command_type,
            "session_id": f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        self.interactions.append(interaction)
        self.save_interactions()

    def save_interactions(self):
        """Save interactions to JSON file."""
        try:
            with open(self.log_file, "w", encoding='utf-8') as f:
                json.dump(self.interactions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving chat interactions: {e}")

# Chatbot Command Handler
class ChatbotCommandHandler:
    """Handles chatbot commands and integrates with trading bot."""

    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.chat_logger = ChatLogger()
        self.llama_llm = None
        self.conversation_memory = ConversationBufferMemory()
        self.trading_paused = False
        self.pause_until = None

        # Initialize Llama LLM (always available with Ollama)
        try:
            self.llama_llm = LlamaLLM(model_name="llama3.2:latest")
            logger.info("Llama LLM initialized successfully with Ollama")
        except Exception as e:
            logger.error(f"Failed to initialize Llama LLM: {e}")
            self.llama_llm = None

    def process_command(self, user_input: str) -> str:
        """Process user command and return response."""
        try:
            user_input = user_input.strip()

            # Check if it's a command (starts with /)
            if user_input.startswith('/'):
                response = self.handle_command(user_input)
                command_type = user_input.split()[0]
            else:
                # Use Llama LLM for general conversation
                response = self.handle_general_query(user_input)
                command_type = "general"

            # Log the interaction
            self.chat_logger.log_interaction(user_input, response, command_type)
            return response

        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def handle_command(self, command: str) -> str:
        """Handle specific bot commands."""
        parts = command.split()
        cmd = parts[0].lower()

        try:
            if cmd == "/start_bot":
                return self.start_bot()
            elif cmd == "/set_risk":
                risk_level = parts[1].upper() if len(parts) > 1 else "MEDIUM"
                return self.set_risk(risk_level)
            elif cmd == "/get_pnl":
                return self.get_pnl()
            elif cmd == "/why_trade":
                ticker = parts[1] if len(parts) > 1 else None
                return self.why_trade(ticker)
            elif cmd == "/list_positions":
                return self.list_positions()
            elif cmd == "/set_ticker":
                ticker = parts[1] if len(parts) > 1 else None
                action = parts[2].upper() if len(parts) > 2 else "ADD"
                return self.set_ticker(ticker, action)
            elif cmd == "/get_signals":
                ticker = parts[1] if len(parts) > 1 else None
                return self.get_signals(ticker)
            elif cmd == "/pause_trading":
                minutes = int(parts[1]) if len(parts) > 1 else 30
                return self.pause_trading(minutes)
            elif cmd == "/resume_trading":
                return self.resume_trading()
            elif cmd == "/get_performance":
                period = parts[1] if len(parts) > 1 else "1d"
                return self.get_performance(period)
            elif cmd == "/set_allocation":
                percentage = float(parts[1]) if len(parts) > 1 else 25.0
                return self.set_allocation(percentage)
            else:
                return f"Unknown command: {cmd}. Type /help for available commands."

        except Exception as e:
            logger.error(f"Error handling command {cmd}: {e}")
            return f"Error executing command {cmd}: {str(e)}"

    def handle_general_query(self, query: str) -> str:
        """Handle general queries using Llama LLM."""
        if not self.llama_llm:
            return "AI chat is not available. Please make sure Ollama is running locally."

        try:
            # Add context about the trading bot
            context = f"""You are an AI assistant for an Indian stock trading bot. Current context:
- Portfolio Value: Rs.{self.trading_bot.portfolio.get_metrics()['total_value']:,.2f}
- Active Positions: {len(self.trading_bot.portfolio.holdings)}
- Trading Mode: {self.trading_bot.portfolio.mode}
- Available Commands: /start_bot, /set_risk, /get_pnl, /why_trade, /list_positions, /set_ticker, /get_signals, /pause_trading, /resume_trading, /get_performance, /set_allocation

User Query: {query}

Please provide a helpful response about trading, markets, or the user's portfolio. Keep responses concise and relevant."""

            # Use direct LLM call instead of ConversationChain
            response = self.llama_llm._call(context)
            return response

        except Exception as e:
            logger.error(f"Error with Llama LLM: {e}")
            return "Sorry, I'm having trouble with the local AI service. Please make sure Ollama is running."

    # Command Implementation Methods
    def start_bot(self) -> str:
        """Start the trading bot."""
        try:
            logger.info("Bot start command received via chat interface")
            return "[SUCCESS] Trading bot is active and monitoring markets. Check logs for detailed activity."
        except Exception as e:
            return f"[ERROR] Error starting bot: {str(e)}"

    def set_risk(self, risk_level: str) -> str:
        """Set risk level and update stop-loss."""
        try:
            risk_mapping = {
                "LOW": 0.03,    # 3% stop-loss
                "MEDIUM": 0.05, # 5% stop-loss
                "HIGH": 0.08    # 8% stop-loss
            }

            if risk_level not in risk_mapping:
                return f"[ERROR] Invalid risk level. Use: LOW, MEDIUM, or HIGH"

            new_stop_loss = risk_mapping[risk_level]
            self.trading_bot.config["stop_loss_pct"] = new_stop_loss
            self.trading_bot.executor.stop_loss_pct = new_stop_loss

            return f"[SUCCESS] Risk level set to {risk_level}. Stop-loss updated to {new_stop_loss*100}%"
        except Exception as e:
            return f"[ERROR] Error setting risk level: {str(e)}"

    def get_pnl(self) -> str:
        """Get current portfolio P&L metrics."""
        try:
            metrics = self.trading_bot.portfolio.get_metrics()
            starting_balance = self.trading_bot.portfolio.starting_balance
            total_return = metrics['total_value'] - starting_balance
            return_pct = (total_return / starting_balance) * 100

            pnl_report = f"""
[DATA] **Portfolio Metrics**
Rs. Total Value: Rs.{metrics['total_value']:,.2f}
Cash: Cash: Rs.{metrics['cash']:,.2f}
[UP] Holdings Value: Rs.{metrics['total_exposure']:,.2f}
Target: Total Return: Rs.{total_return:,.2f} ({return_pct:+.2f}%)
[SUCCESS] Realized P&L: Rs.{metrics['realized_pnl']:,.2f}
[DATA] Unrealized P&L: Rs.{metrics['unrealized_pnl']:,.2f}
Positions: Active Positions: {len(self.trading_bot.portfolio.holdings)}
            """
            return pnl_report.strip()
        except Exception as e:
            return f"[ERROR] Error getting P&L: {str(e)}"

    def why_trade(self, ticker: str) -> str:
        """Explain why a particular ticker should be traded."""
        try:
            if not ticker:
                return "[ERROR] Please specify a ticker. Example: /why_trade RELIANCE.NS"

            # Get analysis for the ticker
            analysis = self.trading_bot.run_analysis(ticker)
            if not analysis.get("success"):
                return f"[ERROR] Could not analyze {ticker}: {analysis.get('message', 'Unknown error')}"

            # Extract key metrics
            technical = analysis["technical_indicators"]
            current_price = analysis["stock_data"]["current_price"]["INR"]

            # Determine recommendation based on signals
            buy_signals = 0
            sell_signals = 0

            # RSI analysis
            rsi = technical.get("RSI", 50)
            if rsi < 30:
                buy_signals += 1
                rsi_signal = "oversold (bullish)"
            elif rsi > 70:
                sell_signals += 1
                rsi_signal = "overbought (bearish)"
            else:
                rsi_signal = "neutral"

            # Moving average analysis
            sma_20 = technical.get("SMA_20", current_price)
            if current_price > sma_20:
                buy_signals += 1
                ma_signal = "above SMA (bullish)"
            else:
                sell_signals += 1
                ma_signal = "below SMA (bearish)"

            # MACD analysis
            macd = technical.get("MACD", 0)
            macd_signal_line = technical.get("MACD_Signal", 0)
            if macd > macd_signal_line:
                buy_signals += 1
                macd_signal = "bullish crossover"
            else:
                sell_signals += 1
                macd_signal = "bearish crossover"

            # Overall recommendation
            if buy_signals > sell_signals:
                recommendation = "[+] BUY"
                reason = "Technical indicators suggest upward momentum"
            elif sell_signals > buy_signals:
                recommendation = "[-] SELL"
                reason = "Technical indicators suggest downward pressure"
            else:
                recommendation = "[=] HOLD"
                reason = "Mixed signals, wait for clearer direction"

            analysis_text = f"""
Target: **Trade Analysis for {ticker}**
Rs. Current Price: Rs.{current_price:.2f}
[DATA] Recommendation: {recommendation}

**Technical Signals:**
- RSI ({rsi:.1f}): {rsi_signal}
- Moving Average: {ma_signal}
- MACD: {macd_signal}

**Reasoning:** {reason}

WARNING: This is for educational purposes only. Always do your own research before trading.
"""
            return analysis_text.strip()

        except Exception as e:
            return f"[ERROR] Error analyzing trade: {str(e)}"

    def list_positions(self) -> str:
        """List all open positions."""
        try:
            holdings = self.trading_bot.portfolio.holdings
            if not holdings:
                return "No open positions currently."

            positions_text = "[DATA] **Current Positions:**\n"
            for ticker, data in holdings.items():
                current_value = data['qty'] * data['avg_price']
                positions_text += f"- {ticker}: {data['qty']} shares @ Rs.{data['avg_price']:.2f} (Rs.{current_value:,.2f})\n"

            return positions_text.strip()
        except Exception as e:
            return f"[ERROR] Error listing positions: {str(e)}"

    def set_ticker(self, ticker: str, action: str) -> str:
        """Add or remove ticker from watchlist."""
        try:
            if not ticker:
                return "[ERROR] Please specify a ticker. Example: /set_ticker RELIANCE.NS ADD"

            current_tickers = self.trading_bot.config["tickers"]

            if action == "ADD":
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    return f"[SUCCESS] Added {ticker} to watchlist. Total tickers: {len(current_tickers)}"
                else:
                    return f"INFO: {ticker} is already in watchlist."

            elif action == "REMOVE":
                if ticker in current_tickers:
                    current_tickers.remove(ticker)
                    return f"[SUCCESS] Removed {ticker} from watchlist. Total tickers: {len(current_tickers)}"
                else:
                    return f"INFO: {ticker} is not in watchlist."

            else:
                return "[ERROR] Invalid action. Use ADD or REMOVE."

        except Exception as e:
            return f"[ERROR] Error managing ticker: {str(e)}"

    def get_signals(self, ticker: str) -> str:
        """Get current trading signals for a ticker."""
        try:
            if not ticker:
                return "[ERROR] Please specify a ticker. Example: /get_signals TCS.NS"

            analysis = self.trading_bot.run_analysis(ticker)
            if not analysis.get("success"):
                return f"[ERROR] Could not get signals for {ticker}: {analysis.get('message', 'Unknown error')}"

            technical = analysis["technical_indicators"]
            current_price = analysis["stock_data"]["current_price"]["INR"]

            signals_text = f"""
Target: **Trading Signals for {ticker}**
Rs. Current Price: Rs.{current_price:.2f}

[DATA] **Technical Signals:**
- RSI: {technical['rsi']:.2f} {'[-] Overbought' if technical['rsi'] > 70 else '[+] Oversold' if technical['rsi'] < 30 else '[=] Neutral'}
- MACD: {technical['macd']:.4f} {'[+] Bullish' if technical['macd'] > 0 else '[-] Bearish'}
- SMA Trend: {'[+] Bullish' if technical['sma_50'] > technical['sma_200'] else '[-] Bearish'}
- Bollinger Bands: {'[-] Overbought' if current_price > technical['bb_upper'] else '[+] Oversold' if current_price < technical['bb_lower'] else '[=] Normal'}

[UP] **Overall Signal:** {analysis['recommendation']}
            """
            return signals_text.strip()
        except Exception as e:
            return f"[ERROR] Error getting signals: {str(e)}"

    def pause_trading(self, minutes: int) -> str:
        """Pause trading for specified minutes."""
        try:
            self.trading_paused = True
            self.pause_until = datetime.now() + timedelta(minutes=minutes)
            return f"PAUSED: Trading paused for {minutes} minutes until {self.pause_until.strftime('%H:%M:%S')}"
        except Exception as e:
            return f"[ERROR] Error pausing trading: {str(e)}"

    def resume_trading(self) -> str:
        """Resume trading."""
        try:
            self.trading_paused = False
            self.pause_until = None
            return "RESUMED: Trading resumed successfully."
        except Exception as e:
            return f"[ERROR] Error resuming trading: {str(e)}"

    def get_performance(self, period: str) -> str:
        """Get performance report for specified period."""
        try:
            metrics = self.trading_bot.portfolio.get_metrics()
            starting_balance = self.trading_bot.portfolio.starting_balance
            current_cash = metrics['cash']

            # Calculate performance metrics based on cash invested
            cash_invested = starting_balance - current_cash
            holdings_value = metrics['total_value'] - current_cash
            total_return = holdings_value - cash_invested
            return_pct = (total_return / cash_invested) * 100 if cash_invested > 0 else 0

            # Get trade statistics
            trades = self.trading_bot.portfolio.trade_log
            recent_trades = []

            # Filter trades based on period
            now = datetime.now()
            if period == "1d":
                cutoff = now - timedelta(days=1)
            elif period == "1w":
                cutoff = now - timedelta(weeks=1)
            elif period == "1m":
                cutoff = now - timedelta(days=30)
            else:
                cutoff = now - timedelta(days=1)

            for trade in trades:
                try:
                    trade_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
                    if trade_time >= cutoff:
                        recent_trades.append(trade)
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error parsing trade timestamp: {e}")
                    continue

            performance_text = f"""
[UP] **Performance Report ({period})**
Rs. Current Value: Rs.{metrics['total_value']:,.2f}
Target: Total Return: Rs.{total_return:,.2f} ({return_pct:+.2f}%)
[DATA] Realized P&L: Rs.{metrics['realized_pnl']:,.2f}
[UP] Unrealized P&L: Rs.{metrics['unrealized_pnl']:,.2f}
Trades: Recent Trades: {len(recent_trades)}
Positions: Active Positions: {len(self.trading_bot.portfolio.holdings)}
            """
            return performance_text.strip()
        except Exception as e:
            return f"[ERROR] Error getting performance: {str(e)}"

    def set_allocation(self, percentage: float) -> str:
        """Set maximum capital allocation per trade."""
        try:
            if percentage <= 0 or percentage > 100:
                return "[ERROR] Allocation must be between 0 and 100%"

            allocation_decimal = percentage / 100
            self.trading_bot.config["max_capital_per_trade"] = allocation_decimal
            self.trading_bot.executor.max_capital_per_trade = allocation_decimal

            return f"[SUCCESS] Maximum allocation per trade set to {percentage}%"
        except Exception as e:
            return f"[ERROR] Error setting allocation: {str(e)}"

def get_fyers_client():
    """Get Fyers client for real-time data - same as web_backend"""
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        app_id = os.getenv("FYERS_APP_ID")
        access_token = os.getenv("FYERS_ACCESS_TOKEN")

        if not app_id or not access_token or not FYERS_AVAILABLE:
            return None

        fyers = fyersModel.FyersModel(
            client_id=app_id,
            is_async=False,
            token=access_token,
            log_path=""
        )
        return fyers
    except Exception as e:
        logger.warning(f"Fyers client creation failed: {e}")
        return None

def fyers_to_yfinance_format(fyers_data, ticker):
    """Convert Fyers data to yfinance-like DataFrame format"""
    try:
        if not fyers_data or fyers_data.get("s") != "ok" or not fyers_data.get("d"):
            return None

        data = fyers_data["d"][0]["v"]

        # Create a simple DataFrame with current price as Close
        import pandas as pd
        from datetime import datetime

        df = pd.DataFrame({
            'Open': [data.get('open_price', data.get('lp', 0))],
            'High': [data.get('high_price', data.get('lp', 0))],
            'Low': [data.get('low_price', data.get('lp', 0))],
            'Close': [data.get('lp', 0)],
            'Volume': [data.get('volume', 0)]
        }, index=[datetime.now()])

        return df
    except Exception as e:
        logger.warning(f"Error converting Fyers data: {e}")
        return None

def get_stock_data_fyers_or_yf(ticker, period="1d"):
    """Get stock data from Fyers first, fallback to yfinance - SAME LOGIC"""
    # Validate ticker format and handle special cases
    if ticker.startswith('$'):
        logger.error(f"Invalid ticker symbol format: {ticker} - symbols should not start with '$'")
        # Try to correct common misformatted symbols
        corrected_ticker = ticker[1:] + '.NS'  # Remove $ and add .NS suffix
        logger.info(f"Attempting to correct ticker to: {corrected_ticker}")
        ticker = corrected_ticker
    elif '.' not in ticker and not ticker.isdigit():
        # If no exchange suffix and not a numeric security ID, assume NSE
        ticker = ticker + '.NS'
    
    # Try Fyers first
    fyers_client = get_fyers_client()
    if fyers_client:
        try:
            # Convert ticker format for Fyers - handle both NSE and BSE
            if ticker.endswith('.NS'):
                fyers_symbol = f"NSE:{ticker.replace('.NS', '')}-EQ"
            elif ticker.endswith('.BO'):
                fyers_symbol = f"BSE:{ticker.replace('.BO', '')}-EQ"
            else:
                # Default to NSE if no exchange specified
                fyers_symbol = f"NSE:{ticker}-EQ"

            if period == "1d":
                # For current day data, use quotes
                quotes = fyers_client.quotes({"symbols": fyers_symbol})
                if quotes and quotes.get("s") == "ok":
                    df = fyers_to_yfinance_format(quotes, ticker)
                    if df is not None and not df.empty:
                        logger.debug(f"Using Fyers current data for {ticker}")
                        return df
            else:
                # For historical data, use Fyers historical API
                df = get_fyers_historical_data(fyers_client, fyers_symbol, ticker, period)
                if df is not None and not df.empty:
                    logger.debug(f"Using Fyers historical data for {ticker} ({period})")
                    return df
        except Exception as e:
            logger.warning(f"Fyers failed for {ticker}: {e}")

    # Fallback to yfinance - EXACT SAME LOGIC
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if not df.empty:
            logger.info(f"Using Yahoo Finance data for {ticker} ({period})")
            return df
        else:
            logger.error(f"No data found for {ticker} (period={period}) (Yahoo error = \"No data found, symbol may be delisted\")")
            # Try to suggest a corrected symbol if possible
            if ticker.startswith('$'):
                suggested_ticker = ticker[1:] + '.NS'
                logger.info(f"Suggested correction: Try '{suggested_ticker}' instead of '{ticker}'")
    except Exception as e:
        logger.warning(f"Yahoo Finance failed for {ticker}: {e}")

    return None

def get_fyers_historical_data(fyers_client, fyers_symbol, ticker, period):
    """Get historical data from Fyers API"""
    try:
        from datetime import datetime, timedelta
        import pandas as pd

        # Convert period to date range
        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=730)
            resolution = "D"  # Daily
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
            resolution = "D"
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
            resolution = "D"
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
            resolution = "D"
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
            resolution = "D"
        else:
            start_date = end_date - timedelta(days=1)
            resolution = "1"  # 1 minute

        # Format dates for Fyers API
        range_from = start_date.strftime("%Y-%m-%d")
        range_to = end_date.strftime("%Y-%m-%d")

        # Get historical data from Fyers
        historical_data = fyers_client.history({
            "symbol": fyers_symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        })

        if historical_data and historical_data.get("s") == "ok" and historical_data.get("candles"):
            # Convert Fyers historical data to yfinance format
            candles = historical_data["candles"]

            # Fyers candles format: [timestamp, open, high, low, close, volume]
            data = []
            for candle in candles:
                data.append({
                    'Open': candle[1],
                    'High': candle[2],
                    'Low': candle[3],
                    'Close': candle[4],
                    'Volume': candle[5]
                })

            # Create DataFrame with proper datetime index
            df = pd.DataFrame(data)
            df.index = pd.to_datetime([candle[0] for candle in candles], unit='s')
            df.index.name = 'Datetime'

            return df

    except Exception as e:
        logger.warning(f"Error getting Fyers historical data for {ticker}: {e}")

    return None

class DataFeed:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                # Use Fyers first, fallback to yfinance - SAME LOGIC
                df = get_stock_data_fyers_or_yf(ticker)
                if not df.empty:
                    latest = df.iloc[-1]
                    data[ticker] = {
                        "price": latest["Close"],
                        "volume": latest["Volume"]
                    }
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data
        }

class VirtualPortfolio:
    def __init__(self, config):
        self.starting_balance = config["starting_balance"]
        self.cash = self.starting_balance
        self.holdings = {}  # {asset: {qty, avg_price}}
        self.trade_log = []
        self.mode = config.get("mode", "paper")
        self.realized_pnl = 0  # P&L from completed trades
        self.unrealized_pnl = 0  # P&L from current holdings
        self.trade_callbacks = []  # Callbacks to notify when trades are executed

        # Initialize Dhan API only if we have credentials
        if config.get("dhan_client_id") and config.get("dhan_access_token"):
            try:
                self.api = dhanhq(
                    client_id=config["dhan_client_id"],
                    access_token=config["dhan_access_token"]
                )
                logger.info("Dhan API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Dhan API: {e}")
                self.api = None
        else:
            self.api = None
            logger.warning("Dhan API credentials not provided. Running in simulation mode.")

        self.config = config

        # Set file paths based on mode - use absolute path to data folder
        import os
        # Get the project root directory (parent of backend folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, "data")

        if self.mode == "live":
            self.portfolio_file = os.path.join(data_dir, "portfolio_india_live.json")
            self.trade_log_file = os.path.join(data_dir, "trade_log_india_live.json")
        else:
            self.portfolio_file = os.path.join(data_dir, "portfolio_india_paper.json")
            self.trade_log_file = os.path.join(data_dir, "trade_log_india_paper.json")

        self.initialize_files()
        self.load_portfolio_data()

    def load_portfolio_data(self):
        """Load existing portfolio data from files."""
        try:
            # Load portfolio data
            if os.path.exists(self.portfolio_file):
                with open(self.portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                    self.cash = portfolio_data.get('cash', self.starting_balance)
                    self.holdings = portfolio_data.get('holdings', {})
                    self.realized_pnl = portfolio_data.get('realized_pnl', 0)
                    self.unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
                    # Update starting_balance if it exists in file
                    if 'starting_balance' in portfolio_data:
                        self.starting_balance = portfolio_data['starting_balance']
                    logger.info(f"Loaded portfolio: Cash=Rs.{self.cash:.2f}, Holdings={len(self.holdings)} positions")

            # Load trade log data
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'r') as f:
                    self.trade_log = json.load(f)
                    logger.info(f"Loaded {len(self.trade_log)} trades from trade log")
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            # Keep default values if loading fails

    def initialize_files(self):
        """Initialize portfolio and trade log JSON files if they don't exist."""
        # Use absolute paths to project root directories
        import os
        from pathlib import Path
        
        # Get project root directory (parent of backend folder)
        backend_dir = Path(__file__).resolve().parent
        project_root = backend_dir.parent
        data_dir = project_root / 'data'
        logs_dir = project_root / 'logs'
        
        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Update file paths to use the correct data directory
        if self.mode == "live":
            self.portfolio_file = str(data_dir / "portfolio_india_live.json")
            self.trade_log_file = str(data_dir / "trade_log_india_live.json")
        else:
            self.portfolio_file = str(data_dir / "portfolio_india_paper.json")
            self.trade_log_file = str(data_dir / "trade_log_india_paper.json")

        # Only create portfolio file if it doesn't exist - preserve existing data
        if not os.path.exists(self.portfolio_file):
            initial_portfolio = {
                "cash": self.starting_balance,
                "holdings": {},
                "starting_balance": self.starting_balance,
                "realized_pnl": 0,
                "unrealized_pnl": 0
            }
            with open(self.portfolio_file, "w") as f:
                json.dump(initial_portfolio, f, indent=4)

        # Only create trade log file if it doesn't exist - preserve existing data
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, "w") as f:
                json.dump([], f, indent=4)

        # Initialize paper trading specific logs directory
        if self.mode == "paper":
            logs_dir.mkdir(parents=True, exist_ok=True)
            paper_trade_log_path = logs_dir / f"paper_trade_{datetime.now().strftime('%Y%m%d')}.txt"
            self.paper_trade_log = str(paper_trade_log_path)
            
            # Initialize paper trade log file with header if it doesn't exist
            if not paper_trade_log_path.exists():
                with open(paper_trade_log_path, "w", encoding='utf-8') as f:
                    f.write(f"=== PAPER TRADING SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"Starting Balance: Rs.{self.starting_balance:,.2f}\n")
                    f.write("=" * 80 + "\n\n")
    def initialize_portfolio(self, balance=None):
        """Reset or initialize portfolio with a given balance."""
        if balance is not None:
            self.starting_balance = balance
        self.cash = self.starting_balance
        self.holdings = {}
        self.trade_log = []
        self.save_portfolio()
        self.save_trade_log()
        
    def add_trade_callback(self, callback):
        """Add a callback function to be called when trades are executed"""
        self.trade_callbacks.append(callback)

    def notify_trade_callbacks(self, trade_data):
        """Notify all registered callbacks about a trade execution"""
        for callback in self.trade_callbacks:
            try:
                callback(trade_data)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}") 

    def buy(self, asset, qty, price):
        """Execute a buy order in live or paper trading mode."""
        if qty <= 0:
            logger.warning(f"Invalid buy quantity: {qty} for {asset}")
            return False

        cost = qty * price
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: {asset}, qty: {qty}, price: {price}")
            return False

        try:
            # Only place actual order in live mode
            if self.mode == "live" and self.api:
                order_result = self.api.place_order(
                    security_id=self.get_security_id(asset),
                    exchange_segment="NSE_EQ",
                    transaction_type="BUY",
                    order_type="MARKET",
                    quantity=qty,
                    price=0,  # Market order uses 0 for price
                    validity="DAY"
                )
                logger.info(f"Live order placed: {order_result}")
            else:
                logger.info(f"Paper trade executed: BUY {qty} {asset} at Rs.{price}")

            # Update portfolio regardless of mode
            self.cash -= cost
            if asset in self.holdings:
                current_qty = self.holdings[asset]["qty"]
                current_avg_price = self.holdings[asset]["avg_price"]
                new_qty = current_qty + qty
                new_avg_price = ((current_avg_price * current_qty) + (price * qty)) / new_qty
                self.holdings[asset] = {"qty": new_qty, "avg_price": new_avg_price}
            else:
                self.holdings[asset] = {"qty": qty, "avg_price": price}

            trade_data = {
                "asset": asset,
                "action": "buy",
                "qty": qty,
                "price": price,
                "mode": self.mode,
                "timestamp": str(datetime.now())
            }

            self.log_trade(trade_data)

            # Update unrealized P&L for all holdings
            self.update_unrealized_pnl()
            self.save_portfolio()

            # Notify callbacks about the trade
            self.notify_trade_callbacks(trade_data)

            return True

        except Exception as e:
            logger.error(f"Error executing buy order for {asset}: {e}")
            return False

    def sell(self, asset, qty, price):
        """Execute a sell order in live or paper trading mode."""
        # Enforce global sell disable flag
        enable_sell = str(os.getenv("ENABLE_SELL", "true")).lower() not in ("false", "0", "no", "off")
        if not enable_sell:
            logger.info("Sell disabled by configuration (ENABLE_SELL=false). Skipping sell for %s", asset)
            return False

        if qty <= 0:
            logger.warning(f"Invalid sell quantity: {qty} for {asset}")
            return False

        if asset not in self.holdings or self.holdings[asset]["qty"] < qty:
            logger.warning(f"Insufficient holdings for sell order: {asset}, qty: {qty}")
            return False

        try:
            # Only place actual order in live mode
            if self.mode == "live" and self.api:
                order_result = self.api.place_order(
                    security_id=self.get_security_id(asset),
                    exchange_segment="NSE_EQ",
                    transaction_type="SELL",
                    order_type="MARKET",
                    quantity=qty,
                    price=0,  # Market order uses 0 for price
                    validity="DAY"
                )
                logger.info(f"Live order placed: {order_result}")
            else:
                logger.info(f"Paper trade executed: SELL {qty} {asset} at Rs.{price}")

            # Update portfolio regardless of mode
            revenue = qty * price
            self.cash += revenue

            # Calculate realized P&L for this trade
            avg_price = self.holdings[asset]["avg_price"]
            realized_pnl_for_trade = (price - avg_price) * qty
            self.realized_pnl += realized_pnl_for_trade

            current_qty = self.holdings[asset]["qty"]
            if current_qty == qty:
                del self.holdings[asset]
            else:
                self.holdings[asset]["qty"] -= qty

            trade_data = {
                "asset": asset,
                "action": "sell",
                "qty": qty,
                "price": price,
                "mode": self.mode,
                "timestamp": str(datetime.now()),
                "realized_pnl": realized_pnl_for_trade
            }

            self.log_trade(trade_data)

            # Update unrealized P&L for remaining holdings
            self.update_unrealized_pnl()
            self.save_portfolio()

            # Notify callbacks about the trade
            self.notify_trade_callbacks(trade_data)

            return True

        except Exception as e:
            logger.error(f"Error executing sell order for {asset}: {e}")
            return False

    def get_security_id(self, ticker):
        """Fetch security ID for a ticker using dynamic resolution system."""
        try:
            # Import the dynamic security ID method from dhan_client
            from dhan_client import DhanAPIClient
            
            # Create a temporary client instance for security ID lookup
            # Use dummy credentials since we only need the mapping function
            temp_client = DhanAPIClient("temp", "temp")
            
            # Use the dynamic security ID resolution
            security_id = temp_client.get_security_id(ticker)
            logger.info(f"Found security ID for {ticker}: {security_id}")
            return security_id
            
        except ValueError as e:
            logger.error(f"Security ID not found for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching security ID for {ticker}: {e}")
            return None


    def get_value(self, current_prices):
        """Calculate total portfolio value based on current prices."""
        total_value = self.cash
        for asset, data in self.holdings.items():
            price = current_prices.get(asset, {}).get("price", 0)
            total_value += data["qty"] * price
        return total_value

    def get_metrics(self):
        """Return comprehensive portfolio metrics with professional calculations."""
        current_prices = self.get_current_prices()

        # Calculate portfolio values
        total_invested = sum(data["qty"] * data["avg_price"] for data in self.holdings.values())  # Cost basis
        current_holdings_value = 0
        total_unrealized = 0

        # Calculate current market value and unrealized P&L
        for ticker, data in self.holdings.items():
            qty = data["qty"]
            avg_price = data["avg_price"]
            invested_amount = qty * avg_price

            if ticker in current_prices:
                # Handle both dict format {"price": value} and direct value
                if isinstance(current_prices[ticker], dict):
                    current_price = current_prices[ticker].get("price", avg_price)
                else:
                    current_price = current_prices[ticker]
                current_value = qty * current_price
                current_holdings_value += current_value
                unrealized_pnl = current_value - invested_amount
                total_unrealized += unrealized_pnl
            else:
                current_holdings_value += invested_amount

        # Total portfolio value = Cash + Current holdings value
        total_portfolio_value = self.cash + current_holdings_value

        # Calculate returns based on initial investment
        initial_balance = self.starting_balance
        total_return = total_portfolio_value - initial_balance
        total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0

        # Calculate realized P&L from completed trades
        realized_pnl = sum(
            (t["price"] - self.holdings.get(t["asset"], {}).get("avg_price", t["price"])) * t["qty"]
            for t in self.trade_log if t["action"] == "sell"
        )

        # Professional metrics
        cash_percentage = (self.cash / total_portfolio_value) * 100 if total_portfolio_value > 0 else 100
        invested_percentage = (total_invested / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

        metrics = {
            "cash": round(self.cash, 2),
            "cash_percentage": round(cash_percentage, 2),
            "holdings": self.holdings,
            "total_value": round(total_portfolio_value, 2),
            "current_holdings_value": round(current_holdings_value, 2),
            "total_invested": round(total_invested, 2),
            "invested_percentage": round(invested_percentage, 2),
            "total_exposure": round(current_holdings_value, 2),  # Current market value of holdings
            "exposure_ratio": round((total_invested / total_portfolio_value) * 100, 2) if total_portfolio_value > 0 else 0,
            "unrealized_pnl": round(total_unrealized, 2),
            "unrealized_pnl_pct": round((total_unrealized / total_invested) * 100, 2) if total_invested > 0 else 0,
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round((realized_pnl / initial_balance) * 100, 2) if initial_balance > 0 else 0,
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "profit_loss": round(total_return, 2),
            "profit_loss_pct": round(total_return_pct, 2),
            "positions": len(self.holdings),
            "initial_balance": initial_balance,
            "trades_today": len([t for t in self.trade_log if t.get("date", "").startswith(datetime.now().strftime("%Y-%m-%d"))]),
            "current_portfolio_value": self.config.get("current_portfolio_value", total_portfolio_value),
            "current_pnl": self.config.get("current_pnl", total_return)
        }
        return metrics

    def log_trade(self, trade):
        """Log a trade to the trade log file."""
        # Don't log trades with 0 quantity
        if trade.get("qty", 0) <= 0:
            logger.warning(f"Skipping trade log for {trade.get('asset')} with quantity {trade.get('qty')}")
            return

        self.trade_log.append(trade)
        self.save_trade_log()

        # Enhanced paper trading logging
        if self.mode == "paper":
            self.log_paper_trade_details(trade)

    def log_paper_trade_details(self, trade):
        """Log detailed paper trade information for Phase 2 objectives."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            action = trade["action"].upper()
            asset = trade["asset"]
            qty = trade["qty"]
            price = trade["price"]

            # Calculate current portfolio metrics
            metrics = self.get_metrics()

            log_entry = f"\n[{timestamp}] === {action} SIGNAL EXECUTED ===\n"
            log_entry += f"Asset: {asset}\n"
            log_entry += f"Action: {action} {qty} shares at Rs.{price:.2f}\n"
            log_entry += f"Trade Value: Rs.{qty * price:,.2f}\n"

            if action == "BUY":
                log_entry += f"Entry Signal: Price Rs.{price:.2f} identified as favorable entry point\n"
                log_entry += f"Position Size: {qty} shares ({(qty * price / metrics['total_value'] * 100):.1f}% of portfolio)\n"
            else:
                if asset in self.holdings:
                    avg_price = self.holdings[asset]["avg_price"]
                    pnl = (price - avg_price) * qty
                    pnl_pct = ((price / avg_price) - 1) * 100
                    log_entry += f"Exit Signal: Price Rs.{price:.2f} vs Entry Rs.{avg_price:.2f}\n"
                    log_entry += f"Trade P&L: Rs.{pnl:,.2f} ({pnl_pct:+.2f}%)\n"

            log_entry += f"Portfolio Cash: Rs.{metrics['cash']:,.2f}\n"
            log_entry += f"Total Portfolio Value: Rs.{metrics['total_value']:,.2f}\n"
            log_entry += f"Unrealized P&L: Rs.{metrics['unrealized_pnl']:,.2f}\n"
            log_entry += "="*60 + "\n"

            with open(self.paper_trade_log, "a", encoding='utf-8') as f:
                f.write(log_entry)

        except Exception as e:
            logger.error(f"Error logging paper trade details: {e}")

    def log_strategy_trigger(self, ticker, analysis, decision_data):
        """Log strategy trigger explanations for paper trading."""
        if self.mode != "paper":
            return

        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            log_entry = f"\n[{timestamp}] === STRATEGY ANALYSIS: {ticker} ===\n"

            # Technical Analysis Summary
            if 'technical_analysis' in analysis:
                tech = analysis['technical_analysis']
                log_entry += f"Technical Indicators:\n"
                log_entry += f"  - RSI: {tech.get('rsi', 'N/A'):.2f} {'(Oversold)' if tech.get('rsi', 50) < 30 else '(Overbought)' if tech.get('rsi', 50) > 70 else '(Neutral)'}\n"
                log_entry += f"  - MACD: {tech.get('macd', 'N/A'):.4f}\n"
                log_entry += f"  - SMA 50: Rs.{tech.get('sma_50', 'N/A'):.2f}\n"
                log_entry += f"  - SMA 200: Rs.{tech.get('sma_200', 'N/A'):.2f}\n"
                log_entry += f"  - Trend: {'Bullish' if tech.get('sma_50', 0) > tech.get('sma_200', 0) else 'Bearish'}\n"

            # Sentiment Analysis Summary
            if 'sentiment_analysis' in analysis:
                sentiment = analysis['sentiment_analysis']['aggregated']
                total = sum(sentiment.values())
                if total > 0:
                    pos_pct = sentiment['positive'] / total * 100
                    neg_pct = sentiment['negative'] / total * 100
                    log_entry += f"Market Sentiment:\n"
                    log_entry += f"  - Positive: {pos_pct:.1f}% ({sentiment['positive']} articles)\n"
                    log_entry += f"  - Negative: {neg_pct:.1f}% ({sentiment['negative']} articles)\n"
                    log_entry += f"  - Overall: {'Bullish' if pos_pct > neg_pct else 'Bearish'}\n"

            # ML/RL Predictions
            if 'ml_analysis' in analysis:
                ml = analysis['ml_analysis']
                current_price = analysis.get('stock_data', {}).get('current_price', {}).get('INR', 0)
                predicted_price = ml.get('predicted_price', current_price)
                price_change = ((predicted_price / current_price) - 1) * 100 if current_price > 0 else 0

                log_entry += f"ML/RL Analysis:\n"
                log_entry += f"  - Current Price: Rs.{current_price:.2f}\n"
                log_entry += f"  - Predicted Price: Rs.{predicted_price:.2f}\n"
                log_entry += f"  - Expected Change: {price_change:+.2f}%\n"
                log_entry += f"  - RL Recommendation: {ml.get('rl_metrics', {}).get('recommendation', 'HOLD')}\n"

            # Decision Summary
            log_entry += f"Decision Factors:\n"
            log_entry += f"  - Buy Score: {decision_data.get('buy_score', 0):.3f}\n"
            log_entry += f"  - Sell Score: {decision_data.get('sell_score', 0):.3f}\n"
            log_entry += f"  - Buy Signals: {decision_data.get('buy_signals', 0)}/4\n"
            log_entry += f"  - Sell Signals: {decision_data.get('sell_signals', 0)}/4\n"
            log_entry += f"  - Final Decision: {decision_data.get('action', 'HOLD')}\n"

            if decision_data.get('action') in ['BUY', 'SELL']:
                log_entry += f"  - Quantity: {decision_data.get('quantity', 0)} shares\n"
                log_entry += f"  - Trade Value: Rs.{decision_data.get('trade_value', 0):,.2f}\n"

            log_entry += "="*60 + "\n"

            with open(self.paper_trade_log, "a", encoding='utf-8') as f:
                f.write(log_entry)

        except Exception as e:
            logger.error(f"Error logging strategy trigger: {e}")

    def generate_paper_pnl_summary(self):
        """Generate comprehensive P&L summary for paper trading."""
        if self.mode != "paper":
            return

        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics = self.get_metrics()

            # Calculate performance metrics
            total_return = metrics['total_value'] - self.starting_balance
            total_return_pct = (total_return / self.starting_balance) * 100

            # Trade statistics
            buy_trades = [t for t in self.trade_log if t['action'] == 'buy']
            sell_trades = [t for t in self.trade_log if t['action'] == 'sell']

            # Calculate realized P&L from completed trades
            realized_pnl = 0
            for sell_trade in sell_trades:
                asset = sell_trade['asset']
                # Find corresponding buy trades for this asset
                asset_buy_trades = [t for t in buy_trades if t['asset'] == asset]
                if asset_buy_trades:
                    avg_buy_price = sum(t['price'] * t['qty'] for t in asset_buy_trades) / sum(t['qty'] for t in asset_buy_trades)
                    realized_pnl += (sell_trade['price'] - avg_buy_price) * sell_trade['qty']

            # Win/Loss ratio
            profitable_trades = sum(1 for t in sell_trades if t['price'] >
                                  (sum(bt['price'] * bt['qty'] for bt in buy_trades if bt['asset'] == t['asset']) /
                                   sum(bt['qty'] for bt in buy_trades if bt['asset'] == t['asset']) if
                                   any(bt['asset'] == t['asset'] for bt in buy_trades) else t['price']))

            win_rate = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0

            summary = f"\n[{timestamp}] === PAPER TRADING P&L SUMMARY ===\n"
            summary += f"Session Duration: {(datetime.now() - datetime.strptime(self.trade_log[0]['timestamp'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds() / 3600:.1f} hours\n" if self.trade_log else "Session Duration: 0 hours\n"
            summary += f"\nPortfolio Performance:\n"
            summary += f"  - Starting Balance: Rs.{self.starting_balance:,.2f}\n"
            summary += f"  - Current Cash: Rs.{metrics['cash']:,.2f}\n"
            summary += f"  - Holdings Value: Rs.{metrics['total_exposure']:,.2f}\n"
            summary += f"  - Total Portfolio Value: Rs.{metrics['total_value']:,.2f}\n"
            summary += f"  - Total Return: Rs.{total_return:,.2f} ({total_return_pct:+.2f}%)\n"
            summary += f"  - Realized P&L: Rs.{realized_pnl:,.2f}\n"
            summary += f"  - Unrealized P&L: Rs.{metrics['unrealized_pnl']:,.2f}\n"

            summary += f"\nTrading Statistics:\n"
            summary += f"  - Total Trades: {len(self.trade_log)}\n"
            summary += f"  - Buy Orders: {len(buy_trades)}\n"
            summary += f"  - Sell Orders: {len(sell_trades)}\n"
            summary += f"  - Win Rate: {win_rate:.1f}%\n"
            summary += f"  - Active Positions: {len(self.holdings)}\n"

            if self.holdings:
                summary += f"\nCurrent Holdings:\n"
                for asset, data in self.holdings.items():
                    current_value = data['qty'] * data['avg_price']  # Simplified - would need current price
                    summary += f"  - {asset}: {data['qty']} shares @ Rs.{data['avg_price']:.2f} (Rs.{current_value:,.2f})\n"

            summary += "="*70 + "\n"

            with open(self.paper_trade_log, "a", encoding='utf-8') as f:
                f.write(summary)

            logger.info("Paper trading P&L summary generated")

        except Exception as e:
            logger.error(f"Error generating paper P&L summary: {e}")

    def save_portfolio(self):
        """Save portfolio state to JSON file."""
        try:
            portfolio_data = {
                "cash": self.cash,
                "holdings": self.holdings,
                "starting_balance": self.starting_balance,
                "realized_pnl": getattr(self, 'realized_pnl', 0),
                "unrealized_pnl": getattr(self, 'unrealized_pnl', 0)
            }
            with open(self.portfolio_file, "w") as f:
                json.dump(portfolio_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def save_trade_log(self):
        """Save trade log to JSON file."""
        try:
            with open(self.trade_log_file, "w") as f:
                json.dump(self.trade_log, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")

    def update_unrealized_pnl(self):
        """Update unrealized P&L based on current market prices."""
        try:
            if not self.holdings:
                self.unrealized_pnl = 0
                return

            total_unrealized = 0

            for ticker, data in self.holdings.items():
                try:
                    # Use Fyers first, fallback to yfinance - SAME LOGIC
                    df = get_stock_data_fyers_or_yf(ticker)
                    if df is not None and not df.empty:
                        current_price = df['Close'].iloc[-1]
                        unrealized_for_stock = (current_price - data['avg_price']) * data['qty']
                        total_unrealized += unrealized_for_stock
                    else:
                        logger.warning(f"No price data available for {ticker}")
                except Exception as e:
                    logger.warning(f"Error fetching price for {ticker}: {e}")

            self.unrealized_pnl = total_unrealized
            logger.info(f"Updated unrealized P&L: Rs.{self.unrealized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error updating unrealized P&L: {e}")

    def get_current_prices(self):
        """Fetch current prices using Yahoo Finance only."""
        import yfinance as yf
        prices = {}
        for asset in self.holdings:
            try:
                # Use Fyers first, fallback to yfinance - SAME LOGIC
                hist = get_stock_data_fyers_or_yf(asset)
                if hist is not None and not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    prices[asset] = {"price": current_price, "volume": volume}
                    logger.debug(f"Fetched price for {asset}: Rs.{current_price:.2f}")
                else:
                    logger.warning(f"No price data available for {asset}")
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
        return prices
    

class TradingExecutor:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.mode = config.get("mode", "paper")
        self.dhanhq = portfolio.api  # Use the dhanhq client from VirtualPortfolio
        self.stop_loss_pct = float(config.get("stop_loss_pct", 0.05))
        self.max_capital_per_trade = float(config.get("max_capital_per_trade", 0.40))  # Increased from 25% to 40%
        self.max_trade_limit = int(config.get("max_trade_limit", 150))
        
        # Initialize live executor for database integration if available
        self.live_executor = None
        if self.mode == "live":
            try:
                from portfolio_manager import DualPortfolioManager
                from live_executor import LiveTradingExecutor
                
                # Initialize the portfolio manager
                self._portfolio_manager = DualPortfolioManager()
                self.live_executor = LiveTradingExecutor(self._portfolio_manager, config)
                logger.info("Live trading database integration initialized successfully")
            except ImportError as e:
                logger.warning(f"Live trading database components not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize live trading database integration: {e}")
        
    def set_live_executor(self, live_executor):
        """Set the live executor for database integration"""
        self.live_executor = live_executor

    def execute_trade(self, action, ticker, qty, price, stop_loss=None, take_profit=None):
        try:
            # Enforce global sell disable flag
            if action.upper() == "SELL":
                enable_sell = str(os.getenv("ENABLE_SELL", "true")).lower() not in ("false", "0", "no", "off")
                if not enable_sell:
                    logger.info("Sell disabled by configuration (ENABLE_SELL=false). Skipping sell for %s", ticker)
                    return {"success": False, "message": "Sell disabled by configuration"}

            # Apply risk management rules with detailed logging
            portfolio_value = self.portfolio.get_value({ticker: {"price": price}})
            max_trade_value = portfolio_value * self.max_capital_per_trade
            trade_value = qty * price

            logger.info(f"=== CAPITAL LIMITS CHECK for {ticker} ===")
            logger.info(f"  Portfolio Value: Rs.{portfolio_value:.2f}")
            logger.info(f"  Max Capital Per Trade: {self.max_capital_per_trade:.1%}")
            logger.info(f"  Max Trade Value: Rs.{max_trade_value:.2f}")
            logger.info(f"  Requested Trade Value: Rs.{trade_value:.2f}")

            # Check if trade exceeds maximum capital per trade
            if trade_value > max_trade_value:
                adjusted_qty = max(1, int(max_trade_value / price))  # Ensure minimum 1 share
                if adjusted_qty < qty:
                    logger.warning(f"Reducing trade size from {qty} to {adjusted_qty} due to capital limits")
                    qty = adjusted_qty
                else:
                    logger.info(f"  Capital limits OK: Trade within limits")
            else:
                logger.info(f"  Capital limits OK: Trade within limits")

            # If quantity becomes 0 or negative, don't execute the trade
            if qty <= 0:
                logger.warning(f"Trade cancelled: Quantity reduced to {qty} due to capital limits")
                return {"success": False, "message": f"Trade cancelled: Insufficient capital for minimum trade size"}

            # Check trade limit
            if len(self.portfolio.trade_log) >= self.max_trade_limit:
                logger.warning(f"Maximum trade limit ({self.max_trade_limit}) reached")
                return {"success": False, "message": "Trade limit exceeded"}

            # Set default stop loss if not provided
            if stop_loss is None:
                if action.upper() == "BUY":
                    stop_loss = price * (1 - self.stop_loss_pct)
                else:
                    stop_loss = price * (1 + self.stop_loss_pct)

            # Execute trade based on mode
            if self.mode == "live":
                # Use database-integrated live executor if available
                if self.live_executor:
                    logger.info(f"Using database-integrated live executor for {action.upper()} {qty} {ticker}")
                    
                    signal_data = {
                        "confidence": 0.5,  # Default confidence
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "current_price": price,  # Pass current price as fallback
                        "quantity": qty  # Pass the calculated quantity
                    }
                    
                    if action.upper() == "BUY":
                        result = self.live_executor.execute_buy_order(ticker, signal_data)
                    else:  # SELL
                        result = self.live_executor.execute_sell_order(ticker, signal_data)
                    
                    if result.get("success"):
                        logger.info(f"Database live trade executed: {action} {qty} units of {ticker} at Rs.{price}")
                        return {
                            "success": True,
                            "action": action,
                            "ticker": ticker,
                            "qty": result.get("quantity", qty),
                            "price": result.get("price", price),
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "mode": self.mode,
                            "order_id": result.get("order_id")
                        }
                    else:
                        logger.error(f"Database live trade failed: {result.get('message')}")
                        return {"success": False, "message": result.get("message", "Live trade failed")}
                
                # Fallback to original live trading logic if no database executor
                elif self.dhanhq:
                    # Fetch security ID for live trading
                    security_id = self.get_security_id(ticker)
                    if security_id is None:
                        error_msg = f"Could not find security ID for {ticker}"
                        logger.error(error_msg)
                        return {"success": False, "message": error_msg}

                    # Place live order
                    order = self.dhanhq.place_order(
                        security_id=security_id,
                        exchange_segment="NSE_EQ",
                        transaction_type=action.upper(),
                        order_type="MARKET",
                        quantity=int(qty),
                        price=0,  # Market order
                        validity="DAY"
                    )
                    logger.info(f"Legacy live trade executed: {action} {qty} units of {ticker} at Rs.{price}")
                else:
                    return {"success": False, "message": "No live trading client available"}
            else:
                # Enhanced paper trading logging
                signal_type = "ENTRY" if action.upper() == "BUY" else "EXIT"
                logger.info(f"PAPER TRADE - {signal_type} SIGNAL: {action.upper()} {qty} units of {ticker} at Rs.{price:.2f}")
                logger.info(f"   Trade Value: Rs.{qty * price:,.2f}")
                logger.info(f"   Stop Loss: Rs.{stop_loss:.2f} ({((stop_loss/price - 1) * 100):+.1f}%)")
                if take_profit:
                    logger.info(f"   Take Profit: Rs.{take_profit:.2f} ({((take_profit/price - 1) * 100):+.1f}%)")
                logger.info(f"   Risk/Reward Ratio: {((take_profit - price) / (price - stop_loss)):.2f}" if take_profit and stop_loss < price else "N/A")
                order = {"order_id": f"PAPER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}

            # Actually update the portfolio
            if action.upper() == "BUY":
                success = self.portfolio.buy(ticker, qty, price)
            else:  # SELL
                success = self.portfolio.sell(ticker, qty, price)

            if success:
                return {
                    "success": True,
                    "action": action,
                    "ticker": ticker,
                    "qty": qty,
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "mode": self.mode,
                    "order": order
                }
            else:
                return {"success": False, "message": f"Failed to update portfolio for {action} {ticker}"}
        except Exception as e:
            logger.error(f"Error executing {action} order for {ticker}: {str(e)}")
            return {"success": False, "message": str(e)}

    def convert_ticker_to_dhan_format(self, ticker):
        """Convert yfinance ticker format to Dhan API format."""
        # Remove .NS, .BO suffixes for Indian stocks
        if ticker.endswith('.NS') or ticker.endswith('.BO'):
            return ticker.split('.')[0]
        return ticker

    def get_security_id(self, ticker):
        """Fetch security ID for a ticker using dynamic resolution system."""
        try:
            # Import the dynamic security ID method from dhan_client
            from dhan_client import DhanAPIClient
            
            # Create a temporary client instance for security ID lookup
            # Use dummy credentials since we only need the mapping function
            temp_client = DhanAPIClient("temp", "temp")
            
            # Use the dynamic security ID resolution
            security_id = temp_client.get_security_id(ticker)
            logger.info(f"Found security ID for {ticker}: {security_id}")
            return security_id
            
        except ValueError as e:
            logger.error(f"Security ID not found for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching security ID for {ticker}: {e}")
            return None

            return None
        
class PerformanceReport:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.report_dir = "reports"
        os.makedirs(self.report_dir, exist_ok=True)

    def generate_report(self):
        """Generate a daily performance report."""
        metrics = self.portfolio.get_metrics()
        total_value = metrics["total_value"]
        starting_value = self.portfolio.starting_balance
        daily_roi = ((total_value / starting_value) - 1) * 100
        cumulative_roi = daily_roi  # Simplified for daily report

        returns = [t["price"] for t in self.portfolio.trade_log if t["action"] == "sell"]
        if len(returns) > 1:
            returns = np.array(returns)
            sharpe_ratio = (np.mean(returns) - 0.02) / np.std(returns) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0

        values = [starting_value] + [metrics["total_value"]]
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "roi_today": daily_roi,
            "cumulative_roi": cumulative_roi,
            "sharpe": sharpe_ratio,
            "drawdown": max_drawdown,
            "trades_executed": len(self.portfolio.trade_log)
        }

        report_file = os.path.join(self.report_dir, f"report_{datetime.now().strftime('%Y%m%d')}.json")
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        return report

class PortfolioTracker:
    def __init__(self, portfolio, config):
        self.portfolio = portfolio
        self.config = config

    def log_metrics(self):
        """Log portfolio metrics once."""
        try:
            metrics = self.portfolio.get_metrics()
            logger.info(f"Portfolio Metrics:")
            logger.info(f"Cash: Rs.{metrics['cash']:.2f}")
            logger.info(f"Holdings: {metrics['holdings']}")
            logger.info(f"Total Value: Rs.{metrics['total_value']:.2f}")
            logger.info(f"Current Portfolio Value (Dhan): Rs.{self.config.get('current_portfolio_value', 0):.2f}")
            logger.info(f"Current PnL (Dhan): Rs.{self.config.get('current_pnl', 0):.2f}")
            logger.info(f"Realized PnL: Rs.{metrics['realized_pnl']:.2f}")
            logger.info(f"Unrealized PnL: Rs.{metrics['unrealized_pnl']:.2f}")
            logger.info(f"Total Exposure: Rs.{metrics['total_exposure']:.2f}")
        except Exception as e:
            logger.error(f"Error logging portfolio metrics: {e}")

class Stock:
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"
    FMPC_API_KEY = os.getenv("FMPC_API_KEY")
    MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
    SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
    SANTIMENT_API = "https://api.santiment.net"
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    def __init__(self, reddit_client_id=None, reddit_client_secret=None, reddit_user_agent=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Initialize Fyers-based ticker mapper
        self.ticker_mapper = FyersTickerMapper()
        logger.info("Dynamic ticker mapper initialized")

        if reddit_client_id and reddit_client_secret and reddit_user_agent:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        else:
            self.reddit = None
        self.COINGECKO_API = "https://api.coingecko.com/api/v3"
        self.last_rate_fetch = None
        self.rate_cache = None
        self.cache_duration = 300
        self.google_news = GNews()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.filterwarnings('ignore')

        # PRODUCTION FIX: Add technical analysis caching for speed
        self.technical_cache = {}
        self.technical_cache_duration = 300  # 5 minutes cache

    def get_company_names(self, ticker):
        """Get company names for news search using dynamic mapping"""
        return self.ticker_mapper.get_company_names(ticker)

    def build_search_query(self, ticker):
        """Build search query using dynamic ticker mapping"""
        company_names = self.get_company_names(ticker)
        if not company_names:
            return ticker.replace(".NS", "").replace(".BO", "")

        # Create OR query with quoted names (limit to top 3 to avoid overly long queries)
        query_parts = [f'"{name}"' for name in company_names[:3]]
        return " OR ".join(query_parts)

    def get_sector_keywords(self, ticker, company_names):
        """Get sector-specific keywords dynamically for any Indian stock"""
        sector_keywords = {
            # Banking & Financial
            'banking': ['loan growth', 'NPA', 'credit growth', 'interest rates', 'deposits'],
            'financial': ['lending', 'credit', 'banking', 'finance', 'loan'],

            # Oil & Gas
            'petroleum': ['refinery', 'crude oil', 'fuel prices', 'margins', 'downstream'],
            'oil': ['crude', 'refining', 'petrochemicals', 'fuel', 'energy'],
            'gas': ['natural gas', 'LNG', 'pipeline', 'exploration', 'upstream'],

            # IT & Technology
            'technology': ['software', 'IT services', 'digital', 'cloud', 'automation'],
            'software': ['tech', 'IT', 'digital transformation', 'cloud computing'],
            'infotech': ['software', 'technology', 'IT services', 'consulting'],

            # Pharma & Healthcare
            'pharma': ['drug', 'medicine', 'FDA approval', 'clinical trials', 'healthcare'],
            'healthcare': ['medical', 'hospital', 'treatment', 'pharmaceutical'],

            # Auto & Manufacturing
            'auto': ['vehicle sales', 'automobile', 'EV', 'electric vehicle', 'manufacturing'],
            'steel': ['iron ore', 'steel prices', 'infrastructure', 'construction'],
            'cement': ['construction', 'infrastructure', 'real estate', 'housing'],

            # Telecom
            'telecom': ['5G', 'spectrum', 'mobile', 'broadband', 'network'],
            'communication': ['telecom', 'mobile services', 'data', 'connectivity'],

            # FMCG & Consumer
            'consumer': ['FMCG', 'rural demand', 'urban consumption', 'brand'],
            'fmcg': ['consumer goods', 'rural market', 'distribution', 'brand'],

            # Power & Utilities
            'power': ['electricity', 'renewable energy', 'solar', 'wind', 'grid'],
            'energy': ['power generation', 'renewable', 'electricity', 'utilities'],

            # Real Estate
            'realty': ['real estate', 'property', 'housing', 'construction', 'land'],
            'housing': ['real estate', 'property development', 'residential', 'commercial']
        }

        # Detect sector from company names and ticker
        detected_keywords = []
        search_text = f"{ticker} {' '.join(company_names)}".lower()

        for sector, keywords in sector_keywords.items():
            if sector in search_text:
                detected_keywords.extend(keywords[:3])  # Top 3 keywords per sector

        return detected_keywords[:5]  # Limit to 5 most relevant keywords

    def build_multi_level_search_queries(self, ticker):
        """Build multiple search query levels for fallback strategy"""
        company_names = self.get_company_names(ticker)
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

        # Get dynamic sector keywords
        sector_keywords = self.get_sector_keywords(ticker, company_names)

        search_levels = []

        # Level 1: Full company names (highest priority)
        if company_names:
            primary_query = " OR ".join([f'"{name}"' for name in company_names[:3]])
            search_levels.append({
                "level": 1,
                "query": primary_query,
                "description": "Full company names",
                "priority": "high"
            })

        # Level 2: Company names with financial context + sector-specific terms
        if company_names:
            contextual_queries = []
            for name in company_names[:2]:
                contextual_queries.extend([
                    f'"{name}" stock',
                    f'"{name}" shares',
                    f'"{name}" earnings',
                    f'"{name}" financial'
                ])

                # Add dynamic sector-specific terms for ANY Indian stock (limit to 2)
                for keyword in sector_keywords[:2]:
                    contextual_queries.append(f'"{name}" {keyword}')

            contextual_query = " OR ".join(contextual_queries[:5])  # Limit to 5 to avoid URL length issues
            search_levels.append({
                "level": 2,
                "query": contextual_query,
                "description": "Company names with financial context",
                "priority": "medium"
            })

        # Level 3: Clean ticker with market context
        market_context_query = f'"{clean_ticker}" AND (NSE OR BSE OR "Indian stock" OR "India market")'
        search_levels.append({
            "level": 3,
            "query": market_context_query,
            "description": "Ticker with market context",
            "priority": "medium"
        })

        # Level 4: Sector-based search
        sector_keywords = self.get_sector_keywords(ticker, company_names)
        if sector_keywords:
            sector_query = " OR ".join([f'"{keyword}"' for keyword in sector_keywords[:3]])
            search_levels.append({
                "level": 4,
                "query": sector_query,
                "description": "Sector-based search",
                "priority": "low"
            })

        # Level 5: Earnings and analyst sentiment (NEW)
        if company_names:
            earnings_queries = []
            for name in company_names[:2]:
                earnings_queries.extend([
                    f'"{name}" earnings beat',
                    f'"{name}" analyst upgrade',
                    f'"{name}" target price',
                    f'"{name}" buy rating',
                    f'"{name}" quarterly results'
                ])
            earnings_query = " OR ".join(earnings_queries[:5])
            search_levels.append({
                "level": 5,
                "query": earnings_query,
                "description": "Earnings and analyst sentiment",
                "priority": "medium"
            })

        # Level 6: Basic ticker (fallback)
        search_levels.append({
            "level": 6,
            "query": clean_ticker,
            "description": "Basic ticker fallback",
            "priority": "low"
        })

        return search_levels

    def get_sector_keywords(self, ticker, company_names):
        """Dynamically determine sector keywords based on company names"""
        if not company_names:
            return []

        sector_keywords = []
        combined_text = " ".join(company_names).lower()

        # Dynamic sector detection
        sector_mappings = {
            'banking': ['bank', 'financial', 'finance', 'credit'],
            'technology': ['tech', 'software', 'it', 'consultancy', 'services'],
            'energy': ['energy', 'power', 'electricity', 'renewable', 'solar', 'wind'],
            'automotive': ['auto', 'motor', 'car', 'vehicle', 'suzuki', 'tata motors'],
            'pharmaceuticals': ['pharma', 'drug', 'medicine', 'healthcare', 'bio'],
            'telecommunications': ['telecom', 'airtel', 'communication', 'mobile'],
            'steel': ['steel', 'iron', 'metal', 'mining'],
            'oil_gas': ['oil', 'gas', 'petroleum', 'refinery'],
            'cement': ['cement', 'construction', 'building'],
            'textiles': ['textile', 'fabric', 'cotton', 'apparel'],
            'chemicals': ['chemical', 'fertilizer', 'pesticide'],
            'real_estate': ['real estate', 'property', 'housing', 'realty'],
            'insurance': ['insurance', 'life', 'general insurance'],
            'ports': ['port', 'shipping', 'logistics', 'cargo'],
            'aviation': ['aviation', 'airline', 'airport', 'aircraft']
        }

        detected_sectors = []
        for sector, keywords in sector_mappings.items():
            if any(keyword in combined_text for keyword in keywords):
                detected_sectors.append(sector)

        # Generate sector-specific search terms
        for sector in detected_sectors[:2]:  # Limit to top 2 sectors
            if sector == 'banking':
                sector_keywords.extend(['banking sector India', 'Indian banks', 'financial services India'])
            elif sector == 'technology':
                sector_keywords.extend(['IT sector India', 'Indian tech companies', 'software services India'])
            elif sector == 'energy':
                sector_keywords.extend(['energy sector India', 'power companies India', 'renewable energy India'])
            elif sector == 'automotive':
                sector_keywords.extend(['auto sector India', 'Indian automobile', 'car manufacturers India'])
            # Add more sector-specific terms as needed

        return sector_keywords[:5]  # Limit to top 5 keywords

    def update_ticker_mapping(self):
        """Manually trigger ticker mapping update"""
        self.ticker_mapper.get_ticker_mapping(force_update=True)
        logger.info("Ticker mapping updated successfully")

    def get_cached_technical_analysis(self, ticker, history):
        """PRODUCTION FIX: Get cached technical analysis or calculate if needed"""
        cache_key = f"{ticker}_{len(history)}"
        current_time = datetime.now()

        # Check if we have valid cached data
        if (cache_key in self.technical_cache and
            (current_time - self.technical_cache[cache_key]['timestamp']).total_seconds() < self.technical_cache_duration):
            logger.info(f"Using cached technical analysis for {ticker}")
            return self.technical_cache[cache_key]['data']

        # Calculate technical indicators
        logger.info(f"Calculating technical analysis for {ticker}")
        technical_data = self.calculate_technical_indicators(history)

        # Cache the results
        self.technical_cache[cache_key] = {
            'data': technical_data,
            'timestamp': current_time
        }

        return technical_data

    def calculate_technical_indicators(self, history):
        """Calculate technical indicators efficiently"""
        try:
            # Create a copy to avoid modifying original data
            data = history.copy()

            # Moving averages
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_200"] = data["Close"].rolling(window=200).mean()
            data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

            # RSI calculation
            def calculate_rsi(prices, periods=14):
                delta = prices.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down.where(roll_down != 0, 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi.clip(0, 100)

            data["RSI"] = calculate_rsi(data["Close"])

            # Bollinger Bands
            data["BB_Middle"] = data["Close"].rolling(window=20).mean()
            bb_std = data["Close"].rolling(window=20).std()
            data["BB_Upper"] = data["BB_Middle"] + 2 * bb_std
            data["BB_Lower"] = data["BB_Middle"] - 2 * bb_std

            # MACD
            exp1 = data["Close"].ewm(span=12, adjust=False).mean()
            exp2 = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = exp1 - exp2
            data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

            # Volume analysis (if available)
            if "Volume" in data.columns:
                data["Volume_SMA"] = data["Volume"].rolling(window=20).mean()
                data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]
            else:
                data["Volume_Ratio"] = 1.0

            # Fill NaN values
            data.fillna({
                "SMA_50": data["Close"],
                "SMA_200": data["Close"],
                "RSI": 50,
                "MACD": 0,
                "Signal_Line": 0,
                "BB_Middle": data["Close"],
                "BB_Upper": data["Close"] * 1.02,
                "BB_Lower": data["Close"] * 0.98,
                "Volume_Ratio": 1.0
            }, inplace=True)

            return data

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return history

    def fetch_exchange_rates(self):
        """Fetch real-time exchange rates for INR, EUR, BTC, ETH"""
        if (self.last_rate_fetch and 
            self.rate_cache and 
            (datetime.now() - self.last_rate_fetch).total_seconds() < self.cache_duration):
            return self.rate_cache

        try:
            url = f"{self.COINGECKO_API}/simple/price?ids=bitcoin,ethereum&vs_currencies=usd,inr,eur"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            btc_usd = data.get("bitcoin", {}).get("usd", 1)
            rates = {
                "bitcoin": {"usd": btc_usd},
                "ethereum": {"usd": data.get("ethereum", {}).get("usd", 1)},
                "inr": {"usd": data.get("bitcoin", {}).get("inr", 1) / btc_usd if btc_usd != 0 else 83},
                "eur": {"usd": data.get("bitcoin", {}).get("eur", 1) / btc_usd if btc_usd != 0 else 0.95}
            }

            self.rate_cache = rates
            self.last_rate_fetch = datetime.now()
            return rates
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {e}")
            fallback = {
                "bitcoin": {"usd": 85000},
                "ethereum": {"usd": 1633},
                "inr": {"usd": 86},
                "eur": {"usd": 0.95}
            }
            self.rate_cache = fallback
            self.last_rate_fetch = datetime.now()
            return fallback

    def convert_price(self, price, exchange_rates):
        """Convert price to different currencies including crypto and fiat"""
        if not isinstance(price, (int, float)) or price == "N/A":
            return {"INR": price}
        try:
            return {
                "INR": round(float(price), 2),
                "USD": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1), 2),
                "EUR": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) * exchange_rates.get("eur", {}).get("usd", 1), 2),
                "BTC": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) / exchange_rates.get("bitcoin", {}).get("usd", 1), 8),
                "ETH": round(float(price) / exchange_rates.get("inr", {}).get("usd", 1) / exchange_rates.get("ethereum", {}).get("usd", 1), 8)
            }
        except Exception as e:
            logger.error(f"Error converting price: {e}")
            return {
                "INR": round(float(price), 2),
                "USD": price,
                "EUR": price,
                "BTC": price,
                "ETH": price
            }

    def convert_np_types(self, data):
        if isinstance(data, dict):
            return {k: self.convert_np_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_np_types(item) for item in data]
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif pd.isna(data):
            return None
        else:
            return data

    def get_simple_news_sources(self):
        """Get focused news sources to avoid rate limits"""
        return [
            'economictimes.indiatimes.com', 'moneycontrol.com',
            'business-standard.com', 'reuters.com', 'bloomberg.com'
        ]

    def build_source_specific_query(self, ticker, source_type='primary'):
        """Build search queries optimized for specific Indian news sources"""
        company_names = self.get_company_names(ticker)
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

        if source_type == 'primary':
            # For major financial news sources, use comprehensive queries
            if company_names:
                base_query = " OR ".join([f'"{name}"' for name in company_names[:2]])
                enhanced_query = f"({base_query}) AND (stock OR shares OR earnings OR financial OR results)"
                return enhanced_query

        elif source_type == 'secondary':
            # For secondary sources, use simpler but effective queries
            if company_names:
                return f'"{company_names[0]}" stock India'

        elif source_type == 'regional':
            # For regional sources, include location context
            if company_names:
                return f'"{company_names[0]}" AND (Mumbai OR Delhi OR Bangalore OR India)'

        elif source_type == 'specialized':
            # For specialized financial sources, use technical terms
            if company_names:
                return f'"{company_names[0]}" AND (NSE OR BSE OR trading OR investment)'

        # Fallback to basic query
        return clean_ticker

    def build_simple_market_query(self, ticker):
        """Build simple, focused query to avoid rate limits"""
        company_names = self.get_company_names(ticker)
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

        # Simple company-focused query
        if company_names:
            return f'"{company_names[0]}" (stock OR shares OR earnings OR results)'
        else:
            return f'{clean_ticker} stock India'

    def get_market_context_multiplier(self):
        """Get dynamic market context multiplier based on current market conditions"""
        try:
            from datetime import datetime
            current_hour = datetime.now().hour

            # Higher weight during market hours (9 AM to 4 PM IST)
            if 9 <= current_hour <= 16:
                return 1.3  # 30% higher weight during market hours
            elif 7 <= current_hour <= 9 or 16 <= current_hour <= 18:
                return 1.1  # 10% higher weight during pre/post market
            else:
                return 1.0  # Normal weight during off-market hours
        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Error calculating market hours multiplier: {e}")
            return 1.0  # Default multiplier if calculation fails

    def multi_level_search(self, ticker, search_function, max_levels=3):
        """Execute multi-level search strategy with fallback"""
        search_levels = self.build_multi_level_search_queries(ticker)

        for level_info in search_levels[:max_levels]:
            try:
                logger.info(f"Trying search level {level_info['level']}: {level_info['description']}")
                logger.debug(f"Query: {level_info['query']}")

                # Execute search with current level query
                results = search_function(level_info['query'])

                # Check if we got meaningful results
                if self.has_meaningful_results(results):
                    logger.info(f"Success at level {level_info['level']}: Found meaningful results")
                    return results
                else:
                    logger.info(f"Level {level_info['level']} returned insufficient results, trying next level")

            except Exception as e:
                logger.warning(f"Level {level_info['level']} failed: {e}, trying next level")
                continue

        # If all levels fail, return empty results
        logger.warning(f"All search levels failed for {ticker}")
        return {"positive": 0, "negative": 0, "neutral": 0}

    def has_meaningful_results(self, results):
        """Check if search results are meaningful"""
        if not isinstance(results, dict):
            return False

        total_articles = results.get("positive", 0) + results.get("negative", 0) + results.get("neutral", 0)
        return total_articles >= 2  # Minimum threshold for meaningful results

    def newsapi_sentiment(self, ticker):
        """Comprehensive NewsAPI sentiment with global and Indian sources covering all market factors"""
        def search_with_query(query):
            try:
                # Get simple news sources to avoid rate limits
                selected_sources = self.get_simple_news_sources()
                sources_param = ",".join(selected_sources[:5])  # Limit to 5 sources to avoid rate limits

                # Focus on last 24 hours for more actionable sentiment
                from datetime import datetime, timedelta
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

                # Build URL with Indian sources preference and 24-hour filter
                url = f"https://newsapi.org/v2/everything?q={quote(query)}&sources={sources_param}&from={yesterday}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"

                # Check URL length to avoid 400 errors (NewsAPI has ~2000 char limit)
                if len(url) > 1800:
                    logger.warning(f"URL too long ({len(url)} chars), using simplified query")
                    # Use first company name only for very long queries
                    company_names = self.get_company_names(ticker.replace('.NS', ''))
                    simple_query = f'"{company_names[0]}" stock' if company_names else ticker.replace('.NS', '')
                    url = f"https://newsapi.org/v2/everything?q={quote(simple_query)}&from={yesterday}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"

                # Add delay to avoid rate limits
                import time
                time.sleep(1.5)  # 1.5 second delay between requests to avoid 429 errors

                # Fallback to general search if sources fail
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                except (requests.exceptions.RequestException, requests.exceptions.HTTPError, ValueError, KeyError) as e:
                    logger.warning(f"NewsAPI request with sources failed: {e}")
                    # Fallback without specific sources but keep 24-hour filter
                    fallback_url = f"https://newsapi.org/v2/everything?q={quote(query)}&from={yesterday}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"
                    # Check fallback URL length too
                    if len(fallback_url) > 1800:
                        company_names = self.get_company_names(ticker.replace('.NS', ''))
                        simple_query = f'"{company_names[0]}" stock' if company_names else ticker.replace('.NS', '')
                        fallback_url = f"https://newsapi.org/v2/everything?q={quote(simple_query)}&from={yesterday}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"
                    response = requests.get(fallback_url, timeout=10)
                    response.raise_for_status()
                    data = response.json()

                sentiments = {"positive": 0, "negative": 0, "neutral": 0}
                total_articles = data.get("totalResults", 0)

                if total_articles == 0:
                    return sentiments

                logger.info(f"Found {total_articles} articles for query: {query[:50]}...")

                if "articles" in data:
                    for article in data["articles"][:10]:
                        description = article.get("description", "") or article.get("content", "")[:200]
                        if description:
                            sentiment = self.sentiment_analyzer.polarity_scores(description)

                            # ENHANCED: Weight earnings/analyst news higher
                            weight = 1.0
                            earnings_keywords = ['earnings', 'quarterly', 'results', 'beat', 'miss', 'guidance',
                                               'analyst', 'upgrade', 'downgrade', 'target', 'rating']
                            if any(keyword in description.lower() for keyword in earnings_keywords):
                                weight = 2.0  # Double weight for earnings/analyst news

                            if sentiment["compound"] > 0.1:
                                sentiments["positive"] += weight
                            elif sentiment["compound"] < -0.1:
                                sentiments["negative"] += weight
                            else:
                                sentiments["neutral"] += weight

                return sentiments

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    logger.warning(f"NewsAPI rate limit hit for query '{query[:50]}...': {e}")
                    logger.info("Skipping this query to avoid further rate limits")
                else:
                    logger.warning(f"NewsAPI HTTP error for query '{query[:50]}...': {e}")
                return {"positive": 0, "negative": 0, "neutral": 0}
            except Exception as e:
                logger.warning(f"NewsAPI search failed for query '{query[:50]}...': {e}")
                return {"positive": 0, "negative": 0, "neutral": 0}

        try:
            # Use simple, efficient search strategy to avoid rate limits
            logger.info(f"Starting simple NewsAPI search for {ticker}")

            # Single focused query to avoid rate limits
            basic_query = self.build_search_query(ticker)
            result = search_with_query(basic_query)

            # If basic search fails, try one fallback
            if not self.has_meaningful_results(result):
                logger.info(f"Basic search failed, trying company name search for {ticker}")
                company_names = self.get_company_names(ticker.replace('.NS', ''))
                if company_names:
                    fallback_query = f'"{company_names[0]}" stock'
                    result = search_with_query(fallback_query)
                else:
                    result = {"positive": 0, "negative": 0, "neutral": 0}

            return result

        except Exception as e:
            logger.error(f"Error in enhanced NewsAPI sentiment for {ticker}: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(HTTPError))
    def _make_request(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response

    def gnews_sentiment(self, ticker):
        """Comprehensive GNews sentiment with global market factors and Indian focus"""
        def search_with_query(query):
            try:
                # Focus on last 24 hours for more actionable sentiment
                from datetime import datetime, timedelta
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

                encoded_query = quote(query)
                url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&country=in&from={yesterday}&token={self.GNEWS_API_KEY}"
                logger.debug(f"Requesting GNews API: {url}")

                response = self._make_request(url)
                data = response.json()
                logger.debug(f"Response status: {response.status_code}, content: {response.text[:200]}")

                sentiments = {"positive": 0, "negative": 0, "neutral": 0}
                if not data or "articles" not in data or not data["articles"]:
                    return sentiments

                logger.info(f"Found {len(data['articles'])} articles for query: {query[:50]}...")

                for article in data["articles"][:10]:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    content = f"{title} {description}"

                    if content.strip():
                        sentiment = self.sentiment_analyzer.polarity_scores(content)

                        # ENHANCED: Weight earnings/analyst news higher
                        weight = 1.0
                        earnings_keywords = ['earnings', 'quarterly', 'results', 'beat', 'miss', 'guidance',
                                           'analyst', 'upgrade', 'downgrade', 'target', 'rating']
                        if any(keyword in content.lower() for keyword in earnings_keywords):
                            weight = 2.0  # Double weight for earnings/analyst news

                        if sentiment["compound"] > 0.1:
                            sentiments["positive"] += weight
                        elif sentiment["compound"] < -0.1:
                            sentiments["negative"] += weight
                        else:
                            sentiments["neutral"] += weight

                return sentiments

            except Exception as e:
                logger.warning(f"GNews search failed for query '{query[:50]}...': {e}")
                return {"positive": 0, "negative": 0, "neutral": 0}

        try:
            # Use multi-level search strategy
            logger.info(f"Starting multi-level GNews search for {ticker}")
            result = self.multi_level_search(ticker, search_with_query, max_levels=3)

            # If multi-level search fails, try basic search as final fallback
            if not self.has_meaningful_results(result):
                logger.info(f"Multi-level search failed, trying basic search for {ticker}")
                basic_query = self.build_search_query(ticker)
                result = search_with_query(basic_query)

            return result

        except Exception as e:
            logger.error(f"Error in enhanced GNews sentiment for {ticker}: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def reddit_sentiment(self, ticker):
        if not self.reddit:
            return {"positive": 0, "negative": 0, "neutral": 0}
        try:
            subreddit = self.reddit.subreddit("all")
            query = f"${ticker}"
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for submission in subreddit.search(query, limit=10):
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:
                    sentiment = self.sentiment_analyzer.polarity_scores(comment.body)
                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def google_news_sentiment(self, ticker):
        try:
            # Use dynamic search query instead of raw ticker
            search_query = self.build_search_query(ticker)
            logger.info(f"Google News search query for {ticker}: {search_query}")

            news = self.google_news.get_news(search_query)
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}

            if not news:
                logger.warning(f"No Google News articles found for ticker {ticker}")
                return sentiments

            logger.info(f"Found {len(news)} Google News articles for {ticker}")

            for article in news[:10]:
                description = article.get("description", "")
                if description:
                    sentiment = self.sentiment_analyzer.polarity_scores(description)

                    # ENHANCED: Weight earnings/analyst news higher
                    weight = 1.0
                    earnings_keywords = ['earnings', 'quarterly', 'results', 'beat', 'miss', 'guidance',
                                       'analyst', 'upgrade', 'downgrade', 'target', 'rating']
                    if any(keyword in description.lower() for keyword in earnings_keywords):
                        weight = 2.0  # Double weight for earnings/analyst news

                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += weight
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += weight
                    else:
                        sentiments["neutral"] += weight
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Google News sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def fetch_combined_sentiment(self, ticker):
        """Comprehensive market sentiment analysis covering all factors affecting stock prices"""
        try:
            logger.info(f"Starting comprehensive sentiment analysis for {ticker}")

            # Get sentiment from all sources with enhanced coverage
            newsapi_sentiment = self.newsapi_sentiment(ticker)
            gnews_sentiment = self.gnews_sentiment(ticker)
            reddit_sentiment = self.reddit_sentiment(ticker)
            google_sentiment = self.google_news_sentiment(ticker)

            # ENHANCED: Dynamic weighted sentiment aggregation based on market impact
            # Higher weights for sources with better global and financial coverage
            weights = {
                "newsapi": 3.0,      # Highest weight - comprehensive global financial coverage
                "gnews": 2.5,        # High weight - good global and Indian coverage
                "google_news": 2.0,  # High weight - broad market coverage
                "reddit": 1.2        # Medium weight - social sentiment indicator
            }

            # Add market context weighting based on current market conditions
            market_context_multiplier = self.get_market_context_multiplier()
            for source in weights:
                weights[source] *= market_context_multiplier

            # Calculate weighted sentiment scores
            weighted_positive = (
                newsapi_sentiment["positive"] * weights["newsapi"] +
                gnews_sentiment["positive"] * weights["gnews"] +
                reddit_sentiment["positive"] * weights["reddit"] +
                google_sentiment["positive"] * weights["google_news"]
            )

            weighted_negative = (
                newsapi_sentiment["negative"] * weights["newsapi"] +
                gnews_sentiment["negative"] * weights["gnews"] +
                reddit_sentiment["negative"] * weights["reddit"] +
                google_sentiment["negative"] * weights["google_news"]
            )

            weighted_neutral = (
                newsapi_sentiment["neutral"] * weights["newsapi"] +
                gnews_sentiment["neutral"] * weights["gnews"] +
                reddit_sentiment["neutral"] * weights["reddit"] +
                google_sentiment["neutral"] * weights["google_news"]
            )

            # Traditional aggregation (for backward compatibility)
            aggregated = {
                "positive": (newsapi_sentiment["positive"] + gnews_sentiment["positive"] +
                            reddit_sentiment["positive"] + google_sentiment["positive"]),
                "negative": (newsapi_sentiment["negative"] + gnews_sentiment["negative"] +
                            reddit_sentiment["negative"] + google_sentiment["negative"]),
                "neutral": (newsapi_sentiment["neutral"] + gnews_sentiment["neutral"] +
                           reddit_sentiment["neutral"] + google_sentiment["neutral"])
            }

            # NEW: Comprehensive weighted aggregation with market context
            weighted_aggregated = {
                "positive": weighted_positive,
                "negative": weighted_negative,
                "neutral": weighted_neutral,
                "total_weight": sum(weights.values())
            }

            # Calculate sentiment confidence score based on source agreement
            total_sentiment = weighted_positive + weighted_negative + weighted_neutral
            if total_sentiment > 0:
                sentiment_confidence = max(weighted_positive, weighted_negative, weighted_neutral) / total_sentiment
            else:
                sentiment_confidence = 0.0

            # Add comprehensive market sentiment breakdown
            comprehensive_analysis = {
                "sentiment_strength": {
                    "bullish": weighted_positive / sum(weights.values()) if sum(weights.values()) > 0 else 0,
                    "bearish": weighted_negative / sum(weights.values()) if sum(weights.values()) > 0 else 0,
                    "neutral": weighted_neutral / sum(weights.values()) if sum(weights.values()) > 0 else 0
                },
                "confidence_score": sentiment_confidence,
                "market_context_applied": market_context_multiplier,
                "sources_coverage": {
                    "global_financial": "newsapi" in weights,
                    "indian_focus": "gnews" in weights,
                    "broad_market": "google_news" in weights,
                    "social_sentiment": "reddit" in weights
                }
            }

            return {
                "newsapi": newsapi_sentiment,
                "gnews": gnews_sentiment,
                "reddit": reddit_sentiment,
                "google_news": google_sentiment,
                "aggregated": aggregated,
                "weighted_aggregated": weighted_aggregated,
                "comprehensive_analysis": comprehensive_analysis  # NEW: Detailed market sentiment analysis
            }
        except Exception as e:
            logger.error(f"Error fetching comprehensive sentiment: {e}")
            return {
                "newsapi": {"positive": 0, "negative": 0, "neutral": 0},
                "gnews": {"positive": 0, "negative": 0, "neutral": 0},
                "reddit": {"positive": 0, "negative": 0, "neutral": 0},
                "google_news": {"positive": 0, "negative": 0, "neutral": 0},
                "aggregated": {"positive": 0, "negative": 0, "neutral": 0},
                "weighted_aggregated": {"positive": 0, "negative": 0, "neutral": 0, "total_weight": 8.7},
                "comprehensive_analysis": {
                    "sentiment_strength": {"bullish": 0, "bearish": 0, "neutral": 0},
                    "confidence_score": 0.0,
                    "market_context_applied": 1.0,
                    "sources_coverage": {
                        "global_financial": False,
                        "indian_focus": False,
                        "broad_market": False,
                        "social_sentiment": False
                    }
                }
            }

    def _generate_detailed_recommendation(self, stock_data, recommendation, buy_score, sell_score,
                                        price_to_sma200, trend_direction, sentiment_score,
                                        volatility, sharpe_ratio):
        explanation = f"Recommendation for {stock_data['name']} ({stock_data['symbol']}): {recommendation}\n"
        explanation += f"Current Price: Rs.{stock_data['current_price']['INR']:.2f}\n\n"
        explanation += f"Buy Score: {buy_score:.3f}, Sell Score: {sell_score:.3f}\n\n"

        if recommendation in ["STRONG BUY", "BUY"]:
            explanation += "Bullish Factors:\n"
            if price_to_sma200 < 1:
                explanation += "- Price is below the 200-day SMA, indicating potential undervaluation.\n"
            if trend_direction == "UPTREND":
                explanation += "- Stock is in an uptrend (50-day SMA > 200-day SMA).\n"
            if sentiment_score > 0.6:
                explanation += "- Positive market sentiment from news and social media.\n"
            if sharpe_ratio > 0:
                explanation += "- Positive risk-adjusted return (Sharpe Ratio).\n"
        elif recommendation in ["STRONG SELL", "SELL"]:
            explanation += "Bearish Factors:\n"
            if price_to_sma200 > 1:
                explanation += "- Price is above the 200-day SMA, suggesting potential overvaluation.\n"
            if trend_direction == "DOWNTREND":
                explanation += "- Stock is in a downtrend (50-day SMA < 200-day SMA).\n"
            if sentiment_score < 0.4:
                explanation += "- Negative market sentiment from news and social media.\n"
            if sharpe_ratio < 0:
                explanation += "- Negative risk-adjusted return (Sharpe Ratio).\n"
        else:
            explanation += "Neutral Factors:\n"
            explanation += "- Price is stable relative to moving averages.\n"
            explanation += "- Sentiment is balanced, with no strong bullish or bearish signals.\n"

        explanation += f"\nRisk Assessment:\n"
        explanation += f"- Volatility: {volatility:.4f} (higher values indicate higher risk)\n"
        explanation += f"- Sector: {stock_data.get('sector', 'N/A')}\n"
        explanation += f"- Industry: {stock_data.get('industry', 'N/A')}\n"

        return explanation

    def convert_df_to_dict(self, df):
        if df is None or df.empty:
            return {}
        try:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.where(pd.notnull(df), None)
            result = df.to_dict(orient='index')
            return {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in result.items()}
        except Exception as e:
            logger.error(f"Error converting DataFrame to dict: {e}")
            return {}

    def income_statement(self, ticker):
        try:
            # Financial data still uses yfinance (Fyers doesn't provide this)
            stock = yf.Ticker(ticker)
            income_stmt = stock.financials
            if income_stmt.empty:
                return {"success": False, "message": f"No income statement data for {ticker}"}
            income_dict = self.convert_df_to_dict(income_stmt)
            return {
                "success": True,
                "income_statement": income_dict
            }
        except Exception as e:
            logger.error(f"Error fetching income statement: {e}")
            return {"success": False, "message": f"Error fetching income statement: {str(e)}"}

    def balance_sheet(self, ticker):
        try:
            # Financial data still uses yfinance (Fyers doesn't provide this)
            stock = yf.Ticker(ticker)
            balance = stock.balance_sheet
            if balance.empty:
                return {"success": False, "message": f"No balance sheet data for {ticker}"}
            balance_dict = self.convert_df_to_dict(balance)
            return {
                "success": True,
                "balance_sheet": balance_dict
            }
        except Exception as e:
            logger.error(f"Error fetching balance sheet: {e}")
            return {"success": False, "message": f"Error fetching balance sheet: {str(e)}"}

    def cash_flow(self, ticker):
        try:
            # Financial data still uses yfinance (Fyers doesn't provide this)
            stock = yf.Ticker(ticker)
            cashflow = stock.cashflow
            if cashflow.empty:
                return {"success": False, "message": f"No cash flow data for {ticker}"}
            cashflow_dict = self.convert_df_to_dict(cashflow)
            return {
                "success": True,
                "cash_flow": cashflow_dict
            }
        except Exception as e:
            logger.error(f"Error fetching cash flow: {e}")
            return {"success": False, "message": f"Error fetching cash flow: {str(e)}"}

    def calculate_mpt_metrics(self, stock_history, benchmark_tickers):
        try:
            stock_returns = stock_history["Close"].pct_change().dropna()
            if stock_returns.empty:
                return {
                    "annual_return": "N/A",
                    "annual_volatility": "N/A",
                    "sharpe_ratio": "N/A",
                    "beta": "N/A",
                    "alpha": "N/A"
                }

            annual_return = stock_returns.mean() * 252
            annual_volatility = stock_returns.std() * np.sqrt(252)
            risk_free_rate = 0.06  # Adjusted for Indian risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else "N/A"

            beta = "N/A"
            alpha = "N/A"
            for benchmark_ticker in benchmark_tickers:
                # Use Fyers for benchmark data too - SAME LOGIC
                benchmark_history = get_stock_data_fyers_or_yf(benchmark_ticker, period="1y")
                if benchmark_history is not None and not benchmark_history.empty:
                    benchmark_returns = benchmark_history["Close"].pct_change().dropna()
                else:
                    benchmark_returns = pd.Series()

                if not benchmark_returns.empty:
                    aligned_returns = pd.concat([stock_returns, benchmark_returns], axis=1, join='inner')
                    if not aligned_returns.empty:
                        stock_ret = aligned_returns.iloc[:, 0]
                        bench_ret = aligned_returns.iloc[:, 1]
                        covariance = stock_ret.cov(bench_ret)
                        benchmark_variance = bench_ret.var()
                        beta = covariance / benchmark_variance if benchmark_variance != 0 else "N/A"
                        market_return = bench_ret.mean() * 252
                        alpha = annual_return - (risk_free_rate + beta * (market_return - risk_free_rate)) if beta != "N/A" else "N/A"
                        break

            return {
                "annual_return": float(annual_return) if not pd.isna(annual_return) else "N/A",
                "annual_volatility": float(annual_volatility) if not pd.isna(annual_volatility) else "N/A",
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio != "N/A" else "N/A",
                "beta": float(beta) if beta != "N/A" else "N/A",
                "alpha": float(alpha) if alpha != "N/A" else "N/A"
            }
        except Exception as e:
            logger.error(f"Error calculating MPT metrics: {e}")
            return {
                "annual_return": "N/A",
                "annual_volatility": "N/A",
                "sharpe_ratio": "N/A",
                "beta": "N/A",
                "alpha": "N/A"
            }

    def generate_adversarial_financial_data(self, history, epsilon=0.05, noise_factor=0.1, event_prob=0.1):
        try:
            adv_history = history.copy()
            price_std = history['Close'].std()
            volume_std = history['Volume'].std()
            
            min_price = 0.01
            max_price = history['Close'].max() * 2
            max_volume = history['Volume'].max() * 10
            
            for i in range(len(adv_history)):
                perturbation = np.random.uniform(-epsilon, epsilon) * price_std
                noise = np.random.normal(0, noise_factor * price_std)
                
                adv_history['Close'].iloc[i] += perturbation + noise
                adv_history['Open'].iloc[i] += perturbation + noise
                adv_history['High'].iloc[i] = max(adv_history['High'].iloc[i] + perturbation + noise, 
                                                adv_history['Open'].iloc[i], 
                                                adv_history['Close'].iloc[i])
                adv_history['Low'].iloc[i] = min(adv_history['Low'].iloc[i] + perturbation + noise, 
                                               adv_history['Open'].iloc[i], 
                                               adv_history['Close'].iloc[i])
                
                volume_perturbation = np.random.uniform(-epsilon, epsilon) * volume_std
                adv_history['Volume'].iloc[i] = max(0, adv_history['Volume'].iloc[i] + volume_perturbation)
                
                if np.random.random() < event_prob:
                    event_type = np.random.choice(['crash', 'spike'])
                    if event_type == 'crash':
                        drop_factor = np.random.uniform(0.85, 0.95)
                        adv_history['Close'].iloc[i] *= drop_factor
                        adv_history['Open'].iloc[i] *= drop_factor
                        adv_history['High'].iloc[i] *= drop_factor
                        adv_history['Low'].iloc[i] *= drop_factor
                        adv_history['Volume'].iloc[i] *= 1.5
                    else:
                        spike_factor = np.random.uniform(1.05, 1.15)
                        adv_history['Close'].iloc[i] *= spike_factor
                        adv_history['Open'].iloc[i] *= spike_factor
                        adv_history['High'].iloc[i] *= spike_factor
                        adv_history['Low'].iloc[i] *= spike_factor
                        adv_history['Volume'].iloc[i] *= 1.3
                
                adv_history['Close'].iloc[i] = np.clip(adv_history['Close'].iloc[i], min_price, max_price)
                adv_history['Open'].iloc[i] = np.clip(adv_history['Open'].iloc[i], min_price, max_price)
                adv_history['High'].iloc[i] = np.clip(adv_history['High'].iloc[i], min_price, max_price)
                adv_history['Low'].iloc[i] = np.clip(adv_history['Low'].iloc[i], min_price, max_price)
                adv_history['Volume'].iloc[i] = np.clip(adv_history['Volume'].iloc[i], 0, max_volume)
            
            adv_history.replace([np.inf, -np.inf], np.nan, inplace=True)
            adv_history.fillna({
                'Close': history['Close'].mean(),
                'Open': history['Open'].mean(),
                'High': history['High'].mean(),
                'Low': history['Low'].mean(),
                'Volume': history['Volume'].mean()
            }, inplace=True)
            
            return adv_history
        
        except Exception as e:
            logger.error(f"Error generating adversarial data: {e}")
            return history

    def train_rl_with_adversarial_events(self, history, ml_predicted_price, current_price,
                                       num_episodes=100, adversarial_freq=0.2, max_event_magnitude=0.1, bot_running=True):  # INDUSTRY LEVEL: 1000 episodes for production
        try:
            history = history.copy()
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()

            def calculate_rsi(data, periods=14):
                delta = data.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down.where(roll_down != 0, 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi.clip(0, 100)

            history["RSI"] = calculate_rsi(history["Close"])

            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2

            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()

            history.fillna({
                "SMA_50": history["Close"],
                "SMA_200": history["Close"],
                "RSI": 50,
                "MACD": 0,
                "Volatility": 0
            }, inplace=True)

            class AdversarialStockTradingEnv(gym.Env):
                def __init__(self, history, current_price, ml_predicted_price, 
                           adversarial_freq, max_event_magnitude):
                    super(AdversarialStockTradingEnv, self).__init__()
                    self.history = history
                    self.current_price = current_price
                    self.ml_predicted_price = ml_predicted_price
                    self.adversarial_freq = adversarial_freq
                    self.max_event_magnitude = max_event_magnitude
                    self.max_steps = len(history) - 1
                    self.current_step = 0
                    self.initial_balance = 100000
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    self.max_shares = 100
                    
                    self.action_space = spaces.Discrete(3)
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
                    )
                    
                def reset(self):
                    self.current_step = 0
                    self.balance = self.initial_balance
                    self.shares_held = 0
                    self.net_worth = self.initial_balance
                    self.event_occurred = 0
                    return self._get_observation()
                
                def _get_observation(self):
                    price = float(self.history["Close"].iloc[self.current_step])
                    sma_50 = float(self.history["SMA_50"].iloc[self.current_step] or 0)
                    sma_200 = float(self.history["SMA_200"].iloc[self.current_step] or 0)
                    rsi = float(self.history["RSI"].iloc[self.current_step] or 50)
                    macd = float(self.history["MACD"].iloc[self.current_step] or 0)
                    volatility = float(self.history["Volatility"].iloc[self.current_step] or 0)
                    ml_pred = self.ml_predicted_price if self.current_step == self.max_steps else price
                    return np.array([
                        price, sma_50, sma_200, rsi, macd, volatility,
                        self.balance, self.shares_held, self.net_worth, ml_pred,
                        self.event_occurred
                    ], dtype=np.float32)
                
                def step(self, action):
                    current_price = float(self.history["Close"].iloc[self.current_step])
                    reward = 0
                    
                    if np.random.random() < self.adversarial_freq:
                        event_magnitude = np.random.uniform(-self.max_event_magnitude, 
                                                          self.max_event_magnitude)
                        current_price *= (1 + event_magnitude)
                        self.event_occurred = abs(event_magnitude)
                    else:
                        self.event_occurred = 0
                    
                    if action == 1:
                        shares_to_buy = min(self.max_shares - self.shares_held, 
                                          int(self.balance / current_price))
                        cost = shares_to_buy * current_price
                        if cost <= self.balance:
                            self.balance -= cost
                            self.shares_held += shares_to_buy
                    elif action == 2:
                        shares_to_sell = self.shares_held
                        if shares_to_sell > 0:
                            revenue = shares_to_sell * current_price
                            self.balance += revenue
                            self.shares_held = 0
                            
                    self.net_worth = self.balance + self.shares_held * current_price
                    reward = self.net_worth - self.initial_balance
                    
                    self.current_step += 1
                    done = self.current_step >= self.max_steps
                    
                    if done:
                        reward += (self.ml_predicted_price - current_price) * self.shares_held
                        
                    return self._get_observation(), reward, done, {}
            
            env = AdversarialStockTradingEnv(
                history=history,
                current_price=current_price,
                ml_predicted_price=ml_predicted_price,
                adversarial_freq=adversarial_freq,
                max_event_magnitude=max_event_magnitude
            )
            
            class AdversarialDQNAgent:
                def __init__(self, state_size, action_size):
                    self.state_size = state_size
                    self.action_size = action_size

                    # INDUSTRY LEVEL: Deep Q-Network setup
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.q_network = self._build_dqn().to(self.device)
                    self.target_network = self._build_dqn().to(self.device)
                    self.optimizer = optim.AdamW(self.q_network.parameters(), lr=0.0001, weight_decay=1e-5)

                    # INDUSTRY LEVEL: Experience replay buffer
                    self.memory = []
                    self.memory_size = 50000
                    self.batch_size = 64

                    # INDUSTRY LEVEL: Advanced hyperparameters
                    self.gamma = 0.99
                    self.epsilon = 1.0
                    self.epsilon_min = 0.01
                    self.epsilon_decay = 0.9995
                    self.tau = 0.005  # Soft update parameter
                    self.update_frequency = 4
                    self.step_count = 0

                def _build_dqn(self):
                    # INDUSTRY LEVEL: Deep Q-Network architecture
                    return nn.Sequential(
                        nn.Linear(self.state_size, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(128, self.action_size)
                    )

                def remember(self, state, action, reward, next_state, done):
                    # INDUSTRY LEVEL: Experience replay
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)
                    self.memory.append((state, action, reward, next_state, done))

                def get_action(self, state):
                    # INDUSTRY LEVEL: Epsilon-greedy with neural network
                    if np.random.random() < self.epsilon:
                        return np.random.randint(self.action_size)

                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()

                def replay(self):
                    # INDUSTRY LEVEL: Experience replay training
                    if len(self.memory) < self.batch_size:
                        return

                    batch = random.sample(self.memory, self.batch_size)
                    states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
                    actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
                    rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
                    next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
                    dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

                    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                    next_q_values = self.target_network(next_states).max(1)[0].detach()
                    target_q_values = rewards + (self.gamma * next_q_values * ~dones)

                    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                    self.optimizer.step()

                    # INDUSTRY LEVEL: Epsilon decay
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay

                def soft_update_target_network(self):
                    # INDUSTRY LEVEL: Soft update of target network
                    for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                        target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

                def update(self, state, action, reward, next_state, done=False):
                    # INDUSTRY LEVEL: Store experience and train
                    self.remember(state, action, reward, next_state, done)
                    self.step_count += 1

                    if self.step_count % self.update_frequency == 0:
                        self.replay()
                        self.soft_update_target_network()
            
            agent = AdversarialDQNAgent(state_size=11, action_size=3)  # INDUSTRY LEVEL: Deep Q-Network agent
            
            total_rewards = []
            event_counts = []
            epoch_logs = []
            
            logger.info(f"Training adversarial RL agent...")
            for episode in range(num_episodes):
                # Check if bot should stop
                if not bot_running:
                    logger.info("Bot stop signal received, stopping RL training...")
                    break

                state = env.reset()
                total_reward = 0
                done = False
                episode_events = 0

                while not done:
                    # Check if bot should stop during episode
                    if not bot_running:
                        logger.info("Bot stop signal received, stopping RL episode...")
                        break

                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.update(state, action, reward, next_state, done)  # INDUSTRY LEVEL: Include done parameter
                    state = next_state
                    total_reward += reward
                    if state[-1] > 0:
                        episode_events += 1

                # Break out of episode loop if bot should stop
                if not bot_running:
                    break

                total_rewards.append(total_reward)
                event_counts.append(episode_events)
                if (episode + 1) % 10 == 0:
                    logger.info(f"Adversarial Episode {episode + 1}/{num_episodes}, "
                                f"Total Reward: {total_reward:.2f}, Events: {episode_events}")
                    epoch_logs.append({
                        "episode": episode + 1,
                        "total_reward": total_reward,
                        "average_reward": np.mean(total_rewards[-10:]),
                        "events_triggered": episode_events
                    })
            
            state = env.reset()
            done = False
            actions_taken = []
            net_worth_history = []
            
            while not done:
                action = agent.get_action(state)
                actions_taken.append(action)
                next_state, reward, done, _ = env.step(action)
                net_worth_history.append(env.net_worth)
                state = next_state
                
            final_net_worth = net_worth_history[-1]
            performance = (final_net_worth - env.initial_balance) / env.initial_balance * 100
            
            recommendation = "BUY" if performance > 10 else "SELL" if performance < -5 else "HOLD"
            
            return {
                "success": True,
                "recommendation": recommendation,
                "performance_pct": float(performance),
                "final_net_worth": float(final_net_worth),
                "average_reward": float(np.mean(total_rewards)),
                "average_events_per_episode": float(np.mean(event_counts)),
                "actions_distribution": {
                    "hold": actions_taken.count(0) / len(actions_taken),
                    "buy": actions_taken.count(1) / len(actions_taken),
                    "sell": actions_taken.count(2) / len(actions_taken)
                },
                "epoch_logs": epoch_logs
            }
            
        except Exception as e:
            logger.error(f"Error in adversarial RL training: {e}")
            return {
                "success": False,
                "message": f"Error in adversarial RL training: {str(e)}"
            }

    def adversarial_training_loop(self, X_train, y_train, X_test, y_test, input_size,
                                 seq_length=20, num_epochs=50, adv_lambda=0.1, bot_running=True):  # INDUSTRY LEVEL: 150 epochs for production
        try:
            logger.info("Cleaning input data...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

            logger.info("Converting data to tensors...")
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device, requires_grad=True)
            y_train_tensor = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, 
                                        dtype=torch.float32, device=self.device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device, requires_grad=True)
            y_test_tensor = torch.tensor(y_test.values if isinstance(y_test, pd.Series) else y_test, 
                                        dtype=torch.float32, device=self.device)
            
            def create_sequences(x_data, y_data, seq_length):
                logger.info(f"Creating sequences with length {seq_length}...")
                logger.info(f"Input data shape: {x_data.shape}, Target data shape: {y_data.shape}")
                
                # Check if we have enough data to create sequences
                if len(x_data) <= seq_length:
                    logger.warning(f"Insufficient data for sequence creation. Data length: {len(x_data)}, Required: {seq_length + 1}")
                    # Return empty tensors with correct shape for compatibility
                    empty_x = torch.empty((0, seq_length, x_data.shape[1]), device=self.device, dtype=torch.float32)
                    empty_y = torch.empty((0,), device=self.device, dtype=torch.float32)
                    empty_x.requires_grad_(True)
                    return empty_x, empty_y
                
                xs, ys = [], []
                available_sequences = len(x_data) - seq_length
                logger.info(f"Can create {available_sequences} sequences")
                
                for i in range(available_sequences):
                    seq = x_data[i:i+seq_length].detach().clone()
                    seq.requires_grad_(True)
                    xs.append(seq)
                    ys.append(y_data[i+seq_length])
                
                if len(xs) == 0:
                    logger.warning("No sequences could be created")
                    # Return empty tensors with correct shape for compatibility
                    empty_x = torch.empty((0, seq_length, x_data.shape[1]), device=self.device, dtype=torch.float32)
                    empty_y = torch.empty((0,), device=self.device, dtype=torch.float32)
                    empty_x.requires_grad_(True)
                    return empty_x, empty_y
                
                xs_tensor = torch.stack(xs).to(self.device)
                ys_tensor = torch.stack(ys).to(self.device)
                xs_tensor.requires_grad_(True)
                logger.info(f"Created sequences - X shape: {xs_tensor.shape}, Y shape: {ys_tensor.shape}")
                return xs_tensor, ys_tensor
            
            logger.info("Generating training sequences...")
            X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, seq_length)
            logger.info("Generating test sequences...")
            X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, seq_length)
            
            # Check if we have any sequences to train with
            if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
                logger.error(f"Insufficient data for adversarial training. Train sequences: {X_train_seq.shape[0]}, Test sequences: {X_test_seq.shape[0]}")
                logger.warning("Skipping adversarial training due to insufficient data. Returning None.")
                return None
            
            train_dataset = TensorDataset(X_train_seq, y_train_seq)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # INDUSTRY LEVEL: Increased batch size
            test_dataset = TensorDataset(X_test_seq, y_test_seq)
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # INDUSTRY LEVEL: Increased batch size
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=1):  # INDUSTRY LEVEL: Increased capacity
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    # INDUSTRY LEVEL: Bidirectional LSTM with higher dropout
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                      dropout=0.3, bidirectional=True)
                    # INDUSTRY LEVEL: Multiple FC layers with batch norm and dropout
                    self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
                    self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
                    self.dropout1 = nn.Dropout(0.4)
                    self.batch_norm2 = nn.BatchNorm1d(hidden_size)
                    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                    self.dropout2 = nn.Dropout(0.3)
                    self.fc3 = nn.Linear(hidden_size // 2, output_size)
                    self.relu = nn.ReLU()
                    self.leaky_relu = nn.LeakyReLU(0.1)
                    
                def forward(self, x):
                    # INDUSTRY LEVEL: Proper initialization for bidirectional LSTM
                    h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
                    c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional

                    # LSTM forward pass
                    out, _ = self.lstm(x, (h0, c0))

                    # INDUSTRY LEVEL: Advanced forward pass with residual connections
                    out = out[:, -1, :]  # Take last output

                    # First layer with batch norm and dropout
                    out = self.batch_norm1(out)
                    out = self.fc1(out)
                    out = self.leaky_relu(out)
                    out = self.dropout1(out)

                    # Second layer with batch norm and dropout
                    out = self.batch_norm2(out)
                    out = self.fc2(out)
                    out = self.relu(out)
                    out = self.dropout2(out)

                    # Final output layer
                    out = self.fc3(out)
                    return out
            
            class TransformerModel(nn.Module):
                def __init__(self, input_size, seq_length, num_heads=4, dim_feedforward=512, num_layers=3, output_size=1):  # INDUSTRY LEVEL: Increased capacity
                    super(TransformerModel, self).__init__()
                    # INDUSTRY LEVEL: Proper d_model sizing
                    d_model = 128  # Standard transformer dimension
                    adjusted_input_size = d_model

                    # INDUSTRY LEVEL: Advanced transformer with dropout and layer norm
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=adjusted_input_size,
                        nhead=num_heads,
                        dim_feedforward=dim_feedforward,
                        dropout=0.2,  # INDUSTRY LEVEL: Proper dropout
                        activation='gelu',  # INDUSTRY LEVEL: GELU activation
                        batch_first=True,
                        norm_first=True  # INDUSTRY LEVEL: Pre-norm architecture
                    )
                    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

                    # INDUSTRY LEVEL: Advanced output layers with attention pooling
                    self.input_proj = nn.Linear(input_size, adjusted_input_size)
                    self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, adjusted_input_size))

                    # INDUSTRY LEVEL: Multi-head attention pooling
                    self.attention_pool = nn.MultiheadAttention(adjusted_input_size, num_heads, batch_first=True)
                    self.layer_norm = nn.LayerNorm(adjusted_input_size)

                    # INDUSTRY LEVEL: Advanced output network
                    self.output_net = nn.Sequential(
                        nn.Linear(adjusted_input_size, dim_feedforward),
                        nn.GELU(),
                        nn.Dropout(0.3),
                        nn.Linear(dim_feedforward, dim_feedforward // 2),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(dim_feedforward // 2, output_size)
                    )

                    self.input_size = input_size
                    self.seq_length = seq_length
                    self.d_model = adjusted_input_size
                    
                def forward(self, x):
                    # INDUSTRY LEVEL: Advanced forward pass with positional encoding
                    batch_size, seq_len, _ = x.shape

                    # Project input and add positional encoding
                    x = self.input_proj(x)
                    x = x + self.positional_encoding[:, :seq_len, :]

                    # Transformer encoding
                    x = self.transformer_encoder(x)

                    # INDUSTRY LEVEL: Attention pooling instead of simple averaging
                    # Use the last token as query for attention pooling
                    query = x[:, -1:, :]  # Last token as query
                    pooled_output, _ = self.attention_pool(query, x, x)
                    pooled_output = pooled_output.squeeze(1)  # Remove sequence dimension

                    # Layer normalization and output
                    pooled_output = self.layer_norm(pooled_output)
                    output = self.output_net(pooled_output)

                    return output
            
            logger.info("Initializing LSTM and Transformer models...")
            lstm_model = LSTMModel(input_size=input_size).to(self.device)
            transformer_model = TransformerModel(input_size=input_size, seq_length=seq_length).to(self.device)
            
            # INDUSTRY LEVEL: Advanced loss function and optimizers
            criterion = nn.SmoothL1Loss()  # More robust than MSE for financial data

            # INDUSTRY LEVEL: AdamW with weight decay and learning rate scheduling
            lstm_optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
            transformer_optimizer = optim.AdamW(transformer_model.parameters(), lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999))

            # INDUSTRY LEVEL: Learning rate schedulers
            lstm_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(lstm_optimizer, T_0=10, T_mult=2)
            transformer_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(transformer_optimizer, T_0=10, T_mult=2)
            
            def fgsm_attack(data, epsilon, data_grad):
                if data_grad is None:
                    logger.warning("data_grad is None, skipping perturbation")
                    return data
                sign_data_grad = data_grad.sign()
                perturbed_data = data + epsilon * sign_data_grad
                perturbed_data = perturbed_data.detach().requires_grad_(True)
                return perturbed_data
            
            lstm_logs = {}
            transformer_logs = {}

            # INDUSTRY LEVEL: Early stopping and best model tracking
            best_lstm_loss = float('inf')
            best_transformer_loss = float('inf')
            patience = 20
            patience_counter = 0

            logger.info("Starting adversarial training for LSTM and Transformer...")
            for epoch in range(num_epochs):
                # Check if bot should stop
                if not bot_running:
                    logger.info("Bot stop signal received, stopping adversarial training...")
                    break

                lstm_model.train()
                transformer_model.train()
                lstm_running_loss = 0.0
                transformer_running_loss = 0.0

                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    # Check if bot should stop during batch processing
                    if not bot_running:
                        logger.info("Bot stop signal received, stopping batch processing...")
                        break

                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.clone().detach().requires_grad_(True)

                    lstm_optimizer.zero_grad()
                    with torch.enable_grad():
                        lstm_outputs = lstm_model(inputs)
                        lstm_loss = criterion(lstm_outputs.squeeze(), labels)

                        lstm_loss.backward(retain_graph=True)
                        data_grad = inputs.grad
                        if data_grad is None:
                            logger.warning(f"LSTM data_grad is None in batch {batch_idx}, skipping adversarial step")
                            perturbed_inputs = inputs.clone().detach().requires_grad_(True)
                        else:
                            perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)

                        lstm_optimizer.zero_grad()
                        lstm_adv_outputs = lstm_model(perturbed_inputs)
                        lstm_adv_loss = criterion(lstm_adv_outputs.squeeze(), labels)

                        lstm_total_loss = lstm_loss + adv_lambda * lstm_adv_loss
                        lstm_total_loss.backward()

                    # INDUSTRY LEVEL: Gradient clipping
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    lstm_optimizer.step()
                    lstm_running_loss += lstm_total_loss.item()

                    transformer_optimizer.zero_grad()
                    inputs = inputs.clone().detach().requires_grad_(True)
                    with torch.enable_grad():
                        transformer_outputs = transformer_model(inputs)
                        transformer_loss = criterion(transformer_outputs.squeeze(), labels)

                        transformer_loss.backward(retain_graph=True)
                        data_grad = inputs.grad
                        if data_grad is None:
                            logger.warning(f"Transformer data_grad is None in batch {batch_idx}, skipping adversarial step")
                            perturbed_inputs = inputs.clone().detach().requires_grad_(True)
                        else:
                            perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)

                        transformer_optimizer.zero_grad()
                        transformer_adv_outputs = transformer_model(perturbed_inputs)
                        transformer_adv_loss = criterion(transformer_adv_outputs.squeeze(), labels)

                        transformer_total_loss = transformer_loss + adv_lambda * transformer_adv_loss
                        transformer_total_loss.backward()

                    # INDUSTRY LEVEL: Gradient clipping
                    torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
                    transformer_optimizer.step()
                    transformer_running_loss += transformer_total_loss.item()

                # Break out of epoch loop if bot should stop
                if not bot_running:
                    break

                # INDUSTRY LEVEL: Calculate epoch losses and update schedulers
                lstm_epoch_loss = lstm_running_loss / len(train_loader)
                transformer_epoch_loss = transformer_running_loss / len(train_loader)

                # INDUSTRY LEVEL: Learning rate scheduling
                lstm_scheduler.step()
                transformer_scheduler.step()

                # INDUSTRY LEVEL: Early stopping logic
                current_loss = (lstm_epoch_loss + transformer_epoch_loss) / 2
                if current_loss < best_lstm_loss:
                    best_lstm_loss = current_loss
                    best_transformer_loss = transformer_epoch_loss
                    patience_counter = 0
                    # Save best models (optional - can be added if needed)
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                                f'LSTM Loss: {lstm_epoch_loss:.4f}, '
                                f'Transformer Loss: {transformer_epoch_loss:.4f}, '
                                f'LR: {lstm_optimizer.param_groups[0]["lr"]:.6f}')
                    lstm_logs[f"Epoch_{epoch+1}"] = lstm_epoch_loss
                    transformer_logs[f"Epoch_{epoch+1}"] = transformer_epoch_loss

                # INDUSTRY LEVEL: Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs")
                    break
            
            lstm_model.eval()
            transformer_model.eval()
            lstm_preds = []
            transformer_preds = []
            
            logger.info("Evaluating models...")
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    lstm_outputs = lstm_model(inputs)
                    lstm_preds.extend(lstm_outputs.squeeze().cpu().tolist())
                    transformer_outputs = transformer_model(inputs)
                    transformer_preds.extend(transformer_outputs.squeeze().cpu().tolist())
            
            lstm_mse = mean_squared_error(y_test_seq.cpu().numpy(), lstm_preds)
            lstm_r2 = r2_score(y_test_seq.cpu().numpy(), lstm_preds)
            transformer_mse = mean_squared_error(y_test_seq.cpu().numpy(), transformer_preds)
            transformer_r2 = r2_score(y_test_seq.cpu().numpy(), transformer_preds)
            
            lstm_adv_preds = []
            transformer_adv_preds = []
            
            logger.info("Evaluating adversarial robustness...")
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device).clone().detach().requires_grad_(True)
                lstm_outputs = lstm_model(inputs)
                lstm_loss = criterion(lstm_outputs.squeeze(), labels.to(self.device))
                lstm_loss.backward(retain_graph=True)
                data_grad = inputs.grad
                perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                
                lstm_adv_outputs = lstm_model(perturbed_inputs)
                lstm_adv_preds.extend(lstm_adv_outputs.squeeze().cpu().tolist())
                
                inputs = inputs.clone().detach().requires_grad_(True)
                transformer_outputs = transformer_model(inputs)
                transformer_loss = criterion(transformer_outputs.squeeze(), labels.to(self.device))
                transformer_loss.backward(retain_graph=True)
                data_grad = inputs.grad
                perturbed_inputs = fgsm_attack(inputs, epsilon=0.1, data_grad=data_grad)
                
                transformer_adv_outputs = transformer_model(perturbed_inputs)
                transformer_adv_preds.extend(transformer_adv_outputs.squeeze().cpu().tolist())
            
            lstm_adv_mse = mean_squared_error(y_test_seq.cpu().numpy(), lstm_adv_preds)
            lstm_adv_r2 = r2_score(y_test_seq.cpu().numpy(), lstm_adv_preds)
            transformer_adv_mse = mean_squared_error(y_test_seq.cpu().numpy(), transformer_adv_preds)
            transformer_adv_r2 = r2_score(y_test_seq.cpu().numpy(), transformer_adv_preds)
            
            return {
                "success": True,
                "lstm_metrics": {
                    "mse": float(lstm_mse),
                    "r2": float(lstm_r2),
                    "adv_mse": float(lstm_adv_mse),
                    "adv_r2": float(lstm_adv_r2)
                },
                "transformer_metrics": {
                    "mse": float(transformer_mse),
                    "r2": float(transformer_r2),
                    "adv_mse": float(transformer_adv_mse),
                    "adv_r2": float(transformer_adv_r2)
                },
                "lstm_model": lstm_model,
                "transformer_model": transformer_model,
                "lstm_epoch_logs": lstm_logs,
                "transformer_epoch_logs": transformer_logs
            }
            
        except Exception as e:
            logger.error(f"Error in adversarial training loop: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error in adversarial training: {str(e)}",
                "lstm_epoch_logs": {},
                "transformer_epoch_logs": {}
            }

    def analyze_stock(self, ticker, benchmark_tickers=None, prediction_days=30, training_period="2y", bot_running=True):
        try:
            # Check if bot should stop before starting analysis
            if not bot_running:
                logger.info(f"Bot stop signal received, skipping analysis for {ticker}")
                return {
                    "success": False,
                    "message": "Bot stopped during analysis"
                }

            ticker = ticker.strip().upper()
            logger.info(f"Fetching and analyzing data for {ticker}...")

            # Use Fyers for historical data (1y) - SAME LOGIC
            history = get_stock_data_fyers_or_yf(ticker, period="1y")

            if history is None or history.empty:
                logger.error(f"No price data found for {ticker}.")
                return {
                    "success": False,
                    "message": f"Unable to fetch data for {ticker}: No price data found"
                }

            # Get stock info for additional data (still use yfinance for company info)
            try:
                stock = yf.Ticker(ticker)
                stock_info = stock.info
            except Exception as e:
                logger.warning(f"Could not fetch stock info for {ticker}: {e}")
                stock_info = {}
            current_price = float(history["Close"].iloc[-1])
            exchange_rates = self.fetch_exchange_rates()
            converted_prices = self.convert_price(current_price, exchange_rates)
            # Helper function to safely convert values to float
            def safe_float(value, default=0.0):
                if isinstance(value, str):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return default
            
            market_cap = safe_float(stock_info.get("marketCap", 0.0), 0.0)
            volume = safe_float(stock_info.get("volume", 0.0), 0.0)
            pe_ratio = safe_float(stock_info.get("trailingPE", 0.0), 0.0)
            dividends = stock_info.get("dividendYield", 0.0)
            dividend_yield = safe_float(dividends, 0.0) * 100
            high_52w = safe_float(stock_info.get("fiftyTwoWeekHigh", 0.0), 0.0)
            low_52w = safe_float(stock_info.get("fiftyTwoWeekLow", 0.0), 0.0)
            sector = stock_info.get("sector", "N/A")
            industry = stock_info.get("industry", "N/A")

            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()
            history["EMA_50"] = history["Close"].ewm(span=50, adjust=False).mean()

            def calculate_rsi(data, periods=14):
                delta = data.diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                roll_up = up.ewm(com=periods-1, adjust=False).mean()
                roll_down = down.ewm(com=periods-1, adjust=False).mean()
                rs = roll_up / roll_down.where(roll_down != 0, 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return rsi.clip(0, 100)

            history["RSI"] = calculate_rsi(history["Close"])

            history["BB_Middle"] = history["Close"].rolling(window=20).mean()
            history["BB_Upper"] = history["BB_Middle"] + 2 * history["Close"].rolling(window=20).std()
            history["BB_Lower"] = history["BB_Middle"] - 2 * history["Close"].rolling(window=20).std()

            exp1 = history["Close"].ewm(span=12, adjust=False).mean()
            exp2 = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = exp1 - exp2
            history["Signal_Line"] = history["MACD"].ewm(span=9, adjust=False).mean()
            history["MACD_Histogram"] = history["MACD"] - history["Signal_Line"]

            history["Daily_Return"] = history["Close"].pct_change()
            history["Volatility"] = history["Daily_Return"].rolling(window=30).std()

            mpt_metrics = self.calculate_mpt_metrics(history, benchmark_tickers or ['^NSEI'])

            risk_free_rate = 0.06
            sharpe_ratio = (history["Daily_Return"].mean() - risk_free_rate) / history["Daily_Return"].std() if history["Daily_Return"].std() != 0 else 0

            # Use safe_float for technical indicators
            sma_50 = safe_float(history["SMA_50"].iloc[-1], current_price) if not pd.isna(history["SMA_50"].iloc[-1]) else current_price
            sma_200 = safe_float(history["SMA_200"].iloc[-1], current_price) if not pd.isna(history["SMA_200"].iloc[-1]) else current_price
            ema_50 = safe_float(history["EMA_50"].iloc[-1], current_price) if not pd.isna(history["EMA_50"].iloc[-1]) else current_price
            volatility = safe_float(history["Volatility"].iloc[-1], 0) if not pd.isna(history["Volatility"].iloc[-1]) else 0
            rsi = safe_float(history["RSI"].iloc[-1], 50) if not pd.isna(history["RSI"].iloc[-1]) else 50
            bb_upper = safe_float(history["BB_Upper"].iloc[-1], current_price * 1.1) if not pd.isna(history["BB_Upper"].iloc[-1]) else current_price * 1.1
            bb_lower = safe_float(history["BB_Lower"].iloc[-1], current_price * 0.9) if not pd.isna(history["BB_Lower"].iloc[-1]) else current_price * 0.9
            macd = safe_float(history["MACD"].iloc[-1], 0) if not pd.isna(history["MACD"].iloc[-1]) else 0
            signal_line = safe_float(history["Signal_Line"].iloc[-1], 0) if not pd.isna(history["Signal_Line"].iloc[-1]) else 0
            macd_histogram = safe_float(history["MACD_Histogram"].iloc[-1], 0) if not pd.isna(history["MACD_Histogram"].iloc[-1]) else 0

            momentum = 0
            if len(history) >= 30:
                momentum = (current_price - history["Close"].iloc[-30]) / history["Close"].iloc[-30]

            logger.info(f"Fetching sentiment for {ticker}...")
            sentiment_data = self.fetch_combined_sentiment(ticker)

            # ENHANCED: Use weighted sentiment for better accuracy
            if "weighted_aggregated" in sentiment_data and sentiment_data["weighted_aggregated"]["total_weight"] > 0:
                sentiment = sentiment_data["weighted_aggregated"]
                total_sentiment = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
                sentiment_score = sentiment["positive"] / total_sentiment if total_sentiment > 0 else 0.5
                logger.debug(f"Using weighted sentiment score: {sentiment_score:.3f}")
            else:
                # Fallback to regular aggregation
                sentiment = sentiment_data["aggregated"]
                total_sentiment = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
                sentiment_score = sentiment["positive"] / total_sentiment if total_sentiment > 0 else 0.5
                logger.debug(f"Using regular sentiment score: {sentiment_score:.3f}")

            price_to_sma200 = current_price / sma_200 if sma_200 > 0 else 1
            # price_to_sma50 calculated but not used in current logic
            trend_direction = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"
            volume_trend = "HIGH" if isinstance(volume, (int, float)) and volume > 1000000 else "MODERATE"

            logger.info(f"Fetching institutional investments data for {ticker}...")
            
            # Initialize variables with fallback values
            institutional_holders = None
            major_holders = None
            mutual_fund_holders = None
            
            try:
                # Rate limit handling with exponential backoff (per project specifications)
                max_retries = 3
                base_delay = 2  # Start with 2 seconds
                
                for attempt in range(max_retries):
                    try:
                        # Fetch institutional data with rate limit protection
                        institutional_holders = stock.institutional_holders
                        major_holders = stock.major_holders
                        
                        # Add 0.5s delay between related API calls to reduce rate limiting pressure
                        time.sleep(0.5)
                        mutual_fund_holders = stock.mutualfund_holders
                        
                        logger.info(f"✅ Successfully fetched institutional data for {ticker} on attempt {attempt + 1}")
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_message = str(e).lower()
                        
                        # Check for rate limiting errors with comprehensive detection patterns
                        if any(phrase in error_message for phrase in ["rate limited", "too many requests", "429", "yfrateli"]):
                            if attempt < max_retries - 1:  # Don't sleep on last attempt
                                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                                logger.warning(f"⏱️ Rate limited for {ticker}. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
                                time.sleep(delay)
                                continue
                            else:
                                logger.error(f"❌ Rate limit exceeded for {ticker} after {max_retries} attempts. Using fallback data.")
                                break
                        else:
                            # Non-rate limit error, log and use fallback
                            logger.error(f"❌ Error fetching institutional data for {ticker}: {e}")
                            break
                else:
                    # All retries failed due to rate limiting
                    logger.error(f"❌ All retry attempts failed for {ticker}. Using fallback data.")
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error fetching institutional data for {ticker}: {e}")
            
            # Continue with processing using fallback logic for system resilience
            institutional_data = {}
            if institutional_holders is not None and not institutional_holders.empty:
                top_institutional = institutional_holders.head(5)
                institutional_data["top_holders"] = []
                for _, row in top_institutional.iterrows():
                    holder_data = {
                        "name": row["Holder"] if "Holder" in row else "Unknown",
                        "shares": row["Shares"] if "Shares" in row else 0,
                        "date_reported": str(row["Date Reported"]) if "Date Reported" in row else "Unknown",
                        "pct_out": round(float(row["% Out"]) * 100, 2) if "% Out" in row else 0,
                        "value": row["Value"] if "Value" in row else 0
                    }
                    institutional_data["top_holders"].append(holder_data)
                institutional_data["total_shares_held"] = institutional_holders["Shares"].sum() if "Shares" in institutional_holders else 0
                institutional_data["total_value"] = institutional_holders["Value"].sum() if "Value" in institutional_holders else 0

            if major_holders is not None and not major_holders.empty:
                try:
                    inst_value = major_holders.iloc[0, 0]
                    if isinstance(inst_value, str) and '%' in inst_value:
                        institutional_data["institutional_ownership_pct"] = float(inst_value.strip('%'))
                    else:
                        institutional_data["institutional_ownership_pct"] = float(inst_value)

                    insider_value = major_holders.iloc[1, 0]
                    if isinstance(insider_value, str) and '%' in insider_value:
                        institutional_data["insider_ownership_pct"] = float(insider_value.strip('%'))
                    else:
                        institutional_data["insider_ownership_pct"] = float(insider_value)
                except (IndexError, ValueError, AttributeError) as e:
                    logger.error(f"Error processing major holders data: {e}")
                    institutional_data["institutional_ownership_pct"] = 0
                    institutional_data["insider_ownership_pct"] = 0

            mutual_fund_data = {}
            if mutual_fund_holders is not None and not mutual_fund_holders.empty:
                top_mutual_funds = mutual_fund_holders.head(5)
                mutual_fund_data["top_holders"] = []
                for _, row in top_mutual_funds.iterrows():
                    holder_data = {
                        "name": row["Holder"] if "Holder" in row else "Unknown",
                        "shares": row["Shares"] if "Shares" in row else 0,
                        "date_reported": str(row["Date Reported"]) if "Date Reported" in row else "Unknown",
                        "pct_out": round(float(row["% Out"]) * 100, 2) if "% Out" in row else 0,
                        "value": row["Value"] if "Value" in row else 0
                    }
                    mutual_fund_data["top_holders"].append(holder_data)
                mutual_fund_data["total_shares_held"] = mutual_fund_holders["Shares"].sum() if "Shares" in mutual_fund_holders else 0
                mutual_fund_data["total_value"] = mutual_fund_holders["Value"].sum() if "Value" in mutual_fund_holders else 0

            institutional_confidence = 0
            if institutional_data.get("institutional_ownership_pct", 0) > 70:
                institutional_confidence = 0.3
            elif institutional_data.get("institutional_ownership_pct", 0) > 50:
                institutional_confidence = 0.2
            elif institutional_data.get("institutional_ownership_pct", 0) > 30:
                institutional_confidence = 0.1
            if institutional_data.get("insider_ownership_pct", 0) > 20:
                institutional_confidence += 0.1

            # REMOVED OLD LOGIC: Early recommendation calculation that was causing circular logic bug
            # The proper scoring logic will handle recommendation generation later

            # Set default recommendation for technical analysis - will be overridden by proper scoring
            recommendation = "NEUTRAL"  # Neutral starting point for technical analysis

            support_level = safe_float(min(sma_200, sma_50) * 0.95, current_price * 0.95)
            resistance_level = safe_float(max(current_price * 1.05, sma_50 * 1.05), current_price * 1.05)

            if volatility > 0.03:
                risk_level = "HIGH"
            elif volatility > 0.015:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            timeframe = "LONG_TERM" if trend_direction == "UPTREND" and sentiment_score > 0.6 else "MEDIUM_TERM" if trend_direction == "UPTREND" else "SHORT_TERM"

            stock_data = {
                "symbol": ticker,
                "name": stock_info.get("shortName", ticker),
                "current_price": converted_prices,
                "sector": sector,
                "industry": industry,
                "pe_ratio": pe_ratio,
                "dividends": dividends
            }

            balance_sheet = self.balance_sheet(ticker)
            income_statement = self.income_statement(ticker)
            cash_flow = self.cash_flow(ticker)

            def filter_non_nan(data):
                return {k: v for k, v in data.items() if v not in ["N/A", "nan", None, float('nan'), "null", ""]}

            balance_sheet_filtered = filter_non_nan(balance_sheet.get("balance_sheet", {}))
            income_statement_filtered = filter_non_nan(income_statement.get("income_statement", {}))
            cash_flow_filtered = filter_non_nan(cash_flow.get("cash_flow", {}))

            explanation = self._generate_detailed_recommendation(
                stock_data, recommendation, 0.0, 0.0,  # Placeholder scores - real scores calculated later
                price_to_sma200, trend_direction, sentiment_score,
                volatility, sharpe_ratio
            )

            logger.info(f"Fetching extended data for {ticker} for ML analysis...")
            extended_history = get_stock_data_fyers_or_yf(ticker, period=training_period)

            if extended_history is None or extended_history.empty:
                logger.error(f"Unable to fetch sufficient extended historical data for {ticker}")
                ml_analysis = {
                    "success": False,
                    "message": f"Unable to fetch sufficient historical data for ML analysis of {ticker}"
                }
            else:
                logger.info(f"Generating adversarial financial data for {ticker}...")
                adv_history = self.generate_adversarial_financial_data(extended_history)

                combined_history = pd.concat([extended_history, adv_history]).reset_index(drop=True)

                logger.info(f"Engineering features for ML pattern recognition...")
                data = combined_history[['Close']].copy()

                data['SMA_5'] = combined_history['Close'].rolling(window=5).mean()
                data['SMA_20'] = combined_history['Close'].rolling(window=20).mean()
                data['SMA_50'] = combined_history['Close'].rolling(window=50).mean()
                data['SMA_200'] = combined_history['Close'].rolling(window=200).mean()

                data['EMA_12'] = combined_history['Close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = combined_history['Close'].ewm(span=26, adjust=False).mean()

                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

                delta = combined_history['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss.where(loss != 0, 1e-10)
                data['RSI'] = 100 - (100 / (1 + rs))
                data['RSI'] = data['RSI'].clip(0, 100)

                data['BB_Middle'] = data['SMA_20']
                stddev = combined_history['Close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Middle'] + 2 * stddev
                data['BB_Lower'] = data['BB_Middle'] - 2 * stddev
                data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle'].where(data['BB_Middle'] != 0, 1e-10)

                data['Volume_Change'] = combined_history['Volume'].pct_change()
                data['Volume_SMA_5'] = combined_history['Volume'].rolling(window=5).mean()
                data['Volume_SMA_20'] = combined_history['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = combined_history['Volume'] / data['Volume_SMA_5'].where(data['Volume_SMA_5'] != 0, 1e-10)
                data['Volume_Ratio'] = data['Volume_Ratio'].clip(0, 100)

                data['Price_Change'] = combined_history['Close'].pct_change()
                data['Price_Change_5d'] = combined_history['Close'].pct_change(periods=5)
                data['Price_Change_20d'] = combined_history['Close'].pct_change(periods=20)

                data['Volatility_5d'] = data['Price_Change'].rolling(window=5).std()
                data['Volatility_20d'] = data['Price_Change'].rolling(window=20).std()

                data['Price_to_SMA50'] = combined_history['Close'] / data['SMA_50'].where(data['SMA_50'] != 0, 1e-10) - 1
                data['Price_to_SMA200'] = combined_history['Close'] / data['SMA_200'].where(data['SMA_200'] != 0, 1e-10) - 1

                data['ROC_5'] = (combined_history['Close'] / combined_history['Close'].shift(5) - 1) * 100
                data['ROC_10'] = (combined_history['Close'] / combined_history['Close'].shift(10) - 1) * 100

                obv = pd.Series(index=combined_history.index)
                obv.iloc[0] = 0
                for i in range(1, len(combined_history)):
                    if combined_history['Close'].iloc[i] > combined_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + combined_history['Volume'].iloc[i]
                    elif combined_history['Close'].iloc[i] < combined_history['Close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - combined_history['Volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                data['OBV'] = obv
                data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()

                low_14 = combined_history['Low'].rolling(window=14).min()
                high_14 = combined_history['High'].rolling(window=14).max()
                data['%K'] = (combined_history['Close'] - low_14) / (high_14 - low_14).where(high_14 != low_14, 1e-10) * 100
                data['%D'] = data['%K'].rolling(window=3).mean()

                tr1 = abs(combined_history['High'] - combined_history['Low'])
                tr2 = abs(combined_history['High'] - combined_history['Close'].shift())
                tr3 = abs(combined_history['Low'] - combined_history['Close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()

                plus_dm = combined_history['High'].diff()
                minus_dm = combined_history['Low'].diff().mul(-1)
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

                smoothed_plus_dm = plus_dm.rolling(window=14).sum()
                smoothed_minus_dm = minus_dm.rolling(window=14).sum()
                smoothed_atr = atr.rolling(window=14).sum()

                plus_di = 100 * smoothed_plus_dm / smoothed_atr.where(smoothed_atr !=0 , 1e-10)
                minus_di = 100 * smoothed_minus_dm / smoothed_atr.where(smoothed_atr != 0, 1e-10)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).where((plus_di + minus_di) != 0, 1e-10)
                data['ADX'] = dx.rolling(window=14).mean()
                data['Plus_DI'] = plus_di
                data['Minus_DI'] = minus_di
                data['ADX'] = data['ADX'].clip(0, 100)
                data['Plus_DI'] = data['Plus_DI'].clip(0, 100)
                data['Minus_DI'] = data['Minus_DI'].clip(0, 100)

                data['Sentiment_Score'] = sentiment_score
                data['Trend_Direction'] = 1 if trend_direction == "UPTREND" else -1

                # Replace infinities and fill NaNs with reasonable defaults
                data.replace([np.inf, -np.inf], np.nan, inplace=True)
                data = data.fillna({
                    'SMA_5': data['Close'], 'SMA_20': data['Close'], 'SMA_50': data['Close'], 'SMA_200': data['Close'],
                    'EMA_12': data['Close'], 'EMA_26': data['Close'], 'MACD': 0, 'MACD_Signal': 0, 'MACD_Histogram': 0,
                    'RSI': 50, 'BB_Middle': data['Close'], 'BB_Upper': data['Close'] * 1.1, 'BB_Lower': data['Close'] * 0.9,
                    'BB_Width': 0, 'Volume_Change': 0, 'Volume_SMA_5': combined_history['Volume'],
                    'Volume_SMA_20': combined_history['Volume'], 'Volume_Ratio': 1, 'Price_Change': 0,
                    'Price_Change_5d': 0, 'Price_Change_20d': 0, 'Volatility_5d': 0, 'Volatility_20d': 0,
                    'Price_to_SMA50': 0, 'Price_to_SMA200': 0, 'ROC_5': 0, 'ROC_10': 0, 'OBV': 0, 'OBV_EMA': 0,
                    '%K': 50, '%D': 50, 'ADX': 25, 'Plus_DI': 25, 'Minus_DI': 25, 'Sentiment_Score': 0.5,
                    'Trend_Direction': 0
                })
                data = data.clip(lower=-1e10, upper=1e10)  # Clip extreme values
                data.dropna(inplace=True)

                logger.info("Preparing data for ML prediction...")
                features = [
                    'Close', 'SMA_5', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'MACD',
                    'MACD_Signal', 'MACD_Histogram', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                    'BB_Width', 'Volume_Change', 'Volume_SMA_5', 'Volume_SMA_20', 'Volume_Ratio',
                    'Price_Change', 'Price_Change_5d', 'Price_Change_20d', 'Volatility_5d',
                    'Volatility_20d', 'Price_to_SMA50', 'Price_to_SMA200', 'ROC_5', 'ROC_10',
                    'OBV', 'OBV_EMA', '%K', '%D', 'ADX', 'Plus_DI', 'Minus_DI', 'Sentiment_Score',
                    'Trend_Direction'
                ]
                X = data[features]
                y = data['Close'].shift(-prediction_days)

                if len(X) < prediction_days + 1:
                    logger.error(f"Insufficient data for {ticker} after preprocessing")
                    ml_analysis = {
                        "success": False,
                        "message": f"Insufficient data for ML analysis of {ticker}"
                    }
                else:
                    X = X[:-prediction_days]
                    y = y[:-prediction_days]
                    X = X.dropna()
                    y = y[X.index]

                    # Check for invalid values in X and y
                    logger.info("Checking feature matrix for invalid values...")
                    if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(X.abs() > 1e10):
                        logger.error(f"Invalid values in feature matrix for {ticker}. Skipping ML analysis.")
                        for col in X.columns:
                            if np.any(np.isnan(X[col])) or np.any(np.isinf(X[col])):
                                logger.error(f"Column {col} contains NaN or infinite values")
                        ml_analysis = {
                            "success": False,
                            "message": f"Invalid values in feature matrix for {ticker}"
                        }
                    elif np.any(np.isnan(y)) or np.any(np.isinf(y)):
                        logger.error(f"Invalid values in target variable for {ticker}. Skipping ML analysis.")
                        ml_analysis = {
                            "success": False,
                            "message": f"Invalid values in target variable for {ticker}"
                        }
                    else:
                        scaler_X = MinMaxScaler()
                        scaler_y = MinMaxScaler()
                        X_scaled = scaler_X.fit_transform(X)
                        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y_scaled, test_size=0.2, shuffle=False
                        )

                        logger.info(f"Training TOP 5 INDUSTRY-LEVEL ML models for {ticker}...")

                        # INDUSTRY LEVEL: Import additional ML libraries with proper error handling
                        ml_libraries_available = {}

                        try:
                            import xgboost as xgb
                            ml_libraries_available['xgb'] = xgb
                            logger.info("XGBoost imported successfully")
                        except ImportError as e:
                            logger.warning(f"XGBoost not available: {e}")

                        try:
                            import lightgbm as lgb
                            ml_libraries_available['lgb'] = lgb
                            logger.info("LightGBM imported successfully")
                        except ImportError as e:
                            logger.warning(f"LightGBM not available: {e}")

                        try:
                            import catboost as cb
                            ml_libraries_available['cb'] = cb
                            logger.info("CatBoost imported successfully")
                        except ImportError as e:
                            logger.warning(f"CatBoost not available: {e}")

                        try:
                            from sklearn.svm import SVR
                            from sklearn.neural_network import MLPRegressor
                            from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
                            from sklearn.model_selection import GridSearchCV, cross_val_score
                            from sklearn.preprocessing import RobustScaler
                            ml_libraries_available['sklearn'] = True
                            logger.info("Scikit-learn imported successfully")
                        except ImportError as e:
                            logger.warning(f"Scikit-learn components not available: {e}")
                            ml_libraries_available['sklearn'] = False

                        # INDUSTRY LEVEL: Advanced feature scaling
                        if ml_libraries_available.get('sklearn', False):
                            robust_scaler = RobustScaler()
                            X_train_robust = robust_scaler.fit_transform(X_train)
                            X_test_robust = robust_scaler.transform(X_test)
                        else:
                            # Fallback to simple scaling
                            X_train_robust = X_train
                            X_test_robust = X_test

                        # INDUSTRY LEVEL: TOP 5 ML MODELS FOR FINANCIAL PREDICTION
                        models = {}
                        predictions = {}

                        # 1. XGBoost - Best for tabular financial data
                        if 'xgb' in ml_libraries_available:
                            try:
                                logger.info("Training XGBoost model...")
                                xgb_model = ml_libraries_available['xgb'].XGBRegressor(
                                    n_estimators=200,
                                    max_depth=6,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42,
                                    n_jobs=-1
                                )
                                xgb_model.fit(X_train_robust, y_train)
                                models['xgb'] = xgb_model
                                predictions['xgb'] = xgb_model.predict(X_test_robust)
                                logger.info("XGBoost model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training XGBoost: {e}")

                        # 2. LightGBM - Fast gradient boosting
                        if 'lgb' in ml_libraries_available:
                            try:
                                logger.info("Training LightGBM model...")
                                lgb_model = ml_libraries_available['lgb'].LGBMRegressor(
                                    n_estimators=200,
                                    max_depth=6,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42,
                                    n_jobs=-1,
                                    verbose=-1
                                )
                                lgb_model.fit(X_train_robust, y_train)
                                models['lgb'] = lgb_model
                                predictions['lgb'] = lgb_model.predict(X_test_robust)
                                logger.info("LightGBM model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training LightGBM: {e}")

                        # 3. CatBoost - Categorical features handling
                        if 'cb' in ml_libraries_available:
                            try:
                                logger.info("Training CatBoost model...")
                                cb_model = ml_libraries_available['cb'].CatBoostRegressor(
                                    iterations=200,
                                    depth=6,
                                    learning_rate=0.1,
                                    random_state=42,
                                    verbose=False
                                )
                                cb_model.fit(X_train_robust, y_train)
                                models['cb'] = cb_model
                                predictions['cb'] = cb_model.predict(X_test_robust)
                                logger.info("CatBoost model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training CatBoost: {e}")

                        # 4. Extra Trees - Extremely randomized trees
                        if ml_libraries_available.get('sklearn', False):
                            try:
                                logger.info("Training Extra Trees model...")
                                et_model = ExtraTreesRegressor(
                                    n_estimators=200,
                                    max_depth=10,
                                    random_state=42,
                                    n_jobs=-1
                                )
                                et_model.fit(X_train_robust, y_train)
                                models['et'] = et_model
                                predictions['et'] = et_model.predict(X_test_robust)
                                logger.info("Extra Trees model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training Extra Trees: {e}")

                        # 5. Support Vector Regression - Non-linear patterns
                        if ml_libraries_available.get('sklearn', False):
                            try:
                                logger.info("Training SVR model...")
                                svr_model = SVR(
                                    kernel='rbf',
                                    C=100,
                                    gamma='scale',
                                    epsilon=0.1
                                )
                                svr_model.fit(X_train_robust, y_train)
                                models['svr'] = svr_model
                                predictions['svr'] = svr_model.predict(X_test_robust)
                                logger.info("SVR model trained successfully")
                            except Exception as e:
                                logger.error(f"Error training SVR: {e}")

                        # 6. Multi-layer Perceptron - Neural network
                        if ml_libraries_available.get('sklearn', False):
                            try:
                                logger.info("Training MLP Neural Network...")
                                mlp_model = MLPRegressor(
                                    hidden_layer_sizes=(256, 128, 64),
                                    activation='relu',
                                    solver='adam',
                                    alpha=0.001,
                                    learning_rate='adaptive',
                                    max_iter=500,
                                    random_state=42
                                )
                                mlp_model.fit(X_train_robust, y_train)
                                models['mlp'] = mlp_model
                                predictions['mlp'] = mlp_model.predict(X_test_robust)
                                logger.info("MLP Neural Network trained successfully")
                            except Exception as e:
                                logger.error(f"Error training MLP: {e}")

                        # INDUSTRY LEVEL: ADVANCED ENSEMBLE METHODS
                        if len(models) > 1 and ml_libraries_available.get('sklearn', False):
                            try:
                                logger.info("Creating advanced ensemble models...")

                                # Create ensemble only with successfully trained models
                                ensemble_models = [(name, model) for name, model in models.items()]

                                if len(ensemble_models) >= 2:
                                    voting_ensemble = VotingRegressor(ensemble_models)
                                    voting_ensemble.fit(X_train_robust, y_train)
                                    models['ensemble'] = voting_ensemble
                                    predictions['ensemble'] = voting_ensemble.predict(X_test_robust)
                                    logger.info("Ensemble model created successfully")
                            except Exception as e:
                                logger.error(f"Error creating ensemble: {e}")

                        # Stacking Regressor with top models (if we have enough models)
                        if len(models) >= 3 and ml_libraries_available.get('sklearn', False):
                            try:
                                logger.info("Creating stacking ensemble...")
                                from sklearn.ensemble import StackingRegressor
                                from sklearn.linear_model import LinearRegression

                                # Create stacking only with successfully trained models
                                stacking_estimators = [(name, model) for name, model in models.items() if name != 'ensemble']

                                if len(stacking_estimators) >= 2:
                                    stacking_regressor = StackingRegressor(
                                        estimators=stacking_estimators,
                                        final_estimator=LinearRegression(),
                                        cv=3  # Reduced CV for faster training
                                    )
                                    stacking_regressor.fit(X_train_robust, y_train)
                                    models['stacking'] = stacking_regressor
                                    predictions['stacking'] = stacking_regressor.predict(X_test_robust)
                                    logger.info("Stacking ensemble created successfully")
                            except Exception as e:
                                logger.error(f"Error creating stacking ensemble: {e}")

                        # INDUSTRY LEVEL: MODEL EVALUATION AND SELECTION
                        if len(models) > 0:
                            logger.info(f"Evaluating {len(models)} trained models...")
                            model_scores = {}
                            model_predictions = {}

                            for name, model in models.items():
                                try:
                                    # Use the predictions we already calculated if available
                                    if name in predictions:
                                        y_pred_robust = predictions[name]
                                    else:
                                        y_pred_robust = model.predict(X_test_robust)

                                    y_pred = scaler_y.inverse_transform(y_pred_robust.reshape(-1, 1)).ravel()
                                    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

                                    mse_model = mean_squared_error(y_test_actual, y_pred)
                                    mae_model = mean_absolute_error(y_test_actual, y_pred)
                                    r2_model = r2_score(y_test_actual, y_pred)

                                    # Get prediction for current price
                                    last_features = X.iloc[-1:].values
                                    last_features_robust = robust_scaler.transform(scaler_X.transform(last_features))
                                    pred_scaled = model.predict(last_features_robust)
                                    pred_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                                    model_scores[name] = {
                                        'mse': mse_model,
                                        'mae': mae_model,
                                        'r2': r2_model,
                                        'prediction': pred_price
                                    }
                                    model_predictions[name] = y_pred
                                    logger.info(f"{name} - R2: {r2_model:.4f}, Prediction: {pred_price:.2f}")
                                except Exception as e:
                                    logger.error(f"Error evaluating {name}: {e}")

                            # Select best model based on R2 score
                            if model_scores:
                                best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['r2'])
                                predicted_price = model_scores[best_model_name]['prediction']
                                mse = model_scores[best_model_name]['mse']
                                mae = model_scores[best_model_name]['mae']
                                r2 = model_scores[best_model_name]['r2']
                                logger.info(f"Best ML model: {best_model_name} (R2: {r2:.4f})")
                            else:
                                logger.warning("No models successfully evaluated, using fallback")
                                predicted_price = current_price * 1.01
                                mse = mae = r2 = 0.0
                                best_model_name = "fallback"
                        else:
                            logger.warning("No ML models trained, using fallback prediction")
                            predicted_price = current_price * 1.01
                            mse = mae = r2 = 0.0
                            best_model_name = "fallback"

                        logger.info(f"Best ML model for {ticker}: {best_model_name} (R2: {r2:.4f})")

                        # Store all model results for ensemble decision
                        ensemble_results = {
                            'best_model': best_model_name,
                            'best_prediction': predicted_price,
                            'model_scores': model_scores,
                            'ensemble_prediction': np.mean([score['prediction'] for score in model_scores.values()])
                        }

                        logger.info(f"Performing adversarial training for {ticker}...")
                        adv_training_result = self.adversarial_training_loop(
                            X_train, y_train, X_test, y_test, input_size=X.shape[1], bot_running=bot_running
                        )

                        # Handle case where adversarial training returns None due to insufficient data
                        if adv_training_result is None:
                            logger.warning(f"Adversarial training returned None for {ticker}. Using stacking regressor prediction.")
                            lstm_metrics = {"mse": "N/A", "r2": "N/A", "adv_mse": "N/A", "adv_r2": "N/A"}
                            transformer_metrics = {"mse": "N/A", "r2": "N/A", "adv_mse": "N/A", "adv_r2": "N/A"}
                            lstm_pred = predicted_price
                            transformer_pred = predicted_price
                            ensemble_pred = predicted_price
                        elif adv_training_result["success"]:
                            lstm_model = adv_training_result["lstm_model"]
                            transformer_model = adv_training_result["transformer_model"]
                            lstm_metrics = adv_training_result["lstm_metrics"]
                            transformer_metrics = adv_training_result["transformer_metrics"]

                            last_sequence = X_scaled[-20:]  # Assuming seq_length=20
                            last_sequence = last_sequence[np.newaxis, :, :]
                            last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(self.device)

                            lstm_model.eval()
                            transformer_model.eval()
                            with torch.no_grad():
                                lstm_pred_scaled = lstm_model(last_sequence_tensor).cpu().numpy()
                                transformer_pred_scaled = transformer_model(last_sequence_tensor).cpu().numpy()
                            lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled)[0][0]
                            transformer_pred = scaler_y.inverse_transform(transformer_pred_scaled)[0][0]

                            ensemble_pred = (predicted_price + lstm_pred + transformer_pred) / 3
                        else:
                            logger.warning(f"Adversarial training failed for {ticker}. Using stacking regressor prediction.")
                            lstm_metrics = {"mse": "N/A", "r2": "N/A", "adv_mse": "N/A", "adv_r2": "N/A"}
                            transformer_metrics = {"mse": "N/A", "r2": "N/A", "adv_mse": "N/A", "adv_r2": "N/A"}
                            lstm_pred = predicted_price
                            transformer_pred = predicted_price
                            ensemble_pred = predicted_price

                        logger.info(f"Performing RL training with adversarial events for {ticker}...")
                        rl_result = self.train_rl_with_adversarial_events(
                            extended_history,
                            ensemble_pred,
                            current_price,
                            bot_running=bot_running
                        )

                        # Calculate prediction direction based on current vs predicted price
                        prediction_direction = (float(ensemble_pred) - current_price) / current_price if current_price != 0 else 0
                        
                        ml_analysis = {
                            "success": True,
                            "predicted_price": float(ensemble_pred),
                            "prediction_direction": prediction_direction,  # Required for professional buy logic
                            "confidence": float(r2),
                            "model_accuracy": float(r2),  # Required for professional buy logic
                            "mse": float(mse),
                            "mae": float(mae),
                            "r2_score": float(r2),
                            "best_ml_model": best_model_name,
                            "best_model_metrics": {
                                "mse": float(mse),
                                "mae": float(mae),
                                "r2": float(r2)
                            },
                            "all_model_scores": {name: {
                                "mse": float(scores["mse"]),
                                "mae": float(scores["mae"]),
                                "r2": float(scores["r2"]),
                                "prediction": float(scores["prediction"])
                            } for name, scores in model_scores.items()},
                            "lstm_metrics": lstm_metrics,
                            "transformer_metrics": transformer_metrics,
                            "rl_metrics": rl_result,
                            "rl_recommendation": rl_result.get("recommendation", "HOLD"),  # Required for professional buy logic
                            "rl_confidence": rl_result.get("average_reward", 0.5),  # Required for professional buy logic
                            "rl_sharpe_ratio": rl_result.get("performance_pct", 0.5) / 100 if rl_result.get("performance_pct") else 0.5,  # Required for professional buy logic
                            "ensemble_components": {
                                "best_ml_model": float(predicted_price),
                                "lstm": float(lstm_pred),
                                "transformer": float(transformer_pred),
                                "ensemble_average": float(ensemble_results['ensemble_prediction']),
                                "ensemble_prediction": prediction_direction,  # Required for professional buy logic
                                "ensemble_models_count": len(model_scores)  # Required for professional buy logic
                            },
                            "industry_level_features": {
                                "top_5_ml_models": list(model_scores.keys()),
                                "ensemble_methods": ["VotingRegressor", "StackingRegressor"],
                                "advanced_scaling": "RobustScaler",
                                "model_selection": "Best R2 Score"
                            }
                        }

            result = {
                "success": True,
                "stock_data": stock_data,
                "recommendation": recommendation,
                "buy_score": 0.0,  # Will be calculated in proper scoring logic
                "sell_score": 0.0,  # Will be calculated in proper scoring logic
                "technical_indicators": {
                    "sma_50": float(sma_50),
                    "sma_200": float(sma_200),
                    "ema_50": float(ema_50),
                    "rsi": float(rsi),
                    "macd": float(macd),
                    "signal_line": float(signal_line),
                    "macd_histogram": float(macd_histogram),
                    "bb_upper": float(bb_upper),
                    "bb_lower": float(bb_lower),
                    "volatility": float(volatility),
                    "momentum": float(momentum),
                    "volume_trend": volume_trend
                },
                "fundamental_analysis": {
                    "market_cap": safe_float(market_cap, 0.0),
                    "pe_ratio": safe_float(pe_ratio, 0.0),
                    "price_to_earnings": safe_float(pe_ratio, 0.0),  # Duplicate for compatibility
                    "dividend_yield": safe_float(dividend_yield, 0.0),
                    "52w_high": safe_float(high_52w, 0.0),
                    "52w_low": safe_float(low_52w, 0.0),
                    "balance_sheet": balance_sheet_filtered,
                    "income_statement": income_statement_filtered,
                    "cash_flow": cash_flow_filtered,
                    "price_to_book": safe_float(stock_info.get("priceToBook", 2.0), 2.0),
                    "earnings_growth": safe_float(stock_info.get("earningsGrowth", 0.05), 0.05),  # Default 5% growth
                    "return_on_equity": safe_float(stock_info.get("returnOnEquity", 0.10), 0.10),  # Default 10% ROE
                    "free_cash_flow_yield": safe_float(stock_info.get("freeCashflow", 0.05), 0.05) / safe_float(market_cap, 1.0) if safe_float(market_cap, 1.0) > 0 else 0.05,  # Default 5% FCF yield
                    "debt_to_equity": safe_float(stock_info.get("debtToEquity", 0.5), 0.5),  # Default 0.5 debt-to-equity
                    "payout_ratio": safe_float(stock_info.get("payoutRatio", 0.0), 0.0),  # Default 0% payout ratio
                    "earnings_quality": safe_float(0.5, 0.5),  # Default value
                    "insider_ownership": safe_float(institutional_data.get("insider_ownership_pct", 0.0), 0.0) / 100 if institutional_data else 0.0,  # Use insider ownership if available
                    "sector_pe": safe_float(20.0, 20.0)  # Default sector P/E
                },
                "sentiment_analysis": sentiment_data,
                "mpt_metrics": mpt_metrics,
                "institutional_holders": institutional_data,
                "mutual_fund_holders": mutual_fund_data,
                "support_level": float(support_level),
                "resistance_level": float(resistance_level),
                "risk_level": risk_level,
                "timeframe": timeframe,
                "explanation": explanation,
                "ml_analysis": ml_analysis,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return result

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error analyzing {ticker}: {str(e)}"
            }
    def save_analysis_to_files(self, analysis, output_dir="stock_analysis"):
        try:
            if not analysis.get("success", False):
                logger.error(f"Cannot save analysis: {analysis.get('message', 'Unknown error')}")
                return {"success": False, "message": analysis.get('message', 'Unknown error')}

            os.makedirs(output_dir, exist_ok=True)
            ticker = analysis.get("stock_data", {}).get("symbol", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_ticker = ticker.replace(".", "_")

            json_filename = os.path.join(output_dir, f"{sanitized_ticker}_analysis_{timestamp}.json")
            csv_filename = os.path.join(output_dir, f"{sanitized_ticker}_summary_{timestamp}.csv")
            log_filename = os.path.join(output_dir, "ml_logs.txt")

            # Save full analysis to JSON
            json_data = self.convert_np_types(analysis)
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved full analysis to {json_filename}")

            # Prepare CSV summary
            stock_data = analysis.get("stock_data", {})
            ml_analysis = analysis.get("ml_analysis", {})
            current_price = stock_data.get("current_price", {})
            predicted_price = ml_analysis.get("predicted_price", "N/A")
            
            # Handle current_price and predicted_price
            current_price_usd = current_price.get("USD", "N/A") if isinstance(current_price, dict) else current_price
            predicted_price_usd = predicted_price if isinstance(predicted_price, (int, float)) else "N/A"

            # Calculate predicted change percentage if possible
            predicted_change_pct = "N/A"
            if isinstance(current_price_usd, (int, float)) and isinstance(predicted_price_usd, (int, float)) and current_price_usd != 0:
                predicted_change_pct = ((predicted_price_usd - current_price_usd) / current_price_usd) * 100

            csv_data = {
                "Symbol": ticker,
                "Name": stock_data.get("name", "N/A"),
                "Current_Price_USD": current_price_usd,
                "Predicted_Price": predicted_price_usd,
                "Recommendation": analysis.get("recommendation", "N/A"),
                "Buy_Score": analysis.get("buy_score", "N/A"),
                "Sell_Score": analysis.get("sell_score", "N/A"),
                "Risk_Level": analysis.get("risk_level", "N/A"),
                "Sector": stock_data.get("sector", "N/A"),
                "Industry": stock_data.get("industry", "N/A"),
                "Timestamp": timestamp
            }

            with open(csv_filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_data.keys())
                writer.writeheader()
                writer.writerow(csv_data)
            logger.info(f"Saved summary to {csv_filename}")

            # Log ML, RL, and Stock Analysis Report details
            if ml_analysis.get("success", False):
                log_entry = f"\n[{timestamp}] Analysis for {ticker}\n"
                log_entry += "=" * 50 + "\n"
                log_entry += "Stock Analysis Report\n"
                log_entry += "=" * 50 + "\n"
                log_entry += analysis.get("explanation", "No recommendation explanation available")  # Use top-level explanation
                log_entry += "\n"  # Add spacing after the report
                log_entry += f"Prediction Days: 30\n"
                log_entry += f"Current Price (USD): {current_price_usd}\n"
                log_entry += f"Predicted Price (USD): {predicted_price_usd}\n"
                log_entry += f"Predicted Change (%): {predicted_change_pct:.2f}\n" if predicted_change_pct != "N/A" else "Predicted Change (%): N/A\n"
                log_entry += f"Confidence Score: {ml_analysis.get('confidence', 'N/A')}\n"
                log_entry += f"Pattern: Technical and Sentiment Analysis\n"
                log_entry += "Model Scores:\n"
                logger.debug(f"stacking_regressor_metrics: {ml_analysis.get('stacking_regressor_metrics', {})}")  # Debug logging
                for model, scores in ml_analysis.get("stacking_regressor_metrics", {}).items():
                    log_entry += f"  {model}:\n"
                    if isinstance(scores, dict):
                        for metric, value in scores.items():
                            log_entry += f"    {metric}: {value}\n"
                    else:
                        log_entry += f"    Score: {scores}\n"  # Handle float case
                log_entry += f"Best Model: Stacking Ensemble\n"

                # Log LSTM and Transformer epoch logs
                adv_results = {}  # Update if adversarial results are stored differently
                for model, logs in [
                    ("LSTM", adv_results.get("lstm_epoch_logs", {})),
                    ("Transformer", adv_results.get("transformer_epoch_logs", {}))
                ]:
                    if logs:
                        log_entry += f"{model} Epoch Logs:\n"
                        for epoch, loss in logs.items():
                            log_entry += f"  {epoch}: Loss = {loss:.4f}\n"

                # Log RL results
                rl_results = ml_analysis.get("rl_metrics", {})
                if rl_results.get("success", False):
                    log_entry += "Reinforcement Learning Results:\n"
                    log_entry += f"  Recommendation: {rl_results.get('recommendation', 'N/A')}\n"
                    log_entry += f"  Performance (%): {rl_results.get('performance_pct', 'N/A')}\n"
                    log_entry += f"  Average Reward: {rl_results.get('average_reward', 'N/A')}\n"
                    log_entry += f"  Average Events/Episode: {rl_results.get('average_events_per_episode', 'N/A')}\n"
                    log_entry += "  Actions Distribution:\n"
                    for action, prob in rl_results.get("actions_distribution", {}).items():
                        log_entry += f"    {action}: {prob:.4f}\n"
                    log_entry += "  RL Epoch Logs:\n"
                    for log in rl_results.get("epoch_logs", []):
                        log_entry += (f"    Episode {log.get('episode', 'N/A')}: "
                                    f"Reward = {log.get('total_reward', 'N/A'):.2f}, "
                                    f"Avg Reward = {log.get('average_reward', 'N/A'):.2f}, "
                                    f"Events = {log.get('events_triggered', 'N/A')}\n")

                with open(log_filename, "a", encoding="utf-8") as f:
                    f.write(log_entry)
                logger.info(f"Appended ML, RL, and Stock Analysis Report to {log_filename}")

            return {
                "success": True,
                "json_file": json_filename,
                "csv_file": csv_filename,
                "log_file": log_filename
            }

        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error saving analysis: {str(e)}"
            }

class StockTradingBot:
    def __init__(self, config):
        self.config = config

# Add import for professional buy logic
# Use absolute import to avoid relative import issues when imported by other modules
try:
    from core.professional_buy_integration import ProfessionalBuyIntegration
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from core.professional_buy_integration import ProfessionalBuyIntegration

# Setup logging
import logging
logger = logging.getLogger(__name__)

class StockTradingBot:
    def __init__(self, config):
        self.config = config

        self.timezone = pytz.timezone("Asia/Kolkata")  # Changed to India timezone
        self.data_feed = DataFeed(config["tickers"])
        self.portfolio = VirtualPortfolio(config)
        self.executor = TradingExecutor(self.portfolio, config)
        self.tracker = PortfolioTracker(self.portfolio, config)
        self.reporter = PerformanceReport(self.portfolio)
        self.stock_analyzer = Stock(
            reddit_client_id=config.get("reddit_client_id"),
            reddit_client_secret=config.get("reddit_client_secret"),
            reddit_user_agent=config.get("reddit_user_agent")
        )

        # Initialize professional buy integration
        try:
            from core.professional_buy_integration import ProfessionalBuyIntegration
            from core.professional_buy_config import ProfessionalBuyConfig
            # Use conservative configuration for more professional trading
            professional_config = ProfessionalBuyConfig.get_config()
            self.professional_buy_integration = ProfessionalBuyIntegration(professional_config)
            logger.info("Professional buy integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize professional buy integration: {e}")
            self.professional_buy_integration = None

        # Initialize professional sell integration
        try:
            from core.professional_sell_integration import ProfessionalSellIntegration
            from core.professional_sell_config import ProfessionalSellConfig
            # Use default configuration for professional sell logic
            professional_sell_config = ProfessionalSellConfig.get_config()
            self.professional_sell_integration = ProfessionalSellIntegration(professional_sell_config)
            logger.info("Professional sell integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize professional sell integration: {e}")
            self.professional_sell_integration = None

        # Initialize production core components
        try:
            from core import AdaptiveThresholdManager, IntegratedRiskManager
            self.adaptive_threshold_manager = AdaptiveThresholdManager()
            # Fix: Initialize IntegratedRiskManager with proper configuration
            self.risk_manager = IntegratedRiskManager({
                "max_portfolio_risk_pct": 0.02,  # 2% max portfolio risk (industry standard)
                "max_single_stock_exposure": 0.05    # 5% max position risk
            })
            self.production_core_enabled = True
            logger.info("Production core components initialized successfully")
        except ImportError as e:
            logger.warning(f"Production core components not available: {e}")
            self.adaptive_threshold_manager = None
            self.risk_manager = None
            self.production_core_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize production core components: {e}")
            self.adaptive_threshold_manager = None
            self.risk_manager = None
            self.production_core_enabled = False

        # Initialize continuous learning engine
        try:
            from core.continuous_learning_engine import ContinuousLearningEngine
            self.learning_engine = ContinuousLearningEngine()
            logger.info("Continuous learning engine initialized successfully")
        except ImportError as e:
            logger.warning(f"Continuous learning engine not available: {e}")
            self.learning_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize continuous learning engine: {e}")
            self.learning_engine = None

        # Register callback for trade outcomes
        if self.learning_engine:
            self.portfolio.add_trade_callback(self._on_trade_executed)

        # Initialize unified data service manager
        self.data_service_manager = self._initialize_data_services()

        # Initialize chatbot command handler
        self.chatbot = ChatbotCommandHandler(self)
        self.bot_running = False
        self.command_queue = queue.Queue()
        self.initialize()

        # Initialize unified ML interface
        if ML_INTERFACE_AVAILABLE:
            try:
                self.ml_interface = get_ml_interface()
                logger.info("Unified ML interface initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize unified ML interface: {e}")
                self.ml_interface = None
        else:
            self.ml_interface = None

    def _initialize_data_services(self):
        """Initialize unified data service manager"""
        try:
            # Try to initialize Fyers data service first
            from fyers_data_service import FyersDataService
            fyers_service = FyersDataService()
            # Note: FyersDataService runs as a separate service, no direct connection needed
            logger.info("Fyers data service initialized (runs independently)")
            return {"primary": fyers_service, "fallback": "yahoo"}
        except Exception as e:
            logger.error(f"Error initializing data services: {e}")
            return {"primary": None, "fallback": "yahoo"}

    def initialize(self):
        """Initialize the bot and its components."""
        logger.info("Initializing Stock Trading Bot for Indian market...")
        # Portfolio is already initialized in VirtualPortfolio.__init__()
        # Don't call initialize_portfolio() as it resets the data

    def _on_trade_executed(self, trade_data):
        """Callback method called when a trade is executed, used for learning engine updates."""
        if not self.learning_engine:
            return
            
        try:
            # Extract trade information
            asset = trade_data.get("asset")
            action = trade_data.get("action")
            qty = trade_data.get("qty", 0)
            price = trade_data.get("price", 0)
            realized_pnl = trade_data.get("realized_pnl", 0)
            
            # Calculate profit/loss percentage
            profit_loss_pct = 0
            if action == "sell" and qty > 0 and price > 0:
                # Find the original buy transaction to calculate P&L percentage
                for log_entry in reversed(self.portfolio.trade_log):
                    if (log_entry.get("asset") == asset and 
                        log_entry.get("action") == "buy" and
                        log_entry.get("qty", 0) >= qty):
                        buy_price = log_entry.get("price", price)
                        if buy_price > 0:
                            profit_loss_pct = (price - buy_price) / buy_price
                        break
            
            # Prepare outcome data for learning engine
            outcome_data = {
                "profit_loss": realized_pnl,
                "profit_loss_pct": profit_loss_pct,
                "holding_period": 0,  # Would need to track holding periods
                "max_drawdown": 0.0,  # Would need to track drawdown
                "max_runup": 0.0      # Would need to track runup
            }
            
            # We would need to match this with the original decision data
            # For now, we'll just log that a trade was executed
            logger.debug(f"Trade executed for learning engine: {asset} {action} {qty}@{price}")
            
        except Exception as e:
            logger.debug(f"Non-critical: Error in trade callback for learning engine: {e}")

    def is_market_open(self):
        """Check if the NSE is open."""
        try:
            nse = mcal.get_calendar("NSE")
            now = datetime.now(self.timezone)
            schedule = nse.schedule(start_date=now.date(), end_date=now.date())
            if schedule.empty:
                logger.info(f"Market is closed on {now.date()} (no schedule found).")
                return False
            market_open = schedule.iloc[0]["market_open"].astimezone(self.timezone)
            market_close = schedule.iloc[0]["market_close"].astimezone(self.timezone)
            is_open = market_open <= now <= market_close
            logger.debug(f"Market check: Now={now}, Open={market_open}, Close={market_close}, Is Open={is_open}")
            return is_open
        except Exception as e:
            logger.error(f"Error checking NSE market status: {e}")
            return False

    def make_trading_decision(self, analysis):
        """Crystal clear trading decision logic - no fallbacks, pure professional logic"""
        if not analysis.get("success"):
            logger.warning(f"Skipping trading decision for {analysis.get('stock_data', {}).get('symbol')} due to failed analysis")
            return None

        ticker = analysis["stock_data"]["symbol"]
        current_price = analysis["stock_data"]["current_price"]["INR"]
        ml_analysis = analysis["ml_analysis"]
        technical_indicators = analysis["technical_indicators"]
        sentiment_data = analysis["sentiment_analysis"]
        metrics = self.portfolio.get_metrics()
        available_cash = metrics["cash"]
        total_value = metrics["total_value"]
        history = get_stock_data_fyers_or_yf(ticker, period="1y")

        logger.info(f"=== TRADING DECISION FOR {ticker} ===")
        logger.info(f"Current Price: Rs.{current_price:.2f}")
        logger.info(f"Available Cash: Rs.{available_cash:.2f}")
        logger.info(f"Total Portfolio Value: Rs.{total_value:.2f}")

        # STEP 1: Check for SELL decision first (if we own the stock)
        if ticker in self.portfolio.holdings and self.professional_sell_integration:
            logger.info(f"💼 POSITION DETECTED: {ticker} - Checking for sell opportunities")
            try:
                portfolio_holdings = self.portfolio.holdings
                analysis_data = {
                    "technical_indicators": technical_indicators,
                    "sentiment": sentiment_data,
                    "ml_analysis": ml_analysis,
                    "deep_learning_analysis": analysis.get("deep_learning_analysis", {}),
                    "fundamental_analysis": analysis.get("fundamental_analysis", {})
                }
                
                sell_decision = self.professional_sell_integration.evaluate_professional_sell(
                    ticker=ticker,
                    current_price=current_price,
                    portfolio_holdings=portfolio_holdings,
                    analysis_data=analysis_data,
                    price_history=history
                )
                
                if sell_decision.get("action") == "sell":
                    sell_qty = sell_decision.get("qty", 0)
                    if sell_qty > 0 and portfolio_holdings[ticker]["qty"] >= sell_qty:
                        logger.info(f"🔴 SELL SIGNAL CONFIRMED: {ticker}")
                        logger.info(f"   Quantity: {sell_qty} units")
                        logger.info(f"   Current Price: Rs.{current_price:.2f}")
                        logger.info(f"   Stop Loss: Rs.{sell_decision.get('stop_loss', current_price * 0.95):.2f}")
                        logger.info(f"   Take Profit: Rs.{sell_decision.get('take_profit', current_price * 1.05):.2f}")
                        logger.info(f"   Confidence Score: {sell_decision.get('confidence_score', 0.0):.3f}")
                        logger.info(f"   Reason: {sell_decision.get('reason', 'professional_sell')}")
                        
                        success_result = self.executor.execute_trade(
                            "sell", ticker, sell_qty, current_price,
                            sell_decision.get("stop_loss", current_price * 0.95),
                            sell_decision.get("take_profit", current_price * 1.05)
                        )
                        logger.info(f"✅ SELL ORDER EXECUTED: {sell_qty} {ticker} at Rs.{current_price:.2f}")
                        return {
                            "action": "sell",
                            "ticker": ticker,
                            "qty": sell_qty,
                            "price": current_price,
                            "success": success_result,
                            "confidence_score": sell_decision.get("confidence_score", 0.0),
                            "reason": "professional_sell"
                        }
                    else:
                        logger.info(f"⚠️  SELL SIGNAL INVALID: Insufficient quantity or invalid parameters")
                else:
                    logger.info(f"🟡 NO SELL SIGNAL: Professional logic recommends holding {ticker}")
                    logger.info(f"   Reason: {sell_decision.get('reason', 'no_clear_signal')}")
                    professional_reasoning = sell_decision.get('professional_reasoning', 'No specific reasoning provided')
                    logger.info(f"   Professional Reasoning: {professional_reasoning}")
                    # Log detailed breakdown of why no sell signal
                    if professional_reasoning and professional_reasoning != 'No specific reasoning provided':
                        logger.info(f"   🔍 DETAILED SELL ANALYSIS: {professional_reasoning}")
                    else:
                        # Provide additional context when no reasoning is provided
                        logger.info(f"   🔍 ANALYSIS CONTEXT:")
                        logger.info(f"      - Position Quantity: {portfolio_holdings.get(ticker, {}).get('qty', 0)}")
                        logger.info(f"      - Entry Price: Rs.{portfolio_holdings.get(ticker, {}).get('avg_price', 0.0):.2f}")
                        logger.info(f"      - Current Price: Rs.{current_price:.2f}")
                        logger.info(f"      - Unrealized P&L: {((current_price - portfolio_holdings.get(ticker, {}).get('avg_price', current_price)) / portfolio_holdings.get(ticker, {}).get('avg_price', current_price) * 100):.2f}%")
            except Exception as e:
                logger.error(f"❌ ERROR IN SELL EVALUATION for {ticker}: {e}")
                logger.exception("Full traceback:")

        # STEP 2: Check for BUY decision (if we have cash and don't own too much)
        if self.professional_buy_integration and available_cash > 1:  # Minimum cash threshold changed to Rs.1
            logger.info(f"💰 CASH AVAILABLE: Rs.{available_cash:.2f} - Checking for buy opportunities")
            try:
                portfolio_data = {
                    "total_value": total_value,
                    "available_cash": available_cash,
                    "holdings": self.portfolio.holdings
                }
                
                analysis_data = {
                    "technical_indicators": technical_indicators,
                    "sentiment_analysis": sentiment_data,
                    "ml_analysis": ml_analysis,
                    "deep_learning_analysis": analysis.get("deep_learning_analysis", {}),
                    "fundamental_analysis": analysis.get("fundamental_analysis", {})
                }
                
                buy_decision = self.professional_buy_integration.evaluate_professional_buy(
                    ticker=ticker,
                    current_price=current_price,
                    portfolio_context=portfolio_data,
                    analysis_data=analysis_data
                )
                
                if buy_decision.get("action") == "buy":
                    buy_qty = buy_decision.get("qty", 0)
                    trade_value = buy_qty * current_price
                    if buy_qty > 0 and trade_value <= available_cash:
                        logger.info(f"🟢 BUY SIGNAL CONFIRMED: {ticker}")
                        logger.info(f"   Quantity: {buy_qty} units")
                        logger.info(f"   Current Price: Rs.{current_price:.2f}")
                        logger.info(f"   Stop Loss: Rs.{buy_decision.get('stop_loss', current_price * 0.95):.2f}")
                        logger.info(f"   Take Profit: Rs.{buy_decision.get('take_profit', current_price * 1.15):.2f}")
                        logger.info(f"   Confidence Score: {buy_decision.get('confidence_score', 0.0):.3f}")
                        logger.info(f"   Reason: {buy_decision.get('reason', 'professional_buy')}")
                        
                        success_result = self.executor.execute_trade(
                            "buy", ticker, buy_qty, current_price,
                            buy_decision.get("stop_loss", current_price * 0.95),
                            buy_decision.get("take_profit", current_price * 1.15)
                        )
                        logger.info(f"✅ BUY ORDER EXECUTED: {buy_qty} {ticker} at Rs.{current_price:.2f}")
                        return {
                            "action": "buy",
                            "ticker": ticker,
                            "qty": buy_qty,
                            "price": current_price,
                            "success": success_result,
                            "confidence_score": buy_decision.get("confidence_score", 0.0),
                            "reason": "professional_buy"
                        }
                    else:
                        logger.info(f"⚠️  BUY SIGNAL INVALID: Insufficient cash or invalid parameters")
                else:
                    logger.info(f"🟡 NO BUY SIGNAL: Professional logic recommends holding {ticker}")
                    logger.info(f"   Reason: {buy_decision.get('reason', 'no_clear_signal')}")
                    professional_reasoning = buy_decision.get('professional_reasoning', 'No specific reasoning provided')
                    logger.info(f"   Professional Reasoning: {professional_reasoning}")
                    # Log detailed breakdown of why no buy signal
                    if professional_reasoning and professional_reasoning != 'No specific reasoning provided':
                        logger.info(f"   🔍 DETAILED BUY ANALYSIS: {professional_reasoning}")
                    else:
                        # Provide additional context when no reasoning is provided
                        logger.info(f"   🔍 ANALYSIS CONTEXT:")
                        logger.info(f"      - Available Cash: Rs.{available_cash:.2f}")
                        logger.info(f"      - Cash Threshold: Rs.1.00")
                        logger.info(f"      - Cash Sufficient: {'Yes' if available_cash > 1 else 'No'}")
                        logger.info(f"      - Current Price: Rs.{current_price:.2f}")
                        logger.info(f"      - Professional Logic Enabled: {'Yes' if self.professional_buy_integration else 'No'}")
            except Exception as e:
                logger.error(f"❌ ERROR IN BUY EVALUATION for {ticker}: {e}")
                logger.exception("Full traceback:")

        # STEP 3: Default to HOLD
        logger.info(f"🟡 HOLD DECISION: {ticker} - No clear buy/sell signal")
        logger.info(f"   🔍 DETAILED HOLD ANALYSIS:")
        logger.info(f"      - Position Status: {'Holding' if ticker in self.portfolio.holdings else 'Not Holding'}")
        logger.info(f"      - Available Cash: Rs.{available_cash:.2f} ({'Sufficient' if available_cash > 1 else 'Insufficient'})")  # Changed threshold to Rs.1
        logger.info(f"      - Professional Logic Enabled: {self.professional_buy_integration is not None}")
        if ticker in self.portfolio.holdings:
            holding_info = self.portfolio.holdings.get(ticker, {})
            logger.info(f"      - Holding Quantity: {holding_info.get('qty', 0)}")
            logger.info(f"      - Entry Price: Rs.{holding_info.get('avg_price', 0.0):.2f}")
        logger.info(f"=== END TRADING DECISION FOR {ticker} ===")
        return {
            "action": "hold",
            "ticker": ticker,
            "qty": 0,
            "price": current_price,
            "success": True,
            "confidence_score": 0.5,
            "reason": "no_clear_signal"
        }

    def analyze_stock(self, ticker, benchmark_tickers=None, prediction_days=30, training_period="2y", bot_running=True):

        # Calculate unrealized PnL
        current_prices = self.portfolio.get_current_prices()
        current_ticker_price = current_prices.get(ticker, {"price": current_price})["price"]
        unrealized_pnl = (
            (current_ticker_price - self.portfolio.holdings[ticker]["avg_price"])
            * self.portfolio.holdings[ticker]["qty"]
            if ticker in self.portfolio.holdings else 0
        )

        # PRODUCTION FIX: Weighted signal system (replaces binary signals)
        # Calculate weighted signal strength instead of binary signals

        # ENHANCED: Optimize signal weights based on current market conditions and recent performance
        if market_regime == "TRENDING":
            # In trending markets, emphasize trend-following signals
            optimized_weights = {
                "technical": 0.40,    # Higher weight for technical in trends
                "sentiment": 0.15,    # Moderate sentiment weight
                "ml": 0.10,          # Reduce ML weight in trending markets
                "rsi": 0.15,         # RSI for momentum confirmation
                "macd": 0.15,        # MACD for trend confirmation
                "breakout": 0.03,    # Minimal breakout weight
                "volume": 0.02       # Minimal volume weight
            }
        elif market_regime == "VOLATILE":
            # In volatile markets, emphasize risk-aware signals
            optimized_weights = {
                "technical": 0.25,    # Reduce technical weight in volatility
                "sentiment": 0.05,    # Minimize sentiment in volatile markets
                "ml": 0.20,          # Higher ML weight for pattern recognition
                "rsi": 0.20,         # Higher RSI weight for momentum
                "macd": 0.15,        # Moderate MACD weight
                "breakout": 0.10,    # Higher breakout weight for volatility
                "volume": 0.05       # Moderate volume weight
            }
        else:  # RANGE_BOUND
            # In range-bound markets, balanced approach with breakout emphasis
            optimized_weights = {
                "technical": 0.30,    # Moderate technical weight
                "sentiment": 0.12,    # Moderate sentiment weight
                "ml": 0.18,          # Good ML weight for range prediction
                "rsi": 0.12,         # Moderate RSI weight
                "macd": 0.10,        # Lower MACD weight in ranges
                "breakout": 0.12,    # Higher breakout weight for range breaks
                "volume": 0.06       # Moderate volume weight
            }
        
        logger.debug(f"Optimized signal weights for {market_regime}: {optimized_weights}")
        
        # CRITICAL FIX: Weighted signal score calculation moved to AFTER signal_strengths are populated
        # This prevents the bug where weighted_signal_score was calculated with all zero values
        weighted_signal_score = 0.0  # Placeholder - will be calculated after signal_strengths are populated
        
        # Use optimized weights for signal breakdown
        signal_weights = optimized_weights

        # Get moving averages from technical indicators
        ma_20 = technical_indicators.get("SMA_20", current_ticker_price)
        ma_50 = technical_indicators.get("SMA_50", current_ticker_price)

        # Get RSI value early to avoid undefined variable error
        rsi_value = technical_indicators.get("rsi", 50)

        # Get volume data
        volume_data = history.get("Volume", pd.Series())
        volume_sma = volume_data.rolling(window=20).mean().iloc[-1] if not volume_data.empty else None
        current_volume = volume_data.iloc[-1] if not volume_data.empty else None

        # ENHANCED: Professional technical signal with multi-timeframe analysis and adaptive thresholds
        technical_signals = {}
        
        # Get current technical indicators
        current_rsi = technical_indicators.get("rsi", 50)
        current_macd = technical_indicators.get("macd", 0)
        current_signal = technical_indicators.get("signal_line", 0)
        current_bb_upper = technical_indicators.get("bb_upper", current_ticker_price * 1.02)
        current_bb_lower = technical_indicators.get("bb_lower", current_ticker_price * 0.98)
        
        # ENHANCED: Multi-timeframe RSI analysis with adaptive thresholds
        def get_adaptive_rsi_thresholds(volatility, market_regime):
            """Calculate adaptive RSI thresholds based on market conditions"""
            if market_regime == "VOLATILE":
                # In volatile markets, use wider bands
                oversold, overbought = 25, 75
                moderate_oversold, moderate_overbought = 35, 65
            elif market_regime == "TRENDING":
                # In trending markets, use standard bands
                oversold, overbought = 30, 70
                moderate_oversold, moderate_overbought = 40, 60
            else:  # RANGE_BOUND
                # In range-bound markets, use tighter bands for more signals
                oversold, overbought = 35, 65
                moderate_oversold, moderate_overbought = 45, 55
            
            # Adjust for volatility
            volatility_factor = min(max(volatility * 50, 0.5), 2.0)  # 0.5 to 2.0 range
            adjustment = (volatility_factor - 1.0) * 5  # -2.5 to +5 adjustment
            
            return {
                'oversold': max(15, oversold - adjustment),
                'moderate_oversold': max(25, moderate_oversold - adjustment),
                'moderate_overbought': min(75, moderate_overbought + adjustment),
                'overbought': min(85, overbought + adjustment)
            }
        
        rsi_thresholds = get_adaptive_rsi_thresholds(volatility, market_regime)
        logger.debug(f"Adaptive RSI thresholds: {rsi_thresholds}")
        
        # Enhanced RSI signal calculation
        if current_rsi < rsi_thresholds['oversold']:  # Extremely oversold
            rsi_signal = 1.0
            rsi_strength = "extremely_oversold"
        elif current_rsi < rsi_thresholds['moderate_oversold']:  # Moderately oversold
            range_size = rsi_thresholds['moderate_oversold'] - rsi_thresholds['oversold']
            rsi_signal = 0.7 + (rsi_thresholds['moderate_oversold'] - current_rsi) / range_size * 0.3
            rsi_strength = "moderately_oversold"
        elif current_rsi < 50:  # Below neutral but not oversold
            range_size = 50 - rsi_thresholds['moderate_oversold']
            rsi_signal = 0.3 + (rsi_thresholds['moderate_oversold'] - current_rsi) / range_size * 0.4
            rsi_strength = "below_neutral"
        elif current_rsi < rsi_thresholds['moderate_overbought']:  # Above neutral, good momentum
            range_size = rsi_thresholds['moderate_overbought'] - 50
            rsi_signal = 0.3 + (current_rsi - 50) / range_size * 0.4
            rsi_strength = "healthy_uptrend"
        elif current_rsi < rsi_thresholds['overbought']:  # Moderately overbought
            range_size = rsi_thresholds['overbought'] - rsi_thresholds['moderate_overbought']
            rsi_signal = 0.3 - (current_rsi - rsi_thresholds['moderate_overbought']) / range_size * 0.2
            rsi_strength = "moderately_overbought"
        else:  # Extremely overbought
            rsi_signal = 0.1
            rsi_strength = "extremely_overbought"
        
        technical_signals['rsi'] = max(0.0, rsi_signal)
        logger.debug(f"Enhanced RSI analysis: {current_rsi:.1f} -> {rsi_strength} -> signal: {technical_signals['rsi']:.3f}")
        
        # ENHANCED: Multi-timeframe moving average analysis
        ma_20 = technical_indicators.get("SMA_20", current_ticker_price)
        ma_50 = technical_indicators.get("SMA_50", current_ticker_price)
        sma_50 = technical_indicators.get("sma_50", current_ticker_price)
        sma_200 = technical_indicators.get("sma_200", current_ticker_price)
        ema_50 = technical_indicators.get("ema_50", current_ticker_price)
        
        # Use the most reliable MA data available
        if ma_20 and ma_50:
            short_ma, long_ma = ma_20, ma_50
        elif sma_50 and sma_200:
            short_ma, long_ma = sma_50, sma_200
        else:
            short_ma, long_ma = current_ticker_price, current_ticker_price
        
        # Multi-timeframe trend analysis
        trend_signals = []
        
        # Short-term trend (price vs short MA)
        if current_ticker_price > short_ma * 1.02:  # 2% above short MA
            short_trend = 1.0
        elif current_ticker_price > short_ma:
            short_trend = (current_ticker_price - short_ma) / (short_ma * 0.02) * 0.8 + 0.2
        elif current_ticker_price > short_ma * 0.98:  # Within 2% below
            short_trend = 0.2 + (current_ticker_price - short_ma * 0.98) / (short_ma * 0.02) * 0.2
        else:
            short_trend = 0.1
        trend_signals.append(('short_term', short_trend, 0.4))  # 40% weight
        
        # Medium-term trend (short MA vs long MA)
        if short_ma > long_ma * 1.01:  # Short MA clearly above long MA
            medium_trend = 1.0
        elif short_ma > long_ma:
            medium_trend = 0.7 + (short_ma - long_ma) / (long_ma * 0.01) * 0.3
        elif short_ma > long_ma * 0.99:  # Close to crossing
            medium_trend = 0.4 + (short_ma - long_ma * 0.99) / (long_ma * 0.01) * 0.3
        else:
            medium_trend = 0.2
        trend_signals.append(('medium_term', medium_trend, 0.4))  # 40% weight
        
        # Long-term trend momentum (if EMA available)
        if ema_50 and abs(ema_50 - current_ticker_price) > 0:
            ema_distance = (current_ticker_price - ema_50) / ema_50
            if ema_distance > 0.05:  # 5% above EMA
                long_trend = 1.0
            elif ema_distance > 0:
                long_trend = 0.5 + ema_distance / 0.05 * 0.5
            elif ema_distance > -0.05:
                long_trend = 0.3 + (ema_distance + 0.05) / 0.05 * 0.2
            else:
                long_trend = 0.1
            trend_signals.append(('long_term', long_trend, 0.2))  # 20% weight
        
        # Calculate weighted trend signal
        total_weight = sum(weight for _, _, weight in trend_signals)
        weighted_trend = sum(signal * weight for _, signal, weight in trend_signals) / total_weight
        technical_signals['trend'] = weighted_trend
        
        logger.debug(f"Multi-timeframe trend analysis:")
        for name, signal, weight in trend_signals:
            logger.debug(f"  {name}: {signal:.3f} (weight: {weight:.1f})")
        logger.debug(f"  Weighted trend signal: {weighted_trend:.3f}")
        
        # ENHANCED: MACD with divergence detection
        macd_signals = []
        
        # Basic MACD signal
        if current_macd > current_signal:
            macd_diff = current_macd - current_signal
            if abs(current_signal) > 0.001:
                macd_strength = min(abs(macd_diff) / abs(current_signal), 2.0)
            else:
                macd_strength = 1.0
            basic_macd = min(0.5 + macd_strength * 0.5, 1.0)
        else:
            basic_macd = 0.2
        macd_signals.append(('basic_crossover', basic_macd, 0.6))
        
        # MACD histogram trend (if available)
        macd_histogram = technical_indicators.get("macd_histogram", 0)
        if macd_histogram > 0:
            histogram_signal = min(macd_histogram * 10, 1.0)  # Scale histogram
        else:
            histogram_signal = 0.1
        macd_signals.append(('histogram_trend', histogram_signal, 0.4))
        
        # Calculate weighted MACD signal
        total_macd_weight = sum(weight for _, _, weight in macd_signals)
        weighted_macd = sum(signal * weight for _, signal, weight in macd_signals) / total_macd_weight
        technical_signals['macd'] = weighted_macd
        
        logger.debug(f"Enhanced MACD analysis: basic={basic_macd:.3f}, histogram={histogram_signal:.3f}, weighted={weighted_macd:.3f}")
        
        # ENHANCED: Bollinger Bands with volatility context
        bb_range = current_bb_upper - current_bb_lower
        bb_position = (current_ticker_price - current_bb_lower) / bb_range if bb_range > 0 else 0.5
        
        if bb_position < 0.2:  # Near lower band - potential bounce
            bb_signal = 0.8 + (0.2 - bb_position) * 1.0  # 0.8 to 1.0
        elif bb_position < 0.4:  # Lower half - moderate signal
            bb_signal = 0.5 + (0.4 - bb_position) * 1.5  # 0.5 to 0.8
        elif bb_position < 0.6:  # Middle range - neutral
            bb_signal = 0.4
        elif bb_position < 0.8:  # Upper half - moderate signal
            bb_signal = 0.3
        else:  # Near upper band - potential resistance
            bb_signal = 0.2
        
        technical_signals['bollinger'] = bb_signal
        logger.debug(f"Bollinger Bands analysis: position={bb_position:.3f}, signal={bb_signal:.3f}")
        
        # Calculate overall technical signal strength with improved weighting
        technical_weights = {
            'rsi': 0.3,        # RSI gets high weight due to reliability
            'trend': 0.35,     # Trend analysis is critical
            'macd': 0.25,      # MACD for momentum confirmation
            'bollinger': 0.1   # Bollinger for volatility context
        }
        
        weighted_technical = sum(
            technical_signals.get(signal, 0.0) * weight 
            for signal, weight in technical_weights.items()
        )
        
        signal_strengths["technical"] = weighted_technical
        
        logger.debug(f"Technical signal components:")
        for signal, strength in technical_signals.items():
            weight = technical_weights.get(signal, 0)
            contribution = strength * weight
            logger.debug(f"  {signal}: {strength:.3f} × {weight:.2f} = {contribution:.3f}")
        logger.debug(f"Overall technical signal strength: {weighted_technical:.3f}")

        # ENHANCED: Advanced sentiment signal with confidence weighting and market context
        if processed_sentiment_score != 0:  # Only apply when sentiment is available
            base_sentiment = min(max(abs(processed_sentiment_score) / 0.12, 0.0), 1.0)  # More aggressive normalization
            
            # Enhanced market context multipliers
            context_multipliers = []
            
            # Market regime multiplier
            if market_regime == "TRENDING":
                regime_multiplier = 1.4  # Sentiment very important in trending markets
            elif market_regime == "VOLATILE":
                regime_multiplier = 0.6  # Sentiment less reliable in volatile markets
            else:  # RANGE_BOUND
                regime_multiplier = 1.0  # Normal sentiment weight
            context_multipliers.append(regime_multiplier)
            
            # Volatility-based adjustment
            if volatility > 0.04:  # High volatility
                volatility_multiplier = 0.7
            elif volatility > 0.02:  # Medium volatility
                volatility_multiplier = 1.0
            else:  # Low volatility
                volatility_multiplier = 1.2  # Sentiment more reliable in low volatility
            context_multipliers.append(volatility_multiplier)
            
            # Technical confirmation bonus
            if signal_strengths.get("technical", 0) > 0.6:  # Strong technical signal
                technical_confirmation = 1.2  # Sentiment + technical alignment
            elif signal_strengths.get("technical", 0) > 0.4:  # Moderate technical
                technical_confirmation = 1.0
            else:  # Weak technical
                technical_confirmation = 0.8  # Reduce sentiment weight when technical is weak
            context_multipliers.append(technical_confirmation)
            
            # Calculate final multiplier
            final_multiplier = min(sum(context_multipliers) / len(context_multipliers), 1.5)  # Cap at 1.5x
            
            # Apply sentiment direction
            if processed_sentiment_score > 0:
                signal_strengths["sentiment"] = base_sentiment * final_multiplier
            else:
                signal_strengths["sentiment"] = 0.0  # Only positive sentiment for buy signals
            
            logger.debug(f"Enhanced sentiment signal: {signal_strengths['sentiment']:.3f}")
            logger.debug(f"  Base: {base_sentiment:.3f}, Regime: {regime_multiplier:.2f}, Volatility: {volatility_multiplier:.2f}, Technical: {technical_confirmation:.2f}, Final: {final_multiplier:.2f}")
        else:
            signal_strengths["sentiment"] = 0.0  # Zero when no sentiment data
            logger.debug("Sentiment signal: 0.000 (no sentiment data available)")

        # ENHANCED: Professional ML signal strength with comprehensive validation and ensemble optimization
        if ml_analysis.get("success") and ml_analysis.get("predicted_price"):  # Fixed: Check ML analysis success directly
            # Get comprehensive ML metrics for intelligent validation
            ml_r2 = ml_analysis.get("r2_score", 0.0)
            ml_mse = ml_analysis.get("mse", float('inf'))
            ml_mae = ml_analysis.get("mae", float('inf'))
            ml_confidence = ml_analysis.get("confidence", 0.0)
            predicted_price = ml_analysis.get("predicted_price", current_ticker_price)
            best_model = ml_analysis.get("best_ml_model", "unknown")
            
            # AGGRESSIVE DEBUG: Log all ML analysis components
            logger.info(f"=== ML ANALYSIS DEBUG for {ticker} ===")
            logger.info(f"  ML Success: {ml_analysis.get('success')}")
            logger.info(f"  R2 Score: {ml_r2:.6f}")
            logger.info(f"  Predicted Price: {predicted_price:.2f}")
            logger.info(f"  Current Price: {current_ticker_price:.2f}")
            logger.info(f"  Best Model: {best_model}")
            
            # Enhanced ensemble analysis
            ensemble_components = ml_analysis.get("ensemble_components", {})
            all_model_scores = ml_analysis.get("all_model_scores", {})
            
            logger.debug(f"ML Analysis - R2: {ml_r2:.3f}, MSE: {ml_mse:.1f}, MAE: {ml_mae:.2f}, Confidence: {ml_confidence:.3f}, Best: {best_model}")
            
            # ENHANCED: Smart model selection based on comprehensive metrics
            usable_models = []
            for model_name, metrics in all_model_scores.items():
                model_r2 = metrics.get("r2", -999)
                model_mse = metrics.get("mse", float('inf'))
                model_prediction = metrics.get("prediction", current_ticker_price)
                
                # Filter out clearly poor models
                if model_r2 > -2.0 and model_mse < 50000:  # Basic quality threshold
                    price_change = abs(model_prediction - current_ticker_price) / current_ticker_price
                    if price_change < 0.3:  # Prediction within 30% of current price
                        usable_models.append({
                            'name': model_name,
                            'r2': model_r2,
                            'mse': model_mse,
                            'prediction': model_prediction,
                            'quality_score': max(0, model_r2) + 1/(1 + model_mse/1000)  # Combined quality metric
                        })
            
            logger.debug(f"Usable models: {len(usable_models)}/{len(all_model_scores)}")
            
            # Calculate base ML score with improved logic
            if usable_models:
                # Use the best quality model instead of just the "best" model
                best_quality_model = max(usable_models, key=lambda x: x['quality_score'])
                ensemble_prediction = best_quality_model['prediction']
                ensemble_quality = best_quality_model['quality_score']
                
                # Also consider LSTM/Transformer if available
                lstm_prediction = ensemble_components.get("lstm", current_ticker_price)
                transformer_prediction = ensemble_components.get("transformer", current_ticker_price)
                
                # Weighted ensemble of top performers
                predictions = [ensemble_prediction]
                weights_list = [0.5]  # Base model gets 50%
                
                if abs(lstm_prediction - current_ticker_price) / current_ticker_price < 0.2:  # LSTM within 20%
                    predictions.append(lstm_prediction)
                    weights_list.append(0.25)
                    
                if abs(transformer_prediction - current_ticker_price) / current_ticker_price < 0.2:  # Transformer within 20%
                    predictions.append(transformer_prediction)
                    weights_list.append(0.25)
                
                # Normalize weights
                total_weight = sum(weights_list)
                weights_list = [w/total_weight for w in weights_list]
                
                # Calculate weighted prediction
                smart_ensemble_prediction = sum(p*w for p, w in zip(predictions, weights_list))
                
                logger.debug(f"Smart ensemble: {len(predictions)} models, prediction: {smart_ensemble_prediction:.2f}")
                
                # Use smart ensemble prediction instead of simple ML prediction
                predicted_price = smart_ensemble_prediction
            else:
                # Fall back to original prediction if no usable models
                predicted_price = ml_analysis.get("predicted_price", current_ticker_price)
                ensemble_quality = 0.1  # Low quality fallback
                logger.debug("No usable models found, using fallback prediction")
            
            # Enhanced validation with realistic bounds
            price_change_pct = (predicted_price - current_ticker_price) / current_ticker_price if current_ticker_price > 0 else 0
            
            # AGGRESSIVE DEBUG: Log price change calculation
            logger.info(f"  Price Change: {price_change_pct:.6f} ({price_change_pct*100:.3f}%)")
            
            # Validate prediction reasonableness
            if abs(price_change_pct) > 0.5:  # More than 50% change is suspicious
                logger.info(f"  CAPPING extreme prediction: {price_change_pct:.3f} -> ±0.5")
                price_change_pct = 0.5 if price_change_pct > 0 else -0.5
            
            # Calculate base ML score with quality weighting - FIXED: Always calculate if price_change_pct exists
            base_ml_score = min(max(abs(price_change_pct) * 6, 0.0), 1.0)  # Increased scaling for stronger signals
            
            # AGGRESSIVE DEBUG: Log base score calculation
            logger.info(f"  Base ML Score: {base_ml_score:.6f} (|{price_change_pct:.6f}| * 6)")
            
            # Apply multiple quality multipliers
            quality_multipliers = []
            
            # R2-based quality (more lenient for negative R2)
            if ml_r2 > 0.2:  # Good quality
                r2_multiplier = 1.0
            elif ml_r2 > 0.0:  # Positive but low quality
                r2_multiplier = 0.7 + ml_r2 * 1.5  # More generous scaling
            elif ml_r2 > -0.3:  # Slightly negative but potentially useful
                r2_multiplier = 0.5 + (ml_r2 + 0.3) * 1.67  # 0.5 to 1.0
            else:  # Very poor quality
                r2_multiplier = 0.3  # Still allow some signal
            quality_multipliers.append(r2_multiplier)
            
            # AGGRESSIVE DEBUG: Log R2 multiplier calculation
            logger.info(f"  R2 Multiplier: {r2_multiplier:.6f} (R2: {ml_r2:.6f})")
            
            # Confidence-based quality (more lenient)
            if abs(ml_confidence) > 0.4:  # High confidence
                confidence_multiplier = 1.0
            elif abs(ml_confidence) > 0.2:  # Medium confidence
                confidence_multiplier = 0.8
            elif abs(ml_confidence) > 0.1:  # Low confidence
                confidence_multiplier = 0.6
            else:  # Very low confidence
                confidence_multiplier = 0.4  # Still allow some signal
            quality_multipliers.append(confidence_multiplier)
            
            # Model consensus quality (if multiple models agree)
            if usable_models and len(usable_models) > 1:
                predictions_range = max([m['prediction'] for m in usable_models]) - min([m['prediction'] for m in usable_models])
                relative_range = predictions_range / current_ticker_price
                if relative_range < 0.1:  # Models agree within 10%
                    consensus_multiplier = 1.2
                elif relative_range < 0.2:  # Models agree within 20%
                    consensus_multiplier = 1.0
                else:  # Models disagree significantly
                    consensus_multiplier = 0.7
                quality_multipliers.append(consensus_multiplier)
            
            # Calculate final quality multiplier
            final_quality = min(sum(quality_multipliers) / len(quality_multipliers), 1.0)
            
            # AGGRESSIVE DEBUG: Log quality calculation
            logger.info(f"  Quality Multipliers: {quality_multipliers}")
            logger.info(f"  Final Quality: {final_quality:.6f}")
            
            # FIXED: Use ML signal regardless of direction for buy signals, but weight appropriately
            if price_change_pct > 0.002:  # Very low threshold (0.2%)
                signal_strengths["ml"] = base_ml_score * final_quality
                logger.info(f"  ML POSITIVE signal: change={price_change_pct:.6f}, base={base_ml_score:.6f}, quality={final_quality:.6f}, FINAL={signal_strengths['ml']:.6f}")
            elif price_change_pct > -0.005:  # Small negative prediction - still some buy potential
                signal_strengths["ml"] = base_ml_score * final_quality * 0.4  # Less penalty
                logger.info(f"  ML SLIGHTLY NEGATIVE signal: change={price_change_pct:.6f}, base={base_ml_score:.6f}, quality={final_quality:.6f}, FINAL={signal_strengths['ml']:.6f}")
            else:  # Strong negative prediction
                signal_strengths["ml"] = 0.0
                logger.info(f"  ML NEGATIVE signal: change={price_change_pct:.6f}, no buy signal, FINAL=0.000000")
            
            logger.debug(f"Quality components - R2: {r2_multiplier:.3f}, Confidence: {confidence_multiplier:.3f}, Final: {final_quality:.3f}")
        else:
            signal_strengths["ml"] = 0.0
            logger.info(f"=== ML ANALYSIS DEBUG for {ticker} ===")
            logger.info(f"  ML Analysis Failed: success={ml_analysis.get('success')}, has_predicted_price={bool(ml_analysis.get('predicted_price'))}")            
            logger.info(f"  ML signal: 0.000000 (no ML analysis available or failed)")

        # ENHANCED: RSI signal strength with progressive thresholds
        logger.debug(f"RSI value: {rsi_value:.2f}")
        
        if rsi_value < 45:  # Expanded from 40 to capture more oversold signals
            # Progressive RSI scoring - stronger signals for more oversold conditions
            if rsi_value < 25:  # Extremely oversold
                signal_strengths["rsi"] = 1.0
                logger.debug(f"RSI extremely oversold: {rsi_value:.2f} -> 1.000")
            elif rsi_value < 35:  # Very oversold
                signal_strengths["rsi"] = 0.8 + (35 - rsi_value) / 10 * 0.2  # 0.8 to 1.0
                logger.debug(f"RSI very oversold: {rsi_value:.2f} -> {signal_strengths['rsi']:.3f}")
            else:  # Moderately oversold (35-45)
                signal_strengths["rsi"] = (45 - rsi_value) / 10 * 0.6  # 0.0 to 0.6
                logger.debug(f"RSI moderately oversold: {rsi_value:.2f} -> {signal_strengths['rsi']:.3f}")
        elif rsi_value > 55:  # Also capture momentum in trending markets
            # Upward momentum signal (but weaker than oversold)
            if rsi_value < 65:  # Healthy uptrend
                signal_strengths["rsi"] = (rsi_value - 55) / 10 * 0.3  # 0.0 to 0.3
                logger.debug(f"RSI upward momentum: {rsi_value:.2f} -> {signal_strengths['rsi']:.3f}")
            else:  # Potential overbought (reduce signal)
                signal_strengths["rsi"] = max(0.1, 0.3 - (rsi_value - 65) / 10 * 0.2)  # Diminishing returns
                logger.debug(f"RSI potential overbought: {rsi_value:.2f} -> {signal_strengths['rsi']:.3f}")
        else:
            # Neutral RSI (45-55) - minimal signal but not zero
            signal_strengths["rsi"] = 0.05  # Small base signal to avoid complete zero
            logger.debug(f"RSI neutral: {rsi_value:.2f} -> {signal_strengths['rsi']:.3f}")

        # ENHANCED: Professional MACD signal strength with trend validation and robust bounds checking
        macd_val = technical_indicators.get("macd", 0)
        signal_val = technical_indicators.get("signal_line", 0)
        
        logger.debug(f"MACD values - MACD: {macd_val:.4f}, Signal: {signal_val:.4f}")
        
        # Validate MACD values for sanity
        if abs(macd_val) > 1000 or abs(signal_val) > 1000:
            logger.warning(f"Extreme MACD values detected: MACD={macd_val:.4f}, Signal={signal_val:.4f}. Using default.")
            signal_strengths["macd"] = 0.3
            logger.debug("MACD extreme values, using default signal: 0.300")
        elif macd_val != 0 and signal_val != 0:  # Ensure we have valid MACD data
            macd_diff = macd_val - signal_val
            
            # Additional bounds checking for extreme differences
            if abs(macd_diff) > 100:
                logger.warning(f"Extreme MACD difference detected: {macd_diff:.4f}. Capping calculation.")
                # Cap the difference to prevent extreme signals
                macd_diff = 100 if macd_diff > 0 else -100
            
            if macd_val > signal_val:  # Bullish MACD crossover
                # Enhanced bounds checking for signal value denominator
                if abs(signal_val) > 0.01:  # Stricter threshold to avoid division by very small numbers
                    strength_ratio = abs(macd_diff) / abs(signal_val)
                    # Additional bounds checking on strength ratio
                    if strength_ratio > 10.0:  # Cap extreme ratios
                        logger.debug(f"Capping extreme MACD strength ratio: {strength_ratio:.3f} -> 10.0")
                        strength_ratio = 10.0
                    signal_strengths["macd"] = min(strength_ratio * 0.1, 1.0)  # More conservative scaling (was 0.5)
                elif abs(signal_val) > 0.001:  # Moderate small numbers
                    # Use reduced signal strength for small denominators
                    strength_ratio = abs(macd_diff) / abs(signal_val)
                    strength_ratio = min(strength_ratio, 5.0)  # Cap at 5.0
                    signal_strengths["macd"] = min(strength_ratio * 0.05, 0.5)  # Very conservative scaling
                else:
                    # Very small signal line - use conservative default
                    signal_strengths["macd"] = 0.4  # Reduced from 0.7 for safety
                
                logger.debug(f"MACD bullish: diff={macd_diff:.4f}, strength={signal_strengths['macd']:.3f}")
            else:  # Bearish MACD
                # Still provide minimal signal for potential reversal
                signal_strengths["macd"] = 0.1
                logger.debug(f"MACD bearish: diff={macd_diff:.4f}, minimal signal=0.100")
        else:
            # Default signal when MACD data unavailable
            signal_strengths["macd"] = 0.3
            logger.debug("MACD unavailable, using default signal: 0.300")

        # ENHANCED: Professional breakout signal with volume and momentum confirmation
        if resistance_level and resistance_level > 0:
            resistance_breakout_threshold = resistance_level * 1.01
            
            if current_ticker_price > resistance_level:
                # Price above resistance - calculate breakout strength
                breakout_distance = current_ticker_price - resistance_level
                resistance_range = resistance_level * 0.03  # 3% range for normalization
                
                # Base breakout strength
                base_strength = min(breakout_distance / resistance_range, 1.0)
                
                # Volume confirmation bonus
                volume_bonus = 1.0
                if volume_ratio > 1.5:  # High volume confirmation
                    volume_bonus = 1.2
                elif volume_ratio > 1.2:  # Moderate volume
                    volume_bonus = 1.1
                elif volume_ratio < 0.8:  # Low volume - penalty
                    volume_bonus = 0.8
                
                signal_strengths["breakout"] = min(base_strength * volume_bonus, 1.0)
                logger.debug(f"Breakout above resistance: distance={breakout_distance:.2f}, volume_ratio={volume_ratio:.2f}, strength={signal_strengths['breakout']:.3f}")
            else:
                # Price below resistance - calculate proximity strength
                proximity_range = resistance_level * 0.03  # 3% range below resistance
                proximity_distance = current_ticker_price - (resistance_level * 0.97)
                
                if proximity_distance > 0:  # Within 3% of resistance
                    proximity_strength = proximity_distance / proximity_range
                    signal_strengths["breakout"] = min(proximity_strength * 0.6, 0.6)  # Max 0.6 for proximity
                    logger.debug(f"Near resistance: proximity={proximity_distance:.2f}, strength={signal_strengths['breakout']:.3f}")
                else:
                    signal_strengths["breakout"] = 0.1  # Minimal signal when far from resistance
                    logger.debug(f"Far from resistance: {current_ticker_price:.2f} vs {resistance_level:.2f}, minimal signal")
        else:
            # No resistance data - use price momentum as proxy
            if ma_20 and current_ticker_price > ma_20 * 1.02:  # 2% above MA20
                signal_strengths["breakout"] = 0.4
                logger.debug("No resistance data, using MA20 momentum: 0.400")
            else:
                signal_strengths["breakout"] = 0.2
                logger.debug("No resistance/momentum data, default: 0.200")

        # ENHANCED: Professional volume signal with adaptive thresholds and market context
        volume_ratio = technical_indicators.get("volume_ratio", 1.0)
        
        # Get additional volume context if available
        historical_volume_data = history.get("Volume", pd.Series()) if history is not None else pd.Series()
        
        logger.debug(f"Volume ratio: {volume_ratio:.2f}")
        
        if volume_ratio > 1.0:
            # Progressive volume signal strength with market context adjustment
            if volume_ratio > 4.0:  # Exceptional volume (rare events)
                base_volume_signal = 1.0
                logger.debug(f"Exceptional volume spike: {volume_ratio:.2f} -> base=1.000")
            elif volume_ratio > 2.5:  # Very high volume
                base_volume_signal = 0.85 + (volume_ratio - 2.5) * 0.1  # 0.85 to 1.0
                logger.debug(f"Very high volume: {volume_ratio:.2f} -> base={base_volume_signal:.3f}")
            elif volume_ratio > 1.8:  # High volume
                base_volume_signal = 0.7 + (volume_ratio - 1.8) * 0.21  # 0.7 to 0.85
                logger.debug(f"High volume: {volume_ratio:.2f} -> base={base_volume_signal:.3f}")
            elif volume_ratio > 1.3:  # Above average volume
                base_volume_signal = 0.5 + (volume_ratio - 1.3) * 0.4  # 0.5 to 0.7
                logger.debug(f"Above average volume: {volume_ratio:.2f} -> base={base_volume_signal:.3f}")
            elif volume_ratio > 1.1:  # Slightly above average
                base_volume_signal = 0.3 + (volume_ratio - 1.1) * 1.0  # 0.3 to 0.5
                logger.debug(f"Slightly above average volume: {volume_ratio:.2f} -> base={base_volume_signal:.3f}")
            else:  # Minimal above average (1.0-1.1)
                base_volume_signal = (volume_ratio - 1.0) * 3  # 0.0 to 0.3
                logger.debug(f"Minimal above average volume: {volume_ratio:.2f} -> base={base_volume_signal:.3f}")
            
            # Market regime adjustment for volume signals
            if market_regime == "VOLATILE":
                # In volatile markets, high volume is more meaningful
                regime_multiplier = 1.2
            elif market_regime == "TRENDING":
                # In trending markets, volume confirms direction
                regime_multiplier = 1.1
            else:  # RANGE_BOUND
                # In range-bound markets, volume spikes may indicate breakouts
                regime_multiplier = 1.0
            
            signal_strengths["volume"] = min(base_volume_signal * regime_multiplier, 1.0)
            logger.debug(f"Volume signal after regime adjustment: {signal_strengths['volume']:.3f} (regime: {market_regime}, multiplier: {regime_multiplier:.2f})")
        else:
            # Below average volume - provide context-dependent minimal signal
            if market_regime == "TRENDING" and volume_ratio > 0.7:
                # In trending markets, even moderate volume can be okay
                signal_strengths["volume"] = volume_ratio * 0.4  # 0.0 to 0.28
                logger.debug(f"Below average volume in trending market: {volume_ratio:.2f} -> {signal_strengths['volume']:.3f}")
            elif volume_ratio > 0.5:
                # Moderate low volume
                signal_strengths["volume"] = volume_ratio * 0.3  # 0.0 to 0.15
                logger.debug(f"Moderate low volume: {volume_ratio:.2f} -> {signal_strengths['volume']:.3f}")
            else:
                # Very low volume - minimal signal
                signal_strengths["volume"] = 0.05  # Always provide minimal signal
                logger.debug(f"Very low volume: {volume_ratio:.2f} -> 0.050 (minimal)")

        # CRITICAL FIX: Now calculate weighted signal score AFTER all signal_strengths are populated
        weighted_signal_score = sum(
            signal_strengths.get(signal, 0.0) * weight
            for signal, weight in signal_weights.items()
        )
        
        # Apply market regime bonus/penalty  
        if market_regime == "TRENDING" and weighted_signal_score > 0.3:
            regime_bonus = min(weighted_signal_score * 0.1, 0.1)  # Up to 10% bonus
            weighted_signal_score += regime_bonus
            logger.debug(f"Trending market bonus: +{regime_bonus:.3f}")
        elif market_regime == "VOLATILE" and weighted_signal_score > 0.4:
            regime_penalty = min(weighted_signal_score * 0.05, 0.05)  # Up to 5% penalty
            weighted_signal_score -= regime_penalty
            logger.debug(f"Volatile market penalty: -{regime_penalty:.3f}")
        
        # Final bounds check
        weighted_signal_score = max(0.0, min(weighted_signal_score, 1.0))
        
        logger.info(f"CORRECTED WEIGHTED SIGNAL SCORE: {weighted_signal_score:.3f}")

        # DEBUG: Log weighted signal breakdown
        logger.info(f"=== WEIGHTED SIGNAL BREAKDOWN for {ticker} ===")
        for signal, strength in signal_strengths.items():
            weight = signal_weights[signal]
            contribution = strength * weight
            logger.info(f"  {signal}: strength={strength:.3f} × weight={weight:.3f} = {contribution:.3f}")
        logger.info(f"  TOTAL WEIGHTED SIGNAL SCORE: {weighted_signal_score:.3f}")

        # Convert to legacy buy_signals for compatibility (but use weighted logic)
        buy_signals = int(weighted_signal_score * 7)  # Scale to 0-7 for logging compatibility

        # Calculate weighted sell signal score (similar to buy signals)
        sell_signal_strengths = {}

        # Technical sell signal strength
        sell_signal_strengths["technical"] = min(max(sell_technical_score / 0.4, 0.0), 1.0)

        # Sentiment sell signal strength
        sell_signal_strengths["sentiment"] = min(max(abs(sentiment_score) / 0.2, 0.0), 1.0) if sentiment_score < 0 else 0.0

        # ML sell signal strength
        sell_signal_strengths["ml"] = min(max(abs(ml_score) / 0.1, 0.0), 1.0) if ml_score < 0 else 0.0

        # RSI overbought signal strength
        if rsi_value > 65:
            sell_signal_strengths["rsi"] = min((rsi_value - 65) / 15, 1.0)  # Stronger as RSI gets higher
        else:
            sell_signal_strengths["rsi"] = 0.0

        # MACD bearish signal strength with robust bounds checking
        if macd_val != 0 and signal_val != 0:  # Ensure valid MACD data
            # Validate MACD values for sanity
            if abs(macd_val) > 1000 or abs(signal_val) > 1000:
                logger.warning(f"Extreme MACD values in sell calculation: MACD={macd_val:.4f}, Signal={signal_val:.4f}")
                sell_signal_strengths["macd"] = 0.0
            elif macd_val < signal_val:  # Bearish crossover
                macd_diff = abs(macd_val - signal_val)
                # Cap extreme differences
                if macd_diff > 100:
                    logger.warning(f"Extreme MACD difference in sell: {macd_diff:.4f}. Capping.")
                    macd_diff = 100
                
                # Enhanced bounds checking for denominator
                if abs(signal_val) > 0.01:  # Stricter threshold
                    strength_ratio = macd_diff / abs(signal_val)
                    # Cap extreme ratios
                    if strength_ratio > 10.0:
                        strength_ratio = 10.0
                    sell_signal_strengths["macd"] = min(strength_ratio * 0.1, 1.0)  # Conservative scaling
                elif abs(signal_val) > 0.001:  # Moderate small numbers
                    strength_ratio = min(macd_diff / abs(signal_val), 5.0)
                    sell_signal_strengths["macd"] = min(strength_ratio * 0.05, 0.5)
                else:
                    sell_signal_strengths["macd"] = 0.3  # Conservative default
            else:
                sell_signal_strengths["macd"] = 0.0
        else:
            sell_signal_strengths["macd"] = 0.0

        # Support breakdown signal strength
        if current_ticker_price < support_level:
            sell_signal_strengths["breakdown"] = min((support_level - current_ticker_price) / (support_level * 0.02), 1.0)
        else:
            sell_signal_strengths["breakdown"] = 0.0

        # Risk management signal strength
        if unrealized_pnl < 0:
            sell_signal_strengths["risk"] = min(abs(unrealized_pnl) / (0.05 * total_value), 1.0)
        else:
            sell_signal_strengths["risk"] = 0.0

        # Calculate weighted sell signal score
        sell_signal_weights = {
            "technical": 0.25, "sentiment": 0.15, "ml": 0.15, "rsi": 0.15,
            "macd": 0.10, "breakdown": 0.10, "risk": 0.10
        }

        weighted_sell_signal_score = sum(
            sell_signal_strengths.get(signal, 0.0) * sell_signal_weights[signal]
            for signal in sell_signal_weights.keys()
        )

        # Log detailed sell signal breakdown
        logger.info(f"=== WEIGHTED SELL SIGNAL BREAKDOWN for {ticker} ===")
        for signal, strength in sell_signal_strengths.items():
            weight = sell_signal_weights[signal]
            contribution = strength * weight
            logger.info(f"  {signal}: strength={strength:.3f} × weight={weight:.3f} = {contribution:.3f}")
        logger.info(f"  TOTAL WEIGHTED SELL SIGNAL SCORE: {weighted_sell_signal_score:.3f}")

        # Convert to legacy sell_signals for compatibility
        sell_signals = int(weighted_sell_signal_score * 7)

        # Portfolio Constraints (REMOVED SECTOR EXPOSURE LIMITS)
        max_exposure_per_stock = total_value * 0.25
        current_stock_exposure = self.portfolio.holdings.get(ticker, {"qty": 0})["qty"] * current_ticker_price

        # DEBUG: Log portfolio constraints (simplified - no sector limits)
        logger.info(f"=== PORTFOLIO CONSTRAINTS for {ticker} ===")
        logger.info(f"  Total Portfolio Value: Rs.{total_value:.2f}")
        logger.info(f"  Available Cash: Rs.{available_cash:.2f}")
        logger.info(f"  Max Stock Exposure (25%): Rs.{max_exposure_per_stock:.2f}")
        logger.info(f"  Current Stock Exposure: Rs.{current_stock_exposure:.2f}")
        logger.info(f"  Remaining Stock Capacity: Rs.{max_exposure_per_stock - current_stock_exposure:.2f}")
        logger.info(f"  SECTOR EXPOSURE LIMITS: DISABLED (removed constraint)")

        # Calculate ATR for volatility-based sizing
        high_low = history['High'] - history['Low']
        high_close = abs(history['High'] - history['Close'].shift())
        low_close = abs(history['Low'] - history['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        atr = float(atr) if not pd.isna(atr) else volatility * current_ticker_price

        # ENHANCED: Professional Kelly Criterion with comprehensive win rate estimation and risk adjustment
        def calculate_advanced_kelly_fraction():
            """Calculate Kelly fraction using multiple sophisticated approaches"""
            
            # Method 1: Historical Performance-Based Kelly
            historical_kelly = 0.15  # Default
            
            # Get recent trading history for this specific ticker
            ticker_trades = [t for t in self.portfolio.trade_log if t.get("asset") == ticker]
            
            if len(ticker_trades) >= 10:  # Need sufficient data
                # Calculate actual win rate and average win/loss for this ticker
                completed_trades = []
                for i, trade in enumerate(ticker_trades):
                    if trade["action"] == "buy":
                        # Look for corresponding sell trade
                        for j in range(i+1, len(ticker_trades)):
                            if ticker_trades[j]["action"] == "sell":
                                pnl = (ticker_trades[j]["price"] - trade["price"]) * trade["qty"]
                                pnl_pct = pnl / (trade["price"] * trade["qty"])
                                completed_trades.append(pnl_pct)
                                break
                
                if len(completed_trades) >= 5:
                    wins = [t for t in completed_trades if t > 0]
                    losses = [t for t in completed_trades if t < 0]
                    
                    if len(wins) > 0 and len(losses) > 0:
                        actual_win_rate = len(wins) / len(completed_trades)
                        avg_win = np.mean(wins)
                        avg_loss = abs(np.mean(losses))
                        
                        # Kelly formula: f = (p*b - q) / b, where b = avg_win/avg_loss
                        if avg_loss > 0:
                            b = avg_win / avg_loss
                            historical_kelly = (actual_win_rate * b - (1 - actual_win_rate)) / b
                            historical_kelly = max(0.02, min(historical_kelly, 0.40))  # Bounds: 2-40%
                            
                            logger.debug(f"Historical Kelly for {ticker}: win_rate={actual_win_rate:.3f}, avg_win={avg_win:.3f}, avg_loss={avg_loss:.3f}, kelly={historical_kelly:.3f}")
            
            # Method 2: Signal Quality-Based Win Rate Estimation
            signal_quality_kelly = 0.15  # Default
            
            # Estimate win probability based on signal strength and market conditions
            avg_signal_strength = sum(signal_strengths.values()) / len(signal_strengths) if signal_strengths else 0
            
            # Base win rate estimation
            if avg_signal_strength > 0.8:  # Very strong signals
                estimated_win_rate = 0.78
            elif avg_signal_strength > 0.6:  # Strong signals
                estimated_win_rate = 0.68 + (avg_signal_strength - 0.6) * 0.5  # 68-78%
            elif avg_signal_strength > 0.4:  # Moderate signals
                estimated_win_rate = 0.58 + (avg_signal_strength - 0.4) * 0.5  # 58-68%
            elif avg_signal_strength > 0.2:  # Weak signals
                estimated_win_rate = 0.48 + (avg_signal_strength - 0.2) * 0.5  # 48-58%
            else:  # Very weak signals
                estimated_win_rate = 0.45
            
            # Adjust for market regime
            if market_regime == "TRENDING":
                estimated_win_rate *= 1.05  # 5% boost in trending markets
            elif market_regime == "VOLATILE":
                estimated_win_rate *= 0.93  # 7% penalty in volatile markets
            
            # Adjust for confidence score
            confidence_multiplier = 0.90 + (final_buy_score * 0.20)  # 90-110% based on confidence
            estimated_win_rate *= confidence_multiplier
            
            # Bound the win rate
            estimated_win_rate = max(0.45, min(estimated_win_rate, 0.85))
            
            # Calculate expected returns with enhanced validation
            ml_expected_return = 0.0
            if predicted_price and current_ticker_price > 0:
                ml_return_raw = (predicted_price / current_ticker_price - 1)
                
                # Enhanced ML return validation
                historical_volatility = history['Close'].pct_change().std() * np.sqrt(252) if history is not None else 0.20
                max_reasonable_return = min(0.25, historical_volatility * 3)  # Cap at 25% or 3x volatility
                
                ml_expected_return = max(-0.10, min(ml_return_raw, max_reasonable_return))  # -10% to reasonable max
            
            # Technical expected return (based on support/resistance levels)
            technical_expected_return = 0.0
            if resistance_level and current_ticker_price > 0:
                technical_return_raw = (resistance_level / current_ticker_price - 1)
                technical_expected_return = max(0.0, min(technical_return_raw, 0.20))  # 0-20% cap
            
            # Sentiment expected return
            sentiment_expected_return = 0.0
            if processed_sentiment_score > 0:
                sentiment_expected_return = processed_sentiment_score * 0.08  # Up to 8% from sentiment
            
            # Weighted expected return calculation
            total_expected_return = (
                ml_expected_return * 0.4 +
                technical_expected_return * 0.4 +
                sentiment_expected_return * 0.2
            )
            
            # Ensure positive expected return for Kelly calculation
            total_expected_return = max(0.01, min(total_expected_return, 0.15))  # 1-15% range
            
            # Average loss estimation (more conservative)
            estimated_avg_loss = min(0.08, volatility * 4)  # Up to 8% or 4x volatility
            
            # Kelly calculation with enhanced formula
            if estimated_avg_loss > 0:
                odds_ratio = total_expected_return / estimated_avg_loss
                signal_quality_kelly = (estimated_win_rate * odds_ratio - (1 - estimated_win_rate)) / odds_ratio
                signal_quality_kelly = max(0.02, min(signal_quality_kelly, 0.35))  # 2-35% bounds
            
            logger.debug(f"Signal-based Kelly: win_rate={estimated_win_rate:.3f}, exp_return={total_expected_return:.3f}, avg_loss={estimated_avg_loss:.3f}, kelly={signal_quality_kelly:.3f}")
            
            # Method 3: Confidence and Market-Weighted Kelly
            confidence_kelly = 0.15  # Default
            
            # Base Kelly from confidence
            confidence_kelly = 0.08 + (final_buy_score * 0.25)  # 8-33% based on buy score
            
            # Weighted signal adjustment
            if weighted_signal_score > 0.5:
                confidence_kelly *= (1.0 + (weighted_signal_score - 0.5))  # Boost for strong signals
            else:
                confidence_kelly *= (0.7 + weighted_signal_score * 0.6)  # Reduce for weak signals
            
            # Market regime adjustment
            regime_multipliers = {
                "TRENDING": 1.15,     # More aggressive in trends
                "VOLATILE": 0.75,     # More conservative in volatility
                "RANGE_BOUND": 0.95   # Slightly conservative in ranges
            }
            confidence_kelly *= regime_multipliers.get(market_regime, 1.0)
            
            confidence_kelly = max(0.05, min(confidence_kelly, 0.40))  # 5-40% bounds
            
            logger.debug(f"Confidence-based Kelly: base={0.08 + (final_buy_score * 0.25):.3f}, signal_adj={weighted_signal_score:.3f}, regime_mult={regime_multipliers.get(market_regime, 1.0):.2f}, final={confidence_kelly:.3f}")
            
            # Final Kelly: Weighted combination of all methods
            kelly_weights = []
            kelly_values = []
            
            # Weight historical Kelly higher if we have good data
            if len(ticker_trades) >= 10:
                kelly_weights.append(0.5)  # 50% weight for historical
                kelly_values.append(historical_kelly)
                kelly_weights.append(0.3)  # 30% for signal-based
                kelly_values.append(signal_quality_kelly)
                kelly_weights.append(0.2)  # 20% for confidence-based
                kelly_values.append(confidence_kelly)
            else:
                kelly_weights.append(0.6)  # 60% for signal-based
                kelly_values.append(signal_quality_kelly)
                kelly_weights.append(0.4)  # 40% for confidence-based
                kelly_values.append(confidence_kelly)
            
            # Calculate weighted Kelly
            final_kelly = sum(k * w for k, w in zip(kelly_values, kelly_weights))
            
            # Apply additional risk adjustments
            
            # Volatility adjustment (more sophisticated)
            if volatility > 0.04:  # High volatility (>4%)
                vol_adjustment = 0.70
            elif volatility > 0.025:  # Medium volatility (2.5-4%)
                vol_adjustment = 0.85
            elif volatility < 0.015:  # Low volatility (<1.5%)
                vol_adjustment = 1.15
            else:  # Normal volatility
                vol_adjustment = 1.0
            
            final_kelly *= vol_adjustment
            
            # Portfolio heat adjustment (reduce Kelly if portfolio is already stressed)
            portfolio_metrics = self.portfolio.get_metrics()
            current_positions = len([h for h in self.portfolio.holdings.values() if h["qty"] > 0])
            
            if current_positions >= 8:  # Many positions - reduce Kelly
                position_adjustment = 0.80
            elif current_positions >= 5:  # Several positions - slightly reduce
                position_adjustment = 0.90
            else:  # Few positions - normal Kelly
                position_adjustment = 1.0
            
            final_kelly *= position_adjustment
            
            # Recent performance adjustment
            if hasattr(self.portfolio, 'trade_log') and len(self.portfolio.trade_log) > 0:
                recent_trades = [t for t in self.portfolio.trade_log if 
                               datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S.%f") > datetime.now() - timedelta(days=5)]
                
                if len(recent_trades) >= 3:
                    recent_success_rate = len([t for t in recent_trades if t.get("success", False)]) / len(recent_trades)
                    
                    if recent_success_rate > 0.7:  # Good recent performance
                        performance_adjustment = 1.1
                    elif recent_success_rate < 0.4:  # Poor recent performance
                        performance_adjustment = 0.8
                    else:
                        performance_adjustment = 1.0
                    
                    final_kelly *= performance_adjustment
            
            # Final bounds with more aggressive limits for profitable system
            final_kelly = max(0.08, min(final_kelly, 0.35))  # 8-35% range (was 12-35%)
            
            return {
                "final_kelly": final_kelly,
                "components": {
                    "historical_kelly": historical_kelly,
                    "signal_quality_kelly": signal_quality_kelly,
                    "confidence_kelly": confidence_kelly
                },
                "adjustments": {
                    "volatility": vol_adjustment,
                    "position_count": position_adjustment,
                    "estimated_win_rate": estimated_win_rate,
                    "total_expected_return": total_expected_return
                },
                "weights": dict(zip(["historical", "signal_quality", "confidence"], kelly_weights)) if len(kelly_weights) == 3 else {"signal_quality": kelly_weights[0], "confidence": kelly_weights[1]}
            }
        
        # Calculate the advanced Kelly fraction
        kelly_result = calculate_advanced_kelly_fraction()
        kelly_fraction = kelly_result["final_kelly"]
        
        # Enhanced logging for Kelly calculation
        logger.info(f"=== ADVANCED KELLY FRACTION CALCULATION for {ticker} ====")
        logger.info(f"  Final Kelly Fraction: {kelly_fraction:.3f} (8-35% bounds)")
        
        components = kelly_result["components"]
        logger.info(f"  Component Kellys:")
        for name, value in components.items():
            logger.info(f"    - {name}: {value:.3f}")
        
        weights = kelly_result["weights"]
        logger.info(f"  Component Weights: {weights}")
        
        adjustments = kelly_result["adjustments"]
        logger.info(f"  Risk Adjustments:")
        logger.info(f"    - Volatility factor: {adjustments['volatility']:.2f}")
        logger.info(f"    - Position count factor: {adjustments['position_count']:.2f}")
        logger.info(f"    - Estimated win rate: {adjustments['estimated_win_rate']:.3f}")
        logger.info(f"    - Expected return: {adjustments['total_expected_return']:.3f}")

        # Liquidity Check
        avg_daily_volume = history["Volume"].rolling(window=20).mean().iloc[-1]
        max_trade_volume = avg_daily_volume * 0.015
        max_qty_by_volume = max_trade_volume if not pd.isna(max_trade_volume) else float('inf')

        # Stop-Loss and Take-Profit
        stop_loss = support_level * 0.97 if support_level > 0 else current_ticker_price * 0.94
        take_profit = resistance_level * 1.03 if resistance_level > 0 else current_ticker_price * 1.12

        # Trailing Stop-Loss
        if ticker in self.portfolio.holdings:
            # avg_price = self.portfolio.holdings[ticker]["avg_price"]  # Not used in current logic
            trailing_stop_pct = 0.06
            highest_price = max(history["Close"].iloc[-30:]) if history is not None and not history.empty else current_ticker_price
            trailing_stop = highest_price * (1 - trailing_stop_pct)
            if current_ticker_price < trailing_stop:
                logger.info(f"Trailing Stop-Loss triggered for {ticker}: Price Rs.{current_ticker_price:.2f} < Trailing Stop Rs.{trailing_stop:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": trailing_stop,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "trailing_stop_loss"
                }

        # Stop-Loss and Take-Profit
        if ticker in self.portfolio.holdings:
            if current_ticker_price < stop_loss:
                logger.info(f"Stop-Loss triggered for {ticker}: Price Rs.{current_ticker_price:.2f} < Stop-Loss Rs.{stop_loss:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "stop_loss"
                }
            elif current_ticker_price > take_profit:
                logger.info(f"Take-Profit triggered for {ticker}: Price Rs.{current_ticker_price:.2f} > Take-Profit Rs.{take_profit:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "take_profit"
                }

        # Max Holding Period
        max_holding_days = 60
        if ticker in self.portfolio.holdings:
            first_trade = min(
                (datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S.%f") for t in self.portfolio.trade_log if t["asset"] == ticker and t["action"] == "buy"),
                default=datetime.now()
            )
            holding_days = (datetime.now() - first_trade).days
            if holding_days > max_holding_days and unrealized_pnl < 0:
                logger.info(f"Max holding period exceeded for {ticker}: {holding_days} days, Unrealized PnL: Rs.{unrealized_pnl:.2f}")
                holding_qty = self.portfolio.holdings[ticker]["qty"]
                success = self.executor.execute_trade(ticker, "sell", holding_qty, current_ticker_price)
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "qty": holding_qty,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": success,
                    "confidence_score": final_sell_score,
                    "signals": sell_signals,
                    "reason": "max_holding_period"
                }

        # PRODUCTION FIX: More aggressive buy/sell decisions
        buy_qty = 0
        # Lowered signal requirement from 2 to 1 for more trading activity

        # PRODUCTION FIX: Weighted signal threshold system
        # Define minimum weighted signal threshold based on market regime
        # Enhanced market regime detection
        # FIXED: Lowered all thresholds to allow more trades to execute
        if market_regime == "STRONG_UPTREND":
            min_weighted_signal_threshold = 0.04  # Most aggressive in strong uptrends (FIXED: Reduced from 0.12 to 0.04)
        elif market_regime == "WEAK_UPTREND":
            min_weighted_signal_threshold = 0.06  # Moderately aggressive in weak uptrends (FIXED: Reduced from 0.15 to 0.06)
        elif market_regime == "STRONG_DOWNTREND":
            min_weighted_signal_threshold = 0.12  # Most conservative in strong downtrends (FIXED: Reduced from 0.30 to 0.12)
        elif market_regime == "WEAK_DOWNTREND":
            min_weighted_signal_threshold = 0.10  # Moderately conservative in weak downtrends (FIXED: Reduced from 0.25 to 0.10)
        elif market_regime == "HIGH_VOLATILITY":
            min_weighted_signal_threshold = 0.11  # Conservative in highly volatile markets (FIXED: Reduced from 0.28 to 0.11)
        elif market_regime == "LOW_VOLATILITY":
            min_weighted_signal_threshold = 0.07  # More opportunities in low volatility (FIXED: Reduced from 0.18 to 0.07)
        else:  # RANGE_BOUND
            min_weighted_signal_threshold = 0.08  # Moderate threshold for range-bound markets (FIXED: Reduced from 0.20 to 0.08)

        # Use more flexible signal system - either condition can trigger buy, with special handling for volatile markets
        volatile_market_adjustment = False
        if market_regime == "VOLATILE" and weighted_signal_score >= 0.10:  # Strong signals in volatile market (FIXED: Reduced from 0.25 to 0.10)
            # Allow lower buy score threshold when weighted signals are strong
            adjusted_buy_threshold = confidence_threshold * 0.75  # 25% reduction for strong signals
            volatile_market_adjustment = True
            logger.info(f"VOLATILE market adjustment: Reduced threshold from {confidence_threshold:.3f} to {adjusted_buy_threshold:.3f}")
        else:
            adjusted_buy_threshold = confidence_threshold

        # DEBUG: Log buy decision pre-checks with weighted system
        logger.info(f"=== BUY DECISION PRE-CHECKS for {ticker} ===")
        logger.info(f"  Buy Score: {final_buy_score:.3f} > Threshold: {adjusted_buy_threshold:.3f} = {'PASS' if final_buy_score > adjusted_buy_threshold else 'FAIL'}")
        if volatile_market_adjustment:
            logger.info(f"    (Volatile market adjustment applied: {confidence_threshold:.3f} -> {adjusted_buy_threshold:.3f})")
        logger.info(f"  Weighted Signal Score: {weighted_signal_score:.3f} > Threshold: {min_weighted_signal_threshold:.3f} = {'PASS' if weighted_signal_score >= min_weighted_signal_threshold else 'FAIL'}")
        logger.info(f"  Legacy Buy Signals (for reference): {buy_signals}/7")
        logger.info(f"  Market Regime: {market_regime} (threshold: {min_weighted_signal_threshold:.3f})")
            
        if final_buy_score > adjusted_buy_threshold or weighted_signal_score >= min_weighted_signal_threshold:
            # Early cash validation - skip expensive calculations if stock is unaffordable
            if current_ticker_price > available_cash:
                logger.info(f"[SKIP] {ticker}: Price Rs.{current_ticker_price:.2f} > Available Cash Rs.{available_cash:.2f}")
                buy_qty = 0  # Ensure buy_qty is 0 for unaffordable stocks
            elif available_cash < 100:  # Minimum trade value in INR
                logger.info(f"[SKIP] SKIPPING BUY for {ticker}: Insufficient cash (Rs.{available_cash:.2f} < Rs.100)")
                buy_qty = 0  # Ensure buy_qty is 0 for insufficient cash
            else:
                logger.info(f"[BUY] PROCEEDING with BUY calculation for {ticker}")

                # Position sizing calculation with adaptive Kelly fraction
                # Adjust Kelly fraction based on market conditions and volatility
                market_condition_factor = {
                    "TRENDING": 0.9,    # More aggressive in trending markets
                    "VOLATILE": 0.6,    # More conservative in volatile markets
                    "RANGE_BOUND": 0.8  # Moderate in range-bound markets
                }.get(market_regime, 0.7)
                
                volatility_adjustment = max(0.6, min(1.0, 1.0 / (1 + volatility)))
                
                # ENHANCEMENT: Standardize Position Sizing Usage
                # Instead of the current approach, use the DynamicPositionSizer consistently
                from utils.dynamic_position_sizer import get_position_sizer
                import pandas as pd
                
                # Create a DataFrame with historical data for the position sizer
                # This is a simplified version - in practice, you'd use real historical data
                historical_data = pd.DataFrame({
                    'Close': df['Close'].tail(50).values if len(df) >= 50 else df['Close'].values
                })
                
                # Prepare portfolio data for the position sizer
                portfolio_data = {
                    'total_value': total_value,
                    'cash': available_cash,
                    'holdings': self.portfolio.holdings
                }
                
                # Get the position sizer instance
                position_sizer = get_position_sizer(initial_capital=available_cash)
                
                # Determine market regime from context
                market_regime = market_regime or "NORMAL"
                
                # Calculate position size using the DynamicPositionSizer
                position_result = position_sizer.calculate_position_size(
                    symbol=ticker,
                    signal_strength=final_buy_score,
                    current_price=current_ticker_price,
                    volatility=volatility,
                    historical_data=historical_data,
                    portfolio_data=portfolio_data,
                    market_regime=market_regime
                )
                
                # Extract the calculated quantity
                buy_qty = position_result['quantity']
                
                # ENHANCEMENT: Add Position Sizing Logging
                logger.info(f"=== POSITION SIZING for {ticker} ===")
                logger.info(f"  Method used: {position_result['method_used']}")
                logger.info(f"  Base size: {position_result['base_size']:.3f}")
                logger.info(f"  Constrained size: {position_result['constrained_size']:.3f}")
                logger.info(f"  Actual size: {position_result['actual_size']:.3f}")
                logger.info(f"  Position value: Rs.{position_result['position_value']:,.2f}")
                logger.info(f"  Quantity: {position_result['quantity']} shares")
                logger.info(f"  Stop loss: Rs.{position_result['stop_loss']:.2f}")
                logger.info(f"  Constraints applied: {position_result['constraints_applied']}")
                
                # ENHANCEMENT: Enhance Cash Constraint Handling
                # Ensure the position sizer always respects available cash constraints
                if buy_qty * current_ticker_price > available_cash:
                    buy_qty = int(available_cash / current_ticker_price)
                    logger.info(f"  [CASH ADJUST] Reduced quantity to {buy_qty} shares to fit available cash")
                
                # Log the sizing components for debugging
                sizing_components = position_result.get('sizing_components', {})
                if sizing_components:
                    logger.info(f"  Sizing components:")
                    logger.info(f"    Kelly size: {sizing_components.get('kelly_size', 0):.3f}")
                    logger.info(f"    Volatility size: {sizing_components.get('volatility_size', 0):.3f}")
                    logger.info(f"    Risk parity size: {sizing_components.get('risk_parity_size', 0):.3f}")
                
                # Log risk metrics
                risk_metrics = position_result.get('risk_metrics', {})
                if risk_metrics:
                    logger.info(f"  Risk metrics:")
                    logger.info(f"    Portfolio risk: {risk_metrics.get('portfolio_risk_pct', 0):.2%}")
                    logger.info(f"    Max loss: {risk_metrics.get('max_loss_pct', 0):.2%}")
                    logger.info(f"    Stop loss price: Rs.{risk_metrics.get('stop_loss_price', 0):.2f}")
                    logger.info(f"    Sharpe estimate: {risk_metrics.get('sharpe_estimate', 0):.3f}")
                
                logger.info(f"  FINAL BUY QUANTITY: {buy_qty} shares (Rs.{buy_qty * current_ticker_price:.2f})")
        else:
            logger.info(f"[SKIP] BUY PRE-CHECKS FAILED for {ticker}")
            logger.info(f"   - Buy Score Check: {'PASS' if final_buy_score > confidence_threshold else 'FAIL'}")
            logger.info(f"   - Weighted Signal Check: {'PASS' if weighted_signal_score >= min_weighted_signal_threshold else 'FAIL'}")
            buy_qty = 0  # Ensure buy_qty is 0 when pre-checks fail

        # Sell Quantity Calculation with weighted signals
        sell_qty = 0
        min_weighted_sell_threshold = min_weighted_signal_threshold * 0.8  # Slightly lower threshold for sells

        # ENHANCED SELL LOGIC: More responsive sell conditions
        if ticker in self.portfolio.holdings:
            holding_qty = self.portfolio.holdings[ticker]["qty"]
            
            # Only proceed if we actually have shares to sell
            if holding_qty > 0:
                # Check various sell conditions
                stop_loss_triggered = current_ticker_price < stop_loss
                take_profit_triggered = current_ticker_price > take_profit
                sell_signal_strong = final_sell_score > sell_confidence_threshold * 0.7  # Lower threshold for selling
                weighted_sell_signal_met = weighted_sell_signal_score >= min_weighted_sell_threshold
                emergency_sell = unrealized_pnl < -0.05 * total_value  # 5% portfolio loss
                
                # Check if we're in testing mode (from logs)
                testing_mode = os.getenv("TESTING_MODE", "false").lower() == "true"
                
                # Log sell decision factors for debugging
                logger.info(f"=== SELL DECISION ANALYSIS for {ticker} ===")
                logger.info(f"  Position Size: {holding_qty} shares")
                logger.info(f"  Stop Loss Triggered: {stop_loss_triggered} (Price: {current_ticker_price:.2f} < Stop: {stop_loss:.2f})")
                logger.info(f"  Take Profit Triggered: {take_profit_triggered} (Price: {current_ticker_price:.2f} > Take Profit: {take_profit:.2f})")
                logger.info(f"  Sell Signal Strong: {sell_signal_strong} (Score: {final_sell_score:.3f} > Threshold: {sell_confidence_threshold * 0.7:.3f})")
                logger.info(f"  Weighted Sell Signal: {weighted_sell_signal_met} (Score: {weighted_sell_signal_score:.3f} >= Threshold: {min_weighted_sell_threshold:.3f})")
                logger.info(f"  Emergency Sell Condition: {emergency_sell} (Unrealized P&L: Rs.{unrealized_pnl:.2f} < {-0.05 * total_value:.2f})")
                logger.info(f"  Testing Mode Active: {testing_mode}")
                
                # Execute sell if any significant condition is met
                if stop_loss_triggered or take_profit_triggered or emergency_sell or sell_signal_strong or weighted_sell_signal_met or testing_mode:
                    # In testing mode, be more aggressive with selling
                    if testing_mode:
                        logger.info("🔧 TESTING MODE ACTIVE: Lowering requirements for sell execution")
                        # In testing mode, even modest sell signals should trigger sells
                        sell_qty = max(1, int(holding_qty * 0.5))  # Sell 50% in testing mode
                        logger.info(f"  [TESTING MODE SELL] Selling {sell_qty} of {holding_qty} shares")
                    elif stop_loss_triggered or emergency_sell:
                        # Emergency sell - liquidate entire position
                        sell_qty = holding_qty
                        logger.info(f"  [EMERGENCY SELL] Selling full position: {sell_qty} shares")
                    elif take_profit_triggered:
                        # Take profit - liquidate entire position
                        sell_qty = holding_qty
                        logger.info(f"  [TAKE PROFIT] Selling full position: {sell_qty} shares")
                    elif final_sell_score > 0.5 or weighted_sell_signal_score > 0.25:
                        # Strong sell signal - liquidate entire position
                        sell_qty = holding_qty
                        logger.info(f"  [STRONG SELL] Selling full position: {sell_qty} shares")
                    else:
                        # Moderate sell signal - partial position
                        # Scale sell quantity based on signal strength (MORE GRANULAR AND CONSISTENT)
                        # Implement more granular exit strategies based on confidence levels
                        if weighted_sell_signal_score >= 0.70:  # Very High confidence
                            sell_factor = min(max(final_sell_score * 3.0, 0.8), 1.0)  # 80-100%
                        elif weighted_sell_signal_score >= 0.50:  # High confidence
                            sell_factor = min(max(final_sell_score * 2.5, 0.6), 0.9)  # 60-90%
                        elif weighted_sell_signal_score >= 0.30:  # Medium confidence
                            sell_factor = min(max(final_sell_score * 2.0, 0.4), 0.7)  # 40-70%
                        elif weighted_sell_signal_score >= 0.15:  # Low confidence
                            sell_factor = min(max(final_sell_score * 1.5, 0.2), 0.5)  # 20-50%
                        else:  # Very low confidence
                            sell_factor = min(max(final_sell_score * 1.0, 0.1), 0.3)  # 10-30%
                        
                        sell_qty = max(1, int(holding_qty * sell_factor))  # Minimum 1 share
                        logger.info(f"  [PARTIAL SELL] Selling {sell_qty} of {holding_qty} shares ({sell_factor:.2f} factor)")
                    
                    # Ensure we don't try to sell more than we own
                    sell_qty = min(sell_qty, holding_qty)
                    logger.info(f"  FINAL SELL QUANTITY: {sell_qty} shares")
                else:
                    logger.info(f"  [NO SELL] No compelling sell conditions met")
                    sell_qty = 0

        # Validate Exposure Limits (REMOVED SECTOR EXPOSURE VALIDATION)
        if buy_qty > 0:
            new_stock_exposure = current_stock_exposure + buy_qty * current_ticker_price

            logger.info(f"=== EXPOSURE VALIDATION for {ticker} ===")
            logger.info(f"  New Stock Exposure: Rs.{new_stock_exposure:.2f} vs Max: Rs.{max_exposure_per_stock:.2f}")
            logger.info(f"  Sector Exposure Validation: DISABLED (removed constraint)")

            if new_stock_exposure > max_exposure_per_stock:
                old_qty = buy_qty
                buy_qty = (max_exposure_per_stock - current_stock_exposure) / current_ticker_price if current_ticker_price > 0 else 0
                buy_qty = max(1, int(buy_qty + 0.5)) if buy_qty > 0.1 else 0  # Round up, minimum 1 share
                logger.info(f"  [LIMIT] STOCK EXPOSURE LIMIT EXCEEDED: Reduced quantity from {old_qty:.2f} to {buy_qty} shares")
            else:
                logger.info(f"  [OK] STOCK EXPOSURE LIMIT OK: Keeping {buy_qty} shares")

        # Backoff Logic
        recent_trades = [t for t in self.portfolio.trade_log if datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S.%f") > datetime.now() - timedelta(hours=12)]
        recent_pnl = sum(
            (t["price"] - self.portfolio.holdings.get(t["asset"], {"avg_price": t["price"]})["avg_price"]) * t["qty"]
            for t in recent_trades if t["action"] == "sell"
        )
        trade_frequency = len(recent_trades)
        backoff = (recent_pnl < -0.015 * total_value or trade_frequency > 8) and not (
            current_ticker_price < stop_loss or unrealized_pnl < -0.06 * total_value
        )

        # DEBUG: Log backoff logic
        logger.info(f"=== BACKOFF LOGIC for {ticker} ===")
        logger.info(f"  Recent Trades (12h): {trade_frequency}")
        logger.info(f"  Recent P&L: Rs.{recent_pnl:.2f}")
        logger.info(f"  P&L Threshold (-1.5%): Rs.{-0.015 * total_value:.2f}")
        logger.info(f"  Trade Frequency Limit: 8")
        logger.info(f"  Unrealized P&L: Rs.{unrealized_pnl:.2f}")
        logger.info(f"  Emergency Override (price < stop_loss OR unrealized < -6%): {'YES' if (current_ticker_price < stop_loss or unrealized_pnl < -0.06 * total_value) else 'NO'}")
        logger.info(f"  BACKOFF ACTIVE: {'YES' if backoff else 'NO'}")

        # Execute Trades
        trade = None
        # PRODUCTION FIX: Updated execution logic with new signal requirements

        # DEBUG: Final BUY execution checks with weighted signals
        logger.info(f"=== FINAL BUY EXECUTION CHECKS for {ticker} ===")
        logger.info(f"  1. buy_qty > 0: {buy_qty} > 0 = {'PASS' if buy_qty > 0 else 'FAIL'}")
        logger.info(f"  2. final_buy_score > confidence_threshold: {final_buy_score:.3f} > {adjusted_buy_threshold:.3f} = {'PASS' if final_buy_score > adjusted_buy_threshold else 'FAIL'}")
        if volatile_market_adjustment:
            logger.info(f"    (Volatile market threshold adjustment: {confidence_threshold:.3f} -> {adjusted_buy_threshold:.3f})")
        logger.info(f"  3. weighted_signal_score >= threshold: {weighted_signal_score:.3f} >= {min_weighted_signal_threshold:.3f} = {'PASS' if weighted_signal_score >= min_weighted_signal_threshold else 'FAIL'}")
        logger.info(f"  4. buy_qty * price <= available_cash: Rs.{buy_qty * current_ticker_price:.2f} <= Rs.{available_cash:.2f} = {'PASS' if buy_qty * current_ticker_price <= available_cash else 'FAIL'}")
        logger.info(f"  5. not backoff: {'PASS' if not backoff else 'FAIL'}")
        logger.info(f"  Legacy buy_signals (reference): {buy_signals}/7")

        # Enhanced buy conditions with stronger validation
        all_conditions_met = (
            buy_qty > 0
            and (
                # Use adjusted threshold for volatile markets
                final_buy_score > adjusted_buy_threshold
                or (weighted_signal_score >= min_weighted_signal_threshold)
            )
            and buy_qty * current_ticker_price <= available_cash
            and not backoff
        )

        logger.info(f"  ALL CONDITIONS MET: {'[EXECUTING BUY]' if all_conditions_met else '[BUY BLOCKED]'}")

        # IMPROVED TRADE EXECUTION LOGIC
        # Execute buy if conditions are met
        if all_conditions_met:
            logger.info(
                f"Executing BUY for {ticker}: {buy_qty:.0f} units at Rs.{current_ticker_price:.2f}, "
                f"Position Value: Rs.{buy_qty * current_price:.2f} ({(buy_qty * current_price / total_value * 100):.2f}% of portfolio), "
                f"Stop-Loss: Rs.{stop_loss:.2f}, Take-Profit: Rs.{take_profit:.2f}, ATR: Rs.{atr:.2f}, Kelly Fraction: {kelly_fraction:.2f}"
            )
            success_result = self.executor.execute_trade("buy", ticker, buy_qty, current_ticker_price, stop_loss, take_profit)

            # Handle the case where trade execution returns detailed result
            if isinstance(success_result, dict):
                success = success_result.get("success", False)
                actual_qty = success_result.get("qty", buy_qty)
            else:
                success = success_result
                actual_qty = buy_qty if success else 0

            trade = {
                "action": "buy",
                "ticker": ticker,
                "qty": actual_qty,
                "price": current_ticker_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": success_result,
                "confidence_score": final_buy_score,
                "signals": buy_signals,
                "reason": "signal_based"
            }
        # Execute sell if we have sell quantity (this is the key fix)
        elif sell_qty > 0 and ticker in self.portfolio.holdings and self.portfolio.holdings[ticker]["qty"] > 0:
            logger.info(
                f"Executing SELL for {ticker}: {sell_qty:.0f} units at Rs.{current_ticker_price:.2f}, "
                f"Position Value: Rs.{sell_qty * current_ticker_price:.2f}, "
                f"Stop-Loss: Rs.{stop_loss:.2f}, Take-Profit: Rs.{take_profit:.2f}, ATR: Rs.{atr:.2f}"
            )
            success_result = self.executor.execute_trade("sell", ticker, sell_qty, current_ticker_price, stop_loss, take_profit)

            # Handle the case where trade execution returns detailed result
            if isinstance(success_result, dict):
                success = success_result.get("success", False)
                actual_qty = success_result.get("qty", sell_qty)
            else:
                success = success_result
                actual_qty = sell_qty if success else 0

            trade = {
                "action": "sell",
                "ticker": ticker,
                "qty": actual_qty,
                "price": current_ticker_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "success": success_result,
                "confidence_score": final_sell_score,
                "signals": sell_signals,
                "reason": "signal_based"
            }
        else:
            # PRODUCTION FIX: Regime-specific HOLD conditions
            if market_regime == "TRENDING":
                # In trending markets, still allow HOLD if conditions are met
                hold_conditions = (
                    abs(final_buy_score - final_sell_score) < 0.06
                    or (support_level * 0.98 < current_ticker_price < resistance_level * 1.02)
                    or (buy_signals < 2 and sell_signals < 2)
                )
                logger.info(f"TRENDING market: Applying standard HOLD conditions for {ticker}")
            elif market_regime == "VOLATILE":
                # In volatile markets, only HOLD if very uncertain
                hold_conditions = (abs(final_buy_score - final_sell_score) < 0.03)
                logger.info(f"VOLATILE market: Reduced HOLD conditions for {ticker}")
            else:  # RANGE_BOUND
                # Traditional HOLD conditions for range-bound markets
                hold_conditions = (
                    abs(final_buy_score - final_sell_score) < 0.06
                    or (support_level * 0.98 < current_ticker_price < resistance_level * 1.02)
                    or (buy_signals < 2 and sell_signals < 2)
                )

            if hold_conditions:
                logger.info(f"HOLD {ticker}: {market_regime} market conditions, "
                            f"Price Rs.{current_ticker_price:.2f} within Support Rs.{support_level:.2f} "
                            f"and Resistance Rs.{resistance_level:.2f}, "
                            f"Buy Score={final_buy_score:.2f}, Sell Score={final_sell_score:.2f}")
                trade = {
                    "action": "hold",
                    "ticker": ticker,
                    "qty": 0,
                    "price": current_ticker_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "success": True,
                    "confidence_score": max(final_buy_score, final_sell_score),
                    "signals": max(buy_signals, sell_signals),
                    "reason": "neutral_conditions"
                }

        if trade is None:
            logger.info(f"No trade executed for {ticker}: Buy Score={final_buy_score:.2f}, "
                        f"Sell Score={final_sell_score:.2f}, Buy Signals={buy_signals}, "
                        f"Sell Signals={sell_signals}, Buy Qty={buy_qty:.0f}, Sell Qty={sell_qty:.0f}, "
                        f"Backoff={backoff}, Cash=Rs.{available_cash:.2f}, ATR=Rs.{atr:.2f}")

        # Log the final trade decision with detailed reasoning
        if trade:
            logger.info(f"=== FINAL TRADE DECISION for {ticker} ===")
            logger.info(f"Action: {trade['action'].upper()}")
            logger.info(f"Quantity: {trade['qty']}")
            logger.info(f"Price: Rs.{trade['price']:.2f}")
            logger.info(f"Confidence Score: {trade['confidence_score']:.3f}")
            logger.info(f"Signals: {trade['signals']}")
            logger.info(f"Reason: {trade['reason']}")
            if trade['action'] in ['buy', 'sell']:
                logger.info(f"Stop Loss: Rs.{trade['stop_loss']:.2f}")
                logger.info(f"Take Profit: Rs.{trade['take_profit']:.2f}")
                logger.info(f"Position Value: Rs.{trade['qty'] * trade['price']:.2f}")

        # Log strategy trigger for paper trading
        if self.portfolio.mode == "paper":
            decision_data = {
                "buy_score": final_buy_score,
                "sell_score": final_sell_score,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "action": trade["action"].upper() if trade else "HOLD",
                "quantity": trade["qty"] if trade else 0,
                "trade_value": (trade["qty"] * trade["price"]) if trade else 0
            }
            self.portfolio.log_strategy_trigger(ticker, analysis, decision_data)

        # Send trade outcome to continuous learning engine
        if self.learning_engine and trade:
            try:
                # Prepare decision data for learning
                decision_data = {
                    "symbol": ticker,
                    "action": trade["action"].upper(),
                    "confidence": trade.get("confidence_score", 0.5),
                    "quantity": trade["qty"],
                    "price": trade["price"],
                    "stop_loss": trade["stop_loss"],
                    "take_profit": trade["take_profit"],
                    "signals": trade.get("signals", 0),
                    "reason": trade.get("reason", "unknown"),
                    "market_context": {
                        "volatility": volatility,
                        "market_regime": market_regime,
                        "support_level": support_level,
                        "resistance_level": resistance_level
                    },
                    "technical_indicators": {
                        "rsi": technical_indicators.get("rsi", 50),
                        "macd": technical_indicators.get("macd", 0),
                        "sma_trend": "UP" if technical_indicators.get("sma_50", 0) > technical_indicators.get("sma_200", 0) else "DOWN"
                    }
                }
                
                # For now, we'll use a placeholder for outcome data
                # In a real implementation, this would be updated when the trade is closed
                outcome_data = {
                    "profit_loss": 0.0,
                    "profit_loss_pct": 0.0,
                    "holding_period": 0,
                    "max_drawdown": 0.0,
                    "max_runup": 0.0
                }
                
                # Send to learning engine (async)
                import asyncio
                try:
                    # Create a new event loop if one doesn't exist
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Schedule the learning task
                    if loop.is_running():
                        # If we're in a running loop, create a task
                        asyncio.create_task(self.learning_engine.learn_from_decision_outcome(decision_data, outcome_data))
                    else:
                        # If we're not in a running loop, run until complete
                        loop.run_until_complete(self.learning_engine.learn_from_decision_outcome(decision_data, outcome_data))
                except Exception as e:
                    logger.debug(f"Non-critical: Could not send to learning engine immediately: {e}")
                    # Fallback to a simpler approach
                    try:
                        # Try to record the decision for later learning
                        if hasattr(self.learning_engine, 'record_decision'):
                            self.learning_engine.record_decision(decision_data)
                    except Exception as e2:
                        logger.debug(f"Could not record decision for learning: {e2}")
                        
            except Exception as e:
                logger.debug(f"Non-critical: Learning engine integration issue: {e}")

        return trade

    def run_analysis(self, ticker):
        """Run analysis for a given ticker and return the result."""
        analysis = self.stock_analyzer.analyze_stock(
            ticker,
            benchmark_tickers=self.config.get("benchmark_tickers", ["^NSEI"]),
            prediction_days=self.config.get("prediction_days", 30),
            training_period=self.config.get("period", "1y"),
            bot_running=self.bot_running
        )
        
        # Validate ML models if ML interface is available
        if self.ml_interface:
            try:
                validation_result = self.ml_interface.validate_all_models()
                if validation_result.get("success", False):
                    logger.info(f"ML models validation completed for {ticker}")
                    # Log any issues found
                    validation_results = validation_result.get("validation_results", {})
                    overall_health = validation_results.get("overall_health", {})
                    if overall_health.get("status") != "healthy":
                        logger.warning(f"ML models health status: {overall_health.get('status', 'unknown')}")
                else:
                    logger.warning(f"ML models validation failed: {validation_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error during ML models validation: {e}")
        
        return analysis

    def run(self):
        """Main bot loop to run analysis, make trades, and generate reports."""
        logger.info("Starting Stock Trading Bot for Indian market...")
        self.bot_running = True

        while self.bot_running:
            try:
                # Check if bot should stop
                if not self.bot_running:
                    logger.info("Bot stop signal received, exiting main loop...")
                    break

                # Check if trading is paused
                if self.chatbot.trading_paused:
                    if self.chatbot.pause_until and datetime.now() >= self.chatbot.pause_until:
                        self.chatbot.trading_paused = False
                        self.chatbot.pause_until = None
                        logger.info("Trading pause expired, resuming trading...")
                    else:
                        logger.info("Trading is paused, waiting...")
                        time.sleep(60)  # Wait 1 minute
                        continue

                # if not self.is_market_open():
                #     logger.info("NSE market is closed, waiting...")
                #     time.sleep(300)  # Wait 5 minutes
                #     continue

                logger.info("Logging portfolio metrics at start of trading cycle...")
                self.tracker.log_metrics()

                for ticker in self.config["tickers"]:
                    # Check if bot should stop before processing each ticker
                    if not self.bot_running:
                        logger.info("Bot stop signal received, stopping ticker processing...")
                        break

                    logger.info(f"Processing {ticker}...")
                    analysis = self.run_analysis(ticker)
                    if analysis.get("success"):
                        save_result = self.stock_analyzer.save_analysis_to_files(analysis)
                        if save_result.get("success"):
                            logger.info(f"Saved analysis files: {save_result}")
                        else:
                            logger.warning(f"Failed to save analysis: {save_result.get('message')}")
                        trade = self.make_trading_decision(analysis)
                        if trade and trade["success"]:
                            logger.info(f"Trade executed: {trade}")
                    else:
                        logger.warning(f"Analysis failed for {ticker}: {analysis.get('message')}")

                # Check if bot should stop before generating report
                if not self.bot_running:
                    logger.info("Bot stop signal received, skipping report generation...")
                    break

                report = self.reporter.generate_report()
                logger.info(f"Daily Report: {report}")

                logger.info("Logging portfolio metrics at end of trading cycle...")
                self.tracker.log_metrics()

                # Generate P&L summary for paper trading every cycle
                if self.portfolio.mode == "paper":
                    self.portfolio.generate_paper_pnl_summary()

                time.sleep(self.config.get("sleep_interval", 300))
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)

        logger.info("Stock Trading Bot stopped successfully")

def main():
    config = {
        "tickers": [],  # Empty by default - users can add tickers manually
        "starting_balance": 10000,  # Default starting balance in INR
        "current_portfolio_value": 10000,  # Initial portfolio value
        "current_pnl": 0,  # Initial PnL
        "mode": "paper",
        "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
        "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
        "period": "3y",
        "prediction_days": 30,
        "benchmark_tickers": ["^NSEI"],
        "sleep_interval": 300  # 5 minutes
    }

    bot = StockTradingBot(config)
    bot.run()
def signal_handler(_sig, _frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Bot shutdown signal received. Shutting down gracefully...")
    print("\n[BOT] Bot shut down successfully!")
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Indian Stock Trading Bot')
    parser.add_argument('--mode', choices=['live', 'paper'],
                       default=os.getenv('TRADING_MODE', 'paper'),
                       help='Trading mode: live or paper (default: paper)')
    return parser.parse_args()

def validate_environment_variables(mode: str) -> bool:
    """Validate required environment variables based on mode"""
    required_vars = {
        "common": ["NEWSAPI_KEY", "GNEWS_API_KEY"],
        "live": ["DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN"],
        "fyers": ["FYERS_APP_ID", "FYERS_ACCESS_TOKEN"]
    }

    missing_vars = []

    # Check common variables
    for var in required_vars["common"]:
        if not os.getenv(var):
            missing_vars.append(var)

    # Check mode-specific variables
    if mode == "live":
        for var in required_vars["live"]:
            if not os.getenv(var):
                missing_vars.append(var)

    # Check optional Fyers variables
    fyers_vars = required_vars["fyers"]
    fyers_available = all(os.getenv(var) for var in fyers_vars)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all required variables are set")
        return False

    if not fyers_available:
        logger.warning("Fyers API credentials not found - will use Yahoo Finance fallback")

    logger.info("Environment variable validation passed")
    return True

def main_with_mode():
    """Main function with mode selection and enhanced configuration."""
    try:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)

        # Parse command line arguments
        args = parse_arguments()

        # Load environment variables
        load_dotenv()

        # Validate environment variables
        if not validate_environment_variables(args.mode):
            logger.error("Environment validation failed. Exiting...")
            return

        # Enhanced configuration with risk management and configurable paths
        config = {
            "tickers": [],  # Empty by default - users can add tickers manually
            "starting_balance": 10000,  # Rs.10 thousand
            "current_portfolio_value": 10000,
            "current_pnl": 0,
            "mode": os.getenv("MODE", "paper"),
            "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
            "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
            "period": "3y",
            "prediction_days": 30,
            "benchmark_tickers": ["^NSEI"],
            "sleep_interval": 300,  # 5 minutes
            # Risk management settings from .env
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
            "max_capital_per_trade": float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.25")),
            "max_trade_limit": int(os.getenv("MAX_TRADE_LIMIT", "150")),
            # Configurable paths
            "data_dir": os.getenv("DATA_DIR", "data"),
            "log_dir": os.getenv("LOG_DIR", "logs"),
            "reports_dir": os.getenv("REPORTS_DIR", "reports"),
            "analysis_dir": os.getenv("ANALYSIS_DIR", "stock_analysis")
        }

        # Display mode information
        mode_display = " LIVE TRADING" if args.mode == "live" else " PAPER TRADING"
        logger.info(f"Starting Indian Stock Trading Bot in {mode_display} mode")

        # Display startup banner - REMOVED as requested by user

        if args.mode == "live":
            if not config["dhan_client_id"] or not config["dhan_access_token"]:
                logger.error("Dhan API credentials not found in .env file. Cannot run in live mode.")
                print(" Error: Dhan API credentials required for live trading!")
                return
            print("WARNING: Live trading mode enabled. Real money will be used!")
            confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirmation != "CONFIRM":
                print(" Live trading cancelled.")
                return

        # Initialize and run bot
        bot = StockTradingBot(config)
        bot.run()

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user.")
        print("\n Bot shut down successfully!")
    except Exception as e:
        logger.error(f"Critical error in main function: {e}")
        print(f" Critical error: {e}")


if __name__ == "__main__":
    
    main_with_mode()