import json
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
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
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pickle
from bs4 import BeautifulSoup
from collections import defaultdict
from requests.exceptions import HTTPError
from dotenv import load_dotenv
from dhanhq import dhanhq
import argparse
import sys
import signal
# threading removed - not used in current implementation
import queue
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing import Optional, List


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


class DynamicTickerMapper:
    """Dynamic ticker-to-company name mapping for Indian stock market"""

    def __init__(self, cache_file="data/indian_ticker_mapping.pkl"):
        self.cache_file = cache_file
        self.ticker_mapping = {}
        self.last_updated = None
        self.update_frequency = timedelta(days=7)  # Update weekly

        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Load cached data if available
        self.load_cache()

    def get_ticker_mapping(self, force_update=False):
        """Get ticker mapping with automatic updates"""
        if self.should_update() or force_update:
            logger.info("Updating ticker mapping from NSE/BSE APIs...")
            self.update_mapping()
            self.save_cache()

        return self.ticker_mapping

    def should_update(self):
        """Check if mapping needs update"""
        if not self.last_updated:
            return True
        return datetime.now() - self.last_updated > self.update_frequency

    def fetch_nse_stock_list(self):
        """Fetch all NSE stocks with company names from official NSE API"""
        try:
            stock_mapping = {}

            # NSE API endpoints for different indices
            nse_endpoints = [
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20100",
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20100",
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MICROCAP%20250"
            ]

            # Enhanced headers to mimic real browser behavior
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': 'https://www.nseindia.com/',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }

            session = requests.Session()
            session.headers.update(headers)

            # Enhanced session establishment with proper cookie handling
            try:
                logger.info("Establishing NSE session...")
                # Visit main page to get cookies
                main_response = session.get("https://www.nseindia.com", timeout=15)
                logger.info(f"Main page response: {main_response.status_code}")

                # Visit market data page to establish proper session
                market_response = session.get("https://www.nseindia.com/market-data", timeout=15)
                logger.info(f"Market data page response: {market_response.status_code}")

                time.sleep(2)  # Allow session to stabilize
            except Exception as e:
                logger.warning(f"Session establishment failed: {e}")

            for endpoint in nse_endpoints:
                try:
                    logger.info(f"Fetching data from: {endpoint}")
                    response = session.get(endpoint, timeout=20)

                    logger.info(f"Response status: {response.status_code}")
                    logger.info(f"Response headers: {dict(response.headers)}")

                    if response.status_code == 200:
                        # Check if response is actually JSON
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' not in content_type:
                            logger.warning(f"Unexpected content type: {content_type}")
                            logger.warning(f"Response text preview: {response.text[:200]}")
                            continue

                        try:
                            data = response.json()
                        except ValueError as e:
                            logger.warning(f"JSON parsing failed: {e}")
                            logger.warning(f"Response text preview: {response.text[:200]}")
                            continue

                        stocks_data = data.get('data', [])
                        if not stocks_data:
                            logger.warning(f"No data found in response for {endpoint}")
                            continue

                        for stock in stocks_data:
                            symbol = stock.get('symbol')
                            company_name = stock.get('companyName', '')

                            if symbol and company_name:
                                ticker = f"{symbol}.NS"
                                variations = self.create_name_variations(company_name)
                                stock_mapping[ticker] = variations

                        logger.info(f"Fetched {len(stocks_data)} stocks from this endpoint")
                    elif response.status_code == 401:
                        logger.warning(f"Authentication required for {endpoint}. Trying alternative approach...")
                        # Try with additional headers for authentication
                        auth_headers = headers.copy()
                        auth_headers.update({
                            'X-Requested-With': 'XMLHttpRequest',
                            'Origin': 'https://www.nseindia.com'
                        })

                        auth_response = session.get(endpoint, headers=auth_headers, timeout=20)
                        if auth_response.status_code == 200:
                            try:
                                data = auth_response.json()
                                stocks_data = data.get('data', [])
                                for stock in stocks_data:
                                    symbol = stock.get('symbol')
                                    company_name = stock.get('companyName', '')
                                    if symbol and company_name:
                                        ticker = f"{symbol}.NS"
                                        variations = self.create_name_variations(company_name)
                                        stock_mapping[ticker] = variations
                                logger.info(f"Successfully fetched {len(stocks_data)} stocks with auth headers")
                            except ValueError:
                                logger.warning(f"Auth attempt also failed for {endpoint}")
                        else:
                            logger.warning(f"Auth attempt failed with status: {auth_response.status_code}")
                    else:
                        logger.warning(f"Failed to fetch from {endpoint}: Status {response.status_code}")
                        logger.warning(f"Response text preview: {response.text[:200]}")

                except Exception as e:
                    logger.warning(f"Error fetching from {endpoint}: {e}")
                    continue

                time.sleep(3)  # Increased delay to be more respectful

            logger.info(f"Total stocks fetched from NSE: {len(stock_mapping)}")
            return stock_mapping

        except Exception as e:
            logger.error(f"Error in fetch_nse_stock_list: {e}")
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
        """Update mapping from multiple sources with enhanced error reporting"""
        all_mappings = {}

        # Test API connectivity first
        connectivity = self.test_api_connectivity()
        logger.info(f"API connectivity status: NSE Main={connectivity['nse_main']}, "
                   f"NSE Market={connectivity['nse_market_data']}, BSE={connectivity['bse_api']}")

        # Method 1: Try NSE API
        try:
            nse_data = self.fetch_nse_stock_list()
            all_mappings.update(nse_data)
            logger.info(f"✅ Successfully fetched {len(nse_data)} stocks from NSE API")
        except Exception as e:
            logger.warning(f"❌ NSE API failed: {e}")

        # Method 2: Try BSE API
        try:
            bse_data = self.fetch_bse_stock_list()
            # Merge with existing data
            for ticker, names in bse_data.items():
                if ticker in all_mappings:
                    # Combine name variations
                    all_mappings[ticker].extend(names)
                    all_mappings[ticker] = list(set(all_mappings[ticker]))
                else:
                    all_mappings[ticker] = names
            logger.info(f"✅ Successfully added {len(bse_data)} stocks from BSE API")
        except Exception as e:
            logger.warning(f"❌ BSE API failed: {e}")

        # Method 3: Fallback to essential stocks if we don't have enough data
        if len(all_mappings) < 50:
            logger.warning(f"⚠️  Only {len(all_mappings)} stocks fetched from APIs. Using fallback hardcoded mapping.")
            fallback_mapping = self.get_fallback_mapping()
            for ticker, names in fallback_mapping.items():
                if ticker not in all_mappings:
                    all_mappings[ticker] = names
            logger.info(f"✅ Added {len(fallback_mapping)} fallback stocks")
        else:
            logger.info(f"✅ Sufficient stocks ({len(all_mappings)}) fetched from APIs")

        self.ticker_mapping = all_mappings
        self.last_updated = datetime.now()

        logger.info(f"🎯 Final mapping contains {len(self.ticker_mapping)} stocks")

        # Log sample of stocks for verification
        sample_stocks = list(self.ticker_mapping.keys())[:5]
        logger.info(f"📊 Sample stocks: {sample_stocks}")

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
        self.log_file = "../data/chat_interactions.json"
        os.makedirs("../data", exist_ok=True)
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
- Portfolio Value: ₹{self.trading_bot.portfolio.get_metrics()['total_value']:,.2f}
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
            return "✅ Trading bot is active and monitoring markets. Check logs for detailed activity."
        except Exception as e:
            return f"❌ Error starting bot: {str(e)}"

    def set_risk(self, risk_level: str) -> str:
        """Set risk level and update stop-loss."""
        try:
            risk_mapping = {
                "LOW": 0.03,    # 3% stop-loss
                "MEDIUM": 0.05, # 5% stop-loss
                "HIGH": 0.08    # 8% stop-loss
            }

            if risk_level not in risk_mapping:
                return f"❌ Invalid risk level. Use: LOW, MEDIUM, or HIGH"

            new_stop_loss = risk_mapping[risk_level]
            self.trading_bot.config["stop_loss_pct"] = new_stop_loss
            self.trading_bot.executor.stop_loss_pct = new_stop_loss

            return f"✅ Risk level set to {risk_level}. Stop-loss updated to {new_stop_loss*100}%"
        except Exception as e:
            return f"❌ Error setting risk level: {str(e)}"

    def get_pnl(self) -> str:
        """Get current portfolio P&L metrics."""
        try:
            metrics = self.trading_bot.portfolio.get_metrics()
            starting_balance = self.trading_bot.portfolio.starting_balance
            total_return = metrics['total_value'] - starting_balance
            return_pct = (total_return / starting_balance) * 100

            pnl_report = f"""
📊 **Portfolio Metrics**
💰 Total Value: ₹{metrics['total_value']:,.2f}
💵 Cash: ₹{metrics['cash']:,.2f}
📈 Holdings Value: ₹{metrics['total_exposure']:,.2f}
🎯 Total Return: ₹{total_return:,.2f} ({return_pct:+.2f}%)
✅ Realized P&L: ₹{metrics['realized_pnl']:,.2f}
📊 Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}
🏢 Active Positions: {len(self.trading_bot.portfolio.holdings)}
            """
            return pnl_report.strip()
        except Exception as e:
            return f"❌ Error getting P&L: {str(e)}"

    def why_trade(self, ticker: str) -> str:
        """Explain why a particular ticker should be traded."""
        try:
            if not ticker:
                return "❌ Please specify a ticker. Example: /why_trade RELIANCE.NS"

            # Get analysis for the ticker
            analysis = self.trading_bot.run_analysis(ticker)
            if not analysis.get("success"):
                return f"❌ Could not analyze {ticker}: {analysis.get('message', 'Unknown error')}"

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
                recommendation = "🟢 BUY"
                reason = "Technical indicators suggest upward momentum"
            elif sell_signals > buy_signals:
                recommendation = "🔴 SELL"
                reason = "Technical indicators suggest downward pressure"
            else:
                recommendation = "🟡 HOLD"
                reason = "Mixed signals, wait for clearer direction"

            analysis_text = f"""
🎯 **Trade Analysis for {ticker}**
💰 Current Price: ₹{current_price:.2f}
📊 Recommendation: {recommendation}

**Technical Signals:**
• RSI ({rsi:.1f}): {rsi_signal}
• Moving Average: {ma_signal}
• MACD: {macd_signal}

**Reasoning:** {reason}

⚠️ This is for educational purposes only. Always do your own research before trading.
"""
            return analysis_text.strip()

        except Exception as e:
            return f"❌ Error analyzing trade: {str(e)}"

    def list_positions(self) -> str:
        """List all open positions."""
        try:
            holdings = self.trading_bot.portfolio.holdings
            if not holdings:
                return "📭 No open positions currently."

            positions_text = "📊 **Current Positions:**\n"
            for ticker, data in holdings.items():
                current_value = data['qty'] * data['avg_price']
                positions_text += f"• {ticker}: {data['qty']} shares @ ₹{data['avg_price']:.2f} (₹{current_value:,.2f})\n"

            return positions_text.strip()
        except Exception as e:
            return f"❌ Error listing positions: {str(e)}"

    def set_ticker(self, ticker: str, action: str) -> str:
        """Add or remove ticker from watchlist."""
        try:
            if not ticker:
                return "❌ Please specify a ticker. Example: /set_ticker RELIANCE.NS ADD"

            current_tickers = self.trading_bot.config["tickers"]

            if action == "ADD":
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    return f"✅ Added {ticker} to watchlist. Total tickers: {len(current_tickers)}"
                else:
                    return f"ℹ️ {ticker} is already in watchlist."

            elif action == "REMOVE":
                if ticker in current_tickers:
                    current_tickers.remove(ticker)
                    return f"✅ Removed {ticker} from watchlist. Total tickers: {len(current_tickers)}"
                else:
                    return f"ℹ️ {ticker} is not in watchlist."

            else:
                return "❌ Invalid action. Use ADD or REMOVE."

        except Exception as e:
            return f"❌ Error managing ticker: {str(e)}"

    def get_signals(self, ticker: str) -> str:
        """Get current trading signals for a ticker."""
        try:
            if not ticker:
                return "❌ Please specify a ticker. Example: /get_signals TCS.NS"

            analysis = self.trading_bot.run_analysis(ticker)
            if not analysis.get("success"):
                return f"❌ Could not get signals for {ticker}: {analysis.get('message', 'Unknown error')}"

            technical = analysis["technical_indicators"]
            current_price = analysis["stock_data"]["current_price"]["INR"]

            signals_text = f"""
🎯 **Trading Signals for {ticker}**
💰 Current Price: ₹{current_price:.2f}

📊 **Technical Signals:**
• RSI: {technical['rsi']:.2f} {'🔴 Overbought' if technical['rsi'] > 70 else '🟢 Oversold' if technical['rsi'] < 30 else '🟡 Neutral'}
• MACD: {technical['macd']:.4f} {'🟢 Bullish' if technical['macd'] > 0 else '🔴 Bearish'}
• SMA Trend: {'🟢 Bullish' if technical['sma_50'] > technical['sma_200'] else '🔴 Bearish'}
• Bollinger Bands: {'🔴 Overbought' if current_price > technical['bb_upper'] else '🟢 Oversold' if current_price < technical['bb_lower'] else '🟡 Normal'}

📈 **Overall Signal:** {analysis['recommendation']}
            """
            return signals_text.strip()
        except Exception as e:
            return f"❌ Error getting signals: {str(e)}"

    def pause_trading(self, minutes: int) -> str:
        """Pause trading for specified minutes."""
        try:
            self.trading_paused = True
            self.pause_until = datetime.now() + timedelta(minutes=minutes)
            return f"⏸️ Trading paused for {minutes} minutes until {self.pause_until.strftime('%H:%M:%S')}"
        except Exception as e:
            return f"❌ Error pausing trading: {str(e)}"

    def resume_trading(self) -> str:
        """Resume trading."""
        try:
            self.trading_paused = False
            self.pause_until = None
            return "▶️ Trading resumed successfully."
        except Exception as e:
            return f"❌ Error resuming trading: {str(e)}"

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
                except:
                    continue

            performance_text = f"""
📈 **Performance Report ({period})**
💰 Current Value: ₹{metrics['total_value']:,.2f}
🎯 Total Return: ₹{total_return:,.2f} ({return_pct:+.2f}%)
📊 Realized P&L: ₹{metrics['realized_pnl']:,.2f}
📈 Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}
🔄 Recent Trades: {len(recent_trades)}
🏢 Active Positions: {len(self.trading_bot.portfolio.holdings)}
            """
            return performance_text.strip()
        except Exception as e:
            return f"❌ Error getting performance: {str(e)}"

    def set_allocation(self, percentage: float) -> str:
        """Set maximum capital allocation per trade."""
        try:
            if percentage <= 0 or percentage > 100:
                return "❌ Allocation must be between 0 and 100%"

            allocation_decimal = percentage / 100
            self.trading_bot.config["max_capital_per_trade"] = allocation_decimal
            self.trading_bot.executor.max_capital_per_trade = allocation_decimal

            return f"✅ Maximum allocation per trade set to {percentage}%"
        except Exception as e:
            return f"❌ Error setting allocation: {str(e)}"

class DataFeed:
    def __init__(self, tickers):
        self.tickers = tickers

    def get_live_prices(self):
        """Fetch live prices for specified tickers using yfinance."""
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="1d", interval="1m")
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
            self.api = dhanhq(
                client_id=config["dhan_client_id"],
                access_token=config["dhan_access_token"]
            )
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
        # Ensure parent data directory exists (don't create local data folder)
        os.makedirs("../data", exist_ok=True)

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

        # Initialize paper trading specific logs directory in parent folder
        if self.mode == "paper":
            os.makedirs("../logs", exist_ok=True)
            self.paper_trade_log = f"../logs/paper_trade_{datetime.now().strftime('%Y%m%d')}.txt"
            # Initialize paper trade log file with header
            if not os.path.exists(self.paper_trade_log):
                with open(self.paper_trade_log, "w", encoding='utf-8') as f:
                    f.write(f"=== PAPER TRADING SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"Starting Balance: Rs.{self.starting_balance:,.2f}\n")
                    f.write("="*80 + "\n\n")

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
        """Fetch security ID for a ticker from Dhan API or a mapping."""
        try:
            # Convert ticker to Dhan format (remove .NS, .BO suffixes)
            dhan_symbol = ticker.split('.')[0] if ticker.endswith(('.NS', '.BO')) else ticker

            # Use fetch_security_list instead of get_instruments
            instruments = self.api.fetch_security_list("compact")
            for inst in instruments:
                if inst.get("trading_symbol") == dhan_symbol and inst.get("exchange_segment") == "NSE":
                    return inst.get("security_id")
            logger.error(f"Security ID not found for {ticker} (converted to {dhan_symbol})")
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
        """Return portfolio metrics including PnL and exposure."""
        current_prices = self.get_current_prices()
        metrics = {
            "cash": self.cash,
            "holdings": self.holdings,
            "total_value": self.get_value(current_prices),
            "current_portfolio_value": self.config.get("current_portfolio_value", 0),
            "current_pnl": self.config.get("current_pnl", 0),
            "realized_pnl": sum(
                (t["price"] - self.holdings.get(t["asset"], {}).get("avg_price", t["price"])) * t["qty"]
                for t in self.trade_log if t["action"] == "sell"
            ),
            "unrealized_pnl": sum(
                (current_prices.get(asset, {}).get("price", 0) - data["avg_price"]) * data["qty"]
                for asset, data in self.holdings.items()
            ),
            "total_exposure": sum(
                data["qty"] * current_prices.get(asset, {}).get("price", 0)
                for asset, data in self.holdings.items()
            )
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
            log_entry += f"Action: {action} {qty} shares at ₹{price:.2f}\n"
            log_entry += f"Trade Value: ₹{qty * price:,.2f}\n"

            if action == "BUY":
                log_entry += f"Entry Signal: Price ₹{price:.2f} identified as favorable entry point\n"
                log_entry += f"Position Size: {qty} shares ({(qty * price / metrics['total_value'] * 100):.1f}% of portfolio)\n"
            else:
                if asset in self.holdings:
                    avg_price = self.holdings[asset]["avg_price"]
                    pnl = (price - avg_price) * qty
                    pnl_pct = ((price / avg_price) - 1) * 100
                    log_entry += f"Exit Signal: Price ₹{price:.2f} vs Entry ₹{avg_price:.2f}\n"
                    log_entry += f"Trade P&L: ₹{pnl:,.2f} ({pnl_pct:+.2f}%)\n"

            log_entry += f"Portfolio Cash: ₹{metrics['cash']:,.2f}\n"
            log_entry += f"Total Portfolio Value: ₹{metrics['total_value']:,.2f}\n"
            log_entry += f"Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}\n"
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
                log_entry += f"  • RSI: {tech.get('rsi', 'N/A'):.2f} {'(Oversold)' if tech.get('rsi', 50) < 30 else '(Overbought)' if tech.get('rsi', 50) > 70 else '(Neutral)'}\n"
                log_entry += f"  • MACD: {tech.get('macd', 'N/A'):.4f}\n"
                log_entry += f"  • SMA 50: ₹{tech.get('sma_50', 'N/A'):.2f}\n"
                log_entry += f"  • SMA 200: ₹{tech.get('sma_200', 'N/A'):.2f}\n"
                log_entry += f"  • Trend: {'Bullish' if tech.get('sma_50', 0) > tech.get('sma_200', 0) else 'Bearish'}\n"

            # Sentiment Analysis Summary
            if 'sentiment_analysis' in analysis:
                sentiment = analysis['sentiment_analysis']['aggregated']
                total = sum(sentiment.values())
                if total > 0:
                    pos_pct = sentiment['positive'] / total * 100
                    neg_pct = sentiment['negative'] / total * 100
                    log_entry += f"Market Sentiment:\n"
                    log_entry += f"  • Positive: {pos_pct:.1f}% ({sentiment['positive']} articles)\n"
                    log_entry += f"  • Negative: {neg_pct:.1f}% ({sentiment['negative']} articles)\n"
                    log_entry += f"  • Overall: {'Bullish' if pos_pct > neg_pct else 'Bearish'}\n"

            # ML/RL Predictions
            if 'ml_analysis' in analysis:
                ml = analysis['ml_analysis']
                current_price = analysis.get('stock_data', {}).get('current_price', {}).get('INR', 0)
                predicted_price = ml.get('predicted_price', current_price)
                price_change = ((predicted_price / current_price) - 1) * 100 if current_price > 0 else 0

                log_entry += f"ML/RL Analysis:\n"
                log_entry += f"  • Current Price: ₹{current_price:.2f}\n"
                log_entry += f"  • Predicted Price: ₹{predicted_price:.2f}\n"
                log_entry += f"  • Expected Change: {price_change:+.2f}%\n"
                log_entry += f"  • RL Recommendation: {ml.get('rl_metrics', {}).get('recommendation', 'HOLD')}\n"

            # Decision Summary
            log_entry += f"Decision Factors:\n"
            log_entry += f"  • Buy Score: {decision_data.get('buy_score', 0):.3f}\n"
            log_entry += f"  • Sell Score: {decision_data.get('sell_score', 0):.3f}\n"
            log_entry += f"  • Buy Signals: {decision_data.get('buy_signals', 0)}/4\n"
            log_entry += f"  • Sell Signals: {decision_data.get('sell_signals', 0)}/4\n"
            log_entry += f"  • Final Decision: {decision_data.get('action', 'HOLD')}\n"

            if decision_data.get('action') in ['BUY', 'SELL']:
                log_entry += f"  • Quantity: {decision_data.get('quantity', 0)} shares\n"
                log_entry += f"  • Trade Value: ₹{decision_data.get('trade_value', 0):,.2f}\n"

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
            summary += f"  • Starting Balance: ₹{self.starting_balance:,.2f}\n"
            summary += f"  • Current Cash: ₹{metrics['cash']:,.2f}\n"
            summary += f"  • Holdings Value: ₹{metrics['total_exposure']:,.2f}\n"
            summary += f"  • Total Portfolio Value: ₹{metrics['total_value']:,.2f}\n"
            summary += f"  • Total Return: ₹{total_return:,.2f} ({total_return_pct:+.2f}%)\n"
            summary += f"  • Realized P&L: ₹{realized_pnl:,.2f}\n"
            summary += f"  • Unrealized P&L: ₹{metrics['unrealized_pnl']:,.2f}\n"

            summary += f"\nTrading Statistics:\n"
            summary += f"  • Total Trades: {len(self.trade_log)}\n"
            summary += f"  • Buy Orders: {len(buy_trades)}\n"
            summary += f"  • Sell Orders: {len(sell_trades)}\n"
            summary += f"  • Win Rate: {win_rate:.1f}%\n"
            summary += f"  • Active Positions: {len(self.holdings)}\n"

            if self.holdings:
                summary += f"\nCurrent Holdings:\n"
                for asset, data in self.holdings.items():
                    current_value = data['qty'] * data['avg_price']  # Simplified - would need current price
                    summary += f"  • {asset}: {data['qty']} shares @ ₹{data['avg_price']:.2f} (₹{current_value:,.2f})\n"

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

            import yfinance as yf
            total_unrealized = 0

            for ticker, data in self.holdings.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
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
                stock = yf.Ticker(asset)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    prices[asset] = {"price": current_price, "volume": volume}
                    logger.info(f"Fetched price for {asset}: Rs.{current_price:.2f}")
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
        self.max_capital_per_trade = float(config.get("max_capital_per_trade", 0.25))
        self.max_trade_limit = int(config.get("max_trade_limit", 10))

    def execute_trade(self, action, ticker, qty, price, stop_loss=None, take_profit=None):
        try:
            # Apply risk management rules
            portfolio_value = self.portfolio.get_value({ticker: {"price": price}})
            max_trade_value = portfolio_value * self.max_capital_per_trade
            trade_value = qty * price

            # Check if trade exceeds maximum capital per trade
            if trade_value > max_trade_value:
                adjusted_qty = int(max_trade_value / price)
                if adjusted_qty < qty:
                    logger.warning(f"Reducing trade size from {qty} to {adjusted_qty} due to capital limits")
                    qty = adjusted_qty

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
            if self.mode == "live" and self.dhanhq:
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
                logger.info(f"Live trade executed: {action} {qty} units of {ticker} at Rs.{price}")
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

    def get_security_id(self, symbol, exchange="NSE"):
        """Fetch security ID for a ticker from Dhan API."""
        try:
            # Convert ticker to Dhan format
            dhan_symbol = self.convert_ticker_to_dhan_format(symbol)

            # Use fetch_security_list instead of get_instruments
            instruments = self.dhanhq.fetch_security_list("compact")
            for inst in instruments:
                if inst.get("trading_symbol") == dhan_symbol and inst.get("exchange_segment") == exchange:
                    return inst.get("security_id")
            logger.error(f"Security ID not found for {symbol} (converted to {dhan_symbol})")
            return None
        except Exception as e:
            logger.error(f"Error fetching security ID for {symbol}: {str(e)}")
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

        # Initialize dynamic ticker mapper
        self.ticker_mapper = DynamicTickerMapper()
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

    def build_multi_level_search_queries(self, ticker):
        """Build multiple search query levels for fallback strategy"""
        company_names = self.get_company_names(ticker)
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

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

        # Level 2: Company names with financial context
        if company_names:
            contextual_queries = []
            for name in company_names[:2]:
                contextual_queries.extend([
                    f'"{name}" stock',
                    f'"{name}" shares',
                    f'"{name}" earnings',
                    f'"{name}" financial'
                ])
            contextual_query = " OR ".join(contextual_queries[:4])
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

        # Level 5: Basic ticker (fallback)
        search_levels.append({
            "level": 5,
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

    def get_indian_news_sources(self):
        """Get dynamic list of Indian financial news sources"""
        return {
            'primary': [
                'economictimes.indiatimes.com',
                'moneycontrol.com',
                'business-standard.com',
                'livemint.com'
            ],
            'secondary': [
                'financialexpress.com',
                'zeebiz.com',
                'cnbctv18.com',
                'businesstoday.in'
            ],
            'regional': [
                'thehindubusinessline.com',
                'deccanherald.com',
                'indianexpress.com',
                'timesofindia.indiatimes.com'
            ],
            'specialized': [
                'capitalmarket.com',
                'equitymaster.com',
                'investmentguruindia.com',
                'stockguru.com'
            ]
        }

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
        """Enhanced NewsAPI sentiment with multi-level search and Indian sources"""
        def search_with_query(query):
            try:
                # Get Indian news sources dynamically
                indian_sources = self.get_indian_news_sources()
                all_sources = indian_sources['primary'] + indian_sources['secondary']
                sources_param = ",".join(all_sources[:10])  # Limit to avoid URL length issues

                # Build URL with Indian sources preference
                url = f"https://newsapi.org/v2/everything?q={quote(query)}&sources={sources_param}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"

                # Fallback to general search if sources fail
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                except:
                    # Fallback without specific sources
                    url = f"https://newsapi.org/v2/everything?q={quote(query)}&apiKey={self.NEWSAPI_KEY}&language=en&sortBy=publishedAt"
                    response = requests.get(url, timeout=10)
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
                            if sentiment["compound"] > 0.1:
                                sentiments["positive"] += 1
                            elif sentiment["compound"] < -0.1:
                                sentiments["negative"] += 1
                            else:
                                sentiments["neutral"] += 1

                return sentiments

            except Exception as e:
                logger.warning(f"NewsAPI search failed for query '{query[:50]}...': {e}")
                return {"positive": 0, "negative": 0, "neutral": 0}

        try:
            # Use multi-level search strategy
            logger.info(f"Starting multi-level NewsAPI search for {ticker}")
            result = self.multi_level_search(ticker, search_with_query, max_levels=3)

            # If multi-level search fails, try basic search as final fallback
            if not self.has_meaningful_results(result):
                logger.info(f"Multi-level search failed, trying basic search for {ticker}")
                basic_query = self.build_search_query(ticker)
                result = search_with_query(basic_query)

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
        """Enhanced GNews sentiment with multi-level search and Indian focus"""
        def search_with_query(query):
            try:
                encoded_query = quote(query)
                url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&country=in&token={self.GNEWS_API_KEY}"
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
                        if sentiment["compound"] > 0.1:
                            sentiments["positive"] += 1
                        elif sentiment["compound"] < -0.1:
                            sentiments["negative"] += 1
                        else:
                            sentiments["neutral"] += 1

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
                    if sentiment["compound"] > 0.1:
                        sentiments["positive"] += 1
                    elif sentiment["compound"] < -0.1:
                        sentiments["negative"] += 1
                    else:
                        sentiments["neutral"] += 1
            return sentiments
        except Exception as e:
            logger.error(f"Error fetching Google News sentiment: {e}")
            return {"positive": 0, "negative": 0, "neutral": 0}

    def fetch_combined_sentiment(self, ticker):
        try:
            newsapi_sentiment = self.newsapi_sentiment(ticker)
            gnews_sentiment = self.gnews_sentiment(ticker)
            reddit_sentiment = self.reddit_sentiment(ticker)
            google_sentiment = self.google_news_sentiment(ticker)

            aggregated = {
                "positive": (newsapi_sentiment["positive"] + gnews_sentiment["positive"] +
                            reddit_sentiment["positive"] + google_sentiment["positive"]),
                "negative": (newsapi_sentiment["negative"] + gnews_sentiment["negative"] +
                            reddit_sentiment["negative"] + google_sentiment["negative"]),
                "neutral": (newsapi_sentiment["neutral"] + gnews_sentiment["neutral"] +
                           reddit_sentiment["neutral"] + google_sentiment["neutral"])
            }

            return {
                "newsapi": newsapi_sentiment,
                "gnews": gnews_sentiment,
                "reddit": reddit_sentiment,
                "google_news": google_sentiment,
                "aggregated": aggregated
            }
        except Exception as e:
            logger.error(f"Error fetching combined sentiment: {e}")
            return {
                "newsapi": {"positive": 0, "negative": 0, "neutral": 0},
                "gnews": {"positive": 0, "negative": 0, "neutral": 0},
                "reddit": {"positive": 0, "negative": 0, "neutral": 0},
                "google_news": {"positive": 0, "negative": 0, "neutral": 0},
                "aggregated": {"positive": 0, "negative": 0, "neutral": 0}
            }

    def _generate_detailed_recommendation(self, stock_data, recommendation, buy_score, sell_score,
                                        price_to_sma200, trend_direction, sentiment_score,
                                        volatility, sharpe_ratio):
        explanation = f"Recommendation for {stock_data['name']} ({stock_data['symbol']}): {recommendation}\n"
        explanation += f"Current Price: ₹{stock_data['current_price']['INR']:.2f}\n\n"
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
                benchmark = yf.Ticker(benchmark_ticker)
                benchmark_history = benchmark.history(period="2y")
                benchmark_returns = benchmark_history["Close"].pct_change().dropna()

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
                                       num_episodes=100, adversarial_freq=0.2, max_event_magnitude=0.1, bot_running=True):
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
            
            class AdversarialQLearningAgent:
                def __init__(self, state_size, action_size):
                    self.state_size = state_size
                    self.action_size = action_size
                    self.q_table = {}
                    self.alpha = 0.1
                    self.gamma = 0.95
                    self.epsilon = 0.1
                    self.event_threshold = 0.05
                    
                def discretize_state(self, state):
                    rounded_state = np.round(state[:-1], 2)
                    event = 1 if state[-1] > self.event_threshold else 0
                    return tuple(np.append(rounded_state, event))
                
                def get_action(self, state):
                    state_key = self.discretize_state(state)
                    if np.random.random() < self.epsilon:
                        return np.random.randint(self.action_size)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    return np.argmax(self.q_table[state_key])
                
                def update(self, state, action, reward, next_state):
                    state_key = self.discretize_state(state)
                    next_state_key = self.discretize_state(next_state)
                    
                    if state_key not in self.q_table:
                        self.q_table[state_key] = np.zeros(self.action_size)
                    if next_state_key not in self.q_table:
                        self.q_table[next_state_key] = np.zeros(self.action_size)
                        
                    q_value = self.q_table[state_key][action]
                    next_max = np.max(self.q_table[next_state_key])
                    new_q_value = q_value + self.alpha * (reward + self.gamma * next_max - q_value)
                    self.q_table[state_key][action] = new_q_value
            
            agent = AdversarialQLearningAgent(state_size=11, action_size=3)
            
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
                    agent.update(state, action, reward, next_state)
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
                                 seq_length=20, num_epochs=50, adv_lambda=0.1, bot_running=True):
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
                xs, ys = [], []
                for i in range(len(x_data) - seq_length):
                    seq = x_data[i:i+seq_length].detach().clone()
                    seq.requires_grad_(True)
                    xs.append(seq)
                    ys.append(y_data[i+seq_length])
                xs_tensor = torch.stack(xs).to(self.device)
                ys_tensor = torch.stack(ys).to(self.device)
                xs_tensor.requires_grad_(True)
                return xs_tensor, ys_tensor
            
            logger.info("Generating training sequences...")
            X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, seq_length)
            logger.info("Generating test sequences...")
            X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, seq_length)
            
            train_dataset = TensorDataset(X_train_seq, y_train_seq)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_dataset = TensorDataset(X_test_seq, y_test_seq)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, output_size)
                    
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            class TransformerModel(nn.Module):
                def __init__(self, input_size, seq_length, num_heads=4, dim_feedforward=128, num_layers=2, output_size=1):
                    super(TransformerModel, self).__init__()
                    adjusted_input_size = ((input_size + num_heads - 1) // num_heads) * num_heads
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=adjusted_input_size,
                        nhead=num_heads,
                        dim_feedforward=dim_feedforward,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                    self.fc = nn.Linear(adjusted_input_size * seq_length, output_size)
                    self.input_size = input_size
                    self.seq_length = seq_length
                    self.input_proj = nn.Linear(input_size, adjusted_input_size)
                    
                def forward(self, x):
                    x = self.input_proj(x)
                    out = self.transformer_encoder(x)
                    out = out.reshape(out.shape[0], -1)
                    out = self.fc(out)
                    return out
            
            logger.info("Initializing LSTM and Transformer models...")
            lstm_model = LSTMModel(input_size=input_size).to(self.device)
            transformer_model = TransformerModel(input_size=input_size, seq_length=seq_length).to(self.device)
            
            criterion = nn.MSELoss()
            lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
            transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
            
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

                    transformer_optimizer.step()
                    transformer_running_loss += transformer_total_loss.item()

                # Break out of epoch loop if bot should stop
                if not bot_running:
                    break

                if (epoch + 1) % 10 == 0:
                    lstm_epoch_loss = lstm_running_loss / len(train_loader)
                    transformer_epoch_loss = transformer_running_loss / len(train_loader)
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                                f'LSTM Loss: {lstm_epoch_loss:.4f}, '
                                f'Transformer Loss: {transformer_epoch_loss:.4f}')
                    lstm_logs[f"Epoch_{epoch+1}"] = lstm_epoch_loss
                    transformer_logs[f"Epoch_{epoch+1}"] = transformer_epoch_loss
            
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

    def analyze_stock(self, ticker, benchmark_tickers=None, prediction_days=30, training_period="7y", bot_running=True):
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

            stock = yf.Ticker(ticker)
            history = stock.history(period="2y")

            if history.empty:
                logger.error(f"No price data found for {ticker}.")
                return {
                    "success": False,
                    "message": f"Unable to fetch data for {ticker}: No price data found"
                }

            stock_info = stock.info
            current_price = float(history["Close"].iloc[-1])
            exchange_rates = self.fetch_exchange_rates()
            converted_prices = self.convert_price(current_price, exchange_rates)
            market_cap = stock_info.get("marketCap", "N/A")
            volume = stock_info.get("volume", "N/A")
            pe_ratio = stock_info.get("trailingPE", "N/A")
            dividends = stock_info.get("dividendYield", "N/A")
            dividend_yield = float(dividends) * 100 if isinstance(dividends, (int, float)) else "N/A"
            high_52w = stock_info.get("fiftyTwoWeekHigh", "N/A")
            low_52w = stock_info.get("fiftyTwoWeekLow", "N/A")
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

            sma_50 = float(history["SMA_50"].iloc[-1]) if not pd.isna(history["SMA_50"].iloc[-1]) else current_price
            sma_200 = float(history["SMA_200"].iloc[-1]) if not pd.isna(history["SMA_200"].iloc[-1]) else current_price
            ema_50 = float(history["EMA_50"].iloc[-1]) if not pd.isna(history["EMA_50"].iloc[-1]) else current_price
            volatility = float(history["Volatility"].iloc[-1]) if not pd.isna(history["Volatility"].iloc[-1]) else 0
            rsi = float(history["RSI"].iloc[-1]) if not pd.isna(history["RSI"].iloc[-1]) else 50
            bb_upper = float(history["BB_Upper"].iloc[-1]) if not pd.isna(history["BB_Upper"].iloc[-1]) else current_price * 1.1
            bb_lower = float(history["BB_Lower"].iloc[-1]) if not pd.isna(history["BB_Lower"].iloc[-1]) else current_price * 0.9
            macd = float(history["MACD"].iloc[-1]) if not pd.isna(history["MACD"].iloc[-1]) else 0
            signal_line = float(history["Signal_Line"].iloc[-1]) if not pd.isna(history["Signal_Line"].iloc[-1]) else 0
            macd_histogram = float(history["MACD_Histogram"].iloc[-1]) if not pd.isna(history["MACD_Histogram"].iloc[-1]) else 0

            momentum = 0
            if len(history) >= 30:
                momentum = (current_price - history["Close"].iloc[-30]) / history["Close"].iloc[-30]

            logger.info(f"Fetching sentiment for {ticker}...")
            sentiment_data = self.fetch_combined_sentiment(ticker)

            sentiment = sentiment_data["aggregated"]
            total_sentiment = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]
            sentiment_score = sentiment["positive"] / total_sentiment if total_sentiment > 0 else 0.5

            price_to_sma200 = current_price / sma_200 if sma_200 > 0 else 1
            # price_to_sma50 calculated but not used in current logic
            trend_direction = "UPTREND" if sma_50 > sma_200 else "DOWNTREND"
            volume_trend = "HIGH" if isinstance(volume, (int, float)) and volume > 1000000 else "MODERATE"

            logger.info(f"Fetching institutional investments data for {ticker}...")
            institutional_holders = stock.institutional_holders
            major_holders = stock.major_holders

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

            mutual_fund_holders = stock.mutualfund_holders
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

            buy_score = (
                (1 - price_to_sma200) * 0.2 +
                (1 if trend_direction == "UPTREND" else -0.5) * 0.2 +
                (sentiment_score - 0.5) * 0.25 +
                (sharpe_ratio * 0.15) +
                (0.5 - volatility) * 0.1 +
                institutional_confidence * 0.1
            )
            sell_score = (
                (price_to_sma200 - 1) * 0.2 +
                (1 if trend_direction == "DOWNTREND" else -0.5) * 0.2 +
                ((1 - sentiment_score) - 0.5) * 0.25 +
                (-sharpe_ratio * 0.15) +
                (volatility - 0.5) * 0.1 +
                (-institutional_confidence * 0.1)
            )

            if buy_score > 0.2:
                recommendation = "STRONG BUY"
            elif buy_score > 0:
                recommendation = "BUY"
            elif sell_score > 0.2:
                recommendation = "STRONG SELL"
            elif sell_score > 0:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"

            support_level = min(sma_200, sma_50) * 0.95
            resistance_level = max(current_price * 1.05, sma_50 * 1.05)

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
                stock_data, recommendation, buy_score, sell_score,
                price_to_sma200, trend_direction, sentiment_score,
                volatility, sharpe_ratio
            )

            logger.info(f"Fetching extended data for {ticker} for ML analysis...")
            extended_history = stock.history(period=training_period)

            if extended_history.empty:
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

                        logger.info(f"Training ML models for {ticker}...")
                        base_models = [
                            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                            ('xgb', XGBRegressor(n_estimators=100, random_state=42))
                        ]
                        meta_model = LinearRegression()
                        stacking_regressor = StackingRegressor(
                            estimators=base_models,
                            final_estimator=meta_model
                        )

                        stacking_regressor.fit(X_train, y_train)
                        y_pred_scaled = stacking_regressor.predict(X_test)
                        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                        mse = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred)
                        mae = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred)
                        r2 = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred)

                        last_features = X.iloc[-1:].values
                        last_features_scaled = scaler_X.transform(last_features)
                        predicted_price_scaled = stacking_regressor.predict(last_features_scaled)
                        predicted_price = scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

                        logger.info(f"Performing adversarial training for {ticker}...")
                        adv_training_result = self.adversarial_training_loop(
                            X_train, y_train, X_test, y_test, input_size=X.shape[1], bot_running=bot_running
                        )

                        if adv_training_result["success"]:
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

                        ml_analysis = {
                            "success": True,
                            "predicted_price": float(ensemble_pred),
                            "confidence": float(r2),
                            "mse": float(mse),
                            "mae": float(mae),
                            "r2_score": float(r2),
                            "stacking_regressor_metrics": {
                                "mse": float(mse),
                                "mae": float(mae),
                                "r2": float(r2)
                            },
                            "lstm_metrics": lstm_metrics,
                            "transformer_metrics": transformer_metrics,
                            "rl_metrics": rl_result,
                            "ensemble_components": {
                                "stacking_regressor": float(predicted_price),
                                "lstm": float(lstm_pred),
                                "transformer": float(transformer_pred)
                            }
                        }

            result = {
                "success": True,
                "stock_data": stock_data,
                "recommendation": recommendation,
                "buy_score": float(buy_score),
                "sell_score": float(sell_score),
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
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "dividend_yield": dividend_yield,
                    "52w_high": high_52w,
                    "52w_low": low_52w,
                    "balance_sheet": balance_sheet_filtered,
                    "income_statement": income_statement_filtered,
                    "cash_flow": cash_flow_filtered
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

        # Initialize chatbot command handler
        self.chatbot = ChatbotCommandHandler(self)
        self.bot_running = False
        self.command_queue = queue.Queue()
        self.initialize()

    def initialize(self):
        """Initialize the bot and its components."""
        logger.info("Initializing Stock Trading Bot for Indian market...")
        # Portfolio is already initialized in VirtualPortfolio.__init__()
        # Don't call initialize_portfolio() as it resets the data

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
        if not analysis.get("success"):
            logger.warning(f"Skipping trading decision for {analysis.get('stock_data', {}).get('symbol')} due to failed analysis")
            return None

        ticker = analysis["stock_data"]["symbol"]
        recommendation = analysis["recommendation"]
        current_price = analysis["stock_data"]["current_price"]["INR"]  # Changed to INR
        ml_analysis = analysis["ml_analysis"]
        technical_indicators = analysis["technical_indicators"]
        sentiment_data = analysis["sentiment_analysis"]
        risk_level = analysis["risk_level"]
        support_level = analysis["support_level"]
        resistance_level = analysis["resistance_level"]
        volatility = technical_indicators["volatility"]
        metrics = self.portfolio.get_metrics()
        available_cash = metrics["cash"]
        total_value = metrics["total_value"]
        history = yf.Ticker(ticker).history(period="2y")

        # Signal weights
        weights = {
            "technical": 0.4,
            "sentiment": 0.3,
            "ml": 0.2,
            "rl": 0.1
        }
        scores = {"buy": 0.0, "sell": 0.0}

        # Technical Score
        technical_score = 0.0
        if recommendation in ["BUY", "STRONG BUY"]:
            technical_score += 0.6
            if technical_indicators["rsi"] < 35:
                technical_score += 0.3
            if technical_indicators["macd"] > technical_indicators["signal_line"]:
                technical_score += 0.2
            if current_price < technical_indicators["bb_lower"]:
                technical_score += 0.2
            if technical_indicators["sma_50"] > technical_indicators["sma_200"]:
                technical_score += 0.2
            scores["buy"] += technical_score * weights["technical"]
        elif recommendation in ["SELL", "STRONG SELL"]:
            technical_score -= 0.6
            if technical_indicators["rsi"] > 65:
                technical_score -= 0.3
            if technical_indicators["macd"] < technical_indicators["signal_line"]:
                technical_score -= 0.2
            if current_price > technical_indicators["bb_upper"]:
                technical_score -= 0.2
            if technical_indicators["sma_50"] < technical_indicators["sma_200"]:
                technical_score -= 0.2
            scores["sell"] += abs(technical_score) * weights["technical"]

        # Sentiment Score
        sentiment_score = 0.0
        aggregated = sentiment_data["aggregated"]
        total_sentiment = aggregated["positive"] + aggregated["negative"] + aggregated["neutral"]
        sentiment_ratio = aggregated["positive"] / total_sentiment if total_sentiment > 0 else 0.5
        sentiment_score = (sentiment_ratio - 0.5) * 2
        if sentiment_score > 0:
            scores["buy"] += sentiment_score * weights["sentiment"]
        else:
            scores["sell"] += abs(sentiment_score) * weights["sentiment"]

        # ML Score
        ml_score = 0.0
        if ml_analysis.get("success"):
            ensemble_r2 = ml_analysis["r2_score"]
            predicted_price = ml_analysis["predicted_price"]
            price_change_pct = ((predicted_price - current_price) / current_price) if current_price > 0 else 0
            if price_change_pct > 0.03 and ensemble_r2 > 0.4:
                ml_score += min(price_change_pct * 2, 1.0) * ensemble_r2
            elif price_change_pct < -0.03 and ensemble_r2 > 0.4:
                ml_score -= min(abs(price_change_pct) * 2, 1.0) * ensemble_r2
            if ml_score > 0:
                scores["buy"] += ml_score * weights["ml"]
            else:
                scores["sell"] += abs(ml_score) * weights["ml"]

        # RL Score
        if ml_analysis.get("rl_metrics", {}).get("success"):
            rl_recommendation = ml_analysis["rl_metrics"]["recommendation"]
            if rl_recommendation == "SELL":
                scores["sell"] += 0.3 * weights["rl"]
            elif rl_recommendation == "BUY":
                scores["buy"] += 0.3 * weights["rl"]

        final_buy_score = scores["buy"]
        final_sell_score = scores["sell"]

        # Confidence Thresholds
        base_confidence_threshold = 0.12
        confidence_threshold = min(base_confidence_threshold + (volatility * 0.4), 0.18)
        if risk_level == "HIGH":
            confidence_threshold += 0.04
        elif risk_level == "LOW":
            confidence_threshold -= 0.04
        sell_confidence_threshold = confidence_threshold * 0.75

        # Calculate unrealized PnL
        current_prices = self.portfolio.get_current_prices()
        current_ticker_price = current_prices.get(ticker, {"price": current_price})["price"]
        unrealized_pnl = (
            (current_ticker_price - self.portfolio.holdings[ticker]["avg_price"])
            * self.portfolio.holdings[ticker]["qty"]
            if ticker in self.portfolio.holdings else 0
        )

        # Signals
        buy_signals = sum([
            technical_score > 0,
            sentiment_score > 0,
            ml_score > 0,
            ml_analysis.get("rl_metrics", {}).get("recommendation") == "BUY"
        ])
        sell_signals = sum([
            technical_score < 0,
            sentiment_score < 0,
            ml_score < 0,
            current_ticker_price < support_level * 0.97,
            current_ticker_price > resistance_level * 1.03,
            unrealized_pnl < -0.06 * total_value,
            ml_analysis.get("rl_metrics", {}).get("recommendation") == "SELL"
        ])

        # Portfolio Constraints
        max_exposure_per_stock = total_value * 0.25
        current_stock_exposure = self.portfolio.holdings.get(ticker, {"qty": 0})["qty"] * current_ticker_price
        sector = analysis["stock_data"]["sector"]
        sector_exposure = sum(
            data["qty"] * price["price"]
            for asset, data in self.portfolio.holdings.items()
            for price in [self.portfolio.get_current_prices().get(asset, {"price": 0})]
            if analysis["stock_data"].get("sector") == sector
        )
        max_sector_exposure = total_value * 0.4

        # Calculate ATR for volatility-based sizing
        high_low = history['High'] - history['Low']
        high_close = abs(history['High'] - history['Close'].shift())
        low_close = abs(history['Low'] - history['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        atr = float(atr) if not pd.isna(atr) else volatility * current_ticker_price

        # Kelly Criterion for position sizing
        win_prob = min(final_buy_score / 0.4, 1.0) if final_buy_score > 0 else 0.5
        expected_return = (ml_analysis.get("predicted_price", current_ticker_price) / current_ticker_price - 1) if current_ticker_price > 0 else 0
        # loss_prob = 1 - win_prob  # Calculated but not used
        kelly_fraction = (win_prob * (expected_return + 1) - 1) / expected_return if expected_return > 0 else 0.15
        kelly_fraction = max(min(kelly_fraction, 0.3), 0.1)

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
            highest_price = max(history["Close"].iloc[-30:]) if not history.empty else current_ticker_price
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

        # Buy Quantity Calculation
        buy_qty = 0
        if final_buy_score > confidence_threshold and buy_signals >= 2:
            if available_cash < 5000:  # Minimum trade value in INR
                logger.info(f"Skipping BUY for {ticker}: Insufficient cash (Rs.{available_cash:.2f} < Rs.5000)")
            else:
                position_size_pct = kelly_fraction * 0.6
                target_position_value = total_value * position_size_pct
                volatility_factor = max(0.4, min(1.6, 1.0 / (1 + atr / current_ticker_price))) if current_ticker_price > 0 else 1.0
                target_position_value *= volatility_factor
                confidence_factor = 0.6 + (final_buy_score / 0.4) * 0.4
                confidence_factor = min(max(confidence_factor, 0.6), 1.0)
                target_position_value *= confidence_factor
                target_position_value = max(5000, min(target_position_value, max_exposure_per_stock - current_stock_exposure))
                target_position_value = min(target_position_value, max_sector_exposure - sector_exposure)
                buy_qty = target_position_value / current_ticker_price if current_ticker_price > 0 else 0
                buy_qty = min(buy_qty, max_qty_by_volume, available_cash / current_ticker_price)
                buy_qty = max(0, int(buy_qty))

        # Sell Quantity Calculation
        sell_qty = 0
        if ticker in self.portfolio.holdings and final_sell_score > sell_confidence_threshold and sell_signals >= 2:
            holding_qty = self.portfolio.holdings[ticker]["qty"]
            price_to_take_profit = (current_ticker_price - take_profit) / take_profit if take_profit > 0 else 0
            price_to_stop_loss = (stop_loss - current_ticker_price) / stop_loss if stop_loss > 0 else 0
            sell_factor = min(max(final_sell_score * 1.5, abs(price_to_take_profit), abs(price_to_stop_loss)), 1.0)
            if final_sell_score > 0.4 or current_ticker_price < stop_loss or current_ticker_price > take_profit:
                sell_qty = holding_qty
            else:
                sell_qty = holding_qty * sell_factor
                volatility_factor = max(0.4, min(1.6, 1.0 / (1 + atr / current_ticker_price))) if current_ticker_price > 0 else 1.0
                sell_qty *= volatility_factor
            sell_qty = max(0, int(min(sell_qty, holding_qty)))

        # Validate Exposure Limits
        if buy_qty > 0:
            new_stock_exposure = current_stock_exposure + buy_qty * current_ticker_price
            new_sector_exposure = sector_exposure + buy_qty * current_ticker_price
            if new_stock_exposure > max_exposure_per_stock or new_sector_exposure > max_sector_exposure:
                buy_qty = min(
                    (max_exposure_per_stock - current_stock_exposure) / current_ticker_price,
                    (max_sector_exposure - sector_exposure) / current_ticker_price
                ) if current_ticker_price > 0 else 0
                buy_qty = max(0, int(buy_qty))

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

        # Execute Trades
        trade = None
        if (
            buy_qty > 0
            and final_buy_score > confidence_threshold
            and buy_signals >= 2
            and buy_qty * current_ticker_price <= available_cash
            and not backoff
        ):
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
        elif (
            sell_qty > 0
            and final_sell_score > sell_confidence_threshold
            and sell_signals >= 2
            and ticker in self.portfolio.holdings
            and not backoff
        ):
            logger.info(
                f"Executing SELL for {ticker}: {sell_qty:.0f} units at Rs.{current_ticker_price:.2f}, "
                f"Position Value: Rs.{sell_qty * current_price:.2f} ({(sell_qty * current_price / total_value * 100):.2f}% of portfolio), "
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
            hold_conditions = (
                abs(final_buy_score - final_sell_score) < 0.06
                or (support_level * 0.98 < current_ticker_price < resistance_level * 1.02)
                or (buy_signals < 2 and sell_signals < 2)
            )
            if hold_conditions:
                logger.info(f"HOLD {ticker}: Neutral market conditions, "
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

        return trade

    def run_analysis(self, ticker):
        """Run analysis for a given ticker and return the result."""
        return self.stock_analyzer.analyze_stock(
            ticker,
            benchmark_tickers=self.config.get("benchmark_tickers", ["^NSEI"]),
            prediction_days=self.config.get("prediction_days", 30),
            training_period=self.config.get("period", "3y"),
            bot_running=self.bot_running
        )

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
    print("\n🤖 Bot shut down successfully!")
    sys.exit(0)
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Indian Stock Trading Bot')
    parser.add_argument('--mode', choices=['live', 'paper'],
                       default=os.getenv('TRADING_MODE', 'paper'),
                       help='Trading mode: live or paper (default: paper)')
    return parser.parse_args()

def main_with_mode():
    """Main function with mode selection and enhanced configuration."""
    try:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)

        # Parse command line arguments
        args = parse_arguments()

        # Load environment variables
        load_dotenv()

        # Enhanced configuration with risk management
        config = {
            "tickers": [],  # Empty by default - users can add tickers manually
            "starting_balance": 10000,  # ₹10 thousand
            "current_portfolio_value": 10000,
            "current_pnl": 0,
            "mode": args.mode,
            "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
            "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
            "period": "3y",
            "prediction_days": 30,
            "benchmark_tickers": ["^NSEI"],
            "sleep_interval": 300,  # 5 minutes
            # Risk management settings from .env
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
            "max_capital_per_trade": float(os.getenv("MAX_CAPITAL_PER_TRADE", "0.25")),
            "max_trade_limit": int(os.getenv("MAX_TRADE_LIMIT", "10"))
        }

        # Display mode information
        mode_display = " LIVE TRADING" if args.mode == "live" else "📝 PAPER TRADING"
        logger.info(f"Starting Indian Stock Trading Bot in {mode_display} mode")

        # Display startup banner - REMOVED as requested by user

        if args.mode == "live":
            if not config["dhan_client_id"] or not config["dhan_access_token"]:
                logger.error("Dhan API credentials not found in .env file. Cannot run in live mode.")
                print(" Error: Dhan API credentials required for live trading!")
                return
            print("⚠️  WARNING: Live trading mode enabled. Real money will be used!")
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