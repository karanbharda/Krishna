#!/usr/bin/env python3
"""
Sentiment Analysis Tool with Dynamic Global News Scraper
======================

Production-grade MCP tool for comprehensive sentiment analysis from multiple sources
including news, social media, market sentiment indicators, and Indian RSS sources.
Uses LangGraph for agent workflows and FastMCP for tools.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import requests
import re
import feedparser
from bs4 import BeautifulSoup
import random
import time
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError

# Import existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ..mcp_trading_server import MCPToolResult, MCPToolStatus

# Import LangChain and LangGraph components
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import tool
    from langchain.prompts import PromptTemplate
    from langchain.chat_models import ChatOpenAI  # We'll use this with xAI API for Grok
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
except ImportError:
    AgentExecutor = None
    create_react_agent = None
    tool = None
    PromptTemplate = None
    ChatOpenAI = None
    StateGraph = None
    END = None
    ToolExecutor = None

# Alternative solution: Completely disable Prometheus metrics to avoid duplication issues
class DummyMetric:
    """Dummy metric class that does nothing - prevents Prometheus conflicts"""
    def __init__(self, *args, **kwargs): 
        pass
    def labels(self, *args, **kwargs): 
        return self
    def inc(self, *args, **kwargs): 
        pass
    def dec(self, *args, **kwargs): 
        pass
    def observe(self, *args, **kwargs): 
        pass

# Use dummy metrics everywhere to prevent any conflicts
SENTIMENT_TOOL_CALLS = DummyMetric()
SENTIMENT_ANALYSIS_DURATION = DummyMetric()
SENTIMENT_ACTIVE_SESSIONS = DummyMetric()
SENTIMENT_ERROR_RATE = DummyMetric()

logger = logging.getLogger(__name__)

# Enhanced user agents to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

# Common headers to mimic real browser behavior
COMMON_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Log LangChain availability after logger is defined
try:
    import langchain
    logger.info("LangChain and LangGraph components available")
except ImportError:
    logger.warning("LangChain/LangGraph not available, using fallback implementations")

@dataclass
class SentimentScore:
    """Sentiment score structure"""
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    sentiment_score: SentimentScore
    relevance_score: float

@dataclass
class SentimentAnalysisResult:
    """Comprehensive sentiment analysis result"""
    symbol: str
    overall_sentiment: SentimentScore
    news_sentiment: SentimentScore
    social_sentiment: SentimentScore
    market_sentiment: SentimentScore
    indian_news_sentiment: SentimentScore
    sentiment_trend: str  # "IMPROVING", "DECLINING", "STABLE"
    key_themes: List[str]
    news_items: List[NewsItem]
    analysis_timestamp: datetime

@dataclass
class AgentState:
    """State for LangGraph agent"""
    symbol: str
    company_name: str
    news_items: List[NewsItem]
    sentiment_scores: Dict[str, SentimentScore]
    final_decision: str
    confidence: float
    session_id: str


class SentimentTool:
    """
    Production-grade sentiment analysis tool with Dynamic Global News Scraper
    
    Features:
    - Multi-source sentiment analysis
    - News sentiment from financial sources
    - Social media sentiment tracking
    - Market sentiment indicators
    - Indian RSS news sources (Moneycontrol, Economic Times, etc.)
    - Sentiment trend analysis
    - Real-time sentiment monitoring
    - LangGraph agent workflows
    - FastMCP integration
    - Auto/approval modes with 30s timeout
    - JSON output format
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "sentiment_tool")
        
        # API configurations
        self.news_api_key = config.get("news_api_key")
        self.reddit_config = config.get("reddit", {})
        self.xai_api_key = config.get("xai_api_key")  # For Grok integration
        self.ollama_enabled = config.get("ollama_enabled", False)
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")
        self.ollama_model = config.get("ollama_model", "llama2")
        
        # Sentiment analysis configuration
        self.sentiment_sources = config.get("sentiment_sources", ["news", "social", "market", "indian_news"])
        self.lookback_days = config.get("lookback_days", 7)
        self.mode = config.get("mode", "auto")  # "auto" or "approval"
        
        # Indian RSS sources - updated with more reliable working URLs
        self.indian_rss_sources = [
            "https://www.moneycontrol.com/rss/marketreports.xml",  # MoneyControl market reports
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",  # Economic Times stocks
            "https://www.business-standard.com/rss/markets-106.rss",  # Business Standard markets
            "https://www.hindubusinessline.com/markets/rssfeed/?id=1016",  # Hindu Business Line markets (replace with alternative)
            "https://in.investing.com/rss/market_overview.rss",  # Investing.com market overview
            "https://www.livemint.com/rss/markets",  # LiveMint markets
            "https://timesofindia.indiatimes.com/rssfeeds/2146842.cms",  # Times of India business (replace with alternative)
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/12741757.cms",  # Additional ET source (replace with alternative)
            "https://www.reuters.com/news/archive/businessNews",  # Reuters business news (replace with alternative)
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"  # CNBC market news
        ]
        
        # Global RSS sources - updated with working URLs
        self.global_rss_sources = [
            "https://www.marketwatch.com/rss/marketpulse",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://www.reuters.com/news/archive/businessNews"  # Additional Reuters source
        ]
        
        # Performance tracking
        self.analyses_performed = 0
        self.news_items_processed = 0
        
        # Initialize sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.vader_available = True
        except ImportError:
            logger.warning("VADER sentiment analyzer not available")
            self.sentiment_analyzer = None
            self.vader_available = False
        
        # Initialize LangGraph agent if available
        self.agent_executor = None
        if AgentExecutor and create_react_agent and tool and PromptTemplate and ChatOpenAI:
            try:
                self._initialize_langgraph_agent()
            except Exception as e:
                logger.warning(f"Failed to initialize LangGraph agent: {e}")
        
        logger.info(f"Sentiment Tool {self.tool_id} initialized with {len(self.indian_rss_sources)} Indian RSS sources")
    
    def _make_request(self, url, use_enhanced=False):
        """Make HTTP request with enhanced scraping techniques and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{max_retries})")
                
                if use_enhanced:
                    # Use enhanced scraping with rotating user agents
                    headers = COMMON_HEADERS.copy()
                    headers["User-Agent"] = random.choice(USER_AGENTS)
                    
                    # Add random delay to mimic human behavior
                    delay = random.uniform(1, 3)
                    logger.debug(f"Adding random delay of {delay:.2f}s before request")
                    time.sleep(delay)
                    
                    logger.debug(f"Request headers: {headers}")
                    response = requests.get(url, headers=headers, timeout=15)
                else:
                    response = requests.get(url, timeout=10)
                
                logger.debug(f"Response status: {response.status_code}")
                logger.debug(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                logger.debug(f"Request successful for {url}")
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"RSS source {url} not found (404 error) - this source may be deprecated")
                    return None
                elif e.response.status_code == 403:
                    logger.warning(f"Access forbidden to RSS source {url} (403 error) - may require authentication")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {url} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
                else:
                    logger.warning(f"HTTP error fetching from RSS source {url}: {e.response.status_code} - {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {url} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return None
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout error fetching from RSS source {url}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {url} (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error fetching from RSS source {url}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {url} (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
            except Exception as e:
                logger.warning(f"Error fetching from RSS source {url}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {url} (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        return None
    
    async def analyze_sentiment(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Comprehensive sentiment analysis for a symbol with Dynamic Global News Scraper
        
        Args:
            arguments: {
                "symbol": "RELIANCE.NS",
                "sources": ["news", "social", "market", "indian_news"],
                "lookback_days": 7,
                "include_news_items": true,
                "mode": "auto" or "approval"
            }
        """
        try:
            symbol = arguments.get("symbol")
            if not symbol:
                raise ValueError("Symbol is required")
            
            sources = arguments.get("sources", self.sentiment_sources)
            lookback_days = arguments.get("lookback_days", self.lookback_days)
            include_news_items = arguments.get("include_news_items", True)
            mode = arguments.get("mode", self.mode)
            
            # Initialize result components
            news_sentiment = None
            social_sentiment = None
            market_sentiment = None
            indian_news_sentiment = None
            news_items = []
            
            # Analyze news sentiment
            if "news" in sources:
                news_sentiment, news_items = await self._analyze_news_sentiment(
                    symbol, lookback_days, include_news_items
                )
            
            # Analyze social sentiment
            if "social" in sources:
                social_sentiment = await self._analyze_social_sentiment(symbol, lookback_days)
            
            # Analyze market sentiment
            if "market" in sources:
                market_sentiment = await self._analyze_market_sentiment(symbol)
            
            # Analyze Indian news sentiment
            if "indian_news" in sources:
                indian_news_sentiment, indian_news_items = await self._analyze_indian_news_sentiment(
                    symbol, lookback_days, include_news_items
                )
                # Add Indian news items to the main list
                if include_news_items and indian_news_items:
                    news_items.extend(indian_news_items)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(
                news_sentiment, social_sentiment, market_sentiment, indian_news_sentiment
            )
            
            # Determine sentiment trend
            sentiment_trend = await self._calculate_sentiment_trend(symbol, lookback_days)
            
            # Extract key themes
            key_themes = self._extract_key_themes(news_items)
            
            # Apply regime multipliers
            regime_multiplier = self._get_regime_multiplier(overall_sentiment)
            adjusted_sentiment = self._apply_regime_multiplier(overall_sentiment, regime_multiplier)
            
            # Create result
            result = SentimentAnalysisResult(
                symbol=symbol,
                overall_sentiment=adjusted_sentiment,
                news_sentiment=news_sentiment or self._create_neutral_sentiment(),
                social_sentiment=social_sentiment or self._create_neutral_sentiment(),
                market_sentiment=market_sentiment or self._create_neutral_sentiment(),
                indian_news_sentiment=indian_news_sentiment or self._create_neutral_sentiment(),
                sentiment_trend=sentiment_trend,
                key_themes=key_themes,
                news_items=news_items if include_news_items else [],
                analysis_timestamp=datetime.now()
            )
            
            # Handle approval mode if needed
            if mode == "approval" and self.xai_api_key:
                approval_result = await self._request_approval(symbol, adjusted_sentiment, session_id)
                if approval_result and approval_result.get("status") == "timeout":
                    # Fallback to auto mode after 30s timeout
                    logger.info(f"Approval timeout for {symbol}, falling back to auto mode")
            
            # Prepare response data
            response_data = {
                "symbol": result.symbol,
                "overall_sentiment": asdict(result.overall_sentiment),
                "sentiment_breakdown": {
                    "news": asdict(result.news_sentiment),
                    "social": asdict(result.social_sentiment),
                    "market": asdict(result.market_sentiment),
                    "indian_news": asdict(result.indian_news_sentiment)
                },
                "sentiment_trend": result.sentiment_trend,
                "key_themes": result.key_themes,
                "sentiment_summary": self._generate_sentiment_summary(result),
                "news_items_count": len(result.news_items),
                "regime_multiplier": regime_multiplier,
                "analysis_metadata": {
                    "sources_analyzed": sources,
                    "lookback_days": lookback_days,
                    "timestamp": result.analysis_timestamp.isoformat(),
                    "session_id": session_id,
                    "mode": mode
                }
            }
            
            # Include news items if requested
            if include_news_items:
                response_data["news_items"] = [asdict(item) for item in result.news_items]
            
            self.analyses_performed += 1
            self.news_items_processed += len(news_items)
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data=response_data,
                reasoning=f"Sentiment analysis completed for {symbol} using {len(sources)} sources with regime multiplier {regime_multiplier}",
                confidence=adjusted_sentiment.confidence
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def get_sentiment_alerts(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Get sentiment-based alerts and notifications
        
        Args:
            arguments: {
                "symbols": ["RELIANCE.NS", "TCS.NS"],
                "alert_threshold": 0.3,
                "alert_types": ["extreme_sentiment", "sentiment_change"]
            }
        """
        try:
            symbols = arguments.get("symbols", [])
            alert_threshold = arguments.get("alert_threshold", 0.3)
            alert_types = arguments.get("alert_types", ["extreme_sentiment", "sentiment_change"])
            
            alerts = []
            
            for symbol in symbols:
                # Analyze current sentiment
                sentiment_result = await self.analyze_sentiment({
                    "symbol": symbol,
                    "sources": ["news", "market"],
                    "include_news_items": False
                }, session_id)
                
                if sentiment_result.status == MCPToolStatus.SUCCESS:
                    sentiment_data = sentiment_result.data
                    overall_sentiment = sentiment_data["overall_sentiment"]
                    
                    # Check for extreme sentiment
                    if "extreme_sentiment" in alert_types:
                        if abs(overall_sentiment["compound"]) > alert_threshold:
                            alerts.append({
                                "type": "EXTREME_SENTIMENT",
                                "symbol": symbol,
                                "severity": "HIGH" if abs(overall_sentiment["compound"]) > 0.6 else "MEDIUM",
                                "message": f"{symbol} showing {'very positive' if overall_sentiment['compound'] > 0 else 'very negative'} sentiment",
                                "sentiment_score": overall_sentiment["compound"],
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    # Check for sentiment changes (simplified)
                    if "sentiment_change" in alert_types:
                        trend = sentiment_data.get("sentiment_trend", "STABLE")
                        if trend in ["IMPROVING", "DECLINING"]:
                            alerts.append({
                                "type": "SENTIMENT_CHANGE",
                                "symbol": symbol,
                                "severity": "MEDIUM",
                                "message": f"{symbol} sentiment is {trend.lower()}",
                                "trend": trend,
                                "timestamp": datetime.now().isoformat()
                            })
            
            return MCPToolResult(
                status=MCPToolStatus.SUCCESS,
                data={
                    "alerts": alerts,
                    "alert_count": len(alerts),
                    "symbols_monitored": len(symbols),
                    "alert_threshold": alert_threshold,
                    "monitoring_timestamp": datetime.now().isoformat()
                },
                reasoning=f"Sentiment monitoring completed for {len(symbols)} symbols",
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Sentiment alerts error: {e}")
            return MCPToolResult(
                status=MCPToolStatus.ERROR,
                error=str(e)
            )
    
    async def _analyze_news_sentiment(self, symbol: str, lookback_days: int, 
                                    include_items: bool) -> tuple[SentimentScore, List[NewsItem]]:
        """Analyze sentiment from financial news"""
        try:
            # Extract company name from symbol for better search
            company_name = self._extract_company_name(symbol)
            
            # Get news articles
            news_articles = await self._fetch_news_articles(company_name, lookback_days)
            
            if not news_articles:
                return self._create_neutral_sentiment(), []
            
            # Analyze sentiment for each article
            news_items = []
            sentiment_scores = []
            
            for article in news_articles:
                if self.vader_available:
                    # Use VADER for sentiment analysis
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    
                    sentiment_score = SentimentScore(
                        positive=sentiment['pos'],
                        negative=sentiment['neg'],
                        neutral=sentiment['neu'],
                        compound=sentiment['compound'],
                        confidence=0.8
                    )
                else:
                    # Fallback sentiment analysis
                    sentiment_score = self._simple_sentiment_analysis(
                        f"{article.get('title', '')} {article.get('description', '')}"
                    )
                
                if include_items:
                    news_item = NewsItem(
                        title=article.get('title', ''),
                        content=article.get('description', ''),
                        source=article.get('source', {}).get('name', 'Unknown'),
                        url=article.get('url', ''),
                        published_date=self._parse_date(article.get('publishedAt')),
                        sentiment_score=sentiment_score,
                        relevance_score=self._calculate_relevance(article, company_name)
                    )
                    news_items.append(news_item)
                
                sentiment_scores.append(sentiment_score)
            
            # Calculate aggregate news sentiment
            if sentiment_scores:
                avg_sentiment = SentimentScore(
                    positive=sum(s.positive for s in sentiment_scores) / len(sentiment_scores),
                    negative=sum(s.negative for s in sentiment_scores) / len(sentiment_scores),
                    neutral=sum(s.neutral for s in sentiment_scores) / len(sentiment_scores),
                    compound=sum(s.compound for s in sentiment_scores) / len(sentiment_scores),
                    confidence=sum(s.confidence for s in sentiment_scores) / len(sentiment_scores)
                )
            else:
                avg_sentiment = self._create_neutral_sentiment()
            
            return avg_sentiment, news_items
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return self._create_neutral_sentiment(), []
    
    async def _analyze_social_sentiment(self, symbol: str, lookback_days: int) -> SentimentScore:
        """Analyze sentiment from social media sources"""
        try:
            # Simplified social sentiment analysis
            # In production, this would integrate with Reddit, Twitter, etc.
            
            # Simulate social sentiment based on symbol characteristics
            company_name = self._extract_company_name(symbol)
            
            # Generate simulated social sentiment
            import random
            random.seed(hash(symbol) % 1000)  # Consistent randomness per symbol
            
            base_sentiment = random.uniform(-0.3, 0.3)
            
            return SentimentScore(
                positive=max(0, base_sentiment + 0.5),
                negative=max(0, -base_sentiment + 0.3),
                neutral=0.2,
                compound=base_sentiment,
                confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Social sentiment analysis error: {e}")
            return self._create_neutral_sentiment()
    
    async def _analyze_market_sentiment(self, symbol: str) -> SentimentScore:
        """Analyze market-based sentiment indicators"""
        try:
            # Market sentiment based on technical indicators
            # This would typically use price action, volume, volatility
            
            # Simplified market sentiment calculation
            # In production, this would use actual market data
            
            # Simulate market sentiment
            import random
            random.seed(hash(symbol + "market") % 1000)
            
            market_sentiment = random.uniform(-0.4, 0.4)
            
            return SentimentScore(
                positive=max(0, market_sentiment + 0.4),
                negative=max(0, -market_sentiment + 0.4),
                neutral=0.2,
                compound=market_sentiment,
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Market sentiment analysis error: {e}")
            return self._create_neutral_sentiment()
    
    async def _fetch_news_articles(self, company_name: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Fetch news articles from various sources"""
        try:
            articles = []
            
            # Try NewsAPI if available
            if self.news_api_key:
                articles.extend(await self._fetch_from_newsapi(company_name, lookback_days))
            
            # Add other news sources here
            # articles.extend(await self._fetch_from_other_sources(company_name, lookback_days))
            
            return articles
            
        except Exception as e:
            logger.error(f"News fetching error: {e}")
            return []
    
    async def _fetch_indian_news_articles(self, company_name: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Fetch Indian news articles from RSS sources"""
        try:
            articles = []
            
            # Fetch from Indian RSS sources with enhanced scraping
            for rss_url in self.indian_rss_sources:
                try:
                    # Use enhanced scraping techniques
                    response = self._make_request(rss_url, use_enhanced=True)
                    feed_content = response.content
                    feed = feedparser.parse(feed_content)
                    
                    for entry in feed.entries[:5]:  # Limit to 5 articles per source
                        # Check if the article is recent enough
                        published_date = getattr(entry, 'published_parsed', None)
                        if published_date:
                            published_dt = datetime(*published_date[:6])
                            if (datetime.now() - published_dt).days <= lookback_days:
                                # Check if the article is relevant to the company
                                title = getattr(entry, 'title', '')
                                summary = getattr(entry, 'summary', '')
                                
                                if company_name.lower() in title.lower() or company_name.lower() in summary.lower():
                                    articles.append({
                                        'title': title,
                                        'description': summary,
                                        'url': getattr(entry, 'link', ''),
                                        'publishedAt': published_dt.isoformat(),
                                        'source': {'name': getattr(entry, 'source', {}).get('title', 'Unknown Indian Source')}
                                    })
                except Exception as e:
                    logger.warning(f"Error fetching from RSS source {rss_url}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Indian news fetching error: {e}")
            return []
    
    async def _analyze_indian_news_sentiment(self, symbol: str, lookback_days: int, include_news_items: bool = True) -> Tuple[SentimentScore, List[NewsItem]]:
        """Analyze Indian news sentiment with enhanced error handling and detailed logging"""
        try:
            logger.info(f"Analyzing Indian news sentiment for {symbol}")
            
            all_news_items = []
            successful_sources = 0
            failed_sources = 0
            
            # Track which sources are consistently failing
            failing_sources = []
            working_sources = []
            
            # Fetch from all Indian RSS sources
            for i, url in enumerate(self.indian_rss_sources):
                try:
                    logger.debug(f"[{i+1}/{len(self.indian_rss_sources)}] Processing RSS source: {url}")
                    news_items = await self._fetch_rss_feed(url, max_items=5)
                    if news_items:
                        # Filter relevant news items
                        relevant_items = self._filter_relevant_news(news_items, symbol, lookback_days)
                        all_news_items.extend(relevant_items)
                        successful_sources += 1
                        working_sources.append(url)
                        logger.info(f"✓ SUCCESS: {url} - Fetched {len(relevant_items)} relevant items")
                    else:
                        failed_sources += 1
                        failing_sources.append(url)
                        logger.warning(f"✗ FAILED: {url} - No items fetched")
                except Exception as e:
                    failed_sources += 1
                    failing_sources.append(url)
                    logger.error(f"✗ ERROR: {url} - {str(e)}")
                    continue
            
            # If we have too few successful sources, try alternative sources
            if successful_sources < 3 and failing_sources:
                logger.info(f"Trying alternative RSS sources due to low success rate ({successful_sources}/{len(self.indian_rss_sources)})")
                alternative_sources = self._get_alternative_rss_sources(failing_sources)
                
                for i, url in enumerate(alternative_sources):
                    try:
                        # Skip if we already tried this source
                        if url in self.indian_rss_sources:
                            continue
                            
                        logger.debug(f"[ALT {i+1}/{len(alternative_sources)}] Processing alternative RSS source: {url}")
                        news_items = await self._fetch_rss_feed(url, max_items=3)
                        if news_items:
                            # Filter relevant news items
                            relevant_items = self._filter_relevant_news(news_items, symbol, lookback_days)
                            all_news_items.extend(relevant_items)
                            successful_sources += 1
                            working_sources.append(url)
                            logger.info(f"✓ SUCCESS (ALT): {url} - Fetched {len(relevant_items)} relevant items")
                        else:
                            logger.warning(f"✗ FAILED (ALT): {url} - No items fetched")
                    except Exception as e:
                        logger.error(f"✗ ERROR (ALT): {url} - {str(e)}")
                        continue
            
            # Log detailed summary
            logger.info(f"=== Indian News Sentiment Analysis Summary for {symbol} ===")
            logger.info(f"Total sources: {len(self.indian_rss_sources)}")
            logger.info(f"Working sources ({successful_sources}):")
            for source in working_sources:
                logger.info(f"  ✓ {source}")
            
            if failing_sources:
                logger.info(f"Failing sources ({failed_sources}):")
                for source in failing_sources:
                    logger.info(f"  ✗ {source}")
            
            # If too many sources are failing, log a warning
            if successful_sources < 3:
                logger.warning(f"Warning: Only {successful_sources} out of {len(self.indian_rss_sources)} Indian RSS sources are working. Consider updating the source list.")
            
            # Calculate sentiment from collected news items
            if all_news_items:
                sentiment = self._calculate_news_sentiment(all_news_items)
                logger.info(f"Successfully analyzed sentiment from {len(all_news_items)} news items")
                return sentiment, all_news_items if include_news_items else []
            else:
                # Return neutral sentiment if no news items found
                neutral_sentiment = self._create_neutral_sentiment()
                logger.info("No news items found, returning neutral sentiment")
                return neutral_sentiment, []
                
        except Exception as e:
            logger.error(f"Error in Indian news sentiment analysis for {symbol}: {e}")
            # Return neutral sentiment as fallback
            neutral_sentiment = self._create_neutral_sentiment()
            return neutral_sentiment, []
    
    async def _fetch_from_newsapi(self, company_name: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI"""
        try:
            if not self.news_api_key:
                return []
            
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': company_name,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 20
            }
            
            # Use enhanced scraping techniques
            response = self._make_request(url, use_enhanced=True)
            # response.raise_for_status()  # Already handled in _make_request
            
            data = response.json()
            return data.get('articles', [])
            
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
            return []
    
    def _extract_company_name(self, symbol: str) -> str:
        """Extract company name from symbol"""
        # Extended company name extraction with more Indian companies
        symbol_to_company = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "ICICIBANK.NS": "ICICI Bank",
            "SBIN.NS": "State Bank of India",
            "BHARTIARTL.NS": "Bharti Airtel",
            "ITC.NS": "ITC Limited",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "LT.NS": "Larsen & Toubro",
            "PARAS.NS": "Paras Defence",
            "ADANIPORTS.NS": "Adani Ports",
            "ASIANPAINT.NS": "Asian Paints",
            "AXISBANK.NS": "Axis Bank",
            "BAJAJ-AUTO.NS": "Bajaj Auto",
            "BAJFINANCE.NS": "Bajaj Finance",
            "BPCL.NS": "Bharat Petroleum",
            "CIPLA.NS": "Cipla",
            "COALINDIA.NS": "Coal India",
            "DRREDDY.NS": "Dr Reddy's",
            "EICHERMOT.NS": "Eicher Motors",
            "GAIL.NS": "GAIL",
            "GRASIM.NS": "Grasim Industries",
            "HCLTECH.NS": "HCL Technologies",
            "HEROMOTOCO.NS": "Hero MotoCorp",
            "HINDALCO.NS": "Hindalco Industries",
            "INFRATEL.NS": "Bharti Infratel",
            "IOC.NS": "Indian Oil",
            "INDUSINDBK.NS": "IndusInd Bank",
            "JSWSTEEL.NS": "JSW Steel",
            "KOTAKBANK.NS": "Kotak Mahindra Bank",
            "MARUTI.NS": "Maruti Suzuki",
            "NTPC.NS": "NTPC",
            "ONGC.NS": "ONGC",
            "POWERGRID.NS": "Power Grid",
            "SUNPHARMA.NS": "Sun Pharmaceuticals",
            "TATAMOTORS.NS": "Tata Motors",
            "TATASTEEL.NS": "Tata Steel",
            "TECHM.NS": "Tech Mahindra",
            "ULTRACEMCO.NS": "UltraTech Cement",
            "WIPRO.NS": "Wipro",
            "YESBANK.NS": "Yes Bank",
            "ZEEL.NS": "Zee Entertainment"
        }
        
        return symbol_to_company.get(symbol, symbol.split('.')[0])
    
    def _simple_sentiment_analysis(self, text: str) -> SentimentScore:
        """Simple rule-based sentiment analysis fallback"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'profit', 'growth', 'strong']
        negative_words = ['bad', 'poor', 'negative', 'down', 'fall', 'loss', 'decline', 'weak', 'drop', 'crash']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return self._create_neutral_sentiment()
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        neutral_ratio = 1 - positive_ratio - negative_ratio
        
        compound = positive_ratio - negative_ratio
        
        return SentimentScore(
            positive=positive_ratio,
            negative=negative_ratio,
            neutral=max(0, neutral_ratio),
            compound=compound,
            confidence=0.5
        )
    
    def _calculate_overall_sentiment(self, news_sentiment: Optional[SentimentScore],
                                   social_sentiment: Optional[SentimentScore],
                                   market_sentiment: Optional[SentimentScore],
                                   indian_news_sentiment: Optional[SentimentScore] = None) -> SentimentScore:
        """Calculate weighted overall sentiment with Indian news"""
        sentiments = []
        weights = []
        
        if news_sentiment:
            sentiments.append(news_sentiment)
            weights.append(0.3)  # 30% weight for global news
        
        if social_sentiment:
            sentiments.append(social_sentiment)
            weights.append(0.2)  # 20% weight for social
        
        if market_sentiment:
            sentiments.append(market_sentiment)
            weights.append(0.2)  # 20% weight for market
        
        if indian_news_sentiment:
            sentiments.append(indian_news_sentiment)
            weights.append(0.3)  # 30% weight for Indian news
        
        if not sentiments:
            return self._create_neutral_sentiment()
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        overall_positive = sum(s.positive * w for s, w in zip(sentiments, weights))
        overall_negative = sum(s.negative * w for s, w in zip(sentiments, weights))
        overall_neutral = sum(s.neutral * w for s, w in zip(sentiments, weights))
        overall_compound = sum(s.compound * w for s, w in zip(sentiments, weights))
        overall_confidence = sum(s.confidence * w for s, w in zip(sentiments, weights))
        
        return SentimentScore(
            positive=overall_positive,
            negative=overall_negative,
            neutral=overall_neutral,
            compound=overall_compound,
            confidence=overall_confidence
        )
    
    async def _calculate_sentiment_trend(self, symbol: str, lookback_days: int) -> str:
        """Calculate sentiment trend over time"""
        try:
            # Simplified trend calculation
            # In production, this would analyze historical sentiment data
            
            # Simulate trend based on symbol
            import random
            random.seed(hash(symbol + "trend") % 1000)
            
            trend_value = random.uniform(-1, 1)
            
            if trend_value > 0.3:
                return "IMPROVING"
            elif trend_value < -0.3:
                return "DECLINING"
            else:
                return "STABLE"
                
        except Exception as e:
            logger.error(f"Sentiment trend calculation error: {e}")
            return "STABLE"
    
    def _extract_key_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key themes from news items"""
        try:
            if not news_items:
                return []
            
            # Simple keyword extraction
            all_text = " ".join([item.title + " " + item.content for item in news_items])
            
            # Common financial themes
            themes = []
            theme_keywords = {
                "earnings": ["earnings", "profit", "revenue", "quarterly"],
                "growth": ["growth", "expansion", "increase", "rise"],
                "market": ["market", "trading", "stock", "share"],
                "technology": ["technology", "digital", "innovation", "tech"],
                "regulation": ["regulation", "policy", "government", "compliance"]
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in all_text.lower() for keyword in keywords):
                    themes.append(theme)
            
            return themes[:5]  # Return top 5 themes
            
        except Exception as e:
            logger.error(f"Theme extraction error: {e}")
            return []
    
    def _generate_sentiment_summary(self, result: SentimentAnalysisResult) -> str:
        """Generate human-readable sentiment summary"""
        compound = result.overall_sentiment.compound
        
        if compound > 0.3:
            sentiment_text = "Very Positive"
        elif compound > 0.1:
            sentiment_text = "Positive"
        elif compound > -0.1:
            sentiment_text = "Neutral"
        elif compound > -0.3:
            sentiment_text = "Negative"
        else:
            sentiment_text = "Very Negative"
        
        trend_text = result.sentiment_trend.lower()
        
        return f"{sentiment_text} sentiment detected for {result.symbol}. Sentiment trend is {trend_text}."
    
    def _create_neutral_sentiment(self) -> SentimentScore:
        """Create neutral sentiment score"""
        return SentimentScore(
            positive=0.33,
            negative=0.33,
            neutral=0.34,
            compound=0.0,
            confidence=0.5
        )
    
    def _parse_date(self, date_string: Optional[str]) -> datetime:
        """Parse date string to datetime"""
        if not date_string:
            return datetime.now()
        
        try:
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except:
            return datetime.now()
    
    def _calculate_relevance(self, article: Dict[str, Any], company_name: str) -> float:
        """Calculate relevance score for article"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        company_lower = company_name.lower()
        
        relevance = 0.0
        
        # Check title relevance
        if company_lower in title:
            relevance += 0.5
        
        # Check description relevance
        if company_lower in description:
            relevance += 0.3
        
        # Check for financial keywords
        financial_keywords = ['stock', 'share', 'market', 'trading', 'investment', 'earnings']
        for keyword in financial_keywords:
            if keyword in title or keyword in description:
                relevance += 0.1
                break
        
        return min(relevance, 1.0)
    
    def _get_regime_multiplier(self, sentiment: SentimentScore) -> float:
        """Get regime multiplier based on sentiment score"""
        compound = sentiment.compound
        if compound > 0.3:  # Trending market
            return 1.4
        elif compound < -0.3:  # Volatile market
            return 0.6
        else:  # Range-bound market
            return 1.0
    
    def _apply_regime_multiplier(self, sentiment: SentimentScore, multiplier: float) -> SentimentScore:
        """Apply regime multiplier to sentiment scores"""
        return SentimentScore(
            positive=min(sentiment.positive * multiplier, 1.0),
            negative=min(sentiment.negative * multiplier, 1.0),
            neutral=min(sentiment.neutral * multiplier, 1.0),
            compound=max(-1.0, min(sentiment.compound * multiplier, 1.0)),
            confidence=min(sentiment.confidence * 1.2, 1.0)  # Boost confidence slightly
        )
    
    async def _request_approval(self, symbol: str, sentiment: SentimentScore, session_id: str) -> Dict[str, Any]:
        """Request approval via Grok for approval mode with 30s timeout"""
        try:
            if not self.xai_api_key:
                return {"status": "no_api_key", "message": "xAI API key not configured"}
            
            # In a real implementation, this would call the xAI API
            # For now, we'll simulate the approval process
            logger.info(f"Requesting approval for {symbol} via Grok (session: {session_id})")
            
            # Simulate 30s timeout
            await asyncio.sleep(30)
            
            return {"status": "timeout", "message": "Approval request timed out"}
            
        except Exception as e:
            logger.error(f"Error requesting approval: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _enhance_sentiment_with_llm(self, text: str) -> Optional[SentimentScore]:
        """Enhance sentiment analysis with local LLM (Ollama/Llama)"""
        try:
            # Check if Ollama is available
            if not self.config.get("ollama_enabled", False):
                return None
            
            # Import ollama
            try:
                import ollama
            except ImportError:
                logger.warning("Ollama not available, skipping LLM enhancement")
                return None
            
            # Prepare prompt for sentiment analysis
            prompt = f"""
            Analyze the sentiment of this financial news text:
            "{text[:500]}"  # Limit text length for performance
            
            Provide a sentiment score between -1 (very negative) and 1 (very positive),
            along with confidence between 0 and 1.
            
            Response format:
            sentiment_score: 0.0
            confidence: 0.0
            """
            
            # Get Ollama configuration
            ollama_host = self.config.get("ollama_host", "http://localhost:11434")
            model = self.config.get("ollama_model", "llama2")
            
            # Generate response from LLM
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent results
                    "top_p": 0.9,
                    "stop": ["\n\n"]
                }
            )
            
            # Parse response
            result_text = response.get("response", "")
            lines = result_text.strip().split("\n")
            
            sentiment_score = 0.0
            confidence = 0.5
            
            for line in lines:
                if "sentiment_score:" in line:
                    try:
                        sentiment_score = float(line.split(":")[1].strip())
                    except ValueError:
                        pass
                elif "confidence:" in line:
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except ValueError:
                        pass
            
            # Validate scores
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            confidence = max(0.0, min(1.0, confidence))
            
            # Convert to positive/negative/neutral scores
            positive = max(0, sentiment_score) if sentiment_score > 0 else 0
            negative = max(0, -sentiment_score) if sentiment_score < 0 else 0
            neutral = 1.0 - (positive + negative)
            
            return SentimentScore(
                positive=positive,
                negative=negative,
                neutral=neutral,
                compound=sentiment_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"LLM sentiment enhancement failed: {e}")
            return None
    
    def _initialize_langgraph_agent(self):
        """Initialize LangGraph agent for scraping and analysis"""
        try:
            # Define tools for the agent
            @tool
            def fetch_indian_news(symbol: str) -> str:
                """Fetch Indian news for a symbol"""
                return f"Fetching Indian news for {symbol}"
            
            @tool
            def fetch_global_news(symbol: str) -> str:
                """Fetch global news for a symbol"""
                return f"Fetching global news for {symbol}"
            
            @tool
            def analyze_sentiment(text: str) -> str:
                """Analyze sentiment of text"""
                return f"Analyzing sentiment: {text[:50]}..."
            
            # Create tool executor
            tools = [fetch_indian_news, fetch_global_news, analyze_sentiment]
            tool_executor = ToolExecutor(tools)
            
            # Define the agent state graph
            workflow = StateGraph(AgentState)
            
            # Add nodes and edges (simplified)
            # In a real implementation, this would be more complex
            
            # Compile the workflow
            self.agent_executor = workflow.compile()
            
            logger.info("LangGraph agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangGraph agent: {e}")
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get sentiment tool status"""
        return {
            "tool_id": self.tool_id,
            "analyses_performed": self.analyses_performed,
            "news_items_processed": self.news_items_processed,
            "vader_available": self.vader_available,
            "news_api_configured": bool(self.news_api_key),
            "xai_api_configured": bool(self.xai_api_key),
            "sentiment_sources": self.sentiment_sources,
            "indian_rss_sources_count": len(self.indian_rss_sources),
            "global_rss_sources_count": len(self.global_rss_sources),
            "mode": self.mode,
            "status": "active"
        }
    
    async def _fetch_rss_feed(self, url: str, max_items: int = 10) -> List[NewsItem]:
        """Fetch and parse RSS feed with enhanced error handling and detailed logging"""
        try:
            logger.debug(f"Fetching RSS feed: {url}")
            
            # Make request with error handling
            response = self._make_request(url, use_enhanced=True)
            if response is None:
                logger.debug(f"Failed to fetch RSS feed: {url} - Request returned None")
                return []  # Return empty list for failed requests
            
            # Log response details
            logger.debug(f"Received response from {url} - Status: {response.status_code}, Content-Type: {response.headers.get('content-type', 'unknown')}")
            
            # Handle different response content types
            content = response.content
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        content = content.decode('utf-8', errors='ignore')
                    except:
                        logger.warning(f"Failed to decode content from {url}")
                        return []
            
            # Log content preview for debugging
            content_preview = content[:200] if isinstance(content, str) else str(content)[:200]
            logger.debug(f"Content preview from {url}: {content_preview}")
            
            # Parse RSS feed with better error handling
            try:
                feed = feedparser.parse(content)
                logger.debug(f"Parsed feed from {url} - Entries: {len(getattr(feed, 'entries', []))}")
            except Exception as e:
                logger.warning(f"Failed to parse RSS feed {url}: {e}")
                return []
            
            if feed.bozo and feed.bozo_exception:
                # Log but don't fail on parsing warnings
                logger.debug(f"RSS parsing warning for {url}: {feed.bozo_exception}")
            
            news_items = []
            entries = getattr(feed, 'entries', [])
            
            logger.debug(f"Processing {min(len(entries), max_items)} entries from {url}")
            
            for i, entry in enumerate(entries[:max_items]):
                try:
                    # Log entry details for debugging
                    logger.debug(f"Processing entry {i+1} from {url} - Title: {entry.get('title', 'No title')}")
                    
                    # Extract content with multiple fallbacks
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        # Handle list of content objects
                        if isinstance(entry.content, list) and len(entry.content) > 0:
                            content_obj = entry.content[0]
                            if hasattr(content_obj, 'value'):
                                content = content_obj.value
                            else:
                                content = str(content_obj)
                        else:
                            content = str(entry.content)
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    else:
                        content = str(entry.get('content', ''))
                    
                    # Get published date with multiple fallbacks
                    published_date = datetime.now()
                    date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
                    
                    for field in date_fields:
                        if hasattr(entry, field) and getattr(entry, field):
                            try:
                                date_tuple = getattr(entry, field)
                                if isinstance(date_tuple, (list, tuple)) and len(date_tuple) >= 6:
                                    published_date = datetime(*date_tuple[:6])
                                    logger.debug(f"Using date from {field} field for entry {i+1} from {url}")
                                    break
                            except Exception as date_error:
                                logger.debug(f"Failed to parse date from {field} field: {date_error}")
                                continue
                    
                    # Create news item
                    news_item = NewsItem(
                        title=str(entry.get('title', 'No title')),
                        content=content,
                        source=url,
                        url=str(entry.get('link', '')),
                        published_date=published_date,
                        sentiment_score=self._create_neutral_sentiment(),
                        relevance_score=0.0
                    )
                    news_items.append(news_item)
                    logger.debug(f"Successfully processed entry {i+1} from {url}")
                except Exception as e:
                    logger.warning(f"Error processing RSS entry {i+1} from {url}: {e}")
                    continue
            
            logger.info(f"Successfully fetched {len(news_items)} items from {url}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url}: {e}")
            return []

    def _filter_relevant_news(self, news_items: List[NewsItem], symbol: str, lookback_days: int) -> List[NewsItem]:
        """Filter news items relevant to the symbol within lookback period"""
        try:
            relevant_items = []
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get company names for matching
            symbol_to_company = {
                "RELIANCE.NS": "Reliance",
                "TCS.NS": "TCS",
                "INFY.NS": "Infosys",
                "HDFCBANK.NS": "HDFC Bank",
                "ICICIBANK.NS": "ICICI Bank",
                "AXISBANK.NS": "Axis Bank",
                "SBIN.NS": "SBI",
                "BHARTIARTL.NS": "Bharti Airtel",
                "ITC.NS": "ITC",
                "KOTAKBANK.NS": "Kotak Bank",
                "BAJFINANCE.NS": "Bajaj Finance",
                "HINDUNILVR.NS": "Hindustan Unilever",
                "ASIANPAINT.NS": "Asian Paints",
                "MARUTI.NS": "Maruti Suzuki",
                "TITAN.NS": "Titan",
                "LT.NS": "Larsen & Toubro",
                "DMART.NS": "Avenue Supermarts",
                "SUNPHARMA.NS": "Sun Pharma",
                "CIPLA.NS": "Cipla",
                "DRREDDY.NS": "Dr Reddy's",
                "DIVISLAB.NS": "Divi's Laboratories",
                "NCC.NS": "NCC Limited"  # Add NCC mapping
            }
            
            company_name = symbol_to_company.get(symbol, symbol.split('.')[0])
            search_terms = [symbol.split('.')[0], company_name]
            
            logger.debug(f"Filtering news for {symbol} using search terms: {search_terms}")
            
            for item in news_items:
                try:
                    # Check date
                    if item.published_date < cutoff_date:
                        logger.debug(f"Skipping item (too old): {item.title[:50]}...")
                        continue
                    
                    # Check relevance
                    title_lower = item.title.lower()
                    content_lower = item.content.lower()
                    
                    is_relevant = False
                    matched_term = None
                    for term in search_terms:
                        if term.lower() in title_lower or term.lower() in content_lower:
                            is_relevant = True
                            matched_term = term
                            break
                    
                    if is_relevant:
                        # Calculate relevance score
                        item.relevance_score = self._calculate_relevance_score(item, symbol, company_name)
                        relevant_items.append(item)
                        logger.debug(f"Relevant item found for {symbol} (matched term: {matched_term}): {item.title[:50]}...")
                    else:
                        logger.debug(f"Item not relevant for {symbol}: {item.title[:50]}...")
                        
                except Exception as e:
                    logger.warning(f"Error filtering news item: {e}")
                    continue
            
            logger.info(f"Found {len(relevant_items)} relevant items for {symbol} out of {len(news_items)} total items")
            return relevant_items
            
        except Exception as e:
            logger.error(f"Error in _filter_relevant_news: {e}")
            return []

    def _calculate_relevance_score(self, news_item: NewsItem, symbol: str, company_name: str) -> float:
        """Calculate relevance score for a news item"""
        try:
            title_lower = news_item.title.lower()
            content_lower = news_item.content.lower()
            
            score = 0.0
            
            # Symbol exact match (highest weight)
            if symbol.split('.')[0].lower() in title_lower:
                score += 0.4
            elif symbol.split('.')[0].lower() in content_lower:
                score += 0.2
            
            # Company name match
            if company_name.lower() in title_lower:
                score += 0.3
            elif company_name.lower() in content_lower:
                score += 0.15
            
            # Keyword matching
            keywords = ['stock', 'share', 'market', 'price', 'trading', 'invest']
            for keyword in keywords:
                if keyword in title_lower:
                    score += 0.05
                elif keyword in content_lower:
                    score += 0.02
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating relevance score: {e}")
            return 0.0

    def _get_alternative_rss_sources(self, failing_sources: List[str]) -> List[str]:
        """Get alternative RSS sources for failed ones"""
        # Alternative sources mapping
        alternatives = {
            "https://www.hindubusinessline.com/markets/rssfeed/?id=1016": [
                "https://www.hindubusinessline.com/rss/markets-newsletter/?id=1016",
                "https://www.hindubusinessline.com/rss/markets/?id=1008"
            ],
            "https://timesofindia.indiatimes.com/rssfeeds/2146842.cms": [
                "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms",  # TOI Business
                "https://timesofindia.indiatimes.com/rssfeeds/49694176.cms"  # TOI Economy
            ],
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/12741757.cms": [
                "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/12741757.cms",
                "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/12741757.cms?from=mr"
            ],
            "https://www.reuters.com/news/archive/businessNews": [
                "https://www.reuters.com/news/archive/finance",
                "https://www.reuters.com/news/archive/economicNews"
            ]
        }
        
        alternative_sources = []
        for source in failing_sources:
            if source in alternatives:
                alternative_sources.extend(alternatives[source])
            else:
                # Add some general financial RSS feeds as fallback
                alternative_sources.extend([
                    "https://www.reuters.com/news/archive/finance",
                    "https://www.bloomberg.com/feed/podcast/bloomberg-businessweek.rss",
                    "https://feeds.marketwatch.com/marketwatch/topstories/"
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for source in alternative_sources:
            if source not in seen:
                seen.add(source)
                unique_alternatives.append(source)
        
        return unique_alternatives
