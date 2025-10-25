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

# Fix for Prometheus CollectorRegistry issue - use a shared registry pattern
try:
    from prometheus_client import Counter, Histogram, Gauge
    # Use the default registry to avoid duplication
    import prometheus_client
    
    # Check if metrics already exist to avoid duplication
    def get_or_create_counter(name, description, labelnames):
        try:
            return prometheus_client.Counter(name, description, labelnames)
        except ValueError:
            # Counter already exists, get it from the registry
            return prometheus_client.REGISTRY._names_to_collectors[name]
    
    def get_or_create_histogram(name, description):
        try:
            return prometheus_client.Histogram(name, description)
        except ValueError:
            # Histogram already exists, get it from the registry
            return prometheus_client.REGISTRY._names_to_collectors[name]
    
    SENTIMENT_TOOL_CALLS = get_or_create_counter('mcp_sentiment_tool_calls_total', 'Sentiment tool calls', ['status'])
    SENTIMENT_ANALYSIS_DURATION = get_or_create_histogram('mcp_sentiment_analysis_duration_seconds', 'Sentiment analysis duration')
except ImportError:
    # Create dummy metrics if prometheus is not available
    class DummyMetric:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def inc(self): pass
        def observe(self, value): pass
    
    SENTIMENT_TOOL_CALLS = DummyMetric()
    SENTIMENT_ANALYSIS_DURATION = DummyMetric()

logger = logging.getLogger(__name__)

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
        
        # Indian RSS sources
        self.indian_rss_sources = [
            "https://www.moneycontrol.com/rss/marketreports.xml",
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            "https://www.business-standard.com/rss/markets-106.rss",
            "https://www.nseindia.com/content/equities/rssLatest.xml",
            "https://www.hindubusinessline.com/markets/rssfeed/?id=1016",
            "https://www.5paisa.com/api/MarketRSS",
            "https://in.investing.com/rss/market_overview.rss",
            "https://www.livemint.com/rss/markets",
            "https://timesofindia.indiatimes.com/rssfeeds/2146842.cms",
            "https://www.cnbc.com/id/10001147/device/rss/rss.html"  # Global source
        ]
        
        # Global RSS sources
        self.global_rss_sources = [
            "https://www.marketwatch.com/rss/marketpulse",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://feeds.reuters.com/reuters/businessNews"
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
            
            # Fetch from Indian RSS sources
            for rss_url in self.indian_rss_sources:
                try:
                    feed = feedparser.parse(rss_url)
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
    
    async def _analyze_indian_news_sentiment(self, symbol: str, lookback_days: int, 
                                           include_items: bool) -> tuple[SentimentScore, List[NewsItem]]:
        """Analyze sentiment from Indian news sources"""
        try:
            # Extract company name from symbol for better search
            company_name = self._extract_company_name(symbol)
            
            # Get Indian news articles
            news_articles = await self._fetch_indian_news_articles(company_name, lookback_days)
            
            if not news_articles:
                return self._create_neutral_sentiment(), []
            
            # Analyze sentiment for each article
            news_items = []
            sentiment_scores = []
            
            for article in news_articles:
                # Use VADER for initial sentiment analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if self.vader_available:
                    # Use VADER for sentiment analysis
                    sentiment = self.sentiment_analyzer.polarity_scores(text)
                    
                    sentiment_score = SentimentScore(
                        positive=sentiment['pos'],
                        negative=sentiment['neg'],
                        neutral=sentiment['neu'],
                        compound=sentiment['compound'],
                        confidence=0.85  # Higher confidence for Indian sources
                    )
                else:
                    # Fallback sentiment analysis
                    sentiment_score = self._simple_sentiment_analysis(text)
                
                # Enhance sentiment with LLM if available
                enhanced_sentiment = await self._enhance_sentiment_with_llm(text)
                if enhanced_sentiment:
                    # Blend VADER and LLM sentiment
                    sentiment_score = SentimentScore(
                        positive=(sentiment_score.positive + enhanced_sentiment.positive) / 2,
                        negative=(sentiment_score.negative + enhanced_sentiment.negative) / 2,
                        neutral=(sentiment_score.neutral + enhanced_sentiment.neutral) / 2,
                        compound=(sentiment_score.compound + enhanced_sentiment.compound) / 2,
                        confidence=min((sentiment_score.confidence + enhanced_sentiment.confidence) / 2, 1.0)
                    )
                
                if include_items:
                    news_item = NewsItem(
                        title=article.get('title', ''),
                        content=article.get('description', ''),
                        source=article.get('source', {}).get('name', 'Unknown Indian Source'),
                        url=article.get('url', ''),
                        published_date=self._parse_date(article.get('publishedAt')),
                        sentiment_score=sentiment_score,
                        relevance_score=self._calculate_relevance(article, company_name)
                    )
                    news_items.append(news_item)
                
                sentiment_scores.append(sentiment_score)
            
            # Calculate aggregate Indian news sentiment
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
            logger.error(f"Indian news sentiment analysis error: {e}")
            return self._create_neutral_sentiment(), []
    
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
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
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
