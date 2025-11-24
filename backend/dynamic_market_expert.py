#!/usr/bin/env python3
"""
Dynamic Market Expert - Simplified but Powerful
Professional stock market expert with live Fyers data
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import from main trading bot
try:
    from testindia import GroqLLM
except ImportError:
    GroqLLM = None

# Import Fyers directly
try:
    from fyers_apiv3 import fyersModel
    import os

    class FyersMarketData:
        """Direct Fyers market data integration"""

        def __init__(self):
            self.fyers = None
            self._initialize_fyers()

        def _initialize_fyers(self):
            """Initialize Fyers API"""
            try:
                app_id = os.getenv("FYERS_APP_ID")
                access_token = os.getenv("FYERS_ACCESS_TOKEN")

                if app_id and access_token:
                    self.fyers = fyersModel.FyersModel(
                        client_id=app_id,
                        token=access_token,
                        log_path=""
                    )
                    logger.info("Fyers API initialized successfully")
                else:
                    logger.error("Fyers credentials not found")
            except Exception as e:
                logger.error(f"Error initializing Fyers: {e}")

        def get_live_quotes(self, symbols):
            """Get live quotes for symbols"""
            if not self.fyers:
                return {"error": "Fyers API not available"}

            try:
                # Convert symbols to Fyers format if needed
                fyers_symbols = []
                for symbol in symbols:
                    if not symbol.startswith("NSE:"):
                        fyers_symbols.append(
                            f"NSE:{symbol.replace('.NS', '')}-EQ")
                    else:
                        fyers_symbols.append(symbol)

                response = self.fyers.quotes(
                    {"symbols": ",".join(fyers_symbols)})
                return response
            except Exception as e:
                logger.error(f"Error getting quotes: {e}")
                return {"error": str(e)}

except ImportError:
    logger.error("Fyers API not available")
    FyersMarketData = None

logger = logging.getLogger(__name__)


class DynamicMarketExpert:
    """
    Professional Stock Market Expert with Live Data Integration
    Simplified but powerful - no complex graphs, just intelligent responses
    """

    def __init__(self):
        """Initialize the market expert"""

        # Initialize core components
        if FyersMarketData:
            self.fyers_data = FyersMarketData()
        else:
            self.fyers_data = None

        # Initialize Groq LLM if API key is available
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            try:
                from testindia import GroqLLM
                self.llm = GroqLLM(model_name="llama-3.1-8b-instant",
                                   api_key=groq_api_key)
                logger.info("Groq LLM initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Groq LLM: {e}")
                self.llm = None
        else:
            self.llm = None

        # Stock symbol mapping
        self.symbol_map = {
            'reliance': 'RELIANCE',
            'tcs': 'TCS',
            'hdfc': 'HDFCBANK',
            'hdfc bank': 'HDFCBANK',
            'infosys': 'INFY',
            'icici': 'ICICIBANK',
            'icici bank': 'ICICIBANK',
            'sbi': 'SBIN',
            'state bank': 'SBIN',
            'bharti': 'BHARTIARTL',
            'airtel': 'BHARTIARTL',
            'bharti airtel': 'BHARTIARTL',
            'itc': 'ITC',
            'wipro': 'WIPRO',
            'tatasteel': 'TATASTEEL',
            'tata steel': 'TATASTEEL',
            'adani': 'ADANIPORTS',
            'bajaj': 'BAJFINANCE',
            'maruti': 'MARUTI',
            'asian paints': 'ASIANPAINT',
            'hindustan unilever': 'HINDUNILVR',
            'hul': 'HINDUNILVR',
            'lt': 'LT',
            'larsen': 'LT',
            'ongc': 'ONGC',
            'ntpc': 'NTPC',
            'powergrid': 'POWERGRID',
            'coal india': 'COALINDIA',
            'sun pharma': 'SUNPHARMA',
            'dr reddy': 'DRREDDY'
        }

        logger.info("Dynamic Market Expert initialized successfully")

    def extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from user query"""

        symbols = []
        query_lower = query.lower()

        # Check for exact matches
        for keyword, symbol in self.symbol_map.items():
            if keyword in query_lower:
                if symbol not in symbols:
                    symbols.append(symbol)

        # If no symbols found but query seems stock-related, use popular ones
        if not symbols:
            stock_keywords = ['stock', 'share', 'price',
                              'market', 'trading', 'invest', 'buy', 'sell']
            if any(keyword in query_lower for keyword in stock_keywords):
                # Return popular stocks for general market queries
                symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']

        return symbols

    def get_live_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get live market data for symbols with Fyers primary and yfinance fallback"""

        if not symbols:
            return {}

        market_data = {}

        # Try Fyers first if available
        if self.fyers_data and self.fyers_data.fyers:
            try:
                # Convert to Fyers format
                fyers_symbols = [f"{symbol}.NS" for symbol in symbols]

                # Fetch live data
                quotes_response = self.fyers_data.get_live_quotes(
                    fyers_symbols)

                if quotes_response.get('code') == 200:
                    for quote in quotes_response.get('d', []):
                        symbol_name = quote.get('n', '').replace(
                            'NSE:', '').replace('-EQ', '')
                        price_data = quote.get('v', {})

                        market_data[symbol_name] = {
                            'price': price_data.get('lp', 0),
                            'change': price_data.get('ch', 0),
                            'change_pct': price_data.get('chp', 0),
                            'volume': price_data.get('volume', 0),
                            'high': price_data.get('h', 0),
                            'low': price_data.get('l', 0),
                            'open': price_data.get('o', 0)
                        }

                    if market_data:
                        logger.info(
                            f"[SUCCESS] Fetched live data from Fyers for {len(market_data)} symbols")
                        return market_data
                else:
                    logger.warning(f"Fyers API error: {quotes_response}")

            except Exception as e:
                logger.warning(f"Fyers API failed: {e}")

        # Fallback to yfinance if Fyers fails or unavailable
        logger.info("🔄 Falling back to yfinance for live market data")
        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    # Convert symbol to yfinance format
                    yf_symbol = f"{symbol}.NS" if not symbol.endswith(
                        '.NS') else symbol

                    ticker = yf.Ticker(yf_symbol)

                    # Get current data
                    info = ticker.info
                    hist = ticker.history(period="2d", interval="1d")

                    if not hist.empty and info:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(
                            hist) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close *
                                      100) if prev_close != 0 else 0

                        market_data[symbol] = {
                            'price': float(current_price),
                            'change': float(change),
                            'change_pct': float(change_pct),
                            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                            'high': float(hist['High'].iloc[-1]),
                            'low': float(hist['Low'].iloc[-1]),
                            'open': float(hist['Open'].iloc[-1])
                        }

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue

            if market_data:
                logger.info(
                    f"[SUCCESS] Fetched live data from yfinance for {len(market_data)} symbols")
            else:
                logger.warning(
                    "[ERROR] No market data available from any source")

        except Exception as e:
            logger.error(f"yfinance fallback failed: {e}")

        return market_data

    def create_market_summary(self, market_data: Dict[str, Any]) -> str:
        """Create a formatted market data summary"""

        if not market_data:
            return "No live market data available at the moment."

        summary = "**Live Market Data:**\n"
        for symbol, data in market_data.items():
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"
            summary += f"{change_emoji} **{symbol}**: Rs.{data['price']:.2f} ({data['change']:+.2f}, {data['change_pct']:+.2f}%)\n"
            summary += f"   High: Rs.{data['high']:.2f} | Low: Rs.{data['low']:.2f} | Volume: {data['volume']:,}\n"

        return summary

    def generate_professional_analysis(self, query: str, market_data: Dict[str, Any]) -> str:
        """Generate professional market analysis - ENHANCED REAL-TIME VERSION"""

        # Use enhanced analysis for better insights
        return self.generate_enhanced_analysis(query, market_data)

    def create_intelligent_analysis(self, query: str, market_data: Dict[str, Any]) -> str:
        """Create intelligent analysis without LLM delays"""

        if not market_data:
            return "I'm your professional stock market advisor! Ask me about specific Indian stocks like Reliance, TCS, HDFC Bank, Infosys, etc. for live prices and expert analysis."

        # Analyze query intent
        query_lower = query.lower()
        is_price_query = any(word in query_lower for word in [
                             'price', 'cost', 'value', 'trading at'])
        is_performance_query = any(word in query_lower for word in [
                                   'performing', 'performance', 'doing', 'movement'])
        is_buy_query = any(word in query_lower for word in [
                           'buy', 'invest', 'purchase', 'should i'])
        is_comparison = any(word in query_lower for word in [
                            'vs', 'versus', 'compare', 'between'])

        # Create dynamic response based on query type and live data
        response = "📊 **Professional Market Analysis**\n\n"

        # Add live data summary
        total_stocks = len(market_data)
        positive_stocks = sum(
            1 for data in market_data.values() if data['change'] >= 0)
        avg_change = sum(data['change_pct']
                         for data in market_data.values()) / total_stocks

        # Market sentiment analysis
        if avg_change > 2:
            market_sentiment = "strong bullish momentum"
            advice = "Consider profit booking on existing positions"
        elif avg_change > 0.5:
            market_sentiment = "positive momentum"
            advice = "Good time for selective buying"
        elif avg_change > -0.5:
            market_sentiment = "sideways movement"
            advice = "Wait and watch approach recommended"
        elif avg_change > -2:
            market_sentiment = "mild weakness"
            advice = "Look for better entry points"
        else:
            market_sentiment = "significant weakness"
            advice = "Avoid fresh buying, consider stop-losses"

        # Query-specific responses
        if is_price_query:
            response += f"**Current Prices & Analysis:**\n"
            for symbol, data in market_data.items():
                change_emoji = "[+]" if data['change'] >= 0 else "[-]"
                trend = "uptrend" if data['change'] > 0 else "downtrend" if data['change'] < 0 else "flat"
                response += f"{change_emoji} **{symbol}**: Rs.{data['price']:.2f} ({data['change_pct']:+.2f}%) - Currently in {trend}\n"

        elif is_performance_query:
            response += f"**Performance Analysis:**\n"
            response += f"Market showing {market_sentiment} with {positive_stocks}/{total_stocks} stocks in green.\n\n"

            for symbol, data in market_data.items():
                if abs(data['change_pct']) > 2:
                    performance = "strong performance"
                elif abs(data['change_pct']) > 1:
                    performance = "moderate movement"
                else:
                    performance = "stable trading"

                change_emoji = "[+]" if data['change'] >= 0 else "[-]"
                response += f"{change_emoji} **{symbol}**: {performance} at Rs.{data['price']:.2f} with {data['volume']:,} volume\n"

        elif is_buy_query:
            response += f"**Investment Recommendation:**\n"
            response += f"Current market shows {market_sentiment}. {advice}.\n\n"

            for symbol, data in market_data.items():
                change_emoji = "[+]" if data['change'] >= 0 else "[-]"

                if data['change_pct'] > 3:
                    recommendation = "Overbought - wait for correction"
                elif data['change_pct'] > 1:
                    recommendation = "Positive momentum - can consider"
                elif data['change_pct'] > -1:
                    recommendation = "Neutral zone - selective approach"
                elif data['change_pct'] > -3:
                    recommendation = "Weakness - wait for stability"
                else:
                    recommendation = "Oversold - potential opportunity"

                response += f"{change_emoji} **{symbol}** at ₹{data['price']:.2f}: {recommendation}\n"

        elif is_comparison:
            response += f"**Comparative Analysis:**\n"
            sorted_stocks = sorted(
                market_data.items(), key=lambda x: x[1]['change_pct'], reverse=True)

            for i, (symbol, data) in enumerate(sorted_stocks):
                rank = "#1" if i == 0 else "#2" if i == 1 else "#3" if i == 2 else "#"
                change_emoji = "[+]" if data['change'] >= 0 else "[-]"
                response += f"{rank} **{symbol}**: Rs.{data['price']:.2f} ({data['change_pct']:+.2f}%) {change_emoji}\n"

        else:
            # General analysis
            response += f"**Market Overview:**\n"
            response += f"Showing {market_sentiment} with average change of {avg_change:+.2f}%.\n\n"

            for symbol, data in market_data.items():
                change_emoji = "[+]" if data['change'] >= 0 else "[-]"
                response += f"{change_emoji} **{symbol}**: Rs.{data['price']:.2f} ({data['change_pct']:+.2f}%) | Vol: {data['volume']:,}\n"

        # Add professional insight
        response += f"\n💡 **Professional Insight:** {advice}. "

        if any(data['volume'] > 5000000 for data in market_data.values()):
            response += "High volume indicates strong institutional interest."
        else:
            response += "Moderate volume suggests cautious market participation."

        return response

    def create_fallback_analysis(self, query: str, market_data: Dict[str, Any]) -> str:
        """Create intelligent fallback analysis"""

        if not market_data:
            return """I'm your professional stock market expert! 📊

**Available Services:**
🔍 **Live Stock Analysis** - Ask about specific Indian stocks like Reliance, TCS, HDFC Bank, Infosys
💹 **Market Insights** - Get professional analysis on market trends and movements
📈 **Technical Analysis** - Price patterns, support/resistance levels, indicators
💼 **Investment Advice** - Portfolio strategy and risk management

**Try asking:** "What's the current price of Reliance?" or "How is TCS performing today?"

*Note: Using yfinance data as Fyers token needs refresh*"""

        # Analyze the data
        total_stocks = len(market_data)
        positive_stocks = sum(
            1 for data in market_data.values() if data['change'] >= 0)
        negative_stocks = total_stocks - positive_stocks

        avg_change = sum(data['change_pct']
                         for data in market_data.values()) / total_stocks

        # Create intelligent response
        response = f"📊 **Professional Market Analysis for your query:**\n\n"

        # Market sentiment
        if avg_change > 1:
            sentiment = "positive momentum"
        elif avg_change < -1:
            sentiment = "bearish pressure"
        else:
            sentiment = "mixed sentiment"

        response += f"**Market Overview:** {positive_stocks}/{total_stocks} stocks showing gains, indicating {sentiment} in your selected stocks.\n\n"

        # Individual stock insights
        for symbol, data in market_data.items():
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"

            if abs(data['change_pct']) > 2:
                movement = "significant movement"
            elif abs(data['change_pct']) > 1:
                movement = "moderate movement"
            else:
                movement = "stable trading"

            response += f"{change_emoji} **{symbol}** at ₹{data['price']:.2f} ({data['change_pct']:+.2f}%) - {movement} with volume of {data['volume']:,} shares.\n"

        # Professional advice
        response += f"\n💡 **Professional Insight:** "
        if avg_change > 2:
            response += "Strong bullish momentum. Consider profit booking on existing positions."
        elif avg_change < -2:
            response += "Weakness observed. Wait for better entry points or consider stop-losses."
        else:
            response += "Consolidation phase. Good time for selective stock picking based on fundamentals."

        return response

    def process_query(self, user_query: str) -> str:
        """Process user query with enhanced real-time analysis"""

        try:
            query_lower = user_query.lower()

            # Enhanced symbol selection based on query intelligence
            symbols = self.extract_symbols(user_query)

            # If no specific symbols, intelligently select based on query type
            if not symbols:
                if any(word in query_lower for word in ['penny', 'cheap', 'low price', 'under 100', 'under 50']):
                    symbols = self.get_penny_stocks()
                elif any(word in query_lower for word in ['top 5', 'best 5', 'top five', 'best five']):
                    symbols = self.get_top_performers()
                elif any(word in query_lower for word in ['high volume', 'most traded', 'active']):
                    symbols = self.get_high_volume_stocks()
                elif any(word in query_lower for word in ['tech', 'it', 'software', 'technology']):
                    symbols = ["TCS.NS", "INFY.NS",
                               "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
                elif any(word in query_lower for word in ['bank', 'finance', 'financial']):
                    symbols = ["HDFCBANK.NS", "ICICIBANK.NS",
                               "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
                elif any(word in query_lower for word in ['pharma', 'healthcare', 'medical']):
                    symbols = ["SUNPHARMA.NS", "DRREDDY.NS",
                               "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"]
                else:
                    # Default diverse portfolio for general queries
                    symbols = ["RELIANCE.NS", "TCS.NS",
                               "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

            # Get live market data
            market_data = self.get_live_market_data(symbols)

            # Generate enhanced professional analysis
            if market_data:
                response = self.generate_enhanced_analysis(
                    user_query, market_data)
            else:
                # Fallback with intelligent response
                response = self.generate_fallback_response(user_query)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize for the technical difficulty. As your market expert, I'm here to help with stock analysis and investment advice. Please try asking about specific Indian stocks like Reliance, TCS, or HDFC Bank."

    def get_penny_stocks(self) -> List[str]:
        """Get list of penny stocks (under Rs. 100)"""
        return [
            "SUZLON.NS", "YESBANK.NS", "VODAFONE.NS", "JPASSOCIAT.NS",
            "RPOWER.NS", "JETAIRWAYS.NS", "RCOM.NS", "SAIL.NS",
            "NMDC.NS", "COALINDIA.NS"
        ]

    def get_top_performers(self) -> List[str]:
        """Get top performing stocks"""
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
            "BHARTIARTL.NS", "ITC.NS", "LT.NS"
        ]

    def get_high_volume_stocks(self) -> List[str]:
        """Get high volume trading stocks"""
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "SBIN.NS", "BHARTIARTL.NS", "INFY.NS", "ITC.NS",
            "HINDUNILVR.NS", "KOTAKBANK.NS"
        ]

    def generate_enhanced_analysis(self, query: str, market_data: Dict[str, Any]) -> str:
        """Generate enhanced real-time analysis with actionable insights"""

        query_lower = query.lower()

        # Real-time market overview
        response = f"🔴 **Live Market Overview** (as of {datetime.now().strftime('%H:%M PM')})\n\n"

        # Calculate market sentiment
        positive_stocks = sum(
            1 for data in market_data.values() if data['change'] > 0)
        total_stocks = len(market_data)
        sentiment_pct = (positive_stocks / total_stocks) * \
            100 if total_stocks > 0 else 0

        if sentiment_pct > 60:
            sentiment = "Positive"
            sentiment_desc = "Strong bullish momentum"
        elif sentiment_pct > 40:
            sentiment = "Neutral"
            sentiment_desc = "Mixed signals, selective approach"
        else:
            sentiment = "Negative"
            sentiment_desc = "Bearish pressure, defensive strategy"

        response += f"**Market Sentiment:** {sentiment} with average change of {sentiment_desc}\n\n"

        # Specific analysis based on query type
        if any(word in query_lower for word in ['penny', 'cheap', 'low price']):
            response += self.analyze_penny_stocks(market_data)
        elif any(word in query_lower for word in ['top 5', 'best 5', 'buy']):
            response += self.analyze_top_picks(market_data)
        elif any(word in query_lower for word in ['analysis', 'today', 'market']):
            response += self.analyze_market_overview(market_data)
        else:
            response += self.analyze_specific_stocks(market_data)

        # Add professional insight
        response += f"\n💡 **Professional Insight:** {self.get_market_insight(market_data)}"

        return response

    def analyze_penny_stocks(self, market_data: Dict[str, Any]) -> str:
        """Analyze penny stocks with risk assessment"""

        analysis = "**🎯 Penny Stock Analysis:**\n"

        penny_picks = []
        for symbol, data in market_data.items():
            if data['price'] < 100:  # Penny stock threshold
                risk_level = "High" if data['price'] < 50 else "Medium"
                momentum = "Positive" if data['change'] > 0 else "Negative"

                penny_picks.append({
                    'symbol': symbol,
                    'price': data['price'],
                    'change': data['change'],
                    'change_pct': data['change_pct'],
                    'risk': risk_level,
                    'momentum': momentum
                })

        if penny_picks:
            # Sort by performance
            penny_picks.sort(key=lambda x: x['change_pct'], reverse=True)

            for i, stock in enumerate(penny_picks[:5], 1):
                change_emoji = "[+]" if stock['change'] >= 0 else "[-]"
                analysis += f"{change_emoji} **{stock['symbol']}** at ₹{stock['price']:.2f} ({stock['change']:+.2f}%)\n"
                analysis += f"   Risk: {stock['risk']} | Momentum: {stock['momentum']}\n"
        else:
            analysis += "No penny stocks in current selection. Consider SUZLON, YESBANK for penny stock exposure.\n"

        analysis += "\n⚠️ **Risk Warning:** Penny stocks are highly volatile. Only invest what you can afford to lose.\n"

        return analysis

    def analyze_top_picks(self, market_data: Dict[str, Any]) -> str:
        """Analyze top stock picks with buy recommendations"""

        analysis = "**🏆 Top Investment Recommendations:**\n"

        # Sort stocks by performance and fundamentals
        stock_scores = []
        for symbol, data in market_data.items():
            # Simple scoring algorithm
            price_score = min(data['change_pct'] * 2, 10)  # Performance weight
            volume_score = min(data['volume'] / 1000000, 5)  # Volume weight
            total_score = price_score + volume_score

            stock_scores.append({
                'symbol': symbol,
                'score': total_score,
                'data': data
            })

        # Sort by score
        stock_scores.sort(key=lambda x: x['score'], reverse=True)

        for i, stock in enumerate(stock_scores[:5], 1):
            data = stock['data']
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"

            # Recommendation logic
            if data['change_pct'] > 2:
                recommendation = "Strong Buy"
            elif data['change_pct'] > 0:
                recommendation = "Buy"
            elif data['change_pct'] > -2:
                recommendation = "Hold"
            else:
                recommendation = "Wait"

            analysis += f"{change_emoji} **{stock['symbol']}** at ₹{data['price']:.2f} ({data['change']:+.2f}%)\n"
            analysis += f"   Recommendation: {recommendation} | Volume: {data['volume']:,}\n"

        return analysis

    def analyze_market_overview(self, market_data: Dict[str, Any]) -> str:
        """Provide comprehensive market overview"""

        analysis = "**📊 Live Market Analysis:**\n"

        # Market statistics
        total_stocks = len(market_data)
        gainers = sum(1 for data in market_data.values() if data['change'] > 0)
        losers = total_stocks - gainers

        avg_change = sum(data['change_pct']
                         for data in market_data.values()) / total_stocks
        total_volume = sum(data['volume'] for data in market_data.values())

        analysis += f"**Market Statistics:**\n"
        analysis += f"• Gainers: {gainers} | Losers: {losers}\n"
        analysis += f"• Average Change: {avg_change:+.2f}%\n"
        analysis += f"• Total Volume: {total_volume:,}\n\n"

        # Top performers
        sorted_stocks = sorted(market_data.items(),
                               key=lambda x: x[1]['change_pct'], reverse=True)

        analysis += "**Top Performers:**\n"
        for symbol, data in sorted_stocks[:3]:
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"
            analysis += f"{change_emoji} **{symbol}**: ₹{data['price']:.2f} ({data['change']:+.2f}%)\n"

        analysis += "\n**Underperformers:**\n"
        for symbol, data in sorted_stocks[-3:]:
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"
            analysis += f"{change_emoji} **{symbol}**: ₹{data['price']:.2f} ({data['change']:+.2f}%)\n"

        return analysis

    def analyze_specific_stocks(self, market_data: Dict[str, Any]) -> str:
        """Analyze specific stocks with detailed insights"""

        analysis = "**🔍 Stock Analysis:**\n"

        for symbol, data in market_data.items():
            change_emoji = "[+]" if data['change'] >= 0 else "[-]"

            # Technical analysis
            if data['price'] > data['high'] * 0.95:
                technical = "Near day high - momentum strong"
            elif data['price'] < data['low'] * 1.05:
                technical = "Near day low - potential support"
            else:
                technical = "Trading in range - watch for breakout"

            analysis += f"{change_emoji} **{symbol}**: ₹{data['price']:.2f} ({data['change']:+.2f}%)\n"
            analysis += f"   Range: ₹{data['low']:.2f} - ₹{data['high']:.2f} | {technical}\n"

        return analysis

    def get_market_insight(self, market_data: Dict[str, Any]) -> str:
        """Generate professional market insight"""

        avg_change = sum(data['change_pct']
                         for data in market_data.values()) / len(market_data)
        high_volume_stocks = sum(
            1 for data in market_data.values() if data['volume'] > 1000000)

        if avg_change > 1:
            return "Strong bullish momentum across sectors. Consider increasing equity exposure with proper risk management."
        elif avg_change > 0:
            return "Moderate positive sentiment. Selective stock picking recommended with focus on fundamentals."
        elif avg_change > -1:
            return "Mixed market conditions. Maintain balanced portfolio and wait for clearer direction."
        else:
            return "Bearish pressure evident. Consider defensive stocks and maintain higher cash allocation."

    def generate_fallback_response(self, query: str) -> str:
        """Generate intelligent fallback response when live data unavailable"""

        query_lower = query.lower()

        response = f"🤖 **Market Expert Analysis** (Live data temporarily unavailable)\n\n"
        response += f"**Your Query:** {query}\n\n"

        if any(word in query_lower for word in ['penny', 'cheap']):
            response += "**Penny Stock Strategy:**\n"
            response += "• Focus on companies with strong fundamentals\n"
            response += "• Look for stocks under ₹100 with good volume\n"
            response += "• Consider SUZLON, SAIL, NMDC for penny exposure\n"
            response += "• ⚠️ High risk - only invest spare money\n"

        elif any(word in query_lower for word in ['top 5', 'best']):
            response += "**Top Stock Recommendations:**\n"
            response += "• Large Cap: RELIANCE, TCS, HDFCBANK\n"
            response += "• IT Sector: INFY, WIPRO, TECHM\n"
            response += "• Banking: ICICIBANK, KOTAKBANK, SBIN\n"
            response += "• Diversified portfolio approach recommended\n"

        else:
            response += "**General Market Guidance:**\n"
            response += "• Focus on fundamentally strong companies\n"
            response += "• Maintain proper risk management\n"
            response += "• Consider sector diversification\n"
            response += "• Monitor market trends and news\n"

        response += "\n💡 **Tip:** Ask about specific stocks like 'Reliance price' or 'TCS analysis' for live data!"

        return response

# Test function


def test_expert():
    """Test the dynamic market expert"""

    expert = DynamicMarketExpert()

    test_queries = [
        "What is the price of Tata Steel?",
        "How is Reliance performing?",
        "Should I buy TCS stock?",
        "Compare HDFC Bank vs ICICI Bank"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = expert.process_query(query)
        print(f"Response: {response[:300]}...")


if __name__ == "__main__":
    test_expert()
