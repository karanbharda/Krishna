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

# Import from main trading bot
try:
    from testindia import LlamaLLM
except ImportError:
    LlamaLLM = None

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
                        fyers_symbols.append(f"NSE:{symbol.replace('.NS', '')}-EQ")
                    else:
                        fyers_symbols.append(symbol)

                response = self.fyers.quotes({"symbols": ",".join(fyers_symbols)})
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

        if LlamaLLM:
            try:
                self.llm = LlamaLLM(model_name="llama3.2:latest")
            except Exception as e:
                logger.error(f"Error initializing LLM: {e}")
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
            stock_keywords = ['stock', 'share', 'price', 'market', 'trading', 'invest', 'buy', 'sell']
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
                quotes_response = self.fyers_data.get_live_quotes(fyers_symbols)

                if quotes_response.get('code') == 200:
                    for quote in quotes_response.get('d', []):
                        symbol_name = quote.get('n', '').replace('NSE:', '').replace('-EQ', '')
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
                        logger.info(f"[SUCCESS] Fetched live data from Fyers for {len(market_data)} symbols")
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
                    yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol

                    ticker = yf.Ticker(yf_symbol)

                    # Get current data
                    info = ticker.info
                    hist = ticker.history(period="2d", interval="1d")

                    if not hist.empty and info:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_close
                        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

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
                logger.info(f"[SUCCESS] Fetched live data from yfinance for {len(market_data)} symbols")
            else:
                logger.warning("[ERROR] No market data available from any source")

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
        """Generate professional market analysis - FAST VERSION"""

        # Skip LLM for speed - use intelligent structured response
        return self.create_intelligent_analysis(query, market_data)
    
    def create_intelligent_analysis(self, query: str, market_data: Dict[str, Any]) -> str:
        """Create intelligent analysis without LLM delays"""

        if not market_data:
            return "I'm your professional stock market advisor! Ask me about specific Indian stocks like Reliance, TCS, HDFC Bank, Infosys, etc. for live prices and expert analysis."

        # Analyze query intent
        query_lower = query.lower()
        is_price_query = any(word in query_lower for word in ['price', 'cost', 'value', 'trading at'])
        is_performance_query = any(word in query_lower for word in ['performing', 'performance', 'doing', 'movement'])
        is_buy_query = any(word in query_lower for word in ['buy', 'invest', 'purchase', 'should i'])
        is_comparison = any(word in query_lower for word in ['vs', 'versus', 'compare', 'between'])

        # Create dynamic response based on query type and live data
        response = "📊 **Professional Market Analysis**\n\n"

        # Add live data summary
        total_stocks = len(market_data)
        positive_stocks = sum(1 for data in market_data.values() if data['change'] >= 0)
        avg_change = sum(data['change_pct'] for data in market_data.values()) / total_stocks

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
            sorted_stocks = sorted(market_data.items(), key=lambda x: x[1]['change_pct'], reverse=True)

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
        positive_stocks = sum(1 for data in market_data.values() if data['change'] >= 0)
        negative_stocks = total_stocks - positive_stocks
        
        avg_change = sum(data['change_pct'] for data in market_data.values()) / total_stocks
        
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
        """Process user query and return professional response"""
        
        try:
            # Extract symbols from query
            symbols = self.extract_symbols(user_query)
            
            # Get live market data
            market_data = self.get_live_market_data(symbols)
            
            # Generate professional analysis
            if market_data:
                response = self.generate_professional_analysis(user_query, market_data)
            else:
                # No live data available
                response = f"""I understand you're asking about: "{user_query}"

I'm your professional stock market expert, but I'm currently unable to fetch live market data. 

I can help you with:
📊 **Live Stock Analysis** - Ask about specific stocks like Reliance, TCS, HDFC Bank
💼 **Investment Advice** - Get professional insights on market trends
📈 **Technical Analysis** - Understand price movements and patterns
🎯 **Portfolio Strategy** - Risk management and allocation advice

Please try asking about a specific stock, and I'll provide live data and professional analysis!"""
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize for the technical difficulty. As your market expert, I'm here to help with stock analysis and investment advice. Please try asking about specific Indian stocks like Reliance, TCS, or HDFC Bank."

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
