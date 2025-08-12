import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CorrelationManager:
    """Manages position sizing based on portfolio correlations"""
    
    def __init__(self, lookback_period: int = 30):
        self.lookback_period = lookback_period
        self.correlation_threshold = 0.7  # High correlation threshold
        self.correlation_cache = {}
        self.price_history = {}
        
    def update_price_history(self, ticker: str, prices: pd.Series):
        """Update price history for a ticker"""
        self.price_history[ticker] = prices.tail(self.lookback_period)
        self._update_correlation_cache()
        
    def get_correlation_adjusted_size(self,
                                    ticker: str,
                                    base_position_size: float,
                                    current_portfolio: Dict[str, Dict]) -> float:
        """
        Adjust position size based on correlations with existing portfolio
        """
        try:
            if not self.correlation_cache or ticker not in self.correlation_cache:
                return base_position_size
                
            # Get correlations with current holdings
            portfolio_correlations = []
            for holding_ticker in current_portfolio:
                if holding_ticker in self.correlation_cache[ticker]:
                    portfolio_correlations.append(
                        abs(self.correlation_cache[ticker][holding_ticker])
                    )
            
            if not portfolio_correlations:
                return base_position_size
                
            # Calculate correlation-based adjustment
            avg_correlation = np.mean(portfolio_correlations)
            max_correlation = max(portfolio_correlations)
            
            # Reduce position size based on correlations
            if max_correlation > self.correlation_threshold:
                # Significant correlation found, reduce position size
                reduction_factor = 1 - (max_correlation - self.correlation_threshold)
                adjusted_size = base_position_size * reduction_factor
                
                logger.info(f"Correlation adjustment for {ticker}: "
                          f"factor={reduction_factor:.2f}, "
                          f"avg_corr={avg_correlation:.2f}, "
                          f"max_corr={max_correlation:.2f}")
                
                return max(adjusted_size, base_position_size * 0.3)  # Floor at 30%
            
            return base_position_size
            
        except Exception as e:
            logger.error(f"Error in correlation adjustment: {e}")
            return base_position_size
            
    def _update_correlation_cache(self):
        """Update correlation cache for all tickers"""
        try:
            tickers = list(self.price_history.keys())
            if len(tickers) < 2:
                return
                
            returns_data = {}
            for ticker in tickers:
                returns_data[ticker] = self.price_history[ticker].pct_change().dropna()
                
            for i, ticker1 in enumerate(tickers):
                if ticker1 not in self.correlation_cache:
                    self.correlation_cache[ticker1] = {}
                    
                for ticker2 in tickers[i+1:]:
                    if len(returns_data[ticker1]) == len(returns_data[ticker2]):
                        corr = returns_data[ticker1].corr(returns_data[ticker2])
                        self.correlation_cache[ticker1][ticker2] = corr
                        
                        if ticker2 not in self.correlation_cache:
                            self.correlation_cache[ticker2] = {}
                        self.correlation_cache[ticker2][ticker1] = corr
                        
        except Exception as e:
            logger.error(f"Error updating correlation cache: {e}")
            
    def get_portfolio_diversification_score(self, portfolio: Dict[str, Dict]) -> float:
        """Calculate overall portfolio diversification score"""
        try:
            if len(portfolio) < 2:
                return 1.0
                
            correlations = []
            tickers = list(portfolio.keys())
            
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:
                    if (ticker1 in self.correlation_cache and 
                        ticker2 in self.correlation_cache[ticker1]):
                        correlations.append(abs(self.correlation_cache[ticker1][ticker2]))
                        
            if not correlations:
                return 1.0
                
            # Convert to diversification score (1 - avg correlation)
            avg_correlation = np.mean(correlations)
            return 1 - avg_correlation
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 1.0
