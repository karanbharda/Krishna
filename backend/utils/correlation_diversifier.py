"""
Phase 3: Correlation-Based Diversification System
Implements intelligent portfolio balancing through correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CorrelationDiversifier:
    """
    Manages portfolio diversification using correlation analysis and optimization
    """
    
    def __init__(self, max_correlation: float = 0.7, min_positions: int = 3):
        self.max_correlation = max_correlation
        self.min_positions = min_positions
        self.correlation_matrix = None
        self.sector_mapping = {}
        self.diversification_scores = {}
        
        # Risk budgeting parameters
        self.max_sector_allocation = 0.30  # 30% max per sector
        self.target_portfolio_vol = 0.15   # 15% target volatility
        self.rebalance_threshold = 0.05    # 5% threshold for rebalancing
        
        logger.info("✅ Correlation Diversifier initialized")
    
    def analyze_portfolio_correlation(self, holdings: Dict, price_data: Dict) -> Dict:
        """Analyze correlation structure of current portfolio"""
        try:
            if not holdings or len(holdings) < 2:
                return {'diversification_score': 1.0, 'correlations': {}, 'recommendations': []}
            
            # Calculate correlation matrix
            symbols = list(holdings.keys())
            correlation_matrix = self._calculate_correlation_matrix(symbols, price_data)
            
            if correlation_matrix is None:
                return {'diversification_score': 0.5, 'correlations': {}, 'recommendations': []}
            
            # Calculate diversification metrics
            diversification_score = self._calculate_diversification_score(correlation_matrix, holdings)
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            max_correlation = self._find_max_correlation(correlation_matrix)
            
            # Identify problematic correlations
            high_corr_pairs = self._find_high_correlation_pairs(correlation_matrix, symbols)
            
            # Generate recommendations
            recommendations = self._generate_diversification_recommendations(
                correlation_matrix, holdings, symbols, high_corr_pairs
            )
            
            result = {
                'diversification_score': diversification_score,
                'average_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'high_correlation_pairs': high_corr_pairs,
                'correlation_matrix': correlation_matrix.to_dict() if correlation_matrix is not None else {},
                'recommendations': recommendations,
                'sector_allocation': self._analyze_sector_allocation(holdings),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Portfolio correlation analysis: Score={diversification_score:.3f}, Avg Corr={avg_correlation:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio correlation: {e}")
            return {'diversification_score': 0.0, 'correlations': {}, 'recommendations': []}
    
    def _calculate_correlation_matrix(self, symbols: List[str], price_data: Dict) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for given symbols"""
        try:
            returns_data = {}
            
            # Calculate returns for each symbol
            for symbol in symbols:
                if symbol in price_data and len(price_data[symbol]) > 30:
                    prices = pd.Series(price_data[symbol])
                    returns = prices.pct_change().dropna()
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return None
            
            # Align all return series to same dates
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:  # Need sufficient data
                return None
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def _calculate_diversification_score(self, corr_matrix: pd.DataFrame, holdings: Dict) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        try:
            if corr_matrix is None or len(corr_matrix) < 2:
                return 1.0
            
            # Weight-based diversification ratio
            symbols = list(corr_matrix.index)
            total_value = sum(holdings[s].get('current_value', 0) for s in symbols if s in holdings)
            
            if total_value == 0:
                return 1.0
            
            weights = np.array([
                holdings[s].get('current_value', 0) / total_value 
                for s in symbols if s in holdings
            ])
            
            # Portfolio variance vs equally weighted variance
            corr_array = corr_matrix.values
            portfolio_var = np.dot(weights, np.dot(corr_array, weights))
            equal_weight_var = np.mean(corr_array)
            
            # Diversification ratio (simplified)
            diversification_ratio = 1 / portfolio_var if portfolio_var > 0 else 1
            
            # Normalize to 0-1 scale
            score = min(1.0, max(0.0, diversification_ratio / len(symbols)))
            return score
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0.5
    
    def _calculate_average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average correlation excluding diagonal"""
        try:
            if corr_matrix is None or len(corr_matrix) < 2:
                return 0.0
            
            # Get upper triangle excluding diagonal
            upper_triangle = np.triu(corr_matrix.values, k=1)
            non_zero_elements = upper_triangle[upper_triangle != 0]
            
            if len(non_zero_elements) == 0:
                return 0.0
            
            return float(np.mean(non_zero_elements))
            
        except Exception as e:
            logger.error(f"Error calculating average correlation: {e}")
            return 0.0
    
    def _find_max_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Find maximum correlation excluding diagonal"""
        try:
            if corr_matrix is None or len(corr_matrix) < 2:
                return 0.0
            
            # Set diagonal to 0 to exclude self-correlation
            corr_copy = corr_matrix.copy()
            np.fill_diagonal(corr_copy.values, 0)
            
            return float(corr_copy.abs().max().max())
            
        except Exception as e:
            logger.error(f"Error finding max correlation: {e}")
            return 0.0
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, symbols: List[str]) -> List[Tuple]:
        """Find pairs with correlation above threshold"""
        try:
            high_corr_pairs = []
            
            if corr_matrix is None:
                return high_corr_pairs
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) > self.max_correlation:
                        high_corr_pairs.append((symbols[i], symbols[j], correlation))
            
            return high_corr_pairs
            
        except Exception as e:
            logger.error(f"Error finding high correlation pairs: {e}")
            return []
    
    def _generate_diversification_recommendations(self, 
                                                corr_matrix: pd.DataFrame,
                                                holdings: Dict,
                                                symbols: List[str],
                                                high_corr_pairs: List[Tuple]) -> List[str]:
        """Generate recommendations to improve diversification"""
        recommendations = []
        
        try:
            # Check for high correlations
            if high_corr_pairs:
                recommendations.append(f"⚠️ Found {len(high_corr_pairs)} highly correlated pairs (>{self.max_correlation:.2f})")
                
                for pair in high_corr_pairs[:3]:  # Show top 3
                    symbol1, symbol2, corr = pair
                    recommendations.append(f"  • {symbol1} & {symbol2}: {corr:.3f} correlation")
                
                recommendations.append("💡 Consider reducing position in one of the correlated assets")
            
            # Check portfolio size
            if len(holdings) < self.min_positions:
                recommendations.append(f"📊 Portfolio has only {len(holdings)} positions. Consider adding {self.min_positions - len(holdings)} more for better diversification")
            
            # Check sector concentration
            sector_allocation = self._analyze_sector_allocation(holdings)
            for sector, allocation in sector_allocation.items():
                if allocation > self.max_sector_allocation:
                    recommendations.append(f"🏭 {sector} sector is {allocation:.1%} of portfolio (limit: {self.max_sector_allocation:.1%})")
            
            # Check position concentration
            total_value = sum(h.get('current_value', 0) for h in holdings.values())
            if total_value > 0:
                for symbol, holding in holdings.items():
                    allocation = holding.get('current_value', 0) / total_value
                    if allocation > 0.25:  # 25% position limit
                        recommendations.append(f"📈 {symbol} is {allocation:.1%} of portfolio - consider reducing concentration")
            
            if not recommendations:
                recommendations.append("✅ Portfolio diversification looks good!")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["❌ Unable to generate diversification recommendations"]
    
    def _analyze_sector_allocation(self, holdings: Dict) -> Dict[str, float]:
        """Analyze allocation by sector (simplified)"""
        try:
            sector_allocation = {}
            total_value = sum(h.get('current_value', 0) for h in holdings.values())
            
            if total_value == 0:
                return sector_allocation
            
            for symbol, holding in holdings.items():
                # Simplified sector mapping based on symbol prefix
                sector = self._get_sector(symbol)
                allocation = holding.get('current_value', 0) / total_value
                
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0
                sector_allocation[sector] += allocation
            
            return sector_allocation
            
        except Exception as e:
            logger.error(f"Error analyzing sector allocation: {e}")
            return {}
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)"""
        # Simplified sector classification
        tech_symbols = ['TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH']
        bank_symbols = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK']
        energy_symbols = ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL']
        
        symbol_clean = symbol.replace('.NS', '').replace('.BO', '')
        
        if symbol_clean in tech_symbols:
            return 'Technology'
        elif symbol_clean in bank_symbols:
            return 'Banking'
        elif symbol_clean in energy_symbols:
            return 'Energy'
        else:
            return 'Other'
    
    def optimize_portfolio_weights(self, symbols: List[str], 
                                 expected_returns: np.ndarray,
                                 correlation_matrix: pd.DataFrame,
                                 current_weights: np.ndarray) -> Dict:
        """Optimize portfolio weights for better diversification"""
        try:
            if correlation_matrix is None or len(symbols) < 2:
                return {'optimized_weights': current_weights, 'improvement': 0.0}
            
            n_assets = len(symbols)
            
            # Objective function: minimize portfolio variance while maintaining diversification
            def objective(weights):
                portfolio_var = np.dot(weights, np.dot(correlation_matrix.values, weights))
                concentration_penalty = np.sum(weights ** 2)  # Penalize concentration
                return portfolio_var + 0.5 * concentration_penalty
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            ]
            
            # Bounds: each weight between 1% and 30%
            bounds = [(0.01, 0.30) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimized_weights = result.x
                
                # Calculate improvement
                current_var = objective(current_weights)
                optimized_var = objective(optimized_weights)
                improvement = (current_var - optimized_var) / current_var
                
                return {
                    'optimized_weights': optimized_weights,
                    'current_variance': current_var,
                    'optimized_variance': optimized_var,
                    'improvement': improvement,
                    'rebalancing_needed': improvement > self.rebalance_threshold
                }
            else:
                logger.warning("Portfolio optimization failed")
                return {'optimized_weights': current_weights, 'improvement': 0.0}
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio weights: {e}")
            return {'optimized_weights': current_weights, 'improvement': 0.0}
    
    def suggest_diversification_trades(self, holdings: Dict, 
                                     correlation_analysis: Dict,
                                     available_symbols: List[str]) -> List[Dict]:
        """Suggest trades to improve diversification"""
        try:
            suggestions = []
            
            # Find uncorrelated assets to add
            high_corr_pairs = correlation_analysis.get('high_correlation_pairs', [])
            current_symbols = list(holdings.keys())
            
            # Suggest reducing highly correlated positions
            for symbol1, symbol2, corr in high_corr_pairs:
                holding1_value = holdings.get(symbol1, {}).get('current_value', 0)
                holding2_value = holdings.get(symbol2, {}).get('current_value', 0)
                
                # Suggest reducing the smaller position
                if holding1_value < holding2_value:
                    reduce_symbol = symbol1
                    reduce_value = holding1_value * 0.5
                else:
                    reduce_symbol = symbol2
                    reduce_value = holding2_value * 0.5
                
                suggestions.append({
                    'action': 'REDUCE',
                    'symbol': reduce_symbol,
                    'reason': f'High correlation ({corr:.3f}) with other holdings',
                    'suggested_reduction': reduce_value,
                    'priority': 'HIGH' if abs(corr) > 0.8 else 'MEDIUM'
                })
            
            # Suggest adding uncorrelated assets
            for symbol in available_symbols[:5]:  # Check top 5 available
                if symbol not in current_symbols:
                    avg_correlation = self._estimate_correlation_with_portfolio(symbol, current_symbols)
                    
                    if avg_correlation < 0.5:  # Low correlation
                        suggestions.append({
                            'action': 'ADD',
                            'symbol': symbol,
                            'reason': f'Low correlation ({avg_correlation:.3f}) with portfolio',
                            'suggested_allocation': 0.05,  # 5% allocation
                            'priority': 'MEDIUM'
                        })
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating trade suggestions: {e}")
            return []
    
    def _estimate_correlation_with_portfolio(self, symbol: str, portfolio_symbols: List[str]) -> float:
        """Estimate correlation of symbol with existing portfolio (simplified)"""
        try:
            # Simplified correlation estimation based on sector
            symbol_sector = self._get_sector(symbol)
            
            portfolio_sectors = [self._get_sector(s) for s in portfolio_symbols]
            same_sector_count = portfolio_sectors.count(symbol_sector)
            
            # Higher same-sector count implies higher correlation
            estimated_correlation = same_sector_count / len(portfolio_symbols)
            return estimated_correlation
            
        except Exception as e:
            logger.error(f"Error estimating correlation: {e}")
            return 0.5
    
    def get_diversification_summary(self, analysis: Dict) -> str:
        """Get human-readable diversification summary"""
        try:
            score = analysis.get('diversification_score', 0)
            avg_corr = analysis.get('average_correlation', 0)
            recommendations = analysis.get('recommendations', [])
            
            summary = f"Portfolio Diversification Analysis:\\n"
            summary += f"Diversification Score: {score:.1%}\\n"
            summary += f"Average Correlation: {avg_corr:.3f}\\n\\n"
            
            if score > 0.7:
                summary += "✅ Well diversified portfolio\\n"
            elif score > 0.4:
                summary += "⚠️ Moderately diversified\\n"
            else:
                summary += "❌ Poor diversification\\n"
            
            summary += "\\nRecommendations:\\n"
            for rec in recommendations[:5]:
                summary += f"• {rec}\\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Diversification analysis unavailable"


# Global instance
_correlation_diversifier = None

def get_correlation_diversifier() -> CorrelationDiversifier:
    """Get global correlation diversifier instance"""
    global _correlation_diversifier
    if _correlation_diversifier is None:
        _correlation_diversifier = CorrelationDiversifier()
    return _correlation_diversifier