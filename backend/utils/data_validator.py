"""
Data Validation and Outlier Detection System
Implements comprehensive data validation with statistical outlier detection for trading data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation"""
    completeness: float  # 0-1 scale
    accuracy: float      # 0-1 scale
    consistency: float   # 0-1 scale
    timeliness: float    # 0-1 scale
    outlier_count: int
    validation_score: float  # Overall quality score 0-1


class DataValidator:
    """
    Comprehensive data validation system with outlier detection
    """
    
    def __init__(self):
        # Validation thresholds
        self.completeness_threshold = 0.95  # 95% required fields present
        self.accuracy_threshold = 0.98      # 98% data points valid
        self.consistency_threshold = 0.95   # 95% consistent data
        self.timeliness_threshold = 0.99    # 99% recent data
        
        # Outlier detection parameters
        self.z_score_threshold = 3.0        # Standard deviations for outlier detection
        self.iqr_multiplier = 1.5           # IQR multiplier for outlier detection
        self.volatility_window = 30         # Days for volatility calculation
        
        # Historical data storage for outlier detection
        self.historical_data = {}
        self.validation_stats = {}
        
        logger.info("✅ Data Validator initialized")
    
    def validate_stock_data(self, symbol: str, data: pd.DataFrame) -> DataQualityMetrics:
        """
        Validate stock data with comprehensive checks
        
        Args:
            symbol: Stock symbol
            data: DataFrame with stock data
            
        Returns:
            DataQualityMetrics with validation results
        """
        try:
            # Perform all validation checks
            completeness = self._check_completeness(data)
            accuracy = self._check_accuracy(data)
            consistency = self._check_consistency(data)
            timeliness = self._check_timeliness(data)
            outlier_count = self._detect_outliers(symbol, data)
            
            # Calculate overall validation score
            validation_score = np.mean([completeness, accuracy, consistency, timeliness])
            
            metrics = DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                outlier_count=outlier_count,
                validation_score=validation_score
            )
            
            # Store validation stats
            self.validation_stats[symbol] = {
                'timestamp': datetime.now(),
                'metrics': metrics,
                'data_points': len(data)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            # Return poor quality metrics on error
            return DataQualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                consistency=0.0,
                timeliness=0.0,
                outlier_count=len(data) if data is not None else 0,
                validation_score=0.0
            )
    
    def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness"""
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            # Required columns for stock data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            column_completeness = 1.0 - (len(missing_columns) / len(required_columns))
            
            # Check for missing values in existing columns
            if len(data.columns) > 0:
                missing_values_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                data_completeness = 1.0 - missing_values_ratio
            else:
                data_completeness = 0.0
            
            # Weighted completeness score
            completeness = 0.7 * column_completeness + 0.3 * data_completeness
            return max(0.0, min(1.0, completeness))
            
        except Exception as e:
            logger.error(f"Error checking completeness: {e}")
            return 0.0
    
    def _check_accuracy(self, data: pd.DataFrame) -> float:
        """Check data accuracy"""
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            total_points = len(data)
            valid_points = 0
            
            # Check for negative prices (invalid)
            if 'Close' in data.columns:
                valid_close = (data['Close'] >= 0) & (data['Close'] < 1000000)  # Reasonable bounds
                valid_points += valid_close.sum()
            
            if 'Volume' in data.columns:
                valid_volume = (data['Volume'] >= 0)
                valid_points += valid_volume.sum()
            
            # Check for price consistency (High >= Low, etc.)
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                price_consistent = (
                    (data['High'] >= data['Low']) & 
                    (data['High'] >= data['Close']) & 
                    (data['Low'] <= data['Close'])
                )
                valid_points += price_consistent.sum()
            
            accuracy = valid_points / (total_points * 3) if total_points > 0 else 0.0
            return max(0.0, min(1.0, accuracy))
            
        except Exception as e:
            logger.error(f"Error checking accuracy: {e}")
            return 0.0
    
    def _check_consistency(self, data: pd.DataFrame) -> float:
        """Check data consistency"""
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            # Check for duplicate dates
            if 'Date' in data.columns:
                duplicate_dates = data['Date'].duplicated().sum()
                date_consistency = 1.0 - (duplicate_dates / len(data))
            else:
                date_consistency = 1.0
            
            # Check for monotonic date progression
            if 'Date' in data.columns and len(data) > 1:
                date_sorted = data['Date'].is_monotonic_increasing
                sort_consistency = 1.0 if date_sorted else 0.5
            else:
                sort_consistency = 1.0
            
            # Check for reasonable price movements
            if 'Close' in data.columns and len(data) > 1:
                price_changes = data['Close'].pct_change().dropna()
                extreme_changes = (abs(price_changes) > 0.5).sum()  # >50% daily change
                price_consistency = 1.0 - (extreme_changes / len(price_changes))
            else:
                price_consistency = 1.0
            
            consistency = np.mean([date_consistency, sort_consistency, price_consistency])
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return 0.0
    
    def _check_timeliness(self, data: pd.DataFrame) -> float:
        """Check data timeliness"""
        try:
            if data is None or len(data) == 0:
                return 0.0
            
            # Check if we have recent data (within last 2 business days)
            if 'Date' in data.columns and len(data) > 0:
                latest_date = pd.to_datetime(data['Date'].iloc[-1])
                days_old = (datetime.now() - latest_date).days
                
                # Consider data fresh if within 3 business days
                if days_old <= 3:
                    timeliness = 1.0
                elif days_old <= 7:
                    timeliness = 0.8
                elif days_old <= 30:
                    timeliness = 0.5
                else:
                    timeliness = 0.1
            else:
                timeliness = 0.5  # Neutral score if no date info
            
            return timeliness
            
        except Exception as e:
            logger.error(f"Error checking timeliness: {e}")
            return 0.0
    
    def _detect_outliers(self, symbol: str, data: pd.DataFrame) -> int:
        """Detect outliers in stock data using multiple sophisticated methods"""
        try:
            if data is None or len(data) == 0:
                return 0
            
            outlier_count = 0
            
            # Update historical data
            self._update_historical_data(symbol, data)
            
            # Detect outliers in price data using multiple methods
            if 'Close' in data.columns:
                # Z-score method
                z_outliers = self._detect_z_score_outliers(symbol, data['Close'])
                outlier_count += z_outliers
                
                # IQR method
                iqr_outliers = self._detect_iqr_outliers(data['Close'])
                outlier_count += iqr_outliers
                
                # Statistical method using historical data
                stat_outliers = self._detect_statistical_outliers(symbol, data['Close'])
                outlier_count += stat_outliers
                
                # Modified Z-score method (more robust to outliers)
                modified_z_outliers = self._detect_modified_z_score_outliers(data['Close'])
                outlier_count += modified_z_outliers
                
                # Grubbs' test for outliers
                grubbs_outliers = self._detect_grubbs_outliers(data['Close'])
                outlier_count += grubbs_outliers
            
            # Detect volume outliers
            if 'Volume' in data.columns:
                vol_outliers = self._detect_volume_outliers(symbol, data['Volume'])
                outlier_count += vol_outliers
                
                # Volume spike detection using rolling statistics
                volume_spike_outliers = self._detect_volume_spike_outliers(data['Volume'])
                outlier_count += volume_spike_outliers
            
            # Detect OHLC consistency outliers
            ohlc_outliers = self._detect_ohlc_outliers(data)
            outlier_count += ohlc_outliers
            
            return outlier_count
            
        except Exception as e:
            logger.error(f"Error detecting outliers for {symbol}: {e}")
            return 0
    
    def _update_historical_data(self, symbol: str, data: pd.DataFrame):
        """Update historical data for outlier detection"""
        try:
            if symbol not in self.historical_data:
                self.historical_data[symbol] = pd.DataFrame()
            
            # Append new data
            self.historical_data[symbol] = pd.concat([
                self.historical_data[symbol], 
                data
            ]).drop_duplicates(subset=['Date'] if 'Date' in data.columns else None)
            
            # Keep only recent data (last 2 years)
            if 'Date' in self.historical_data[symbol].columns:
                cutoff_date = datetime.now() - timedelta(days=730)
                self.historical_data[symbol] = self.historical_data[symbol][
                    pd.to_datetime(self.historical_data[symbol]['Date']) >= cutoff_date
                ]
            
        except Exception as e:
            logger.error(f"Error updating historical data for {symbol}: {e}")
    
    def _detect_z_score_outliers(self, symbol: str, series: pd.Series) -> int:
        """Detect outliers using Z-score method"""
        try:
            if len(series) < 3:
                return 0
            
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 3:
                return 0
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(clean_series))
            outliers = (z_scores > self.z_score_threshold).sum()
            
            if outliers > 0:
                logger.debug(f"Z-score outliers detected for {symbol}: {outliers}")
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in Z-score outlier detection: {e}")
            return 0
    
    def _detect_iqr_outliers(self, series: pd.Series) -> int:
        """Detect outliers using IQR method"""
        try:
            if len(series) < 4:
                return 0
            
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 4:
                return 0
            
            # Calculate IQR
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            # Count outliers
            outliers = ((clean_series < lower_bound) | (clean_series > upper_bound)).sum()
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in IQR outlier detection: {e}")
            return 0
    
    def _detect_statistical_outliers(self, symbol: str, series: pd.Series) -> int:
        """Detect outliers using statistical methods with historical data"""
        try:
            if symbol not in self.historical_data or len(self.historical_data[symbol]) < 30:
                return 0
            
            historical_data = self.historical_data[symbol]
            if 'Close' not in historical_data.columns:
                return 0
            
            # Get historical close prices
            historical_prices = historical_data['Close'].dropna()
            if len(historical_prices) < 30:
                return 0
            
            # Calculate historical statistics
            historical_mean = historical_prices.mean()
            historical_std = historical_prices.std()
            
            if historical_std == 0:
                return 0
            
            # Calculate current deviations
            current_prices = series.dropna()
            if len(current_prices) == 0:
                return 0
            
            # Z-score based on historical data
            z_scores = np.abs((current_prices - historical_mean) / historical_std)
            outliers = (z_scores > self.z_score_threshold).sum()
            
            if outliers > 0:
                logger.debug(f"Statistical outliers detected for {symbol}: {outliers}")
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in statistical outlier detection: {e}")
            return 0
    
    def _detect_volume_outliers(self, symbol: str, volume_series: pd.Series) -> int:
        """Detect volume outliers"""
        try:
            if len(volume_series) < 10:
                return 0
            
            clean_volume = volume_series.dropna()
            if len(clean_volume) < 10:
                return 0
            
            # Calculate average volume
            avg_volume = clean_volume.mean()
            
            if avg_volume == 0:
                return 0
            
            # Detect volume spikes (5x average volume)
            volume_spikes = (clean_volume > (5 * avg_volume)).sum()
            
            if volume_spikes > 0:
                logger.debug(f"Volume spikes detected for {symbol}: {volume_spikes}")
            
            return volume_spikes
            
        except Exception as e:
            logger.error(f"Error in volume outlier detection: {e}")
            return 0
    
    def _detect_modified_z_score_outliers(self, series: pd.Series) -> int:
        """Detect outliers using Modified Z-Score (more robust to outliers)"""
        try:
            if len(series) < 5:
                return 0
            
            clean_series = series.dropna()
            if len(clean_series) < 5:
                return 0
            
            # Calculate median absolute deviation
            median = clean_series.median()
            mad = np.median(np.abs(clean_series - median))
            
            if mad == 0:
                return 0
            
            # Modified Z-score (0.6745 is a constant for normal distribution)
            modified_z_scores = 0.6745 * (clean_series - median) / mad
            outliers = (np.abs(modified_z_scores) > self.z_score_threshold).sum()
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in modified Z-score outlier detection: {e}")
            return 0
    
    def _detect_grubbs_outliers(self, series: pd.Series) -> int:
        """Detect outliers using Grubbs' test"""
        try:
            if len(series) < 6:  # Grubbs' test requires at least 6 data points
                return 0
            
            clean_series = series.dropna()
            if len(clean_series) < 6:
                return 0
            
            # Grubbs' test implementation
            n = len(clean_series)
            mean = clean_series.mean()
            std = clean_series.std()
            
            if std == 0:
                return 0
            
            # Calculate Grubbs' statistic for each point
            grubbs_stats = np.abs(clean_series - mean) / std
            
            # Critical value for Grubbs' test (approximate)
            # For a more accurate implementation, you'd use statistical tables
            critical_value = (n - 1) / np.sqrt(n) * np.sqrt(
                (stats.t.ppf(1 - 0.05/(2*n), n-2))**2 / 
                (n - 2 + (stats.t.ppf(1 - 0.05/(2*n), n-2))**2)
            )
            
            outliers = (grubbs_stats > critical_value).sum()
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in Grubbs' test outlier detection: {e}")
            return 0
    
    def _detect_volume_spike_outliers(self, volume_series: pd.Series) -> int:
        """Detect volume spikes using rolling statistics"""
        try:
            if len(volume_series) < 20:
                return 0
            
            clean_volume = volume_series.dropna()
            if len(clean_volume) < 20:
                return 0
            
            # Calculate rolling statistics
            rolling_mean = clean_volume.rolling(window=20, min_periods=10).mean()
            rolling_std = clean_volume.rolling(window=20, min_periods=10).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)
            
            # Calculate Z-scores for volume
            volume_z_scores = (clean_volume - rolling_mean) / rolling_std
            
            # Detect extreme volume spikes (3 standard deviations above rolling mean)
            volume_spikes = (volume_z_scores > 3).sum()
            
            return volume_spikes
            
        except Exception as e:
            logger.error(f"Error in volume spike outlier detection: {e}")
            return 0
    
    def _detect_ohlc_outliers(self, data: pd.DataFrame) -> int:
        """Detect outliers in OHLC price relationships"""
        try:
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_columns):
                return 0
            
            if len(data) < 5:
                return 0
            
            outliers = 0
            
            for idx, row in data.iterrows():
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                
                # Check for invalid OHLC relationships
                if (high_price < low_price or 
                    high_price < open_price or 
                    high_price < close_price or
                    low_price > open_price or
                    low_price > close_price or
                    abs(high_price - low_price) > (high_price + low_price) * 0.5):  # Extreme range
                    outliers += 1
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error in OHLC outlier detection: {e}")
            return 0
    
    def is_data_quality_acceptable(self, symbol: str, metrics: DataQualityMetrics) -> bool:
        """Check if data quality meets acceptable thresholds"""
        try:
            # Check individual thresholds
            completeness_ok = metrics.completeness >= self.completeness_threshold
            accuracy_ok = metrics.accuracy >= self.accuracy_threshold
            consistency_ok = metrics.consistency >= self.consistency_threshold
            timeliness_ok = metrics.timeliness >= self.timeliness_threshold
            
            # Check overall quality score
            quality_ok = metrics.validation_score >= 0.8
            
            # Check outlier count (should be minimal)
            outliers_ok = metrics.outlier_count <= max(1, len(self.historical_data.get(symbol, [])) * 0.05)
            
            acceptable = completeness_ok and accuracy_ok and consistency_ok and timeliness_ok and quality_ok and outliers_ok
            
            if not acceptable:
                logger.warning(f"Data quality issues for {symbol}: "
                             f"Completeness={metrics.completeness:.2f}, "
                             f"Accuracy={metrics.accuracy:.2f}, "
                             f"Consistency={metrics.consistency:.2f}, "
                             f"Timeliness={metrics.timeliness:.2f}, "
                             f"Outliers={metrics.outlier_count}")
            
            return acceptable
            
        except Exception as e:
            logger.error(f"Error checking data quality acceptability: {e}")
            return False
    
    def get_fallback_data(self, symbol: str, requested_data: pd.DataFrame) -> pd.DataFrame:
        """Get fallback data when validation fails"""
        try:
            # Return cached historical data if available
            if symbol in self.historical_data and len(self.historical_data[symbol]) > 0:
                logger.info(f"Using fallback data for {symbol} from historical cache")
                return self.historical_data[symbol].tail(50)  # Last 50 data points
            
            # If no historical data, return last known good data from requested data
            if requested_data is not None and len(requested_data) > 0:
                logger.info(f"Using fallback data for {symbol} from requested data")
                return requested_data.tail(20)  # Last 20 data points
            
            # Return empty DataFrame as last resort
            logger.warning(f"No fallback data available for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting fallback data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_validation_report(self, symbol: str) -> Dict:
        """Get detailed validation report for a symbol"""
        try:
            if symbol in self.validation_stats:
                stats = self.validation_stats[symbol]
                metrics = stats['metrics']
                
                return {
                    'symbol': symbol,
                    'timestamp': stats['timestamp'].isoformat(),
                    'data_points': stats['data_points'],
                    'quality_metrics': {
                        'completeness': metrics.completeness,
                        'accuracy': metrics.accuracy,
                        'consistency': metrics.consistency,
                        'timeliness': metrics.timeliness,
                        'validation_score': metrics.validation_score,
                        'outlier_count': metrics.outlier_count
                    },
                    'status': 'ACCEPTABLE' if self.is_data_quality_acceptable(symbol, metrics) else 'POOR_QUALITY'
                }
            else:
                return {
                    'symbol': symbol,
                    'status': 'NO_DATA',
                    'message': 'No validation data available'
                }
                
        except Exception as e:
            logger.error(f"Error generating validation report for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'ERROR',
                'message': str(e)
            }


# Global instance
_data_validator = None


def get_data_validator() -> DataValidator:
    """Get global data validator instance"""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataValidator()
    return _data_validator