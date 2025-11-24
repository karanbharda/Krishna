"""
Phase 2: Advanced Feature Engineering System
Expands from 38 to 100+ features with correlation analysis and intelligent feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging

# Try to import TA-Lib with fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic feature engineering only")

# Try to import scipy with fallback
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import monitoring
try:
    from backend.utils.monitoring import log_model_performance
    MONITORING_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Monitoring not available for Feature Engineer")
    MONITORING_AVAILABLE = False

if 'logger' not in globals():
    logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering system that creates 100+ features from price and volume data
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.correlation_matrix = None
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.selected_features = []
        
        # Feature categories for organized engineering
        self.feature_categories = {
            'price_based': ['sma', 'ema', 'wma', 'dema', 'tema', 'trima'],
            'momentum': ['rsi', 'stoch', 'williams', 'cci', 'roc', 'momentum', 'mfi', 'ultosc', 'stochrsi', 'ppo'],
            'volatility': ['atr', 'bb', 'keltner', 'dc', 'ultosc', 'natr'],
            'volume': ['obv', 'ad', 'fi', 'nvi', 'pvi', 'vpt', 'adosc', 'vroc', 'vwap', 'eom'],
            'trend': ['macd', 'ppo', 'trix', 'dmi', 'aroon', 'sar', 'adx', 'dx', 'linearreg_slope', 'tsf', 'ht_trendmode'],
            'cycles': ['sine', 'leadsin', 'dcperiod', 'dcphase', 'ht_phasor'],
            'pattern': ['cdl_patterns', 'morning_star', 'evening_star', 'hammer'],
            'statistical': ['linear_reg', 'std_dev', 'var', 'beta', 'correl', 'tsf', 'linearreg_intercept', 'linearreg_angle', 'zscore'],
            'custom': ['price_position', 'volume_profile', 'support_resistance', 'gap', 'body_size', 'shadows', 'price_accel', 'close_to_sma']
        }
        
        logger.info("Advanced Feature Engineering System initialized")
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 100+ features from OHLCV data
        """
        try:
            if len(data) < 200:  # Need sufficient data for complex indicators
                logger.warning(f"Insufficient data for full feature engineering: {len(data)} rows")
                return self._engineer_basic_features(data)
            
            logger.info(f"Engineering 100+ features from {len(data)} data points")
            
            # Create feature dataframe
            features_df = data.copy()
            
            # 1. Price-based features (15 features)
            features_df = self._add_price_features(features_df)
            
            # 2. Momentum indicators (20 features)
            features_df = self._add_momentum_features(features_df)
            
            # 3. Volatility features (15 features)
            features_df = self._add_volatility_features(features_df)
            
            # 4. Volume features (12 features)
            features_df = self._add_volume_features(features_df)
            
            # 5. Trend indicators (18 features)
            features_df = self._add_trend_features(features_df)
            
            # 6. Cycle indicators (8 features)
            features_df = self._add_cycle_features(features_df)
            
            # 7. Candlestick patterns (20 features)
            features_df = self._add_pattern_features(features_df)
            
            # 8. Statistical features (15 features)
            features_df = self._add_statistical_features(features_df)
            
            # 9. Custom engineered features (12 features)
            features_df = self._add_custom_features(features_df)
            
            # Fill NaN values with forward fill then backward fill
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ Successfully engineered {len(features_df.columns)} total features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return self._engineer_basic_features(data)
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based moving averages and derived features"""
        try:
            high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
            open_price = df['Open'].values
            
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            
            # Exponential Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            
            # Weighted Moving Average
            df['wma_20'] = talib.WMA(close, timeperiod=20)
            
            # Double Exponential Moving Average
            df['dema_20'] = talib.DEMA(close, timeperiod=20)
            
            # Triple Exponential Moving Average
            df['tema_20'] = talib.TEMA(close, timeperiod=20)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding price features: {e}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators and momentum-based features"""
        try:
            high = df['High'].values.astype(np.double)
            low = df['Low'].values.astype(np.double)
            close = df['Close'].values.astype(np.double)
            volume = df['Volume'].values.astype(np.double)
            
            # RSI with different periods
            for period in [7, 14, 21]:
                df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            
            # Stochastic Oscillator
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
            
            # Williams %R
            for period in [14, 21]:
                df[f'willr_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
            
            # Commodity Channel Index
            df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            
            # Rate of Change
            for period in [10, 20]:
                df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
            
            # Momentum
            df['mom_10'] = talib.MOM(close, timeperiod=10)
            
            # Money Flow Index
            df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # Ultimate Oscillator
            df['ultosc'] = talib.ULTOSC(high, low, close)
            
            # Relative Strength Index Stochastic
            df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(close)
            
            # Percentage Price Oscillator
            df['ppo'] = talib.PPO(close)
            
            # Chande Momentum Oscillator
            df['cmo_14'] = talib.CMO(close, timeperiod=14)
            
            # Rate of Change Percentage
            df['rocp_10'] = talib.ROCP(close, timeperiod=10)
            
            # Triple Smoothed Rate of Change
            df['trix_14'] = talib.TRIX(close, timeperiod=14)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding momentum features: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        try:
            high, low, close = df['High'].values, df['Low'].values, df['Close'].values
            
            # Average True Range
            for period in [14, 21]:
                df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Keltner Channel (approximated)
            df['kc_middle'] = talib.EMA(close, timeperiod=20)
            atr_20 = talib.ATR(high, low, close, timeperiod=20)
            df['kc_upper'] = df['kc_middle'] + 2 * atr_20
            df['kc_lower'] = df['kc_middle'] - 2 * atr_20
            
            # Donchian Channel
            df['dc_upper'] = talib.MAX(high, timeperiod=20)
            df['dc_lower'] = talib.MIN(low, timeperiod=20)
            df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
            
            # True Range
            df['true_range'] = talib.TRANGE(high, low, close)
            
            # Normalized Average True Range
            df['natr'] = talib.NATR(high, low, close, timeperiod=14)
            
            # Standard Deviation
            df['std_dev_20'] = talib.STDDEV(close, timeperiod=20)
            
            # Variance
            df['var_20'] = talib.VAR(close, timeperiod=20)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding volatility features: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            high = df['High'].values.astype(np.double)
            low = df['Low'].values.astype(np.double)
            close = df['Close'].values.astype(np.double)
            volume = df['Volume'].values.astype(np.double)
            
            # On Balance Volume
            df['obv'] = talib.OBV(close, volume)
            
            # Accumulation/Distribution Line
            df['ad'] = talib.AD(high, low, close, volume)
            
            # Chaikin A/D Oscillator
            df['adosc'] = talib.ADOSC(high, low, close, volume)
            
            # Volume Rate of Change
            df['vroc_10'] = talib.ROC(volume, timeperiod=10)
            
            # Volume Moving Averages
            df['vol_sma_20'] = talib.SMA(volume, timeperiod=20)
            df['vol_ratio'] = volume / df['vol_sma_20']
            
            # Price Volume Trend
            close_series = pd.Series(close)
            volume_series = pd.Series(volume)
            df['pvt'] = ((close_series - close_series.shift(1)) / close_series.shift(1) * volume_series).cumsum()
            
            # Volume Weighted Average Price (approximated)
            df['vwap'] = (close_series * volume_series).rolling(20).sum() / volume_series.rolling(20).sum()
            
            # Force Index
            df['force_index'] = (close_series - close_series.shift(1)) * volume_series
            
            # Volume Oscillator
            df['vol_osc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # Chaikin Money Flow
            df['cmf'] = self._calculate_cmf(high, low, close, volume, 20)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding volume features: {e}")
            return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        try:
            high, low, close = df['High'].values, df['Low'].values, df['Close'].values
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
            
            # Average Directional Index
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Aroon
            df['aroon_up'], df['aroon_down'] = talib.AROON(high, low, timeperiod=14)
            df['aroon_osc'] = talib.AROONOSC(high, low, timeperiod=14)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(high, low)
            
            # TRIX
            df['trix'] = talib.TRIX(close, timeperiod=14)
            
            # Directional Movement Index
            df['dx'] = talib.DX(high, low, close, timeperiod=14)
            
            # Linear Regression Slope
            df['linearreg_slope'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
            
            # Time Series Forecast
            df['tsf'] = talib.TSF(close, timeperiod=14)
            
            # Hilbert Transform Trend vs Cycle Mode
            df['ht_trendmode'] = talib.HT_TRENDMODE(close)
            
            # Minus Directional Indicator
            df['mdi'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Plus Directional Indicator
            df['pdi'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            
            # Directional Movement Index Rating
            df['adxr'] = talib.ADXR(high, low, close, timeperiod=14)
            
            # Linear Regression Angle
            df['linearreg_angle'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding trend features: {e}")
            return df
    
    def _add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle indicators"""
        try:
            close = df['Close'].values
            
            # Hilbert Transform - Sine Wave
            df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(close)
            
            # Hilbert Transform - Dominant Cycle Period
            df['ht_dcperiod'] = talib.HT_DCPERIOD(close)
            
            # Hilbert Transform - Dominant Cycle Phase
            df['ht_dcphase'] = talib.HT_DCPHASE(close)
            
            # Hilbert Transform - Phasor Components
            df['ht_phasor_inphase'], df['ht_phasor_quad'] = talib.HT_PHASOR(close)
            
            # Mesa Adaptive Moving Average
            df['mama'], df['fama'] = talib.MAMA(close)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding cycle features: {e}")
            return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features"""
        try:
            open_price, high, low, close = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
            
            # Major candlestick patterns
            patterns = [
                'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
                'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
                'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
                'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE']
            
            for pattern in patterns[:15]:  # Limit to avoid too many features
                try:
                    pattern_func = getattr(talib, pattern)
                    df[pattern.lower()] = pattern_func(open_price, high, low, close)
                except:
                    continue
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding pattern features: {e}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical and mathematical features"""
        try:
            close = df['Close'].values
            high, low = df['High'].values, df['Low'].values
            
            # Standard deviation
            for period in [10, 20]:
                df[f'stddev_{period}'] = talib.STDDEV(close, timeperiod=period)
            
            # Variance
            df['var_10'] = talib.VAR(close, timeperiod=10)
            
            # Linear Regression
            df['linearreg'] = talib.LINEARREG(close, timeperiod=14)
            
            # Pearson Correlation Coefficient
            df['correl'] = talib.CORREL(high, low, timeperiod=30)
            
            # Beta
            df['beta'] = talib.BETA(high, low, timeperiod=5)
            
            # Time Series Forecast
            df['tsf_14'] = talib.TSF(close, timeperiod=14)
            
            # Linear Regression Intercept
            df['linearreg_intercept'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14)
            
            # Linear Regression Angle
            df['linearreg_angle'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
            
            # Z-Score (custom)
            rolling_mean = pd.Series(close).rolling(20).mean()
            rolling_std = pd.Series(close).rolling(20).std()
            df['zscore_20'] = (close - rolling_mean) / rolling_std
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding statistical features: {e}")
            return df
    
    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features"""
        try:
            # Price position within range
            df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Gap analysis
            df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
            df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
            
            # Intraday range
            df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
            
            # Body size (candle body)
            df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
            
            # Upper and lower shadows
            df['upper_shadow'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / df['Close']
            df['lower_shadow'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / df['Close']
            
            # Price acceleration
            df['price_accel'] = df['Close'].diff().diff()
            
            # Volume-price trend
            df['vpt'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
            
            # Relative position to moving averages
            if 'sma_20' in df.columns:
                df['close_to_sma20'] = df['Close'] / df['sma_20'] - 1
            
            # Volatility ratio
            if 'atr_14' in df.columns:
                df['volatility_ratio'] = df['atr_14'] / df['Close']
            
            # Support and resistance levels (simplified)
            df['resistance_level'] = df['High'].rolling(20).max()
            df['support_level'] = df['Low'].rolling(20).min()
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding custom features: {e}")
            return df
    
    def _calculate_cmf(self, high, low, close, volume, period):
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        cmf = pd.Series(mf_volume).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
        return cmf
    
    def _engineer_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic feature engineering for insufficient data"""
        try:
            features_df = data.copy()
            close = data['Close'].values
            
            # Basic moving averages
            features_df['sma_5'] = talib.SMA(close, timeperiod=5)
            features_df['sma_10'] = talib.SMA(close, timeperiod=10)
            features_df['sma_20'] = talib.SMA(close, timeperiod=20)
            
            # Basic momentum
            features_df['rsi_14'] = talib.RSI(close, timeperiod=14)
            
            # Basic MACD
            features_df['macd'], features_df['macd_signal'], features_df['macd_hist'] = talib.MACD(close)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ Basic feature engineering completed: {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error in basic feature engineering: {e}")
            return data
    
    def analyze_feature_correlations(self, features_df: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Analyze correlations between features and with target"""
        try:
            # Calculate correlation matrix
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            corr_matrix = features_df[numeric_cols].corr()
            
            # Find highly correlated features (>0.95)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            # Target correlations
            if target_col in features_df.columns:
                target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
            else:
                target_corrs = None
            
            self.correlation_matrix = corr_matrix
            
            return {
                'correlation_matrix': corr_matrix,
                'high_correlation_pairs': high_corr_pairs,
                'target_correlations': target_corrs,
                'total_features': len(numeric_cols)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    def select_best_features(self, features_df: pd.DataFrame, target: pd.Series, k: int = 50) -> List[str]:
        """Select k best features using multiple statistical tests and ensemble methods"""
        try:
            # Get numeric features only
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Remove target if it exists in features
            if target.name in numeric_features.columns:
                numeric_features = numeric_features.drop(columns=[target.name])
            
            # Handle NaN values
            numeric_features = numeric_features.fillna(numeric_features.median())
            target_clean = target.fillna(target.median())
            
            # Align indices
            common_idx = numeric_features.index.intersection(target_clean.index)
            numeric_features = numeric_features.loc[common_idx]
            target_clean = target_clean.loc[common_idx]
            
            # Multiple feature selection methods
            feature_scores = {}
            
            # 1. F-regression
            try:
                f_selector = SelectKBest(score_func=f_regression, k='all')
                f_selector.fit(numeric_features, target_clean)
                f_scores = dict(zip(numeric_features.columns, f_selector.scores_))
                # Normalize scores
                f_max = max(f_scores.values())
                if f_max > 0:
                    f_scores = {k: v/f_max for k, v in f_scores.items()}
                feature_scores['f_regression'] = f_scores
            except Exception as e:
                logger.warning(f"F-regression feature selection failed: {e}")
                feature_scores['f_regression'] = {col: 1.0 for col in numeric_features.columns}
            
            # 2. Mutual Information (if scipy available)
            if SCIPY_AVAILABLE:
                try:
                    mi_scores = {}
                    for col in numeric_features.columns:
                        try:
                            mi_score = mutual_info_regression(numeric_features[[col]], target_clean)[0]
                            mi_scores[col] = mi_score
                        except:
                            mi_scores[col] = 0.0
                    # Normalize scores
                    mi_max = max(mi_scores.values())
                    if mi_max > 0:
                        mi_scores = {k: v/mi_max for k, v in mi_scores.items()}
                    feature_scores['mutual_info'] = mi_scores
                except Exception as e:
                    logger.warning(f"Mutual info feature selection failed: {e}")
                    feature_scores['mutual_info'] = {col: 1.0 for col in numeric_features.columns}
            else:
                feature_scores['mutual_info'] = {col: 1.0 for col in numeric_features.columns}
            
            # 3. Correlation-based selection
            try:
                correlations = {}
                for col in numeric_features.columns:
                    try:
                        corr = abs(np.corrcoef(numeric_features[col], target_clean)[0, 1])
                        correlations[col] = corr if not np.isnan(corr) else 0.0
                    except:
                        correlations[col] = 0.0
                feature_scores['correlation'] = correlations
            except Exception as e:
                logger.warning(f"Correlation feature selection failed: {e}")
                feature_scores['correlation'] = {col: 1.0 for col in numeric_features.columns}
            
            # Ensemble feature selection - combine all methods
            ensemble_scores = {}
            method_weights = {'f_regression': 0.4, 'mutual_info': 0.3, 'correlation': 0.3}
            
            for col in numeric_features.columns:
                score = 0.0
                for method, weight in method_weights.items():
                    if method in feature_scores and col in feature_scores[method]:
                        score += feature_scores[method][col] * weight
                ensemble_scores[col] = score
            
            # Select top k features based on ensemble scores
            sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in sorted_features[:min(k, len(sorted_features))]]
            
            # Store feature scores
            self.feature_importance = ensemble_scores
            self.selected_features = selected_features
            
            logger.info(f"✅ Selected {len(selected_features)} best features from {len(numeric_features.columns)} using ensemble method")
            
            # Log top 10 features
            top_features = sorted_features[:10]
            logger.info(f"Top 10 features: {[(feat, f'{score:.4f}') for feat, score in top_features]}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            # Fallback to simple selection
            numeric_cols = list(features_df.select_dtypes(include=[np.number]).columns)
            if target.name in numeric_cols:
                numeric_cols.remove(target.name)
            return numeric_cols[:k]

    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance score"""
        if not self.feature_importance:
            return []
        
        return sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

    def select_features_by_stability(self, features_df: pd.DataFrame, target: pd.Series, k: int = 50) -> List[str]:
        """Select features based on stability over time"""
        try:
            # Get numeric features only
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Remove target if it exists in features
            if target.name in numeric_features.columns:
                numeric_features = numeric_features.drop(columns=[target.name])
            
            # Handle NaN values
            numeric_features = numeric_features.fillna(numeric_features.median())
            target_clean = target.fillna(target.median())
            
            # Align indices
            common_idx = numeric_features.index.intersection(target_clean.index)
            numeric_features = numeric_features.loc[common_idx]
            target_clean = target_clean.loc[common_idx]
            
            # Calculate stability scores based on rolling window correlations
            stability_scores = {}
            window_size = min(30, len(numeric_features) // 4)  # Use 25% of data or 30, whichever is smaller
            
            if window_size < 10:
                # Not enough data for stability analysis, fall back to regular selection
                return self.select_best_features(features_df, target, k)
            
            # Calculate rolling correlations for each feature
            for col in numeric_features.columns:
                try:
                    rolling_corrs = []
                    for i in range(0, len(numeric_features) - window_size, window_size // 2):
                        window_end = i + window_size
                        if window_end <= len(numeric_features):
                            feat_window = numeric_features[col].iloc[i:window_end]
                            target_window = target_clean.iloc[i:window_end]
                            
                            # Calculate correlation for this window
                            corr = np.corrcoef(feat_window, target_window)[0, 1]
                            if not np.isnan(corr):
                                rolling_corrs.append(abs(corr))
                    
                    # Stability is measured by low standard deviation of correlations
                    if len(rolling_corrs) > 1:
                        stability = 1.0 / (1.0 + np.std(rolling_corrs))  # Higher stability = lower std
                        avg_corr = np.mean(rolling_corrs)
                        stability_scores[col] = stability * avg_corr  # Combine stability with average predictive power
                    else:
                        stability_scores[col] = 0.5  # Default score
                except Exception as e:
                    logger.warning(f"Error calculating stability for {col}: {e}")
                    stability_scores[col] = 0.0
            
            # Select top k features based on stability scores
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, score in sorted_features[:min(k, len(sorted_features))]]
            
            # Update feature importance
            self.feature_importance = stability_scores
            self.selected_features = selected_features
            
            logger.info(f"✅ Selected {len(selected_features)} stable features from {len(numeric_features.columns)}")
            
            # Log top 10 stable features
            top_features = sorted_features[:10]
            logger.info(f"Top 10 stable features: {[(feat, f'{score:.4f}') for feat, score in top_features]}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error selecting stable features: {e}")
            # Fallback to regular feature selection
            return self.select_best_features(features_df, target, k)        
    
    def get_feature_summary(self) -> Dict:
        """Get summary of engineered features"""
        summary = {
            'total_features_engineered': len(self.feature_importance) if self.feature_importance else 0,
            'selected_features_count': len(self.selected_features),
            'feature_categories': self.feature_categories,
            'top_10_features': sorted(self.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10] if self.feature_importance else []
        }
        
        # Log performance metrics if monitoring is available
        if MONITORING_AVAILABLE:
            try:
                metrics = {
                    "features_engineered": summary['total_features_engineered'],
                    "features_selected": summary['selected_features_count'],
                    "quality_score": min(summary['total_features_engineered'] / 100.0, 1.0)  # Normalize to 0-1
                }
                log_model_performance("FeatureEngineer", metrics, summary)
            except Exception as e:
                logger.debug(f"Failed to log feature engineer performance: {e}")
        
        return summary


# Global instance
_feature_engineer = None

def get_feature_engineer() -> AdvancedFeatureEngineer:
    """Get the global feature engineer instance"""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = AdvancedFeatureEngineer()
    return _feature_engineer