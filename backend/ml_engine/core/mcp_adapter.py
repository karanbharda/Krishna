"""
MCP Adapter - Wrapper for orchestrator integration
Provides tool-style interfaces for stock prediction system
"""

import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..analysis.stock_analysis_complete import (
    EnhancedDataIngester,
    FeatureEngineer,
    predict_stock_price,
    train_ml_models,
    DATA_CACHE_DIR,
    FEATURE_CACHE_DIR,
    MODEL_DIR,
    LOGS_DIR
)

logger = logging.getLogger(__name__)

# Request/Response logging
MCP_LOG_DIR = LOGS_DIR / "mcp_requests"
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)


class MCPAdapter:
    """
    MCP-style adapter for stock prediction system
    Provides orchestrator-friendly tool interfaces with comprehensive logging
    """
    
    def __init__(self):
        self.ingester = EnhancedDataIngester()
        self.engineer = FeatureEngineer()
        self.request_counter = 0
        
    def _log_request(self, tool_name: str, request_data: Dict) -> str:
        """Log incoming request"""
        self.request_counter += 1
        request_id = f"{tool_name}_{int(time.time())}_{self.request_counter}"
        
        log_entry = {
            "request_id": request_id,
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "request": request_data
        }
        
        log_file = MCP_LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}_requests.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"MCP Request [{request_id}]: {tool_name}")
        return request_id
    
    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types and other non-JSON-serializable types to Python native types"""
        import numpy as np
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return obj
    
    def _log_response(self, request_id: str, response_data: Dict, duration_ms: float):
        """Log outgoing response"""
        # Convert numpy types to Python native types for JSON serialization
        sanitized_response = self._convert_to_json_serializable(response_data)
        
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": duration_ms,
            "response": sanitized_response
        }
        
        log_file = MCP_LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}_responses.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except TypeError as e:
            logger.error(f"Failed to log response: {e}")
            logger.error(f"Problematic data: {log_entry}")
            # Try to log with default=str as fallback
            try:
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry, default=str) + '\n')
            except Exception as e2:
                logger.error(f"Failed to log response even with default=str: {e2}")
        
        logger.info(f"MCP Response [{request_id}]: {duration_ms:.2f}ms")
    
    def predict(
        self,
        symbols: List[str],
        horizon: str = "intraday",
        risk_profile: Optional[str] = None,
        stop_loss_pct: Optional[float] = None,
        capital_risk_pct: Optional[float] = None,
        drawdown_limit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        MCP Tool: predict
        
        Args:
            symbols: List of stock symbols to predict
            horizon: Time horizon (intraday, short, long)
            risk_profile: Optional risk profile override
        
        Returns:
            Dict with metadata and predictions array
        """
        start_time = time.time()
        request_data = {
            "symbols": symbols,
            "horizon": horizon,
            "risk_profile": risk_profile
        }
        request_id = self._log_request("predict", request_data)
        
        try:
            predictions = []
            
            for symbol in symbols:
                try:
                    print(f"\n{'='*80}", flush=True)
                    print(f"[API REQUEST] Processing {symbol} for {horizon} horizon", flush=True)
                    print(f"{'='*80}\n", flush=True)
                    logger.info(f"[{request_id}] Predicting {symbol} ({horizon})")
                    
                    # STEP 1: Ensure data exists
                    json_path = DATA_CACHE_DIR / f"{symbol}_all_data.json"
                    
                    if not json_path.exists():
                        print(f"[STEP 1/4] Data not found for {symbol}. Fetching from Yahoo Finance...", flush=True)
                        logger.info(f"[{request_id}] Data not found for {symbol}. Fetching...")
                        try:
                            self.ingester.fetch_all_data(symbol, period="2y")
                            print(f"[STEP 1/4] [OK] Data fetched successfully!\n", flush=True)
                            logger.info(f"[{request_id}] Data fetched for {symbol}")
                        except Exception as e:
                            print(f"[STEP 1/4] [FAIL] Failed to fetch data: {e}\n", flush=True)
                            logger.error(f"[{request_id}] Failed to fetch data for {symbol}: {e}")
                            predictions.append({
                                "symbol": symbol,
                                "horizon": horizon,
                                "error": f"Data fetch failed: {str(e)}"
                            })
                            continue
                    else:
                        print(f"[STEP 1/4] [OK] Data already cached\n", flush=True)
                    
                    # STEP 2: Ensure features are calculated
                    features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
                    if not features_path.exists():
                        print(f"[STEP 2/4] Features not found. Calculating 50+ technical indicators...", flush=True)
                        logger.info(f"[{request_id}] Features not found for {symbol}. Calculating...")
                        all_data = self.ingester.load_all_data(symbol)
                        if all_data:
                            df = all_data.get('price_history')
                            if df is not None and not df.empty:
                                features_df = self.engineer.calculate_all_features(df, symbol)
                                self.engineer.save_features(features_df, symbol)
                                print(f"[STEP 2/4] [OK] Features calculated successfully!\n", flush=True)
                                logger.info(f"[{request_id}] Features calculated for {symbol}")
                    
                    else:
                        print(f"[STEP 2/4] [OK] Features already calculated\n", flush=True)
                    
                    # STEP 3: Check if models exist for this horizon
                    model_files = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*"))
                    
                    if not model_files:
                        print(f"[STEP 3/4] Models not found. Training 4 ML models (RF+LGB+XGB+DQN)...", flush=True)
                        print(f"            This will take 60-90 seconds...\n", flush=True)
                        logger.info(f"[{request_id}] Models not found for {symbol} ({horizon}). Training...")
                        try:
                            from ..analysis.stock_analysis_complete import train_ml_models
                            training_result = train_ml_models(symbol, horizon, verbose=True)
                            
                            # Handle both dict and bool return formats
                            success = training_result.get('success', False) if isinstance(training_result, dict) else training_result
                            
                            if not success:
                                print(f"[STEP 3/4] [FAIL] Training failed\n", flush=True)
                                logger.error(f"[{request_id}] Training failed for {symbol}")
                                predictions.append({
                                    "symbol": symbol,
                                    "horizon": horizon,
                                    "error": "Model training failed"
                                })
                                continue
                            print(f"[STEP 3/4] [OK] All 4 models trained successfully!\n", flush=True)
                            logger.info(f"[{request_id}] Models trained for {symbol} ({horizon})")
                        except Exception as e:
                            print(f"[STEP 3/4] [FAIL] Training error: {e}\n", flush=True)
                            logger.error(f"[{request_id}] Training failed for {symbol}: {e}", exc_info=True)
                            predictions.append({
                                "symbol": symbol,
                                "horizon": horizon,
                                "error": f"Training failed: {str(e)}"
                            })
                            continue
                    else:
                        print(f"[STEP 3/4] [OK] Models already trained\n", flush=True)
                    
                    # STEP 4: Get prediction
                    print(f"[STEP 4/4] Generating prediction using ensemble of 4 models...", flush=True)
                    prediction = predict_stock_price(symbol, horizon=horizon, verbose=True)
                    print(f"[STEP 4/4] [OK] Prediction generated!\n", flush=True)
                    
                    if prediction:
                        # Override risk_profile if specified
                        if risk_profile:
                            prediction["risk_profile"] = risk_profile
                            prediction["horizon_details"]["risk_profile"] = risk_profile
                        
                        predictions.append(prediction)
                        
                        # Log to main predictions file
                        self._log_prediction_to_file(prediction)
                        
                        logger.info(f"[{request_id}] [OK] {symbol}: {prediction['action']} "
                                  f"(confidence: {prediction['confidence']:.4f})")
                    else:
                        logger.warning(f"[{request_id}] [FAIL] {symbol}: No prediction returned")
                        predictions.append({
                            "symbol": symbol,
                            "horizon": horizon,
                            "error": "Prediction failed - models may need training"
                        })
                        
                except Exception as e:
                    logger.error(f"[{request_id}] Error predicting {symbol}: {e}", exc_info=True)
                    predictions.append({
                        "symbol": symbol,
                        "horizon": horizon,
                        "error": str(e)
                    })
            
            # Determine risk profile from horizon if not specified
            if not risk_profile:
                risk_profiles = {
                    "intraday": "high",
                    "short": "moderate",
                    "long": "low"
                }
                risk_profile = risk_profiles.get(horizon, "moderate")
            
            response = {
                "metadata": {
                    "count": len(predictions),
                    "horizon": horizon,
                    "risk_profile": risk_profile,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "predictions": predictions
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, response, duration_ms)
            
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Critical error: {e}", exc_info=True)
            error_response = {
                "metadata": {
                    "request_id": request_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "predictions": []
            }
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, error_response, duration_ms)
            return error_response
    
    def scan_all(
        self,
        symbols: Optional[List[str]] = None,
        horizon: str = "intraday",
        min_confidence: float = 0.5,
        stop_loss_pct: Optional[float] = None,
        capital_risk_pct: Optional[float] = None,
        drawdown_limit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        MCP Tool: scan_all
        
        Batch scoring of multiple symbols with filtering and risk parameters
        
        Args:
            symbols: List of symbols to scan (uses default universe if None)
            horizon: Time horizon for predictions
            min_confidence: Minimum confidence threshold for shortlist
        
        Returns:
            Dict with shortlisted high-confidence predictions
        """
        start_time = time.time()
        
        # Default universe if no symbols provided
        if not symbols:
            symbols = [
                "RPOWER.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", 
                "RELIANCE.NS", "TATASTEEL.NS", "WIPRO.NS", "ITC.NS",
                "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"
            ]
        
        request_data = {
            "symbols": symbols,
            "horizon": horizon,
            "min_confidence": min_confidence
        }
        request_id = self._log_request("scan_all", request_data)
        
        try:
            logger.info(f"[{request_id}] Scanning {len(symbols)} symbols...")
            
            all_predictions = []
            shortlist = []
            
            for symbol in symbols:
                try:
                    print(f"\n{'='*80}", flush=True)
                    print(f"[SCAN] Processing {symbol} ({len(symbols)} total)", flush=True)
                    print(f"{'='*80}\n", flush=True)
                    logger.info(f"[{request_id}] Processing {symbol}...")
                    
                    # STEP 1: Ensure data exists
                    json_path = DATA_CACHE_DIR / f"{symbol}_all_data.json"
                    
                    if not json_path.exists():
                        print(f"[STEP 1/4] Fetching data from Yahoo Finance...", flush=True)
                        logger.info(f"[{request_id}] Data not found for {symbol}. Fetching...")
                        try:
                            self.ingester.fetch_all_data(symbol, period="2y")
                            print(f"[STEP 1/4] [OK] Data fetched!\n", flush=True)
                            logger.info(f"[{request_id}] Data fetched for {symbol}")
                        except Exception as e:
                            print(f"[STEP 1/4] [FAIL] Fetch failed: {e}\n", flush=True)
                            logger.error(f"[{request_id}] Failed to fetch data for {symbol}: {e}")
                            continue
                    else:
                        print(f"[STEP 1/4] [OK] Data cached\n", flush=True)
                    
                    # STEP 2: Ensure features are calculated
                    features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
                    if not features_path.exists():
                        print(f"[STEP 2/4] Calculating 50+ technical indicators...", flush=True)
                        logger.info(f"[{request_id}] Features not found for {symbol}. Calculating...")
                        all_data = self.ingester.load_all_data(symbol)
                        if all_data:
                            df = all_data.get('price_history')
                            if df is not None and not df.empty:
                                features_df = self.engineer.calculate_all_features(df, symbol)
                                self.engineer.save_features(features_df, symbol)
                                print(f"[STEP 2/4] [OK] Features calculated!\n", flush=True)
                                logger.info(f"[{request_id}] Features calculated for {symbol}")
                    
                    else:
                        print(f"[STEP 2/4] [OK] Features cached\n", flush=True)
                    
                    # STEP 3: Check if models exist for this horizon
                    model_files = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*"))
                    
                    if not model_files:
                        print(f"[STEP 3/4] Training 4 ML models (60-90 seconds)...", flush=True)
                        logger.info(f"[{request_id}] Models not found for {symbol} ({horizon}). Training...")
                        try:
                            from ..analysis.stock_analysis_complete import train_ml_models
                            training_result = train_ml_models(symbol, horizon, verbose=True)
                            
                            # Handle both dict and bool return formats
                            success = training_result.get('success', False) if isinstance(training_result, dict) else training_result
                            
                            if not success:
                                print(f"[STEP 3/4] [FAIL] Training failed\n", flush=True)
                                logger.error(f"[{request_id}] Training failed for {symbol}")
                                continue
                            print(f"[STEP 3/4] [OK] Models trained!\n", flush=True)
                            logger.info(f"[{request_id}] Models trained for {symbol} ({horizon})")
                        except Exception as e:
                            print(f"[STEP 3/4] [FAIL] Training error: {e}\n", flush=True)
                            logger.error(f"[{request_id}] Training failed for {symbol}: {e}", exc_info=True)
                            continue
                    else:
                        print(f"[STEP 3/4] [OK] Models cached\n", flush=True)
                    
                    # STEP 4: Get prediction
                    print(f"[STEP 4/4] Generating prediction...", flush=True)
                    prediction = predict_stock_price(symbol, horizon=horizon, verbose=True)
                    print(f"[STEP 4/4] [OK] Done!\n", flush=True)
                    
                    if prediction:
                        all_predictions.append(prediction)
                        
                        # Add to shortlist if meets confidence threshold
                        if prediction.get('confidence', 0) >= min_confidence:
                            shortlist.append(prediction)
                            logger.info(f"[{request_id}] [OK] SHORTLIST: {symbol} "
                                      f"({prediction['action']}, conf: {prediction['confidence']:.4f})")
                        
                        # Log prediction
                        self._log_prediction_to_file(prediction)
                        
                except Exception as e:
                    logger.error(f"[{request_id}] Error scanning {symbol}: {e}")
            
            # Sort shortlist by score (descending)
            shortlist.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            response = {
                "metadata": {
                    "total_scanned": len(symbols),
                    "predictions_generated": len(all_predictions),
                    "shortlist_count": len(shortlist),
                    "horizon": horizon,
                    "min_confidence": min_confidence,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "shortlist": shortlist,
                "all_predictions": all_predictions
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, response, duration_ms)
            
            logger.info(f"[{request_id}] Scan complete: {len(shortlist)}/{len(all_predictions)} "
                       f"passed threshold ({duration_ms:.2f}ms)")
            
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Scan error: {e}", exc_info=True)
            error_response = {
                "metadata": {
                    "error": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                "shortlist": [],
                "all_predictions": []
            }
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, error_response, duration_ms)
            return error_response
    
    def analyze(
        self,
        symbol: str,
        horizons: Optional[List[str]] = None,
        stop_loss_pct: Optional[float] = None,
        capital_risk_pct: Optional[float] = None,
        drawdown_limit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        MCP Tool: analyze
        
        Deep analysis of a single ticker across multiple horizons
        
        Args:
            symbol: Stock symbol to analyze
            horizons: List of horizons to analyze (default: all 3)
        
        Returns:
            Dict with multi-horizon analysis
        """
        start_time = time.time()
        
        if not horizons:
            horizons = ["intraday", "short", "long"]
        
        request_data = {
            "symbol": symbol,
            "horizons": horizons
        }
        request_id = self._log_request("analyze", request_data)
        
        try:
            logger.info(f"[{request_id}] Analyzing {symbol} across {len(horizons)} horizons...")
            
            predictions = []
            
            # First ensure data and features exist (only once, not per horizon)
            json_path = DATA_CACHE_DIR / f"{symbol}_all_data.json"
            
            if not json_path.exists():
                print(f"\n[ANALYZE] Fetching data for {symbol}...", flush=True)
                logger.info(f"[{request_id}] Data not found for {symbol}. Fetching...")
                try:
                    self.ingester.fetch_all_data(symbol, period="2y")
                    print(f"[ANALYZE] [OK] Data fetched!\n", flush=True)
                except Exception as e:
                    print(f"[ANALYZE] [FAIL] Data fetch failed: {e}\n", flush=True)
                    logger.error(f"[{request_id}] Failed to fetch data: {e}")
                    return {
                        "metadata": {
                            "symbol": symbol,
                            "error": f"Data fetch failed: {str(e)}",
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat()
                        },
                        "predictions": []
                    }
            else:
                print(f"[ANALYZE] [OK] Data cached for {symbol}\n", flush=True)
            
            # Ensure features are calculated
            features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
            if not features_path.exists():
                print(f"[ANALYZE] Calculating features for {symbol}...", flush=True)
                logger.info(f"[{request_id}] Features not found. Calculating...")
                all_data = self.ingester.load_all_data(symbol)
                if all_data:
                    df = all_data.get('price_history')
                    if df is not None and not df.empty:
                        features_df = self.engineer.calculate_all_features(df, symbol)
                        self.engineer.save_features(features_df, symbol)
                        print(f"[ANALYZE] [OK] Features calculated!\n", flush=True)
            else:
                print(f"[ANALYZE] [OK] Features cached for {symbol}\n", flush=True)
            
            # Now process each horizon
            for horizon in horizons:
                try:
                    print(f"\n{'='*80}", flush=True)
                    print(f"[ANALYZE] Processing {symbol} - {horizon.upper()} horizon", flush=True)
                    print(f"{'='*80}\n", flush=True)
                    
                    # Check if models exist for this horizon
                    model_files = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*"))
                    
                    if not model_files:
                        print(f"[ANALYZE] Training models for {horizon} horizon (60-90 seconds)...", flush=True)
                        logger.info(f"[{request_id}] Models not found for {symbol} ({horizon}). Training...")
                        try:
                            from ..analysis.stock_analysis_complete import train_ml_models
                            training_result = train_ml_models(symbol, horizon, verbose=True)
                            
                            # Handle both dict and bool return formats
                            success = training_result.get('success', False) if isinstance(training_result, dict) else training_result
                            
                            if not success:
                                print(f"[ANALYZE] [FAIL] Training failed for {horizon}\n", flush=True)
                                logger.error(f"[{request_id}] Training failed for {horizon}")
                                predictions.append({
                                    "symbol": symbol,
                                    "horizon": horizon,
                                    "error": "Model training failed"
                                })
                                continue
                            print(f"[ANALYZE] [OK] Models trained for {horizon}!\n", flush=True)
                        except Exception as e:
                            print(f"[ANALYZE] [FAIL] Training error: {e}\n", flush=True)
                            logger.error(f"[{request_id}] Training error for {horizon}: {e}", exc_info=True)
                            predictions.append({
                                "symbol": symbol,
                                "horizon": horizon,
                                "error": f"Training failed: {str(e)}"
                            })
                            continue
                    else:
                        print(f"[ANALYZE] [OK] Models exist for {horizon}\n", flush=True)
                    
                    # Generate prediction
                    print(f"[ANALYZE] Generating {horizon} prediction...", flush=True)
                    prediction = predict_stock_price(symbol, horizon=horizon, verbose=True)
                    
                    if prediction:
                        print(f"[ANALYZE] [OK] {horizon} prediction: {prediction['action']} "
                              f"({prediction['predicted_return']:+.2f}%, conf: {prediction['confidence']:.4f})\n", flush=True)
                        predictions.append(prediction)
                        self._log_prediction_to_file(prediction)
                        logger.info(f"[{request_id}] [OK] {horizon}: {prediction['action']} "
                                  f"(conf: {prediction['confidence']:.4f})")
                    else:
                        print(f"[ANALYZE] [FAIL] {horizon} prediction failed\n", flush=True)
                        logger.warning(f"[{request_id}] [FAIL] {horizon}: No prediction")
                        predictions.append({
                            "symbol": symbol,
                            "horizon": horizon,
                            "error": "Prediction failed"
                        })
                        
                except Exception as e:
                    print(f"[ANALYZE] [FAIL] Error on {horizon}: {e}\n", flush=True)
                    logger.error(f"[{request_id}] Error on {horizon}: {e}")
                    predictions.append({
                        "symbol": symbol,
                        "horizon": horizon,
                        "error": str(e)
                    })
            
            # Calculate consensus across horizons
            actions = [p.get('action') for p in predictions if 'action' in p]
            avg_confidence = sum(p.get('confidence', 0) for p in predictions if 'confidence' in p) / len(predictions) if predictions else 0
            
            consensus = None
            if len(set(actions)) == 1:
                consensus = f"Strong {actions[0]} - All horizons agree"
            elif actions.count('LONG') > actions.count('SHORT'):
                consensus = "Bullish - Majority LONG signals"
            elif actions.count('SHORT') > actions.count('LONG'):
                consensus = "Bearish - Majority SHORT signals"
            else:
                consensus = "Mixed signals - Exercise caution"
            
            response = {
                "metadata": {
                    "symbol": symbol,
                    "horizons": horizons,
                    "count": len(predictions),
                    "average_confidence": round(avg_confidence, 4),
                    "consensus": consensus,
                    "risk_parameters": {
                        "stop_loss_pct": stop_loss_pct,
                        "capital_risk_pct": capital_risk_pct,
                        "drawdown_limit_pct": drawdown_limit_pct
                    },
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "predictions": predictions
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, response, duration_ms)
            
            logger.info(f"[{request_id}] Analysis complete: {consensus} ({duration_ms:.2f}ms)")
            
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Analysis error: {e}", exc_info=True)
            error_response = {
                "metadata": {
                    "symbol": symbol,
                    "error": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                "predictions": []
            }
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, error_response, duration_ms)
            return error_response
    
    def _log_prediction_to_file(self, prediction: Dict):
        """Log prediction to main predictions file"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            sanitized_prediction = self._convert_to_json_serializable(prediction)
            log_file = LOGS_DIR / "mcp_predictions.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(sanitized_prediction) + '\n')
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def confirm(
        self,
        symbol: str,
        decision: str,
        session_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP Tool: confirm
        Confirm or reject a trade decision
        
        Args:
            symbol: Stock symbol
            decision: "approve" or "reject"
            session_id: Session identifier
            reason: Optional reason for the decision
        
        Returns:
            Dict with confirmation result
        """
        start_time = time.time()
        request_data = {
            "symbol": symbol,
            "decision": decision,
            "session_id": session_id,
            "reason": reason
        }
        request_id = self._log_request("confirm", request_data)
        
        try:
            logger.info(f"[{request_id}] Trade confirmation for {symbol}: {decision} (session: {session_id})")
            
            # In a real implementation, this would interact with the trading system
            # For now, we'll just log the decision
            result = {
                "symbol": symbol,
                "decision": decision,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "confirmed" if decision == "approve" else "rejected",
                "reason": reason
            }
            
            response = {
                "metadata": {
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                },
                "result": result
            }
            
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, response, duration_ms)
            
            logger.info(f"[{request_id}] Trade confirmation processed: {decision}")
            
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Error confirming trade for {symbol}: {e}", exc_info=True)
            error_response = {
                "metadata": {
                    "request_id": request_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "result": {}
            }
            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, error_response, duration_ms)
            return error_response
