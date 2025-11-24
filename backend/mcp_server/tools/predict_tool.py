#!/usr/bin/env python3
"""
MCP Predict Tool
===============

LightGBM + RL prediction tool for the Model Context Protocol server
with standardized JSON responses and Groq API integration.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

from ..mcp_trading_server import MCPToolResult
# Import stock analysis components
from utils.ml_components.stock_analysis_complete import (
    EnhancedDataIngester,
    FeatureEngineer,
    predict_stock_price,
    train_ml_models,
    DATA_CACHE_DIR,
    FEATURE_CACHE_DIR,
    MODEL_DIR,
    LOGS_DIR
)
import json
import time
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Request/Response logging
MCP_LOG_DIR = LOGS_DIR / "mcp_requests"
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class PredictTool:
    """
    MCP Predict Tool
    Generates stock price predictions using ensemble ML models
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "predict_tool")
        self.lightgbm_enabled = config.get("lightgbm_enabled", True)
        self.rl_model_type = config.get("rl_model_type", "linucb")

        # Initialize stock analysis components
        self.ingester = EnhancedDataIngester()
        self.engineer = FeatureEngineer()
        self.request_counter = 0

        logger.info(f"Predict Tool {self.tool_id} initialized")

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

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_requests.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.info(f"MCP Request [{request_id}]: {tool_name}")
        return request_id

    def _convert_to_json_serializable(self, obj):
        """Recursively convert numpy types and other non-JSON-serializable types to Python native types"""
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

        log_file = MCP_LOG_DIR / \
            f"{datetime.now().strftime('%Y%m%d')}_responses.jsonl"
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
                logger.error(
                    f"Failed to log response even with default=str: {e2}")

        logger.info(f"MCP Response [{request_id}]: {duration_ms:.2f}ms")

    def _predict_single_symbol(self, symbol: str, horizon: str, request_id: str) -> Dict[str, Any]:
        """Predict for a single symbol"""
        try:
            logger.info(f"[{request_id}] Predicting {symbol} ({horizon})")

            # STEP 1: Ensure data exists
            json_path = DATA_CACHE_DIR / f"{symbol}_all_data.json"

            if not json_path.exists():
                logger.info(
                    f"[{request_id}] Data not found for {symbol}. Fetching...")
                try:
                    self.ingester.fetch_all_data(symbol, period="2y")
                    logger.info(f"[{request_id}] Data fetched for {symbol}")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Failed to fetch data for {symbol}: {e}")
                    return {
                        "symbol": symbol,
                        "horizon": horizon,
                        "error": f"Data fetch failed: {str(e)}"
                    }

            # STEP 2: Ensure features are calculated
            features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
            if not features_path.exists():
                logger.info(
                    f"[{request_id}] Features not found for {symbol}. Calculating...")
                all_data = self.ingester.load_all_data(symbol)
                if all_data:
                    df = all_data.get('price_history')
                    # Fix: Check if df is a DataFrame and not empty, or if it's a dict check if it has data
                    if df is not None:
                        # Handle both DataFrame and dict cases
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            features_df = self.engineer.calculate_all_features(
                                df, symbol)
                            self.engineer.save_features(features_df, symbol)
                            logger.info(
                                f"[{request_id}] Features calculated for {symbol}")
                        elif isinstance(df, dict) and len(df) > 0:
                            # Convert dict to DataFrame if possible
                            try:
                                df_converted = pd.DataFrame(df)
                                if not df_converted.empty:
                                    features_df = self.engineer.calculate_all_features(
                                        df_converted, symbol)
                                    self.engineer.save_features(
                                        features_df, symbol)
                                    logger.info(
                                        f"[{request_id}] Features calculated for {symbol}")
                            except Exception as e:
                                logger.warning(
                                    f"[{request_id}] Could not convert dict to DataFrame for {symbol}: {e}")
                        elif isinstance(df, list) and len(df) > 0:
                            # Convert list to DataFrame if possible
                            try:
                                df_converted = pd.DataFrame(df)
                                if not df_converted.empty:
                                    features_df = self.engineer.calculate_all_features(
                                        df_converted, symbol)
                                    self.engineer.save_features(
                                        features_df, symbol)
                                    logger.info(
                                        f"[{request_id}] Features calculated for {symbol}")
                            except Exception as e:
                                logger.warning(
                                    f"[{request_id}] Could not convert list to DataFrame for {symbol}: {e}")
                        else:
                            logger.warning(
                                f"[{request_id}] No valid price history data for {symbol}")

            # STEP 3: Check if models exist for this horizon
            model_files = list(MODEL_DIR.glob(f"{symbol}_{horizon}_*"))

            if not model_files:
                logger.info(
                    f"[{request_id}] Models not found for {symbol} ({horizon}). Training...")
                try:
                    training_result = train_ml_models(
                        symbol, horizon, verbose=True)

                    # Handle both dict and bool return formats
                    success = training_result.get('success', False) if isinstance(
                        training_result, dict) else training_result

                    if not success:
                        logger.error(
                            f"[{request_id}] Training failed for {symbol}")
                        return {
                            "symbol": symbol,
                            "horizon": horizon,
                            "error": "Model training failed"
                        }
                    logger.info(
                        f"[{request_id}] Models trained for {symbol} ({horizon})")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Training failed for {symbol}: {e}", exc_info=True)
                    return {
                        "symbol": symbol,
                        "horizon": horizon,
                        "error": f"Training failed: {str(e)}"
                    }

            # STEP 4: Get prediction
            prediction = predict_stock_price(
                symbol, horizon=horizon, verbose=True)

            if prediction:
                logger.info(f"[{request_id}] [OK] {symbol}: {prediction['action']} "
                            f"(confidence: {prediction['confidence']:.4f})")
                return prediction
            else:
                logger.warning(
                    f"[{request_id}] [FAIL] {symbol}: No prediction returned")
                return {
                    "symbol": symbol,
                    "horizon": horizon,
                    "error": "Prediction failed - models may need training"
                }

        except Exception as e:
            logger.error(
                f"[{request_id}] Error predicting {symbol}: {e}", exc_info=True)
            return {
                "symbol": symbol,
                "horizon": horizon,
                "error": str(e)
            }

    async def predict(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Generate predictions for specified symbols

        Args:
            arguments: Tool arguments containing symbols and parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with predictions
        """
        start_time = time.time()

        try:
            # Extract parameters
            symbols = arguments.get("symbols", [])
            horizon = arguments.get("horizon", "intraday")
            risk_profile = arguments.get("risk_profile")

            if not symbols:
                return MCPToolResult(
                    status="ERROR",
                    error="No symbols provided",
                    metadata={
                        "session_id": session_id,
                        "tool_id": self.tool_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )

            request_data = {
                "symbols": symbols,
                "horizon": horizon,
                "risk_profile": risk_profile
            }
            request_id = self._log_request("predict", request_data)

            # Process each symbol
            predictions = []
            for symbol in symbols:
                prediction = self._predict_single_symbol(
                    symbol, horizon, request_id)
                predictions.append(prediction)

            # Determine risk profile from horizon if not specified
            if not risk_profile:
                risk_profiles = {
                    "intraday": "high",
                    "short": "moderate",
                    "long": "low"
                }
                risk_profile = risk_profiles.get(horizon, "moderate")

            result = {
                "metadata": {
                    "count": len(predictions),
                    "horizon": horizon,
                    "risk_profile": risk_profile,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "predictions": predictions
            }

            # Calculate average confidence
            confidence = 0.0
            if predictions:
                valid_predictions = [
                    p for p in predictions if "confidence" in p]
                if valid_predictions:
                    confidence = sum(
                        p["confidence"] for p in valid_predictions) / len(valid_predictions)

            # Convert numpy types to Python native types for JSON serialization
            sanitized_result = self._convert_to_json_serializable(result)
            sanitized_confidence = self._convert_to_json_serializable(
                confidence)

            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, sanitized_result, duration_ms)

            execution_time = time.time() - start_time

            return MCPToolResult(
                status="SUCCESS",
                data=sanitized_result,
                confidence=sanitized_confidence,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "horizon": horizon,
                    "symbols_count": len(symbols),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error in predict tool: {e}", exc_info=True)
            return MCPToolResult(
                status="ERROR",
                error=str(e),
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "timestamp": datetime.now().isoformat()
                }
            )


# Tool availability flag
PREDICT_TOOL_AVAILABLE = True
