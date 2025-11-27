#!/usr/bin/env python3
"""
MCP Scan All Tool
===============

Batch stock scanning tool for the Model Context Protocol server
with RL agent integration and standardized JSON responses.
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

logger = logging.getLogger(__name__)

# Request/Response logging
MCP_LOG_DIR = LOGS_DIR / "mcp_requests"
MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class ScanAllTool:
    """
    MCP Scan All Tool
    Batch scans all cached stocks and ranks by RL agent
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_id = config.get("tool_id", "scan_all_tool")
        self.rl_model_type = config.get("rl_model_type", "linucb")

        # Initialize stock analysis components
        self.ingester = EnhancedDataIngester()
        self.engineer = FeatureEngineer()
        self.request_counter = 0

        # Tool interconnections
        self.analyze_tool = None
        self.predict_tool = None

        logger.info(f"Scan All Tool {self.tool_id} initialized")

    def connect_tools(self, tool_registry: Dict[str, Any]):
        """Connect to other tools for interconnection"""
        if "analyze" in tool_registry:
            self.analyze_tool = tool_registry["analyze"]
        if "predict" in tool_registry:
            self.predict_tool = tool_registry["predict"]
        logger.info(f"Scan All Tool {self.tool_id} connected to other tools")

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

    def _scan_single_symbol(self, symbol: str, horizon: str, min_confidence: float, request_id: str) -> Dict[str, Any]:
        """Scan a single symbol"""
        try:
            logger.info(f"[{request_id}] Processing {symbol}...")

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
                    # Try to initialize data dynamically
                    try:
                        from utils.ml_components.stock_analysis_complete import _initialize_symbol_data
                        if _initialize_symbol_data(symbol, verbose=False):
                            logger.info(
                                f"[{request_id}] Data initialized for {symbol} via dynamic initialization")
                        else:
                            return None
                    except Exception as init_e:
                        logger.error(
                            f"[{request_id}] Failed to initialize data for {symbol}: {init_e}")
                        return None

            # STEP 2: Ensure features are calculated
            features_path = FEATURE_CACHE_DIR / f"{symbol}_features.json"
            if not features_path.exists():
                logger.info(
                    f"[{request_id}] Features not found for {symbol}. Calculating...")
                all_data = self.ingester.load_all_data(symbol)
                if all_data:
                    df = all_data.get('price_history')
                    if df is not None and not df.empty:
                        features_df = self.engineer.calculate_all_features(
                            df, symbol)
                        self.engineer.save_features(features_df, symbol)
                        logger.info(
                            f"[{request_id}] Features calculated for {symbol}")

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
                        # Try to train models dynamically
                        try:
                            from utils.ml_components.stock_analysis_complete import _train_symbol_models
                            if _train_symbol_models(symbol, horizon, verbose=False):
                                logger.info(
                                    f"[{request_id}] Models trained for {symbol} via dynamic training")
                            else:
                                return None
                        except Exception as train_e:
                            logger.error(
                                f"[{request_id}] Failed to train models for {symbol}: {train_e}")
                            return None
                    logger.info(
                        f"[{request_id}] Models trained for {symbol} ({horizon})")
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Training failed for {symbol}: {e}", exc_info=True)
                    # Try to train models dynamically
                    try:
                        from utils.ml_components.stock_analysis_complete import _train_symbol_models
                        if _train_symbol_models(symbol, horizon, verbose=False):
                            logger.info(
                                f"[{request_id}] Models trained for {symbol} via dynamic training")
                        else:
                            return None
                    except Exception as train_e:
                        logger.error(
                            f"[{request_id}] Failed to train models for {symbol}: {train_e}")
                        return None

            # STEP 4: Get prediction
            prediction = predict_stock_price(
                symbol, horizon=horizon, verbose=True)

            if prediction and prediction.get('confidence', 0) >= min_confidence:
                logger.info(f"[{request_id}] [OK] SHORTLIST: {symbol} "
                            f"({prediction['action']}, conf: {prediction['confidence']:.4f})")
                return prediction
            elif prediction:
                logger.info(f"[{request_id}] [FILTERED] {symbol} "
                            f"({prediction['action']}, conf: {prediction['confidence']:.4f} < {min_confidence})")
                return None
            else:
                logger.warning(
                    f"[{request_id}] [FAIL] {symbol}: No prediction returned")
                return None

        except Exception as e:
            logger.error(f"[{request_id}] Error scanning {symbol}: {e}")
            return None

    async def scan_all(self, arguments: Dict[str, Any], session_id: str) -> MCPToolResult:
        """
        Batch scan all stocks and generate ranked shortlist

        Args:
            arguments: Tool arguments containing scan parameters
            session_id: Session identifier

        Returns:
            MCPToolResult with scan results
        """
        start_time = time.time()

        try:
            # Extract parameters
            min_score = arguments.get("min_score", 0.5)
            max_results = arguments.get("max_results", 50)
            sectors = arguments.get("sectors", [])
            market_caps = arguments.get("market_caps", [])
            sort_by = arguments.get("sort_by", "score")
            horizon = arguments.get("horizon", "intraday")

            # Default symbols if none provided
            default_symbols = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
                "AXISBANK.NS", "MARUTI.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "TATAMOTORS.NS"
            ]

            symbols = arguments.get("symbols", default_symbols)

            request_data = {
                "symbols": symbols,
                "horizon": horizon,
                "min_score": min_score,
                "max_results": max_results
            }
            request_id = self._log_request("scan_all", request_data)

            # Process each symbol
            all_predictions = []
            shortlist = []
            failed_symbols = []
            low_confidence_symbols = []

            for symbol in symbols:
                prediction = self._scan_single_symbol(
                    symbol, horizon, min_score, request_id)
                if prediction:
                    all_predictions.append(prediction)
                    # Add to shortlist if meets confidence threshold
                    if prediction.get('confidence', 0) >= min_score:
                        shortlist.append(prediction)
                    else:
                        low_confidence_symbols.append({
                            "symbol": symbol,
                            "confidence": prediction.get('confidence', 0),
                            "action": prediction.get('action', 'UNKNOWN')
                        })
                else:
                    failed_symbols.append(symbol)

            # Sort shortlist by score (descending)
            shortlist.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Limit results if needed
            if max_results > 0:
                shortlist = shortlist[:max_results]

            result = {
                "metadata": {
                    "total_scanned": len(symbols),
                    "predictions_generated": len(all_predictions),
                    "shortlist_count": len(shortlist),
                    "failed_symbols_count": len(failed_symbols),
                    "low_confidence_count": len(low_confidence_symbols),
                    "horizon": horizon,
                    "min_confidence": min_score,
                    "timestamp": datetime.now().isoformat(),
                    "request_id": request_id
                },
                "shortlist": shortlist,
                "all_predictions": all_predictions,
                "diagnostics": {
                    "failed_symbols": failed_symbols,
                    # Limit to first 10 for brevity
                    "low_confidence_symbols": low_confidence_symbols[:10]
                }
            }

            # Calculate average confidence from shortlist
            confidence = 0.0
            if shortlist:
                valid_predictions = [p for p in shortlist if "confidence" in p]
                if valid_predictions:
                    confidence = sum(
                        p["confidence"] for p in valid_predictions) / len(valid_predictions)
            elif all_predictions:  # If no shortlist but we have some predictions
                valid_predictions = [
                    p for p in all_predictions if "confidence" in p]
                if valid_predictions:
                    confidence = sum(
                        p["confidence"] for p in valid_predictions) / len(valid_predictions)

            duration_ms = (time.time() - start_time) * 1000
            self._log_response(request_id, result, duration_ms)

            execution_time = time.time() - start_time

            # If no predictions were generated, provide diagnostic information
            if not all_predictions and not shortlist:
                logger.warning(f"[{request_id}] No predictions generated for any symbols. "
                               f"Failed symbols: {len(failed_symbols)}, "
                               f"Total symbols scanned: {len(symbols)}")

                # Return a result with diagnostic information
                result["diagnostics"]["issue_summary"] = (
                    "No predictions were generated. This typically happens when: "
                    "1) Stock data is unavailable or incomplete, "
                    "2) Models haven't been trained for these symbols, "
                    "3) Technical indicators cannot be calculated due to insufficient data, "
                    "4) All predictions fell below the confidence threshold. "
                    "The system will attempt to automatically fetch and process data for new symbols."
                )

            # If we have valid predictions, also generate analysis using the analyze tool
            valid_predictions = shortlist if shortlist else all_predictions
            if valid_predictions:
                try:
                    # Import analyze tool dynamically
                    from .analyze_tool import AnalyzeTool
                    analyze_tool = AnalyzeTool({
                        "tool_id": "analyze_tool_for_scan"
                    })

                    # Generate analysis for the predictions
                    analysis_arguments = {
                        # Limit to first 10 for performance
                        "predictions": valid_predictions[:10],
                        "analysis_depth": "comprehensive",
                        "include_risk_assessment": True
                    }

                    analysis_result = await analyze_tool.analyze(analysis_arguments, session_id)
                    if analysis_result.status == "SUCCESS":
                        result["analysis"] = analysis_result.data
                except Exception as analyze_error:
                    logger.warning(
                        f"Failed to generate analysis for scan results: {analyze_error}")

            return MCPToolResult(
                status="SUCCESS" if (
                    shortlist or all_predictions) else "WARNING",
                data=result,
                confidence=confidence,
                execution_time=execution_time,
                metadata={
                    "session_id": session_id,
                    "tool_id": self.tool_id,
                    "symbols_scanned": len(symbols),
                    "shortlist_count": len(shortlist),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Error in scan_all tool: {e}", exc_info=True)
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
SCAN_ALL_TOOL_AVAILABLE = True
