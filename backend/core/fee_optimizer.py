import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class FeeOptimizer:
    """Dynamic fee optimization for trading decisions"""
    
    def __init__(self):
        self.min_trade_value = 100  # Minimum trade value in INR
        self.base_brokerage_rate = 0.0003  # 0.03%
        self.stt_rate = 0.00025  # Securities Transaction Tax
        self.exchange_txn_charge = 0.0000345  # Exchange Transaction Charges
        self.gst = 0.18  # GST on brokerage and transaction charges
        
    def calculate_optimal_trade_size(self, 
                                   current_price: float,
                                   expected_return: float,
                                   target_position_value: float,
                                   volatility: float) -> Dict[str, float]:
        """
        Calculate optimal trade size considering fees and expected return
        """
        try:
            # Calculate base fees for different position sizes
            position_sizes = np.linspace(0.5, 1.0, num=10) * target_position_value
            optimal_size = target_position_value
            best_net_return = float('-inf')
            
            for size in position_sizes:
                # Skip if below minimum trade value
                if size < self.min_trade_value:
                    continue
                    
                quantity = int(size / current_price)
                trade_value = quantity * current_price
                
                # Calculate total fees
                fees = self._calculate_total_fees(trade_value)
                
                # Calculate expected gross return
                gross_return = trade_value * expected_return
                
                # Calculate net return after fees
                net_return = gross_return - fees
                
                # Adjust for risk using volatility
                risk_adjusted_return = net_return / (1 + volatility)
                
                if risk_adjusted_return > best_net_return:
                    best_net_return = risk_adjusted_return
                    optimal_size = size
            
            # Calculate final metrics
            optimal_quantity = int(optimal_size / current_price)
            final_trade_value = optimal_quantity * current_price
            total_fees = self._calculate_total_fees(final_trade_value)
            
            return {
                "optimal_quantity": optimal_quantity,
                "trade_value": final_trade_value,
                "total_fees": total_fees,
                "expected_net_return": best_net_return
            }
            
        except Exception as e:
            logger.error(f"Error in fee optimization: {e}")
            return {
                "optimal_quantity": int(target_position_value / current_price),
                "trade_value": target_position_value,
                "total_fees": self._calculate_total_fees(target_position_value),
                "expected_net_return": 0
            }
    
    def _calculate_total_fees(self, trade_value: float) -> float:
        """Calculate total trading fees"""
        brokerage = min(trade_value * self.base_brokerage_rate, 20)  # Cap at Rs.20
        stt = trade_value * self.stt_rate
        exchange_charges = trade_value * self.exchange_txn_charge
        gst_amount = (brokerage + exchange_charges) * self.gst
        
        return brokerage + stt + exchange_charges + gst_amount
