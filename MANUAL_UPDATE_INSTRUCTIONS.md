# Manual Update Instructions for professional_sell_logic.py

## File to Update
`backend/core/professional_sell_logic.py`

## Method to Replace
Find the `_check_immediate_exit_conditions` method (around line 506-531)

## Current Method (TO BE REPLACED)
```python
    def _check_immediate_exit_conditions(self, position: PositionMetrics, stop_levels: Dict) -> Optional[SellDecision]:
        """Check for immediate exit conditions (stop-loss, etc.)"""

        # Hard stop-loss hit
        if position.current_price <= stop_levels["active_stop"]:
            reason = SellReason.STOP_LOSS
            if stop_levels["profit_protection_stop"] and position.current_price <= stop_levels["profit_protection_stop"]:
                reason = SellReason.PROFIT_PROTECTION
            elif position.current_price <= stop_levels["trailing_stop"]:
                reason = SellReason.TRAILING_STOP

            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=reason,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=stop_levels["active_stop"],
                take_profit_price=0.0,
                trailing_stop_price=stop_levels["trailing_stop"],
                reasoning=f"Stop-loss triggered: {position.current_price:.2f} <= {stop_levels['active_stop']:.2f}"
            )

        return None
```

## New Method (REPLACE WITH THIS)
```python
    def _check_immediate_exit_conditions(self, position: PositionMetrics, stop_levels: Dict) -> Optional[SellDecision]:
        """Check for immediate exit conditions (stop-loss, target price, emergency loss, etc.)"""

        # PRIORITY 1: Emergency loss threshold (8-10% loss)
        if position.unrealized_pnl_pct <= -self.emergency_loss_threshold:
            logger.warning(f"🚨 EMERGENCY LOSS TRIGGERED: {position.unrealized_pnl_pct:.2%} loss exceeds threshold {-self.emergency_loss_threshold:.2%}")
            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=SellReason.RISK_MANAGEMENT,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=position.current_price,
                take_profit_price=0.0,
                trailing_stop_price=0.0,
                reasoning=f"EMERGENCY SELL: Loss {position.unrealized_pnl_pct:.2%} exceeds emergency threshold {-self.emergency_loss_threshold:.2%}"
            )
        
        # PRIORITY 2: Database stop-loss hit
        if position.db_stop_loss and position.current_price <= position.db_stop_loss:
            logger.warning(f"🛑 DATABASE STOP-LOSS HIT: Current price {position.current_price:.2f} <= DB stop-loss {position.db_stop_loss:.2f}")
            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=SellReason.STOP_LOSS,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=position.db_stop_loss,
                take_profit_price=position.db_target_price or 0.0,
                trailing_stop_price=stop_levels["trailing_stop"],
                reasoning=f"Database stop-loss triggered: {position.current_price:.2f} <= {position.db_stop_loss:.2f}"
            )
        
        # PRIORITY 3: Database target price hit
        if position.db_target_price and position.current_price >= position.db_target_price:
            logger.info(f"🎯 DATABASE TARGET PRICE HIT: Current price {position.current_price:.2f} >= DB target {position.db_target_price:.2f}")
            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=SellReason.TAKE_PROFIT,
                confidence=1.0,
                urgency=0.9,
                signals_triggered=[],
                stop_loss_price=position.db_stop_loss or stop_levels["active_stop"],
                take_profit_price=position.db_target_price,
                trailing_stop_price=stop_levels["trailing_stop"],
                reasoning=f"Database target price achieved: {position.current_price:.2f} >= {position.db_target_price:.2f}"
            )

        # PRIORITY 4: Hard stop-loss hit (calculated)
        if position.current_price <= stop_levels["active_stop"]:
            reason = SellReason.STOP_LOSS
            if stop_levels["profit_protection_stop"] and position.current_price <= stop_levels["profit_protection_stop"]:
                reason = SellReason.PROFIT_PROTECTION
            elif position.current_price <= stop_levels["trailing_stop"]:
                reason = SellReason.TRAILING_STOP

            return SellDecision(
                should_sell=True,
                sell_quantity=position.quantity,
                sell_percentage=1.0,
                reason=reason,
                confidence=1.0,
                urgency=1.0,
                signals_triggered=[],
                stop_loss_price=stop_levels["active_stop"],
                take_profit_price=0.0,
                trailing_stop_price=stop_levels["trailing_stop"],
                reasoning=f"Stop-loss triggered: {position.current_price:.2f} <= {stop_levels['active_stop']:.2f}"
            )

        return None
```

## Steps to Update

1. Open `backend/core/professional_sell_logic.py`
2. Find the `_check_immediate_exit_conditions` method (around line 506)
3. Select the entire method (from `def _check_immediate_exit_conditions` to the final `return None`)
4. Delete the selected method
5. Copy and paste the NEW METHOD from above
6. Save the file
7. Verify no indentation errors (method should be indented at class level - 4 spaces)

## What Changed

- **Added Priority 1**: Emergency loss threshold check (8-10% loss)
- **Added Priority 2**: Database stop-loss check
- **Added Priority 3**: Database target price check
- **Kept Priority 4**: Original calculated stop-loss logic (unchanged)

## Verification

After updating, verify:
- No indentation errors
- Method is properly indented inside the `ProfessionalSellLogic` class
- All return statements are properly aligned
- The method signature matches exactly

## Testing

After the update, test with:
1. A position with 10%+ loss (should trigger emergency sell)
2. A position with database stop-loss set (should respect it)
3. A position with database target price set (should take profit)
4. Normal positions (should use existing logic)
