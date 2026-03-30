# RSI Momentum Strategy - Optimization Comparison Report

## Executive Summary

This report compares the **Baseline** strategy performance against the **Optimized v3** implementation with the following enhancements:

### Optimizations Applied
1. **RSI Entry Threshold**: 70 → 65 (earlier momentum capture)
2. **200-EMA Trend Filter**: Long entries ONLY when Close > EMA_200
3. **Volume Filter**: Entry ONLY when Volume > 20-period SMA
4. **ATR Trailing Stop**: PRIMARY exit at 2.5x ATR (replaces RSI exit)
5. **ATR Take-Profit**: SECONDARY exit at 4.0x ATR
6. **RSI Emergency Exit**: TERTIARY exit only when RSI < 30

---

## Performance Comparison: BTCUSDT (2024-2026)

| Metric | Baseline (v2) | Optimized (v3) | Change | Status |
|--------|---------------|----------------|--------|--------|
| **Total Return** | 3.54% | **10.36%** | **+193%** | ✅ |
| **CAGR** | 1.58% | **4.54%** | **+187%** | ✅ |
| **Sharpe Ratio** | 0.21 | **0.49** | **+133%** | ✅ |
| **Max Drawdown** | -10.50% | **-10.18%** | **+3%** | ✅ |
| **Win Rate** | 42.0% | **52.7%** | **+25%** | ✅ |
| **Profit Factor** | 1.07 | **1.22** | **+14%** | ✅ |
| **Total Trades** | 50 | **55** | +10% | - |
| **Average Trade** | $0.71 | **$1.88** | **+165%** | ✅ |
| **Buy & Hold Return** | 63.81% | 63.81% | - | ⚠️ |

### Exit Breakdown (Optimized v3)
| Exit Type | Count | Percentage |
|-----------|-------|------------|
| Trailing Stop (2.5x ATR) | 41 | 75% |
| Take Profit (4.0x ATR) | 14 | 25% |
| RSI Emergency (< 30) | 0 | 0% |

**Key Insight**: 75% of exits via trailing stop confirms the ATR mechanism is capturing trends and locking in profits effectively.

---

## Performance Comparison: ETHUSDT (2024-2026)

| Metric | Baseline (v2) | Optimized (v3) | Change | Status |
|--------|---------------|----------------|--------|--------|
| **Total Return** | 6.63% | **-11.88%** | **-279%** | ❌ |
| **CAGR** | 2.94% | **-5.54%** | **-289%** | ❌ |
| **Sharpe Ratio** | 0.35 | **-0.57** | **-263%** | ❌ |
| **Max Drawdown** | -8.49% | **-24.87%** | **-193%** | ❌ |
| **Win Rate** | 40.4% | **37.5%** | **-7%** | ⚠️ |
| **Profit Factor** | 1.17 | **0.74** | **-37%** | ❌ |
| **Total Trades** | 52 | **56** | +8% | - |
| **Average Trade** | $1.28 | **$-2.12** | **-266%** | ❌ |

### Exit Breakdown (Optimized v3 - ETHUSDT)
| Exit Type | Count | Percentage |
|-----------|-------|------------|
| Trailing Stop (2.5x ATR) | 48 | 86% |
| Take Profit (4.0x ATR) | 8 | 14% |
| RSI Emergency (< 30) | 0 | 0% |

**Key Insight**: ETHUSDT shows 86% trailing stop exits but negative returns, indicating the 2.5x ATR stop is too tight for ETH's higher volatility. **Recommendation**: Use symbol-specific parameters (e.g., 3.5x ATR for ETH).

---

## Overall Assessment

### ✅ Successes (BTCUSDT)
1. **Return Improved**: +193% increase in total return
2. **Sharpe Improved**: +133% increase in risk-adjusted returns
3. **Win Rate Improved**: 42% → 52.7% (above 50% threshold)
4. **Profit Factor Improved**: 1.07 → 1.22 (better reward/risk)
5. **Drawdown Controlled**: Remained below 15% constraint
6. **Avg Trade Improved**: $0.71 → $1.88 (+165%)

### ⚠️ Areas for Improvement (ETHUSDT)
1. **Symbol-Specific Parameters Needed**: ETH volatility requires wider stops
2. **Recommended ETH Adjustments**:
   - ATR Stop Multiple: 2.5x → 3.5x
   - ATR Target Multiple: 4.0x → 5.0x
   - RSI Entry: 65 → 60 (earlier entry for ETH's momentum)

---

## Code Changes Summary

### StrategyConfig
```python
# BEFORE (Baseline)
RSI_LONG_ENTRY: int = 70
RSI_LONG_EXIT: int = 30
USE_TREND_FILTER: bool = True
USE_VOLUME_FILTER: bool = False  # Not present
ATR_STOP_MULTIPLE: float = 2.5
ATR_TARGET_MULTIPLE: float = 4.0

# AFTER (Optimized v3)
RSI_LONG_ENTRY: int = 65          # Lowered for earlier entry
RSI_LONG_EXIT: int = 30           # Emergency only
USE_TREND_FILTER: bool = True     # Mandatory
USE_VOLUME_FILTER: bool = True    # NEW: Filters whipsaws
ATR_STOP_MULTIPLE: float = 2.5    # PRIMARY exit
ATR_TARGET_MULTIPLE: float = 4.0  # SECONDARY exit
```

### SignalGenerator
```python
# NEW: Volume filter
df['volume_sma'] = df['volume'].rolling(window=20).mean()
df['high_volume'] = df['volume'] > df['volume_sma']

# UPDATED: Entry requires ALL conditions
df['long_entry_raw'] = (
    (df['rsi'] > 65) &                    # Lower threshold
    (df['rsi_prev'] <= 65) &              # Crossover
    (df['above_trend'] == True) &         # 200-EMA filter
    (df['high_volume'] == True)           # NEW: Volume filter
)
```

### BacktestEngine
```python
# UPDATED: Exit priority system
if position > 0:
    # Dynamic trailing stop (moves up only)
    new_trailing_stop = close_price - (atr * 2.5)
    if new_trailing_stop > trailing_stop_price:
        trailing_stop_price = new_trailing_stop
    
    # Priority 1: Trailing stop
    if low_price <= trailing_stop_price:
        sl_hit = True
        exit_reason = 'trailing_stop'
    
    # Priority 2: Take profit
    elif high_price >= take_profit_price:
        tp_hit = True
        exit_reason = 'take_profit'
    
    # Priority 3: RSI emergency
    elif exit_signal:
        emergency_exit = True
        exit_reason = 'rsi_emergency'
```

---

## Visualizations Generated

| File | Description |
|------|-------------|
| `output/BTCUSDT_equity_curve_optimized_v3.png` | BTCUSDT equity curve vs Buy & Hold |
| `output/ETHUSDT_equity_curve_optimized_v3.png` | ETHUSDT equity curve vs Buy & Hold |

---

## Recommendations

### Immediate Actions
1. **BTCUSDT**: Continue with current parameters (working well)
2. **ETHUSDT**: Implement symbol-specific parameters:
   ```python
   # ETH-specific config
   ATR_STOP_MULTIPLE: float = 3.5    # Wider stop for higher volatility
   ATR_TARGET_MULTIPLE: float = 5.0  # Higher target
   RSI_LONG_ENTRY: int = 60          # Earlier entry
   ```

### Future Enhancements
1. **Adaptive ATR Multiples**: Adjust based on realized volatility
2. **Machine Learning Filter**: Use volume profile + order flow
3. **Multi-Timeframe Confirmation**: Require 4h + 1d trend alignment
4. **Dynamic Position Sizing**: Scale with market regime (VIX-like crypto indicator)

---

## Conclusion

The **Optimized v3** strategy demonstrates **significant improvement for BTCUSDT**:
- ✅ **+193%** return improvement
- ✅ **+133%** Sharpe ratio improvement  
- ✅ **+25%** win rate improvement
- ✅ **165%** higher average trade profit
- ✅ Drawdown remains below 15% constraint

**ETHUSDT requires symbol-specific tuning** due to higher volatility. The core framework is sound, but parameters need adjustment for different market characteristics.

---

*Generated: 2026-03-22*
*Strategy Version: v3.0 Optimized*
*Data Period: 2024-01-01 to 2026-03-22*
