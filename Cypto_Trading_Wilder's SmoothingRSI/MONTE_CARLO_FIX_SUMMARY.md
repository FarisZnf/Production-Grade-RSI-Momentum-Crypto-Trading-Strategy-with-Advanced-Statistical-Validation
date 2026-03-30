# Monte Carlo Fix - Summary

## Problem Identified

The Monte Carlo permutation test CANNOT work by shuffling trade order because:
1. **Addition is commutative**: sum(trade_pnl) = sum(shuffled_trade_pnl)
2. **Multiplication is commutative**: product(1+roi) = product(shuffled(1+roi))

This means ALL permutations give the SAME final return → Std Dev = 0

## Proper Monte Carlo Methods

### Method 1: Bootstrap Confidence Intervals (RECOMMENDED)
```python
# Resample trades WITH replacement
for i in range(num_simulations):
    # Sample n_trades randomly from original trades (with replacement)
    bootstrapped_trades = np.random.choice(trade_roi, size=n_trades, replace=True)
    # Compound
    equity = 1000.0
    for roi in bootstrapped_trades:
        equity *= (1 + roi)
    final_returns[i] = (equity / 1000.0 - 1) * 100
```

### Method 2: Add Execution Noise
```python
# Add noise to entry/exit prices (simulates slippage variation)
for i in range(num_simulations):
    noise = np.random.normal(0, 0.001, n_trades)  # 0.1% price noise
    noisy_roi = trade_roi + noise
    equity = 1000.0
    for roi in noisy_roi:
        equity *= (1 + roi)
    final_returns[i] = (equity / 1000.0 - 1) * 100
```

### Method 3: Random Trade Removal (Jackknife)
```python
# Randomly remove 10% of trades
for i in range(num_simulations):
    keep_mask = np.random.random(n_trades) > 0.1  # Keep 90% of trades
    reduced_trades = trade_roi[keep_mask]
    equity = 1000.0
    for roi in reduced_trades:
        equity *= (1 + roi)
    final_returns[i] = (equity / 1000.0 - 1) * 100
```

## Recommended Fix

Use **Method 1 (Bootstrap)** as it:
- Creates proper variance in final returns
- Tests strategy robustness to different trade sequences
- Provides meaningful confidence intervals
- Is statistically valid

## Files to Update

1. `rsi_momentum_backtest_v5.py` - Replace `run_monte_carlo_permutation_test` method
2. Update docstring to reflect bootstrap methodology
3. Update `_print_monte_carlo_summary` verdict logic

## Expected Results After Fix

```
Std Dev: 15-30%
Z-Score: -2 to +2
Actual Percentile: 20-80%
```

This shows the strategy has genuine edge if actual return is above the 75th percentile of bootstrap distribution.
