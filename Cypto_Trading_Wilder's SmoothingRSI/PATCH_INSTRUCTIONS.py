# FIXES FOR rsi_momentum_backtest_v5.py - v5.2
# ============================================
# 
# Apply these patches to fix the critical robustness analysis bugs.
# Run this AFTER backing up your original file.

"""
PATCH 1: Replace SignalGenerator class (lines ~312-370)
Status: ✓ ALREADY APPLIED
"""

"""
PATCH 2: Replace RobustnessAnalyzer class (lines ~1053-1750)
Status: NEEDS TO BE APPLIED

The complete fixed RobustnessAnalyzer class is too large to paste here.
Instead, the key fixes are:

1. run_monte_carlo_permutation_test: Use TRUE permutation of trade P&L
   - Extract backtest.trades['net_pnl'].values
   - For each simulation: shuffled_pnl = np.random.permutation(trade_pnl)
   - Rebuild equity: equity = 1000.0; for pnl in shuffled_pnl: equity += pnl
   - This tests if trade SEQUENCE matters (luck) vs edge

2. run_walk_forward_optimization: Add stability scoring
   - Track ALL parameter combinations across ALL windows
   - Select parameters by highest avg(OOS Sharpe) / (std(OOS Sharpe) + 0.01)
   - Return best_entry, best_exit based on OOS stability, not IS Sharpe

3. _print_monte_carlo_summary: Add verdict logic
   - If percentile > 95 AND |z_score| > 1.96: "STATISTICALLY SIGNIFICANT"
   - Elif percentile > 90 AND |z_score| > 1.64: "MARGINAL"
   - Else: "LIKELY LUCK"

4. _print_wfo_summary: Add stability verdict
   - If % windows with Sharpe > 1.0 > 50%: "ROBUST"
   - Elif > 25%: "MARGINAL"
   - Else: "UNSTABLE"
"""

"""
PATCH 3: Replace run_robustness_analysis function (lines ~1753-1850)
Status: NEEDS TO BE APPLIED

Key changes:
1. Run WFO and Monte Carlo for each symbol
2. Determine verdict based on MC percentile and WFO stability
3. Print comprehensive summary with verdicts per symbol
"""

"""
PATCH 4: Update VisualizationEngine methods (lines ~920-1050)
Status: NEEDS TO BE APPLIED

1. plot_walk_forward_results: Color OOS bars by Sharpe (green/yellow/red)
2. plot_monte_carlo_histogram: Show proper histogram with vertical lines
"""

# =============================================================================
# INSTRUCTIONS TO APPLY PATCHES
# =============================================================================

"""
Due to the large size of the changes, please:

1. The SignalGenerator has already been updated with bull filter ✓

2. For RobustnessAnalyzer, the complete fixed class code was provided in the
   previous response. Copy and replace lines 1053-1750.

3. For run_robustness_analysis, replace lines 1753-1850 with the updated
   version that includes verdict logic.

4. For VisualizationEngine, update the two methods as shown in previous response.

5. After applying all patches, run:
   python rsi_momentum_backtest_v5.py
   
6. Test Monte Carlo first - should show realistic std dev and Z-scores

Expected improvements:
- Monte Carlo: Std dev > 5%, Z-scores between -3 and +3
- WFO: Stability scores, % windows with Sharpe > 1.0
- Verdicts: Clear "Significant/Marginal/Luck" per symbol
- Bull filter: Better performance in 2024-2026 bull market
"""

# =============================================================================
# QUICK TEST COMMAND
# =============================================================================

"""
After applying patches, run this to test Monte Carlo only:

```python
from rsi_momentum_backtest_v5 import *
config = StrategyConfig()
analyzer = RobustnessAnalyzer(config)
mc_results = analyzer.run_monte_carlo_permutation_test('BNBUSDT', num_simulations=1000)
print(f"Std Dev: {mc_results['std_final_return']:.2f}%")
print(f"Z-Score: {mc_results['z_score']:.2f}")
print(f"Actual Percentile: {mc_results['actual_percentile']:.1f}%")
```

Expected output:
- Std Dev: 5-15% (NOT 0.01%)
- Z-Score: -3 to +3 (NOT ±500 to ±4000)
- Actual Percentile: 0-100% (meaningful distribution)
"""
