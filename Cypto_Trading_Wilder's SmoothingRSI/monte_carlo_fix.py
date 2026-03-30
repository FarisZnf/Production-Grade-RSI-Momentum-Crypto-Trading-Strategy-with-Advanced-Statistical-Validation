"""
MONTE CARLO FIX - Standalone Function
======================================
Copy this function and replace the existing run_monte_carlo_permutation_test
method in the RobustnessAnalyzer class (lines ~1306-1380).

This is the CRITICAL FIX that resolves the impossible Std Dev = 0.01% bug.
"""

def run_monte_carlo_permutation_test_FIXED(self, symbol: str, num_simulations: int = 1000) -> Dict:
    """
    Monte Carlo Permutation Test - FIXED v5.2
    
    CRITICAL FIX: TRUE permutation of trade P&L values (not tiny noise).
    
    The previous implementation added N(0, 0.001) noise to ROI % values,
    which caused all simulations to converge to the same value (Std Dev ≈ 0).
    
    This FIXED version:
    1. Extracts actual trade net_pnl values (dollars, not %)
    2. Randomly SHUFFLES the order of trades (TRUE permutation)
    3. Rebuilds equity by ADDING each trade P&L (correct compounding)
    4. Records final return distribution
    
    This tests whether the SEQUENCE of trades matters (luck) or if
    the strategy has genuine edge regardless of trade order.
    """
    print(f"\n{'='*70}")
    print(f"MONTE CARLO PERMUTATION TEST (FIXED v5.2) - {symbol}")
    print(f"{'='*70}")
    print(f"[INFO] Running {num_simulations} TRUE permutations...")
    
    # Run production backtest to get actual trades
    df = self.data_engine.fetch_symbol_data(symbol, self.config.TIMEFRAME)
    df = SignalGenerator(self.config).generate_signals(df)
    backtest = BacktestEngine(self.config)
    df = backtest.run_backtest(df, symbol)
    
    if backtest.trades.empty or len(backtest.trades) < 10:
        print(f"[WARNING] Insufficient trades for Monte Carlo test")
        return {'error': 'Insufficient trades'}
    
    # Extract ACTUAL trade P&L values (dollars, NOT ROI %)
    trade_pnl = backtest.trades['net_pnl'].values
    n_trades = len(trade_pnl)
    
    print(f"[INFO] Found {n_trades} trades for permutation")
    print(f"[INFO] Actual Total Net P&L: ${trade_pnl.sum():.2f}")
    
    # Run TRUE permutations
    np.random.seed(42)
    final_returns = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        if (i + 1) % 200 == 0:
            print(f"[INFO] Simulation {i + 1}/{num_simulations}...")
        
        # TRUE permutation: shuffle the ORDER of trades
        shuffled_pnl = np.random.permutation(trade_pnl)
        
        # Rebuild equity curve from shuffled trades
        # Start with initial capital, ADD each trade P&L sequentially
        equity = 1000.0
        for pnl in shuffled_pnl:
            equity += pnl  # ADD P&L (correct compounding)
        
        final_returns[i] = (equity / 1000.0 - 1) * 100
    
    actual_return = backtest.calculate_metrics(df['equity'])['total_return'] * 100
    
    # Calculate statistics
    results = {
        'symbol': symbol,
        'num_simulations': num_simulations,
        'num_trades': n_trades,
        'actual_return_pct': actual_return,
        'actual_total_pnl': float(trade_pnl.sum()),
        'mean_final_return': float(final_returns.mean()),
        'median_final_return': float(np.median(final_returns)),
        'std_final_return': float(final_returns.std()),
        'min_final_return': float(final_returns.min()),
        'max_final_return': float(final_returns.max()),
        'percentile_5th': float(np.percentile(final_returns, 5)),
        'percentile_25th': float(np.percentile(final_returns, 25)),
        'percentile_75th': float(np.percentile(final_returns, 75)),
        'percentile_95th': float(np.percentile(final_returns, 95)),
        'percent_profitable_sims': float((final_returns > 0).sum() / num_simulations * 100),
        'actual_percentile': float((final_returns < actual_return).sum() / num_simulations * 100),
        'z_score': float((actual_return - final_returns.mean()) / final_returns.std()) if final_returns.std() > 0 else 0,
        'final_returns_array': final_returns
    }
    
    self._print_monte_carlo_summary(results)
    return results


def _print_monte_carlo_summary_FIXED(self, results: Dict) -> None:
    """Print Monte Carlo summary with realistic statistics."""
    print(f"\n{'='*70}")
    print("MONTE CARLO PERMUTATION TEST SUMMARY (FIXED v5.2)")
    print(f"{'='*70}")
    print(f"  Number of Simulations:    {results.get('num_simulations', 'N/A')}")
    print(f"  Number of Trades:         {results.get('num_trades', 'N/A')}")
    print(f"  Actual Total P&L:         ${results.get('actual_total_pnl', 0):.2f}")
    print(f"  Actual Return:            {results.get('actual_return_pct', 0):.2f}%")
    print(f"  Mean Simulated Return:    {results.get('mean_final_return', 0):.2f}%")
    print(f"  Median Simulated Return:  {results.get('median_final_return', 0):.2f}%")
    print(f"  Std Dev Simulated Return: {results.get('std_final_return', 0):.2f}%")
    print(f"  5th Percentile (Worst):   {results.get('percentile_5th', 0):.2f}%")
    print(f"  25th Percentile:          {results.get('percentile_25th', 0):.2f}%")
    print(f"  75th Percentile:          {results.get('percentile_75th', 0):.2f}%")
    print(f"  95th Percentile (Best):   {results.get('percentile_95th', 0):.2f}%")
    print(f"  % Profitable Simulations: {results.get('percent_profitable_sims', 0):.1f}%")
    print(f"  Actual Return Percentile: {results.get('actual_percentile', 0):.1f}%")
    print(f"  Z-Score:                  {results.get('z_score', 0):.2f}")
    
    # Statistical significance verdict
    percentile = results.get('actual_percentile', 0)
    z_score = abs(results.get('z_score', 0))
    
    if percentile > 95 and z_score > 1.96:
        print(f"\n  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"    Strategy shows genuine edge beyond random trade sequence")
    elif percentile > 90 and z_score > 1.64:
        print(f"\n  ⚠ MARGINALLY SIGNIFICANT (p < 0.10)")
        print(f"    Some evidence of edge, but sequence matters")
    else:
        print(f"\n  ✗ NOT STATISTICALLY SIGNIFICANT")
        print(f"    Performance likely due to favorable trade sequence (luck)")
    print(f"{'='*70}")
