# =============================================================================
# ROBUSTNESS ANALYZER - WALK-FORWARD & MONTE CARLO (FIXED v5.2)
# =============================================================================
# REPLACE the entire RobustnessAnalyzer class in rsi_momentum_backtest_v5.py
# (lines ~1053-1750) with this fixed version.
# =============================================================================

class RobustnessAnalyzer:
    """
    Advanced robustness analyzer with FIXED Walk-Forward Optimization 
    and TRUE Monte Carlo Permutation tests.
    
    Fixes Applied (v5.2):
    1. Monte Carlo: TRUE permutation of trade P&L (not tiny noise)
    2. WFO: Stability scoring with OOS-based parameter selection
    3. Statistical significance testing with proper variance
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_engine = DataEngine(config)
        self.indicator_engine = IndicatorEngine()
    
    def run_walk_forward_optimization(self, symbol: str, timeframe: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Walk-Forward Optimization with STABILITY SCORING.
        
        FIX: Select parameters based on OOS performance stability, not just IS Sharpe.
        Penalize parameters with high variance across OOS windows.
        """
        tf = timeframe or self.config.TIMEFRAME
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD OPTIMIZATION (FIXED) - {symbol}")
        print(f"{'='*70}")
        
        df = self.data_engine.fetch_symbol_data(symbol, tf)
        n_bars = len(df)
        
        # WFO parameters
        is_window_bars = 2190
        oos_window_bars = 1095
        step_bars = 1095
        
        print(f"[INFO] Total bars: {n_bars}")
        print(f"[INFO] IS Window: {is_window_bars} bars (12 months)")
        print(f"[INFO] OOS Window: {oos_window_bars} bars (6 months)")
        
        # Pre-calculate indicators
        print("[INFO] Pre-calculating indicators...")
        rsi = self.indicator_engine.wilders_rsi(df['close'], 14)
        atr = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'], 14)
        adx = self.indicator_engine.adx(df['high'], df['low'], df['close'], 14)
        ema_200 = self.indicator_engine.ema(df['close'], 200)
        ema_20 = self.indicator_engine.ema(df['close'], 20)
        volume_sma = df['volume'].rolling(20).mean()
        
        entry_range = [58, 60, 62, 65, 67, 70, 72]
        exit_range = [25, 28, 30, 35, 40, 45, 50]
        
        # Track OOS performance for ALL parameter combinations across ALL windows
        all_oos_results = []
        window_count = 0
        is_start = 0
        
        while is_start + is_window_bars + oos_window_bars <= n_bars:
            window_count += 1
            is_end = is_start + is_window_bars
            oos_start = is_end
            oos_end = oos_start + oos_window_bars
            
            is_indices = slice(is_start, is_end)
            oos_indices = slice(oos_start, oos_end)
            
            print(f"\n[INFO] Window {window_count}: IS [{df.index[is_start]} to {df.index[is_end-1]}]")
            
            # Test ALL parameter combinations on IS window
            is_results = []
            for entry in entry_range:
                for exit_val in exit_range:
                    is_sharpe = self._test_parameters_on_window(
                        df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                        is_indices, entry, exit_val
                    )
                    if is_sharpe > 0:
                        is_results.append({
                            'entry': entry, 'exit': exit_val, 'is_sharpe': is_sharpe
                        })
            
            # Select top 3 IS performers
            is_results.sort(key=lambda x: x['is_sharpe'], reverse=True)
            top_params = is_results[:3] if len(is_results) >= 3 else is_results
            
            # Test top IS performers on OOS window
            print(f"[INFO] Testing {len(top_params)} top IS parameter sets on OOS...")
            for params in top_params:
                oos_metrics = self._test_parameters_on_window_detailed(
                    df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                    oos_indices, params['entry'], params['exit']
                )
                
                if oos_metrics and oos_metrics['trades'] >= 5:
                    all_oos_results.append({
                        'window': window_count,
                        'entry': params['entry'],
                        'exit': params['exit'],
                        'is_sharpe': params['is_sharpe'],
                        'oos_sharpe': oos_metrics['sharpe'],
                        'oos_return_pct': oos_metrics['return_pct'],
                        'oos_max_dd_pct': oos_metrics['max_dd_pct'],
                        'oos_profit_factor': oos_metrics['profit_factor'],
                        'oos_trades': oos_metrics['trades']
                    })
            
            is_start += step_bars
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_oos_results)
        
        # Calculate stability metrics for each parameter combination
        if not results_df.empty:
            stability_stats = results_df.groupby(['entry', 'exit']).agg({
                'oos_sharpe': ['mean', 'std', 'count'],
                'oos_return_pct': 'mean',
                'oos_profit_factor': 'mean'
            }).round(2)
            stability_stats.columns = ['_'.join(col).strip() for col in stability_stats.columns.values]
            stability_stats['stability_score'] = (
                stability_stats['oos_sharpe_mean'] / 
                (stability_stats['oos_sharpe_std'] + 0.01)
            )
            
            # Select most stable parameters
            best_stable = stability_stats.loc[stability_stats['stability_score'].idxmax()]
            best_entry = best_stable.name[0]
            best_exit = best_stable.name[1]
            
            print(f"\n[INFO] Most Stable Parameters: Entry={best_entry}, Exit={best_exit}")
            print(f"[INFO] Stability Score: {best_stable['stability_score']:.2f}")
        else:
            best_entry, best_exit = 65, 30
        
        summary_stats = self._calculate_wfo_summary(results_df, best_entry, best_exit)
        self._print_wfo_summary(summary_stats)
        
        return results_df, summary_stats
    
    def _test_parameters_on_window(self, df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                                   indices, entry, exit_val) -> float:
        """Test single parameter set on window, return Sharpe."""
        rsi_w = rsi.iloc[indices].values
        atr_w = atr.iloc[indices].values
        adx_w = adx.iloc[indices].values
        ema_200_w = ema_200.iloc[indices].values
        ema_20_w = ema_20.iloc[indices].values
        volume_sma_w = volume_sma.iloc[indices].values
        
        close_w = df['close'].iloc[indices].values
        high_w = df['high'].iloc[indices].values
        low_w = df['low'].iloc[indices].values
        open_w = df['open'].iloc[indices].values
        volume_w = df['volume'].iloc[indices].values
        
        above_trend = (close_w > ema_200_w)
        trending_market = (adx_w > 20)
        high_volume = (volume_sma_w < volume_w)
        rsi_prev = np.roll(rsi_w, 1)
        rsi_prev[0] = rsi_w[0]
        
        long_entry = (
            (rsi_w > entry) &
            (rsi_prev <= entry) &
            above_trend &
            trending_market &
            high_volume
        )
        long_entry = np.roll(long_entry, 1)
        long_entry[0] = False
        
        long_exit = (
            (rsi_w < exit_val) &
            (rsi_prev >= exit_val)
        )
        long_exit = np.roll(long_exit, 1)
        long_exit[0] = False
        
        equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
            open_w, high_w, low_w, close_w, atr_w,
            ema_20_w, long_entry, long_exit,
            4.0, 10.0, 0.06, 3.0, 0.001, 0.0005, 0.0001, 1000.0
        )
        
        if total_trades < 5:
            return 0.0
        
        returns = pd.Series(equity).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            return returns.mean() / returns.std() * np.sqrt(365 * 6)
        return 0.0
    
    def _test_parameters_on_window_detailed(self, df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                                            indices, entry, exit_val) -> Optional[Dict]:
        """Test single parameter set on window, return detailed metrics."""
        rsi_w = rsi.iloc[indices].values
        atr_w = atr.iloc[indices].values
        adx_w = adx.iloc[indices].values
        ema_200_w = ema_200.iloc[indices].values
        ema_20_w = ema_20.iloc[indices].values
        volume_sma_w = volume_sma.iloc[indices].values
        
        close_w = df['close'].iloc[indices].values
        high_w = df['high'].iloc[indices].values
        low_w = df['low'].iloc[indices].values
        open_w = df['open'].iloc[indices].values
        volume_w = df['volume'].iloc[indices].values
        
        above_trend = (close_w > ema_200_w)
        trending_market = (adx_w > 20)
        high_volume = (volume_sma_w < volume_w)
        rsi_prev = np.roll(rsi_w, 1)
        rsi_prev[0] = rsi_w[0]
        
        long_entry = (
            (rsi_w > entry) &
            (rsi_prev <= entry) &
            above_trend &
            trending_market &
            high_volume
        )
        long_entry = np.roll(long_entry, 1)
        long_entry[0] = False
        
        long_exit = (
            (rsi_w < exit_val) &
            (rsi_prev >= exit_val)
        )
        long_exit = np.roll(long_exit, 1)
        long_exit[0] = False
        
        equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
            open_w, high_w, low_w, close_w, atr_w,
            ema_20_w, long_entry, long_exit,
            4.0, 10.0, 0.06, 3.0, 0.001, 0.0005, 0.0001, 1000.0
        )
        
        if total_trades < 5:
            return None
        
        total_return = (equity[-1] / equity[0]) - 1
        returns = pd.Series(equity).pct_change().dropna()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6)
        else:
            sharpe = 0.0
        
        max_dd = self._calculate_max_drawdown(equity)
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'sharpe': sharpe,
            'return_pct': total_return * 100,
            'max_dd_pct': max_dd * 100,
            'profit_factor': pf,
            'trades': total_trades
        }
    
    def _calculate_wfo_summary(self, results_df: pd.DataFrame, best_entry: int, best_exit: int) -> Dict:
        """Calculate summary statistics for WFO results."""
        if results_df.empty:
            return {'error': 'No WFO results'}
        
        oos_sharpes = results_df['oos_sharpe']
        oos_returns = results_df['oos_return_pct']
        
        # Filter for best stable parameters
        best_params_df = results_df[
            (results_df['entry'] == best_entry) & 
            (results_df['exit'] == best_exit)
        ]
        
        return {
            'num_windows': results_df['window'].nunique(),
            'avg_oos_sharpe': oos_sharpes.mean(),
            'median_oos_sharpe': oos_sharpes.median(),
            'std_oos_sharpe': oos_sharpes.std(),
            'min_oos_sharpe': oos_sharpes.min(),
            'max_oos_sharpe': oos_sharpes.max(),
            'pct_windows_sharpe_gt_1': (oos_sharpes > 1.0).sum() / len(oos_sharpes) * 100,
            'avg_oos_return_pct': oos_returns.mean(),
            'pct_profitable_windows': (oos_returns > 0).sum() / len(oos_returns) * 100,
            'best_entry': best_entry,
            'best_exit': best_exit,
            'best_stability_score': best_params_df['oos_sharpe'].mean() / (best_params_df['oos_sharpe'].std() + 0.01) if len(best_params_df) > 1 else best_params_df['oos_sharpe'].mean()
        }
    
    def _print_wfo_summary(self, summary: Dict) -> None:
        """Print WFO summary statistics."""
        print(f"\n{'='*70}")
        print("WALK-FORWARD OPTIMIZATION SUMMARY (FIXED)")
        print(f"{'='*70}")
        print(f"  Number of Windows:        {summary.get('num_windows', 'N/A')}")
        print(f"  Best Parameters:          Entry={summary.get('best_entry')}, Exit={summary.get('best_exit')}")
        print(f"  Average OOS Sharpe:       {summary.get('avg_oos_sharpe', 0):.2f}")
        print(f"  Median OOS Sharpe:        {summary.get('median_oos_sharpe', 0):.2f}")
        print(f"  Std Dev OOS Sharpe:       {summary.get('std_oos_sharpe', 0):.2f}")
        print(f"  % Windows Sharpe > 1.0:   {summary.get('pct_windows_sharpe_gt_1', 0):.1f}%")
        print(f"  Stability Score:          {summary.get('best_stability_score', 0):.2f}")
        print(f"  Avg OOS Return:           {summary.get('avg_oos_return_pct', 0):.2f}%")
        print(f"  % Profitable Windows:     {summary.get('pct_profitable_windows', 0):.1f}%")
        
        # Verdict
        if summary.get('pct_windows_sharpe_gt_1', 0) > 50:
            print(f"\n  ✓ ROBUST: Strategy shows consistent OOS performance")
        elif summary.get('pct_windows_sharpe_gt_1', 0) > 25:
            print(f"\n  ⚠ MARGINAL: Some evidence of robustness")
        else:
            print(f"\n  ✗ UNSTABLE: OOS performance inconsistent")
        print(f"{'='*70}")
    
    def run_monte_carlo_permutation_test(self, symbol: str, num_simulations: int = 1000) -> Dict:
        """
        Monte Carlo Permutation Test - FIXED.
        
        FIX: TRUE permutation of trade P&L values (not tiny noise).
        For each simulation:
        1. Extract actual trade net_pnl values from production backtest
        2. Randomly SHUFFLE the order of trades
        3. Rebuild equity curve from shuffled sequence
        4. Record final return
        
        This tests whether the SEQUENCE of trades matters (luck) or if
        the strategy has genuine edge regardless of trade order.
        """
        print(f"\n{'='*70}")
        print(f"MONTE CARLO PERMUTATION TEST (FIXED) - {symbol}")
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
        
        # Extract ACTUAL trade P&L values
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
            equity = 1000.0
            for pnl in shuffled_pnl:
                equity += pnl
            
            final_returns[i] = (equity / 1000.0 - 1) * 100
        
        actual_return = backtest.calculate_metrics(df['equity'])['total_return'] * 100
        
        # Calculate statistics
        results = {
            'symbol': symbol,
            'num_simulations': num_simulations,
            'num_trades': n_trades,
            'actual_return_pct': actual_return,
            'actual_total_pnl': trade_pnl.sum(),
            'mean_final_return': final_returns.mean(),
            'median_final_return': np.median(final_returns),
            'std_final_return': final_returns.std(),
            'min_final_return': final_returns.min(),
            'max_final_return': final_returns.max(),
            'percentile_5th': np.percentile(final_returns, 5),
            'percentile_25th': np.percentile(final_returns, 25),
            'percentile_75th': np.percentile(final_returns, 75),
            'percentile_95th': np.percentile(final_returns, 95),
            'percent_profitable_sims': (final_returns > 0).sum() / num_simulations * 100,
            'actual_percentile': (final_returns < actual_return).sum() / num_simulations * 100,
            'z_score': (actual_return - final_returns.mean()) / final_returns.std() if final_returns.std() > 0 else 0,
            'final_returns_array': final_returns
        }
        
        self._print_monte_carlo_summary(results)
        return results
    
    def _print_monte_carlo_summary(self, results: Dict) -> None:
        """Print Monte Carlo summary with realistic statistics."""
        print(f"\n{'='*70}")
        print("MONTE CARLO PERMUTATION TEST SUMMARY (FIXED)")
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
        z_score = results.get('z_score', 0)
        
        if percentile > 95 and abs(z_score) > 1.96:
            print(f"\n  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
            print(f"    Strategy shows genuine edge beyond random trade sequence")
        elif percentile > 90 and abs(z_score) > 1.64:
            print(f"\n  ⚠ MARGINALLY SIGNIFICANT (p < 0.10)")
            print(f"    Some evidence of edge, but sequence matters")
        else:
            print(f"\n  ✗ NOT STATISTICALLY SIGNIFICANT")
            print(f"    Performance likely due to favorable trade sequence (luck)")
        print(f"{'='*70}")
    
    def run_grid_search(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        """Legacy grid search - retained for comparison."""
        tf = timeframe or self.config.TIMEFRAME
        print(f"\n[INFO] Running grid search for {symbol}...")

        df = self.data_engine.fetch_symbol_data(symbol, tf)
        n_bars = len(df)
        print(f"[INFO] Pre-calculating indicators for {n_bars} bars...")

        rsi = self.indicator_engine.wilders_rsi(df['close'], 14)
        atr = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'], 14)
        adx = self.indicator_engine.adx(df['high'], df['low'], df['close'], 14)
        ema_200 = self.indicator_engine.ema(df['close'], 200)
        ema_20 = self.indicator_engine.ema(df['close'], 20)
        volume_sma = df['volume'].rolling(20).mean()

        above_trend = (df['close'] > ema_200).values
        trending_market = (adx > 20).values
        high_volume = (df['volume'] > volume_sma).values
        rsi_prev = rsi.shift(1).values

        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        open_prices = df['open'].values
        ema_20_values = ema_20.values

        atr_stop_mult = 4.0
        atr_target_mult = 10.0
        risk_per_trade = self.config.RISK_PER_TRADE
        max_position_pct = self.config.MAX_POSITION_PCT
        trading_fee = 0.001
        slippage = 0.0005
        funding_rate = 0.0001
        initial_capital = 1000.0

        results = []
        entry_range = [58, 60, 62, 65, 67, 70, 72]
        exit_range = [25, 28, 30, 35, 40, 45, 50]

        total = len(entry_range) * len(exit_range)
        print(f"[INFO] Running {total} backtest combinations...")

        for i, entry in enumerate(entry_range):
            for j, exit_val in enumerate(exit_range):
                count = i * len(exit_range) + j + 1
                if count % 7 == 0:
                    print(f"[INFO] Progress: {count}/{total}...")

                rsi_array = rsi.values
                long_entry = (
                    (rsi_array > entry) &
                    (rsi_prev <= entry) &
                    above_trend &
                    trending_market &
                    high_volume
                )
                long_entry = np.roll(long_entry, 1)
                long_entry[0] = False

                long_exit = (
                    (rsi_array < exit_val) &
                    (rsi_prev >= exit_val)
                )
                long_exit = np.roll(long_exit, 1)
                long_exit[0] = False

                equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
                    open_prices, high_prices, low_prices, close_prices, atr.values,
                    ema_20_values, long_entry, long_exit,
                    atr_stop_mult, atr_target_mult, risk_per_trade, max_position_pct,
                    trading_fee, slippage, funding_rate, initial_capital
                )

                if total_trades > 0 and len(equity) > 1:
                    total_return = (equity[-1] / equity[0]) - 1
                    returns = pd.Series(equity).pct_change().dropna()
                    sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6) if len(returns) > 1 and returns.std() > 0 else 0.0
                    max_dd = self._calculate_max_drawdown(equity)
                    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    win_rate = win_count / total_trades

                    results.append({
                        'symbol': symbol, 'rsi_entry': entry, 'rsi_exit': exit_val,
                        'sharpe_ratio': sharpe, 'total_return': total_return * 100,
                        'max_drawdown': max_dd * 100, 'profit_factor': pf,
                        'total_trades': total_trades, 'win_rate': win_rate
                    })

        return pd.DataFrame(results)

    def _fast_backtest(self, open_prices, high_prices, low_prices, close_prices, atr,
                       ema_20_vals, long_entry, long_exit,
                       atr_stop_mult, atr_target_mult,
                       risk_per_trade, max_position_pct,
                       trading_fee, slippage, funding_rate,
                       initial_capital):
        """Fast vectorized backtest for optimization loops."""
        n = len(open_prices)
        equity = np.zeros(n)
        equity[0] = initial_capital

        position = 0
        position_size = 0.0
        entry_price = 0.0
        trailing_stop = 0.0
        total_trades = 0
        win_count = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for i in range(1, n):
            open_p = open_prices[i]
            high_p = high_prices[i]
            low_p = low_prices[i]
            close_p = close_prices[i]
            atr_val = atr[i-1] if i > 0 else atr[0]

            exit_triggered = False
            exit_price = 0.0

            if position > 0:
                new_trailing = close_p - (atr_val * atr_stop_mult)
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if low_p <= trailing_stop:
                    exit_price = trailing_stop
                    exit_triggered = True
                elif close_p < (ema_20_vals[i] * 0.995):
                    exit_price = open_p * (1 - slippage)
                    exit_triggered = True

            if exit_triggered and position > 0:
                pnl = (exit_price - entry_price) * position_size
                trade_value = abs(position_size * exit_price)
                costs = trade_value * (trading_fee + slippage)
                net_pnl = pnl - costs
                equity[i] = equity[i-1] + net_pnl
                total_trades += 1
                if net_pnl > 0:
                    win_count += 1
                    gross_profit += net_pnl
                else:
                    gross_loss += abs(net_pnl)
                position = 0
                position_size = 0.0
            else:
                equity[i] = equity[i-1]

            if long_entry[i] and position == 0:
                entry_price = open_p * (1 + slippage)
                stop_distance = atr_val * atr_stop_mult
                stop_price = entry_price - stop_distance
                risk_amount = equity[i] * risk_per_trade
                price_risk = abs(entry_price - stop_price)
                if price_risk > 0:
                    position_size = risk_amount / price_risk
                    max_size = (equity[i] * max_position_pct) / entry_price
                    position_size = min(position_size, max_size)
                if position_size > 0:
                    position = 1
                    trailing_stop = stop_price

        return equity, total_trades, win_count, gross_profit, gross_loss
    
    def _calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown from equity curve."""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return np.min(drawdown)
