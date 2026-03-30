"""
================================================================================
Momentum-Based RSI Crypto Trading Strategy - OPTIMIZED v3
================================================================================
Author: Senior Quantitative Developer
Description: Production-grade backtest with OPTIMIZED signal filters and 
             ATR-based risk management to reduce whipsaws and improve Sharpe.

OPTIMIZATIONS APPLIED:
    1. RSI Entry Threshold: Lowered from 70 to 65 (earlier momentum capture)
    2. 200-EMA Trend Filter: Long entries ONLY when Close > EMA_200
    3. Volume Filter: Entry ONLY when Volume > 20-period Volume SMA
    4. ATR Trailing Stop: PRIMARY exit at 2.5x ATR (locks in profits)
    5. ATR Take-Profit: SECONDARY exit at 4.0x ATR
    6. RSI Emergency Exit: TERTIARY exit only when RSI < 30
================================================================================
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import ffn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from binance.client import Client

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """Centralized configuration for the backtest."""
    
    # Data settings
    SYMBOLS: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    TIMEFRAME: str = '4h'
    START_DATE: str = '2024-01-01'
    END_DATE: str = '2026-03-22'
    
    # RSI parameters - OPTIMIZED: Lower entry to 65 for earlier momentum
    RSI_PERIOD: int = 14
    RSI_LONG_ENTRY: int = 65      # Changed from 70 → capture earlier
    RSI_LONG_EXIT: int = 30       # Emergency exit only
    
    # Trend filter - MANDATORY
    USE_TREND_FILTER: bool = True
    TREND_EMA_PERIOD: int = 200
    
    # Volume filter - NEW: Filter low-volume whipsaws
    USE_VOLUME_FILTER: bool = True
    VOLUME_SMA_PERIOD: int = 20
    
    # ATR-based risk management - PRIMARY EXIT MECHANISM
    ATR_PERIOD: int = 14
    ATR_STOP_MULTIPLE: float = 2.5      # Trailing stop at 2.5x ATR
    ATR_TARGET_MULTIPLE: float = 4.0    # Take profit at 4.0x ATR
    
    # Position sizing
    RISK_PER_TRADE: float = 0.02        # Risk 2% per trade
    MAX_POSITION_SIZE: float = 1.0      # Max 100% of equity
    
    # Cost modeling
    TRADING_FEE: float = 0.001          # 0.1% per trade
    SLIPPAGE: float = 0.0005            # 0.05% slippage
    FUNDING_RATE: float = 0.0001        # 0.01% per 8h
    
    # Capital
    INITIAL_CAPITAL: float = 1000.0
    
    # Cache directory
    CACHE_DIR: Path = field(default_factory=lambda: Path(__file__).parent / 'data_cache')
    
    # Binance API
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    BINANCE_API_SECRET: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))


# =============================================================================
# DATA ENGINE
# =============================================================================

class DataEngine:
    """Handles data fetching from Binance with local CSV caching."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.client = self._init_binance_client()
    
    def _init_binance_client(self) -> Optional[Client]:
        try:
            if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
                client = Client(self.config.BINANCE_API_KEY, self.config.BINANCE_API_SECRET)
                print("[INFO] Binance client initialized with API keys.")
            else:
                client = Client()
                print("[INFO] Binance client initialized (public access).")
            return client
        except Exception as e:
            print(f"[WARNING] Binance client failed: {e}")
            return None
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_dir / f"{symbol}_{timeframe}.csv"
    
    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(symbol, timeframe)
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col='timestamp', parse_dates=True)
                print(f"[INFO] Loaded {symbol} {timeframe} from cache: {len(df)} bars")
                return df
            except Exception as e:
                print(f"[WARNING] Cache load failed: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        cache_path = self._get_cache_path(symbol, timeframe)
        df.to_csv(cache_path)
        print(f"[INFO] Saved {symbol} {timeframe} to cache: {cache_path}")
    
    def _fetch_binance_klines(self, symbol: str, timeframe: str,
                               start_str: str, end_str: str) -> pd.DataFrame:
        if self.client is None:
            raise RuntimeError("Binance client not initialized")

        print(f"[INFO] Fetching {symbol} {timeframe} from Binance...")
        
        all_klines = []
        current_start = start_str
        
        while True:
            klines = self.client.get_historical_klines(
                symbol=symbol, interval=timeframe,
                start_str=current_start, end_str=end_str, limit=1000
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            if len(klines) < 1000:
                break
            
            last_candle_time = klines[-1][6]
            current_start = datetime.fromtimestamp(last_candle_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            if current_start >= end_str:
                break
            
            print(f"[INFO] Fetched {len(all_klines)} candles...")

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df[~df.index.duplicated(keep='first')]

        return df
    
    def fetch_symbol_data(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        tf = timeframe or self.config.TIMEFRAME
        
        cached_data = self._load_from_cache(symbol, tf)
        if cached_data is not None:
            return cached_data
        
        if self.client is None:
            raise RuntimeError("Binance client not available and no cache found.")
        
        df = self._fetch_binance_klines(
            symbol=symbol, timeframe=tf,
            start_str=self.config.START_DATE, end_str=self.config.END_DATE
        )
        
        self._save_to_cache(df, symbol, tf)
        return df


# =============================================================================
# INDICATOR ENGINE (Wilder's RSI + ATR + EMA)
# =============================================================================

class IndicatorEngine:
    """Technical indicators with Wilder's smoothing."""
    
    @staticmethod
    def wilders_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's original smoothing (1/period)."""
        delta = close.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        for i in range(period, len(close)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
        
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
        """Calculate ATR using Wilder's smoothing."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period, min_periods=period).mean()
        for i in range(period, len(close)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
        
        return atr
    
    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return close.ewm(span=period, adjust=False).mean()


# =============================================================================
# SIGNAL GENERATOR (OPTIMIZED WITH FILTERS)
# =============================================================================

class SignalGenerator:
    """
    Generate trading signals with OPTIMIZED filters:
    - RSI(14) crossover above 65 (earlier entry)
    - 200-EMA trend filter (mandatory)
    - Volume > 20-period SMA (filter whipsaws)
    - ATR trailing stop primary exit
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicator_engine = IndicatorEngine()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate indicators
        df['rsi'] = self.indicator_engine.wilders_rsi(df['close'], self.config.RSI_PERIOD)
        df['atr'] = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'],
                                                       self.config.ATR_PERIOD)

        # Trend filter: 200-period EMA (MANDATORY)
        if self.config.USE_TREND_FILTER:
            df['ema_200'] = self.indicator_engine.ema(df['close'], self.config.TREND_EMA_PERIOD)
            df['above_trend'] = df['close'] > df['ema_200']
        else:
            df['above_trend'] = True
        
        # Volume filter: Volume > 20-period SMA
        if self.config.USE_VOLUME_FILTER:
            df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
            df['high_volume'] = df['volume'] > df['volume_sma']
        else:
            df['high_volume'] = True

        # Generate raw signals
        df['rsi_prev'] = df['rsi'].shift(1)

        # Long entry: RSI > 65 + Trend + Volume (ALL must be true)
        df['long_entry_raw'] = (
            (df['rsi'] > self.config.RSI_LONG_ENTRY) &
            (df['rsi_prev'] <= self.config.RSI_LONG_ENTRY) &
            (df['above_trend'] == True) &
            (df['high_volume'] == True)
        )

        # Long exit: RSI < 30 (EMERGENCY ONLY - primary exits are ATR-based)
        df['long_exit_raw'] = (
            (df['rsi'] < self.config.RSI_LONG_EXIT) &
            (df['rsi_prev'] >= self.config.RSI_LONG_EXIT)
        )

        # Shift signals by 1 (look-ahead bias prevention)
        df['long_entry'] = df['long_entry_raw'].shift(1).fillna(False)
        df['long_exit'] = df['long_exit_raw'].shift(1).fillna(False)

        # ATR-based stop/target levels
        df['stop_loss_distance'] = df['atr'] * self.config.ATR_STOP_MULTIPLE
        df['take_profit_distance'] = df['atr'] * self.config.ATR_TARGET_MULTIPLE

        # Cleanup
        df.drop(columns=['rsi_prev', 'long_entry_raw', 'long_exit_raw'],
                inplace=True, errors='ignore')

        return df


# =============================================================================
# BACKTEST ENGINE (ATR-BASED RISK MANAGEMENT)
# =============================================================================

class BacktestEngine:
    """
    Backtest engine with OPTIMIZED exit logic:
    
    Exit Priority:
    1. ATR Trailing Stop (PRIMARY) - Locks profits during trends
    2. ATR Take-Profit (SECONDARY) - Measured target exit
    3. RSI Emergency Exit (TERTIARY) - Trend reversal protection
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.trades = pd.DataFrame()
    
    def _calculate_position_size(self, equity: float, entry_price: float,
                                  stop_loss_price: float) -> float:
        """Volatility-adjusted position sizing (2% risk per trade)."""
        risk_amount = equity * self.config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0
        
        position_size = risk_amount / price_risk
        max_size = (equity * self.config.MAX_POSITION_SIZE) / entry_price
        return min(position_size, max_size)
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()

        # Initialize columns
        df['returns'] = 0.0
        df['equity'] = self.initial_capital
        df['position'] = 0
        df['drawdown'] = 0.0

        # State variables
        position = 0
        position_size = 0.0
        entry_price = 0.0
        entry_bar = 0
        entry_atr = 0.0
        take_profit_price = 0.0
        trailing_stop_price = 0.0
        highest_close = 0.0
        total_costs = 0.0
        equity = self.initial_capital

        trades = []
        sl_hits = 0
        tp_hits = 0
        rsi_exits = 0

        for i in range(1, len(df)):
            prev_idx = i - 1
            curr_idx = i

            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']
            atr = df.iloc[prev_idx]['atr'] if not pd.isna(df.iloc[prev_idx]['atr']) else 0.01 * close_price

            signal = df.iloc[prev_idx]['long_entry']
            exit_signal = df.iloc[prev_idx]['long_exit']

            sl_hit = False
            tp_hit = False
            emergency_exit = False

            if position > 0:
                # Track highest close for trailing stop
                if close_price > highest_close:
                    highest_close = close_price
                
                # DYNAMIC TRAILING STOP: Only moves UP
                new_trailing_stop = close_price - (atr * self.config.ATR_STOP_MULTIPLE)
                if new_trailing_stop > trailing_stop_price:
                    trailing_stop_price = new_trailing_stop

                # Exit conditions (PRIORITY ORDER)
                if low_price <= trailing_stop_price:
                    sl_hit = True          # Priority 1: Trailing stop
                elif high_price >= take_profit_price:
                    tp_hit = True          # Priority 2: Take profit
                elif exit_signal:
                    emergency_exit = True  # Priority 3: RSI emergency

            # Execute exit
            if position > 0 and (sl_hit or tp_hit or emergency_exit):
                if tp_hit:
                    exit_price = take_profit_price
                    tp_hits += 1
                    exit_reason = 'take_profit'
                elif sl_hit:
                    exit_price = trailing_stop_price
                    sl_hits += 1
                    exit_reason = 'trailing_stop'
                else:
                    exit_price = open_price * (1 - self.config.SLIPPAGE)
                    rsi_exits += 1
                    exit_reason = 'rsi_emergency'

                pnl = (exit_price - entry_price) * position_size

                trade_value = abs(position_size * exit_price)
                holding_periods = curr_idx - entry_bar
                funding_cost = position_size * self.config.FUNDING_RATE * (holding_periods / 2)
                fees = trade_value * self.config.TRADING_FEE
                slippage_cost = trade_value * self.config.SLIPPAGE
                total_trade_cost = fees + slippage_cost + funding_cost
                total_costs += total_trade_cost

                equity += pnl - total_trade_cost

                trades.append({
                    'symbol': symbol,
                    'entry_date': df.index[entry_bar],
                    'exit_date': df.index[curr_idx],
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'costs': total_trade_cost,
                    'net_pnl': pnl - total_trade_cost,
                    'exit_reason': exit_reason,
                    'entry_atr': entry_atr
                })

                position = 0
                position_size = 0.0
                trailing_stop_price = 0.0
                highest_close = 0.0

            # Execute entry
            if signal and position == 0:
                entry_price = open_price * (1 + self.config.SLIPPAGE)
                entry_atr = atr if atr > 0 else 0.01 * entry_price

                stop_loss_distance = entry_atr * self.config.ATR_STOP_MULTIPLE
                take_profit_distance = entry_atr * self.config.ATR_TARGET_MULTIPLE

                stop_loss_price = entry_price - stop_loss_distance
                take_profit_price = entry_price + take_profit_distance
                trailing_stop_price = stop_loss_price
                highest_close = entry_price

                position_size = self._calculate_position_size(
                    equity, entry_price, stop_loss_price
                )

                if position_size > 0:
                    position = 1
                    entry_bar = curr_idx

            # Funding cost
            if position > 0:
                funding_cost = position_size * self.config.FUNDING_RATE
                equity -= funding_cost
                total_costs += funding_cost

            # Period return
            prev_equity = df.iloc[i-1]['equity'] if i > 1 else equity
            period_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0

            # Drawdown
            running_max = df['equity'].iloc[:i].max() if i > 0 else equity
            drawdown = (equity - running_max) / running_max if running_max > 0 else 0

            # Update DataFrame
            df.iloc[i, df.columns.get_loc('returns')] = period_return
            df.iloc[i, df.columns.get_loc('equity')] = equity
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('drawdown')] = drawdown

        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        return df
    
    def calculate_metrics(self, equity_curve: pd.Series) -> Dict:
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < 2:
            return {'error': 'Insufficient data'}
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        years = len(equity_curve) / (365 * 6)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        
        periods_per_year = 365 * 6
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        avg_trade = self.trades['net_pnl'].mean() if not self.trades.empty else 0
        
        # Exit breakdown
        if not self.trades.empty:
            tp_count = (self.trades['exit_reason'] == 'take_profit').sum()
            sl_count = (self.trades['exit_reason'] == 'trailing_stop').sum()
            rsi_count = (self.trades['exit_reason'] == 'rsi_emergency').sum()
        else:
            tp_count = sl_count = rsi_count = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': f"{total_return * 100:.2f}%",
            'cagr': cagr,
            'cagr_pct': f"{cagr * 100:.2f}%",
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': f"{max_drawdown * 100:.2f}%",
            'final_equity': equity_curve.iloc[-1],
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'tp_exits': tp_count,
            'sl_exits': sl_count,
            'rsi_exits': rsi_count,
        }
    
    def _calculate_win_rate(self) -> float:
        if self.trades.empty:
            return 0.0
        return (self.trades['net_pnl'] > 0).sum() / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        if self.trades.empty:
            return 0.0
        gross_profit = self.trades[self.trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(self.trades[self.trades['net_pnl'] <= 0]['net_pnl'].sum())
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')


# =============================================================================
# VISUALIZATION
# =============================================================================

class VisualizationEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.output_dir = config.CACHE_DIR.parent / 'output'
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_equity_curve(self, strategy_equity: pd.Series,
                          buy_hold_equity: pd.Series,
                          symbol: str, metrics: Dict,
                          filename_suffix: str = '') -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        strategy_norm = strategy_equity / strategy_equity.iloc[0] * 100
        buyhold_norm = buy_hold_equity / buy_hold_equity.iloc[0] * 100
        
        ax1 = axes[0]
        ax1.plot(strategy_norm.index, strategy_norm.values,
                label='Optimized RSI Strategy', linewidth=2, color='#2E86AB')
        ax1.plot(buyhold_norm.index, buyhold_norm.values,
                label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_ylabel('Normalized Value ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Optimized Strategy Performance\n'
                     f'Return: {metrics.get("total_return_pct", "N/A")} | '
                     f'Sharpe: {metrics.get("sharpe_ratio", 0):.2f} | '
                     f'MDD: {metrics.get("max_drawdown_pct", "N/A")} | '
                     f'Win Rate: {metrics.get("win_rate", 0)*100:.1f}%',
                     fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        ax2 = axes[1]
        drawdown = (strategy_equity - strategy_equity.cummax()) / strategy_equity.cummax() * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color='#F18F01', alpha=0.6, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        suffix = f'_{filename_suffix}' if filename_suffix else ''
        save_path = self.output_dir / f'{symbol}_equity_curve{suffix}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved: {save_path}")
        plt.show()


# =============================================================================
# BASELINE
# =============================================================================

def calculate_buy_hold_equity(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    first_open = df['open'].iloc[0]
    shares = initial_capital / first_open
    return shares * df['close']


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = StrategyConfig()
    
    print("\n" + "="*70)
    print("OPTIMIZED RSI MOMENTUM STRATEGY BACKTEST (v3)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Symbols: {config.SYMBOLS}")
    print(f"  Timeframe: {config.TIMEFRAME}")
    print(f"  Date Range: {config.START_DATE} to {config.END_DATE}")
    print(f"\nOptimizations:")
    print(f"  RSI Entry: {config.RSI_LONG_ENTRY} (lowered from 70)")
    print(f"  Trend Filter: 200-EMA (mandatory)")
    print(f"  Volume Filter: Volume > 20-SMA")
    print(f"  Exit: ATR Trailing Stop (2.5x) + Take Profit (4.0x)")
    print(f"\nCost Model:")
    print(f"  Fee: {config.TRADING_FEE*100:.2f}% | Slippage: {config.SLIPPAGE*100:.3f}%")
    print("="*70)
    
    data_engine = DataEngine(config)
    signal_generator = SignalGenerator(config)
    backtest_engine = BacktestEngine(config)
    viz_engine = VisualizationEngine(config)
    
    results = {}
    
    for symbol in config.SYMBOLS:
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'='*60}")
        
        df = data_engine.fetch_symbol_data(symbol)
        print(f"[INFO] Data: {len(df)} bars ({df.index.min()} to {df.index.max()})")
        
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{'='*40}")
        print("PERFORMANCE METRICS")
        print(f"{'='*40}")
        print(f"  Total Return:     {metrics['total_return_pct']}")
        print(f"  CAGR:             {metrics['cagr_pct']}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']}")
        print(f"  Win Rate:         {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
        print(f"  Total Trades:     {metrics['total_trades']}")
        print(f"  Avg Trade:        ${metrics['avg_trade']:.2f}")
        print(f"\n  Exit Breakdown:")
        print(f"    Take Profit:    {metrics['tp_exits']} ({metrics['tp_exits']/metrics['total_trades']*100:.0f}%)")
        print(f"    Trailing Stop:  {metrics['sl_exits']} ({metrics['sl_exits']/metrics['total_trades']*100:.0f}%)")
        print(f"    RSI Emergency:  {metrics['rsi_exits']} ({metrics['rsi_exits']/metrics['total_trades']*100:.0f}%)")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return*100:.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'optimized_v3'
        )
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"\nOutput: {config.CACHE_DIR.parent / 'output'}")
    
    return results


if __name__ == "__main__":
    results = main()
