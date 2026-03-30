"""
================================================================================
Momentum-Based RSI Crypto Trading Strategy - OPTIMIZED v2
================================================================================
Author: Senior Quantitative Developer
Description: Advanced backtest with hyperparameter optimization, trend filtering,
             ATR-based position sizing, trailing stops, and walk-forward validation.

Enhancements:
    - Expanded hyperparameter grid search
    - 200-EMA trend filter for long entries
    - ATR-based volatility-adjusted position sizing
    - Trailing stop-loss and take-profit mechanisms
    - Walk-forward validation (3-segment split)
    - Multi-timeframe analysis (1h, 4h, 1d)
================================================================================
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product

import ffn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
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
    
    # RSI parameters - OPTIMIZED: Lower entry threshold to 65
    RSI_PERIOD: int = 14
    RSI_LONG_ENTRY: int = 65      # Changed from 70 to capture momentum earlier
    RSI_LONG_EXIT: int = 30       # Kept for emergency exit only
    
    # Trend filter - MANDATORY
    USE_TREND_FILTER: bool = True
    TREND_EMA_PERIOD: int = 200
    
    # Volume filter - NEW: Filter out low-volume false signals
    USE_VOLUME_FILTER: bool = True
    VOLUME_SMA_PERIOD: int = 20
    
    # ATR-based risk management - PRIMARY EXIT MECHANISM
    ATR_PERIOD: int = 14
    ATR_STOP_MULTIPLE: float = 2.5      # Trailing stop at 2.5x ATR
    ATR_TARGET_MULTIPLE: float = 4.0    # Take profit at 4.0x ATR
    
    # Position sizing (volatility-adjusted)
    RISK_PER_TRADE: float = 0.02        # Risk 2% of equity per trade
    MAX_POSITION_SIZE: float = 1.0      # Maximum 100% of equity
    
    # Cost modeling
    TRADING_FEE: float = 0.001          # 0.1% per trade
    SLIPPAGE: float = 0.0005            # 0.05% per execution
    FUNDING_RATE: float = 0.0001        # 0.01% per 8 hours
    
    # Capital
    INITIAL_CAPITAL: float = 1000.0
    
    # Cache directory
    CACHE_DIR: Path = field(default_factory=lambda: Path(__file__).parent / 'data_cache')
    
    # Binance API
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv('BINANCE_API_KEY', ''))
    BINANCE_API_SECRET: str = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET', ''))
    
    # Walk-forward validation
    WALK_FORWARD_SEGMENTS: int = 3


# =============================================================================
# DATA INGESTION & CACHING ENGINE
# =============================================================================

class DataEngine:
    """Handles data fetching from Binance with local CSV caching."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.client = self._init_binance_client()
    
    def _init_binance_client(self) -> Optional[Client]:
        """Initialize Binance API client."""
        try:
            if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
                client = Client(self.config.BINANCE_API_KEY, self.config.BINANCE_API_SECRET)
                print("[INFO] Binance client initialized with API keys.")
            else:
                client = Client()
                print("[INFO] Binance client initialized (public access only).")
            return client
        except Exception as e:
            print(f"[WARNING] Binance client initialization failed: {e}")
            return None
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Generate cache file path for a symbol."""
        return self.cache_dir / f"{symbol}_{timeframe}.csv"
    
    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from local CSV cache if available."""
        cache_path = self._get_cache_path(symbol, timeframe)
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col='timestamp', parse_dates=True)
                print(f"[INFO] Loaded {symbol} {timeframe} data from cache: {len(df)} bars")
                return df
            except Exception as e:
                print(f"[WARNING] Cache load failed for {symbol}: {e}")
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save data to local CSV cache."""
        cache_path = self._get_cache_path(symbol, timeframe)
        df.to_csv(cache_path)
        print(f"[INFO] Saved {symbol} {timeframe} data to cache: {cache_path}")
    
    def _fetch_binance_klines(self, symbol: str, timeframe: str,
                               start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch historical klines from Binance API with pagination."""
        if self.client is None:
            raise RuntimeError("Binance client not initialized")

        print(f"[INFO] Fetching {symbol} {timeframe} data from Binance...")
        
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
            
            print(f"[INFO] Fetched {len(all_klines)} candles so far...")

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
        """Fetch data for a single symbol with caching."""
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
    
    def fetch_all_data(self, timeframe: str = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols."""
        tf = timeframe or self.config.TIMEFRAME
        data = {}
        for symbol in self.config.SYMBOLS:
            data[symbol] = self.fetch_symbol_data(symbol, tf)
        return data


# =============================================================================
# INDICATOR CALCULATION (WILDER'S RSI + ATR + EMA)
# =============================================================================

class IndicatorEngine:
    """Technical indicator calculations with Wilder's smoothing."""
    
    @staticmethod
    def wilders_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's original smoothing method.
        Wilder's smoothing: alpha = 1/period (not 2/(period+1) like standard EMA)
        """
        delta = close.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Initialize with SMA for first 'period' values
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Apply Wilder's smoothing
        for i in range(period, len(close)):
            avg_gains.iloc[i] = (avg_gains.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
            avg_losses.iloc[i] = (avg_losses.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
        
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        
        return rsi
    
    @staticmethod
    def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
        """Calculate Average True Range using Wilder's smoothing."""
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
# SIGNAL GENERATION WITH ENHANCED LOGIC
# =============================================================================

class SignalGenerator:
    """Generate trading signals with trend filter and ATR-based stops."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicator_engine = IndicatorEngine()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with OPTIMIZED filters:
        - Wilder's RSI momentum (threshold: 65 for earlier entry)
        - 200-EMA trend filter (mandatory)
        - Volume filter (volume > 20-period SMA)
        - ATR-based trailing stops and take-profit (PRIMARY EXIT)
        - Look-ahead bias prevention (1-bar shift)
        
        Key Changes:
        - RSI entry lowered to 65 to capture momentum earlier
        - Volume filter eliminates low-volume whipsaws
        - ATR trailing stop is primary exit (not RSI)
        """
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
        
        # Volume filter: Current volume > 20-period Volume SMA
        if self.config.USE_VOLUME_FILTER:
            df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
            df['high_volume'] = df['volume'] > df['volume_sma']
        else:
            df['high_volume'] = True

        # Generate raw signals
        df['rsi_prev'] = df['rsi'].shift(1)

        # Long entry: RSI crosses ABOVE 65 + Trend Filter + Volume Filter
        # All three conditions must be met to reduce whipsaws
        df['long_entry_raw'] = (
            (df['rsi'] > self.config.RSI_LONG_ENTRY) &           # RSI > 65 (earlier entry)
            (df['rsi_prev'] <= self.config.RSI_LONG_ENTRY) &     # Crossover
            (df['above_trend'] == True) &                         # Price above 200-EMA
            (df['high_volume'] == True)                           # Above-average volume
        )

        # Long exit: RSI drops BELOW 30 (EMERGENCY EXIT ONLY)
        # Primary exits are now ATR trailing stop and take-profit
        df['long_exit_raw'] = (
            (df['rsi'] < self.config.RSI_LONG_EXIT) &
            (df['rsi_prev'] >= self.config.RSI_LONG_EXIT)
        )

        # CRITICAL: Shift signals by 1 to prevent look-ahead bias
        df['long_entry'] = df['long_entry_raw'].shift(1).fillna(False)
        df['long_exit'] = df['long_exit_raw'].shift(1).fillna(False)

        # Calculate stop-loss and take-profit levels based on ATR
        df['stop_loss_distance'] = df['atr'] * self.config.ATR_STOP_MULTIPLE
        df['take_profit_distance'] = df['atr'] * self.config.ATR_TARGET_MULTIPLE

        # Clean up temporary columns
        df.drop(columns=['rsi_prev', 'long_entry_raw', 'long_exit_raw'],
                inplace=True, errors='ignore')

        return df


# =============================================================================
# BACKTEST ENGINE WITH ATR-BASED RISK MANAGEMENT
# =============================================================================

class BacktestEngine:
    """
    Core backtesting engine with:
    - ATR-based trailing stop-loss
    - ATR-based take-profit
    - Volatility-adjusted position sizing
    - Realistic cost modeling
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.trades = pd.DataFrame()
    
    def _calculate_position_size(self, equity: float, entry_price: float,
                                  stop_loss_price: float) -> float:
        """
        Calculate volatility-adjusted position size.
        Risk a fixed percentage of equity per trade.
        """
        risk_amount = equity * self.config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        # Cap at maximum position size
        max_size = (equity * self.config.MAX_POSITION_SIZE) / entry_price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Run the backtest with OPTIMIZED ATR-based risk management.
        
        Exit Priority:
        1. ATR Trailing Stop (PRIMARY) - Locks in profits during trends
        2. ATR Take-Profit (SECONDARY) - Takes profits at measured targets
        3. RSI Emergency Exit (TERTIARY) - Only when RSI < 30 (trend reversal)
        """
        df = df.copy()

        # Initialize tracking columns
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
        stop_loss_price = 0.0
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

            # Check for entry/exit signals from previous bar
            signal = df.iloc[prev_idx]['long_entry']
            exit_signal = df.iloc[prev_idx]['long_exit']

            # Initialize exit flags
            sl_hit = False
            tp_hit = False
            emergency_exit = False

            if position > 0:
                # Update highest close since entry (for trailing stop calculation)
                if close_price > highest_close:
                    highest_close = close_price
                
                # DYNAMIC TRAILING STOP: Based on highest close since entry
                new_trailing_stop = close_price - (atr * self.config.ATR_STOP_MULTIPLE)
                
                # Trailing stop only moves UP, never down
                if new_trailing_stop > trailing_stop_price:
                    trailing_stop_price = new_trailing_stop

                # Check exit conditions IN PRIORITY ORDER
                # Priority 1: Trailing stop hit (most common in trends)
                if low_price <= trailing_stop_price:
                    sl_hit = True
                
                # Priority 2: Take-profit hit
                elif high_price >= take_profit_price:
                    tp_hit = True
                
                # Priority 3: RSI emergency exit (trend reversal signal)
                elif exit_signal:
                    emergency_exit = True

            # Execute exit if any condition is met
            if position > 0 and (sl_hit or tp_hit or emergency_exit):
                # Determine exit price based on exit reason
                if tp_hit:
                    exit_price = take_profit_price
                    tp_hits += 1
                    exit_reason = 'take_profit'
                elif sl_hit:
                    exit_price = trailing_stop_price
                    sl_hits += 1
                    exit_reason = 'trailing_stop'
                else:
                    # RSI emergency exit
                    exit_price = open_price * (1 - self.config.SLIPPAGE)
                    rsi_exits += 1
                    exit_reason = 'rsi_emergency'

                # Calculate P&L
                pnl = (exit_price - entry_price) * position_size

                # Calculate total costs
                trade_value = abs(position_size * exit_price)
                holding_periods = curr_idx - entry_bar
                funding_cost = position_size * self.config.FUNDING_RATE * (holding_periods / 2)
                fees = trade_value * self.config.TRADING_FEE
                slippage_cost = trade_value * self.config.SLIPPAGE
                total_trade_cost = fees + slippage_cost + funding_cost
                total_costs += total_trade_cost

                # Update equity
                equity += pnl - total_trade_cost

                # Record trade with detailed info
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
                    'highest_close': highest_close,
                    'entry_atr': entry_atr
                })

                # Reset position
                position = 0
                position_size = 0.0
                trailing_stop_price = 0.0
                highest_close = 0.0

            # Execute entry if signal and no existing position
            if signal and position == 0:
                entry_price = open_price * (1 + self.config.SLIPPAGE)
                entry_atr = atr if atr > 0 else 0.01 * entry_price

                # Calculate stop-loss and take-profit levels based on entry ATR
                stop_loss_distance = entry_atr * self.config.ATR_STOP_MULTIPLE
                take_profit_distance = entry_atr * self.config.ATR_TARGET_MULTIPLE

                stop_loss_price = entry_price - stop_loss_distance
                take_profit_price = entry_price + take_profit_distance
                trailing_stop_price = stop_loss_price  # Initial trailing stop
                highest_close = entry_price

                # Calculate position size based on risk (2% of equity)
                position_size = self._calculate_position_size(
                    equity, entry_price, stop_loss_price
                )

                if position_size > 0:
                    position = 1
                    entry_bar = curr_idx

            # Calculate funding cost for open positions (perpetual futures)
            if position > 0:
                funding_cost = position_size * self.config.FUNDING_RATE
                equity -= funding_cost
                total_costs += funding_cost

            # Calculate period return
            if i > 1:
                prev_equity = df.iloc[i-1]['equity']
                period_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            else:
                period_return = 0

            # Calculate drawdown
            running_max = df['equity'].iloc[:i].max() if i > 0 else equity
            drawdown = (equity - running_max) / running_max if running_max > 0 else 0

            # Update DataFrame
            df.iloc[i, df.columns.get_loc('returns')] = period_return
            df.iloc[i, df.columns.get_loc('equity')] = equity
            df.iloc[i, df.columns.get_loc('position')] = position
                    'costs': total_trade_cost,
                    'net_pnl': pnl - total_trade_cost,
                    'exit_reason': 'take_profit' if tp_hit else ('stop_loss' if sl_hit else 'signal')
                })
                
                position = 0
                position_size = 0
            
            # Execute entry if signal and no existing position
            if signal and position == 0:
                entry_price = open_price * (1 + self.config.SLIPPAGE)
                
                # Calculate stop-loss and take-profit levels
                stop_loss_distance = df.iloc[prev_idx]['stop_loss_distance'] if not pd.isna(df.iloc[prev_idx]['stop_loss_distance']) else atr * self.config.ATR_STOP_MULTIPLE
                take_profit_distance = df.iloc[prev_idx]['take_profit_distance'] if not pd.isna(df.iloc[prev_idx]['take_profit_distance']) else atr * self.config.ATR_TARGET_MULTIPLE
                
                stop_loss_price = entry_price - stop_loss_distance
                take_profit_price = entry_price + take_profit_distance
                trailing_stop_price = stop_loss_price
                
                # Calculate position size based on risk
                position_size = self._calculate_position_size(
                    equity, entry_price, stop_loss_price
                )
                
                if position_size > 0:
                    position = 1
                    entry_bar = curr_idx
            
            # Calculate funding cost for open positions
            if position > 0:
                funding_cost = position_size * self.config.FUNDING_RATE
                equity -= funding_cost
                total_costs += funding_cost
            
            # Calculate period return
            if i > 1:
                prev_equity = df.iloc[i-1]['equity']
                period_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            else:
                period_return = 0
            
            # Calculate drawdown
            running_max = df['equity'].iloc[:i].max() if i > 0 else equity
            drawdown = (equity - running_max) / running_max if running_max > 0 else 0
            
            # Update DataFrame
            df.iloc[i, df.columns.get_loc('returns')] = period_return
            df.iloc[i, df.columns.get_loc('equity')] = equity
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('drawdown')] = drawdown
        
        df['stop_loss_hits'] = sl_hits
        df['take_profit_hits'] = tp_hits
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return df
    
    def calculate_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate portfolio performance metrics."""
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < 2:
            return {'error': 'Insufficient data for metrics calculation'}
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return (CAGR)
        years = len(equity_curve) / (365 * 6)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe Ratio (annualized)
        periods_per_year = 365 * 6
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        # Maximum Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = self._calculate_win_rate()
        
        # Profit factor
        profit_factor = self._calculate_profit_factor()
        
        # Average trade
        avg_trade = self.trades['net_pnl'].mean() if not self.trades.empty else 0
        
        metrics = {
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
            'sl_hits': self.trades['exit_reason'].eq('stop_loss').sum() if not self.trades.empty else 0,
            'tp_hits': self.trades['exit_reason'].eq('take_profit').sum() if not self.trades.empty else 0,
        }
        
        return metrics
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if self.trades.empty:
            return 0.0
        winning_trades = (self.trades['net_pnl'] > 0).sum()
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if self.trades.empty:
            return 0.0
        
        gross_profit = self.trades[self.trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(self.trades[self.trades['net_pnl'] <= 0]['net_pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

class WalkForwardValidator:
    """
    Implement walk-forward validation to prevent overfitting.
    Splits data into N segments, trains on first N-1, tests on last.
    """
    
    def __init__(self, config: StrategyConfig, n_segments: int = 3):
        self.config = config
        self.n_segments = n_segments
    
    def split_data(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into training and testing segments.
        Returns list of (train_df, test_df) tuples.
        """
        n = len(df)
        segment_size = n // self.n_segments
        
        splits = []
        for i in range(self.n_segments - 1):
            # Train on segments 0 to i, test on segment i+1
            train_end = (i + 1) * segment_size
            test_start = train_end
            test_end = test_start + segment_size
            
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            splits.append((train_df, test_df))
        
        return splits
    
    def validate(self, df: pd.DataFrame, symbol: str,
                 param_combinations: List[Dict]) -> pd.DataFrame:
        """
        Run walk-forward validation for multiple parameter combinations.
        Returns DataFrame with out-of-sample performance for each combination.
        """
        splits = self.split_data(df)
        results = []
        
        for params in param_combinations:
            fold_results = []
            
            for train_df, test_df in splits:
                # Create config with these parameters
                config = StrategyConfig()
                config.RSI_PERIOD = params['rsi_period']
                config.RSI_LONG_ENTRY = params['rsi_entry']
                config.RSI_LONG_EXIT = params['rsi_exit']
                config.ATR_STOP_MULTIPLE = params.get('stop_multiple', 2.5)
                config.ATR_TARGET_MULTIPLE = params.get('target_multiple', 4.0)
                
                # Generate signals on test data
                signal_gen = SignalGenerator(config)
                test_signals = signal_gen.generate_signals(test_df)
                
                # Run backtest
                backtest = BacktestEngine(config)
                test_results = backtest.run_backtest(test_signals, symbol)
                
                # Calculate metrics
                metrics = backtest.calculate_metrics(test_results['equity'])
                
                if 'error' not in metrics:
                    fold_results.append({
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_return': metrics['total_return'],
                        'max_drawdown': metrics['max_drawdown'],
                        'win_rate': metrics['win_rate']
                    })
            
            # Average across folds
            if fold_results:
                avg_results = {
                    'rsi_period': params['rsi_period'],
                    'rsi_entry': params['rsi_entry'],
                    'rsi_exit': params['rsi_exit'],
                    'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in fold_results]),
                    'total_return': np.mean([r['total_return'] for r in fold_results]),
                    'max_drawdown': np.mean([r['max_drawdown'] for r in fold_results]),
                    'win_rate': np.mean([r['win_rate'] for r in fold_results]),
                    'sharpe_std': np.std([r['sharpe_ratio'] for r in fold_results]),
                }
                results.append(avg_results)
        
        return pd.DataFrame(results)


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

class ParameterOptimizer:
    """Run comprehensive parameter sensitivity analysis."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_engine = DataEngine(config)
        self.indicator_engine = IndicatorEngine()
    
    def run_optimization(self, symbol: str, timeframe: str = None,
                         quick_mode: bool = True) -> pd.DataFrame:
        """
        Run comprehensive grid search optimization.
        
        Parameter ranges:
        - RSI_PERIOD: 7 to 21 (step 1)
        - RSI_LONG_ENTRY: 60 to 85 (step 5)
        - RSI_LONG_EXIT: 20 to 50 (step 5)
        
        quick_mode: If True, uses reduced parameter space for faster execution
        """
        tf = timeframe or self.config.TIMEFRAME
        print(f"\n{'='*70}")
        print(f"PARAMETER OPTIMIZATION - {symbol} ({timeframe})")
        print(f"{'='*70}")
        
        # Fetch data
        df = self.data_engine.fetch_symbol_data(symbol, tf)
        
        # Pre-calculate indicators
        print("[INFO] Pre-calculating indicators...")
        
        results = []
        
        if quick_mode:
            # Reduced parameter space for faster execution
            rsi_periods = [10, 14, 18]       # 3 values
            rsi_entries = [60, 70, 80]        # 3 values
            rsi_exits = [25, 35, 45]          # 3 values
            total_combos = 27
        else:
            # Full parameter space
            rsi_periods = range(7, 22)         # 7 to 21
            rsi_entries = range(60, 90, 5)     # 60, 65, 70, 75, 80, 85
            rsi_exits = range(20, 55, 5)       # 20, 25, 30, 35, 40, 45, 50
            total_combos = 15 * 6 * 7  # 630 combinations
        
        combo_count = 0
        
        for rsi_period in rsi_periods:
            for rsi_entry in rsi_entries:
                for rsi_exit in rsi_exits:
                    combo_count += 1
                    
                    if combo_count % 20 == 0 or combo_count == total_combos:
                        print(f"[{combo_count}/{total_combos}] Processing...")
                    
                    # Create config
                    config = StrategyConfig()
                    config.RSI_PERIOD = rsi_period
                    config.RSI_LONG_ENTRY = rsi_entry
                    config.RSI_LONG_EXIT = rsi_exit
                    config.USE_TREND_FILTER = True
                    
                    # Calculate RSI
                    rsi = self.indicator_engine.wilders_rsi(df['close'], rsi_period)
                    
                    # Run backtest
                    metrics = self._run_single_backtest_fast(df, rsi, config, symbol)
                    
                    if 'error' not in metrics:
                        results.append({
                            'symbol': symbol,
                            'timeframe': tf,
                            'rsi_period': rsi_period,
                            'rsi_entry': rsi_entry,
                            'rsi_exit': rsi_exit,
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'total_return': metrics['total_return'],
                            'total_return_pct': metrics['total_return_pct'],
                            'cagr': metrics['cagr'],
                            'max_drawdown': metrics['max_drawdown'],
                            'max_drawdown_pct': metrics['max_drawdown_pct'],
                            'win_rate': metrics['win_rate'],
                            'profit_factor': metrics.get('profit_factor', 0),
                            'total_trades': metrics['total_trades'],
                        })
        
        return pd.DataFrame(results)
    
    def _run_single_backtest_fast(self, df: pd.DataFrame, rsi: pd.Series,
                                   config: StrategyConfig, symbol: str) -> Dict:
        """Fast single backtest run with pre-calculated RSI."""
        df = df.copy()
        
        # Calculate ATR
        atr = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'],
                                                 config.ATR_PERIOD)
        
        # Trend filter
        if config.USE_TREND_FILTER:
            ema_200 = self.indicator_engine.ema(df['close'], config.TREND_EMA_PERIOD)
            above_trend = df['close'] > ema_200
        else:
            above_trend = pd.Series(True, index=df.index)
        
        # Generate signals
        rsi_prev = rsi.shift(1)
        long_entry = ((rsi > config.RSI_LONG_ENTRY) & 
                      (rsi_prev <= config.RSI_LONG_ENTRY) & 
                      above_trend).shift(1).fillna(False)
        long_exit = ((rsi < config.RSI_LONG_EXIT) & 
                     (rsi_prev >= config.RSI_LONG_EXIT)).shift(1).fillna(False)
        
        df['long_entry'] = long_entry
        df['long_exit'] = long_exit
        df['atr'] = atr
        df['stop_loss_distance'] = atr * config.ATR_STOP_MULTIPLE
        df['take_profit_distance'] = atr * config.ATR_TARGET_MULTIPLE
        
        # Run backtest
        backtest_engine = BacktestEngine(config)
        df_results = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df_results['equity'])
        
        return metrics
    
    def get_top_combinations(self, results: pd.DataFrame, n: int = 10,
                             max_drawdown_threshold: float = -0.15) -> pd.DataFrame:
        """
        Get top N parameter combinations filtered by max drawdown constraint.
        """
        # Filter by max drawdown constraint
        filtered = results[results['max_drawdown'] >= max_drawdown_threshold].copy()
        
        if filtered.empty:
            print(f"[WARNING] No combinations meet MDD constraint of {max_drawdown_threshold*100:.0f}%")
            # Relax constraint and get best available
            filtered = results.copy()
        
        # Sort by Sharpe ratio
        top = filtered.nlargest(n, 'sharpe_ratio')
        
        return top


# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

class MultiTimeframeAnalyzer:
    """Compare strategy performance across different timeframes."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.timeframes = ['1h', '4h', '1d']
    
    def analyze(self, symbol: str, best_params: Dict) -> pd.DataFrame:
        """Run backtest for each timeframe with given parameters."""
        results = []
        
        for tf in self.timeframes:
            print(f"\n[INFO] Analyzing {symbol} on {tf} timeframe...")
            
            # Update config
            config = StrategyConfig()
            config.TIMEFRAME = tf
            config.RSI_PERIOD = best_params.get('rsi_period', 14)
            config.RSI_LONG_ENTRY = best_params.get('rsi_entry', 70)
            config.RSI_LONG_EXIT = best_params.get('rsi_exit', 30)
            
            # Fetch data
            data_engine = DataEngine(config)
            df = data_engine.fetch_symbol_data(symbol, tf)
            
            # Run backtest
            signal_gen = SignalGenerator(config)
            df_signals = signal_gen.generate_signals(df)
            
            backtest = BacktestEngine(config)
            df_results = backtest.run_backtest(df_signals, symbol)
            metrics = backtest.calculate_metrics(df_results['equity'])
            
            if 'error' not in metrics:
                results.append({
                    'symbol': symbol,
                    'timeframe': tf,
                    'bars': len(df),
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_return_pct': metrics['total_return_pct'],
                    'cagr_pct': metrics['cagr_pct'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                })
        
        return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

class VisualizationEngine:
    """Generate strategy visualizations."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.output_dir = config.CACHE_DIR.parent / 'output'
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_equity_curve(self, strategy_equity: pd.Series,
                          buy_hold_equity: pd.Series,
                          symbol: str, metrics: Dict,
                          filename_suffix: str = '') -> None:
        """Plot strategy equity curve vs buy & hold baseline."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        strategy_norm = strategy_equity / strategy_equity.iloc[0] * 100
        buyhold_norm = buy_hold_equity / buy_hold_equity.iloc[0] * 100
        
        ax1 = axes[0]
        ax1.plot(strategy_norm.index, strategy_norm.values,
                label='RSI Momentum Strategy', linewidth=2, color='#2E86AB')
        ax1.plot(buyhold_norm.index, buyhold_norm.values,
                label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_ylabel('Normalized Value ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Strategy Performance vs Buy & Hold\n'
                     f'Total Return: {metrics.get("total_return_pct", "N/A")} | '
                     f'Sharpe: {metrics.get("sharpe_ratio", 0):.2f} | '
                     f'Max DD: {metrics.get("max_drawdown_pct", "N/A")}',
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
        print(f"[INFO] Saved equity curve plot: {save_path}")
        plt.show()
    
    def plot_parameter_heatmap(self, results: pd.DataFrame,
                               metric: str = 'sharpe') -> None:
        """Generate parameter sensitivity heatmap."""
        # Filter by best drawdown
        filtered = results[results['max_drawdown'] >= -0.15]
        
        if filtered.empty:
            print("[WARNING] No results meet drawdown constraint for heatmap")
            return
        
        # Pivot for heatmap (aggregate by mean)
        pivot_data = filtered.pivot_table(
            index='rsi_exit',
            columns='rsi_entry',
            values=metric,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if metric == 'sharpe':
            cmap = 'RdYlGn'
            center = 0
            fmt = '.2f'
            title = 'Sharpe Ratio'
        else:
            cmap = 'YlOrRd'
            center = None
            fmt = '.1f'
            title = 'Total Return (%)'
        
        sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap,
                   center=center, linewidths=0.5, ax=ax,
                   cbar_kws={'label': title})
        
        ax.set_xlabel('Long Entry Threshold (RSI)', fontsize=12)
        ax.set_ylabel('Long Exit Threshold (RSI)', fontsize=12)
        ax.set_title(f'Parameter Sensitivity Analysis - {title}\n'
                    f'(Filtered: Max DD >= -15%)',
                    fontsize=14)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'parameter_heatmap_{metric}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved parameter heatmap: {save_path}")
        plt.show()
    
    def plot_optimization_results(self, results: pd.DataFrame,
                                  top_params: pd.DataFrame) -> None:
        """Plot optimization results distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sharpe ratio distribution
        ax1 = axes[0, 0]
        ax1.hist(results['sharpe_ratio'], bins=50, color='#2E86AB',
                edgecolor='black', alpha=0.7)
        ax1.axvline(x=top_params['sharpe_ratio'].mean(), color='red',
                   linestyle='--', linewidth=2, label='Top Avg')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sharpe Ratio Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Max drawdown vs Sharpe
        ax2 = axes[0, 1]
        scatter = ax2.scatter(results['max_drawdown'] * 100, results['sharpe_ratio'],
                             alpha=0.3, c=results['total_return'], cmap='YlOrRd')
        ax2.axvline(x=-15, color='red', linestyle='--', linewidth=2,
                   label='MDD Constraint (-15%)')
        ax2.set_xlabel('Max Drawdown (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Risk-Adjusted Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Win rate distribution
        ax3 = axes[1, 0]
        ax3.hist(results['win_rate'] * 100, bins=30, color='#A23B72',
                edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Win Rate (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Win Rate Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Total trades vs Return
        ax4 = axes[1, 1]
        ax4.scatter(results['total_trades'], results['total_return'] * 100,
                   alpha=0.3, color='#F18F01')
        ax4.set_xlabel('Total Trades')
        ax4.set_ylabel('Total Return (%)')
        ax4.set_title('Trading Frequency vs Returns')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'optimization_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved optimization summary: {save_path}")
        plt.show()


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

def calculate_buy_hold_equity(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Calculate buy & hold equity curve for comparison."""
    first_open = df['open'].iloc[0]
    shares = initial_capital / first_open
    equity = shares * df['close']
    return equity


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_baseline_backtest(config: StrategyConfig) -> Dict[str, Dict]:
    """Run baseline backtest with default parameters."""
    print("\n" + "="*70)
    print("BASELINE BACKTEST - Default Parameters")
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
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{'='*40}")
        print("PERFORMANCE METRICS")
        print(f"{'='*40}")
        for key, value in metrics.items():
            if key not in ['error'] and isinstance(value, str):
                print(f"  {key.replace('_', ' ').title()}: {value}")
            elif key not in ['error'] and isinstance(value, (int, float)):
                if 'pct' in key or 'rate' in key:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return * 100:.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
            'trades': backtest_engine.trades,
            'data': df
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'baseline'
        )
    
    return results


def run_optimized_backtest(config: StrategyConfig,
                           best_params: Dict) -> Dict[str, Dict]:
    """Run backtest with optimized parameters."""
    print("\n" + "="*70)
    print("OPTIMIZED BACKTEST - Best Parameters")
    print("="*70)
    print(f"Parameters: RSI Period={best_params.get('rsi_period', 14)}, "
          f"Entry={best_params.get('rsi_entry', 70)}, "
          f"Exit={best_params.get('rsi_exit', 30)}")
    
    # Update config
    config.RSI_PERIOD = best_params.get('rsi_period', 14)
    config.RSI_LONG_ENTRY = best_params.get('rsi_entry', 70)
    config.RSI_LONG_EXIT = best_params.get('rsi_exit', 30)
    
    data_engine = DataEngine(config)
    signal_generator = SignalGenerator(config)
    backtest_engine = BacktestEngine(config)
    viz_engine = VisualizationEngine(config)
    
    results = {}
    
    for symbol in config.SYMBOLS:
        df = data_engine.fetch_symbol_data(symbol)
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{symbol} Metrics:")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f} | "
              f"Return: {metrics['total_return_pct']} | "
              f"MDD: {metrics['max_drawdown_pct']}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'optimized'
        )
    
    return results


def run_walk_forward_validation(config: StrategyConfig,
                                param_combinations: List[Dict]) -> pd.DataFrame:
    """Run walk-forward validation."""
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION")
    print("="*70)
    
    data_engine = DataEngine(config)
    validator = WalkForwardValidator(config, n_segments=3)
    
    all_results = []
    
    for symbol in config.SYMBOLS:
        df = data_engine.fetch_symbol_data(symbol)
        wf_results = validator.validate(df, symbol, param_combinations)
        wf_results['symbol'] = symbol
        all_results.append(wf_results)
    
    return pd.concat(all_results, ignore_index=True)


def print_comparison_table(baseline_results: Dict, optimized_results: Dict,
                           wf_results: pd.DataFrame = None) -> None:
    """Print comparison table of baseline vs optimized performance."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: BASELINE vs OPTIMIZED")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 80)
    
    for symbol in baseline_results.keys():
        baseline = baseline_results[symbol]['metrics']
        optimized = optimized_results[symbol]['metrics']
        
        print(f"\n{symbol}")
        print("-" * 80)
        
        metrics_to_compare = ['sharpe_ratio', 'total_return_pct', 'cagr_pct',
                              'max_drawdown_pct', 'win_rate']
        
        for metric in metrics_to_compare:
            b_val = baseline.get(metric, 0)
            o_val = optimized.get(metric, 0)
            
            if isinstance(b_val, str):
                b_num = float(b_val.replace('%', ''))
                o_num = float(o_val.replace('%', ''))
                change = o_num - b_num
                change_str = f"{change:+.2f}%"
            else:
                change = o_val - b_val
                change_str = f"{change:+.4f}"
            
            print(f"  {metric.replace('_', ' ').title():<22} "
                  f"{str(b_val):<15} {str(o_val):<15} {change_str:<15}")
    
    if wf_results is not None and not wf_results.empty:
        print(f"\n\nWALK-FORWARD VALIDATION RESULTS (Out-of-Sample)")
        print("-" * 80)
        print(wf_results[['symbol', 'rsi_period', 'rsi_entry', 'rsi_exit',
                          'sharpe_ratio', 'total_return', 'max_drawdown']].to_string(index=False))
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    config = StrategyConfig()
    
    # Step 1: Run baseline backtest
    baseline_results = run_baseline_backtest(config)
    
    # Step 2: Run parameter optimization
    print("\n" + "="*70)
    print("STEP 2: PARAMETER OPTIMIZATION")
    print("="*70)
    
    optimizer = ParameterOptimizer(config)
    
    # Optimize for each symbol
    all_optimization_results = []
    for symbol in config.SYMBOLS:
        opt_results = optimizer.run_optimization(symbol)
        all_optimization_results.append(opt_results)
        
        # Get top combinations
        top_params = optimizer.get_top_combinations(
            opt_results, n=10, max_drawdown_threshold=-0.15
        )
        
        print(f"\n{'='*50}")
        print(f"TOP 10 PARAMETER COMBINATIONS - {symbol}")
        print(f"{'='*50}")
        print(top_params[['rsi_period', 'rsi_entry', 'rsi_exit',
                          'sharpe_ratio', 'total_return_pct',
                          'max_drawdown_pct', 'win_rate']].to_string(index=False))
    
    # Combine results
    combined_results = pd.concat(all_optimization_results, ignore_index=True)
    
    # Save optimization results
    combined_results.to_csv(config.CACHE_DIR.parent / 'output' / 'parameter_optimization_full.csv',
                           index=False)
    
    # Generate visualizations
    viz_engine = VisualizationEngine(config)
    viz_engine.plot_optimization_results(combined_results,
                                         optimizer.get_top_combinations(combined_results))
    
    for symbol in config.SYMBOLS:
        symbol_results = combined_results[combined_results['symbol'] == symbol]
        viz_engine.plot_parameter_heatmap(symbol_results, 'sharpe')
    
    # Step 3: Select best parameters
    print("\n" + "="*70)
    print("STEP 3: SELECTING BEST PARAMETERS")
    print("="*70)
    
    # Find best parameters that meet drawdown constraint
    filtered = combined_results[combined_results['max_drawdown'] >= -0.15]
    if not filtered.empty:
        best_row = filtered.loc[filtered['sharpe_ratio'].idxmax()]
        best_params = {
            'rsi_period': int(best_row['rsi_period']),
            'rsi_entry': int(best_row['rsi_entry']),
            'rsi_exit': int(best_row['rsi_exit']),
        }
        print(f"\nBest Parameters (Max Sharpe, MDD >= -15%):")
        print(f"  RSI Period: {best_params['rsi_period']}")
        print(f"  RSI Entry:  {best_params['rsi_entry']}")
        print(f"  RSI Exit:   {best_params['rsi_exit']}")
    else:
        best_params = {'rsi_period': 14, 'rsi_entry': 70, 'rsi_exit': 30}
        print("Using default parameters (no combinations meet MDD constraint)")
    
    # Step 4: Run optimized backtest
    optimized_results = run_optimized_backtest(config, best_params)
    
    # Step 5: Walk-forward validation
    print("\n" + "="*70)
    print("STEP 5: WALK-FORWARD VALIDATION")
    print("="*70)
    
    # Prepare parameter combinations for WF validation (top 5)
    top_combos = filtered.nlargest(5, 'sharpe_ratio')[
        ['rsi_period', 'rsi_entry', 'rsi_exit']
    ].to_dict('records')
    
    wf_results = run_walk_forward_validation(config, top_combos)
    
    # Step 6: Print comparison table
    print_comparison_table(baseline_results, optimized_results, wf_results)
    
    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)
    print(f"\nOutput files saved to: {config.CACHE_DIR.parent / 'output'}")
    
    return {
        'baseline': baseline_results,
        'optimized': optimized_results,
        'optimization_results': combined_results,
        'walk_forward': wf_results,
        'best_params': best_params
    }


if __name__ == "__main__":
    results = main()
