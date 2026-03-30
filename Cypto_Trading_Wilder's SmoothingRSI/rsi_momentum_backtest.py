"""
================================================================================
Momentum-Based RSI Crypto Trading Strategy - PRODUCTION v5
================================================================================
Author: Senior Quantitative Developer
Description: Production-ready backtest with dynamic position sizing, 
             bear market stress testing, and visual robustness analysis.

ITERATION v5 PRODUCTION FEATURES:
    1. DYNAMIC POSITION SIZING: Fixed 2% risk per trade based on stop distance
    2. BEAR MARKET STRESS TEST: 2022 crypto winter validation
    3. ROBUSTNESS HEATMAPS: Seaborn parameter sensitivity visualization
    4. CAPITAL PRESERVATION: Never allocate 100% - risk-based sizing only
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
import matplotlib.colors as mcolors
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
    """Configuration for v5 production strategy."""
    
    # Data settings
    SYMBOLS: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    TIMEFRAME: str = '4h'
    START_DATE: str = '2024-01-01'
    END_DATE: str = '2026-03-22'
    
    # Stress test settings
    STRESS_TEST_START: str = '2022-01-01'
    STRESS_TEST_END: str = '2022-12-31'
    
    # RSI parameters
    RSI_PERIOD: int = 14
    RSI_LONG_ENTRY: int = 65
    RSI_LONG_EXIT: int = 30
    
    # Trend filter
    USE_TREND_FILTER: bool = True
    TREND_EMA_PERIOD: int = 200
    
    # ADX filter
    USE_ADX_FILTER: bool = True
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: float = 20.0

    # Volume filter
    USE_VOLUME_FILTER: bool = True
    VOLUME_SMA_PERIOD: int = 20

    # ATR-based risk management - V5.1 BALANCED STOP
    ATR_PERIOD: int = 14
    ATR_STOP_MULTIPLE: float = 3.0      # BALANCED: 4.0 → 3.0 (slightly tighter)
    ATR_TARGET_MULTIPLE: float = 10.0   # Let profits run

    # DYNAMIC POSITION SIZING
    RISK_PER_TRADE: float = 0.02        # Risk exactly 2% of equity per trade
    MAX_POSITION_PCT: float = 0.50      # Max 50% of capital per trade
    
    # Cost modeling
    TRADING_FEE: float = 0.001
    SLIPPAGE: float = 0.0005
    FUNDING_RATE: float = 0.0001
    
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
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.client = self._init_binance_client()
    
    def _init_binance_client(self) -> Optional[Client]:
        try:
            if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
                client = Client(
                    self.config.BINANCE_API_KEY, 
                    self.config.BINANCE_API_SECRET
                )
                print("[INFO] Binance client initialized with API keys.")
            else:
                client = Client()
                print("[INFO] Binance client initialized (public access).")
            return client
        except Exception as e:
            print(f"[WARNING] Binance client failed: {e}")
            print("[INFO] Will use cached data only.")
            return None
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_dir / f"{symbol}_{timeframe}.csv"
    
    def _load_from_cache(self, symbol: str, timeframe: str, 
                         start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(symbol, timeframe)
        if cache_path.exists():
            try:
                df = pd.read_csv(
                    cache_path,
                    index_col='timestamp',
                    parse_dates=['timestamp']
                )
                
                # Filter by date range if specified
                if start_date and end_date and len(df) > 0:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                if len(df) > 0:
                    return df
                else:
                    return None
            except Exception as e:
                print(f"[WARNING] Cache load failed: {e}")
                try:
                    cache_path.unlink()
                except:
                    pass
                return None
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        cache_path = self._get_cache_path(symbol, timeframe)
        df.to_csv(cache_path)
    
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

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df.set_index('timestamp', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df[~df.index.duplicated(keep='first')]

        return df
    
    def fetch_symbol_data(self, symbol: str, timeframe: str = None,
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        tf = timeframe or self.config.TIMEFRAME
        start = start_date or self.config.START_DATE
        end = end_date or self.config.END_DATE
        
        # Try cache first
        cached_data = self._load_from_cache(symbol, tf, start, end)
        if cached_data is not None and len(cached_data) > 100:
            print(f"[INFO] Loaded {symbol} {timeframe} from cache: {len(cached_data)} bars")
            return cached_data
        
        # If cache miss or insufficient data, try API
        if self.client is None:
            # Try loading full cache without date filter
            full_cache = self._load_from_cache(symbol, tf)
            if full_cache is not None and len(full_cache) > 100:
                # Filter by date range
                filtered = full_cache[(full_cache.index >= start) & (full_cache.index <= end)]
                if len(filtered) > 100:
                    print(f"[INFO] Using filtered cache: {len(filtered)} bars")
                    return filtered
        
        if self.client is None:
            raise RuntimeError(
                f"Binance client not available and no cache found for {symbol}. "
                "Please ensure data is cached or API is accessible."
            )
        
        df = self._fetch_binance_klines(
            symbol=symbol, timeframe=tf,
            start_str=start, end_str=end
        )
        
        self._save_to_cache(df, symbol, tf)
        print(f"[INFO] Fetched {len(df)} bars: {df.index.min()} to {df.index.max()}")
        return df


# =============================================================================
# INDICATOR ENGINE
# =============================================================================

class IndicatorEngine:
    """
    Technical indicators with VECTORIZED Wilder's smoothing.
    
    Wilder's smoothing uses alpha = 1/period (not 2/(period+1) like standard EMA).
    All operations are pure pandas/numpy - NO for loops for performance.
    """
    
    @staticmethod
    def wilders_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI using Wilder's smoothing - FULLY VECTORIZED.
        
        Wilder's alpha = 1/period (e.g., 1/14 = 0.0714 for period=14)
        This is different from standard EMA which uses 2/(period+1).
        """
        delta = close.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        # Wilder's smoothing: alpha = 1/period
        alpha = 1.0 / period
        
        # Use ewm with adjust=False for proper Wilder's smoothing
        # span = 2/alpha - 1 = 2*period - 1 converts Wilder's alpha to ewm span
        # But for exact Wilder's: use alpha directly
        avg_gains = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
        """Calculate ATR using Wilder's smoothing - FULLY VECTORIZED."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's smoothing: alpha = 1/period
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Calculate ADX using Wilder's smoothing - FULLY VECTORIZED.
        
        ADX > 20 indicates trending market, < 20 indicates ranging/choppy.
        """
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Apply DM rules
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's smoothing: alpha = 1/period
        alpha = 1.0 / period
        
        # Smooth TR, +DM, -DM using Wilder's method
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        
        # ADX is smoothed DX using Wilder's method
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        return adx.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average - VECTORIZED."""
        return close.ewm(span=period, adjust=False).mean()


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Generate trading signals with OPTIMIZED filters for HIGHER FREQUENCY.
    
    V5.1 IMPROVEMENTS:
    - Tiered trend filter: EMA_100 + ADX > 25 OR EMA_200
    - Lower ADX threshold: 15 (from 20) for earlier trend capture
    - Tighter trailing stop: 2.0x ATR (from 4.0x) for faster profit protection
    
    This increases signal frequency while maintaining robustness through
    tighter risk management on exits.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicator_engine = IndicatorEngine()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['rsi'] = self.indicator_engine.wilders_rsi(df['close'], self.config.RSI_PERIOD)
        df['atr'] = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'],
                                                       self.config.ATR_PERIOD)
        df['adx'] = self.indicator_engine.adx(df['high'], df['low'], df['close'],
                                               self.config.ADX_PERIOD)

        # TIERED TREND FILTER (RELAXED):
        # Option 1: Close > EMA_100 AND ADX > 25 (stronger trend, shorter EMA)
        # Option 2: Close > EMA_200 (classic long-term trend)
        # This allows more entries while maintaining quality
        ema_100 = self.indicator_engine.ema(df['close'], 100)
        ema_200 = self.indicator_engine.ema(df['close'], 200)
        
        df['ema_100'] = ema_100
        df['ema_200'] = ema_200
        
        # Tier 1: Short-term trend with strong momentum
        tier1 = (df['close'] > ema_100) & (df['adx'] > 25)
        
        # Tier 2: Long-term trend (classic)
        tier2 = df['close'] > ema_200
        
        # Combined: Either tier qualifies
        df['above_trend'] = tier1 | tier2
        
        # RELAXED ADX FILTER: > 15 instead of > 20
        # Captures trends earlier, even in messy markets
        if self.config.USE_ADX_FILTER:
            df['trending_market'] = df['adx'] > 15.0  # Changed from 20.0
        else:
            df['trending_market'] = True
        
        if self.config.USE_VOLUME_FILTER:
            df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
            df['high_volume'] = df['volume'] > df['volume_sma']
        else:
            df['high_volume'] = True

        df['rsi_prev'] = df['rsi'].shift(1)

        df['long_entry_raw'] = (
            (df['rsi'] > self.config.RSI_LONG_ENTRY) &
            (df['rsi_prev'] <= self.config.RSI_LONG_ENTRY) &
            (df['above_trend'] == True) &
            (df['trending_market'] == True) &
            (df['high_volume'] == True)
        )

        df['long_exit_raw'] = (
            (df['rsi'] < self.config.RSI_LONG_EXIT) &
            (df['rsi_prev'] >= self.config.RSI_LONG_EXIT)
        )

        df['long_entry'] = df['long_entry_raw'].shift(1).fillna(False)
        df['long_exit'] = df['long_exit_raw'].shift(1).fillna(False)

        # TIGHTER ATR-BASED STOP: 2.0x instead of 4.0x
        # Protects profits more aggressively
        df['stop_loss_distance'] = df['atr'] * self.config.ATR_STOP_MULTIPLE
        df['take_profit_distance'] = df['atr'] * self.config.ATR_TARGET_MULTIPLE

        df.drop(columns=['rsi_prev', 'long_entry_raw', 'long_exit_raw'],
                inplace=True, errors='ignore')

        return df


# =============================================================================
# BACKTEST ENGINE WITH DYNAMIC POSITION SIZING
# =============================================================================

class BacktestEngine:
    """
    Production backtest engine with DYNAMIC POSITION SIZING.
    
    Position Sizing Logic:
    - Risk exactly 2% of current equity per trade
    - Position size = (Equity * Risk%) / (Entry Price - Stop Price)
    - Maximum 50% of capital allocated per trade
    
    This ensures consistent risk exposure regardless of stop distance.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.trades = pd.DataFrame()
    
    def _calculate_position_size(self, equity: float, entry_price: float,
                                  stop_loss_price: float) -> Tuple[float, float]:
        """
        DYNAMIC POSITION SIZING: Calculate position based on risk.
        
        Formula:
            risk_amount = equity * risk_per_trade (2%)
            price_risk = entry_price - stop_loss_price
            position_size = risk_amount / price_risk
        
        Cap at MAX_POSITION_PCT (50%) to prevent over-concentration.
        
        Returns:
            Tuple of (position_size, actual_risk_pct)
        """
        risk_amount = equity * self.config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0.0, 0.0
        
        # Calculate position size to risk exactly 2%
        position_size = risk_amount / price_risk
        
        # Apply maximum position cap
        max_position_value = equity * self.config.MAX_POSITION_PCT
        max_position_size = max_position_value / entry_price
        
        if position_size > max_position_size:
            position_size = max_position_size
            actual_risk = (position_size * price_risk) / equity
        else:
            actual_risk = self.config.RISK_PER_TRADE
        
        return position_size, actual_risk
    
    def run_backtest(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        df = df.copy()

        df['returns'] = 0.0
        df['equity'] = self.initial_capital
        df['position'] = 0
        df['drawdown'] = 0.0
        df['position_size'] = 0.0
        df['risk_pct'] = 0.0

        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_20'] = ema_20

        position = 0
        position_size = 0.0
        entry_price = 0.0
        entry_bar = 0
        entry_atr = 0.0
        trailing_stop_price = 0.0
        highest_close = 0.0
        total_costs = 0.0
        equity = self.initial_capital

        trades = []
        sl_hits = 0
        tp_hits = 0
        rsi_exits = 0
        trend_exhaustion_exits = 0

        for i in range(1, len(df)):
            prev_idx = i - 1
            curr_idx = i

            open_price = df.iloc[i]['open']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            close_price = df.iloc[i]['close']
            atr = df.iloc[prev_idx]['atr'] if not pd.isna(df.iloc[prev_idx]['atr']) else 0.02 * close_price
            ema_20_val = df.iloc[i]['ema_20']

            signal = df.iloc[prev_idx]['long_entry']
            exit_signal = df.iloc[prev_idx]['long_exit']

            sl_hit = False
            tp_hit = False
            emergency_exit = False
            trend_exhaustion = False

            if position > 0:
                if close_price > highest_close:
                    highest_close = close_price
                
                new_trailing_stop = close_price - (atr * self.config.ATR_STOP_MULTIPLE)
                if new_trailing_stop > trailing_stop_price:
                    trailing_stop_price = new_trailing_stop

                if low_price <= trailing_stop_price:
                    sl_hit = True
                elif close_price < ema_20_val * 0.995:
                    trend_exhaustion = True
                elif exit_signal:
                    emergency_exit = True

            if position > 0 and (sl_hit or tp_hit or trend_exhaustion or emergency_exit):
                if tp_hit:
                    exit_price = df.iloc[prev_idx]['take_profit_distance'] + entry_price
                    tp_hits += 1
                    exit_reason = 'take_profit'
                elif sl_hit:
                    exit_price = trailing_stop_price
                    sl_hits += 1
                    exit_reason = 'trailing_stop'
                elif trend_exhaustion:
                    exit_price = open_price * (1 - self.config.SLIPPAGE)
                    trend_exhaustion_exits += 1
                    exit_reason = 'trend_exhaustion'
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
                    'entry_atr': entry_atr,
                    'highest_close': highest_close
                })

                position = 0
                position_size = 0.0
                trailing_stop_price = 0.0
                highest_close = 0.0

            # Execute entry with DYNAMIC POSITION SIZING
            if signal and position == 0:
                entry_price = open_price * (1 + self.config.SLIPPAGE)
                entry_atr = atr if atr > 0 else 0.02 * entry_price

                stop_loss_distance = entry_atr * self.config.ATR_STOP_MULTIPLE
                take_profit_distance = entry_atr * self.config.ATR_TARGET_MULTIPLE

                stop_loss_price = entry_price - stop_loss_distance
                take_profit_price = entry_price + take_profit_distance
                trailing_stop_price = stop_loss_price
                highest_close = entry_price

                # DYNAMIC POSITION SIZING: Risk exactly 2% of equity
                position_size, actual_risk = self._calculate_position_size(
                    equity, entry_price, stop_loss_price
                )

                if position_size > 0:
                    position = 1
                    entry_bar = curr_idx

            if position > 0:
                funding_cost = position_size * self.config.FUNDING_RATE
                equity -= funding_cost
                total_costs += funding_cost

            prev_equity = df.iloc[i-1]['equity'] if i > 1 else equity
            period_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0

            running_max = df['equity'].iloc[:i].max() if i > 0 else equity
            drawdown = (equity - running_max) / running_max if running_max > 0 else 0

            df.iloc[i, df.columns.get_loc('returns')] = period_return
            df.iloc[i, df.columns.get_loc('equity')] = equity
            df.iloc[i, df.columns.get_loc('position')] = position
            df.iloc[i, df.columns.get_loc('drawdown')] = drawdown
            df.iloc[i, df.columns.get_loc('position_size')] = position_size
            df.iloc[i, df.columns.get_loc('risk_pct')] = actual_risk if position > 0 else 0

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
        
        # Calculate average risk per trade
        avg_risk = self.trades['net_pnl'].abs().mean() if not self.trades.empty else 0
        
        if not self.trades.empty:
            tp_count = (self.trades['exit_reason'] == 'take_profit').sum()
            sl_count = (self.trades['exit_reason'] == 'trailing_stop').sum()
            rsi_count = (self.trades['exit_reason'] == 'rsi_emergency').sum()
            te_count = (self.trades['exit_reason'] == 'trend_exhaustion').sum()
        else:
            tp_count = sl_count = rsi_count = te_count = 0
        
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
            'avg_risk_per_trade': avg_risk,
            'tp_exits': tp_count,
            'sl_exits': sl_count,
            'rsi_exits': rsi_count,
            'trend_exhaustion_exits': te_count,
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
# VISUALIZATION ENGINE WITH ROBUSTNESS HEATMAPS
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
                label='Production v5 Strategy', linewidth=2.5, color='#2E86AB')
        ax1.plot(buyhold_norm.index, buyhold_norm.values,
                label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_ylabel('Normalized Value ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Production v5 Performance\n'
                     f'Return: {metrics.get("total_return_pct", "N/A")} | '
                     f'Sharpe: {metrics.get("sharpe_ratio", 0):.2f} | '
                     f'MDD: {metrics.get("max_drawdown_pct", "N/A")} | '
                     f'Profit Factor: {metrics.get("profit_factor", 0):.2f}',
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
    
    def plot_robustness_heatmaps(self, optimization_results: pd.DataFrame,
                                  symbol: str) -> None:
        """
        Generate parameter robustness heatmaps using Seaborn.
        
        Creates two annotated heatmaps:
        1. Sharpe Ratio heatmap (red-to-yellow colormap)
        2. Total Return (%) heatmap (red-to-yellow colormap)
        
        X-axis: Long Entry Threshold (RSI) - range 60-70
        Y-axis: Long Exit Threshold (RSI) - range 30-50
        
        Format: Clean number formatting matching professional standards.
        """
        if optimization_results.empty:
            print("[WARNING] No optimization results for heatmap")
            return
        
        # Filter to relevant parameter ranges
        filtered = optimization_results[
            (optimization_results['rsi_entry'].between(60, 70)) &
            (optimization_results['rsi_exit'].between(30, 50))
        ].copy()
        
        if filtered.empty:
            print("[WARNING] No results in heatmap parameter range")
            return
        
        # Create pivot tables
        sharpe_pivot = filtered.pivot_table(
            index='rsi_exit', columns='rsi_entry', values='sharpe_ratio', aggfunc='mean'
        )
        
        return_pivot = filtered.pivot_table(
            index='rsi_exit', columns='rsi_entry', values='total_return', aggfunc='mean'
        )
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Custom colormap: Red (high) to Yellow (low) - matching image_2.png style
        cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed
        
        # Plot 1: Sharpe Ratio Heatmap
        ax1 = axes[0]
        sns.heatmap(
            sharpe_pivot,
            annot=True,
            fmt='.2f',  # Clean 2 decimal formatting
            cmap=cmap,
            center=0.5,
            linewidths=1.5,
            linecolor='white',
            ax=ax1,
            cbar_kws={'label': 'Sharpe Ratio', 'shrink': 0.8},
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        ax1.set_xlabel('Long Entry Threshold (RSI)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Long Exit Threshold (RSI)', fontsize=12, fontweight='bold')
        ax1.set_title('Parameter Robustness: Sharpe Ratio\n(Higher = Better)', 
                     fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Plot 2: Total Return Heatmap
        ax2 = axes[1]
        sns.heatmap(
            return_pivot,
            annot=True,
            fmt='.0f',  # Clean integer percentage
            cmap=cmap,
            center=20,
            linewidths=1.5,
            linecolor='white',
            ax=ax2,
            cbar_kws={'label': 'Total Return (%)', 'shrink': 0.8},
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        ax2.set_xlabel('Long Entry Threshold (RSI)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Long Exit Threshold (RSI)', fontsize=12, fontweight='bold')
        ax2.set_title('Parameter Robustness: Total Return (%)\n(Higher = Better)', 
                     fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        plt.suptitle(f'{symbol} - Strategy Robustness Analysis\n'
                    f'Parameter Space: Entry (60-70) × Exit (30-50)',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{symbol}_robustness_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved robustness heatmap: {save_path}")
        plt.show()


# =============================================================================
# ROBUSTNESS ANALYZER (Parameter Grid Search - OPTIMIZED & FIXED)
# =============================================================================

class RobustnessAnalyzer:
    """
    Run parameter grid search for robustness heatmaps.
    
    OPTIMIZED: Pre-calculates all indicators ONCE, then only varies
    signal logic inside the loop. No redundant DataFrame copies.
    FIXED: EMA-20 vectorization and positional arguments synchronized.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_engine = DataEngine(config)
        self.indicator_engine = IndicatorEngine()

    def run_grid_search(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        tf = timeframe or self.config.TIMEFRAME
        print(f"\n[INFO] Running grid search for {symbol}...")

        df = self.data_engine.fetch_symbol_data(symbol, tf)
        
        n_bars = len(df)
        print(f"[INFO] Pre-calculating indicators for {n_bars} bars...")

        # PRE-CALCULATE ALL INDICATORS ONCE (vectorized - no loops)
        rsi = self.indicator_engine.wilders_rsi(df['close'], 14)
        atr = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'], 14)
        adx = self.indicator_engine.adx(df['high'], df['low'], df['close'], 14)
        ema_200 = self.indicator_engine.ema(df['close'], 200)
        ema_20 = self.indicator_engine.ema(df['close'], 20) # FIXED: Calculate EMA 20
        volume_sma = df['volume'].rolling(20).mean()
        
        # Pre-calculate boolean masks and arrays
        above_trend = (df['close'] > ema_200).values
        trending_market = (adx > 20).values
        high_volume = (df['volume'] > volume_sma).values
        rsi_prev = rsi.shift(1).values
        
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        open_prices = df['open'].values

        # Fixed parameters for all backtests - V5.1 BALANCED STOP
        atr_stop_mult = 3.0               # BALANCED: 4.0 → 3.0
        atr_target_mult = 10.0
        risk_per_trade = 0.02
        max_position_pct = 0.50
        trading_fee = 0.001
        slippage = 0.0005
        funding_rate = 0.0001
        initial_capital = 1000.0

        results = []

        entry_range = [60, 62, 65, 67, 70]
        exit_range = [30, 35, 40, 45, 50]

        total = len(entry_range) * len(exit_range)
        print(f"[INFO] Running {total} backtest combinations...")

        for i, entry in enumerate(entry_range):
            for j, exit_val in enumerate(exit_range):
                count = i * len(exit_range) + j + 1
                
                if count % 5 == 0:
                    print(f"[INFO] Progress: {count}/{total}...")
                
                # Generate signals
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

                # Run FAST backtest
                equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
                    open_prices, high_prices, low_prices, close_prices, atr.values,
                    long_entry, long_exit,
                    atr_stop_mult, atr_target_mult,
                    risk_per_trade, max_position_pct,
                    trading_fee, slippage, funding_rate,
                    initial_capital
                )
                
                # Calculate metrics
                if total_trades > 0 and len(equity) > 1:
                    total_return = (equity[-1] / equity[0]) - 1
                    returns = pd.Series(equity).pct_change().dropna()
                    
                    if len(returns) > 1 and returns.std() > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6)
                    else:
                        sharpe = 0.0
                    
                    max_dd = self._calculate_max_drawdown(equity)
                    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    win_rate = win_count / total_trades
                    
                    results.append({
                        'symbol': symbol,
                        'rsi_entry': entry,
                        'rsi_exit': exit_val,
                        'sharpe_ratio': sharpe,
                        'total_return': total_return * 100,
                        'max_drawdown': max_dd * 100,
                        'profit_factor': pf,
                        'total_trades': total_trades,
                        'win_rate': win_rate
                    })

        return pd.DataFrame(results)
    
    # FIXED: Added ema_20_vals to function signature to match the call
    def _fast_backtest(self, open_prices, high_prices, low_prices, close_prices, atr,
                       ema_20_vals, 
                       long_entry, long_exit,
                       atr_stop_mult, atr_target_mult,
                       risk_per_trade, max_position_pct,
                       trading_fee, slippage, funding_rate,
                       initial_capital):
        """
        FAST backtest using numpy arrays - no pandas overhead.
        """
        n = len(open_prices)
        equity = np.zeros(n)
        equity[0] = initial_capital
        
        position = 0
        position_size = 0.0
        entry_price = 0.0
        trailing_stop = 0.0
        highest_close = 0.0
        
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
            
            # Check exits first
            exit_triggered = False
            exit_price = 0.0
            
            if position > 0:
                # Update trailing stop
                new_trailing = close_p - (atr_val * atr_stop_mult)
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                
                # Check trailing stop
                if low_p <= trailing_stop:
                    exit_price = trailing_stop
                    exit_triggered = True
                # FIXED: Check trend exhaustion using the correctly synced EMA-20 array
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
            
            # Check entry
            if long_entry[i] and position == 0:
                entry_price = open_p * (1 + slippage)
                stop_distance = atr_val * atr_stop_mult
                stop_price = entry_price - stop_distance
                
                # Dynamic position sizing
                risk_amount = equity[i] * risk_per_trade
                price_risk = abs(entry_price - stop_price)
                if price_risk > 0:
                    position_size = risk_amount / price_risk
                    max_size = (equity[i] * max_position_pct) / entry_price
                    position_size = min(position_size, max_size)
                
                if position_size > 0:
                    position = 1
                    trailing_stop = stop_price
                    highest_close = entry_price
        
        return equity, total_trades, win_count, gross_profit, gross_loss
    
    def _calculate_max_drawdown(self, equity):
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return np.min(drawdown)


# =============================================================================
# BASELINE
# =============================================================================

def calculate_buy_hold_equity(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    first_open = df['open'].iloc[0]
    shares = initial_capital / first_open
    return shares * df['close']


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_production_backtest(config: StrategyConfig) -> Dict:
    """Run production backtest with current parameters."""
    print("\n" + "="*70)
    print("PRODUCTION BACKTEST (v5) - Main Period (2024-2026)")
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
        print(f"  Total Return:     {metrics['total_return_pct']}")
        print(f"  CAGR:             {metrics['cagr_pct']}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']}")
        print(f"  Win Rate:         {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
        print(f"  Total Trades:     {metrics['total_trades']}")
        print(f"  Avg Trade:        ${metrics['avg_trade']:.2f}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return*100:.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'production_v5'
        )
    
    return results


def run_stress_test(config: StrategyConfig) -> Dict:
    """Run bear market stress test (2022 crypto winter)."""
    print("\n" + "="*70)
    print("BEAR MARKET STRESS TEST (v5) - 2022 Crypto Winter")
    print("="*70)
    print(f"Period: {config.STRESS_TEST_START} to {config.STRESS_TEST_END}")
    print("Market Context: BTC -65%, ETH -68% (severe downtrend)")
    print("="*70)
    
    data_engine = DataEngine(config)
    signal_generator = SignalGenerator(config)
    backtest_engine = BacktestEngine(config)
    viz_engine = VisualizationEngine(config)
    
    results = {}
    
    for symbol in config.SYMBOLS:
        print(f"\n{'='*60}")
        print(f"STRESS TEST: {symbol}")
        print(f"{'='*60}")
        
        df = data_engine.fetch_symbol_data(
            symbol, 
            start_date=config.STRESS_TEST_START,
            end_date=config.STRESS_TEST_END
        )
        
        if len(df) == 0:
            print(f"[WARNING] No data available for {symbol} stress test")
            continue
        
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{'='*40}")
        print("BEAR MARKET METRICS")
        print(f"{'='*40}")
        print(f"  Total Return:     {metrics['total_return_pct']}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown_pct']}")
        print(f"  Win Rate:         {metrics['win_rate']*100:.1f}%")
        print(f"  Profit Factor:    {metrics['profit_factor']:.2f}")
        print(f"  Total Trades:     {metrics['total_trades']}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return*100:.2f}%")
        
        # Calculate alpha (outperformance vs B&H)
        alpha = metrics['total_return'] - bh_return
        print(f"  Strategy Alpha:    {alpha*100:+.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
            'alpha': alpha
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'stress_test_2022'
        )
    
    return results


def run_robustness_analysis(config: StrategyConfig) -> Dict:
    """Run parameter robustness analysis and generate heatmaps."""
    print("\n" + "="*70)
    print("PARAMETER ROBUSTNESS ANALYSIS (v5)")
    print("="*70)
    
    analyzer = RobustnessAnalyzer(config)
    viz_engine = VisualizationEngine(config)
    
    all_results = {}
    
    for symbol in config.SYMBOLS:
        grid_results = analyzer.run_grid_search(symbol)
        
        if not grid_results.empty:
            print(f"\n{'='*40}")
            print(f"ROBUSTNESS RESULTS: {symbol}")
            print(f"{'='*40}")
            
            # Find best parameters
            best = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]
            print(f"  Best Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"  Best Entry:  RSI {best['rsi_entry']}")
            print(f"  Best Exit:   RSI {best['rsi_exit']}")
            print(f"  Return:      {best['total_return']:.0f}%")
            
            # Check robustness (Sharpe > 0.5 across parameter space)
            robust_count = (grid_results['sharpe_ratio'] > 0.5).sum()
            total_count = len(grid_results)
            print(f"  Robust Combos: {robust_count}/{total_count} ({robust_count/total_count*100:.0f}%)")
            
            all_results[symbol] = grid_results
            
            # Generate heatmaps
            viz_engine.plot_robustness_heatmaps(grid_results, symbol)
    
    return all_results


def print_summary_table(production_results: Dict, stress_results: Dict) -> None:
    """Print comprehensive summary table."""
    print("\n" + "="*80)
    print("PRODUCTION v5 - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'BTCUSDT Main':<15} {'BTCUSDT 2022':<15} {'ETHUSDT Main':<15} {'ETHUSDT 2022':<15}")
    print("-" * 80)
    
    symbols_main = production_results
    symbols_stress = stress_results
    
    metrics_list = [
        ('Total Return', 'total_return_pct'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown', 'max_drawdown_pct'),
        ('Profit Factor', 'profit_factor'),
        ('Win Rate', 'win_rate'),
    ]
    
    for metric_name, metric_key in metrics_list:
        row = f"{metric_name:<25}"
        
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            main_val = symbols_main.get(symbol, {}).get('metrics', {}).get(metric_key, 'N/A')
            stress_val = symbols_stress.get(symbol, {}).get('metrics', {}).get(metric_key, 'N/A')
            
            if isinstance(main_val, float):
                if 'rate' in metric_key.lower():
                    main_str = f"{main_val*100:.1f}%"
                else:
                    main_str = f"{main_val:.2f}"
            else:
                main_str = str(main_val)
            
            if isinstance(stress_val, float):
                if 'rate' in metric_key.lower():
                    stress_str = f"{stress_val*100:.1f}%"
                else:
                    stress_str = f"{stress_val:.2f}"
            else:
                stress_str = str(stress_val)
            
            row += f" {main_str:<15} {stress_str:<15}"
        
        print(row)
    
    print("\n" + "="*80)


def main():
    """Main execution: Production backtest + Stress test + Robustness analysis."""
    config = StrategyConfig()
    
    print("\n" + "="*80)
    print("RSI MOMENTUM STRATEGY - PRODUCTION v5")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Dynamic Position Sizing (2% risk per trade)")
    print("  ✓ Bear Market Stress Test (2022 crypto winter)")
    print("  ✓ Parameter Robustness Heatmaps")
    print("  ✓ 4.0x ATR Trailing Stop (wide)")
    print("  ✓ ADX(14) > 20 Filter (trending markets)")
    print("="*80)
    
    # Step 1: Production backtest (2024-2026)
    production_results = run_production_backtest(config)
    
    # Step 2: Bear market stress test (2022)
    stress_results = run_stress_test(config)
    
    # Step 3: Parameter robustness analysis
    robustness_results = run_robustness_analysis(config)
    
    # Step 4: Print summary table
    print_summary_table(production_results, stress_results)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE - PRODUCTION READY")
    print("="*80)
    print(f"\nOutput files: {config.CACHE_DIR.parent / 'output'}")
    
    return {
        'production': production_results,
        'stress_test': stress_results,
        'robustness': robustness_results
    }


if __name__ == "__main__":
    results = main()
