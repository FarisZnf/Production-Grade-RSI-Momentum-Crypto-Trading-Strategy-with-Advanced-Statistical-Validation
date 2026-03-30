
"""
================================================================================
Momentum-Based RSI Crypto Trading Strategy - PRODUCTION v5 (3x LEVERAGE)
================================================================================
Author: Senior Quantitative Developer
Description: Production-ready backtest with dynamic position sizing,
             bear market stress testing, visual robustness analysis,
             and Excel trade log export.

ITERATION v5 (LEVERAGE EDITION) FEATURES:
    1. DYNAMIC POSITION SIZING: Risk 6% per trade (Simulating 3x Leverage)
    2. MAX POSITION CAP: Increased to 300% to allow leverage exposure
    3. SYMBOL CHANGE: BTCUSDT replaced with BNBUSDT
    4. BEAR MARKET STRESS TEST: 2022 crypto winter validation
    5. OPTIMIZED VECTORIZATION: Fast execution for grid search
    6. EXCEL TRADE LOG: Detailed trade export in .xlsx format
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
    """Configuration for v5 production strategy (3x Leverage Simulation)."""
    
    # Data settings - SOLUSDT added, SUIUSDT removed
    SYMBOLS: List[str] = field(default_factory=lambda: ['BNBUSDT', 'ETHUSDT', 'SOLUSDT'])
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

    # ATR-based risk management
    ATR_PERIOD: int = 14
    ATR_STOP_MULTIPLE: float = 4.0
    ATR_TARGET_MULTIPLE: float = 10.0

    # DYNAMIC POSITION SIZING (LEVERAGE 3X SIMULATION)
    RISK_PER_TRADE: float = 0.06        # Risk 6% per trade (2% * 3x Leverage)
    MAX_POSITION_PCT: float = 3.00      # Max 300% of capital per trade

    # Bull market adaptive filter (NEW - helps beat Buy & Hold)
    USE_BULL_FILTER: bool = True

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
    
    def _load_from_cache(self, symbol: str, timeframe: str, 
                         start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        cache_path = self._get_cache_path(symbol, timeframe)
        if cache_path.exists():
            try:
                df = pd.read_csv(
                    cache_path,
                    index_col='timestamp',
                    parse_dates=['timestamp'],
                    date_parser=lambda x: pd.to_datetime(x, utc=True)
                )
                df.index = df.index.tz_localize(None)
                
                # Filter by date range if specified
                if start_date and end_date:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                return df
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
        if cached_data is not None and len(cached_data) > 0:
            print(f"[INFO] Loaded {symbol} {timeframe} from cache: {len(cached_data)} bars")
            return cached_data
        
        if self.client is None:
            raise RuntimeError("Binance client not available and no cache found.")
        
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
    """
    
    @staticmethod
    def wilders_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gains = delta.where(delta > 0, 0.0)
        losses = (-delta).where(delta < 0, 0.0)
        
        alpha = 1.0 / period
        
        avg_gains = gains.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        alpha = 1.0 / period
        
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        return adx.replace([np.inf, -np.inf], np.nan)
    
    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================
# SIGNAL GENERATOR (WITH BULL MARKET ADAPTIVE FILTER)
# =============================================================================

class SignalGenerator:
    """
    Signal generator with bull market adaptive filter.
    
    NEW (v5.2): In bull regimes (price > 200-EMA AND ADX > 25),
    reduces RSI entry threshold by 5 points for more aggressive entries.
    This helps the strategy capture more upside in strong bull markets
    and beat Buy & Hold performance.
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

        # Trend filter
        if self.config.USE_TREND_FILTER:
            df['ema_200'] = self.indicator_engine.ema(df['close'], self.config.TREND_EMA_PERIOD)
            df['above_trend'] = df['close'] > df['ema_200']
        else:
            df['above_trend'] = True

        # ADX filter
        if self.config.USE_ADX_FILTER:
            df['trending_market'] = df['adx'] > self.config.ADX_THRESHOLD
        else:
            df['trending_market'] = True
        
        # BULL MARKET REGIME FILTER (NEW - helps beat Buy & Hold)
        if getattr(self.config, 'USE_BULL_FILTER', False):
            df['bull_regime'] = (df['close'] > df['ema_200']) & (df['adx'] > 25)
            # Reduce RSI entry threshold by 5 in bull regime (more aggressive)
            df['rsi_entry_threshold'] = np.where(
                df['bull_regime'],
                self.config.RSI_LONG_ENTRY - 5,
                self.config.RSI_LONG_ENTRY
            )
        else:
            df['bull_regime'] = False
            df['rsi_entry_threshold'] = self.config.RSI_LONG_ENTRY

        # Volume filter
        if self.config.USE_VOLUME_FILTER:
            df['volume_sma'] = df['volume'].rolling(window=self.config.VOLUME_SMA_PERIOD).mean()
            df['high_volume'] = df['volume'] > df['volume_sma']
        else:
            df['high_volume'] = True

        df['rsi_prev'] = df['rsi'].shift(1)

        # Long entry with adaptive threshold (bull filter)
        df['long_entry_raw'] = (
            (df['rsi'] > df['rsi_entry_threshold']) &
            (df['rsi_prev'] <= df['rsi_entry_threshold']) &
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

        df['stop_loss_distance'] = df['atr'] * self.config.ATR_STOP_MULTIPLE
        df['take_profit_distance'] = df['atr'] * self.config.ATR_TARGET_MULTIPLE

        # Cleanup (remove intermediate columns)
        df.drop(columns=['rsi_prev', 'long_entry_raw', 'long_exit_raw', 
                        'rsi_entry_threshold', 'bull_regime'],
                inplace=True, errors='ignore')

        return df


# =============================================================================
# BACKTEST ENGINE WITH DYNAMIC POSITION SIZING
# =============================================================================

class BacktestEngine:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.trades = pd.DataFrame()
    
    def _calculate_position_size(self, equity: float, entry_price: float,
                                  stop_loss_price: float) -> Tuple[float, float]:
        risk_amount = equity * self.config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk <= 0:
            return 0.0, 0.0
        
        position_size = risk_amount / price_risk
        
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
        entry_equity = self.initial_capital  # FIX: Lacak saldo awal trade
        trailing_stop_price = 0.0
        highest_close = 0.0
        total_costs = 0.0
        equity = self.initial_capital

        trades = []

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
                    exit_reason = 'take_profit'
                elif sl_hit:
                    exit_price = trailing_stop_price
                    exit_reason = 'trailing_stop'
                elif trend_exhaustion:
                    exit_price = open_price * (1 - self.config.SLIPPAGE)
                    exit_reason = 'trend_exhaustion'
                else:
                    exit_price = open_price * (1 - self.config.SLIPPAGE)
                    exit_reason = 'rsi_emergency'

                pnl = (exit_price - entry_price) * position_size
                trade_value = abs(position_size * exit_price)
                holding_periods = curr_idx - entry_bar
                funding_cost = position_size * self.config.FUNDING_RATE * (holding_periods / 2)
                fees = trade_value * self.config.TRADING_FEE
                slippage_cost = trade_value * self.config.SLIPPAGE
                total_trade_cost = fees + slippage_cost + funding_cost

                equity += pnl - total_trade_cost
                
                # FIX: Hitung persentase ROI nyata pada Ekuitas Akun
                roi_on_equity = (pnl - total_trade_cost) / entry_equity

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
                    'entry_equity': entry_equity,     # Disimpan untuk Monte Carlo
                    'roi_on_equity': roi_on_equity,   # Disimpan untuk Monte Carlo
                    'exit_reason': exit_reason,
                    'entry_atr': entry_atr,
                    'highest_close': highest_close
                })

                position = 0
                position_size = 0.0
                trailing_stop_price = 0.0
                highest_close = 0.0

            if signal and position == 0:
                entry_price = open_price * (1 + self.config.SLIPPAGE)
                entry_atr = atr if atr > 0 else 0.02 * entry_price

                stop_loss_distance = entry_atr * self.config.ATR_STOP_MULTIPLE
                stop_loss_price = entry_price - stop_loss_distance
                
                position_size, actual_risk = self._calculate_position_size(
                    equity, entry_price, stop_loss_price
                )

                if position_size > 0:
                    position = 1
                    entry_bar = curr_idx
                    entry_equity = equity  # FIX: Rekam ekuitas tepat saat posisi dibuka
                    trailing_stop_price = stop_loss_price
                    highest_close = entry_price

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
    
    def export_trade_log(self, symbol: str, output_dir: Path) -> Optional[Path]:
        """
        Export detailed trade log to Excel file format.
        
        V5 FEATURE: Professional trade log with all trade details,
        performance metrics, and summary statistics.
        
        Args:
            symbol: Trading symbol (e.g., 'BNBUSDT')
            output_dir: Directory to save Excel file
        
        Returns:
            Path to saved Excel file, or None if no trades
        """
        if self.trades.empty:
            print(f"[INFO] No trades to export for {symbol}")
            return None
        
        # Prepare trade log DataFrame with enhanced columns
        trade_log = self.trades.copy()
        
        # Add calculated columns
        if not trade_log.empty:
            # Duration
            trade_log['duration_hours'] = (
                trade_log['exit_date'] - trade_log['entry_date']
            ).dt.total_seconds() / 3600
            
            # Return %
            trade_log['return_pct'] = (
                (trade_log['exit_price'] - trade_log['entry_price']) / 
                trade_log['entry_price'] * 100
            )
            
            # P&L per unit
            trade_log['pnl_per_unit'] = trade_log['exit_price'] - trade_log['entry_price']
            
            # ROI % (including costs)
            trade_log['roi_pct'] = (
                trade_log['net_pnl'] / 
                (trade_log['entry_price'] * trade_log['position_size']) * 100
            )
            
            # Format dates
            trade_log['entry_date'] = trade_log['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
            trade_log['exit_date'] = trade_log['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Round numeric columns
            numeric_cols = ['entry_price', 'exit_price', 'position_size', 'pnl', 
                           'costs', 'net_pnl', 'duration_hours', 'return_pct', 
                           'pnl_per_unit', 'roi_pct', 'entry_atr', 'highest_close']
            for col in numeric_cols:
                if col in trade_log.columns:
                    trade_log[col] = trade_log[col].round(4)
            
            # Rename columns for readability
            column_names = {
                'entry_date': 'Entry Date',
                'exit_date': 'Exit Date',
                'direction': 'Direction',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'position_size': 'Position Size',
                'pnl': 'Gross P&L ($)',
                'costs': 'Total Costs ($)',
                'net_pnl': 'Net P&L ($)',
                'exit_reason': 'Exit Reason',
                'duration_hours': 'Duration (Hours)',
                'return_pct': 'Return (%)',
                'pnl_per_unit': 'P&L per Unit ($)',
                'roi_pct': 'ROI (%)',
                'entry_atr': 'Entry ATR',
                'highest_close': 'Highest Close'
            }
            trade_log = trade_log.rename(columns=column_names)
        
        # Create Excel writer with multiple sheets
        excel_path = output_dir / f'{symbol}_trade_log.xlsx'
        
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Sheet 1: Detailed Trade Log
                trade_log.to_excel(writer, sheet_name='Trade Log', index=False)
                
                # Sheet 2: Summary Statistics
                summary_data = {
                    'Metric': [
                        'Total Trades',
                        'Winning Trades',
                        'Losing Trades',
                        'Win Rate (%)',
                        'Total Net P&L ($)',
                        'Total Gross Profit ($)',
                        'Total Gross Loss ($)',
                        'Profit Factor',
                        'Average Net P&L ($)',
                        'Largest Winning Trade ($)',
                        'Largest Losing Trade ($)',
                        'Average Duration (Hours)',
                        'Total Costs ($)'
                    ],
                    'Value': [
                        len(self.trades),
                        (self.trades['net_pnl'] > 0).sum(),
                        (self.trades['net_pnl'] <= 0).sum(),
                        f"{(self.trades['net_pnl'] > 0).sum() / len(self.trades) * 100:.2f}%",
                        f"{self.trades['net_pnl'].sum():.2f}",
                        f"{self.trades[self.trades['net_pnl'] > 0]['net_pnl'].sum():.2f}",
                        f"{abs(self.trades[self.trades['net_pnl'] <= 0]['net_pnl'].sum()):.2f}",
                        f"{self.trades[self.trades['net_pnl'] > 0]['net_pnl'].sum() / abs(self.trades[self.trades['net_pnl'] <= 0]['net_pnl'].sum()):.2f}" if (self.trades['net_pnl'] <= 0).sum() > 0 else 'N/A',
                        f"{self.trades['net_pnl'].mean():.2f}",
                        f"{self.trades['net_pnl'].max():.2f}",
                        f"{self.trades['net_pnl'].min():.2f}",
                        f"{trade_log['Duration (Hours)'].mean():.2f}" if 'Duration (Hours)' in trade_log.columns else 'N/A',
                        f"{self.trades['costs'].sum():.2f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 3: Exit Breakdown
                exit_breakdown = {
                    'Exit Reason': ['Trailing Stop', 'Take Profit', 'RSI Emergency', 'Trend Exhaustion'],
                    'Count': [
                        (self.trades['exit_reason'] == 'trailing_stop').sum(),
                        (self.trades['exit_reason'] == 'take_profit').sum(),
                        (self.trades['exit_reason'] == 'rsi_emergency').sum(),
                        (self.trades['exit_reason'] == 'trend_exhaustion').sum()
                    ],
                    'Percentage': [
                        f"{(self.trades['exit_reason'] == 'trailing_stop').sum() / len(self.trades) * 100:.1f}%",
                        f"{(self.trades['exit_reason'] == 'take_profit').sum() / len(self.trades) * 100:.1f}%",
                        f"{(self.trades['exit_reason'] == 'rsi_emergency').sum() / len(self.trades) * 100:.1f}%",
                        f"{(self.trades['exit_reason'] == 'trend_exhaustion').sum() / len(self.trades) * 100:.1f}%"
                    ]
                }
                breakdown_df = pd.DataFrame(exit_breakdown)
                breakdown_df.to_excel(writer, sheet_name='Exit Analysis', index=False)
                
                # Format Excel sheets
                workbook = writer.book
                worksheet1 = writer.sheets['Trade Log']
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#2E86AB',
                    'font_color': 'white',
                    'border': 1
                })
                
                money_format = workbook.add_format({'num_format': '$#,##0.00'})
                pct_format = workbook.add_format({'num_format': '0.00%'})
                datetime_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm'})
                
                # Apply column widths
                worksheet1.set_column('A:A', 18)  # Entry Date
                worksheet1.set_column('B:B', 18)  # Exit Date
                worksheet1.set_column('C:C', 12)  # Direction
                worksheet1.set_column('D:D', 12)  # Entry Price
                worksheet1.set_column('E:E', 12)  # Exit Price
                worksheet1.set_column('F:F', 15)  # Position Size
                worksheet1.set_column('G:G', 12)  # Gross P&L
                worksheet1.set_column('H:H', 12)  # Costs
                worksheet1.set_column('I:I', 12)  # Net P&L
                worksheet1.set_column('J:J', 18)  # Exit Reason
                worksheet1.set_column('K:K', 15)  # Duration
                worksheet1.set_column('L:L', 12)  # Return %
                worksheet1.set_column('M:M', 12)  # ROI %
                
                print(f"[INFO] Exported trade log to Excel: {excel_path}")
                
        except Exception as e:
            print(f"[WARNING] Failed to export trade log: {e}")
            return None
        
        return excel_path
    
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
                label='Production v5 Strategy (3x Leverage)', linewidth=2.5, color='#2E86AB')
        ax1.plot(buyhold_norm.index, buyhold_norm.values,
                label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
        ax1.set_ylabel('Normalized Value ($)', fontsize=12)
        ax1.set_title(f'{symbol} - Production v5 Performance (6% Risk)\n'
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

    def plot_robustness_heatmaps(self, optimization_results: pd.DataFrame,
                                  symbol: str) -> None:
        """Generate parameter robustness heatmaps with EXPANDED parameter space."""
        if optimization_results.empty:
            print("[WARNING] No optimization results for heatmap")
            return

        # Filter to expanded parameter ranges (matches robustness matrix)
        filtered = optimization_results[
            (optimization_results['rsi_entry'].between(58, 72)) &
            (optimization_results['rsi_exit'].between(25, 50))
        ].copy()

        if filtered.empty:
            print("[WARNING] No results in heatmap parameter range")
            return

        sharpe_pivot = filtered.pivot_table(
            index='rsi_exit', columns='rsi_entry', values='sharpe_ratio', aggfunc='mean'
        )

        return_pivot = filtered.pivot_table(
            index='rsi_exit', columns='rsi_entry', values='total_return', aggfunc='mean'
        )

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        cmap = plt.cm.RdYlBu_r

        ax1 = axes[0]
        sns.heatmap(
            sharpe_pivot,
            annot=True,
            fmt='.2f',
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
        ax1.set_title('Parameter Robustness: Sharpe Ratio (Expanded Matrix)\n(Higher = Better)',
                     fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=10)

        ax2 = axes[1]
        sns.heatmap(
            return_pivot,
            annot=True,
            fmt='.0f',
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
        ax2.set_title('Parameter Robustness: Total Return (%) (Expanded Matrix)\n(Higher = Better)',
                     fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=10)

        plt.suptitle(f'{symbol} - Strategy Robustness Analysis (v5 - 49 Combinations)\n'
                    f'Parameter Space: Entry (58-72) × Exit (25-50)',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'{symbol}_robustness_heatmap.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved robustness heatmap: {save_path}")
    
    def plot_walk_forward_results(self, wfo_df: pd.DataFrame, symbol: str) -> None:
        """Plot Walk-Forward Optimization results."""
        if wfo_df.empty:
            print("[WARNING] No WFO results to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        window_nums = wfo_df['window'].values
        
        ax1 = axes[0]
        ax1.bar(window_nums - 0.2, wfo_df['is_sharpe'].values, width=0.4,
               label='In-Sample Sharpe', color='#2E86AB', alpha=0.7)
        ax1.bar(window_nums + 0.2, wfo_df['oos_sharpe'].values, width=0.4,
               label='Out-of-Sample Sharpe', color='#A23B72', alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=1.0, color='green', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Window Number', fontsize=12)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.set_title(f'{symbol} - Walk-Forward IS vs OOS Performance', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        oos_cumulative = (1 + wfo_df['oos_return_pct'] / 100).cumprod()
        ax2.plot(window_nums, oos_cumulative.values, linewidth=2, color='#F18F01',
                marker='o', markersize=8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Window Number', fontsize=12)
        ax2.set_ylabel('Cumulative Return', fontsize=12)
        ax2.set_title(f'{symbol} - Cumulative OOS Returns Across Windows', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{symbol}_walk_forward_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved WFO plot: {save_path}")
        plt.show()
    
    def plot_monte_carlo_histogram(self, results: Dict, symbol: str) -> None:
        """Plot Monte Carlo permutation test results histogram."""
        if 'final_returns_array' not in results:
            print("[WARNING] No Monte Carlo results to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        final_returns = results['final_returns_array']
        actual_return = results['actual_return_pct']
        
        # Handle case where all returns are identical (no variance)
        if results['std_final_return'] == 0 or np.isnan(results['std_final_return']):
            # Plot as vertical line instead of histogram
            ax.axvline(x=actual_return, color='purple', linestyle='-', linewidth=3,
                      label=f'Actual Return ({actual_return:.1f}%)')
            ax.axvline(x=results['mean_final_return'], color='green', linestyle='--', linewidth=2,
                      label=f'Mean Simulated ({results["mean_final_return"]:.1f}%)')
            ax.set_title(f'{symbol} - Monte Carlo Results (All Simulations Identical)\n'
                        f'Actual Percentile: {results["actual_percentile"]:.1f}% | '
                        f'Z-Score: {results["z_score"]:.2f}', fontsize=14)
        else:
            # Normal histogram
            ax.hist(final_returns, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(x=results['percentile_5th'], color='red', linestyle='--', linewidth=2,
                      label=f'5th Percentile ({results["percentile_5th"]:.1f}%)')
            ax.axvline(x=results['median_final_return'], color='green', linestyle='--', linewidth=2,
                      label=f'Median ({results["median_final_return"]:.1f}%)')
            ax.axvline(x=results['percentile_95th'], color='orange', linestyle='--', linewidth=2,
                      label=f'95th Percentile ({results["percentile_95th"]:.1f}%)')
            ax.axvline(x=actual_return, color='purple', linestyle='-', linewidth=3,
                      label=f'Actual Return ({actual_return:.1f}%)')
            ax.set_title(f'{symbol} - Monte Carlo Permutation Test\n'
                        f'{results["num_simulations"]} Simulations | '
                        f'Actual Percentile: {results["actual_percentile"]:.1f}% | '
                        f'Z-Score: {results["z_score"]:.2f}', fontsize=14)
        
        ax.set_xlabel('Final Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        stats_text = (f"Mean: {results['mean_final_return']:.2f}%\n"
                     f"Std Dev: {results['std_final_return']:.2f}%\n"
                     f"% Profitable: {results['percent_profitable_sims']:.1f}%")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        save_path = self.output_dir / f'{symbol}_monte_carlo_histogram.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved Monte Carlo plot: {save_path}")
        plt.show()


# =============================================================================
# ROBUSTNESS ANALYZER - WALK-FORWARD & MONTE CARLO (ADVANCED)
# =============================================================================

class RobustnessAnalyzer:
    """
    Advanced robustness analyzer with Walk-Forward Optimization and Monte Carlo tests.
    
    Methods:
    1. run_walk_forward_optimization() - Rolling IS/OOS validation
    2. run_monte_carlo_permutation_test() - Statistical significance testing
    3. run_grid_search() - Legacy grid search (retained for comparison)
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data_engine = DataEngine(config)
        self.indicator_engine = IndicatorEngine()
    
    def run_walk_forward_optimization(self, symbol: str, timeframe: str = None) -> Tuple[pd.DataFrame, Dict]:
        tf = timeframe or self.config.TIMEFRAME
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD OPTIMIZATION - {symbol}")
        print(f"{'='*70}")
        
        df = self.data_engine.fetch_symbol_data(symbol, tf)
        n_bars = len(df)
        
        # FIX: Ubah dari 12B/6B menjadi 6 Bulan / 2 Bulan (Timeframe 4H = 6 bar/hari)
        is_window_bars = 1095  # 6 Bulan In-Sample
        oos_window_bars = 365  # 2 Bulan Out-of-Sample
        step_bars = 365        # Bergeser 2 bulan ke depan setiap iterasi
        
        print(f"[INFO] Total bars: {n_bars}")
        print(f"[INFO] IS Window: {is_window_bars} bars (6 months)")
        print(f"[INFO] OOS Window: {oos_window_bars} bars (2 months)")
        print(f"[INFO] Step Size: {step_bars} bars (2 months)")
        
        print("[INFO] Pre-calculating indicators...")
        rsi = self.indicator_engine.wilders_rsi(df['close'], 14)
        atr = self.indicator_engine.wilders_atr(df['high'], df['low'], df['close'], 14)
        adx = self.indicator_engine.adx(df['high'], df['low'], df['close'], 14)
        ema_200 = self.indicator_engine.ema(df['close'], 200)
        ema_20 = self.indicator_engine.ema(df['close'], 20)
        volume_sma = df['volume'].rolling(20).mean()
        
        entry_range = [58, 60, 62, 65, 67, 70, 72]
        exit_range = [25, 28, 30, 35, 40, 45, 50]
        
        results = []
        window_count = 0
        is_start = 0
        
        while is_start + is_window_bars + oos_window_bars <= n_bars:
            window_count += 1
            is_end = is_start + is_window_bars
            oos_start = is_end
            oos_end = oos_start + oos_window_bars
            
            is_indices = slice(is_start, is_end)
            oos_indices = slice(oos_start, oos_end)
            
            print(f"\n[INFO] Window {window_count}: IS [{df.index[is_start].strftime('%Y-%m-%d')} to {df.index[is_end-1].strftime('%Y-%m-%d')}]")
            best_params, best_sharpe = self._optimize_is_window(
                df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                is_indices, entry_range, exit_range
            )
            
            if best_params is None:
                print(f"[WARNING] No valid trades in IS window {window_count}")
                is_start += step_bars
                continue
            
            print(f"[INFO] Best IS Params: Entry={best_params['entry']}, Exit={best_params['exit']}, Sharpe={best_sharpe:.2f}")
            
            oos_metrics = self._evaluate_oos_window(
                df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                oos_indices, best_params['entry'], best_params['exit']
            )
            
            if oos_metrics is not None:
                results.append({
                    'window': window_count,
                    'is_start': df.index[is_start],
                    'oos_start': df.index[oos_start],
                    'best_entry': best_params['entry'],
                    'best_exit': best_params['exit'],
                    'is_sharpe': best_sharpe,
                    'oos_sharpe': oos_metrics['sharpe'],
                    'oos_return_pct': oos_metrics['return_pct'],
                    'oos_max_dd_pct': oos_metrics['max_dd_pct'],
                    'oos_profit_factor': oos_metrics['profit_factor'],
                    'oos_trades': oos_metrics['trades']
                })
            
            is_start += step_bars
        
        results_df = pd.DataFrame(results)
        summary_stats = self._calculate_wfo_summary(results_df)
        self._print_wfo_summary(summary_stats)
        
        return results_df, summary_stats
    
    def _optimize_is_window(self, df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                            indices, entry_range, exit_range) -> Tuple[Optional[Dict], float]:
        """Optimize parameters on In-Sample window."""
        rsi_is = rsi.iloc[indices].values
        atr_is = atr.iloc[indices].values
        adx_is = adx.iloc[indices].values
        ema_200_is = ema_200.iloc[indices].values
        ema_20_is = ema_20.iloc[indices].values
        volume_sma_is = volume_sma.iloc[indices].values
        
        close_is = df['close'].iloc[indices].values
        high_is = df['high'].iloc[indices].values
        low_is = df['low'].iloc[indices].values
        open_is = df['open'].iloc[indices].values
        volume_is = df['volume'].iloc[indices].values
        
        above_trend = (close_is > ema_200_is)
        trending_market = (adx_is > 20)
        high_volume = (volume_sma_is < volume_is)
        rsi_prev = np.roll(rsi_is, 1)
        rsi_prev[0] = rsi_is[0]
        
        best_sharpe = -np.inf
        best_params = None
        
        for entry in entry_range:
            for exit_val in exit_range:
                long_entry = (
                    (rsi_is > entry) &
                    (rsi_prev <= entry) &
                    above_trend &
                    trending_market &
                    high_volume
                )
                long_entry = np.roll(long_entry, 1)
                long_entry[0] = False
                
                long_exit = (
                    (rsi_is < exit_val) &
                    (rsi_prev >= exit_val)
                )
                long_exit = np.roll(long_exit, 1)
                long_exit[0] = False
                
                equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
                    open_is, high_is, low_is, close_is, atr_is,
                    ema_20_is, long_entry, long_exit,
                    4.0, 10.0, 0.06, 3.0, 0.001, 0.0005, 0.0001, 1000.0
                )
                
                if total_trades > 10:
                    returns = pd.Series(equity).pct_change().dropna()
                    if returns.std() > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6)
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {'entry': entry, 'exit': exit_val}
        
        return best_params, best_sharpe if best_params else 0.0
    
    def _evaluate_oos_window(self, df, rsi, atr, adx, ema_200, ema_20, volume_sma,
                             indices, entry, exit_val) -> Optional[Dict]:
        """Evaluate fixed parameters on Out-of-Sample window."""
        rsi_oos = rsi.iloc[indices].values
        atr_oos = atr.iloc[indices].values
        adx_oos = adx.iloc[indices].values
        ema_200_oos = ema_200.iloc[indices].values
        ema_20_oos = ema_20.iloc[indices].values
        volume_sma_oos = volume_sma.iloc[indices].values
        
        close_oos = df['close'].iloc[indices].values
        high_oos = df['high'].iloc[indices].values
        low_oos = df['low'].iloc[indices].values
        open_oos = df['open'].iloc[indices].values
        volume_oos = df['volume'].iloc[indices].values
        
        above_trend = (close_oos > ema_200_oos)
        trending_market = (adx_oos > 20)
        high_volume = (volume_sma_oos < volume_oos)
        rsi_prev = np.roll(rsi_oos, 1)
        rsi_prev[0] = rsi_oos[0]
        
        long_entry = (
            (rsi_oos > entry) &
            (rsi_prev <= entry) &
            above_trend &
            trending_market &
            high_volume
        )
        long_entry = np.roll(long_entry, 1)
        long_entry[0] = False
        
        long_exit = (
            (rsi_oos < exit_val) &
            (rsi_prev >= exit_val)
        )
        long_exit = np.roll(long_exit, 1)
        long_exit[0] = False
        
        equity, total_trades, win_count, gross_profit, gross_loss = self._fast_backtest(
            open_oos, high_oos, low_oos, close_oos, atr_oos,
            ema_20_oos, long_entry, long_exit,
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
    
    def _calculate_wfo_summary(self, results_df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for WFO results."""
        if results_df.empty:
            return {'error': 'No WFO results'}
        
        oos_sharpes = results_df['oos_sharpe']
        oos_returns = results_df['oos_return_pct']
        
        return {
            'num_windows': len(results_df),
            'avg_oos_sharpe': oos_sharpes.mean(),
            'median_oos_sharpe': oos_sharpes.median(),
            'std_oos_sharpe': oos_sharpes.std(),
            'min_oos_sharpe': oos_sharpes.min(),
            'max_oos_sharpe': oos_sharpes.max(),
            'avg_oos_return_pct': oos_returns.mean(),
            'median_oos_return_pct': oos_returns.median(),
            'pct_profitable_windows': (oos_returns > 0).sum() / len(oos_returns) * 100,
            'stability_score': oos_sharpes.mean() / oos_sharpes.std() if oos_sharpes.std() > 0 else np.inf,
            'avg_oos_trades': results_df['oos_trades'].mean(),
            'avg_oos_profit_factor': results_df['oos_profit_factor'].mean()
        }
    
    def _print_wfo_summary(self, summary: Dict) -> None:
        """Print WFO summary statistics."""
        print(f"\n{'='*70}")
        print("WALK-FORWARD OPTIMIZATION SUMMARY")
        print(f"{'='*70}")
        print(f"  Number of Windows:        {summary.get('num_windows', 'N/A')}")
        print(f"  Average OOS Sharpe:       {summary.get('avg_oos_sharpe', 0):.2f}")
        print(f"  Median OOS Sharpe:        {summary.get('median_oos_sharpe', 0):.2f}")
        print(f"  Std Dev OOS Sharpe:       {summary.get('std_oos_sharpe', 0):.2f}")
        print(f"  Min OOS Sharpe:           {summary.get('min_oos_sharpe', 0):.2f}")
        print(f"  Max OOS Sharpe:           {summary.get('max_oos_sharpe', 0):.2f}")
        print(f"  Stability Score:          {summary.get('stability_score', 0):.2f}")
        print(f"  Avg OOS Return:           {summary.get('avg_oos_return_pct', 0):.2f}%")
        print(f"  % Profitable Windows:     {summary.get('pct_profitable_windows', 0):.1f}%")
        print(f"  Avg OOS Trades:           {summary.get('avg_oos_trades', 0):.1f}")
        print(f"  Avg OOS Profit Factor:    {summary.get('avg_oos_profit_factor', 0):.2f}")
        print(f"{'='*70}")
    
    def run_monte_carlo_bootstrap(self, symbol: str, num_simulations: int = 2000,
                                   use_block_bootstrap: bool = True, block_size: int = 10) -> Dict:
        print(f"\n{'='*80}")
        print(f"MONTE CARLO BOOTSTRAP SIMULATION (v6.1 FIXED) - {symbol}")
        print(f"{'='*80}")
        
        df = self.data_engine.fetch_symbol_data(symbol, self.config.TIMEFRAME)
        df = SignalGenerator(self.config).generate_signals(df)
        backtest = BacktestEngine(self.config)
        df = backtest.run_backtest(df, symbol)
        
        if backtest.trades.empty or len(backtest.trades) < 10:
            print(f"[ERROR] Insufficient trades (< 10). Cannot run bootstrap.")
            return {'error': 'Insufficient trades'}
        
        # FIX: Ambil True Equity ROI (sudah memperhitungkan 3x leverage / 6% risk)
        trade_rois = backtest.trades['roi_on_equity'].values
        n_trades = len(trade_rois)
        
        # FIX: Dynamic Block Sizing untuk mencegah Zero Variance
        actual_block_size = block_size
        if use_block_bootstrap:
            actual_block_size = min(block_size, max(1, n_trades // 4))
            if actual_block_size < 2:
                print(f"[INFO] Too few trades ({n_trades}) for block bootstrap. Falling back to Pure Bootstrap.")
                use_block_bootstrap = False
        
        print(f"[INFO] Configuration:")
        print(f"  Simulations:          {num_simulations:,}")
        print(f"  Method:               {'Block Bootstrap' if use_block_bootstrap else 'Pure Bootstrap'}")
        if use_block_bootstrap:
            print(f"  Block Size:           {actual_block_size} trades (Dynamically adjusted)")
        print(f"  Original Trades:      {n_trades}")
        print(f"  Mean Equity ROI:      {trade_rois.mean()*100:.2f}% per trade")
        
        print(f"\n[INFO] Running {num_simulations:,} bootstrap simulations...")
        
        np.random.seed(42)
        final_returns = np.zeros(num_simulations)
        max_drawdowns = np.zeros(num_simulations)
        sharpe_ratios = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            # Bootstrap resampling
            if use_block_bootstrap:
                # Manual block sampling logic
                n_blocks = int(np.ceil(n_trades / actual_block_size))
                blocks = [trade_rois[j:j+actual_block_size] for j in range(0, n_trades, actual_block_size) if len(trade_rois[j:j+actual_block_size]) > 0]
                sampled_blocks = [blocks[np.random.randint(0, len(blocks))] for _ in range(n_blocks)]
                sampled_rois = np.concatenate(sampled_blocks)[:n_trades]
            else:
                sampled_rois = np.random.choice(trade_rois, size=n_trades, replace=True)
            
            # Compound equity curve
            equity = 1000.0
            equity_curve = [equity]
            for roi in sampled_rois:
                equity *= (1 + roi)
                equity_curve.append(equity)
            equity_curve = np.array(equity_curve)
            
            final_returns[i] = (equity_curve[-1] / 1000.0 - 1) * 100
            
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdowns[i] = np.min(drawdown) * 100
            
            trade_returns = pd.Series(equity_curve).pct_change().dropna()
            if len(trade_returns) > 2 and trade_returns.std() > 0:
                sharpe_ratios[i] = trade_returns.mean() / trade_returns.std() * np.sqrt(2190)
            else:
                sharpe_ratios[i] = 0.0
        
        # Calculate actual strategy metrics
        actual_return = backtest.calculate_metrics(df['equity'])['total_return'] * 100
        actual_max_dd = backtest.calculate_metrics(df['equity'])['max_drawdown'] * 100
        actual_sharpe = backtest.calculate_metrics(df['equity'])['sharpe_ratio']
        
        # === FULL DICTIONARY RESULTS (MENCEGAH KEYERROR) ===
        results = {
            'symbol': symbol,
            'num_simulations': num_simulations,
            'num_trades_original': n_trades,
            
            # --- FINAL RETURN DISTRIBUTION ---
            'final_return_mean': float(final_returns.mean()),
            'final_return_median': float(np.median(final_returns)),
            'final_return_std': float(final_returns.std()),
            'final_return_min': float(np.min(final_returns)),
            'final_return_max': float(np.max(final_returns)),
            'final_return_p5': float(np.percentile(final_returns, 5)),
            'final_return_p95': float(np.percentile(final_returns, 95)),
            
            # --- ACTUAL STRATEGY PERFORMANCE ---
            'actual_final_return': float(actual_return),
            'actual_max_dd': float(actual_max_dd),
            'actual_sharpe': float(actual_sharpe),
            
            # --- PROBABILISTIC METRICS ---
            'prob_beat_actual': float((final_returns > actual_return).sum() / num_simulations * 100),
            'prob_positive_return': float((final_returns > 0).sum() / num_simulations * 100),
            'prob_sharpe_gt_1': float((sharpe_ratios > 1.0).sum() / num_simulations * 100),
            # Asumsi DD bernilai negatif (misal -25%), maka < -20 berarti lebih buruk dari -20%
            'prob_dd_worse_than_20': float((max_drawdowns < -20.0).sum() / num_simulations * 100), 
            
            # --- ACTUAL STRATEGY PERCENTILE RANKING ---
            'actual_return_percentile': float((final_returns < actual_return).sum() / num_simulations * 100),
            'actual_sharpe_percentile': float((sharpe_ratios < actual_sharpe).sum() / num_simulations * 100),
            'actual_dd_percentile': float((max_drawdowns < actual_max_dd).sum() / num_simulations * 100),
            
            # --- Z-SCORES ---
            'z_score_return': float((actual_return - final_returns.mean()) / final_returns.std()) if final_returns.std() > 0 else 0.0,
            'z_score_sharpe': float((actual_sharpe - sharpe_ratios.mean()) / sharpe_ratios.std()) if sharpe_ratios.std() > 0 else 0.0,
            
            # --- RAW ARRAYS (For plotting) ---
            'final_returns_array': final_returns,
            'max_drawdowns_array': max_drawdowns,
            'sharpe_ratios_array': sharpe_ratios
        }
        
        self._print_monte_carlo_bootstrap_summary(results)
        return results
    
    def _block_bootstrap_sample(self, trade_rois: np.ndarray, n_trades: int, block_size: int) -> np.ndarray:
        """
        Block bootstrap: sample blocks of trades with replacement.
        Preserves autocorrelation and volatility clustering within blocks.
        """
        n_blocks = int(np.ceil(n_trades / block_size))
        
        # Create non-overlapping blocks
        blocks = []
        for i in range(0, n_trades, block_size):
            block = trade_rois[i:i+block_size]
            if len(block) > 0:
                blocks.append(block)
        
        if len(blocks) == 0:
            return trade_rois
        
        # Sample blocks with replacement
        sampled_blocks = []
        for _ in range(n_blocks):
            block_idx = np.random.randint(0, len(blocks))
            sampled_blocks.append(blocks[block_idx])
        
        # Concatenate and trim to original length
        sampled_rois = np.concatenate(sampled_blocks)[:n_trades]
        
        # Pad if needed
        if len(sampled_rois) < n_trades:
            padding = np.random.choice(trade_rois, size=n_trades - len(sampled_rois), replace=True)
            sampled_rois = np.concatenate([sampled_rois, padding])
        
        return sampled_rois[:n_trades]
    
    def _print_monte_carlo_bootstrap_summary(self, results: Dict) -> None:
        """Print Monte Carlo bootstrap summary."""
        print(f"\n{'='*80}")
        print(f"MONTE CARLO BOOTSTRAP SIMULATION RESULTS")
        print(f"{'='*80}")
        
        print(f"\n--- FINAL RETURN DISTRIBUTION ---")
        print(f"  Mean:                   {results['final_return_mean']:.2f}%")
        print(f"  Median:                 {results['final_return_median']:.2f}%")
        print(f"  Std Dev:                {results['final_return_std']:.2f}%")
        print(f"  Range:                  [{results['final_return_min']:.2f}%, {results['final_return_max']:.2f}%]")
        print(f"  95% CI:                 [{results['final_return_p5']:.2f}%, {results['final_return_p95']:.2f}%]")
        
        print(f"\n--- ACTUAL STRATEGY PERFORMANCE ---")
        print(f"  Actual Final Return:    {results['actual_final_return']:.2f}%")
        print(f"  Actual Max DD:          {results['actual_max_dd']:.2f}%")
        print(f"  Actual Sharpe:          {results['actual_sharpe']:.2f}")
        
        print(f"\n--- PROBABILISTIC METRICS ---")
        print(f"  % Sims Beat Actual:     {results['prob_beat_actual']:.1f}%")
        print(f"  % Positive Return:      {results['prob_positive_return']:.1f}%")
        print(f"  % Sharpe > 1.0:         {results['prob_sharpe_gt_1']:.1f}%")
        print(f"  % DD Worse than -20%:   {results['prob_dd_worse_than_20']:.1f}%")
        
        print(f"\n--- ACTUAL STRATEGY PERCENTILE RANKING ---")
        print(f"  Return Percentile:      {results['actual_return_percentile']:.1f}%")
        print(f"  Sharpe Percentile:      {results['actual_sharpe_percentile']:.1f}%")
        print(f"  DD Percentile:          {results['actual_dd_percentile']:.1f}%")
        
        print(f"\n--- Z-SCORES ---")
        print(f"  Return Z-Score:         {results['z_score_return']:.2f}")
        print(f"  Sharpe Z-Score:         {results['z_score_sharpe']:.2f}")
        
        # Statistical significance verdict
        percentile = results['actual_return_percentile']
        z_score = abs(results['z_score_return'])
        
        print(f"\n--- STATISTICAL SIGNIFICANCE VERDICT ---")
        if percentile > 95 and z_score > 1.96:
            print(f"  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
            print(f"    Strategy shows genuine edge beyond bootstrap variance")
        elif percentile > 90 and z_score > 1.64:
            print(f"  ⚠ MARGINALLY SIGNIFICANT (p < 0.10)")
            print(f"    Some evidence of edge, but bootstrap shows variability")
        elif percentile > 75:
            print(f"  ⚠ WEAK EVIDENCE")
            print(f"    Actual performance better than 75% of bootstrap samples")
        else:
            print(f"  ✗ NOT STATISTICALLY SIGNIFICANT")
            print(f"    Performance within normal bootstrap variance (could be luck)")
        
        print(f"{'='*80}")

    def _print_monte_carlo_summary(self, results: Dict) -> None:
        """Print Monte Carlo summary statistics."""
        print(f"\n{'='*70}")
        print("MONTE CARLO PERMUTATION TEST SUMMARY")
        print(f"{'='*70}")
        print(f"  Number of Simulations:    {results.get('num_simulations', 'N/A')}")
        print(f"  Number of Trades:         {results.get('num_trades', 'N/A')}")
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
        
        if results.get('actual_percentile', 0) > 95:
            print(f"\n  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
            print(f"    Strategy shows genuine edge beyond random chance")
        elif results.get('actual_percentile', 0) > 90:
            print(f"\n  ⚠ MARGINALLY SIGNIFICANT (p < 0.10)")
            print(f"    Some evidence of edge, but more data needed")
        else:
            print(f"\n  ✗ NOT STATISTICALLY SIGNIFICANT")
            print(f"    Performance could be due to luck")
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
        print(f"[INFO] Running {total} backtest combinations (EXPANDED MATRIX)...")

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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# IMPORT MODUL MONTE CARLO EKSTERNAL
from monte_carlo_bootstrap_v6 import MonteCarloConfig, MonteCarloAnalyzer

def run_production_backtest(config: StrategyConfig) -> Dict:
    print("\n" + "="*70)
    print("PRODUCTION BACKTEST (v5 - 3x Leverage) - Main Period (2024-2026)")
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
        if df is None or len(df) == 0:
            print(f"[WARNING] No data available for {symbol}")
            continue
            
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{'='*40}")
        print("PERFORMANCE METRICS")
        print(f"{'='*40}")
        print(f"  Total Return:     {metrics.get('total_return_pct', 'N/A')}")
        print(f"  CAGR:             {metrics.get('cagr_pct', 'N/A')}")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 'N/A')}")
        print(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
        print(f"  Avg Trade:        ${metrics.get('avg_trade', 0):.2f}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return*100:.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
            'trades': backtest_engine.trades,
        }

        # 1. Plot Equity Curve
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'production_v5'
        )
        
        # 2. Export Excel Log
        backtest_engine.export_trade_log(symbol, viz_engine.output_dir)

        # 3. JALANKAN MONTE CARLO BOOTSTRAP (4-Panel Chart)
        if not backtest_engine.trades.empty:
            mc_config = MonteCarloConfig(num_simulations=2000, initial_capital=1000.0)
            mc_analyzer = MonteCarloAnalyzer(mc_config)
            mc_analyzer.run_simulation(backtest_engine.trades, symbol)

    return results


def run_stress_test(config: StrategyConfig) -> Dict:
    print("\n" + "="*70)
    print("BEAR MARKET STRESS TEST (v5 - 3x Leverage) - 2022 Crypto Winter")
    print("="*70)
    print(f"Period: {config.STRESS_TEST_START} to {config.STRESS_TEST_END}")
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
        
        if df is None or len(df) == 0:
            print(f"[WARNING] No data available for {symbol} stress test")
            continue
        
        df = signal_generator.generate_signals(df)
        df = backtest_engine.run_backtest(df, symbol)
        metrics = backtest_engine.calculate_metrics(df['equity'])
        
        print(f"\n{'='*40}")
        print("BEAR MARKET METRICS")
        print(f"{'='*40}")
        print(f"  Total Return:     {metrics.get('total_return_pct', 'N/A')}")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 'N/A')}")
        print(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
        
        buy_hold_equity = calculate_buy_hold_equity(df, config.INITIAL_CAPITAL)
        bh_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0]) - 1
        print(f"\n  Buy & Hold Return: {bh_return*100:.2f}%")
        
        if 'total_return' in metrics:
            alpha = metrics['total_return'] - bh_return
            print(f"  Strategy Alpha:    {alpha*100:+.2f}%")
        
        results[symbol] = {
            'metrics': metrics,
            'equity_curve': df['equity'],
            'buy_hold_equity': buy_hold_equity,
        }
        
        viz_engine.plot_equity_curve(
            df['equity'], buy_hold_equity, symbol, metrics, 'stress_test_2022'
        )
    
    return results


def run_robustness_analysis(config: StrategyConfig) -> Dict:
    """
    Run comprehensive robustness analysis with WFO and Monte Carlo Bootstrap tests.
    """
    print("\n" + "="*70)
    print("ADVANCED ROBUSTNESS ANALYSIS (v6 - Bootstrap Monte Carlo)")
    print("="*70)

    analyzer = RobustnessAnalyzer(config)
    viz_engine = VisualizationEngine(config)

    all_results = {
        'wfo': {},
        'monte_carlo': {},
        'grid_search': {}
    }

    for symbol in config.SYMBOLS:
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS ANALYSIS: {symbol}")
        print(f"{'='*70}")

        # 1. Walk-Forward Optimization
        wfo_results, wfo_stats = analyzer.run_walk_forward_optimization(symbol)
        all_results['wfo'][symbol] = {
            'results': wfo_results,
            'stats': wfo_stats
        }

        # Plot WFO results
        viz_engine.plot_walk_forward_results(wfo_results, symbol)

        # 2. Monte Carlo Bootstrap Test (FIXED v6.0)
        print(f"\n[INFO] Running Monte Carlo Bootstrap Test...")
        mc_results = analyzer.run_monte_carlo_bootstrap(
            symbol,
            num_simulations=2000,
            use_block_bootstrap=True,
            block_size=10
        )
        all_results['monte_carlo'][symbol] = mc_results

        # 3. Legacy Grid Search (retained for comparison)
        print(f"\n[INFO] Running legacy grid search for comparison...")
        grid_results = analyzer.run_grid_search(symbol)

        if not grid_results.empty:
            print(f"\n{'='*40}")
            print(f"GRID SEARCH RESULTS: {symbol}")
            print(f"{'='*40}")

            best = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]
            print(f"  Best Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"  Best Entry:  RSI {best['rsi_entry']}")
            print(f"  Best Exit:   RSI {best['rsi_exit']}")
            print(f"  Return:      {best['total_return']:.0f}%")

            robust_count = (grid_results['sharpe_ratio'] > 0.5).sum()
            total_count = len(grid_results)
            print(f"  Robust Combos: {robust_count}/{total_count} ({robust_count/total_count*100:.0f}%)")

            all_results['grid_search'][symbol] = grid_results

            # Generate heatmaps
            viz_engine.plot_robustness_heatmaps(grid_results, symbol)

    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBUSTNESS SUMMARY")
    print("="*80)

    for symbol in config.SYMBOLS:
        print(f"\n{symbol}:")
        print("-" * 60)

        # WFO Summary
        wfo_stats = all_results['wfo'].get(symbol, {}).get('stats', {})
        if wfo_stats and 'error' not in wfo_stats:
            print(f"  Walk-Forward Optimization:")
            print(f"    Avg OOS Sharpe:     {wfo_stats.get('avg_oos_sharpe', 0):.2f}")
            print(f"    Stability Score:    {wfo_stats.get('stability_score', 0):.2f}")
            print(f"    % Profitable:       {wfo_stats.get('pct_profitable_windows', 0):.1f}%")

        # Monte Carlo Summary
        mc_stats = all_results['monte_carlo'].get(symbol, {})
        if mc_stats and 'error' not in mc_stats:
            print(f"  Monte Carlo Bootstrap:")
            print(f"    Actual Return:      {mc_stats.get('actual_final_return', 0):.2f}%")
            print(f"    Return Percentile:  {mc_stats.get('actual_return_percentile', 0):.1f}%")
            print(f"    Z-Score:            {mc_stats.get('z_score_return', 0):.2f}")
            print(f"    5th Percentile:     {mc_stats.get('final_return_p5', 0):.2f}%")
            print(f"    95th Percentile:    {mc_stats.get('final_return_p95', 0):.2f}%")

            # Verdict
            percentile = mc_stats.get('actual_return_percentile', 0)
            z_score = abs(mc_stats.get('z_score_return', 0))
            if percentile > 95 and z_score > 1.96:
                print(f"    Verdict:          ✓ STATISTICALLY SIGNIFICANT")
            elif percentile > 90 and z_score > 1.64:
                print(f"    Verdict:          ⚠ MARGINALLY SIGNIFICANT")
            else:
                print(f"    Verdict:          ✗ NOT STATISTICALLY SIGNIFICANT")

        # Grid Search Summary
        grid_results = all_results['grid_search'].get(symbol, pd.DataFrame())
        if not grid_results.empty:
            best = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]
            print(f"  Grid Search:")
            print(f"    Best Sharpe:        {best['sharpe_ratio']:.2f}")
            print(f"    Best Return:        {best['total_return']:.0f}%")

    print("\n" + "="*80)

    return all_results


def print_summary_table(production_results: Dict, stress_results: Dict) -> None:
    print("\n" + "="*80)
    print("PRODUCTION v5 (3x Leverage) - COMPREHENSIVE SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'BNBUSDT Main':<15} {'BNBUSDT 2022':<15} {'ETHUSDT Main':<15} {'ETHUSDT 2022':<15}")
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
        
        for symbol in ['BNBUSDT', 'ETHUSDT']:
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

from monte_carlo_bootstrap_v6 import MonteCarloConfig, MonteCarloAnalyzer

def run_monte_carlo(backtest_engine: BacktestEngine, symbol: str):
    if backtest_engine.trades.empty:
        print("[WARNING] No trades found to run Monte Carlo.")
        return
        
    mc_config = MonteCarloConfig(num_simulations=2000, initial_capital=1000.0)
    mc_analyzer = MonteCarloAnalyzer(mc_config)
    mc_analyzer.run_simulation(backtest_engine.trades, symbol)

def main():
    config = StrategyConfig()
    
    print("\n" + "="*80)
    print("RSI MOMENTUM STRATEGY - PRODUCTION v5 (3x LEVERAGE)")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Dynamic Position Sizing (6% risk per trade - Simulating 3x Leverage)")
    print("  ✓ Max Position Size Cap increased to 300%")
    print("  ✓ Symbol Swapped: BNBUSDT added, BTCUSDT removed")
    print("  ✓ Bear Market Stress Test (2022 crypto winter)")
    print("  ✓ Parameter Robustness Heatmaps")
    print("  ✓ 4.0x ATR Trailing Stop (wide)")
    print("  ✓ ADX(14) > 20 Filter (trending markets)")
    print("="*80)
    
    production_results = run_production_backtest(config)
    stress_results = run_stress_test(config)
    robustness_results = run_robustness_analysis(config)
    
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
