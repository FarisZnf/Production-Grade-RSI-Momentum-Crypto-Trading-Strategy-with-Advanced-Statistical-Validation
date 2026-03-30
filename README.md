# Momentum-Based RSI Crypto Trading Strategy Backtest

A production-grade Python backtesting framework for testing Wilder's RSI momentum strategies on cryptocurrency perpetual futures.

## Strategy Overview

This implements a **momentum-based** (not mean-reversion) RSI strategy:

| Signal | Condition | Action |
|--------|-----------|--------|
| **Long Entry** | RSI(14) crosses **ABOVE 70** | Buy (momentum breakout) |
| **Long Exit** | RSI(14) drops **BELOW 30** | Sell (momentum exhaustion) |

### Key Features

- ✅ **Wilder's Original RSI Smoothing** - Uses Wilder's modified EMA (1/period smoothing), not standard SMA-based RSI
- ✅ **Look-Ahead Bias Prevention** - All signals shifted by 1 period; trades execute on next bar open
- ✅ **Realistic Cost Modeling** - Trading fees (0.1%), slippage (0.05%), funding rates (0.01%/period)
- ✅ **Local Data Caching** - CSV caching for fast subsequent runs
- ✅ **Parameter Sensitivity Analysis** - Heatmap visualization to test strategy robustness
- ✅ **Professional Metrics** - Sharpe Ratio, CAGR, Max Drawdown via `ffn` library

## Installation

```bash
pip install -r requirements.txt
```

### Optional: Binance API Keys

For unlimited API access, set environment variables:

```bash
# Windows (PowerShell)
$env:BINANCE_API_KEY="your_api_key"
$env:BINANCE_API_SECRET="your_api_secret"

# Linux/Mac
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

Without API keys, the script will use public access (rate-limited).

## Usage

### Run Full Backtest

```bash
python rsi_momentum_backtest.py
```

This will:
1. Fetch historical 4h candle data for BTCUSDT and ETHUSDT (Aug 2017 - Feb 2026)
2. Cache data locally in `data_cache/` folder
3. Calculate Wilder's RSI(14)
4. Generate trading signals with look-ahead bias prevention
5. Run backtest with realistic cost modeling
6. Generate performance metrics and visualizations
7. Run parameter sensitivity analysis (RSI thresholds 70-78 entry, 20-30 exit)

### Output Files

| Location | Contents |
|----------|----------|
| `data_cache/` | Cached OHLCV data (CSV) |
| `output/` | Equity curves, heatmaps, trade distributions (PNG) |
| `output/parameter_optimization.csv` | Full optimization results |

## Configuration

Edit `StrategyConfig` class in the script:

```python
class StrategyConfig:
    # Data settings
    SYMBOLS = ['BTCUSDT', 'ETHUSDT']
    TIMEFRAME = '4h'
    START_DATE = '2017-08-01'
    END_DATE = '2026-02-01'
    
    # RSI parameters
    RSI_PERIOD = 14
    RSI_LONG_ENTRY = 70
    RSI_LONG_EXIT = 30
    
    # Cost modeling
    TRADING_FEE = 0.001      # 0.1%
    SLIPPAGE = 0.0005        # 0.05%
    FUNDING_RATE = 0.0001    # 0.01% per 4h
    
    # Capital
    INITIAL_CAPITAL = 1000.0
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      StrategyConfig                          │
│                  (Centralized Settings)                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│  DataEngine   │    │ IndicatorEngine │    │ SignalGenerator│
│  - Binance API│    │ - Wilder's RSI  │    │ - 1-bar shift │
│  - CSV Cache  │    │ - ATR (Wilder)  │    │ - Look-ahead  │
└───────────────┘    └─────────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ BacktestEngine  │
                    │ - Cost Modeling │
                    │ - P&L Tracking  │
                    │ - ffn Metrics   │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│ VizEngine     │    │ ParameterOptimizer│   │ Buy&Hold Calc │
│ - Equity Plot │    │ - Grid Search   │    │ - Baseline    │
│ - Heatmaps    │    │ - Robustness    │    │ - Comparison  │
└───────────────┘    └─────────────────┘    └───────────────┘
```

## Wilder's RSI Implementation

The script implements J. Welles Wilder's **original** smoothing method:

```python
# Standard EMA smoothing factor:
alpha_ema = 2 / (period + 1)  # e.g., 0.133 for period=14

# Wilder's smoothing factor:
alpha_wilder = 1 / period     # e.g., 0.071 for period=14

# Wilder's formula:
WilderMA[i] = (WilderMA[i-1] * (period - 1) + Value[i]) / period
```

This produces **smoother, less reactive** signals compared to standard RSI implementations.

## Cost Model Details

| Cost Type | Value | Application |
|-----------|-------|-------------|
| Trading Fee | 0.1% | Applied on entry AND exit |
| Slippage | 0.05% | Market order impact on execution |
| Funding Rate | 0.01% / 4h | Perpetual futures holding cost |

**Example**: On a $10,000 position held for 5 days (30 x 4h bars):
- Entry fee: $10
- Exit fee: $10
- Slippage: $5 (entry) + $5 (exit)
- Funding: $10,000 × 0.0001 × 15 = $15
- **Total costs: ~$45**

## Performance Metrics

The backtest calculates:

- **Total Return (%)**: Cumulative return over the test period
- **CAGR (%)**: Annualized compound growth rate
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown (%)**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Disclaimer

This backtest is for **educational and research purposes only**. Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Always conduct your own research before trading.

## License

MIT License - Feel free to modify and distribute.
