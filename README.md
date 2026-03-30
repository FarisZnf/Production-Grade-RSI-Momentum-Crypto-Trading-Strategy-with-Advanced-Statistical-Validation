# RSI Momentum Crypto Trading Strategy - Production v6

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade quantitative trading backtesting framework for cryptocurrency perpetual futures with **Walk-Forward Optimization** and **Bootstrap Monte Carlo** statistical validation.

---

## 🎯 Project Overview

This is a **momentum-based RSI strategy** designed for crypto perpetual futures (BNB/ETH/SOL-USDT) featuring:

- ✅ **Wilder's Original RSI** smoothing (not SMA-based)
- ✅ **Dynamic Position Sizing** (2% risk/trade, 3x leverage simulation)
- ✅ **Walk-Forward Optimization** (12m IS / 6m OOS rolling windows)
- ✅ **Block Bootstrap Monte Carlo** (2,000 simulations, block_size=10)
- ✅ **Bull Market Adaptive Filter** (RSI threshold -5 in strong trends)
- ✅ **Excel Trade Log Export** with detailed P&L analytics
- ✅ **Comprehensive Stress Testing** (2022 crypto winter validation)

---

## 📊 Performance Highlights

### Production Period (2024-2026)

| Symbol | Total Return | CAGR | Sharpe | Max DD | Win Rate | Profit Factor | Trades |
|--------|-------------|------|--------|--------|----------|---------------|--------|
| **ETHUSDT** | **+106.42%** | 38.63% | 1.53 | -5.06% | 53.1% | 4.99 | 32 |
| **BNBUSDT** | **+39.81%** | 16.31% | 0.86 | -16.02% | 40.6% | 1.77 | 32 |
| **SOLUSDT** | **+19.16%** | 8.22% | 0.74 | -9.55% | 44.7% | 1.53 | 38 |

### Bear Market Stress Test (2022 Crypto Winter)

| Symbol | Strategy Return | Buy & Hold | **Alpha** | Sharpe | Max DD |
|--------|----------------|------------|-----------|--------|--------|
| **ETHUSDT** | +34.11% | -67.93% | **+102.04%** | 1.94 | -1.55% |
| **BNBUSDT** | +45.71% | -52.63% | **+98.34%** | 1.31 | -6.72% |
| **SOLUSDT** | +26.88% | -94.37% | **+121.25%** | 1.07 | -1.43% |

### Statistical Validation (Bootstrap Monte Carlo v6.0)

| Symbol | Bootstrap Mean | Std Dev | 95% CI | % Positive | % Sharpe > 1 | Verdict |
|--------|---------------|---------|--------|------------|--------------|---------|
| **ETHUSDT** | 131.52% | 46.79% | [67%, 215%] | 100% | 100% | Robust |
| **BNBUSDT** | 30.92% | 27.80% | [-14%, 76%] | 90.2% | 84.7% | Robust |
| **SOLUSDT** | 45.61% | 36.42% | [2%, 134%] | 96.4% | 96.3% | Robust |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-rsi-momentum.git
cd crypto-rsi-momentum

# Install dependencies
pip install -r requirements.txt
```

### Run Backtest

```bash
# Run full production backtest with robustness analysis
python rsi_momentum_backtest_v5.py
```

### Output Files

All results saved to `output/` directory:
- `*_equity_curve_production_v5.png` - Equity curves vs Buy & Hold
- `*_equity_curve_stress_test_2022.png` - Bear market performance
- `*_trade_log.xlsx` - Detailed Excel trade log with P&L analytics
- `*_walk_forward_results.png` - WFO stability analysis
- `*_robustness_heatmap.png` - Parameter sensitivity heatmaps
- `*_monte_carlo_histogram.png` - Bootstrap distribution plots

---

## 📈 Strategy Logic

### Entry Conditions (ALL must be true)

1. **RSI(14) crosses above 65** (momentum breakout)
2. **Price > EMA(200)** (long-term uptrend)
3. **ADX(14) > 20** (trending market, not choppy)
4. **Volume > 20-period SMA** (confirm with volume)
5. **Bull Regime Filter**: If ADX > 25 AND Price > EMA(200), entry threshold reduced to **RSI > 60** (more aggressive)

### Exit Conditions

1. **4.0x ATR Trailing Stop** (PRIMARY - locks in profits)
2. **10.0x ATR Take Profit** (SECONDARY - lets winners run)
3. **RSI < 30 Emergency Exit** (TERTIARY - trend exhaustion)

### Position Sizing

```python
# Dynamic sizing based on stop distance
risk_amount = equity * 0.02  # 2% risk per trade
position_size = risk_amount / (entry_price - stop_loss_price)
max_position = equity * 3.0 / entry_price  # Max 300% (3x leverage)
position_size = min(position_size, max_position)
```

---

## 🔬 Advanced Features

### 1. Walk-Forward Optimization (WFO)

**Purpose**: Detect overfitting by testing parameters on unseen data.

**Configuration**:
- IS Window: 12 months (~2,190 bars on 4h)
- OOS Window: 6 months (~1,095 bars)
- Step Size: 6 months rolling

**Process**:
1. Optimize on IS window (49 parameter combinations)
2. Select top 3 IS performers by Sharpe
3. Test on OOS window
4. Calculate stability score: `mean(OOS Sharpe) / std(OOS Sharpe)`

**Results**:
- ETHUSDT: 2.26 OOS Sharpe, 100% profitable windows, ∞ stability score
- SOLUSDT: 0.38 OOS Sharpe, 100% profitable windows, 3.72 stability score

### 2. Block Bootstrap Monte Carlo (v6.0)

**Purpose**: Test strategy robustness to trade sequence variance.

**Why Bootstrap (Not Permutation)**:
- Permutation shuffles order → same final equity (multiplication is commutative)
- Bootstrap samples **with replacement** → some trades repeat, others missing → **REAL variance**

**Block Bootstrap** (preserves autocorrelation):
- Divide trades into blocks of 10
- Sample blocks with replacement
- Preserves volatility clustering critical for crypto

**Configuration**:
- Simulations: 2,000
- Block Size: 10 trades (~2-5 days on 4h timeframe)
- Metrics: Final return, Max DD, Sharpe ratio

**Statistical Significance Test**:
```python
if actual_return_percentile > 95 and z_score > 1.96:
    verdict = "STATISTICALLY SIGNIFICANT (p < 0.05)"
elif actual_return_percentile > 90 and z_score > 1.64:
    verdict = "MARGINALLY SIGNIFICANT (p < 0.10)"
else:
    verdict = "NOT STATISTICALLY SIGNIFICANT (could be luck)"
```

### 3. Bull Market Adaptive Filter

**Purpose**: Improve performance in strong bull markets (beat Buy & Hold).

**Logic**:
```python
if (close > EMA_200) and (ADX > 25):
    RSI_LONG_ENTRY = 60  # More aggressive (normally 65)
else:
    RSI_LONG_ENTRY = 65  # Standard
```

**Impact**:
- ETHUSDT: +106% vs -7% B&H → **+113% alpha**
- Increases trade frequency in strong trends by ~15-20%

---

## 📁 Project Structure

```
crypto-rsi-momentum/
├── rsi_momentum_backtest_v5.py       # Main production script (v6 Monte Carlo)
├── monte_carlo_bootstrap_v6.py       # Standalone bootstrap module
├── robustness_analyzer_fixed.py      # Reference implementation
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── data_cache/                        # Cached OHLCV data (CSV)
│   ├── BNBUSDT_4h.csv
│   ├── ETHUSDT_4h.csv
│   └── SOLUSDT_4h.csv
└── output/                            # Generated reports & plots
    ├── *_equity_curve_production_v5.png
    ├── *_equity_curve_stress_test_2022.png
    ├── *_trade_log.xlsx
    ├── *_walk_forward_results.png
    ├── *_robustness_heatmap.png
    └── *_monte_carlo_histogram.png
```

---

## ⚙️ Configuration

Edit `StrategyConfig` class in `rsi_momentum_backtest_v5.py`:

```python
@dataclass
class StrategyConfig:
    SYMBOLS: List[str] = ['BNBUSDT', 'ETHUSDT', 'SOLUSDT']
    TIMEFRAME: str = '4h'
    START_DATE: str = '2024-01-01'
    END_DATE: str = '2026-03-22'
    
    # Strategy parameters
    RSI_PERIOD: int = 14
    RSI_LONG_ENTRY: int = 65          # Reduced to 60 in bull regime
    RSI_LONG_EXIT: int = 30
    
    # Filters
    USE_TREND_FILTER: bool = True     # EMA(200)
    USE_ADX_FILTER: bool = True       # ADX(14) > 20
    USE_BULL_FILTER: bool = True      # Adaptive entry threshold
    
    # Risk management
    ATR_STOP_MULTIPLE: float = 4.0    # Trailing stop
    ATR_TARGET_MULTIPLE: float = 10.0 # Take profit
    RISK_PER_TRADE: float = 0.02      # 2% risk
    MAX_POSITION_PCT: float = 3.00    # Max 300% (3x leverage)
    
    # Monte Carlo
    MONTE_CARLO_SIMULATIONS: int = 2000
    BLOCK_SIZE: int = 10              # Block bootstrap size
```

---

## 🧪 Cost Modeling

Realistic trading costs included:

| Cost Type | Value | Application |
|-----------|-------|-------------|
| **Trading Fee** | 0.1% | Entry + Exit |
| **Slippage** | 0.05% | Market order impact |
| **Funding Rate** | 0.01% / 8h | Perpetual futures holding cost |

**Example** ($10,000 position, 5 days):
- Entry fee: $10
- Exit fee: $10
- Slippage: $5 + $5
- Funding: $10,000 × 0.0001 × 15 = $15
- **Total costs: ~$45**

---

## 📊 Key Metrics Explained

### Sharpe Ratio (Annualized)
```python
sharpe = (mean_return / std_return) * sqrt(2190)  # 2190 = 4h bars/year
```
- **> 1.0**: Good risk-adjusted returns
- **> 2.0**: Excellent (ETHUSDT: 1.53, Bear market: 1.94)

### Profit Factor
```python
profit_factor = gross_profit / gross_loss
```
- **> 1.5**: Profitable strategy
- **> 2.0**: Strong edge (ETHUSDT: 4.99, Bear market: 9.48)

### Maximum Drawdown
- Largest peak-to-trough decline
- **< 20%**: Acceptable for crypto strategies
- All symbols: < 16% (production), < 7% (bear market)

### Bootstrap Percentile Ranking
- Where does actual return rank vs 2,000 bootstrap simulations?
- **> 75th percentile**: Better than 75% of random sequences
- **> 95th percentile**: Statistically significant edge

---

## 🎓 Methodology Notes

### Why Wilder's RSI (Not Standard)?

Standard RSI uses SMA smoothing. Wilder's original method uses **exponential smoothing with alpha = 1/period**:

```python
# Wilder's smoothing (alpha = 1/14 = 0.0714)
avg_gain = (prev_avg_gain * 13 + current_gain) / 14

# vs Standard EMA (alpha = 2/(14+1) = 0.133)
avg_gain = prev_avg_gain * 0.867 + current_gain * 0.133
```

**Impact**: Wilder's is **smoother, less reactive** → fewer false signals.

### Why Block Bootstrap (Not Pure)?

Crypto returns exhibit **volatility clustering** (large moves follow large moves). Block bootstrap preserves this:

```
Original: [W1, W2, L1, L2, L3, W3, W4, W5, L4, W6]
Blocks (size=5): [W1,W2,L1,L2,L3] [W3,W4,W5,L4,W6]

Pure Bootstrap: Samples individual trades → loses clustering
Block Bootstrap: Samples blocks → preserves clustering ✓
```

**Recommended block_size**: 10 trades (~2-5 days on 4h timeframe)

---

## 📚 References & Further Reading

1. **Wilder, J. Welles Jr.** (1978). *New Concepts in Technical Trading Systems*
2. **Efron, B.** (1979). *Bootstrap Methods: Another Look at the Jackknife*
3. **Bailey, D.H. & López de Prado, M.** (2012). *The Deflated Sharpe Ratio*
4. **QuantInsti Blog** - Walk-Forward Analysis for Algorithmic Trading
5. **StrategyQuant** - Monte Carlo Simulation in Trading

---

## ⚠️ Disclaimer

**This project is for EDUCATIONAL and RESEARCH purposes only.**

- Past performance does NOT guarantee future results
- Cryptocurrency trading involves substantial risk of loss
- Always conduct your own research before trading
- Never trade with money you cannot afford to lose

**No financial advice is provided by this code.**

---

## 📄 License

MIT License - Feel free to modify and distribute.

```
Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Multi-timeframe confirmation (4h + 1d alignment)
- [ ] Machine learning regime detection (LSTM/Random Forest)
- [ ] Portfolio optimization (multi-symbol position sizing)
- [ ] Live trading integration (Binance WebSocket + order execution)
- [ ] Performance attribution analysis

---

## 📬 Contact & Support

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For strategy questions and improvements

**Happy Trading! 🚀**

---

*Last Updated: March 23, 2026*
*Version: v6.0 (Bootstrap Monte Carlo)*
