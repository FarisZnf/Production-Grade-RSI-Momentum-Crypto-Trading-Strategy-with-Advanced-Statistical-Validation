"""
================================================================================
MONTE CARLO BOOTSTRAP SIMULATION - PRODUCTION GRADE v6.1 (FIXED)
================================================================================
Author: Senior Quantitative Developer
Description: Statistically valid Monte Carlo simulation using bootstrap resampling
             to test strategy robustness and measure sequence risk.

FIXED IN v6.1:
- Integration with Equity ROI (accounts for position sizing and leverage).
- Dynamic Block Sizing to prevent Zero Variance on low-frequency strategies.
- Comprehensive 4-panel Seaborn visualization (Returns, MDD, Sharpe, Curves).
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo bootstrap simulation."""
    num_simulations: int = 2000          # Jumlah simulasi bootstrap
    use_block_bootstrap: bool = True     # Mempertahankan korelasi trade berurutan
    block_size: int = 5                  # Ukuran default blok (akan disesuaikan otomatis)
    initial_capital: float = 1000.0      # Modal awal simulasi
    periods_per_year: int = 2190         # Faktor tahunan untuk timeframe 4H (365 * 6)
    plot_samples: int = 50               # Jumlah garis kurva ekuitas acak yang digambar
    save_plots: bool = True
    output_dir: str = './output'


# =============================================================================
# MONTE CARLO ANALYZER
# =============================================================================

class MonteCarloAnalyzer:
    def __init__(self, config: MonteCarloConfig):
        self.config = config

    def run_simulation(self, trade_df: pd.DataFrame, symbol: str) -> Dict:
        """
        Menjalankan simulasi Monte Carlo Bootstrap berdasarkan log transaksi.
        Menerima DataFrame dari BacktestEngine (harus memiliki 'roi_on_equity').
        """
        print(f"\n{'='*80}")
        print(f"MONTE CARLO BOOTSTRAP SIMULATION - {symbol}")
        print(f"{'='*80}")

        if trade_df is None or trade_df.empty:
            print("[ERROR] Trade DataFrame is empty. Cannot run Monte Carlo.")
            return {}

        # Memastikan kolom roi_on_equity tersedia
        if 'roi_on_equity' not in trade_df.columns:
            print("[WARNING] 'roi_on_equity' not found. Attempting to calculate...")
            if 'net_pnl' in trade_df.columns and 'entry_equity' in trade_df.columns:
                trade_df['roi_on_equity'] = trade_df['net_pnl'] / trade_df['entry_equity']
            else:
                raise ValueError("Missing 'roi_on_equity' and cannot calculate it. Please update BacktestEngine.")

        trade_rois = trade_df['roi_on_equity'].values
        n_trades = len(trade_rois)

        if n_trades < 10:
            print(f"[WARNING] Only {n_trades} trades found. Monte Carlo results may lack statistical significance.")

        # DYNAMIC BLOCK SIZING (Mencegah Zero Variance / Garis Lurus)
        actual_block_size = self.config.block_size
        use_block = self.config.use_block_bootstrap
        
        if use_block and n_trades < (actual_block_size * 4):
            actual_block_size = max(1, n_trades // 4)
            if actual_block_size < 2:
                use_block = False
                print("[INFO] Not enough trades for Block Bootstrap. Falling back to Pure Bootstrap.")
            else:
                print(f"[INFO] Adjusted block size to {actual_block_size} due to low trade count.")

        print(f"[INFO] Original Trades: {n_trades}")
        print(f"[INFO] Simulations:     {self.config.num_simulations:,}")
        print(f"[INFO] Method:          {'Block Bootstrap' if use_block else 'Pure Bootstrap'} (Block Size: {actual_block_size})")

        # Inisialisasi array hasil
        final_returns = np.zeros(self.config.num_simulations)
        max_drawdowns = np.zeros(self.config.num_simulations)
        sharpe_ratios = np.zeros(self.config.num_simulations)
        sample_curves = []

        np.random.seed(42)

        for i in range(self.config.num_simulations):
            # Resampling (Mengacak urutan trade)
            if use_block:
                n_blocks = int(np.ceil(n_trades / actual_block_size))
                blocks = [trade_rois[j:j+actual_block_size] for j in range(0, n_trades, actual_block_size)]
                sampled_blocks = [blocks[np.random.randint(0, len(blocks))] for _ in range(n_blocks)]
                sampled_rois = np.concatenate(sampled_blocks)[:n_trades]
            else:
                sampled_rois = np.random.choice(trade_rois, size=n_trades, replace=True)

            # Rekonstruksi kurva ekuitas
            equity_curve = self.config.initial_capital * np.cumprod(1 + sampled_rois)
            equity_curve = np.insert(equity_curve, 0, self.config.initial_capital) # Tambahkan modal awal di indeks 0

            # Simpan sampel kurva untuk di-plot
            if i < self.config.plot_samples:
                sample_curves.append(equity_curve)

            # Kalkulasi Metrik Simulasi
            final_returns[i] = (equity_curve[-1] / self.config.initial_capital - 1) * 100

            running_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - running_max) / running_max
            max_drawdowns[i] = np.min(dd) * 100

            # Estimasi Sharpe Ratio per iterasi
            trades_per_year = n_trades / (2.25) if n_trades > 0 else 1  # Asumsi 2.25 tahun data (2024 - Q1 2026)
            if len(sampled_rois) > 2 and np.std(sampled_rois) > 0:
                sharpe = (np.mean(sampled_rois) / np.std(sampled_rois)) * np.sqrt(max(1, trades_per_year))
            else:
                sharpe = 0.0
            sharpe_ratios[i] = sharpe

        # Kalkulasi Metrik Strategi Aktual (Tanpa diacak)
        actual_equity = self.config.initial_capital * np.cumprod(1 + trade_rois)
        actual_equity = np.insert(actual_equity, 0, self.config.initial_capital)
        actual_return = (actual_equity[-1] / self.config.initial_capital - 1) * 100
        act_running_max = np.maximum.accumulate(actual_equity)
        actual_dd = np.min((actual_equity - act_running_max) / act_running_max) * 100
        actual_trades_per_year = n_trades / 2.25 if n_trades > 0 else 1
        actual_sharpe = (np.mean(trade_rois) / np.std(trade_rois)) * np.sqrt(max(1, actual_trades_per_year)) if np.std(trade_rois) > 0 else 0

        results = {
            'symbol': symbol,
            'final_returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'sharpe_ratios': sharpe_ratios,
            'sample_curves': sample_curves,
            'actual_equity': actual_equity,
            'actual_return': actual_return,
            'actual_dd': actual_dd,
            'actual_sharpe': actual_sharpe,
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'p5_return': np.percentile(final_returns, 5),
            'p95_return': np.percentile(final_returns, 95),
            'prob_profit': (final_returns > 0).sum() / self.config.num_simulations * 100
        }

        self._plot_results(results)
        self._print_summary(results)
        
        return results

    def _plot_results(self, res: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"{res['symbol']} - Monte Carlo Bootstrap Analysis\n"
                     f"Simulations: {self.config.num_simulations:,} | Method: {'Block' if self.config.use_block_bootstrap else 'Pure'} Bootstrap", 
                     fontsize=16, fontweight='bold')

        # 1. Final Returns Distribution
        ax = axes[0, 0]
        sns.histplot(res['final_returns'], bins=50, kde=True, ax=ax, color='#4A90E2', edgecolor='black')
        ax.axvline(res['actual_return'], color='purple', linestyle='-', linewidth=3, label=f"Actual: {res['actual_return']:.2f}%")
        ax.axvline(res['median_return'], color='green', linestyle='--', linewidth=2, label=f"Median: {res['median_return']:.2f}%")
        ax.axvline(res['p5_return'], color='red', linestyle=':', linewidth=2, label=f"5th %ile: {res['p5_return']:.2f}%")
        ax.set_title('Bootstrap Distribution of Final Returns')
        ax.set_xlabel('Final Return (%)')
        ax.set_ylabel('Density')
        ax.legend()

        # 2. Max Drawdown Distribution
        ax = axes[0, 1]
        sns.histplot(res['max_drawdowns'], bins=50, kde=True, ax=ax, color='#F5A623', edgecolor='black')
        ax.axvline(res['actual_dd'], color='purple', linestyle='-', linewidth=3, label=f"Actual: {res['actual_dd']:.2f}%")
        ax.axvline(np.median(res['max_drawdowns']), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(res['max_drawdowns']):.2f}%")
        ax.axvline(np.percentile(res['max_drawdowns'], 5), color='red', linestyle=':', linewidth=2, label=f"5th %ile: {np.percentile(res['max_drawdowns'], 5):.2f}%")
        ax.set_title('Bootstrap Distribution of Max Drawdown')
        ax.set_xlabel('Max Drawdown (%)')
        ax.set_ylabel('Density')
        ax.legend()

        # 3. Sharpe Ratio Distribution
        ax = axes[1, 0]
        sns.histplot(res['sharpe_ratios'], bins=50, kde=True, ax=ax, color='#50E3C2', edgecolor='black')
        ax.axvline(res['actual_sharpe'], color='purple', linestyle='-', linewidth=3, label=f"Actual: {res['actual_sharpe']:.2f}")
        ax.axvline(np.median(res['sharpe_ratios']), color='green', linestyle='--', linewidth=2, label=f"Median: {np.median(res['sharpe_ratios']):.2f}")
        ax.set_title('Bootstrap Distribution of Sharpe Ratio')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Density')
        ax.legend()

        # 4. Sample Equity Curves
        ax = axes[1, 1]
        for curve in res['sample_curves']:
            norm_curve = (curve / self.config.initial_capital) * 100
            ax.plot(norm_curve, color='gray', alpha=0.15, linewidth=1)
        
        actual_norm = (res['actual_equity'] / self.config.initial_capital) * 100
        ax.plot(actual_norm, color='purple', linewidth=3, label='Actual Strategy')
        ax.set_title(f"Sample Bootstrap Equity Curves (n={len(res['sample_curves'])})")
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Normalized Equity (%)')
        ax.legend()

        plt.tight_layout()
        
        if self.config.save_plots:
            save_path = Path(self.config.output_dir) / f"{res['symbol']}_monte_carlo.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Plot saved to {save_path}")
        
        plt.show()

    def _print_summary(self, res: Dict):
        print(f"\n{'='*40}")
        print(f"MONTE CARLO RESULTS: {res['symbol']}")
        print(f"{'='*40}")
        print(f"  Actual Return:       {res['actual_return']:.2f}%")
        print(f"  MC Median Return:    {res['median_return']:.2f}%")
        print(f"  MC Mean Return:      {res['mean_return']:.2f}%")
        print(f"  5th Percentile:      {res['p5_return']:.2f}% (Worst 5%)")
        print(f"  95th Percentile:     {res['p95_return']:.2f}% (Best 5%)")
        print(f"  Prob > 0% Return:    {res['prob_profit']:.1f}%")
        print(f"\n  Actual Max DD:       {res['actual_dd']:.2f}%")
        print(f"  MC Median Max DD:    {np.median(res['max_drawdowns']):.2f}%")
        print(f"  5th Percentile DD:   {np.percentile(res['max_drawdowns'], 5):.2f}% (Worst 5%)")
        print(f"\n  Actual Sharpe:       {res['actual_sharpe']:.2f}")
        print(f"  MC Median Sharpe:    {np.median(res['sharpe_ratios']):.2f}")
        print(f"{'='*40}\n")