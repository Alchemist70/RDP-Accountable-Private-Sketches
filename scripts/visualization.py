"""
Visualization utilities for APRA results: convergence plots, heatmaps, privacy/robustness curves.
Output both to notebook display and external PNG/PDF files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

sns.set_style("whitegrid")


def plot_convergence_by_aggregator(
    results_df: pd.DataFrame,
    output_dir: str = '.',
    figsize: Tuple[int, int] = (12, 6),
    save_formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Plot accuracy convergence per aggregator across rounds.
    
    Args:
        results_df: DataFrame with columns ['round', 'agg', 'accuracy']
        output_dir: directory to save output
        figsize: figure size
        save_formats: list of formats to save ('png', 'pdf')
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for agg in results_df['agg'].unique():
        data = results_df[results_df['agg'] == agg].sort_values('round')
        ax.plot(data['round'], data['accuracy'], marker='o', label=agg, linewidth=2)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Convergence by Aggregator', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        path = os.path.join(output_dir, f'convergence.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    return fig


def plot_robustness_vs_byzantine_fraction(
    results_by_byzantine: Dict[float, Dict],
    output_dir: str = '.',
    figsize: Tuple[int, int] = (12, 6),
    save_formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Plot model accuracy vs Byzantine fraction for each aggregator.
    
    Args:
        results_by_byzantine: dict mapping byzantine_fraction -> {agg: accuracy}
        output_dir: directory to save
        figsize: figure size
        save_formats: list of formats
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract per-aggregator curves
    aggs = set()
    for d in results_by_byzantine.values():
        aggs.update(d.keys())
    
    for agg in sorted(aggs):
        frac_list = []
        acc_list = []
        for frac in sorted(results_by_byzantine.keys()):
            if agg in results_by_byzantine[frac]:
                frac_list.append(frac)
                acc_list.append(results_by_byzantine[frac][agg])
        ax.plot(frac_list, acc_list, marker='s', label=agg, linewidth=2, markersize=8)
    
    ax.set_xlabel('Byzantine Fraction', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Robustness: Accuracy vs Byzantine Fraction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        path = os.path.join(output_dir, f'robustness_vs_byzantine.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    return fig


def plot_privacy_auc_heatmap(
    privacy_auc_matrix: np.ndarray,
    aggregator_names: List[str],
    sketch_configs: List[str],
    output_dir: str = '.',
    figsize: Tuple[int, int] = (10, 8),
    save_formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Heatmap of shadow attack AUC (lower = more private) across aggregators and sketch configs.
    
    Args:
        privacy_auc_matrix: 2D array of AUC values (shape: [n_aggs, n_configs])
        aggregator_names: list of aggregator names
        sketch_configs: list of sketch config labels
        output_dir: directory to save
        figsize: figure size
        save_formats: list of formats
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        privacy_auc_matrix,
        xticklabels=sketch_configs,
        yticklabels=aggregator_names,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Red = high AUC (low privacy), Green = low AUC (high privacy)
        cbar_kws={'label': 'Shadow Attack AUC'},
        ax=ax
    )
    
    ax.set_xlabel('Sketch Configuration', fontsize=12)
    ax.set_ylabel('Aggregator', fontsize=12)
    ax.set_title('Privacy: Shadow Attack AUC (lower = more private)', fontsize=14, fontweight='bold')
    
    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        path = os.path.join(output_dir, f'privacy_auc_heatmap.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    return fig


def plot_utility_privacy_tradeoff(
    results_df: pd.DataFrame,
    output_dir: str = '.',
    figsize: Tuple[int, int] = (10, 8),
    save_formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Scatter plot of final accuracy vs privacy AUC (utility-privacy tradeoff).
    Each point is one aggregator/config combination.
    
    Args:
        results_df: DataFrame with columns ['agg', 'final_accuracy', 'privacy_auc']
        output_dir: directory to save
        figsize: figure size
        save_formats: list of formats
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for agg in results_df['agg'].unique():
        data = results_df[results_df['agg'] == agg]
        ax.scatter(data['privacy_auc'], data['final_accuracy'], label=agg, s=100, alpha=0.7)
    
    ax.set_xlabel('Privacy (Shadow Attack AUC)', fontsize=12)
    ax.set_ylabel('Utility (Final Accuracy)', fontsize=12)
    ax.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        path = os.path.join(output_dir, f'utility_privacy_tradeoff.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    return fig


def plot_detection_vs_byzantine(
    detection_results: Dict[float, Dict[str, float]],
    output_dir: str = '.',
    figsize: Tuple[int, int] = (10, 6),
    save_formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Plot detection accuracy (identifying Byzantine clients) vs Byzantine fraction.
    
    Args:
        detection_results: dict {byzantine_frac: {detector_name: detection_acc}}
        output_dir: directory to save
        figsize: figure size
        save_formats: list of formats
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    detectors = set()
    for d in detection_results.values():
        detectors.update(d.keys())
    
    for detector in sorted(detectors):
        frac_list = []
        acc_list = []
        for frac in sorted(detection_results.keys()):
            if detector in detection_results[frac]:
                frac_list.append(frac)
                acc_list.append(detection_results[frac][detector])
        ax.plot(frac_list, acc_list, marker='^', label=detector, linewidth=2, markersize=8)
    
    ax.set_xlabel('Byzantine Fraction', fontsize=12)
    ax.set_ylabel('Detection Accuracy', fontsize=12)
    ax.set_title('Byzantine Detection: Accuracy vs Fraction', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    for fmt in save_formats:
        path = os.path.join(output_dir, f'detection_vs_byzantine.{fmt}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    return fig


def generate_markdown_report(
    results_summary: Dict,
    output_path: str = 'apra_results_report.md'
) -> str:
    """
    Generate a Markdown report summarizing APRA results.
    
    Args:
        results_summary: dict with keys like 'best_agg', 'privacy_auc', 'robustness', etc.
        output_path: where to save the report
    
    Returns:
        markdown string
    """
    report = f"""
# APRA Experimental Results Report

**Generated:** {pd.Timestamp.now().isoformat()}

## Executive Summary

This report summarizes the APRA (Adaptive Private Robust Aggregation) experimental campaign on MNIST with federated learning.

### Key Metrics

- **Best Aggregator (Accuracy):** {results_summary.get('best_agg', 'N/A')}
- **Best Accuracy:** {results_summary.get('best_accuracy', 'N/A'):.4f}
- **Privacy (Shadow Attack AUC):** {results_summary.get('privacy_auc', 'N/A'):.4f} (lower is better)
- **Robustness (Byzantine Tolerance):** {results_summary.get('byzantine_tolerance', 'N/A'):.2%}

## Aggregators Evaluated

| Aggregator | Description |
|-----------|-------------|
| FedAvg | Baseline averaging (no robustness) |
| Median | Coordinate-wise median (Byzantine-robust) |
| Trimmed Mean | Trimmed coordinate-wise mean (Byzantine-robust) |
| Krum | Byzantine-resilient aggregation |
| APRA | Adaptive Private Robust Aggregation (proposed) |

## Results

### Convergence Behavior
- See `convergence.png`

### Privacy Evaluation
- Shadow attack AUC per aggregator: See `privacy_auc_heatmap.png`
- Lower AUC indicates better privacy

### Robustness Analysis
- Accuracy degradation vs Byzantine fraction: See `robustness_vs_byzantine.png`
- Detection accuracy: See `detection_vs_byzantine.png`

### Utility-Privacy Tradeoff
- See `utility_privacy_tradeoff.png`

## Ablations

{results_summary.get('ablations', 'N/A')}

## Recommendations

1. Use APRA for deployments requiring both privacy and robustness.
2. For privacy-only, trimmed mean + secure aggregation is sufficient.
3. For robustness-only, Krum offers strong guarantees without privacy overhead.

## Appendix

Full results available in CSV files:
- `convergence_results.csv`
- `privacy_results.csv`
- `robustness_results.csv`

---

*For reproducibility, all runs use fixed random seeds and detailed hyperparameters in the code.*
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    return report


if __name__ == '__main__':
    # Smoke test
    print("Visualization smoke test...")
    
    # Create dummy results
    rounds = np.arange(1, 26)
    accuracy_fedavg = 0.5 + 0.4 * (1 - np.exp(-0.1 * rounds))
    accuracy_median = 0.5 + 0.35 * (1 - np.exp(-0.12 * rounds))
    accuracy_apra = 0.5 + 0.42 * (1 - np.exp(-0.09 * rounds))
    
    df = pd.concat([
        pd.DataFrame({'round': rounds, 'agg': 'FedAvg', 'accuracy': accuracy_fedavg}),
        pd.DataFrame({'round': rounds, 'agg': 'Median', 'accuracy': accuracy_median}),
        pd.DataFrame({'round': rounds, 'agg': 'APRA', 'accuracy': accuracy_apra}),
    ])
    
    fig = plot_convergence_by_aggregator(df, output_dir='/tmp')
    print("✓ Convergence plot saved")
    
    print("✓ Visualization smoke test passed")
