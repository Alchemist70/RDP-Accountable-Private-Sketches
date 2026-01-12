#!/usr/bin/env python
"""Comprehensive analysis and visualization of APRA results."""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def load_results_csv(csv_path: str):
    """Load results CSV."""
    grid_data = defaultdict(lambda: defaultdict(list))
    agg_data = defaultdict(list)
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return grid_data, agg_data
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agg = row['agg']
            sd = int(row['sketch_dim'])
            ns = int(row['n_sketches'])
            zt = float(row['z_thresh'])
            round_num = int(row['round'])
            acc = float(row['accuracy'])
            
            key = (sd, ns, zt)
            grid_data[key][agg].append((round_num, acc))
            agg_data[agg].append(acc)
    
    return grid_data, agg_data

def plot_convergence_by_grid(grid_data, output_dir):
    """Plot convergence curves for each grid point."""
    os.makedirs(output_dir, exist_ok=True)
    
    grid_points = sorted(grid_data.keys())
    n_grids = len(grid_points)
    cols = 2
    rows = (n_grids + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = {'apra_weighted': '#1f77b4', 'apra_basic': '#ff7f0e', 'trimmed': '#2ca02c', 'median': '#d62728'}
    
    for idx, (sd, ns, zt) in enumerate(grid_points):
        ax = axes[idx]
        for agg in sorted(grid_data[(sd, ns, zt)].keys()):
            rounds_accs = sorted(grid_data[(sd, ns, zt)][agg])
            rounds, accs = zip(*rounds_accs)
            color = colors.get(agg, 'black')
            ax.plot(rounds, accs, marker='o', label=agg, color=color, linewidth=2)
        
        ax.set_title(f'Grid: sd={sd}, ns={ns}, zt={zt}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # hide extra subplots
    for idx in range(len(grid_points), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_by_grid.png'), dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'convergence_by_grid.png')}")
    plt.close()

def plot_final_accuracy_heatmap(grid_data, output_dir):
    """Plot heatmap of final accuracies by grid and aggregator."""
    os.makedirs(output_dir, exist_ok=True)
    
    grid_points = sorted(grid_data.keys())
    aggs = sorted(set(agg for gp in grid_data.values() for agg in gp.keys()))
    
    # matrix: rows=grid points, cols=aggs
    matrix = np.zeros((len(grid_points), len(aggs)))
    for i, (sd, ns, zt) in enumerate(grid_points):
        for j, agg in enumerate(aggs):
            if agg in grid_data[(sd, ns, zt)] and grid_data[(sd, ns, zt)][agg]:
                rounds_accs = grid_data[(sd, ns, zt)][agg]
                final_acc = rounds_accs[-1][1]  # last round accuracy
                matrix[i, j] = final_acc
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # labels
    grid_labels = [f"sd={sd}\nns={ns}\nzt={zt}" for sd, ns, zt in grid_points]
    ax.set_xticks(np.arange(len(aggs)))
    ax.set_yticks(np.arange(len(grid_points)))
    ax.set_xticklabels(aggs, fontsize=10)
    ax.set_yticklabels(grid_labels, fontsize=9)
    
    # add values to cells
    for i in range(len(grid_points)):
        for j in range(len(aggs)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}', ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Final Test Accuracy by Grid Point and Aggregator', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_accuracy_heatmap.png'), dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'final_accuracy_heatmap.png')}")
    plt.close()

def plot_agg_comparison(agg_data, output_dir):
    """Box plot comparing aggregators across all runs."""
    os.makedirs(output_dir, exist_ok=True)
    
    aggs = sorted(agg_data.keys())
    data_to_plot = [agg_data[agg] for agg in aggs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_to_plot, labels=aggs, patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors[:len(aggs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Distribution of Test Accuracy Across All Runs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregator_comparison.png'), dpi=100, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'aggregator_comparison.png')}")
    plt.close()

def generate_analysis_report(csv_path, shadow_csv_path, output_dir):
    """Generate comprehensive analysis report."""
    grid_data, agg_data = load_results_csv(csv_path)
    
    # load shadow data if available
    shadow_data = {}
    if os.path.exists(shadow_csv_path):
        with open(shadow_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                agg = row['agg']
                auc = float(row['shadow_auc'])
                if agg not in shadow_data:
                    shadow_data[agg] = []
                shadow_data[agg].append(auc)
    
    # compute stats
    stats = {}
    for agg in sorted(agg_data.keys()):
        accs = agg_data[agg]
        stats[agg] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'min': float(np.min(accs)),
            'max': float(np.max(accs)),
            'final_rounds_mean': float(np.mean(accs[-5:])) if len(accs) >= 5 else float(np.mean(accs)),
        }
    
    # save stats to JSON
    stats_path = os.path.join(output_dir, 'analysis_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")
    
    # generate plots
    plot_convergence_by_grid(grid_data, output_dir)
    plot_final_accuracy_heatmap(grid_data, output_dir)
    plot_agg_comparison(agg_data, output_dir)
    
    # print summary
    print("\n=== ACCURACY SUMMARY ===")
    for agg in sorted(stats.keys()):
        s = stats[agg]
        print(f"{agg:15s} - mean: {s['mean']:.4f} ± {s['std']:.4f}, "
              f"range: [{s['min']:.4f}, {s['max']:.4f}], final-5: {s['final_rounds_mean']:.4f}")
    
    if shadow_data:
        print("\n=== SHADOW ATTACK SUMMARY ===")
        for agg in sorted(shadow_data.keys()):
            aucs = shadow_data[agg]
            mean_auc = np.mean(aucs)
            print(f"{agg:15s} - mean AUC: {mean_auc:.4f} (lower = more private)")

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'apra_mnist_runs_full'
    csv_path = os.path.join(results_dir, 'apra_mnist_results.csv')
    shadow_csv_path = os.path.join(results_dir, 'shadow_aucs_all_grids.csv')
    
    generate_analysis_report(csv_path, shadow_csv_path, results_dir)
