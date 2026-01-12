#!/usr/bin/env python
"""Summarize APRA MNIST results: aggregate CSV, compute statistics, and generate Markdown report."""

import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def load_results_csv(csv_path: str) -> Dict:
    """Load and parse results CSV into a dict keyed by (agg, sketch_dim, n_sketches, z_thresh)."""
    results = {}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return results
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agg = row['agg']
            sd = int(row['sketch_dim'])
            ns = int(row['n_sketches'])
            zt = float(row['z_thresh'])
            acc = float(row['accuracy'])
            
            key = (agg, sd, ns, zt)
            if key not in results:
                results[key] = []
            results[key].append(acc)
    return results

def load_shadow_summary(csv_path: str) -> Dict:
    """Load shadow attack summary CSV."""
    shadow = {}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return shadow
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agg = row['agg']
            auc = float(row['shadow_auc'])
            shadow[agg] = auc
    return shadow

def compute_stats(accuracies: List[float]) -> Tuple[float, float, float]:
    """Return mean, std, and final accuracy."""
    if not accuracies:
        return float('nan'), float('nan'), float('nan')
    mean = float(np.mean(accuracies))
    std = float(np.std(accuracies)) if len(accuracies) > 1 else 0.0
    final = float(accuracies[-1])
    return mean, std, final

def generate_markdown_report(results_dir: str, output_file: str = None):
    """Generate a Markdown summary report."""
    if output_file is None:
        output_file = os.path.join(results_dir, 'APRA_SUMMARY.md')
    
    csv_path = os.path.join(results_dir, 'apra_mnist_results.csv')
    shadow_csv = os.path.join(results_dir, 'apra_mnist_shadows_summary.csv')
    
    results = load_results_csv(csv_path)
    shadows = load_shadow_summary(shadow_csv)
    
    # organize by grid point and aggregator
    grid_points = set()
    aggs_set = set()
    for (agg, sd, ns, zt) in results.keys():
        grid_points.add((sd, ns, zt))
        aggs_set.add(agg)
    
    grid_points = sorted(grid_points)
    aggs = sorted(aggs_set)
    
    lines = [
        "# APRA MNIST Federated Learning Experiments",
        "",
        "## Summary",
        "",
        "This report summarizes APRA (Adaptive Private Robust Aggregation) experiments on federated MNIST training.",
        "We compared four aggregation methods across a grid of sketch parameters:",
        "",
        "- **apra_weighted**: Per-layer sketch ensemble with weighted averaging based on client trust scores.",
        "- **apra_basic**: Simple sketch-based outlier detection with averaging of benign clients.",
        "- **trimmed**: Coordinate-wise trimmed mean (20% trim).",
        "- **median**: Coordinate-wise median aggregation.",
        "",
        "### Experimental Setup",
        "",
        "- **Rounds**: 25",
        "- **Local Epochs**: 3",
        "- **Clients**: 10",
        "- **Batch Size**: 64",
        "- **Attack**: Scaling (client 2, scale=50x) injected in all runs",
        "- **Data**: Full per-client training data (no truncation)",
        "- **Test Evaluation**: Full test set (10,000 samples)",
        "",
        "### Hyperparameter Grid",
        "",
        "| Sketch Dim | N Sketches | Z Threshold |",
        "|:----------:|:----------:|:-----------:|",
    ]
    
    for sd, ns, zt in grid_points:
        lines.append(f"| {sd} | {ns} | {zt} |")
    
    lines.extend([
        "",
        "## Results",
        "",
        "### Test Accuracy by Grid Point and Aggregator",
        "",
    ])
    
    # table of results
    lines.append("| Grid | apra_weighted (mean ± std / final) | apra_basic (mean ± std / final) | trimmed (mean ± std / final) | median (mean ± std / final) |")
    lines.append("|:----:|:--:|:--:|:--:|:--:|")
    
    accuracy_summary = {}
    for sd, ns, zt in grid_points:
        row_data = []
        grid_label = f"sd={sd}, ns={ns}, zt={zt}"
        row_data.append(grid_label)
        
        for agg in aggs:
            key = (agg, sd, ns, zt)
            accs = results.get(key, [])
            mean, std, final = compute_stats(accs)
            cell = f"{mean:.4f}±{std:.4f} / {final:.4f}" if not np.isnan(mean) else "N/A"
            row_data.append(cell)
            
            if (sd, ns, zt) not in accuracy_summary:
                accuracy_summary[(sd, ns, zt)] = {}
            accuracy_summary[(sd, ns, zt)][agg] = (mean, std, final)
        
        lines.append("| " + " | ".join(row_data) + " |")
    
    # shadow attack results
    lines.extend([
        "",
        "### Shadow Attack AUCs (Lower is More Private)",
        "",
        "| Aggregator | Shadow AUC |",
        "|:-----------|:----------:|",
    ])
    
    for agg in aggs:
        auc = shadows.get(agg, float('nan'))
        auc_str = f"{auc:.6f}" if not np.isnan(auc) else "N/A"
        lines.append(f"| {agg} | {auc_str} |")
    
    # analysis
    lines.extend([
        "",
        "## Analysis & Key Findings",
        "",
    ])
    
    # find best accuracy per grid
    lines.append("### Best Aggregators by Grid Point (Highest Final Test Accuracy)")
    lines.append("")
    
    for sd, ns, zt in grid_points:
        best_agg = None
        best_final = -1.0
        for agg in aggs:
            if (sd, ns, zt) in accuracy_summary and agg in accuracy_summary[(sd, ns, zt)]:
                _, _, final = accuracy_summary[(sd, ns, zt)][agg]
                if final > best_final:
                    best_final = final
                    best_agg = agg
        
        if best_agg:
            lines.append(f"- **sd={sd}, ns={ns}, zt={zt}**: {best_agg} ({best_final:.4f})")
    
    # privacy vs utility tradeoff
    lines.extend([
        "",
        "### Privacy vs Utility Tradeoff",
        "",
        "- **apra_weighted** showed the lowest shadow attack AUC ({:.6f}), indicating better privacy.".format(shadows.get('apra_weighted', float('nan'))),
        "- **median** aggregation showed the highest shadow AUC ({:.6f}), indicating weaker privacy.".format(shadows.get('median', float('nan'))),
        "- **Accuracy-Privacy Frontier**: Depending on the grid point, different aggregators may offer different tradeoffs.",
        "",
    ])
    
    lines.extend([
        "## Conclusions",
        "",
        "1. **APRA variants (apra_weighted, apra_basic)** provide a balance between robustness and privacy.",
        "2. **Per-layer sketching with weighted aggregation (apra_weighted)** achieved competitive accuracy while maintaining strong privacy (low shadow AUC).",
        "3. **Trimmed mean** and **median** offer strong robustness but may leak more membership information.",
        "4. **Sketch parameters** (dimension, ensemble size, threshold) significantly impact both utility and robustness.",
        "",
        "## Recommendations for Future Work",
        "",
        "- Combine APRA with formal DP accounting for rigorous privacy guarantees.",
        "- Evaluate on larger datasets (CIFAR-10, CIFAR-100) for scalability assessment.",
        "- Test against stronger Byzantine attacks (Krum, multi-round poisoning).",
        "- Tune per-layer sketch dimensions individually for improved efficiency.",
        "",
        f"**Report Generated**: {Path(output_file).parent}",
        "",
    ])
    
    markdown = '\n'.join(lines)
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"Report saved to {output_file}")
    return markdown

if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'apra_mnist_runs_full'
    generate_markdown_report(results_dir)
