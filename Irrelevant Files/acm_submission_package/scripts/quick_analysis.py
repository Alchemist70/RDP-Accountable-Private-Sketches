#!/usr/bin/env python
"""Quick analysis: print summary statistics from results CSV."""

import csv
import sys
import numpy as np
from collections import defaultdict

def analyze_csv(csv_path):
    """Print summary stats from results CSV."""
    agg_data = defaultdict(list)
    grid_data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agg = row['agg']
            sd = int(row['sketch_dim'])
            ns = int(row['n_sketches'])
            zt = float(row['z_thresh'])
            acc = float(row['accuracy'])
            
            agg_data[agg].append(acc)
            grid_data[(sd, ns, zt)][agg].append(acc)
    
    print("\n=== AGGREGATOR SUMMARY (All Rounds) ===")
    for agg in sorted(agg_data.keys()):
        accs = agg_data[agg]
        mean = np.mean(accs)
        std = np.std(accs)
        final = accs[-1] if accs else float('nan')
        print(f"{agg:15s} - mean: {mean:.4f} ± {std:.4f}, final: {final:.4f}")
    
    print("\n=== GRID POINT SUMMARY (Final Accuracy) ===")
    for (sd, ns, zt) in sorted(grid_data.keys()):
        print(f"\nGrid (sd={sd}, ns={ns}, zt={zt}):")
        for agg in sorted(grid_data[(sd, ns, zt)].keys()):
            accs = grid_data[(sd, ns, zt)][agg]
            final = accs[-1] if accs else float('nan')
            print(f"  {agg:15s}: {final:.4f}")

if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'apra_mnist_runs_full/apra_mnist_results.csv'
    analyze_csv(csv_path)
