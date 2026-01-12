#!/usr/bin/env python3
"""
Regenerate the max_sensitivity_vs_nsk figure with all three curves (dims=32, 64, 128)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reproduce the empirical sensitivity function from the notebook
def empirical_sensitivity(D=10000, dims=64, n_sketches=4, delta=1.0, trials=200, seed=0):
    """Simple CountSketch simulation for sensitivity estimation."""
    rng = np.random.default_rng(seed)
    vals = []
    for t in range(trials):
        # Simulate CountSketch: hash and sign functions
        h = rng.integers(0, dims, D)  # random bucket assignment
        s = rng.choice([-1, 1], D)    # random signs
        
        # Single-coordinate update
        idx = rng.integers(0, D)
        delta_val = delta
        
        # Compute sketch for both original and perturbed
        s_orig = np.zeros((n_sketches, dims))
        s_pert = np.zeros((n_sketches, dims))
        
        for j in range(n_sketches):
            # Different hash/sign per sketch repetition
            h_j = rng.integers(0, dims, D)
            s_j = rng.choice([-1, 1], D)
            
            s_orig[j, h_j[idx]] += s_j[idx] * 0
            s_pert[j, h_j[idx]] += s_j[idx] * delta_val
        
        # L2 distance between sketches
        diff = np.linalg.norm(s_pert.ravel() - s_orig.ravel())
        vals.append(diff)
    
    return np.array(vals)

# Parameter sweep: dims and n_sketches
dims_list = [32, 64, 128]
n_list = [1, 2, 4, 8]
results = []

print("Generating sensitivity data...")
for dims in dims_list:
    for nsk in n_list:
        vals = empirical_sensitivity(D=10000, dims=dims, n_sketches=nsk, delta=1.0, trials=300, seed=123)
        max_val = float(vals.max())
        results.append({'dims': dims, 'n_sketches': nsk, 'max': max_val, 'median': float(np.median(vals))})
        print(f"  dims={dims}, n_sketches={nsk}: max={max_val:.3f}")

rd = pd.DataFrame(results)

# Generate the plot with all three curves
plt.figure(figsize=(7.2, 4.8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
for i, dims in enumerate(dims_list):
    subset = rd[rd['dims'] == dims].sort_values('n_sketches')
    plt.plot(subset['n_sketches'], subset['max'], marker='o', linewidth=2, markersize=8, 
             label=f'dims={dims}', color=colors[i])

plt.xlabel('n_sketches', fontsize=12)
plt.ylabel('empirical max L2', fontsize=12)
plt.title('Empirical max sensitivity vs n_sketches', fontsize=13)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save in both SVG and PDF
import os
os.makedirs('paper_figures', exist_ok=True)

svg_path = os.path.join('paper_figures', 'max_sensitivity_vs_nsk_fixed.svg')
pdf_path = os.path.join('paper_figures', 'max_sensitivity_vs_nsk_fixed.pdf')

plt.savefig(svg_path, format='svg', dpi=150, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved to: {svg_path}")
print(f"✅ Saved to: {pdf_path}")

plt.show()
