"""Plot FARPA sweep results CSV and save PNG summarizing malicious trust behavior.

Reads `farpa_sweep_results.csv` by default and produces `scripts/figs/farpa_sweep_trust.png`.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CSV = os.environ.get('FARPA_SWEEP_CSV', os.path.join(ROOT, 'farpa_sweep_results.csv'))
OUT_DIR = os.path.join(ROOT, 'scripts', 'figs')
OUT_PNG = os.path.join(OUT_DIR, 'farpa_sweep_trust.png')

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV):
    print('Sweep CSV not found at', CSV)
    sys.exit(1)

df = pd.read_csv(CSV)
# Convert types
for c in ['sketch_dim_per_layer', 'eps_sketch', 'z_thresh']:
    if c in df.columns:
        df[c] = df[c].astype(float)

# We'll plot trust_malicious0 as heatmaps across sketch_dim vs eps for a fixed z_thresh slices
z_values = sorted(df['z_thresh'].unique())
fig, axes = plt.subplots(1, len(z_values), figsize=(5 * max(1, len(z_values)), 4), squeeze=False)

for i, z in enumerate(z_values):
    sub = df[df['z_thresh'] == z]
    pivot = sub.pivot(index='sketch_dim_per_layer', columns='eps_sketch', values='trust_malicious0')
    ax = axes[0, i]
    im = ax.imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'trust_malicious0 (z={z})')
    ax.set_xlabel('eps_sketch')
    ax.set_ylabel('sketch_dim_per_layer')
    # xticks/yticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(r)) for r in pivot.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
# also save vector formats
try:
    svg_path = os.path.splitext(OUT_PNG)[0] + '.svg'
    pdf_path = os.path.splitext(OUT_PNG)[0] + '.pdf'
    plt.savefig(svg_path)
    plt.savefig(pdf_path)
    print('Saved plot to', OUT_PNG, svg_path, pdf_path)
except Exception:
    print('Saved plot to', OUT_PNG)
