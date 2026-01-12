#!/usr/bin/env python3
"""
Plot per-run epsilon values from a merged RDP JSON produced by merge_rdp_aggregates.py

Usage:
  python scripts/plot_rdp_multi.py --input paper_figures/rdp_smokegrid_demo_multi.json --outdir paper_figures
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--outdir', type=Path, default=Path('paper_figures'))
args = p.parse_args()

j = json.loads(args.input.read_text(encoding='utf-8'))
entries = j.get('entries', [])
eps = [e.get('epsilon') for e in entries if e.get('epsilon') is not None]
labels = [f"run{i}" for i in range(len(eps))]

args.outdir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8,4))
plt.bar(labels, eps, color='C2')
plt.xticks(rotation=45)
plt.ylabel('epsilon')
plt.title('Per-run epsilon values (merged smoke-grid)')
plt.tight_layout()
outp = args.outdir / 'rdp_smokegrid_eps_multi.png'
plt.savefig(outp, dpi=150)
print('Wrote', outp)
