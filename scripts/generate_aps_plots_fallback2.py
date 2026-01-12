#!/usr/bin/env python3
"""
Fallback APS+ plotting: plots per-client sigma allocations and approximate per-client epsilons
when the `dp_accounting` package is not available. Uses a simple RDP approximation for Gaussian
mechanism without sampling amplification.

Usage:
  python scripts/generate_aps_plots_fallback2.py --alloc paper_figures/aps_plus_convex_tight.json --steps 100 --delta 1e-6 --outdir paper_figures

Note: This is an approximation used only when the project's dp_accounting package is unavailable.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def approx_epsilon_from_sigma(sigma, steps, delta, orders=None):
    # RDP for Gaussian: R_alpha(alpha) = alpha / (2 sigma^2) per mechanism
    # Compose over `steps` rounds: R_total(alpha) = steps * alpha / (2 sigma^2)
    # Convert to (epsilon, delta): epsilon = min_alpha R_total(alpha) - log(delta)/(alpha-1)
    if orders is None:
        orders = np.concatenate([np.arange(2.0, 64.0, 0.5), np.arange(64, 201, 1.0)])
    best = float('inf')
    for a in orders:
        rdp = steps * (a / (2.0 * (sigma ** 2)))
        eps = rdp - (math.log(1.0 / delta) / (a - 1.0))
        if eps < best:
            best = eps
    return float(best)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--alloc', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--delta', type=float, default=1e-6)
    p.add_argument('--outdir', type=Path, default=Path('paper_figures'))
    args = p.parse_args(argv)

    alloc = json.loads(args.alloc.read_text(encoding='utf-8'))
    alloc_list = alloc.get('alloc', [])
    ids = [a['id'] for a in alloc_list]
    sigmas = [float(a['sigma']) for a in alloc_list]

    # compute approximate per-client epsilons (no sampling amplification)
    epsilons = [approx_epsilon_from_sigma(s, args.steps, args.delta) for s in sigmas]

    args.outdir.mkdir(parents=True, exist_ok=True)
    # plot sigmas
    plt.figure(figsize=(10,4))
    plt.bar(ids, sigmas, color='C0')
    plt.xticks(rotation=45)
    plt.ylabel('sigma')
    plt.title('Per-client sigma allocations (APS+)')
    plt.tight_layout()
    sig_path = args.outdir / 'aps_plus_sigmas_fallback.png'
    plt.savefig(sig_path, dpi=150)
    plt.close()
    print('Wrote', sig_path)

    # plot epsilons
    plt.figure(figsize=(10,4))
    plt.bar(ids, epsilons, color='C1')
    plt.xticks(rotation=45)
    plt.ylabel('epsilon (approx)')
    plt.title(f'Per-client epsilon (approx, delta={args.delta})')
    plt.tight_layout()
    eps_path = args.outdir / 'aps_plus_epsilons_fallback.png'
    plt.savefig(eps_path, dpi=150)
    plt.close()
    print('Wrote', eps_path)

if __name__ == '__main__':
    main()
