#!/usr/bin/env python3
"""
Plot per-client sigma allocations and per-client epsilons using dp_accounting.

Usage:
  python scripts/plot_allocation_and_eps.py --alloc paper_figures/aps_plus_convex_tight.json --q 0.01 --steps 100 --delta 1e-6 --outdir paper_figures
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import dp_accounting.dp_event as dp_event
    from dp_accounting.rdp import RdpAccountant
except Exception:
    raise


def client_epsilon(q, sigma, steps, orders=None, target_delta=1e-6):
    if orders is None:
        orders = np.concatenate([np.arange(2, 64, 1.0), np.arange(64, 201, 5.0)])
    acct = RdpAccountant(orders=orders)
    g = dp_event.GaussianDpEvent(noise_multiplier=float(sigma))
    if q < 1.0:
        ev = dp_event.PoissonSampledDpEvent(sampling_probability=float(q), event=g)
    else:
        ev = g
    acct.compose(ev, count=int(steps))
    eps, opt_order = acct.get_epsilon_and_optimal_order(float(target_delta))
    return float(eps)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--alloc', type=Path, required=True)
    p.add_argument('--q', type=float, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--delta', type=float, default=1e-6)
    p.add_argument('--outdir', type=Path, default=Path('paper_figures'))
    args = p.parse_args(argv)

    alloc = json.loads(args.alloc.read_text(encoding='utf-8'))
    alloc_list = alloc.get('alloc', [])
    ids = [a['id'] for a in alloc_list]
    sigmas = [float(a['sigma']) for a in alloc_list]

    # compute per-client epsilons
    epsilons = [client_epsilon(args.q, s, args.steps, target_delta=args.delta) for s in sigmas]

    args.outdir.mkdir(parents=True, exist_ok=True)
    # plot sigmas
    plt.figure(figsize=(10,4))
    plt.bar(ids, sigmas, color='C0')
    plt.xticks(rotation=45)
    plt.ylabel('sigma')
    plt.title('Per-client sigma allocations')
    plt.tight_layout()
    sig_path = args.outdir / 'aps_plus_sigmas.png'
    plt.savefig(sig_path, dpi=150)
    plt.close()
    print('Wrote', sig_path)

    # plot epsilons
    plt.figure(figsize=(10,4))
    plt.bar(ids, epsilons, color='C1')
    plt.xticks(rotation=45)
    plt.ylabel('epsilon')
    plt.title('Per-client epsilon (at delta={})'.format(args.delta))
    plt.tight_layout()
    eps_path = args.outdir / 'aps_plus_epsilons.png'
    plt.savefig(eps_path, dpi=150)
    plt.close()
    print('Wrote', eps_path)

if __name__ == '__main__':
    main()
