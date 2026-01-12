#!/usr/bin/env python3
"""
APS+ Global optimizer: find a scalar multiplier on a heuristic base allocation
such that the total composed epsilon (across all clients/events) does not
exceed a target global epsilon under RDP accounting (via dp_accounting).

Usage:
  python scripts/aps_plus_global_optimizer.py --clients clients.json --q 0.01 --steps 100 --target-eps 5.0 --target-delta 1e-6 --base-sigma 1.0 --out paper_figures/aps_plus_global.json

The script assumes `dp_accounting` and `numpy` are available (run in `tfpriv`).
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import math
import sys

try:
    import numpy as np
    import dp_accounting.dp_event as dp_event
    from dp_accounting.rdp import RdpAccountant
except Exception as e:
    print("ERROR: This script requires dp_accounting and numpy. Run in tfpriv environment.", file=sys.stderr)
    raise


def heuristic_base_sigmas(clients, base_sigma=1.0, min_sigma=0.1, max_sigma=10.0):
    scores = [max(0.0, float(c.get("importance", 1.0))) for c in clients]
    if sum(scores) <= 0:
        scores = [1.0 for _ in scores]
    total = sum(scores)
    rel = [1.0 / math.sqrt(s / total) for s in scores]
    median_rel = sorted(rel)[len(rel)//2]
    scale = base_sigma / median_rel if median_rel > 0 else 1.0
    sigmas = []
    for r in rel:
        s = float(max(min_sigma, min(max_sigma, r * scale)))
        sigmas.append(s)
    return sigmas


def total_epsilon_for_all(q, sigmas, steps, target_delta, orders=None):
    if orders is None:
        orders = np.concatenate([np.arange(2, 64, 0.5), np.arange(64, 512, 1.0)])
    acct = RdpAccountant(orders=orders)
    for sigma in sigmas:
        g = dp_event.GaussianDpEvent(noise_multiplier=float(sigma))
        if q < 1.0:
            ev = dp_event.PoissonSampledDpEvent(sampling_probability=float(q), event=g)
        else:
            ev = g
        acct.compose(ev, count=int(steps))
    eps, opt_order = acct.get_epsilon_and_optimal_order(float(target_delta))
    return float(eps)


def find_multiplier_global(clients, base_sigmas, q, steps, target_eps, target_delta, tol=1e-3, max_iters=40):
    lo = 0.01
    hi = 100.0
    best = None
    orders = np.concatenate([np.arange(2, 64, 0.5), np.arange(64, 512, 1.0)])
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        sigmas = [s * mid for s in base_sigmas]
        total_eps = total_epsilon_for_all(q, sigmas, steps, target_delta, orders=orders)
        if total_eps <= target_eps:
            best = (mid, sigmas, total_eps)
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    if best is None:
        return mid, sigmas, total_eps
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=Path, required=True, help="JSON file listing clients and importance scores")
    p.add_argument("--q", type=float, required=True, help="Sampling probability q (per-client)")
    p.add_argument("--steps", type=int, default=100, help="Number of rounds/steps per client")
    p.add_argument("--target-eps", type=float, required=True, help="Target global epsilon for the system")
    p.add_argument("--target-delta", type=float, default=1e-6, help="Target delta")
    p.add_argument("--base-sigma", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("paper_figures/aps_plus_global.json"))
    args = p.parse_args(argv)

    clients = json.loads(args.clients.read_text(encoding='utf-8'))
    base_sigmas = heuristic_base_sigmas(clients, base_sigma=args.base_sigma)
    m, sigmas, total_eps = find_multiplier_global(clients, base_sigmas, args.q, args.steps, args.target_eps, args.target_delta)

    out = {"multiplier": float(m), "total_epsilon": float(total_eps), "alloc": []}
    for c, s in zip(clients, sigmas):
        out['alloc'].append({"id": c.get('id'), "importance": c.get('importance'), "sigma": float(s)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote global allocation to {args.out}")

if __name__ == "__main__":
    main()
