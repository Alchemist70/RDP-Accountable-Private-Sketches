#!/usr/bin/env python3
"""
APS+ constrained optimizer: scale per-client sigma allocations so that each
client's composed privacy (via dp_accounting) does not exceed a target epsilon.

Requires: `dp_accounting` (run in `tfpriv`).

Usage:
  python scripts/aps_plus_constrained.py --clients clients.json --q 0.01 --steps 100 --target-eps 2.0 --target-delta 1e-6 --out paper_figures/aps_plus_constrained.json

Clients JSON format: list of {"id": "c0", "importance": 1.0}
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


def allocate_sigmas(clients, base_sigma=1.0, min_sigma=0.1, max_sigma=10.0):
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


def client_epsilon(q, sigma, steps, orders=None, target_delta=1e-6):
    if orders is None:
        orders = np.concatenate([np.arange(2, 64, 0.5), np.arange(64, 512, 1.0)])
    acct = RdpAccountant(orders=orders)
    g = dp_event.GaussianDpEvent(noise_multiplier=float(sigma))
    if q < 1.0:
        ev = dp_event.PoissonSampledDpEvent(sampling_probability=float(q), event=g)
    else:
        ev = g
    acct.compose(ev, count=int(steps))
    eps, opt_order = acct.get_epsilon_and_optimal_order(float(target_delta))
    return float(eps)


def find_multiplier(clients, base_sigmas, q, steps, target_eps, target_delta, tol=1e-3, max_iters=40):
    # Binary search multiplier m so that max_i eps_i(sigma_i * m) <= target_eps
    lo = 0.01
    hi = 100.0
    best = None
    orders = np.concatenate([np.arange(2, 64, 0.5), np.arange(64, 512, 1.0)])
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        sigmas = [s * mid for s in base_sigmas]
        epsilons = [client_epsilon(q, s, steps, orders=orders, target_delta=target_delta) for s in sigmas]
        worst = max(epsilons)
        if worst <= target_eps:
            best = (mid, sigmas, epsilons)
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    if best is None:
        # return final attempted allocation
        return mid, sigmas, epsilons
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=Path, required=True, help="JSON file listing clients and importance scores")
    p.add_argument("--q", type=float, required=True, help="Sampling probability q (per-client)")
    p.add_argument("--steps", type=int, default=100, help="Number of rounds/steps per client")
    p.add_argument("--target-eps", type=float, required=True, help="Target per-client epsilon")
    p.add_argument("--target-delta", type=float, default=1e-6, help="Target delta")
    p.add_argument("--base-sigma", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("paper_figures/aps_plus_constrained.json"))
    args = p.parse_args(argv)

    clients = json.loads(args.clients.read_text(encoding='utf-8'))
    base_sigmas = allocate_sigmas(clients, base_sigma=args.base_sigma)
    m, sigmas, epsilons = find_multiplier(clients, base_sigmas, args.q, args.steps, args.target_eps, args.target_delta)

    out = {"multiplier": float(m), "alloc": []}
    for c, s, e in zip(clients, sigmas, epsilons):
        out['alloc'].append({"id": c.get('id'), "importance": c.get('importance'), "sigma": float(s), "epsilon": float(e)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote allocation to {args.out}")

if __name__ == "__main__":
    main()
