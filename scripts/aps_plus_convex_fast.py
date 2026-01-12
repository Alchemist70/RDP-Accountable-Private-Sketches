#!/usr/bin/env python3
"""
Faster convex APS+ optimizer variant.
- Uses a coarser set of Rényi orders by default (fewer evaluations).
- Warm-starts from `paper_figures/aps_plus_global.json` if available.
- Uses lower max iterations and relaxed tolerance to speed up.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys

try:
    import numpy as np
    from scipy.optimize import minimize
    import dp_accounting.dp_event as dp_event
    from dp_accounting.rdp import RdpAccountant
except Exception as e:
    print("ERROR: This script requires numpy, scipy, and dp_accounting. Run in tfpriv environment.", file=sys.stderr)
    raise


def total_epsilon_for_all(q, sigmas, steps, target_delta, orders=None):
    if orders is None:
        orders = np.concatenate([np.arange(2, 64, 1.0), np.arange(64, 201, 5.0)])
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


def objective(sigmas, importances):
    sigmas = np.asarray(sigmas)
    return float(np.sum(np.asarray(importances) * (sigmas ** 2)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=Path, required=True)
    p.add_argument("--q", type=float, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--target-eps", type=float, required=True)
    p.add_argument("--target-delta", type=float, default=1e-6)
    p.add_argument("--min-sigma", type=float, default=0.1)
    p.add_argument("--max-sigma", type=float, default=10.0)
    p.add_argument("--base-sigma", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("paper_figures/aps_plus_convex_fast.json"))
    p.add_argument("--warmstart", type=Path, default=Path("paper_figures/aps_plus_global.json"), help="Optional warm-start JSON from global optimizer")
    args = p.parse_args(argv)

    clients = json.loads(args.clients.read_text(encoding='utf-8'))
    importances = [max(0.0, float(c.get('importance', 1.0))) for c in clients]
    n = len(clients)
    total = sum(importances) if sum(importances) > 0 else n
    rel = [1.0 / ((imp / total) ** 0.5) for imp in importances]
    median_rel = sorted(rel)[n//2]
    scale = args.base_sigma / median_rel if median_rel > 0 else 1.0
    x0 = np.array([max(args.min_sigma, min(args.max_sigma, r * scale)) for r in rel], dtype=float)

    # warm-start from global optimizer if available
    if args.warmstart.exists():
        try:
            g = json.loads(args.warmstart.read_text(encoding='utf-8'))
            alloc = g.get('alloc', [])
            if alloc and len(alloc) == n:
                ws = np.array([float(a.get('sigma', x0[i])) for i, a in enumerate(alloc)], dtype=float)
                x0 = np.clip(ws, args.min_sigma, args.max_sigma)
                print('Warm-started from', args.warmstart)
        except Exception:
            pass

    def constr_fun(x):
        x_clipped = np.clip(x, args.min_sigma, args.max_sigma)
        eps = total_epsilon_for_all(args.q, x_clipped, args.steps, args.target_delta)
        return float(args.target_eps - eps)

    cons = ({'type': 'ineq', 'fun': constr_fun},)
    bounds = [(args.min_sigma, args.max_sigma) for _ in range(n)]

    # Reduced iteration settings for speed
    res = minimize(lambda s: objective(s, importances), x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 60, 'ftol':1e-3})

    final_sigmas = np.clip(res.x, args.min_sigma, args.max_sigma)
    final_eps = total_epsilon_for_all(args.q, final_sigmas, args.steps, args.target_delta)

    out = {"success": bool(res.success), "message": res.message, "final_total_epsilon": float(final_eps), "alloc": []}
    for c, s in zip(clients, final_sigmas):
        out['alloc'].append({"id": c.get('id'), "importance": c.get('importance'), "sigma": float(s)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote convex-fast allocation to {args.out} (total_eps={final_eps})')

if __name__ == '__main__':
    main()
