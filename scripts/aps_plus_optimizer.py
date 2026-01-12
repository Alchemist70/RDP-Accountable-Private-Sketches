#!/usr/bin/env python3
"""
APS+ optimizer prototype (heuristic).

Given a list of clients with importance scores (higher means more valuable to
protect / allocate budget to), produce a suggested per-client Gaussian noise
stddev (`sigma`) allocation under a simple heuristic. This is a prototype and
intended for experimentation only; it does not perform formal RDP composition
or guarantee a global privacy budget. Use this to generate candidate
allocations which can be evaluated with `privacy_accounting.py`.

Usage example:
  python scripts/aps_plus_optimizer.py --clients clients.json --out alloc.json

Where `clients.json` is a list of {"id":..., "importance":...}.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import math


def allocate_sigmas(clients, base_sigma=1.0, min_sigma=0.1, max_sigma=10.0):
    """Heuristic allocation: assign sigma inversely proportional to sqrt(importance).

    More important clients get lower noise (smaller sigma) to improve detection,
    but clipped to [min_sigma, max_sigma]. The base_sigma scales the overall
    magnitude.
    """
    # Normalize importances
    scores = [max(0.0, float(c.get("importance", 1.0))) for c in clients]
    if sum(scores) <= 0:
        scores = [1.0 for _ in scores]
    total = sum(scores)
    # desired relative allocation ~ 1/sqrt(importance) -> smaller sigma for high importance
    rel = [1.0 / math.sqrt(s / total) for s in scores]
    # scale so that median sigma ~ base_sigma
    median_rel = sorted(rel)[len(rel)//2]
    scale = base_sigma / median_rel if median_rel > 0 else 1.0
    sigmas = []
    for r in rel:
        s = float(max(min_sigma, min(max_sigma, r * scale)))
        sigmas.append(s)
    return sigmas


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=Path, required=True, help="JSON file with list of clients and importance scores")
    p.add_argument("--out", type=Path, default=Path("paper_figures/aps_plus_alloc.json"))
    p.add_argument("--base-sigma", type=float, default=1.0)
    p.add_argument("--min-sigma", type=float, default=0.1)
    p.add_argument("--max-sigma", type=float, default=10.0)
    args = p.parse_args(argv)

    clients = json.loads(args.clients.read_text(encoding="utf-8"))
    sigmas = allocate_sigmas(clients, base_sigma=args.base_sigma, min_sigma=args.min_sigma, max_sigma=args.max_sigma)
    alloc = []
    for c, s in zip(clients, sigmas):
        alloc.append({"id": c.get("id"), "importance": c.get("importance"), "sigma": s})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"alloc": alloc}, indent=2), encoding="utf-8")
    print(f"Wrote allocation to {args.out}")


if __name__ == "__main__":
    main()
