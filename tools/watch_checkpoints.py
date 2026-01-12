#!/usr/bin/env python3
"""Watch for new round_*.npz files under apra_mnist_runs_full and print them as they appear.

Usage: python tools/watch_checkpoints.py [--root apra_mnist_runs_full] [--interval 5] [--target 800]

Press Ctrl-C to stop.
"""
import time
from pathlib import Path
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='apra_mnist_runs_full')
parser.add_argument('--interval', type=float, default=5.0)
parser.add_argument('--target', type=int, default=800)
args = parser.parse_args()

root = Path(args.root)
if not root.exists():
    print(f"Root {root} not found")
    raise SystemExit(1)

seen = set()
# initialize seen with existing files
for p in root.rglob('round_*.npz'):
    seen.add(p.resolve())

print(f"Watching {root} for new round_*.npz (starting with {len(seen)} files). Target: {args.target}")
try:
    while True:
        current = set(root.rglob('round_*.npz'))
        new = sorted([p for p in current if p.resolve() not in seen], key=lambda p: p.stat().st_mtime)
        for p in new:
            seen.add(p.resolve())
            ts = datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{ts}] NEW: {p} ({len(seen)}/{args.target})")
        if len(seen) >= args.target:
            print(f"\nReached target {args.target} files. Exiting watcher.")
            break
        time.sleep(args.interval)
except KeyboardInterrupt:
    print('\nWatcher interrupted by user. Exiting.')
