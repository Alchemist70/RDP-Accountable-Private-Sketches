#!/usr/bin/env python
"""One-shot checker: prints initial status of run.log and round_*.npz files.
"""
from pathlib import Path
import os
from datetime import datetime

OUTDIR = Path('apra_mnist_runs_full')
AGGS = ['apra_weighted', 'apra_basic', 'trimmed', 'median']

def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def check_once():
    if not OUTDIR.exists():
        print(f"{ts()} OUTDIR not found: {OUTDIR}")
        return
    grids = sorted([p for p in OUTDIR.iterdir() if p.is_dir()])
    if not grids:
        print(f"{ts()} No grid directories found in {OUTDIR}")
        return
    for grid in grids:
        print(f"--- Grid: {grid.name} ---")
        for agg in AGGS:
            agg_dir = grid / agg
            log_path = agg_dir / 'run.log'
            log_exists = log_path.exists()
            log_size = log_path.stat().st_size if log_exists else 0
            rounds = 0
            if agg_dir.exists():
                rounds = len([p for p in agg_dir.iterdir() if p.is_file() and p.name.startswith('round_') and p.name.endswith('.npz')])
            print(f"  {agg}: log_exists={log_exists} size={log_size} bytes, rounds={rounds}")
            if log_exists and log_size>0:
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                        tail = ''.join(f.readlines()[-10:])
                    # show if there are errors
                    err_lines = [l for l in tail.splitlines() if 'Traceback' in l or 'Exception' in l]
                    if err_lines:
                        print('    --> Error markers found in log:')
                        for el in err_lines[:5]:
                            print('      ', el)
                except Exception as e:
                    print('    --> Could not read log:', e)

if __name__ == '__main__':
    print(f"{ts()} One-shot monitor check starting")
    check_once()
    print(f"{ts()} One-shot monitor check finished")
