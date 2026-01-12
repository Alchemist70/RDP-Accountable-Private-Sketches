#!/usr/bin/env python3
from pathlib import Path
import argparse

ROOT = Path('apra_mnist_runs_full')


def scan(root=ROOT):
    rows = []
    if not root.exists():
        print(f"Root {root} does not exist")
        return rows
    for grid in sorted([p for p in root.iterdir() if p.is_dir()]):
        # look for aggregator subdirs (or possibly nested same-name folder)
        aggs = [p for p in grid.iterdir() if p.is_dir()]
        if not aggs:
            # maybe nested same-name folder structure
            aggs = [p for p in (grid / grid.name).iterdir() if p.is_dir()] if (grid / grid.name).exists() else []
        for agg in sorted(aggs):
            npz_files = sorted(agg.glob('round_*.npz'))
            count = len(npz_files)
            rounds = [int(f.stem.split('_')[1]) for f in npz_files if '_' in f.stem]
            missing = [r for r in range(1,26) if r not in rounds]
            last_mod = max((f.stat().st_mtime for f in npz_files), default=0)
            rows.append((grid.name, agg.name, count, missing, last_mod))
    return rows


if __name__ == '__main__':
    rows = scan()
    total = sum(r[2] for r in rows)
    print(f"Total round_*.npz files: {total}")
    incomplete = [(g,a,c,m) for g,a,c,m,lt in [(r[0],r[1],r[2],r[3],r[4]) for r in rows] if c < 25]
    print(f"Incomplete aggregator dirs: {len(incomplete)}")
    for g,a,c,m in incomplete:
        print(f"{g} / {a}: {c}/25  missing: {m}")
    if not incomplete:
        print('All complete')
