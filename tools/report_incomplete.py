#!/usr/bin/env python3
import os
from pathlib import Path

ROOT = Path('apra_mnist_runs_full')
AGGS = ['apra_basic','apra_weighted','trimmed','median']
TOTAL_EXPECTED = 25

if not ROOT.exists():
    print(f"Root folder {ROOT} not found")
    raise SystemExit(1)

total_found = 0
incomplete = []
print('Report of APRA runs under', ROOT)
for grid_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
    # grid_dir may contain aggregator subdirs or be nested grid/grid (some dirs had repeated names)
    # If grid_dir has subdirs matching AGGS, use them; otherwise if grid_dir contains files, assume it's the agg folder
    subdirs = [d for d in grid_dir.iterdir() if d.is_dir()]
    # heuristic: if any subdir is in AGGS, treat grid_dir as grid; else if grid_dir name in AGGS treat as agg container
    if any(d.name in AGGS for d in subdirs):
        grid_name = grid_dir.name
        for agg in sorted([d for d in subdirs if d.name in AGGS], key=lambda x: x.name):
            files = list((agg).glob('round_*.npz'))
            count = len(files)
            total_found += count
            missing = [i for i in range(1, TOTAL_EXPECTED+1) if (agg / f'round_{i:03d}.npz').exists() is False]
            status = 'OK' if count==TOTAL_EXPECTED else 'MISSING'
            print(f"{grid_name:30} | {agg.name:12} | {count:2d}/{TOTAL_EXPECTED} | {status} | Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if count != TOTAL_EXPECTED:
                incomplete.append((grid_name, agg.name, count, missing))
    else:
        # maybe nested double folder (e.g., sd64_ns2_zt3.0/sd64_ns2_zt3.0)
        nested = [d for d in subdirs if d.name == grid_dir.name]
        if nested:
            # examine nested path for aggs
            sub = nested[0]
            for aggname in AGGS:
                agg = sub / aggname
                if agg.exists():
                    files = list(agg.glob('round_*.npz'))
                    count = len(files)
                    total_found += count
                    missing = [i for i in range(1, TOTAL_EXPECTED+1) if (agg / f'round_{i:03d}.npz').exists() is False]
                    status = 'OK' if count==TOTAL_EXPECTED else 'MISSING'
                    print(f"{grid_dir.name:30} | {aggname:12} | {count:2d}/{TOTAL_EXPECTED} | {status} | Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
                    if count != TOTAL_EXPECTED:
                        incomplete.append((grid_dir.name, aggname, count, missing))
        else:
            # maybe agg folder directly
            if grid_dir.name in AGGS:
                agg = grid_dir
                files = list(agg.glob('round_*.npz'))
                count = len(files)
                total_found += count
                missing = [i for i in range(1, TOTAL_EXPECTED+1) if (agg / f'round_{i:03d}.npz').exists() is False]
                status = 'OK' if count==TOTAL_EXPECTED else 'MISSING'
                print(f"{grid_dir.name:30} | {grid_dir.name:12} | {count:2d}/{TOTAL_EXPECTED} | {status} | Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")
                if count != TOTAL_EXPECTED:
                    incomplete.append((grid_dir.name, grid_dir.name, count, missing))

print('\nTOTAL ROUND FILES FOUND:', total_found)
print('INCOMPLETE TASKS:', len(incomplete))
if incomplete:
    print('\nTop incomplete examples:')
    for i,(g,a,c,m) in enumerate(incomplete[:20],1):
        print(f" {i}. Grid: {g}, Agg: {a}, Count: {c}, Missing sample: {m[:6]}")

# Exit code 0
