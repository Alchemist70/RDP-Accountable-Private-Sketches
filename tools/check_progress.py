#!/usr/bin/env python3
from pathlib import Path
import time
import math

ROOT = Path('apra_mnist_runs_full')
AGGS = ['apra_basic','apra_weighted','trimmed','median']
TOTAL_EXPECTED = 25
now = time.time()

print('Progress check: newest round file timestamps and counts')
print('-'*80)
for grid_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
    # find agg subdirs
    subdirs = [d for d in grid_dir.iterdir() if d.is_dir()]
    if any(d.name in AGGS for d in subdirs):
        grid_name = grid_dir.name
        for agg in sorted([d for d in subdirs if d.name in AGGS], key=lambda x: x.name):
            files = list(agg.glob('round_*.npz'))
            count = len(files)
            if files:
                newest = max(files, key=lambda p: p.stat().st_mtime)
                age = now - newest.stat().st_mtime
                age_s = int(age)
                print(f"{grid_name:25} | {agg.name:12} | {count:2d}/{TOTAL_EXPECTED} | newest: {newest.name} | age(s): {age_s}")
            else:
                print(f"{grid_name:25} | {agg.name:12} | {count:2d}/{TOTAL_EXPECTED} | newest: None")
    else:
        # check nested
        nested = [d for d in subdirs if d.name == grid_dir.name]
        if nested:
            sub = nested[0]
            for aggname in AGGS:
                agg = sub / aggname
                if agg.exists():
                    files = list(agg.glob('round_*.npz'))
                    count = len(files)
                    if files:
                        newest = max(files, key=lambda p: p.stat().st_mtime)
                        age_s = int(now - newest.stat().st_mtime)
                        print(f"{grid_dir.name:25} | {aggname:12} | {count:2d}/{TOTAL_EXPECTED} | newest: {newest.name} | age(s): {age_s}")
                    else:
                        print(f"{grid_dir.name:25} | {aggname:12} | {count:2d}/{TOTAL_EXPECTED} | newest: None")

print('-'*80)
# summary newest overall
all_files = list(ROOT.rglob('round_*.npz'))
print('Total round files:', len(all_files))
if all_files:
    newest_all = max(all_files, key=lambda p: p.stat().st_mtime)
    print('Newest checkpoint overall:', newest_all, 'age(s):', int(now - newest_all.stat().st_mtime))

