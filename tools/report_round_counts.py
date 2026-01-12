import os
import re
from collections import defaultdict

ROOT = os.path.join(os.getcwd(), 'apra_mnist_runs_full')
pattern_strict = re.compile(r'^round_\d{3}\.npz$')
pattern_suspect = re.compile(r'^round_\d{3}.*')

per_agg = defaultdict(int)
suspect_files = []
all_strict = []

if not os.path.isdir(ROOT):
    print(f"Directory not found: {ROOT}")
    raise SystemExit(1)

for dirpath, dirs, files in os.walk(ROOT):
    for fn in files:
        if fn.startswith('round_'):
            rel_dir = os.path.relpath(dirpath, ROOT)
            parts = rel_dir.split(os.sep)
            if len(parts) >= 2:
                grid = parts[0]
                agg = parts[1]
            else:
                grid = parts[0] if parts else '<root>'
                agg = '<unknown>'

            if pattern_strict.match(fn):
                per_agg[(grid, agg)] += 1
                all_strict.append(os.path.join(dirpath, fn))
            elif pattern_suspect.match(fn):
                suspect_files.append(os.path.join(dirpath, fn))

# Print summary
print(f"Total strict 'round_###.npz' files: {len(all_strict)}")

keys = sorted(set([k for k in per_agg.keys()]))
print(f"Aggregators found (strict-counted): {len(keys)}\n")

missing = []
extra = []
for k in keys:
    count = per_agg[k]
    grid, agg = k
    print(f"{grid}/{agg}: {count}")
    if count < 25:
        missing.append((grid, agg, count))
    elif count > 25:
        extra.append((grid, agg, count))

if missing:
    print("\nAggregators with <25 rounds:")
    for g,a,c in missing:
        print(f" - {g}/{a}: {c} (needs {25-c})")
else:
    print('\nNo aggregators with <25 rounds')

if extra:
    print('\nAggregators with >25 rounds:')
    for g,a,c in extra:
        print(f" - {g}/{a}: {c} (excess {c-25})")
else:
    print('\nNo aggregators with >25 rounds')

if suspect_files:
    print('\nSuspicious filenames (not exact round_###.npz):')
    for p in suspect_files:
        print(' - ' + p)
else:
    print('\nNo suspicious filenames found')
