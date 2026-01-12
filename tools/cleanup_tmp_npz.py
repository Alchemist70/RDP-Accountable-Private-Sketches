"""Robust cleanup utility for stray checkpoint filenames.

This will scan `apra_mnist_runs_full` for any file whose name starts
with `round_` and attempts to canonicalize it to the exact
`round_###.npz` form. For each canonical name we:

- collect all candidate files (e.g. `round_001.npz.npz`, `round_001.npz.`,
  `round_001.npz.tmp.npz`, etc.)
- if a canonical `round_###.npz` already exists, remove other candidates
  (they are duplicates)
- otherwise pick the largest candidate (best-effort) and rename it to
  the canonical name, removing other candidates.

This is idempotent and safe to re-run. It keeps the largest available
file for each round when a single canonical file is missing.
"""
from pathlib import Path
from collections import defaultdict
import shutil
import re

ROOT = Path('apra_mnist_runs_full')
if not ROOT.exists():
    print('No apra_mnist_runs_full directory found. Exiting.')
    raise SystemExit(0)

rx_round = re.compile(r'^(round_\d{3})(.*)$', flags=re.IGNORECASE)

# Map canonical relative path -> list of candidate Path objects
candidates = defaultdict(list)

for p in ROOT.rglob('round_*'):
    if not p.is_file():
        continue
    m = rx_round.match(p.name)
    if not m:
        continue
    base = m.group(1)  # e.g. 'round_001'
    # canonical name we'll aim for
    canonical_name = base + '.npz'
    # canonical path is in same directory
    canonical_path = p.with_name(canonical_name)
    candidates[(p.parent.resolve(), canonical_name)].append(p)

total_candidates = sum(len(v) for v in candidates.values())
print(f'Found {total_candidates} round-* candidate files across {len(candidates)} canonical slots')

processed = 0
for (dirpath, canonical_name), files in sorted(candidates.items()):
    dirpath = Path(dirpath)
    canonical_path = dirpath / canonical_name
    # If canonical exists already, remove all other candidates safely
    if canonical_path.exists():
        for f in files:
            if f.resolve() == canonical_path.resolve():
                continue
            try:
                print(f'Removing duplicate candidate: {f} (canonical exists)')
                f.unlink()
            except Exception as e:
                print(f'  Failed to remove {f}: {e}')
        processed += 1
        continue

    # No canonical exists: pick the best candidate (largest file) to promote
    best = None
    best_size = -1
    for f in files:
        try:
            sz = f.stat().st_size
        except Exception:
            sz = -1
        if sz > best_size:
            best = f
            best_size = sz

    if best is None:
        continue

    # Promote best candidate to canonical
    try:
        print(f'Promoting {best} -> {canonical_path} (size={best_size})')
        try:
            best.replace(canonical_path)
        except Exception:
            shutil.move(str(best), str(canonical_path))
        # Remove any remaining candidates (if any)
        for f in files:
            if f.resolve() == canonical_path.resolve():
                continue
            if f.exists():
                try:
                    print(f'  Removing sibling candidate: {f}')
                    f.unlink()
                except Exception as e:
                    print(f'    Failed to remove {f}: {e}')
    except Exception as e:
        print(f'Error promoting {best} to {canonical_path}: {e}')

    processed += 1

print(f'Processed {processed} canonical slots. Cleanup complete.')
