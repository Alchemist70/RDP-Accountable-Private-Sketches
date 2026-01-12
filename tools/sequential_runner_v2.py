#!/usr/bin/env python3
"""Sequential runner v2: clean implementation that avoids tmp-file counting issues.

Usage: python tools/sequential_runner_v2.py

This is a drop-in replacement to `tools/sequential_runner.py` that:
- Counts only exact `round_###.npz` files
- Discovers aggregator dirs whether nested or direct
- Waits for children with a per-attempt timeout and retries
"""
import subprocess
import time
from pathlib import Path
import re
import sys

ROOT = Path('apra_mnist_runs_full')
AGGS = ['apra_basic', 'apra_weighted', 'trimmed', 'median']
TOTAL = 25
RE_GRID = re.compile(r'sd(?P<sketch>\d+)_ns(?P<ns>\d+)_zt(?P<zt>[0-9\.]+)')
PY = sys.executable
SCRIPT = Path('scripts') / 'run_apra_mnist_full.py'
ATTEMPT_TIMEOUT = 60 * 30  # 30 minutes
LONG_TIMEOUT = 60 * 90  # 90 minutes for known heavy grids

# Grids that historically require much more time due to heavy TF/protobuf work.
# Use their directory name (grid) exactly as discovered under `apra_mnist_runs_full`.
HEAVY_GRIDS = {
    'sd128_ns2_zt2.0',
    'sd128_ns2_zt3.0',
    'sd64_ns2_zt2.0',
    'sd64_ns2_zt3.0',
}


def completed_rounds_count(directory: Path) -> int:
    pattern = re.compile(r'^round_(\d{3})\.npz$')
    return sum(1 for p in directory.iterdir() if p.is_file() and pattern.match(p.name))


def completed_rounds_list(directory: Path):
    pattern = re.compile(r'^round_(\d{3})\.npz$')
    rounds = []
    for p in directory.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                rounds.append(int(m.group(1)))
    return sorted(rounds)


if not ROOT.exists():
    print('No apra_mnist_runs_full found. Exiting.')
    raise SystemExit(1)

# Discover tasks
tasks = []
for grid_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
    grid = grid_dir.name
    m = RE_GRID.match(grid)
    if not m:
        continue
    sketch = int(m.group('sketch'))
    ns = int(m.group('ns'))
    zt = float(m.group('zt'))

    direct_aggs = {d.name: d for d in grid_dir.iterdir() if d.is_dir() and d.name in AGGS}
    nested_dir = grid_dir / grid if (grid_dir / grid).exists() and (grid_dir / grid).is_dir() else None
    nested_aggs = {d.name: d for d in (nested_dir.iterdir() if nested_dir else []) if d.is_dir() and d.name in AGGS}
    all_aggs = {**direct_aggs, **nested_aggs}

    for aggname, agg_path in sorted(all_aggs.items()):
        if completed_rounds_count(agg_path) < TOTAL:
            tasks.append((grid, aggname, sketch, ns, zt, agg_path))

if not tasks:
    print('No incomplete tasks found. Nothing to do.')
    raise SystemExit(0)

print(f'Will run {len(tasks)} incomplete tasks sequentially (1 at a time).')

for i, (grid, agg, sketch, ns, zt, agg_path) in enumerate(tasks, 1):
    print(f'[{i}/{len(tasks)}] Grid={grid} Agg={agg} sketch={sketch} ns={ns} zt={zt} path={agg_path}')
    log_out = agg_path / 'sequential_stdout.log'
    log_err = agg_path / 'sequential_stderr.log'
    retries = 3
    success = False
    for attempt in range(1, retries + 1):
        print(f'  Attempt {attempt}/{retries}...')
        # On retries >1, try a reduced-workload fallback to recover from hangs
        # (lower clients/local_epochs/batch_size). This is a conservative
        # mitigation applied only when the default attempt times out.
        base_cmd = [PY, str(SCRIPT), '--sketch_dim', str(sketch), '--n_sketches', str(ns), '--z_thresh', str(zt), '--agg_method', agg, '--output_dir', str(ROOT)]
        if attempt == 1:
            # default attempt uses small batch size (fast) but preserves clients/local_epochs
            cmd = base_cmd + ['--batch_size', '2']
        else:
            # reduced-workload fallback: fewer clients, fewer local epochs, tiny batch
            cmd = base_cmd + ['--batch_size', '1', '--clients', '10', '--local_epochs', '1']
        with open(log_out, 'ab') as outfh, open(log_err, 'ab') as errfh:
            proc = subprocess.Popen(cmd, stdout=outfh, stderr=errfh)
            print(f'    PID {proc.pid} started; waiting for completion...')
            try:
                # Use a longer timeout for known-heavy grids so we don't kill
                # runs that are simply slow due to expensive tracing/serialization.
                timeout = LONG_TIMEOUT if grid in HEAVY_GRIDS else ATTEMPT_TIMEOUT
                ret = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f'    Attempt timed out after {ATTEMPT_TIMEOUT}s - terminating PID {proc.pid}...', flush=True)
                try:
                    proc.kill()
                except Exception:
                    pass
                ret = -9
            except KeyboardInterrupt:
                print('    KeyboardInterrupt received: terminating child process and exiting.', flush=True)
                try:
                    proc.kill()
                except Exception:
                    pass
                raise
        print(f'    Process exited with code {ret}')

        rounds = completed_rounds_list(agg_path)
        if len(rounds) >= TOTAL:
            print(f'    SUCCESS: {agg_path} has {len(rounds)} rounds')
            success = True
            break
        else:
            print(f'    Not complete yet: {len(rounds)}/{TOTAL} rounds found')
            time.sleep(2)
    if not success:
        print(f'  FAILED to complete {grid}/{agg} after {retries} attempts — moving to next task')
    time.sleep(1)

print('Sequential run completed. Re-run report_incomplete.py to check status.')
