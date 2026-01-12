#!/usr/bin/env python3
"""Sequential runner: runs each incomplete aggregator task one at a time until all rounds present.

Usage: python tools/sequential_runner.py

The script reads the `apra_mnist_runs_full` directory, finds incomplete aggregators (25 rounds expected), and invokes
`python scripts/run_apra_mnist_full.py` with parsed grid parameters. It logs per-task stdout/stderr to the aggregator
folder as `sequential_stdout.log` and `sequential_stderr.log` and retries up to 3 times on failure.
"""
import subprocess
import time
from pathlib import Path
import re
import sys

ROOT = Path('apra_mnist_runs_full')
AGGS = ['apra_basic','apra_weighted','trimmed','median']
TOTAL = 25
RE_GRID = re.compile(r'sd(?P<sketch>\d+)_ns(?P<ns>\d+)_zt(?P<zt>[0-9\.]+)')
# Python executable and script path
PY = sys.executable
SCRIPT = Path('scripts') / 'run_apra_mnist_full.py'
# Max seconds to allow a single attempt to run before force-killing and retrying.
# Empirically chosen to allow long experiments while still recovering from hangs.
ATTEMPT_TIMEOUT = 60 * 30  # 30 minutes

def completed_rounds_count(directory: Path) -> int:
    """Return number of completed rounds matching exact 'round_###.npz' filenames."""
    pattern = re.compile(r'^round_(\d{3})\.npz$')
    return sum(1 for p in directory.iterdir() if p.is_file() and pattern.match(p.name))

def completed_rounds_list(directory: Path):
    """Return sorted list of integer round indices present as 'round_###.npz'."""
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

# Build list of incomplete tasks
tasks = []
for grid_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
    # search inside for AGGS
    subdirs = [d for d in grid_dir.iterdir() if d.is_dir()]
    if any(d.name in AGGS for d in subdirs):
        grid = grid_dir.name
        m = RE_GRID.match(grid)
        if not m:
            print('Skipping malformed grid dir', grid)
            continue
        sketch = int(m.group('sketch'))
        ns = int(m.group('ns'))
        zt = float(m.group('zt'))
        for agg in sorted([d for d in subdirs if d.name in AGGS], key=lambda x: x.name):
            # Count only exact checkpoint filenames like 'round_001.npz' (avoid '*.tmp.npz')
            def completed_rounds_count(directory: Path):
                pattern = re.compile(r'^round_(\d{3})\.npz$')
                files = [p for p in directory.iterdir() if p.is_file() and pattern.match(p.name)]
                return len(files)

            if completed_rounds_count(agg) < TOTAL:
                tasks.append((grid, agg.name, sketch, ns, zt, agg))
    else:
        # maybe nested structure
        nested = [d for d in subdirs if d.name == grid_dir.name]
        if nested:
            sub = nested[0]
            grid = grid_dir.name
            m = RE_GRID.match(grid)
            if not m:
                continue
            sketch = int(m.group('sketch'))
            ns = int(m.group('ns'))
            zt = float(m.group('zt'))
            for aggname in AGGS:
                agg = sub/aggname
                if agg.exists():
                    # Recompute completed rounds using strict filename match

                        if completed_rounds_count(agg) < TOTAL:
                            tasks.append((grid, aggname, sketch, ns, zt, agg))

if not tasks:
    print('No incomplete tasks found. Nothing to do.')
    rounds = completed_rounds_list(agg)
    if len(rounds) >= TOTAL:
        print(f'    SUCCESS: {agg_path} has {len(rounds)} rounds')
    else:
        print(f'    Not complete yet: {len(rounds)}/{TOTAL} rounds found')

print(f'Will run {len(tasks)} incomplete tasks sequentially (1 at a time).')

for i, (grid, agg, sketch, ns, zt, agg_path) in enumerate(tasks, 1):
    print(f'[{i}/{len(tasks)}] Grid={grid} Agg={agg} sketch={sketch} ns={ns} zt={zt} path={agg_path}')
    log_out = agg_path / 'sequential_stdout.log'
    log_err = agg_path / 'sequential_stderr.log'
    retries = 3
    success = False
    for attempt in range(1, retries+1):
        print(f'  Attempt {attempt}/{retries}...')
        cmd = [PY, str(SCRIPT), '--sketch_dim', str(sketch), '--n_sketches', str(ns), '--z_thresh', str(zt), '--agg_method', agg, '--output_dir', str(ROOT), '--batch_size', '2']
        with open(log_out, 'ab') as outfh, open(log_err, 'ab') as errfh:
            proc = subprocess.Popen(cmd, stdout=outfh, stderr=errfh)
            print(f'    PID {proc.pid} started; waiting for completion...')
            try:
                ret = proc.wait(timeout=ATTEMPT_TIMEOUT)
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
        # Check whether aggregator has all rounds
        files = list(agg_path.glob('round_*.npz'))
        if len(files) >= TOTAL:
            print(f'    SUCCESS: {agg_path} has {len(files)} rounds')
            success = True
            break
        else:
            print(f'    Not complete yet: {len(files)}/{TOTAL} rounds found')
            time.sleep(2)
    if not success:
        print(f'  FAILED to complete {grid}/{agg} after {retries} attempts — moving to next task')
    # small delay between tasks
    time.sleep(1)

print('Sequential run completed. Re-run report_incomplete.py to check status.')
