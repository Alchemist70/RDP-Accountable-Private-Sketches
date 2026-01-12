#!/usr/bin/env python
"""
Direct parallel runner for incomplete APRA-MNIST tasks.
Bypasses resume_now.py launcher - runs tasks directly using multiprocessing.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# Config
OUTPUT_ROOT = Path("apra_mnist_runs_full")
PYTHON_EXE = sys.executable
MAX_PROCS = 3
TARGET_ROUNDS = 25
RETRIES = 3
AGG_METHODS = ["apra_basic", "apra_weighted", "trimmed", "median"]
# Timeout per attempt in seconds. If a run hangs beyond this, it will be killed and retried.
ATTEMPT_TIMEOUT = 60 * 30  # 30 minutes


def completed_rounds_count(directory: Path) -> int:
    """Count only files named exactly round_###.npz to avoid tmp duplicates."""
    import re as _re
    pattern = _re.compile(r'^round_(\d{3})\.npz$')
    return sum(1 for p in directory.iterdir() if p.is_file() and pattern.match(p.name))

def get_incomplete_tasks():
    """Find all tasks that haven't completed 25 rounds."""
    tasks = []
    
    if not OUTPUT_ROOT.exists():
        return tasks
    
    # Iterate over all grid directories
    for grid_dir in OUTPUT_ROOT.iterdir():
        if not grid_dir.is_dir():
            continue

        # Iterate over aggregation methods (only known aggregator dirs)
        for agg_name in AGG_METHODS:
            agg_dir = grid_dir / agg_name
            if not agg_dir.exists() or not agg_dir.is_dir():
                continue

            agg_method = agg_name
            grid_name = grid_dir.name

            # Count completed rounds using strict filename match
            completed = completed_rounds_count(agg_dir)

            if completed < TARGET_ROUNDS:
                tasks.append({
                    'grid': grid_name,
                    'agg': agg_method,
                    'completed': completed,
                    'remaining': TARGET_ROUNDS - completed
                })
    
    return sorted(tasks, key=lambda x: -x['remaining'])  # Most incomplete first

def run_task(task):
    """Run a single task using run_apra_mnist_full.py"""
    grid = task['grid']
    agg = task['agg']
    
    # Parse grid parameters from name (e.g., "sd64_ns1_zt2.0")
    try:
        parts = grid.split('_')
        sketch_dim = int(parts[0][2:])  # sd64 -> 64
        n_sketches = int(parts[1][2:])  # ns1 -> 1
        z_thresh = float(parts[2][2:])  # zt2.0 -> 2.0
    except Exception as e:
        print(f"[WARN] Unexpected grid name '{grid}', skipping task: {e}", flush=True)
        return 2
    
    cmd = [
        PYTHON_EXE, "-u",
        "scripts/run_apra_mnist_full.py",
        "--sketch_dim", str(sketch_dim),
        "--n_sketches", str(n_sketches),
        "--z_thresh", str(z_thresh),
        "--agg_method", agg,
        "--batch_size", "2",
        "--output_dir", "apra_mnist_runs_full"
    ]
    
    print(f"[{grid:20s} / {agg:12s}] Starting (resuming {task['completed']}/{TARGET_ROUNDS})...", flush=True)

    # Ensure agg directory exists for launcher logs
    agg_path = OUTPUT_ROOT / grid / agg
    agg_path.mkdir(parents=True, exist_ok=True)
    launcher_stderr = agg_path / 'launcher_stderr.log'
    launcher_stdout = agg_path / 'launcher_stdout.log'

    # Retry loop
    for attempt in range(1, RETRIES + 1):
        try:
            # Stream stdout/stderr to files to avoid capturing large outputs in memory
            with open(launcher_stdout, 'ab') as outf, open(launcher_stderr, 'ab') as errf:
                proc = subprocess.Popen(cmd, stdout=outf, stderr=errf)
                print(f"    PID {proc.pid} started; waiting (timeout={ATTEMPT_TIMEOUT}s)...", flush=True)
                try:
                    rc = proc.wait(timeout=ATTEMPT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print(f"    PID {proc.pid} timed out after {ATTEMPT_TIMEOUT}s - killing...", flush=True)
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    rc = -9

            if rc == 0:
                print(f"[{grid:20s} / {agg:12s}] ✓ COMPLETE (attempt {attempt})", flush=True)
                return 0
            else:
                print(f"[{grid:20s} / {agg:12s}] ✗ FAILED (exit {rc}) attempt {attempt}", flush=True)
                if attempt == RETRIES:
                    # Print tail of stderr for debugging
                    try:
                        tail = launcher_stderr.read_text(encoding='utf-8')[-800:]
                        print(f"  Error (tail): {tail}", flush=True)
                    except Exception:
                        pass
                    return rc
                print(f"  Retrying {grid}/{agg} (attempt {attempt+1}/{RETRIES})...", flush=True)
        except KeyboardInterrupt:
            print(f"[{grid:20s} / {agg:12s}] Interrupted by user, aborting.", flush=True)
            return 2
        except Exception as e:
            print(f"[{grid:20s} / {agg:12s}] ✗ EXCEPTION: {e} (attempt {attempt})", flush=True)
            if attempt == RETRIES:
                return 1
            print(f"  Retrying after exception...", flush=True)
    return 1

def main():
    tasks = get_incomplete_tasks()
    
    if not tasks:
        print("✓ All tasks complete!")
        return
    
    print(f"Found {len(tasks)} incomplete tasks (need {sum(t['remaining'] for t in tasks)} more checkpoints)\n")
    
    # Show incomplete status
    grids_status = defaultdict(lambda: {'total': 0, 'completed': 0})
    for task in tasks:
        grids_status[task['grid']]['total'] += 1
        grids_status[task['grid']]['completed'] += 0  # Will update below
    
    for grid_dir in OUTPUT_ROOT.iterdir():
        if grid_dir.is_dir():
            grid = grid_dir.name
            if grid in grids_status:
                completed_aggs = sum(1 for agg_dir in grid_dir.iterdir() 
                                    if agg_dir.is_dir() and completed_rounds_count(agg_dir) >= TARGET_ROUNDS)
                grids_status[grid]['completed'] = completed_aggs
    
    print("Status by grid:")
    for grid in sorted(grids_status.keys()):
        completed = grids_status[grid]['completed']
        total = grids_status[grid]['total']
        print(f"  {grid}: {completed}/{total} aggregators complete")
    print()
    
    # Run tasks in parallel
    print(f"Running up to {MAX_PROCS} tasks in parallel...\n")
    with Pool(processes=MAX_PROCS) as pool:
        results = pool.map(run_task, tasks)
    
    print(f"\n✓ Batch complete. {sum(1 for r in results if r == 0)}/{len(tasks)} tasks succeeded.")

if __name__ == '__main__':
    main()
