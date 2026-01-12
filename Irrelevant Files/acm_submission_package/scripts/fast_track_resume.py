#!/usr/bin/env python
"""
Fast-track resume for incomplete grids: launches ALL 4 aggregators in parallel per grid.

Usage:
    python scripts/fast_track_resume.py --output_dir apra_mnist_runs_full

This script:
1. Scans for incomplete grids (those not at 25/25 rounds for all 4 aggregators)
2. Launches 4 detached processes per grid (one per aggregator)
3. All processes run in parallel, reducing wall time from ~4x to ~1x
"""

import os
import sys
import argparse
import subprocess
import itertools
from pathlib import Path

def count_rounds_per_aggregator(grid_path):
    """Count completed rounds per aggregator in a grid directory."""
    aggs = ['apra_weighted', 'apra_basic', 'trimmed', 'median']
    round_counts = {}
    for agg in aggs:
        agg_dir = os.path.join(grid_path, agg)
        if os.path.isdir(agg_dir):
            rounds = [f for f in os.listdir(agg_dir) if f.startswith('round_') and f.endswith('.npz')]
            round_counts[agg] = len(rounds)
        else:
            round_counts[agg] = 0
    return round_counts


def is_grid_complete(grid_path, target_rounds=25):
    """Check if all aggregators have target_rounds completed."""
    round_counts = count_rounds_per_aggregator(grid_path)
    return all(count >= target_rounds for count in round_counts.values())


def extract_grid_params(grid_name):
    """Parse grid name 'sd64_ns2_zt2.0' -> (64, 2, 2.0)"""
    parts = grid_name.split('_')
    sd = int(parts[0][2:])
    ns = int(parts[1][2:])
    zt = float(parts[2][2:])
    return sd, ns, zt


def run_detached_process(cmd, log_path=None):
    """Launch a detached Python process on Windows/POSIX."""
    if log_path:
        logf = open(log_path, 'ab')
    else:
        logf = subprocess.PIPE
    
    env = os.environ.copy()
    cwd = os.getcwd()
    prev = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = cwd + os.pathsep + prev if prev else cwd
    
    if os.name == 'nt':
        # Windows: use DETACHED_PROCESS
        creationflags = 0x00000008
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, shell=True,
                           creationflags=creationflags, env=env)
    else:
        # POSIX: use setsid
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, shell=True,
                           preexec_fn=os.setsid, env=env)
    return p


def main():
    parser = argparse.ArgumentParser(description='Fast-track resume with parallel aggregators')
    parser.add_argument('--output_dir', default='apra_mnist_runs_full',
                       help='Output directory containing grids')
    parser.add_argument('--rounds', type=int, default=25, help='Target rounds per grid')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--attack', default='none', help='Attack type')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without launching')
    
    args = parser.parse_args()
    
    outdir = args.output_dir
    if not os.path.isdir(outdir):
        print(f"Output directory not found: {outdir}")
        sys.exit(1)
    
    # Find all grids
    grids = sorted([d for d in os.listdir(outdir) 
                   if os.path.isdir(os.path.join(outdir, d)) and d.startswith('sd')])
    
    if not grids:
        print(f"No grids found in {outdir}")
        sys.exit(0)
    
    print(f"Found {len(grids)} grids")
    print("-" * 70)
    
    # Check which are incomplete and need resuming
    incomplete = []
    for grid in grids:
        grid_path = os.path.join(outdir, grid)
        round_counts = count_rounds_per_aggregator(grid_path)
        max_rounds = max(round_counts.values()) if round_counts else 0
        
        if max_rounds < args.rounds:
            incomplete.append((grid, round_counts))
            status_str = ', '.join([f"{agg}={count}" for agg, count in sorted(round_counts.items())])
            print(f"{grid:20s}: {max_rounds}/25  ({status_str})")
        else:
            print(f"{grid:20s}: ✓ COMPLETE")
    
    if not incomplete:
        print("\n✓ All grids complete!")
        return 0
    
    print("-" * 70)
    print(f"\nResuming {len(incomplete)} incomplete grids with parallel aggregators")
    print("-" * 70)
    
    # Build and launch commands for incomplete grids
    launched_cmds = []
    agg_methods = ['apra_weighted', 'apra_basic', 'trimmed', 'median']
    
    for grid, round_counts in incomplete:
        sd, ns, zt = extract_grid_params(grid)
        grid_path = os.path.join(outdir, grid)
        
        # Launch all 4 aggregators in parallel for this grid
        for agg_method in agg_methods:
            cmd = (f"python -u scripts/run_apra_mnist_full.py "
                  f"--sketch_dim {sd} --n_sketches {ns} --z_thresh {zt} "
                  f"--rounds {args.rounds} --local_epochs {args.local_epochs} "
                  f"--batch_size {args.batch_size} --clients {args.clients} "
                  f"--attack {args.attack} --output_dir {grid_path} "
                  f"--agg_method {agg_method}")
            
            launched_cmds.append((grid, agg_method, cmd))
    
    print(f"\nTotal commands to launch: {len(launched_cmds)}")
    print(f"  {len(incomplete)} grids × 4 aggregators = {len(launched_cmds)} processes")
    
    if args.dry_run:
        print("\n[DRY RUN] Commands that would be launched:")
        for grid, agg, cmd in launched_cmds[:4]:
            print(f"  {grid} / {agg}")
        if len(launched_cmds) > 4:
            print(f"  ... and {len(launched_cmds) - 4} more")
        return 0
    
    print("\nLaunching processes...")
    
    pids = []
    for grid, agg_method, cmd in launched_cmds:
        grid_path = os.path.join(outdir, grid)
        log_path = os.path.join(grid_path, f"{agg_method}.log")
        
        p = run_detached_process(cmd, log_path=log_path)
        pid = getattr(p, 'pid', None)
        pids.append(pid)
        print(f"  ✓ Launched {grid:20s} / {agg_method:15s} (pid={pid})")
    
    print("-" * 70)
    print(f"\n✓ Launched {len(pids)} detached processes")
    print(f"\nMonitor progress with:")
    print(f"  python scripts/monitor_sweep.py --output_dir {outdir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
