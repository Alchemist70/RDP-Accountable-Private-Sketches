#!/usr/bin/env python
"""
Monitor grid sweep progress: display per-grid, per-aggregator status.

Usage:
    python scripts/monitor_sweep.py --output_dir apra_mnist_runs_full
"""

import os
import sys
import argparse
import time


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


def main():
    parser = argparse.ArgumentParser(description='Monitor grid sweep progress')
    parser.add_argument('--output_dir', default='apra_mnist_runs_full', help='Output directory')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (seconds)')
    parser.add_argument('--target_rounds', type=int, default=25, help='Target rounds per grid')
    
    args = parser.parse_args()
    
    outdir = args.output_dir
    if not os.path.isdir(outdir):
        print(f"Output directory not found: {outdir}")
        sys.exit(1)
    
    grids = sorted([d for d in os.listdir(outdir)
                   if os.path.isdir(os.path.join(outdir, d)) and d.startswith('sd')])
    
    if not grids:
        print(f"No grids found in {outdir}")
        sys.exit(0)
    
    check_num = 0
    while True:
        check_num += 1
        print(f"\n[{time.strftime('%H:%M:%S')}] Check #{check_num}")
        print("-" * 70)
        
        all_complete = True
        for grid in grids:
            grid_path = os.path.join(outdir, grid)
            round_counts = count_rounds_per_aggregator(grid_path)
            max_rounds = max(round_counts.values()) if round_counts else 0
            
            if max_rounds >= args.target_rounds:
                print(f"  {grid:20s}: ✓ {max_rounds}/{args.target_rounds}")
            else:
                print(f"  {grid:20s}:   {max_rounds}/{args.target_rounds}", end='')
                details = ', '.join([f"{agg}={c}" for agg, c in sorted(round_counts.items())])
                print(f"  ({details})")
                all_complete = False
        
        if all_complete:
            print("\n" + "="*70)
            print("✓ ALL GRIDS COMPLETE!")
            print("="*70)
            break
        
        print(f"\nWaiting {args.interval}s until next check...")
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)
