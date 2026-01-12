#!/usr/bin/env python
"""Direct runner for incomplete tasks - bypass launcher issues."""
import os
import sys
import subprocess
import time

outdir = 'apra_mnist_runs_full'

# Incomplete tasks based on last status check
incomplete = [
    ('sd128_ns1_zt2.0', 'apra_basic', 79),
    ('sd128_ns1_zt2.0', 'apra_weighted', 79),
    ('sd128_ns1_zt2.0', 'median', 79),
    ('sd128_ns1_zt2.0', 'trimmed', 79),
    ('sd128_ns1_zt3.0', 'apra_basic', 15),
    ('sd128_ns1_zt3.0', 'apra_weighted', 15),
    ('sd128_ns1_zt3.0', 'median', 15),
    ('sd128_ns1_zt3.0', 'trimmed', 15),
    ('sd128_ns2_zt2.0', 'apra_basic', 24),
    ('sd128_ns2_zt2.0', 'apra_weighted', 24),
    ('sd128_ns2_zt2.0', 'median', 24),
    ('sd128_ns2_zt2.0', 'trimmed', 24),
    ('sd64_ns2_zt2.0', 'apra_basic', 79),
    ('sd64_ns2_zt2.0', 'apra_weighted', 79),
    ('sd64_ns2_zt2.0', 'median', 79),
    ('sd64_ns2_zt2.0', 'trimmed', 79),
    ('sd64_ns2_zt3.0', 'apra_basic', 80),
    ('sd64_ns2_zt3.0', 'apra_weighted', 80),
    ('sd64_ns2_zt3.0', 'median', 80),
    ('sd64_ns2_zt3.0', 'trimmed', 80),
]

print(f"Running {len(incomplete)} incomplete tasks sequentially...")
print()

for grid, agg, existing in incomplete:
    # Parse grid name
    parts = grid.replace('sd', '').replace('_ns', ' ').replace('_zt', ' ').split()
    sd, ns, zt = int(parts[0]), int(parts[1]), float(parts[2])
    
    cmd = (
        f"python -u scripts/run_apra_mnist_full.py "
        f"--sketch_dim {sd} --n_sketches {ns} --z_thresh {zt} "
        f"--rounds 25 --local_epochs 3 --batch_size 4 "
        f"--clients 100 --attack layer_backdoor "
        f"--output_dir {outdir} --agg_method {agg}"
    )
    
    print(f"[{grid:15} / {agg:12}] Running (resuming {existing}/25)...")
    result = subprocess.run(cmd, shell=True)
    print(f"[{grid:15} / {agg:12}] Completed (exit code: {result.returncode})")
    print()
    time.sleep(1)

print("All tasks completed!")
