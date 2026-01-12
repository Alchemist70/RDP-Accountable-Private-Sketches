#!/usr/bin/env python
"""Test single aggregator method to diagnose errors."""

import subprocess
import sys

# Test apra_basic on one grid to see the error
cmd = (
    "python -u scripts/run_apra_mnist_full.py "
    "--sketch_dim 128 --n_sketches 1 --z_thresh 2.0 "
    "--rounds 25 --local_epochs 3 --batch_size 32 "
    "--clients 100 --attack layer_backdoor "
    "--output_dir apra_mnist_runs_full --agg_method apra_basic"
)


def run_apra_basic():
    print("=" * 70)
    print("Testing apra_basic aggregator")
    print("=" * 70)
    print(f"Command: {cmd}\n")
    # Run without suppressing output
    result = subprocess.run(cmd, shell=True)
    return result


def test_apra_basic_runner():
    """Pytest entry: run the apra_basic runner and assert it exits with code 0.

    This test will execute the script as a subprocess. It may be slow; use -k
    to select or skip as needed. If the environment is not configured to run
    the full experiment, this test can be skipped by the user.
    """
    res = run_apra_basic()
    assert res.returncode == 0, f"apra_basic runner failed: returncode={res.returncode} stdout={res.stdout} stderr={res.stderr}"


if __name__ == '__main__':
    res = run_apra_basic()
    sys.exit(res.returncode)
