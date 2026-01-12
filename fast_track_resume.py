"""
fast_track_resume.py

Launch per-grid, per-aggregator runs in parallel while skipping already-completed runs.
This script uses the same CLI for `scripts/run_apra_mnist_full.py` but supplies a `--run_tag` equal
to the aggregator name so that outputs do not collide.

Usage:
  python fast_track_resume.py --output_dir apra_mnist_results --rounds 25 --max_procs 6

"""
import os
import subprocess
import argparse
import time
import math

SKETCH_DIMS = [64, 128]
N_SKETCHES = [1, 2]
Z_THRESH = [2.0, 3.0]
AGGS = ['apra_weighted', 'apra_basic', 'trimmed', 'median']


def results_dir_for(sketch_dim, n_sketches, z_thresh, output_dir, run_tag):
    tag_suffix = f'_{run_tag}' if run_tag else ''
    return os.path.join(output_dir, f'sd{sketch_dim}_ns{n_sketches}_zt{z_thresh}{tag_suffix}')


def is_completed(results_dir, rounds):
    csv_path = os.path.join(results_dir, 'results.csv')
    if not os.path.exists(results_dir):
        return False
    # If a results.csv exists, assume completion if it has >= rounds rows (header + rows)
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            return len(lines) - 1 >= rounds
        except Exception:
            return False
    # Fall back to checking number of checkpoint files in aggregator dir
    # This heuristic assumes checkpoints are saved under an aggregator subfolder when run tag not used.
    # For safety, consider it not completed.
    return False


def build_command(sketch_dim, n_sketches, z_thresh, agg, output_dir, rounds, clients, local_epochs, batch_size, seed):
    cmd = [
        'python', 'scripts/run_apra_mnist_full.py',
        '--sketch_dim', str(sketch_dim),
        '--n_sketches', str(n_sketches),
        '--z_thresh', str(z_thresh),
        '--agg_method', agg,
        '--output_dir', output_dir,
        '--run_tag', agg,
        '--rounds', str(rounds),
        '--clients', str(clients),
        '--local_epochs', str(local_epochs),
        '--batch_size', str(batch_size),
        '--seed', str(seed)
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='apra_mnist_results', help='Base output directory used by experiments')
    parser.add_argument('--rounds', type=int, default=25)
    parser.add_argument('--clients', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_procs', type=int, default=6, help='Maximum concurrent processes')
    parser.add_argument('--poll_interval', type=float, default=5.0, help='Seconds to wait between process polls')
    args = parser.parse_args()

    to_launch = []
    for sd in SKETCH_DIMS:
        for ns in N_SKETCHES:
            for z in Z_THRESH:
                for agg in AGGS:
                    results_dir = results_dir_for(sd, ns, z, args.output_dir, agg)
                    if is_completed(results_dir, args.rounds):
                        print(f'[SKIP] {results_dir} already completed (>= {args.rounds} rounds)')
                        continue
                    cmd = build_command(sd, ns, z, agg, args.output_dir, args.rounds, args.clients, args.local_epochs, args.batch_size, args.seed)
                    to_launch.append((sd, ns, z, agg, results_dir, cmd))

    print(f'Prepared {len(to_launch)} runs to launch')
    processes = []  # list of tuples (Popen, meta)
    idx = 0
    while idx < len(to_launch) or processes:
        # Launch until hitting max_procs or no more to launch
        while idx < len(to_launch) and len(processes) < args.max_procs:
            sd, ns, z, agg, results_dir, cmd = to_launch[idx]
            print(f'Launching {agg} for sd{sd}_ns{ns}_zt{z} -> {results_dir}')
            # Ensure parent directory exists so child can create its own results dir
            os.makedirs(os.path.dirname(results_dir) or '.', exist_ok=True)
            p = subprocess.Popen(cmd)
            processes.append((p, (sd, ns, z, agg, results_dir)))
            idx += 1
            time.sleep(0.5)

        # Poll processes
        time.sleep(args.poll_interval)
        alive = []
        for p, meta in processes:
            ret = p.poll()
            if ret is None:
                alive.append((p, meta))
            else:
                sd, ns, z, agg, results_dir = meta
                print(f'Process finished for {agg} sd{sd}_ns{ns}_zt{z} (exit {ret})')
        processes = alive

    print('All requested runs launched and completed (or are running to completion).')


if __name__ == '__main__':
    main()
