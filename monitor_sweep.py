"""
monitor_sweep.py

Simple progress monitor for APRA runs launched with `--run_tag` (aggregator-specific folders).
It reports checkpoint counts per aggregator and whether the run appears complete.

Usage:
  python monitor_sweep.py --output_dir apra_mnist_results --rounds 25
"""
import os
import argparse
import glob

SKETCH_DIMS = [64, 128]
N_SKETCHES = [1, 2]
Z_THRESH = [2.0, 3.0]
AGGS = ['apra_weighted', 'apra_basic', 'trimmed', 'median']


def results_dir_for(sketch_dim, n_sketches, z_thresh, output_dir, run_tag):
    tag_suffix = f'_{run_tag}' if run_tag else ''
    return os.path.join(output_dir, f'sd{sketch_dim}_ns{n_sketches}_zt{z_thresh}{tag_suffix}')


def count_checkpoints(agg_dir):
    if not os.path.exists(agg_dir):
        return 0
    files = glob.glob(os.path.join(agg_dir, 'round_*.npz'))
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='apra_mnist_results')
    parser.add_argument('--rounds', type=int, default=25)
    args = parser.parse_args()

    summary = []
    for sd in SKETCH_DIMS:
        for ns in N_SKETCHES:
            for z in Z_THRESH:
                grid_line = []
                all_complete = True
                for agg in AGGS:
                    results_dir = results_dir_for(sd, ns, z, args.output_dir, agg)
                    agg_dir = os.path.join(results_dir, agg)
                    cnt = count_checkpoints(agg_dir)
                    status = '✓' if cnt >= args.rounds else f'{cnt}/{args.rounds}'
                    grid_line.append((agg, status))
                    if cnt < args.rounds:
                        all_complete = False
                grid_name = f'sd{sd}_ns{ns}_zt{z}'
                status_str = 'COMPLETE' if all_complete else 'INCOMPLETE'
                print(f'{grid_name}: {status_str}  ' + '  '.join([f"{a}={s}" for a, s in grid_line]))

if __name__ == '__main__':
    main()
