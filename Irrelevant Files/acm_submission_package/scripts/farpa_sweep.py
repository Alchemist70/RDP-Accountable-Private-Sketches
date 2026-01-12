"""Parameter sweep for FARPA.

Runs FARPA over combinations of sketch_dim_per_layer, eps_sketch, and z_thresh,
collects per-client trust scores and benign_count, and writes results to CSV.
"""
import os
import sys
import csv
import itertools
import numpy as np

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl_helpers import farpa_aggregate


def make_client(base, noise_scale=0.01, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    w0 = base[0] + rng.normal(0, noise_scale, size=base[0].shape)
    w1 = base[1] + rng.normal(0, noise_scale, size=base[1].shape)
    return [w0.astype(np.float32), w1.astype(np.float32)]


def run_sweep(out_csv: str = 'farpa_sweep_results.csv'):
    rng = np.random.RandomState(123)
    base = [np.zeros((8, 8), dtype=np.float32), np.zeros((16,), dtype=np.float32)]
    n_clients = 10
    local_weights = [make_client(base, noise_scale=0.01, rng=rng) for _ in range(n_clients)]
    # inject two malicious clients
    local_weights[0][0] += 5.0
    local_weights[1][0] += -4.0

    sketch_dims = [16, 32, 64]
    eps_list = [0.1, 0.5, 1.0, 5.0]
    z_thresh_list = [1.5, 2.5, 3.5]

    fieldnames = ['sketch_dim_per_layer', 'eps_sketch', 'z_thresh', 'benign_count', 'trust_mean', 'trust_std', 'trust_median', 'trust_malicious0', 'trust_malicious1']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sketch_dim, eps_sketch, z_thresh in itertools.product(sketch_dims, eps_list, z_thresh_list):
            agg, trust, debug = farpa_aggregate(
                local_weights,
                sketch_dim_per_layer=sketch_dim,
                n_sketches=2,
                eps_sketch=eps_sketch,
                sketch_noise_mech='laplace',
                z_thresh=z_thresh,
                fallback='trimmed_mean',
                seed=42,
            )
            trust_arr = np.asarray(trust, dtype=np.float64)
            row = {
                'sketch_dim_per_layer': sketch_dim,
                'eps_sketch': eps_sketch,
                'z_thresh': z_thresh,
                'benign_count': int(debug.get('benign_count', -1)),
                'trust_mean': float(np.mean(trust_arr)),
                'trust_std': float(np.std(trust_arr)),
                'trust_median': float(np.median(trust_arr)),
                'trust_malicious0': float(trust_arr[0]),
                'trust_malicious1': float(trust_arr[1]),
            }
            writer.writerow(row)
            print(f"Wrote row: dim={sketch_dim} eps={eps_sketch} z={z_thresh} benign={row['benign_count']}")


if __name__ == '__main__':
    out = os.environ.get('FARPA_SWEEP_OUT', 'farpa_sweep_results.csv')
    run_sweep(out)
