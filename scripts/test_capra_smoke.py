"""Quick CAPRA smoke test.

Runs a synthetic FARPA/CAPRA aggregation on small synthetic weights to verify switching.
"""
import os
import sys
import numpy as np
# Ensure repo root is on path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fl_helpers import aggregate_dispatcher


def make_synthetic_weights(num_clients=8, layer_shapes=[(16,), (32,)]):
    local = []
    for i in range(num_clients):
        client = []
        for s in layer_shapes:
            # benign clients near zero, malicious large offsets for last two clients
            if i >= num_clients - 2:
                client.append(np.ones(s, dtype=np.float32) * 5.0)
            else:
                client.append(np.random.normal(0.0, 0.5, size=s).astype(np.float32))
        local.append(client)
    return local


def run_smoke():
    local = make_synthetic_weights(num_clients=10)
    agg, meta = aggregate_dispatcher(
        'capra',
        local,
        sketch_dim_canonical=64,
        n_sketches_canonical=2,
        sketch_dim_fast=8,
        n_sketches_fast=1,
        eps_sketch=1.0,
        time_budget_ms=50.0,
        probe_samples=3,
        seed=1,
    )
    print('CAPRA mode:', meta.get('mode'))
    print('Probe info:', meta.get('probe'))
    print('Eps used:', meta.get('eps_used'))
    print('Trust scores (first 8):', np.array(meta.get('trust_scores')[:8]))


if __name__ == '__main__':
    run_smoke()
