"""Smoke test for FARPA aggregator.
Creates a small synthetic federated update set with a couple of strong outliers
and compares FARPA's output and trust scores to a baseline aggregator.
"""
import os
import sys
import numpy as np

# Ensure repo root is on sys.path so imports like `from fl_helpers import ...` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl_helpers import farpa_aggregate

try:
    from fl_helpers import apra_aggregate
    HAVE_APRA = True
except Exception:
    HAVE_APRA = False


def make_client(base, noise_scale=0.01, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    w0 = base[0] + rng.normal(0, noise_scale, size=base[0].shape)
    w1 = base[1] + rng.normal(0, noise_scale, size=base[1].shape)
    return [w0.astype(np.float32), w1.astype(np.float32)]


def main():
    rng = np.random.RandomState(123)
    base = [np.zeros((8, 8), dtype=np.float32), np.zeros((16,), dtype=np.float32)]
    n_clients = 10
    local_weights = [make_client(base, noise_scale=0.01, rng=rng) for _ in range(n_clients)]

    # Inject two malicious clients with large offsets
    local_weights[0][0] += 5.0
    local_weights[1][0] += -4.0

    print("Running FARPA smoke test with {} clients (2 malicious).".format(n_clients))

    configs = [
        {'sketch_dim_per_layer': 16, 'eps_sketch': 0.5, 'label': 'conservative'},
        {'sketch_dim_per_layer': 64, 'eps_sketch': 5.0, 'label': 'tuned'},
    ]

    for cfg in configs:
        print(f"\nRunning FARPA config: {cfg['label']} (dim={cfg['sketch_dim_per_layer']}, eps={cfg['eps_sketch']})")
        agg_farpa, trust_scores, debug = farpa_aggregate(
            local_weights,
            sketch_dim_per_layer=cfg['sketch_dim_per_layer'],
            n_sketches=2,
            eps_sketch=cfg['eps_sketch'],
            sketch_noise_mech='laplace',
            fallback='trimmed_mean',
            seed=42,
        )

        print("FARPA debug:")
        for k, v in debug.items():
            print(f"  {k}: {v}")
        print("Trust scores (first 8):", np.round(trust_scores[:8], 3))
        # show trust scores for injected malicious clients (indices 0 and 1)
        print("Trust[malicious 0] =", round(float(trust_scores[0]), 4))
        print("Trust[malicious 1] =", round(float(trust_scores[1]), 4))

    # Compute naive mean for comparison
    stacked0 = np.stack([w[0].astype(np.float64) for w in local_weights], axis=0)
    naive_mean0 = np.mean(stacked0, axis=0)

    farpa_layer0 = np.asarray(agg_farpa[0], dtype=np.float64)
    diff_norm = np.linalg.norm(farpa_layer0 - naive_mean0)
    print(f"L2 norm difference between FARPA and naive mean (layer0): {diff_norm:.4f}")

    if HAVE_APRA:
        try:
            # apra_aggregate signature: sketch_dim, trim_fraction, z_thresh, seed
            agg_apra = apra_aggregate(local_weights, sketch_dim=64, z_thresh=3.0, seed=42)
            apra_layer0 = np.asarray(agg_apra[0], dtype=np.float64)
            diff_apra = np.linalg.norm(apra_layer0 - naive_mean0)
            print(f"L2 norm difference between APRA and naive mean (layer0): {diff_apra:.4f}")
        except Exception as e:
            print("APRA run failed:", e)

    print("Done.")


if __name__ == '__main__':
    main()
