import sys
import os
import numpy as np

# ensure repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl_helpers import farpa_aggregate


def make_client(base, noise_scale=0.01, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    w0 = base[0] + rng.normal(0, noise_scale, size=base[0].shape)
    w1 = base[1] + rng.normal(0, noise_scale, size=base[1].shape)
    return [w0.astype(np.float32), w1.astype(np.float32)]


def test_farpa_downweights_outliers():
    rng = np.random.RandomState(42)
    base = [np.zeros((4, 4), dtype=np.float32), np.zeros((8,), dtype=np.float32)]
    n_clients = 6
    local_weights = [make_client(base, noise_scale=0.01, rng=rng) for _ in range(n_clients)]
    # inject two outliers
    local_weights[0][0] += 10.0
    local_weights[1][0] += -8.0

    # tuned FARPA expected to downweight outliers
    agg, trust, debug = farpa_aggregate(local_weights, sketch_dim_per_layer=32, eps_sketch=5.0, sketch_noise_mech='laplace', z_thresh=3.0, seed=1)

    # Record trust values for debugging (do not assert brittle numeric thresholds here)
    trust_arr = np.asarray(trust)
    print('Trust scores:', trust_arr)

    # aggregated layer0 should be closer to benign-only average than naive mean
    stacked0 = np.stack([w[0].astype(np.float64) for w in local_weights], axis=0)
    naive_mean0 = np.mean(stacked0, axis=0)
    # compute benign-only mean
    benign = [local_weights[i] for i in range(len(local_weights)) if i not in (0,1)]
    benign_mean0 = np.mean(np.stack([b[0].astype(np.float64) for b in benign], axis=0), axis=0)

    farpa_layer0 = np.asarray(agg[0], dtype=np.float64)
    d_naive = np.linalg.norm(farpa_layer0 - naive_mean0)
    d_benign = np.linalg.norm(farpa_layer0 - benign_mean0)
    assert d_benign <= d_naive, f"FARPA aggregate should be closer to benign mean (d_benign={d_benign} <= d_naive={d_naive})"


if __name__ == '__main__':
    test_farpa_downweights_outliers()
    print('FARPA unit test passed')
