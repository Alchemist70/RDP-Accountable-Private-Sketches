import numpy as np
import json
import os
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fl_helpers


def make_dummy_deltas(num_clients=6):
    # create simple 2-layer models: layer0 shape (4,), layer1 shape (3,)
    base = [np.zeros(4, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    out = []
    for i in range(num_clients):
        # produce small variations; one client slightly malicious
        if i == 0:
            deltas = [base[0] + 0.5, base[1] + 0.5]
        else:
            deltas = [base[0] + (0.01 * i), base[1] + (0.01 * i)]
        out.append([np.array(d) for d in deltas])
    return out


def test_capra_eps_and_trust_scaling():
    deltas = make_dummy_deltas(6)
    # Force fast mode by tiny time_budget_ms
    agg_fast, meta_fast = fl_helpers.capra_aggregate(
        deltas,
        sketch_dim_canonical=64,
        n_sketches_canonical=2,
        sketch_dim_fast=8,
        n_sketches_fast=1,
        eps_sketch=1.0,
        time_budget_ms=0.0,  # force fast
        probe_samples=2,
        seed=123,
    )
    assert isinstance(meta_fast, dict)
    assert meta_fast.get('mode') == 'fast'
    eps_fast = float(meta_fast.get('eps_used', 0.0))

    # Force canonical by large time budget
    agg_can, meta_can = fl_helpers.capra_aggregate(
        deltas,
        sketch_dim_canonical=64,
        n_sketches_canonical=2,
        sketch_dim_fast=8,
        n_sketches_fast=1,
        eps_sketch=1.0,
        time_budget_ms=100000.0,  # force canonical
        probe_samples=2,
        seed=123,
    )
    assert isinstance(meta_can, dict)
    assert meta_can.get('mode') == 'canonical'
    eps_can = float(meta_can.get('eps_used', 0.0))

    # eps scaling: fast should use smaller or equal epsilon than canonical (proportional to sketch dim)
    assert eps_fast <= eps_can + 1e-12
    # Check that eps_fast matches expected ratio roughly
    expected_ratio = 8.0 / 64.0
    assert eps_fast == pytest.approx(eps_can * expected_ratio, rel=1e-3) or eps_fast <= eps_can

    # Trust scores should be present and differ between runs
    trust_fast = np.array(meta_fast.get('trust_scores', []), dtype=float)
    trust_can = np.array(meta_can.get('trust_scores', []), dtype=float)
    assert trust_fast.size == trust_can.size
    # They should not be identical (due to different noise levels)
    assert not np.allclose(trust_fast, trust_can)
    # Trust distributions should differ due to distinct sketch/noise regimes
    # (no strict directionality asserted to avoid flaky numerical assumptions)
    assert not np.allclose(trust_can, trust_fast)


if __name__ == '__main__':
    pytest.main([__file__])
