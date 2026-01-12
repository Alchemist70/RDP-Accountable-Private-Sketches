import pytest
import math
import os
import sys

# ensure repo root is on sys.path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from privacy_accounting import PrivacyLedger, ledger


def test_ledger_compute_rdp_with_tf_privacy():
    try:
        import importlib
        importlib.import_module('tensorflow_privacy.privacy.analysis.rdp_accountant')
    except Exception:
        pytest.skip('tensorflow_privacy not installed; skipping tf-privacy integration test')

    # Create a fresh ledger
    lg = PrivacyLedger()
    # Record two gaussian mechanism groups with different sigmas/q
    lg.record_mechanism('gaussian', sigma=1.0, sampling_rate=1.0, steps=2)
    lg.record_mechanism('gaussian', sigma=2.0, sampling_rate=1.0, steps=1)

    eps, delta = lg.compute_rdp_via_tensorflow_privacy(target_delta=1e-6)
    assert isinstance(eps, float)
    assert math.isfinite(eps)
    assert float(delta) == pytest.approx(1e-6)

    # ensure ledger instance (module-level) untouched
    assert hasattr(ledger, 'record')
