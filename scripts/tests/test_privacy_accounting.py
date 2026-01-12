import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from privacy_accounting import ledger


def test_privacy_ledger_basic_and_advanced():
    ledger.clear()
    ledger.record(0.5, 1e-5)
    ledger.record(0.25, 1e-6)
    eps_basic, delta_basic = ledger.basic_composition()
    assert pytest.approx(eps_basic, rel=1e-6) == 0.75
    assert delta_basic == pytest.approx(1e-5 + 1e-6)

    eps_adv, delta_adv = ledger.advanced_composition(delta_prime=1e-6)
    assert eps_adv >= eps_basic
    assert delta_adv >= delta_basic


if __name__ == '__main__':
    pytest.main([__file__])
