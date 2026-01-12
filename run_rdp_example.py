"""run_rdp_example.py

Example script to compute RDP composition using tensorflow-privacy's rdp_accountant.
Run inside your `tfpriv` conda environment:

(C:/Users/rravi/miniconda3/shell/condabin/conda-hook.ps1) ; (conda activate tfpriv) ; python run_rdp_example.py

The script prints (epsilon, delta) for a target delta and example records.
"""
import json
import numpy as np
import sys

try:
    # Preferred import (some tf-privacy versions expose rdp_accountant here)
    from tensorflow_privacy.privacy.analysis import rdp_accountant
    _rdp_accountant_available = True
except Exception:
    # rdp_accountant not available in this tf-privacy installation; fall back later
    print('Warning: tensorflow-privacy rdp_accountant not available in this Python env; will try fallback accounting methods.')
    rdp_accountant = None
    _rdp_accountant_available = False


def main():
    # Example records: list of Gaussian sampling mechanisms
    records = [
        { 'mechanism': 'gaussian', 'sampling_rate': 0.01, 'sigma': 1.0, 'steps': 100 },
        { 'mechanism': 'gaussian', 'sampling_rate': 0.02, 'sigma': 0.8, 'steps': 200 },
    ]

    # Orders to evaluate (typical grid)
    orders = np.concatenate([np.arange(2, 64, 0.5), np.arange(64, 512, 1.0)])

    total_rdp = np.zeros_like(orders, dtype=float)

    for rec in records:
        mech = rec.get('mechanism', 'gaussian').lower()
        if mech != 'gaussian':
            print(f"Skipping non-gaussian mechanism: {mech}")
            continue
        q = float(rec.get('sampling_rate', 1.0))
        sigma = float(rec['sigma'])
        steps = int(rec.get('steps', 1))

        # compute_rdp(q, noise_multiplier, steps, orders)
        per_order_rdp = rdp_accountant.compute_rdp(q, sigma, 1, orders)
        # multiply per-step RDP by steps
        per_order_rdp = np.array(per_order_rdp, dtype=float) * float(steps)
        total_rdp += per_order_rdp

    target_delta = 1e-6
    try:
        eps, opt_order = rdp_accountant.get_privacy_spent(orders, total_rdp, target_delta)
    except Exception as e:
        print("Error computing privacy spent:", e)
        sys.exit(3)

    print("Records:")
    print(json.dumps(records, indent=2))
    print(f"Target delta = {target_delta:e}")
    print(f"Composed (epsilon, delta) = ({eps:.6g}, {target_delta:e}), optimal order = {opt_order}")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print('Primary RDP-accountant flow failed; attempting fallback using compute_dp_sgd_privacy if available:', e)
        try:
            from tensorflow_privacy import compute_dp_sgd_privacy
            # Example parameters for a fallback DP-SGD epsilon computation
            n = 60000
            batch_size = 256
            noise_multiplier = 1.0
            epochs = 10
            delta = 1e-6
            eps, opt = compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta)
            print(f"Fallback DP-SGD (n={n}, batch={batch_size}, noise={noise_multiplier}, epochs={epochs}, delta={delta}) -> eps={eps:.6g}, opt_order={opt}")
        except Exception as e2:
            print('Fallback also failed or is not available:', e2)
            sys.exit(4)
