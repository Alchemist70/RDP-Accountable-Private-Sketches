"""run_rdp_dp_accounting.py

Compute composed RDP -> (epsilon, delta) using the installed `dp_accounting` library.
This script composes arbitrary Gaussian sampling records (sampling_rate, sigma, steps)
and prints the resulting (epsilon, delta).

Run inside `tfpriv`:
  (C:/Users/rravi/miniconda3/shell/condabin/conda-hook.ps1) ; (conda activate tfpriv) ; python run_rdp_dp_accounting.py
"""
import json
import sys

try:
    import dp_accounting.dp_event as dp_event
    from dp_accounting.dp_event_builder import DpEventBuilder
    from dp_accounting.rdp import RdpAccountant
except Exception as e:
    print('ERROR: dp_accounting imports failed:', e)
    sys.exit(2)


def compose_records_and_get_eps(records, target_delta=1e-6, orders=None):
    # Create an RDP accountant
    acct = RdpAccountant(orders=orders)

    for rec in records:
        mech = rec.get('mechanism', 'gaussian').lower()
        if mech != 'gaussian':
            print('Skipping non-gaussian mechanism:', mech)
            continue
        q = float(rec.get('sampling_rate', 1.0))
        sigma = float(rec['sigma'])
        steps = int(rec.get('steps', 1))

        # Basic event: Gaussian mechanism with given noise multiplier
        g = dp_event.GaussianDpEvent(noise_multiplier=sigma)
        if q < 1.0:
            # model as Poisson-sampled Gaussian event
            ev = dp_event.PoissonSampledDpEvent(sampling_probability=q, event=g)
        else:
            ev = g

        acct.compose(ev, count=steps)

    eps, opt_order = acct.get_epsilon_and_optimal_order(target_delta)
    return float(eps), float(opt_order)


def main():
    # Example records (same as earlier run_rdp_example)
    records = [
        { 'mechanism': 'gaussian', 'sampling_rate': 0.01, 'sigma': 1.0, 'steps': 100 },
        { 'mechanism': 'gaussian', 'sampling_rate': 0.02, 'sigma': 0.8, 'steps': 200 },
    ]

    target_delta = 1e-6
    eps, opt_order = compose_records_and_get_eps(records, target_delta=target_delta)

    print('Records:')
    print(json.dumps(records, indent=2))
    print(f'Target delta = {target_delta:e}')
    print(f'Composed (epsilon, delta) = ({eps:.6g}, {target_delta:e}), optimal order = {opt_order}')


if __name__ == '__main__':
    main()
