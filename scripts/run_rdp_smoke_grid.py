"""
Run a small RDP smoke-grid using dp_accounting and write per-run metadata JSON files.
This script is intended to be run in the `tfpriv` conda environment where `dp_accounting` is available.
"""
import os
import json
from pprint import pprint
try:
    # provenance helper from privacy_accounting (best-effort)
    from privacy_accounting import write_provenance_metadata
except Exception:
    write_provenance_metadata = None

try:
    import dp_accounting.dp_event as dp_event
    from dp_accounting.rdp import RdpAccountant
except Exception as e:
    raise SystemExit('dp_accounting is required for this script; run in tfpriv environment. Error: ' + str(e))

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='rdp_smoke_outputs', help='Output directory for per-run RDP JSONs')
    p.add_argument('--seed', type=int, default=None, help='Optional seed/provenance field')
    return p.parse_args()


args_cli = parse_args()
OUTDIR = args_cli.outdir
os.makedirs(OUTDIR, exist_ok=True)

# Write provenance metadata for this smoke run (best-effort)
try:
    if write_provenance_metadata is not None:
        write_provenance_metadata(OUTDIR, args={'script': 'run_rdp_smoke_grid.py'}, seed=args_cli.seed, extra={'note': 'smoke-grid run'})
        print('Wrote provenance metadata to', OUTDIR)
except Exception:
    pass

# small grid of demo records to exercise composition
GRID = [
    # single-record variants
    {'mechanism': 'gaussian', 'sampling_rate': 0.01, 'sigma': 1.0, 'steps': 100},
    {'mechanism': 'gaussian', 'sampling_rate': 0.02, 'sigma': 0.8, 'steps': 200},
    # variants to probe sigma and q
    {'mechanism': 'gaussian', 'sampling_rate': 0.01, 'sigma': 0.5, 'steps': 100},
    {'mechanism': 'gaussian', 'sampling_rate': 0.05, 'sigma': 1.0, 'steps': 50},
]

TARGET_DELTA = 1e-6
ORDERS = list(__import__('numpy').concatenate([__import__('numpy').arange(2, 64, 0.5), __import__('numpy').arange(64, 512, 1.0)]))

acct = RdpAccountant(orders=ORDERS)

# We'll compose each GRID entry separately and write metadata_{i}.json
for i, rec in enumerate(GRID):
    acct = RdpAccountant(orders=ORDERS)
    mech = rec.get('mechanism', 'gaussian').lower()
    if mech != 'gaussian':
        print('Skipping non-gaussian record:', rec)
        continue
    q = float(rec.get('sampling_rate', 1.0))
    sigma = float(rec['sigma'])
    steps = int(rec.get('steps', 1))
    g = dp_event.GaussianDpEvent(noise_multiplier=sigma)
    if q < 1.0:
        ev = dp_event.PoissonSampledDpEvent(sampling_probability=q, event=g)
    else:
        ev = g
    acct.compose(ev, count=steps)
    eps, opt_order = acct.get_epsilon_and_optimal_order(TARGET_DELTA)
    out = {
        'record': rec,
        'target_delta': TARGET_DELTA,
        'epsilon': float(eps),
        'optimal_order': float(opt_order)
    }
    # Attach selected provenance fields into the per-run JSON (normalize keys)
    try:
        prov_path = os.path.join(OUTDIR, 'metadata_run.json')
        if os.path.exists(prov_path):
            with open(prov_path, 'r', encoding='utf-8') as _pf:
                prov = json.load(_pf)
            # Copy canonical provenance fields if present
            for k in ('git_sha', 'python_version', 'packages', 'args'):
                if k in prov:
                    out[k] = prov[k]
            # also include timestamp if available
            if 'timestamp' in prov:
                out['run_timestamp'] = prov['timestamp']
    except Exception:
        pass

    out_path = os.path.join(OUTDIR, f'metadata_rdp_{i:02d}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('Wrote', out_path)
    pprint(out)

print('Smoke-grid complete; outputs in', OUTDIR)
