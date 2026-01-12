import tempfile
import os
import json
import sys
import pathlib

# Ensure repository root is on sys.path so `scripts` can be imported during pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.aggregate_rdp_smokegrid import aggregate_rdp_smokegrid


def make_sample(outdir):
    os.makedirs(outdir, exist_ok=True)
    samples = [
        {'record': {'mechanism': 'gaussian', 'sigma': 1.0, 'sampling_rate': 0.01, 'steps': 10}, 'target_delta': 1e-6, 'epsilon': 0.5, 'optimal_order': 10},
        {'record': {'mechanism': 'gaussian', 'sigma': 0.8, 'sampling_rate': 0.02, 'steps': 20}, 'target_delta': 1e-6, 'epsilon': 1.0, 'optimal_order': 5},
    ]
    for i, s in enumerate(samples):
        with open(os.path.join(outdir, f'metadata_rdp_{i:02d}.json'), 'w', encoding='utf-8') as f:
            json.dump(s, f)


def test_aggregate_creates_outputs(tmp_path):
    outdir = tmp_path / 'rdp_smoke_outputs'
    outfig = tmp_path / 'paper_figures'
    make_sample(str(outdir))
    # run aggregation
    out_path = aggregate_rdp_smokegrid(outdir=str(outdir), outfig=str(outfig))
    assert os.path.exists(out_path)
    agg_file = os.path.join(str(outfig), 'rdp_smokegrid_demo.json')
    assert os.path.exists(agg_file)
    with open(agg_file, 'r', encoding='utf-8') as f:
        agg = json.load(f)
    assert agg['count'] == 2
    assert 'eps_mean' in agg and agg['eps_mean'] == (0.5 + 1.0) / 2
    demo_path = os.path.join(str(outfig), 'rdp_demo_result.json')
    assert os.path.exists(demo_path)
    with open(demo_path, 'r', encoding='utf-8') as f:
        demo = json.load(f)
    assert 'eps' in demo
