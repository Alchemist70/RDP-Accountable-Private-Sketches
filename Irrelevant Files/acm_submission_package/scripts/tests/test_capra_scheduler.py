"""Unit test for CAPRA scheduler/fallback behavior.

This test runs a very short APRA experiment configured to use CAPRA with a tiny
time budget so CAPRA will select fast mode and schedule recomputation. The test
verifies that a recompute meta file is created after the experiment.
"""
import os
import sys
import tempfile
import shutil
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.run_apra_mnist_full import APRAMNISTExperiment


def test_capra_scheduler_creates_recompute_meta():
    outdir = os.path.join(tempfile.gettempdir(), f'apra_test_capra_{int(time.time())}')
    try:
        exp = APRAMNISTExperiment(
            sketch_dim=64,
            n_sketches=2,
            z_thresh=3.0,
            rounds=3,
            clients=8,
            local_epochs=1,
            batch_size=8,
            attack='none',
            byzantine_fraction=0.0,
            output_dir=outdir,
            seed=1,
            agg_method='capra',
            capra_time_budget_ms=0.00001,
            capra_fast_dim=8,
            capra_fast_n_sketches=1,
            run_tag='testcapra'
        )

        # Force CAPRA to be aggressive (very small budget) so fast mode selected
        exp.run()

        # check for recomputed meta files in CAPRA aggregator dir
        agg_dir = exp.agg_dirs.get('capra') if hasattr(exp, 'agg_dirs') else os.path.join(outdir, f'sd{exp.sketch_dim}_ns{exp.n_sketches}_zt{exp.z_thresh}', 'capra')
        found = False
        if os.path.isdir(agg_dir):
            for fname in os.listdir(agg_dir):
                if fname.endswith('_recomputed_meta.json'):
                    found = True
                    break

        assert found, f'No recomputed meta file found in {agg_dir}'
        print('CAPRA scheduler test: recompute meta found')

    finally:
        try:
            if os.path.exists(outdir):
                shutil.rmtree(outdir)
        except Exception:
            pass


if __name__ == '__main__':
    test_capra_scheduler_creates_recompute_meta()
    print('test_capra_scheduler passed')
