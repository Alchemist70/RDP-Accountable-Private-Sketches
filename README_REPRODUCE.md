# Reproducibility README

This file documents minimal steps to reproduce key experiments and package artifacts for submission.

Prerequisites
- Create a Python environment with the packages listed in `requirements.txt` or `environment.yml`.

Quick test run (small representative grid)
```powershell
$env:PYTHONPATH='.'; conda activate tfpriv; python -u scripts\run_apra_mnist.py --rounds 2 --local-epochs 1 --clients 4 --outdir tmp_apra_test --sketch-dims 64 --n-sketches 1 --z-thresh 2.0 --sketch-noise-mech gaussian
```

This will run a tiny grid and write outputs to `tmp_apra_test/` including per-aggregator subdirectories and `metadata.json` files describing composed privacy (ε, δ).

Generate publication plots (example)
```powershell
python -u scripts\plot_farpa_sweep.py
python -u scripts\plot_krum_sweep.py --summary results/krum_poisoning_summary.csv --detailed results/krum_poisoning_detailed.csv
```

Package artifacts
```powershell
python -u scripts\export_artifacts.py --out submission_artifacts.zip --dirs apra_mnist_runs_full tmp_apra_test --patterns "*.csv" "*.svg" "*.npz" "*.json"
```

Notes
- Each experiment `run_dir` will include a `metadata.json` file with `git_sha`, `python_version`, CLI args, and composed privacy estimates.
- For formal privacy guarantees replace the simple RDP approximation with a full RDP accountant (e.g., `tensorflow_privacy` or `opacus`) and re-run the experiments.
 - For formal privacy guarantees use the `tfpriv` conda env. Example:

```powershell
conda create -n tfpriv python=3.11 -y
conda activate tfpriv
pip install -r requirements.txt
pip install tensorflow-privacy
```

Then run experiments with `--sketch-noise-mech gaussian` under `tfpriv` so ledger can compute formal RDP via `tensorflow_privacy`.
