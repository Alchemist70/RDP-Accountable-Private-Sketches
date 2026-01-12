# IJIT Submission: Reproducibility Guide

This artifact bundle enables independent verification of PrivateSketch privacy accounting, detection performance, and deployment costs.

## Quick Start

### Prerequisites
```bash
# Create environment
conda create -n ijit-reproduce python=3.11 -y
conda activate ijit-reproduce
pip install -r requirements.txt
```

### Minimal Verification Test
Run a tiny smoke-grid to verify the RDP ledger logic:

```bash
export PYTHONPATH=.
python -u scripts/run_apra_mnist.py \
  --rounds 2 \
  --local-epochs 1 \
  --clients 4 \
  --outdir verification_test \
  --sketch-dims 64 \
  --n-sketches 1 \
  --z-thresh 2.0 \
  --sketch-noise-mech gaussian
```

**Output**: Creates `verification_test/` with per-aggregator subdirectories and `metadata.json` containing composed privacy (ε, δ).

## Reviewer Checklist for RDP Verification

Per manuscript Section 10.2 (Privacy-audit ledger and reproducibility), verify the RDP ledger independently:

1. **Load ledger JSON**
   ```python
   import json
   with open('verification_test/metadata.json') as f:
       metadata = json.load(f)
   print(metadata['composed_rdp'])
   ```

2. **Recompute RDP per order**
   ```python
   # See privacy_accounting.py
   from scripts.privacy_accounting import compute_rdp_from_ledger
   rdp_dict = compute_rdp_from_ledger(metadata['ledger'])
   ```

3. **Convert to (ε, δ)**
   ```python
   epsilon, delta = convert_rdp_to_eps_delta(rdp_dict, target_delta=1e-6)
   print(f"ε={epsilon:.3f}, δ={delta:.2e}")
   ```

4. **Verify smoke-grid matches recomputation**
   - Compare reported ε, δ in Tables 4-5 (manuscript) against recomputed values
   - Generate visualization: `python -u scripts/plot_rdp_multi.py --input verification_test/`

## Full Experiment Suite

For the complete smoke-grid shown in Section 6:

```bash
# This requires 2-4 hours depending on hardware
python -u scripts/run_apra_mnist.py \
  --rounds 20 \
  --local-epochs 5 \
  --clients 100 \
  --outdir full_smoke_grid \
  --sketch-dims 64 128 256 \
  --z-thresh 1.5 2.0 2.5 \
  --sketch-noise-mech gaussian \
  --n-sketches 3
```

## Key Files in This Bundle

```
submission_package/
├── manuscript.pdf              # IJIT submission paper
├── manuscript.tex              # LaTeX source
├── svjour3.cls                 # Springer document class
├── ijit.bib                    # Bibliography
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment spec
├── REPRODUCIBILITY.md          # This file
├── scripts/
│   ├── run_apra_mnist.py       # Main experiment harness
│   ├── privacy_accounting.py   # RDP computation engine
│   ├── aps_plus_constrained.py # APS+ optimizer
│   ├── plot_rdp_multi.py       # Smoke-grid visualization
│   └── merge_rdp_aggregates.py # Aggregate RDP traces
└── [figures, tables, supporting TeX files]
```

## Privacy Accounting Details

The RDP ledger (stored in `metadata.json`) contains:
- Per-round tuples: `(sample_rate, sigma, num_steps, mechanism)`
- Composed RDP values per order α ∈ {1.5, 2, 4, 8, 16, 32, 64}
- Noise mechanism (Gaussian or Laplace)

The smoke-grid plots (Figures 5-7, Appendix) are generated from these ledgers and allow independent verification without running new experiments.

## Artifact Verification Certificate

After running the verification test, you can confirm:
- [ ] `verification_test/metadata.json` exists and is parseable
- [ ] RDP values for α ∈ {2, 4, 8} are positive and finite
- [ ] Converted (ε, δ) match Table 4-5 within 0.01
- [ ] No Python errors or warnings during computation
- [ ] Generated plots match Figures 5-7 style and scale

## Formal Privacy Guarantees (Optional)

For privacy-certified RDP values using TensorFlow Privacy:

```bash
# Install TensorFlow Privacy
pip install tensorflow-privacy

# Re-run experiments with formal accountant
python -u scripts/run_apra_mnist.py \
  --rounds 5 \
  --outdir formal_privacy_test \
  --use-tfpriv
```

This enables `--sketch-noise-mech gaussian-tfpriv` for certified RDP via `tensorflow_privacy.privacy.analysis.rdp_accountant`.

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Install: `pip install tensorflow` |
| `ModuleNotFoundError: No module named 'scipy'` | Install: `pip install scipy numpy` |
| Out of memory on small GPU | Reduce `--clients` to 10 or run CPU-only |
| Plots not generating | Verify `matplotlib` installed: `pip install matplotlib` |

## Contact & Support

For questions about reproducibility:
1. Check Section 10 (Deployment and Reproducibility) in the manuscript
2. Verify all files in `scripts/` and `requirements.txt` are present
3. Confirm Python 3.11+ environment is active

## Expected Runtime

| Experiment | Clients | Duration | Output Size |
|------------|---------|----------|------------|
| Verification test | 4 | ~30 sec | ~5 MB |
| Small grid | 20 | ~5 min | ~50 MB |
| Full smoke-grid | 100 | 2-4 hrs | ~500 MB |

-------------------------------------------

**Artifact Status**: Ready for independent verification  
**Last Updated**: December 11, 2025  
**Verification Checklist**: See Section 10.2 of manuscript
