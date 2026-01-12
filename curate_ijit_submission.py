#!/usr/bin/env python3
"""
Curate essential files for IJIT submission into a clean ijit_submission/ folder.

This script copies only the files required for journal submission, organizing them logically:
- Manuscript (PDF, TeX, BibTeX)
- Figures (PNG/PDF)
- Reproducibility documentation
- Core code (orchestrator, algorithms, accounting)
- Requirements (environment files)

Run: python curate_ijit_submission.py
"""

import os
import shutil
from pathlib import Path

# Define source and destination
WORKSPACE_ROOT = Path(__file__).parent
SUBMISSION_PACKAGE = WORKSPACE_ROOT / "submission_package"
IJIT_BIN = WORKSPACE_ROOT / "ijit_bin"
OUTPUT_DIR = WORKSPACE_ROOT / "ijit_submission"

# Essential files mapping: source_path -> destination filename (all in root)
# Naming convention: [category]_filename to keep them organized in flat structure
ESSENTIAL_FILES = {
    # Manuscript
    "submission_package/manuscript.pdf": "manuscript.pdf",
    "submission_package/manuscript.tex": "manuscript.tex",
    "submission_package/ijit.bib": "ijit.bib",
    "submission_package/titlepage.pdf": "titlepage.pdf",
    "submission_package/svjour3.cls": "svjour3.cls",
    
    # Cover letter
    "ijit_bin/COVER_LETTER.txt": "COVER_LETTER.txt",
    
    # Reproducibility documentation
    "submission_package/REPRODUCIBILITY.md": "REPRODUCIBILITY.md",
    "submission_package/README_REPRODUCE.md": "README_REPRODUCE.md",
    
    # Requirements and environment
    "submission_package/requirements.txt": "requirements.txt",
    "submission_package/environment.yml": "environment.yml",
    
    # Figures
    "submission_package/detection_roc_curves.png": "fig_detection_roc_curves.png",
    "submission_package/privacy_utility_tradeoff.png": "fig_privacy_utility_tradeoff.png",
    "submission_package/convergence_by_grid.png": "fig_convergence_by_grid.png",
    "submission_package/convergence_detailed_by_attack.png": "fig_convergence_detailed_by_attack.png",
    "submission_package/multi_attack_comparison.png": "fig_multi_attack_comparison.png",
    "submission_package/communication_vs_accuracy.png": "fig_communication_vs_accuracy.png",
    "submission_package/rdp_composition.png": "fig_rdp_composition.png",
    "submission_package/ablation_sketch_dimension.png": "fig_ablation_sketch_dimension.png",
    "submission_package/privacy_utility_tradeoff_eps.png": "fig_privacy_utility_tradeoff_eps.png",
    
    # LaTeX tables (TeX files)
    "submission_package/farpa_summary_table.tex": "table_farpa_summary.tex",
    "submission_package/comm_runtime_table.tex": "table_comm_runtime.tex",
    "submission_package/rdp_table.tex": "table_rdp.tex",
    
    # Core reproducibility code
    "submission_package/notebooks/apra_orchestrator.ipynb": "code_apra_orchestrator.ipynb",
    "submission_package/scripts/run_apra_mnist_full.py": "code_run_apra_mnist_full.py",
    "submission_package/scripts/apra.py": "code_apra.py",
    "submission_package/scripts/aps_plus_constrained.py": "code_aps_plus_constrained.py",
    "submission_package/privacy_accounting.py": "code_privacy_accounting.py",
    "submission_package/scripts/eval_all_grids_shadows.py": "code_eval_all_grids_shadows.py",
    "submission_package/scripts/analyze_and_plot.py": "code_analyze_and_plot.py",
    "submission_package/scripts/summarize_apra_results.py": "code_summarize_apra_results.py",
}

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def copy_file(src, dst_filename):
    """Copy file from src to ijit_submission root with new filename."""
    src_path = WORKSPACE_ROOT / src
    if not src_path.exists():
        print(f"  ✗ SOURCE NOT FOUND: {src}")
        return False
    
    dst_path = OUTPUT_DIR / dst_filename
    ensure_dir(dst_path.parent)
    
    try:
        shutil.copy2(src_path, dst_path)
        size_kb = src_path.stat().st_size / 1024
        print(f"  ✓ {dst_filename:<50s} ({size_kb:>7.1f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ ERROR copying {src}: {e}")
        return False

def main():
    print("="*80)
    print("IJIT SUBMISSION CURATION")
    print("="*80)
    
    # Clean and create output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    ensure_dir(OUTPUT_DIR)
    print(f"Created: {OUTPUT_DIR}\n")
    
    # Copy files
    copied = 0
    failed = 0
    
    for src, dst_filename in sorted(ESSENTIAL_FILES.items()):
        if copy_file(src, dst_filename):
            copied += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: {copied} files copied, {failed} errors")
    print("="*80)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    total_mb = total_size / (1024 * 1024)
    
    print(f"\nTotal submission size: {total_mb:.1f} MB")
    print(f"Location: {OUTPUT_DIR}")
    
    # Print directory structure
    print("\nFiles in ijit_submission/ (flat structure):")
    files = sorted([f for f in OUTPUT_DIR.iterdir() if f.is_file()])
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  ├─ {f.name:<50s} ({size_kb:>7.1f} KB)")
    
    # Print submission checklist
    print("\n" + "="*80)
    print("SUBMISSION CHECKLIST")
    print("="*80)
    print("\nSubmit all these files individually to IJIT (flat directory):\n")
    
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}. □ {f.name}")
    
    print("\n" + "="*80)
    print("✓ CURATION COMPLETE")
    print("="*80)

def print_tree(path, prefix="", indent=""):
    """Print flat file list."""
    entries = sorted([f for f in path.iterdir() if f.is_file()])
    for f in entries:
        size_kb = f.stat().st_size / 1024
        print(f"{indent}{prefix}{f.name:<50s} ({size_kb:>7.1f} KB)")

if __name__ == "__main__":
    main()
