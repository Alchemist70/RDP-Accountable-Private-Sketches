"""
Plotting helper to visualize krum poisoning sweep results.

Generates:
- `results/krum_poisoning_summary.png`: mean ± std final accuracy vs byzantine fraction.
- `results/krum_poisoning_curves.png`: per-round average learning curves (APRA vs Krum) per byz.

Usage:
    python scripts/plot_krum_sweep.py --summary results/krum_poisoning_summary.csv --detailed results/krum_poisoning_detailed.csv
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict


def load_summary(path):
    data = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            data.append({
                'seed': int(r['seed']),
                'byz': float(r['byzantine_fraction']),
                'scale': float(r['scale_factor']),
                'apra_final': float(r['apra_final']),
                'krum_final': float(r['krum_final'])
            })
    return data


def load_detailed(path):
    # returns dict[(byz, scale, method)] -> list of lists (per-seed rounds)
    rows = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    perkey = {}
    for r in rows:
        seed = int(float(r['seed']))
        byz = float(r['byzantine_fraction'])
        scale = float(r['scale_factor'])
        rnd = int(r['round'])
        method = r['method']
        acc = float(r['acc'])
        k = (byz, scale, method)
        if k not in perkey:
            perkey[k] = defaultdict(list)
        perkey[k][seed].append((rnd, acc))

    # normalize per-seed lists by sorting by round
    out = {}
    for k, seedmap in perkey.items():
        lists = []
        for seed, pairs in seedmap.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            lists.append([a for (_, a) in pairs_sorted])
        out[k] = lists
    return out


def plot_summary(summary, outpath='results/krum_poisoning_summary.png'):
    # compute mean/std by byz for each method
    byz_vals = sorted(list({row['byz'] for row in summary}))
    apra_means = []
    apra_stds = []
    krum_means = []
    krum_stds = []
    for b in byz_vals:
        ap = [r['apra_final'] for r in summary if r['byz'] == b]
        kr = [r['krum_final'] for r in summary if r['byz'] == b]
        apra_means.append(np.mean(ap) if ap else np.nan)
        apra_stds.append(np.std(ap) if ap else np.nan)
        krum_means.append(np.mean(kr) if kr else np.nan)
        krum_stds.append(np.std(kr) if kr else np.nan)

    plt.figure()
    plt.errorbar(byz_vals, apra_means, yerr=apra_stds, label='APRA (CAPRA)', marker='o')
    plt.errorbar(byz_vals, krum_means, yerr=krum_stds, label='Krum', marker='o')
    plt.xlabel('Byzantine fraction')
    plt.ylabel('Final accuracy (mean ± std)')
    plt.title('Final accuracy vs Byzantine fraction')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # save PNG and vector formats
    plt.savefig(outpath)
    try:
        svg_path = os.path.splitext(outpath)[0] + '.svg'
        pdf_path = os.path.splitext(outpath)[0] + '.pdf'
        plt.savefig(svg_path)
        plt.savefig(pdf_path)
    except Exception:
        pass
    plt.close()


def plot_curves(detailed, outpath='results/krum_poisoning_curves.png'):
    # For each byz value, plot mean per-round curve for both methods (averaged across seeds and scales)
    byz_map = defaultdict(lambda: defaultdict(list))
    for (byz, scale, method), lists in detailed.items():
        for l in lists:
            byz_map[byz][method].append(l)

    n_byz = len(sorted(byz_map.keys()))
    fig, axes = plt.subplots(n_byz, 1, figsize=(6, 3 * max(1, n_byz)), squeeze=False)
    for idx, b in enumerate(sorted(byz_map.keys())):
        ax = axes[idx, 0]
        for method, lists in byz_map[b].items():
            maxlen = max(len(l) for l in lists)
            arr = np.array([l + [np.nan] * (maxlen - len(l)) for l in lists])
            mean_curve = np.nanmean(arr, axis=0)
            std_curve = np.nanstd(arr, axis=0)
            rounds = np.arange(1, len(mean_curve) + 1)
            ax.plot(rounds, mean_curve, label=method)
            ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        ax.set_title(f'Byz={b}')
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # save PNG and vector formats
    plt.savefig(outpath)
    try:
        svg_path = os.path.splitext(outpath)[0] + '.svg'
        pdf_path = os.path.splitext(outpath)[0] + '.pdf'
        plt.savefig(svg_path)
        plt.savefig(pdf_path)
    except Exception:
        pass
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=str, default='results/krum_poisoning_summary.csv')
    parser.add_argument('--detailed', type=str, default='results/krum_poisoning_detailed.csv')
    parser.add_argument('--out_summary_png', type=str, default='results/krum_poisoning_summary.png')
    parser.add_argument('--out_curves_png', type=str, default='results/krum_poisoning_curves.png')
    args = parser.parse_args()

    summary = load_summary(args.summary)
    detailed = load_detailed(args.detailed)
    plot_summary(summary, outpath=args.out_summary_png)
    plot_curves(detailed, outpath=args.out_curves_png)
    # Vector formats were saved alongside PNGs in the plot functions
    print(f"Saved plots: {args.out_summary_png}, {args.out_curves_png}")


if __name__ == '__main__':
    main()
