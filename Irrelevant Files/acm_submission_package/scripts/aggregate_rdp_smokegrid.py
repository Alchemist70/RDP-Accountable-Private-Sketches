"""
Aggregate per-run RDP metadata in `rdp_smoke_outputs` into a consolidated JSON under `paper_figures`.
"""
import os
import json
import glob
import statistics

def aggregate_rdp_smokegrid(indir: str = 'rdp_smoke_outputs', outfig: str = 'paper_figures') -> str:
    """Aggregate per-run RDP JSONs from `outdir` and write consolidated outputs into `outfig`.

    Returns the path to the aggregated JSON written.
    """
    os.makedirs(outfig, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(indir, 'metadata_rdp_*.json')))
    entries = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                entries.append(obj)
        except Exception as e:
            print('Failed to read', p, e)

    if not entries:
        raise SystemExit(f'No entries found in {indir}')

    eps_vals = [e.get('epsilon') for e in entries if e.get('epsilon') is not None]
    summary = {
        'count': len(entries),
        'eps_min': min(eps_vals) if eps_vals else None,
        'eps_max': max(eps_vals) if eps_vals else None,
        'eps_mean': statistics.mean(eps_vals) if eps_vals else None,
        'eps_median': statistics.median(eps_vals) if eps_vals else None,
        'entries': entries,
    }

    out_path = os.path.join(outfig, 'rdp_smokegrid_demo.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Wrote', out_path)

    # Also update rdp_demo_result.json with the median entry (closest to median epsilon)
    median = summary['eps_median']
    closest = None
    best_diff = None
    for e in entries:
        eps = e.get('epsilon')
        if eps is None:
            continue
        diff = abs(eps - median)
        if closest is None or diff < best_diff:
            closest = e
            best_diff = diff
    if closest is not None:
        demo_path = os.path.join(outfig, 'rdp_demo_result.json')
        demo_obj = {
            'eps': closest.get('epsilon'),
            'delta': closest.get('target_delta'),
            'optimal_order': closest.get('optimal_order'),
            'record': closest.get('record'),
        }
        with open(demo_path, 'w', encoding='utf-8') as f:
            json.dump(demo_obj, f, indent=2)
        print('Updated', demo_path)
    else:
        print('No suitable demo entry to update')

    return out_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--indir', type=str, default='rdp_smoke_outputs', help='Input directory with per-run metadata JSONs')
    p.add_argument('--outdir', type=str, default='paper_figures', help='Output directory for aggregated JSON')
    args = p.parse_args()
    aggregate_rdp_smokegrid(indir=args.indir, outfig=args.outdir)
