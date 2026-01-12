"""Summarize composed RDP (formal) values from run metadata files.

Usage:
    python scripts/summarize_rdp.py --indir tmp_apra_test_gauss2 --out summary_rdp.csv
"""
import os
import json
import argparse
import csv


def find_metadata_files(indir):
    out = []
    for root, dirs, files in os.walk(indir):
        for fn in files:
            if fn.startswith('metadata_') and fn.endswith('.json'):
                out.append(os.path.join(root, fn))
    return out


def parse_meta(path):
    try:
        with open(path, 'r') as f:
            j = json.load(f)
    except Exception:
        return None
    return j


def summarize(indir, outcsv):
    metas = find_metadata_files(indir)
    rows = []
    for m in metas:
        rel = os.path.relpath(m, indir)
        parts = rel.split(os.sep)
        run_folder = parts[0] if parts else ''
        agg = os.path.basename(m).replace('metadata_', '').replace('.json', '')
        data = parse_meta(m) or {}
        comp = data.get('composed_rdp_formal')
        if isinstance(comp, dict) and 'eps' in comp:
            eps = comp.get('eps')
            delta = comp.get('delta')
            note = ''
        else:
            eps = ''
            delta = ''
            note = str(comp) if comp is not None else data.get('notes','')

        # try to extract parameters from run folder name
        sd = ns = zt = ''
        try:
            # run_folder like sd64_ns1_zt2.0 or sd64_ns1_zt2.0_tag
            rf = run_folder
            parts = rf.split('_')
            for p in parts:
                if p.startswith('sd'):
                    sd = p.replace('sd','')
                if p.startswith('ns'):
                    ns = p.replace('ns','')
                if p.startswith('zt'):
                    zt = p.replace('zt','')
        except Exception:
            pass

        rows.append({'run': run_folder, 'agg': agg, 'sketch_dim': sd, 'n_sketches': ns, 'z_thresh': zt, 'eps': eps, 'delta': delta, 'note': note, 'meta_path': m})

    # write CSV
    fieldnames = ['run','agg','sketch_dim','n_sketches','z_thresh','eps','delta','note','meta_path']
    with open(outcsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return outcsv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--indir', required=True)
    p.add_argument('--out', default='summary_rdp.csv')
    args = p.parse_args()
    out = summarize(args.indir, args.out)
    print('Wrote', out)


if __name__ == '__main__':
    main()
