#!/usr/bin/env python3
"""
Create a simple demo JSON `paper_figures/rdp_smokegrid_demo.json` that contains a list
of entries suitable for `scripts/populate_rdp_table.py` from a merged json with an
`entries` field.

Usage:
  python scripts/make_demo_json.py --in paper_figures/rdp_smokegrid_demo_multi.json --out paper_figures/rdp_smokegrid_demo.json
"""
import json
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument('--in', dest='inp', type=Path, required=True)
p.add_argument('--out', dest='out', type=Path, default=Path('paper_figures/rdp_smokegrid_demo.json'))
args = p.parse_args()

j = json.loads(args.inp.read_text(encoding='utf-8'))
entries = j.get('entries') if isinstance(j, dict) else None
if entries is None:
    raise SystemExit('Input JSON does not contain an "entries" list')

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(entries, indent=2), encoding='utf-8')
print('Wrote', args.out)
