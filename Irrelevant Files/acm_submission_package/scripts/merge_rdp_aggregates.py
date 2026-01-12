#!/usr/bin/env python3
"""
Merge multiple `rdp_smokegrid_demo.json` files into a single merged demo JSON.
Writes `paper_figures/rdp_smokegrid_demo_multi.json`.
"""
import json
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--inputs', nargs='+', type=Path, required=True)
p.add_argument('--out', type=Path, default=Path('paper_figures/rdp_smokegrid_demo_multi.json'))
args = p.parse_args()

all_entries = []
for ip in args.inputs:
    j = json.loads(ip.read_text(encoding='utf-8'))
    # accept either a summary with 'entries' or a list
    if isinstance(j, dict) and 'entries' in j:
        all_entries.extend(j['entries'])
    elif isinstance(j, list):
        all_entries.extend(j)
    elif isinstance(j, dict):
        # maybe dict of named runs
        for v in j.values():
            if isinstance(v, dict) and 'epsilon' in v:
                all_entries.append(v)

out = {
    'count': len(all_entries),
    'entries': all_entries,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
print('Wrote', args.out)
