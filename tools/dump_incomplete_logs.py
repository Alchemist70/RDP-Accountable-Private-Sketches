#!/usr/bin/env python3
from pathlib import Path
import os
from tools.report_incomplete_tasks import scan

ROOT = Path('apra_mnist_runs_full')


def tail(path, n=200):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            to_read = min(size, 32*1024)
            f.seek(max(0, size - to_read))
            raw = f.read().decode('utf-8', errors='replace')
            lines = raw.splitlines()
            return lines[-n:]
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    rows = scan()
    incomplete = [r for r in rows if r[2] < 25]
    for grid, agg, count, missing, last_mod in incomplete:
        # locate the agg dir
        grid_path = ROOT / grid
        agg_path = grid_path / agg
        if not agg_path.exists():
            # maybe nested
            nested = grid_path / grid
            if nested.exists():
                agg_path = nested / agg
        print('\n' + '='*80)
        print(f"{grid} / {agg}: {count}/25  missing: {missing}")
        for logname in ('launcher_stderr.log', 'stderr.log', 'launcher_stdout.log', 'stdout.log'):
            p = agg_path / logname
            lines = tail(p, n=200)
            print('\n--', logname, '--')
            if lines is None:
                print('  [missing]')
            else:
                for L in lines:
                    print(L)

