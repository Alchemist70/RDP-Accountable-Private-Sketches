#!/usr/bin/env python
"""Monitor script for APRA runs.

Polls `apra_mnist_runs_full/*/run.log` every 30s and reports:
- first non-empty outputs (prints last 50 lines)
- any occurrences of 'Traceback' or 'Exception' in logs
- checkpoint progress: counts of `round_*.npz` per aggregator and new rounds

Run with: python -u scripts/monitor_runs.py
"""
from pathlib import Path
import time
import sys
import glob
import os
from datetime import datetime

OUTDIR = Path('apra_mnist_runs_full')
POLL_INTERVAL = 30
AGGS = ['apra_weighted', 'apra_basic', 'trimmed', 'median']
TAIL_LINES = 50


def ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tail(path, n=TAIL_LINES):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = max(1024, n * 200)
            pos = max(0, end - size)
            f.seek(pos)
            data = f.read().decode(errors='replace')
            lines = data.splitlines()
            return '\n'.join(lines[-n:])
    except Exception as e:
        return f'<error reading log: {e}>'


def find_grids():
    if not OUTDIR.exists():
        return []
    return sorted([p for p in OUTDIR.iterdir() if p.is_dir()])


def check_once(prev):
    """Scan once. `prev` is a dict tracking seen sizes/counts."""
    grids = find_grids()
    new_prev = {}
    events = []

    for grid in grids:
        gname = grid.name
        new_prev[gname] = {'logs': {}, 'round_counts': {}}
        for agg in AGGS:
            agg_dir = grid / agg
            # track run.log
            log_path = agg_dir / 'run.log'
            prev_size = prev.get(gname, {}).get('logs', {}).get(agg, 0)
            cur_size = 0
            if log_path.exists():
                try:
                    cur_size = log_path.stat().st_size
                except Exception:
                    cur_size = prev_size
            new_prev[gname]['logs'][agg] = cur_size
            if cur_size > 0 and prev_size == 0:
                # first non-empty
                content = tail(log_path)
                short = '\n'.join(content.splitlines()[:20]) if content else ''
                events.append((gname, agg, 'FIRST_NONEMPTY', short))
                if 'Traceback' in content or 'Exception' in content:
                    events.append((gname, agg, 'ERROR_DETECTED', '\n'.join([l for l in content.splitlines() if 'Traceback' in l or 'Exception' in l][:20])))
            elif cur_size > prev_size:
                # growth since last check; report tail
                content = tail(log_path)
                if 'Traceback' in content or 'Exception' in content:
                    events.append((gname, agg, 'ERROR_DETECTED', '\n'.join([l for l in content.splitlines() if 'Traceback' in l or 'Exception' in l][:20])))

            # track rounds
            round_files = []
            if agg_dir.exists():
                round_files = [p for p in agg_dir.iterdir() if p.is_file() and p.name.startswith('round_') and p.name.endswith('.npz')]
            cur_round_count = len(round_files)
            prev_round = prev.get(gname, {}).get('round_counts', {}).get(agg, 0)
            new_prev[gname]['round_counts'][agg] = cur_round_count
            if cur_round_count > prev_round:
                events.append((gname, agg, 'NEW_ROUNDS', f'{prev_round} -> {cur_round_count}'))

    # Also detect grids that disappeared
    prev_grids = set(prev.keys())
    cur_grids = set(new_prev.keys())
    removed = prev_grids - cur_grids
    for r in removed:
        events.append((r, None, 'GRID_REMOVED', ''))

    return new_prev, events


def print_event(ev):
    gname, agg, etype, payload = ev
    header = f"[{ts()}] {etype}: {gname} {('/'+agg) if agg else ''}"
    print(header)
    if payload:
        print(payload)
    print('-' * 72)


def main():
    print(f"{ts()} Monitor starting. Poll interval: {POLL_INTERVAL}s. Watching: {OUTDIR}")
    prev = {}
    try:
        # do an immediate initial scan
        prev, events = check_once(prev)
        if not events:
            print(f"{ts()} Initial scan: no non-empty logs or new rounds detected yet.")
        else:
            for ev in events:
                print_event(ev)

        while True:
            time.sleep(POLL_INTERVAL)
            prev, events = check_once(prev)
            if events:
                for ev in events:
                    print_event(ev)
            else:
                print(f"{ts()} Poll: no changes")

    except KeyboardInterrupt:
        print(f"{ts()} Monitor stopped by user (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == '__main__':
    main()
