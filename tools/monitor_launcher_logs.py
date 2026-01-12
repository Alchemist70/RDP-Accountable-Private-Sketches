import argparse
import time
import re
from pathlib import Path
from datetime import datetime

RE_CHECK_ROUND = re.compile(r"Round\s+(\d{1,3})/25")


def tail_files(paths, interval=30, last_n=200):
    paths = [Path(p) for p in paths]
    positions = {p: 0 for p in paths}
    last_round = {p: 0 for p in paths}
    print(f"Starting monitor for {len(paths)} files; polling every {interval}s")
    try:
        while True:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('\n' + '='*80)
            print(f"{now} - Monitor tick")
            all_done = True
            for p in paths:
                print('\n' + '-'*20)
                print(f"File: {p}")
                if not p.exists():
                    print("  [missing]")
                    all_done = False
                    continue
                size = p.stat().st_size
                # Read last_n lines
                try:
                    with p.open('rb') as fh:
                        # read last ~32k bytes or file length
                        to_read = min(size, 32*1024)
                        fh.seek(max(0, size - to_read))
                        raw = fh.read().decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"  [error reading file] {e}")
                    all_done = False
                    continue
                lines = raw.splitlines()
                tail = lines[-last_n:]
                # Print tail (limited)
                print('\n'.join(tail))
                # Extract highest round seen
                rounds = [int(m.group(1)) for line in tail for m in [RE_CHECK_ROUND.search(line)] if m]
                if rounds:
                    mr = max(rounds)
                    if mr > last_round[p]:
                        print(f"  -> Progress: Round {mr}/25 (increased)")
                    else:
                        print(f"  -> Progress: Round {last_round[p]}/25")
                    last_round[p] = max(last_round[p], mr)
                else:
                    print("  -> Progress: Round 0/25")
                    all_done = False
                if last_round[p] < 25:
                    all_done = False
            if all_done:
                print("\nAll monitored files reached Round 25/25. Exiting monitor.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\nMonitor interrupted by user. Exiting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', required=True, help='Paths to launcher_stderr.log files')
    parser.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    parser.add_argument('--lines', type=int, default=200, help='How many tail lines to print per file')
    args = parser.parse_args()
    tail_files(args.paths, interval=args.interval, last_n=args.lines)
