"""Collect experiment artifacts into a zip for submission.

Usage:
    python -u scripts/export_artifacts.py --out submission_artifacts.zip --dirs apra_mnist_runs_full tmp_apra_test --patterns "*.csv" "*.svg" "*.npz"

This script finds files under provided directories matching patterns and stores them in a zip.
"""
import argparse
import os
import fnmatch
import zipfile


def collect_files(dirs, patterns):
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for pat in patterns:
                for fn in fnmatch.filter(filenames, pat):
                    files.append(os.path.join(root, fn))
    return files


def main():
    parser = argparse.ArgumentParser(description='Export artifacts into a zip file')
    parser.add_argument('--out', required=True, help='Output zip path')
    parser.add_argument('--dirs', nargs='+', required=True, help='Directories to scan')
    parser.add_argument('--patterns', nargs='+', default=['*.csv','*.png','*.svg','*.pdf','*.npz','*.json'], help='Filename patterns to include')
    args = parser.parse_args()

    files = collect_files(args.dirs, args.patterns)
    if not files:
        print('No files found matching patterns in given dirs')
        return
    print(f'Adding {len(files)} files to {args.out}')
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = os.path.relpath(f)
            zf.write(f, arcname)
    print('Done')

if __name__ == '__main__':
    main()
