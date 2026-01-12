"""Package selected artifacts into a zip file for submission.

Usage:
    python scripts/package_artifacts.py --out artifacts.zip --paths results/*.png apra_mnist_runs_full/apra_mnist_results.csv
"""
import argparse
import zipfile
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='artifacts.zip')
    parser.add_argument('--paths', type=str, nargs='+', required=True, help='glob patterns to include')
    args = parser.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))

    files = sorted(set(files))
    if not files:
        print('No files matched the provided paths')
        return

    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            arcname = os.path.relpath(f, start=os.getcwd())
            z.write(f, arcname=arcname)
            print(f'Added {f} as {arcname}')

    print(f'Packaged {len(files)} files into {args.out}')


if __name__ == '__main__':
    main()
