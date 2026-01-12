"""
Parameterized sweep harness to compare APRA/CAPRA vs Krum under scaling attacks.
Writes CSV of results to `results/krum_poisoning_sweep.csv` by default.

Usage (example):
    python scripts/sweep_krum_poisoning.py --seeds 1 2 3 --byz 0.0 0.25 0.5 --rounds 3 --clients 8

This is intentionally lightweight and re-uses the same small CNN used in smoke tests.
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
from tensorflow import keras

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fl_helpers


def build_small_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(8, 3, activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def split_clients(x, y, num_clients):
    per = len(x) // num_clients
    clients = []
    for i in range(num_clients):
        start = i * per
        end = start + per if i < num_clients - 1 else len(x)
        clients.append((x[start:end], y[start:end]))
    return clients


def local_train_and_delta(model, x, y, local_epochs=1, batch_size=32):
    orig = [w.copy() for w in model.get_weights()]
    model.fit(x, y, epochs=local_epochs, batch_size=batch_size, verbose=0)
    new = model.get_weights()
    delta = [np.array(n) - np.array(o) for n, o in zip(new, orig)]
    model.set_weights(orig)
    return delta


def run_single_experiment(seed, byz_frac, scale_factor, rounds=3, clients=8):
    np.random.seed(int(seed))
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # slice the dataset first to avoid allocating the full dataset in float32
    x_train = x_train[:4000]
    y_train = y_train[:4000].flatten()
    x_test = x_test[:800]
    y_test = y_test[:800].flatten()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    clients_data = split_clients(x_train, y_train, clients)

    # APRA/CAPRA baseline: record per-round accuracies
    apra_round_accs = []
    global_model = build_small_cnn()
    for r in range(1, rounds + 1):
        deltas = []
        for cid, (cx, cy) in enumerate(clients_data):
            cm = build_small_cnn()
            cm.set_weights(global_model.get_weights())
            d = local_train_and_delta(cm, cx, cy, local_epochs=1)
            deltas.append(d)
        agg, meta = fl_helpers.aggregate_dispatcher('capra', deltas, sketch_dim_per_layer=32, n_sketches=1, time_budget_ms=50.0, probe_samples=3, seed=seed + r)
        new = [w + a for w, a in zip(global_model.get_weights(), agg)]
        global_model.set_weights(new)
        _, acc = global_model.evaluate(x_test, y_test, verbose=0)
        apra_round_accs.append(float(acc))

    # Krum under scaling attack: record per-round accuracies
    krum_round_accs = []
    global_model_k = build_small_cnn()
    num_byz = max(1, int(np.ceil(byz_frac * clients)))
    for r in range(1, rounds + 1):
        deltas = []
        for cid, (cx, cy) in enumerate(clients_data):
            cm = build_small_cnn()
            cm.set_weights(global_model_k.get_weights())
            d = local_train_and_delta(cm, cx, cy, local_epochs=1)
            if cid < num_byz:
                d = [dd * scale_factor for dd in d]
            deltas.append(d)
        agg_k = fl_helpers.federated_krum(deltas)
        newk = [w + a for w, a in zip(global_model_k.get_weights(), agg_k)]
        global_model_k.set_weights(newk)
        _, acck = global_model_k.evaluate(x_test, y_test, verbose=0)
        krum_round_accs.append(float(acck))

    return {
        'apra_rounds': apra_round_accs,
        'krum_rounds': krum_round_accs,
        'apra_final': apra_round_accs[-1] if apra_round_accs else float('nan'),
        'krum_final': krum_round_accs[-1] if krum_round_accs else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--byz', type=float, nargs='+', default=[0.0, 0.25, 0.5])
    parser.add_argument('--scales', type=float, nargs='+', default=[10.0])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--out_summary', type=str, default='results/krum_poisoning_summary.csv')
    parser.add_argument('--out_detailed', type=str, default='results/krum_poisoning_detailed.csv')
    parser.add_argument('--resume', action='store_true', help='Skip experiments already present in out_summary')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    # Summary header
    summary_header = ['timestamp', 'seed', 'byzantine_fraction', 'scale_factor', 'rounds', 'clients', 'apra_final', 'krum_final']
    detailed_header = ['timestamp', 'seed', 'byzantine_fraction', 'scale_factor', 'round', 'method', 'acc']

    # build set of completed combos if resume requested
    completed = set()
    summary_exists = os.path.exists(args.out_summary)
    detailed_exists = os.path.exists(args.out_detailed)
    if args.resume and summary_exists:
        try:
            with open(args.out_summary, 'r', newline='') as fh_read:
                r = csv.DictReader(fh_read)
                for row in r:
                    try:
                        completed.add((int(float(row['seed'])), float(row['byzantine_fraction']), float(row['scale_factor'])))
                    except Exception:
                        # ignore malformed rows
                        continue
        except Exception:
            completed = set()

    # Open files for append if they exist, otherwise create and write headers
    open_mode_summary = 'a' if summary_exists else 'w'
    open_mode_detailed = 'a' if detailed_exists else 'w'
    with open(args.out_summary, open_mode_summary, newline='') as sum_fh, open(args.out_detailed, open_mode_detailed, newline='') as det_fh:
        sum_writer = csv.writer(sum_fh)
        det_writer = csv.writer(det_fh)
        if not summary_exists:
            sum_writer.writerow(summary_header)
            sum_fh.flush()
        if not detailed_exists:
            det_writer.writerow(detailed_header)
            det_fh.flush()

        total = len(args.seeds) * len(args.byz) * len(args.scales)
        count = 0
        for seed in args.seeds:
            for byz in args.byz:
                for scale in args.scales:
                    if args.resume and (seed, float(byz), float(scale)) in completed:
                        print(f"Skipping completed: seed={seed} byz={byz} scale={scale}")
                        continue
                    count += 1
                    t0 = time.time()
                    print(f"Running ({count}/{total}) seed={seed} byz={byz} scale={scale}")
                    out = run_single_experiment(seed, byz, scale, rounds=args.rounds, clients=args.clients)
                    # write summary row
                    sum_writer.writerow([time.time(), seed, byz, scale, args.rounds, args.clients, out['apra_final'], out['krum_final']])
                    sum_fh.flush()
                    # write detailed per-round rows
                    for ri, a in enumerate(out['apra_rounds'], start=1):
                        det_writer.writerow([time.time(), seed, byz, scale, ri, 'apra', a])
                    for ri, k in enumerate(out['krum_rounds'], start=1):
                        det_writer.writerow([time.time(), seed, byz, scale, ri, 'krum', k])
                    det_fh.flush()
                    print(f"Done seed={seed} byz={byz} scale={scale} -> apra_final={out['apra_final']:.4f}, krum_final={out['krum_final']:.4f} (elapsed {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
