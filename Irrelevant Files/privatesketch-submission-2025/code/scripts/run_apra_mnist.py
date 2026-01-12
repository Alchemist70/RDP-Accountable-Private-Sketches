"""APRA MNIST federated smoke experiment.

This script runs a small federated training loop on MNIST using APRA aggregation.
It supports injecting three attack types: scaling, label_flip, and backdoor.

Warning: this script imports TensorFlow and will take longer to run. It's intended
for controlled experiments (few rounds, small local epochs).
"""
import os
import sys
import time
import json
import subprocess
import sys
import numpy as np
import tensorflow as tf
from typing import List, Tuple

# Ensure parent directory is on sys.path so scripts can import project modules like `fl_helpers`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fl_helpers import apra_aggregate, federated_trimmed_mean, federated_median, shadow_model_membership_attack, apra_weighted_aggregate, aggregate_dispatcher
import privacy_accounting


NUM_CLIENTS = 10
ROUNDS = 25
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
MAX_SAMPLES_PER_CLIENT = 3000  # Limit per-client data to avoid memory pressure during training
OUTPUT_DIR = os.environ.get('APRA_OUTPUT_DIR', 'apra_mnist_runs')


def load_mnist_clients(num_clients=NUM_CLIENTS):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    labels = np.argmax(y_train, axis=1)
    sorted_idx = np.argsort(labels)
    splits = np.array_split(sorted_idx, num_clients)
    clients = []
    for s in splits:
        x_c, y_c = x_train[s], y_train[s]
        # Limit per-client data to reduce memory pressure during training
        if len(x_c) > MAX_SAMPLES_PER_CLIENT:
            x_c = x_c[:MAX_SAMPLES_PER_CLIENT]
            y_c = y_c[:MAX_SAMPLES_PER_CLIENT]
        clients.append((x_c, y_c))
    return clients, (x_test, y_test)


def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def inject_attack(clients: List[Tuple[np.ndarray, np.ndarray]], attack_type: str, attacker_idx: int = 0, **kwargs):
    if attack_type == 'scaling':
        scale = float(kwargs.get('scale', 50.0))
        x, y = clients[attacker_idx]
        # scale their (simulated) updates by transforming x slightly: we simulate by scaling weights later
        # For simplicity, we will mark attacker by index and later multiply its weights in aggregation
        clients[attacker_idx] = (x, y)
        return ('scaling', attacker_idx, scale)
    elif attack_type == 'layer_backdoor':
        # inject backdoor to first layer (conv2d kernel) only
        x, y = clients[attacker_idx]
        return ('layer_backdoor', attacker_idx, {})
    elif attack_type == 'label_flip':
        x, y = clients[attacker_idx]
        # flip labels to other class randomly
        y_flipped = np.roll(y, shift=1, axis=1)
        clients[attacker_idx] = (x, y_flipped)
        return ('label_flip', attacker_idx, {})
    elif attack_type == 'backdoor':
        x, y = clients[attacker_idx]
        # inject tiny backdoor pattern: add 1.0 to corner pixels (not realistic but simple)
        x_bd = np.copy(x)
        x_bd[:, 0, 0, 0] = np.clip(x_bd[:, 0, 0, 0] + 0.9, 0.0, 1.0)
        clients[attacker_idx] = (x_bd, y)
        return ('backdoor', attacker_idx, {})
    return (None, None, None)


def _save_checkpoint(weights: List[np.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # save list of arrays as a npz
    np.savez_compressed(path, *weights)


def _run_federated_train(clients, x_test, y_test, aggregation_fn, agg_name: str, sketch_params: dict, attack_meta, rounds: int, local_epochs: int, batch_size: int, out_dir: str):
    model = create_mnist_model()
    global_weights = model.get_weights()
    stats = []
    t0 = time.time()
    for r in range(rounds):
        local_weights = []
        for cid, (x_c, y_c) in enumerate(clients):
            local_model = create_mnist_model()
            local_model.set_weights(global_weights)
            # use full per-client data (no truncation)
            local_model.fit(x_c, y_c, epochs=local_epochs, batch_size=batch_size, verbose=0)
            w = local_model.get_weights()
            if attack_meta[0] == 'scaling' and cid == attack_meta[1]:
                w = [np.asarray(arr) * float(attack_meta[2]) for arr in w]
            elif attack_meta[0] == 'layer_backdoor' and cid == attack_meta[1]:
                # inject small perturbation to first layer (conv2d)
                w[0] = w[0] + np.random.normal(0, 0.1, size=w[0].shape).astype(w[0].dtype)
            local_weights.append(w)

        # aggregate
        if agg_name == 'apra_weighted':
            new_global = apra_weighted_aggregate(local_weights, sketch_dim_per_layer=sketch_params.get('sketch_dim_per_layer', 32), n_sketches=sketch_params.get('n_sketches', 2), z_thresh=sketch_params.get('z_thresh', 3.0), seed=r)
        elif agg_name == 'apra_basic':
            new_global = apra_aggregate(local_weights, sketch_dim=sketch_params.get('sketch_dim', 128), trim_fraction=0.2, z_thresh=sketch_params.get('z_thresh', 3.0), seed=r)
        elif agg_name == 'trimmed':
            new_global = federated_trimmed_mean(local_weights, trim_fraction=0.2)
        elif agg_name == 'median':
            new_global = federated_median(local_weights)
        elif agg_name == 'farpa':
            # Use dispatcher to call FARPA
            agg_out, meta = aggregate_dispatcher(
                'farpa',
                local_weights,
                sketch_dim_per_layer=sketch_params.get('sketch_dim_per_layer', 32),
                n_sketches=sketch_params.get('n_sketches', 2),
                eps_sketch=sketch_params.get('eps_sketch', 0.5),
                z_thresh=sketch_params.get('z_thresh', 3.0),
                sketch_noise_mech=sketch_params.get('sketch_noise_mech', 'laplace'),
                seed=r,
            )
            new_global = agg_out
        elif agg_name == 'capra':
            # CAPRA adaptive aggregation via dispatcher
            # determine fast dim if provided; else auto inside capra_aggregate
            capra_kwargs = {
                'sketch_dim_canonical': sketch_params.get('sketch_dim_per_layer', 32),
                'n_sketches_canonical': sketch_params.get('n_sketches', 2),
                'sketch_dim_fast': sketch_params.get('capra_fast_dim', 0) or max(8, sketch_params.get('sketch_dim_per_layer', 32) // 4),
                'n_sketches_fast': sketch_params.get('capra_fast_n_sketches', 1),
                'eps_sketch': sketch_params.get('eps_sketch', 1.0),
                'time_budget_ms': sketch_params.get('capra_time_budget_ms', 200.0),
                'z_thresh': sketch_params.get('z_thresh', 3.0),
                'seed': r,
            }
            agg_out, meta = aggregate_dispatcher('capra', local_weights, **capra_kwargs)
            new_global = agg_out
        else:
            # default plain average
            # compute plain average
            n = len(local_weights)
            avg = []
            for layer_vals in zip(*local_weights):
                stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
                avg.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
            new_global = avg

        model.set_weights(new_global)
        global_weights = new_global

        # eval on full test set
        res = model.evaluate(x_test, y_test, verbose=0)
        acc = res[1] if isinstance(res, (list, tuple, np.ndarray)) and len(res) > 1 else float('nan')
        elapsed = time.time() - t0
        stats.append({'round': r + 1, 'accuracy': float(acc), 'elapsed_s': elapsed})

        # save per-round checkpoint and stats
        ckpt_path = os.path.join(out_dir, f"{agg_name}", f"round_{r+1:03d}.npz")
        _save_checkpoint(global_weights, ckpt_path)

        print(f"[{agg_name}] Round {r+1}/{rounds} - Test accuracy (subset): {acc:.4f} - elapsed: {elapsed:.1f}s")

    total_time = time.time() - t0
    return stats, total_time, global_weights


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=ROUNDS)
    parser.add_argument('--local-epochs', type=int, default=LOCAL_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--attack', type=str, default='scaling')
    parser.add_argument('--clients', type=int, default=NUM_CLIENTS)
    parser.add_argument('--outdir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--sketch-dims', type=str, default='64,128', help='comma separated sketch dims for sweep (per-layer param)')
    parser.add_argument('--n-sketches', type=str, default='1,2', help='comma separated n_sketches values')
    parser.add_argument('--z-thresh', type=str, default='2.0,3.0', help='comma separated z_thresh values')
    parser.add_argument('--sketch-noise-mech', type=str, default='laplace', help='noise mechanism for sketches: laplace|gaussian')
    # CAPRA-specific options
    parser.add_argument('--capra-time-budget-ms', type=float, default=200.0, help='time budget (ms) for CAPRA probe to decide fast mode')
    parser.add_argument('--capra-fast-dim', type=int, default=0, help='fast-mode per-layer sketch dim (0 -> auto = canonical//4)')
    parser.add_argument('--capra-fast-n-sketches', type=int, default=1, help='fast-mode n_sketches')
    args = parser.parse_args()

    NUM_CLIENTS = int(args.clients)
    ROUNDS = int(args.rounds)
    LOCAL_EPOCHS = int(args.local_epochs)
    BATCH_SIZE = int(args.batch_size)
    OUTPUT_DIR = args.outdir

    clients, (x_test, y_test) = load_mnist_clients(NUM_CLIENTS)
    attack_meta = inject_attack(clients, args.attack, attacker_idx=2)

    # build sweep grid
    sketch_dims = [int(x) for x in args.sketch_dims.split(',') if x.strip()]
    n_sketches_list = [int(x) for x in args.n_sketches.split(',') if x.strip()]
    z_thresh_list = [float(x) for x in args.z_thresh.split(',') if x.strip()]

    # ensure output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # CSV header
    csv_lines = ["agg,sketch_dim,n_sketches,z_thresh,round,accuracy,elapsed_s"]

    # iterate grid and run experiments for apra_weighted, apra_basic, trimmed, median, farpa, capra
    aggs = ['apra_weighted', 'apra_basic', 'trimmed', 'median', 'farpa', 'capra']
    for sd in sketch_dims:
        for ns in n_sketches_list:
            for zt in z_thresh_list:
                sketch_params = {'sketch_dim_per_layer': max(8, sd // 4), 'n_sketches': ns, 'sketch_dim': sd, 'z_thresh': zt}
                print(f"\n--- Running grid sd={sd}, ns={ns}, zt={zt} ---")
                for agg in aggs:
                    run_dir = os.path.join(OUTPUT_DIR, f"sd{sd}_ns{ns}_zt{zt}")
                    os.makedirs(run_dir, exist_ok=True)
                    # clear per-experiment privacy ledger so each aggregator's run records are isolated
                    try:
                        privacy_accounting.ledger.clear()
                    except Exception:
                        pass

                    # write provenance metadata for this run directory and ensure a per-aggregator metadata file exists
                    meta_path = os.path.join(run_dir, f'metadata_{agg}.json')
                    try:
                        # write run-level provenance (metadata_run.json)
                        privacy_accounting.write_provenance_metadata(run_dir, args=vars(args), seed=None)
                    except Exception:
                        pass

                    # ensure per-aggregator metadata file exists so post-run updates can append composed budgets
                    if not os.path.exists(meta_path):
                        try:
                            meta = {
                                'git_sha': None,
                                'python_version': sys.version,
                                'args': vars(args),
                                'seed': None,
                                'notes': 'metadata written before run; updated after completion'
                            }
                            # attempt to get git sha for the aggregator metadata as well
                            try:
                                git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.dirname(__file__)), stderr=subprocess.DEVNULL).decode('utf-8').strip()
                                meta['git_sha'] = git_sha
                            except Exception:
                                pass
                            with open(meta_path, 'w') as mf:
                                json.dump(meta, mf, indent=2)
                        except Exception:
                            pass
                    # pass CAPRA params through sketch_params
                    sketch_params['capra_time_budget_ms'] = float(args.capra_time_budget_ms)
                    sketch_params['capra_fast_dim'] = int(args.capra_fast_dim) if int(args.capra_fast_dim) > 0 else 0
                    sketch_params['capra_fast_n_sketches'] = int(args.capra_fast_n_sketches)
                    # global sketch noise mechanism (laplace|gaussian)
                    sketch_params['sketch_noise_mech'] = str(args.sketch_noise_mech)

                    stats, tot_t, final_weights = _run_federated_train(
                        clients,
                        x_test,
                        y_test,
                        None,
                        agg,
                        sketch_params,
                        attack_meta,
                        rounds=ROUNDS,
                        local_epochs=LOCAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        out_dir=run_dir,
                    )
                    # append to csv_lines
                    for s in stats:
                        csv_lines.append(f"{agg},{sd},{ns},{zt},{s['round']},{s['accuracy']:.6f},{s['elapsed_s']:.2f}")
                    # After run: collect composed privacy numbers from ledger and update metadata.json
                    try:
                        basic_eps, basic_delta = privacy_accounting.ledger.basic_composition()
                        adv_eps, adv_delta = privacy_accounting.ledger.advanced_composition(delta_prime=1e-6)
                        rdp_eps, rdp_delta = privacy_accounting.ledger.rdp_approx_composition(delta_prime=1e-6)
                        # attempt formal RDP via tensorflow_privacy if available
                        formal_rdp = None
                        try:
                            formal_eps, formal_delta = privacy_accounting.ledger.compute_rdp_via_tensorflow_privacy(target_delta=1e-6)
                            formal_rdp = {'eps': float(formal_eps), 'delta': float(formal_delta)}
                        except Exception as e:
                            formal_rdp = {'error': str(e)}
                        # update metadata
                        try:
                            with open(meta_path, 'r') as mf:
                                meta = json.load(mf)
                        except Exception:
                            meta = {}
                        meta.update({
                            'composed_basic': {'eps': basic_eps, 'delta': basic_delta},
                            'composed_advanced': {'eps': adv_eps, 'delta': adv_delta},
                            'composed_rdp_approx': {'eps': rdp_eps, 'delta': rdp_delta},
                            'composed_rdp_formal': formal_rdp,
                            'run_time_s': tot_t,
                        })
                        with open(meta_path, 'w') as mf:
                            json.dump(meta, mf, indent=2)
                    except Exception:
                        pass

    # write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'apra_mnist_results.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"All runs complete. Results saved to {csv_path}")
