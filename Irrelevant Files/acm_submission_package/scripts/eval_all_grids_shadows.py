#!/usr/bin/env python
"""Evaluate shadow membership attacks on all checkpoints from the grid sweep."""

import os
import glob
import csv
import numpy as np
import tensorflow as tf
from fl_helpers import shadow_model_membership_attack

def load_mnist_clients(num_clients=10):
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
        clients.append((x_train[s], y_train[s]))
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


def load_weights_from_checkpoint(ckpt_path):
    """Load weights from npz checkpoint."""
    data = np.load(ckpt_path)
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    return weights


def run_shadow_eval(outdir: str, num_samples_for_attack: int = 256):
    """Evaluate shadow attacks on all final round checkpoints in the grid sweep."""
    print(f"Loading MNIST data...")
    clients, (x_test, y_test) = load_mnist_clients(num_clients=10)
    full_train_x = np.concatenate([c[0] for c in clients], axis=0)
    full_train_y = np.concatenate([c[1] for c in clients], axis=0)
    
    # find all grid subdirectories
    grid_dirs = sorted(glob.glob(os.path.join(outdir, 'sd*_ns*_zt*')))
    
    shadow_results = []
    
    for grid_dir in grid_dirs:
        # parse grid params from dirname
        dirname = os.path.basename(grid_dir)
        parts = dirname.split('_')
        sd = int(parts[0][2:])
        ns = int(parts[1][2:])
        zt = float(parts[2][2:])
        
        print(f"\n--- Processing grid {dirname} ---")
        
        # find aggregator checkpoints
        agg_dirs = sorted(glob.glob(os.path.join(grid_dir, 'apra*')) + glob.glob(os.path.join(grid_dir, 'trimmed')) + glob.glob(os.path.join(grid_dir, 'median')))
        
        for agg_dir in agg_dirs:
            agg_name = os.path.basename(agg_dir)
            # find latest checkpoint (highest round number)
            ckpts = glob.glob(os.path.join(agg_dir, 'round_*.npz'))
            if not ckpts:
                print(f"No checkpoints found in {agg_dir}")
                continue
            
            latest_ckpt = sorted(ckpts)[-1]
            round_num = int(os.path.basename(latest_ckpt).split('_')[1].split('.')[0])
            
            print(f"  {agg_name} round {round_num}...", end=' ', flush=True)
            
            try:
                weights = load_weights_from_checkpoint(latest_ckpt)
                
                def model_fn():
                    m = create_mnist_model()
                    try:
                        m.set_weights(weights)
                    except Exception:
                        pass
                    return m
                
                # prepare member/nonmember examples
                member_x = clients[0][0][:num_samples_for_attack]
                member_y = clients[0][1][:num_samples_for_attack]
                non_x = x_test[:num_samples_for_attack]
                non_y = y_test[:num_samples_for_attack]
                
                # run shadow attack (lightweight)
                shadow_res = shadow_model_membership_attack(
                    model_fn=model_fn,
                    full_train=(full_train_x, full_train_y),
                    full_holdout=(x_test, y_test),
                    member_examples=(member_x, member_y),
                    nonmember_examples=(non_x, non_y),
                    num_shadows=2,
                    shadow_size=500,
                    shadow_epochs=1,
                    attacker_epochs=3,
                    top_k=3,
                )
                auc = shadow_res.get('auc')
                print(f"AUC={auc:.4f}")
                shadow_results.append({
                    'sketch_dim': sd,
                    'n_sketches': ns,
                    'z_thresh': zt,
                    'agg': agg_name,
                    'round': round_num,
                    'shadow_auc': auc
                })
            except Exception as e:
                print(f"Error: {e}")
    
    # save results
    csv_path = os.path.join(outdir, 'shadow_aucs_all_grids.csv')
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['sketch_dim', 'n_sketches', 'z_thresh', 'agg', 'round', 'shadow_auc'])
        writer.writeheader()
        writer.writerows(shadow_results)
    
    print(f"\nShadow AUC results saved to {csv_path}")
    return shadow_results


if __name__ == '__main__':
    import sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'apra_mnist_runs_full'
    run_shadow_eval(outdir)
