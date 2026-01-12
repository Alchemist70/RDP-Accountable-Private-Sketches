"""Protector comparison multi-seed harness.

This script:
- Trains victim models under different protection strategies (none, fixed, DP-SGD, secure-agg, APS).
- Uses tensorflow-privacy DP optimizer when available; otherwise falls back to a simulated DP post-hoc.
- Simulates secure aggregation via per-client local training + averaging.
- Runs a shadow-model membership attack to compute shadow AUC.
- Runs across multiple seeds and an epsilon sweep for 'fixed' and 'aps'.
- Saves per-seed CSV and aggregated summaries and plots.

Conservative default settings: num_shadows=6, shadow_size=800, shadow_epochs=1, attacker_epochs=8
"""
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from fl_helpers import apply_protector_by_name, shadow_model_membership_attack, evaluate_on_heldout, protector_apply_dpsgd
# Try to import tensorflow-privacy's DPKerasSGDOptimizer dynamically. Use importlib to avoid
# static import errors in environments where tensorflow-privacy isn't installed.
try:
    import importlib
    _mod = importlib.import_module('tensorflow_privacy.privacy.optimizers.dp_optimizer_keras')
    DPKerasSGDOptimizer = getattr(_mod, 'DPKerasSGDOptimizer', None)
    _HAS_TFP = DPKerasSGDOptimizer is not None
except Exception:
    DPKerasSGDOptimizer = None
    _HAS_TFP = False

# Suppress TF verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_CLIENTS = 5

# loader and model factory (same as notebook)
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


def _federated_aggregate(global_weights, local_weights_list):
    avg = []
    for layer_vals in zip(*local_weights_list):
        avg.append(np.mean(np.stack([np.asarray(v) for v in layer_vals], axis=0), axis=0))
    return avg


def train_victim_under_protector(name, params, clients, seed, create_model_fn):
    """Train a victim model under the specified protector. Returns a compiled Keras model."""
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))

    if name == 'dpsgd':
        # Central DP training using tensorflow-privacy if available
        model = create_model_fn()
        if _HAS_TFP and DPKerasSGDOptimizer is not None:
            clip = float(params.get('clip_norm', 1.0))
            noise = float(params.get('noise_multiplier', 1.0))
            lr = float(params.get('learning_rate', 0.01))
            microbatches = int(params.get('microbatches', 64))
            opt = DPKerasSGDOptimizer(l2_norm_clip=clip, noise_multiplier=noise, num_microbatches=microbatches, learning_rate=lr)
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            # central training on concatenated data
            X = np.concatenate([c[0] for c in clients], axis=0)
            Y = np.concatenate([c[1] for c in clients], axis=0)
            model.fit(X[:2000], Y[:2000], epochs=int(params.get('epochs', 3)), batch_size=int(params.get('batch_size', 64)), verbose=0)
            return model
        else:
            # fallback: use simulated dpsgd by training centrally and adding Gaussian noise to weights
            model = create_model_fn()
            X = np.concatenate([c[0] for c in clients], axis=0)
            Y = np.concatenate([c[1] for c in clients], axis=0)
            model.fit(X[:2000], Y[:2000], epochs=3, batch_size=64, verbose=0)
            w = model.get_weights()
            noisy = protector_apply_dpsgd(w, epsilon=float(params.get('epsilon', 1.0)), noise_multiplier=float(params.get('noise_multiplier', 1.0)), clip_norm=float(params.get('clip_norm', 1.0)))
            model.set_weights(noisy)
            return model

    if name in ('secure_agg', 'aps'):
        # Simulate federated local training + secure aggregation
        global_model = create_model_fn()
        global_weights = global_model.get_weights()
        local_weights_list = []
        for cid, (x_c, y_c) in enumerate(clients):
            local_model = create_model_fn()
            local_model.set_weights(global_weights)
            if len(x_c) > 0:
                local_model.fit(x_c[:200], y_c[:200], epochs=int(params.get('local_epochs', 1)), batch_size=64, verbose=0)
            local_weights_list.append(local_model.get_weights())

        new_global = _federated_aggregate(global_weights, local_weights_list)
        # apply APS protection to aggregated weights if requested
        if name == 'aps':
            aps_inst = params.get('aps_instance')
            if aps_inst is None:
                from fl_helpers import AdaptivePrivacyShield
                aps_inst = AdaptivePrivacyShield(default_epsilon=params.get('epsilon', 1.0))
            new_global = aps_inst.apply_hybrid_protection(new_global, float(params.get('epsilon', 1.0)))
        global_model.set_weights(new_global)
        return global_model

    # default: central training, then optionally apply post-hoc protector
    model = create_model_fn()
    X = np.concatenate([c[0] for c in clients], axis=0)
    Y = np.concatenate([c[1] for c in clients], axis=0)
    model.fit(X[:2000], Y[:2000], epochs=int(params.get('epochs', 3)), batch_size=int(params.get('batch_size', 64)), verbose=0)
    if name == 'fixed':
        w = model.get_weights()
        w2 = apply_protector_by_name('fixed', w, epsilon=float(params.get('epsilon', 1.0)))
        model.set_weights(w2)
    return model


def main():
    seeds = [42, 123, 456, 789, 1013]
    clients, (x_test, y_test) = load_mnist_clients(num_clients=NUM_CLIENTS)
    member_x = clients[0][0][:500]
    member_y = clients[0][1][:500]
    non_x = x_test[:500]
    non_y = y_test[:500]

    # protector base params
    base_protectors = [
        ('none', {}),
        ('fixed', {'epsilon': 1.0}),
        ('dpsgd', {'epsilon': 1.0, 'noise_multiplier': 1.0, 'clip_norm': 1.0, 'epochs': 3}),
        ('secure_agg', {'local_epochs': 1}),
        ('aps', {'epsilon': 1.0}),
    ]

    # epsilon sweep for fixed and aps
    eps_sweep = [0.5, 1.0, 2.0]

    results = []

    for seed in seeds:
        for name, params in base_protectors:
            # for fixed and aps run sweep over epsilons
            if name in ('fixed', 'aps'):
                for eps in eps_sweep:
                    params_copy = dict(params)
                    params_copy['epsilon'] = eps
                    if name == 'aps':
                        from fl_helpers import AdaptivePrivacyShield
                        params_copy['aps_instance'] = AdaptivePrivacyShield(default_epsilon=eps)
                    print(f"Seed {seed}: protector={name}, epsilon={eps}")
                    t0 = time.time()
                    victim = train_victim_under_protector(name, params_copy, clients, seed, create_mnist_model)
                    elapsed = time.time() - t0
                    util = evaluate_on_heldout(victim, x_test, y_test, sample_limit=1000).get('accuracy', float('nan'))
                    # shadow attack
                    shadow_res = shadow_model_membership_attack(
                        model_fn=create_mnist_model,
                        full_train=(np.concatenate([c[0] for c in clients], axis=0), np.concatenate([c[1] for c in clients], axis=0)),
                        full_holdout=(x_test, y_test),
                        member_examples=(member_x, member_y),
                        nonmember_examples=(non_x, non_y),
                        num_shadows=6,
                        shadow_size=800,
                        shadow_epochs=1,
                        attacker_epochs=8,
                        top_k=3,
                    )
                    auc = float(shadow_res.get('auc', float('nan')))
                    results.append({'seed': seed, 'protector': name, 'epsilon': eps, 'utility': float(util), 'shadow_auc': auc, 'time_s': elapsed})
                    print(f"  util={util:.4f}, shadow_auc={auc:.4f}, time={elapsed:.1f}s")
            else:
                params_copy = dict(params)
                t0 = time.time()
                print(f"Seed {seed}: protector={name}")
                victim = train_victim_under_protector(name, params_copy, clients, seed, create_mnist_model)
                elapsed = time.time() - t0
                util = evaluate_on_heldout(victim, x_test, y_test, sample_limit=1000).get('accuracy', float('nan'))
                shadow_res = shadow_model_membership_attack(
                    model_fn=create_mnist_model,
                    full_train=(np.concatenate([c[0] for c in clients], axis=0), np.concatenate([c[1] for c in clients], axis=0)),
                    full_holdout=(x_test, y_test),
                    member_examples=(member_x, member_y),
                    nonmember_examples=(non_x, non_y),
                    num_shadows=6,
                    shadow_size=800,
                    shadow_epochs=1,
                    attacker_epochs=8,
                    top_k=3,
                )
                auc = float(shadow_res.get('auc', float('nan')))
                results.append({'seed': seed, 'protector': name, 'epsilon': float('nan'), 'utility': float(util), 'shadow_auc': auc, 'time_s': elapsed})
                print(f"  util={util:.4f}, shadow_auc={auc:.4f}, time={elapsed:.1f}s")

    # save CSV
    out_csv = os.path.join('scripts', 'protector_results_multiseed.csv')
    with open(out_csv, 'w', newline='') as f:
        fieldnames = ['seed','protector','epsilon','utility','shadow_auc','time_s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('\nSaved results to', out_csv)

    # Aggregate and save plots: average shadow_auc per protector and epsilon
    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby(['protector','epsilon']).agg({'shadow_auc':['mean','std'],'utility':['mean','std']}).reset_index()
    out_agg = os.path.join('scripts', 'protector_results_agg.csv')
    agg.to_csv(out_agg, index=False)
    print('Saved aggregated results to', out_agg)

    # simple plot for baseline epsilon (nan) vs fixed/aps sweep: plot mean shadow_auc per protector/epsilon
    plt.figure(figsize=(10,5))
    for prot in df['protector'].unique():
        sub = df[df['protector'] == prot].groupby('epsilon')['shadow_auc'].mean()
        plt.plot(sub.index.astype(str), sub.values, marker='o', label=prot)
    plt.xlabel('epsilon')
    plt.ylabel('mean shadow AUC')
    plt.title('Protector sweep: mean shadow AUC across seeds')
    plt.legend()
    out_png = os.path.join('scripts', 'protector_sweep.png')
    plt.savefig(out_png)
    print('Saved sweep plot to', out_png)


if __name__ == '__main__':
    main()
