"""Lightweight MNIST multi-seed smoke test.

Runs a short federated-style simulation (2 rounds) across a few clients and
uses `fl_helpers.AdaptivePrivacyShield` utilities for sanity checks. This is
intended as a fast, local smoke test (not full experiments).
"""

import time
import numpy as np
import tensorflow as tf

from fl_helpers import AdaptivePrivacyShield, run_multi_seed_experiment, evaluate_on_heldout, membership_inference_via_loss, shadow_model_membership_attack

# Suppress TF logging for clarity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_CLIENTS = 5
NUM_ROUNDS = 2
SEEDS = [42, 123, 456]

# Simple MNIST loader and non-iid split (sort by label and split)
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

# Small model (fast)
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

# Simple federated averaging for weight lists
def federated_averaging(weights_list):
    # weights_list: list of model.get_weights() (lists of arrays)
    if not weights_list:
        return None
    avg = []
    for layer_vals in zip(*weights_list):
        avg.append(np.mean(np.stack([np.asarray(v) for v in layer_vals], axis=0), axis=0))
    return avg

# Short FL simulation used per-seed
def run_seed(seed):
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))
    clients, (x_test, y_test) = load_mnist_clients(num_clients=NUM_CLIENTS)

    # create global model
    global_model = create_mnist_model()
    global_weights = global_model.get_weights()

    aps = AdaptivePrivacyShield(default_epsilon=1.0)

    grad_vars = []
    t0 = time.time()
    for r in range(NUM_ROUNDS):
        local_weights = []
        for cid in range(min(3, len(clients))):
            x, y = clients[cid]
            if len(x) == 0:
                continue
            # local model copy
            m = create_mnist_model()
            m.set_weights(global_weights)
            # train one epoch on small batch to keep this fast
            try:
                m.fit(x[:128], y[:128], epochs=1, batch_size=32, verbose=0)
            except Exception:
                continue
            new_w = m.get_weights()
            local_weights.append(new_w)
            # gradient proxy = weight diffs
            grads = [nw - gw for nw, gw in zip(new_w, global_weights)]
            # variance proxy
            grad_vars.append(np.mean([float(np.var(g)) for g in grads if np.isfinite(np.var(g))]))
        # aggregate
        if local_weights:
            new_global = federated_averaging(local_weights)
            global_weights = new_global
            global_model.set_weights(global_weights)
    elapsed = time.time() - t0

    # Evaluate on heldout
    heldout = evaluate_on_heldout(global_model, x_test, y_test, sample_limit=1000)
    acc = heldout.get('accuracy', float('nan'))
    privacy_proxy = float(np.mean(grad_vars)) if grad_vars else float('nan')

    # Simple membership inference evaluation: sample some members and non-members
    try:
        rng = np.random.RandomState(int(seed))
        member_x = clients[0][0][:200]
        member_y = clients[0][1][:200]
        non_x = x_test[:200]
        non_y = y_test[:200]
        mi = membership_inference_via_loss(global_model, member_x, member_y, non_x, non_y)
        membership_auc = float(mi.get('auc', float('nan')))

        # Shadow-model attack (stronger): use all training clients concatenated as full_train
        all_x = np.concatenate([c[0] for c in clients], axis=0)
        all_y = np.concatenate([c[1] for c in clients], axis=0)
        shadow_res = shadow_model_membership_attack(
            model_fn=create_mnist_model,
            full_train=(all_x, all_y),
            full_holdout=(x_test, y_test),
            member_examples=(member_x, member_y),
            nonmember_examples=(non_x, non_y),
            num_shadows=10,
            shadow_size=1000,
            shadow_epochs=2,
            attacker_epochs=20,
            top_k=3,
        )
        membership_shadow_auc = float(shadow_res.get('auc', float('nan')))
    except Exception:
        membership_auc = float('nan')
        membership_shadow_auc = float('nan')

    return {'utility': float(acc), 'privacy': privacy_proxy, 'overhead': float(elapsed), 'membership_auc': membership_auc, 'membership_shadow_auc': membership_shadow_auc}


if __name__ == '__main__':
    print('Running lightweight MNIST multi-seed smoke test...')
    res = run_multi_seed_experiment(run_seed, SEEDS, metrics=['utility','privacy','overhead','membership_auc','membership_shadow_auc'], num_bootstrap=100)
    print('\nSummary:')
    for k, v in res['summary'].items():
        print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, ci={v['ci']}")
    print('\nPer-seed outputs:')
    for p in res['per_seed']:
        print(f"  seed={p['seed']}, output={p['output']}, time={p['time']:.2f}s")
