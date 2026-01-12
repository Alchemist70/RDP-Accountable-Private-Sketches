"""Smoke-run the federated training loop (mirrors the notebook loop) and save basic metrics.

Saves: scripts/federated_smoke_metrics.csv
"""
import os
import csv
import time
import numpy as np
import tensorflow as tf

# Try to import APS from fl_helpers if available
try:
    from fl_helpers import AdaptivePrivacyShield
    aps = AdaptivePrivacyShield(default_epsilon=1.0)
except Exception:
    class _SimpleAPS:
        def __init__(self, default_epsilon=1.0):
            self.default_epsilon = float(default_epsilon)
        def calculate_attack_risk(self, gradients, round_num=0):
            try:
                norms = [np.sqrt(np.sum([np.sum((g.astype(np.float64))**2) for g in client_grads])) for client_grads in gradients]
                return float(np.mean(norms))
            except Exception:
                return 0.0
        def apply_hybrid_protection(self, weights, epsilon):
            return weights
        def update_privacy_budget(self, client_id, consumed=0.0):
            return self.default_epsilon
    aps = _SimpleAPS(default_epsilon=1.0)

# utility functions

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


def federated_averaging(local_weights_list):
    if not local_weights_list:
        return None
    avg = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        avg.append(np.mean(stacked, axis=0))
    return avg


def client_update(global_model, clients, client_id, round_num, local_epochs=1, batch_size=64):
    x_c, y_c = clients[client_id]
    if len(x_c) == 0:
        return global_model.get_weights(), float('nan'), getattr(aps, 'default_epsilon', 1.0)
    local_model = create_mnist_model()
    local_model.set_weights(global_model.get_weights())
    try:
        history = local_model.fit(x_c[:200], y_c[:200], epochs=local_epochs, batch_size=batch_size, verbose=0)
        loss = float(history.history.get('loss', [float('nan')])[-1])
    except Exception as e:
        print(f"client_update training error for client {client_id}: {e}")
        return global_model.get_weights(), float('nan'), getattr(aps, 'default_epsilon', 1.0)
    epsilon = getattr(aps, 'default_epsilon', 1.0)
    try:
        if hasattr(aps, 'update_privacy_budget'):
            try:
                eps_new = aps.update_privacy_budget(client_id, consumed=0.0)
                epsilon = float(eps_new) if eps_new is not None else epsilon
            except TypeError:
                try:
                    eps_new = aps.update_privacy_budget(client_id)
                    epsilon = float(eps_new) if eps_new is not None else epsilon
                except Exception:
                    pass
    except Exception:
        pass
    return local_model.get_weights(), loss, float(epsilon)


def load_mnist_clients(num_clients=10):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    labels = np.argmax(y_train, axis=1)
    sorted_idx = np.argsort(labels)
    splits = np.array_split(sorted_idx, num_clients)
    clients = []
    for s in splits:
        clients.append((x_train[s], y_train[s]))
    return clients, (x_test.reshape(-1,28,28,1).astype('float32')/255.0, None)


def main():
    NUM_CLIENTS = 10
    NUM_ROUNDS = 5
    clients, _ = load_mnist_clients(NUM_CLIENTS)
    global_model = create_mnist_model()

    # metrics
    global_losses = []
    local_losses = []
    privacy_budgets = []
    attack_risks = []

    for round_num in range(NUM_ROUNDS):
        print(f"\nFederated Round {round_num + 1}/{NUM_ROUNDS}")
        local_weights = []
        round_losses = []
        round_epsilons = []
        round_gradients = []

        for client_id in range(NUM_CLIENTS):
            try:
                x_train_c, y_train_c = clients[client_id]
                print(f"Client {client_id + 1} x_train shape: {x_train_c.shape}")
                old_weights = global_model.get_weights()
                client_weights, client_loss, epsilon = client_update(global_model, clients, client_id, round_num)
                new_weights = client_weights
                local_weights.append(new_weights)
                round_losses.append(client_loss)
                round_epsilons.append(epsilon)
                gradients = [(new - old) for new, old in zip(new_weights, old_weights)]
                round_gradients.append(gradients)
                print(f"Client {client_id + 1} - Loss: {client_loss:.4f}, Privacy Budget (ε): {epsilon:.4f}")
            except Exception as e:
                print(f"Error for client {client_id + 1}: {e}")
                continue

        round_risk = aps.calculate_attack_risk(round_gradients, round_num)
        attack_risks.append(round_risk)

        global_weights = federated_averaging(local_weights)
        if global_weights is None:
            print("[WARNING] No valid client updates this round. Skipping aggregation.")
            global_losses.append(float('nan'))
            local_losses.append([])
            privacy_budgets.append(float('nan'))
            attack_risks.append(float('nan'))
            continue
        global_model.set_weights(global_weights)

        avg_round_loss = np.mean([l for l in round_losses if not np.isnan(l)]) if round_losses else float('nan')
        avg_round_epsilon = np.mean(round_epsilons) if round_epsilons else float('nan')
        global_losses.append(avg_round_loss)
        local_losses.append(round_losses)
        privacy_budgets.append(avg_round_epsilon)

        print(f"Average Round Loss: {avg_round_loss:.4f}")
        print(f"Average Privacy Budget (ε): {avg_round_epsilon:.4f}")
        print(f"Attack Risk Score: {round_risk:.4f}")

    # save CSV
    out_csv = os.path.join('scripts', 'federated_smoke_metrics.csv')
    rows = []
    for r in range(NUM_ROUNDS):
        rows.append({'round': r+1, 'avg_loss': global_losses[r], 'avg_epsilon': privacy_budgets[r], 'attack_risk': attack_risks[r]})
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round','avg_loss','avg_epsilon','attack_risk'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print('\nSaved federated smoke metrics to', out_csv)


if __name__ == '__main__':
    main()
