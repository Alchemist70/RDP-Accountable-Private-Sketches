"""Tabular scaling experiment runner (fast sanity run).
Saves CSV: scripts/tabular_scaling_results.csv
Saves PNG:  scripts/tabular_scaling.png

This script intentionally uses reduced sizes for a quick run: num_clients=[5,10,20], samples_per_client=128, rounds=3
"""
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Helpers (copied/adapted from notebook)
def create_non_iid_data(num_clients, samples_per_client, imbalance_factor=0.8):
    client_data = []
    for i in range(num_clients):
        mean_shift = i * imbalance_factor
        x = np.random.normal(loc=mean_shift, scale=2.0, size=(samples_per_client, 10))
        threshold = mean_shift * 0.5
        y = (np.sum(x * np.linspace(0.1, 1.0, 10), axis=1) > threshold).astype(np.float32)
        y = np.expand_dims(y, axis=1)
        client_data.append((x.astype(np.float32), y))
    return client_data


def create_tabular_model(input_dim=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def federated_averaging(local_weights_list):
    if not local_weights_list:
        return None
    avg = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        avg.append(np.mean(stacked, axis=0))
    return avg


def _safe_client_update(global_model, clients, client_id, round_num, local_epochs=1, batch_size=64):
    x_c, y_c = clients[client_id]
    if len(x_c) == 0:
        return global_model.get_weights(), float('nan'), 1.0
    try:
        local_model = tf.keras.models.clone_model(global_model)
        loss_fn = getattr(global_model, 'loss', None) or 'binary_crossentropy'
        local_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        local_model.set_weights(global_model.get_weights())
        history = local_model.fit(x_c[:200], y_c[:200], epochs=local_epochs, batch_size=batch_size, verbose=0)
        loss_val = float(history.history.get('loss', [float('nan')])[-1])
        return local_model.get_weights(), loss_val, 1.0
    except Exception as e:
        print(f"Fallback client_update failed: {e}")
        return global_model.get_weights(), float('nan'), 1.0


def run_scaling_experiment_tabular(num_clients_list, samples_per_client, rounds=3):
    scaling_results = []
    for num_clients in num_clients_list:
        print(f"\nTesting with {num_clients} clients...")
        client_data_local = create_non_iid_data(num_clients, samples_per_client)
        model = create_tabular_model()

        accuracies = []
        privacy_scores = []
        start_time = time.time()

        for round_num in range(rounds):
            local_weights = []
            round_gradients = []
            for client_id in range(num_clients):
                w, _, _ = _safe_client_update(model, client_data_local, client_id, round_num)
                local_weights.append(w)
                old_weights = model.get_weights()
                gradients = [(new - old) for new, old in zip(w, old_weights)]
                round_gradients.append(gradients)
            global_weights = federated_averaging(local_weights)
            if global_weights is None:
                print('[WARNING] No aggregated weights')
                continue
            model.set_weights(global_weights)

            # Evaluate across clients
            test_accuracies = []
            for cid in range(num_clients):
                x_test_c, y_test_c = client_data_local[cid]
                try:
                    eval_res = model.evaluate(x_test_c, y_test_c, verbose=0)
                    if isinstance(eval_res, (list, tuple, np.ndarray)) and len(eval_res) >= 2:
                        acc = float(eval_res[1])
                    else:
                        preds = model.predict(x_test_c, verbose=0)
                        acc = float(np.mean((preds.flatten() > 0.5).astype(np.float32) == y_test_c.flatten()))
                except Exception as e:
                    print(f"Evaluation failed for client {cid}: {e}")
                    acc = 0.0
                test_accuracies.append(acc)

            accuracies.append(np.mean(test_accuracies))
            # Simple privacy proxy: mean variance of gradients
            grad_vars = []
            for gl in round_gradients:
                for g in gl:
                    if g is None:
                        continue
                    g_np = g.numpy() if isinstance(g, tf.Tensor) else g
                    grad_vars.append(float(np.var(g_np)))
            privacy_score = float(np.mean(grad_vars)) if grad_vars else 0.0
            privacy_scores.append(privacy_score)

        total_time = time.time() - start_time
        scaling_results.append({'num_clients': num_clients,
                                 'final_accuracy': accuracies[-1] if accuracies else 0.0,
                                 'avg_privacy_score': float(np.mean(privacy_scores)) if privacy_scores else 0.0,
                                 'time_per_round': total_time / rounds})
    return scaling_results


if __name__ == '__main__':
    # Reduced quick run
    num_clients_list = [5, 10, 20]
    samples_per_client = 128
    rounds = 3

    print('Running quick tabular scaling experiment...')
    results = run_scaling_experiment_tabular(num_clients_list, samples_per_client, rounds=rounds)

    out_csv = os.path.join('scripts', 'tabular_scaling_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_clients','final_accuracy','avg_privacy_score','time_per_round'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print('\nSaved tabular scaling CSV to', out_csv)

    # Plot results
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot([r['num_clients'] for r in results], [r['final_accuracy'] for r in results], 'bo-')
    plt.xlabel('Number of Clients')
    plt.ylabel('Final Accuracy')
    plt.title('Accuracy Scaling')
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot([r['num_clients'] for r in results], [r['avg_privacy_score'] for r in results], 'ro-')
    plt.xlabel('Number of Clients')
    plt.ylabel('Avg Privacy Score')
    plt.title('Privacy Scaling')
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot([r['num_clients'] for r in results], [r['time_per_round'] for r in results], 'go-')
    plt.xlabel('Number of Clients')
    plt.ylabel('Time per Round (s)')
    plt.title('Computational Scaling')
    plt.grid(True)

    out_png = os.path.join('scripts', 'tabular_scaling.png')
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved plot to', out_png)

    print('\nScaling results:')
    for r in results:
        print(r)
