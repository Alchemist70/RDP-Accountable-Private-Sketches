"""
Small CIFAR-10 smoke experiment for APRA/CAPRA/FARPA runners.
Quick FL loop: split CIFAR-10 into N clients, run a few rounds, and exercise aggregators.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fl_helpers


def build_small_cnn(input_shape=(32,32,3), num_classes=10):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
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
    orig_weights = [w.copy() for w in model.get_weights()]
    model.fit(x, y, epochs=local_epochs, batch_size=batch_size, verbose=0)
    new_weights = model.get_weights()
    delta = [np.array(nw) - np.array(ow) for nw, ow in zip(new_weights, orig_weights)]
    # revert model to orig weights for client isolation
    model.set_weights(orig_weights)
    return delta


def run_smoke(agg_method='capra', rounds=3, clients=10, local_epochs=1, seed=123):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # reduce dataset for speed
    x_train = x_train[:5000]
    y_train = y_train[:5000].flatten()
    x_test = x_test[:1000]
    y_test = y_test[:1000].flatten()

    clients_data = split_clients(x_train, y_train, clients)

    global_model = build_small_cnn()
    # initialize weights
    global_model.set_weights(global_model.get_weights())

    results = []
    for r in range(1, rounds + 1):
        client_deltas = []
        for cid, (cx, cy) in enumerate(clients_data):
            # create client model copy
            client_model = build_small_cnn()
            client_model.set_weights(global_model.get_weights())
            delta = local_train_and_delta(client_model, cx, cy, local_epochs=local_epochs)
            client_deltas.append(delta)

        # call dispatcher
        try:
            if agg_method == 'capra':
                agg, meta = fl_helpers.aggregate_dispatcher('capra', client_deltas, sketch_dim_per_layer=64, n_sketches=1, eps_sketch=1.0, time_budget_ms=50.0, probe_samples=3, seed=seed + r)
            elif agg_method == 'farpa':
                agg, meta = fl_helpers.aggregate_dispatcher('farpa', client_deltas, sketch_dim_per_layer=64, n_sketches=1, eps_sketch=1.0, seed=seed + r)
            elif agg_method == 'krum':
                agg = fl_helpers.federated_krum(client_deltas)
                meta = {'method': 'krum'}
            else:
                agg, meta = fl_helpers.aggregate_dispatcher(agg_method, client_deltas)
        except Exception as e:
            print('Aggregation failed:', e)
            agg = fl_helpers.federated_trimmed_mean(client_deltas, trim_fraction=0.2)
            meta = {'fallback': True}

        # apply agg to global model
        new_global = [np.array(w) + np.array(a) for w, a in zip(global_model.get_weights(), agg)]
        global_model.set_weights(new_global)

        # evaluate
        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {r}: acc={acc:.4f} meta_mode={meta.get('mode') if isinstance(meta, dict) else meta}")
        results.append({'round': r, 'acc': float(acc), 'meta': meta})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', type=str, default='capra')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--clients', type=int, default=8)
    parser.add_argument('--local_epochs', type=int, default=1)
    args = parser.parse_args()
    run_smoke(agg_method=args.agg, rounds=args.rounds, clients=args.clients, local_epochs=args.local_epochs)
