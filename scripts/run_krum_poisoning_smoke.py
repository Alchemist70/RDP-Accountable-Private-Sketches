"""
Run a short comparison between Krum and APRA under a simple scaling attack.
This is a smoke harness (few rounds, small model) to exercise the poisoning logic.
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


def run_comparison(rounds=3, clients=8, byzantine_fraction=0.25, scale_factor=10.0):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[:4000]
    y_train = y_train[:4000].flatten()
    x_test = x_test[:800]
    y_test = y_test[:800].flatten()

    clients_data = split_clients(x_train, y_train, clients)

    # Baseline APRA/Capra run
    global_model = build_small_cnn()
    for r in range(1, rounds + 1):
        deltas = []
        for cid, (cx, cy) in enumerate(clients_data):
            cm = build_small_cnn()
            cm.set_weights(global_model.get_weights())
            d = local_train_and_delta(cm, cx, cy, local_epochs=1)
            deltas.append(d)
        agg, meta = fl_helpers.aggregate_dispatcher('capra', deltas, sketch_dim_per_layer=32, n_sketches=1, time_budget_ms=1.0, probe_samples=3, seed=r)
        new = [w + a for w, a in zip(global_model.get_weights(), agg)]
        global_model.set_weights(new)
    loss, acc_apra = global_model.evaluate(x_test, y_test, verbose=0)

    # Krum under poisoning
    global_model_k = build_small_cnn()
    num_byz = max(1, int(np.ceil(byzantine_fraction * clients)))
    for r in range(1, rounds + 1):
        deltas = []
        for cid, (cx, cy) in enumerate(clients_data):
            cm = build_small_cnn()
            cm.set_weights(global_model_k.get_weights())
            d = local_train_and_delta(cm, cx, cy, local_epochs=1)
            # apply scaling attack to first num_byz clients
            if cid < num_byz:
                d = [dd * scale_factor for dd in d]
            deltas.append(d)
        agg_k = fl_helpers.federated_krum(deltas)
        newk = [w + a for w, a in zip(global_model_k.get_weights(), agg_k)]
        global_model_k.set_weights(newk)
    lossk, acc_krum = global_model_k.evaluate(x_test, y_test, verbose=0)

    print(f"APRA/Capra acc: {acc_apra:.4f} | Krum (poisoned) acc: {acc_krum:.4f} | byz_frac={byzantine_fraction}")
    return {'apra_acc': float(acc_apra), 'krum_acc': float(acc_krum)}


if __name__ == '__main__':
    run_comparison()
