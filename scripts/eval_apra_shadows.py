import os
import numpy as np
import tensorflow as tf
from fl_helpers import shadow_model_membership_attack

# Duplicate minimal data/model loaders to match training script

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

OUTDIR = 'apra_mnist_runs_short'
GRID_DIR = os.path.join(OUTDIR, 'sd128_ns2_zt3.0')
AGGS = ['apra_weighted', 'apra_basic', 'trimmed', 'median']

clients, (x_test, y_test) = load_mnist_clients(num_clients=10)
full_train_x = np.concatenate([c[0] for c in clients], axis=0)
full_train_y = np.concatenate([c[1] for c in clients], axis=0)

results = []
for agg in AGGS:
    ckpt = os.path.join(GRID_DIR, agg, 'round_020.npz')
    if not os.path.exists(ckpt):
        print('Checkpoint not found for', agg, ckpt)
        continue
    data = np.load(ckpt)
    # rebuild weights list from arrays in npz (they were saved positionally)
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    # prepare model_fn that returns a model with these weights
    def model_fn():
        m = create_mnist_model()
        try:
            m.set_weights(weights)
        except Exception:
            pass
        return m

    # member/nonmember examples for shadow attack
    member_x = clients[0][0][:256]
    member_y = clients[0][1][:256]
    non_x = x_test[:256]
    non_y = y_test[:256]

    print(f'Running shadow attack evaluation for {agg}...')
    shadow_res = shadow_model_membership_attack(
        model_fn=model_fn,
        full_train=(full_train_x, full_train_y),
        full_holdout=(x_test, y_test),
        member_examples=(member_x, member_y),
        nonmember_examples=(non_x, non_y),
        num_shadows=3,
        shadow_size=500,
        shadow_epochs=1,
        attacker_epochs=5,
        top_k=3,
    )
    auc = shadow_res.get('auc')
    print(f"{agg} shadow AUC: {auc}")
    results.append((agg, auc))

# write summary CSV
with open(os.path.join(OUTDIR, 'apra_mnist_shadows_summary.csv'), 'w') as f:
    f.write('agg,shadow_auc\n')
    for agg, auc in results:
        f.write(f"{agg},{auc}\n")

print('Done. Summary saved to', os.path.join(OUTDIR, 'apra_mnist_shadows_summary.csv'))
