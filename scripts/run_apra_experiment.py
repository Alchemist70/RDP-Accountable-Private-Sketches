"""APRA smoke test script.

This script demonstrates the APRA aggregation utilities implemented in
`fl_helpers.py` using synthetic weight updates and a simple malicious
client injection (scaling attack).

Run as a quick sanity check; it's lightweight and does not require TF.
"""
import numpy as np
import time
from typing import List

# To keep this smoke test lightweight we reimplement a small subset of the
# APRA utilities locally here so the script does not trigger heavy imports
# (tensorflow, sklearn) when executed.


def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(w).ravel() for w in weights]) if weights else np.array([])


def random_projection_sketch(weights: List[np.ndarray], sketch_dim: int = 128, seed: int = 0) -> np.ndarray:
    vec = _flatten_weights(weights)
    if vec.size == 0:
        return np.zeros((sketch_dim,), dtype=np.float32)
    rng = np.random.RandomState(int(seed))
    proj = rng.normal(loc=0.0, scale=1.0 / np.sqrt(float(max(1, vec.size))), size=(vec.size, sketch_dim)).astype(np.float32)
    sketch = vec.astype(np.float32).dot(proj)
    norm = np.linalg.norm(sketch) + 1e-12
    return (sketch / norm).astype(np.float32)


def apra_detect_outliers(sketches: List[np.ndarray], z_thresh: float = 3.0) -> List[bool]:
    if not sketches:
        return []
    S = np.vstack([np.asarray(s).ravel() for s in sketches])
    med = np.median(S, axis=0)
    dists = np.linalg.norm(S - med[None, :], axis=1)
    med_dist = np.median(dists)
    mad = np.median(np.abs(dists - med_dist)) + 1e-12
    threshold = med_dist + z_thresh * mad
    good = (dists <= threshold).tolist()
    return good


def ensemble_sketch(weights: List[np.ndarray], sketch_dim: int = 128, n_sketches: int = 3, seed: int = 0) -> np.ndarray:
    """Return concatenated ensemble of multiple random-projection sketches (lightweight local copy)."""
    if n_sketches <= 1:
        return random_projection_sketch(weights, sketch_dim=sketch_dim, seed=seed)
    sketches = [random_projection_sketch(weights, sketch_dim=sketch_dim, seed=seed + i) for i in range(n_sketches)]
    return np.concatenate(sketches, axis=0)


def per_layer_ensemble_sketch(weights: List[np.ndarray], sketch_dim: int = 64, n_sketches: int = 2, seed: int = 0) -> np.ndarray:
    """Compute per-layer ensemble sketches and concatenate them.

    This preserves per-layer signal and is often more robust to targeted layer attacks.
    """
    parts = []
    for li, layer in enumerate(weights):
        # for each layer, compute a small sketch
        s = random_projection_sketch([layer], sketch_dim=sketch_dim, seed=seed + li)
        if n_sketches > 1:
            extra = [random_projection_sketch([layer], sketch_dim=sketch_dim, seed=seed + li + j + 1) for j in range(n_sketches - 1)]
            s = np.concatenate([s] + extra, axis=0)
        parts.append(s)
    return np.concatenate(parts, axis=0)


def apra_weighted_aggregate(local_weights_list: List[List[np.ndarray]], sketch_dim_per_layer: int = 32, n_sketches: int = 2, z_thresh: float = 3.0, eps: float = 1e-6, seed: int = 0) -> List[np.ndarray]:
    """Weighted aggregation using per-layer sketches to compute trust scores.

    Steps:
    - For each client compute per-layer sketches and a single concatenated sketch.
    - Compute distances to median sketch and convert to trust weights via softmax-like mapping.
    - Use trust weights to compute a weighted average of weights per coordinate.
    """
    n = len(local_weights_list)
    if n == 0:
        return []
    # compute per-client concatenated per-layer sketch
    sketches = [per_layer_ensemble_sketch(w, sketch_dim=sketch_dim_per_layer, n_sketches=n_sketches, seed=seed + i) for i, w in enumerate(local_weights_list)]
    S = np.vstack([s.ravel() for s in sketches])
    med = np.median(S, axis=0)
    # use cosine distances for robustness
    dists = np.array([1.0 - float(np.dot(s.ravel(), med) / ((np.linalg.norm(s) + 1e-12) * (np.linalg.norm(med) + 1e-12))) for s in sketches])
    # map distances to positive trust scores (smaller dist -> higher trust)
    inv = 1.0 / (dists + eps)
    weights = inv / np.sum(inv)

    # weighted aggregation per-layer
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)  # shape (n_clients, *shape)
        # reshape to (n_clients, -1)
        orig_shape = stacked.shape[1:]
        flat = stacked.reshape((stacked.shape[0], -1))
        # apply weights along axis 0
        weighted = np.tensordot(weights, flat, axes=(0, 0))
        out.append(weighted.reshape(orig_shape).astype(layer_vals[0].dtype))
    return out


def federated_trimmed_mean(local_weights_list: List[List[np.ndarray]], trim_fraction: float = 0.2) -> List[np.ndarray]:
    if not local_weights_list:
        return []
    n_clients = len(local_weights_list)
    k = int(np.floor(trim_fraction * n_clients))
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        if k <= 0:
            out.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
            continue
        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[k:n_clients - k, ...]
        out.append(np.mean(trimmed, axis=0).astype(layer_vals[0].dtype))
    return out


def federated_median(local_weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    if not local_weights_list:
        return []
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        med = np.median(stacked, axis=0)
        out.append(med.astype(layer_vals[0].dtype))
    return out


def apra_aggregate(local_weights_list: List[List[np.ndarray]], sketch_dim: int = 128, trim_fraction: float = 0.2, z_thresh: float = 3.0, seed: int = 0) -> List[np.ndarray]:
    n = len(local_weights_list)
    if n == 0:
        return []
    sketches = [random_projection_sketch(w, sketch_dim=sketch_dim, seed=seed + i) for i, w in enumerate(local_weights_list)]
    benign_mask = apra_detect_outliers(sketches, z_thresh=z_thresh)
    benign_indices = [i for i, ok in enumerate(benign_mask) if ok]
    if len(benign_indices) >= max(1, int(np.ceil(0.5 * n))):
        selected = [local_weights_list[i] for i in benign_indices]
        avg = []
        for layer_vals in zip(*selected):
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            avg.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
        return avg
    return federated_trimmed_mean(local_weights_list, trim_fraction=trim_fraction)


def synth_client_updates(n_clients: int, layer_shapes: List[tuple], benign_scale: float = 0.01, seed: int = 0):
    rng = np.random.RandomState(seed)
    clients = []
    for i in range(n_clients):
        weights = []
        for s in layer_shapes:
            # small random update
            w = rng.normal(loc=0.0, scale=benign_scale, size=s).astype(np.float32)
            weights.append(w)
        clients.append(weights)
    return clients


def inject_scaling_attack(clients: List[List[np.ndarray]], attacker_idx: int = 0, scale: float = 50.0):
    # amplify attacker's update
    attacked = [np.copy(w) for w in clients[attacker_idx]]
    for i in range(len(attacked)):
        attacked[i] = attacked[i] * scale
    clients[attacker_idx] = attacked


def avg_weights(local_weights_list: List[List[np.ndarray]]):
    n = len(local_weights_list)
    if n == 0:
        return []
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        out.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
    return out


def weight_distance(a: List[np.ndarray], b: List[np.ndarray]):
    if not a or not b:
        return float('nan')
    s = 0.0
    for x, y in zip(a, b):
        s += float(np.linalg.norm(np.asarray(x).ravel() - np.asarray(y).ravel()))
    return s


def run_smoke():
    print("[APRA] Running smoke test: synthetic clients + scaling attack")
    n_clients = 10
    layer_shapes = [(50,), (20,)]
    clients = synth_client_updates(n_clients, layer_shapes, benign_scale=0.02, seed=123)

    # Save benign average reference (mean of all true benign clients before attack)
    benign_avg = avg_weights(clients)

    # Attack scenarios to test
    attack_configs = [
        ('scaling', {'attacker_idx': 2, 'scale': 50.0}),
        ('label_flip', {'attacker_idx': 3}),
        ('additive_backdoor', {'attacker_idx': 1, 'magnitude': 0.5}),
    ]

    sketch_dims = [32, 64, 128]
    n_sketches = [1, 2, 4]

    for attack_name, params in attack_configs:
        # make fresh clients copy for each attack
        base_clients = synth_client_updates(n_clients, layer_shapes, benign_scale=0.02, seed=123)
        if attack_name == 'scaling':
            inject_scaling_attack(base_clients, attacker_idx=params['attacker_idx'], scale=params['scale'])
        elif attack_name == 'label_flip':
            # flip sign of attacker's updates
            idx = params['attacker_idx']
            for li in range(len(base_clients[idx])):
                base_clients[idx][li] = -base_clients[idx][li]
        elif attack_name == 'additive_backdoor':
            idx = params['attacker_idx']
            mag = params.get('magnitude', 0.5)
            # add a small fixed vector to first layer
            base_clients[idx][0] = base_clients[idx][0] + mag

        print('\n--- Attack:', attack_name, '---')
        for sd in sketch_dims:
            for ns in n_sketches:
                # compute apra with ensemble size ns and sketch dim sd
                # reusing local implementations in this script
                sketches = [ensemble_sketch(w, sketch_dim=sd, n_sketches=ns, seed=42 + i) for i, w in enumerate(base_clients)]
                mask = apra_detect_outliers(sketches, z_thresh=3.0)
                apra_agg = apra_aggregate(base_clients, sketch_dim=sd, trim_fraction=0.2, z_thresh=3.0, seed=42)
                # weighted per-layer APRA aggregation
                apra_w = apra_weighted_aggregate(base_clients, sketch_dim_per_layer=max(8, sd // 4), n_sketches=ns, z_thresh=3.0, seed=42)
                plain_avg = avg_weights(base_clients)
                trimmed = federated_trimmed_mean(base_clients, trim_fraction=0.2)
                med = federated_median(base_clients)
                print(f"sd={sd}, n_sketches={ns} -> mask_true={sum(mask)}/{len(mask)}; distances: plain={weight_distance(plain_avg, benign_avg):.4f}, trimmed={weight_distance(trimmed, benign_avg):.4f}, median={weight_distance(med, benign_avg):.4f}, apra_basic={weight_distance(apra_agg, benign_avg):.4f}, apra_weighted={weight_distance(apra_w, benign_avg):.4f}")

    print("[APRA] Tuning smoke tests complete.")


if __name__ == '__main__':
    run_smoke()
