"""
APRA: Adaptive Private Robust Aggregation
Core module implementing sketching, robust detection, and Byzantine attack injection.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('apra')


# ============================================================================
# SKETCHING UTILITIES
# ============================================================================

def random_projection_sketch(weights: np.ndarray, sketch_dim: int, seed: Optional[int] = None, chunk_size: int = 10000) -> np.ndarray:
    """
    Memory-efficient random projection sketch via chunked processing.
    Avoids allocating full (D x k) matrix by processing flattened weight vector in chunks.
    Uses reusable random matrix per chunk to reduce memory fragmentation.
    
    Args:
        weights: ndarray or list of ndarrays (per-layer weights)
        sketch_dim: target sketch dimension k
        seed: random seed for reproducibility
        chunk_size: number of elements to process per chunk
    
    Returns:
        normalized sketch vector of shape (sketch_dim,)
    """
    if isinstance(weights, (list, tuple)):
        vec = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
    else:
        vec = np.asarray(weights).astype(np.float32).flatten()
    
    n = vec.size
    rng = np.random.RandomState(seed)
    sk = np.zeros(sketch_dim, dtype=np.float32)
    scale = 1.0 / np.sqrt(float(max(1, n)))
    
    # Reuse a single random projection matrix to avoid repeated allocation
    # Size is min(chunk_size, n) x sketch_dim to save memory
    proj_size = min(chunk_size, n)
    proj = rng.normal(loc=0.0, scale=scale, size=(proj_size, sketch_dim)).astype(np.float32)
    
    for i in range(0, n, chunk_size):
        end_idx = min(i + chunk_size, n)
        c = vec[i:end_idx]
        actual_chunk = len(c)
        
        # If chunk is smaller, regenerate smaller projection
        if actual_chunk < proj_size:
            proj_chunk = rng.normal(loc=0.0, scale=scale, size=(actual_chunk, sketch_dim)).astype(np.float32)
        else:
            proj_chunk = proj
        
        sk += c.dot(proj_chunk[:actual_chunk, :])
    
    norm = np.linalg.norm(sk)
    if norm > 0:
        sk /= norm
    return sk


def count_sketch(weights: np.ndarray, sketch_dim: int, num_hashes: int = 3, seed: Optional[int] = None) -> np.ndarray:
    """
    Count-sketch: hash-based sketch for fast streaming aggregation.
    
    Args:
        weights: flattened weight vector
        sketch_dim: target sketch dimension
        num_hashes: number of hash functions
        seed: random seed
    
    Returns:
        count-sketch vector of shape (sketch_dim,)
    """
    vec = np.asarray(weights).astype(np.float32).flatten()
    rng = np.random.RandomState(seed)
    sk = np.zeros(sketch_dim, dtype=np.float32)
    
    for h in range(num_hashes):
        seed_h = seed + h if seed is not None else None
        rng_h = np.random.RandomState(seed_h)
        # Hash indices and signs
        indices = rng_h.randint(0, sketch_dim, size=len(vec))
        signs = 2 * rng_h.randint(0, 2, size=len(vec)) - 1  # {-1, 1}
        sk += np.bincount(indices, weights=vec * signs, minlength=sketch_dim)
    
    return sk / num_hashes


def hadamard_sketch(weights: np.ndarray, sketch_dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Fast Hadamard transform sketch (simplified via random projection).
    In practice, can use scipy.linalg for true Hadamard if needed.
    
    Args:
        weights: flattened weight vector
        sketch_dim: target sketch dimension
        seed: random seed
    
    Returns:
        Hadamard sketch vector of shape (sketch_dim,)
    """
    return random_projection_sketch(weights, sketch_dim, seed=seed)


# ============================================================================
# ROBUST AGGREGATION & DETECTION
# ============================================================================

def coordinate_wise_median(updates: List[np.ndarray]) -> np.ndarray:
    """
    Compute coordinate-wise median across clients.
    
    Args:
        updates: list of weight vectors from clients
    
    Returns:
        aggregated weight vector (median per coordinate)
    """
    # Support updates provided as per-layer lists or as flattened arrays
    if len(updates) == 0:
        return np.array([])

    if isinstance(updates[0], (list, tuple)):
        flattened = [np.concatenate([np.asarray(w).flatten() for w in u]) for u in updates]
        stacked = np.stack(flattened, axis=0)
    else:
        stacked = np.stack(updates, axis=0)

    return np.median(stacked, axis=0)


def coordinate_wise_trimmed_mean(updates: List[np.ndarray], trim_fraction: float = 0.2) -> np.ndarray:
    """
    Compute coordinate-wise trimmed mean (remove top/bottom trim_fraction).
    
    Args:
        updates: list of weight vectors from clients
        trim_fraction: fraction to trim from each end (e.g., 0.2 = trim 20% largest, 20% smallest)
    
    Returns:
        aggregated weight vector (trimmed mean per coordinate)
    """
    # Support updates provided as per-layer lists or flattened arrays
    if len(updates) == 0:
        return np.array([])

    if isinstance(updates[0], (list, tuple)):
        flattened = [np.concatenate([np.asarray(w).flatten() for w in u]) for u in updates]
        stacked = np.stack(flattened, axis=0)
    else:
        stacked = np.stack(updates, axis=0)

    num_to_trim = max(1, int(len(updates) * trim_fraction))
    trimmed = np.sort(stacked, axis=0)
    if num_to_trim * 2 >= trimmed.shape[0]:
        # If trimming would remove all rows, fall back to mean
        return np.mean(trimmed, axis=0)
    trimmed = trimmed[num_to_trim:-num_to_trim, ...]
    return np.mean(trimmed, axis=0)


def krum_aggregation(updates: List[np.ndarray], num_to_exclude: int = 0) -> np.ndarray:
    """
    Krum aggregation: select client update with smallest sum-of-distances to other clients.
    Byzantine-robust (tolerates up to num_to_exclude bad clients).
    
    Args:
        updates: list of weight vectors from clients
        num_to_exclude: number of worst clients to exclude (default 0 = Krum)
    
    Returns:
        selected/aggregated weight vector
    """
    updates = [u.flatten() for u in updates]
    n = len(updates)
    
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(updates[i] - updates[j], ord=2)
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, compute sum of k-nearest neighbors (k = n - num_to_exclude - 2)
    k = n - num_to_exclude - 2
    scores = np.sum(np.sort(distances, axis=1)[:, :k], axis=1)
    
    # Return the client with smallest score
    selected_idx = np.argmin(scores)
    return updates[selected_idx]


def sketch_based_robust_detector(client_sketches: List[np.ndarray], method: str = 'median') -> Dict:
    """
    Detect Byzantine behavior using sketches (no access to raw updates).
    
    Args:
        client_sketches: list of sketch vectors from clients
        method: detection method ('median', 'mle', 'quantile')
    
    Returns:
        dict with aggregated_sketch, outlier_scores, suspicious_clients
    """
    sketches = np.array(client_sketches)
    n_clients = len(sketches)
    
    if method == 'median':
        agg_sketch = np.median(sketches, axis=0)
        # Outlier score = L2 distance to median
        outlier_scores = np.linalg.norm(sketches - agg_sketch, axis=1)
    elif method == 'mle':
        # Maximum likelihood: assume Gaussian; detect via Mahalanobis distance
        mean_sketch = np.mean(sketches, axis=0)
        cov = np.cov(sketches.T)
        # Add regularization to avoid singular matrix
        cov += 1e-6 * np.eye(cov.shape[0])
        inv_cov = np.linalg.inv(cov)
        outlier_scores = np.array([
            np.sqrt((s - mean_sketch).dot(inv_cov).dot(s - mean_sketch))
            for s in sketches
        ])
        agg_sketch = mean_sketch
    elif method == 'quantile':
        # Quantile-based: use element-wise quantiles
        agg_sketch = np.quantile(sketches, q=0.5, axis=0)
        outlier_scores = np.linalg.norm(sketches - agg_sketch, axis=1)
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    # Identify suspicious clients (e.g., top 10% by outlier score)
    threshold = np.quantile(outlier_scores, q=0.9)
    suspicious_clients = np.where(outlier_scores > threshold)[0]
    
    return {
        'aggregated_sketch': agg_sketch,
        'outlier_scores': outlier_scores,
        'suspicious_clients': suspicious_clients,
        'threshold': threshold
    }


def adaptive_weighted_aggregation(updates: List[np.ndarray], aps_scores: np.ndarray) -> np.ndarray:
    """
    Aggregate updates with adaptive weighting from APS (Adaptive Privacy Shield).
    Higher APS score = lower trust = lower weight.
    
    Args:
        updates: list of updates; each update is either:
                 - a list of weight arrays (one per layer), or
                 - a single flattened array
        aps_scores: per-client APS scores (trust/risk metric)
    
    Returns:
        adaptively weighted aggregated update
    """
    # Flatten each update if it's a list of arrays
    flattened_updates = []
    for u in updates:
        if isinstance(u, (list, tuple)):
            # u is a list of arrays (per-layer); flatten and concatenate
            flat = np.concatenate([np.array(arr).flatten() for arr in u])
        else:
            # u is already a flat array
            flat = np.array(u).flatten()
        flattened_updates.append(flat)
    
    updates_array = np.array(flattened_updates)
    n_clients = len(updates_array)
    
    # Convert APS scores to weights: invert so high score (high risk) gets low weight
    # Assume aps_scores in [0, 1]; normalize to weights
    weights = 1.0 / (1.0 + aps_scores)
    weights = weights / np.sum(weights)  # normalize
    
    aggregated = np.average(updates_array, axis=0, weights=weights)
    return aggregated


# ============================================================================
# BYZANTINE ATTACK INJECTION
# ============================================================================

def inject_scaling_attack(update: np.ndarray, scale_factor: float = 10.0) -> np.ndarray:
    """
    Scaling attack: multiply update by a large factor to poison aggregation.
    
    Args:
        update: client weight vector
        scale_factor: multiplier (e.g., 10x)
    
    Returns:
        poisoned update
    """
    return update * scale_factor


def inject_label_flip_attack(logits: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Label-flip attack: flip predictions to wrong class during training.
    (In FL, this modifies the gradient/update computed on flipped labels.)
    
    Args:
        logits: model logits/predictions
        num_classes: number of classes
    
    Returns:
        flipped logits (attack corrupts training target)
    """
    # Flip to random wrong class
    flipped = logits.copy()
    true_label = np.argmax(flipped)
    # Flip to a wrong class
    wrong_classes = [c for c in range(num_classes) if c != true_label]
    if wrong_classes:
        flipped[true_label] = -1e6
        flipped[wrong_classes[0]] = 1e6
    return flipped


def inject_backdoor_attack(update: np.ndarray, trigger_pattern: Optional[np.ndarray] = None, target_label: int = 0, scale: float = 0.5) -> np.ndarray:
    """
    Backdoor attack: poison update to cause model to misclassify specific inputs.
    Simplified: add a scaled perturbation in the direction that increases target_label logit.
    
    Args:
        update: client weight vector
        trigger_pattern: optional trigger pattern (else random)
        target_label: label to poison toward
        scale: attack magnitude
    
    Returns:
        backdoored update
    """
    if trigger_pattern is None:
        trigger_pattern = np.random.randn(*update.shape)
    
    # Add scaled perturbation
    poisoned = update + scale * trigger_pattern
    return poisoned


def inject_gaussian_noise_attack(update: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
    """
    Add Gaussian noise attack (simple model poisoning).
    
    Args:
        update: client weight vector
        noise_std: std dev of Gaussian noise
    
    Returns:
        noisy poisoned update
    """
    noise = np.random.normal(0, noise_std, size=update.shape)
    return update + noise


# ============================================================================
# APRA ORCHESTRATION
# ============================================================================

class APRAContext:
    """
    Context for APRA aggregation with privacy, robustness, and sketching.
    """
    
    def __init__(
        self,
        sketch_dim: int = 64,
        sketch_method: str = 'random_projection',
        robust_method: str = 'trimmed_mean',
        detection_method: str = 'median',
        aps_enabled: bool = False,
        byzantine_fraction: float = 0.0,
        attack_type: Optional[str] = None
    ):
        """
        Args:
            sketch_dim: sketch dimension k
            sketch_method: 'random_projection', 'count_sketch', 'hadamard'
            robust_method: 'median', 'trimmed_mean', 'krum', 'fedavg'
            detection_method: 'median', 'mle', 'quantile'
            aps_enabled: whether to use APS weighting
            byzantine_fraction: fraction of clients to poison
            attack_type: 'scaling', 'label_flip', 'backdoor', 'gaussian_noise', None
        """
        self.sketch_dim = sketch_dim
        self.sketch_method = sketch_method
        self.robust_method = robust_method
        self.detection_method = detection_method
        self.aps_enabled = aps_enabled
        self.byzantine_fraction = byzantine_fraction
        self.attack_type = attack_type
        
        self.sketch_fn = {
            'random_projection': random_projection_sketch,
            'count_sketch': count_sketch,
            'hadamard': hadamard_sketch,
        }.get(sketch_method, random_projection_sketch)
        
        self.robust_fn = {
            'median': coordinate_wise_median,
            'trimmed_mean': coordinate_wise_trimmed_mean,
            'krum': krum_aggregation,
            # 'apra_basic' uses the trimmed-mean style robust aggregator by default
            'apra_basic': coordinate_wise_trimmed_mean,
        }.get(robust_method, coordinate_wise_trimmed_mean)
    
    def apply_attack(self, update: np.ndarray, attack_seed: Optional[int] = None) -> np.ndarray:
        """Apply Byzantine attack if configured."""
        if self.attack_type is None:
            return update
        
        if attack_seed is not None:
            np.random.seed(attack_seed)
        
        if self.attack_type == 'scaling':
            return inject_scaling_attack(update, scale_factor=10.0)
        elif self.attack_type == 'gaussian_noise':
            return inject_gaussian_noise_attack(update, noise_std=1.0)
        elif self.attack_type == 'backdoor':
            return inject_backdoor_attack(update, scale=0.5)
        else:
            logger.warning(f"Unknown attack type: {self.attack_type}")
            return update
    
    def aggregate(
        self,
        client_updates: List[np.ndarray],
        aps_scores: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Aggregate client updates with privacy, robustness, and sketching.
        
        Args:
            client_updates: list of per-client weight updates
            aps_scores: optional per-client APS scores for weighting
            seed: random seed
        
        Returns:
            aggregated_update, metadata dict
        """
        n_clients = len(client_updates)
        metadata = {
            'n_clients': n_clients,
            'sketch_dim': self.sketch_dim,
            'sketch_method': self.sketch_method,
            'robust_method': self.robust_method,
        }
        
        # Inject attacks
        n_byzantine = max(1, int(n_clients * self.byzantine_fraction))
        byzantine_indices = np.random.choice(n_clients, size=n_byzantine, replace=False)
        poisoned_updates = []
        for i, update in enumerate(client_updates):
            if i in byzantine_indices:
                poisoned = self.apply_attack(update, attack_seed=seed + i if seed else None)
                poisoned_updates.append(poisoned)
            else:
                poisoned_updates.append(update)
        
        metadata['byzantine_count'] = n_byzantine
        metadata['byzantine_indices'] = byzantine_indices.tolist()
        
        # Compute sketches
        sketches = []
        for i, update in enumerate(poisoned_updates):
            sk = self.sketch_fn(update, self.sketch_dim, seed=seed + i if seed else None)
            sketches.append(sk)
        
        # Detect outliers via sketch-based detector
        detection_result = sketch_based_robust_detector(sketches, method=self.detection_method)
        metadata['detection_result'] = {
            'suspicious_clients': detection_result['suspicious_clients'].tolist(),
            'threshold': float(detection_result['threshold']),
        }
        
        # Robust aggregation
        if self.robust_method == 'krum':
            aggregated = self.robust_fn(poisoned_updates, num_to_exclude=n_byzantine)
        elif self.robust_method == 'trimmed_mean':
            aggregated = self.robust_fn(poisoned_updates, trim_fraction=0.2)
        elif self.aps_enabled and aps_scores is not None:
            aggregated = adaptive_weighted_aggregation(poisoned_updates, aps_scores)
        else:
            aggregated = self.robust_fn(poisoned_updates)
        
        metadata['aggregation_method'] = self.robust_method
        return aggregated, metadata


if __name__ == '__main__':
    # Smoke test
    print("APRA smoke test...")
    
    # Create dummy updates
    updates = [np.random.randn(100) for _ in range(10)]
    
    # Test sketching
    sk = random_projection_sketch(updates[0], sketch_dim=16, seed=42)
    print(f"Sketch shape: {sk.shape}, norm: {np.linalg.norm(sk)}")
    
    # Test aggregation
    ctx = APRAContext(sketch_dim=16, robust_method='trimmed_mean', byzantine_fraction=0.1)
    agg, meta = ctx.aggregate(updates, seed=42)
    print(f"Aggregated shape: {agg.shape}")
    print(f"Metadata: {meta}")
    print("✓ Smoke test passed")
