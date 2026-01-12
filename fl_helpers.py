"""Helper utilities for the privacy-enhanced FL notebook.

Provides:
- A consolidated AdaptivePrivacyShield implementation (safe, documented).
- compute_image_metrics: MSE and SSIM (SSIM uses skimage if available, otherwise falls back).
- compute_bootstrap_ci: bootstrap confidence intervals.
- run_multi_seed_experiment: run an experiment function across seeds and compute summary stats.
- evaluate_on_heldout: convenience wrapper to evaluate a Keras model on a provided test set.

Usage (from notebook):
from fl_helpers import AdaptivePrivacyShield, compute_image_metrics, compute_bootstrap_ci, run_multi_seed_experiment, evaluate_on_heldout

"""

from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import time
import tensorflow as tf
from health_monitor import probe_sketch_cost
from privacy_accounting import ledger

# Try to import SSIM from skimage, else provide a fallback
try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


class AdaptivePrivacyShield:
    """Canonical Adaptive Privacy Shield implementation.

    Methods:
    - assess_data_sensitivity(gradients)
    - calculate_attack_risk(gradient_list, round_num)
    - update_privacy_budget(sensitivity, attack_risk, base_epsilon)
    - apply_hybrid_protection(gradients, epsilon)

    This implementation is intentionally self-contained and conservative.
    It focuses on numeric stability and clear inputs/outputs so it can be
    used from the notebook or as a module.
    """

    def __init__(self, default_epsilon: float = 1.0, min_epsilon: float = 0.01, max_epsilon: float = 10.0):
        self.default_epsilon = float(default_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.max_epsilon = float(max_epsilon)
        self.round_history: List[float] = []
        self.round_window = 5

    def assess_data_sensitivity(self, gradients: List[np.ndarray]) -> float:
        grads = [g if isinstance(g, np.ndarray) else np.asarray(g) for g in gradients]
        # Guard against empty
        if not grads:
            return 0.0
        total_variance = float(np.mean([np.var(g) for g in grads]))
        max_vals = [float(np.max(np.abs(g))) if g.size > 0 else 0.0 for g in grads]
        outlier_score = float(np.mean([np.sum(np.abs(g) > (np.std(g) * 2)) / (g.size + 1e-12) for g in grads]))
        sensitivity = 0.4 * np.log1p(total_variance) + 0.4 * (np.mean(max_vals) / (1.0 + np.mean(max_vals))) + 0.2 * outlier_score
        return float(np.clip(sensitivity, 0.0, 1.0))

    def calculate_attack_risk(self, gradient_list: List[List[np.ndarray]], round_num: int) -> float:
        if not gradient_list:
            return 0.5
        # convert to numpy
        grad_list = [[np.asarray(g) for g in grads] for grads in gradient_list]
        similarities: List[float] = []
        for i in range(len(grad_list)):
            for j in range(i + 1, len(grad_list)):
                try:
                    flat1 = np.concatenate([g.flatten() for g in grad_list[i]])
                    flat2 = np.concatenate([g.flatten() for g in grad_list[j]])
                    # numeric safeguards
                    flat1 = np.where(np.isfinite(flat1), flat1, 0.0)
                    flat2 = np.where(np.isfinite(flat2), flat2, 0.0)
                    n1 = np.linalg.norm(flat1) + 1e-12
                    n2 = np.linalg.norm(flat2) + 1e-12
                    sim = float(np.dot(flat1, flat2) / (n1 * n2))
                    similarities.append(abs(np.clip(sim, -1.0, 1.0)))
                except Exception:
                    continue
        if not similarities:
            return 0.5
        avg_sim = float(np.mean(similarities))
        temporal_risk = float(np.mean(self.round_history[-self.round_window:])) if self.round_history else 0.5
        risk = 0.7 * avg_sim + 0.3 * temporal_risk
        self.round_history.append(risk)
        if len(self.round_history) > self.round_window:
            self.round_history.pop(0)
        return float(np.clip(risk, 0.0, 1.0))

    def update_privacy_budget(self, sensitivity: float, attack_risk: float, base_epsilon: float) -> float:
        epsilon_scale = (1.0 - float(sensitivity)) * (1.0 - float(attack_risk))
        epsilon = float(base_epsilon) * (0.5 + epsilon_scale)
        return float(np.clip(epsilon, self.min_epsilon, self.max_epsilon))

    def apply_hybrid_protection(self, gradients: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
        protected = []
        for g in gradients:
            arr = np.asarray(g)
            # simple sensitivity mask: high magnitude entries are "sensitive"
            std = float(np.std(arr)) + 1e-12
            sensitive_mask = np.abs(arr) > (2.0 * std)
            eps_noise = 0.7 * float(epsilon)
            eps_clip = 0.3 * float(epsilon)
            noise_scale = np.where(sensitive_mask, 1.0 / (eps_noise + 1e-12), 0.3 / (eps_noise + 1e-12))
            # Laplace noise (vectorized)
            noise = np.random.laplace(0.0, noise_scale, arr.shape).astype(arr.dtype)
            clip_threshold = np.where(sensitive_mask, 1.0 / (eps_clip + 1e-12), 3.0 / (eps_clip + 1e-12))
            clipped = np.clip(arr, -clip_threshold, clip_threshold)
            protected.append((clipped + noise).astype(arr.dtype))
        return protected


def protector_apply_fixed_epsilon(weights: List[np.ndarray], epsilon: float = 1.0) -> List[np.ndarray]:
    """Apply a simple fixed-epsilon protection to a list of weight arrays.

    This is a lightweight, deterministic simulation: clip each weight array to a
    threshold and add Laplace noise scaled by epsilon. Returns new weight list.
    """
    out = []
    for w in weights:
        arr = np.asarray(w)
        clip = 1.0 / (epsilon + 1e-12)
        clipped = np.clip(arr, -clip, clip)
        noise = np.random.laplace(0.0, scale=0.5 / (epsilon + 1e-12), size=arr.shape).astype(arr.dtype)
        out.append((clipped + noise).astype(arr.dtype))
    return out


def protector_apply_dpsgd(weights: List[np.ndarray], epsilon: float = 1.0, noise_multiplier: float = 1.0, clip_norm: float = 1.0) -> List[np.ndarray]:
    """Simulate a DPSGD-style protection on model weights.

    For this lightweight simulation we:
    - Clip each weight array elementwise to +-clip_norm
    - Add Gaussian noise with std = noise_multiplier * clip_norm / (epsilon + tiny)

    Note: This is a simulation for experimental comparison only.
    """
    out = []
    for w in weights:
        arr = np.asarray(w)
        # Preserve original dtype for returned arrays
        orig_dtype = arr.dtype
        # Do computations in a floating type (prefer float32 for TF models)
        working_dtype = np.float32 if np.issubdtype(orig_dtype, np.floating) else np.float32
        arr_f = arr.astype(working_dtype)
        # Clip elementwise to clip_norm then add gaussian noise
        clipped = np.clip(arr_f, -clip_norm, clip_norm)
        std = float(noise_multiplier * clip_norm / (max(epsilon, 1e-12)))
        noise = np.random.normal(loc=0.0, scale=std, size=arr.shape).astype(working_dtype)
        # Recast back to original dtype before returning
        res = (clipped + noise).astype(orig_dtype)
        out.append(res)
    return out


def protector_apply_secure_aggregation(weights: List[np.ndarray], noise_scale: float = 0.0, group_size: int = 5) -> List[np.ndarray]:
    """Simulate secure aggregation post-processing on model weights.

    This lightweight simulation optionally adds small gaussian noise to emulate
    the approximate effect of aggregation noise or quantization.
    """
    out = []
    for w in weights:
        arr = np.asarray(w)
        if noise_scale > 0.0:
            noise = np.random.normal(0.0, noise_scale, size=arr.shape).astype(arr.dtype)
            out.append((arr + noise).astype(arr.dtype))
        else:
            out.append(arr.copy())
    return out


def apply_protector_by_name(name: str, weights: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """Dispatcher to apply a named protector to model weights.

    Supported names: 'none', 'fixed', 'dpsgd', 'secure_agg', 'aps'
    For 'aps', caller should pass 'aps_instance' keyword with an AdaptivePrivacyShield instance.
    """
    name = (name or 'none').lower()
    if name == 'none':
        return [w.copy() for w in weights]
    if name == 'fixed':
        return protector_apply_fixed_epsilon(weights, epsilon=float(kwargs.get('epsilon', 1.0)))
    if name == 'dpsgd':
        return protector_apply_dpsgd(weights, epsilon=float(kwargs.get('epsilon', 1.0)), noise_multiplier=float(kwargs.get('noise_multiplier', 1.0)), clip_norm=float(kwargs.get('clip_norm', 1.0)))
    if name == 'secure_agg':
        return protector_apply_secure_aggregation(weights, noise_scale=float(kwargs.get('noise_scale', 0.0)), group_size=int(kwargs.get('group_size', 5)))
    if name == 'aps':
        aps_inst = kwargs.get('aps_instance')
        eps = float(kwargs.get('epsilon', 1.0))
        if aps_inst is None:
            # if no instance provided, instantiate a default one
            aps_inst = AdaptivePrivacyShield(default_epsilon=eps)
        return aps_inst.apply_hybrid_protection(weights, eps)
    # fallback: identity
    return [w.copy() for w in weights]


def compute_image_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
    """Compute image-level metrics between original and reconstructed arrays.

    original and reconstructed are arrays of shape (N, H, W[, C]).
    Returns mean MSE and mean SSIM (if available) and per-sample arrays.
    """
    orig = np.asarray(original)
    recon = np.asarray(reconstructed)
    if orig.shape != recon.shape:
        raise ValueError(f"Original shape {orig.shape} and reconstructed shape {recon.shape} must match")
    # flatten channels when computing MSE
    n = orig.shape[0]
    mses = np.mean((orig - recon) ** 2, axis=tuple(range(1, orig.ndim)))
    mean_mse = float(np.mean(mses))
    mean_ssim = None
    ssims = None
    if _HAS_SKIMAGE:
        ssims = []
        for i in range(n):
            try:
                if orig.ndim == 4 and orig.shape[-1] == 3:
                    # color
                    s = ssim(orig[i], recon[i], multichannel=True, data_range=orig[i].max() - orig[i].min() + 1e-12)
                else:
                    s = ssim(orig[i].squeeze(), recon[i].squeeze(), data_range=orig[i].max() - orig[i].min() + 1e-12)
                ssims.append(float(s))
            except Exception:
                ssims.append(float('nan'))
        mean_ssim = float(np.nanmean(ssims)) if ssims else None
    return {
        'mean_mse': mean_mse,
        'per_sample_mse': mses,
        'mean_ssim': mean_ssim,
        'per_sample_ssim': ssims,
    }


def compute_bootstrap_ci(data: List[float], num_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
    data = np.asarray(data)
    if data.size == 0:
        return (0.0, 0.0)
    if data.size == 1:
        return (float(data[0]), float(data[0]))
    boot_means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=data.size, replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, (1.0 - confidence) * 100.0 / 2.0))
    upper = float(np.percentile(boot_means, 100.0 - (1.0 - confidence) * 100.0 / 2.0))
    return (lower, upper)


def run_multi_seed_experiment(
    run_fn: Callable[[int], Dict[str, Any]],
    seeds: List[int],
    metrics: List[str] = ('utility', 'privacy', 'overhead'),
    num_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Run a provided run_fn over multiple random seeds.

    run_fn(seed) -> dictionary with keys matching metrics (each numeric or list).
    Returns a summary dict with mean/std/ci for each metric.
    """
    all_results: Dict[str, List[float]] = {m: [] for m in metrics}
    per_seed_outputs = []
    for seed in seeds:
        np.random.seed(int(seed))
        t0 = time.time()
        out = run_fn(int(seed))
        elapsed = time.time() - t0
        per_seed_outputs.append({'seed': seed, 'output': out, 'time': elapsed})
        for m in metrics:
            val = out.get(m)
            if isinstance(val, (list, tuple, np.ndarray)):
                # collapse to scalar if list (mean)
                try:
                    s = float(np.mean(val))
                except Exception:
                    s = float('nan')
            else:
                try:
                    s = float(val)
                except Exception:
                    s = float('nan')
            all_results[m].append(s)

    summary = {}
    for m in metrics:
        arr = all_results.get(m, [])
        if len(arr) == 0:
            summary[m] = {'mean': float('nan'), 'std': float('nan'), 'ci': (float('nan'), float('nan'))}
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            ci = compute_bootstrap_ci(arr, num_bootstrap=num_bootstrap) if len(arr) > 1 else (mean, mean)
            summary[m] = {'mean': mean, 'std': std, 'ci': ci}

    return {'per_seed': per_seed_outputs, 'summary': summary}


def evaluate_on_heldout(model, x_test: np.ndarray, y_test: np.ndarray, sample_limit: int = 1000) -> Dict[str, float]:
    """Evaluate a Keras model on an explicit hold-out set with sampling fallback.

    Returns a dict with 'loss' and 'accuracy' (accuracy may be nan if not returned by model.evaluate).
    """
    try:
        if x_test.shape[0] > sample_limit:
            x_eval = x_test[:sample_limit]
            y_eval = y_test[:sample_limit]
        else:
            x_eval = x_test
            y_eval = y_test

        # Primary evaluation attempt
        res = model.evaluate(x_eval, y_eval, verbose=0)

        # Parse evaluate() result robustly
        loss = float('nan')
        acc = float('nan')
        if isinstance(res, (list, tuple, np.ndarray)):
            try:
                loss = float(res[0])
            except Exception:
                loss = float('nan')
            if len(res) > 1:
                try:
                    acc = float(res[1])
                except Exception:
                    acc = float('nan')
        else:
            try:
                loss = float(res)
            except Exception:
                loss = float('nan')

        # If evaluate did not return an accuracy (or returned non-finite), fall back to predict+argmax
        need_fallback = False
        try:
            if not np.isfinite(acc):
                need_fallback = True
            else:
                # In some setups evaluate returns a scalar loss and no accuracy; detect this
                if not isinstance(res, (list, tuple, np.ndarray)) or len(res) < 2:
                    need_fallback = True
                # Also fallback if accuracy is exactly 0.0 but the model was not compiled with accuracy
                elif hasattr(model, 'metrics_names') and 'accuracy' not in getattr(model, 'metrics_names', []) and 'acc' not in getattr(model, 'metrics_names', []):
                    # if evaluate returned an accuracy value but the model metrics don't include accuracy name,
                    # it's safer to recompute via predict to avoid misleading zeros
                    if float(acc) == 0.0:
                        need_fallback = True
        except Exception:
            need_fallback = True

        if need_fallback:
            try:
                # Predict-based accuracy fallback
                n = min(len(x_eval), sample_limit)
                preds = model.predict(x_eval[:n], batch_size=128, verbose=0)
                preds = np.asarray(preds)
                if preds.ndim > 1 and preds.shape[1] > 1:
                    pred_labels = np.argmax(preds, axis=1)
                    true_labels = np.argmax(y_eval[:n], axis=1) if y_eval.ndim > 1 else y_eval[:n].flatten()
                else:
                    pred_labels = (preds.flatten() > 0.5).astype(int)
                    true_labels = np.argmax(y_eval[:n], axis=1) if y_eval.ndim > 1 else y_eval[:n].flatten()
                acc = float(np.mean(pred_labels == true_labels))
            except Exception:
                # leave acc as nan if fallback fails
                acc = float('nan')

        return {'loss': loss, 'accuracy': acc}
    except Exception:
        return {'loss': float('nan'), 'accuracy': float('nan')}


def safe_evaluate(model, x_test: np.ndarray, y_test: np.ndarray, sample_limit: int = 1000) -> Dict[str, float]:
    """Backward-compatible alias for evaluate_on_heldout.

    Kept for readability in calling code that expects a ``safe_evaluate`` symbol.
    """
    return evaluate_on_heldout(model, x_test, y_test, sample_limit=sample_limit)


def safe_accuracy(model, x_test: np.ndarray, y_test: np.ndarray, sample_limit: int = 1000) -> float:
    """Return a single float accuracy using the robust evaluate->predict fallback.

    This convenience function returns a plain float (nan when not computable).
    """
    res = evaluate_on_heldout(model, x_test, y_test, sample_limit=sample_limit)
    try:
        acc = float(res.get('accuracy', float('nan')))
    except Exception:
        acc = float('nan')
    return acc


def membership_inference_via_loss(
    model,
    x_member: np.ndarray,
    y_member: np.ndarray,
    x_nonmember: np.ndarray,
    y_nonmember: np.ndarray,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Perform a simple membership inference attack using per-sample loss.

    The attacker scores examples by negative loss (lower loss => more likely member).
    We compute an empirical AUC using ranks so we avoid adding sklearn as a dependency.

    Returns a dict with:
    - 'auc': ROC AUC value (0.0-1.0, 0.5 = random)
    - 'member_loss_mean', 'nonmember_loss_mean'
    """
    try:
        # helper to get per-sample loss (works for categorical_crossentropy and similar)
        loss_fn = tf.keras.losses.get(model.loss)
    except Exception:
        # fallback to categorical_crossentropy
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def _per_sample_losses(x, y):
        try:
            # predict logits/probs then compute per-sample loss via loss_fn
            preds = model.predict(x, batch_size=batch_size, verbose=0)
            # convert to tensors
            t_preds = tf.convert_to_tensor(preds)
            t_y = tf.convert_to_tensor(y)
            losses = loss_fn(t_y, t_preds)
            return np.asarray(losses).astype(float)
        except Exception:
            # as a fallback, try model(x) path
            try:
                t_preds = model(x, training=False)
                t_y = tf.convert_to_tensor(y)
                losses = loss_fn(t_y, t_preds)
                return np.asarray(losses).astype(float)
            except Exception:
                # cannot compute
                return np.array([])

    m_losses = _per_sample_losses(x_member, y_member)
    nm_losses = _per_sample_losses(x_nonmember, y_nonmember)

    # If either is empty, return nan
    if m_losses.size == 0 or nm_losses.size == 0:
        return {'auc': float('nan'), 'member_loss_mean': float(np.nan), 'nonmember_loss_mean': float(np.nan)}

    # For membership inference, smaller loss => more likely member. Use score = -loss
    pos_scores = -m_losses  # positive class = members
    neg_scores = -nm_losses

    # compute AUC via rank-sum method (Mann-Whitney U relation)
    try:
        all_scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        # argsort descending so higher score -> higher rank
        order = np.argsort(-all_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(all_scores) + 1)
        # sum ranks for positive class
        sum_ranks_pos = np.sum(ranks[labels == 1])
        n1 = float(pos_scores.size)
        n2 = float(neg_scores.size)
        if n1 <= 0 or n2 <= 0:
            auc = float('nan')
        else:
            # U statistic for positive class
            U1 = sum_ranks_pos - (n1 * (n1 + 1.0) / 2.0)
            auc = U1 / (n1 * n2)
    except Exception:
        auc = float('nan')

    return {'auc': float(auc), 'member_loss_mean': float(np.mean(m_losses)), 'nonmember_loss_mean': float(np.mean(nm_losses))}


def shadow_model_membership_attack(
    model_fn: Callable[[], Any],
    full_train: Tuple[np.ndarray, np.ndarray],
    full_holdout: Tuple[np.ndarray, np.ndarray],
    member_examples: Tuple[np.ndarray, np.ndarray],
    nonmember_examples: Tuple[np.ndarray, np.ndarray],
    num_shadows: int = 4,
    shadow_size: int = 500,
    shadow_epochs: int = 2,
    attacker_epochs: int = 10,
    top_k: int = 3,
    batch_size: int = 128,
    debug: bool = False,
) -> Dict[str, Any]:
    """A lightweight shadow-model membership inference attack.

    - model_fn: callable returning a compiled Keras model (shadow models will be trained with it)
    - full_train: (x_train, y_train) used as the data pool for shadow training
    - full_holdout: (x_holdout, y_holdout) used as non-member pool for shadow training
    - member_examples/nonmember_examples: data for the target model evaluation (the victim)

    This trains num_shadows small shadow models over random subsets and trains a small attacker
    network on concatenated shadow outputs (top_k confidences + loss) to predict membership.

    Returns { 'auc': float, 'attacker_history': history_dict }
    """
    try:
        x_pool, y_pool = full_train
        x_hold, y_hold = full_holdout
        sx, sy = member_examples
        nx, ny = nonmember_examples
    except Exception:
        return {'auc': float('nan'), 'error': 'bad_inputs'}

    attacker_X = []
    attacker_y = []

    # loss function for computing per-sample loss
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    rng = np.random.RandomState(1234)
    n_pool = x_pool.shape[0]
    n_hold = x_hold.shape[0]

    trained_shadows = 0
    for s in range(int(num_shadows)):
        # sample shadow train and shadow non-member
        idx = rng.choice(n_pool, size=min(shadow_size, n_pool), replace=False)
        idx2 = rng.choice(n_hold, size=min(shadow_size, n_hold), replace=False)
        sx_train, sy_train = x_pool[idx], y_pool[idx]
        sx_non, sy_non = x_hold[idx2], y_hold[idx2]

        shadow = model_fn()
        # ensure shadow is compiled with a sensible metric for consistency
        try:
            mnames = getattr(shadow, 'metrics_names', None)
            if not mnames or ('accuracy' not in mnames and 'acc' not in mnames):
                try:
                    shadow.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                except Exception:
                    # ignore compile failures and attempt to fit anyway
                    pass
        except Exception:
            pass

        try:
            shadow.fit(sx_train, sy_train, epochs=shadow_epochs, batch_size=batch_size, verbose=0)
            trained_shadows += 1
            if debug:
                # report some shadow stats
                try:
                    preds_s = shadow.predict(sx_train[:100], batch_size=128, verbose=0)
                    preds_n = shadow.predict(sx_non[:100], batch_size=128, verbose=0)
                    if isinstance(preds_s, np.ndarray) and preds_s.size:
                        print(f"[DEBUG] shadow {s}: preds_s.shape={preds_s.shape}, sample_top1={np.argmax(preds_s[:5],axis=1).tolist()}")
                    if isinstance(preds_n, np.ndarray) and preds_n.size:
                        print(f"[DEBUG] shadow {s}: preds_n.shape={preds_n.shape}, sample_top1_non={np.argmax(preds_n[:5],axis=1).tolist()}")
                except Exception:
                    pass
        except Exception:
            # if training fails, skip this shadow
            if debug:
                print(f"[DEBUG] shadow {s} training failed, skipping")
            continue

        # get predictions and losses on member/non-member shadow sets
        preds_mem = shadow.predict(sx_train, batch_size=batch_size, verbose=0)
        preds_non = shadow.predict(sx_non, batch_size=batch_size, verbose=0)
        losses_mem = np.asarray(loss_fn(tf.convert_to_tensor(sy_train), tf.convert_to_tensor(preds_mem))).astype(float)
        losses_non = np.asarray(loss_fn(tf.convert_to_tensor(sy_non), tf.convert_to_tensor(preds_non))).astype(float)

        if debug:
            try:
                print(f"[DEBUG] shadow {s}: losses_mem.mean={float(np.mean(losses_mem)) if losses_mem.size else 'nan'}, losses_non.mean={float(np.mean(losses_non)) if losses_non.size else 'nan'}")
            except Exception:
                pass

        # features: top_k confidences (sorted descending) and loss
        def topk_feats(probs):
            # get top_k probabilities for each sample
            sorted_idx = np.argsort(-probs, axis=1)
            topk = np.take_along_axis(probs, sorted_idx[:, :top_k], axis=1)
            return topk

        mem_feats = topk_feats(preds_mem)
        non_feats = topk_feats(preds_non)

        for i in range(mem_feats.shape[0]):
            row = np.concatenate([mem_feats[i], [losses_mem[i]]])
            attacker_X.append(row)
            attacker_y.append(1.0)
        for i in range(non_feats.shape[0]):
            row = np.concatenate([non_feats[i], [losses_non[i]]])
            attacker_X.append(row)
            attacker_y.append(0.0)

    attacker_X = np.asarray(attacker_X)
    attacker_y = np.asarray(attacker_y)

    if attacker_X.shape[0] < 10:
        return {'auc': float('nan'), 'error': 'not_enough_shadow_data', 'trained_shadows': int(trained_shadows)}

    # shuffle and train a tiny attacker network
    perm = rng.permutation(attacker_X.shape[0])
    attacker_X = attacker_X[perm]
    attacker_y = attacker_y[perm]

    # simple attacker model
    inp_dim = attacker_X.shape[1]
    atk = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(inp_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    atk.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # train attacker briefly
    try:
        hist = atk.fit(attacker_X, attacker_y, epochs=attacker_epochs, batch_size=256, verbose=0)
    except Exception:
        return {'auc': float('nan'), 'error': 'attacker_train_fail', 'trained_shadows': int(trained_shadows)}

    # report attacker training AUC if available
    attacker_auc = float('nan')
    try:
        ev = atk.evaluate(attacker_X, attacker_y, verbose=0)
        # ev may be [loss, AUC]
        if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > 1:
            attacker_auc = float(ev[1])
        elif isinstance(ev, (float, int)):
            attacker_auc = float(ev)
    except Exception:
        attacker_auc = float('nan')

    if debug:
        try:
            print(f"[DEBUG] attacker trained on {attacker_X.shape[0]} examples; attacker_auc={attacker_auc}")
        except Exception:
            pass

    # prepare target features
    try:
        # If examples are (x,y) arrays, use the victim model to predict
        if isinstance(member_examples[0], np.ndarray) and member_examples[0].ndim == 2 and member_examples[0].shape[1] > 1:
            mem_preds = member_examples[0]
            non_preds = nonmember_examples[0]
        else:
            victim = model_fn()
            mem_preds = victim.predict(member_examples[0], batch_size=batch_size, verbose=0)
            non_preds = victim.predict(nonmember_examples[0], batch_size=batch_size, verbose=0)
    except Exception:
        return {'auc': float('nan'), 'error': 'victim_pred_fail'}

    # compute per-sample losses for victim examples
    try:
        mem_losses = np.asarray(loss_fn(tf.convert_to_tensor(member_examples[1]), tf.convert_to_tensor(mem_preds))).astype(float)
        non_losses = np.asarray(loss_fn(tf.convert_to_tensor(nonmember_examples[1]), tf.convert_to_tensor(non_preds))).astype(float)
    except Exception:
        mem_losses = np.zeros(mem_preds.shape[0])
        non_losses = np.zeros(non_preds.shape[0])

    def topk_feats_from_preds(probs):
        sorted_idx = np.argsort(-probs, axis=1)
        return np.take_along_axis(probs, sorted_idx[:, :top_k], axis=1)

    mem_feats = topk_feats_from_preds(mem_preds)
    non_feats = topk_feats_from_preds(non_preds)

    X_target = np.vstack([np.hstack([mem_feats, mem_losses.reshape(-1, 1)]), np.hstack([non_feats, non_losses.reshape(-1, 1)])])
    y_target = np.concatenate([np.ones(mem_feats.shape[0]), np.zeros(non_feats.shape[0])])

    # attacker scores
    scores = atk.predict(X_target, batch_size=batch_size, verbose=0).reshape(-1)

    # compute AUC via rank-sum
    try:
        pos_scores = scores[y_target == 1]
        neg_scores = scores[y_target == 0]
        all_scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        order = np.argsort(-all_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(all_scores) + 1)
        sum_ranks_pos = np.sum(ranks[labels == 1])
        n1 = float(pos_scores.size)
        n2 = float(neg_scores.size)
        U1 = sum_ranks_pos - (n1 * (n1 + 1.0) / 2.0)
        auc = U1 / (n1 * n2) if n1 > 0 and n2 > 0 else float('nan')
    except Exception:
        auc = float('nan')

    return {'auc': float(auc), 'attacker_history': { 'trained_shadows': int(trained_shadows), 'attacker_epochs': int(attacker_epochs), 'attacker_auc': attacker_auc }}


# --------------------- APRA / Robust + Private utilities ---------------------
def _flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of numpy arrays into a single 1-D vector."""
    return np.concatenate([np.asarray(w).ravel() for w in weights]) if weights else np.array([])


def random_projection_sketch(weights: List[np.ndarray], sketch_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Compute a memory-efficient random-projection sketch using sparse projections.

    Uses sparse random projections (Johnson-Lindenstrauss style) to avoid allocating
    large dense matrices. Returns a normalized sketch vector of length ``sketch_dim``.
    """
    vec = _flatten_weights(weights)
    if vec.size == 0:
        return np.zeros((sketch_dim,), dtype=np.float32)

    rng = np.random.RandomState(int(seed))
    vec_f32 = vec.astype(np.float32)

    # Chunked random projection: avoid allocating a (D x k) matrix by projecting
    # the flattened vector in manageable chunks and accumulating results.
    n = vec_f32.size
    sketch = np.zeros((sketch_dim,), dtype=np.float32)
    scale = 1.0 / np.sqrt(float(max(1, n)))
    chunk_size = 10000
    for i in range(0, n, chunk_size):
        c = vec_f32[i:i+chunk_size]
        # small dense projection for the chunk only (chunk_size x sketch_dim)
        proj = rng.normal(loc=0.0, scale=scale, size=(c.size, sketch_dim)).astype(np.float32)
        sketch += c.dot(proj)

    # Normalize and return
    norm = np.linalg.norm(sketch) + 1e-12
    return (sketch / norm).astype(np.float32)


def ensemble_sketch(weights: List[np.ndarray], sketch_dim: int = 128, n_sketches: int = 3, seed: int = 0) -> np.ndarray:
    """Return concatenated ensemble of multiple random-projection sketches.

    This reduces variance in the sketch representation and improves detection.
    """
    if n_sketches <= 1:
        return random_projection_sketch(weights, sketch_dim=sketch_dim, seed=seed)
    sketches = [random_projection_sketch(weights, sketch_dim=sketch_dim, seed=seed + i) for i in range(n_sketches)]
    return np.concatenate(sketches, axis=0)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel().astype(np.float32)
    b = np.asarray(b).ravel().astype(np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    sim = float(np.dot(a, b) / (na * nb))
    # distance in [0,2]
    return float(1.0 - sim)


def apra_detect_outliers(sketches: List[np.ndarray], z_thresh: float = 3.0, metric: str = 'euclidean') -> List[bool]:
    """Detect outlier sketches using median+MAD thresholding with optional metric.

    metric: 'euclidean' or 'cosine'
    Returns True for benign sketches.
    """
    if not sketches:
        return []
    S = np.vstack([np.asarray(s).ravel() for s in sketches])
    if metric == 'cosine':
        # compute distance-to-median using cosine distance
        med = np.median(S, axis=0)
        dists = np.array([_cosine_distance(row, med) for row in S])
    else:
        med = np.median(S, axis=0)
        dists = np.linalg.norm(S - med[None, :], axis=1)
    med_dist = np.median(dists)
    mad = np.median(np.abs(dists - med_dist)) + 1e-12
    threshold = med_dist + z_thresh * mad
    good = (dists <= threshold).tolist()
    return good


def federated_trimmed_mean(local_weights_list: List[List[np.ndarray]], trim_fraction: float = 0.2) -> List[np.ndarray]:
    """Compute coordinate-wise trimmed mean across clients for each layer.

    trim_fraction: fraction to trim from each tail (0.2 -> trim 20% smallest and 20% largest)
    """
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
        # sort along client axis
        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[k:n_clients - k, ...]
        out.append(np.mean(trimmed, axis=0).astype(layer_vals[0].dtype))
    return out


def federated_median(local_weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Compute coordinate-wise median aggregation across clients."""
    if not local_weights_list:
        return []
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        med = np.median(stacked, axis=0)
        out.append(med.astype(layer_vals[0].dtype))
    return out


def apra_aggregate(local_weights_list: List[List[np.ndarray]], sketch_dim: int = 128, trim_fraction: float = 0.2, z_thresh: float = 3.0, seed: int = 0) -> List[np.ndarray]:
    """APRA aggregation: detect outliers from sketches and aggregate benign client updates.

    Steps:
    1. Compute random-projection sketches per client.
    2. Detect outliers via median+MAD on sketch distances.
    3. If enough benign clients remain, average their weights; otherwise fall back to trimmed mean.
    """
    n = len(local_weights_list)
    if n == 0:
        return []
    # default uses an ensemble of small sketches
    sketches = [ensemble_sketch(w, sketch_dim=sketch_dim, n_sketches=2, seed=seed + i) for i, w in enumerate(local_weights_list)]
    benign_mask = apra_detect_outliers(sketches, z_thresh=z_thresh, metric='cosine')
    benign_indices = [i for i, ok in enumerate(benign_mask) if ok]
    # require at least ceil(50% + 1) clients to proceed with simple averaging
    if len(benign_indices) >= max(1, int(np.ceil(0.5 * n))):
        selected = [local_weights_list[i] for i in benign_indices]
        # plain federated averaging across selected
        avg = []
        for layer_vals in zip(*selected):
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            avg.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
        return avg
    # fallback: trimmed mean
    return federated_trimmed_mean(local_weights_list, trim_fraction=trim_fraction)


def per_layer_ensemble_sketch(weights: List[np.ndarray], sketch_dim: int = 64, n_sketches: int = 2, seed: int = 0) -> np.ndarray:
    """Compute concatenated per-layer ensemble sketches.

    For each layer in weights we compute a small random-projection sketch and
    concatenate them. This preserves per-layer signal used by apra_weighted_aggregate.
    """
    parts = []
    for li, layer in enumerate(weights):
        s = random_projection_sketch([layer], sketch_dim=sketch_dim, seed=seed + li)
        if n_sketches > 1:
            extra = [random_projection_sketch([layer], sketch_dim=sketch_dim, seed=seed + li + j + 1) for j in range(n_sketches - 1)]
            s = np.concatenate([s] + extra, axis=0)
        parts.append(s)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def apra_weighted_aggregate(local_weights_list: List[List[np.ndarray]], sketch_dim_per_layer: int = 32, n_sketches: int = 2, z_thresh: float = 3.0, eps: float = 1e-6, seed: int = 0) -> List[np.ndarray]:
    """Weighted aggregation using per-layer sketches to compute trust scores.

    Steps:
    - Compute per-client per-layer sketches and a concatenated sketch per client.
    - Compute distances to the median sketch (cosine distance) and convert to trust
      weights via an inverse-distance mapping.
    - Use trust weights to compute a weighted average of weights per coordinate.
    """
    n = len(local_weights_list)
    if n == 0:
        return []
    sketches = [per_layer_ensemble_sketch(w, sketch_dim=sketch_dim_per_layer, n_sketches=n_sketches, seed=seed + i) for i, w in enumerate(local_weights_list)]
    S = np.vstack([s.ravel() for s in sketches])
    med = np.median(S, axis=0)
    # cosine distances
    dists = np.array([1.0 - float(np.dot(s.ravel(), med) / ((np.linalg.norm(s) + 1e-12) * (np.linalg.norm(med) + 1e-12))) for s in sketches])
    inv = 1.0 / (dists + eps)
    weights = inv / np.sum(inv)

    # weighted aggregation per-layer
    out = []
    for layer_vals in zip(*local_weights_list):
        stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
        orig_shape = stacked.shape[1:]
        flat = stacked.reshape((stacked.shape[0], -1))
        weighted = np.tensordot(weights, flat, axes=(0, 0))
        out.append(weighted.reshape(orig_shape).astype(layer_vals[0].dtype))
    return out


def aggregate_dispatcher(name: str, local_weights_list: List[List[np.ndarray]], **kwargs):
    """Common dispatcher to call different aggregators by name.

    Supported names: 'apra', 'apra_weighted', 'farpa', 'capra', 'trimmed_mean', 'median', 'mean'
    Returns a tuple (aggregated_weights, meta) where `meta` may be None or
    a dict for aggregators that return extra info (e.g., FARPA returns trust/debug).
    """
    name = (name or '').lower()
    if name in ('apra', 'apra_aggregate'):
        # apra_aggregate returns weights only
        agg = apra_aggregate(local_weights_list, sketch_dim=kwargs.get('sketch_dim', 128), z_thresh=kwargs.get('z_thresh', 3.0), seed=kwargs.get('seed', 0))
        return agg, None
    if name in ('apra_weighted', 'apra-weighted'):
        agg = apra_weighted_aggregate(local_weights_list, sketch_dim_per_layer=kwargs.get('sketch_dim_per_layer', 32), n_sketches=kwargs.get('n_sketches', 2), z_thresh=kwargs.get('z_thresh', 3.0), eps=kwargs.get('eps', 1e-6), seed=kwargs.get('seed', 0))
        return agg, None
    if name in ('farpa', 'farpa_aggregate'):
        agg, trust, debug = farpa_aggregate(
            local_weights_list,
            sketch_dim_per_layer=kwargs.get('sketch_dim_per_layer', 32),
            n_sketches=kwargs.get('n_sketches', 2),
            eps_sketch=kwargs.get('eps_sketch', 0.5),
            delta=kwargs.get('delta', 1e-5),
            sketch_noise_mech=kwargs.get('sketch_noise_mech', 'laplace'),
            z_thresh=kwargs.get('z_thresh', 3.0),
            fallback=kwargs.get('fallback', 'trimmed_mean'),
            seed=kwargs.get('seed', 0),
            aps_instance=kwargs.get('aps_instance', None),
            adapt_with_aps=kwargs.get('adapt_with_aps', True),
        )
        return agg, {'trust_scores': trust, 'debug': debug}
    if name in ('capra', 'capra_aggregate'):
        # CAPRA wrapper: adaptive fast/canonical switching based on lightweight health probe
        # Expected kwargs: sketch_dim_canonical, n_sketches_canonical, sketch_dim_fast, n_sketches_fast,
        # eps_sketch (base for canonical), time_budget_ms, probe_samples
        try:
            agg, meta = capra_aggregate(local_weights_list, **kwargs)
            return agg, meta
        except Exception as e:
            # Fall back to FARPA if capra fails
            logger = kwargs.get('logger')
            if logger is not None:
                logger.exception('CAPRA aggregation failed, falling back to FARPA: %s', e)
            else:
                print('CAPRA aggregation failed, falling back to FARPA:', e)
            agg, trust, debug = farpa_aggregate(local_weights_list, sketch_dim_per_layer=kwargs.get('sketch_dim_per_layer', 32), n_sketches=kwargs.get('n_sketches', 2), eps_sketch=kwargs.get('eps_sketch', 0.5), delta=kwargs.get('delta', 1e-5), sketch_noise_mech=kwargs.get('sketch_noise_mech', 'laplace'), z_thresh=kwargs.get('z_thresh', 3.0), fallback=kwargs.get('fallback', 'trimmed_mean'), seed=kwargs.get('seed', 0))
            return agg, {'trust_scores': trust, 'debug': debug}
    if name in ('trimmed_mean', 'trimmed'):
        agg = federated_trimmed_mean(local_weights_list, trim_fraction=kwargs.get('trim_fraction', 0.2))
        return agg, None
    if name in ('median', 'federated_median'):
        agg = federated_median(local_weights_list)
        return agg, None
    # default: mean
    if local_weights_list and isinstance(local_weights_list[0], (list, tuple)):
        out = []
        for layer_idx in range(len(local_weights_list[0])):
            layer_vals = [w[layer_idx] for w in local_weights_list]
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            out.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
        return out, None
    else:
        stacked = np.stack([np.asarray(w, dtype=np.float64) for w in local_weights_list], axis=0)
        return np.mean(stacked, axis=0), None


def farpa_aggregate(
    local_weights_list: List[List[np.ndarray]],
    sketch_dim_per_layer: int = 32,
    n_sketches: int = 2,
    eps_sketch: float = 0.5,
    delta: float = 1e-5,
    sketch_noise_mech: str = 'laplace',
    z_thresh: float = 3.0,
    fallback: str = 'trimmed_mean',
    seed: int = 0,
    aps_instance: Optional[AdaptivePrivacyShield] = None,
    adapt_with_aps: bool = True,
) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
    """FARPA: Federated Adaptive Robust Private Aggregation.

    Steps (brief):
    1. Compute per-layer ensemble sketches per client.
    2. Apply light local-DP to sketches (Laplace/Gaussian) to protect sketch exchange.
    3. Compute noise-aware distances to median and derive trust scores.
    4. Use trust scores for weighted aggregation; fall back to trimmed mean if insufficient benign clients.

    Returns (aggregated_weights, trust_scores, debug_info)
    """
    debug: Dict[str, Any] = {}
    if not local_weights_list:
        return [], [], {'error': 'no_clients'}

    n = len(local_weights_list)
    # Compute per-client concatenated per-layer sketches
    sketches = [per_layer_ensemble_sketch(w, sketch_dim=sketch_dim_per_layer, n_sketches=n_sketches, seed=seed + i) for i, w in enumerate(local_weights_list)]
    S = np.vstack([s.ravel() for s in sketches]) if sketches else np.zeros((0,))

    # Apply local DP to sketches (simple additive noise)
    rng = np.random.RandomState(int(seed))
    noisy_sketches = []
    # Conservative per-coordinate sensitivity (assume 1.0 unless better bound provided)
    sens = 1.0
    if sketch_noise_mech.lower() == 'laplace':
        b = sens / (max(eps_sketch, 1e-12))
        for s in sketches:
            noise = rng.laplace(0.0, b, size=s.shape).astype(s.dtype)
            noisy_sketches.append((s + noise).astype(np.float32))
        noise_variance_per_coord = 2.0 * (b ** 2)
    else:
        # gaussian
        sigma = (sens * np.sqrt(2.0 * np.log(1.25 / max(delta, 1e-12)))) / max(eps_sketch, 1e-12)
        for s in sketches:
            noise = rng.normal(0.0, sigma, size=s.shape).astype(s.dtype)
            noisy_sketches.append((s + noise).astype(np.float32))
        noise_variance_per_coord = sigma ** 2

    debug['noise_variance_per_coord'] = float(noise_variance_per_coord)

    # Record per-round sketch epsilon into privacy ledger for accounting
    try:
        ledger.record(float(eps_sketch), float(delta))
    except Exception:
        pass
    # Additionally record mechanism-level params for formal RDP accounting when possible
    try:
        if sketch_noise_mech.lower() == 'laplace':
            # Laplace b parameter
            b = sens / (max(eps_sketch, 1e-12))
            ledger.record_mechanism('laplace', b=b, sensitivity=sens, steps=1)
        elif sketch_noise_mech.lower() == 'gaussian':
            # Gaussian: record all relevant parameters for formal accounting
            sigma = (sens * np.sqrt(2.0 * np.log(1.25 / max(delta, 1e-12)))) / max(eps_sketch, 1e-12)
            # We don't have explicit sampling_rate/steps in this function signature.
            # Use conservative defaults: sampling_rate=1.0 (full participation), steps=1
            sampling_rate = 1.0
            steps = 1
            ledger.record_mechanism('gaussian', sigma=sigma, sampling_rate=sampling_rate, steps=steps, sensitivity=sens)
        else:
            # Unknown/noise mechanism: record as generic
            ledger.record_mechanism(str(sketch_noise_mech).lower(), sensitivity=sens, steps=1)
    except Exception:
        pass

    # Distances to median (cosine distance by default)
    med = np.median(np.vstack([ns.ravel() for ns in noisy_sketches]), axis=0)
    dists = np.array([1.0 - float(np.dot(ns.ravel(), med) / ((np.linalg.norm(ns) + 1e-12) * (np.linalg.norm(med) + 1e-12))) for ns in noisy_sketches])

    # Estimate expected noise contribution to L2 distances (approx)
    k = noisy_sketches[0].ravel().size if noisy_sketches else 0
    expected_noise_l2_var = k * noise_variance_per_coord
    debug['expected_noise_l2_var'] = float(expected_noise_l2_var)

    # Compute median and mad of raw distances
    med_dist = np.median(dists)
    mad = np.median(np.abs(dists - med_dist)) + 1e-12

    # Adjust threshold by incorporating expected noise (convert var to STD for comparison)
    noise_std_effect = np.sqrt(expected_noise_l2_var) / (np.sqrt(max(1, k)) + 1e-12)
    # z-adjusted threshold
    threshold = med_dist + z_thresh * (mad + noise_std_effect)
    debug['threshold'] = float(threshold)
    debug['med_dist'] = float(med_dist)
    debug['mad'] = float(mad)

    benign_mask = (dists <= threshold).tolist()
    benign_indices = [i for i, ok in enumerate(benign_mask) if ok]
    debug['benign_count'] = len(benign_indices)

    # trust scores: invert normalized distance (clamp to [0,1])
    maxd = float(np.max(dists)) if dists.size else 1.0
    trust_scores = [float(1.0 - (d / (maxd + 1e-12))) for d in dists]

    # Optionally incorporate APS-derived budget/risk signals
    client_epsilons = None
    if aps_instance is not None and adapt_with_aps:
        client_epsilons = []
        for i, w in enumerate(local_weights_list):
            try:
                sens_est = aps_instance.assess_data_sensitivity([np.asarray(g) for g in w])
                risk = aps_instance.calculate_attack_risk([[np.asarray(g) for g in w]], 0)
                eps_new = aps_instance.update_privacy_budget(sens_est, risk, aps_instance.default_epsilon)
                client_epsilons.append(float(eps_new))
            except Exception:
                client_epsilons.append(float(aps_instance.default_epsilon))
        debug['client_epsilons'] = client_epsilons

    # If enough benign clients, aggregate their weights via weighted avg using trust scores limited to benign set
    if len(benign_indices) >= max(1, int(np.ceil(0.5 * n))):
        selected = [local_weights_list[i] for i in benign_indices]
        sel_weights = np.array([trust_scores[i] for i in benign_indices], dtype=np.float64)
        # normalize trust weights
        wsum = float(np.sum(sel_weights)) if sel_weights.size else 1.0
        if wsum <= 0:
            sel_weights = np.ones_like(sel_weights)
            wsum = float(np.sum(sel_weights))
        norm_weights = sel_weights / (wsum + 1e-12)

        # Weighted aggregation per-layer
        out = []
        for layer_vals in zip(*selected):
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            # flatten per-client, apply weights by tensordot
            orig_shape = stacked.shape[1:]
            flat = stacked.reshape((stacked.shape[0], -1))
            weighted = np.tensordot(norm_weights, flat, axes=(0, 0))
            out.append(weighted.reshape(orig_shape).astype(layer_vals[0].dtype))
        return out, trust_scores, debug

    # Fallback aggregator
    if fallback == 'trimmed_mean':
        agg = federated_trimmed_mean(local_weights_list, trim_fraction=0.2)
        return agg, trust_scores, debug
    elif fallback == 'median':
        agg = federated_median(local_weights_list)
        return agg, trust_scores, debug
    else:
        # default to simple average
        if isinstance(local_weights_list[0], (list, tuple)):
            out = []
            for layer_idx in range(len(local_weights_list[0])):
                layer_vals = [w[layer_idx] for w in local_weights_list]
                stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
                out.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
            return out, trust_scores, debug
        else:
            stacked = np.stack([np.asarray(w, dtype=np.float64) for w in local_weights_list], axis=0)
            return np.mean(stacked, axis=0), trust_scores, debug


# ============================================================================
# ROBUST AGGREGATION FUNCTIONS (Phase 1 extension)
# ============================================================================

def federated_median(local_weights_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Coordinate-wise median aggregation (Byzantine-robust).
    
    Args:
        local_weights_list: list of weight arrays per client
    
    Returns:
        aggregated weights (coordinate-wise median)
    """
    if not local_weights_list:
        return []
    
    # Handle per-layer
    if isinstance(local_weights_list[0], (list, tuple)):
        # Per-layer weights
        out = []
        for layer_idx in range(len(local_weights_list[0])):
            layer_vals = [w[layer_idx] for w in local_weights_list]
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            orig_shape = stacked.shape[1:]
            flat = stacked.reshape((stacked.shape[0], -1))
            median = np.median(flat, axis=0)
            out.append(median.reshape(orig_shape).astype(layer_vals[0].dtype))
        return out
    else:
        # Single array
        stacked = np.stack([np.asarray(w, dtype=np.float64) for w in local_weights_list], axis=0)
        return np.median(stacked, axis=0)


def federated_trimmed_mean(local_weights_list: List[np.ndarray], trim_fraction: float = 0.2) -> List[np.ndarray]:
    """
    Coordinate-wise trimmed mean aggregation (removes top/bottom trim_fraction).
    More robust to Byzantine than FedAvg.
    
    Args:
        local_weights_list: list of weight arrays per client
        trim_fraction: fraction to trim from each end (default 0.2 = 20%)
    
    Returns:
        aggregated weights (trimmed mean)
    """
    if not local_weights_list:
        return []
    
    num_to_trim = max(1, int(len(local_weights_list) * trim_fraction))
    
    if isinstance(local_weights_list[0], (list, tuple)):
        # Per-layer
        out = []
        for layer_idx in range(len(local_weights_list[0])):
            layer_vals = [w[layer_idx] for w in local_weights_list]
            stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
            orig_shape = stacked.shape[1:]
            flat = stacked.reshape((stacked.shape[0], -1))
            
            # Sort and trim
            sorted_flat = np.sort(flat, axis=0)
            trimmed = sorted_flat[num_to_trim:-num_to_trim, :]
            trimmed_mean = np.mean(trimmed, axis=0)
            
            out.append(trimmed_mean.reshape(orig_shape).astype(layer_vals[0].dtype))
        return out
    else:
        # Single array
        stacked = np.stack([np.asarray(w, dtype=np.float64) for w in local_weights_list], axis=0)
        sorted_stacked = np.sort(stacked, axis=0)
        trimmed = sorted_stacked[num_to_trim:-num_to_trim, :]
        return np.mean(trimmed, axis=0)


def federated_krum(local_weights_list: List[np.ndarray], num_to_exclude: int = 0) -> List[np.ndarray]:
    """
    Krum aggregation: select client update with smallest sum-of-distances to others.
    Byzantine-resilient.
    
    Args:
        local_weights_list: list of weight arrays per client
        num_to_exclude: number of worst clients to exclude (default 0 = Krum)
    
    Returns:
        selected/aggregated weights
    """
    if not local_weights_list:
        return []
    
    n = len(local_weights_list)

    # Flatten all weights. Supports both per-layer lists (list of arrays)
    # and single-array client updates. For per-layer weights we concatenate
    # per-layer flattened arrays to form the client's full vector.
    if isinstance(local_weights_list[0], (list, tuple)):
        flattened = [np.concatenate([np.asarray(layer).ravel() for layer in w]) for w in local_weights_list]
    else:
        flattened = [np.asarray(w).ravel() for w in local_weights_list]

    # Ensure all flattened vectors have same length by padding shorter ones with zeros
    max_len = max([f.size for f in flattened]) if flattened else 0
    if any(f.size != max_len for f in flattened):
        padded = []
        for f in flattened:
            if f.size < max_len:
                pf = np.zeros((max_len,), dtype=f.dtype)
                pf[:f.size] = f
                padded.append(pf)
            else:
                padded.append(f)
        flattened = padded

    # Compute pairwise L2 distances
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist = float(np.linalg.norm(flattened[i] - flattened[j], ord=2))
            except Exception:
                dist = float('inf')
            distances[i, j] = dist
            distances[j, i] = dist

    # Determine k (number of closest neighbors to sum). Krum uses k = n - f - 2
    # but ensure k is within [1, n-1] to avoid invalid slicing for small n.
    k = int(n - int(num_to_exclude) - 2)
    if k < 1:
        k = max(1, n - 1)
    if k > n - 1:
        k = n - 1

    # For each client, compute sum of k-nearest neighbors (excluding self)
    sorted_dist = np.sort(distances, axis=1)
    # sorted_dist[:, 0] is zero (distance to self); sum nearest k non-self distances
    scores = np.sum(sorted_dist[:, 1:k + 1], axis=1)

    # Select client with smallest score
    selected_idx = int(np.argmin(scores))
    selected = local_weights_list[selected_idx]

    return selected


def federated_robust_aggregate(
    local_weights_list: List[np.ndarray],
    method: str = 'trimmed_mean',
    **kwargs
) -> List[np.ndarray]:
    """
    Dispatcher for robust aggregation methods.
    
    Args:
        local_weights_list: list of per-client weights
        method: 'fedavg', 'median', 'trimmed_mean', 'krum'
        **kwargs: method-specific arguments
    
    Returns:
        aggregated weights
    """
    if method == 'median':
        return federated_median(local_weights_list)
    elif method == 'trimmed_mean':
        trim_frac = kwargs.get('trim_fraction', 0.2)
        return federated_trimmed_mean(local_weights_list, trim_fraction=trim_frac)
    elif method == 'krum':
        num_exclude = kwargs.get('num_to_exclude', 0)
        return federated_krum(local_weights_list, num_to_exclude=num_exclude)
    elif method == 'fedavg':
        # Standard averaging
        if isinstance(local_weights_list[0], (list, tuple)):
            out = []
            for layer_idx in range(len(local_weights_list[0])):
                layer_vals = [w[layer_idx] for w in local_weights_list]
                stacked = np.stack([np.asarray(v, dtype=np.float64) for v in layer_vals], axis=0)
                out.append(np.mean(stacked, axis=0).astype(layer_vals[0].dtype))
            return out
        else:
            stacked = np.stack([np.asarray(w, dtype=np.float64) for w in local_weights_list], axis=0)
            return np.mean(stacked, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


# ============================================================================
# LOCAL DIFFERENTIAL PRIVACY
# ============================================================================

def protector_apply_local_dp(
    weights: np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply client-side local differential privacy (Laplace mechanism).
    Add calibrated Laplace noise to weights before sending to server.
    
    Args:
        weights: client weight update
        epsilon: privacy budget (smaller = more noise)
        sensitivity: L1 sensitivity of weights (global bound)
        seed: random seed
    
    Returns:
        noisy weights satisfying epsilon-DP
    """
    rng = np.random.RandomState(seed)
    noise_scale = sensitivity / (epsilon + 1e-12)
    noise = rng.laplace(0, noise_scale, size=weights.shape)
    return weights + noise


def protector_apply_gaussian_dp(
    weights: np.ndarray,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply Gaussian DP (often used in DP-SGD).
    
    Args:
        weights: client weight update
        epsilon: privacy budget
        delta: failure probability
        sensitivity: L2 sensitivity
        seed: random seed
    
    Returns:
        noisy weights satisfying (epsilon, delta)-DP
    """
    rng = np.random.RandomState(seed)
    # Gaussian noise scale for (eps, delta)-DP
    sigma = (np.sqrt(2 * np.log(1.25 / delta)) * sensitivity) / epsilon
    noise = rng.normal(0, sigma, size=weights.shape)
    return weights + noise


def capra_aggregate(
    local_weights_list: List[List[np.ndarray]],
    sketch_dim_canonical: int = 64,
    n_sketches_canonical: int = 2,
    sketch_dim_fast: int = 16,
    n_sketches_fast: int = 1,
    # Backwards-compatible aliases accepted by aggregate_dispatcher/clients
    sketch_dim_per_layer: Optional[int] = None,
    n_sketches: Optional[int] = None,
    eps_sketch: float = 1.0,
    delta: float = 1e-5,
    sketch_noise_mech: str = 'laplace',
    z_thresh: float = 3.0,
    time_budget_ms: float = 200.0,
    probe_samples: int = 3,
    seed: int = 0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Simple CAPRA wrapper: adapt between fast and canonical FARPA depending on probe estimate.

    Returns aggregated weights and meta dict containing 'mode' and health probe info.
    """
    meta: Dict[str, Any] = {}
    if not local_weights_list:
        return [], {'error': 'no_clients'}

    # support legacy/dispatcher arg names: if caller passed sketch_dim_per_layer or n_sketches,
    # treat them as canonical-mode defaults unless explicit canonical args were provided.
    if sketch_dim_per_layer is not None and (sketch_dim_canonical is None or sketch_dim_canonical == 64):
        try:
            sketch_dim_canonical = int(sketch_dim_per_layer)
        except Exception:
            pass
    if n_sketches is not None and (n_sketches_canonical is None or n_sketches_canonical == 2):
        try:
            n_sketches_canonical = int(n_sketches)
        except Exception:
            pass

    # sample a few clients for probing
    sample = local_weights_list[: max(1, min(len(local_weights_list), probe_samples))]
    probe = probe_sketch_cost(sample, sketch_dim_per_layer=sketch_dim_canonical, n_sketches=n_sketches_canonical)
    meta['probe'] = probe

    est_total_ms = probe.get('avg_sketch_time_ms', 0.0) + probe.get('avg_serialize_time_ms', 0.0)
    # Decide mode
    if est_total_ms > float(time_budget_ms):
        mode = 'fast'
        chosen_dim = sketch_dim_fast
        chosen_n = n_sketches_fast
        # scale eps conservatively: reduce epsilon proportionally to sketch_dim ratio
        r = float(sketch_dim_fast) / max(1.0, float(sketch_dim_canonical))
        eps_use = max(1e-6, float(eps_sketch) * r)
    else:
        mode = 'canonical'
        chosen_dim = sketch_dim_canonical
        chosen_n = n_sketches_canonical
        eps_use = float(eps_sketch)

    meta['mode'] = mode
    meta['chosen_sketch_dim'] = int(chosen_dim)
    meta['chosen_n_sketches'] = int(chosen_n)
    meta['eps_used'] = float(eps_use)

    # Call FARPA with chosen params
    agg, trust, debug = farpa_aggregate(
        local_weights_list,
        sketch_dim_per_layer=chosen_dim,
        n_sketches=chosen_n,
        eps_sketch=eps_use,
        delta=delta,
        sketch_noise_mech=sketch_noise_mech,
        z_thresh=z_thresh,
        fallback='trimmed_mean',
        seed=seed,
    )

    meta['trust_scores'] = trust
    meta['debug'] = debug
    return agg, meta


if __name__ == '__main__':
    print("fl_helpers robustness extensions loaded.")


