"""
Extended APRA-MNIST experiment runner integrating:
- APS (Adaptive Privacy Shield) for adaptive privacy
- Robust aggregation (median, trimmed mean, Krum, APRA)
- Privacy evaluation (shadow attack)
- Comprehensive logging and checkpointing
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Use pure Python protobuf
# Force CPU-only execution to avoid GPU memory/fragmentation issues seen in runs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Don't set TF_XLA_FLAGS - let TensorFlow use defaults
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU for JAX backend if used
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory preallocation

import argparse
import json
import sys
import csv
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

# Avoid forcing experimental eager-mode at module import time. Use the configured
# runtime setting below (we explicitly disable eager execution for performance).

# Critical protobuf compatibility settings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.get_logger().setLevel('ERROR')

# Use graph mode for performance, avoid eager issues
tf.config.run_functions_eagerly(False)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Reduce TensorFlow memory usage
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Add scripts and parent directory to path
sys.path.insert(0, os.path.dirname(__file__))  # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # parent directory
# Also add workspace root (two levels up) so imports from repository root (e.g., fl_helpers.py)
# are available when running from submission_package scripts/ directory.
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import fl_helpers
import apra
import privacy_accounting

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APRAMNISTExperiment:
    """Full APRA-MNIST federated learning experiment."""
    
    def __init__(
        self,
        sketch_dim: int = 64,
        n_sketches: int = 1,
        z_thresh: float = 2.0,
        rounds: int = 25,
        clients: int = 100,
        local_epochs: int = 3,
        batch_size: int = 1,
        attack: str = 'none',
        byzantine_fraction: float = 0.0,
        output_dir: str = '.',
        seed: int = 42,
        agg_method: str = 'apra_weighted',
        run_tag: str = '',
        capra_time_budget_ms: float = 200.0,
        capra_fast_dim: Optional[int] = None,
        capra_fast_n_sketches: int = 1,
        sketch_noise_mech: str = 'laplace',
    ):
        """
        Args:
            sketch_dim: sketch dimension for APRA
            n_sketches: number of sketches per layer
            z_thresh: outlier detection threshold
            rounds: number of FL rounds
            clients: number of participating clients
            local_epochs: local SGD epochs
            batch_size: local batch size
            attack: type of attack ('none', 'scaling', 'backdoor', 'label_flip')
            byzantine_fraction: fraction of clients to poison
            output_dir: where to save results
            seed: random seed
            agg_method: aggregation method ('fedavg', 'median', 'trimmed', 'apra_weighted', etc.)
        """
        self.sketch_dim = sketch_dim
        self.n_sketches = n_sketches
        self.z_thresh = z_thresh
        self.rounds = rounds
        self.clients = clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.attack = attack
        self.byzantine_fraction = byzantine_fraction
        self.output_dir = output_dir
        self.seed = seed
        self.agg_method = agg_method
        # CAPRA runtime configuration
        self.capra_time_budget_ms = float(capra_time_budget_ms)
        self.capra_fast_dim = int(capra_fast_dim) if capra_fast_dim is not None else None
        self.capra_fast_n_sketches = int(capra_fast_n_sketches)
        # normalize and validate sketch noise mechanism
        self.sketch_noise_mech = str(sketch_noise_mech).lower()
        if self.sketch_noise_mech not in ('laplace', 'gaussian'):
            raise ValueError(f"Invalid sketch_noise_mech '{sketch_noise_mech}'. Allowed: 'laplace' or 'gaussian'.")
        # Allowed aggregators (must match runner/launcher expectations)
        self._allowed_aggs = ['apra_weighted', 'apra_basic', 'trimmed', 'median', 'farpa', 'capra']
        if self.agg_method not in self._allowed_aggs:
            raise ValueError(f"Invalid agg_method '{self.agg_method}'. Allowed: {self._allowed_aggs}")
        
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        # If run_tag provided, append to grid folder to avoid CSV overwrite when running aggregators in parallel
        tag_suffix = f'_{run_tag}' if run_tag else ''
        self.results_dir = os.path.join(output_dir, f'sd{sketch_dim}_ns{n_sketches}_zt{z_thresh}{tag_suffix}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create subdirs for aggregators
        self.agg_dirs = {agg: os.path.join(self.results_dir, agg) for agg in self._allowed_aggs}
        for d in self.agg_dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Load/prepare MNIST
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_mnist()
        
        # Initialize APS
        self.aps = fl_helpers.AdaptivePrivacyShield(default_epsilon=1.0)
        
        # Initialize APRA context
        self.apra_ctx = apra.APRAContext(
            sketch_dim=sketch_dim,
            sketch_method='random_projection',
            robust_method='trimmed_mean' if agg_method == 'trimmed' else agg_method,
            detection_method='median',
            aps_enabled=(agg_method == 'apra_weighted'),
            byzantine_fraction=byzantine_fraction,
            attack_type=None if attack == 'none' else attack
        )
        
        # Results tracking
        self.results = []
        self.csv_path = os.path.join(self.results_dir, 'results.csv')
    
    def _load_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess MNIST."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)
        return x_train, y_train, x_test, y_test
    
    def _build_model(self) -> keras.Model:
        """Build minimal MNIST model (single layer for stability)."""
        model = keras.Sequential([
            keras.layers.Dense(10, activation='softmax', input_shape=(784,))
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _partition_clients(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Partition MNIST into IID shards per client."""
        n_samples = len(self.x_train)
        samples_per_client = n_samples // self.clients
        
        client_data = {}
        for c in range(self.clients):
            start = c * samples_per_client
            end = (c + 1) * samples_per_client if c < self.clients - 1 else n_samples
            client_data[c] = (self.x_train[start:end], self.y_train[start:end])
        
        return client_data
    
    def _client_train(self, model: keras.Model, x: np.ndarray, y: np.ndarray) -> keras.Model:
        """Train model on client data for local_epochs."""
        model.fit(
            x, y,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False
        )
        return model
    
    def _evaluate_global_model(self, model: keras.Model) -> float:
        """Evaluate global model on test set."""
        _, acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(acc)
    
    def run(self):
        """Execute full APRA-MNIST experiment."""
        logger.info(f"Starting APRA-MNIST experiment: sketch_dim={self.sketch_dim}, agg={self.agg_method}")
        
        # Partition clients
        client_data = self._partition_clients()
        
        # Initialize global model
        global_model = self._build_model()
        initial_weights = global_model.get_weights()
        
        # Run FL rounds
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Round {round_num}/{self.rounds}")
            
            # Client local training
            client_weights = []
            client_deltas = []
            
            for client_id in range(self.clients):
                # Create client model
                client_model = self._build_model()
                client_model.set_weights(global_model.get_weights())
                
                # Local training
                x, y = client_data[client_id]
                self._client_train(client_model, x, y)
                
                # Get delta (update)
                new_w = client_model.get_weights()
                delta = [np.array(nw) - np.array(ow) for nw, ow in zip(new_w, global_model.get_weights())]
                client_deltas.append(delta)
                client_weights.append(new_w)
            
            # Apply APS scoring
            aps_scores = np.array([
                self.aps.assess_data_sensitivity(delta)
                for delta in client_deltas
            ])

            # Aggregate using APRA or dispatcher for FARPA/CAPRA
            if self.agg_method == 'farpa':
                # Use aggregate_dispatcher from fl_helpers for FARPA
                try:
                    agg, meta = fl_helpers.aggregate_dispatcher(
                        'farpa',
                        client_deltas,
                        sketch_dim_per_layer=self.sketch_dim,
                        n_sketches=self.n_sketches,
                        eps_sketch=1.0,
                        sketch_noise_mech=self.sketch_noise_mech,
                        z_thresh=self.z_thresh,
                        seed=self.seed + round_num,
                        aps_instance=self.aps,
                        adapt_with_aps=True,
                    )
                    aggregated_delta = agg
                    apra_meta = meta
                except Exception as e:
                    logger.exception('FARPA aggregation failed, falling back to APRA trimmed mean: %s', e)
                    aggregated_delta = fl_helpers.federated_trimmed_mean(client_deltas, trim_fraction=0.2)
                    apra_meta = {'fallback': 'trimmed_mean', 'error': str(e)}
            elif self.agg_method == 'capra':
                # Use CAPRA (adaptive FARPA wrapper)
                try:
                    capra_kwargs = dict(
                        sketch_dim_canonical=self.sketch_dim,
                        n_sketches_canonical=self.n_sketches,
                        sketch_dim_fast=(self.capra_fast_dim if self.capra_fast_dim is not None else max(8, self.sketch_dim // 4)),
                        n_sketches_fast=self.capra_fast_n_sketches,
                        eps_sketch=1.0,
                        sketch_noise_mech=self.sketch_noise_mech,
                        z_thresh=self.z_thresh,
                        time_budget_ms=self.capra_time_budget_ms,
                        probe_samples=3,
                        seed=self.seed + round_num,
                    )
                    agg, meta = fl_helpers.aggregate_dispatcher('capra', client_deltas, **capra_kwargs)
                    aggregated_delta = agg
                    apra_meta = meta
                except Exception as e:
                    logger.exception('CAPRA aggregation failed, falling back to APRA trimmed mean: %s', e)
                    aggregated_delta = fl_helpers.federated_trimmed_mean(client_deltas, trim_fraction=0.2)
                    apra_meta = {'fallback': 'trimmed_mean', 'error': str(e)}

                # If CAPRA used fast mode, schedule canonical recompute for this round
                # Keep a copy of client_deltas for recompute to avoid re-running clients
                try:
                    if isinstance(apra_meta, dict) and apra_meta.get('mode') == 'fast':
                        if not hasattr(self, '_capra_recompute_queue'):
                            self._capra_recompute_queue = []
                        # store shallow copy of deltas (arrays themselves are fine)
                        self._capra_recompute_queue.append({
                            'round': round_num,
                            'client_deltas': [ [np.array(x) for x in d] for d in client_deltas ],
                            'sketch_dim_canonical': capra_kwargs.get('sketch_dim_canonical'),
                            'n_sketches_canonical': capra_kwargs.get('n_sketches_canonical'),
                            'eps_sketch': capra_kwargs.get('eps_sketch'),
                            'seed': capra_kwargs.get('seed'),
                        })
                        logger.info('CAPRA: scheduled canonical recompute for round %d', round_num)
                except Exception:
                    # non-fatal: continue
                    pass
            else:
                # Aggregate using APRA context
                aggregated_delta, apra_meta = self.apra_ctx.aggregate(
                    client_deltas,
                    aps_scores=aps_scores if self.agg_method == 'apra_weighted' else None,
                    seed=self.seed + round_num
                )
            
            # Update global model
            new_global_weights = [
                np.array(gw) + np.array(ad)
                for gw, ad in zip(global_model.get_weights(), aggregated_delta)
            ]
            global_model.set_weights(new_global_weights)
            
            # Evaluate
            acc = self._evaluate_global_model(global_model)
            logger.info(f"  Accuracy: {acc:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(round_num, global_model, acc, apra_meta, aps_scores)
            
            # Save to CSV
            self.results.append({
                'round': round_num,
                'accuracy': acc,
                'agg': self.agg_method,
                'sketch_dim': self.sketch_dim,
                'n_sketches': self.n_sketches,
                'z_thresh': self.z_thresh,
                'attack': self.attack,
                'byzantine_fraction': self.byzantine_fraction,
            })
        
        # Save results CSV using native Python
        if self.results:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        logger.info(f"Results saved to {self.csv_path}")
        # Post-run: process CAPRA recompute queue (canonical recomputation for rounds that used fast-mode)
        if hasattr(self, '_capra_recompute_queue') and self._capra_recompute_queue:
            logger.info('Processing CAPRA recompute queue (%d entries)...', len(self._capra_recompute_queue))
            for entry in self._capra_recompute_queue:
                rnum = entry['round']
                # First, compute (or re-use) the canonical aggregate for this round. Any failure here
                # should be logged and we continue to the next queued entry.
                try:
                    canonical_agg, trust_canonical, debug_canonical = fl_helpers.farpa_aggregate(
                        entry['client_deltas'],
                        sketch_dim_per_layer=entry.get('sketch_dim_canonical', self.sketch_dim),
                        n_sketches=entry.get('n_sketches_canonical', self.n_sketches),
                        eps_sketch=entry.get('eps_sketch', 1.0),
                        delta=1e-5,
                        sketch_noise_mech=self.sketch_noise_mech,
                        z_thresh=self.z_thresh,
                        fallback='trimmed_mean',
                        seed=entry.get('seed', self.seed + rnum),
                    )
                except Exception:
                    logger.exception('CAPRA: failed canonical recompute for round %d', rnum)
                    continue

                out_dir = self.agg_dirs.get(self.agg_method, self.results_dir)
                ckpt_path = os.path.join(out_dir, f'round_{rnum:03d}.npz')

                if os.path.exists(ckpt_path):
                    try:
                        data = np.load(ckpt_path, allow_pickle=True)

                        # extract weight arrays w0, w1, ... into list
                        stored_weights = []
                        i = 0
                        while True:
                            key = f'w{i}'
                            if key in data:
                                stored_weights.append(np.array(data[key]))
                                i += 1
                            else:
                                break

                        # parse saved apra metadata if present to recover fast params used
                        fast_params = None
                        apra_meta_loaded = None
                        if 'apra_metadata' in data:
                            try:
                                import json as _json
                                raw = data['apra_metadata'].tolist() if hasattr(data['apra_metadata'], 'tolist') else data['apra_metadata']
                                apra_meta_loaded = _json.loads(raw)
                                if isinstance(apra_meta_loaded, dict) and apra_meta_loaded.get('mode') == 'fast':
                                    fast_params = dict(
                                        sketch_dim_per_layer=int(apra_meta_loaded.get('chosen_sketch_dim', entry.get('sketch_dim_canonical') // 4)),
                                        n_sketches=int(apra_meta_loaded.get('chosen_n_sketches', 1)),
                                        eps_sketch=float(apra_meta_loaded.get('eps_used', 0.5)),
                                    )
                            except Exception:
                                fast_params = None

                        # If we couldn't recover fast params from checkpoint, fall back to guessed fast params
                        if fast_params is None:
                            fast_params = dict(
                                sketch_dim_per_layer=entry.get('sketch_dim_canonical', self.sketch_dim) // 4,
                                n_sketches=1,
                                eps_sketch=entry.get('eps_sketch', 1.0) * 0.25,
                            )

                        # Recompute fast aggregate (to get fast aggregated delta)
                        try:
                            fast_agg, fast_trust, fast_debug = fl_helpers.farpa_aggregate(
                                entry['client_deltas'],
                                sketch_dim_per_layer=int(fast_params.get('sketch_dim_per_layer')),
                                n_sketches=int(fast_params.get('n_sketches')),
                                eps_sketch=float(fast_params.get('eps_sketch')),
                                delta=1e-5,
                                sketch_noise_mech=self.sketch_noise_mech,
                                z_thresh=self.z_thresh,
                                fallback='trimmed_mean',
                                seed=entry.get('seed', self.seed + rnum),
                            )
                        except Exception:
                            fast_agg = None

                        # If we successfully computed both canonical and fast aggregated deltas and have stored_weights,
                        # compute adjusted weights = stored_weights + (canonical_agg - fast_agg)
                        if stored_weights and (canonical_agg is not None) and (fast_agg is not None):
                            try:
                                adjusted = [None] * len(stored_weights)
                                for idx in range(len(stored_weights)):
                                    adjusted[idx] = (
                                        stored_weights[idx].astype(np.float64)
                                        + (np.asarray(canonical_agg[idx], dtype=np.float64) - np.asarray(fast_agg[idx], dtype=np.float64))
                                    ).astype(stored_weights[idx].dtype)

                                # atomically write adjusted weights to tmp npz then replace checkpoint
                                import tempfile as _temp
                                tmp_dir = os.path.dirname(ckpt_path)
                                with _temp.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix='.npz') as tfp:
                                    tmp_name = tfp.name

                                save_dict = {}
                                for i, w in enumerate(adjusted):
                                    save_dict[f'w{i}'] = w

                                # preserve other checkpoint fields if present (accuracy, aps_scores, existing apra_metadata)
                                try:
                                    if 'accuracy' in data:
                                        save_dict['accuracy'] = float(data['accuracy'].tolist() if hasattr(data['accuracy'], 'tolist') else data['accuracy'])
                                except Exception:
                                    pass
                                try:
                                    if 'aps_scores' in data:
                                        save_dict['aps_scores'] = np.array(data['aps_scores'])
                                except Exception:
                                    pass

                                # preserve apra_metadata but annotate recomputed
                                try:
                                    import json as _json
                                    save_dict['apra_metadata'] = _json.dumps({
                                        'recomputed': True,
                                        'recomputed_round': rnum,
                                        'orig_apra_meta': apra_meta_loaded,
                                    })
                                except Exception:
                                    save_dict['apra_metadata'] = json.dumps({'recomputed': True, 'recomputed_round': rnum})

                                np.savez(tmp_name, **save_dict)
                                try:
                                    os.replace(tmp_name, ckpt_path)
                                except Exception:
                                    os.rename(tmp_name, ckpt_path)
                                logger.info('CAPRA: atomic overwrite succeeded for round %d', rnum)
                            except Exception:
                                logger.exception('CAPRA: failed atomic overwrite for round %d', rnum)
                                # fallback: save recompute meta only
                                ckpt_meta = {
                                    'recomputed': True,
                                    'recomputed_round': rnum,
                                    'trust_canonical': trust_canonical,
                                    'debug_canonical': debug_canonical,
                                }
                                meta_path = os.path.join(out_dir, f'round_{rnum:03d}_recomputed_meta.json')
                                try:
                                    with open(meta_path, 'w') as mf:
                                        json.dump(ckpt_meta, mf)
                                except Exception:
                                    logger.exception('Failed to write recompute meta for round %d', rnum)
                                logger.info('CAPRA: recomputed canonical aggregate for round %d (meta saved, overwrite failed)', rnum)
                        else:
                            # fallback: save recompute meta only
                            ckpt_meta = {
                                'recomputed': True,
                                'recomputed_round': rnum,
                                'trust_canonical': trust_canonical,
                                'debug_canonical': debug_canonical,
                            }
                            meta_path = os.path.join(out_dir, f'round_{rnum:03d}_recomputed_meta.json')
                            try:
                                with open(meta_path, 'w') as mf:
                                    json.dump(ckpt_meta, mf)
                            except Exception:
                                logger.exception('Failed to write recompute meta for round %d', rnum)
                            logger.info('CAPRA: recomputed canonical aggregate for round %d (meta saved)', rnum)
                    except Exception:
                        logger.exception('CAPRA: failed processing checkpoint for round %d', rnum)
                else:
                    # no checkpoint file found; just save meta
                    ckpt_meta = {
                        'recomputed': True,
                        'recomputed_round': rnum,
                        'trust_canonical': trust_canonical,
                        'debug_canonical': debug_canonical,
                    }
                    meta_path = os.path.join(out_dir, f'round_{rnum:03d}_recomputed_meta.json')
                    try:
                        with open(meta_path, 'w') as mf:
                            json.dump(ckpt_meta, mf)
                    except Exception:
                        logger.exception('Failed to write recompute meta for round %d', rnum)
                    logger.info('CAPRA: recomputed canonical aggregate for round %d (meta saved, no checkpoint)', rnum)

        logger.info(f"Experiment complete!")
    
    def _save_checkpoint(self, round_num: int, model: keras.Model, acc: float, apra_meta: Dict, aps_scores: np.ndarray):
        """Save round checkpoint."""
        ckpt_path = os.path.join(self.agg_dirs[self.agg_method], f'round_{round_num:03d}.npz')

        # Prepare checkpoint data: save weights separately
        save_dict = {
            'accuracy': acc,
            'apra_metadata': json.dumps(apra_meta),
            'aps_scores': aps_scores.astype(np.float32)
        }

        # Add weights with keys w0, w1, w2, ... to avoid shape mismatch
        for i, w in enumerate(model.get_weights()):
            save_dict[f'w{i}'] = w.astype(np.float32)

        # Save atomically using a temporary file in the same directory.
        # Use a NamedTemporaryFile to avoid any chance of double-extension bugs
        # and then atomically replace the final checkpoint.
        tmp_dir = os.path.dirname(ckpt_path)
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix='.npz') as tfp:
                tmp_name = tfp.name
            # np.savez accepts a filename; if given a full path ending with
            # '.npz' it will not append another '.npz'. Write to the tmp file
            # then atomically replace the target.
            try:
                np.savez(tmp_name, **save_dict)
                try:
                    os.replace(tmp_name, ckpt_path)
                except Exception:
                    # fallback to rename
                    os.rename(tmp_name, ckpt_path)
            except Exception:
                # If tmp write failed, attempt direct save as last resort
                try:
                    np.savez(ckpt_path, **save_dict)
                except Exception:
                    logger.exception('Failed to write checkpoint %s', ckpt_path)
                finally:
                    # cleanup tmp file if it still exists
                    try:
                        if os.path.exists(tmp_name):
                            os.remove(tmp_name)
                    except Exception:
                        pass
        except Exception:
            # As a final best-effort fallback, try direct save and log
            try:
                np.savez(ckpt_path, **save_dict)
            except Exception:
                logger.exception('Failed to write checkpoint %s (final fallback)', ckpt_path)


def main():
    parser = argparse.ArgumentParser(description='APRA-MNIST Federated Learning Experiment')
    parser.add_argument('--sketch_dim', type=int, default=64, help='Sketch dimension')
    parser.add_argument('--n_sketches', type=int, default=1, help='Number of sketches per layer')
    parser.add_argument('--z_thresh', type=float, default=2.0, help='Outlier threshold (z-score)')
    parser.add_argument('--rounds', type=int, default=25, help='Number of FL rounds')
    parser.add_argument('--clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=1, help='Local SGD epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--attack', type=str, default='none', help='Attack type')
    parser.add_argument('--byzantine_fraction', type=float, default=0.0, help='Byzantine client fraction')
    parser.add_argument('--output_dir', type=str, default='apra_mnist_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--agg_method', type=str, default='apra_weighted', help='Aggregation method')
    parser.add_argument('--sketch_noise_mech', type=str, default='laplace', help='Sketch noise mechanism (laplace|gaussian)')
    parser.add_argument('--run_tag', type=str, default='', help='Optional run tag to isolate output directories (useful for parallel runs)')
    
    args = parser.parse_args()
    
    exp = APRAMNISTExperiment(
        sketch_dim=args.sketch_dim,
        n_sketches=args.n_sketches,
        z_thresh=args.z_thresh,
        rounds=args.rounds,
        clients=args.clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        attack=args.attack,
        byzantine_fraction=args.byzantine_fraction,
        output_dir=args.output_dir,
        seed=args.seed,
        agg_method=args.agg_method,
        run_tag=args.run_tag,
        sketch_noise_mech=args.sketch_noise_mech,
    )
    
    # write canonical provenance metadata for this run folder
    try:
        privacy_accounting.write_provenance_metadata(exp.results_dir, args=vars(args), seed=args.seed)
    except Exception:
        pass

    exp.run()


if __name__ == '__main__':
    main()
