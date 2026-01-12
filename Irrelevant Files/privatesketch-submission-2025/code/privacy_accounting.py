from typing import List, Dict, Tuple
import math

class PrivacyLedger:
    """Minimal privacy accounting ledger.

    Records per-round (epsilon, delta) tuples and provides simple composition summaries:
    - basic composition (sum of epsilons)
    - advanced (approx) composition: using a simplified bound
    """
    def __init__(self):
        self.records: List[Tuple[float, float]] = []
        # mechanism_records stores detailed mechanism-level info for formal accounting
        # Each entry is a dict with keys like: {'mech': 'gaussian', 'sigma': float, 'sampling_rate': float, 'steps': int}
        self.mechanism_records: List[Dict] = []

    def record(self, eps: float, delta: float = 1e-5) -> None:
        try:
            e = float(eps)
        except Exception:
            e = 0.0
        try:
            d = float(delta)
        except Exception:
            d = 1e-12
        self.records.append((e, d))

    def record_mechanism(self, mech: str, **kwargs) -> None:
        """Record a mechanism-level entry for RDP-style accounting.

        Examples:
            record_mechanism('gaussian', sigma=1.0, sampling_rate=1.0, steps=1)
            record_mechanism('laplace', b=2.0, sensitivity=1.0, steps=1)
        """
        try:
            entry = {'mech': str(mech)}
            entry.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in kwargs.items()})
            self.mechanism_records.append(entry)
        except Exception:
            # best-effort: ignore malformed mechanism records
            pass

    def clear(self) -> None:
        self.records = []

    def basic_composition(self) -> Tuple[float, float]:
        """Return (eps_total, delta_total) under naive composition: sum(eps), sum(delta)."""
        if not self.records:
            return 0.0, 0.0
        eps_total = sum(r[0] for r in self.records)
        delta_total = sum(r[1] for r in self.records)
        return float(eps_total), float(delta_total)

    def advanced_composition(self, delta_prime: float = 1e-6) -> Tuple[float, float]:
        """Compute an (epsilon, delta) bound using a simplified advanced composition approximation.

        Uses: epsilon_total <= sum eps_i + 2 * sqrt(sum eps_i^2 * log(1/delta_prime)).
        Returns (epsilon_bound, delta_total + delta_prime)
        """
        if not self.records:
            return 0.0, 0.0
        sum_eps = sum(r[0] for r in self.records)
        sum_eps_sq = sum((r[0] ** 2) for r in self.records)
        if sum_eps_sq <= 0:
            eps_bound = sum_eps
        else:
            eps_bound = sum_eps + 2.0 * math.sqrt(sum_eps_sq * max(1e-12, math.log(1.0 / max(1e-12, delta_prime))))
        delta_total = sum(r[1] for r in self.records) + delta_prime
        return float(eps_bound), float(delta_total)

    def rdp_approx_composition(self, delta_prime: float = 1e-6) -> Tuple[float, float]:
        """Return a conservative (eps, delta) bound using an RDP-style approximation.

        This routine provides a practical, conservative approximation suitable for
        engineering evaluation. It uses the recorded per-round epsilons (interpreted
        as ``epsilon`` for each mechanism) and composes them using a moments-like
        bound:

            eps_total ~= sum_eps + sqrt(2 * sum_eps_sq * log(1/delta_prime))

        and returns (eps_total, delta_total + delta_prime).

        This is not a substitute for a formal RDP accountant (e.g., numerical
        optimization over Rényi orders) but is useful for reporting an approximate
        composed privacy budget for experiments.
        """
        if not self.records:
            return 0.0, 0.0
        sum_eps = sum(r[0] for r in self.records)
        sum_eps_sq = sum((r[0] ** 2) for r in self.records)
        # conservative bound
        try:
            tail = 2.0 * math.sqrt(max(0.0, sum_eps_sq) * max(1e-12, math.log(1.0 / max(1e-12, delta_prime))))
        except Exception:
            tail = 0.0
        eps_bound = sum_eps + tail
        delta_total = sum(r[1] for r in self.records) + delta_prime
        return float(eps_bound), float(delta_total)

        def compute_rdp_via_tensorflow_privacy(self, target_delta: float = 1e-6, mechanism_records: List[Dict] = None) -> Tuple[float, float]:
                """Compute (eps, delta) using `tensorflow_privacy`'s accountant.

                Args:
                        target_delta: Target delta for (eps, delta)-DP.
                        mechanism_records: Optional explicit list of mechanism dicts (if None, uses self.mechanism_records).

                Notes for maintainers:
                - The import of `tensorflow_privacy` is performed dynamically at runtime
                    (via `importlib`) to avoid static-analysis and IDE diagnostics when the
                    package is not installed in the developer's environment. This allows
                    the repository to be edited and linted without requiring `tensorflow_privacy`.
                - This helper currently supports Gaussian mechanisms recorded via
                    `record_mechanism('gaussian', sigma=..., sampling_rate=..., steps=...)`.
                - If `tensorflow_privacy` is not present, this function raises
                    `ImportError` so caller code can fall back to approximate composition.
                """
        # lazy dynamic import to avoid static-analysis/IDE missing-imports warnings
        # (use importlib so analyzers like Pylance don't try to resolve the package
        # at static analysis time). We only import when this function is called.
        try:
            import importlib
            tp_mod = importlib.import_module("tensorflow_privacy.privacy.analysis")
            # try several possible exported helper names used across tf-privacy versions
            import types
            compute_dp = None
            for candidate in ("compute_dp_sgd_privacy", "compute_dp_sgd_privacy_lib", "compute_dp_sgd_privacy_tf"):
                obj = getattr(tp_mod, candidate, None)
                if isinstance(obj, types.ModuleType):
                    # some distributions expose the implementation as a submodule
                    compute_dp = getattr(obj, 'compute_dp_sgd_privacy', None) or getattr(obj, 'compute_dp', None)
                elif callable(obj):
                    compute_dp = obj
                if callable(compute_dp):
                    break
            if compute_dp is None:
                # fall back to scanning for any callable with 'compute' in the name
                for name in dir(tp_mod):
                    if 'compute' in name:
                        candidate_obj = getattr(tp_mod, name)
                        if callable(candidate_obj):
                            compute_dp = candidate_obj
                            break
                        if isinstance(candidate_obj, types.ModuleType):
                            compute_dp = getattr(candidate_obj, 'compute_dp_sgd_privacy', None) or getattr(candidate_obj, 'compute_dp', None)
                            if callable(compute_dp):
                                break
            if compute_dp is None:
                raise ImportError("tensorflow_privacy.privacy.analysis.compute_dp helper not found")
        except Exception as e:
            raise ImportError('tensorflow_privacy not available') from e

        # Collect gaussian mechanism records
        records = mechanism_records if mechanism_records is not None else self.mechanism_records
        gauss_records = [r for r in records if r.get('mech') == 'gaussian']
        if not gauss_records:
            raise ValueError('No gaussian mechanism records available for tf-privacy accountant')

        # Group gaussian records by (sigma, sampling_rate) so we can compute RDP per group
        groups = {}
        for r in gauss_records:
            sigma = float(r.get('sigma'))
            samp = float(r.get('sampling_rate', 1.0))
            steps = int(r.get('steps', 1))
            key = (sigma, samp)
            groups.setdefault(key, 0)
            groups[key] += steps

        # Import tf-privacy rdp accountant helpers robustly
        try:
            import importlib
            rdp_mod = importlib.import_module('tensorflow_privacy.privacy.analysis.rdp_accountant')
            compute_rdp = getattr(rdp_mod, 'compute_rdp', None)
            get_privacy_spent = getattr(rdp_mod, 'get_privacy_spent', None)
            # some versions expose helper at top-level analysis module
            if compute_rdp is None:
                top = importlib.import_module('tensorflow_privacy.privacy.analysis')
                compute_rdp = getattr(top, 'compute_rdp', None)
                get_privacy_spent = getattr(top, 'get_privacy_spent', None)
        except Exception as e:
            raise ImportError('tensorflow_privacy rdp accountant not available') from e

        if compute_rdp is None or get_privacy_spent is None:
            raise ImportError('Required tf-privacy rdp_accountant helpers not found')

        # Choose a sensible set of Rényi orders to evaluate
        try:
            import numpy as _np
            orders = _np.concatenate((_np.arange(2, 64, dtype=float), _np.arange(64, 201, 10, dtype=float)))
            orders = orders.tolist()
        except Exception:
            orders = [float(x) for x in list(range(2, 200, 1))]

        # Sum RDP across groups (for each order)
        total_rdp = None
        for (sigma, sampling_rate), group_steps in groups.items():
            # compute_rdp signature: compute_rdp(q, noise_multiplier, steps, orders)
            try:
                rdp_vals = compute_rdp(sampling_rate, float(sigma), int(group_steps), orders)
            except TypeError:
                # try alternative signature: compute_rdp(steps, noise_multiplier, orders)
                rdp_vals = compute_rdp(int(group_steps), float(sigma), orders)
            # rdp_vals expected list-like same length as orders
            if total_rdp is None:
                total_rdp = [float(x) for x in rdp_vals]
            else:
                total_rdp = [a + float(b) for a, b in zip(total_rdp, rdp_vals)]

        if total_rdp is None:
            raise ValueError('No gaussian mechanism records composed')

        # Convert RDP -> (eps, opt_order) using tf-privacy helper
        try:
            eps, opt_order = get_privacy_spent(orders, total_rdp, target_delta)
            return float(eps), float(target_delta)
        except Exception as e:
            raise RuntimeError('Failed to compute privacy spent via tf-privacy') from e


# Module-level ledger instance for easy import/use
ledger = PrivacyLedger()


def write_provenance_metadata(outdir: str, args: dict = None, seed: int = None, extra: dict = None) -> str:
    """Write a canonical provenance file `metadata_run.json` into `outdir`.

    Fields included: git_sha (if available), python_version, key package versions (tensorflow, tensorflow_privacy),
    provided CLI args, seed, and timestamp (ISO). Returns the path written.
    """
    import json, sys, subprocess, datetime
    meta = {}
    meta['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'
    meta['python_version'] = sys.version
    if seed is not None:
        meta['seed'] = int(seed)
    if args is not None:
        try:
            meta['args'] = dict(args)
        except Exception:
            meta['args'] = str(args)
    if extra is not None:
        try:
            meta['extra'] = dict(extra)
        except Exception:
            meta['extra'] = str(extra)

    # try to get git sha
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.', stderr=subprocess.DEVNULL).decode('utf-8').strip()
        meta['git_sha'] = git_sha
    except Exception:
        meta['git_sha'] = None

    # collect a few package versions if available
    pkgs = {}
    try:
        import importlib
        for name in ('tensorflow', 'tensorflow_privacy'):
            try:
                mod = importlib.import_module(name)
                pkgs[name] = getattr(mod, '__version__', str(mod))
            except Exception:
                pkgs[name] = None
    except Exception:
        pass
    if pkgs:
        meta['packages'] = pkgs

    # write file
    import os
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'metadata_run.json')
    try:
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return path
