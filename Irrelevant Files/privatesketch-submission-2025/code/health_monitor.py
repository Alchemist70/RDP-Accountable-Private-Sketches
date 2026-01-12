import time
import pickle
from typing import Any, Dict, List


def probe_sketch_cost(local_weights_sample: List[List[Any]], sketch_dim_per_layer: int, n_sketches: int, n_trials: int = 2) -> Dict[str, float]:
    """Estimate rough compute+serialization cost for sketching a small sample.

    Returns a dict with keys: 'avg_sketch_time_ms', 'avg_serialize_time_ms', 'payload_size_bytes'
    """
    # Keep this probe dependency-free and lightweight: time local sketch construction and pickle serialization
    times = []
    ser_times = []
    sizes = []
    sample = local_weights_sample[: max(1, min(len(local_weights_sample), 3))]
    for _ in range(n_trials):
        # measure sketch compute by simulating a simple projection cost: we avoid calling fl_helpers to keep
        # this module import-light — callers can pass measured times too.
        t0 = time.perf_counter()
        # perform a cheap vector operation to approximate compute
        for w in sample:
            for layer in w:
                _ = (layer.reshape(-1)[:max(1, int(min(layer.size, sketch_dim_per_layer)))].astype('float32') * 1.0).sum()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

        # serialization time (pickle)
        t0 = time.perf_counter()
        payload = pickle.dumps(sample)
        t1 = time.perf_counter()
        ser_times.append((t1 - t0) * 1000.0)
        sizes.append(len(payload))

    return {
        'avg_sketch_time_ms': float(sum(times) / len(times)),
        'avg_serialize_time_ms': float(sum(ser_times) / len(ser_times)),
        'payload_size_bytes': float(sum(sizes) / len(sizes)),
    }


if __name__ == '__main__':
    print('health_monitor probe ready')
