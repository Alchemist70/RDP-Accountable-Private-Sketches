import numpy as np
import fl_helpers

# create synthetic weight list (mimic layers)
w1 = np.random.randn(1000, 10).astype(np.float32)
w2 = np.random.randn(2000).astype(np.float32)
weights = [w1, w2]
print('Total flattened size:', sum([w.size for w in weights]))
sk = fl_helpers.random_projection_sketch(weights, sketch_dim=64, seed=123)
print('Returned sketch shape:', sk.shape, 'norm:', np.linalg.norm(sk))
