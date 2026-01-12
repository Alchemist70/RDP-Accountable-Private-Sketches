import numpy as np

def random_projection_sketch_sparse(weights, sketch_dim, seed=None, chunk_size=5000):
    if isinstance(weights, (list, tuple)):
        vec = np.concatenate([w.flatten() for w in weights]).astype(np.float32)
    else:
        vec = np.asarray(weights).astype(np.float32).flatten()
    n = vec.size
    rng = np.random.RandomState(seed)
    sk = np.zeros(sketch_dim, dtype=np.float32)
    scale = 1.0 / (np.sqrt(float(max(1, n))))
    for i in range(0, n, chunk_size):
        c = vec[i:i+chunk_size]
        proj = rng.normal(loc=0.0, scale=scale, size=(c.size, sketch_dim)).astype(np.float32)
        sk += c.dot(proj)
    norm = np.linalg.norm(sk)
    if norm>0:
        sk /= norm
    return sk

if __name__=='__main__':
    vec = np.random.randn(87050).astype(np.float32)
    sk = random_projection_sketch_sparse(vec, 64, seed=42, chunk_size=5000)
    print('Sketch shape:', sk.shape, 'norm:', np.linalg.norm(sk))
