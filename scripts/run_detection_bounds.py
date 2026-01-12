import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

np.random.seed(0)

def simulate_detection(d, n_clients=20, n_trials=200, attacker_shift=3.0, threshold_factor=1.5):
    detections = 0
    for t in range(n_trials):
        # benign clients
        benign = np.random.normal(0.0, 1.0, size=(n_clients-1, d))
        # attacker
        attacker = np.random.normal(attacker_shift, 1.0, size=(1, d))
        allv = np.vstack([benign, attacker])
        # median vector (coordinate-wise)
        med = np.median(allv, axis=0)
        # cosine distances to median
        def cosdist(a, b):
            na = np.linalg.norm(a) + 1e-12
            nb = np.linalg.norm(b) + 1e-12
            return 1.0 - np.dot(a, b) / (na * nb)
        dists = np.array([cosdist(v, med) for v in allv])
        # threshold set as median + threshold_factor * mad
        med_dist = np.median(dists)
        mad = np.median(np.abs(dists - med_dist)) + 1e-12
        threshold = med_dist + threshold_factor * mad
        # detect attacker if its distance > threshold
        if dists[-1] > threshold:
            detections += 1
    return detections / float(n_trials)

if __name__ == '__main__':
    ds = [4,8,16,32,64,128]
    probs = [simulate_detection(d) for d in ds]
    plt.figure(figsize=(6,4))
    plt.plot(ds, probs, marker='o')
    plt.xlabel('Sketch dimension (d)')
    plt.ylabel('Detection probability')
    plt.title('Detection probability vs sketch dimension (toy sim)')
    plt.grid(True)
    out_path = os.path.join('results','detection_vs_dimension.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print('Saved plot to', out_path)
