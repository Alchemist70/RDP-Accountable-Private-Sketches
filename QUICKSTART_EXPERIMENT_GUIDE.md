# PrivateSketch & APRA: Quick-Start Experiment Guide

## TL;DR - Run Your First Experiment (5 Minutes)

```bash
# Open Jupyter notebook
jupyter notebook apra_orchestrator.ipynb

# Run all cells (Shift+Enter repeatedly)
# You'll see:
# - ✓ Privacy budget computation
# - ✓ Sketch compression ratio
# - ✓ Byzantine attack detection results
# - ✓ Final model accuracy

# Output: metrics.json with complete results
```

---

## Guided Tour: Understanding the Code

### 1. Core Library: `fl_helpers.py`

**What it does**: Implements sketching, noise, detection, privacy accounting

**Key Classes**:
```python
# Sketching
RandomProjection(dimension=128)     # Random matrix projection
CountSketch(dimension=128)          # Hashing-based sketch

# Privacy
DifferentialPrivacyNoise(sigma=0.1) # Gaussian noise
RDPComposer(eps_target=1.0)         # Track privacy budget

# Detection  
ByzantineDetector(threshold=3.0)    # Median + MAD detector
AdaptiveAllocator(eps_budget=1.0)   # APS+ allocator

# Federated Learning
FederatedAverager(clients=10)        # Averaging aggregator
LocalSGDClient(model, lr=0.01)       # Client optimizer
```

**Example Usage**:
```python
import fl_helpers as flh

# Initialize components
sketch = flh.RandomProjection(dimension=128)
noise = flh.DifferentialPrivacyNoise(sigma=0.1)
detector = flh.ByzantineDetector(threshold=3.0)

# Sketch a gradient update
gradient = np.random.randn(10000)  # 10k-dimensional update
sketched = sketch.compress(gradient)  # 128 dimensions

# Add noise
noisy_sketch = noise.add(sketched)  # Add Gaussian noise

# Detect attacks (collect from all clients first)
all_sketches = [...]  # 10 clients' sketches
is_byzantine = detector.detect(all_sketches)  # Boolean array
```

---

### 2. Orchestrator: `apra_orchestrator.ipynb`

**What it does**: Full federated learning pipeline for one experiment

**Pipeline Stages**:

```
Stage 1: Setup
├── Initialize 10 clients
├── Load MNIST dataset
├── Distribute data (IID)
└── Set privacy budget ε=1.0

Stage 2: Training Loop (25 rounds)
├── Client-side: Local SGD training
├── Sketching: Compress updates (100GB → 1MB)
├── Noise: Add privacy-preserving Gaussian noise
├── Server-side: Collect sketches from all clients
├── Detection: Flag Byzantine clients (median + MAD)
├── Aggregation: Average honest sketches only
├── Privacy Accounting: Track cumulative ε
└── Evaluation: Test accuracy on validation set

Stage 3: Results
├── Plot accuracy curves
├── Show privacy accounting over rounds
├── Report detection statistics
└── Save metrics.json
```

**How to Modify**:

```python
# Change number of clients
n_clients = 100  # instead of 10

# Change Byzantine fraction
byzantine_fraction = 0.3  # 30% malicious clients

# Change sketch dimension
sketch_dim = 64  # smaller = more privacy, less utility

# Change privacy budget
epsilon_target = 0.5  # tighter privacy

# Change dataset
dataset = 'cifar10'  # instead of 'mnist'

# Change attack type
attack_type = 'label_flipping'  # or 'random_noise', 'scaling'
```

---

### 3. Privacy Accounting: `privacy_accounting.py`

**What it does**: Compute formal privacy guarantees using RDP

**Key Concepts**:

```python
import privacy_accounting as pa

# Create composer
composer = pa.RDPComposer(num_steps=25, noise_multiplier=0.1)

# After each round, add privacy cost
for round_i in range(25):
    # Add noise for this round
    composer.record_gaussian_noise(sigma=0.1)
    
    # Query cumulative privacy
    eps_current = composer.get_epsilon(delta=1e-5)
    print(f"Round {round_i}: ε = {eps_current:.4f}")

# Final privacy report
final_eps = composer.get_epsilon(delta=1e-5)
print(f"Final privacy: ({final_eps:.4f}, 1e-5)-DP")

# What this means:
# - Any dataset differing by 1 person is indistinguishable
# - Attacker's inference error is bounded by e^ε ≈ 1.01× (if ε=0.01)
# - A value of ε ≤ 1 is considered "strong" privacy
# - A value of ε ≤ 0.1 is considered "very strong" privacy
```

---

## Experiment Scenarios & How to Run Them

### Scenario 1: Basic Baseline (IID, No Attacks)

**Goal**: Establish baseline accuracy without Byzantine clients

**Configuration**:
```python
config = {
    'n_clients': 10,
    'data_distribution': 'iid',
    'byzantine_fraction': 0.0,  # No attacks
    'sketch_dim': 128,
    'num_rounds': 25,
    'epsilon_budget': 1.0,
    'attack_type': None,
}
```

**Expected Output**:
- Accuracy: ~98% on MNIST
- Privacy: ε ≈ 1.0 (as configured)
- Detection: 0 false positives (no Byzantine clients to detect)

**Time**: ~2 minutes

---

### Scenario 2: Byzantine Robustness (With Attacks)

**Goal**: Test detection of malicious clients

**Configuration**:
```python
config = {
    'n_clients': 10,
    'data_distribution': 'iid',
    'byzantine_fraction': 0.3,  # 30% malicious
    'byzantine_attacks': ['label_flipping', 'random_noise', 'scaling'],
    'sketch_dim': 128,
    'num_rounds': 25,
    'epsilon_budget': 1.0,
}
```

**Expected Output**:
- Accuracy: ~95% on MNIST (slight drop due to Byzantine presence)
- Detection Rate: >95% of Byzantine clients correctly identified
- Privacy: ε ≈ 1.0 (unaffected by Byzantine activity)

**Time**: ~5 minutes

---

### Scenario 3: Privacy-Utility Trade-off

**Goal**: Compare accuracy vs privacy under different privacy budgets

**Configuration**:
```python
for epsilon in [0.1, 0.5, 1.0, 5.0, 10.0]:
    config = {
        'n_clients': 10,
        'data_distribution': 'iid',
        'byzantine_fraction': 0.0,
        'sketch_dim': 128,
        'num_rounds': 25,
        'epsilon_budget': epsilon,  # Vary this
    }
    # Run experiment, record accuracy
    
# Plot: x-axis = epsilon (privacy), y-axis = accuracy (utility)
# This shows the Pareto frontier
```

**Expected Output**:
- ε=0.1: Accuracy ~85% (very private, noisy)
- ε=1.0: Accuracy ~98% (standard privacy)
- ε=10.0: Accuracy ~99.5% (weak privacy, high utility)

**Time**: ~15 minutes (5 runs × 2 min)

---

### Scenario 4: Non-IID Data Distribution

**Goal**: Test robustness when clients have different data distributions

**Configuration**:
```python
config = {
    'n_clients': 20,
    'data_distribution': 'non_iid',  # Each client has data from few classes
    'non_iid_distribution': 'label_skew',  # Class imbalance
    'label_skew_param': 0.5,  # Controls heterogeneity
    'byzantine_fraction': 0.2,
    'sketch_dim': 128,
    'num_rounds': 50,  # More rounds for convergence
    'epsilon_budget': 1.0,
}
```

**Expected Output**:
- Accuracy: ~92% (lower than IID, but still good)
- Convergence: Slower (more rounds needed)
- Detection: Still >90% effective

**Time**: ~8 minutes

---

### Scenario 5: Sketch Dimension Sensitivity

**Goal**: Find optimal sketch dimension

**Configuration**:
```python
for sketch_dim in [32, 64, 128, 256, 512]:
    config = {
        'n_clients': 10,
        'byzantine_fraction': 0.2,
        'sketch_dim': sketch_dim,  # Vary this
        'num_rounds': 25,
        'epsilon_budget': 1.0,
    }
    # Run experiment, record accuracy and detection rate
    
# Plot 1: Accuracy vs sketch dimension
# Plot 2: Detection rate vs sketch dimension
# Plot 3: Communication cost vs sketch dimension
```

**Expected Output**:
- sketch_dim=32: Communication 1.2MB, Accuracy 90%, Detection 85%
- sketch_dim=128: Communication 4.9MB, Accuracy 98%, Detection 95%
- sketch_dim=512: Communication 19.5MB, Accuracy 99%, Detection 98%

**Key Insight**: Diminishing returns after sketch_dim=128

**Time**: ~10 minutes

---

## How to Interpret Results

### Key Metrics Table

| Metric | What It Means | Good Value | Red Flag |
|--------|---------------|-----------|----------|
| **Accuracy** | Final model performance on test set | >95% on MNIST | <85% |
| **Privacy (ε)** | Privacy level (lower=more private) | ε ≤ 1.0 | ε > 10.0 |
| **Detection Rate** | % of Byzantine clients caught | >90% | <70% |
| **False Positive Rate** | % honest clients flagged as Byzantine | <5% | >10% |
| **Convergence Rounds** | Rounds to reach 95% accuracy | ~20-30 | >100 |
| **Communication** | Total data sent to server | 100MB-1GB | >10GB |

### Visualizations to Generate

**Plot 1: Accuracy Curve**
```
Accuracy (%)
|
99 |----___
   |       \___
97 |            \___
   |                \---___
95 |                       \---
   |________________________
    0      10      20      30 (Round #)
   
Shows: Convergence speed, final accuracy, variance
```

**Plot 2: Privacy Over Time**
```
Privacy (ε)
|
1.0|  ___
   | /   \___
0.7|/        \____
   |              \____
0.5|____________________
    0      10      20      30 (Round #)
    
Shows: Privacy cost accumulation, total budget
```

**Plot 3: Detection Performance**
```
Detection Rate (%)
|
100|████ ████ ████
   |
 95|████ ████ ████
   |
 90|████ ████ ████
   |
 85|████ ████ ████
   |
   +----+----+----
   label random scaling
   flip  noise
   
Shows: Effectiveness against different attacks
```

---

## Common Issues & Fixes

### Issue 1: "Out of Memory"
**Cause**: Sketch dimension too large, too many clients

**Fix**:
```python
# Reduce sketch dimension
config['sketch_dim'] = 64  # was 256

# Reduce batch size
config['batch_size'] = 16  # was 32

# Reduce number of rounds
config['num_rounds'] = 10  # was 25
```

---

### Issue 2: "Detection Rate Too Low"
**Cause**: Threshold too high, Byzantine attacks too subtle

**Fix**:
```python
# Lower detection threshold
config['detection_threshold'] = 2.0  # was 3.0

# Increase Byzantine attack strength
config['attack_magnitude'] = 10.0  # was 1.0

# Increase Byzantine fraction (more easy to detect)
config['byzantine_fraction'] = 0.5  # was 0.1
```

---

### Issue 3: "Accuracy Drops Too Much"
**Cause**: Noise too high, sketch dimension too small

**Fix**:
```python
# Increase epsilon (lower noise)
config['epsilon_budget'] = 2.0  # was 1.0

# Increase sketch dimension
config['sketch_dim'] = 256  # was 128

# Use gentler noise schedule
config['noise_schedule'] = 'adaptive'  # was 'uniform'
```

---

## Next Steps for Ambitious Researchers

### After Running Basic Experiments (Week 1)

**Hypothesis-Driven Experiments**:
1. "Does sketching help with Byzantine robustness?" 
   → Compare: full-dimensional vs sketched Byzantine detection
   
2. "What's the minimum sketch dimension?"
   → Grid search: sketch_dim from 4 to 1024

3. "Can we detect adaptive Byzantine attacks?"
   → Design attack that adjusts based on detection feedback

4. "How does non-IID data affect privacy?"
   → Compare privacy cost: IID vs non-IID for same accuracy

### After Understanding the Method (Weeks 2-4)

**Novel Contributions**:
1. **Theoretical**: Derive convergence rate for sketched FL
2. **Empirical**: Scale experiments to 1000+ clients, ImageNet dataset
3. **Methodological**: Design better Byzantine detection (e.g., multi-round)
4. **Practical**: Optimize for specific hardware (GPU, TPU)

---

## Computing Resources & Timeline

| Task | Time | CPU/GPU | Notes |
|------|------|---------|-------|
| Basic experiment (10 clients, 25 rounds) | 2-3 min | CPU | Can run on laptop |
| Non-IID experiment (20 clients) | 5-8 min | CPU | Slow, consider GPU |
| Large-scale (1000 clients, CIFAR-10) | 30+ min | GPU | Requires acceleration |
| Comprehensive grid search (50+ configs) | 2-4 hours | GPU | Parallelizable |

**Recommended Setup**: 
- Start: Laptop CPU (for debugging)
- Scale: Cloud GPU (GCP, AWS, Azure) for large experiments

---

## Quick Reference: Important Parameters

```python
# Federated Learning
n_clients = 10                    # Number of participants
n_rounds = 25                     # Training rounds
local_epochs = 1                  # Local training steps per round
learning_rate = 0.01              # Client optimizer LR

# Data & Distribution
dataset = 'mnist'                 # or 'cifar10', 'femnist'
data_distribution = 'iid'         # or 'non_iid'
byzantine_fraction = 0.0          # Fraction of malicious clients
attack_type = 'label_flipping'    # or 'random_noise', 'scaling'

# Sketching
sketch_dim = 128                  # Compression dimension
sketch_type = 'random'            # or 'countsketch'

# Privacy
epsilon_budget = 1.0              # Total privacy budget
noise_multiplier = 0.1            # Noise scale (σ)
delta = 1e-5                      # Failure probability

# Detection
detection_threshold = 3.0         # MAD multiplier for detection
aggregation_method = 'median'     # or 'trimmed_mean', 'average'
```

---

## Resources for Deeper Understanding

### Understanding Sketching
- Blog: "A Gentle Introduction to Random Projections"
- Paper: Woodruff (2014) "Sketching as a Tool for Numerical Linear Algebra"
- Video: MIT Open Courseware on Randomized Algorithms

### Understanding Byzantine Robustness  
- Blog: "Byzantine-Robust Aggregation for Federated Learning"
- Paper: Blanchard et al. (2017) "Machine Learning with Adversaries"
- Video: "Byzantine Robust Optimization" lecture

### Understanding Differential Privacy
- Blog: "Differential Privacy Under the Hood" (Lê Diem)
- Paper: Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
- Tool: DPCtorch (interactive DP tuning)

---

## Getting Help

**Slack**: #privatesketch-research  
**Office Hours**: Monday 2-3 PM + By appointment  
**Code Review**: Post in #code-review for feedback  
**Questions**: Tag @ml-faculty in relevant channel

---

**Ready? Start with the 5-minute quickstart above!**

*Last Updated*: November 28, 2025  
*Version*: 1.0
