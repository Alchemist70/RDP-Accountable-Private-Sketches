import os
import itertools
import subprocess
import sys

# Parameters matching the sweep
sketch_dims = [64, 128]
n_sketches = [1, 2]
z_thresh = [2.0, 3.0]
rounds = 25
local_epochs = 3
batch_size = 32
clients = 100
attack = 'layer_backdoor'
outdir = 'apra_mnist_runs_full'

# Helper to check completion: require each aggregator dir to have rounds files for final round
def trial_complete(run_dir, final_round=rounds):
    if not os.path.isdir(run_dir):
        return False
    # expected aggregators
    aggs = ['apra_weighted','apra_basic','trimmed','median']
    for agg in aggs:
        agg_dir = os.path.join(run_dir, agg)
        if not os.path.isdir(agg_dir):
            return False
        # check for round_{:03d}.npz
        expected = os.path.join(agg_dir, f'round_{final_round:03d}.npz')
        if not os.path.exists(expected):
            return False
    return True

# Build command
def build_cmd(run_dir, sd, ns, zt):
    # quote run_dir
    run_dir_q = run_dir.replace('\\','/')
    # use unbuffered Python (-u) so stdout/stderr is flushed to run.log promptly
    cmd = (
        f"python -u scripts/run_apra_mnist.py --rounds {rounds} --local-epochs {local_epochs} --batch-size {batch_size} "
        f"--clients {clients} --attack {attack} --sketch-dims \"{sd}\" --n-sketches \"{ns}\" --z-thresh \"{zt}\" --outdir \"{run_dir_q}\""
    )
    # redirect stdout/stderr to run.log in run_dir
    log = os.path.join(run_dir, 'run.log')
    # Windows redirection
    cmd = f"{cmd} > \"{log}\" 2>&1"
    return cmd

launched = []
for sd, ns, zt in itertools.product(sketch_dims, n_sketches, z_thresh):
    run_dir = os.path.join(outdir, f'sd{sd}_ns{ns}_zt{zt}')
    if trial_complete(run_dir):
        print('Skipping complete:', run_dir)
        continue
    os.makedirs(run_dir, exist_ok=True)
    cmd = build_cmd(run_dir, sd, ns, zt)
    print('Launching:', run_dir)
    # On Windows, use DETACHED_PROCESS to detach
    if os.name == 'nt':
        creationflags = 0x00000008
        p = subprocess.Popen(cmd, shell=True, creationflags=creationflags)
    else:
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    launched.append({'pid': getattr(p,'pid',None), 'run_dir': run_dir, 'cmd': cmd})

print('\nLaunched summary:')
for l in launched:
    print(l['pid'], l['run_dir'])

print('\nDone.')
