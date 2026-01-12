import os
import subprocess
import time

# Suppress protobuf version warnings that can cause decoding failures
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def resume_incomplete_grids_parallel(outdir='apra_mnist_runs_full', max_procs=3):
    sketch_dims = [64, 128]
    n_sketches_list = [1, 2]
    z_threshs = [2.0, 3.0]
    rounds = 25
    local_epochs = 3
    batch_size = 32
    clients = 100
    attack = 'layer_backdoor'
    
    to_launch = []
    for sd in sketch_dims:
        for ns in n_sketches_list:
            for zt in z_threshs:
                grid_name = f'sd{sd}_ns{ns}_zt{zt}'
                grid_path = os.path.join(outdir, grid_name)
                
                for agg in ['apra_weighted', 'apra_basic', 'trimmed', 'median']:
                    agg_dir = os.path.join(grid_path, agg)
                    existing_ckpts = len([f for f in os.listdir(agg_dir) if f.startswith('round_') and f.endswith('.npz')]) if os.path.isdir(agg_dir) else 0
                    
                    if existing_ckpts < rounds:
                        cmd = (
                            f"python -u scripts/run_apra_mnist_full.py "
                            f"--sketch_dim {sd} --n_sketches {ns} --z_thresh {zt} "
                            f"--rounds {rounds} --local_epochs {local_epochs} --batch_size {batch_size} "
                            f"--clients {clients} --attack {attack} "
                            f"--output_dir {outdir} --agg_method {agg}"
                        )
                        to_launch.append({'grid': grid_name, 'agg': agg, 'existing': existing_ckpts, 'cmd': cmd})
    
    print(f"Launching {len(to_launch)} incomplete tasks...")
    print()
    
    processes = []
    idx = 0
    
    while idx < len(to_launch) or processes:
        while idx < len(to_launch) and len(processes) < max_procs:
            task = to_launch[idx]
            agg_dir = os.path.join(outdir, task['grid'], task['agg'])
            os.makedirs(agg_dir, exist_ok=True)
            stderr_file = open(os.path.join(agg_dir, 'stderr.log'), 'w')
            stdout_file = open(os.path.join(agg_dir, 'stdout.log'), 'w')
            p = subprocess.Popen(task['cmd'], shell=True, stdout=stdout_file, stderr=stderr_file)
            processes.append({'pid': p.pid, 'task': task, 'process': p, 'stdout': stdout_file, 'stderr': stderr_file})
            print(f"  [{task['grid']:15} / {task['agg']:12}] PID {p.pid:5} (resuming {task['existing']}/25)")
            idx += 1
            time.sleep(0.3)
        
        if processes:
            time.sleep(2)
            alive = []
            for proc_info in processes:
                if proc_info['process'].poll() is None:
                    alive.append(proc_info)
                else:
                    proc_info['stdout'].close()
                    proc_info['stderr'].close()
            processes = alive
    
    print()
    print(f"[OK] All {len(to_launch)} tasks launched")

resume_incomplete_grids_parallel(outdir='apra_mnist_runs_full', max_procs=3)
