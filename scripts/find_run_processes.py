import psutil
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        cmd=' '.join(p.info['cmdline'] or [])
        if 'run_apra_mnist.py' in cmd or 'run_apra_mnist' in cmd:
            print('FOUND', p.pid, cmd)
    except Exception:
        pass
print('Done')
