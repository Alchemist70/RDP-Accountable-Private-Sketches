import psutil
for p in psutil.process_iter(['pid','name','cmdline','create_time']):
    try:
        if 'python' in (p.info['name'] or '').lower():
            print(p.info['pid'], p.info['create_time'], ' '.join(p.info['cmdline'] or []))
    except Exception:
        pass
