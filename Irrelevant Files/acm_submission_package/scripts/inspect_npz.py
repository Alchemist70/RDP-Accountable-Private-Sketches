import numpy as np, sys
p = sys.argv[1]
try:
    d = np.load(p)
    print('keys:', list(d.keys()))
    for k in d.files:
        try:
            a = d[k]
            print(k, getattr(a, 'shape', 'scalar'), type(a))
        except Exception as e:
            print('key read error', k, e)
except Exception as e:
    print('load error', e)
