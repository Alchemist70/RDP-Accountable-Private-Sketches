import inspect
from tensorflow_privacy import compute_dp_sgd_privacy

print('SIGNATURE:', inspect.signature(compute_dp_sgd_privacy))
print('\nDOC:')
print(compute_dp_sgd_privacy.__doc__)

try:
    import tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib as lib
    print('\nFound compute_dp_sgd_privacy_lib, members:', [n for n in dir(lib) if not n.startswith('_')][:50])
except Exception:
    pass
