"""inspect_tfp.py

Utility to introspect installed tensorflow_privacy module in the active Python env.
Run this inside `tfpriv` to print available submodules and attributes related to RDP-accounting.
"""
import importlib
import sys

mods_to_try = [
    'tensorflow_privacy',
    'tensorflow_privacy.privacy',
    'tensorflow_privacy.privacy.analysis',
    'tensorflow_privacy.privacy.analysis.rdp_accountant',
    'tensorflow_privacy.privacy.analysis.rdp',
    'tensorflow_privacy.privacy.analysis.accountant',
]

for m in mods_to_try:
    try:
        mod = importlib.import_module(m)
        print(f"IMPORT OK: {m} -> {getattr(mod, '__file__', 'builtin/module')}" )
        print(sorted([name for name in dir(mod) if not name.startswith('_')])[:50])
    except Exception as e:
        print(f"IMPORT FAIL: {m} -> {e}")

print('\nPython sys.path:')
print('\n'.join(sys.path[:10]))
