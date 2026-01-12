import os
for root, dirs, files in os.walk('apra_mnist_runs_full'):
    for f in files:
        if f.endswith('.csv'):
            print(os.path.join(root, f))
