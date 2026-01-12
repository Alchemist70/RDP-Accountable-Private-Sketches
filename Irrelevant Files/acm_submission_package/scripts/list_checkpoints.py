import os
for grid in sorted(os.listdir('apra_mnist_runs_full')):
    gd = os.path.join('apra_mnist_runs_full', grid)
    if not os.path.isdir(gd):
        continue
    print('\nGrid:', grid)
    for agg in sorted(os.listdir(gd)):
        aggd = os.path.join(gd, agg)
        if not os.path.isdir(aggd):
            continue
        files = [f for f in os.listdir(aggd) if f.startswith('round_') and f.endswith('.npz')]
        print('  ', agg, len(files))
