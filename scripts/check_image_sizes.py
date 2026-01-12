import os
files=['paper_overleaf_upload/single_coord_sensitivity.pdf','paper_overleaf_upload/rdp_smokegrid_eps_multi.png','paper_overleaf_upload/aps_plus_epsilons.png','scripts/figs/farpa_sweep_trust.png','privacy_utility_tradeoff.png','paper_figures/aps_plus_sigmas.png','paper_figures/aps_plus_epsilons.png','paper_figures/aps_plus_sigmas_fallback.png','paper_figures/aps_plus_epsilons_fallback.png','scripts/figs/farpa_sweep_trust.svg','results/krum_poisoning_summary.png','results/krum_poisoning_curves.png','apra_mnist_runs_short/convergence_by_grid.png']
for p in files:
    if os.path.exists(p):
        print(p, os.path.getsize(p), 'bytes')
    else:
        print(p, 'MISSING')
