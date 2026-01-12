import re
s=open('paper_acm_draft.pdf','rb').read().decode('latin1')
pat=re.compile(r'rdp_smokegrid_eps_multi|aps_plus_epsilons|farpa_sweep_trust|single_coord_sensitivity|privacy_utility_tradeoff')
ms=pat.findall(s)
print('matches in PDF:', ms)
