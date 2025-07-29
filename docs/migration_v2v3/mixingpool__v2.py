import starsim as ss

mp_pars = dict(
    src = ss.AgeGroup(0, 15),
    dst = ss.AgeGroup(15, None),
    beta = 1,
    contacts = ss.poisson(lam=5),
    diseases = 'ncd'
)
mp = ss.MixingPool(**mp_pars) 