pars = dict(
    diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1, beta=ss.beta(0.01)),
    demographics = ss.Births(unit='year', dt=0.25),
    networks = ss.RandomNet(unit='week'),
    n_agents = small,
    verbose = 0,
) 