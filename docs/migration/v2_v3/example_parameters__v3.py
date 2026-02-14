pars = dict(
    diseases = ss.SIS(dt=ss.days(1), init_prev=0.1, beta=ss.peryear(0.01)),
    demographics = ss.Births(dt=0.25),
    networks = ss.RandomNet(dt=ss.weeks(1)),
    n_agents = small,
    verbose = 0,
) 