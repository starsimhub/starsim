sim = ss.Sim(
    n_agents = 1000,
    pars = dict(
      networks = dict(
        type = 'random',
        n_contacts = 4
      ),
      diseases = dict(
        type      = 'sir',
        init_prev = 0.01,
        dur_inf   = ss.years(10),
        p_death   = 0,
        beta      = ss.peryear(0.06),
      )
    ),
    dur = ss.years(10),
    dt  = ss.years(0.05)
) 