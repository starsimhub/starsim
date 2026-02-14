import starsim as ss

pars = dict(
    diseases = ss.SIS(dt=ss.days(1.0), init_prev=0.1),
    demographics = ss.Births(dt=ss.years(0.25)),
    networks = ss.RandomNet(dt='week'),
)

sim = ss.Sim(pars, dt=ss.days(2), start='2000-01-01', stop='2002-01-01')
sim.run()