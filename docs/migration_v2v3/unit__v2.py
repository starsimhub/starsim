import starsim as ss

pars = dict(
    diseases = ss.SIS(unit='day', dt=1.0, init_prev=0.1),
    demographics = ss.Births(dt=0.25, unit='year'),
    networks = ss.RandomNet(unit='week'),
)

sim = ss.Sim(pars, unit='day', dt=2, start='2000-01-01', stop='2002-01-01')
sim.run()