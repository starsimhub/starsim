siskw = dict(dur_inf=ss.dur(50, 'day'), beta=ss.beta(0.01, 'day'), waning=ss.rate(0.005, 'day'))
kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

print('Year-year')
sis1 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
sim1 = ss.Sim(unit='year', dt=1/365, diseases=sis1, label='year-year', **kw)

print('Day-day')
sis2 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
sim2 = ss.Sim(unit='day', dt=1.0, diseases=sis2, label='day-day', **kw)

print('Day-year')
sis3 = ss.SIS(unit='day', dt=1.0, **sc.dcp(siskw))
sim3 = ss.Sim(unit='year', dt=1/365, diseases=sis3, label='day-year', **kw)

print('Year-day')
sis4 = ss.SIS(unit='year', dt=1/365, **sc.dcp(siskw))
sim4 = ss.Sim(unit='day', dt=1.0, diseases=sis4, label='year-day', **kw) 