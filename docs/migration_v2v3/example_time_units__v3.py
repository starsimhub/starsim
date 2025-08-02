siskw = dict(dur_inf=ss.datedur(days=50), beta=ss.perday(0.01), waning=ss.perday(0.005))
kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', copy_inputs=False, verbose=0)

print('Year-year')
sis1 = ss.SIS(dt=1/365, **sc.dcp(siskw))
sim1 = ss.Sim(dt=1/365, diseases=sis1, label='year-year', **kw)

print('Day-day')
sis2 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
sim2 = ss.Sim(dt=ss.days(1), diseases=sis2, label='day-day', **kw)

print('Day-year')
sis3 = ss.SIS(dt=ss.days(1), **sc.dcp(siskw))
sim3 = ss.Sim(dt=1/365, diseases=sis3, label='day-year', **kw)

print('Year-day')
sis4 = ss.SIS(dt=1/365, **sc.dcp(siskw))
sim4 = ss.Sim(dt=ss.days(1), diseases=sis4, label='year-day', **kw) 