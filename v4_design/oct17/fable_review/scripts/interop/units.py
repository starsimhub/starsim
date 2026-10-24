import warnings, numpy as np, sciris as sc, starsim as ss
ex = sc.path('/home/cliffk/idm/lingua_franca_design/examples')
sim = ss.load(ex/'flepimop/seir_2pop/original/config_sample_2pop.yml', verbose=0)
sim.init()
d = sim.diseases[0]
print('Ro par raw:', repr(d.pars.Ro.raw), type(d.pars.Ro.raw))
print('gamma par raw:', repr(d.pars.gamma.raw))
spec = sim.to_spec()
print('spec pars:', spec['modules'][0]['pars'])
print('spec start/stop:', spec['config']['start'], spec['config']['stop'], 'npts', sim.t.npts, 'dt', sim.t.dt)
sim.run()
code = sim.to_code()
ns = {}; exec(code, ns); sim2 = ns['sim']; sim2.run(verbose=0)
a = sim.results[d.name].n_I.values; b = sim2.results[d.name].n_I.values
print('max |diff| n_I import vs to_code:', np.abs(a-b).max(), 'peak', a.max())
# What does perday*perday do?
try:
    print('perday(2.5)*perday(0.2) =', repr(ss.perday(2.5)*ss.perday(0.2)))
except Exception as E:
    print('perday*perday error:', E)
# Same model with the sim in weeks: does Ro get rescaled as a rate?
sim3 = ss.load(ex/'flepimop/seir_2pop/original/config_sample_2pop.yml', verbose=0, unit='week')
sim3.run()
c = sim3.results[d.name].n_I.values
print('weeks: peak n_I', c.max(), 'vs days', a.max(), ' npts', len(c))
print('imported survives new()?', hasattr(sim.new(solver='ode'), 'imported'))
import copy
print('imported survives dcp?', hasattr(sc.dcp(sim), 'imported'))
