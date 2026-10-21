"""
Test importing EpiModel models, from the JSON that dump_epimodel.R writes (so R is not needed)
"""
import copy
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from starsim.interop.epimodel import load_epimodel

# EpiModel's introduction (lingua_franca_design/examples/epimodel/sis-network), as dump_epimodel.R writes it
sis = dict(format='epimodel', runs=[dict(
    kind = 'netsim',
    type = 'SIS',
    networks = [dict(
        n = 500,
        formation = '~edges + concurrent + degrange(from = 4)',
        terms = [dict(term='edges', args=[], text='edges'), dict(term='concurrent', args=[], text='concurrent'), dict(term='degrange', args={'from':4}, text='degrange(from = 4)')],
        stats = ['edges', 'concurrent', 'deg4+'],
        target_stats = [175, 110, 0],
        dissolution = '~offset(edges)',
        duration = 50,
        d_rate = 0,
        attributes = [],
    )],
    param = {'inf.prob':0.4, 'act.rate':2, 'rec.rate':0.05, 'vital':False},
    init = {'i.num':10},
    control = dict(type='SIS', nsteps=500, nsims=5),
    modules = {'initialize.FUN': dict(builtin=True)},
)], functions=[])

# A dcm() SIR with arrivals and departures, which agrees with EpiModel's to 3e-7 of the largest value
dcm = dict(format='epimodel', runs=[dict(
    kind = 'dcm',
    type = 'SIR',
    networks = [],
    param = {'inf.prob':0.2, 'act.rate':1, 'rec.rate':0.05, 'a.rate':1/95, 'ds.rate':0.01, 'di.rate':0.0125, 'dr.rate':0.01},
    init = {'s.num':1000, 'i.num':1, 'r.num':0},
    control = dict(type='SIR', nsteps=500, dt=0.5, odemethod='lsoda'),
    modules = [],
)], functions=[])


def test_epimodel_netsim():
    """ The built-in SIS on a network: the transitions, the network's targets, and the prevalence """
    sim = load_epimodel(sis, verbose=0)
    assert not sim.imported.incomplete, 'the introduction should import completely'
    net = sim.pars.networks
    assert isinstance(net, ssl.DegreeNet)
    assert (net.pars.n_contacts, net.pars.concurrency, net.pars.max_contacts, net.pars.dur) == (0.7, 0.22, 3, 50), 'the network should have EpiModel\'s targets'
    sim.run()
    res = sim.results.sis
    assert len(res.n_I) == 500, 'EpiModel\'s 500 times are 499 steps'
    prev = res.n_I[-200:].mean() # EpiModel: 227.5, against 234 over 200 runs (its recovery in the step of infection makes it lower)
    assert 180 < prev < 290, f'the equilibrium should be near EpiModel\'s, not {prev}'
    code = sim.to_code()
    assert 'ssl.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)' in code
    return sim


def test_epimodel_dcm():
    """ dcm() is an ODE with rates, and agrees with a hand-written model """
    sim = load_epimodel(dcm, verbose=0)
    sim.run()
    S, I, R = ss.states('S I R')
    inf_prob, act_rate, rec_rate = 0.2, ss.perday(1), ss.perday(0.05)
    a_rate, ds_rate, di_rate, dr_rate = ss.perday(1/95), ss.perday(0.01), ss.perday(0.0125), ss.perday(0.01)
    dis = ss.Disease(
        infection = S >> I @ inf_prob*act_rate*I,
        recovery  = I >> R @ rec_rate,
        arrival   = ss.null >> S @ a_rate*(S + I + R).n,
        departure_S = S >> ss.null @ ds_rate,
        departure_I = I >> ss.null @ di_rate,
        departure_R = R >> ss.null @ dr_rate,
        init = dict(S=1000, I=1), name='sir',
    )
    ref = ss.Sim(dis, solver='ode', total_pop=1001, unit='day', dt=ss.days(0.5), start=0, stop=499, verbose=0).run()
    for k in 'SIR':
        a, b = sim.results.sir[f'n_{k}'].values, ref.results.sir[f'n_{k}'].values
        assert np.allclose(a, b, rtol=1e-8), f'{k} should match the hand-written model'
    assert sim.results.sir.n_I.values.max() > 100, 'the epidemic should take off'
    return sim


def test_epimodel_listed():
    """ What has no declared form is listed: other ERGM terms, custom modules, and their source is kept """
    d = copy.deepcopy(sis)
    run = d['runs'][0]
    net = run['networks'][0]
    net['terms'] += [dict(term='nodematch', args=['risk'], text='nodematch("risk")'), dict(term='gwesp', args=dict(decay=0.5, fixed=True), text='gwesp(0.5, fixed = TRUE)')]
    net['target_stats'] += [140, 20]
    net['attributes'] = dict(risk=[1]*100 + [0]*400)
    run['modules']['recovery.FUN'] = dict(builtin=False, name='recover', source='recover <- function(dat, at) {\n  dat\n}')
    run['param']['inter.eff'] = 0.8
    second = copy.deepcopy(run)
    second['param']['inf.prob'] = 0.2
    d['runs'].append(second)

    sim = load_epimodel(d, verbose=0) # Warns ss.errors.ImportIncomplete for what is listed
    listed = sc.strjoin([k for k,why in sim.imported.incomplete])
    for key in ['gwesp', 'recovery.FUN', 'inter.eff']:
        assert key in listed, f'{key} should be listed as not translated'
    assert 'nodematch' not in listed, 'nodematch should be imported'
    assert sim.pars.networks.pars.match == dict(risk=0.8)
    assert 'function(dat, at)' in sim.imported.modules.recover
    assert sim.imported.scenarios.run1 == {'inf.prob': 0.2}, 'the other call of netsim() should be a scenario'
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_epimodel_netsim()
    sim2 = test_epimodel_dcm()
    sim3 = test_epimodel_listed()
    T.toc()
