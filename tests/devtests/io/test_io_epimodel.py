"""
Test importing EpiModel models, from the JSON that dump_epimodel.R writes (so R is not needed), and exporting to EpiModel (run in EpiModel's environment, if it is there)
"""
import os
import copy
import tempfile
import subprocess
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import paths
from starsim.converters.epimodel import load_epimodel, to_epimodel, here

lf = sc.path(os.getenv('STARSIM_LF', paths.root/'lingua_franca')) # The lingua_franca repo, for EpiModel's environment
rscript = lf/'_envs'/'epimodel'/'bin'/'Rscript'

# EpiModel's introduction (docs/models/epimodel/sis-network in the lingua_franca repo), as dump_epimodel.R writes it
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

# The SEIR of EpiModel's tutorial on new models in dcm() (with one case fatality ratio), as dump_epimodel.R writes it; EpiModel gives at most 99388.06 infectious, on day 363
seir = dict(format='epimodel', runs=[dict(
    kind = 'dcm',
    type = None,
    networks = [],
    param = {'R0':1.9, 'e.dur':10, 'i.dur':14, 'cfr':0.5},
    init = {'s.num':1e6, 'e.num':10, 'i.num':0, 'r.num':0, 'se.flow':0, 'ei.flow':0, 'ir.flow':0, 'd.flow':0},
    control = dict(nsteps=500, dt=1, odemethod='lsoda'),
    modules = {'new.mod': dict(builtin=False, name='SEIR', source="""function(t, t0, parms) {
  with(as.list(c(t0, parms)), {
    # Population size
    num <- s.num + e.num + i.num + r.num
    ce <- R0 / i.dur
    lambda <- ce * i.num/num
    dS <- -lambda*s.num
    dE <- lambda*s.num - (1/e.dur)*e.num
    dI <- (1/e.dur)*e.num - (1 - cfr)*(1/i.dur)*i.num - cfr*(1/i.dur)*i.num
    dR <- (1 - cfr)*(1/i.dur)*i.num
    list(c(dS, dE, dI, dR,
           se.flow = lambda * s.num,
           ei.flow = (1/e.dur) * e.num,
           ir.flow = (1 - cfr)*(1/i.dur) * i.num,
           d.flow = cfr*(1/i.dur)*i.num),
         num = num,
         i.prev = i.num / num)
  })
}""")},
)], functions=[])


def make_seird():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    seird = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, [R, D], dur=ss.lognormal(ss.days(8), ss.days(3)), p=[None, 0.1]),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return ss.Sim(seird, total_pop=1e4, unit='day', stop=100, verbose=0)


def test_epimodel_netsim():
    """ The built-in SIS on a network: the transitions, the network's targets, and the prevalence """
    sim = load_epimodel(sis, verbose=0)
    assert not sim.imported.incomplete, 'the introduction should import completely'
    net = sim.pars.networks
    assert isinstance(net, ss.DegreeNet)
    assert (net.pars.n_contacts, net.pars.concurrency, net.pars.max_contacts, net.pars.dur) == (0.7, 0.22, 3, 50), 'the network should have EpiModel\'s targets'
    sim.run()
    res = sim.results.sis
    assert len(res.I) == 500, 'EpiModel\'s 500 times are 499 steps'
    prev = res.I[-200:].mean() # EpiModel: 227.5, against 234 over 200 runs (its recovery in the step of infection makes it lower)
    assert 180 < prev < 290, f'the equilibrium should be near EpiModel\'s, not {prev}'
    code = sim.imported.code
    assert "ss.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50, acts=ss.freqperday(2), name='net')" in code
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
        arrival   = ss.unborn >> S @ a_rate*(S + I + R).n,
        departure_S = S >> ss.dead @ ds_rate,
        departure_I = I >> ss.dead @ di_rate,
        departure_R = R >> ss.dead @ dr_rate,
        init_n = dict(S=1000, I=1), name='sir',
    )
    ref = ss.Sim(dis, solver='ode', total_pop=1001, unit='day', dt=ss.days(0.5), start=0, stop=499, verbose=0).run()
    for k in 'SIR':
        a, b = sim.results.sir[k].values, ref.results.sir[k].values
        assert np.allclose(a, b, rtol=1e-8), f'{k} should match the hand-written model'
    assert sim.results.sir.I.values.max() > 100, 'the epidemic should take off'
    return sim


def test_epimodel_listed():
    """ What has no declared form is listed: other ERGM terms, custom modules, and their source is kept """
    d = copy.deepcopy(sis)
    run = d['runs'][0]
    net = run['networks'][0]
    net['terms'] += [dict(term='nodefactor', args=['risk'], text='nodefactor("risk")'), dict(term='nodematch', args=['risk'], text='nodematch("risk")'), dict(term='gwesp', args=dict(decay=0.5, fixed=True), text='gwesp(0.5, fixed = TRUE)')]
    net['stats'] += ['nodefactor.risk.1', 'nodematch.risk', 'gwesp.fixed.0.5']
    net['target_stats'] += [150, 140, 20]
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
    assert 'nodematch' not in listed and 'nodefactor' not in listed, 'nodematch and nodefactor should be imported'
    assert sim.pars.networks.pars.match == dict(risk=0.8)
    assert sc.tolist(sim.pars.networks.pars.n_contacts.values) == [0.5, 1.5], 'nodefactor should give the mean degree of each level: the 100 at risk 1 have 150 of the 350'
    assert 'function(dat, at)' in sim.imported.modules.recover
    assert sim.imported.scenarios.run1 == {'inf.prob': 0.2}, 'the other call of netsim() should be a scenario'
    return sim


def test_epimodel_new_mod():
    """ A model function of dcm() (new.mod) is read as its flows, splitting each derivative into terms """
    sim = load_epimodel(seir, verbose=0)
    assert not sim.imported.incomplete, "the tutorial's SEIR should import completely"
    sim.run()
    assert list(sim.diseases.seir.graph.transitions) == ['s_to_e', 'e_to_i', 'i_to_r', 'i_to_out'], 'the death of a share cfr of the infectious should be an outflow'
    I = sim.results.seir.i.values
    assert np.isclose(I.max(), 99388.06, rtol=1e-6) and I.argmax() == 362, "the peak should be EpiModel's"
    return sim


def test_epimodel_export(tmp_path):
    """ An export is a dcm() with a model function, which EpiModel runs and dump_epimodel.R reads back """
    sim = make_seird()
    text, summary, lost = to_epimodel(sim)
    sc.savetext(tmp_path/'seird.R', text)
    assert '7 exponential stages' in lost[0]
    assert 'infection.flow <- beta * S.num * I / N' in text and 'new.mod = seird' in text and 'outcome_8.flow' in text
    if not rscript.exists():
        pytest.skip(f'needs EpiModel\'s environment ({rscript}; set STARSIM_LF)')
    sim.run()
    res = sim.results.seird
    out = subprocess.run([str(rscript), os.path.join(here, 'dump_epimodel.R'), str(tmp_path/'seird.R'), str(tmp_path/'seird.json')], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    back = load_epimodel(tmp_path/'seird.json', verbose=0).run() # Read back as Starsim
    I = sum(back.results.seird[f'I_{k}'].values for k in range(1, 8))
    assert np.allclose(I, res.I.values, rtol=1e-9) and np.allclose(back.results.seird.S.values, res.S.values, rtol=1e-9), 'the round trip should give the same ODE'
    code = f'suppressMessages(library(EpiModel)); pdf(NULL); source("{tmp_path/"seird.R"}"); write.csv(as.data.frame(mod), "{tmp_path/"seird.csv"}", row.names=FALSE)'
    out = subprocess.run([str(rscript), '-e', code], capture_output=True, text=True) # Run in EpiModel
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path/'seird.csv')
    I = df[[f'I_{k}.num' for k in range(1, 8)]].sum(axis=1).values
    assert np.allclose(I, res.I.values, rtol=0, atol=1e-4*res.I.values.max()), 'EpiModel\'s ODE should agree with Starsim\'s' # 3e-6 of the peak (lsoda's tolerance)
    return back


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_epimodel_netsim()
    sim2 = test_epimodel_dcm()
    sim3 = test_epimodel_listed()
    sim4 = test_epimodel_new_mod()
    sim5 = test_epimodel_export(sc.path(tempfile.mkdtemp()))
    T.toc()
