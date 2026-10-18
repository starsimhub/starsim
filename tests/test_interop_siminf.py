"""
Test importing SimInf models
"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import paths
from starsim.interop.siminf import load_siminf, read_script, event_register, as_matrix, predefined, Unknown

examples = paths.lfd/'examples/siminf' # The originals, if present
kw = dict(verbose=False)

script = '''
library(SimInf)
transitions <- c("@ -> mu*N -> S", "S -> beta*S*I/N -> I", "I -> gamma*I -> R",
                 "S -> mu*S -> @", "I -> mu*I -> @", "R -> mu*R -> @", "N <- S + I + R")
compartments <- c("S", "I", "R")
n <- 3
u0 <- data.frame(S = rep(990, n), I = c(10, 0, 0), R = rep(0, n))
u0$I[2] <- 5
model <- mparse(transitions = transitions, compartments = compartments,
                gdata = c(beta = 0.3, gamma = 0.1, mu = 1/(70*365)), u0 = u0, tspan = 1:200)
result <- run(model)
plot(result, I ~ .)
'''


def test_r_reader():
    sc.heading('Testing reading the literal parts of an R script')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.R')
        sc.savetext(path, script)
        args = read_script(path)
    assert args['model'] == 'mparse'
    assert args['compartments'] == ['S', 'I', 'R']
    assert args['gdata'] == dict(beta=0.3, gamma=0.1, mu=1/(70*365))
    assert list(args['u0'].I) == [10, 5, 0] # The assignment u0$I[2] <- 5 is applied
    assert np.array_equal(args['tspan'], np.arange(1, 201))
    return args


def test_mparse():
    sc.heading('Testing an mparse() model with births, deaths, and nodes')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'model.R')
        sc.savetext(path, script)
        sim = load_siminf(path, **kw)
    choices = '\n'.join(sim.imported.choices)
    assert 'exposure to I' in choices # beta*S*I/N, with N <- S + I + R
    assert 'mu * (S.n + I.n + R.n)' in choices # Births in proportion to everyone
    assert sim.pars.solver == 'gillespie'
    sim.run()
    res = sim.results.model
    assert res.n_I.values.max() > 50
    assert np.isclose((res.n_S + res.n_I + res.n_R).values[-1], 2985, rtol=0.02) # Births and deaths balance
    assert res.n_I.strata.shape[1] == 3 # By node
    return sim


def test_dict():
    sc.heading('Testing a model given as Python data, with a catalytic transition and parameters by node')
    model = dict(transitions=['S + I -> beta*S*I/(S+I) -> 2*I', 'I -> gamma*I -> S'], compartments=['S', 'I'],
                 u0=dict(S=[95, 90], I=[5, 10]), ldata=dict(beta=[0.5, 0.0], gamma=[0.1, 0.1]), tspan=list(range(1, 101)))
    sim = load_siminf(model, **kw)
    sim.run()
    I = sim.results.siminf.n_I.strata
    assert I[-1, 0] > 30 # Node 1 keeps an endemic infection
    assert I[-1, 1] == 0 # Node 2 has beta = 0, so its infected recover
    bad = dict(model, transitions=['S -> beta*S -> 2*I'])
    sim = load_siminf(bad, **kw)
    assert 'not one person moving' in sim.imported.incomplete[0][1]
    return sim


def test_events():
    sc.heading('Testing SimInf events as a movement register')
    compartments = predefined['SISe3']['compartments']
    E, N = [as_matrix(predefined['SISe3'][k], compartments) for k in 'EN']
    ev = pd.DataFrame(dict(event=['enter', 'exit', 'intTrans', 'extTrans'], time=[1, 2, 3, 4], node=[1, 2, 1, 1], dest=[0, 0, 0, 2],
                           n=[3, 1, 2, 1], proportion=0, select=[1, 4, 4, 6], shift=[0, 0, 1, 0]))
    reg, notes = event_register(ev, compartments, E, N)
    assert list(reg.states) == ['S_1', 'S_1,I_1', 'S_1,I_1', 'S_3,I_3']
    assert reg.to_states[2] == 'S_2,I_2' # Aging: shift 1 moves age category 1 to 2
    assert np.isnan(reg['from'][0]) and np.isnan(reg['to'][1]) and reg['to'][3] == 2
    assert not notes
    return reg


def test_examples():
    sc.heading('Testing the SimInf examples (if present)')
    folder = examples / 'sir_herds'
    if not folder.exists():
        pytest.skip(paths.no_lfd)
    script = folder/'original'/'events_SIR.R'
    assert [k for k,v in read_script(script).items() if isinstance(v, Unknown)] == ['u0', 'events'] # Computed in R, by u0_SIR() and events_SIR()
    sim = load_siminf(script, u0=folder/'reference'/'u0.csv', events=folder/'reference'/'events.csv.gz', stop=100, **kw) # So passed in, as R wrote them
    assert not sim.imported.incomplete
    sim.run()
    ref = pd.read_csv(folder/'reference'/'results.csv')
    ref = ref[(ref.series == 'S') & (ref.time == 100)].iloc[0]
    S = sim.results.events_SIR.n_S.values[100]
    assert ref.low - 50 <= S <= ref.high + 50, f'Expected S on day 100 near SimInf\'s {ref.low:n} to {ref.high:n}, not {S:n}'

    folder = examples / 'sise3'
    sim = load_siminf(folder/'original'/'events_SISe3.R', u0=folder/'reference'/'u0.csv', events=folder/'reference'/'events.csv.gz', stop=20, **kw)
    sim.run()
    res = sim.results.events_SISe3
    assert res.n_phi.values[-1] > 0 # The infected shed into phi
    assert all(sim.diseases[0].graph.states[s].people for s in ['S_2', 'I_3']) # People, though only aging joins them to S_1
    return sim


if __name__ == '__main__':
    T = sc.timer()
    args = test_r_reader()
    sim1 = test_mparse()
    sim2 = test_dict()
    reg = test_events()
    sim3 = test_examples()
    T.toc()
