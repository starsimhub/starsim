"""
Test importing SimInf models, and exporting to SimInf (run in SimInf's environment, if it is there)
"""
import os
import tempfile
import subprocess
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import paths
from starsim.converters.siminf import load_siminf, read_script, event_register, as_matrix, predefined, to_siminf

lf = sc.path(os.getenv('STARSIM_LF', paths.root/'lingua_franca')) # The lingua_franca repo, for SimInf's environment
rscript = lf/'_envs'/'siminf'/'bin'/'Rscript'

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
        sc.savetext(path := f'{tmp}/model.R', script)
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
        sc.savetext(path := f'{tmp}/model.R', script)
        sim = load_siminf(path, **kw)
    choices = '\n'.join(sim.imported.choices)
    assert 'exposure to I' in choices # beta*S*I/N, with N <- S + I + R
    assert 'mu * (S.n + I.n + R.n)' in choices # Births in proportion to everyone
    assert isinstance(sim.init().solver, ss.solvers.gillespie)
    sim.run()
    res = sim.results.model
    assert res.I.values.max() > 50
    assert np.isclose((res.S + res.I + res.R).values[-1], 2985, rtol=0.02) # Births and deaths balance
    assert res.I.strata.shape[1] == 3 # By node
    return sim


def test_dict():
    sc.heading('Testing a model given as Python data, with a catalytic transition and parameters by node')
    model = dict(transitions=['S + I -> beta*S*I/(S+I) -> 2*I', 'I -> gamma*I -> S'], compartments=['S', 'I'],
                 u0=dict(S=[95, 90], I=[5, 10]), ldata=dict(beta=[0.5, 0.0], gamma=[0.1, 0.1]), tspan=list(range(1, 101)))
    sim = load_siminf(model, **kw)
    sim.run()
    I = sim.results.siminf.I.strata
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


def make_seird():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    seird = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=ss.lognormal(ss.days(8), ss.days(3))),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return ss.Sim(seird, total_pop=1e4, unit='day', stop=100, verbose=0)


def test_export(tmp_path):
    sc.heading('Testing export to SimInf, read back, and run in SimInf')
    sim = make_seird()
    text, summary, lost = to_siminf(sim)
    sc.savetext(tmp_path/'seird.R', text)
    assert '"(double)N <- S + E + I + R + D"' in text and '"I_7 -> 0.1 * 7.0 * I_7 / dur_I -> D"' in text and 'tspan <- 1:100' in text
    assert '7 exponential stages' in lost[0]
    back = load_siminf(tmp_path/'seird.R', solver='ode', verbose=False) # Read back as Starsim, as an ODE
    sim.run(); back.run()
    res = sim.results.seird
    I = sum(back.results.seird[f'I_{k}'].values for k in range(1, 8))
    assert np.allclose(I, res.I.values, rtol=1e-9) and np.allclose(back.results.seird.S.values, res.S.values, rtol=1e-9), 'the round trip should give the same ODE'
    if not rscript.exists():
        pytest.skip(f'needs SimInf\'s environment ({rscript}; set STARSIM_LF)')
    code = f'pdf(NULL); source("{tmp_path/"seird.R"}"); set.seed(1); m <- mparse(transitions = transitions, compartments = compartments, gdata = gdata, u0 = u0[rep(1, 100), ], tspan = tspan); ' \
           f'write.csv(aggregate(. ~ time, trajectory(run(m))[, -1], mean), "{tmp_path/"seird.csv"}", row.names = FALSE)' # The mean of 100 nodes
    out = subprocess.run([str(rscript), '-e', code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path/'seird.csv')
    I = df[[f'I_{k}' for k in range(1, 8)]].sum(axis=1).values
    assert np.abs(I - res.I.values[df.time]).max() < 0.03*res.I.values.max(), 'the mean of SimInf\'s runs should be near the ODE' # 0.4% with this seed
    return back


if __name__ == '__main__':
    T = sc.timer()
    args = test_r_reader()
    sim1 = test_mparse()
    sim2 = test_dict()
    reg = test_events()
    sim3 = test_export(sc.path(tempfile.mkdtemp()))
    T.toc()
