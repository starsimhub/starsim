"""
Test importing epydemix models, and exporting to epydemix (run in epydemix's own environment, `_envs/epydemix` in the lingua_franca repo, if it is there)
"""
import json
import warnings
import subprocess
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import paths
from starsim.converters.epydemix import load_epydemix, to_epydemix

kw = dict(verbose=False)
epydemix_python = paths.lf/'_envs'/'epydemix'/'bin'/'python'


def seird_model():
    """ An SEIR with deaths in two age groups, as epydemix.py writes it """
    days = 61
    return dict(
        format = 'epydemix',
        name = 'seird',
        compartments = ['S', 'E', 'I', 'R', 'D'],
        transitions = [
            dict(source='S', target='E', kind='mediated', params=['beta*r', 'I']),
            dict(source='E', target='I', kind='spontaneous', params='eps'),
            dict(source='I', target='R', kind='spontaneous', params='mu*(1-IFR)'),
            dict(source='I', target='D', kind='spontaneous', params='mu*IFR'),
        ],
        parameters = dict(
            beta = dict(kind='number', value=0.3),
            r    = dict(kind='time', value=[1.0]*20 + [0.5]*(days - 20)), # Contacts halve on day 20
            eps  = dict(kind='number', value=0.25),
            mu   = dict(kind='number', value=0.2),
            IFR  = dict(kind='age', value=[0.01, 0.1]),
        ),
        overrides = dict(eps=[dict(start_date='2020-02-01', end_date='2020-02-10', value=dict(kind='number', value=0.5))]),
        interventions = [dict(layer='school', start_date='2020-01-15', end_date='2020-02-15', name='closure', reduction_factor=0.0, new_matrix=None)],
        population = dict(name='Toy', groups=['0-19', '20+'], Nk=[6000.0, 4000.0], contacts=dict(home=[[2.0, 1.0], [1.5, 2.0]], school=[[3.0, 0.0], [0.0, 0.0]])),
        run = dict(start_date='2020-01-01', end_date='2020-03-01', dt=1.0, initial=dict(S=[5990.0, 3990.0], E=[0.0, 0.0], I=[10.0, 10.0], R=[0.0, 0.0], D=[0.0, 0.0]), parameters={}),
    )


def test_epydemix():
    sc.heading('Testing importing an epydemix model')
    with pytest.warns(ss.errors.ImportIncomplete): # The school closure is listed, not translated
        sim = load_epydemix(seird_model(), solver='ode', **kw)
    assert [k for k,_ in sim.imported.incomplete] == ['intervention closure']
    sim.run()
    dis = sim.diseases[0]
    assert list(dis.transitions.keys()) == ['S_to_E', 'E_to_I', 'I_to_D_or_R'], 'The two exits from I should be one branch'
    res = sim.results.epydemix
    assert res.I.select(age=0).values[0] == 10, 'Each group should start with its initial numbers'
    total = sum(res[c].values for c in 'SEIRD')
    assert np.allclose(total, 10_000)

    # The same model by hand
    S, E, I, R, D = ss.states(5)
    r = ss.timeseries({'2020-01-01': 1.0, '2020-01-21': 0.5})
    eps = ss.timeseries({'2020-01-01': 0.25, '2020-02-01': 0.5, '2020-02-11': 0.25})
    seird = ss.Disease(
        S_to_E = S >> E @ 0.3*r*I,
        E_to_I = E >> I @ eps,
        I_to_D_or_R = ss.Transition(I, {D: ss.by([0.01, 0.1], dims=['age']), R: None}, rate=0.2),
        init_n = dict(I=ss.by([10, 10], dims=['age'])),
        name = 'epydemix',
    )
    seird.stratify(age={0: 0.6, 20: 0.4})
    ref = ss.Sim(seird, networks=ss.contacts([[5.0, 1.0], [1.5, 2.0]]), total_pop=10_000, unit='day', start='2020-01-01', stop='2020-03-02',
                 solver='ode', aging=False, verbose=0).run() # A solver by name is that solver's defaults, as load_epydemix(solver='ode') gives
    for c in 'SEIRD':
        ours, theirs = res[c].values, ref.results.epydemix[c].values
        assert np.allclose(ours, theirs, rtol=1e-9), f'{c} should match the model written by hand'
    return sim


def sir(**kwargs):
    """ An SIR, in days unless given another unit """
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.01, name='sir')
    return ss.Sim(dis, **sc.mergedicts(dict(total_pop=1e5, unit='day', stop=60, verbose=0), kwargs))


def seird_export(births=False):
    """ An SEIR with a branch to death and Erlang stages, which epydemix carries whole; with births=True, births and deaths, which it has no kind of transition for """
    S, E, I, R, D = ss.states('S E I R D')
    trs = dict(
        infection = ss.Transition(S, E, rate=ss.perday(0.4)*I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=ss.lognormal(ss.days(8), ss.days(3))),
    )
    if births:
        trs.update(births=ss.birth(S, rate=ss.peryear(0.02)), deaths=ss.death([S, E, I, R], rate=ss.peryear(0.01)))
    dis = ss.Disease(**trs, init_prev=0.001, name='seird')
    return ss.Sim(dis, total_pop=1e5, unit='day', start='2020-03-01', stop='2020-06-01', verbose=0)


def test_epydemix_export():
    sc.heading('Testing export to epydemix')
    text, summary, lost = to_epydemix(seird_export())
    compile(text, 'seird.py', 'exec')
    assert text.count("kind='mediated'") == 7, 'An infection should be a mediated transition for each stage of I'
    assert "start_date='2020-03-01', end_date='2020-05-31'" in text, 'The dates should be kept, the last that of the last step'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        text, summary, lost = to_epydemix(seird_export(births=True))
    assert sum('moves people into or out of the model' in line for line in lost) == 11
    text, summary, lost = to_epydemix(sir(unit='week', stop=10))
    assert "params='(1 / dur_I) / 7'" in text and 'dt=7.0' in text, 'The rates should be per day, and the step in days'
    return text


@pytest.mark.skipif(not epydemix_python.exists(), reason='needs epydemix\'s environment (_envs/epydemix in the lingua_franca repo; set STARSIM_LF)')
def test_epydemix_export_run(tmp_path):
    sc.heading('Testing export to epydemix, run in epydemix and read back')
    dumper = sc.path(ss.__file__).parent/'converters'/'epydemix.py'
    run = ('import sys, json, runpy, numpy as np; ns = runpy.run_path(sys.argv[1]); res = ns["model"].run_simulations(**ns["run"], Nsim=20, rng=1); '
           'json.dump({c:np.mean([tr.compartments[f"{c}_total"] for tr in res.trajectories], axis=0).tolist() for c in ns["model"].compartments}, open(sys.argv[2], "w"))')
    for name,sim in dict(sir=sir(), seird=seird_export()).items():
        script, dumped, ran = [tmp_path/f'{name}.{ext}' for ext in ['py', 'json', 'out.json']]
        sc.savetext(script, to_epydemix(sim)[0])
        for args in [[dumper, script, dumped], ['-c', run, script, ran]]: # The JSON that load_epydemix() reads, and the mean of 20 of epydemix's runs
            out = subprocess.run([str(epydemix_python), *map(str, args)], capture_output=True, text=True, cwd=tmp_path, timeout=300)
            assert out.returncode == 0, out.stderr
        back = load_epydemix(str(dumped), solver='ode', **kw)
        assert back.imported.incomplete == []
        sim.run(); back.run()
        ours, again = sim.results[sim.diseases[0].name], back.results.epydemix
        for state in 'SR':
            assert np.allclose(ours[state], again[state], rtol=1e-9, atol=1e-9), f'{state} should be the same read back'

        # epydemix is a chain binomial in daily steps, which Starsim's binomial solver is too: the means of their runs agree
        theirs = json.load(open(ran))
        runs = [load_epydemix(str(dumped), rand_seed=seed, **kw).run().results.epydemix for seed in range(10)]
        for state in ['S', 'E' if 'E' in theirs else 'I']:
            binomial = np.mean([res[state].values for res in runs], axis=0)
            n = len(theirs[state]) # A count on a date in epydemix is at the end of its step
            assert np.abs(binomial[1:n+1] - theirs[state]).max() < 0.05*binomial.max(), f'{state} should be the same in epydemix, on average'
    return back


if __name__ == '__main__':
    T = sc.timer()
    sim1 = test_epydemix()
    text = test_epydemix_export()
    T.toc()
