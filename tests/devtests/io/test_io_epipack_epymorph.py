"""
Test importing epipack and epymorph models, and exporting to epipack and epymorph

The models are as `starsim/converters/epipack.py` and `epymorph.py` write them, run as
scripts in those frameworks' own environments (`_envs/` in the lingua_franca repo);
the reference numbers are from epipack 0.1.5 and epymorph 1.2.0. The exports are run in
those environments, if they are there.
"""
import json
import warnings
import subprocess
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import paths
from scipy.integrate import solve_ivp
from starsim.converters import sniff
from starsim.converters.epipack import load_epipack, split, to_epipack
from starsim.converters.epymorph import load_epymorph, to_epymorph

kw = dict(verbose=False)
epipack_python = paths.lf/'_envs'/'epipack'/'bin'/'python'
epymorph_python = paths.lf/'_envs'/'epymorph'/'bin'/'python'
has_epymorph = epymorph_python.exists()


def sir_sim():
    """ An SIR, in days """
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(7)), init_prev=0.01, name='sir')
    return ss.Sim(dis, total_pop=1000, unit='day', stop=60, verbose=0)


def seird_sim():
    """ Births, deaths from every state, a branch, and a lognormal dwell time, which is written as Erlang stages """
    S, E, I, R, D = ss.states('S E I R D')
    dis = ss.Disease(
        births    = ss.birth(S, rate=ss.peryear(0.02)),
        infection = ss.Transition(S, E, rate=ss.perday(0.4) * I),
        onset     = ss.Transition(E, I, dur=ss.days(4)),
        outcome   = ss.Transition(I, {R: None, D: 0.1}, dur=ss.lognormal(ss.days(8), ss.days(3))),
        deaths    = ss.death([S, E, I, R], rate=ss.peryear(0.01)),
        init_prev = 0.01, name='seird')
    return ss.Sim(dis, total_pop=1e5, unit='day', stop=120, verbose=0)


def sirs_epipack():
    """ The SIRS in epipack's README, as epipack.py writes it """
    return dict(
        format = 'epipack',
        name = 'EpiModel',
        compartments = ['S', 'I', 'R'],
        population = 1000.0,
        dynamic = False,
        events = [
            dict(kind='linear', rate='1.0*I', changes=dict(I=-1, R=1)),
            dict(kind='linear', rate='0.07142857142857142*R', changes=dict(R=-1, S=1)),
            dict(kind='quadratic', rate='2.5*S*I', changes=dict(S=-1, I=1)),
        ],
        parameters = {},
        initial = dict(S=990.0, I=10.0, R=0.0),
        run = dict(tmax=40.0, solver='ode'),
        incomplete = [],
    )


def births_epipack():
    """ A symbolic SIRS with births and deaths, which divides by the population as it changes """
    return dict(
        format = 'epipack',
        name = 'SymbolicEpiModel',
        compartments = ['S', 'I', 'R'],
        population = 1000.0,
        dynamic = True,
        events = [
            dict(kind='birth', rate='1000*mu', changes=dict(S=1)),
            dict(kind='linear', rate='I*rho', changes=dict(I=-1, R=1)),
            dict(kind='linear', rate='R*omega', changes=dict(R=-1, S=1)),
            dict(kind='linear', rate='S*mu', changes=dict(S=-1)),
            dict(kind='linear', rate='I*mu', changes=dict(I=-1)),
            dict(kind='linear', rate='R*mu', changes=dict(R=-1)),
            dict(kind='linear', rate='0.05*I', changes=dict(I=-1)),
            dict(kind='quadratic', rate='I*S*eta', changes=dict(S=-1, I=1)),
        ],
        parameters = dict(eta=2.5, rho=1.0, omega=1/14, mu=0.01),
        initial = dict(S=990.0, I=10.0, R=0.0),
        run = dict(tmax=60.0, solver='ode'),
        incomplete = [],
    )


def math_rate():
    """ A rate of no usual form, and one that varies in time, which epipack carries as they are """
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.5)*I.frac*np.maximum(np.exp(-2*I.frac), 0.2)), recovery=ss.Transition(I, R, rate=ss.seasonal(ss.perday(0.1), amplitude=0.2)), init_prev=0.01, name='behavior')
    return ss.Sim(dis, total_pop=1000, unit='day', stop=60, verbose=0)


def test_epipack_export():
    sc.heading('Testing export to epipack')
    text, summary, lost = to_epipack(seird_sim()) # Births, deaths, a branch, and Erlang stages
    compile(text, 'seird.py', 'exec')
    assert text.count('beta, E, I_') == 7, 'An infection should be a transmission process for each stage of I'
    assert '(None, 5.479452054794521e-05 * (S + E + I + R + D), S), # births' in text
    assert 'allow_nonzero_column_sums=True' in text
    text, summary, lost = to_epipack(math_rate())
    assert lost == [], 'Every rate should be carried'
    assert 'sympy.Max(sympy.exp(-2 * (I / N)), 0.2)' in text and "t = sympy.Symbol('t')" in text and 'ignore_rate_position_checks=True' in text
    return text


@pytest.mark.skipif(not epipack_python.exists(), reason='needs epipack\'s environment (_envs/epipack in the lingua_franca repo; set STARSIM_LF)')
def test_epipack_export_run(tmp_path):
    sc.heading('Testing export to epipack, run in epipack and read back')
    dumper = sc.path(ss.__file__).parent/'converters'/'epipack.py'
    run = 'import sys, json, runpy; ns = runpy.run_path(sys.argv[1]); res = ns["model"].integrate(ns["times"]); json.dump({str(k):v.tolist() for k,v in res.items()}, open(sys.argv[2], "w"))'
    for make in [sir_sim, seird_sim, math_rate]:
        sim = make()
        script, dumped, ran = [tmp_path/f'{make.__name__}.{ext}' for ext in ['py', 'json', 'out.json']]
        sc.savetext(script, to_epipack(sim)[0])
        for args in [[dumper, script, dumped], ['-c', run, script, ran]]: # The JSON that load_epipack() reads, and epipack's own integration
            out = subprocess.run([str(epipack_python), '-W', 'ignore', *map(str, args)], capture_output=True, text=True, cwd=tmp_path, timeout=300)
            assert out.returncode == 0, out.stderr
        sim.run()
        ours, theirs = sim.results[sim.diseases[0].name], json.load(open(ran))
        for state in ['S', 'R']:
            assert np.allclose(ours[state], theirs[state], rtol=1e-4, atol=1e-3), f'{state} should be the same in epipack'
        if make is not math_rate: # Which load_epipack() reads but for the time
            back = load_epipack(str(dumped), **kw).run()
            assert back.imported.incomplete == []
            for state in ['S', 'R']:
                assert np.allclose(ours[state], back.results.epipack[state], rtol=1e-9, atol=1e-9), f'{state} should be the same read back'
    return


def sirs_epymorph():
    """ epymorph's SIRS and centroids movement in three places, as epymorph.py writes it """
    return dict(
        format = 'epymorph',
        name = 'SIRS',
        compartments = ['S', 'I', 'R'],
        places = ['a', 'b', 'c'],
        edges = [dict(source='S', target='I', rate='I*S*beta/N'), dict(source='I', target='R', rate='I*gamma'), dict(source='R', target='S', rate='R*xi')],
        parameters = dict(beta=dict(kind='number', value=0.4), gamma=dict(kind='number', value=0.25), xi=dict(kind='number', value=0.01)),
        initial = [[199900.0, 100.0, 0.0], [100000.0, 0.0, 0.0], [50000.0, 0.0, 0.0]],
        start = '2020-01-01',
        days = 120,
        movement = [[0.0, 0.009995, 0.004021], [0.009295, 0.0, 0.006078], [0.004379, 0.007259, 0.0]],
        incomplete = [],
    )


def peak(res, state, place):
    """ The height and day of a state's peak in a place """
    vals = np.asarray(res[state].select(patch=place).values)
    return vals.max(), vals.argmax()


def test_epipack():
    sc.heading('Testing importing an epipack model')
    d = sirs_epipack()
    sim = load_epipack(d, stop=41, **kw).run()
    res = sim.results.epipack
    for state,ref in dict(S=411.8700587, I=40.7136001, R=547.4163412).items(): # EpiModel.integrate() at t=40
        assert np.isclose(res[state].values[40], ref, rtol=1e-4), f'{state} should match epipack at t=40'
    assert 'S_to_I' in sim.diseases[0].transitions
    assert 'exposure to I' in sim.imported.code, 'A fixed population equal to the number of people should be read as the total'
    return sim


def test_epipack_population():
    sc.heading('Testing how epipack divides by its population')
    sim = load_epipack(births_epipack(), stop=61, **kw).run() # The population changes, and is corrected for
    res = sim.results.epipack
    for state,ref in dict(S=380.7655678, I=38.9148224, R=478.4695958).items(): # SymbolicEpiModel.integrate() at t=60
        assert np.isclose(res[state].values[60], ref, rtol=1e-4), f'{state} should match epipack at t=60'

    # Not corrected for: the same terms stay divided by the 1,000 at the start as the population falls
    d = births_epipack()
    d['dynamic'] = False
    fixed = load_epipack(d, stop=61, **kw)
    assert '/ 1000' in fixed.imported.code or '/1000' in fixed.imported.code, 'A model that changes in size should keep the fixed division'
    fixed.run()
    assert fixed.results.epipack.I.values[60] != res.I.values[60]

    # A population that isn't the number of people isn't the total, and epipack's default of 1 divides by nothing
    for size in [2000, 1]:
        d = sirs_epipack()
        d['population'] = float(size)
        sim = load_epipack(d, stop=3, **kw).run()
        deriv = lambda t,y: [y[2]/14 - 2.5*y[0]*y[1]/size, 2.5*y[0]*y[1]/size - y[1], y[1] - y[2]/14] # epipack's equations
        ref = solve_ivp(deriv, [0, 2], [990, 10, 0], rtol=1e-8, atol=1e-8).y[:, -1]
        vals = [sim.results.epipack[state].values[2] for state in 'SIR']
        assert np.allclose(vals, ref, rtol=1e-3), f'A fixed population of {size} should be kept in the division'
    return


def test_epipack_reading():
    sc.heading('Testing what epipack events become, and what is listed')
    assert split(dict(A=-1, B=1)) == [('A', 'B')]
    assert split(dict(A=-1, B=1, C=1)) == [('A', 'B'), (None, 'C')], 'A fission is a flow and an inflow'
    assert split(dict(A=-1, B=-1, C=2)) == [('A', 'C'), ('B', 'C')]
    assert split(dict(A=1)) == [(None, 'A')]
    assert split(dict(A=-1)) == [('A', None)]

    # A parameter with no value is listed, and can be given
    d = births_epipack()
    d['parameters']['eta'] = None
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = load_epipack(sc.dcp(d), stop=5, **kw)
    assert ('parameter eta' in [k for k,_ in sim.imported.incomplete])
    assert 'parameter eta' not in [k for k,_ in load_epipack(d, stop=5, pars=dict(eta=2.5), **kw).imported.incomplete]

    # What the dump couldn't read is listed; the run's solver is used; time in a rate is listed
    d = sirs_epipack()
    d['incomplete'] = [['network', 'its 2000 nodes and their links']]
    d['run'] = dict(tmax=5, solver='gillespie')
    d['events'][0]['rate'] = '(1.0 + cos(t))*I'
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = load_epipack(d, **kw)
    assert {'network', 'time'} <= {k for k,_ in sim.imported.incomplete}
    assert isinstance(sim.init().solver, ss.solvers.gillespie)
    return


def test_epymorph():
    sc.heading('Testing importing an epymorph model, against its own runs')
    sims = [load_epymorph(sirs_epymorph(), rand_seed=i, **kw).run() for i in range(5)]
    assert isinstance(sims[0].solver, ss.solvers.tau)
    assert len(sims[0].results.epymorph.I.values) >= 120
    refs = dict(a=(16980, 48), b=(9099, 54), c=(4488, 56)) # The mean of 20 BasicSimulator runs: the peak infectious by place, and its day
    for place,(height,day) in refs.items():
        vals = np.mean([sim.results.epymorph.I.select(patch=place).values for sim in sims], axis=0)
        assert abs(vals.max()/height - 1) < 0.05, f'The peak in {place} should match epymorph\'s'
        assert abs(vals.argmax() - day) <= 3, f'The day of the peak in {place} should match epymorph\'s'
    return sims


def test_epymorph_places():
    sc.heading('Testing what epymorph parameters and movement become')

    # Without movement the places stay apart, as epymorph's nodes do
    d = sirs_epymorph()
    d['movement'] = np.zeros((3, 3)).tolist()
    res = load_epymorph(d, **kw).run().results.epymorph
    assert peak(res, 'I', 'a')[0] > 1000
    assert peak(res, 'I', 'b')[0] == 0 and peak(res, 'I', 'c')[0] == 0, 'Places with no movement should stay apart'

    # Parameters by place and in time; births and deaths; a total that is not every state, which stays as written
    d = sirs_epymorph()
    days = d['days']
    d['parameters']['beta'] = dict(kind='place', value=[0.4, 0.5, 0.6])
    d['parameters']['gamma'] = dict(kind='time', value=[0.25]*60 + [0.5]*(days - 60))
    d['parameters']['xi'] = dict(kind='time_place', value=(np.ones((days, 3))*[0.01, 0.02, 0.03]).tolist())
    d['edges'] += [dict(source=None, target='S', rate='0.001*N'), dict(source='S', target=None, rate='0.001*S')]
    d['incomplete'] = [['immobile compartments', "['H'] are immobile in epymorph; here everyone commutes alike"]]
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = load_epymorph(d, **kw)
    assert {'immobile compartments', 'parameter xi'} == {k for k,_ in sim.imported.incomplete}
    assert "ss.by([0.4, 0.5, 0.6], dims=['patch'])" in sim.imported.code
    assert 'ss.timeseries' in sim.imported.code
    sim.run()
    return sim


def sir_births(**kwargs):
    """ An SIR with births and deaths, to export """
    pars = dict(diseases=ss.SIR(beta=ss.perday(0.3), dur_inf=ss.days(10), init_prev=0.01), demographics=[ss.birth(rate=ss.peryear(0.5)), ss.death(rate=ss.peryear(0.5))],
                total_pop=10_000, unit='day', stop=ss.days(60), solver='ode', verbose=0)
    return ss.Sim(**(pars | kwargs))


def stages(res, state):
    """ A state's counts, summed over its stages (I_1, I_2, ...) where it was written as them """
    return res[state].values if state in res else sum(res[k].values for k in res.keys() if k.startswith(f'{state}_') and k[len(state)+1:].isdigit())


def test_epymorph_export():
    sc.heading('Testing exporting to epymorph')
    text, summary, lost = to_epymorph(sir_births())
    compile(text, 'sir.py', 'exec')
    assert 'N = Max(1, S + I + R)' in text, 'The total should be guarded as epymorph guards it'
    assert 'edge(BIRTH, S' in text and 'edge(S, DEATH' in text
    assert any('Poisson' in line for line in lost)

    # A seasonal beta is a parameter by day; a sim in years has its rates in days
    sim = sir_births(diseases=ss.SIR(beta=ss.seasonal(mean=ss.perday(0.3), amplitude=0.3), dur_inf=ss.days(10)), demographics=None, start='2020-01-01', stop='2020-03-01')
    text, summary, lost = to_epymorph(sim)
    assert 't = np.arange(days)' in text and "TimeFrame.of('2020-01-01', days)" in text
    assert any('"beta", which varies in time' in line for line in lost)
    text, summary, lost = to_epymorph(ss.Sim(diseases=ss.SIR(beta=36.5, dur_inf=0.1), total_pop=1000, unit='year', start=2000, stop=2001, solver='ode', verbose=0))
    assert 'beta * S * I / N / 365.0' in text, 'The rates should be in days, with the parameters as the sim has them'
    return text


@pytest.mark.skipif(not has_epymorph, reason='needs epymorph\'s environment (STARSIM_LF)')
def test_epymorph_roundtrip(tmp_path):
    sc.heading('Testing an export to epymorph, run in epymorph and read back')
    sim = sir_births()
    text, summary, lost = to_epymorph(sim)
    script, dumped = tmp_path/'sir.py', tmp_path/'sir.json'
    script.write_text(text)
    subprocess.run([epymorph_python, ss.converters.epymorph.__file__, f'{script}:build_rume', str(dumped)], check=True, capture_output=True)
    back = load_epymorph(str(dumped), solver='ode', **kw).run()
    res = sim.run().results.sir
    for state in ['S', 'I', 'R']:
        assert np.allclose(stages(back.results.epymorph, state), res[state].values, rtol=1e-9), f'{state} should be the same once read back'

    # epymorph's own runs: on average, Euler's method with a step of a day
    code = f'import runpy, json, numpy as np; from epymorph.kit import BasicSimulator, default_rng; rume = runpy.run_path({str(script)!r}, run_name="x")["rume"]; ' \
           'print(json.dumps(np.mean([BasicSimulator(rume).run(rng_factory=default_rng(i)).compartments[:, 0, 1:3].sum(axis=1) for i in range(10)], axis=0).tolist()))'
    out = subprocess.run([epymorph_python, '-c', code], check=True, capture_output=True, text=True)
    infected = np.array(json.loads(out.stdout.splitlines()[-1]))
    euler = sim.new(solver=ss.solvers.ode(integrator='euler', substeps=1)).run().results.sir.I.values[1:] # epymorph's values are at the end of each day
    assert abs(infected.max()/euler.max() - 1) < 0.03, 'The peak should match Euler\'s method with a step of a day'
    assert abs(infected.argmax() - euler.argmax()) <= 2
    return back


def test_sniff():
    sc.heading('Testing telling epipack and epymorph files apart')
    for format,make in dict(epipack=sirs_epipack, epymorph=sirs_epymorph).items():
        filename = sc.path(f'{format}_test_sniff.json')
        try:
            sc.savejson(filename, make())
            assert sniff(str(filename)) == format
        finally:
            filename.unlink(missing_ok=True)
    return


if __name__ == '__main__':
    T = sc.timer()
    test_epipack()
    test_epipack_population()
    test_epipack_reading()
    test_epipack_export()
    test_epymorph()
    test_epymorph_places()
    test_epymorph_export()
    test_sniff()
    T.toc()
