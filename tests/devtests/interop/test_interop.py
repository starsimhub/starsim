"""
Test importing models from other frameworks (ss.interop.load()), writing a sim as code, and comparing two models
"""

# %% Imports and settings
import runpy
import tempfile
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.interop
import paths
import corpus

kw = dict(verbose=0)

# An EMULSION model with a fixed stay, a per-step probability, deaths, a capacity, and things that are not translated
emulsion_yaml = """
model_name: seirq
time_info: {time_unit: days, delta_t: 1, total_duration: '2*50'}
levels:
  population: {aggregation_type: compartment}
processes:
  population: [infection]
grouping:
  population:
    infection: {machine_name: health_state, key_variables: [health_state]}
state_machines:
  health_state:
    states:
      - S: {name: Susceptible}
      - E: {name: Exposed}
      - I: {name: Infectious}
      - Q: {name: Quarantined, duration: quarantine_duration}
      - R: {name: Resistant}
      - D: {autoremove: yes}
    transitions:
      - {from: S, to: E, rate: 'force_of_infection'}
      - {from: E, to: I, proba: p_onset}
      - {from: I, to: R, rate: recovery}
      - {from: I, to: Q, rate: detection, when: 'some_period'}
      - {from: Q, to: R, proba: 1}
      - {from: I, to: D, rate: mortality}
parameters:
  initial_population_size: {value: 1000}
  initial_infected: {value: 10}
  transmission_I: {value: 0.5}
  p_onset: {value: 0.2}
  recovery: {value: 0.1}
  mortality: {value: 0.001}
  quarantine_duration: {value: '2*5'}
  quarantine_size: {value: 20}
  detection: {value: '0.05*StrictLessThan(total_Q, quarantine_size)'}
  force_of_infection: {value: 'transmission_I * total_I / total_population'}
  prevalence_pct: {value: '100 * (total_E + total_I) / total_population'}
initial_conditions:
  population:
    - population:
        - total: 'initial_population_size'
        - vars: [S]
          amount: 'initial_population_size - initial_infected'
        - vars: [I]
          amount: 'initial_infected'
outputs:
  population: {period: 1, extra_vars: [prevalence_pct]}
"""


# Plain state names, a probability per day with a weekly step, births, and a metapopulation level, which is listed as not translated
emulsion_weekly = """
time_info: {time_unit: days, delta_t: 7, total_duration: 14}
levels: {population: {aggregation_type: compartment}, metapop: {contains: [population], aggregation_type: metapopulation}}
state_machines:
  health_state:
    states: [S, I, R]
    transitions: [{from: S, to: I, rate: 0}, {from: I, to: R, proba: 0.1}]
    productions: [{from: R, to: S, rate: 0.01}]
prototypes: {population: [{infected: {health_state: I}}]}
initial_conditions: {population: [{prototype: infected, amount: 1000}]}
"""


def emulsion_file(folder):
    """ Write the EMULSION model to a file """
    filename = sc.path(folder) / 'seirq.yaml'
    sc.savetext(filename, emulsion_yaml)
    return filename


def test_rates():
    sc.heading('Testing how rates from other frameworks are read')
    src = ss.interop.Source('rates', format='test')
    for s in ['S', 'I', 'R']:
        src.add_state(s, init=dict(S=990, I=10).get(s, 0))
    src.add_state('W', people=False)
    for p,v in dict(beta=0.3, gamma=0.1, xi=10, K=1e4, mu=0.01, pop=1000, K2=2000).items():
        src.add_par(p, v)
    src.totals = ['N']
    flows = [
        ('S', 'I', 'beta * S * I / N', 'total'),       # 1 and 2: the source taken out, then exposure
        ('S', 'I', 'beta * I / (S + I + R)', 'person'), # 2: the total written as a sum of every state of people
        ('S', 'I', 'beta * S * I', 'total'),           # 3: mass action, kept as counts
        ('I', 'R', 'gamma * I', 'total'),              # 4: spontaneous
        ('S', 'I', 'beta * W / (W + K)', 'person'),    # 5 and 6: a count that is not people
        (None, 'W', 'xi * I', 'person'),               # An inflow is a total
        (None, 'S', 'mu * N', 'person'),
        ('S', 'I', 'beta * S * I / pop', 'total'),     # 2: a parameter equal to the number of people is the total
        ('S', 'I', 'beta * S * I / K2', 'total'),      # 3: one that is not, is not
    ]
    rates = ss.interop.core.Rates(src)
    out = [rates.classify(sc.objdict(src=a, dst=b, rate=r, per=per))[0] for a,b,r,per in flows]
    assert out == ['beta * I', 'beta * I', 'beta * I.n', 'gamma', 'beta * W.n / (W.n + K)', 'xi * I.n', 'mu * (S + I + R).n', 'beta * I', 'beta * I.n / K2'], out
    rules = [rates.classify(sc.objdict(src=a, dst=b, rate=r, per=per))[2] for a,b,r,per in flows]
    assert rules[:5] == [2, 2, 3, 4, 5] and rules[7:] == [2, 3]
    return out


def test_emulsion(tmp_path):
    sc.heading('Testing an EMULSION import')
    filename = emulsion_file(tmp_path)
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.interop.load(filename, **kw) # The format is told from the file
    imp = sim.imported
    assert imp.format == 'EMULSION'
    assert any('when' in name for name,why in imp.incomplete), 'The calendar condition should be listed as not translated'
    sim.init()
    dis = sim.diseases.seirq
    assert set(dis.states.keys()) == {'S', 'E', 'I', 'Q', 'R'} # D is removed: an outflow
    assert sim.pars.solver == 'binomial'
    assert 'force_of_infection' in dis.graph.transitions # A rate given by a named expression names its transition
    assert dis.graph.transitions.force_of_infection.contact
    sim.run()
    res = sim.results.seirq
    full = res.Q.values[:-1] >= 20 # The capacity is a condition on the count at the start of a step
    assert full.any() and (res.detection.values[1:][full] == 0).all()
    assert np.isclose(res.prevalence_pct[10], 100*(res.E[10] + res.I[10])/(res.S[10] + res.E[10] + res.I[10] + res.Q[10] + res.R[10]))

    # The same model as an ODE agrees with the stochastic one on average
    ode = ss.interop.load(filename, format='emulsion', **kw)
    ode.pars.solver = 'ode'
    ode.run()
    runs = ss.interop.load(filename, format='emulsion', **kw).run(n_runs=50).results.seirq.R.values[:, -1].mean()
    assert np.isclose(runs, ode.results.seirq.R[-1], rtol=0.1)

    # A probability is per time unit, as EMULSION reads it, not per step
    sc.savetext(tmp_path/'weekly.yaml', emulsion_weekly)
    with pytest.warns(ss.errors.ImportIncomplete):
        weekly = ss.interop.load(tmp_path/'weekly.yaml', format='emulsion', **kw)
    listed = [name for name,why in weekly.imported.incomplete]
    assert 'levels: metapop' in listed and 'health_state: production R -> S' in listed # Listed, not dropped
    assert 'import-model' in weekly.imported.prompt # What to ask an AI assistant to finish it
    left = weekly.new(solver='ode', substeps=None).run().results.weekly.I[1]/1000 # The ODE with its own substeps (EMULSION's compartments take whole steps)
    assert np.isclose(left, 0.9**7, rtol=1e-3) # 47.8% still infectious after a 7-day step, as in EMULSION (90% if the probability were per step)

    # The originals: EMULSION's SIR as compartments, individuals, and hybrid groups, against 200 EMULSION runs
    if paths.lfd.exists():
        folder = paths.lfd/'examples/emulsion/sir'
        ref = sc.dataframe.read_csv(folder/'reference/results.csv')
        for kind in ['compart', 'IBM', 'hybrid']:
            sir = ss.interop.load(folder/f'original/{kind}_SIR.yaml', **kw)
            assert not sir.imported.incomplete and 'n=100;' in repr(sir) # 100 animals, not the default number of agents
            final = sir.run(n_runs=200).results[f'{kind}_SIR'].R.values[:, -1].mean() # With one infected at the start, many runs die out, so the mean needs many
            series = dict(compart='compartment', IBM='ibm', hybrid='hybrid')[kind] + '_R'
            expected = ref[(ref.series == series) & (ref.time == 100)].value.iloc[0]
            rtol = 0.1 if kind == 'compart' else 0.15 # As agents, more runs die out at a daily step (24% of them, against 19% at a tenth of a day), a difference of order dt
            assert np.isclose(final, expected, rtol=rtol), f'{kind}: {final} recovered at day 100 against EMULSION\'s {expected}'
    return sim


def test_to_code(tmp_path):
    sc.heading('Testing sim.to_code(): the code builds the same model')
    def same(sim):
        code = sim.to_code()
        ns = {}
        exec(code, ns)
        a, b = [{k:v for k,v in s.to_spec().items() if k != 'meta'} for s in [sim, ns['sim']]]
        for m in a['modules'] + b['modules']:
            m['guessed'] = sorted(g for g in m['guessed'] if not g.endswith(('.name', '.import'))) # Named explicitly in the code
        assert a == b
        return code

    # A hand-written model with a branch, a distribution, an outcome, a result, and a stratification
    S, I, R, D = ss.states('S I R D')
    V = ss.State(rel_sus=0.2, label='Vaccinated')
    dur = ss.par(ss.lognorm_ex(mean=ss.days(10), std=ss.days(3)), name='dur')
    sir = ss.Disease(S, I, R, D, V, infection=S >> I @ ss.par(ss.perday(0.3), bounds=[0.1, 1], name='beta') * I,
                     recovery=ss.Transition(I, [R, D], dur=dur, p=[0.9, None]), vax=S >> V @ ss.perday(0.01),
                     cases=ss.outcome('infection', delay=ss.days(3)), prev=I/(S + I + R), init_n=dict(I=10), name='sir')
    sir.stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2])
    same(ss.Sim(sir, total_pop=1000, stop=ss.days(50), unit='day', **kw))

    # An imported model: the code says how it was read and what was not translated
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.interop.load(emulsion_file(tmp_path), **kw)
    code = same(sim)
    assert 'Imported from EMULSION' in code and '# Not translated (ImportIncomplete):' in code

    # Every Starsim name is written with ss., and a function from its source, with the values it uses, so the code builds the same model
    def make(beta=0.3, recovery=ss.days(10)):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ recovery, init_prev=0.01, name='sir')
    same(ss.Sim(make(beta=ss.by(age=[0.3, 0.15])), stratify=dict(age=[0, 15]), unit='day', stop=20, **kw))
    sim = ss.Sim(make(recovery=clearance), unit='day', stop=20, **kw)
    sim.to_code(tmp_path/'clearance.py')
    assert runpy.run_path(tmp_path/'clearance.py')['sim'].hash() == sim.hash()

    # Only a rate as a whole has a unit: R0 in R0 * gamma * I is written as a number, so the code gives the same epidemic in any unit
    R0, gamma = ss.par(2.5, name='R0'), ss.par(0.2, name='gamma')
    sim = ss.Sim(make(beta=R0*gamma, recovery=gamma), total_pop=1000, unit='day', stop=100, **kw)
    ns = {}
    exec(sim.to_code(), ns)
    peak = lambda s: s.run().results.sir.I.max()
    assert np.isclose(peak(ns['sim'].new(unit='week', dt=ss.days(1))), peak(sim), rtol=1e-3)

    # What cannot be written is listed in the code and in a warning
    with pytest.warns(ss.errors.LossyExport):
        code = ss.Sim(make(beta=0.05), networks=ss.RandomNet(), n_agents=100, **kw).to_code()
    assert 'network "randomnet"' in code
    return code


def test_round_trip(tmp_path):
    sc.heading('Testing the round trip of every corpus model: to_code(), then save() and load()')
    def results(sim):
        return {k:np.asarray(v.values, dtype=float) for k,v in sim.results.flatten().items() if isinstance(v, ss.Result)}

    def check(name, sim):
        fn = tmp_path/f'{name}.py'
        sim.to_code(fn)
        new = runpy.run_path(fn)['sim'] # From a file, so that the functions in it have their source
        assert new.hash() == sim.hash(), f'{name}: {ss.diff(sim, new, output=True)}'
        a, b = results(sim.run()), results(new.run())
        assert all(np.allclose(a[k], b[k]) for k in a), f'{name}: the code gives different results'
        sim.save(tmp_path/f'{name}.sim') # After a run, so that every module shrinks
        loaded = results(ss.load(tmp_path/f'{name}.sim'))
        assert all(np.allclose(a[k], loaded[k]) for k in a), f'{name}: the loaded sim has different results'
        return

    # Every corpus model that to_code() can write in full: deliveries, contact matrices, transfers, a duration with no unit
    lossy = ['coinfection', 'network', 'threshold', 'starsim3', 'poisson_weekly'] # Networks of agents, a lambda, and data
    for name,make in corpus.models.items():
        if name in lossy:
            with pytest.warns(ss.errors.LossyExport):
                make().to_code()
        else:
            check(name, make())

    # What the fuzzer found: time series keys and a modifier's times as numbers (not dates), conditions, a matrix with its own age groups, and a start in years; and a prior
    S, I, R = ss.states('S I R')
    beta = ss.par(ss.timeseries({0: 0.3, 20.5: 0.2}), name='beta')
    dur = ss.par(ss.days(10), bounds=[5, 15], prior=ss.uniform(5, 15), name='dur')
    sir = ss.Disease(infection=S >> I @ beta * I, recovery=I >> R @ dur, init_n=dict(I=50), name='sir').stratify(risk=['lo', 'hi'])
    mixing = ss.contacts([[3, 1], [1, 2]], ages=[0, 15])
    distancing = ss.modify(sir.pars.beta, multiply=0.5, start=12.5, stop=30, where=sir.risk == 'hi')
    vaccine = ss.vaccinate(ss.Vaccine(efficacy=0.8, waning=ss.days(50), name='vx'), coverage=0.3, interval=ss.days(7), start=5, where=ss.ppl.age < 15)
    check('fuzzed', ss.Sim(sir, networks=mixing, interventions=[distancing, vaccine], stratify=dict(age=[0, 15]), unit='day', stop=60, total_pop=10_000, **kw))
    check('years', ss.Sim(corpus.make_sir(beta=ss.perday(0.3), dur=ss.days(10)), start=2000, stop=2001, total_pop=10_000, **kw)) # Rates per day, so the unit is days and 2000 is in years
    return


clearance_rate = 0.2
def clearance(sim):
    """ A rate given as a function, for test_to_code() """
    return clearance_rate


def test_diff(tmp_path):
    sc.heading('Testing ss.diff(): an import and a hand translation are the same model')
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.interop.load(emulsion_file(tmp_path), **kw)
    def hand(beta=0.5):
        S, E, I, Q, R = ss.states('S E I Q R')
        seirq = ss.Disease(
            infection  = S >> E @ ss.perday(beta) * I,
            onset      = E >> I @ -np.log(1 - 0.2),
            recovery   = I >> R @ ss.perday(0.1),
            detection  = I >> Q @ (0.05 * (Q.n < 20)),
            release    = Q >> R @ ss.constant(ss.days(10)),
            death      = I >> ss.null @ ss.perday(0.001),
            init_n     = dict(I=10),
            name       = 'seirq',
        )
        return ss.Sim(seirq, total_pop=1000, stop=ss.days(100), solver='binomial', substeps=1, **kw)
    text = ss.diff(sim, hand(), output=True)
    lines = dict(line.split(':', 1) for line in text.splitlines())
    assert "'force_of_infection' (called 'infection'): the same" in lines['model']
    assert 'removed' not in lines['model'].replace("result 'prevalence_pct' removed", '') # Only the extra result differs
    assert lines['pars'].strip() == 'same' and lines['config'].strip() == 'same'
    changed = ss.diff(hand(), hand(beta=0.8), output=True)
    assert 'model:    same' in changed and '0.5/day -> 0.8/day' in changed
    return text


def test_no_disease():
    sc.heading('Testing what an import of networks only gives')
    S = ss.State(init_n=100) # A disease with no transitions, as an import of a fully custom model may give, runs and stays as it is
    sim = ss.Sim(ss.Disease(S, name='d'), solver='binomial', total_pop=100, stop=10, verbose=0).run()
    assert np.ptp(sim.results.d.S.values) == 0
    sim = ss.Sim(networks=ss.SexualNet(), n_agents=100, unit='week', stop=10, verbose=0) # A sim of networks only keeps its unit in code
    assert "unit='week'" in sim.to_code()
    return sim


if __name__ == '__main__':
    T = sc.timer()
    with tempfile.TemporaryDirectory() as tmp:
        folder = sc.path(tmp)
        test_rates()
        test_emulsion(folder)
        test_to_code(folder)
        test_round_trip(folder)
        test_diff(folder)
        test_no_disease()
    T.toc()
