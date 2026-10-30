"""
Test importing models from other frameworks (ss.import_from()), as the Starsim code it writes
"""

# %% Imports and settings
import tempfile
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
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
    src = ss.converters.Source('rates', format='test')
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
    rates = ss.converters.core.Rates(src)
    out = [rates.classify(sc.obj(src=a, dst=b, rate=r, per=per))[0] for a,b,r,per in flows]
    assert out == ['beta * I', 'beta * I', 'beta * I.n', 'gamma', 'beta * W.n / (W.n + K)', 'xi * I.n', 'mu * (S + I + R).n', 'beta * I', 'beta * I.n / K2'], out
    rules = [rates.classify(sc.obj(src=a, dst=b, rate=r, per=per))[2] for a,b,r,per in flows]
    assert rules[:5] == [2, 2, 3, 4, 5] and rules[7:] == [2, 3]
    return out


def test_emulsion(tmp_path):
    sc.heading('Testing an EMULSION import')
    filename = emulsion_file(tmp_path)
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.import_from(filename, **kw) # The format is told from the file
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
    ode = ss.import_from(filename, format='emulsion', **kw)
    ode.pars.solver = 'ode'
    ode.run()
    runs = ss.import_from(filename, format='emulsion', **kw).run(n_runs=50).results.seirq.R.values[:, -1].mean()
    assert np.isclose(runs, ode.results.seirq.R[-1], rtol=0.1)

    # A probability is per time unit, as EMULSION reads it, not per step
    sc.savetext(tmp_path/'weekly.yaml', emulsion_weekly)
    with pytest.warns(ss.errors.ImportIncomplete):
        weekly = ss.import_from(tmp_path/'weekly.yaml', format='emulsion', **kw)
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
            sir = ss.import_from(folder/f'original/{kind}_SIR.yaml', **kw)
            assert not sir.imported.incomplete and 'n=100;' in repr(sir) # 100 animals, not the default number of agents
            final = sir.run(n_runs=200).results[f'{kind}_SIR'].R.values[:, -1].mean() # With one infected at the start, many runs die out, so the mean needs many
            series = dict(compart='compartment', IBM='ibm', hybrid='hybrid')[kind] + '_R'
            expected = ref[(ref.series == series) & (ref.time == 100)].value.iloc[0]
            rtol = 0.1 if kind == 'compart' else 0.15 # As agents, more runs die out at a daily step (24% of them, against 19% at a tenth of a day), a difference of order dt
            assert np.isclose(final, expected, rtol=rtol), f'{kind}: {final} recovered at day 100 against EMULSION\'s {expected}'
    return sim


def test_code(tmp_path):
    sc.heading('Testing the code of an import: it builds the sim, and stands alone')
    def check(sim):
        """ The code builds the same sim, run on its own """
        space = {}
        exec(sim.imported.code, space)
        new = space['sim']
        new.pars.verbose = 0
        assert ss.debugtools.check_sims_match(sim.run(), new.run()), 'The code should give the same results'
        return sim.imported.code

    # An imported model: the code says how it was read and what was not translated
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.import_from(emulsion_file(tmp_path), **kw)
    code = check(sim)
    assert 'Imported from EMULSION' in code and '# Not translated (ImportIncomplete):' in code and 'force_of_infection = S >> E @ ' in code

    # Every kind of value an importer gives: bounds and a prior, a time series, a seasonal value, values by group, a dwell time as a distribution, a branch, an outcome with a delay and a likelihood, a result, and a stratification
    src = ss.converters.Source('model', format='test')
    src.add_state('S', init=950)
    src.add_state('I', init=ss.by(risk=[30, 20]), label='Infectious')
    src.add_state('R')
    src.add_state('D')
    src.add_par('beta', ss.timeseries({0: 0.3, 20.5: 0.2}, mode='linear'), bounds=[0.1, 1], prior=ss.uniform(0.1, 1))
    src.add_par('season', ss.seasonal(mean=1.0, amplitude=0.2, peak=0.25))
    src.add_par('k', 5.0)
    src.add_flow('S', 'I', 'beta * season * S * I / N', per='total', name='infection')
    src.add_flow('I', ['R', 'D'], dur=ss.lognorm_ex(mean=ss.days(10), std=ss.days(3)), p=['0.99', None], name='recovery')
    src.add_outcome('cases', 'infection', p=0.5, delay=ss.days(3), likelihood=lambda ps: ss.nbinom(n=ps['k']))
    src.results['prev'] = 'I / N'
    src.totals = ['N']
    src.stratify(risk=['lo', 'hi'], split=[0.6, 0.4], adjust={'rel_sus': [1, 2]})
    src.config.update(unit='day', stop=ss.days(60), solver='ode')
    sim = ss.converters.build(src, verbose=False)
    code = check(sim)
    for text in ["beta = ss.par(ss.timeseries({0: 0.3, 20.5: 0.2}, mode='linear'), bounds=[0.1, 1], prior=ss.uniform(low=0.1, high=1))",
                 'infection = S >> I @ (beta * season * I)', 'p=[0.99, None]', 'likelihood=ss.nbinom(n=k, p=0.5)', 'cases = ss.outcome(p_cases*infection, delay=ss.days(3)',
                 "model.stratify(risk=['lo', 'hi'], split=[0.6, 0.4], adjust=dict(rel_sus=[1, 2]))"]:
        assert text in code, f'Expected "{text}" in the code'
    assert sim.results.model.cases.sum() > 0

    # A value the code cannot say is refused, with what to do instead
    src.config.networks = ss.RandomNet()
    with pytest.raises(TypeError, match='ss.converters.Code'):
        ss.converters.build(src, verbose=False)
    src.config.networks = ss.converters.Code('ss.RandomNet()')
    assert 'networks=ss.RandomNet()' in ss.converters.write(src).code

    # Values written as code keep what they are: a table's index (so data is compared at its own times), and a rate's unit
    code_value = ss.converters.core.code_value
    for val in [pd.DataFrame(dict(I=[5.0, 20, 40]), index=[10, 20, 30]), pd.DataFrame(dict(I=[1.0, 2]), index=pd.to_datetime(['2020-01-01', '2020-02-01']))]:
        assert eval(code_value(val), dict(pd=pd)).equals(val)
    rate = eval(code_value(ss.Rate(0.3, ss.weeks(2))), dict(ss=ss))
    assert rate.value == 0.3 and rate.unit == ss.weeks(2)

    # Several sources: a disease's variable is not a name another source defines (here a state)
    host = ss.converters.Source('host', format='test')
    for state in 'SIR':
        host.add_state(state, init=990 if state == 'S' else 10*(state == 'I'))
    host.add_par('beta', 0.3)
    host.add_flow('S', 'I', 'beta*S*I/(S+I+R)', per='total', name='infection')
    vector = ss.converters.Source('vector', format='test')
    vector.add_state('V', init=100)
    vector.add_state('host', init=0)
    vector.add_flow('V', 'host', 0.05, name='bite')
    sim = ss.converters.build([host, vector], verbose=False, stop=10, unit='day').run()
    assert sim.results.host.I[-1] > 10 and sim.results.vector.host[-1] > 0
    return code


def test_save_load(tmp_path):
    sc.heading('Testing that every corpus model can be saved and loaded after a run')
    def results(sim):
        return {k:np.asarray(v.values, dtype=float) for k,v in sim.results.flatten().items() if isinstance(v, ss.Result)}
    for name,make in corpus.models.items():
        sim = make().run()
        a = results(sim)
        sim.save(tmp_path/f'{name}.sim') # After a run, so that every module shrinks
        loaded = results(ss.load(tmp_path/f'{name}.sim'))
        assert all(np.allclose(a[k], loaded[k]) for k in a), f'{name}: the loaded sim has different results'
    return


def test_hand(tmp_path):
    sc.heading('Testing that an import and a hand translation give the same results')
    with pytest.warns(ss.errors.ImportIncomplete):
        sim = ss.import_from(emulsion_file(tmp_path), **kw)
    def hand(beta=0.5):
        S, E, I, Q, R = ss.states('S E I Q R')
        seirq = ss.Disease(
            force_of_infection = S >> E @ ss.perday(beta) * I,
            onset      = E >> I @ -np.log(1 - 0.2),
            recovery   = I >> R @ ss.perday(0.1),
            detection  = I >> Q @ (0.05 * (Q.n < 20)),
            release    = Q >> R @ ss.constant(ss.days(10)),
            death      = I >> ss.null @ ss.perday(0.001),
            init_n     = dict(I=10),
            name       = 'seirq',
        )
        return ss.Sim(seirq, total_pop=1000, stop=ss.days(100), solver='binomial', substeps=1, **kw)
    a, b = [s.new(solver='ode').run().results.seirq for s in [sim, hand()]] # As an ODE, since the import's random numbers are drawn for its other parts too
    for state in 'SEIQR':
        assert np.allclose(a[state], b[state]), state
    assert not np.allclose(hand(beta=0.8).new(solver='ode').run().results.seirq.I, b.I)
    return a


def test_no_disease():
    sc.heading('Testing what an import of networks only gives')
    S = ss.State(init_n=100) # A disease with no transitions, as an import of a fully custom model may give, runs and stays as it is
    sim = ss.Sim(ss.Disease(S, name='d'), solver='binomial', total_pop=100, stop=10, verbose=0).run()
    assert np.ptp(sim.results.d.S.values) == 0
    return sim


if __name__ == '__main__':
    T = sc.timer()
    with tempfile.TemporaryDirectory() as tmp:
        folder = sc.path(tmp)
        test_rates()
        test_emulsion(folder)
        test_code(folder)
        test_save_load(folder)
        test_hand(folder)
        test_no_disease()
    T.toc()
