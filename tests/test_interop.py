"""
Test importing models from other frameworks (ss.load(..., format=...)), writing a sim as code, and comparing two models
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

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
    for p,v in dict(beta=0.3, gamma=0.1, xi=10, K=1e4, mu=0.01).items():
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
    ]
    rates = ss.interop.core.Rates(src)
    out = [rates.classify(sc.objdict(src=a, dst=b, rate=r, per=per))[0] for a,b,r,per in flows]
    assert out == ['beta * I', 'beta * I', 'beta * I.n', 'gamma', 'beta * W.n / (W.n + K)', 'xi * I.n', 'mu * (S + I + R).n'], out
    rules = [rates.classify(sc.objdict(src=a, dst=b, rate=r, per=per))[2] for a,b,r,per in flows]
    assert rules[:5] == [2, 2, 3, 4, 5]
    return out


def test_emulsion(tmp_path):
    sc.heading('Testing an EMULSION import')
    filename = emulsion_file(tmp_path)
    with pytest.warns(RuntimeWarning, match='W1601'):
        sim = ss.load(filename, **kw) # The format is told from the file
    imp = sim.imported
    assert imp.format == 'EMULSION'
    assert any('when' in name for name,why in imp.incomplete), 'The calendar condition should be listed as not translated'
    sim.init()
    dis = sim.diseases.seirq
    assert set(dis.graph.states.keys()) == {'S', 'E', 'I', 'Q', 'R'} # D is removed: an outflow
    assert sim.pars.solver == 'binomial'
    assert 'force_of_infection' in dis.graph.transitions # A rate given by a named expression names its transition
    assert dis.graph.transitions.force_of_infection.contact
    sim.run()
    res = sim.results.seirq
    full = res.n_Q.values[:-1] >= 20 # The capacity is a condition on the count at the start of a step
    assert full.any() and (res.detection.values[1:][full] == 0).all()
    assert np.isclose(res.prevalence_pct[10], 100*(res.n_E[10] + res.n_I[10])/(res.n_S[10] + res.n_E[10] + res.n_I[10] + res.n_Q[10] + res.n_R[10]))

    # The same model as an ODE agrees with the stochastic one on average
    ode = ss.load(filename, format='emulsion', **kw)
    ode.pars.solver = 'ode'
    ode.run()
    runs = ss.load(filename, format='emulsion', **kw).run(n_runs=50).results.seirq.n_R.values[:, -1].mean()
    assert np.isclose(runs, ode.results.seirq.n_R[-1], rtol=0.1)
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
                     cases=ss.outcome('infection', delay=ss.days(3)), prev=I/(S + I + R), init=dict(I=10), name='sir')
    sir.stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2])
    same(ss.Sim(sir, total_pop=1000, stop=ss.days(50), unit='day', **kw))

    # An imported model: the code says how it was read and what was not translated
    with pytest.warns(RuntimeWarning, match='W1601'):
        sim = ss.load(emulsion_file(tmp_path), **kw)
    code = same(sim)
    assert 'Imported from EMULSION' in code and '# Not translated (W1601 ImportIncomplete):' in code
    return code


def test_diff(tmp_path):
    sc.heading('Testing ss.diff(): an import and a hand translation are the same model')
    with pytest.warns(RuntimeWarning, match='W1601'):
        sim = ss.load(emulsion_file(tmp_path), **kw)
    def hand(beta=0.5):
        S, E, I, Q, R = ss.states('S E I Q R')
        seirq = ss.Disease(
            infection  = S >> E @ ss.perday(beta) * I,
            onset      = E >> I @ -np.log(1 - 0.2),
            recovery   = I >> R @ ss.perday(0.1),
            detection  = I >> Q @ (0.05 * (Q.n < 20)),
            release    = Q >> R @ ss.constant(ss.days(10)),
            death      = I >> ss.null @ ss.perday(0.001),
            init       = dict(I=10),
            name       = 'seirq',
        )
        return ss.Sim(seirq, total_pop=1000, stop=ss.days(100), solver='binomial', **kw)
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
    S = ss.State(init=100) # A disease with no transitions, as an import of a fully custom model may give, runs and stays as it is
    sim = ss.Sim(ss.Disease(S, name='d'), solver='binomial', stop=10, verbose=0).run()
    assert np.ptp(sim.results.d.n_S.values) == 0
    sim = ss.Sim(networks=ss.PartnerNet(), n_agents=100, unit='week', stop=10, verbose=0) # A sim of networks only keeps its unit in code
    assert "unit='week'" in sim.to_code()
    return sim


if __name__ == '__main__':
    T = sc.timer()
    folder = sc.path(sc.thisdir()) / 'temp'
    folder.mkdir(exist_ok=True)
    test_rates()
    test_emulsion(folder)
    test_to_code(folder)
    test_diff(folder)
    test_no_disease()
    T.toc()
