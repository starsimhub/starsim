"""
Test importing models from other frameworks (ss.import_from()): the rules for reading rates, the Starsim code an import writes, and one import of each format, of an SIR written as that framework writes it, against the same SIR written by hand

The tests against the frameworks' own examples are in the lingua_franca repo's `tests/conversion/`; the exporters are tested in test_io_export.py.
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

kw = dict(verbose=0)


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


def test_code():
    sc.heading('Testing the code of an import: it builds the sim, and stands alone')

    # Every kind of value an importer gives: bounds and a prior, a time series, a seasonal value, values by group, a dwell time as a distribution, a branch, an outcome with a delay and a likelihood, a result, and a stratification
    src = ss.converters.Source('model', format='test')
    src.add_state('S', init=950)
    src.add_state('I', init=ss.by(risk=[30, 20]), label='Infectious')
    src.add_state('R')
    src.add_state('D')
    src.add_par('beta', ss.timeseries({0: 0.3, 20.5: 0.2}, interpolation='linear'), bounds=[0.1, 1], prior=ss.uniform(0.1, 1))
    src.add_par('season', ss.seasonal(mean=1.0, amplitude=0.2, peak=0.25))
    src.add_par('k', 5.0)
    src.add_flow('S', 'I', 'beta * season * S * I / N', per='total', name='infection')
    src.add_flow('I', ['R', 'D'], dur=ss.lognormal(mean=ss.days(10), std=ss.days(3)), p=['0.99', None], name='recovery')
    src.add_outcome('cases', 'infection', p=0.5, delay=ss.days(3), likelihood=lambda ps: ss.nbinom(n=ps['k']))
    src.results['prev'] = 'I / N'
    src.totals = ['N']
    src.stratify(risk={'lo': 0.6, 'hi': 0.4}, adjust={'rel_sus': [1, 2]})
    src.config.update(unit='day', stop=ss.days(60), solver='ode')
    sim = ss.converters.build(src, verbose=False)
    code = sim.imported.code
    assert 'infection = S >> I @ (beta * season * I)' in code and '{R: 0.99, D: None}' in code
    space = {}
    exec(code, space)
    new = space['sim']
    new.pars.verbose = 0
    assert ss.debugtools.check_sims_match(sim.run(), new.run()), 'The code should give the same results'
    assert sim.results.model.cases.sum() > 0


#%% An SIR (beta 0.3, gamma 0.1, 990 susceptible and 10 infectious, 50 days) in each format

emulsion = """
model_name: sir
time_info: {time_unit: days, delta_t: 1, total_duration: 50}
levels: {population: {aggregation_type: compartment}}
processes: {population: [infection]}
grouping: {population: {infection: {machine_name: health_state, key_variables: [health_state]}}}
state_machines:
  health_state:
    states: [S: {}, I: {}, R: {}]
    transitions:
      - {from: S, to: I, rate: 'beta * total_I / total_population'}
      - {from: I, to: R, rate: gamma}
parameters: {beta: {value: 0.3}, gamma: {value: 0.1}}
initial_conditions: {population: [{population: [{total: 1000}, {vars: [S], amount: 990}, {vars: [I], amount: 10}]}]}
"""

odin = """
deriv(S) <- -beta * S * I / N
deriv(I) <- beta * S * I / N - gamma * I
deriv(R) <- gamma * I
N <- S + I + R
initial(S) <- 990
initial(I) <- 10
initial(R) <- 0
beta <- user(0.3)
gamma <- user(0.1)
"""

def species(s, n):
    return f'<species id="{s}" compartment="c" initialAmount="{n}" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'

def reaction(name, src, dst, math):
    ref = lambda s: f'<speciesReference species="{s}" stoichiometry="1" constant="true"/>'
    law = f'<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{math}</math></kineticLaw>'
    return f'<reaction id="{name}" reversible="false"><listOfReactants>{ref(src)}</listOfReactants><listOfProducts>{ref(dst)}</listOfProducts>{law}</reaction>'

sbml = f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1"><model id="sir">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>{species('S', 990)}{species('I', 10)}{species('R', 0)}</listOfSpecies>
  <listOfParameters><parameter id="beta" value="0.3" constant="true"/><parameter id="gamma" value="0.1" constant="true"/></listOfParameters>
  <listOfReactions>
    {reaction('infection', 'S', 'I', '<apply><divide/><apply><times/><ci>beta</ci><ci>S</ci><ci>I</ci></apply><apply><plus/><ci>S</ci><ci>I</ci><ci>R</ci></apply></apply>')}
    {reaction('recovery', 'I', 'R', '<apply><times/><ci>gamma</ci><ci>I</ci></apply>')}
  </listOfReactions>
</model></sbml>
"""

siminf = """
library(SimInf)
transitions <- c("S -> beta*S*I/(S+I+R) -> I", "I -> gamma*I -> R")
u0 <- data.frame(S = 990, I = 10, R = 0)
model <- mparse(transitions = transitions, compartments = c("S", "I", "R"), gdata = c(beta = 0.3, gamma = 0.1), u0 = u0, tspan = 1:50)
"""

flepimop = dict(name='sir', start_date='2020-01-01', end_date='2020-02-20', compartments=dict(infection_stage=['S', 'I', 'R']),
    initial_conditions=dict(method='SetInitialConditions', initial_conditions_file='ic.csv'),
    seir=dict(integration=dict(method='rk4', dt=1), parameters=dict(beta=dict(value=0.3), gamma=dict(value=0.1)), transitions=[
        dict(source=['S'], destination=['I'], rate=['beta'], proportional_to=['source', ['I']], proportion_exponent=['1', '1']),
        dict(source=['I'], destination=['R'], rate=['gamma'], proportional_to=['source'], proportion_exponent=['1'])]))

op = lambda name, left, right: {'bin_op': {'op': name, 'left': left, 'right': right}}
camdl = {'ir_version': '0.40', 'model': { # camdl's IR, as camdlc writes it, trimmed of what the importer does not read
    'name': 'sir', 'time_unit': 'days',
    'compartments': [{'name': 'S'}, {'name': 'I'}, {'name': 'R'}],
    'transitions': [
        {'name': 'infection', 'stoichiometry': [['S', -1], ['I', 1]], 'rate': op('div', op('mul', op('mul', {'param': 'beta'}, {'pop': 'S'}), {'pop': 'I'}), {'binding_ref': 'N'})},
        {'name': 'recovery', 'stoichiometry': [['I', -1], ['R', 1]], 'rate': op('mul', {'param': 'gamma'}, {'pop': 'I'})},
    ],
    'parameters': [{'name': 'beta', 'value': {'mode': 'required'}}, {'name': 'gamma', 'value': {'mode': 'required'}}], # Given with pars=, as camdl is given them with --param
    'initial_conditions': {'S': {'deterministic': {'const': 990.0}}, 'I': {'deterministic': {'const': 10.0}}},
    'simulation': {'t_start': 0.0, 't_end': 50.0, 'dt': None},
    'bindings': [{'name': 'N', 'expr': {'pop_sum': ['S', 'I', 'R']}}],
}}

epimodel = dict(format='epimodel', functions=[], runs=[dict(kind='dcm', type='SIR', networks=[], modules={}, # EpiModel's built-in SIR in dcm(), as dump_epimodel.R writes it
    param={'inf.prob': 0.3, 'act.rate': 1, 'rec.rate': 0.1}, init={'s.num': 990, 'i.num': 10, 'r.num': 0}, control=dict(nsteps=51, dt=1, odemethod='rk4'))])

epipack = dict(format='epipack', name='sir', compartments=['S', 'I', 'R'], population=1000.0, dynamic=False, parameters={}, incomplete=[],
    events=[dict(kind='quadratic', rate='0.3*S*I', changes=dict(S=-1, I=1)), dict(kind='linear', rate='0.1*I', changes=dict(I=-1, R=1))], # epipack divides a quadratic rate by the population
    initial=dict(S=990.0, I=10.0, R=0.0), run=dict(tmax=50.0, solver='ode'))

epymorph = dict(format='epymorph', name='sir', compartments=['S', 'I', 'R'], places=['a'], start='2020-01-01', days=50, movement=[[0.0]], incomplete=[],
    edges=[dict(source='S', target='I', rate='I*S*beta/N'), dict(source='I', target='R', rate='I*gamma')],
    parameters=dict(beta=dict(kind='number', value=0.3), gamma=dict(kind='number', value=0.1)), initial=[[990.0, 10.0, 0.0]])

epydemix = dict(format='epydemix', name='sir', compartments=['S', 'I', 'R'], overrides={}, interventions=[],
    transitions=[dict(source='S', target='I', kind='mediated', params=['beta', 'I']), dict(source='I', target='R', kind='spontaneous', params='gamma')],
    parameters=dict(beta=dict(kind='number', value=0.3), gamma=dict(kind='number', value=0.1)),
    population=dict(name='Toy', groups=['all'], Nk=[1000.0], contacts=dict(all=[[1.0]])),
    run=dict(start_date='2020-01-01', end_date='2020-02-20', dt=1.0, initial=dict(S=[990.0], I=[10.0], R=[0.0]), parameters={}))

memilio = dict(format='memilio', model='osir', groups=1, states=['Susceptible', 'Infected', 'Recovered'], other={},
    parameters=dict(TransmissionProbabilityOnContact=dict(kind='age', value=[0.3]), TimeInfected=dict(kind='age', value=[10.0])),
    populations=dict(Susceptible=[990.0], Infected=[10.0], Recovered=[0.0]),
    contacts=dict(matrices=[dict(baseline=[[1.0]], minimum=[[0.0]], dampings=[])], samples=[dict(time=0.0, matrix=[[1.0]])]),
    run=dict(t0=0.0, tmax=50.0, dt=0.1, function='simulate'))

summer2 = dict(format='summer2', compartments=['S', 'I', 'R'], infectious=['I'], initial_population=dict(S=990, I=10, R=0), times=[0, 50], timestep=1.0,
    initial=[dict(name='S', strata={}, value=990.0), dict(name='I', strata={}, value=10.0), dict(name='R', strata={}, value=0.0)],
    flows=[dict(name='infection', kind='InfectionFrequencyFlow', sources=['S'], dest='I', param=0.3), dict(name='recovery', kind='TransitionFlow', sources=['I'], dest='R', param=0.1)],
    stratifications=[], outputs={})

# The format, the model (a file's text, or what the framework's script writes), the arguments to import it, and the solver it is read as (as the framework runs it)
formats = [
    ('emulsion', 'sir.yaml', emulsion, {}, ss.solvers.binomial),
    ('odin', 'sir.R', odin, dict(stop=50), ss.solvers.ode),
    ('sbml', 'sir.xml', sbml, dict(stop=50), ss.solvers.ode),
    ('siminf', 'sir.R', siminf, {}, ss.solvers.gillespie),
    ('flepimop', 'sir.yml', flepimop, {}, ss.solvers.ode),
    ('camdl', None, camdl, dict(pars=dict(beta=0.3, gamma=0.1)), ss.solvers.ode),
    ('epimodel', None, epimodel, {}, ss.solvers.ode),
    ('epipack', None, epipack, {}, ss.solvers.ode),
    ('epymorph', None, epymorph, {}, ss.solvers.tau),
    ('epydemix', None, epydemix, {}, ss.solvers.binomial),
    ('memilio', None, memilio, {}, ss.solvers.ode),
    ('summer2', None, summer2, {}, ss.solvers.ode),
]


@pytest.mark.parametrize('fmt, filename, model, kwargs, solver', formats, ids=[f[0] for f in formats])
def test_import(tmp_path, fmt, filename, model, kwargs, solver):
    sc.heading(f'Testing importing an SIR from {fmt}')
    if fmt == 'flepimop': # A configuration, and its initial conditions beside it
        sc.saveyaml(tmp_path/filename, model)
        pd.DataFrame(dict(mc_name=['S', 'I'], amount=[990, 10])).to_csv(tmp_path/'ic.csv', index=False)
        model = tmp_path/filename
    elif filename:
        sc.savetext(tmp_path/filename, model)
        model = tmp_path/filename
    sim = ss.import_from(model, format=fmt, **kwargs, **kw)
    assert not sim.imported.incomplete
    assert isinstance(sim.init().solver, solver)
    res = sim.new(solver='ode').run().results[sim.diseases[0].name]
    I = res.Infected if fmt == 'memilio' else res.I

    # By hand
    S, I_, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I_ @ ss.perday(0.3)*I_, recovery=I_ >> R @ ss.perday(0.1), init_n=dict(I=10), name='sir')
    ref = ss.Sim(sir, total_pop=1000, unit='day', stop=50, **kw).run().results.sir
    assert np.allclose(I.values[:51], ref.I.values, rtol=1e-9), 'The import should be the SIR written by hand'


def test_atomica(tmp_path):
    sc.heading('Testing importing an Atomica framework and databook')
    sheets = dict( # An SIR with a probability, a duration, a rate, a number, and a function
        About = [['Name', 'Description'], ['tiny', 'A tiny SIR']],
        Compartments = [['Code Name', 'Display Name', 'Is Source', 'Is Sink', 'Is Junction', 'Databook Page'],
                        ['sus', 'Susceptible', 'n', 'n', 'n', 'state_variables'], ['inf', 'Infected', 'n', 'n', 'n', None],
                        ['rec', 'Recovered', 'n', 'n', 'n', None], ['dead', 'Dead', 'n', 'y', 'n', None]],
        Characteristics = [['Code Name', 'Display Name', 'Databook Page', 'Components', 'Denominator'],
                           ['ch_all', 'Total', 'state_variables', 'sus, inf, rec', None], ['ch_prev', 'Prevalence', 'state_variables', 'inf', 'ch_all']],
        Parameters = [['Code Name', 'Display Name', 'Format', 'Databook Page', 'Default Value', 'Function'],
                      ['beta', 'Transmission probability', 'Probability', 'parameters', None, None],
                      ['foi', 'Force of infection', 'Rate', None, None, 'beta * ch_prev'],
                      ['dur', 'Duration of infection', 'Duration', 'parameters', None, None],
                      ['mort', 'Death rate', 'Rate', 'parameters', 0.02, None],
                      ['vacc', 'Number vaccinated', 'Number', 'parameters', None, None]],
        Transitions = [[None, 'sus', 'inf', 'rec', 'dead'], ['sus', None, 'foi', 'vacc', None], ['inf', None, None, 'dur', 'mort'], ['rec', None, None, None, None], ['dead', None, None, None, None]],
    )
    fw, db = tmp_path/'framework.xlsx', tmp_path/'databook.xlsx'
    with pd.ExcelWriter(fw) as w:
        for name,rows in sheets.items():
            pd.DataFrame(rows).to_excel(w, sheet_name=name, header=False, index=False)
    years = list(range(2000, 2005))
    block = lambda label, units, const, vals: [[label, 'Provenance', 'Units', 'Uncertainty', 'Constant', None, *years], ['adults', None, units, None, const, 'OR', *vals], [None]*(6 + len(years))]
    state = block('Susceptible', 'Number', None, [900]*5) + block('Total', 'Number', None, [1000]*5) + block('Prevalence', 'Fraction', None, [0.05]*5)
    par = block('Transmission probability', 'Probability', 0.6, [None]*5) + block('Duration of infection', 'Duration', 5, [None]*5) + block('Number vaccinated', 'Number', None, [10, 10, 20, 20, 20])
    with pd.ExcelWriter(db) as w:
        pd.DataFrame([['Abbreviation', 'Full Name'], ['adults', 'Adults']]).to_excel(w, sheet_name='Population Definitions', header=False, index=False)
        pd.DataFrame(state).to_excel(w, sheet_name='State Variables', header=False, index=False)
        pd.DataFrame(par).to_excel(w, sheet_name='Parameters', header=False, index=False)

    sim = ss.import_from(fw, format='atomica', databook=db, dt=1, **kw)
    res = sim.run().results.tiny
    assert np.allclose([res.sus[0], res.inf[0], res.rec[0]], [900, 50, 50]), 'The initial numbers should be solved from the characteristics'
    assert sim.t.yearvec[-1] == 2009, 'Atomica runs to five years after the data, if there are no births'

    # Atomica's step by hand: rates and probabilities move value × dt, durations dt/value, numbers value × dt people
    x = np.array([900.0, 50, 50, 0])
    vacc = np.interp(np.arange(2000, 2010), years, [10, 10, 20, 20, 20])
    for i in range(9):
        s, inf, r, d = x
        new = 0.6*inf/(s + inf + r)*s
        sr, ir, dead = min(vacc[i], s), inf/5, 0.02*inf
        x = np.array([s - new - sr, inf + new - ir - dead, r + sr + ir, d + dead])
    assert np.allclose([res.sus[-1], res.inf[-1], res.rec[-1], res.dead[-1]], x), 'Should match Atomica\'s Euler step'
