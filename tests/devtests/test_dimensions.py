"""
Test dimensions that are more than a property of people: states with dimensions of their own, strains, and places
"""

# %% Imports and settings
import sys
import math
import subprocess
import tempfile
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.integrate as sp_integrate
import scipy.stats as stats
from scipy.special import expit
import sciris as sc
import starsim as ss
import corpus

kw = dict(verbose=0)
solvers = ['ode', 'binomial', 'abm']


def at_start(sim):
    """ The results with the number in each state before the first step at index 0, for a compartmental sim (results count the number after each step) """
    sim.init()
    for mod in sim.declared:
        mod.engine.start_step()
        mod.engine.record_stocks()
        mod.engine.update_results()
    return sim.results


def run(mod, solver, **kwargs):
    """ Run a module with one solver, as 10,000 agents if it is agent-based """
    pars = dict(n_agents=10_000) if solver == 'abm' else {}
    return ss.Sim(mod, solver=solver, **pars, **kw, **kwargs).run()


def make_malaria():
    """ Ross-Macdonald: humans and mosquitoes as states of one disease """
    a = 0.3 # Bites per mosquito per day
    b = 0.5 # Probability an infectious bite infects a human
    c = 0.5 # Probability a bite on an infectious human infects the mosquito
    Sh, Eh, Ih, Rh = ss.states(4)
    Sv, Ev, Iv     = ss.states(3, people=False)
    Nh             = (Sh + Eh + Ih + Rh).n
    malaria = ss.Disease(
        infection  = Sh >> Eh @ a * b * Iv.n / Nh,
        latency    = Eh >> Ih @ ss.days(12),
        recovery   = Ih >> Rh @ ss.days(100),
        waning     = Rh >> Sh @ ss.days(365),
        emergence  = ss.null >> Sv @ ss.perday(5_000),
        biting     = Sv >> Ev @ a * c * Ih.n / Nh,
        incubation = Ev >> Iv @ ss.days(10),
        mortality  = Sv + Ev + Iv >> ss.null @ ss.days(10),
        init       = dict(Ih=0.01, Sv=50_000),
        first      = 'Sh',
        name       = 'malaria',
    )
    return malaria


def make_tb():
    """ TB with drug resistance as a strain, acquired during treatment """
    susceptible, latent, active, on_treatment = ss.states(4)
    tb = ss.Disease(
        infection  = susceptible >> latent @ ss.peryear(8) * active,
        activation = latent >> active @ ss.peryear(0.05),
        diagnosis  = active >> on_treatment @ ss.peryear(0.7),
        cure       = on_treatment >> susceptible @ ss.months(6),
        failure    = on_treatment >> active @ ss.peryear(0.2),
        init       = dict(active=0.01),
        name       = 'tb',
        strains    = ss.Strains(['DS', 'MDR'], name='resistance', split=[1.0, 0.0], infection=[1.0, 0.7], cure=[1.0, 0.3]),
    )
    tb.stratify(acquired=ss.transfer(resistance=('DS', 'MDR'), rate=ss.peryear(0.02), states=['on_treatment']))
    return tb


def make_towns(**kwargs):
    """ An SIR model in three towns, with nobody infected """
    towns = ss.geo.geography(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    S, I, R = ss.states('S I R')
    sir = ss.Disease(
        infection = ss.Transition(S, I, rate=0.3 * I),
        recovery  = ss.Transition(I, R, dur=10),
        init_prev = 0, # Seeded by seed_north()
        name = 'sir',
    )
    return sir.stratify(patch=towns, **kwargs)


def seed_north(sim):
    """ Infect 50 people in the first town, on the first step """
    if sim.ti == 1:
        sir = sim.diseases.sir
        if sim.solver == 'abm':
            uids = (sim.people.patch == 0).uids[:5] # Each agent stands for 10 people
            sir.S[uids] = False
            sir.I[uids] = True
        else:
            engine = sir.engine
            engine.y[engine.state_comps['S'][0], 0] -= 50
            engine.y[engine.state_comps['I'][0], 0] += 50
    return


# %% Define the tests

@sc.timer()
def test_states():
    sc.heading('Testing dimensions that apply to some states only')
    sims = sc.obj()
    for solver in solvers:
        malaria = make_malaria()
        states = malaria.states
        malaria.stratify(species=['gambiae', 'funestus'], split=[0.75, 0.25], states=[states.Sv, states.Ev, states.Iv], adjust={'mortality': [1.0, 2.0]})
        sims[solver] = run(malaria, solver, unit='day', stop=365, total_pop=20_000, stratify=dict(age=[0, 5, 15]))
    res = sims.ode.results.malaria
    assert [s.name for s in malaria.graph.count_states] == ['Sv', 'Ev', 'Iv'] # Nobody moves between humans and mosquitoes, so the mosquitoes are counts
    assert np.allclose(res.Sv.by('species').values[0], [37_500, 12_500], rtol=0.1) # Split by species (after the first day, in which funestus dies twice as fast) ...
    assert np.allclose(res.emergence.by('species').values[5], [3750, 1250])
    assert np.allclose(sum(res[f'{s}h'] for s in 'SEIR'), 20_000) # ... and left out of the population
    n_Iv = res.Iv.by('species').values[-1]
    assert n_Iv[1] < n_Iv[0]/6, 'Expected fewer of the species that is rarer and dies faster'
    assert res.Ih.by('age').values.shape[1] == 3
    with pytest.raises(KeyError): # A state has only the dimensions it was stratified by
        res.Iv.by('age')
    with pytest.raises(KeyError):
        res.Ih.by('species')
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.malaria
        assert np.allclose(other.Iv.by('species').values[-1], n_Iv, rtol=0.2), f'Expected {solver} to match the ODE'
        assert np.isclose(other.Ih[-1], res.Ih[-1], rtol=0.1)

    # A dimension of the people in some states: they take a level when they enter one, and lose it when they leave
    def make():
        S, I = ss.states(2)
        hiv = ss.Disease(infection=S >> I @ 0.2 * I, recovery=I >> S @ 0.05, init_prev=0.05, name='hiv')
        hiv.stratify(stage=['early', 'mid', 'late'], split=[1, 0, 0], states=[I], adjust={'recovery': [1.0, 0.5, 0.25]},
                     decline=ss.transfer(stage=('early', 'mid'), rate=0.05), decline2=ss.transfer(stage=('mid', 'late'), rate=0.05))
        return hiv
    staged = sc.obj({solver:run(make(), solver, unit='day', stop=300, total_pop=20_000) for solver in solvers})
    res = staged.ode.results.hiv
    assert np.allclose(res.infection.by('stage').values.sum(axis=0)[1:], 0) # Everyone starts in the first stage
    assert getattr(res.S, 'strata', None) is None # Susceptible people have no stage
    for solver in ['binomial', 'abm']:
        other = staged[solver].results.hiv
        assert np.allclose(other.I.by('stage').values[-1], res.I.by('stage').values[-1], rtol=0.15), f'Expected {solver} to match the ODE'

    with pytest.raises(KeyError, match='states= names a state "Evv"'):
        make_malaria().stratify(species=['gambiae', 'funestus'], states=['Sv', 'Evv'])

    # A dimension of some states only, entered with a split that depends on age, as summer2's clinical strata
    def clinical(adjust=None):
        S, E, I, R = ss.states('S E I R')
        seir = ss.Disease(infection=ss.Transition(S, E, rate=0.4 * I), onset=ss.Transition(E, I, dur=3), recovery=ss.Transition(I, R, dur=7), init=dict(I=0.01), name='seir')
        seir.stratify(age=[0, 20, 60], split=[0.3, 0.5, 0.2])
        seir.stratify(clinical=['mild', 'severe'], states=['I'], split=ss.by([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]], dims=['age', 'clinical']), adjust=adjust)
        return seir
    for solver,agents in [('ode', {}), ('binomial', {}), ('abm', dict(n_agents=50_000))]:
        res = ss.Sim(clinical(), solver=solver, stop=100, **agents, **kw).run().results.seir
        severe = [res.I.select(age=age, clinical='severe').values.sum()/res.I.select(age=age).values.sum() for age in ['0-19', '20-59', '60+']]
        assert np.allclose(severe, [0.1, 0.3, 0.8], atol=0.03), f'Expected the share severe to depend on age with {solver}'
        assert np.allclose([res.S.select(age=age).values[0] for age in ['0-19', '20-59', '60+']], np.array([0.3, 0.5, 0.2])*res.S.values[0], rtol=0.02)
    # A flow into such a dimension is recorded at the level people arrive at
    for solver,agents in [('ode', {}), ('abm', dict(n_agents=20_000))]:
        onset = ss.Sim(clinical(), solver=solver, stop=100, **agents, **kw).run().results.seir.onset.by('clinical').values.sum(axis=0)
        assert np.isclose(onset[1]/onset.sum(), 0.3*0.1 + 0.5*0.3 + 0.2*0.8, atol=0.02)

    # A starting count by level of such a dimension is that many people, and the first state has everyone else
    seir = clinical()
    seir.graph.states.I.init, seir.graph.states.I.init_n = None, ss.by(clinical=[10, 0])
    res = at_start(ss.Sim(seir, total_pop=1000, stop=2, **kw)).seir
    assert np.isclose(res.I.select(clinical='mild').values[0], 10) and np.isclose(res.I.values[0], 10)
    assert np.isclose(res.S.values[0], 990)

    with pytest.raises(ValueError, match='have no level of "clinical"'): # People in E have no clinical level, so an adjustment of their onset cannot apply
        ss.Sim(clinical(adjust={'onset': [1, 2]}), **kw).init()

    # Infectiousness by clinical level: none at all stops transmission, and the solvers agree on the rest
    assert ss.Sim(clinical(adjust={'rel_trans': [0, 0]}), stop=100, **kw).run().results.seir.R.values[-1] < 101
    sizes = [ss.Sim(clinical(adjust={'rel_trans': [0.5, 2]}), solver=solver, stop=150, total_pop=20_000, **agents, **kw).run().results.seir.R.values[-1]
             for solver,agents in [('ode', {}), ('abm', dict(n_agents=20_000))]]
    assert np.isclose(*sizes, rtol=0.05)
    with pytest.raises(ValueError, match=r'the people it is for \(S\) have no level of "clinical"'): # Susceptible people have no clinical level
        ss.Sim(clinical(adjust={'rel_sus': [1, 2]}), **kw).init()
    with pytest.raises(KeyError, match='so there is no level to take it from'): # Nor is there a level to take a value by clinical level from (before, it took the first)
        S, E, I, R = ss.states('S E I R')
        seir = ss.Disease(infection=ss.Transition(S, E, rate=0.4 * I), onset=ss.Transition(E, I, rate=ss.by(clinical=[0.2, 0.4])), recovery=ss.Transition(I, R, dur=7), init=dict(I=0.01), name='seir')
        seir.stratify(clinical=['mild', 'severe'], states=['I'], split=[0.8, 0.2])
        ss.Sim(seir, **kw).init()
    return sims


@sc.timer()
def test_strains():
    sc.heading('Testing a strain: a new infection takes the strain of its source')
    sims = sc.obj()
    for solver in solvers:
        sims[solver] = run(make_tb(), solver, start=2000, stop=2050, dt=ss.years(0.1), total_pop=100_000)
    tb = sims.ode.diseases.tb
    res = tb.results
    assert tb.graph.strats[0].states == ['latent', 'active', 'on_treatment'] # A susceptible person has no strain
    assert getattr(res.susceptible, 'strata', None) is None
    by_strain = res.infection.cum.by('resistance').values[-1]
    assert res.active.by('resistance').values[0, 1] < 0.01 # Nobody starts with a resistant strain (so after the first step there is almost nobody)
    assert res.acquired.cum[-1] > 0 and by_strain[1] > 0 # It is acquired during treatment, and then transmitted
    assert by_strain[1] < 0.1*by_strain[0]
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.tb
        assert np.isclose(other.infection.cum[-1], res.infection.cum[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.isclose(other.active.by('resistance').values[-1, 0], res.active.by('resistance').values[-1, 0], rtol=0.1)
        assert 0 < other.infection.cum.by('resistance').values[-1, 1] < 0.1*by_strain[0]

    # Without acquired resistance, the resistant strain never appears; and a fitter strain takes over
    def make(adjust):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.01, name='sis',
                          strains=ss.Strains(['wild', 'variant'], split=[0.9, 0.1], infection=adjust))
    for solver in solvers:
        res = run(make([1.0, 1.5]), solver, stop=300, total_pop=50_000).results.sis
        n_I = res.I.by('strain').values
        assert n_I[0, 0] > 5*n_I[0, 1] and n_I[-1, 1] > 5*n_I[-1, 0], f'Expected the more transmissible strain to take over with {solver}'
    # Cross-immunity between strains: complete protection is the base model; none lets people in R be infected again with the other strain
    def flu(m=None, combine='max'):
        S, I, R = ss.states('S I R')
        strains = ss.Strains(['A', 'B'])
        if m is not None:
            strains.immunity = ss.immunity(due_to=R, rel_sus=m, combine=combine)
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=7), init_prev=0.01, name='flu', strains=strains)
    total = lambda *args, kwargs={}, **kws: ss.Sim(flu(*args, **kws), unit='day', stop=200, **kwargs, **kw).run().results.flu.infection.sum()
    once = total()
    assert np.isclose(total(np.ones((2, 2))), once)
    again = total(np.eye(2), kwargs=dict(solver='ode')) # Due to R, by the strain people carry there (the last), which a compartment keeps
    abm = total(np.eye(2), kwargs=dict(solver='abm', n_agents=10_000))
    assert again > 1.5*once and np.isclose(again, abm, rtol=0.1) # So A, B, A again, ...
    for combine in ['sum', 'multiply', lambda p: p.max(0)]: # With protection from one strain only, the rules agree
        assert total(np.eye(2), combine=combine, kwargs=dict(solver='abm', n_agents=10_000)) == abm
    with pytest.raises(ss.CapabilityError): # A function has no compartmental form (nor a gradient)
        total(np.ones((2, 2)), combine=lambda p: p.max(0), kwargs=dict(solver='ode'))
    with pytest.raises(ValueError, match='but there are 2 strains'):
        total(np.eye(3))

    # A declared ratio whose denominator has no strain (susceptibles carry none) keeps its strata: the denominator is the whole of it for each strain
    def amr():
        S, I = ss.states('S I')
        infection = ss.Transition(S, I, rate=ss.perday(0.1) * I)
        return ss.Disease(S, I, infection=infection, recovery=ss.Transition(I, S, dur=ss.days(20)), incidence=infection/S, prev=I/(S + I), init_prev=0.05, name='d',
                          strains=ss.Strains(['sens', 'res'], name='amr'))
    for solver in ['ode', 'abm']:
        r = run(amr(), solver, unit='day', stop=50).results.d
        n = (r.S + r.I)[-1]
        assert np.isclose(r.incidence.select(amr='res')[-1], r.infection.select(amr='res')[-1]/r.S[-1])
        assert np.allclose(r.incidence.by('amr').values.sum(axis=1)[1:], r.incidence[1:]) and r.incidence[-1] > 0
        assert np.isclose(r.prev.select(amr='sens')[-1], r.I.select(amr='sens')[-1]/n)

    return sims


@sc.timer()
def test_carriage():
    sc.heading('Testing strains carried together: superinfection, and a transition that changes the strains carried')
    def make():
        S, L, I = ss.states(3)
        strains = ss.Strains(['A', 'B'], max_strains=2, immunity=ss.immunity(due_to=L, rel_sus=0.5)) # Someone with A who is infected with B carries both, at half the rate
        A, B = strains
        return ss.Disease(
            infection  = S >> L @ ss.peryear(6)*I,
            activation = ss.Transition(L, I, rate=ss.peryear(0.5), where=~(A & B)),
            together   = ss.Transition(L, I, rate=ss.peryear(0.2), where=A & B), # Two strains progress together more slowly
            to_A       = ss.Transition(L, I, rate=ss.peryear(0.3), move=(A, B) >> A), # Or only one of the two progresses
            to_B       = ss.Transition(L, I, rate=ss.peryear(0.2), move=(A, B) >> B),
            recovery   = I >> S @ ss.peryear(1),
            init       = dict(I=0.02),
            name       = 'dis',
            strains    = strains,
        )

    # The reference: someone with A and B passes on each with half their infectiousness
    N = 100_000
    def deriv(t, y):
        S, LA, LB, LAB, IA, IB, IAB = y
        la, lb = 6*(IA + IAB/2)/N, 6*(IB + IAB/2)/N
        return [-(la + lb)*S + IA + IB + IAB, la*S - lb/2*LA - 0.5*LA, lb*S - la/2*LB - 0.5*LB, lb/2*LA + la/2*LB - 0.7*LAB,
                0.5*LA + 0.3*LAB - IA, 0.5*LB + 0.2*LAB - IB, 0.2*LAB - IAB]
    ref = sp_integrate.solve_ivp(deriv, [0, 20], [0.98*N, 0, 0, 0, 0.01*N, 0.01*N, 0], rtol=1e-10, atol=1e-6).y[:, -1]
    for solver in solvers:
        res = run(make(), solver, start=2000, stop=2020, dt=ss.years(0.02), total_pop=N).results.dis
        levels = lambda r: np.array([r.select(strain=label)[-1] for label in ['A', 'B', 'A+B']]) # Each set carried
        n_L, n_I = levels(res.L), levels(res.I)
        rtol = 1e-3 if solver == 'ode' else 0.15
        assert np.allclose(n_L, ref[1:4], rtol=rtol) and np.allclose(n_I, ref[4:], rtol=rtol), f'Expected {solver} to match the reference: {n_L}, {n_I} vs {ref[1:]}'
    with pytest.raises(ValueError): # Sets of strains are made by ss.Strains
        make().stratify(other=['A', 'B', 'A+B'])
    return res


def make_resistance(beta=4, immunity=True, cure=(0.9, 0.3), **kwargs):
    """ Two strains in one TB-like course: superinfection of the latent, cure for each strain, resistance emerging at failure """
    S, L, I, Tx, R = ss.states('S L I Tx R')
    strains = ss.Strains(['DS', 'RIF'], max_strains=2, split=[0.8, 0.2], rel_trans=[1, 0.6], cure=list(cure),
                         immunity=ss.immunity(due_to=L, rel_sus=0.7) if immunity else None)
    DS, RIF = strains
    return ss.Disease(infection=S >> L @ ss.peryear(beta)*I, activation=L >> I @ ss.peryear(0.3), diagnosis=I >> Tx @ ss.peryear(2),
                      treatment=ss.Transition(Tx, [R, I], p='cure', rate=ss.peryear(2), per='strain'),
                      failure=ss.Transition(Tx, I, rate=ss.peryear(0.2), move=DS >> (DS, RIF)), relapse=R >> S @ ss.years(2),
                      name='tb', strains=strains, **kwargs)


@sc.timer()
def test_strain_emergence():
    sc.heading('Testing emergence: a change of strains at treatment failure is not an infection')
    res = ss.Sim(make_resistance(beta=0, init=dict(Tx=0.5)), n_agents=4000, start=2000, stop=2003, dt=1/52, **kw).run().results.tb
    assert res.infection.sum() == 0 and res.failure.sum() > 0
    assert res.I.select(strain='DS+RIF')[-1] > 0 # DS acquired resistance, carried with it
    for solver in ['ode', 'abm']: # A delivery that changes strains reaches only the people carrying the strain it changes
        S, I, R = ss.states('S I R')
        strains = ss.Strains(['DS', 'RIF'], split=[1, 0])
        DS, RIF = strains
        tb = ss.Disease(infection=S >> I @ ss.perday(0)*I, recovery=I >> R @ ss.days(1000), init=dict(I=0.3), name='tb', strains=strains)
        mut = ss.Delivery(move=DS >> RIF, coverage=0.5, at=[3], name='mut')
        r = run(tb, solver, interventions=mut, unit='day', stop=6, total_pop=10_000).results
        assert np.isclose(r.mut.eligible.sum(), 3000, rtol=0.05) and np.isclose(r.mut.delivered.sum(), r.tb.I.select(strain='RIF')[-1], rtol=0.05)
    return res


@sc.timer()
def test_strain_import():
    sc.heading('Testing a strain imported on a date, as a delivery')
    for solver in ['ode', 'abm']:
        S, E, I, R = ss.states('S E I R')
        strains = ss.Strains(['w', 'v'], split=[1, 0])
        dis = ss.Disease(infection=S >> E @ ss.perday(0.25)*I, latent=E >> I @ ss.perday(1/3), recovery=I >> R @ ss.perday(1/7), init=dict(I=0.001), strains=strains, name='cov')
        imp = ss.Delivery(move=S >> E, strain='v', at=[50], capacity=20, name='import_v')
        res = ss.Sim(dis, interventions=imp, unit='day', stop=60, solver=solver, n_agents=20_000, total_pop=20_000, **kw).run().results.cov
        assert res.E.by('strain')['v'][50] == 0 and 10 < res.E.by('strain')['v'][51] <= 20, solver
    return


@sc.timer()
def test_strain_carriers():
    sc.heading('Testing where strains are carried (an infection, and what leads back to one), sets at the start, and how a set transmits and progresses')
    S, L, N, I, R = ss.states('S L N I R')
    foi = ss.peryear(8)*I
    def cleared(**kwargs): # R leads back to disease only by a new infection, so carries no strain
        return ss.Disease(S, L, I, R, infection=S >> L @ foi, reinfection=R >> L @ foi, activation=L >> I @ ss.peryear(1), clearance=L >> R @ ss.peryear(1),
                          recovery=I >> R @ ss.peryear(1), init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B'], max_strains=2, **kwargs))
    def staged(imm=None): # N is neither infectious nor entered by infection, but leads to I, so carries its strains
        return ss.Disease(S, L, N, I, R, infection=S >> L @ foi, progress=L >> N @ ss.peryear(1), worsen=N >> I @ ss.peryear(0.5), recovery=I >> R @ ss.peryear(1),
                          init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B'], max_strains=2, immunity=imm))
    for solver in ['ode', 'abm']:
        res = run(cleared(), solver, start=2000, stop=2010, total_pop=20_000).results.d
        assert res.L.select(strain='A+B')[-1] == 0 and res.L[-1] > 0, solver # Reinfected from R with one strain, not added to one cleared
        res = run(cleared(split={'A': 0.5, 'A+B': 0.5}), solver, start=2000, stop=2001, total_pop=20_000).results.d
        assert np.isclose(res.I.select(strain='A')[0], res.I.select(strain='A+B')[0], rtol=0.15) and res.I.select(strain='B')[0] == 0, solver
    plain = run(staged(), 'ode', start=2000, stop=2010, total_pop=1e5).results.d
    superinfected = run(staged(ss.immunity(due_to=N, rel_sus=0.5)), 'ode', start=2000, stop=2010, total_pop=1e5).results.d
    assert np.isclose(superinfected.N[-1], plain.N[-1]) and superinfected.N.select(strain='A+B')[-1] > 0 # People in N carry another strain as well, and stay in N
    with pytest.raises(ValueError, match='is given by "strain", the strains people carry'): # A share by strain would be of people at a level of the strains, where nobody is before they are infected
        run(ss.Disease(S, I, infection=S >> I @ ss.peryear(1)*I, init=dict(I=ss.by(strain=[0.1, 0, 0.1])), name='d', strains=ss.Strains(['A', 'B'], max_strains=2)), 'ode', stop=2001)
    for way in ['rate', 'modifier']: # A transmission is worked out for the people it infects, who carry no strain yet: ss.Strains(..., infection=[...]) says it
        with pytest.raises(ValueError, match='who carry no strain yet, so it would apply to none'):
            d = ss.Disease(S, I, infection=S >> I @ (ss.by(strain=[1, 1.5]) if way == 'rate' else 1)*ss.peryear(1)*I, init=dict(I=0.01), name='d', strains=ss.Strains(['A', 'B']))
            run(d, 'ode', stop=2001, connectors=ss.modify(d.infection, multiply=ss.by(strain=[1, 1.5])) if way == 'modifier' else None)

    # A carrier of A and B: passes on one at a time (the default), or each at its own rate; progresses with one at a time, each at its own rate (alone=True)
    def together(init, transmit='one', **values):
        S, L, I = ss.states('S L I')
        return ss.Disease(S, L, I, infection=S >> I @ ss.peryear(1)*I, activation=ss.Transition(L, I, rate=ss.peryear(1), per='strain', alone=True), init=init,
                          name='d', strains=ss.Strains(['A', 'B'], max_strains=2, split={'A+B': 1}, transmit=transmit, **values))
    first = lambda transmit: run(together(dict(I=0.001), transmit), 'ode', stop=2001, dt=0.01, total_pop=1e6).results.d.infection[1]
    assert np.isclose(first('each'), 2*first('one'), rtol=0.02)
    res = run(together(dict(L=0.5), activation=[1, 3]), 'ode', stop=2001, dt=0.01, total_pop=1e6).results.d
    assert np.isclose(res.activation[1], 0.5e6*4*0.01, rtol=0.05) # The rate of a set is the sum of its strains'
    assert np.isclose(res.I.select(strain='B')[1] - res.I.select(strain='B')[0], 3*res.I.select(strain='A')[1] - 3*res.I.select(strain='A')[0], rtol=0.05) # ... and the strain moving on is chosen in proportion

    return res


@sc.timer()
def test_strain_counts():
    sc.heading('Testing results with strains carried together: each person once in n_I, and in each strain they carry')
    for solver in ['abm', 'ode']:
        res = run(make_resistance(init=dict(I=0.05)), solver, start=2000, stop=2010, dt=1/12).results.tb
        by_strain = res.L.by('strain').values[-1]
        both = res.L.select(strain='DS+RIF')[-1]
        assert both > 0 and np.isclose(by_strain.sum(), res.L[-1] + both) and list(res.L.by('strain').columns) == ['DS', 'RIF'], solver
    return res


@sc.timer()
def test_strain_solvers():
    sc.heading('Testing a model of strains in a shared course (superinfection, cure for each, emergence) as the ODE and as agents')
    out = {solver:run(make_resistance(init=dict(I=0.05)), solver, start=2000, stop=2015, dt=ss.days(7), total_pop=20_000).results.tb for solver in ['ode', 'abm']}
    for state in ['L', 'I', 'Tx']:
        for label in ['DS', 'RIF', 'DS+RIF']:
            ode, abm = out['ode'][state].select(strain=label)[-1], out['abm'][state].select(strain=label)[-1]
            assert np.isclose(abm, ode, rtol=0.15, atol=20), f'{state} {label}: agents {abm:n}, ODE {ode:n}'
    return out


@sc.timer()
def test_space():
    sc.heading('Testing places, with and without movement between them')
    mobility = sc.obj(
        none    = None,
        gravity = ss.geo.gravity(k=2e-4), # Commuting, with the share of time away from the gravity model
        away    = ss.geo.Mobility(ss.geo.gravity(), away=0.2), # Commuting, with 20% of time spent away
        visit   = ss.geo.Mobility(ss.geo.gravity(), away=0.2, kind='visit'), # The same, but travelers do not infect the people they visit
        migrate = ss.geo.Mobility(ss.geo.gravity(k=2e-5), kind='migrate'),
        force   = ss.geo.Mobility(ss.geo.gravity(), away=0.2, kind='force'), # Nobody moves; 20% of each town's force of infection goes to the others
    )
    peaks = sc.obj()
    for key,mob in mobility.items():
        for solver in solvers:
            sim = ss.Sim(make_towns(mobility=mob), stop=150, unit='day', solver=solver, total_pop=100_000, **(dict(n_agents=10_000) if solver == 'abm' else {}), **kw)
            sim.init()
            sim.loop.insert(seed_north, label='sim.start_step')
            sim.run()
            res = sim.results.sir
            attack = res.infection.cum.by('patch').values[-1]/[50_000, 30_000, 20_000]
            if key == 'none': # People mix within their own town, so the epidemic stays where it started
                assert attack[0] > 0.9 and np.allclose(attack[1:], 0)
            else:
                assert np.all(attack > 0.85), f'Expected the epidemic to reach every town with {key} and {solver}, not {attack}'
            if solver == 'abm': # Each agent's place is its group too: one 16-bit array, not three
                assert sim.people.patch.raw.dtype == np.int16 and sim.groups.group_raw() is sim.people.patch.raw
            peaks[f'{key}_{solver}'] = res.I.by('patch').values.argmax(axis=0)
    assert peaks.gravity_ode[0] < peaks.gravity_ode[1] < peaks.gravity_ode[2] # The nearer town is reached first
    assert np.ptp(peaks.away_ode) < 3 < np.ptp(peaks.gravity_ode) # The more time people spend away, the closer together the towns peak
    text = sim.describe(output=True)
    assert 'patch (space): north, south, east' in text and "kind='force'" in text

    # The kind of a dimension comes from its value, not its name: places are an ss.geo.geography (or kind='space'), and a name such as region is noted
    def regions(levels):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(region=[10, 0])), name='sir')
        return ss.Sim(sir.stratify(region=levels), total_pop=20_000, unit='day', stop=200, solver='ode', **kw).run()
    plain, places = regions(['a', 'b']), regions(ss.geo.geography(names=['a', 'b']))
    assert np.all(plain.results.sir.infection.cum.by('region').values[-1] > 4000) # Its levels mix: a name says nothing of the kind
    assert places.results.sir.infection.cum.by('region').values[-1, 1] == 0 # Places are apart

    # A moving force of infection (LASER's coupling), in which M[i,k] of place i's force goes to k, goes only where M sends it; it is the same as a contact matrix over the places
    for M,reached in [([[0, 0.1], [0, 0]], True), ([[0, 0], [0.1, 0]], False)]:
        M = np.array(M)
        def make(mobility=None):
            S, I, R = ss.states('S I R')
            sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(place=[10, 0])), name='sir')
            return sir.stratify(place=['a', 'b'], kind='space', mobility=mobility)
        for solver in ['ode', 'abm']:
            attack = run(make(ss.geo.Mobility(matrix=M, kind='force')), solver, total_pop=20_000, unit='day', stop=200).results.sir.infection.cum.by('place').values[-1]
            assert attack[0] > 5000 and (attack[1] > 5000) == reached, f'Expected b to be reached: {reached}, with {solver}, not {attack}'
            contacts = run(make(), solver, networks=ss.contacts(M.T + np.diag(1 - M.sum(axis=1)), dims='place'), total_pop=20_000, unit='day', stop=200).results.sir.infection.cum.by('place').values[-1]
            assert np.allclose(contacts, attack), f'Expected the contact matrix to give the same run as kind="force" with {solver}: {contacts} against {attack}'

    # A movement matrix may be sparse, for many places; the time share is kept sparse either way, with the same run
    finals = []
    for matrix in [[[0, 0.1], [0.05, 0]], sps.csr_array([[0, 0.1], [0.05, 0]])]:
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(place=[10, 0])), name='sir')
        sim = run(sir.stratify(place=['a', 'b'], kind='space', mobility=ss.geo.Mobility(matrix=matrix, kind='commute')), 'ode', total_pop=20_000, unit='day', stop=100)
        assert sps.issparse(sim.mixing.time_share)
        finals.append(sim.results.sir.infection.cum.by('place').values[-1])
    assert np.array_equal(*finals) and finals[0][1] > 1000 # The same with either

    # A count has one value per place, and a count in a rate is the count in the same place
    S, I, R = ss.states(3)
    W = ss.State(people=False)
    cholera = ss.Disease(
        infection = S >> I @ 0.5 * W.n / (W.n + 1e5),
        recovery  = I >> R @ ss.days(5),
        shedding  = ss.null >> W @ 10 * I.n,
        decay     = W >> ss.null @ ss.days(30),
        init      = dict(I=0.001),
        name      = 'cholera',
    )
    cholera.stratify(patch=['upstream', 'downstream'], kind='space', adjust={'shedding': [1.0, 0.0]})
    for solver in ['ode', 'abm']:
        res = run(cholera, solver, total_pop=100_000, unit='day', stop=200).results.cholera
        assert res.W.by('patch').values[:, 1].max() == 0 # Nobody sheds downstream ...
        assert res.infection.cum.by('patch').values[-1, 1] == 0 # ... so nobody is infected there
        assert res.infection.cum.by('patch').values[-1, 0] > 40_000

    # The passes over the agents are split between threads for a large population (here, for a small one): the results are those of one thread
    defaults = ss.solvers.backends.passes.min_threaded, ss.solvers.backends.passes.chunk_size
    ss.solvers.backends.passes.min_threaded, ss.solvers.backends.passes.chunk_size = 0, 1000
    try:
        sims = []
        for threads in [1, 0]: # One thread, then as many as Numba has
            with ss.options.context(threads=threads):
                sim = ss.Sim(make_towns(mobility=ss.geo.gravity(k=2e-4)), stop=60, unit='day', n_agents=20_000, total_pop=100_000, **kw).init()
                sim.loop.insert(seed_north, label='sim.start_step')
                sims.append(sim.run())
    finally:
        ss.solvers.backends.passes.min_threaded, ss.solvers.backends.passes.chunk_size = defaults
    infections = [sim.results.sir.infection.values for sim in sims]
    assert infections[0].sum() > 0 and np.array_equal(*infections) # The same people infected, with threads or without

    # The share of time away can be fitted: the mixing is worked out again for each value tried
    def commuters(away):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=ss.by(patch=[10, 0, 0])), name='sir')
        return sir.stratify(patch=ss.geo.geography(names=['a', 'b', 'c'], pop=[1e5, 1e5, 1e5], x=[0, 100, 200], y=[0, 0, 0]), mobility=ss.geo.Mobility(ss.geo.gravity(), away=away))
    pars = dict(total_pop=3e5, unit='day', stop=150, **kw)
    for solver in ['ode', 'binomial']:
        truth = ss.Sim(commuters(0.02), solver=solver, **pars).run().results.sir.infection
        data = sc.dataframe(date=truth.timevec, **{f'infection[patch={p}]': truth.select(patch=p) for p in 'abc'})
        away = ss.par(0.1, bounds=[0.002, 0.3])
        fit = ss.Sim(commuters(away), data=data, solver=solver, **pars).fit(verbose=0)
        rtol = 0.05 if solver == 'ode' else 0.5 # A stochastic fit takes fresh seeds at each point, so it does not reproduce the data's own draws (0.023 to 0.029 over three seeds of the fit)
        assert np.isclose(fit.pars.away, 0.02, rtol=rtol), f'Expected to recover away=0.02 with {solver}, not {fit.pars.away}' # Before, a TypeError at init
    sim = ss.Sim(commuters(away), n_agents=3000, solver='abm', **pars).init()
    shares = [sim.new(pars=dict(away=val)).init().mixing.time_share.diagonal() for val in [0.01, 0.2]] # As agents too: the share of time at home
    assert np.allclose(shares[0], 0.99) and np.allclose(shares[1], 0.8)
    restarted = ss.Sim(commuters(away), solver='ode', **pars).init().restart(pars=dict(away=0.05)).run().results.sir.I.values # The mixing is worked out again on a restart
    assert np.allclose(restarted, ss.Sim(commuters(0.05), solver='ode', **pars).run().results.sir.I.values)

    # A gravity model needs a geography
    with pytest.raises(ValueError, match='which has no geography'):
        I, R = ss.states('I R')
        sir = ss.Disease(recovery=ss.Transition(I, R, dur=3)).stratify(patch=['a', 'b'], mobility=ss.geo.gravity())
        ss.Sim(sir, **kw).init()

    # The radiation model has no parameters: of the people who travel from a place, the share who go to each other place comes from the populations of the places nearer than it
    line = ss.geo.geography(names=['a', 'b', 'c', 'd'], pop=[4e4, 1e4, 2e4, 3e4], x=[0, 10, 30, 70], y=[0, 0, 0, 0])
    M = ss.geo.radiation().matrix(ss.dimensions.Strat('patch', line))
    assert np.allclose(M[0], np.array([0, 4/20, 8/35, 12/70])/(1 - 4/10)) and np.allclose(M.sum(axis=1), 1) # m·n/((m + s)(m + n + s)), for the finite population (with no ties in distance, each row sums to 1)
    sim = ss.Sim(make_towns(mobility=ss.geo.radiation()), unit='day', **kw).init()
    assert np.allclose(sim.mixing.time_share.diagonal(), 0.9) # As for gravity, 10% of time away unless away= is given
    sim = ss.Sim(make_towns(mobility=ss.geo.radiation(away=0.01)), stop=150, unit='day', solver='ode', total_pop=100_000, **kw).init()
    sim.loop.insert(seed_north, label='sim.start_step')
    peaks = sim.run().results.sir.I.by('patch').values.argmax(axis=0)
    assert peaks[0] < peaks[1] < peaks[2] # South is nearer to the north than east is, and bigger
    with pytest.raises(ValueError, match='which has no geography'):
        ss.Sim(ss.Disease(recovery=ss.Transition(I, R, dur=3)).stratify(patch=['a', 'b'], mobility=ss.geo.radiation()), **kw).init()

    # Places from your own shapes (a GeoDataFrame, or a file of them), each at the centroid of its shape
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError: # Without geopandas, a file of shapes says how to read it
        with pytest.raises(ImportError, match='pip install geopandas'):
            ss.geo.geography('districts.shp')
    else:
        shapes = gpd.GeoDataFrame(dict(name=['a', 'b', 'c'], population=[1e4, 2e4, 3e4]), geometry=[box(36, -1, 37, 0), box(37, -1, 38, 0), box(36, 0, 37, 1)], crs='EPSG:4326')
        places = ss.geo.geography(shapes)
        assert np.allclose(places.lon, [36.5, 37.5, 36.5]) and np.allclose(places.pop, [1e4, 2e4, 3e4])
        assert np.isclose(places.distances[0, 1], 111.2, rtol=1e-3) # A degree of longitude at the equator, in km
        with tempfile.TemporaryDirectory() as folder:
            path = f'{folder}/districts.geojson'
            shapes.to_file(path)
            assert np.allclose(ss.geo.geography(path).distances, places.distances)
        assert np.allclose(ss.geo.geography(shapes.set_crs(None, allow_override=True)).distances[0, 1], 1) # With no coordinate reference system, in the shapes' own units

    # Newborns from ss.Births, written as code, are placed by the split, as declared births are (before, all in the first place)
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(0.5) * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.1, name='sis')
    sis.stratify(patch=['a', 'b', 'c'], kind='space', split=[0.4, 0.3, 0.3])
    sim = ss.Sim(sis, n_agents=5_000, demographics=[ss.Births(birth_rate=ss.peryear(40)), ss.Deaths(death_rate=ss.peryear(40))], start=2000, stop=2020, verbose=0).run()
    n = sim.results.sis.S.by('patch').values[-1] + sim.results.sis.I.by('patch').values[-1]
    assert np.allclose(n/n.sum(), [0.4, 0.3, 0.3], atol=0.03) # Was 0.73, 0.13, 0.13
    return peaks


@sc.timer()
def test_unnamed_data():
    sc.heading('Testing that a model with no named data does not import starsim_data')
    code = """
import sys
import numpy as np
import starsim as ss
S, I, R = ss.states(3)
sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
towns = ss.geo.geography(names=['a', 'b'], pop=[6000, 4000], x=[0, 10], y=[0, 0])
ss.Sim(sir.stratify(patch=towns, mobility=ss.geo.gravity()), networks=ss.contacts([[3, 1], [1, 2]], ages=[0, 18]), unit='day', stop=20, verbose=0).run()
ss.Sim(sir, networks=ss.RandomNet(), n_agents=500, unit='day', stop=20, verbose=0).run()
ss.Sim(ss.SIS(), stratify=dict(age=[0, 15, 65]), solver='ode', verbose=0).run()
print('starsim_data' in sys.modules)
"""
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.split()[-1] == 'False', 'A model with no named places or matrices imported starsim_data'
    return


@sc.timer()
def test_mixing():
    sc.heading('Testing contact matrices by dimension, and where people start')
    def make():
        S, I, R = ss.states('S I R')
        sir = ss.Disease(
            infection = ss.Transition(S, I, rate=2 * I),
            recovery  = ss.Transition(I, R, rate=1/3),
            init_n    = dict(I=10),
            name      = 'sir',
        )
        sir.stratify(age=['young', 'old']) # Labels, so not the age that people have
        sir.stratify(location=['urban', 'rural'], split=[0.7, 0.3])
        return sir
    mixing = ss.contacts(dict(age=[[0.2, 0.3], [0.5, 0.7]], location=[[0.8, 0.2], [0.2, 0.8]]))
    sims = sc.obj({solver:run(make(), solver, networks=mixing, total_pop=10_000, dt=0.1, stop=20) for solver in solvers})
    res = sims.ode.results.sir
    assert sims.ode.groups.strats[0].kind == 'plain'
    attack = {(a,l):res.R.select(age=a, location=l)[-1]/res.S.select(age=a, location=l)[0] for a in ['young', 'old'] for l in ['urban', 'rural']}
    assert attack['old', 'urban'] > attack['young', 'urban'] # Old people have more contacts
    assert np.isclose(attack['young', 'urban'], attack['young', 'rural']) # The two kinds of place mix alike and are seeded as their people are split, so they end alike (before, a strict > held by a rounding error)
    for solver in ['binomial', 'abm']:
        assert np.isclose(sims[solver].results.sir.R[-1], res.R[-1], rtol=0.1), f'Expected {solver} to match the ODE'
    assert sims.abm.results.sir.I[0] <= 10 # As agents, the number who start in each state is exact (less anyone who has already recovered)
    with pytest.raises(ValueError, match='has 3 rows, but "location" has 2 levels'):
        ss.Sim(make(), networks=ss.contacts(np.ones((3, 3)), dims='location'), **kw).init()
    assert mixing.pars.n_contacts[1, 2] == 0.5*0.2 # From (old, urban) to (young, rural)
    T = sc.timer()
    ss.Sim(make(), networks=ss.contacts(np.eye(500) + 0.01, dims='place'), **kw) # A matrix over many places is copied into the sim as numbers, not as 250,000 objects
    assert T.toc(output=True) < 5 # About 0.1 s; 774 places took 170 s

    # Where people start can be given for each level of a dimension
    def start(solver):
        sir = make_towns()
        sir.graph.states.I.init, sir.graph.states.I.init_n = None, ss.by(patch=[0, 500, 0])
        return run(sir, solver, total_pop=100_000, unit='day', stop=5)
    for solver in solvers:
        n_I = start(solver).results.sir.I.by('patch').values[0]
        assert n_I[0] == 0 and n_I[2] == 0 and 400 < n_I[1] < 700 # After the first day

    # People age between age groups only if they are born or die, unless the sim says otherwise
    youngest = dict()
    for use_aging in [None, True, False]:
        S, I = ss.states('S I')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), name='sir').stratify(age=[0, 5, 20]) # Nobody is infected, so everyone stays susceptible
        youngest[use_aging] = ss.Sim(sir, start=2000, stop=2010, use_aging=use_aging, **kw).run().results.sir.S.by('age').values[-1, 0]
    assert youngest[False] > 2*youngest[True] and youngest[None] == youngest[False] # A closed population does not age by default
    S, I = ss.states('S I')
    born = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), births=ss.birth(S, rate=ss.peryear(0.02)), deaths=ss.death(rate=ss.peryear(0.02)), name='sir').stratify(age=[0, 5, 20])
    assert ss.Sim(born, start=2000, stop=2001, **kw).init().pars.use_aging # With births and deaths, people age

    # A split over two dimensions at once, a value for every combination of levels, and a contact matrix within each place
    def make_places(mobility=None):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.05) * I), recovery=ss.Transition(I, R, dur=ss.days(10)),
                         init_n=dict(I=ss.by(pd.DataFrame(dict(age=['0-4'], place=['a'], value=[10])))), name='sir')
        sir.stratify(age=[0, 5, 50])
        sir.stratify(place=['a', 'b'], kind='space', split=ss.by([[3000, 1000], [5000, 2000], [2000, 500]], dims=['age', 'place']), mobility=mobility)
        return ss.Sim(sir, networks=ss.contacts([[12, 4, 1], [4, 8, 2], [1, 2, 3]], ages=[0, 5, 50]), total_pop=13_500, unit='day', stop=100, **kw)
    start = at_start(make_places()).sir
    n = [sum(start[k].select(age=age, place=place)[0] for k in 'SIR') for age in ['0-4', '50+'] for place in ['a', 'b']]
    assert np.allclose(n, [3000, 1000, 2000, 500])
    assert start.I.select(age='0-4', place='a')[0] == 10 and start.I[0] == 10
    alone = make_places().run().results.sir
    assert alone.infection.cum.by('place').values[-1, 1] == 0 and alone.infection.cum[-1] > 5000 # Nobody moves, so the epidemic stays in a
    many = make_places().new(solver='binomial').run(n_runs=3).results.sir # Many runs keep their values by group too
    assert many.I.select(age='0-4', place='a').values.shape == (3, 101) and np.all(many.infection.cum.by('place')['b'].values == 0)
    moving = make_places(ss.geo.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate'))
    assert moving.run().results.sir.infection.cum.by('place').values[-1, 1] > 1000
    agents = make_places().new(n_agents=13_500).run()
    shares = [np.mean((agents.people.place[agents.people.age < 5] == 0)), np.mean((agents.people.place[agents.people.age >= 50] == 0))]
    assert np.allclose(shares, [0.75, 0.8], atol=0.05) # As agents, where people live depends on their age

    # As agents, the contact matrix acts within each place too, with commuting between them
    assert agents.results.sir.infection.cum.by('place').values[-1, 1] == 0
    commute = ss.geo.Mobility(matrix=[[0.9, 0.1], [0.1, 0.9]], kind='commute')
    ode, abm = [make_places(commute).new(**pars).run().results.sir.infection.cum.by('place').values[-1] for pars in [{}, dict(n_agents=13_500)]]
    assert np.allclose(abm, ode, rtol=0.05) and ode[1] > 1000

    # So do a network and a mixing pool of everyone: as agents, contacts are within each place, or where people commute to
    def make_net(net, mobility=None, **pars):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.03) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_n=dict(I=ss.by(patch=[20, 0, 0])), name='sir')
        sir.stratify(patch=['a', 'b', 'c'], kind='space', split=[0.5, 0.3, 0.2], mobility=mobility)
        return ss.Sim(sir, networks=net, total_pop=30_000, unit='day', stop=120, **kw, **pars)
    commute = ss.geo.Mobility(matrix=[[0, 0.05, 0], [0.05, 0, 0.05], [0, 0.05, 0]], kind='commute')
    for net in [ss.RandomNet(n_contacts=10, dur=0, name='net'), ss.MixingPool(n_contacts=10, name='net')]:
        alone = make_net(net, n_agents=30_000).run().results.sir.infection.cum.by('patch').values[-1]
        assert alone[0] > 10_000 and alone[1] == alone[2] == 0
        ode, abm = [make_net(sc.dcp(net), commute, **pars).run().results.sir.infection.cum.by('patch').values[-1] for pars in [dict(solver='ode', approx=True), dict(n_agents=30_000)]]
        assert np.allclose(abm, ode, rtol=0.05) and ode[2] > 1000

    # An explicit contact matrix over the places is used whatever the dimension is called
    for solver in ['ode', 'abm']:
        final = {dim:corpus.patch_matrix(dim=dim, solver=solver, verbose=0).run().results.sir.R.by(dim).values[-1] for dim in ['patch', 'zone']}
        assert np.allclose(final['patch'], final['zone'], rtol=0.01) and final['patch'][1] > 9000 # Not [9,802, 0] with patch

    # A contact matrix is aggregated to the model's age groups by population, so the ODE matches agents, who have ages
    def attack(solver):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.03 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir').stratify(age=[0, 50])
        sim = ss.Sim(sir, networks=ss.contacts([[12, 1, 1], [1, 2, 1], [1, 1, 1]], ages=[0, 5, 50]), people=ss.people('Kenya'), use_aging=False, unit='day', stop=300, solver=solver, **kw)
        res = sim.run().results.sir
        return res.infection.cum[-1]/res.S[0]
    assert np.isclose(attack('ode'), attack('abm'), rtol=0.05) # 0.58 and 0.57; 0.50 by band width
    S, I, R = ss.states('S I R') # A count in a rate is used literally, so the contacts do not apply to it, which is noted
    literal = ss.Disease(infection=S >> I @ 0.05*I.n/(S + I + R).n, recovery=I >> R @ ss.days(10), init=dict(I=0.01), name='sir').stratify(age=[0, 15])
    assert any('literally' in note for note in ss.Sim(literal, networks=ss.contacts([[3, 1], [1, 2]], ages=[0, 15]), **kw).init().notes)
    for bad in [dict(matrix=[[-5, 0], [0, 1]], ages=[0, 20]), dict(matrix=[[1, 0], [0, 1]], ages=[20, 0]), dict(matrix=[[1, np.nan], [1, 1]])]:
        with pytest.raises(ValueError):
            ss.contacts(**bad)

    # A split over two dimensions is the share in each level given the other, whose own split, if given, is kept (FPsim review: it was replaced by the table's totals)
    for solver in ['ode', 'abm']:
        A = ss.State('A')
        d = ss.Disease(A, stay=ss.Transition(A, A, rate=0), first='A', name='d')
        d.stratify(k=['x', 'y'], split=ss.by([[0.8, 0.2], [0.4, 0.6]], dims=['age', 'k']))
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        n_A = at_start(ss.Sim(d, solver=solver, stratify=dict(age=[0, 30], split=[0.9, 0.1]), stop=1, **size, **kw)).d.A
        assert np.allclose(n_A.by('age').values[0], [9000, 1000], rtol=0.03)
        assert np.allclose([n_A.select(age=a, k='y')[0] for a in ['0-29', '30+']], [1800, 600], rtol=0.1)

    # Where people start, given at levels where the split puts nobody, is an error, not zero (FPsim review: everyone starts at ever=False)
    def make_init(table):
        A, B = ss.states('A B')
        d = ss.Disease(A, B, ab=A >> B @ ss.peryear(0), init=dict(B=ss.by(np.array(table), dims=['age', 'ever'])), first='A', name='d')
        d.stratify(ever=[False, True], split=[1, 0])
        return ss.Sim(d, total_pop=1000, stratify=dict(age=[0, 30]), stop=1, **kw)
    with pytest.raises(ValueError, match='ever=True'):
        make_init([[0, 0.5], [0, 0.5]]).init()
    assert any('ever=True' in note for note in make_init([[0, 0.5], [0.2, 0.5]]).init().notes) # Some of it: a note
    return sims


@sc.timer()
def test_ragged():
    sc.heading('Testing a dimension for some groups only: age groups for women')
    def make():
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=0.2 * I), recovery=ss.Transition(I, R, dur=10), init=dict(I=ss.by(age=[0.004, 0.002, 0.001, 0.0])), name='sir')
        sir.stratify(sex=['female', 'male'])
        sir.stratify(age=[15, 25, 35, 50], where=dict(sex='female'), adjust=dict(recovery=[2, 1, 1, 0.5]))
        return sir
    sims = sc.obj({solver:run(make(), solver, unit='day', stop=150, total_pop=100_000) for solver in solvers})
    ode = sims.ode
    assert np.all(ode.groups.group_levels[ode.groups.group_coords[:, 0] == 1, 1] == -1) # Men have no age group
    res = ode.results.sir
    start = at_start(ss.Sim(make(), unit='day', stop=150, total_pop=100_000, **kw)).sir
    split = next(strat for strat in ode.groups.strats if strat.name == 'age').split
    assert np.isclose(start.I.select(sex='male')[0], 50_000*(split @ [0.004, 0.002, 0.001, 0.0]), rtol=0.01) # A value for each age group is, for men, the mean over them, weighted by the split
    assert np.allclose(res.R.by('age').values.sum(axis=1), res.R.select(sex='female').values) # Selecting an age group selects women only
    for solver in ['binomial', 'abm']:
        other = sims[solver].results.sir
        assert np.allclose(other.R.by('sex')[-1], res.R.by('sex')[-1], rtol=0.05), f'Expected {solver} to match the ODE'
        assert np.allclose(other.R.by('age')[-1], res.R.by('age')[-1], rtol=0.1), f'Expected {solver} to match the ODE'

    # A condition on the dimension leaves out the people who have no level of it
    for solver in ['ode', 'abm']:
        sir = make()
        sim = run(sir, solver, unit='day', stop=150, total_pop=100_000, interventions=ss.modify(sir.rel_sus, multiply=0, where=sir.age == '15-24'))
        r = sim.results.sir
        assert r.infection.cum.select(age='15-24')[-1] == 0 and r.infection.cum.select(sex='male')[-1] > 10_000

    # A dimension within a ragged one, set on the sim; contacts by a ragged dimension have no level for the others to mix as
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.2 * I), recovery=ss.Transition(I, R, dur=10), init=dict(I=0.001), name='sir')
    sim = ss.Sim(sir, stratify=[dict(sex=['female', 'male']), dict(age=[15, 25, 35, 50], where=dict(sex='female')), dict(parity=['0', '1+'], split=[0.4, 0.6], where=dict(age=['25-34', '35-49']))], total_pop=10_000, unit='day', stop=50, **kw).run()
    split = next(strat for strat in sim.groups.strats if strat.name == 'age').split
    assert (sim.groups.group_levels[:, 2] >= 0).sum() == 4 and np.isclose(sim.results.sir.S.by('parity').values[0].sum(), 5_000*split[1:3].sum(), rtol=0.01) # Women aged 25 to 49
    with pytest.raises(ValueError, match='so there is no level for the others to mix as'):
        ss.Sim(make(), networks=ss.contacts([[3, 1, 1, 1], [1, 3, 1, 1], [1, 1, 3, 1], [1, 1, 1, 3]], ages=[15, 25, 35, 50]), **kw).init()
    with pytest.raises(ValueError, match='so there is no level of it for the people in I to mix as'): # So do contacts by a dimension that the infectious state does not have
        ss.Sim(sir.copy().stratify(age=[0, 15], states=['S', 'R']), networks=ss.contacts([[3, 1], [1, 3]], ages=[0, 15]), **kw).init()
    with pytest.raises(KeyError, match='but the model is not stratified by "sex"'):
        ss.Sim(sir.copy().stratify(age=[0, 15], where=dict(sex='female')), **kw).init()

    # As agents, a sex dimension is people.female, which a sexual network pairs on
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.5 * I), recovery=ss.Transition(I, R, dur=30), init_prev=0.05, name='sir')
    sir.stratify(sex=['female', 'male'], split=[0.6, 0.4], adjust={'infection': [0, 1]})
    sim = ss.Sim(sir, networks=ss.SexualNet(), n_agents=5_000, unit='day', stop=50, **kw).run()
    res, women = sim.results.sir, sim.people.female.uids
    assert np.isclose(len(women), 3_000, rtol=0.05) and np.all(sim.people.sex[women] == 0) # The split is the share of women
    assert sim.diseases.sir.I[women].sum() + sim.diseases.sir.R[women].sum() == res.I.select(sex='female')[-1] + res.R.select(sex='female')[-1] # Not 51 women infected through the network and 0 in the results
    return sims


@sc.timer()
def test_properties_in_rates():
    sc.heading('Testing people\'s properties and conditions in rates, as a regression')

    # A probability of use as a regression on a property, the same as agents and as compartments, and stopped by ss.change on its intercept (before, 3,531 starts with and without it)
    def make(solver, change=False):
        b0, b_urban = ss.par(-1.0, bounds=[-3, 1]), ss.par(1.0)
        none, on = ss.states('none on')
        start = ss.Transition(none, on, rate=ss.peryear(1)*expit(b0 + b_urban*ss.ppl.urban))
        meth = ss.Disease(start=start, stop=ss.Transition(on, none, rate=ss.peryear(0.5)), b0=b0, b_urban=b_urban, first='none', name='meth')
        intv = ss.change('meth.pars.b0', at=2005, to=-np.inf) if change else None
        sim = ss.Sim(meth, interventions=intv, n_agents=20_000, start=2000, stop=2010, dt=1/12, solver=solver, verbose=0)
        sim.stratify(urban=[False, True], split=[0.4, 0.6])
        return sim.run()
    on = [make(solver).results.meth.on[-1] for solver in ['abm', 'ode']]
    expected = 20_000*(0.4*expit(-1)/(expit(-1) + 0.5) + 0.6*0.5/(0.5 + 0.5)) # At equilibrium
    assert np.isclose(on[1], expected, rtol=0.01) and np.isclose(on[0], expected, rtol=0.03)
    for solver in ['abm', 'ode']:
        assert make(solver, change=True).results.meth.start[61:].sum() == 0

    # A condition in a rate is 1 or 0 for each person; on states alone joined by |, it is still an error
    S, I, R = ss.states('S I R')
    sir = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.peryear(1)*(1 + 2*(ss.ppl.age > 50)), init=dict(I=0.01), name='sir')
    sim = ss.Sim(sir, n_agents=10_000, stop=5, dt=0.1, use_aging=False, verbose=0).run()
    ode = ss.Sim(sir, total_pop=10_000, stop=5, dt=0.1, use_aging=False, solver='ode', verbose=0)
    ode.stratify(age=[0, 50])
    ode.run()
    assert np.isclose(sim.results.sir.R[-1], ode.results.sir.R[-1], rtol=0.15)
    with pytest.raises(TypeError, match=r'0.3\*\(Ia \+ Is\)'):
        ss.Disease(S, I, R, infection=S >> I @ 0.3*(S | I))

    # Shown as people.urban and expit()
    sim = make('abm')
    assert 'expit(b0 + b_urban * people.urban)' in ss.expressions.format_expr(sim.diseases[0].graph.transitions[0].rate, exact=True)

    # A property that is not one of the people's says what is
    bad = ss.Disease(S, I, R, infection=S >> I @ 0.3*I, recovery=I >> R @ ss.peryear(1)*ss.ppl.agee, name='bad')
    with pytest.raises(KeyError, match='age'):
        ss.Sim(bad, n_agents=100, verbose=0).init()
    return sim


@sc.timer()
def test_by_properties():
    sc.heading('Testing values by a property of people that is not a dimension, and tables with bin edges')
    def make(mu, solver='auto', dims=None, **kw):
        S, D = ss.states('S D')
        sim = ss.Sim(ss.Disease(S, D, death=S >> D @ mu, name='d'), solver=solver, use_aging=False, verbose=0, **sc.mergedicts(dict(stop=10), kw))
        for dim,levels in (dims or {}).items():
            sim.stratify(**{dim:levels})
        return sim

    # Death rates by sex with no sex dimension: read from people.female as agents; an explicit ODE says to stratify
    mu = ss.by(sex=dict(female=ss.peryear(0.02), male=ss.peryear(0.05)))
    sim = make(mu, n_agents=20_000).run()
    assert sim.solver == 'abm'
    female, dead = sim.people.female.raw[sim.people.auids], sim.diseases.d.D.raw[sim.people.auids]
    for sel,rate in [(female, 0.02), (~female, 0.05)]:
        p = 1 - np.exp(-10*rate)
        assert abs(dead[sel].mean() - p) < 3*np.sqrt(p*(1 - p)/sel.sum())
    with pytest.raises(ss.CapabilityError, match='stratify'):
        make(mu, solver='ode', total_pop=20_000).init()
    ode = make(mu, solver='ode', total_pop=20_000, dims=dict(sex=['female', 'male'])).run() # The same keys, by a dimension
    assert np.allclose(ode.results.d.D.by('sex').values[-1], 10_000*(1 - np.exp(-10*np.array([0.02, 0.05]))), rtol=1e-3)
    with pytest.raises(KeyError, match=r'dict\(female='):
        make(ss.by(sex=[ss.peryear(0.02), ss.peryear(0.05)]), n_agents=100).init()
    with pytest.raises(KeyError, match='lower edge'):
        make(ss.by(age=[ss.peryear(0.02), ss.peryear(0.05)]), n_agents=100).init()

    # A table with bin edges: each person's own age as agents; as compartments, only bins that each group is inside (bins that split a group need agents)
    table = ss.by(dict(age=[0, 20, 40, 60], value=[ss.peryear(v) for v in [0.01, 0.02, 0.03, 0.04]]))
    abm = make(table, n_agents=50_000, solver='abm').run()
    ode = make(table, total_pop=50_000, solver='ode', dims=dict(age=[0, 10, 20, 40, 60])).run() # Groups inside the bins: exact
    assert np.isclose(abm.results.d.D[-1], ode.results.d.D[-1], rtol=0.03)
    with pytest.raises(ss.CapabilityError, match='split'):
        make(table, total_pop=50_000, solver='ode', dims=dict(age=[0, 30, 60])).init()
    assert make(table, total_pop=50_000, dims=dict(age=[0, 30, 60])).init().solver == 'abm'

    # A row for people with no value (None), and a product of values by different dimensions
    ppl = ss.People(10_000, extra_states=[ss.FloatArr('since_birth', default=ss.choice([np.nan, 1.0, 10.0], p=[0.5, 0.25, 0.25]))])
    rate = ss.by(dict(since_birth=[None, 0, 5], value=[ss.peryear(0.0), ss.peryear(0.5), ss.peryear(0.1)]))
    sim = make(rate, people=ppl, stop=1).run()
    since, dead = sim.people.since_birth.raw[sim.people.auids], sim.diseases.d.D.raw[sim.people.auids]
    assert dead[np.isnan(since)].sum() == 0 and dead[since == 1].mean() > 2*dead[since == 10].mean()
    both = ss.by(sex=dict(female=1.0, male=2.0))*ss.by(urban={False:1.0, True:3.0})
    assert both.dims == ['sex', 'urban']

    # A distribution for each level of a property, as a dwell time
    A, B = ss.states('A B')
    dur = ss.by(sex=dict(female=ss.lognorm_ex(ss.months(6), ss.months(2)), male=ss.lognorm_ex(ss.months(12), ss.months(4))))
    sim = ss.Sim(ss.Disease(A, B, go=ss.Transition(A, B, dur=dur), first='A', name='d'), n_agents=10_000, unit='month', stop=3, verbose=0).init()
    uids = sim.people.auids
    drawn = sim.diseases.d.engine.sample_dwell(sim.diseases.d.graph.transitions['go'], uids) # In months
    female = sim.people.female.raw[uids]
    assert np.isclose(drawn[female].mean(), 6, rtol=0.05) and np.isclose(drawn[~female].mean(), 12, rtol=0.05)

    # Several non-exponential dwell times out of a state, each for one sex: a track for each as compartments, as agents as ever
    def clearance(where_m=ss.ppl.male):
        S, A, R = ss.states('S A R')
        clear_f = ss.Transition(A, R, dur=ss.lognorm_ex(ss.months(8), ss.months(2)), where=ss.ppl.female)
        clear_m = ss.Transition(A, S, dur=ss.lognorm_ex(mean=ss.by(sex=dict(female=ss.months(6), male=ss.months(6))), std=ss.months(3)), where=where_m)
        return ss.Disease(S, A, R, clear_f=clear_f, clear_m=clear_m, init=dict(S=0.5, A=0.5), name='d').stratify(sex=['female', 'male'])
    left = sc.obj()
    for solver,pop in [('ode', dict(total_pop=20_000)), ('binomial', dict(total_pop=20_000)), ('abm', dict(n_agents=20_000))]:
        n = ss.Sim(clearance(), unit='month', dt=0.25, stop=10, solver=solver, verbose=0, **pop).run().results.d.A.by('sex').values
        left[solver] = n[32]/n[0] # Still infected at 8 months
    erlang = [stats.gamma.sf(8, 16, scale=8/16), stats.gamma.sf(8, 4, scale=6/4)] # 16 and 4 stages
    assert np.allclose(left.ode, erlang, atol=1e-3) and np.allclose(left.binomial, erlang, atol=0.03) and np.allclose(left.abm, [0.476, 0.208], atol=0.02) # Lognormal as agents (those who leave at 7.875 months or later, at the step nearest)
    sim = ss.Sim(clearance(where_m=ss.ppl.female), stop=10, verbose=0).init() # Both for women: none as compartments, so agents
    assert sim.solver == 'abm'
    with pytest.raises(ss.CapabilityError, match='same people'):
        ss.Sim(clearance(where_m=ss.ppl.female), stop=10, solver='ode', verbose=0).init()

    # For the same people: the two clocks race, as a compartment for each pair of their Erlang stages, so the share leaving by each is exact
    def race(k=3):
        S, A, B, C = ss.states('S A B C')
        return ss.Disease(S, A, B, C, to_b=ss.Transition(A, B, dur=ss.gamma(k, scale=ss.months(10/k))), to_c=ss.Transition(A, C, dur=ss.gamma(2, scale=ss.months(7))), init=dict(A=1.0), first='S', name='d')
    exact = sp_integrate.quad(lambda t: stats.gamma.pdf(t, 3, scale=10/3)*stats.gamma.sf(t, 2, scale=7), 0, np.inf)[0] # P(the first clock ends first)
    for solver,pop in [('ode', dict(total_pop=1e6)), ('binomial', dict(total_pop=1e6)), ('abm', dict(n_agents=20_000))]:
        r = ss.Sim(race(), unit='month', dt=0.25, stop=100, solver=solver, verbose=0, **pop).run().results.d
        assert np.isclose(r.B[-1]/(r.B[-1] + r.C[-1]), exact, atol=1e-4 if solver == 'ode' else 0.01), f'{solver}: {r.B[-1]/(r.B[-1] + r.C[-1]):.4f} against {exact:.4f}'
    with pytest.raises(ss.CapabilityError, match='combination of their Erlang stages'):
        ss.Sim(race(k=300), unit='month', stop=10, solver='ode', verbose=0).init()
    return sim


@sc.timer()
def test_counters():
    sc.heading('Testing counters: increment= on transitions, deliveries, and transfers')
    def make(solver, twins=0.0, **kw):
        NP, P = ss.states('NP P')
        conceive = ss.Transition(NP, P, rate=ss.by(age=[ss.peryear(0), ss.peryear(0.25), ss.peryear(0.15), ss.peryear(0.05)]))
        single = ss.Transition(P, NP, dur=ss.months(9), p=1 - twins, increment='parity')
        double = ss.Transition(P, NP, dur=ss.months(9), p=twins, increment=dict(parity=2)) # Twins add 2
        sim = ss.Sim(ss.Disease(NP, P, conceive=conceive, single=single, double=double, first='NP', name='preg'), solver=solver, start=2000, stop=2010, dt=ss.months(1), use_aging=False, verbose=0, **kw)
        sim.stratify(age=[0, 15, 25, 35])
        sim.stratify(parity=list(range(8)))
        return sim.run()
    def mean_parity(sim): # By age group, of those 15 and over
        res = sim.results.preg
        n = [res.NP.select(age=a).by('parity').values[-1] + res.P.select(age=a).by('parity').values[-1] for a in range(1, 4)]
        return np.array([(k*np.arange(8)).sum()/k.sum() for k in n])

    # Mean parity by age, agents against the ODE (0.7% apart at most, measured), and twins add 2
    ode, abm = mean_parity(make('ode', total_pop=100_000)), mean_parity(make('abm', n_agents=100_000))
    assert np.allclose(abm, ode, rtol=0.01) and ode[0] > 1.5
    sim = make('abm', twins=1.0, n_agents=2000)
    parity = sim.people.parity[sim.people.auids]
    assert np.all(parity % 2 == 0) and parity.max() > 7 # The last level is open-ended, and agents keep counting
    assert np.allclose(mean_parity(make('ode', twins=0.5, total_pop=5000)), 1.5*ode, rtol=0.02) # A little under, since the ODE's 7+ is 7

    # A delivery adds a dose to the people it reaches, a transfer takes a level of immunity off at a rate, and recovery counts episodes
    def campaign(solver, **kw):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S, I, R, infection=S >> I @ 0.2*I, recovery=ss.Transition(I, R, dur=ss.days(10), increment='episodes'), init=dict(I=0.01), name='sir')
        sir.stratify(waning=ss.transfer(increment=dict(immunity=-1), rate=ss.peryear(0.5)), immunity=[0, 1, 2, 3], split=[0, 0, 0, 1])
        camp = ss.Delivery(increment='doses', coverage=0.6, at=[1, 2, 3], name='camp')
        sim = ss.Sim(sir, interventions=camp, solver=solver, start=0, stop=5, dt=ss.days(1), unit='year', verbose=0, **kw)
        sim.stratify(doses=[0, 1, 2, 3], episodes=[0, 1, 2])
        return sim.run()
    def total(sim, dim):
        res = sim.results.sir
        return res.S.by(dim).values[-1] + res.I.by(dim).values[-1] + res.R.by(dim).values[-1]
    ode, abm = campaign('ode', total_pop=20_000), campaign('abm', n_agents=20_000)
    exact = 20_000*np.array([0.4**3, 3*0.6*0.4**2, 3*0.6**2*0.4, 0.6**3]) # Doses after three rounds at 60%: binomial
    assert np.allclose(total(ode, 'doses'), exact) and np.allclose(total(abm, 'doses'), exact, rtol=0.05)
    m = 2.5 # Levels lost in 5 years at 0.5 per year: Poisson
    lost = np.array([1 - sum(np.exp(-m)*m**k/math.factorial(k) for k in range(3))] + [np.exp(-m)*m**k/math.factorial(k) for k in [2, 1, 0]])
    assert np.allclose(total(ode, 'immunity'), 20_000*lost, rtol=1e-3) and np.allclose(total(abm, 'immunity'), 20_000*lost, rtol=0.05)
    assert np.isclose(total(ode, 'episodes')[1], ode.results.sir.recovery.values.sum())
    with pytest.raises(ValueError, match='whole number'):
        I, R = ss.states('I R')
        ss.Transition(I, R, dur=10, increment=dict(episodes=0.5))
    with pytest.raises(KeyError, match='stratify'):
        I, R = ss.states('I R')
        ss.Sim(ss.Disease(ss.Transition(I, R, dur=10, increment='episodes'), init=dict(I=0.1), first='I', name='d'), n_agents=100, verbose=0).run()
    return abm


@sc.timer()
def test_rows():
    sc.heading('Testing p= with one entry per destination, by age, and a function for each destination')
    methods = ['pill', 'iud', 'inj']
    def make(solver, fn=False, **kw):
        none = ss.State('none')
        on = {m: ss.State(m) for m in methods}
        b0 = ss.par(-0.5)
        p_use = expit(b0 + 1.0*ss.ppl.urban)
        mix = [ss.par(ss.by(age=v), name=f'mix_{m}') for m,v in zip(methods, [[0.6/1.3, 0.2/1.5], [0.6/1.3, 1.0/1.5], [0.1/1.3, 0.3/1.5]])] # The switching row by age, one entry per method
        row = [lambda sim, uids, v=v: np.full(len(uids), v) for v in [0.5, 0.2, 0.2, 0.1]] # A function for each destination
        trs = {f'from_{src.name}':ss.Transition(src, [none, *on.values()], dur=ss.months(12), p=row if fn else [1 - p_use] + [p_use*m for m in mix]) for src in [none, *on.values()]}
        sim = ss.Sim(ss.Disease(none, *on.values(), **trs, b0=b0, first='none', name='contra'), solver=solver, start=2000, stop=2008, dt=ss.months(1), use_aging=False, verbose=0, **kw)
        sim.stratify(age=[0, 30])
        sim.stratify(urban=[False, True], split=[0.4, 0.6])
        return sim
    ode = make('ode', total_pop=20_000).run()
    abm = make('abm', n_agents=20_000).run()
    n = lambda sim: np.array([sim.results.contra[m][-1] for m in ['none'] + methods])
    young = ode.groups.strats[0].split[0] # Everyone re-decides each year from the same row, so the shares are those of the row
    use = 0.4*expit(-0.5) + 0.6*expit(0.5)
    expected = 20_000*np.array([1 - use] + list(use*(young*np.array([0.6, 0.6, 0.1])/1.3 + (1 - young)*np.array([0.2, 1.0, 0.3])/1.5)))
    assert np.allclose(n(ode), expected, rtol=0.01) and np.allclose(n(abm), expected, rtol=0.05)
    assert np.allclose(n(make('abm', fn=True, n_agents=20_000).run()), 20_000*np.array([0.5, 0.2, 0.2, 0.1]), rtol=0.06)
    assert ss.expressions.format_expr(abm.diseases.contra.graph.transitions[0].p[1], exact=True) == 'expit(b0 + (1.0 * people.urban)) * mix_pill'
    return abm


def test_contacts_dims():
    sc.heading('Testing one contact matrix over several stratifications together')
    sex, risk = np.array([[0.1, 1.0], [1.0, 0.1]]), np.array([[1.0, 0.5], [0.5, 3.0]])
    def run(net, solver='ode', seed=1):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.1) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
        sim = ss.Sim(sir, networks=net, stratify=[dict(sex=['female', 'male']), dict(risk=['low', 'high'], split=[0.8, 0.2])], total_pop=20_000, n_agents=20_000, solver=solver, unit='day', stop=120, rand_seed=seed, **kw)
        return sim.run().results.sir.R.values

    # A separable matrix given whole is the product of one for each
    for solver in ['ode', 'abm']:
        assert np.allclose(run(ss.contacts(dict(sex=sex, risk=risk)), solver), run(ss.contacts(np.kron(sex, risk), dims=['sex', 'risk']), solver))

    # One that is not: men mix with women, and the high-risk with each other more among men; as (2, 2, 2, 2) or (4, 4), first stratification slowest
    m = np.zeros((2, 2, 2, 2))
    m[0, :, 1, :] = [[0.8, 0.2], [0.5, 1.5]] # Women's contacts with men
    m[1, :, 0, :] = [[0.8, 0.5], [0.2, 2.0]] # Men's contacts with women
    ode = run(ss.contacts(m, dims=['sex', 'risk']))
    assert np.allclose(ode, run(ss.contacts(m.reshape(4, 4), dims=['sex', 'risk'])))
    abm = np.mean([run(ss.contacts(m, dims=['sex', 'risk']), 'abm', seed) for seed in range(3)], axis=0)
    assert np.isclose(abm[-1], ode[-1], rtol=0.1), (abm[-1], ode[-1]) # As with any contact matrix (about 7% apart for the product form here too)
    with pytest.raises(ValueError, match='risk'):
        run(ss.contacts(np.ones((6, 6)), dims=['sex', 'risk']))
    return


def test_contacts_ages():
    sc.heading('Testing a contact matrix with its own age groups: beside another dimension, its notes, and its results by group')
    C, area = [[10, 4, 1], [4, 8, 2], [1, 2, 3]], [[0.9, 0.1], [0.2, 0.8]]
    def make(stratify=True):
        S, E, I, R = ss.states('S E I R')
        m = ss.Disease(S >> E @ 0.14*I, E >> I @ ss.days(10), I >> R @ ss.days(8), init_n=dict(I=10), name='measles')
        if stratify:
            m.stratify(age={0: 0.3, 15: 0.6, 65: 0.1})
            m.stratify(area={'urban': 0.7, 'rural': 0.3})
        return m
    pars = dict(total_pop=100_000, n_agents=10_000, stop=ss.days(30), rand_seed=1, **kw)
    for solver in ['abm', 'ode']: # ages= combines with another dimension's matrix as age= does (it was "unrecognized argument area")
        sims = [ss.Sim(make(), networks=net, solver=solver, **pars).run() for net in [ss.contacts(dict(age=C, area=area)), ss.contacts(dict(age=C, area=area), ages=[0, 15, 65])]]
        assert np.allclose(*[sim.results.measles.infection.values for sim in sims])
        assert sum('not reciprocal' in note and '"age"' in note for note in sims[1].notes) == 1 # Checked over its own groups, once, as agents too (it was never checked for ages=, and then only in a compartmental run)
    res = sims[0].results.measles.infection
    assert np.allclose(res.by('age').select(area='rural').values, res.select(area='rural').by('age').values) # In either order (it was "is not stratified")
    assert 'contact matrix by age and area' in sims[0].describe(output=True) # It was "by age"

    # Over other age groups than the model's: one note for all the layers, saying "split" when the model has more groups
    sim = ss.Sim(make(False), networks=ss.contacts(ages=[0, 15, 65], matrix=C), stratify=dict(age=[0, 5, 18, 65]), solver='ode', **pars).init()
    assert any('split from 3 age groups into' in note for note in sim.notes)
    assert 'sim.diseases.measles.equations()' in make().equations(output=True) # The disease as written knows nothing of the sim's contacts, and says where they are (it was silent)
    assert 'C[i, j]' in ss.Sim(make(), networks=ss.contacts(ages=[0, 15, 65], matrix=C), solver='ode', **pars).init().diseases.measles.equations(output=True)
    return



def test_by_place_times_local_count():
    """ A value by place times a count in the same place (a vector's bites per person), as agents and as an ODE, and 'patch' in people """
    out = {}
    for solver in ['abm', 'ode']:
        S, I = ss.states('S I')
        V = ss.State(people=False)
        f = ss.par(ss.by(patch=[0.2, 0.8]), name='f')
        sis = ss.Disease(infection=ss.Transition(S, I, rate=f*ss.perday(0.1)*V.n/(S + I).n, acquisition=True), recovery=I >> S @ ss.days(20),
                         hatch=ss.null >> V @ ss.perday(1)*(S + I).n, death=V >> ss.null @ ss.days(1), init=dict(I=0.0, V=ss.by(patch=[500, 500])), name='sis')
        sis.stratify(patch=['a', 'b'], kind='space')
        pop = dict(n_agents=1000) if solver == 'abm' else dict(total_pop=1000)
        sim = ss.Sim(sis, solver=solver, unit='day', stop=60, verbose=0, **pop).run()
        out[solver] = sim.results.sis.I.by('patch')[-1]
        if solver == 'abm':
            assert 'patch' in sim.people and 'nonexistent' not in sim.people
    for solver, inf in out.items():
        assert inf[1] > 2*inf[0], f'[{solver}] Expected about four times the infections where f is four times as high: {inf}'
    return out

def test_by_series():
    """ A value by group that varies in time, for every step at once (as a whole run tabulates it): the same as a step at a time """
    S, I = ss.states('S I')
    season = lambda p: ss.timeseries({ss.years(k/4): ss.perday(0.2 + 0.05*k + p) for k in range(5)}, mode='linear', period=ss.years(1))
    beta = ss.by(patch=[season(0), season(0.1), ss.perday(0.3)], age=[1.0, ss.seasonal(mean=1.5, amplitude=0.2, peak=0.3)])
    sis = ss.Disease(S >> I @ beta*I.frac, I >> S @ ss.days(10), init=dict(I=0.01), name='sis')
    sis.stratify(patch=['a', 'b', 'c'], age=[0, 15])
    sim = ss.Sim(sis, total_pop=1000, unit='day', start='2000-01-01', stop='2001-01-01', verbose=0).init()
    eng = sim.diseases.sis.engine
    years = np.linspace(2000, 2001.5, 17)
    each = np.array([beta.per_group(sim, eng.convert, year=year) for year in years])
    assert np.array_equal(beta.series(years, sim, eng.convert), each), 'All at once should give the values of each year'
    table = ss.by(dict(age=[0, 15], value=[season(0), 2.0])) # A table: taken a year at a time
    assert np.array_equal(table.series(years, sim, eng.convert), np.array([table.per_group(sim, eng.convert, year=year) for year in years]))
    return each


def test_by_function():
    """ MS2: ss.by(patch=fn), with fn(sim) one value per level, is read by every solver, also with another dimension added by a held product """
    def eir(sim):
        return ss.perday(np.array([0.001, 0.01, 0.05]))
    expected = np.array([0.001, 0.01, 0.05])/(np.array([0.001, 0.01, 0.05]) + 0.05) # At equilibrium
    for solver in ['abm', 'ode']:
        for vaccine in [False, True]:
            S, I = ss.states('S I')
            m = ss.Disease(S >> I @ ss.by(patch=eir), I >> S @ ss.days(20), name='m')
            m.stratify(patch=ss.geo.geography(names=['a', 'b', 'c'], pop=[1, 1, 1]))
            intv = [ss.vaccinate(ss.Vaccine(efficacy=dict(acquisition=0.0)), coverage=0.5, at=2000.1)] if vaccine else []
            sim = ss.Sim(diseases=m, interventions=intv, n_agents=15_000, start=2000, dur=ss.days(200), dt=ss.days(1), verbose=0, solver=solver, rand_seed=1).run()
            I = sim.results.m.I.by('patch').values[-2]
            N = I + sim.results.m.S.by('patch').values[-2]
            assert np.allclose(I/N, expected, atol=0.03), (solver, vaccine, I/N)
    S, I = ss.states('S I')
    m = ss.Disease(S >> I @ ss.by(patch=lambda sim: ss.perday(np.ones(2))), name='m')
    m.stratify(patch=ss.geo.geography(names=['a', 'b', 'c']))
    with pytest.raises(ValueError, match='3 levels'):
        ss.Sim(diseases=m, n_agents=300, dur=ss.days(3), dt=ss.days(1), verbose=0).run()
    inflow = lambda rate, S=ss.State('S'): ss.Sim(diseases=ss.Disease(S, arrive=ss.null >> S @ rate, name='m').stratify(pop={'a': 0.2, 'b': 0.8}), solver='ode', total_pop=100, stop=2002, verbose=0).run().results.m.arrive.by('pop')[1]
    assert np.allclose(inflow(ss.by(pop=lambda sim: np.array([10., 5.]))), inflow(ss.by(pop=[10., 5.]))) # An inflow by level is the inflow at each level, not shared out again
    return sim


if __name__ == '__main__':
    mixing  = test_mixing()
    states  = test_states()
    strains = test_strains()
    carriage = test_carriage()
    emerge  = test_strain_emergence()
    imported = test_strain_import()
    counts  = test_strain_counts()
    ssolvers = test_strain_solvers()
    space   = test_space()
    unnamed = test_unnamed_data()
    ragged  = test_ragged()
    props   = test_properties_in_rates()
    by_props = test_by_properties()
    counters = test_counters()
    rows    = test_rows()
    cdims   = test_contacts_dims()
    place_counts = test_by_place_times_local_count()
    by_series = test_by_series()
    by_func = test_by_function()
