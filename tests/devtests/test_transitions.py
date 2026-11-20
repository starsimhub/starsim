"""
Test models written as declared states and transitions, the solvers that run them, and treatments that move people along transitions with a rate of 0
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import corpus
from corpus import make_sir, check_close

kw = dict(verbose=0)


def test_counts():
    sc.heading('Testing states that are not people')
    sim = corpus.cholera(total_pop=100_000).run()
    assert isinstance(sim.solver, ss.solvers.ode) and [s.name for s in sim.diseases.cholera.graph.count_states] == ['W'] # A count never makes a model agent-based
    res = sim.results.cholera
    assert np.allclose(res.S + res.I + res.R, 100_000) and res.W.max() > 1e6 # W is not part of the population
    agents = corpus.cholera(total_pop=100_000, n_agents=5_000).run() # The people are agents and the water stays a count
    assert isinstance(agents.solver, ss.solvers.abm) and np.isclose(agents.results.cholera.W.max(), res.W.max(), rtol=0.2)

    # A fixed delay with a competing exit (a mosquito's incubation, and its mortality), which acts over the whole delay in the ODE as in the chain binomial
    def eip():
        Sh, Ih = ss.states(2)
        Sv, Ev, Iv = ss.states(3, people=False)
        return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-9)*Iv.n/(Sh + Ih).n, emergence=ss.unborn >> Sv @ ss.perday(5000), biting=Sv >> Ev @ ss.perday(0.05),
                          incubation=Ev >> Iv @ ss.constant(ss.days(12)), mortality=Sv + Ev + Iv >> ss.dead @ ss.perday(0.1), init=dict(Sh=1.0, Sv=33_333), default_state='Sh', name='eip')
    for solver in ['ode', 'binomial']:
        Iv = ss.Sim(eip(), solver=solver, total_pop=2000, unit='day', stop=300, **kw).run().results.eip.Iv[-1]
        assert np.isclose(Iv, 5000*0.05/0.15*np.exp(-1.2)/0.1, rtol=0.002), f'{solver}: {Iv:.0f} infectious mosquitoes, not 5,020'



def test_births():
    sc.heading('Testing births, deaths, and migration, as compartments and as agents')
    # Births and deaths of the population, in demographics=: the same people are born into and die from every disease, under every solver
    def flu():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(2) * I), recovery=ss.Transition(I, R, rate=ss.peryear(1)), init_prev=0.05, name='flu')
    sis = lambda: ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.expon(ss.years(2)), name='sis')
    demog = [ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))]
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        sim = ss.Sim(flu(), sis(), demographics=demog, solver=solver, n_agents=2000, total_pop=2000, stop=20, **kw).run()
        n_flu = sum(sim.results.flu[s][-1] for s in 'SIR')
        assert np.isclose(n_flu, sum(sim.results.sis[s][-1] for s in 'SI')) and np.isclose(n_flu, 2000*np.exp(0.03*20), rtol=0.06), solver
    assert isinstance(ss.Sim(flu(), demographics=demog, **kw).init().solver, ss.solvers.ode)
    with pytest.raises(ValueError): # Births in two diseases would count people twice
        ss.Sim(ss.SIR(name='a', births=ss.birth(rate=ss.peryear(0.02))), ss.SIS(name='b', births=ss.birth(rate=ss.peryear(0.02))), **kw).init()

    # Migration: immigrants arrive at any age (and age group), and emigrants leave without dying
    S, I = ss.states(2)
    mig = ss.Disease(S, I, nothing=S >> I @ 0*S.n, arrive=ss.immigration(S, n=ss.peryear(500)), leave=ss.emigration(rate=ss.peryear(0.05)), name='mig').stratify(age=[0, 15, 50])
    sim = ss.Sim(mig, n_agents=10_000, total_pop=10_000, start=2000, stop=2010, **kw).run()
    res = sim.results
    assert 'death' not in res and res.emigration.sum() == res.mig.leave.sum() > 0 and np.isclose(res.alive[-1], 10_000, rtol=0.05)
    assert sim.people.age.mean() > 25 and res.mig.S.by('age').values[-1, -1] > 2000 # Not newborns

    # An Euler step whose outflows would take more than everyone takes everyone, and no more
    S, I = ss.states(2)
    sim = ss.Sim(ss.Disease(fast=S >> I @ 5, init_n=dict(S=100), name='fast'), total_pop=100, start=2000, stop=2002, solver=ss.solvers.ode(integrator='euler', substeps=1), **kw)
    assert np.allclose(sim.run().results.fast.I, [0, 100, 100])


def test_stratify():
    sc.heading('Testing a mixing pool against a random network')
    finals = sc.obj()
    for key,pars in dict(pool_ode=dict(networks=ss.MixingPool(n_contacts=10)), pool_abm=dict(networks=ss.MixingPool(n_contacts=10), n_agents=20_000),
                         net_abm=dict(networks=ss.RandomNet(n_contacts=10), n_agents=20_000), typed_ode=dict(networks=ss.MixingPool(n_contacts=ss.perweek(70)))).items():
        finals[key] = ss.Sim(make_sir(beta=0.03), total_pop=100_000, unit='day', stop=150, **pars, **kw).run().results.sir.infection.cum[-1]
    assert np.allclose(finals[:], finals[0], rtol=0.03), finals
    with pytest.raises(TypeError): # beta and n_contacts both per unit time
        S, I = ss.states('S I')
        ss.Sim(ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.03) * I), init_prev=0.01), networks=ss.MixingPool(n_contacts=ss.perday(10)), **kw).init()


def test_outcomes():
    sc.heading('Testing outcomes')
    def make(beta):
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        cases  = ss.outcome(0.5 * infection, delay=ss.days(5))
        hosp   = ss.outcome(0.05 * infection, delay=ss.days(7), dur=ss.days(10))
        deaths = ss.outcome(0.2 * hosp, delay=ss.lognormal(mean=14, std=5, unit='days'))
        return ss.Disease(infection, I >> R @ ss.days(10), cases, hosp, deaths, prevalence=I/(S + I + R), init_prev=0.01, name='sir')
    pars = dict(total_pop=1e6, start='2020-01-01', stop='2020-06-01', **kw)
    res = ss.Sim(make(0.3), **pars).run().results.sir
    total = res.infection.sum()
    assert np.isclose(res.cases.sum(), 0.5*total, rtol=0.01) and np.isclose(res.deaths.sum(), 0.01*total, rtol=0.02) # Outcomes chain
    assert np.argmax(res.hosp) == np.argmax(res.infection) + 7 and res.n_hosp.max() > res.hosp.max() # The delay, and the number in hospital
    assert np.isclose(res.prevalence.max(), res.I.max()/1e6)

    # Outcomes are stratified like the flows they are counted from, in the kernels (whole and stepped) and as agents
    def by_patch():
        S, I, R = ss.states(3)
        infection = S >> I @ 0.3 * I
        sir = ss.Disease(infection, I >> R @ 0.1, ss.outcome(0.5 * infection, name='cases', delay=ss.days(3)), ss.outcome(infection, name='hosp', dur=ss.days(5)), init_n=dict(I=ss.by(patch=[5, 0])), name='sir')
        sir.stratify(patch=ss.levels(['a', 'b'], mobility=ss.geo.Mobility(matrix=[[0, 0.01], [0.01, 0]], kind='migrate')))
        return ss.Sim(sir, total_pop=1000, unit='day', stop=80, **kw)
    kernel, stepped = by_patch().run().results.sir, by_patch().perf_profile(kind='loop', plot=False).sim.results.sir # Recording the time of each step needs the loop
    agents = by_patch().new(n_agents=1000).run().results.sir
    for res in [kernel, stepped, agents]:
        assert np.allclose(res.cases.select(patch='a') + res.cases.select(patch='b'), res.cases) and np.allclose(res.n_hosp.by('patch').values.sum(axis=1), res.n_hosp)
    assert np.allclose(kernel.cases.select(patch='b'), stepped.cases.select(patch='b'))
    assert np.isclose(agents.cases.select(patch='b').sum(), kernel.cases.select(patch='b').sum(), rtol=0.3)



def test_modify():
    sc.heading('Testing modifiers on dwell times and branches')
    # As agents, a modifier on a dwell time, and a dwell time that changes in time, reach the people already waiting: a cohort
    def cohort(solver, dur, mods=None):
        I, R = ss.states('I R')
        dis = ss.Disease(recovery=I >> R @ dur, default_state='I', name='sir')
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        return ss.Sim(dis, interventions=mods(dis) if mods else None, unit='day', dt=0.1, stop=40, solver=solver, rand_seed=1, verbose=0, **size).run().results.sir.I.values
    shorter = lambda dis: ss.modify(dis.pars.dur_I, multiply=0.5, when=ss.schedule(start=20))
    for dur, mods, i in [(ss.days(10), shorter, 250), (ss.timeseries({0: ss.days(20), 10: ss.days(2)}), None, 150)]:
        assert np.isclose(cohort('abm', dur, mods)[i], cohort('ode', dur, mods)[i], rtol=0.1)

    # A multiplier by level on a branch chosen with p= scales its probability, under every solver
    def branch_frac(solver, modify=False):
        S, D, R = ss.states('S D R')
        d = ss.Disease(S, D, R, onset=S >> D @ ss.peryear(2), death=ss.Transition(D, ss.dead, dur=ss.days(10), p=0.1), recovery=ss.Transition(D, R, dur=ss.days(10), p=None), back=R >> S @ ss.days(5), name='d')
        d.stratify(type=ss.levels(['watery', 'bloody'], states=[D]))
        con = ss.modify(d.death, multiply=5, who=d.type == 'bloody') if modify else ss.modify(d.death, multiply=ss.by(type=[1, 5]))
        res = ss.Sim(d, connectors=con, n_agents=20_000, total_pop=20_000, solver=solver, dt=ss.days(1), start=2000, stop=2001, **kw).run().results.d
        return res.death.by('type').values.sum(0)/res.onset.by('type').values.sum(0)
    for solver in ['ode', 'abm']:
        for frac in [branch_frac(solver), branch_frac(solver, modify=True)]:
            assert np.allclose(frac, [0.1, 0.5], atol=0.02), f'{solver}: {frac}'


def test_functions():
    sc.heading('Testing functions and code in the model')
    severity = lambda sim: 0.1 + 0.3*(sim.people.age > 50) # Works on agents and on age groups
    def make():
        S, I, H, R = ss.states('S I H R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), course=ss.Transition(I, {H: severity, R: None}, dur=7), discharge=ss.Transition(H, R, rate=0.1), init_prev=0.01, name='d')
    with pytest.raises(ss.CapabilityError, match='people.age'): # With no age groups, the function cannot run as compartments, which it says
        ss.Sim(make(), stop=100, unit='day', **kw).run()
    by_age = ss.Sim(make(), stop=100, unit='day', stratify=dict(age=[0, 50]), **kw).run()
    assert isinstance(by_age.solver, ss.solvers.ode) and by_age.results.d.H.select(age='50+').max() > by_age.results.d.H.select(age='0-49').max()
    S, I, R = ss.states('S I R') # A function that takes uids only works on agents
    d2 = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=lambda sim, uids: 0.2/(1 + (sim.people.age[uids] > 50))), init_prev=0.01, name='d2')
    sim = ss.Sim(d2, stop=100, unit='day', **kw).run()
    assert isinstance(sim.solver, ss.solvers.abm) and 'takes uids' in sim.describe()



def test_time_values():
    sc.heading('Testing values that vary in time, and check_dt()')
    def make(beta):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.01, name='sis')
    # A seasonal value peaks once a year, under every solver; the ODE takes it at the time of each stage of a step
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        n_I = ss.Sim(make(ss.seasonal(mean=ss.perday(0.3), amplitude=0.5, peak='January')), start=2020, stop=2023, dt=ss.days(1), solver=solver, total_pop=20_000, **kw).run().results.sis.I[366:731]
        assert (np.argmax(n_I) < 45 or np.argmax(n_I) > 320) and n_I.max() > 1.5*n_I.min(), solver
    for solver in ['ode', 'binomial']: # A branch whose rate varies in time is a wait at that rate, with its flow recorded
        I, H, R = ss.states('I H R')
        severe = ss.Transition(I, H, p=0.25, rate=ss.par(0.2778, name='rate_severe')*ss.seasonal(mean=1.0, amplitude=0.3, peak=0.5))
        sir = ss.Disease(severe=severe, mild=ss.Transition(I, R, p=0.75, rate=0.0909), default_state='I', name='sir')
        res = ss.Sim(sir, total_pop=10_000, stop=60, dt=0.1, unit='day', solver=solver, verbose=0).run().results.sir
        assert res.H[0] == 0 and np.isclose(res.severe.cum[-1], res.H[-1], rtol=1e-3), solver

    fast = make_sir(beta=0.6, dur=4, name='fast') # check_dt() says how much the step matters: for whole steps, and not for an ODE
    df = ss.Sim(fast, solver=ss.solvers.binomial(substeps=1), unit='day', stop=60, total_pop=1e6, **kw).check_dt().table
    assert 0.05 < df.change[df.result == 'I'].iloc[0] < 0.2
    assert ss.Sim(fast, solver='ode', unit='day', stop=60, total_pop=1e6, **kw).check_dt().table.change.max() < 1e-3


def test_stays_and_offspring():
    sc.heading('Testing a rate beside a dwell of whole steps, and litters (ss.birth(when=)), as EMULSION has them')
    # A rate competing with a dwell of whole steps acts in each of them
    S, X, A, B = ss.states('S X A B')
    d = ss.Disease(S, X, A, B, enter=ss.Transition(S, X, rate=ss.peryear(0.5)), timed=ss.Transition(X, A, dur=ss.constant(ss.months(1))), hazard=ss.Transition(X, B, rate=ss.peryear(1.0)), name='d')
    r = ss.Sim(d, n_agents=20_000, start=2000, stop=2010, dt=ss.months(1), verbose=0).run().results.d
    assert np.isclose(r.hazard.values.sum()/(r.hazard.values.sum() + r.timed.values.sum()), 1 - np.exp(-1/12), rtol=0.1)

    # A litter is a birth driven by the transition
    G, L, NB = ss.states(3)
    farrowing = ss.Transition(G, L, dur=ss.days(10))
    herd = ss.Disease(G, L, NB, farrowing=farrowing, weaning=ss.Transition(L, G, dur=ss.days(20)), litter=ss.birth(NB, when=farrowing, children=3), default_state='G', name='herd')
    res = ss.Sim(herd, unit='day', stop=60, n_agents=100, verbose=0).run().results.herd
    assert res.NB[-1] == 3*res.farrowing.values[:-1].sum() > 0
    res = ss.Sim(herd, unit='day', stop=60, solver='ode', verbose=0).run().results.herd
    assert np.isclose(res.NB[-1], 3*res.farrowing.values[:-1].sum())


def test_branches():
    sc.heading('Testing branches by level, and over 1')
    # A branch probability over 1 raises; branches kept to one level each by move= sum to 1 within that level
    for solver in ['abm', 'ode']:
        S, D, R = ss.states('S D R')
        d = ss.Disease(S, D, R, onset=S >> D @ ss.peryear(2), death=ss.Transition(D, ss.dead, dur=ss.days(5), p=ss.by(type=[0.1, 1.5])), recovery=ss.Transition(D, R, dur=ss.days(5), p=None), name='d')
        d.stratify(type=ss.levels(['watery', 'bloody'], states=[D]))
        with pytest.raises(ValueError):
            ss.Sim(d, n_agents=1000, total_pop=1000, solver=solver, start=2000, stop=2001, **kw).run()
    A, B, C = ss.states('A B C')
    trs = {f'{dst.name.lower()}{k}': ss.Transition(A, dst, p=p, dur=ss.months(1), move=dict(k=(k, '1'))) for dst,p in [(B, 0.9), (C, 0.1)] for k in '01'}
    d = ss.Disease(A, B, C, **trs, default_state='A', name='d').stratify(k=['0', '1'])
    res = ss.Sim(d, n_agents=2000, start=2000, stop=2001, dt=ss.months(1), unit='year', **kw).run().results.d
    assert res.A[-1] == 0 and np.isclose(res.C[-1]/2000, 0.1, atol=0.03)


def test_expressions():
    sc.heading('Testing expressions in rates: conditions with ss.by')
    # Arithmetic on a condition and an ss.by, in either order
    age = ss.by(age=[0.1, 0.3])
    older = ss.person.age >= 50
    forms = {'cond + by': (lambda: 0.2*older + age, [0.1, 0.5]), 'by * cond': (lambda: age * (1 + older), [0.1, 0.6])} # The rates under 50, then 50 and over
    pars = dict(start=2000, stop=2002, dt=ss.months(1), verbose=0)
    for name,(form,rates) in forms.items():
        A, B = ss.states('A B')
        d = ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*form(), default_state='A', name='c').stratify(age=[0, 50])
        for solver,size in [('ode', dict(total_pop=10_000)), ('abm', dict(n_agents=10_000))]:
            res = ss.Sim(d, solver=solver, **size, **pars).run().results.c
            moved = res.go.by('age').values[:-1].sum(axis=0)/res.A.by('age').values[0]
            assert np.allclose(moved, 1 - np.exp(-2*np.array(rates)), rtol=0.02 if solver == 'ode' else 0.1), f'{name} ({solver}): {moved}'
    A, B = ss.states('A B')
    with pytest.raises(ValueError): # An unnamed ss.par inside an expression is an error when the model is built
        ss.Disease(A, B, go=A >> B @ ss.peryear(1.0)*(ss.par(age) + 0.2*older), default_state='A', name='c')



def test_modify_dist_dwell():
    sc.heading('Testing modifiers on a dwell-time parameter that is a distribution')
    def make(raw=ss.constant(ss.years(1)), mod=None, solver='abm'):
        A, B = ss.states('A B')
        dis = ss.Disease(A, B, go=ss.Transition(A, B, rate=ss.peryear(2)), back=ss.Transition(B, A, dur=ss.par(raw, name='d')), default_state='A', name='c')
        size = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        return ss.Sim(dis, interventions=mod, start=2000, stop=2006, dt=ss.years(0.25), solver=solver, rand_seed=1, **size, **kw)
    def dwells(sim): # Of those in B, by whether they entered after the start of the modifier
        c = sim.diseases.c
        entered, left = c.ti_B.raw.astype(float), c.ti_exit_B.raw.astype(float)
        dwell = (left - entered)*sim.t.dt.value
        inB = c.B.raw & sim.people.alive.raw
        return dwell[inB & (entered >= 4)], dwell[inB & (entered < 4)]
    for kwargs in [dict(set=ss.constant(ss.years(5))), dict(multiply=5), dict(add=ss.years(4))]:
        after, _ = dwells(make(mod=ss.modify('c.pars.d', when=ss.schedule(start='2001'), **kwargs)).run())
        assert len(after) > 100 and np.allclose(after, 5, atol=0.25), kwargs
    _, before = dwells(make().run())
    assert np.allclose(before, 1, atol=0.25)
    with pytest.raises(ss.CapabilityError): # As compartments, a distribution laid out at the start cannot change
        make(mod=ss.modify('c.pars.d', multiply=5, when=ss.schedule(start='2001')), solver='binomial').init()


def test_algebra():
    """ Each kind of node in an expression, under each operator, in each solver: it runs """
    sc.heading('Testing the expression algebra over the solvers...')
    solvers = ['abm', 'ode', 'binomial']
    def run(solver, *diseases, stratify=dict(age=[0, 15, 50], sex=['female', 'male']), **kwargs):
        sim = ss.Sim(diseases=list(diseases), solver=solver, n_agents=200, total_pop=200, start=2000, stop=2002, dt=ss.months(2), rand_seed=1, verbose=0, **kwargs)
        sim.stratify(**stratify)
        return sim.run()
    def check(label, solver, func):
        sim = func(solver)
        assert all(np.all(np.isfinite(res.values)) for res in sim.results.flatten().values() if isinstance(res, ss.Result)), f'{label} ({solver}): not finite'

    # Each kind of node, as a factor in a rate, under each operator with an ss.by table
    table = lambda: ss.by(dict(age=[0, 15, 50], value=[0.8, 1.0, 1.2]))
    nodes = dict(
        number   = lambda I: ss.par(0.9, name='k'),
        table    = lambda I: ss.by(dict(age=[0, 15, 50], value=[0.5, 1.0, 0.8])),
        count    = lambda I: 1 + I.frac,
        person   = lambda I: ss.person.age/50,
        attr     = lambda I: ss.attr(ss.uniform(0.5, 1.5), name='a'),
        seasonal = lambda I: ss.seasonal(amplitude=0.3),
    )
    ops = {
        '+': lambda x: x + table(),
        '*': lambda x: table()*x,
        'minimum': lambda x: np.minimum(x, table()),
        'exp': lambda x: np.exp(-x),
    }
    def node_model(node, op):
        def make(solver):
            S, I = ss.states('S I')
            return run(solver, ss.Disease(infection=S >> I @ ss.peryear(2)*op(node(I))*I, recovery=I >> S @ ss.years(0.5), init=dict(I=0.1), name='sis'))
        return make
    for nname,node in nodes.items():
        for oname,op in ops.items():
            for solver in solvers:
                check(f'{nname} {oname}', solver, node_model(node, op)) # A trait (an attribute drawn once) at its mean under a compartmental solver



def test_certain_and_substeps():
    sc.heading('Testing a probability of 1 under every solver, and the chain binomial\'s default substeps near threshold')
    S, I, A, B, D = ss.states('S I A B D')
    def make(): # Everyone moves in the step, shared equally between two certain exits; a finite rate competing with them takes nobody
        return ss.Disease(S, I, A, B, D, infection=S >> I @ ss.peryear(0.5), toa=I >> A @ ss.probperyear(1), tob=I >> B @ ss.probperyear(1)*0.5, die=I >> D @ ss.peryear(2), init=dict(I=0.5), name='d')
    for solver in ['abm', 'ode', 'binomial', 'tau', 'gillespie']:
        res = ss.Sim(make(), n_agents=2000, total_pop=2000, solver=solver, start=2000, stop=2001, dt=ss.years(0.1), **kw).run().results.d
        first = res.I[0]
        assert res.D[1] <= 0.01*first, solver
        assert np.isclose(res.A[1] + res.B[1], first + (res.S[0] - res.S[1]), rtol=0.05) and np.isclose(res.A[1], res.B[1], rtol=0.15), solver
    def sis(): # Just above threshold: the default substeps follow the ODE
        S, I = ss.states('S I')
        return ss.Disease(S, I, infection=S >> I @ 1.05*I.frac, recovery=I >> S @ 1.0, init=dict(I=0.1), name='d')
    pars = dict(unit='year', dt=ss.months(1), start=0, stop=50, total_pop=1e9, **kw)
    ode = ss.Sim(sis(), solver='ode', **pars).run().results.d.I[-1]
    sim = ss.Sim(sis(), solver='binomial', **pars).run()
    assert sim.diseases.d.engine.substeps() >= 15 and np.isclose(sim.results.d.I[-1], ode, rtol=0.08)


#%% Treatments

def run_both(make, stop=200, pop=20_000):
    """ Run the sim that `make` returns as an ODE and as agents """
    return sc.obj(ode=make(solver='ode', total_pop=pop, stop=stop, unit='day', **kw).run(), abm=make(solver='abm', n_agents=pop, stop=stop, unit='day', **kw).run())


def make_sti(name='ng', beta=0.015, fail=None):
    """ A bacterial STI as in STIsim, with treatment as an undriven transition and no treated state; `fail` is the event of a treatment failing """
    S, E, A, P, Y, C = ss.states('S E A P Y C')
    trs = dict(
        infection = S >> E @ beta*(A + P + Y + C),
        latency   = ss.Transition(E, {P: 0.4, A: None}, dur=ss.days(5)),
        onset     = P >> Y @ ss.days(10),
        seek      = Y >> C @ ss.days(20),
        clear_a   = A >> S @ ss.days(150),
        clear_y   = (Y + C) >> S @ ss.days(200),
        cure      = (E + A + P + Y + C) >> S @ 0, # Moved only by treatment
    )
    if fail is not None: # Failures lose their symptoms and stop seeking care
        trs['fail'] = ss.Transition(P + Y + C, A, when=fail)
    return ss.Disease(**trs, init_prev=0.1, name=name)


def test_syndromic():
    sc.heading('Testing one treatment for two diseases tracked jointly: the same people are treated for both, and its outcomes are events')
    def make(**kwargs):
        kit = ss.Treatment(cure={'ng.cure': 0.95, 'ct.cure': 0.9}, name='kit')
        ng, ct = make_sti('ng', beta=0.015, fail=kit.failed), make_sti('ct', beta=0.012)
        syndromic = ss.treat(kit, who=ng.C | ct.C, when=ss.schedule(every=ss.days(30)), coverage=0.5) # No disease=: the product names them
        second = ss.treat(ss.Treatment(cure={'ng.cure': 1.0, 'ct.cure': 1.0}), when=syndromic.failed, name='second')
        cure_step = ss.Event(when=syndromic.cured, name='cure_step')
        return ss.Sim([ng, ct], interventions=[syndromic, second, cure_step], **kwargs)
    sims = run_both(make, pop=10_000)
    for solver,sim in sims.items():
        r = sim.results.syndromic
        for dis in ['ng', 'ct']:
            n = [r[key].select(disease=dis).sum() for key in ['cured', 'failed', 'unneeded']]
            assert np.isclose(sum(n), r.delivered.sum()), solver # Each disease sees every recipient: chosen once, before anyone moves
            assert np.isclose(n[0] + sim.results.second.cured.select(disease=dis).sum(), sim.results[dis].cure.sum()), solver
        assert r.unneeded.select(disease='ct').sum() > 0 and sim.results.cure_step.delivered.sum() > 0, solver # Treated for CT because of NG symptoms
        assert np.isclose(sim.results.second.delivered.sum(), r.failed.sum(), rtol=0.05) and sim.results.ng.fail.sum() > 0, solver
    check_close(sims, lambda sim: sim.results.ng.fail.sum(), rtol=0.2)
    check_close(sims, lambda sim: sim.results.syndromic.unneeded.select(disease='ct').sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.cure_step.delivered.sum(), rtol=0.15)
    assert sims.ode.joint_states


if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
