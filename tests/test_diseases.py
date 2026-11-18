"""
Run tests of disease models
"""

# %% Imports and settings
import pytest
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl

n_agents = 2_000


@sc.timer()
def test_sir():
    sc.heading('Testing SIR dynamics')

    # The textbook SIR: a duration is the mean of an exponential, and nobody dies (Starsim 3's lognormal and deaths are ssl.AgentSIR's)
    kw = dict(total_pop=1e6, dur=ss.days(365), solver='ode', verbose=0)
    def closed(sir):
        cl = ss.modify(sir.pars.beta, multiply=0.6, when=ss.schedule(start=ss.days(30), stop=ss.days(90)))
        return ss.Sim(sir, interventions=cl, **kw).run().results.sir
    a = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.days(7), init_n=dict(I=100)))
    b = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.expon(mean=ss.days(7)), init_n=dict(I=100)))
    assert abs(a.infection.sum() - b.infection.sum())/b.infection.sum() < 0.01 # 714,770 each; 785,847 with Starsim 3's lognormal
    assert a.death.sum() == 0
    return a


@sc.timer()
def test_seir():
    sc.heading('Testing SEIR dynamics')

    seir = ss.SEIR(dur_exp=ss.lognormal(mean=ss.days(1)), dur_inf=ss.lognormal(mean=ss.days(10)), beta=ss.perday(0.05))
    sim = ss.Sim(n_agents=n_agents, diseases=seir, networks='random', dur=ss.years(1), dt=ss.days(1), verbose=0).run()
    seir = sim.diseases.seir
    res = sim.results.seir
    assert not (seir.E & seir.I).any(), 'Agents should not be in both E and I'
    assert np.allclose(res.prevalence, (res.E + res.I)/sim.results.alive), 'prevalence should be computed from E plus I'
    assert res.infection.min() >= 0 and res.infection[0] == res.S[0] - res.S[1], 'Seed infections should be excluded from incidence exactly once' # The first step's infections, not the seeds
    assert np.array_equal(res.infection.cum, np.cumsum(res.infection)), 'Cumulative infections should be the running sum of new infections'
    return sim


@sc.timer()
def test_aliases():
    sc.heading('Testing state aliases (of a disease written as code)')

    # A string alias points at another state, and a subclass can replace it with a callable
    class IsolationSIR(ssl.AgentSIR):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(
                ss.BoolState('isolated', label='Isolated'),
                reset = 'infectious', # Remove the inherited string alias before replacing it
                infectious = lambda self: self.infected & ~self.isolated, # Isolated agents don't transmit
            )
            return

        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            self.isolated[uids] = True # Isolate everyone, so nobody transmits
            return

    sir = ssl.AgentSIR()
    assert sir.infectious is sir.infected, 'By default, infectious should be an alias of infected'

    sim = ss.Sim(n_agents=n_agents, diseases=IsolationSIR(init_prev=0.1, beta=ss.perday(0.5)), networks='random',
                 dur=ss.days(50), dt=ss.days(1), verbose=0).run()
    dis = sim.diseases.isolationsir
    res = sim.results.isolationsir
    assert (dis.infectious == (dis.infected & ~dis.isolated)).all(), 'The derived alias should combine the two states'
    assert res.infection.cum[-1] == 0, 'Isolated agents should not transmit'
    assert 'infectious' in res, 'A derived alias should generate an automatic result'
    assert res.infectious.max() == 0, 'Nobody is infectious, since everyone is isolated'

    return sim


@sc.timer()
def test_sis():
    sc.heading('Testing SIS dynamics')

    # The library SIS is declared, so it has an R0 and runs as an ODE when there is no network
    sis = ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.years(5))
    assert np.isclose(sis.R0(), 2.5) # beta × the mean duration
    ode = ss.Sim(sis, stop=100).run()
    assert isinstance(ode.solver, ss.solvers.ode) and np.isclose(ode.results.sis.prevalence[-1], 0.6, atol=0.01) # Endemic at 1 - 1/R0
    rt = ode.diseases.sis.Rt()
    assert np.isclose(rt[0], 2.5, rtol=0.01) and np.isclose(rt[-1], 1, atol=0.01) # R0 at the start, 1 at the endemic state

    # R over time is R0 × S/N in an SIR, and is lowered by a modifier while it acts
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.par(0.3, name='beta')*I, recovery=ss.Transition(I, R, dur=10), name='sir')
    half = ss.modify(sir.pars.beta, multiply=0.5, when=ss.schedule(start=60, stop=80))
    for solver in ['ode', 'abm']:
        run = ss.Sim(sir, interventions=half, total_pop=10_000, n_agents=2000, stop=100, solver=solver, verbose=0).run()
        rt, res = run.diseases.sir.Rt(), run.results.sir
        expected = 3*res.S/(res.S + res.I + res.R)*np.where((res.timevec >= 60) & (res.timevec < 80), 0.5, 1)
        assert np.allclose(rt, expected, rtol=1e-3), f'Expected R over time of R0 × S/N, halved by the modifier, as {solver}'

    # An error while working out the next-generation matrix leaves the compartments as they were
    eng = ss.Sim(ss.SIR(beta=0.1, init_prev=0.1), total_pop=1000, solver='binomial', verbose=0).init().diseases.sir.engine
    before, calls, rates = eng.y.copy(), [0], eng.rates
    def failing():
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError('Not a model error')
        return rates()
    eng.rates = failing
    with pytest.raises(RuntimeError):
        eng.next_generation()
    assert np.array_equal(eng.y, before) and eng.held_n is None
    return run


@sc.timer()
def test_ncd():
    sc.heading('Testing NCDs')

    sim = ss.Sim(n_agents=n_agents, diseases=ssl.NCD(), dt=ss.years(1), verbose=0).run()
    assert sim.results.ncd.affected[-1] > 0

    # A disease written as code that is not an infection (as Starsim 3's ss.Disease), and an infection with no beta
    class Anemia(ss.Disease):
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(p_onset=ss.bernoulli(0.1))
            self.update_pars(**kwargs)
            self.define_states(ss.BoolState('anemic'))

        def step_state(self):
            self.anemic[self.pars.p_onset.filter()] = True

    class Background(ss.AgentDisease): # No beta, so not transmitted
        pass

    sim2 = ss.Sim(diseases=[Anemia(), Background(init_prev=0.1)], networks='random', n_agents=500, verbose=0).run()
    anemia = sim2.diseases.anemia
    assert sim2.results.anemia.anemic[-1] > 400 # Anemia(p_onset=0.1) for 50 years
    assert sim2.results.background.infection.sum() == 0 and sim2.results.background.infected[0] > 0 # Seeded, not transmitted
    assert not hasattr(anemia, 'susceptible') # No states of an infection forced on it

    # An infection written as code with a beta but nothing to transmit it over is an error, not a run with no infections; with beta=0 it is not transmitted by design
    with pytest.raises(ValueError):
        ss.Sim(diseases=ssl.AgentSIR(), n_agents=100, verbose=0).run()
    ss.Sim(diseases=ssl.AgentSIR(beta=0), n_agents=100, verbose=0).run()
    return sim


@sc.timer()
def test_congenital():
    sc.heading('Testing the congenital outcomes of ssl.mnch.CongenitalDisease, through set_congenital(), a hook of ss.AgentDisease')

    sim = ss.Sim(
        n_agents=20_000, # Enough for a stable ratio of outcomes (about 30 of each)
        demographics=[
            ss.Pregnancy(fertility_rate=ss.peryear(0.03), burnin=True),
            ss.Deaths(death_rate=ss.peryear(0.009900990099)),
        ],
        diseases=ssl.mnch.CongenitalDisease(beta=0.2, init_prev=0.2),
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        verbose=0,
    )
    sim.run()
    sb = sim.demographics.pregnancy.results.stillbirth.sum()
    n_congenital = sim.diseases[0].congenital.sum()
    assert sb > 0 and n_congenital > 0 and sim.demographics.pregnancy.results.birth.sum() > 0

    # Stillbirths fire in utero, not after delivery: the outcomes are stillborn, congenital, or normal at 0.3:0.4:0.3, so if stillbirths were
    # scheduled at delivery, they would be logged as neonatal deaths instead, and this ratio would go towards zero
    ratio = sb/n_congenital
    assert 0.3 < ratio < 1.8, f'Stillbirth:congenital ratio {ratio:0.2f} is far from the expected 0.75'

    # The hook on any disease written as code: by default, set_prognoses(); overridden, only those infected before birth reach it
    class Prenatal(ssl.AgentSIR):
        def set_congenital(self, uids, sources=None):
            self.unborn_infected = getattr(self, 'unborn_infected', 0) + len(uids)
            assert (self.sim.people.age[uids] <= 0).all()
            return super().set_congenital(uids, sources)
    kw = dict(n_agents=5000, demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.05)), networks=ss.PrenatalNet(), dur=ss.years(2), dt=ss.years(1/12), verbose=0)
    pre = ss.Sim(diseases=Prenatal(beta=dict(prenatalnet=ss.peryear(5)), init_prev=0.3, dur_inf=ss.constant(ss.years(10)), name='pre'), **kw).run()
    assert pre.diseases.pre.unborn_infected > 0 and pre.diseases.pre.unborn_infected == pre.results.pre.infection.sum()
    return sim


@sc.timer()
def test_current_state():
    sc.heading('Testing current_state(): the index of the state each agent is in')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(name='sir', infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=4), init_prev=0) # Nobody seeded, so only the moves below
    sim = ss.Sim(diseases=sir, n_agents=100, stop=2001, verbose=0).init()
    sir = sim.diseases.sir
    sir.move(sir.I, uids=ss.uids([0, 1])) # From wherever they are
    sir.move(sir.R, uids=ss.uids([2]))
    assert sir.move(sir.S, sir.I, uids=ss.uids([1, 3, 4])) == 2 # Of these, only those in S
    ss.move(sir.I, sir.S, uids=ss.uids([3, 4]))
    cs = sir.current_state()
    assert isinstance(cs, ss.IntArr) and cs.labels == ['S', 'I', 'R']
    assert cs[ss.uids([0, 1, 2, 3])].tolist() == [1, 1, 2, 0]
    assert [cs.labels[i] for i in cs[ss.uids([0, 2])]] == ['I', 'R']
    sub = sir.current_state(['R', sir.I]) # Some of them, by name or as states
    assert sub.labels == ['R', 'I'] and sub[ss.uids([0, 2, 3])].tolist() == [1, 0, -1]
    assert sir.current_state(['R', sir.I], uids=ss.uids([3, 2, 0])).tolist() == [-1, 0, 1] # Some agents only, in the order given
    assert ss.move([sir.S, sir.R], sir.I, uids=ss.uids([2, 3, 5])) == 3 # From several states
    assert sir.current_state(uids=ss.uids([2, 3, 5])).tolist() == [1, 1, 1]
    sir.move(sir.R, uids=ss.uids([2])); sir.move(sir.S, uids=ss.uids([3, 5])) # Back as they were
    with pytest.raises(ValueError): # Read-only
        cs.raw[0] = 2
    with pytest.raises(KeyError, match='"sir" has no state X'):
        sir.current_state(['X'])
    res = sim.run().results.sir
    assert (res.I[0], res.R[0]) == (2, 1) # The start counts the agents moved before the first step

    # A count by group, under every solver
    for kw in [dict(n_agents=2000), dict(solver='ode')]:
        d = ss.Sim(diseases=ss.SIR(init_prev=0.2), stratify=dict(age=[0, 15, 50]), stop=2001, verbose=0, **kw).init().diseases.sir
        by_age = d.count('I', by='age')
        assert by_age.shape == (3,) and np.isclose(by_age.sum(), d.I.n) and np.isclose(d.count([d.I, d.R]), (d.I | d.R).n)
        assert np.isclose(d.count((ss.person.age < 15) & d.I, by=True)[0], by_age[0]) and d.count((ss.person.age < 15) & d.I, by='age')[1:].sum() == 0

    # Only the people for whom a condition holds (where=), under every solver
    for kw in [dict(n_agents=1000), dict(solver='ode')]:
        d = ss.Sim(diseases=ss.SIR(), stratify=dict(age=[0, 15]), stop=2001, verbose=0, **kw).init().diseases.sir
        young = ((ss.person.age < 15) & d.S).n
        assert np.isclose(d.move(d.S, d.R, who=ss.person.age < 15), young) and np.isclose(d.R.n, young)
    S, I, R = ss.states('S I R')
    ode = ss.Sim(diseases=ss.Disease(name='sir', infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=4)), stop=2001, verbose=0).init()
    with pytest.raises(ss.CapabilityError, match='agents'):
        ode.diseases.sir.current_state()
    return cs


@sc.timer()
def test_infect():
    sc.heading('Testing disease.infect(): infections from code are dated and logged as infections')
    class Import(ss.Intervention):
        def step(self):
            if self.ti == 3:
                self.infected = self.sim.diseases.sir.infect(self.sim.people.auids[:50])
    for sir in [ss.SIR(beta=0, init_prev=0), ssl.AgentSIR(name='sir', beta=0, init_prev=0)]: # Declared, and written as code
        sim = ss.Sim(n_agents=1000, diseases=sir, interventions=Import(), analyzers=ss.infection_log(), start=2000, stop=2008, verbose=0).run()
        infected = sim.interventions[0].infected
        log = sim.analyzers[0].to_df()
        ti, new = ('ti_I', 'infection') if sir.graph else ('ti_infected', 'infection')
        assert len(infected) == 50 and (sim.diseases.sir[ti][infected] == 3).all() and sim.results.sir[new][3] == 50
        assert len(log) == 50 and (log.source.to_numpy(dtype=float, na_value=np.nan) == -1).all() # Not NA, which (log.source == -1).all() skips
    return sim


def test_same_transmission():
    sc.heading('Testing that an SIR declared and one written as code transmit the same, by the same code: the same infections, sources, and networks, CRN included')
    seeds = lambda self, sim, uids: (uids % 50 == 0).astype(float) # Every 50th agent, so seeding takes no random numbers

    class CodedSIR(ss.AgentDisease): # Written as code, overriding only what it adds to ss.AgentDisease
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(beta=ss.perday(0.1), dur_inf=10, init_prev=ss.bernoulli(seeds))
            self.define_states(ss.BoolState('recovered'), ss.TimeArr('ti_recovered'))
            self.update_pars(**kwargs)
        def step_state(self):
            done = (self.infected & self.ti_recovered.due()).uids # At the start of the step after dur_inf, as a declared dwell time ends
            self.infected[done] = False
            self.recovered[done] = True
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            self.ti_recovered[uids] = self.ti + 1 + self.pars.dur_inf # Infected at the end of the step (the seeds at the start, step 0)

    kw = dict(n_agents=n_agents, stop=50, unit='day', verbose=0, analyzers=ss.infection_log(), networks=[ss.RandomNet(n_contacts=4), ss.RandomNet(n_contacts=2, name='other')],
              connectors=[ss.modify('sir.rel_sus', multiply=0.5, who=lambda sim: sim.people.age > 40), ss.modify('other.pars.n_contacts', multiply=0.5)],
              interventions=ss.vaccinate(ss.Vaccine(efficacy=0.7), when=10, coverage=0.3))
    dec = ss.Sim(diseases=ss.SIR(name='sir', beta=ss.perday(0.1), dur_inf=ss.constant(ss.days(10)), init_prev=lambda sim, uids: seeds(None, sim, uids)), solver='abm', **kw).run()
    cod = ss.Sim(diseases=CodedSIR(name='sir'), **kw).run()
    d, c = dec.results.sir, cod.results.sir
    assert d.infection.sum() > 500
    assert np.array_equal(d.infection, c.infection) and np.array_equal(d.I, c.infected) and np.array_equal(d.R, c.recovered)
    logs = [sim.analyzers[0].to_df().sort_values('target')[['target', 'source', 'network']].fillna('-').astype(str).values for sim in [dec, cod]]
    assert np.array_equal(*logs) # Who infected whom, and over which network
    return dec, cod


@sc.timer()
def test_mixed_disease():
    sc.heading('Testing a disease with declared and coded parts')
    class Mixed(ss.Disease):
        S, I, R   = ss.states('S I R')
        infection = S >> I @ ss.peryear(4) * I
        recovery  = I >> R @ ss.years(0.5)

        def __init__(self, **kwargs):
            super().__init__(init_prev=0.05, **kwargs)
            self.define_states(ss.FloatArr('years_infected', default=0.0)) # A coded per-agent array beside the declared states
            self.define_pars(rate=1.0) # A parameter that only its code reads

        def step(self):
            super().step() # The declared transitions
            self.years_infected[self.I.uids] += self.t.dt_year

    sim = ss.Sim(diseases=Mixed(), n_agents=500, dt=ss.days(7), stop=2003, verbose=0).run()
    mixed = sim.diseases.mixed
    assert isinstance(sim.solver, ss.solvers.abm) # The coded array needs agents
    assert mixed.results.infection.sum() > 0 and mixed.years_infected.sum() > 0 # Both parts ran
    assert np.isclose(mixed.years_infected.sum(), (mixed.results.I[1:].sum() + mixed.I.n)*sim.t.dt_year) # One step for each step spent infected, the last step's included
    fitted, _ = ss.fitting.bounded_pars(ss.Sim(diseases=Mixed(rate=ss.par(1.0, bounds=[0.5, 2])), n_agents=100, verbose=0).init())
    assert list(fitted.keys()) == ['rate'] # Its code's parameter can be fitted too

    return sim


@sc.timer()
def test_code_seams():
    sc.heading('Testing what code reads from a declared disease: parameters, counts, events and their memory, and products of a kind')
    class TPT(ss.Vaccine):
        pass
    def make(**kwargs):
        p = ss.pars(dict(beta=ss.par(ss.peryear(2), bounds=[ss.peryear(0.5), ss.peryear(5)]), dur=ss.years(1), cure=None), beta=3) # As a class's parameters are made
        S, I, T, R = ss.states('S I T R')
        tb = ss.Disease(S, I, T, R, infection=S >> I @ p.beta*I, diagnosis=I >> T @ p.dur, cure=ss.Transition(T, R, rate=ss.peryear(4)),
                        fast=ss.Transition(T, R, rate=ss.peryear(20), who=TPT.held), init=dict(I=0.05), name='tb', eff=ss.par(0.9)).stratify(age=[0, 15])
        give = ss.vaccinate(TPT(efficacy=0.01, name='tpt'), who=tb.diagnosis.since <= ss.years(1), when=ss.schedule(every=1), coverage=0.5)
        return ss.Sim(tb, interventions=give, start=2000, stop=2004, dt=ss.months(1), verbose=0, **kwargs), p
    sim, p = make(n_agents=2000)
    assert p.beta.value == ss.peryear(3) and p.beta.bounds is not None and p.cure is None
    sim.init()
    tb = sim.diseases.tb
    seen = []
    tb.add_callback(tb.diagnosis, seen.append) # Called from code with the agents who went through it
    sim.run()
    assert list(tb.strats.age.edges) == [0, 15]
    assert sum(len(uids) for uids in seen) == tb.results.diagnosis.sum() == tb.count_diagnosis.raw.sum()
    assert tb.results.fast.sum() > 0 and sim.results.give.delivered.sum() > 0 # Those diagnosed in the last year, then cured by holding a product of the kind
    assert np.isclose(tb.I.n, np.count_nonzero(tb.I.raw & sim.people.alive.raw)*ss.results.people_scale(sim)) and np.isclose(sim.people.alive.n, sim.people.alive.raw.sum()*ss.results.people_scale(sim)) # The people in each state now, after the last step
    assert np.allclose(np.ones(2)*(1 - tb.pars.eff), 0.1) # A declared parameter read by code during a run is its number
    return sim


def test_coded_defaults():
    sc.heading('Testing a bare beta, typed parameters, and results in __init__() of a disease written as code')
    class SI(ss.AgentDisease):
        def __init__(self):
            super().__init__()
            self.define_pars(beta=0.02, init_prev=ss.bernoulli(0.05))
            self.define_results(ss.Result('extra', kind='ratio'))
            return
    def infected(dt):
        sim = ss.Sim(SI(), networks=ss.RandomNet(n_contacts=4), n_agents=1000, dur=ss.days(28), dt=dt, verbose=0, rand_seed=1).run()
        assert 'extra' in sim.results.si
        return sim.results.si.infected[-1]
    daily, weekly = infected(ss.days(1)), infected(ss.days(7))
    assert daily < 1.5*weekly, f'A bare beta should be per day whatever the step: {daily}, {weekly}'

    return


@sc.timer()
def test_define_transitions():
    sc.heading('A declared disease whose structure depends on its arguments, declared on self in __init__()')

    class SIRS(ss.Disease):
        def __init__(self, waning=True, **kwargs):
            super().__init__(init_prev=0.01, **kwargs)
            self.define_states(ss.states('S I R'))
            beta = ss.par(0.3)
            self.define_transitions(
                infection = self.S >> self.I @ beta*self.I,
                recovery  = self.I >> self.R @ ss.days(10),
            )
            if waning: # A transition only some versions have
                self.define_transitions(waning=self.R >> self.S @ ss.days(100))
            self.define_results(prevalence=self.I.frac)

    class ClassSIRS(ss.Disease): # The same model in a class body
        S, I, R = ss.states(3)
        beta = ss.par(0.3)
        infection = S >> I @ beta*I
        recovery  = I >> R @ ss.days(10)
        waning    = R >> S @ ss.days(100)
        prevalence = I.frac

    assert list(SIRS(waning=False).transitions) == ['infection', 'recovery']
    kw = dict(unit='day', stop=100, verbose=0)
    sim = ss.Sim(SIRS(name='sirs'), **kw).run()
    ref = ss.Sim(ClassSIRS(init_prev=0.01, name='sirs'), **kw).run()
    assert np.allclose(sim.results.sirs.prevalence, ref.results.sirs.prevalence)

    class Extra(SIRS): # A subclass declares more after its parent's __init__(), including after a parameter for its own code
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(code_par=1.0)
            self.define_transitions(extra=self.R >> self.I @ ss.peryear(0.1))
    extra = Extra()
    assert 'extra' in extra.transitions and 'code_par' in extra.code_pars
    with pytest.raises(ss.CapabilityError): # A transition assigned to it after it was made would not be part of it
        extra.waning = extra.R >> extra.S @ ss.days(10)
    return sim


def test_init_in_class():
    """ Starting values in a class body are used, and a count can start from the number of people """
    class Vectors(ss.Disease):
        m = ss.par(3.0)
        S, I = ss.states(2)
        V = ss.State(people=False)
        N = (S + I).n
        init = dict(I=0.2, V=m*N)
        recovery = I >> S @ ss.perday(0.1)

    for solver in ['abm', 'ode']:
        sim = ss.Sim(Vectors(name='vec'), n_agents=1000, total_pop=1000, stop=2, unit='day', dt=1, solver=solver, verbose=0).run()
        res = sim.results.vec
        assert res.I[0] == 200, f'Only {res.I[0]} of 1000 started infected, not 200, under {solver}'
        assert res.V[0] == 3000, f'The count started at {res.V[0]}, not 3 per person, under {solver}'
    sim = ss.Sim(Vectors(name='vec', init=dict(I=0.1)), total_pop=1000, stop=2, unit='day', dt=1, solver='ode', verbose=0).run()
    assert sim.results.vec.I[0] == 100, 'init= given to the disease should replace that of the class'

    return


def test_derived_results():
    sc.heading('Testing the infected states and their results, a list of transitions, and | for sources')

    # A list of transitions and outcomes, and | for several sources; infected and prevalence count the states marked infected, beside the user's own
    S, R = ss.states(2)
    Ia, Iw = ss.states(2, infected=True)
    inf_a = S >> Ia @ 0.1*Ia
    inf_w = S >> Iw @ 0.01*(Ia + Iw)
    rec = (Ia | Iw) >> R @ 0.05
    prev_a = Ia/(S + Ia + Iw + R)
    res = ss.Sim(ss.Disease([inf_a, inf_w, rec, prev_a], init=dict(Ia=0.01), name='d'), n_agents=500, verbose=0).run().results.d
    assert np.array_equal(res.infected, res.Ia + res.Iw) and np.allclose(res.prevalence, res.infected/(res.S + res.Ia + res.Iw + res.R))
    rec = Ia + Iw >> R @ 0.05 # The same as (Ia | Iw) >> R
    same = ss.Sim(ss.Disease(inf_a, inf_w, rec, prev_a, init=dict(Ia=0.01), name='d'), n_agents=500, verbose=0).run().results.d
    assert np.array_equal(res.R, same.R)
    with pytest.raises(TypeError):
        Ia | Iw >> R @ 0.05 # Read by Python as Ia | (Iw >> R)

    # Nothing marked: the states of the force of infection are infected (E and I in S >> E @ beta*I), and each state says so; | on states is + in a result too
    S, E, I, R = ss.states(4)
    dis = ss.Disease(S >> E @ 0.1*I, E >> I @ 0.2, I >> R @ 0.05, init=dict(I=0.01), ever=I | R, frac=(I | R).frac, name='d')
    res = ss.Sim(dis, n_agents=200, verbose=0).run().results.d
    assert [s.name for s in dis.graph.states.values() if s.infected] == ['E', 'I']
    assert np.array_equal(res.infected, res.E + res.I) and np.allclose(res.prevalence, res.infected/(res.S + res.E + res.I + res.R))
    assert np.array_equal(res.ever, res.I + res.R) and np.allclose(res.frac, res.ever/(res.S + res.E + res.ever))

    return res


def test_declared_transmit():
    sc.heading('Testing a declared disease with its own transmit(), and functions of self, sim, and uids')
    kw = dict(n_agents=1000, total_pop=1000, start=2000, stop=2003, unit='year', verbose=0)

    class Same(ss.SIR):
        def transmit(self, transition):
            return super().transmit(transition) # What the solver would do

    class Half(ss.SIR):
        def transmit(self): # One transmission: the argument can be left out
            new = super().transmit()
            return new[::2] if isinstance(self.sim.solver, ss.solvers.abm) else new/2

    # As agents, the solver's own transmission is the same run; as an ODE, it is taken once a step, so is first order in dt
    res = {cls.__name__: ss.Sim(cls(beta=ss.peryear(2), dur_inf=ss.years(6), name='sir'), networks='random', solver='abm', **kw).run().results.sir.infection.sum() for cls in [ss.SIR, Same, Half]}
    assert res['SIR'] == res['Same'] > res['Half']
    ode = [ss.Sim(cls(beta=ss.peryear(2), dur_inf=ss.years(6), name='sir'), solver='ode', dt=ss.days(7), **kw).run().results.sir.infection.sum() for cls in [ss.SIR, Same, Half]]
    assert np.isclose(ode[0], ode[1], rtol=0.05) and ode[2] < ode[1]

    return res


@sc.timer()
def test_rel_dists():
    sc.heading('rel_sus= and rel_trans= given as distributions: drawn for each person, or their mean in a compartmental run')
    sir = ss.SIR(rel_trans=ss.gamma(shape=0.5, scale=2), rel_sus=ss.lognormal(mean=1, std=0.5), name='sir')
    sim = ss.Sim(sir, total_pop=2000, stop=10, verbose=0).init()
    assert sim.solver.name == 'abm', 'A distribution across people should make solver="auto" run agents'
    for key, std in [('rel_trans', 2**0.5), ('rel_sus', 0.5)]:
        vals = sim.diseases.sir[key].raw[sim.people.auids]
        assert abs(vals.mean() - 1) < 0.15 and abs(vals.std()/std - 1) < 0.25, f'{key} should be drawn for each person'
    totals = [ss.Sim(ss.SIR(beta=0.1, name='sir', **kw), total_pop=2000, stop=40, solver='ode', verbose=0).run().results.sir.infection.sum() for kw in [dict(), dict(rel_trans=ss.constant(2))]]
    assert totals[1] > 3*totals[0], 'A compartmental run should use the mean of rel_trans'
    with pytest.raises(ss.CapabilityError), ss.options.context(approximations='error'):
        ss.Sim(ss.SIR(rel_sus=ss.gamma(shape=0.5, scale=2)), total_pop=1000, solver='ode', verbose=0).init()
    agent = ssl.AgentSIR(rel_trans=ss.gamma(shape=0.5, scale=2), name='x')
    sim = ss.Sim(agent, n_agents=2000, networks='random', stop=2001, verbose=0).init()
    assert sim.diseases.x.rel_trans.raw[sim.people.auids].std() > 1, 'A disease written as code should draw rel_trans for each person too'
    return


@sc.timer()
def test_coded_strains_checks():
    sc.heading('A disease written as code with no networks is an error, not silence; strains are only for a declared disease')
    with pytest.raises(ValueError, match='no one'):
        ss.Sim(diseases=ssl.AgentSIR(init_prev=0.05), n_agents=500, stop=3, verbose=0).run()
    with pytest.raises(Exception, match='strains'):
        ssl.AgentSIR(strains=ss.Strains(['a', 'b']))
    return


if __name__ == '__main__':
    test_sir()
    test_seir()
    test_aliases()
    test_sis()
    test_ncd()
    test_congenital()
    test_current_state()
    test_infect()
    test_same_transmission()
    test_mixed_disease()
    test_code_seams()
    test_coded_defaults()
    test_define_transitions()
    test_init_in_class()
    test_derived_results()
    test_declared_transmit()
    test_rel_dists()
    test_coded_strains_checks()
