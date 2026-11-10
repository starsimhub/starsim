"""
Run tests of disease models
"""

# %% Imports and settings
import textwrap
import pytest
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import starsim.library as ssl

n_agents = 2_000
do_plot = False
sc.options(interactive=do_plot) # Assume not running interactively


@sc.timer()
def test_sir():
    sc.heading('Testing SIR dynamics')

    networks = ss.RandomNet(n_contacts=ss.poisson(4)) # Contacts Poisson distributed with a mean of 4
    sir = ss.SIR(dur_inf=ss.normal(mean=10)) # Override the default distribution

    # Change pars after creating the SIR instance
    sir.pars.beta = {'random': ss.per(0.1, 'year')}

    # You can also change the parameters of the distribution directly
    sir.pars.dur_inf.set(mean=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(mean = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(n_agents=n_agents, diseases=sir, networks=networks)
    sim.run()
    if do_plot:
        res = sim.results
        plt.figure()
        plt.stackplot(sim.timevec, res.sir.S, res.sir.I, res.sir.R, res.died.cum)
        plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
        plt.title('SIR')

    # The textbook SIR: a duration is the mean of an exponential, and nobody dies (Starsim 3's lognormal and deaths are ssl.AgentSIR's)
    kw = dict(total_pop=1e6, dur=ss.days(365), solver='ode', verbose=0)
    def closed(sir):
        cl = ss.modify(sir.pars.beta, multiply=0.6, when=ss.schedule(start=ss.days(30), stop=ss.days(90)))
        return ss.Sim(sir, interventions=cl, **kw).run().results.sir
    a = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.days(7), init_n=dict(I=100)))
    b = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.expon(mean=ss.days(7)), init_n=dict(I=100)))
    assert abs(a.infection.sum() - b.infection.sum())/b.infection.sum() < 0.01 # 714,770 each; 785,847 with Starsim 3's lognormal
    assert a.death.sum() == 0
    agent = ssl.AgentSIR()
    assert isinstance(agent.pars.dur_inf, ss.lognormal) and agent.pars.p_death.pars.p == 0.01 # Starsim 3's defaults
    return sim


@sc.timer()
def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    base_pars = dict(n_agents=n_agents, networks=dict(type='random'), diseases=dict(type='sir'))

    # Define the parameters to vary
    par_effects = dict(
        beta = [0.01, 0.99],
        n_contacts = [1, 20],
        init_prev = [0.1, 0.9],
        dur_inf = [1, 8],
        p_death = [.01, .1],
    )

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par, (lo, hi) in par_effects.items():
        key = 'networks' if par == 'n_contacts' else 'diseases'
        s0, s1 = [ss.Sim(sc.mergedicts(sc.dcp(base_pars), {key: sc.mergedicts(base_pars[key], {par: val})}), label=f'{par} {val}').run() for val in [lo, hi]]
        if par == 'p_death':
            v0, v1 = s0.results.died.cum[-1], s1.results.died.cum[-1]
        else:
            prev = [s.results.sir.prevalence for s in [s0, s1]]
            v0, v1 = [p[1] if par == 'init_prev' else p.max() if par in ['beta', 'n_contacts'] else p[-1] for p in prev] # The peak for what speeds the epidemic up, which may have burned out by the end
        assert v0 <= v1, f'Expected prevalence to be lower with {par}={lo} than with {par}={hi}, but {v0} > {v1})'

    return s0, s1


@sc.timer()
def test_seir():
    sc.heading('Testing SEIR dynamics')

    # A longer latent period should delay and flatten the epidemic
    sims = []
    for dur_exp in [ss.days(1), ss.days(20)]:
        seir = ss.SEIR(dur_exp=ss.lognormal(mean=dur_exp), dur_inf=ss.lognormal(mean=ss.days(10)), beta=ss.perday(0.05))
        sim = ss.Sim(n_agents=n_agents, diseases=seir, networks='random', dur=ss.years(1), dt=ss.days(1), verbose=0)
        sim.run()
        sims.append(sim)
    short, long = sims
    print(f'  Peak infectious: {short.results.seir.I.max():n} (short) vs {long.results.seir.I.max():n} (long)')
    assert np.argmax(long.results.seir.I) > np.argmax(short.results.seir.I), 'A longer latent period should delay the peak'

    # Check the compartments and incidence
    sim = short
    seir = sim.diseases.seir
    res = sim.results.seir
    assert not (seir.E & seir.I).any(), 'Agents should not be in both E and I'
    assert np.allclose(res.prevalence, (res.E + res.I)/sim.results.alive), 'prevalence should be computed from E plus I'
    assert res.infection.min() >= 0 and res.infection[0] == res.S[0] - res.S[1], 'Seed infections should be excluded from incidence exactly once' # The first step's infections, not the seeds
    assert np.array_equal(res.infection.cum, np.cumsum(res.infection)), 'Cumulative infections should be the running sum of new infections'

    # Latent agents should not transmit: with no infectious period, there should be no epidemic
    latent = ss.SEIR(dur_exp=ss.lognormal(mean=ss.days(1000)), init_prev=0.1, beta=ss.perday(0.5))
    nosim = ss.Sim(n_agents=n_agents, diseases=latent, networks='random', dur=ss.days(100), dt=ss.days(1), verbose=0).run()
    assert nosim.results.seir.infection.cum[-1] == 0, 'Exposed agents should not transmit'

    return sims


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

    # Its new infections are recorded at the step they happen in, which takes them out of S by the next time point
    sim = ss.Sim(n_agents=1000, diseases=ssl.AgentSIR(beta=ss.perday(0.1), dur_inf=ss.days(10), p_death=0), networks='random', dur=ss.days(30), dt=ss.days(1), verbose=0).run()
    res = sim.results.agentsir
    assert np.array_equal(res.infection[:-1], -np.diff(res.susceptible)) # The infections of each step leave S by the next time point
    assert np.count_nonzero(sim.diseases.agentsir.ti_infected == 10) == res.infection[10] # And are dated there too, as a declared disease dates ti_I

    # Aliases can also be defined alongside the states they refer to
    class AliasSIR(ssl.AgentSIR):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(
                ss.BoolState('symptomatic', label='Symptomatic'),
                asymptomatic = lambda self: self.infected & ~self.symptomatic,
                ti_onset = 'ti_infected',
            )
            return

    alias_sir = AliasSIR()
    alias_sir.init_mock()
    assert alias_sir.ti_onset is alias_sir.ti_infected, 'A string alias should resolve to the state it names'
    assert (alias_sir.asymptomatic == (alias_sir.infected & ~alias_sir.symptomatic)).all(), 'A callable alias should be recomputed from its states'
    with pytest.raises(ValueError):
        alias_sir.asymptomatic[ss.uids(0)] = True # A callable alias is derived, so it can't be written to

    # A state can replace an alias or another state, either with overwrite=True or by resetting it by name
    class ReplaceSIR(ssl.AgentSIR):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(
                ss.BoolState('infectious', label='Infectious'), # Replaces the inherited alias
                ss.BoolState('infected', label='Redefined'), # Replaces the inherited state
                overwrite = True,
            )
            return

    rep_sir = ReplaceSIR()
    rep_sir.init_mock()
    assert 'infectious' in rep_sir.state_dict, 'A state should be able to replace an alias'
    assert [state.name for state in rep_sir.auto_state_list].count('infected') == 1, 'Overwriting a state should replace it, not duplicate it'
    assert rep_sir.results.infected.label == 'Redefined', 'The replacement state should be the one that generates the result'

    # An alias that a state would shadow should raise, rather than silently never being consulted
    with pytest.raises(AttributeError):
        ssl.AgentSIR().define_aliases(infected=lambda self: self.susceptible)

    # Removing something that isn't a state or an alias should raise
    with pytest.raises(AttributeError):
        ssl.AgentSIR().define_states(reset='not_a_state')

    # A missing attribute should still raise an AttributeError
    with pytest.raises(AttributeError):
        sir.not_an_attribute
    return sim


@sc.timer()
def test_sis(do_plot=do_plot):
    sc.heading('Testing SIS dynamics')

    sim = ss.Sim(n_agents=n_agents, diseases='sis', networks='random')
    sim.run()
    if do_plot:
        sim.plot()
        sim.diseases.sis.plot()

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
    return sim


@sc.timer()
def test_ncd():
    sc.heading('Testing NCDs')

    sim = ss.Sim(n_agents=n_agents, diseases=ssl.NCD(), dt=ss.years(1), analyzers='infection_log')
    sim.run()
    ncd = sim.diseases[0] # The sim's copy, which ran
    log = sim.analyzers[0].logs[0]

    assert len(log) == log.to_graph().number_of_edges()
    df = log.to_df()  # Check generation of line-list
    assert df.source.isna().all()

    assert ncd.results.not_at_risk.label == 'Not at risk'
    assert ncd.results.affected.label == 'Affected'
    if do_plot:
        plt.figure()
        plt.stackplot(sim.timevec, ncd.results.not_at_risk, ncd.results.at_risk - ncd.results.affected, ncd.results.affected, sim.results.died.cum)
        plt.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
        plt.title('NCD')

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
    assert not hasattr(anemia, 'susceptible') and 'infect' not in anemia.describe() # No states of an infection forced on it

    # An infection written as code with a beta but nothing to transmit it over is an error, not a run with no infections; with beta=0 it is not transmitted by design
    with pytest.raises(ValueError, match='would infect no one'):
        ss.Sim(diseases=ssl.AgentSIR(), n_agents=100, verbose=0).run()
    ss.Sim(diseases=ssl.AgentSIR(beta=0), n_agents=100, verbose=0).run()
    return sim


@sc.timer()
def test_gavi():
    sc.heading('Testing GAVI diseases')
    ss.register_modules(ssl.diseases) # So we can use strings for these

    sims = sc.autolist()
    for disease in ['cholera', 'measles', 'ebola']:
        sim = ss.Sim(diseases=disease, n_agents=n_agents, networks='random').run()
        if do_plot:
            sim.plot()
        sims += sim
    return sims


@sc.timer()
def test_multidisease():
    sc.heading('Testing simulating multiple diseases')

    sir1 = ss.SIR(name='sir1', beta={'randomnet': 0.1})
    sir2 = ss.SIR(name='sir2', beta={'randomnet': 0.2})
    sim = ss.Sim(n_agents=n_agents, diseases=[sir1, sir2], networks=ss.RandomNet(n_contacts=ss.poisson(4)))
    sim.run()
    return sim


@sc.timer()
def test_mtct():
    sc.heading('Test mother-to-child transmission routes')
    sis = ss.SIS(
        beta = dict(
            random = ss.permonth(0.005),
            prenatal = ss.permonth(0.1), # Mother-to-child networks transmit one way
            breastfeeding = ss.permonth(0.1),
        )
    )
    networks = [ss.RandomNet(), ss.PrenatalNet(), ss.BreastfeedingNet()]
    demographics = ss.Pregnancy(fertility_rate=ss.peryear(0.02))
    sim = ss.Sim(dt=ss.months(1), n_agents=n_agents, diseases=sis, networks=networks, demographics=demographics)
    sim.run()
    if do_plot:
        sim.plot()
    res = sim.results
    assert np.array_equal(res.sis.S + res.sis.I, res.alive + res.pregnancy.unborn) # Everyone is in a state of the declared SIS, the unborn too (so they can be infected before birth), though they are not counted among the people
    return sim


@sc.timer()
def test_congenital():
    """
    Test the congenital outcomes of ssl.mnch.CongenitalDisease (an ssl.AgentSIR, whose infections before birth go to set_congenital(), a hook of ss.AgentDisease).
    Uses CongenitalDisease example and verifies outcomes are sampled and fired.
    """
    sc.heading('Testing congenital framework...')

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
    )
    sim.run()

    preg = sim.demographics.pregnancy
    sb           = preg.results['stillbirth'].sum()
    neonatal_deaths = preg.results['neonatal_death'].sum()
    births          = preg.results['birth'].sum()
    n_congenital    = sim.diseases[0].congenital.sum()
    print(f'  Stillbirths: {sb}, Neonatal deaths: {neonatal_deaths}, Births: {births}, Congenital: {n_congenital}')

    assert sb > 0, 'Expected stillbirths from congenital disease outcomes'
    assert n_congenital > 0, 'Expected some congenital infections'
    assert births > 0, 'Expected some live births'

    # Stillbirths must fire in utero, not after delivery. birth_outcomes is
    # choice(a=3, p=[0.3, 0.4, 0.3]) over ['stillborn', 'congenital', 'normal'], so
    # stillbirths and congenital infections should occur in roughly a 0.3:0.4 ratio.
    # If stillbirths were scheduled at (rather than before) delivery, Pregnancy would
    # already have delivered the fetus and they would be logged as neonatal deaths,
    # driving this ratio towards zero -- see ssl.CongenitalDisease.set_congenital().
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
def test_seeding_routes():
    sc.heading('Testing seeding through several transmissions out of the first state: each seeds those it can infect')
    S, M, F = ss.states('S M F')
    dis = ss.Disease(S, M, F, name='g', clear_m=M >> S @ ss.years(1), clear_f=F >> S @ ss.years(1), init_prev=0.3,
                     to_m=ss.Transition(S, M, rate=ss.peract(f2m=0.01, m2f=0)*(M + F)), to_f=ss.Transition(S, F, rate=ss.peract(f2m=0, m2f=0.02)*(M + F)))
    sim = ss.Sim(diseases=dis, networks=ss.SexualNet(), n_agents=5000, verbose=0).init()
    g, female = sim.diseases.g, sim.people.female.raw
    assert not (g.M.raw & female).any() and not (g.F.raw & ~female).any() # Men are seeded as men, women as women
    assert np.isclose((g.M.raw | g.F.raw).sum(), 0.3*5000, rtol=0.1)
    return sim


@sc.timer()
def test_current_state():
    sc.heading('Testing current_state(): the index of the state each agent is in')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(name='sir', infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=4), init_prev=0) # Nobody seeded, so only the moves below
    sim = ss.Sim(diseases=sir, n_agents=100, stop=2001, verbose=0).init()
    sir = sim.diseases.sir
    sir.move(ss.uids([0, 1]), sir.I) # From wherever they are; the arguments are told apart by what they are
    ss.move(sir, [2], 'R')
    assert sir.move(sir.S, sir.I, uids=ss.uids([1, 3, 4])) == 2 # Of these, only those in S
    sir.move('I', 'S', uids=[3, 4]) # By name
    cs = sir.current_state()
    assert isinstance(cs, ss.IntArr) and cs.labels == ['S', 'I', 'R']
    assert cs[ss.uids([0, 1, 2, 3])].tolist() == [1, 1, 2, 0]
    assert [cs.labels[i] for i in cs[ss.uids([0, 2])]] == ['I', 'R']
    sub = sir.current_state(['R', sir.I]) # Some of them, by name or as states
    assert sub.labels == ['R', 'I'] and sub[ss.uids([0, 2, 3])].tolist() == [1, 0, -1]
    with pytest.raises(ValueError): # Read-only
        cs.raw[0] = 2
    with pytest.raises(KeyError, match='"sir" has no state X'):
        sir.current_state(['X'])
    res = sim.run().results.sir
    assert (res.I[0], res.R[0]) == (2, 1) # The start counts the agents moved before the first step

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


@sc.timer()
def test_same_transmission():
    sc.heading('Testing that an SIR declared, one written as code with ss.AgentDisease, and one on a plain ss.Disease transmit the same, by the same code: the same infections, sources, and networks, CRN included')
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

    class PlainSIR(ss.Disease): # The same, as array operations on a plain ss.Disease: S >> I @ beta*I is transmit() and two writes
        allow_modifiers = ('rel_sus', 'rel_trans') # Read by transmit()
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(beta=ss.perday(0.1), dur_inf=10)
            self.define_states(ss.BoolState('S', default=True), ss.BoolState('I'), ss.BoolState('R'), ss.TimeArr('ti_recovered'),
                               ss.FloatArr('rel_sus', default=1.0), ss.FloatArr('rel_trans', default=1.0))
            self.trans_rng = ss.multi_random('source', 'target')
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('infection', dtype=int, scale=True, kind='flow'))
        def init_post(self):
            super().init_post()
            uids = ss.uids(seeds(None, self.sim, self.sim.people.auids).nonzero()[0])
            self.infection_log.add_entries(uids, -1) # With no source, as a declared disease logs its seeds
            self.infect_uids(uids, ti=-1) # The seeds: infected at the end of the step before the first
        def infect_uids(self, uids, ti=None):
            self.S[uids] = False
            self.I[uids] = True
            self.ti_recovered[uids] = (self.ti if ti is None else ti) + 1 + self.pars.dur_inf
        def step_state(self):
            done = (self.I & self.ti_recovered.due()).uids
            self.I[done] = False
            self.R[done] = True
        def step(self):
            super().step()
            new = self.transmit(rel_trans=self.rel_trans.raw*self.I.raw, rel_sus=self.rel_sus.raw*self.S.raw) # Logged with who infected them, and over which network
            self.infect_uids(new)
            self.results.infection[self.ti] = len(new)

    kw = dict(n_agents=n_agents, stop=50, unit='day', verbose=0, analyzers=ss.infection_log(), networks=[ss.RandomNet(n_contacts=4), ss.RandomNet(n_contacts=2, name='other')],
              connectors=[ss.modify('sir.rel_sus', multiply=0.5, who=lambda sim: sim.people.age > 40), ss.modify('other.pars.n_contacts', multiply=0.5)],
              interventions=ss.vaccinate(ss.Vaccine(efficacy=0.7), when=10, coverage=0.3))
    dec = ss.Sim(diseases=ss.SIR(name='sir', beta=ss.perday(0.1), dur_inf=ss.constant(ss.days(10)), init_prev=lambda sim, uids: seeds(None, sim, uids)), solver='abm', **kw).run()
    cod = ss.Sim(diseases=CodedSIR(name='sir'), **kw).run()
    pla = ss.Sim(diseases=PlainSIR(name='sir'), **kw).run()
    d, c, p = dec.results.sir, cod.results.sir, pla.results.sir
    assert d.infection.sum() > 500
    assert np.array_equal(d.infection, c.infection) and np.array_equal(d.I, c.infected) and np.array_equal(d.R, c.recovered)
    assert np.array_equal(d.infection, p.infection) and np.array_equal(d.I, p.I) and np.array_equal(d.R, p.R)
    logs = [sim.analyzers[0].to_df().sort_values('target')[['target', 'source', 'network']].fillna('-').astype(str).values for sim in [dec, cod, pla]]
    assert np.array_equal(logs[0], logs[1]) and np.array_equal(logs[0], logs[2]) # Who infected whom, and over which network
    return dec, cod, pla


@sc.timer()
def test_same_strains():
    sc.heading('Testing that strains declared and written as code are passed on the same, by the same code: one pass, one of several sources at random, then protection')
    seeds = lambda self, sim, uids: (uids % 50 == 0).astype(float) # Every 50th agent, so seeding takes no random numbers
    strains = lambda imm: ss.Strains(['a', 'b'], rel_trans=[1, 2], immunity=ss.immunity(due_to=imm, protection=[[1, 0.5], [0.5, 1]]) if imm else None)

    class CodedSIR(ss.AgentDisease):
        def __init__(self, seed_strain, imm=None, **kwargs):
            super().__init__(strains=strains(imm and 'recovered'))
            self.define_pars(beta=ss.perday(0.1), dur_inf=10, init_prev=ss.bernoulli(seeds))
            self.define_states(ss.BoolState('recovered'), ss.TimeArr('ti_recovered'))
            self.seed_strain = seed_strain # The strain of each seed, as the declared one gives them
            self.update_pars(**kwargs)
        def step_state(self):
            done = (self.infected & self.ti_recovered.due()).uids
            self.infected[done] = False
            self.recovered[done] = True
            self.susceptible[done] = bool(self.strains.immunity) # With protection due to recovery, they can be infected by the other strain (R >> I, as the declared one is)
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            if not self.sim.initialized:
                self.strain[uids] = self.seed_strain[uids]
            self.recovered[uids] = False
            self.ti_recovered[uids] = self.ti + 1 + self.pars.dur_inf

    def same(imm=None):
        kw = dict(n_agents=n_agents, stop=40, unit='day', verbose=0, analyzers=ss.infection_log(), networks=[ss.RandomNet(n_contacts=4), ss.RandomNet(n_contacts=2, name='other')])
        dec = ss.Sim(diseases=ss.SIR(name='sir', beta=ss.perday(0.1), dur_inf=ss.constant(ss.days(10)), init_prev=lambda sim, uids: seeds(None, sim, uids), strains=strains(imm and 'R')), solver='abm', **kw).init()
        cod = ss.Sim(diseases=CodedSIR(name='sir', seed_strain=dec.diseases.sir.engine.strain_raw().copy(), imm=imm), **kw).run()
        dec.run()
        d, c = dec.results.sir, cod.results.sir
        assert np.array_equal(d.infection, c.infection) and np.array_equal(d.I, c.infected) and d.infection.sum() > 1000
        held = cod.diseases.sir.strain.raw >= 0
        assert np.array_equal(dec.diseases.sir.engine.strain_raw()[held], cod.diseases.sir.strain.raw[held]) and (np.bincount(cod.diseases.sir.strain.raw[held]) > 100).all() # The same strain for everyone infected
        logs = [sim.analyzers[0].to_df().sort_values('target')[['target', 'source', 'network']].fillna('-').astype(str).values for sim in [dec, cod]]
        assert np.array_equal(*logs) # Who infected whom (the one drawn), and over which network
        return dec, cod

    out = same()

    # With protection: the declared disease passes on to those in R in a pass of its own, which draws each edge's number after the pass to S;
    # from the step's first numbers, as the coded one's single pass does, the two are the same, protection draws included
    transmit_from = ss.solvers.Agents.transmit_from
    def from_first(self, tr, src):
        rng = self.mod.trans_rng
        rng.jump(to=rng.dists[0].dt_jump_size*(self.mod.t.ti + 1), force=True) # The step's own state (see ss.Dist.jump_dt())
        return transmit_from(self, tr, src)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ss.solvers.Agents, 'transmit_from', from_first)
        same(imm=True)
    return out


@sc.timer()
def test_coded_strains():
    sc.heading('Testing strains of a disease written as code: passed on by the shared transmission, with protection between them')
    class SIRS(ss.AgentDisease):
        def __init__(self, strains=None, **kwargs):
            super().__init__(strains=strains)
            self.define_pars(beta=ss.perday(0.08), dur_inf=8, dur_imm=30, init_prev=ss.bernoulli(0.02))
            self.define_states(ss.BoolState('recovered'), ss.FloatArr('ti_recovered'), ss.FloatArr('ti_waned'))
            self.update_pars(**kwargs)
        def step_state(self):
            done = (self.infected & (self.ti_recovered <= self.ti)).uids
            self.infected[done] = False
            self.recovered[done] = True
            self.ti_waned[done] = self.ti + self.pars.dur_imm
            waned = (self.recovered & (self.ti_waned <= self.ti)).uids
            self.recovered[waned] = False
            self.susceptible[waned] = True
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            self.recovered[uids] = False
            self.ti_recovered[uids] = self.ti + self.pars.dur_inf*self.strain_value('dur', uids)

    strains = ss.Strains(['a', 'b'], rel_trans=[1, 2], dur=[0.5, 1], fractions=[0.9, 0.1], immunity=ss.immunity(due_to='recovered', protection=[[1, 0.2], [0.2, 1]]))
    sim = ss.Sim(diseases=SIRS(strains=strains, name='flu'), networks=ss.RandomNet(n_contacts=4), analyzers=ss.infection_log(), n_agents=n_agents, stop=60, unit='day', verbose=0).run()
    flu = sim.diseases.flu
    log = sim.analyzers[0].to_df()
    src = log.source.to_numpy(dtype=float, na_value=np.nan)
    times = np.bincount(log.target.astype(int), minlength=len(flu.strain.raw)) # Infected more than once, the strain they hold is that of the last
    ok = src >= 0
    ok[ok] = (times[log.target[ok].astype(int)] == 1) & (times[src[ok].astype(int)] == 1)
    assert ok.sum() > 1000 and np.array_equal(flu.strain.raw[log.target[ok].astype(int)], flu.strain.raw[src[ok].astype(int)]) # Each case has its source's strain
    counts = np.bincount(flu.strain[flu.strain >= 0], minlength=2)
    assert counts[1] > counts[0] # The more transmissible strain overtakes the commoner one
    new = flu.infect(flu.susceptible.uids[:5], strain='b')
    assert len(new) == 5 and (flu.strain[new] == 1).all() and np.allclose(flu.ti_recovered[new] - flu.ti, 8)
    with pytest.raises(ss.CapabilityError, match='max_strains'):
        SIRS(strains=ss.Strains(['a', 'b'], max_strains=2))

    # The example in ss.AgentDisease's docstring runs, and a library disease written as code takes strains= as well
    example = ss.AgentDisease.__doc__.split('```python')[1].split('```')[0]
    ns = dict(ss=ss)
    exec(textwrap.dedent(example), ns)
    assert (np.bincount(ns['sim'].diseases.flu.strain[ns['sim'].diseases.flu.strain >= 0], minlength=2) > 0).all()
    sir = ss.Sim(diseases=ssl.AgentSIR(beta=ss.perday(0.1), strains=ss.Strains(['a', 'b'], rel_trans=[1, 2])), networks='random', n_agents=500, unit='day', stop=20, verbose=0).run().diseases.agentsir
    assert sir.strains.names == ['a', 'b'] and (sir.strain[sir.infected] >= 0).all()
    return sim


@sc.timer()
def test_route_betas():
    sc.heading('Testing route_betas(), the beta of each network that transmit() uses, as a hook a disease written as code overrides')
    class SI(ss.AgentDisease):
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(beta=ss.peryear(5), init_prev=0.05)
            self.update_pars(**kwargs)
    class OneWay(SI):
        def route_betas(self):
            betamap = super().route_betas()
            return {k:[b[0], b[1]*0] for k,b in betamap.items()} # From p1 to p2 only
    kw = dict(networks=ss.RandomNet(n_contacts=4, name='random'), n_agents=1000, dur=5, verbose=0)
    both = ss.Sim(diseases=SI(name='si'), **kw).run()
    one = ss.Sim(diseases=OneWay(name='si'), **kw).run()
    assert both.diseases.si.route_betas() == {'random': [ss.peryear(5), ss.peryear(5)]}
    assert 0 < one.results.si.infection.sum() < both.results.si.infection.sum()
    return one


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
    fitted, _ = ss.fit.bounded_pars(ss.Sim(diseases=Mixed(rate=ss.par(1.0, bounds=[0.5, 2])), n_agents=100, verbose=0).init())
    assert list(fitted.keys()) == ['rate'] # Its code's parameter can be fitted too
    with pytest.raises(ss.CapabilityError): # A declared transition never calls set_prognoses()
        type('Bad', (ss.SIS,), dict(set_prognoses=lambda self, uids, sources=None: None))

    # Its code sees those infected from code (disease.infect()) among the infection's events, in the step; and its own results are scaled as the declared ones are
    class Seen(Mixed):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_seen', dtype=float))
        def step(self):
            super().step()
            self.results.n_seen[self.ti] = len(self.infection.uids)
    class Seed(ss.Intervention):
        def step(self):
            if self.ti == 3:
                self.sim.diseases.seen.infect(self.sim.people.auids[:10])
    seen = ss.Sim(diseases=Seen(name='seen'), interventions=Seed(), n_agents=500, total_pop=5000, dt=ss.days(7), stop=2001, verbose=0).run().diseases.seen
    assert seen.results.n_seen[3] >= 10*10 # The 10 seeded, as people
    assert np.allclose(seen.results.n_seen, seen.results.infection) # Each step's infections, seeded or not, in the same scale

    # A modifier that reads what exists only once the sim runs is not worked out before it does
    S, I = ss.states('S I')
    rec = ss.par(ss.peryear(2))
    sis = ss.Disease(infection=S >> I @ ss.peryear(4)*I, recovery=ss.Transition(I, S, rate=rec), init_prev=0.1, name='sis')
    late = lambda sim: ss.peryear(np.full(len(sim.people), 2.0)) if sim.initialized else 1/0 # Fails if read at init
    ss.Sim(diseases=sis, interventions=ss.modify('sis.pars.rec', set=late), n_agents=200, stop=2002, verbose=0).run()
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
    S, I = ss.states('S I')
    sis = ss.Disease(S >> I @ ss.peryear(2)*I, I >> S @ ss.years(1), ss.death(rate=ss.peryear(0.02)), init=dict(I=0.05), name='sis')
    ode = ss.Sim(sis, total_pop=1000, start=2000, stop=2004, verbose=0).run() # The people results agents give the sim
    assert ode.results.alive[0] == 1000 and np.isclose((ode.diseases.sis.S | ode.diseases.sis.I).n, 1000 - ode.results.died.cum[-1]) # The people after the run are the start less every step's deaths
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

    class Early(ss.Disease): # Given a typed parameter, a disease that declares nothing is still written as code
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(beta=ss.peryear(0.1))
            return
    early = Early(beta=ss.par(ss.peryear(0.2)))
    assert early.graph is None and early.pars.beta == ss.peryear(0.2)
    assert not ss.Sim(early, n_agents=100, stop=2, verbose=0).init().declared

    S, I = ss.states('S I') # A disease's own prevalence takes the place of the automatic one
    own = ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> S @ ss.days(10), name='own')
    own.define_results(ss.Result('prevalence'))
    assert not ss.Sim(own, total_pop=100, stop=2, verbose=0).run().results.own.prevalence.values.any() # Left for its own code to fill in
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
    with pytest.raises(ss.CapabilityError): # Declared after the disease was made
        extra.define_results(share_I=extra.I.frac)
    with pytest.raises(ss.CapabilityError, match='subclass'): # Likewise a transition assigned to it, which would not be part of it
        extra.waning = extra.R >> extra.S @ ss.days(10)

    class Fitted(ss.Disease): # A parameter given with bounds is a value, not a declaration, so the states and transitions can follow
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.states('S I'))
            self.define_transitions(infection=self.S >> self.I @ 0.1*self.I)
    assert Fitted(init_prev=ss.par(0.1, bounds=[0, 0.2])).pars.init_prev.bounds == [0, 0.2]

    class Clash(ss.Disease): # A method as a rate is named after its transition (rate_recovery), which a parameter defined in code already is
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.states('S I'))
            self.define_pars(rate_recovery=ss.peryear(0.1))
            self.define_transitions(infection=self.S >> self.I @ 0.1*self.I, recovery=ss.Transition(self.I, self.S, rate=self.recovery_rate))
        def recovery_rate(self, sim, uids):
            return self.pars.rate_recovery
    with pytest.raises(TypeError, match='"rate_recovery" defined in code.*transition recovery'):
        Clash()

    with pytest.raises(ss.CapabilityError): # Declared once, while it is made
        sirs = SIRS()
        sirs.graph
        sirs.define_states(ss.State('V'))
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

    # A parameter of the class inside a distribution in a dwell time is the instance's, so setting it changes the dwell time (it kept the class's)
    class Incubation(ss.Disease):
        dur_inc = ss.par(ss.days(15))
        S, E, I = ss.states(3)
        infection = S >> E @ ss.perday(0.5)*I
        onset = E >> I @ ss.constant(dur_inc)
    dis = Incubation(dur_inc=ss.days(30))
    assert dis.graph.transitions.onset.dur.raw.pars.value is dis.graph.pars.dur_inc, 'The distribution should hold the parameter of the instance'
    assert Incubation.onset.dur.raw.pars.value is Incubation.dur_inc and float(Incubation.dur_inc.raw.value) == 15, 'The class should be unchanged'
    return


def test_private_class_members():
    """ An expression in a class body whose name starts with an underscore is a private intermediate: used, but not a result or an attribute """
    class Saturating(ss.Disease):
        beta = ss.par(0.3)
        S, I, R = ss.states(3)
        _h = np.maximum(beta*I.frac, 1e-12)
        infection = S >> I @ ss.perday(1)*_h
        recovery  = I >> R @ ss.days(10)

    dis = Saturating(name='sat', beta=0.6)
    assert '_h' not in dis.graph.exprs and '_h' not in dis.__dict__, 'A private intermediate should not be a member'
    sim = ss.Sim(dis, total_pop=1000, stop=30, unit='day', solver='ode', verbose=0).run()
    assert '_h' not in sim.results.sat, 'A private intermediate should not be a result'
    class Public(ss.Disease):
        beta = ss.par(0.3)
        S, I, R = ss.states(3)
        h = np.maximum(beta*I.frac, 1e-12)
        infection = S >> I @ ss.perday(1)*h
        recovery  = I >> R @ ss.days(10)
    pub = ss.Sim(Public(name='sat', beta=0.6), total_pop=1000, stop=30, unit='day', solver='ode', verbose=0).run()
    assert 'h' in pub.results.sat and np.array_equal(sim.results.sat.I, pub.results.sat.I), 'A private intermediate should give the same model as a public one, with the instance\'s parameters'

    with pytest.raises(ValueError): # A parameter (or state, transition, ...) can't be private: each instance needs its own
        class PrivatePar(ss.Disease):
            _beta = ss.par(0.3)
            S, I = ss.states(2)
            infection = S >> I @ _beta*I
        PrivatePar()
    return


def test_rel_sus_per_act():
    """ On a network with acts, rel_sus scales each act's hazard for a disease written as code, as for a declared disease """
    class Sus(ss.Connector):
        def step(self):
            self.sim.diseases.x.rel_sus[:] = 0.2
            return

    def run(make, seed):
        disease = make()
        sim = ss.Sim(diseases=disease, networks=ss.SexualNet(acts=ss.perweek(5)), connectors=Sus(), n_agents=20_000, start=2000, stop=2001, dt=ss.months(1), solver='abm', rand_seed=seed, verbose=0).run()
        x = sim.diseases.x
        return 18_000 - (x.susceptible if x.graph is None else x.S).sum() # New infections: 10% are seeds

    def declared():
        S, I = ss.states(2)
        return ss.Disease(infection=S >> I @ (dict(sexualnet=ss.peract(0.03))*I), recovery=I >> S @ ss.years(1000), init_prev=0.1, name='x')

    coded = lambda: ssl.AgentSIR(beta=dict(sexualnet=ss.peract(0.03)), dur_inf=ss.years(1000), p_death=0, init_prev=0.1, name='x')
    a, b = [np.mean([run(make, seed) for seed in range(3)]) for make in [coded, declared]]
    assert np.isclose(a, b, rtol=0.05), f'New infections written as code ({a}) and declared ({b}) should agree'
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
    with pytest.raises(TypeError, match='brackets'):
        Ia | Iw >> R @ 0.05 # Read by Python as Ia | (Iw >> R)

    # Nothing marked: the states of the force of infection are infected (E and I in S >> E @ beta*I), and each state says so; | on states is + in a result too
    S, E, I, R = ss.states(4)
    dis = ss.Disease(S >> E @ 0.1*I, E >> I @ 0.2, I >> R @ 0.05, init=dict(I=0.01), ever=I | R, share=(I | R).frac, name='d')
    res = ss.Sim(dis, n_agents=200, verbose=0).run().results.d
    assert [s.name for s in dis.graph.states.values() if s.infected] == ['E', 'I']
    assert np.array_equal(res.infected, res.E + res.I) and np.allclose(res.prevalence, res.infected/(res.S + res.E + res.I + res.R))
    assert np.array_equal(res.ever, res.I + res.R) and np.allclose(res.share, res.ever/(res.S + res.E + res.ever))

    # Declared, as a result given positionally (infected = E | I, the same as E + I), or by name or position in ss.states(); none for a staged condition
    S, E, I, H, R = ss.states(5)
    infected = E | I | H # H does not transmit, so is infected only if said
    trs = [S >> E @ 0.1*I, E >> I @ 0.2, I >> H @ 0.1, H >> R @ 0.05]
    dis = ss.Disease(*trs, infected, name='d')
    assert [s.name for s in dis.graph.states.values() if s.infected] == ['E', 'I', 'H'] and 'prevalence' in dis.graph.exprs
    for states in [ss.states('S E I H R', infected='E I H'), ss.states(5, infected=range(1, 4))]:
        assert [st.infected for st in states] == [False, True, True, True, False]
    with pytest.raises(ValueError, match='not one of the states'):
        ss.states('S I R', infected='X')
    Hl, C1, C2 = ss.states(3)
    dis = ss.Disease(Hl >> C1 @ 0.01, C1 >> C2 @ 0.1, name='d')
    assert not any(s.infected for s in dis.graph.states.values()) and 'prevalence' not in dis.graph.exprs
    with pytest.raises(ValueError, match='value for each person'): # Not silently dropped
        ss.Disease(S >> I @ 0.1*I, I >> R @ 0.05, p=I/ss.person.n, name='d')
    dis = ss.Disease(*trs, infected=I, name='d') # Exactly what is said: not E, though people are infected into it (by keyword: as a variable, I would have two names)
    assert [s.name for s in dis.graph.states.values() if s.infected] == ['I'] and 'prevalence' in dis.graph.exprs

    # A state called infected keeps the name, with no prevalence; a state no variable holds is named by its label
    susceptible, exposed, infected, recovered = ss.states(4)
    dis = ss.Disease(susceptible >> exposed @ 0.1*infected, exposed >> infected @ 0.1, infected >> recovered @ 0.05, name='d')
    assert 'prevalence' not in dis.graph.exprs and dis.graph.states.infected.infected
    dis = ss.Disease(ss.State('S'), name='d')
    assert list(dis.graph.states.keys()) == ['S'] and dis.graph.states.S.label == 'S'
    return res


@sc.timer()
def test_state_keywords():
    sc.heading('A state that has a name, given by a keyword of another name, raises rather than being renamed or made a result')
    S, I, R = ss.states('S I R')
    trs = [S >> I @ 0.3*I, I >> R @ ss.days(5)]
    with pytest.raises(ValueError, match='default_state=S'):
        ss.Disease(*trs, first=S)
    A = ss.State('A')
    with pytest.raises(ValueError, match='default_state=A'): # Named by its variable
        ss.Disease(first=A, infection=A >> I @ 0.1*I)
    assert ss.Disease(*trs, default_state=R).graph.default_state.name == 'R' # The state itself, as the default
    dis = ss.Disease(*trs, n_first=S.n, N=ss.State('Never')) # A count is a result; a state with no name, nor a variable, takes the keyword
    assert 'n_first' in dis.graph.exprs and 'N' in dis.graph.states
    class Local(ss.Disease): # A state named by keyword and then held in a local variable under another name keeps its name
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(S=ss.State('Susceptible'), I=ss.State('Infected'))
            src = self.I
            self.define_transitions(recovery=ss.Transition(src, self.S, rate=ss.peryear(1)))
    assert {'S', 'I'} <= set(Local().graph.states)
    return


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

    # Several transmissions: transmit() is told which
    class NoArg(ssl.TB):
        def transmit(self):
            return super().transmit()
    with pytest.raises(TypeError, match='transmissions'):
        ss.Sim(NoArg(), solver='abm', **kw).run()

    # A rate as a function of any of self, sim, and uids, by name: the same expression as agents and as an ODE; uids only as agents
    S, I, R = ss.states(3)
    rate = lambda self, sim: self.pars.beta*self.I.n/sim.people.n
    n = [ss.Sim(ss.Disease(infection=S >> I @ rate, recovery=I >> R @ ss.years(1), beta=ss.par(ss.peryear(3)), init_prev=0.05, name='d'), solver=solver, **kw).run().results.d.infection.sum() for solver in ['abm', 'ode']]
    assert np.isclose(n[0], n[1], rtol=0.15)
    with pytest.raises(ss.CapabilityError, match='uids'):
        ss.Sim(ss.Disease(infection=S >> I @ (lambda uids, sim: ss.peryear(0.5)*np.ones(len(uids))), recovery=I >> R @ ss.years(1), name='d'), solver='ode', **kw).run()
    return res


@sc.timer()
def test_init_kinds():
    sc.heading('init= is a dict by state: a number is a share seeded, init_prev=; and init_n= named in its own clash')
    for make in [lambda: ss.SIR(init=0.01), lambda: ssl.Dengue(init=0.001)]:
        with pytest.raises(TypeError, match='init_prev='):
            make()
    S, I, R = ss.states(3)
    with pytest.raises(ValueError, match='init_prev= and init_n= for I'):
        ss.Sim(ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, init_n=dict(I=5)), verbose=0).init()
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    sir   = test_sir()
    s1,s2 = test_sir_epi()
    seir  = test_seir()
    alias = test_aliases()
    sis   = test_sis()
    ncd   = test_ncd()
    gavi  = test_gavi()
    multi = test_multidisease()
    mtct  = test_mtct()
    current = test_current_state()
    infect  = test_infect()
    same  = test_same_transmission()
    strains = test_same_strains()
    coded = test_coded_strains()
    betas = test_route_betas()
    mixed = test_mixed_disease()
    seams = test_code_seams()
    cong  = test_congenital()
    define = test_define_transitions()
    derived = test_derived_results()
    own = test_declared_transmit()
    test_state_keywords()
    test_init_kinds()
