"""
Run tests of disease models
"""

# %% Imports and settings
import textwrap
import warnings
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
    sir = ss.SIR(dur_inf=ss.normal(loc=10)) # Override the default distribution

    # Change pars after creating the SIR instance
    sir.pars.beta = {'random': ss.time.freq(0.1, 'year')}

    # You can also change the parameters of the distribution directly
    sir.pars.dur_inf.set(loc=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(loc = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(n_agents=n_agents, diseases=sir, networks=networks)
    sim.run()
    if do_plot:
        res = sim.results
        plt.figure()
        plt.stackplot(sim.timevec, res.sir.S, res.sir.I, res.sir.R, res.new_deaths.cumsum())
        plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
        plt.title('SIR')

    # The textbook SIR: a duration is the mean of an exponential, and nobody dies (Starsim 3's lognormal and deaths are ssl.AgentSIR's)
    kw = dict(total_pop=1e6, stop=ss.days(365), solver='ode', verbose=0)
    def closed(sir):
        cl = ss.modify(sir.pars.beta, multiply=0.6, start=ss.days(30), stop=ss.days(90))
        return ss.Sim(sir, interventions=cl, **kw).run().results.sir
    a = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.days(7), init_n=dict(I=100)))
    b = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.expon(scale=ss.days(7)), init_n=dict(I=100)))
    assert abs(a.infection.sum() - b.infection.sum())/b.infection.sum() < 0.01 # 714,770 each; 785,847 with Starsim 3's lognormal
    assert a.death.sum() == 0
    agent = ssl.AgentSIR()
    assert isinstance(agent.pars.dur_inf, ss.lognorm_ex) and agent.pars.p_death.pars.p == 0.01 # Starsim 3's defaults
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
            v0, v1 = s0.results.cum_deaths[-1], s1.results.cum_deaths[-1]
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
        seir = ss.SEIR(dur_exp=ss.lognorm_ex(mean=dur_exp), dur_inf=ss.lognorm_ex(mean=ss.days(10)), beta=ss.perday(0.05))
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
    assert res.infection.min() >= 0 and res.infection[0] == 0, 'Seed infections should be excluded from incidence exactly once'
    assert np.array_equal(res.infection.cum, np.cumsum(res.infection)), 'Cumulative infections should be the running sum of new infections'

    # Latent agents should not transmit: with no infectious period, there should be no epidemic
    latent = ss.SEIR(dur_exp=ss.lognorm_ex(mean=ss.days(1000)), init_prev=0.1, beta=ss.perday(0.5))
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
    assert res.cum_infections[-1] == 0, 'Isolated agents should not transmit'
    assert 'infectious' in res, 'A derived alias should generate an automatic result'
    assert res.infectious.max() == 0, 'Nobody is infectious, since everyone is isolated'

    # Its new infections are recorded at the end of the step they happen in, as the counts of its states are (during a step, self.ti is its start)
    sim = ss.Sim(n_agents=1000, diseases=ssl.AgentSIR(beta=ss.perday(0.1), dur_inf=ss.days(10), p_death=0), networks='random', dur=ss.days(30), dt=ss.days(1), verbose=0).run()
    res = sim.results.agentsir
    assert np.array_equal(res.new_infections[1:], -np.diff(res.susceptible)) # The infections of each step leave S in it
    assert np.count_nonzero(sim.diseases.agentsir.ti_infected == 10) == res.new_infections[10] # And are dated there too, as a declared disease dates ti_I

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
    assert ode.solver == 'ode' and np.isclose(ode.results.sis.prevalence[-1], 0.6, atol=0.01) # Endemic at 1 - 1/R0
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
        plt.stackplot(sim.timevec, ncd.results.not_at_risk, ncd.results.at_risk - ncd.results.affected, ncd.results.affected, sim.results.new_deaths.cumsum())
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

    class Background(ss.Infection): # No beta, so not transmitted
        pass

    sim2 = ss.Sim(diseases=[Anemia(), Background(init_prev=0.1)], networks='random', n_agents=500, verbose=0).run()
    anemia = sim2.diseases.anemia
    assert sim2.results.anemia.anemic[-1] > 400 # Anemia(p_onset=0.1) for 50 years
    assert sim2.results.background.new_infections.sum() == 0 # Not transmitted
    assert isinstance(anemia, ss.Disease) and not isinstance(anemia, ss.Infection) and isinstance(sim2.diseases.background, ss.Disease)
    assert not hasattr(anemia, 'susceptible') and 'infect' not in anemia.describe(output=True) # No states of an infection forced on it
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
    demographics = ss.Pregnancy(fertility_rate=ss.time.freq(20, 'year'))
    sim = ss.Sim(dt=ss.months(1), n_agents=n_agents, diseases=sis, networks=networks, demographics=demographics)
    sim.run()
    if do_plot:
        sim.plot()
    res = sim.results
    assert np.array_equal(res.sis.S + res.sis.I, res.alive + res.pregnancy.n_unborn) # Everyone is in a state of the declared SIS, the unborn too (so they can be infected before birth), though they are not counted among the people
    return sim


@sc.timer()
def test_congenital():
    """
    Test the generic congenital outcome framework in the base Infection class.
    Uses CongenitalDisease example and verifies outcomes are sampled and fired.
    """
    sc.heading('Testing congenital framework...')

    sim = ss.Sim(
        n_agents=20_000, # Enough for a stable ratio of outcomes (about 30 of each)
        demographics=[
            ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
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
    neonatal_deaths = preg.results['neonatal_deaths'].sum()
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

    return sim


def family(members, name='hpv', init_prev=None, **kwargs):
    """ A disease whose strains are diseases of their own (ss.Strains of diseases), each with its own course """
    return ss.Disease(name=name, init_prev=init_prev, strains=ss.Strains(members, **kwargs))


def make_genotype(name, beta=3.0, **kwargs):
    """ One genotype of a toy HPV: infection, clearance with immunity that wanes, and cancer (for strains with courses of their own) """
    S, I, R, Ca = ss.states('S I R Ca')
    return ss.Disease(
        infection   = S >> I @ ss.peryear(beta)*I,
        clearance   = I >> R @ ss.years(1),
        reinfection = R >> I @ ss.peryear(beta)*I,
        waning      = R >> S @ ss.years(5),
        cancer      = I >> Ca @ ss.years(20),
        death       = Ca >> ss.null @ ss.years(5),
        X           = ss.State(people=True), # Where someone with cancer from another genotype goes
        infected    = I + Ca,
        name        = name,
        **kwargs,
    )


@sc.timer()
def test_strains_seeding():
    sc.heading('Testing joint seeding: one member for each person seeded, in the shares given')
    names = ['hpv16', 'hpv18', 'hpv31']
    def make(seed):
        hpv = family([make_genotype(name) for name in names], init_prev=0.2, split=[0.5, 0.3, 0.2])
        return ss.Sim(diseases=hpv, n_agents=20_000, start=2000, stop=2005, rand_seed=seed, verbose=0).init()
    held = lambda sim: np.array([sim.diseases[name].I.raw for name in names])
    seeded = held(make(1))
    assert (seeded.sum(0) > 1).sum() == 0 # Nobody is seeded with two
    assert np.isclose(seeded.sum()/20_000, 0.2, atol=0.01) and np.allclose(seeded.sum(1)/seeded.sum(), [0.5, 0.3, 0.2], atol=0.02)
    assert np.array_equal(seeded, held(make(1))) and not np.array_equal(seeded, held(make(2))) # The family's own random numbers
    return seeded


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
def test_strains_errors():
    sc.heading('Testing what strains with courses of their own refuse')
    pair = lambda **kwargs: family([make_genotype('hpv16'), make_genotype('hpv18')], **kwargs)
    build = lambda hpv, **kwargs: ss.Sim(diseases=hpv, n_agents=100, stop=2001, verbose=0, **kwargs).init()
    with pytest.raises(ss.CapabilityError): # A strain written as code, whose states the others cannot see
        family([make_genotype('hpv16'), ssl.AgentSIS()])
    with pytest.raises(KeyError, match='have no state or named sum of states for'): # A state the members do not have
        build(pair(immunity=ss.immunity(due_to='nab', rel_sus=0.5)))
    with pytest.raises(ss.CapabilityError): # A combine= function under sim.gradient()
        ss.Sim(diseases=pair(immunity=ss.immunity(due_to='R', rel_sus=0.5, combine=lambda p: p.max(0))), stop=2001, verbose=0).gradient()
    with pytest.raises(ValueError): # Strains named like a dimension
        build(ss.Disease(name='hpv', strains=ss.Strains([make_genotype('hpv16').stratify(risk=['low', 'high']), make_genotype('hpv18')], name='risk')))
    with pytest.raises(ValueError): # A strain with its own initial values in a disease that seeds
        family([make_genotype('hpv16', init_prev=0.1), make_genotype('hpv18')], init_prev=0.1)
    with pytest.raises(ValueError, match='which would make its exits infinitely fast'): # A protection of 1 on a time spent
        build(pair(immunity=ss.immunity(due_to='R', dur=dict(I=0.6), combine='sum')))
    with pytest.raises(KeyError, match='to move to; add one'): # A shared state with no state to move to
        def sis(name):
            S, I = ss.states('S I')
            return ss.Disease(infection=ss.Transition(S, I, rate=0.1 * I), recovery=ss.Transition(I, S, dur=10), name=name)
        build(family([sis('a'), sis('b')], name='flu', shared='I'))
    with pytest.raises(TypeError): # The protection is the value of what it lowers, not a positional matrix
        ss.immunity(np.eye(2), due_to='R')
    with pytest.raises(ValueError, match='but there are 2 strains'):
        build(pair(immunity=ss.immunity(due_to='R', rel_sus=np.eye(3))))
    with pytest.raises(ValueError): # The disease's own states as well as strains with natural histories of their own
        S, I = ss.states('S I')
        ss.Disease(infection=S >> I @ ss.peryear(1)*I, strains=ss.Strains([make_genotype('hpv16'), make_genotype('hpv18')]))
    assert family([make_genotype('hpv16'), make_genotype('hpv18')]).member_names == ['hpv16', 'hpv18']
    return


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
        ti, new = ('ti_I', 'infection') if sir.graph else ('ti_infected', 'new_infections')
        assert len(infected) == 50 and (sim.diseases.sir[ti][infected] == 3).all() and sim.results.sir[new][3] == 50
        assert len(log) == 50 and (log.source.to_numpy(dtype=float, na_value=np.nan) == -1).all() # Not NA, which (log.source == -1).all() skips

    # Only the seeded cases, who are dated by the first step: no warning that set_prognoses() is missing super()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ss.Sim(n_agents=200, diseases=ssl.AgentSIS(beta=0), start=2000, stop=2002, verbose=0).run()
    assert not any('super()' in str(x.message) for x in w)
    return sim


@sc.timer()
def test_same_transmission():
    sc.heading('Testing that an SIR declared and one written as code transmit the same, by the same code: the same infections, sources, and networks, CRN included')
    seeds = lambda self, sim, uids: (uids % 50 == 0).astype(float) # Every 50th agent, so seeding takes no random numbers

    class CodedSIR(ss.Infection):
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(beta=ss.perday(0.1), dur_inf=10, init_prev=ss.bernoulli(seeds))
            self.define_states(ss.BoolState('recovered'), ss.FloatArr('ti_recovered'))
            self.update_pars(**kwargs)
        def step_state(self):
            done = (self.infected & (self.ti_recovered < self.ti)).uids # At the start of the step after dur_inf, as a declared dwell time ends
            self.infected[done] = False
            self.recovered[done] = True
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            self.infected[uids] = True
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + self.pars.dur_inf - (not self.sim.initialized) # The seeds are infected at the start, as a declared state's are, not in the first step

    kw = dict(n_agents=n_agents, stop=50, unit='day', verbose=0, analyzers=ss.infection_log(), networks=[ss.RandomNet(n_contacts=4), ss.RandomNet(n_contacts=2, name='other')],
              connectors=[ss.modify('sir.rel_sus', multiply=0.5, where=lambda sim: sim.people.age > 40), ss.modify('other.pars.n_contacts', multiply=0.5)],
              interventions=ss.vaccinate(ss.Vaccine(efficacy=0.7), coverage=0.3, at=10))
    dec = ss.Sim(diseases=ss.SIR(name='sir', beta=ss.perday(0.1), dur_inf=ss.constant(ss.days(10)), init_prev=ss.bernoulli(seeds)), solver='abm', **kw).run()
    cod = ss.Sim(diseases=CodedSIR(name='sir'), **kw).run()
    d, c = dec.results.sir, cod.results.sir
    assert np.array_equal(d.infection, c.new_infections) and np.array_equal(d.I, c.infected) and np.array_equal(d.R, c.recovered) and d.infection.sum() > 500
    logs = [sim.analyzers[0].to_df().sort_values('target')[['target', 'source', 'network']].fillna('-').astype(str).values for sim in [dec, cod]]
    assert np.array_equal(*logs) # Who infected whom, and over which network
    return dec, cod


@sc.timer()
def test_same_strains():
    sc.heading('Testing that strains declared and written as code are passed on the same, by the same code: one pass, one of several sources at random, then protection')
    seeds = lambda self, sim, uids: (uids % 50 == 0).astype(float) # Every 50th agent, so seeding takes no random numbers
    strains = lambda imm: ss.Strains(['a', 'b'], rel_trans=[1, 2], immunity=ss.immunity(due_to=imm, rel_sus=[[1, 0.5], [0.5, 1]]) if imm else None)

    class CodedSIR(ss.Infection):
        def __init__(self, seed_strain, imm=None, **kwargs):
            super().__init__(strains=strains(imm and 'recovered'))
            self.define_pars(beta=ss.perday(0.1), dur_inf=10, init_prev=ss.bernoulli(seeds))
            self.define_states(ss.BoolState('recovered'), ss.FloatArr('ti_recovered'))
            self.seed_strain = seed_strain # The strain of each seed, as the declared one gives them
            self.update_pars(**kwargs)
        def step_state(self):
            done = (self.infected & (self.ti_recovered < self.ti)).uids
            self.infected[done] = False
            self.recovered[done] = True
            self.susceptible[done] = bool(self.strains.immunity) # With protection due to recovery, they can be infected by the other strain (R >> I, as the declared one is)
        def set_prognoses(self, uids, sources=None):
            super().set_prognoses(uids, sources)
            if not self.sim.initialized:
                self.strain[uids] = self.seed_strain[uids]
            self.infected[uids] = True
            self.recovered[uids] = False
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + self.pars.dur_inf - (not self.sim.initialized)

    def same(imm=None):
        kw = dict(n_agents=n_agents, stop=40, unit='day', verbose=0, analyzers=ss.infection_log(), networks=[ss.RandomNet(n_contacts=4), ss.RandomNet(n_contacts=2, name='other')])
        dec = ss.Sim(diseases=ss.SIR(name='sir', beta=ss.perday(0.1), dur_inf=ss.constant(ss.days(10)), init_prev=ss.bernoulli(seeds), strains=strains(imm and 'R')), solver='abm', **kw).init()
        cod = ss.Sim(diseases=CodedSIR(name='sir', seed_strain=dec.diseases.sir.engine.strain_raw().copy(), imm=imm), **kw).run()
        dec.run()
        d, c = dec.results.sir, cod.results.sir
        assert np.array_equal(d.infection, c.new_infections) and np.array_equal(d.I, c.infected) and d.infection.sum() > 1000
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
        rng.jump(to=rng.dists[0].dt_jump_size*self.mod.t.ti, force=True)
        return transmit_from(self, tr, src)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ss.solvers.Agents, 'transmit_from', from_first)
        same(imm=True)
    return out


@sc.timer()
def test_coded_strains():
    sc.heading('Testing strains of a disease written as code: passed on by the shared transmission, with protection between them')
    class SIRS(ss.Infection):
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
            self.infected[uids] = True
            self.recovered[uids] = False
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + self.pars.dur_inf*self.strain_value('dur', uids)

    strains = ss.Strains(['a', 'b'], rel_trans=[1, 2], dur=[0.5, 1], split=[0.9, 0.1], immunity=ss.immunity(due_to='recovered', rel_sus=[[1, 0.2], [0.2, 1]]))
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

    # The example in ss.Infection's docstring runs, and a library disease written as code takes strains= as well
    example = ss.Infection.__doc__.split('```python')[1].split('```')[0]
    ns = dict(ss=ss)
    exec(textwrap.dedent(example), ns)
    assert (np.bincount(ns['sim'].diseases.flu.strain[ns['sim'].diseases.flu.strain >= 0], minlength=2) > 0).all()
    sir = ss.Sim(diseases=ssl.AgentSIR(beta=ss.perday(0.1), strains=ss.Strains(['a', 'b'], rel_trans=[1, 2])), networks='random', n_agents=500, unit='day', stop=20, verbose=0).run().diseases.agentsir
    assert sir.strains.names == ['a', 'b'] and (sir.strain[sir.infected] >= 0).all()
    return sim


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
    assert sim.solver == 'abm' # The coded array needs agents
    assert mixed.results.infection.sum() > 0 and mixed.years_infected.sum() > 0 # Both parts ran
    assert np.isclose(mixed.years_infected.sum(), mixed.results.I[1:].sum()*sim.t.dt_year) # One step for each step spent infected
    fitted, _ = ss.fit.bounded_pars(ss.Sim(diseases=Mixed(rate=ss.par(1.0, bounds=[0.5, 2])), n_agents=100, verbose=0).init())
    assert list(fitted.keys()) == ['rate'] # Its code's parameter can be fitted too
    with pytest.raises(ss.CapabilityError, match='never calls'): # A declared transition never calls set_prognoses()
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
                        fast=ss.Transition(T, R, rate=ss.peryear(20), where=TPT.held), init=dict(I=0.05), name='tb', eff=ss.par(0.9)).stratify(age=[0, 15])
        give = ss.vaccinate(TPT(efficacy=0.01, name='tpt'), who=tb.diagnosis.since <= ss.years(1), coverage=0.5, interval=1)
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
    assert np.isclose(tb.count(tb.I), tb.results.I[-1]) and np.isclose(tb.count(), sim.results.alive[-1])
    assert np.allclose(np.ones(2)*(1 - tb.pars.eff), 0.1) # A declared parameter read by code during a run is its number
    S, I = ss.states('S I')
    sis = ss.Disease(S >> I @ ss.peryear(2)*I, I >> S @ ss.years(1), ss.death(rate=ss.peryear(0.02)), init=dict(I=0.05), name='sis')
    ode = ss.Sim(sis, total_pop=1000, start=2000, stop=2004, verbose=0).run() # The people results agents give the sim
    assert np.isclose(ode.results.alive[-1], ode.diseases.sis.count()) and np.isclose(ode.results.cum_deaths[-1], 1000 - ode.results.alive[-1])
    return sim


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

    class Fitted(ss.Disease): # A parameter given with bounds is a value, not a declaration, so the states and transitions can follow
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.states('S I'))
            self.define_transitions(infection=self.S >> self.I @ 0.1*self.I)
    assert Fitted(init_prev=ss.par(0.1, bounds=[0, 0.2])).pars.init_prev.bounds == [0, 0.2]

    with pytest.raises(ss.CapabilityError): # Declared once, while it is made
        sirs = SIRS()
        sirs.graph
        sirs.define_states(ss.State('V'))
    return sim


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
    same_strains = test_same_strains()
    coded_strains = test_coded_strains()
    mixed = test_mixed_disease()
    seams = test_code_seams()
    cong  = test_congenital()
    seeded = test_strains_seeding()
    test_strains_errors()
    define = test_define_transitions()
