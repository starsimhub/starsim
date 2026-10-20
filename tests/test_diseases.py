"""
Run tests of disease models
"""

# %% Imports and settings
import pytest
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import starsim.library as ssl

test_run = True
n_agents = [10_000, 2_000][test_run]
do_plot = True
sc.options(interactive=do_plot) # Assume not running interactively


@sc.timer()
def test_sir():
    sc.heading('Testing SIR dynamics')

    ppl = ss.People(n_agents)
    network_pars = {
        'n_contacts': ss.poisson(4), # Contacts Poisson distributed with a mean of 4
    }
    networks = ss.RandomNet(**network_pars)

    sir_pars = {
        'dur_inf': ss.normal(loc=10),  # Override the default distribution
    }
    sir = ss.SIR(**sir_pars)

    # Change pars after creating the SIR instance
    sir.pars.beta = {'random': ss.time.freq(0.1, 'year')}

    # You can also change the parameters of the distribution directly
    sir.pars.dur_inf.set(loc=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(loc = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(people=ppl, diseases=sir, networks=networks)
    sim.run()

    plt.figure()
    res = sim.results
    plt.stackplot(
        sim.timevec,
        res.sir.n_S,
        res.sir.n_I,
        res.sir.n_R,
        res.new_deaths.cumsum(),
    )
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.xlabel('Year')
    plt.title('SIR')

    # The textbook SIR: a duration is the mean of an exponential, and nobody dies (Starsim 3's lognormal and deaths are ssl.AgentSIR's)
    kw = dict(total_pop=1e6, stop=ss.days(365), solver='ode', verbose=0)
    def closed(sir):
        cl = ss.modify(sir.pars.beta, multiply=0.6, start=ss.days(30), stop=ss.days(90))
        return ss.Sim(sir, interventions=cl, **kw).run().results.sir
    a = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.days(7), init=dict(I=100)))
    b = closed(ss.SIR(beta=ss.par(ss.perday(2.5/7)), dur_inf=ss.expon(scale=ss.days(7)), init=dict(I=100)))
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
    for par, par_val in par_effects.items():
        lo = par_val[0]
        hi = par_val[1]

        # Make baseline pars
        pars0 = sc.dcp(base_pars)
        pars1 = sc.dcp(base_pars)

        if par != 'n_contacts':
            pars0['diseases'] = sc.mergedicts(pars0['diseases'], {par: lo})
            pars1['diseases'] = sc.mergedicts(pars1['diseases'], {par: hi})
        else:
            pars0['networks'] = sc.mergedicts(pars0['networks'], {par: lo})
            pars1['networks'] = sc.mergedicts(pars1['networks'], {par: hi})

        # Run the simulations and pull out the results
        s0 = ss.Sim(pars0, label=f'{par} {par_val[0]}').run()
        s1 = ss.Sim(pars1, label=f'{par} {par_val[1]}').run()

        # Check results
        if par == 'p_death':
            v0 = s0.results.cum_deaths[-1]
            v1 = s1.results.cum_deaths[-1]
        else:
            ind = 1 if par == 'init_prev' else -1
            v0 = s0.results.sir.prevalence[ind]
            v1 = s1.results.sir.prevalence[ind]

        print(f'Checking with varying {par:10s} ... ', end='')
        assert v0 <= v1, f'Expected prevalence to be lower with {par}={lo} than with {par}={hi}, but {v0} > {v1})'
        print(f'✓ ({v0} <= {v1})')

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
    print(f'  Peak infectious: {short.results.seir.n_I.max():n} (short) vs {long.results.seir.n_I.max():n} (long)')
    assert np.argmax(long.results.seir.n_I) > np.argmax(short.results.seir.n_I), 'A longer latent period should delay the peak'

    # Check the compartments and incidence
    sim = short
    seir = sim.diseases.seir
    res = sim.results.seir
    assert not (seir.E & seir.I).any(), 'Agents should not be in both E and I'
    assert (seir.infected == (seir.exposed | seir.infectious)).all(), 'The Starsim 3 name infected should be E plus I'
    assert np.allclose(res.prevalence, (res.n_E + res.n_I)/sim.results.n_alive), 'prevalence should be computed from E plus I'
    assert res.new_infections.min() >= 0 and res.new_infections[0] == 0, 'Seed infections should be excluded from incidence exactly once'
    assert np.array_equal(res.cum_infections, np.cumsum(res.new_infections)), 'Cumulative infections should be the running sum of new infections'

    # Latent agents should not transmit: with no infectious period, there should be no epidemic
    latent = ss.SEIR(dur_exp=ss.lognorm_ex(mean=ss.days(1000)), init_prev=0.1, beta=ss.perday(0.5))
    nosim = ss.Sim(n_agents=n_agents, diseases=latent, networks='random', dur=ss.days(100), dt=ss.days(1), verbose=0).run()
    assert nosim.results.seir.cum_infections[-1] == 0, 'Exposed agents should not transmit'

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
    assert 'n_infectious' in res, 'A derived alias should generate an automatic result'
    assert res.n_infectious.max() == 0, 'Nobody is infectious, since everyone is isolated'

    # Its new infections are recorded at the end of the step they happen in, as the counts of its states are (during a step, self.ti is its start)
    sim = ss.Sim(n_agents=1000, diseases=ssl.AgentSIR(beta=ss.perday(0.1), dur_inf=ss.days(10), p_death=0), networks='random', dur=ss.days(30), dt=ss.days(1), verbose=0).run()
    res = sim.results.agentsir
    assert np.array_equal(res.new_infections[1:], -np.diff(res.n_susceptible)) # The infections of each step leave S in it
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
    assert rep_sir.results.n_infected.label == 'Redefined', 'The replacement state should be the one that generates the result'

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

    pars = dict(
        n_agents = n_agents,
        diseases = 'sis',
        networks = 'random'
    )
    sim = ss.Sim(pars)
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

    ppl = ss.People(n_agents)
    ncd = ss.NCD()
    sim = ss.Sim(people=ppl, diseases=ncd, copy_inputs=False, dt=ss.years(1), analyzers='infection_log') # Since using ncd directly below
    sim.run()
    log = sim.analyzers[0].logs[0]

    assert len(log) == log.to_graph().number_of_edges()
    df = log.to_df()  # Check generation of line-list
    assert df.source.isna().all()

    plt.figure()
    plt.stackplot(
        sim.timevec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk - ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    assert ncd.results.n_not_at_risk.label == 'Not at risk'
    assert ncd.results.n_affected.label == 'Affected'
    plt.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
    plt.xlabel('Year')
    plt.title('NCD')

    # A disease written as code with no states of its own (as Starsim 3's ss.Disease), and an infection with no beta
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
    assert sim2.results.anemia.n_anemic[-1] > 400 # Anemia(p_onset=0.1) for 50 years
    assert sim2.results.background.new_infections.sum() == 0 # Not transmitted
    assert Anemia().transitions is None # Written as code, since it declares nothing
    with pytest.raises(ss.CapabilityError): # A disease that declares nothing and defines no states or step()
        ss.Sim(diseases=type('Empty', (ss.Disease,), {})(), verbose=0).init()
    return sim


@sc.timer()
def test_gavi():
    sc.heading('Testing GAVI diseases')
    ss.register_modules(ssl.diseases) # So we can use strings for these

    sims = sc.autolist()
    for disease in ['cholera', 'measles', 'ebola']:
        pars = dict(
            diseases = disease,
            n_agents = n_agents,
            networks = 'random',
        )
        sim = ss.Sim(pars)
        sim.run()
        sim.plot()
        sims += sim
    return sims


@sc.timer()
def test_multidisease():
    sc.heading('Testing simulating multiple diseases')

    ppl = ss.People(n_agents)
    sir1 = ss.SIR(name='sir1')
    sir2 = ss.SIR(name='sir2')

    sir1.pars.beta = {'randomnet': 0.1}
    sir2.pars.beta = {'randomnet': 0.2}
    networks = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(people=ppl, diseases=[sir1, sir2], networks=networks)
    sim.run()
    return sim


@sc.timer()
def test_mtct():
    sc.heading('Test mother-to-child transmission routes')
    ppl = ss.People(n_agents)
    sis = ss.SIS(
        beta = dict(
            random = ss.permonth(0.005),
            prenatal = ss.permonth(0.1), # Mother-to-child networks transmit one way
            breastfeeding = ss.permonth(0.1),
        )
    )
    networks = [ss.RandomNet(), ss.PrenatalNet(), ss.BreastfeedingNet()]
    demographics = ss.Pregnancy(fertility_rate=ss.time.freq(20, 'year'))
    sim = ss.Sim(dt=ss.months(1), people=ppl, diseases=sis, networks=networks, demographics=demographics)
    sim.run()
    sim.plot()
    res = sim.results
    assert np.array_equal(res.sis.n_S + res.sis.n_I, res.n_alive + res.pregnancy.n_unborn) # Everyone is in a state of the declared SIS, the unborn too (so they can be infected before birth), though they are not counted among the people
    return sim


@sc.timer()
def test_fetal_health(do_plot=do_plot):
    """
    Test the FetalHealth module across three scenarios:
      1. Baseline: no disease — birth weights ~3400g at term
      2. Disease: SIR infection worsens fetal outcomes via a connector
      3. Treatment: treating infected pregnant women partially reverses damage
    """
    sc.heading('Testing fetal health module')

    sim_kw = dict(
        n_agents=5_000,
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
    )
    def demog():
        return [
            ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ]
    sir_kw = dict(dur_inf=ss.lognorm_ex(mean=ss.years(6), std=ss.years(1)), beta=0.1, init_prev=0.1)

    def mean_bw(sim):
        bw = sim.results.fetal_health.mean_birth_weight
        return bw[bw > 0].mean()

    # -- 1. Baseline (no disease) --
    sim_baseline = ss.Sim(
        demographics=demog(),
        custom=ssl.mnch.FetalHealth(),
        networks=ss.PrenatalNet(),
        **sim_kw,
    )
    sim_baseline.run()

    # Basic sanity checks
    res_fh = sim_baseline.results.fetal_health
    total_births = res_fh.n_births.sum()
    assert total_births > 0, 'No births recorded'
    print(f'Total births: {total_births}')

    bw_baseline = mean_bw(sim_baseline)
    print(f'Mean birth weight (baseline): {bw_baseline:.0f}g')
    assert 2500 < bw_baseline < 4000, f'Mean birth weight {bw_baseline:.0f}g outside expected range'

    ga = res_fh.mean_ga_at_birth
    mean_ga = ga[ga > 0].mean()
    print(f'Mean GA at birth: {mean_ga:.1f} weeks')
    assert 36 < mean_ga < 44, f'Mean GA {mean_ga:.1f} weeks outside expected range'

    # -- 2. Disease only (no treatment) --
    sim_disease = ss.Sim(
        demographics=demog(),
        diseases=ssl.AgentSIR(name='sir', **sir_kw),
        connectors=ssl.mnch.fetal_infection(),
        custom=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_disease.run()

    bw_disease = mean_bw(sim_disease)
    print(f'Mean birth weight (disease): {bw_disease:.0f}g')
    assert bw_disease < bw_baseline, f'Disease ({bw_disease:.0f}g) should be lower than baseline ({bw_baseline:.0f}g)'

    ptb_disease = sim_disease.results.pregnancy.n_preterm.sum()
    ptb_baseline = sim_baseline.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — baseline: {ptb_baseline}, disease: {ptb_disease}')
    assert ptb_disease > ptb_baseline, 'Disease scenario should have more preterm births'

    # -- 3. Disease + treatment --
    sim_treated = ss.Sim(
        demographics=demog(),
        diseases=ssl.AgentSIR(name='sir', **sir_kw),
        connectors=ssl.mnch.fetal_infection(),
        interventions=ssl.mnch.treat_pregnant(disease='sir'),
        custom=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_treated.run()

    bw_treated = mean_bw(sim_treated)
    print(f'Mean birth weight (treated): {bw_treated:.0f}g')
    rtol = 0.05 # Generous tolerance for stochastic comparison
    assert bw_treated > bw_disease * (1 - rtol), f'Treated ({bw_treated:.0f}g) should be at least as high as disease-only ({bw_disease:.0f}g)'

    ptb_treated = sim_treated.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — disease: {ptb_disease}, treated: {ptb_treated}')

    print(f'Summary — baseline: {bw_baseline:.0f}g, disease: {bw_disease:.0f}g, treated: {bw_treated:.0f}g')

    if do_plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

        for sim, label in [(sim_baseline, 'Baseline'), (sim_disease, 'Disease'), (sim_treated, 'Treated')]:
            fh = sim.custom['fetal_health']
            born = fh.birth_weight > 0
            ax0.hist(np.asarray(fh.birth_weight[born]), bins=30, alpha=0.4, label=label, edgecolor='k')
        ax0.axvline(2500, color='r', ls='--', label='LBW threshold')
        ax0.set_xlabel('Birth weight (g)')
        ax0.set_ylabel('Count')
        ax0.set_title('Birth weight: baseline vs disease vs treated')
        ax0.legend()

        tvec = sim_baseline.timevec
        ax1.plot(tvec, sim_baseline.results.pregnancy.preterm_rate, label='Baseline')
        ax1.plot(tvec, sim_disease.results.pregnancy.preterm_rate, label='Disease')
        ax1.plot(tvec, sim_treated.results.pregnancy.preterm_rate, label='Treated')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Preterm rate')
        ax1.set_title('Preterm birth rate')
        ax1.legend()

        fig.tight_layout()

    sc.printgreen('✓ Fetal health tests passed')
    return sim_baseline, sim_disease, sim_treated


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
    sb           = preg.results['stillbirths'].sum()
    neonatal_deaths = preg.results['neonatal_deaths'].sum()
    births          = preg.results['births'].sum()
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
    # driving this ratio towards zero -- see Infection.set_congenital.
    ratio = sb/n_congenital
    assert 0.3 < ratio < 1.8, f'Stillbirth:congenital ratio {ratio:0.2f} is far from the expected 0.75'

    sc.printgreen('✓ Congenital framework tests passed')
    return sim


def make_genotype(name, beta=3.0, **kwargs):
    """ One genotype of a toy HPV: infection, clearance with immunity that wanes, and cancer (for ss.Strains) """
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
def test_strains_results():
    sc.heading('Testing a family of diseases: union results and conditions, and one cancer per person')
    names = ['hpv16', 'hpv18', 'hpv31']
    hpv = ss.Strains('hpv', [make_genotype('hpv16'), make_genotype('hpv18', 2.5), make_genotype('hpv31', 2.0)],
        immunity  = ss.immunity(protected='R', rel_sus=[[0.9, 0.3, 0.2], [0.3, 0.9, 0.2], [0.1, 0.1, 0.5]], combine='sum'),
        init_prev = 0.1,
        exclusive = 'Ca',
    )
    counts = []
    def count(sim): # The union and the co-infected, by hand, at every time point
        held = np.array([sim.diseases[name].I.raw | sim.diseases[name].Ca.raw for name in names])
        cancers = np.array([sim.diseases[name].Ca.raw for name in names])
        counts.append((held.any(0).sum(), (held.sum(0) >= 2).sum(), (cancers.sum(0) >= 2).sum()))
    sim = ss.Sim(diseases=hpv, analyzers=count, n_agents=5000, start=2000, stop=2030, dt=0.25, verbose=0).run()
    res = sim.results
    union, coinfected, two_cancers = np.array(counts).T
    assert sim.solver == 'abm' and list(sim.diseases.keys()) == names + ['hpv']
    assert np.array_equal(res.hpv.infected, union) and coinfected.max() > 0 # A union, not a sum: the co-infected are counted once
    assert np.array_equal(res.hpv.cancer, sum(res[name].cancer for name in names)) # A flow is a sum
    assert np.array_equal(res.hpv.infected.by('hpv')[:, 1], res.hpv18.infected) # Each member's own
    assert two_cancers.max() == 0 and res.hpv.n_X[-1] > 0 # One cancer per person: the others move to X
    assert 'Family hpv: 3 members' in sim.describe(output=True)

    # A union is a condition, with a count of members
    sim = ss.Sim(diseases=hpv, n_agents=500, start=2000, stop=2001, verbose=0).init()
    for name in names[:2]:
        sim.diseases[name].move_to(ss.uids([0, 1, 2]), 'I')
    selector = sim.diseases.hpv31.engine.selector
    assert np.array_equal(selector.select(sim.diseases.hpv.infected), np.array([sim.diseases[name].I.raw | sim.diseases[name].Ca.raw for name in names]).any(0))
    assert np.flatnonzero(selector.select(sim.diseases.hpv.infected.count() >= 2)).tolist() == [0, 1, 2]
    return sim


@sc.timer()
def test_strains_immunity():
    sc.heading('Testing protection between the members of a family: the combination rules, a vaccine on top, and shorter stays')
    def make(name, beta=1.0):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=S >> I @ ss.peryear(beta)*I, clearance=I >> R @ ss.years(1), reinfection=R >> I @ ss.peryear(beta)*I, name=name)
    hpv16, hpv18 = make('hpv16'), make('hpv18')
    matrix = [[0.2, 0.5], [0.1, 0.4]] # Rows from, columns against
    def rel_sus(combine):
        """ rel_sus of hpv16 for someone who has had both, someone who has had hpv16 only, and someone who has had neither; all with a vaccine of efficacy 0.5 """
        hpv = ss.Strains('hpv', [hpv16, hpv18], immunity=ss.immunity(protected='R', rel_sus=matrix, combine=combine))
        vaccine = ss.modify(hpv16.rel_sus, multiply=0.5, where=ss.ppl.age >= 0)
        sim = ss.Sim(diseases=hpv, connectors=vaccine, n_agents=100, verbose=0).init()
        sim.diseases.hpv16.move_to(ss.uids([0, 1]), 'R')
        sim.diseases.hpv18.move_to(ss.uids([0]), 'R')
        sim.diseases.hpv.start_step()
        return sim.diseases.hpv16.engine.mult_raw('rel_sus')[:3]
    assert np.allclose(rel_sus('sum'), [(1 - (0.2 + 0.1))*0.5, 0.4, 0.5]) # HP22: (1 − min(1, Σ))·(1 − vaccine) = 0.35, as HPVsim
    assert np.allclose(rel_sus('sum')[0], 0.35)
    assert np.allclose(rel_sus('max'), [0.4, 0.4, 0.5])
    assert np.allclose(rel_sus('multiply'), [(1 - (1 - 0.8*0.9))*0.5, 0.4, 0.5])
    assert np.allclose(rel_sus(lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))), [0.375, 0.4, 0.5]) # The strongest plus half the rest

    # Time in a state: protection 0.5 halves the stay, so after a year exp(-2) are left rather than exp(-1)
    left = []
    for dur in [0.0, 0.5]:
        a, b = make('a', 0.0), make('b', 0.0)
        hpv = ss.Strains('hpv', [a, b], immunity=ss.immunity(protected='R', dur=dict(I=[[0, 0], [dur, 0]])))
        sim = ss.Sim(diseases=hpv, n_agents=20_000, start=2000, stop=2002, dt=1/52, verbose=0).init()
        sim.diseases.b.move_to(sim.people.auids, 'R')
        sim.diseases.a.move_to(sim.people.auids, 'I')
        left.append(sim.run().results.a.n_I[52]/20_000)
    assert np.allclose(left, [np.exp(-1), np.exp(-2)], atol=0.01)
    return sim


@sc.timer()
def test_strains_seeding():
    sc.heading('Testing joint seeding: one member for each person seeded, in the shares given')
    names = ['hpv16', 'hpv18', 'hpv31']
    def make(seed):
        hpv = ss.Strains('hpv', [make_genotype(name) for name in names], init_prev=0.2, split=[0.5, 0.3, 0.2])
        return ss.Sim(diseases=hpv, n_agents=20_000, start=2000, stop=2005, rand_seed=seed, verbose=0).init()
    held = lambda sim: np.array([sim.diseases[name].I.raw for name in names])
    seeded = held(make(1))
    assert (seeded.sum(0) > 1).sum() == 0 # Nobody is seeded with two
    assert np.isclose(seeded.sum()/20_000, 0.2, atol=0.01) and np.allclose(seeded.sum(1)/seeded.sum(), [0.5, 0.3, 0.2], atol=0.02)
    assert np.array_equal(seeded, held(make(1))) and not np.array_equal(seeded, held(make(2))) # The family's own random numbers
    return seeded


@sc.timer()
def test_strains_compartmental():
    sc.heading('Testing a family under the compartmental solvers: members side by side, couplings from shares, against agents')
    def make(name, beta, **kwargs):
        S, I, R, Ca = ss.states('S I R Ca')
        return ss.Disease(infection=S >> I @ ss.peryear(beta)*I, clearance=I >> R @ ss.years(1), reinfection=R >> I @ ss.peryear(beta)*I, waning=R >> S @ ss.years(2),
                          cancer=I >> Ca @ ss.years(50), X=ss.State(people=True), infected=I + Ca, name=name, **kwargs)
    pars = dict(start=2000, stop=2020, dt=1/12, verbose=0)

    # With nothing between them, the members run as they would alone, and the union is from their shares
    alone = ss.Sim(diseases=[make('a', 3, init_prev=0.05), make('b', 2, init_prev=0.05)], solver='ode', total_pop=10_000, **pars).run()
    sim = ss.Sim(diseases=ss.Strains('hpv', [make('a', 3, init_prev=0.05), make('b', 2, init_prev=0.05)]), total_pop=10_000, **pars).run()
    res = sim.results
    assert sim.solver == 'ode' and np.array_equal(res.a.n_I, alone.results.a.n_I) and np.array_equal(res.b.n_I, alone.results.b.n_I) # Bit for bit
    assert np.allclose(res.hpv.n_I, 10_000*(1 - (1 - res.a.n_I/10_000)*(1 - res.b.n_I/10_000)))

    # HP22's rule as shares: rel_sus of a for people in its own R (exact) when everyone is in b's R, with a vaccine on top: (1 − (0.2 + 0.1))·0.5 = 0.35
    hpv = ss.Strains('hpv', [make('a', 1, init=dict(S=0.5, R=0.5)), make('b', 1, init=dict(R=10_000))], immunity=ss.immunity(protected='R', rel_sus=[[0.2, 0.5], [0.1, 0.4]], combine='sum'))
    sim = ss.Sim(diseases=hpv, connectors=ss.modify('a.rel_sus', multiply=0.5), solver='ode', total_pop=10_000, approx=True, **pars).init()
    sim.diseases.hpv.start_step()
    a = sim.diseases.a
    assert np.isclose(a.engine.mult_raw('rel_sus', state=a.graph.states.R), 0.35) and np.isclose(a.engine.mult_raw('rel_sus', state=a.graph.states.S), (1 - 0.1)*0.5)

    # A contract, for 2 and 4 members with HPV-like cross-immunity: side by side (ODE, binomial) against the exact form (the crossing, which a
    # modifier between the members gives, here one that changes nothing) and agents. Side by side is within 10% of exact on these
    for n in [2, 4]:
        matrix = np.full((n, n), 0.3) + np.eye(n)*0.6 # Rows from, columns against
        family = lambda: ss.Strains('hpv', [make(f'g{i}', 3 - 0.3*i) for i in range(n)], immunity=ss.immunity(protected='R', rel_sus=matrix, combine='sum'), init_prev=0.1)
        links = [ss.modify('g0.rel_sus', multiply=1.0, where=ss.conditions.Cond('state', f'g{i}', 'R'), name=f'link{i}') for i in range(1, n)]
        out = sc.objdict()
        for key,solver,connectors,seeds in [('ode', 'ode', [], 1), ('exact', 'ode', links, 1), ('binomial', 'binomial', [], 2), ('abm', 'abm', [], 2)]:
            sims = [ss.Sim(diseases=family(), connectors=sc.dcp(connectors), solver=solver, n_agents=10_000, total_pop=10_000, rand_seed=seed, approx=True, **pars).run() for seed in range(seeds)]
            out[key] = np.mean([[sim.results.hpv.infected[-1], sim.results.hpv.cancer.sum(), sim.results.g0.infected[-1]] for sim in sims], axis=0)
        assert np.allclose(out.ode, out.exact, rtol=0.1) and np.allclose(out.binomial, out.ode, rtol=0.05) and np.allclose(out.abm, out.exact, rtol=0.15), f'{n} members: {out}'

    # One course at a time, from shares: those who get cancer from one member are moved to X in the other, except those with cancer in it already
    hpv = ss.Strains('hpv', [make('a', 3), make('b', 2.5)], immunity=ss.immunity(protected='R', rel_sus=0.3), init_prev=0.1, exclusive='Ca')
    res = ss.Sim(diseases=hpv, solver='ode', total_pop=10_000, approx=True, **pars).run().results
    assert np.isclose(res.a.n_X[-1], res.b.n_Ca[-1]*(1 - res.a.n_Ca[-1]/10_000), rtol=0.05) and np.isclose(res.b.n_X[-1], res.a.n_Ca[-1]*(1 - res.b.n_Ca[-1]/10_000), rtol=0.05)

    # What the solvers say: approx, with auto running a coupled family as agents unless approx=True
    sim = ss.Sim(diseases=family(), **pars).init()
    assert sim.solver == 'abm' and 'approx (members side by side' in sim.solvers(output=True)
    assert ss.Sim(diseases=family(), approx=True, **pars).init().solver == 'ode'
    with pytest.raises(ss.CapabilityError):
        ss.Sim(diseases=family(), solver='ode', approx=False, **pars).init()
    return out


@sc.timer()
def test_strains_spec():
    sc.heading('Testing a family in the model spec: to_code() round trip, ss.diff(), and the hash')
    def family(m=0.3, exclusive='Ca'):
        return ss.Strains('hpv', [make_genotype('hpv16'), make_genotype('hpv18', 2.5)], immunity=ss.immunity(protected='R', rel_sus=[[0.9, m], [m, 0.9]], combine='sum'), init_prev=0.1, split=[0.6, 0.4], exclusive=exclusive)
    sim = ss.Sim(diseases=family(), stop=2010, verbose=0)
    spec = sim.to_spec()['modules'][-1]
    assert spec['type'] == 'Strains' and spec['members'] == ['hpv16', 'hpv18'] and spec['split'] == [0.6, 0.4] and spec['immunity'][0]['combine'] == 'sum'
    ns = {}
    exec(sim.to_code().replace("if __name__ == '__main__':", 'if False:'), ns) # Each member, then the family
    assert ns['sim'].hash() == sim.hash()
    assert 'immunity[0].rel_sus' in ss.diff(sim, ss.Sim(diseases=family(m=0.4), stop=2010, verbose=0), output=True)
    other = [ss.Sim(diseases=f, stop=2010, verbose=0).hash() for f in [family(m=0.4), family(exclusive=None)]]
    assert other[0].model == sim.hash().model and other[0].pars != sim.hash().pars # A value
    assert other[1].model != sim.hash().model # The structure
    return spec


@sc.timer()
def test_strains_errors():
    sc.heading('Testing what a family refuses')
    pair = lambda **kwargs: ss.Strains('hpv', [make_genotype('hpv16'), make_genotype('hpv18')], **kwargs)
    build = lambda hpv, **kwargs: ss.Sim(diseases=hpv, n_agents=100, stop=2001, verbose=0, **kwargs).init()
    with pytest.raises(ss.CapabilityError): # A member written as code, whose states the family cannot see
        ss.Strains('hpv', [make_genotype('hpv16'), ssl.AgentSIS()])
    with pytest.raises(ss.UnknownState): # A state the members do not have
        build(pair(immunity=ss.immunity(protected='nab', rel_sus=0.5)))
    with pytest.raises(ss.CapabilityError): # A combine= function under sim.gradient()
        ss.Sim(diseases=pair(immunity=ss.immunity(protected='R', rel_sus=0.5, combine=lambda p: p.max(0))), stop=2001, verbose=0).gradient()
    with pytest.raises(ValueError): # A family named like a dimension
        build(ss.Strains('risk', [make_genotype('hpv16').stratify(risk=['low', 'high']), make_genotype('hpv18')]))
    with pytest.raises(ValueError): # A member with its own initial values in a family that seeds
        ss.Strains('hpv', [make_genotype('hpv16', init_prev=0.1), make_genotype('hpv18')], init_prev=0.1)
    with pytest.raises(ss.OutOfBounds): # A protection of 1 on a time spent
        build(pair(immunity=ss.immunity(protected='R', dur=dict(I=0.6), combine='sum')))
    with pytest.raises(ss.UnknownState): # exclusive= with no state to move to
        sis = lambda name: ss.Disease(infection=ss.transmission('S -> I', beta=0.1), recovery=ss.progression('I -> S', dur=10), name=name)
        build(ss.Strains('flu', [sis('a'), sis('b')], exclusive='I'))
    with pytest.raises(TypeError): # The protection is the value of what it lowers, not a positional matrix
        ss.immunity(np.eye(2), protected='R')
    with pytest.raises(ss.ShapeError):
        build(pair(immunity=ss.immunity(protected='R', rel_sus=np.eye(3))))
    assert ss.Strains('hpv', members=[make_genotype('hpv16'), make_genotype('hpv18')]).member_names == ['hpv16', 'hpv18']
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
    fetal = test_fetal_health(do_plot=do_plot)
    cong  = test_congenital()
    strains = test_strains_results()
    immune = test_strains_immunity()
    seeded = test_strains_seeding()
    compart = test_strains_compartmental()
    spec    = test_strains_spec()
    test_strains_errors()
