"""
Test connectors and custom interventions

Demonstrates a connector mediating a bidirectional interaction between two diseases
(HIV and SIS), plus a custom intervention that treats one of them. HIV provides a
`cd4` state (lower CD4 = more advanced disease); SIS provides a simple infected/
susceptible cycle that stands in for a co-infection.
"""

import sciris as sc
import numpy as np
import pytest
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl

sc.options(interactive=False) # Assume not running interactively


class hiv_sis(ss.Connector):
    """ Connector for a bidirectional HIV<->SIS interaction """
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            label = 'HIV-SIS',
            rel_sus_sis_hiv    = 2,   # People with HIV are 2x more likely to acquire SIS
            rel_sus_sis_aids   = 5,   # People with AIDS (low CD4) are 5x more likely to acquire SIS
            rel_trans_sis_hiv  = 1.5, # People with HIV are 1.5x more likely to transmit SIS
            rel_trans_sis_aids = 3,   # People with AIDS are 3x more likely to transmit SIS
            rel_sus_hiv_sis    = 2.7, # People with SIS are 2.7x more likely to acquire HIV
            rel_trans_hiv_sis  = 2.7, # People with SIS are 2.7x more likely to transmit HIV
        )
        self.update_pars(**kwargs)
        return

    def step(self):
        """ Specify HIV-SIS interactions """

        diseases = self.sim.diseases
        sis = diseases.sis
        hiv = diseases.hiv
        cd4 = self.sim.people.hiv.cd4

        # People with HIV are more likely to acquire SIS
        sis.rel_sus[cd4 < 500] = self.pars.rel_sus_sis_hiv
        sis.rel_sus[cd4 < 200] = self.pars.rel_sus_sis_aids

        # People with HIV are more likely to transmit SIS
        sis.rel_trans[cd4 < 500] = self.pars.rel_trans_sis_hiv
        sis.rel_trans[cd4 < 200] = self.pars.rel_trans_sis_aids

        # People with SIS are more likely to acquire HIV
        hiv.rel_sus[sis.infected] = self.pars.rel_sus_hiv_sis

        # People with SIS are more likely to transmit HIV
        hiv.rel_trans[sis.infected] = self.pars.rel_trans_hiv_sis
        return


class TreatSIS(ss.Intervention):
    """ Treat (cure) SIS-infected people from a given year onwards """
    def __init__(self, year=2020, prob=0.8):
        super().__init__() # Initialize the intervention
        self.prob = prob # Store the probability of treatment
        self.year = ss.date(year)
        return

    def step(self):
        sim = self.sim
        if sim.now > self.year:
            sis = sim.diseases.sis

            # Define who is eligible for treatment
            eligible_ids = sis.infected.uids
            n_eligible = len(eligible_ids) # Number of people who are eligible

            # Define who receives treatment
            is_treated = np.random.rand(n_eligible) < self.prob  # Compare np.random.rand() to self.prob
            treat_ids = eligible_ids[is_treated]  # Pull out the IDs for the people receiving the treatment
            sis.infected[treat_ids] = False
            sis.susceptible[treat_ids] = True
            sim.diseases.hiv.rel_sus[treat_ids] = 1
            sim.diseases.hiv.rel_trans[treat_ids] = 1
        return


def make_args():
    """ Make people, HIV, SIS, and network """
    pars = dict(n_agents=2000, verbose=0)
    mf = ss.SexualNet(dur=ss.lognormal(mean=5, std=0.5, unit=ss.years))
    hiv = ssl.diseases.HIV(beta={'sexual': [0.0008, 0.0004]}, init_prev=0.2)
    sis = ssl.AgentSIS(name='sis', beta={'sexual': [0.02, 0.01]}, init_prev=0.05) # Written as code, since the connector and TreatSIS set its arrays directly; low enough per act for rel_sus to matter
    return dict(**pars, networks=mf, diseases=[hiv, sis])


@sc.timer()
def test_connectors(do_plot=False):
    """ Test connector example """
    sc.heading('Testing connectors')

    # Sims with a connector, without one, and with a connector and SIS treatment
    args = make_args()
    sims = sc.obj()
    sims.con = ss.Sim(label='With connector', connectors=hiv_sis(), **args).run()
    sims.nocon = ss.Sim(label='Without connector', **args).run()
    sims.treat = ss.Sim(label='With treatment', connectors=hiv_sis(), interventions=TreatSIS(), **args).run()
    diseases = ['sis', 'hiv']
    results = sc.odict({sim.label: sc.obj({d: sim.results[d].infected for d in diseases}) for sim in sims.values()})

    if do_plot:
        plt.figure()
        for i,disease in enumerate(diseases):
            plt.subplot(2,1,i+1)
            for label,res in results.items():
                plt.plot(sims.con.t.yearvec, res[disease], label=label)
            plt.title(f'{disease.upper()} infections')
            plt.axvline(2020)
            plt.legend()
        sc.figlayout()
        plt.show()

    # Check results
    for disease in diseases:
        assert results[0][disease].sum() > results[1][disease].sum(), f'{disease.title()} infections should be higher with connector'
        assert results[0][disease].sum() > results[2][disease].sum(), f'{disease.title()} infections should be lower with treatment'

    # The same interaction with v4's modifiers, on the HIV written as code and a declared SIS
    def run(*connectors):
        hiv = ssl.diseases.HIV(beta={'sexual': [0.008, 0.004]}, init_prev=0.2)
        sis = ss.SIS(beta={'sexual': ss.peract(m2f=0.01, f2m=0.005)}, init_prev=0.05)
        mods = [make(hiv, sis) for make in connectors]
        return ss.Sim(n_agents=2000, networks=ss.SexualNet(), diseases=[hiv, sis], connectors=mods, start=2000, stop=2020, verbose=0).run().results
    base = run()
    stop_hiv = run(lambda hiv, sis: ss.modify(hiv.rel_trans, multiply=0, when=ss.schedule(start=2010))) # A modifier of the rel_trans of a disease written as code
    no_sis = run(lambda hiv, sis: ss.modify(sis.rel_sus, multiply=0, who=hiv.cd4 < 1e6)) # A condition on an array of a disease written as code (everyone's CD4 is below 1e6)
    after = stop_hiv.timevec > 2010
    assert base.hiv.infection[after].sum() > 0 and stop_hiv.hiv.infection[after].sum() == 0 # No HIV transmission after 2010
    assert base.sis.infection.sum() > 0 and no_sis.sis.infection.sum() == 0
    return sims


def test_lazy_arrays():
    """ A lazy condition on another module's array (not a declared state) is resolved in that module """
    sc.heading('Testing lazy conditions on module arrays')
    class HIV(ss.Disease):
        S = ss.State()
        I = ss.State()
        infection = S >> I @ 0.0 * I
        def __init__(self, **kwargs):
            super().__init__(init_prev=0.2, **kwargs)
            self.define_states(ss.FloatArr('cd4', default=ss.choice([300, 800])), ss.BoolArr('flag', default=ss.bernoulli(0.5)))
            return

    class Syph(ss.Disease):
        S = ss.State()
        I = ss.State()
        infection = S >> I @ 0.2 * I

    class Net(ss.RandomNet):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.BoolArr('core', default=ss.bernoulli(0.5)))
            return

    for which in ['cd4', 'flag', 'core']:
        hiv, syph, net = HIV(name='hiv'), Syph(name='syph', init_prev=0.05), Net(name='net')
        where = dict(cd4=hiv.cd4 >= 500, flag=~hiv.flag, core=~net.core)[which]
        protect = ss.modify(syph.rel_sus, multiply=0, who=where)
        sim = ss.Sim([hiv, syph], connectors=protect, networks=net, n_agents=1000, stop=2005, verbose=0, rand_seed=1).init()
        syph = sim.diseases.syph
        seeds = syph.I.uids
        sim.run()
        mods = dict(cd4=sim.diseases.hiv, flag=sim.diseases.hiv, core=sim.networks.net)
        new = syph.I.uids.remove(seeds) # Infected during the run
        arr = mods[which][which]
        protected = (arr[new] >= 500) if which == 'cd4' else ~arr[new]
        assert len(new) and not protected.any(), f'{which}: {protected.sum()} of {len(new)} infected were protected'
    return


@sc.timer()
def test_where_count():
    sc.heading('A condition on a count in where= is worked out at each step, under every solver; in when=, a clear error')
    def make():
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='d')
    def peak(solver, thresh):
        d = make()
        npi = ss.modify(d.infection, multiply=0.5, who=d.I.n > thresh)
        return ss.Sim(d, connectors=npi, n_agents=2000, total_pop=2000, stop=100, solver=solver, rand_seed=1, verbose=0).run().results.d.I.max()
    for solver in ['ode', 'binomial', 'tau', 'gillespie', 'abm']:
        never, once = peak(solver, 2000), peak(solver, 200)
        assert 200 < once < 0.6*never, f'Under {solver}, the peak is {once:n} with a threshold of 200, and {never:n} without one reached'
    d = make()
    vaccinate = ss.vaccinate(ss.Vaccine(efficacy=0.9), who=d.I.n > 200, coverage=0.1) # A delivery takes one too
    assert ss.Sim(d, interventions=vaccinate, n_agents=2000, stop=100, verbose=0).run().results.d.I.max() > 200
    for make_when in [lambda d: ss.modify(d.infection, multiply=0.5, when=d.I.n > 200), lambda d: ss.modify(d.infection, multiply=0.5, when=d.I)]:
        with pytest.raises(TypeError, match='goes in who='):
            make_when(make())
    with pytest.raises(TypeError, match='as who='): # For an event, a condition goes in who=, whether on people or on the population
        ss.vaccinate(ss.Vaccine(efficacy=0.9), when=make().I.n > 200)
    return


def test_keywords():
    sc.heading('who= chooses people (or decides for everyone at once), and when= is an event or a time')
    d = ss.SIR(beta=ss.perday(0.1), name='d')
    assert ss.vaccinate(ss.Vaccine(efficacy=0.9), who=d.I.n > 5).eligible is not None # A count decides for everyone at once, so it is a condition on the population
    with pytest.raises(TypeError, match='give the event as when='): # An event triggers, so it is not who=
        ss.vaccinate(ss.Vaccine(efficacy=0.9), who=d.infection)
    with pytest.raises(TypeError, match='give it as who='): # A condition on people is not when
        ss.vaccinate(ss.Vaccine(efficacy=0.9), when=d.I)
    with pytest.raises(TypeError, match='give the event as when='): # who= of a transition only restricts
        ss.Transition(d.S, d.R, who=d.infection)
    with pytest.raises(TypeError, match='goes in who='): # A state is not a time
        ss.modify(d.pars.beta, multiply=0.5, when=d.I)
    with pytest.raises(AttributeError, match='goes in who='): # when= is given the date, not the sim
        ss.Sim(d, connectors=ss.modify(d.pars.beta, multiply=0.5, when=lambda sim: sim.results), n_agents=100, start='2000-01-01', dur=ss.days(5), verbose=0).run()
    make = lambda intv: ss.Sim(ss.SIR(beta=ss.perday(0.1), init_prev=0.1, name='d'), networks='random', interventions=intv, n_agents=200, unit='day', stop=10, solver='abm', verbose=0).run().results.vx.delivered.sum()
    assert make(ss.vaccinate(ss.Vaccine(efficacy=0), when='d.infection', name='vx')) > 0 # An event in when= triggers it, here by its path

    # The same on a transition: an event, or a condition with no other driver, is what moves people
    S, I, F = ss.states('S I F')
    flag = ss.Transition(I, F, when=ss.person.age >= 0)
    assert flag.trigger is not None and flag.where is None
    tr = S >> I @ ss.perday(1)
    with pytest.raises(ValueError, match='takes no rate='):
        ss.Transition(S, I, rate=ss.perday(1), when=tr)
    timed = ss.Transition(I, F, dur=ss.days(3), who=ss.person.age >= 0) # A condition with a driver is who it applies to
    assert timed.trigger is None and timed.where is not None
    return


def test_member():
    sc.heading('ss.modify() as a member of a disease: the same as given to the sim')
    kw = dict(networks='random', n_agents=500, unit='day', stop=40, solver='abm', verbose=0)

    class Fading(ss.Disease):
        beta = ss.par(ss.perday(0.1))
        S, E, I, R = ss.states('S E I R')
        infection = S >> E @ beta*I
        onset = E >> I @ ss.days(3)
        recovery = I >> R @ ss.days(8)
        fading = ss.modify('rel_trans', multiply=0.5**(I.time_in/ss.days(3)), who=I)
        distancing = ss.modify(beta, multiply=0.5, when=ss.schedule(start=20))
        slower = ss.modify(recovery, multiply=2, when=ss.schedule(start=30))

    held = ss.Sim(Fading(init_prev=0.1, name='a'), **kw).run()
    assert list(held.connectors.keys()) == ['a_fading', 'a_distancing', 'a_slower']
    S, E, I, R = ss.states('S E I R')
    beta = ss.par(ss.perday(0.1))
    d = ss.Disease(S, E, I, R, beta=beta, infection=S >> E @ beta*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(8), init_prev=0.1, name='a')
    given = [ss.modify(d.rel_trans, multiply=0.5**(d.I.time_in/ss.days(3)), who=d.I), ss.modify(d.pars.beta, multiply=0.5, when=ss.schedule(start=20)), ss.modify(d.recovery, multiply=2, when=ss.schedule(start=30))]
    assert np.array_equal(held.results.a.I, ss.Sim(d, connectors=given, **kw).run().results.a.I)

    # Two of the same class, each with its own, and one by keyword, under any solver
    assert ss.Sim(diseases=[Fading(init_prev=0.1, name='x'), Fading(init_prev=0.1, name='y')], **kw).run().results.y.infection.sum() > 0
    S, I, R = ss.states('S I R')
    d = ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.3)*I, recovery=I >> R @ ss.days(8), halved=ss.modify('rel_trans', multiply=0.5), init_prev=0.01, name='d')
    never = ss.Sim(ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.15)*I, recovery=I >> R @ ss.days(8), init_prev=0.01, name='d'), solver='ode', total_pop=500, unit='day', stop=40, verbose=0).run()
    assert np.allclose(ss.Sim(d, solver='ode', total_pop=500, unit='day', stop=40, verbose=0).run().results.d.I, never.results.d.I)

    with pytest.raises(sc.KeyNotFoundError, match='does not belong to a module'): # Not a member, it has no module
        ss.Sim(diseases='sis', networks='random', connectors=ss.modify('rel_trans', multiply=0.5), n_agents=100, verbose=0).init()
    with pytest.raises(ss.CapabilityError, match='declared disease'):
        class Coded(ss.AgentDisease):
            fading = ss.modify('rel_trans', multiply=0.5)
        Coded()
    return


def test_coded():
    sc.heading('ss.modify() reaches the parameters of any module written as code, which are back to their base values after the run')
    class Wealth(ss.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(rate=1.0)
            self.seen = []
        def step(self):
            self.seen.append(self.pars.rate)
    class Link(Wealth, ss.Connector):
        pass
    for solver in ['abm', 'ode']:
        mods = [ss.modify('wealth.pars.rate', multiply=2, when=ss.schedule(start=2)), ss.modify('wealth.pars.rate', add=1, when=ss.schedule(start=4)), ss.modify('link.pars.rate', set=ss.timeseries({1: 3, 3: 4}), when=ss.schedule(start=1))]
        sim = ss.Sim(diseases=ss.SIR(beta=ss.perday(0.1)), modules=Wealth(name='wealth'), connectors=[Link(name='link'), *mods], n_agents=100, total_pop=100, unit='year', stop=5, solver=solver, verbose=0).run()
        wealth, link = sim.get_module('wealth'), sim.get_module('link')
        assert wealth.seen == [1, 1, 2, 2, 3, 3] and link.seen == [1, 3, 3, 4, 4, 4], f'{solver}: {wealth.seen} and {link.seen}' # A step from each year 0 to 5
        assert wealth.pars.rate == 1
    with pytest.raises(ValueError, match='cannot differ between people'):
        ss.Sim(diseases=ss.SIR(beta=ss.perday(0.1)), modules=Wealth(name='wealth'), connectors=ss.modify('wealth.pars.rate', multiply=2, who=ss.person.age > 5), n_agents=100, verbose=0).init()
    return


def test_modifier_tables():
    sc.heading('Modifiers that depend only on time are worked out before a compartmental run, span by span, and kept for a rerun until something they depend on changes')
    def make(start=40, cut=0.5, beta=0.3, wrap=False, **kwargs):
        S, I, R = ss.states(3)
        season = ss.seasonal(amplitude=0.5, peak=ss.days(50), period=ss.days(180))
        rate = ss.par(beta*season, name='beta') if wrap else ss.par(beta, name='beta')*season # A parameter given as an expression varies in time as one with the factor outside
        d = ss.Disease(S, I, R, infection=S >> I @ rate*I, recovery=I >> R @ ss.days(10), init=dict(I=0.001), name='d')
        d.stratify(age={0: 0.3, 20: 0.5, 60: 0.2})
        mods = [ss.modify(d.rel_sus, multiply=ss.by(age=[0.5, 1, 0.8]), when=ss.schedule(start=start, stop=150)),
                ss.modify(d.rel_trans, multiply=ss.par(cut, name='cut'), when=ss.schedule(stop=100)),
                ss.modify(d.recovery, multiply=ss.timeseries({0: 1, 60: 2}), when=ss.schedule(start=20))]
        return ss.Sim(d, interventions=mods, unit='day', stop=200, verbose=0, **kwargs)
    I = lambda sim: sim.results.d.I.values.copy()
    sim = make().run()
    ref = I(sim)
    assert sim._kernel_reason() is None
    assert np.array_equal(I(sim.run()), ref), 'A rerun gives the same'
    assert np.allclose(I(make().run(until=200)), ref, rtol=1e-12), 'A run a step at a time gives the same'
    assert np.array_equal(I(make(wrap=True).run()), ref), 'The seasonal factor inside a parameter varies in time'
    assert np.array_equal(I(sim.restart(pars=dict(beta=0.4)).run()), I(make(beta=0.4).run())), 'A new value of a parameter'
    assert np.array_equal(I(sim.restart(pars=dict(cut=0.2)).run()), I(make(beta=0.4, cut=0.2).run())), 'A new value of a parameter in a modifier'
    sim = make().run()
    sim.interventions()[0].start = 60
    assert np.array_equal(I(sim.run()), I(make(start=60).run())), 'A modifier changed in place'
    sim = make().run()
    sim.interventions()[2].amount.interpolation = 'linear'
    assert not np.array_equal(I(sim.run()), ref), 'A time series changed in place, in how it goes between its values'
    sim = make().run()
    assert np.array_equal(I(sim.new(interventions=make(start=60).pars.interventions).run()), I(make(start=60).run())), 'sim.new() with other modifiers'
    assert np.array_equal(I(sim.new(solver='binomial').run()), I(make(solver='binomial').run())), 'sim.run() with another solver'
    return


def test_sim_diagram():
    sc.heading('A sim\'s diagram and equations show every disease, and how they act on each other')
    hiv, syph = ss.SIS(name='hiv'), ss.SIR(name='syph')
    sim = ss.Sim(diseases=[hiv, syph], connectors=ss.modify(hiv.rel_trans, multiply=3, who=syph.I), verbose=0)
    text = sim.diagram('text')
    assert 'subgraph hiv' in text and 'subgraph syph' in text and 'syph_I -..->' in text # The link, from syphilis to HIV
    eqs = sim.equations()
    assert 'hiv:' in eqs and 'syph:' in eqs and 'syph.I → hiv.infection' in eqs
    fig = sim.diagram(do_show=False)
    assert fig is not None and len(fig.axes[0].patches) > 8
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sims = test_connectors(do_plot=do_plot)
    lazy = test_lazy_arrays()
    test_where_count()
    test_modifier_tables()
    test_sim_diagram()

    T.toc()
