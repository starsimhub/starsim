"""
Test deliveries: routine and campaign delivery by year, products shared between deliveries, tests with several results, and treatment with a capacity and a queue
"""
import warnings
from unittest import mock
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl
import pytest


def make_pars(n_agents=2_000, start=2000, stop=2020):
    """ Lightweight SIS sim params shared across delivery tests """
    return sc.obj(
        n_agents = n_agents,
        start    = start,
        stop     = stop,
        diseases = ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)), # In years, as the deliveries are
        networks = 'random',
        verbose  = 0,
    )


def make_dx():
    """ Minimal diagnostic, as Starsim 3's table of results by state """
    return ss.Diagnostic(sensitivity=0.95, specificity=0.99)


@sc.timer()
def test_products(do_plot=False):
    sc.heading('Testing products: a vaccine, a test with several results, and treatment with a queue')

    # A vaccine for the SIS, offered each year from 2005 to 80% of those not yet vaccinated
    vx_start = 2005
    vaccination = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=ss.schedule(every=ss.years(1), start=vx_start), coverage=0.8)

    # A test with three results, most severe first (Starsim 3's ss.Dx table, with its hierarchy)
    dx_start = 2010
    cytology = ss.Diagnostic(outcomes=dict(high=dict(I=0.9), low={'I': 0.05, None: 0.02}, normal=None))
    screening = ss.screen(cytology, when=ss.schedule(every=ss.years(1), start=dx_start), coverage=0.9)

    sim = ss.Sim(make_pars(n_agents=5e3), interventions=[screening, vaccination])
    sim.run()

    # Checks
    dxres = sim.results.screening
    sisres = sim.results.sis
    y = sim.t.yearvec
    pre_dx = y < dx_start
    post_dx = y > dx_start
    pre_vx = y < vx_start
    post_vx = y > vx_start
    assert dxres.delivered[pre_dx].sum() == 0, 'Expected no one screened before intervention start'
    assert dxres.delivered[post_dx].sum() > 0, 'Expected people screened after intervention start'
    assert dxres.positive[post_dx].sum() == dxres.high[post_dx].sum() + dxres.low[post_dx].sum() > 0, 'Expected positives to be every result but the last'
    assert dxres.high[post_dx].sum() + dxres.low[post_dx].sum() + dxres.normal[post_dx].sum() == dxres.delivered[post_dx].sum()
    assert sisres.infection[pre_vx].mean() > sisres.infection[post_vx].mean(), 'Expected vaccine to reduce infections'

    # Treatment of a fixed number each step (Starsim 3's treat_num): those not served wait, and leave the queue if no longer infected
    sis = ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.years(10))
    treatment = ss.treat(ss.Treatment(cure=0.8), when=sis.infection, capacity=ss.capacity(50, queue=True))
    res = ss.Sim(make_pars() | dict(diseases=sis), interventions=treatment).run().results.treatment
    assert res.delivered.max() == 50 and res.waiting.max() > 0, 'Expected at most 50 treated each step, and a queue'

    # The same for a disease written as code, cured by its clear_infection() (as Starsim 3's ss.Tx moved infected to susceptible)
    sis = ssl.AgentSIS(init_prev=0.1)
    treatment = ss.treat(ss.Treatment(cure=0.8), who=sis.infected, when=ss.schedule(every=1), capacity=ss.capacity(20, queue=True))
    base = ss.Sim(make_pars() | dict(diseases=sis)).run().results.agentsis
    sim = ss.Sim(make_pars() | dict(diseases=sis), interventions=treatment).run()
    assert sim.results.treatment.delivered.max() == 20 and sim.results.treatment.waiting.max() > 0 and sim.results.treatment.cured.sum() > 0
    assert sim.results.agentsis.susceptible.mean() > base.susceptible.mean(), 'Expected those cured to be susceptible again'

    if do_plot:
        sim.plot()

    return sim


@sc.timer()
def test_two_products():
    sc.heading('Testing one product delivered by two deliveries')
    vaccine = ss.Vaccine(efficacy=0.9)
    vx1 = ss.vaccinate(vaccine, who=~vaccine.ever, when=ss.schedule(every=ss.years(1), start=2005), coverage=0.1)
    vx2 = ss.vaccinate(vaccine, who=~vaccine.ever, when=ss.schedule(every=ss.years(1), start=2005), coverage=0.2)
    sim = ss.Sim(make_pars(n_agents=5e3), interventions=[vx1, vx2])
    sim.run()

    res = sim.results
    vaccinated = sim.products()[0].ti_delivered.notnan.sum()
    assert len(sim.products()) == 1
    assert res.vx1.delivered.sum() + res.vx2.delivered.sum() == vaccinated > 0, 'Expected each person to be vaccinated by one delivery, once'

    # Products made by their deliveries are named without a warning, and a delivery keeps the name of its first variable
    vaccine = ss.vaccinate(efficacy=0.9, when=2005, coverage=0.5)
    campaign = ss.vaccinate(efficacy=0.5, when=2010, coverage=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        two = ss.Sim(make_pars(), interventions=[vaccine, campaign]).init()
    assert [p.name for p in two.products()] == ['vaccine_product', 'campaign_product'] # "vaccine" is taken by the first delivery, so each is named for its delivery (it was vaccine_vaccine, and a warning)
    care = ss.treat(cure=0.9, when=2005, coverage=0.5) # Its product would be "treatment", but a later delivery is called that
    sim = ss.Sim(make_pars(), interventions=[care, ss.treat(cure=0.5, when=2010, name='treatment')]).init()
    assert [p.name for p in sim.products()] == ['care_product', 'treatment_product']
    for intervention in [campaign]: # Also bound to a loop variable, and passed through sim.new()
        assert 'campaign' in ss.Sim(make_pars(), interventions=intervention).init().interventions
        assert 'campaign' in ss.Sim(make_pars()).new(interventions=intervention).init().interventions

    # Two products of the same default name, and a product named as its delivery is: each is named for its delivery, without a warning
    t1, t2 = [ss.treat(ss.Treatment(cure=c), when=2005, coverage=0.5) for c in [0.5, 0.9]]
    dst = ss.screen(ss.Diagnostic(sensitivity=0.9, name='dst'), when=2005, name='dst')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        three = ss.Sim(make_pars(), interventions=[t1, t2, dst]).init()
    assert [p.name for p in three.products()] == ['treatment', 't2_product', 'dst_product']
    return sim


@sc.timer()
def test_routine_delivery():
    sc.heading('Testing routine delivery: a window, a coverage that changes by year, and the default years')
    pars = make_pars()

    # Only between the start and the stop (which is not included, so Starsim 3's end_year=2012 is stop=2013)
    start, stop = 2008, 2013
    screening = ss.screen(make_dx(), when=ss.schedule(every=ss.years(1), start=start, stop=stop), coverage=1.0)
    sim = ss.Sim(pars, interventions=screening).run()
    n_screened = sim.results.screening.delivered
    years = sim.t.yearvec # A step's results are in the row of the year it starts
    assert n_screened[years < start].sum() == 0, f'Expected zero screenings before {start}'
    assert n_screened[years >= stop].sum() == 0, f'Expected zero screenings from {stop}'
    assert n_screened[(years >= start) & (years < stop)].min() > 0, f'Expected screenings each year from {start} to {stop}'

    # A coverage that changes by year (Starsim 3's years= with an array of prob=)
    coverage = ss.timeseries(np.arange(2005, 2016), np.linspace(0.05, 0.95, 11), interpolation='linear')
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), when=ss.schedule(every=ss.years(1)), coverage=coverage, name='screening')).run()
    n_screened = sim.results.screening.delivered
    years = sim.t.yearvec
    early = n_screened[(years >= 2005) & (years <= 2007)].mean()
    late = n_screened[(years >= 2013) & (years <= 2015)].mean()
    assert late > 3*early, f'Expected screening to scale up with the coverage; got early={early}, late={late}'

    # No window: the whole sim, offered at every step with a probability per year (Starsim 3's annual_prob=True)
    sim = ss.Sim(make_pars(start=2010, stop=2014), dt=0.5, interventions=ss.screen(make_dx(), coverage=ss.probperyear(0.5), name='screening')).run()
    n_screened = sim.results.screening.delivered
    assert n_screened.min() > 0, 'Expected screening on every step when no window is given'
    assert np.isclose(n_screened[0]/2000, 1 - 0.5**0.5, atol=0.05), 'Expected the probability per year to be per step'

    # A window whose years are close to, but not on, the time grid (Starsim 3 raised an IndexError)
    screening = ss.screen(make_dx(), when=ss.schedule(every=ss.years(1), start=2000, stop=2002), coverage=1.0, name='screening')
    sim = ss.Sim(make_pars(start=2000, stop=2002), dt=7/365, interventions=screening).run()
    assert sim.results.screening.delivered.sum() > 0, 'Expected screenings with a weekly step'
    return sim


@sc.timer()
def test_campaign_delivery():
    sc.heading('Testing campaign delivery: one year, several years with a coverage each, and years outside the sim')
    pars = make_pars()

    # One year only
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), when=[2010], coverage=0.5, name='screening')).run()
    n_screened = sim.results.screening.delivered
    on_year = np.isclose(sim.t.yearvec, 2010) # The results of the step that starts in 2010
    assert n_screened[on_year].sum() > 0 and n_screened[~on_year].sum() == 0, 'Expected screenings in 2010 only'

    # Several years, each with its own coverage
    years = [2005, 2010, 2015]
    coverage = ss.timeseries(years, [0.3, 0.5, 0.7])
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), when=years, coverage=coverage, name='screening')).run()
    n_screened = sim.results.screening.delivered
    on = np.isin(sim.t.yearvec, years) # The results of the steps that start in those years
    assert n_screened[~on].sum() == 0, 'Expected no screenings outside campaign years'
    assert np.all(np.diff(n_screened[on]) > 0), 'Expected each round to reach its own coverage'

    # A campaign, or a start, after the sim ends never runs, so a sim can be run to an earlier stop
    for late in [ss.screen(make_dx(), when=[2030], coverage=0.5, name='screening'), ss.screen(make_dx(), when=ss.schedule(every=ss.years(1), start=2030), coverage=0.5, name='screening'), ss.screen(make_dx(), start=2030, coverage=ss.peryear(0.5), name='screening')]:
        assert ss.Sim(pars, interventions=late).run().results.screening.delivered.sum() == 0
    late = ss.Sim(pars, interventions=ss.modify('sis.pars.beta', multiply=0, start=2030)).run() # So does a modifier
    assert np.array_equal(late.results.sis.prevalence, ss.Sim(pars).run().results.sis.prevalence)
    for early in [ss.screen(make_dx(), when=[1950], coverage=0.5, name='screening'), ss.modify('sis.pars.beta', multiply=0, start=1950)]:
        with pytest.raises(ValueError, match='outside the sim'): # Before it starts is still an error
            ss.Sim(pars, interventions=early).init()

    # Plots mark the rounds, and the start and stop of a modifier; not a delivery on a condition
    sis = sim.diseases.sis
    halve = ss.modify(sis.pars.beta, multiply=0.5, when=ss.schedule(start=2012, stop=2016))
    when = ss.screen(make_dx(), coverage=0.5, when=lambda now: ss.date(now).year % 2 == 0, name='when')
    marked = ss.Sim(pars, interventions=[ss.screen(make_dx(), when=years, coverage=coverage, name='screening'), halve, when]).run()
    marks = ss.results.intervention_marks(marked, fill=True)
    assert len(marks) == 4 and isinstance(marks[-1], tuple), 'Expected a mark at each of the 3 rounds, and a span for the modifier'
    assert ss.results.intervention_marks(marked) == marks[:3] + [marks[-1][0]], 'Expected the modifier marked by its start by default'
    fig = marked.plot('sis.prevalence', do_show=False)
    assert len(fig.axes[0].lines) == 1 + 4 and len(fig.axes[0].patches) == 0
    fig = marked.plot('sis.prevalence', show_interventions='fill', do_show=False)
    assert len(fig.axes[0].lines) == 1 + 3 and len(fig.axes[0].patches) == 1
    assert len(marked.plot('sis.prevalence', show_interventions=False, do_show=False).axes[0].lines) == 1
    plt.close('all')
    return sim


@sc.timer()
def test_state_results_at_start():
    """ An intervention's states are counted at the start of each step, and its own flows at the end, so the first row has the first step's """
    class Program(ss.Intervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.BoolState('never_reached', default=True))
            self.define_aliases(reached=lambda mod: ~mod.never_reached)

        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('new_reached', dtype=int))

        def step(self):
            self.new_reached = 1

        def record_events(self):
            super().record_events()
            self.results.new_reached[self.ti] = self.new_reached

    pars = dict(n_agents=100, diseases='sis', networks='random', verbose=0)
    for kw in [dict(), dict(dt=ss.years(2))]:
        res = ss.Sim(interventions=Program(**kw), **pars).run().results.program
        assert np.all(res.never_reached == 100)
        assert np.all(res.reached == 0)
        assert np.all(res.new_reached == 1) # The first step's at the first time point
    return


def test_disease_resolved_once():
    sc.heading('The disease of a delivery is worked out once, when it is initialized')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    treatment = ss.treat(ss.Treatment(cure=0.8), who=I, when=ss.schedule(every=1), capacity=20, name='treatment')
    sim = ss.Sim(sir, interventions=treatment, unit='day', stop=5, solver='ode', verbose=0).init()
    assert sim.interventions.treatment.__dict__['_disease'] == 'sir'
    return


@sc.timer()
def test_modify_fast_paths():
    """ Dated modifiers, and modifiers on a comparison of counts, stay in the kernels, with the same results """
    S, I, R = ss.states('S I R')
    kw = dict(total_pop=1e6, start='2026-01-01', stop='2026-12-31', dt=ss.days(1), solver='ode', verbose=0)

    def make(**mod):
        beta = ss.par(0.3, name='beta')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
        return ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **{'multiply':0.75, **mod}) if mod else None, **kw).run()

    # The tables of dated modifiers, worked out where they change, are those of every step
    dates = dict(when=ss.schedule(start='2026-03-01', stop='2026-05-02'))
    fast = make(**dates)
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(**dates)
    assert fast._kernel_reason() is None and np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)
    assert not np.array_equal(fast.results.sir.I.values, make().results.sir.I.values) # They had an effect

    # A step function is worked out where it changes, as a date is
    held = lambda: dict(multiply=ss.timeseries({20: 0.7, 40: 0.4, 90: 1.0}), when=ss.schedule(start='2026-01-15'))
    fast = make(**held())
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(**held())
    assert np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)

    # A condition on time alone (a function of the date) is read before the run, for the steps it changes on
    fast = make(when=lambda now: now.day in [1, 15])
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(when=lambda now: now.day in [1, 15])
    assert fast._kernel_reason() is None and np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)

    # A coverage given as a rate is one share of those eligible at each step, in the kernels
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    assert ss.Sim(sir, interventions=ss.vaccinate(efficacy=0.8, coverage=ss.peryear(0.5)), require_kernels=True, **kw).run().complete

    # An amount or share that varies in time is not taken for a dated one
    series = ss.timeseries({30: 1.0, 100: 0.2})
    for kwargs in [dict(multiply=0.5*ss.par(series)), dict(multiply=0.5, frac=series)]:
        beta = ss.par(0.3, name='beta')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
        sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **kwargs), **kw).run()
        assert sim.diseases.sir.engine.changing_steps(0, 365) is None

    # With require_kernels=True, what would fall back to Python raises an error instead
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    make = lambda **mod: ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, **mod), require_kernels=True, **kw)
    assert make(when=ss.schedule(start='2026-03-01')).run().complete and make(who=sir.I.n >= 20_000).run().complete
    for mod in [dict(who=sir.I.n >= 20_000, until=sir.I.n <= 10_000), dict(multiply=ss.timeseries({30: 1.0, 100: 0.2}, interpolation='linear'))]:
        with pytest.raises(ss.CapabilityError, match='require_kernels'):
            ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **{'multiply':0.75, **mod}), require_kernels=True, **kw).run()

    # A comparison of counts is part of the rate: the same as writing it there
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, who=sir.I.n >= 20_000), **kw).run()
    beta2 = ss.par(0.3, name='beta')
    written = ss.Sim(ss.Disease(infection=S >> I @ (beta2*(1 - 0.25*(I.n >= 20_000)))*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir'), **kw).run()
    assert sim._kernel_reason() is None and sim.interventions()[0].inlined
    assert np.allclose(sim.results.sir.I.values, written.results.sir.I.values, rtol=1e-12)

    # What needs memory is applied each step, and the other sim from the same disease is unchanged
    held = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, who=sir.I.n >= 20_000, until=sir.I.n <= 10_000), **kw).run()
    assert not held.interventions()[0].inlined and held._kernel_reason() is not None
    assert ss.Sim(sir, **kw).run().results.sir.I.values.max() > 2.9e5
    return sim


@sc.timer()
def test_old_keywords():
    sc.heading('The messages of interventions name the keywords they now take')
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    late = ss.Event(sir.S >> sir.R, p=0.5, when=ss.on(sir.infection, delay=ss.days(3)), name='late')
    queue = ss.treat(ss.Treatment(cure=0.8), when=sir.infection, capacity=ss.capacity(5, queue=True), name='queue')
    fn = ss.Event(sir.S >> sir.R, p=0.5, who=lambda sim, uids: uids, name='fn')
    sim = ss.Sim(sir, interventions=[late, queue, fn], unit='day', stop=5, n_agents=200, verbose=0).init()
    assert 'when=ss.on(' in late.agent_reasons(sim)[0] and 'ss.capacity(' in queue.agent_reasons(sim)[0] and 'who=' in fn.agent_reasons(sim)[0]
    with pytest.raises(TypeError, match='who='):
        ss.Intervention(eligibility=lambda sim: None)
    return sim


@sc.timer()
def test_check_eligibility():
    sc.heading('An intervention written as code reads who it can reach with check_eligibility(), from who= as a delivery does')
    class Count(ss.Intervention):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_eligible', dtype=int))
        def step(self):
            self.results.n_eligible[self.ti] = len(self.check_eligibility())

    kw = dict(diseases=ss.SIS(), networks='random', n_agents=500, dur=3, verbose=0)
    whos = dict(everyone=None, cond=ss.person.age < 15, fn=lambda sim: sim.people.age < 15, uids=lambda self, sim: (sim.people.age < 15).uids, path='sis.I')
    sims = {k: ss.Sim(interventions=Count(who=who, name='count'), **kw).run() for k,who in whos.items()}
    n = {k: sim.results.count.n_eligible[0] for k,sim in sims.items()}
    young = (sims['cond'].people.age < 15).sum()
    assert n['everyone'] == 500 and n['cond'] == n['fn'] == n['uids'] == young, n
    assert 0 < n['path'] == sims['path'].results.sis.I[0] < 500, n
    return sims


def test_modify_coded_intervention():
    """ ss.modify() changes a parameter of an intervention written as code: one value in self.pars, or, with who=, each person's with current_value() """
    class Tx(ss.Intervention):
        allow_modifiers = ('p_treat',)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(p_treat=0.5)
            self.seen = []
            self.by_person = None
        def step(self):
            self.seen.append(self.pars.p_treat)
            self.by_person = self.current_value('p_treat', self.sim.people.auids)

    sis = ss.SIS()
    tx = Tx(name='tx')
    mods = [ss.modify('tx.pars.p_treat', multiply=0.2, start=2005), ss.modify('tx.pars.p_treat', multiply=0.5, who=sis.I, name='inf')]
    sim = ss.Sim(diseases=sis, networks='random', interventions=[tx, *mods], start=2000, stop=2010, dt=ss.years(1), n_agents=500, verbose=0).run()
    tx = sim.interventions.tx
    assert tx.seen[0] == 0.5 and np.isclose(tx.seen[-1], 0.5), 'p_treat in self.pars is the base value, since the class reads it with current_value()'
    vals = np.asarray(tx.by_person)
    inf = sim.diseases.sis.I[sim.people.auids]
    assert np.allclose(vals[inf], 0.05) and np.allclose(vals[~inf], 0.1), f'Expected 0.05 for the infected and 0.1 otherwise, not {np.unique(vals)}'
    return sim


def test_declared_intervention():
    sc.heading('Testing a declared module given as an intervention')
    class Tx(ss.Disease):
        has_product = False
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            naive, on = ss.states('naive on')
            self.define_states(naive, on)
            self.define_transitions(on >> naive @ ss.peryear(0.5))
    sim = ss.Sim(diseases=ss.SIS(), interventions=Tx(name='tx'), n_agents=500, start=2000, stop=2002, verbose=0).run() # Its move() method is not read as a delivery's move=
    assert 'tx' in sim.interventions
    return sim


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    prod  = test_products(do_plot=do_plot)
    prod2 = test_two_products()
    rout  = test_routine_delivery()
    camp  = test_campaign_delivery()
    once  = test_disease_resolved_once()
    test_old_keywords()
    elig  = test_check_eligibility()

    T.toc()
