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
        diseases = 'sis',
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
    vaccination = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.8, start=vx_start, interval=ss.years(1))

    # A test with three results, most severe first (Starsim 3's ss.Dx table, with its hierarchy)
    dx_start = 2010
    cytology = ss.Diagnostic(results=dict(high=dict(I=0.9), low={'I': 0.05, None: 0.02}, normal=None))
    screening = ss.screen(cytology, coverage=0.9, start=dx_start, interval=ss.years(1))

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
    sis = ss.SIS(beta=ss.peryear(0.5))
    treatment = ss.treat(ss.Treatment(cure=0.8), where=sis.infection, capacity=50, queue=True)
    res = ss.Sim(make_pars() | dict(diseases=sis), interventions=treatment).run().results.treatment
    assert res.delivered.max() == 50 and res.waiting.max() > 0, 'Expected at most 50 treated each step, and a queue'

    # The same for a disease written as code, cured by its clear_infection() (as Starsim 3's ss.Tx moved infected to susceptible)
    sis = ssl.AgentSIS(init_prev=0.1)
    treatment = ss.treat(ss.Treatment(cure=0.8), where=sis.infected, capacity=20, queue=True, interval=1)
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
    vx1 = ss.vaccinate(vaccine, who=~vaccine.ever, coverage=0.1, start=2005, interval=ss.years(1))
    vx2 = ss.vaccinate(vaccine, who=~vaccine.ever, coverage=0.2, start=2005, interval=ss.years(1))
    sim = ss.Sim(make_pars(n_agents=5e3), interventions=[vx1, vx2])
    sim.run()

    res = sim.results
    vaccinated = sim.products()[0].ti_delivered.notnan.sum()
    assert len(sim.products()) == 1
    assert res.vx1.delivered.sum() + res.vx2.delivered.sum() == vaccinated > 0, 'Expected each person to be vaccinated by one delivery, once'

    # Products made by their deliveries are named without a warning, and a delivery keeps the name of its first variable
    vaccine = ss.vaccinate(efficacy=0.9, coverage=0.5, at=2005)
    campaign = ss.vaccinate(efficacy=0.5, coverage=0.5, at=2010)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        two = ss.Sim(make_pars(), interventions=[vaccine, campaign]).init()
    assert [p.name for p in two.products()] == ['vaccine_product', 'campaign_product'] # "vaccine" is taken by the first delivery, so each is named for its delivery (it was vaccine_vaccine, and a warning)
    care = ss.treat(cure=0.9, coverage=0.5, at=2005) # Its product would be "treatment", but a later delivery is called that
    sim = ss.Sim(make_pars(), interventions=[care, ss.treat(cure=0.5, at=2010, name='treatment')]).init()
    assert [p.name for p in sim.products()] == ['care_product', 'treatment_product']
    for intervention in [campaign]: # Also bound to a loop variable, and passed through sim.new()
        assert 'campaign' in ss.Sim(make_pars(), interventions=intervention).init().interventions
        assert 'campaign' in ss.Sim(make_pars()).new(interventions=intervention).init().interventions

    # Two products of the same default name: the warning names the product's own class
    t1, t2 = [ss.treat(ss.Treatment(cure=c), coverage=0.5, at=2005) for c in [0.5, 0.9]]
    with pytest.warns(Warning, match=r"ss\.Treatment\(\.\.\., name="):
        ss.Sim(make_pars(), interventions=[t1, t2]).init()
    return sim


@sc.timer()
def test_routine_delivery():
    sc.heading('Testing routine delivery: a window, a coverage that changes by year, and the default years')
    pars = make_pars()

    # Only between the start and the stop (which is not included, so Starsim 3's end_year=2012 is stop=2013)
    start, stop = 2008, 2013
    screening = ss.screen(make_dx(), coverage=1.0, start=start, stop=stop, interval=ss.years(1))
    sim = ss.Sim(pars, interventions=screening).run()
    n_screened = sim.results.screening.delivered
    years = sim.t.yearvec # A step's results are at its end, so those of the step that starts in a year are at the next
    assert n_screened[years <= start].sum() == 0, f'Expected zero screenings before {start}'
    assert n_screened[years > stop].sum() == 0, f'Expected zero screenings from {stop}'
    assert n_screened[(years > start) & (years <= stop)].min() > 0, f'Expected screenings each year from {start} to {stop}'

    # A coverage that changes by year (Starsim 3's years= with an array of prob=)
    coverage = ss.timeseries(np.arange(2005, 2016), np.linspace(0.05, 0.95, 11), mode='linear')
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), coverage=coverage, interval=ss.years(1), name='screening')).run()
    n_screened = sim.results.screening.delivered
    years = sim.t.yearvec
    early = n_screened[(years >= 2005) & (years <= 2007)].mean()
    late = n_screened[(years >= 2013) & (years <= 2015)].mean()
    assert late > 3*early, f'Expected screening to scale up with the coverage; got early={early}, late={late}'

    # No window: the whole sim, offered at every step with a probability per year (Starsim 3's annual_prob=True)
    sim = ss.Sim(make_pars(start=2010, stop=2014), dt=0.5, interventions=ss.screen(make_dx(), coverage=ss.probperyear(0.5), name='screening')).run()
    n_screened = sim.results.screening.delivered
    assert n_screened[1:].min() > 0, 'Expected screening on every step when no window is given'
    assert np.isclose(n_screened[1]/2000, 1 - 0.5**0.5, atol=0.05), 'Expected the probability per year to be per step'

    # A window whose years are close to, but not on, the time grid (Starsim 3 raised an IndexError)
    screening = ss.screen(make_dx(), coverage=1.0, start=2000, stop=2002, interval=ss.years(1), name='screening')
    sim = ss.Sim(make_pars(start=2000, stop=2002), dt=7/365, interventions=screening).run()
    assert sim.results.screening.delivered.sum() > 0, 'Expected screenings with a weekly step'
    return sim


@sc.timer()
def test_campaign_delivery():
    sc.heading('Testing campaign delivery: one year, several years with a coverage each, and years outside the sim')
    pars = make_pars()

    # One year only
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), coverage=0.5, at=[2010], name='screening')).run()
    n_screened = sim.results.screening.delivered
    on_year = np.isclose(sim.t.yearvec, 2011) # The results of the step that starts in 2010
    assert n_screened[on_year].sum() > 0 and n_screened[~on_year].sum() == 0, 'Expected screenings in 2010 only'

    # Several years, each with its own coverage
    years = [2005, 2010, 2015]
    coverage = ss.timeseries(years, [0.3, 0.5, 0.7])
    sim = ss.Sim(pars, interventions=ss.screen(make_dx(), coverage=coverage, at=years, name='screening')).run()
    n_screened = sim.results.screening.delivered
    on = np.isin(sim.t.yearvec, np.array(years) + 1) # The results of the steps that start in those years
    assert n_screened[~on].sum() == 0, 'Expected no screenings outside campaign years'
    assert np.all(np.diff(n_screened[on]) > 0), 'Expected each round to reach its own coverage'

    # A campaign after the sim ends
    with pytest.raises(ValueError):
        ss.Sim(pars, interventions=ss.screen(make_dx(), coverage=0.5, at=[2030], name='screening')).init()

    # Plots mark the rounds, and the start and stop of a modifier; not a delivery on a condition
    sis = sim.diseases.sis
    halve = ss.modify(sis.pars.beta, multiply=0.5, start=2012, stop=2016)
    when = ss.screen(make_dx(), coverage=0.5, when=lambda t: t.ti % 2 == 0, name='when')
    marked = ss.Sim(pars, interventions=[ss.screen(make_dx(), coverage=coverage, at=years, name='screening'), halve, when]).run()
    marks = ss.results.intervention_marks(marked)
    assert len(marks) == 4 and isinstance(marks[-1], tuple), 'Expected a mark at each of the 3 rounds, and a span for the modifier'
    fig = marked.plot('sis.prevalence', do_show=False)
    assert len(fig.axes[0].lines) == 1 + 3 and len(fig.axes[0].patches) == 1
    assert len(marked.plot('sis.prevalence', show_interventions=False, do_show=False).axes[0].lines) == 1
    plt.close('all')
    return sim


@sc.timer()
def test_state_results_at_start():
    """ An intervention's states are counted at the first time point, and its own results are left for its first step """
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

        def update_results(self):
            super().update_results()
            self.results.new_reached[self.ti] = self.new_reached # Would fail at the start, before the first step

    pars = dict(n_agents=100, diseases='sis', networks='random', verbose=0)
    for kw in [dict(), dict(dt=ss.years(2))]:
        res = ss.Sim(interventions=Program(**kw), **pars).run().results.program
        assert np.all(res.never_reached == 100)
        assert np.all(res.reached == 0)
        assert res.new_reached[0] == 0 and res.new_reached[1] == 1
    return


def test_disease_resolved_once():
    sc.heading('The disease of a delivery is worked out once, when it is initialized')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ ss.perday(0.3)*I, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
    treatment = ss.treat(ss.Treatment(cure=0.8), where=I, capacity=20, interval=1, name='treatment')
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
    dates = dict(start='2026-03-01', stop='2026-05-02')
    fast = make(**dates)
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(**dates)
    assert fast.kernel_only() and np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)
    assert not np.array_equal(fast.results.sir.I.values, make().results.sir.I.values) # They had an effect

    # A step function is worked out where it changes, as a date is
    held = lambda: dict(multiply=ss.timeseries({20: 0.7, 40: 0.4, 90: 1.0}), start='2026-01-15')
    fast = make(**held())
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(**held())
    assert np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)

    # A condition on time alone is read before the run, for the steps it changes on
    fast = make(when=lambda t: t.ti % 7 == 0)
    with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
        slow = make(when=lambda t: t.ti % 7 == 0)
    assert fast.kernel_only() and np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)

    # A coverage given as a rate is one share of those eligible at each step, in the kernels
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    assert ss.Sim(sir, interventions=ss.vaccinate(efficacy=0.8, coverage=ss.peryear(0.5)), nopython=True, **kw).run().complete

    # An amount or share that varies in time is not taken for a dated one
    series = ss.timeseries({30: 1.0, 100: 0.2})
    for kwargs in [dict(multiply=0.5*ss.par(series)), dict(multiply=0.5, share=series)]:
        beta = ss.par(0.3, name='beta')
        sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
        sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **kwargs), **kw).run()
        assert sim.diseases.sir.engine.changing_steps(0, 365) is None

    # With nopython=True, what would fall back to Python raises an error instead
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    make = lambda **mod: ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, **mod), nopython=True, **kw)
    assert make(start='2026-03-01').run().complete and make(where=sir.I.n >= 20_000).run().complete
    for mod in [dict(where=sir.I.n >= 20_000, until=sir.I.n <= 10_000), dict(multiply=ss.timeseries({30: 1.0, 100: 0.2}, mode='linear'))]:
        with pytest.raises(ss.CapabilityError, match='nopython'):
            ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **{'multiply':0.75, **mod}), nopython=True, **kw).run()

    # A comparison of counts is part of the rate: the same as writing it there
    beta = ss.par(0.3, name='beta')
    sir = ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, where=sir.I.n >= 20_000), **kw).run()
    beta2 = ss.par(0.3, name='beta')
    written = ss.Sim(ss.Disease(infection=S >> I @ (beta2*(1 - 0.25*(I.n >= 20_000)))*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir'), **kw).run()
    assert sim.kernel_only() and sim.interventions()[0].inlined
    assert np.allclose(sim.results.sir.I.values, written.results.sir.I.values, rtol=1e-12)

    # What needs memory is applied each step, and the other sim from the same disease is unchanged
    held = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, where=sir.I.n >= 20_000, until=sir.I.n <= 10_000), **kw).run()
    assert not held.interventions()[0].inlined and not held.kernel_only()
    assert ss.Sim(sir, **kw).run().results.sir.I.values.max() > 2.9e5
    return sim


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    prod  = test_products(do_plot=do_plot)
    prod2 = test_two_products()
    rout  = test_routine_delivery()
    camp  = test_campaign_delivery()
    once  = test_disease_resolved_once()

    T.toc()
