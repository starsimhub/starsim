"""
Run tests of what Starsim 3's delivery classes did, with the deliveries: routine and campaign delivery by year, products shared between deliveries, tests with several results, and treatment with a capacity and a queue
"""
import warnings
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl
import pytest


def make_pars(n_agents=2_000, start=2000, stop=2020):
    """ Lightweight SIS sim params shared across delivery tests """
    return sc.objdict(
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
    treatment = ss.treat(ss.Treatment(cure=0.8), on=sis.infection, capacity=50, queue=True)
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

    # Two products of the same default name: the warning names the product's own class (HP85b)
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
    with pytest.raises(ValueError, match='is outside the sim'):
        ss.Sim(pars, interventions=ss.screen(make_dx(), coverage=0.5, at=[2030], name='screening')).init()
    return sim


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    prod  = test_products(do_plot=do_plot)
    prod2 = test_two_products()
    rout  = test_routine_delivery()
    camp  = test_campaign_delivery()

    T.toc()
