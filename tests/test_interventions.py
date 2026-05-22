"""
Run tests of vaccines and products
"""
import sciris as sc
import numpy as np
import starsim as ss
import pytest


def make_delivery_pars(n_agents=2_000, start=2000, stop=2020):
    """ Lightweight SIS sim params shared across delivery tests """
    return sc.objdict(
        n_agents = n_agents,
        start    = start,
        stop     = stop,
        diseases = 'sis',
        networks = 'random',
    )


def make_dx():
    """ Minimal diagnostic product used by delivery tests """
    dx_data = sc.dataframe(
        columns = ['disease', 'state', 'result', 'probability'],
        data = [
            ['sis', 'susceptible', 'positive', 0.01],
            ['sis', 'susceptible', 'negative', 0.99],
            ['sis', 'infected',    'positive', 0.95],
            ['sis', 'infected',    'negative', 0.05],
        ],
    )
    return ss.Dx(df=dx_data)


def run_sir_vaccine(efficacy, leaky=True):
    sc.heading(f'Testing SIR vaccine with {efficacy = } and {leaky = }')

    # parameters
    v_frac      = 0.5    # fraction of population vaccinated
    total_cases = 500    # total cases at which point we check results
    tol         = 3      # tolerance in standard deviations for simulated checks

    # create a basic SIR sim
    sim = ss.Sim(
        n_agents = 1000,
        pars = dict(
          networks = dict(
                type = 'random',
                n_contacts = 4
          ),
          diseases = dict(
                type      = 'sir',
                init_prev = 0.01,
                dur_inf   = ss.years(10),
                p_death   = 0,
                beta      = ss.peryear(0.06),
          )
        ),
        dur = ss.years(10),
        dt  = ss.years(0.05)
    )
    sim.init(verbose=False)

    # work out who to vaccinate
    in_trial = sim.people.sir.susceptible.uids
    n_vac = round(len(in_trial) * v_frac)
    in_vac = np.random.choice(in_trial, n_vac, replace=False)
    in_pla = np.setdiff1d(in_trial, in_vac)
    uids = ss.uids(in_vac)

    # create and apply the vaccination
    vac = ss.simple_vx(efficacy=efficacy, leaky=leaky)
    vac.init_pre(sim)
    vac.administer(sim.people, uids)

    # check the relative susceptibility at the start of the simulation
    rel_susc = sim.people.sir.rel_sus.values
    assert min(rel_susc[in_pla]) == 1.0, 'Placebo arm is not fully susceptible'
    if not leaky:
        assert min(rel_susc[in_vac]) == 0.0, 'Nobody fully vaccinated (all_or_nothing)'
        assert max(rel_susc[in_vac]) == 1.0, 'Vaccine effective in everyone (all_or_nothing)'
        mean = n_vac * (1 - efficacy)
        sd = np.sqrt(n_vac * efficacy * (1 - efficacy))
        assert (np.mean(rel_susc[in_vac]) - mean) / sd < tol, 'Incorrect mean susceptibility in vaccinated (all_or_nothing)'
    else:
        assert max(abs(rel_susc[in_vac] - (1 - efficacy))) < 0.0001, 'Relative susceptibility not 1-efficacy (leaky)'

    # run the simulation until sufficient cases
    old_cases = []
    for idx in range(1000):
        sim.run_one_step()
        susc = sim.people.sir.susceptible.uids
        cases = np.setdiff1d(in_trial, susc)
        if len(cases) > total_cases:
            break
        old_cases = cases

    if len(cases) > total_cases:
        cases = np.concatenate([old_cases, np.random.choice(np.setdiff1d(cases, old_cases), total_cases - len(old_cases), replace=False)])
    vac_cases = np.intersect1d(cases, in_vac)

    # check to see whether the number of cases are as expected
    p = v_frac * (1 - efficacy) / (1 - efficacy * v_frac)
    mean = total_cases * p
    sd = np.sqrt(total_cases * p * (1 - p))
    assert (len(vac_cases) - mean) / sd < tol, 'Incorrect proportion of vaccincated infected'

    # for all or nothing check that fully vaccinated did not get infected
    if not leaky:
        assert len(np.intersect1d(vac_cases, in_vac[rel_susc[in_vac] == 1.0])) == len(vac_cases), 'Not all vaccine cases amongst vaccine failures (all or nothing)'
        assert len(np.intersect1d(vac_cases, in_vac[rel_susc[in_vac] == 0.0])) == 0, 'Vaccine cases amongst fully vaccincated (all or nothing)'

    return sim


@sc.timer()
def test_sir_vaccine_leaky():
    return run_sir_vaccine(0.3, False)


@sc.timer()
def test_sir_vaccine_all_or_nothing():
    return run_sir_vaccine(0.3, True)


@sc.timer()
def test_products(do_plot=False):
    sc.heading('Testing products')

    pars = sc.objdict(
        n_agents = 5e3,
        start = 2000,
        stop = 2020,
        diseases = 'sis',
        networks = 'random',
    )

    dx_data = sc.dataframe(
        columns =
            ['disease', 'state', 'result', 'probability'],
        data = [
            ['sis', 'susceptible', 'positive', 0.01],
            ['sis', 'susceptible', 'negative', 0.99],
            ['sis', 'infected', 'positive', 0.95],
            ['sis', 'infected', 'negative', 0.05],
        ]
    )

    # Using built-in products
    vx_start = 2005
    my_vaccine = ss.simple_vx(efficacy=0.9)
    vaccination = ss.routine_vx(
        product = my_vaccine,  # Product object
        prob = 0.8,
        start_year = vx_start,
    )

    # Using custom products
    dx_start = 2010
    screening = ss.routine_screening(
        product = ss.Dx(df=dx_data),
        prob = 0.9,
        start_year = dx_start,
    )

    # Run the sim
    sim = ss.Sim(pars, interventions=[screening, vaccination])
    sim.run()

    # Checks
    dxres = sim.results.routine_screening
    sisres = sim.results.sis
    y = sim.t.yearvec
    pre_dx = y < dx_start
    post_dx = y > dx_start
    pre_vx = y < vx_start
    post_vx = y > vx_start
    assert dxres.n_screened[pre_dx].sum() == 0, 'Expected no one screened before intervention start'
    assert dxres.n_screened[post_dx].sum() > 0, 'Expected people screened after intervention start'
    assert dxres.n_dx[pre_dx].sum() == 0, 'Expected no one diagnosed before intervention start'
    assert dxres.n_dx[post_dx].sum() > 0, 'Expected people diagnosed after intervention start'
    assert sisres.new_infections[pre_vx].mean() > sisres.new_infections[post_vx].mean(), 'Expected vaccine to reduce prevalence'

    if do_plot:
        sim.plot()

    return sim

class StatefulVx(ss.Vx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_states(
            ss.BoolState('vaccinated'),  # True if a routine vaccine was ever delivered to this agent
        ),
        return

    def administer(self, people, uids):
        self.vaccinated[uids] = True
        self.results.n_vaccinated[self.ti] = len(uids)


@sc.timer()
def test_two_products():

    pars = sc.objdict(
        n_agents = 5e3,
        start = 2000,
        stop = 2020,
        diseases = 'sis',
        networks = 'random',
    )

    my_vaccine = StatefulVx()
    vx1 = ss.routine_vx(name='vx1', product = my_vaccine, prob=0.1,start_year = 2005, eligibility=lambda sim: (~sim.people.statefulvx.vaccinated).uids)
    vx2 = ss.routine_vx(name='vx2', product = my_vaccine, prob=0.2,start_year = 2005, eligibility=lambda sim: (~sim.people.statefulvx.vaccinated).uids)
    sim = ss.Sim(pars, interventions=[vx1,vx2])
    sim.run()

    assert (sim.people.vx1.vaccinated.sum()+sim.people.vx2.vaccinated.sum()) == sim.people.statefulvx.vaccinated.sum()

    return sim



@sc.timer()
def test_routine_delivery_window():
    """ RoutineDelivery should only deliver between start_year and end_year """
    sc.heading('Testing RoutineDelivery start/end window...')

    pars = make_delivery_pars(start=2000, stop=2020)
    start_year, end_year = 2008, 2012
    screening = ss.routine_screening(
        product    = make_dx(),
        prob       = 1.0,
        start_year = start_year,
        end_year   = end_year,
    )
    sim = ss.Sim(pars, interventions=screening).run()

    n_screened = sim.results.routine_screening.n_screened
    years      = sim.t.yearvec
    pre        = years < start_year
    post       = years > end_year

    assert n_screened[pre].sum() == 0, \
        f'Expected zero screenings before {start_year}, got {n_screened[pre].sum()}'
    assert n_screened[post].sum() == 0, \
        f'Expected zero screenings after {end_year}, got {n_screened[post].sum()}'
    in_window = (years >= start_year) & (years <= end_year)
    assert n_screened[in_window].sum() > 0, \
        f'Expected nonzero screenings inside [{start_year}, {end_year}]'

    return sim


@sc.timer()
def test_routine_delivery_prob_array():
    """ RoutineDelivery should accept a per-year prob array matching the year window """
    sc.heading('Testing RoutineDelivery prob-array scaling...')

    pars = make_delivery_pars(start=2000, stop=2020)
    years = np.arange(2005, 2016)               # 11 entries: 2005..2015
    probs = np.linspace(0.05, 0.95, len(years)) # matching ramp 0.05 -> 0.95
    screening = ss.routine_screening(
        product = make_dx(),
        prob    = probs,
        years   = years,
    )
    sim = ss.Sim(pars, interventions=screening).run()

    n_screened = sim.results.routine_screening.n_screened
    yearvec    = sim.t.yearvec
    early_mask = (yearvec >= 2005) & (yearvec <= 2007)
    late_mask  = (yearvec >= 2013) & (yearvec <= 2015)
    early_mean = n_screened[early_mask].mean()
    late_mean  = n_screened[late_mask].mean()

    assert late_mean > early_mean, \
        f'Expected screening to scale up with prob array; got early={early_mean}, late={late_mean}'
    # Late prob is ~19x early prob; require at least 3x to allow stochastic slack
    assert late_mean > 3 * early_mean, \
        f'Expected late screening >> early screening; got early={early_mean}, late={late_mean}'

    return sim


@sc.timer()
def test_routine_delivery_default_years():
    """ Omitting start/end_year should default to sim start/stop """
    sc.heading('Testing RoutineDelivery default year handling...')

    pars = make_delivery_pars(start=2010, stop=2014)
    screening = ss.routine_screening(product=make_dx(), prob=0.5)
    sim = ss.Sim(pars, interventions=screening).run()

    n_screened = sim.results.routine_screening.n_screened
    assert n_screened.sum() > 0, 'Expected screening to occur when no window specified'

    return sim


@sc.timer()
def test_campaign_delivery_single_year():
    """ CampaignDelivery should only deliver at the specified year """
    sc.heading('Testing CampaignDelivery single year...')

    pars = make_delivery_pars(start=2000, stop=2020)
    campaign_year = 2010
    screening = ss.campaign_screening(
        product = make_dx(),
        prob    = 0.5,
        years   = campaign_year,
    )
    sim = ss.Sim(pars, interventions=screening).run()

    n_screened = sim.results.campaign_screening.n_screened
    yearvec    = sim.t.yearvec
    on_year    = np.isclose(yearvec, campaign_year)
    off_year   = ~on_year

    assert n_screened[on_year].sum() > 0, \
        f'Expected screenings on {campaign_year}, got 0'
    assert n_screened[off_year].sum() == 0, \
        f'Expected no screenings off-year, got {n_screened[off_year].sum()}'

    return sim


@sc.timer()
def test_campaign_delivery_multi_year():
    """ CampaignDelivery should deliver at each specified year and nowhere else """
    sc.heading('Testing CampaignDelivery multi year...')

    pars = make_delivery_pars(start=2000, stop=2020)
    campaign_years = [2005, 2010, 2015]
    probs = [0.3, 0.5, 0.7]
    screening = ss.campaign_screening(
        product = make_dx(),
        prob    = probs,
        years   = campaign_years,
    )
    sim = ss.Sim(pars, interventions=screening).run()

    n_screened = sim.results.campaign_screening.n_screened
    yearvec    = sim.t.yearvec
    on_mask    = np.zeros_like(yearvec, dtype=bool)
    for y in campaign_years:
        on_mask |= np.isclose(yearvec, y)

    assert n_screened[~on_mask].sum() == 0, \
        f'Expected no screenings outside campaign years, got {n_screened[~on_mask].sum()}'
    for y in campaign_years:
        n_y = n_screened[np.isclose(yearvec, y)].sum()
        assert n_y > 0, f'Expected screenings in {y}, got {n_y}'

    return sim


@sc.timer()
def test_campaign_delivery_prob_length_mismatch():
    """ CampaignDelivery should reject prob/years length mismatch """
    sc.heading('Testing CampaignDelivery prob/years length validation...')

    pars = make_delivery_pars(start=2000, stop=2020)
    screening = ss.campaign_screening(
        product = make_dx(),
        prob    = [0.1, 0.2],  # length 2, but 3 years
        years   = [2005, 2010, 2015],
    )
    with pytest.raises(ValueError):
        ss.Sim(pars, interventions=screening).init()

    return screening


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    leaky  = test_sir_vaccine_leaky()
    a_or_n = test_sir_vaccine_all_or_nothing()
    prod   = test_products(do_plot=do_plot)
    prod2  = test_two_products()

    s_win    = test_routine_delivery_window()
    s_parr   = test_routine_delivery_prob_array()
    s_def    = test_routine_delivery_default_years()
    s_csing  = test_campaign_delivery_single_year()
    s_cmulti = test_campaign_delivery_multi_year()
    s_cmm    = test_campaign_delivery_prob_length_mismatch()

    T.toc()
