"""
Run tests of vaccines and products
"""
import sciris as sc
import numpy as np
import starsim as ss


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


if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    leaky  = test_sir_vaccine_leaky()
    a_or_n = test_sir_vaccine_all_or_nothing()
    prod   = test_products(do_plot=do_plot)

    T.toc()
