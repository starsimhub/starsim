"""
Test the Timeline object
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl

small = 100
medium = 1000
sc.options(interactive=False)
# ss.options.warnings = 'error' # For additional debugging


@sc.timer()
def test_mixed_timesteps():
    sc.heading('Test behavior of different combinations of timesteps')
    SIS = ssl.AgentSIS # Written as code, since a declared disease cannot yet have a timestep of its own

    siskw = dict(dur_inf=ss.datedur(days=50), beta=ss.perday(0.01), waning=ss.perday(0.005))
    kw = dict(n_agents=medium, start='2001-01-01', stop='2001-07-01', networks='random', verbose=0)

    print('Year-year')
    sis1 = SIS(name='sis', dt=ss.years(1/365), **sc.dcp(siskw))
    sim1 = ss.Sim(dt=ss.years(1/365), diseases=sis1, label='year-year', **kw)

    print('Day-day')
    sis2 = SIS(name='sis', dt=ss.days(1), **sc.dcp(siskw))
    sim2 = ss.Sim(dt=ss.days(1), diseases=sis2, label='day-day', **kw)

    print('Day-year')
    sis3 = SIS(name='sis', dt=ss.days(1), **sc.dcp(siskw))
    sim3 = ss.Sim(dt=ss.years(1/365), diseases=sis3, label='day-year', **kw)

    print('Year-day')
    sis4 = SIS(name='sis', dt=ss.years(1/365), **sc.dcp(siskw))
    sim4 = ss.Sim(dt=ss.days(1), diseases=sis4, label='year-day', **kw)

    msim = ss.parallel(sim1, sim2, sim3, sim4, parallel=False) # Serially, since starting processes takes longer than these short runs

    # Check that all results are close
    threshold = 0.02
    summary = msim.summarize()['value'].unstack().astype(float) # A row per result, and a column per sim
    for key,vals in summary.iterrows():
        if vals.mean():
            ratio = vals.std(ddof=0)/vals.mean()
            assert ratio < threshold, f'Result {key} exceeds threshold: {ratio:n} > {threshold}'
            print(f'✓ Result {key} within threshold: {ratio:n} < {threshold}')
    return msim


@sc.timer()
def test_units(do_plot=False):
    sc.heading('Test behavior of year vs day units')

    sis = ssl.AgentSIS( # Starsim 3's SIS, with waning immunity
        name = 'sis',
        beta = ss.prob(0.05, 'day'),
        init_prev = ss.bernoulli(p=0.1),
        dur_inf = ss.lognorm_ex(mean=ss.datedur(days=10)),
        waning = ss.perday(0.05),
        imm_boost = 1.0,
    )

    rnet = ss.RandomNet(
        n_contacts = 10,
        dur = 0, # Note; network edge durations are required to have the same unit as the network
    )

    pars = dict(
        diseases = sis,
        networks = rnet,
        n_agents = small,
    )

    sims = sc.objdict()
    sims.y = ss.Sim(pars, label='Year', start=2000, stop=2002, dt=1/365, verbose=0)
    sims.d = ss.Sim(pars, label='Day', start='2000-01-01', stop='2002-01-01', dt=ss.days(1), verbose=0)

    for sim in sims.values():
        sim.run()
        if do_plot:
            sim.plot()

    # Check that results match to within stochastic uncertainty
    rtol = 0.05
    vals = [sim.summary.value.sis_cum_infections for sim in [sims.y, sims.d]]
    assert np.isclose(*vals, rtol=rtol), f'Values for cum_infections do not match ({vals})'

    return sims


# Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer('\nTotal time')

    o6 = test_mixed_timesteps()
    o7 = test_units(do_plot)

    T.toc()
