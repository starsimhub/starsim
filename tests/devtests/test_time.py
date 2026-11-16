"""
Test time: modules with timesteps of their own
"""
import sciris as sc
import starsim as ss
import starsim.library as ssl


@sc.timer()
def test_mixed_timesteps():
    sc.heading('Test that a disease and a sim with steps of a day, in days or years, give the same results')
    siskw = dict(dur_inf=ss.days(50), beta=ss.perday(0.01), waning=ss.perday(0.005))
    kw = dict(n_agents=1000, start='2001-01-01', stop='2001-07-01', networks='random', verbose=0)
    sims = []
    for sis_dt, sim_dt in [('year', 'year'), ('day', 'day'), ('day', 'year'), ('year', 'day')]:
        dts = dict(day=ss.days(1), year=ss.years(1/365))
        sis = ssl.AgentSIS(name='sis', dt=dts[sis_dt], **sc.dcp(siskw)) # Written as code, since a declared disease cannot yet have a timestep of its own
        sims.append(ss.Sim(dt=dts[sim_dt], diseases=sis, label=f'{sis_dt}-{sim_dt}', **kw))
    msim = ss.parallel(sims, workers=1) # Serially, since starting processes takes longer than these short runs

    summary = msim.summarize()['value'].unstack().astype(float) # A row per result, and a column per sim
    for key,vals in summary.iterrows():
        if vals.mean():
            ratio = vals.std(ddof=0)/vals.mean()
            assert ratio < 0.02, f'Result {key} differs between the sims: {ratio:n}'
    return msim


if __name__ == '__main__':
    test_mixed_timesteps()
