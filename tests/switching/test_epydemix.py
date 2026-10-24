"""
Switching demo for the epydemix seat: COVID in Kenya, seasonal × time-varying `beta`, fitted, projected, with an age-targeted vaccine.

Passes when: the filter peaks at the truth; `sir.age < 5` selects 6.9M.
Blocked today by: N1, N3, ABC-SMC, `fit.project()`, US states.
Scaled down to an SIR in a million people for the fit, and to four days for the vaccine.
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.fit

ref_peak = 0.5446 # switching/epydemix.md, "The second week" §1: epydemix's peak share infected in the Kenya tutorial (v4: 0.5448)
ref_peak_day = 40 # Same section: epydemix's day of the peak (v4: 39)
ref_final = 0.9963 # Same section: epydemix's final size (v4: 0.9959)
ref_under5 = 6_938_210 # Same report, §4: under-5s in Kenya in 2020 (UN WPP, as ss.People(1).age < 5 selects them)
true_b = 0.5 # The value the filter's data are drawn at (§2, the 25-line repro)
kenya = dict(start='2019-12-01', stop='2020-04-02', use_aging=False, verbose=0)


def make_kenya(beta=0.04):
    """ The epydemix tutorial's SIR, by age, from v4's bundled data """
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=0.1), init_n=dict(I=10), name='sir')
    sir.stratify(age=[0, 5, 20, 50, 65])
    return sir


def make_seasonal(data=None):
    """ An SIR with beta a fitted value times a season times a mobility series, as in epydemix's COVID tutorial """
    days = pd.date_range('2020-02-23', '2020-05-31')
    mobility = ss.timeseries(dict(zip(days.strftime('%Y-%m-%d'), np.linspace(1.0, 0.3, len(days)))))
    b = ss.par(true_b, bounds=[0.1, 1])
    season = ss.seasonal(mean=0.9, amplitude=0.1/0.9, peak=14/365)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=b*season*mobility * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init_n=dict(I=100), name='sir')
    return ss.Sim(sir, total_pop=1e6, unit='day', start='2020-02-23', stop='2020-05-31', solver='binomial', data=data, verbose=0)


def weekly_data():
    """ Weekly new infections from one run at the true value """
    dates = pd.date_range('2020-03-01', periods=13, freq='W')
    truth = make_seasonal(data=ss.simdata(pd.DataFrame({'new_infections': 0.0}, index=dates))).run()
    weekly = ss.results.compare(truth)['new_infections'][1]
    return ss.simdata(pd.DataFrame({'new_infections': np.round(weekly)}, index=dates))


def test_epydemix_kenya():
    sc.heading('epydemix: the Kenya tutorial, 100 chain-binomial runs, against epydemix')
    sim = ss.Sim(make_kenya(), networks=ss.contacts('Kenya'), people=ss.people('Kenya', year=2020), solver='binomial', **kenya)
    res = sim.run(n_runs=100).results.sir
    I, R = [np.asarray(res[key].values).mean(axis=0) for key in ['I', 'R']]
    N = I[0] + R[0] + np.asarray(res.S.values).mean(axis=0)[0]
    assert abs(I.max()/N - ref_peak) < 0.005, f'Peak {I.max()/N:.2%} against epydemix\'s {ref_peak:.2%}'
    assert abs(I.argmax() - ref_peak_day) <= 2, f'Peak on day {I.argmax()} against {ref_peak_day}'
    assert abs(R[-1]/N - ref_final) < 0.002, f'Final size {R[-1]/N:.2%} against {ref_final:.2%}'
    return res


def test_epydemix_vaccine_by_age():
    sc.heading('epydemix: a vaccine for the under-5s and the over-50s, written on the disease\'s age groups')
    def delivered(eligible):
        sir = make_kenya(beta=0)
        vax = ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=1.0, at=['2019-12-02'], where=eligible(sir))
        sim = ss.Sim(sir, people=ss.people('Kenya', year=2020), interventions=vax, solver='ode', **sc.mergedicts(kenya, stop='2019-12-05')).run()
        return np.nansum(sim.results.vax.delivered.values)
    under5 = delivered(lambda sir: sir.age < 5)
    assert np.isclose(under5, ref_under5, rtol=0.01), f'sir.age < 5 vaccinated {under5:,.0f}, not {ref_under5:,}'
    over50 = delivered(lambda sir: sir.age >= 50)
    assert np.isclose(over50, delivered(lambda sir: ss.People(1).age >= 50), rtol=0.01), f'sir.age >= 50 vaccinated {over50:,.0f}'
    return under5, over50


def test_epydemix_filter():
    sc.heading('epydemix: the particle filter peaks at the true value for a seasonal × mobility beta')
    sim = make_seasonal(data=weekly_data()).init()
    filt = ss.fit.ParticleFilter(sim, likelihood=dict(new_infections=ss.poisson()))
    loglik = {}
    for b in [0.4, true_b, 0.6]:
        sim.restart(pars=dict(b=b), rand_seed=0)
        loglik[b] = filt(n_particles=100, seed=0)
    assert max(loglik, key=loglik.get) == true_b, f'Expected the filter to peak at b = {true_b}: {loglik}'
    return loglik


def test_epydemix_posterior_projection():
    sc.heading('epydemix: a posterior with no likelihood, then projections from it')
    sim = make_seasonal(data=weekly_data())
    fit = sim.calibrate(method='abc-smc', particles=50, generations=3, verbose=0) # A distance (wmape), no likelihood
    lo, hi = np.quantile(fit.samples.b, [0.025, 0.975])
    assert lo <= true_b <= hi, f'Expected the posterior of b ({lo:.3f} to {hi:.3f}) to cover {true_b}'
    vaccine = ss.vaccinate(ss.Vaccine(efficacy=0.9), capacity=ss.perday(20_000), start='2020-04-01')
    proj = fit.project(dict(baseline=None, vaccine=vaccine), n=20) # One run per posterior draw
    df = proj.to_df(quantiles=[0.05, 0.5, 0.95])
    assert set(df.scenario) == {'baseline', 'vaccine'}, f'The draws should be runs of each scenario, not scenarios: {df.scenario.unique()[:4]}'
    peak = df[(df.result == 'I') & (df.scenario == 'baseline')].groupby('quantile').value.max()
    assert peak[0.05] < peak[0.95], f'The quantiles should be over the draws: {peak.to_dict()}'
    return proj


@pytest.mark.xfail(reason='The US states\' data are not published by starsim_data yet, so they cannot be downloaded (tests/test_data.py tests them from a copy of its repository)', strict=False)
def test_epydemix_us_state():
    sc.heading('epydemix: Massachusetts by name, as load_epydemix_population() gives it')
    people = ss.people('United States:Massachusetts', year=2020)
    sim = ss.Sim(make_kenya(), networks=ss.contacts('United States:Massachusetts'), people=people, solver='ode', **kenya).run()
    N = sim.results.sir.S[0] + sim.results.sir.I[0]
    assert 6.5e6 < N < 7.5e6, f'Expected Massachusetts\' 7.0 million people (2020 census), not {N:,.0f}'
    return sim


if __name__ == '__main__':
    res    = test_epydemix_kenya()
    vax    = test_epydemix_vaccine_by_age()
    loglik = test_epydemix_filter()
    proj   = test_epydemix_posterior_projection()
    state  = test_epydemix_us_state()
