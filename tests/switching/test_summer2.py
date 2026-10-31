"""
Switching demo for the summer2 seat: the TB model, imported live, fitted by NUTS with time-varying detection.

Passes when: NUTS on the JAX path with no divergences.
Blocked today by: N2, the summer2 importer (N13), the JAX log-density.
The fit is scaled down to the TB model without age groups, with a short chain.
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.fit

ref_n_I = {1990: 3150.376, 2010: 1527.021, 2025: 1477.900} # switching/summer2.md, "The second week" §1: summer2's active TB (all ages) from tb_summer2.py
ref_notifications = {1990: 2341.466, 2010: 2263.245, 2024: 2194.623} # Same section: summer2's notifications (v4: 2,341, 2,260, 2,195)
mixing = np.array([ # Same section: the 6 × 6 contact matrix of the AuTuMN-style TB model
    [3.0, 1.0, 1.5, 1.0, 0.5, 0.3],
    [1.0, 8.0, 2.0, 1.5, 0.8, 0.4],
    [1.5, 2.0, 5.0, 2.5, 1.5, 0.6],
    [1.0, 1.5, 2.5, 4.0, 2.0, 0.8],
    [0.5, 0.8, 1.5, 2.0, 3.0, 1.2],
    [0.3, 0.4, 0.6, 0.8, 1.2, 2.0]]) / 4.0
lik = dict(detection=ss.poisson())


def sigmoid(t, xs=(1950, 1990, 2020), ys=(0, 0.5, 1), curvature=16):
    """ summer2's get_sigmoidal_interpolation_function(), which v4 does not have, sampled every 0.1 years """
    f = lambda u: 1/(1 + np.exp(-curvature*(u - 0.5)))
    i = np.clip(np.searchsorted(xs, t, side='right') - 1, 0, len(xs) - 2)
    u = np.clip((t - np.take(xs, i))/(np.take(xs, i + 1) - np.take(xs, i)), 0, 1)
    return np.take(ys, i) + (np.take(ys, i + 1) - np.take(ys, i))*(f(u) - f(0))/(f(1) - f(0))


def make_tb(ages=True, data=None):
    """ The TB model as a summer2 user writes it in v4: detection is a fitted maximum times a scale-up over time """
    t = np.round(np.arange(1950, 2025.001, 0.1), 3)
    scaleup = ss.timeseries(dict(zip(t, sigmoid(t))), mode='linear')
    beta = ss.par(10.0, bounds=[2, 30])
    detect_max = ss.par(1.5, bounds=[0.2, 4])
    S, E, L, I, T, R = ss.states('S E L I T R')
    tb = ss.Disease(infection=S >> E @ beta*I, reinfection_L=L >> E @ beta*0.21*I, reinfection_R=R >> E @ beta*0.21*I,
                    stabilisation=E >> L @ 2.0, early_activation=E >> I @ 0.2, late_activation=L >> I @ 0.002, self_recovery=I >> R @ 0.2,
                    detection=I >> T @ detect_max*scaleup, treatment_end=T >> R @ 2.0, tb_death=ss.death(I, rate=0.2),
                    death=ss.death(rate=1/70), birth=ss.birth(S, rate=0.02), init_n=dict(I=1000), name='tb')
    if ages:
        tb.stratify(age=[0, 5, 15, 35, 50, 70], split=[1/6]*6, adjust={'early_activation': [3, 1.5, 1, 1, 1, 1], 'rel_trans': [0.1, 0.2, 1, 1, 1, 1]})
    return ss.Sim(tb, networks=ss.contacts(mixing) if ages else None, total_pop=1e6, start=1950, stop=2025, unit='year', data=data, verbose=0)


def notifications():
    """ Notifications from 2000 to 2024, as whole numbers, from a run at the true values (those in a year are at the time point that ends it, as data by calendar year is read) """
    res = make_tb(ages=False).run().results.tb.detection
    return pd.DataFrame(dict(year=range(2000, 2025), detection=np.round(np.asarray(res.values)[51:76])))


def jax_logdensity(sim, x):
    """ The log-density that NUTS samples in JAX, and the one from a run of the sim, at the same values """
    fitted = ss.gradients.find_pars(sim)
    set_pars = lambda x: sim.restart(pars=dict(zip(fitted.keys(), map(float, x))))
    density = ss.gradients.Density(sim, fitted, x, set_pars, likelihood=lik)
    jax = ss.gradients.jax()
    with jax.enable_x64(True):
        in_jax = float(density(jax.numpy.asarray(x)))
    set_pars(x)
    from_sim = ss.fit.loglik(sim.run(), likelihood=lik) + sum(ss.fit.logprior(p, v) for p,v in zip(fitted.values(), x))
    return in_jax, from_sim


def test_summer2_tb():
    sc.heading('summer2: the TB model by age, against summer2')
    res = make_tb().run().results.tb
    for year,ref in ref_n_I.items():
        assert abs(res.I.values[year - 1950] - ref) < 5, f'Active TB in {year}: {res.I.values[year - 1950]:.1f} against summer2\'s {ref}'
    for year,ref in ref_notifications.items():
        assert np.isclose(res.detection.values[year - 1950], ref, rtol=0.005), f'Notifications in {year}: {res.detection.values[year - 1950]:.0f} against {ref:.0f}'
    return res


@pytest.mark.xfail(reason='summer2 1.3.6 needs JAX 0.4 (it imports jax.core.call), so it cannot be installed with Starsim and a live model cannot be imported in the same process; the same model written by starsim/converters/summer2.py in summer2\'s environment imports with nothing missing and matches make_tb(ages=False) to 1e-13 (and tests/devtests/io/test_io_summer2_atomica.py checks it by age against summer2)', strict=False)
def test_summer2_import_live():
    sc.heading('summer2: the live model, with its parameters and its sigmoid, imported')
    summer2 = pytest.importorskip('summer2')
    from summer2.parameters import Parameter
    from summer2.functions.time import get_sigmoidal_interpolation_function
    m = summer2.CompartmentalModel(times=[1950.0, 2025.0], compartments=list('SELITR'), infectious_compartments=['I'], timestep=1.0)
    m.set_initial_population(distribution={'S': 1e6 - 1000, 'I': 1000})
    beta = Parameter('contact_rate')
    m.add_infection_frequency_flow('infection', beta, 'S', 'E')
    m.add_infection_frequency_flow('reinfection_L', beta*Parameter('rr_reinf'), 'L', 'E')
    m.add_infection_frequency_flow('reinfection_R', beta*Parameter('rr_reinf'), 'R', 'E')
    for name,rate,src,dst in [('stabilisation', 2.0, 'E', 'L'), ('early_activation', 0.2, 'E', 'I'), ('late_activation', 0.002, 'L', 'I'), ('self_recovery', 0.2, 'I', 'R'), ('treatment_end', 2.0, 'T', 'R')]:
        m.add_transition_flow(name, rate, src, dst)
    m.add_transition_flow('detection', get_sigmoidal_interpolation_function([1950.0, 1990.0, 2020.0], [0.0, 0.5, 1.0])*Parameter('detect_max'), 'I', 'T')
    m.add_death_flow('tb_death', 0.2, 'I')
    m.add_universal_death_flows('universal_death', death_rate=1/70)
    m.add_crude_birth_flow('birth', 0.02, 'S')
    sim = ss.import_from(m, pars=dict(contact_rate=10.0, rr_reinf=0.21, detect_max=1.5), verbose=0)
    assert not sim.imported.incomplete, f'Not translated: {sim.imported.incomplete}'
    ours = np.asarray(make_tb(ages=False).run().results.tb.detection.values)
    theirs = np.asarray(sim.run().results.summer2.detection.values) # The flow in each year, as make_tb's (summer2's own output is the mean of the rates at either end of the year)
    assert np.allclose(theirs[:len(ours)], ours, rtol=1e-4)
    return sim


def test_summer2_nuts():
    sc.heading('summer2: NUTS on the JAX path, with detection a fitted maximum times a scale-up')
    sim = make_tb(ages=False, data=notifications())
    runner = sim.new().init()
    assert runner.kernel_only(), 'Expected the sim to run whole in the kernels'
    in_jax, from_sim = jax_logdensity(runner, np.array([10.2, 1.53]))
    assert np.isclose(in_jax, from_sim, rtol=1e-6), f'The JAX log-density ({in_jax:.6g}) differs from the sim\'s ({from_sim:.6g})'
    with ss.options.context(warnings='error'): # Falling back to Python in the loop is a failure here
        fit = sim.fit(method='nuts', likelihood=lik, warmup=10, draws=10, chains=2, verbose=0) # About 15 s with JAX's cache warm
    assert fit.diagnostics.divergences == 0, f'{fit.diagnostics.divergences} draws diverged'
    assert np.isclose(fit.pars.detect_max, 1.5, rtol=0.05) and np.isclose(fit.pars.beta, 10, rtol=0.05)
    return fit


if __name__ == '__main__':
    res = test_summer2_tb()
    sim = test_summer2_import_live()
    fit = test_summer2_nuts()
