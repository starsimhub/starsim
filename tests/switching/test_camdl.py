"""
Switching demo for the camdl seat: the polio model with SIAs over `patch=`, fitted by IF2.

Passes when: peak 130,643 on day 145, as camdl's ODE; unit slips caught.
Blocked today by: N28.
The model is camdl's polio_spatial_5 (5 patches, gravity importation, three SIA rounds per patch); the 774-patch build (N28) is checked by the north-star script (phase 6), not here.
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

ref_peak = 130_643 # Peak total I in camdl's ODE and an independent SciPy ODE: switching/camdl.md, "The second week" §3
ref_peak_day = 145 # Day of that peak: same section
ref_infections = 950_610 # Total infections over 3 years without SIAs, camdl's ODE: same section
ref_infections_sia = 863_920 # The same with SIAs: same section
ref_cases = [73, 267, 1060, 1409, 983, 497, 156, 18, 14, 6, 5, 1] # 12 weeks of cases from `camdl simulate sir.camdl --obs` (syn_obs.tsv): switching/camdl.md §1
ref_beta = [0.348, 0.523] # 95% interval of beta from camdl's PMMH, which v4's PMCMC matches: switching/camdl.md §1
ref_gamma = [0.062, 0.303] # The same for gamma

patches = ['north', 'south', 'east', 'west', 'center']
N0 = np.array([150_000, 200_000, 120_000, 100_000, 500_000])
R0 = np.array([2.5, 2.2, 1.9, 2.0, 3.0])
W = np.array([[0.00, 0.10, 0.05, 0.05, 0.15], [0.10, 0.00, 0.10, 0.05, 0.10], [0.05, 0.10, 0.00, 0.10, 0.10],
              [0.05, 0.05, 0.10, 0.00, 0.15], [0.15, 0.10, 0.10, 0.15, 0.00]])
sia_days = dict(north=[180, 545, 910], south=[190, 555, 920], east=[200, 565, 930], west=[200, 565, 930], center=[185, 550, 915])


def make_polio(sia=False):
    """ The polio model by hand, in camdl's words: SEIR by patch, with importation as contacts between patches """
    gamma, kappa = 0.1, 0.01
    S, E, I, R, V = ss.states(5)
    seir = ss.Disease(
        infection   = S >> E @ ss.perday(1) * I,
        progression = E >> I @ ss.perday(0.2),
        recovery    = I >> R @ ss.perday(gamma),
        waning      = V >> S @ ss.perday(0.003),
        init_n      = dict(I=ss.by(patch=[5, 0, 0, 0, 0])),
        name        = 'seir',
    )
    seir.stratify(patch=patches, kind='space', split=N0/N0.sum())
    if sia: # camdl's transfer(S -> V), in each patch on its own days
        sia = [ss.vaccinate(move=seir.S >> seir.V, at=days, coverage=ss.by(patch=[0.8*(q == p) for q in patches]), name=f'sia_{p}') for p, days in sia_days.items()]
    return ss.Sim(seir, networks=ss.contacts(patch=np.diag(R0*gamma) + kappa*W), interventions=sia or None,
                  total_pop=N0.sum(), unit='day', stop=3*365, solver='ode', verbose=0)


def test_camdl_polio():
    sc.heading('The polio model over patch=, without and with SIAs')
    res = make_polio().run().results.seir
    n_I = res.n_I.values
    assert np.isclose(n_I.max(), ref_peak, rtol=1e-3), f'Peak {n_I.max():,.0f}, camdl {ref_peak:,}'
    assert n_I.argmax() == ref_peak_day, f'Peak on day {n_I.argmax()}, camdl {ref_peak_day}'
    assert np.isclose(res.infection.values.sum(), ref_infections, rtol=1e-3)
    with_sia = make_polio(sia=True).run().results.seir
    assert np.isclose(with_sia.infection.values.sum(), ref_infections_sia, rtol=1e-3), f'With SIAs {with_sia.infection.values.sum():,.0f}, camdl {ref_infections_sia:,}'
    return res


def test_camdl_units():
    sc.heading('The unit slips that camdl\'s checker catches')
    slips = dict(
        dur_for_rate    = lambda S, I, R: [S >> I @ ss.days(0.4) * I, I >> R @ ss.perday(0.15)],
        unitless_rate   = lambda S, I, R: [S >> I @ ss.perday(0.4) * I, I >> R @ ss.perday(0.1) * ss.days(5)],
    )
    for key, slip in slips.items():
        S, I, R = ss.states(3)
        with pytest.raises(ss.CodedError): # When the transitions or the disease are built, or at init
            infection, recovery = slip(S, I, R)
            sir = ss.Disease(infection=infection, recovery=recovery, init_n=dict(I=10), name='sir')
            ss.Sim(sir, total_pop=10_000, unit='day', stop=84, verbose=0).init()
            print(f'Not caught: {key}')

    # A count for a share is density-dependent transmission, which is a model of its own, so it is a note; at this size everyone is infected within the first day (before, the substeps did not follow the growth and the run stopped on a NaN)
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ ss.perday(0.4) * I.n, recovery=I >> R @ ss.perday(0.15), init_n=dict(I=10), name='sir')
    sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=84, verbose=0).run()
    assert np.isclose(sim.results.sir.n_I[1], 10_000*np.exp(-0.15), rtol=1e-3) # Everyone infected at once, then a day of recovery
    assert any('density-dependent' in str(note) for note in sim.notes)
    return slips


def test_camdl_if2():
    sc.heading('Maximum likelihood by iterated filtering, as camdl fit run does')
    beta  = ss.par(ss.perday(0.3), bounds=[0.05, 1.0], prior=ss.lognorm_im(mean=-1.0, sigma=0.5))
    gamma = ss.par(ss.perday(0.1), bounds=[0.01, 0.5], prior=ss.lognorm_im(mean=-2.0, sigma=0.5))
    S, I, R = ss.states(3)
    infection = S >> I @ beta * I
    cases = ss.outcome(0.6 * infection, likelihood=ss.nbinom(n=10))
    sir = ss.Disease(infection, I >> R @ gamma, cases, init_n=dict(I=10), name='sir')
    data = pd.DataFrame(dict(time=7*np.arange(1, 13), cases=ref_cases))
    sim = ss.Sim(sir, total_pop=10_000, unit='day', stop=ss.weeks(12), solver='binomial', data=data, verbose=0)
    fit = sim.calibrate(method='if2')
    beta, gamma = fit.pars.beta.value, fit.pars.gamma.value # Per day
    assert ref_beta[0] < beta < ref_beta[1], f'IF2 beta {beta:.3f}, outside camdl\'s PMMH interval {ref_beta}'
    assert ref_gamma[0] < gamma < ref_gamma[1], f'IF2 gamma {gamma:.3f}, outside camdl\'s PMMH interval {ref_gamma}'
    assert fit.converged, fit.why # The chains, from different starts, agree
    return fit


if __name__ == '__main__':
    polio = test_camdl_polio()
    units = test_camdl_units()
    fit   = test_camdl_if2()
