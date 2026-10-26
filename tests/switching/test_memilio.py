"""
Switching demo for the MEmilio seat: 400 counties × 6 ages with commuting and per-county NPIs, projected from priors.

Passes when: within 3x of 1.5 s; agreement as today.
Blocked today by: N18, the per-place hold, `priors=True` (the graph importer is done: test_interop_memilio.py).
Scaled down to 50 and 100 synthetic counties (the report's own synthetic Germany, regenerated here); the 400-county time is checked by the north-star script (phase 6).
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import timing

kw = dict(verbose=0)
ref_peak_50   = 11_561.6 # National peak of symptomatic people, MEmilio's MobilityGraph at 50 counties; switching/memilio.md, "The second week" §1 (memilio_50.npy)
ref_agree     = 0.002    # National curves within 0.2% of the peak at 50 counties; same section
v4_time_50    = 0.075    # Starsim's run at 50 counties (s), normalized to the reference machine (see timing.py), 2026-10-25
ref_time_50   = 0.17     # MEmilio's run at 50 counties (s); same section, table
ref_time_400  = 1.54     # MEmilio's run at 400 counties (s), i.e. its time grows with the number of edges; same section
ref_npi_ratio = 149_298/750_843 # Peak with local NPIs over the peak without, 50 counties; "The second week" §3
ref_study_p50 = 9_763    # Median national peak of symptomatic people, 10-run GraphParameterStudy at 50 counties; "The second week" §2

ages = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
split = [0.047, 0.091, 0.226, 0.338, 0.226, 0.072]
cm = np.array([[3.9547, 1.1002, 2.9472, 2.05, 0.3733, 0.0445], [0.3327, 3.5892, 1.236, 1.9208, 0.2681, 0.0161], [0.246, 0.7124, 5.6518, 3.2939, 0.2043, 0.0109],
               [0.1742, 0.8897, 3.3124, 4.5406, 0.4262, 0.0214], [0.0458, 0.1939, 0.5782, 1.3825, 1.473, 0.0704], [0.1083, 0.1448, 0.4728, 0.9767, 0.6266, 0.1724]]) # MEmilio's German baseline contacts, summed over locations
P = dict(beta=[0.03, 0.06, 0.06, 0.06, 0.09, 0.175], tE=3.335, tINS=1.865, tISy=[7.02625, 7.02625, 7.0665, 6.9385, 6.835, 6.775], tISev=[5., 5., 5.925, 7.55, 8.5, 11.],
         tICr=[6.95, 6.95, 6.86, 17.36, 17.1, 11.6], mu_RINS=[0.25, 0.25, 0.2, 0.2, 0.2, 0.2], mu_ISev=[0.0075, 0.0075, 0.019, 0.0615, 0.165, 0.225],
         mu_ICr=[0.075, 0.075, 0.075, 0.15, 0.3, 0.4], mu_D=[0.05, 0.05, 0.14, 0.14, 0.4, 0.6]) # Midpoints of the ranges in MEmilio's 2020_npis_sarscov2_wildtype_germany.py


def make_germany(n, k=20, seed=1):
    """ The report's synthetic Germany: n counties, gravity commuting to the k nearest, 15% commuting out, 10 seeded counties """
    rng = np.random.default_rng(seed)
    pop = rng.lognormal(np.log(150_000), 0.7, n)
    pop *= 83e6/pop.sum()
    x, y = rng.uniform(0, 600, n), rng.uniform(0, 800, n)
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(dist, np.inf)
    M = np.zeros((n, n))
    for i in range(n):
        near = np.argsort(dist[i])[:k]
        M[i, near] = pop[near]/np.maximum(dist[i, near], 5)**2
    M = 0.15*M/M.sum(1, keepdims=True)
    seeds = np.zeros(n)
    seeds[rng.choice(n, 10, replace=False)] = 1
    return sc.objdict(popsize=pop, x=x, y=y, M=M, seeds=seeds, names=[f'c{i:03d}' for i in range(n)])


def make_secir(d, beta=None, damping=True):
    """ The age-resolved SECIR of the vignette, with commuting between counties """
    A = lambda v: ss.by(age=list(np.broadcast_to(v, 6)))
    D = lambda v: ss.by(age=[ss.days(x) for x in np.broadcast_to(v, 6)])
    beta = beta if beta is not None else A(P['beta'])
    contacts = ss.timeseries({0: 1.0, 29: 1.0, 30: 0.5}, mode='cosine') if damping else 1.0 # Contacts halved from day 30
    S, E, INS, ISy, ISev, ICr, R = ss.states(7)
    secir = ss.Disease(
        infection = S >> E @ beta * contacts * (INS + 0.2*ISy),
        onset     = ss.Transition(E, INS, dur=D(P['tE'])),
        symptoms  = ss.Transition(INS, [ISy, R], dur=D(P['tINS']), p=[A(1 - np.array(P['mu_RINS'])), A(P['mu_RINS'])]),
        severe    = ss.Transition(ISy, [ISev, R], dur=D(P['tISy']), p=[A(P['mu_ISev']), A(1 - np.array(P['mu_ISev']))]),
        critical  = ss.Transition(ISev, [ICr, R], dur=D(P['tISev']), p=[A(P['mu_ICr']), A(1 - np.array(P['mu_ICr']))]),
        recovery  = ss.Transition(ICr, R, dur=D(P['tICr']), p=A(1 - np.array(P['mu_D']))),
        death     = ss.Transition(ICr, ss.null, dur=D(P['tICr']), p=A(P['mu_D'])),
        init_n    = dict(E=ss.by(np.outer(np.full(6, 50), d.seeds), dims=['age', 'patch'])),
        name      = 'secir',
    )
    return secir


def stratify(target, d):
    """ Ages, and counties with MEmilio's commuters away half the day """
    target.stratify(age=ages, split=split)
    target.stratify(patch=ss.geo.geography(names=d.names, pop=d.popsize, x=d.x, y=d.y), mobility=ss.geo.Mobility(matrix=0.5*d.M, kind='commute'))
    return target


def make_sim(d, secir=None, on_sim=False, **kwargs):
    """ Stratified on the disease, or (on_sim=True) on the sim """
    secir = secir or stratify(make_secir(d), d)
    sim = ss.Sim(secir, networks=ss.contacts(cm), total_pop=d.popsize.sum(), stop=ss.days(90), **kwargs, **kw)
    return stratify(sim, d) if on_sim else sim


def timed_run(n):
    sim = make_sim(make_germany(n)).init()
    seconds, _ = timing.timed(sim.run, repeats=1)
    return sim, seconds


def test_memilio_agreement():
    sc.heading('MEmilio: 50 counties × 6 ages with commuting agree with MEmilio')
    sim, _ = timed_run(50)
    peak = sim.results.secir.ISy.values.max()
    assert abs(peak/ref_peak_50 - 1) < ref_agree, f'National peak {peak:,.1f} differs from MEmilio\'s {ref_peak_50:,.1f} by more than {ref_agree:.1%}'
    return sim


def test_memilio_speed():
    sc.heading('MEmilio: speed at 50 counties, and growth with the number of counties')
    timed_run(50) # Warm up
    _, t50 = timed_run(50)
    _, t100 = timed_run(100)
    assert t100/t50 < 3, f'Doubling the counties multiplied the time by {t100/t50:.1f}; MEmilio grows with the edges ({ref_time_50} s → {ref_time_400} s for 8x the counties)'
    timing.check(t50, v4_time_50, '50 counties', other=(ref_time_50, 'MEmilio'))
    return t50, t100


def test_memilio_npi():
    sc.heading('MEmilio: dynamic NPIs, each county on its own')
    d = make_germany(50)
    beta = ss.par(ss.by(age=P['beta']))
    secir = stratify(make_secir(d, beta=beta, damping=False), d)
    incidence = secir.symptoms.incidence(over=ss.days(7), by='patch') # A week's new symptomatic cases in each county, per person
    npi = ss.modify(beta, multiply=0.6, where=incidence > 200e-5, hold=ss.days(14))
    sims = [make_sim(d, secir, interventions=ints).run() for ints in [None, npi]]
    ratio = sims[1].results.secir.ISy.values.max()/sims[0].results.secir.ISy.values.max()
    assert abs(ratio - ref_npi_ratio) < 0.1, f'Local NPIs cut the peak to {ratio:.0%} of baseline, against {ref_npi_ratio:.0%} with the hand-written rule'
    return sims


def test_memilio_projection():
    sc.heading('MEmilio: a parameter study drawn from priors, with percentiles per county')
    d = make_germany(50)
    scale = ss.par(1.0, prior=ss.uniform(0.75, 1.25), name='scale') # MEmilio's ParameterDistributionUniform, ±25% around the midpoints
    sim = make_sim(d, make_secir(d, beta=scale*ss.by(age=P['beta'])), on_sim=True) # sim.stratify(), the spelling in best-practices/metapopulation.md
    # The graph importer (ss.load of a MobilityGraph or GraphParameterStudy script) needs MEmilio itself, so is tested in test_interop_memilio.py
    msim = ss.parallel(sim, n_runs=10, priors=True)
    isy = msim.results.secir.ISy
    q = isy.select(patch='c005').quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
    p05, p50, p95 = isy.quantiles([0.05, 0.5, 0.95]).max(1)
    assert q.shape[0] == 5 and p05 < p50 < p95, 'The runs should differ, since each draws its own beta'
    peaks = np.asarray(isy.values).max(axis=1)
    order = np.argsort(msim.run_pars.scale.values)
    assert np.all(np.diff(peaks[order]) > 0), 'Each run should have the scale it drew: the peak rises with it'
    # MEmilio's study (median peak 9,763) draws every parameter, not only beta, so its median is not this one's: the peak is
    # steeply convex in the scale (3,164 at 0.75, 11,568 at 1, 49,583 at 1.25), so ten draws give a median of 6,000 to 21,000; MEmilio's own ten-run median varies from 9,763 to 16,459 between repeats (W12)
    return msim


if __name__ == '__main__':
    sim  = test_memilio_agreement()
    t    = test_memilio_speed()
    npi  = test_memilio_npi()
    msim = test_memilio_projection()
