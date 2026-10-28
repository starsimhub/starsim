"""
Test the kernels: that a run taken in one call matches the same run taken step by step
"""

# %% Imports and settings
import time
import itertools
from unittest import mock
import pytest
import numpy as np
import sciris as sc
import starsim as ss
from test_dimensions import make_malaria, make_tb

kw = dict(verbose=0)


def make_sir():
    """ An SIR model with reported cases """
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4) * I
    recovery  = I >> R @ ss.perday(0.15)
    cases     = ss.outcome(0.6 * infection)
    return ss.Disease(infection, recovery, cases, init_n=dict(I=10), name='sir')


def make_seir():
    """ Seasonal transmission, a dwell time in stages, and a branch """
    S, E, I, R, D = ss.states('S E I R D')
    seir = ss.Disease(
        infection = ss.Transition(S, E, rate=ss.seasonal(0.4, amplitude=0.3, peak=1) * I),
        latency   = ss.Transition(E, I, dur=ss.erlang(mean=5, k=3)),
        outcome   = ss.Transition(I, [R, D], dur=8, p=[None, 0.1]),
        init      = dict(I=0.01),
        name      = 'seir',
    )
    return seir


def make_plain(beta, **kwargs):
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=8), name='sir', **kwargs)


def make_ages():
    """ Three age groups with a contact matrix, and susceptibility that differs by age """
    return make_plain(0.05, init=dict(I=0.01)).stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2], adjust=dict(infection=[0.5, 1.0, 1.5]))


def make_towns():
    """ Three towns with commuting between them """
    towns = ss.geo.geography(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    return make_plain(0.3, init_n=dict(I=ss.by(patch=[50, 0, 0]))).stratify(patch=towns, mobility=ss.geo.gravity(k=2e-4))


def make_commute():
    """ Three age groups with a contact matrix within each of three towns, with commuting between them, and aging """
    return make_towns().stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2])


def make_delay():
    """ A fixed latent period, a conveyor, with a death rate acting on it as well """
    S, E, I, R = ss.states('S E I R')
    return ss.Disease(infection=ss.Transition(S, E, rate=0.4 * I), latency=ss.Transition(E, I, dur=ss.constant(ss.days(5))),
                      recovery=ss.Transition(I, R, dur=ss.days(8)), death=ss.death([E, I], rate=ss.perday(0.02)), init_prev=0.01, name='delay')


models = sc.objdict(
    sir     = (make_sir,     dict(total_pop=10_000, unit='day', stop=ss.weeks(12))),
    seir    = (make_seir,    dict(unit='day', stop=200)),
    ages    = (make_ages,    dict(unit='day', stop=100, networks=ss.contacts(np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])))),
    towns   = (make_towns,   dict(unit='day', stop=150, total_pop=100_000)),
    commute = (make_commute, dict(unit='day', stop=150, total_pop=100_000, use_aging=True, networks=ss.contacts(np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])))), # Aging, which a model without births or deaths has only if asked for
    malaria = (make_malaria, dict(unit='day', stop=200, total_pop=10_000)),
    tb      = (make_tb,      dict(stop=30, total_pop=100_000)), # Has a transfer, so is stepped
    delay   = (make_delay,   dict(unit='day', stop=150)),
)


def run(key, how, **kwargs):
    """ Run a model with a call to the kernels at each step ('step'), or in one call ('all') """
    make, pars = models[key]
    sim = ss.Sim(make(), **sc.mergedicts(pars, kw, kwargs)).init()
    if how == 'all' and key != 'tb':
        assert sim.kernel_only()
    return sim.run(profile=(how == 'step')) # Recording the time of each step is one of the things that needs the loop


def largest_diff(sim1, sim2):
    """ The largest difference in any result, as a share of that result's largest value """
    out = 0
    for mod in sim1.declared:
        for key,res in mod.results.items():
            if isinstance(res, ss.Result):
                other = sim2.results[mod.name][key]
                for a,b in [(res.values, other.values), (getattr(res, 'strata', None), getattr(other, 'strata', None))]:
                    if a is not None:
                        out = max(out, np.abs(a - b).max()/max(np.abs(b).max(), 1e-12))
    return out


@sc.timer()
def test_ode():
    sc.heading('Testing that the ODE gives the same run in one call to the kernels as step by step')
    for key in models:
        sims = sc.objdict({how:run(key, how) for how in ['step', 'all']})
        diff = largest_diff(sims.all, sims.step)
        print(f'{key:8s} whole run against stepped {diff:0.1e}')
        assert diff < 1e-9, f'{key}: the whole run differs from the stepped one by {diff}' # A value that varies in time is tabulated for a whole run, and worked out at each stage when stepped
    return sims


@sc.timer()
def test_factored_mixing():
    sc.heading('Testing that a contact matrix within places, with commuting, mixes as the product of the time share and the matrix')
    make, pars = models.commute
    sim = ss.Sim(make(), **pars, **kw).init()
    mixing = sim.mixing
    term = mixing.terms[0]
    pre, post = np.kron(mixing.time_share.toarray(), np.eye(3)), np.kron(mixing.time_share.toarray(), term.matrix) # The whole matrix between towns by ages, which is not formed
    rng = np.random.default_rng(1)
    infectious, n = rng.random(sim.groups.n_groups), 1 + rng.random(sim.groups.n_groups)
    num, den = [np.bincount(term.cells, weights=v, minlength=9) @ pre for v in [infectious, n]]
    expected = (post @ (num/den))[term.cells]
    assert np.allclose(mixing.term_exposure(term, infectious, n), expected, rtol=1e-12, atol=0)
    assert len(sim.diseases.sir.engine.movers) == 1 and sim.diseases.sir.engine.movers[0].matrix.nnz == 6 # Aging: from each of the first two age groups in each town to the next
    return expected


@sc.timer()
def test_stages():
    sc.heading('Testing that a dwell time in many stages is as long as it should be under every solver')
    def make():
        S, E, I, R = ss.states('S E I R')
        return ss.Disease(
            infection = ss.Transition(S, E, rate=0.4 * I),
            latency   = ss.Transition(E, I, dur=ss.normal(7, 1.4)), # 25 stages
            recovery  = ss.Transition(I, R, dur=10),
            init      = dict(I=0.001),
            name      = 'seir',
        )
    peaks = sc.objdict()
    for solver in ['ode', 'binomial', 'tau', 'abm']:
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sims = [ss.Sim(make(), unit='day', stop=150, total_pop=200_000, solver=solver, rand_seed=seed, **pars, **kw).run() for seed in range(1 if solver == 'ode' else 3)]
        peaks[solver] = np.mean([np.argmax(sim.results.seir.I) for sim in sims])
        print(f'{solver:8s} infectious peak on day {peaks[solver]:n}')
    assert max(peaks.values()) - min(peaks.values()) < 8 # The chain-binomial solver moves people through as many stages as they pass in a step; one stage a step would make the latent period 25 days, not 7, and the peak weeks later

    # Through the stages in whole steps, with another way out: the same as the ODE
    def cohort(solver):
        E, I, D = ss.states('E I D')
        sim = ss.Sim(ss.Disease(onset=ss.Transition(E, I, dur=ss.normal(7, 1)), death=ss.Transition(E, D, dur=20), first='E', name='d'), unit='day', stop=10, total_pop=1_000_000, solver=solver, **kw)
        return sim.run().results.d
    ode, res = cohort('ode'), cohort('binomial')
    assert np.allclose([res.I[6], res.I[8], res.D[10]], [ode.I[6], ode.I[8], ode.D[10]], rtol=0.01)
    return peaks


@sc.timer()
def test_seasonal_steps():
    sc.heading('Testing that the stochastic solvers follow a value that varies within a step')
    def make(solver, dt, seed=0, pop=5000):
        S, I = ss.states('S I')
        sis = ss.Disease(infection=ss.Transition(S, I, rate=ss.seasonal(mean=ss.perday(0.15), amplitude=0.5, peak=0.25) * I), recovery=ss.Transition(I, S, dur=ss.days(10)), init_prev=0.05, name='sis')
        sim = ss.Sim(sis, solver=solver, dt=dt, stop=ss.years(2), total_pop=pop, rand_seed=seed, **kw).run()
        return sim.results.sis.infection.sum()/pop, sim.solver
    ode = make('ode', ss.days(1))[0]
    out = sc.objdict()
    for solver,pop in dict(gillespie=5000, tau=500_000).items(): # Steps of a quarter of a year: holding beta for the whole step gave 5-6% too many infections
        runs = [make(solver, ss.days(91), seed, pop) for seed in range(3)]
        out[solver] = np.mean([run[0] for run in runs])
        assert runs[0][1] == solver
        assert np.isclose(out[solver], ode, rtol=0.02), f'{solver}: {out[solver]:n} infections per person, against {ode:n} for the ODE'
    with pytest.raises(ss.CapabilityError): # Exact only for rates that are constant within each quarter of a step
        S, I = ss.states('S I')
        sis = ss.Disease(infection=ss.Transition(S, I, rate=ss.seasonal(mean=0.15, amplitude=0.5) * I), recovery=ss.Transition(I, S, dur=10), init_prev=0.05, name='sis')
        ss.Sim(sis, solver='gillespie', approx=False, unit='day', stop=100, **kw).init()

    # A step function next to a value that varies smoothly is held for each step, by the ODE stepped as by a whole run (stepped, it was taken at the middle and end of each step, 1.4% off)
    days = [f'2020-01-{d:02d}' for d in range(1, 32)]
    beta = ss.par(0.5, name='b')*ss.seasonal(mean=0.9, amplitude=0.1)*ss.timeseries(dict(zip(days, np.linspace(1.0, 0.3, len(days))))) # A time series is a step function by default
    def sir():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, rate=ss.perday(0.25)), init_n=dict(I=100), name='sir')
    whole = ss.Sim(sir(), unit='day', start='2020-01-01', stop='2020-02-15', total_pop=1e6, **kw).run()
    stepped = ss.Sim(sir(), unit='day', start='2020-01-01', stop='2020-02-15', total_pop=1e6, **kw).run(until='2020-02-15')
    assert np.allclose(stepped.results.sir.I, whole.results.sir.I, rtol=1e-9)
    return out


@sc.timer()
def test_plain_estimate():
    sc.heading('Testing that the estimate of a plain run counts the substeps of a stiff ODE')
    S, I, R = ss.states(3)
    sis = ss.Disease(S >> I @ ss.perday(2.0)*I.frac, I >> R @ ss.days(1), R >> S @ ss.years(1), init=dict(I=0.01), name='sis')
    sim = ss.Sim(sis, total_pop=1e5, dt=ss.months(1), stop=ss.years(30), solver='ode', **kw).init()
    engine = sim.diseases[0].engine
    engine.init_program()
    seconds = []
    sim.pars.freeze(False) # To ask the engine of this sim, with each setting
    for n_sub in [1, None]: # As if no substeps were taken, then as the run takes them
        sim.pars.substeps = n_sub
        seconds.append(engine.plain_seconds())
    assert seconds[1] > 20*seconds[0] # Recovery in a day under monthly steps: 122 substeps a step, so that no one turns over by more than a quarter in one
    return seconds


@sc.timer()
def test_engine_choice():
    sc.heading('Testing that the engine chosen for a run depends on the model only, not on the cache or the load')
    choices = []
    for compiled in [True, False]:
        clock = itertools.count(0.0, 100.0*(1 - compiled)) # Each call 100 s after the last, as if the machine were busy
        with mock.patch.object(ss.kernels, 'is_compiled', lambda: compiled), mock.patch.object(time, 'perf_counter', lambda: next(clock)): # And a cold cache
            sim = ss.Sim(make_commute(), total_pop=1e5, unit='day', stop=60, solver='ode', **kw).init()
            choices.append((ss.solvers.kernel_seconds(sim), ss.solvers.choose_engine(sim)))
    assert choices[0] == choices[1], f'The estimate and the engine changed with the cache or the clock: {choices}'
    return choices


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    sims = test_ode()
    mix = test_factored_mixing()
    peaks = test_stages()
    seasonal = test_seasonal_steps()
    plain = test_plain_estimate()
    choices = test_engine_choice()

    T.toc()
