"""
Test the kernels: that a run taken in one call, or step by step, matches the same run worked out in Python
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
from test_dimensions import make_malaria, make_tb

kw = dict(verbose=0)


def make_sir():
    """ An SIR model with reported cases, totaled each week """
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4) * I
    recovery  = I >> R @ ss.perday(0.15)
    cases     = ss.outcome(0.6 * infection, every=ss.weeks(1))
    return ss.Disease(infection, recovery, cases, init=dict(I=10), name='sir')


def make_seir():
    """ Seasonal transmission, a dwell time in stages, and a branch """
    seir = ss.Disease(
        infection = ss.transmission('S -> E', beta=ss.seasonal(0.4, amplitude=0.3, peak=1)),
        latency   = ss.progression('E -> I', dur=ss.erlang(mean=5, k=3)),
        outcome   = ss.progression('I -> (R, D)', dur=8, p=[None, 0.1]),
        init      = dict(I=0.01),
        name      = 'seir',
    )
    return seir


def make_plain(beta, **kwargs):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R', dur=8), name='sir', **kwargs)


def make_ages():
    """ Three age groups with a contact matrix, and susceptibility that differs by age """
    return make_plain(0.05, init=dict(I=0.01)).stratify(age=[0, 15, 65], split=[0.3, 0.5, 0.2], adjust=dict(infection=[0.5, 1.0, 1.5]))


def make_towns():
    """ Three towns with commuting between them """
    towns = ss.geo(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
    return make_plain(0.3, init=dict(I=ss.by(patch=[50, 0, 0]))).stratify(patch=towns, mobility=ss.gravity(k=2e-4))


models = sc.objdict(
    sir     = (make_sir,     dict(total_pop=10_000, unit='day', stop=ss.weeks(12))),
    seir    = (make_seir,    dict(unit='day', stop=200)),
    ages    = (make_ages,    dict(unit='day', stop=100, networks=ss.contacts(age=np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])))),
    towns   = (make_towns,   dict(unit='day', stop=150, total_pop=100_000)),
    malaria = (make_malaria, dict(unit='day', stop=200, total_pop=10_000)),
    tb      = (make_tb,      dict(stop=30, total_pop=100_000)), # Has a transfer, so is stepped
)


def run(key, how, **kwargs):
    """ Run a model with everything worked out in Python ('python'), with a call to the kernels at each step ('step'), or in one call ('all') """
    make, pars = models[key]
    sim = ss.Sim(make(), **sc.mergedicts(pars, kw, kwargs)).init()
    for mod in sim.declared:
        mod.engine.use_kernels = how != 'python'
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
    sc.heading('Testing the ODE kernel against Python')
    for key in models:
        sims = sc.objdict({how:run(key, how) for how in ['python', 'step', 'all']})
        diffs = [largest_diff(sims[how], sims.python) for how in ['step', 'all']]
        print(f'{key:8s} step {diffs[0]:0.1e}, whole run {diffs[1]:0.1e}')
        assert max(diffs) < 1e-9, f'{key}: the kernels differ from Python by {max(diffs)}'
    return sims


@sc.timer()
def test_stochastic():
    sc.heading('Testing the stochastic kernels against Python')
    n = 25 # Runs of each
    for solver,keys in dict(binomial=['sir', 'ages'], tau=['sir'], gillespie=['sir', 'towns'], sde=['sir', 'towns']).items():
        pop = 2000 if solver == 'gillespie' else 10_000 # One event at a time is slow in Python
        for key in keys:
            finals = sc.objdict()
            for how in ['python', 'all']:
                runs = [run(key, how, solver=solver, rand_seed=seed, total_pop=pop) for seed in range(n)]
                name = runs[0].declared[0].name
                finals[how] = np.array([[sim.results[name].n_S[-1], sim.results[name].n_I.max()] for sim in runs])
                if solver != 'sde':
                    assert np.all(finals[how] == np.round(finals[how])) # Whole people
            diff = np.abs(finals.all.mean(axis=0) - finals.python.mean(axis=0))
            se = np.sqrt(finals.all.var(axis=0)/n + finals.python.var(axis=0)/n)
            print(f'{solver:8s} {key:8s} difference in means {diff}, standard error {se}')
            assert np.all(diff < 4*se + 1), f'{key}: the {solver} kernel differs from Python by more than chance'
    return finals


@sc.timer()
def test_stages():
    sc.heading('Testing that a dwell time in many stages is as long as it should be under every solver')
    def make():
        return ss.Disease(
            infection = ss.transmission('S -> E', beta=0.4, source='I'),
            latency   = ss.progression('E -> I', dur=ss.normal(7, 1.4)), # 25 stages
            recovery  = ss.progression('I -> R', dur=10),
            init      = dict(I=0.001),
            name      = 'seir',
        )
    peaks = sc.objdict()
    for solver in ['ode', 'binomial', 'tau', 'abm']:
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        sims = [ss.Sim(make(), unit='day', stop=150, total_pop=200_000, solver=solver, rand_seed=seed, **pars, **kw).run() for seed in range(1 if solver == 'ode' else 3)]
        peaks[solver] = np.mean([np.argmax(sim.results.seir.n_I) for sim in sims])
        print(f'{solver:8s} infectious peak on day {peaks[solver]:n}')
    assert max(peaks.values()) - min(peaks.values()) < 6 # The chain-binomial solver moves people through as many stages as they pass in a step; one stage a step would make the latent period 25 days, not 7, and the peak weeks later

    # Through the stages in whole steps, with another way out: the same as the ODE, by the kernels and in Python
    def cohort(solver, kernels=True):
        sim = ss.Sim(ss.Disease(onset=ss.progression('E -> I', dur=ss.normal(7, 1)), death=ss.progression('E -> D', dur=20), init=dict(E=1.0), name='d'), unit='day', stop=10, total_pop=1_000_000, solver=solver, **kw).init()
        sim.diseases.d.engine.use_kernels = kernels
        return sim.run().results.d
    ode = cohort('ode')
    for res in [cohort('binomial'), cohort('binomial', kernels=False)]:
        assert np.allclose([res.n_I[6], res.n_I[8], res.n_D[10]], [ode.n_I[6], ode.n_I[8], ode.n_D[10]], rtol=0.01)
    return peaks


@sc.timer()
def test_seasonal_steps():
    sc.heading('Testing that the stochastic solvers follow a value that varies within a step')
    def make(solver, dt, seed=0):
        sis = ss.Disease(infection=ss.transmission('S -> I', beta=ss.seasonal(mean=ss.perday(0.15), amplitude=0.5, peak=0.25)), recovery=ss.progression('I -> S', dur=ss.days(10)), init_prev=0.05, name='sis')
        return ss.Sim(sis, solver=solver, dt=dt, stop=ss.years(2), total_pop=5000, rand_seed=seed, **kw).run().results.sis.infection.sum()
    ode = make('ode', ss.days(1))
    out = sc.objdict()
    for solver in ['gillespie', 'tau']: # Steps of a quarter of a year: holding beta for the whole step gave 5-6% too many infections
        out[solver] = np.mean([make(solver, ss.days(91), seed) for seed in range(3)])
        assert np.isclose(out[solver], ode, rtol=0.02), f'{solver}: {out[solver]:n} infections, against {ode:n} for the ODE'
    return out


@sc.timer()
def test_scipy():
    sc.heading('Testing the scipy integrators with the right-hand side in the kernels')
    make, pars = models.seir
    out = sc.objdict()
    for kernels in [True, False]:
        sim = ss.Sim(make(), integrator='LSODA', **pars, **kw).init()
        sim.diseases.seir.engine.use_kernels = kernels
        out[str(kernels)] = sim.run().results.seir
    assert np.allclose(out['True'].n_I, out['False'].n_I, rtol=1e-6) and np.allclose(out['True'].infection, out['False'].infection, rtol=1e-6)
    return out


@sc.timer()
def test_restart():
    sc.heading('Testing restarting a sim, and many runs of one')
    make, pars = models.sir

    # A restarted sim gives what a new one does, for another seed or another value
    sim = ss.Sim(make(), solver='binomial', rand_seed=1, **pars, **kw).init()
    sim.run()
    sim.restart(rand_seed=5).run()
    new = ss.Sim(make(), solver='binomial', rand_seed=5, **pars, **kw).run()
    assert largest_diff(sim, new) == 0

    sim = ss.Sim(make_plain(0.3, init=dict(I=0.01)), unit='day', stop=100, **kw).init()
    peaks = [sim.restart(pars=dict(beta=beta)).run().results.sir.n_I.max() for beta in [0.2, 0.3, 0.4]]
    new = [sim.new(pars=dict(beta=beta)).run().results.sir.n_I.max() for beta in [0.2, 0.3, 0.4]]
    assert np.allclose(peaks, new) and peaks[0] < peaks[1] < peaks[2]

    # Many runs are restarts of one sim
    base = ss.Sim(make(), solver='binomial', **pars, **kw)
    msim = base.run(n_runs=20)
    res = msim.results.sir.n_I
    assert res.values.shape == (20, base.new().init().t.npts)
    assert len(msim.sims) == 1 and not base.initialized # One sim was built, and the one that was given is untouched
    assert len(np.unique(res.values.max(axis=1))) > 10 # Each has its own seed
    one = base.new(rand_seed=base.pars.rand_seed + 7).run() # Each is the run that a sim with that seed gives
    for key,multi in msim.results.sir.items():
        if key != 'cases': # Reported cases are drawn afterwards, for all the runs at once
            assert np.allclose(multi.values[7], one.results.sir[key].values), key
    assert set(msim.results.sir.keys()) == {key for key,res in one.results.sir.items() if isinstance(res, ss.Result)}
    assert abs(msim.results.sir.cases.values.sum(axis=1).mean()/msim.results.sir.cum_infection.values[:, -1].mean() - 0.6) < 0.05
    base.pars.rand_seed += 7 # A seed set after the sim was made is used, as is an option of running in parallel
    later = ss.parallel(base, n_runs=2, verbose=0)
    assert len(later.sims) == 1 and np.allclose(later.results.sir.n_I.values[0], msim.results.sir.n_I.values[7])

    # The exact solver runs place by place when places do not affect each other within a step, with the same results as event by event over all places
    def herds(local):
        sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.days(10)), init=dict(I=ss.by(herd=[5]*20)), name='herds')
        sir.stratify(herd=[str(i) for i in range(20)], kind='space')
        out = []
        for seed in range(10):
            sim = ss.Sim(sir, total_pop=20*200, solver='gillespie', unit='day', stop=100, rand_seed=seed, **kw).init()
            engine = sim.diseases.herds.engine
            engine.init_program()
            if not local:
                engine._local_mixing = False
                engine.pack_targets()
            assert engine.program.local == local
            out.append(sim.run().results.herds.n_R[-1])
        return np.mean(out), np.std(out)/np.sqrt(len(out))
    (a, se_a), (b, se_b) = herds(True), herds(False)
    assert abs(a - b) < 3*np.hypot(se_a, se_b) + 1

    # How often a small outbreak dies out
    small = ss.Disease(infection=ss.transmission('S -> I', beta=0.15), recovery=ss.progression('I -> R', dur=10), init=dict(I=2), name='small')
    ext = ss.Sim(small, total_pop=200, solver='gillespie', unit='day', stop=100, **kw).run(n_runs=50).extinction(output=True)
    assert 0.3 < ext.share[0] < 0.9 and ext.runs[0] == 50

    # A sim that needs the loop (here, a modifier with a function of the sim) cannot be restarted, and still runs the usual way
    masks = ss.modify('sir.pars.beta', multiply=0.5, when=lambda sim: sim.ti >= 20)
    sim = ss.Sim(make_plain(0.3, init=dict(I=0.01)), unit='day', stop=100, interventions=masks, **kw).init()
    assert not sim.kernel_only()
    assert sim.run().results.sir.n_I.max() < new[1]
    return msim


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    sims = test_ode()
    finals = test_stochastic()
    peaks = test_stages()
    seasonal = test_seasonal_steps()
    scipy = test_scipy()
    msim = test_restart()

    T.toc()
