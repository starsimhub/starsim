"""
Test the kernels: that a run taken in one call matches the same run taken step by step
"""

# %% Imports and settings
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
    ages    = (make_ages,    dict(unit='day', stop=100, networks=ss.contacts(age=np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])))),
    towns   = (make_towns,   dict(unit='day', stop=150, total_pop=100_000)),
    commute = (make_commute, dict(unit='day', stop=150, total_pop=100_000, use_aging=True, networks=ss.contacts(age=np.array([[8, 3, 1], [3, 6, 2], [1, 2, 3]])))), # Aging, which a model without births or deaths has only if asked for
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
def test_stochastic():
    sc.heading('Testing that the stochastic solvers give the same run in one call to the kernels as step by step, in whole people')
    for solver,keys in dict(binomial=['sir', 'ages', 'commute'], gillespie=['sir', 'towns', 'commute']).items():
        pop = 2000 if solver == 'gillespie' else 10_000
        for key in keys:
            for seed in range(2):
                sims = sc.objdict({how:run(key, how, solver=solver, rand_seed=seed, total_pop=pop) for how in ['step', 'all']})
                assert largest_diff(sims.all, sims.step) == 0, f'{key} under {solver}: the whole run differs from the stepped one'
                n_I = sims.all.results[sims.all.declared[0].name].I.values
                assert np.all(n_I == np.round(n_I)) # Whole people

    # Many places, most of which have no events (skipped until something changes them): the same run as step by step
    herds = make_plain(0.3, init_n=dict(I=ss.by(herd=[2] + [0]*599))).stratify(herd=[str(i) for i in range(600)], kind='space')
    base = ss.Sim(herds, solver='gillespie', unit='day', stop=30, total_pop=600*50, **kw)
    assert largest_diff(base.new().run(), base.new().run(profile=True)) == 0

    # Diseases tracked jointly (74 × 51 compartments, with births) are stepped by the compartments that hold anyone: the same events as stepping every flow, and many times faster
    lognormal = [ss.SIR(dur_inf=ss.lognorm_ex(mean=ss.years(6)), p_death=0.01), ss.SIS(dur_inf=ss.lognorm_ex(mean=ss.years(10)))] # Starsim 3's durations, as Erlang stages
    sim = ss.Sim(lognormal, demographics=[ss.birth(rate=ss.peryear(0.05)), ss.death(rate=ss.peryear(0.02))], total_pop=1000, stop=10, solver='gillespie', **kw).init()
    for ti in range(5):
        sim.run_one_step()
    kn, cross = ss.kernels, sim.crossings[0]
    p, places = cross.program, np.arange(cross.program.n_places)
    out = sc.objdict()
    for how in ['sparse', 'dense']:
        times = []
        for rep in range(3):
            y, flows = cross.y.copy(), np.zeros((p.n_trans + p.n_movers, cross.y.shape[1]))
            args = (y, p, kn.workspace(y, p), places, 0, len(places), 1.0, flows, np.uint64(rep))
            T = sc.timer()
            kn.sparse_events(*args, kn.sparse_maps(p, len(y))) if how == 'sparse' else kn.place_events(*args)
            times.append(T.toc(output=True))
        out[how] = (y, flows, min(times))
    assert np.array_equal(out.sparse[0], out.dense[0]) and np.array_equal(out.sparse[1], out.dense[1]) # The same events
    assert out.sparse[2] < out.dense[2]/3 # About 60 times faster here

    # Strata that mix: the same epidemic as one group
    def strata(K, **pars):
        sir = make_plain(ss.perday(0.3), init_n=dict(I=ss.by(risk=[2] + [0]*(K-1)))).stratify(risk=[f'g{i}' for i in range(K)])
        return ss.Sim(sir, solver='gillespie', unit='day', stop=100, total_pop=200*K, **pars, **kw)
    one = ss.Sim(make_plain(ss.perday(0.3), init_n=dict(I=2)), solver='gillespie', unit='day', stop=100, total_pop=2000, **kw).run(n_runs=60).results.sir.R.values[:, -1]
    ten = strata(10).run(n_runs=60).results.sir.R.values[:, -1]
    assert abs(one.mean() - ten.mean()) < 3*np.hypot(one.std(), ten.std())/np.sqrt(60) # The same final size
    return sims


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
    peaks = [sim.restart(pars=dict(beta=beta)).run().results.sir.I.max() for beta in [0.2, 0.3, 0.4]]
    new = [sim.new(pars=dict(beta=beta)).run().results.sir.I.max() for beta in [0.2, 0.3, 0.4]]
    assert np.allclose(peaks, new) and peaks[0] < peaks[1] < peaks[2]

    # Many runs are restarts of one sim
    base = ss.Sim(make(), solver='binomial', **pars, **kw)
    msim = base.run(n_runs=20)
    res = msim.results.sir.I
    assert res.values.shape == (20, base.new().init().t.npts)
    assert len(msim.sims.runners) == 1 and len(msim.sims) == 20 and not base.initialized # One sim was built (the sim of each run is made when asked for), and the one that was given is untouched
    assert len(np.unique(res.values.max(axis=1))) > 10 # Each has its own seed
    one = base.new(rand_seed=base.pars.rand_seed + 7).run() # Each is the run that a sim with that seed gives
    for key,multi in msim.results.sir.items():
        assert np.allclose(multi.values[7], one.results.sir[key].values), key
    grouped = ss.Sim(models.ages[0](), solver='binomial', **models.ages[1], **kw) # By group too
    many, alone = grouped.run(n_runs=3).results.sir, grouped.new(rand_seed=grouped.pars.rand_seed + 2).run().results.sir
    assert all(np.allclose(many[key].strata[2], alone[key].strata) for key in ['I', 'infection']) and np.allclose(many.I.values[2], alone.I.values)
    assert set(msim.results.sir.keys()) == {key for key,res in one.results.sir.items() if isinstance(res, ss.Result)}
    assert abs(msim.results.sir.cases.values.sum(axis=1).mean()/msim.results.sir.infection.cum.values[:, -1].mean() - 0.6) < 0.05
    base.pars.rand_seed += 7 # A seed set after the sim was made is used, as is an option of running in parallel
    later = ss.parallel(base, n_runs=2, verbose=0)
    assert len(later.sims.runners) == 1 and np.allclose(later.results.sir.I.values[0], msim.results.sir.I.values[7])

    # A sim that has been initialized gives the same distinct runs as one that has not, restarted or copied
    for solver,agents in dict(binomial={}, abm=dict(n_agents=2000)).items():
        fresh = ss.Sim(make(), solver=solver, **agents, **pars, **kw)
        runs = [ss.parallel(sim, n_runs=4, parallel=False).results.sir.I.values for sim in [fresh, fresh.new().init()]]
        assert len(np.unique(runs[0], axis=0)) == 4 and np.array_equal(runs[0], runs[1]), solver # Compared over the whole run, since runs where the epidemic dies out have the same peak
    assert len(ss.parallel(done=fresh.new().run(), fresh=fresh, parallel=False).sims) == 2 # A sim that has run is built again

    # The exact solver runs place by place when places do not affect each other within a step, with the same results as event by event over all places
    def herds(local):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_n=dict(I=ss.by(herd=[5]*20)), name='herds')
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
            out.append(sim.run().results.herds.R[-1])
        return np.mean(out), np.std(out)/np.sqrt(len(out))
    (a, se_a), (b, se_b) = herds(True), herds(False)
    assert abs(a - b) < 3*np.hypot(se_a, se_b) + 1

    # A place where nothing can happen is skipped until something changes it: here, waning that starts on day 30, in a herd where nobody is infected
    wane = 0.05*ss.timeseries({0: 0.0, 29: 0.0, 30: 1.0})
    S, I, R = ss.states('S I R')
    sirs = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.3) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), waning=ss.Transition(R, S, rate=wane),
                      init_n=dict(I=ss.by(herd=[5, 0]), R=ss.by(herd=[0, 50])), name='sirs')
    sirs.stratify(herd=['a', 'b'], kind='space')
    S_b = ss.Sim(sirs, total_pop=200, solver='gillespie', unit='day', stop=60, **kw).run().results.sirs.S.select(herd='b').values
    assert S_b[60] > S_b[29] # Herd b's recovered wane once waning starts

    # How often a small outbreak dies out, from the runs' results
    S, I, R = ss.states('S I R')
    small = ss.Disease(infection=ss.Transition(S, I, rate=0.15 * I), recovery=ss.Transition(I, R, dur=10), init_n=dict(I=2), name='small')
    msim_small = ss.Sim(small, total_pop=200, solver='gillespie', unit='day', stop=100, **kw).run(n_runs=50)
    died = (msim_small.results.small.I.values[:, -1] == 0).mean()
    assert 0.3 < died < 0.9

    # How often it is a major outbreak (more than 10% infected), which dying out does not say for an SIR, since every run dies out in the end
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=2/7 * I), recovery=ss.Transition(I, R, dur=7), init_n=dict(I=1), name='sir')
    msim = ss.Sim(sir, total_pop=1000, solver='gillespie', unit='day', stop=365, **kw).run(n_runs=40)
    major = (msim.results.sir.infection.values.sum(axis=1) > 100).mean()
    assert 0.25 < major < 0.75 and (msim.results.sir.I.values[:, -1] == 0).all()

    # A sim that needs the loop (here, a modifier with a function of the sim) cannot be restarted, and still runs the usual way
    masks = ss.modify('sir.pars.beta', multiply=0.5, when=lambda sim: sim.ti >= 20)
    sim = ss.Sim(make_plain(0.3, init=dict(I=0.01)), unit='day', stop=100, interventions=masks, **kw).init()
    assert not sim.kernel_only() and 'masks' in sim.kernel_reason()
    assert sim.run().results.sir.I.max() < new[1]
    assert base.new().init().kernel_reason() is None
    return msim


@sc.timer()
def test_compile():
    sc.heading('Testing that the kernels can be compiled ahead of time')
    ss.compile(verbose=False)
    assert ss.kernels.is_compiled()
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    stoch = test_stochastic()
    msim = test_restart()
    test_compile()

    T.toc()
