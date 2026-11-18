"""
Contract tests: properties every model and solver must have, checked on the model corpus
(see corpus.py). A failure here is usually a silent wrong answer somewhere else.
"""
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import corpus

stochastic = ['binomial', 'gillespie', 'abm']


def size(solver, n):
    """ A population of n, as agents or as people """
    return dict(n_agents=n) if solver == 'abm' else dict(total_pop=n)


# %% Streams are independent

class Counter(ss.Analyzer):
    """ An analyzer that changes nothing """
    compartmental = True

    def step(self):
        pass


@pytest.mark.parametrize('solver', stochastic)
def test_streams(solver):
    """ Random streams are independent of analyzers, other diseases, stepping from Python, other sims, and other runs; a rate that varies in time is evaluated at the same step either way """
    kw = dict(solver=solver, unit='day', **size(solver, 10_000))
    alone = corpus.sir(**kw).run().results.sir.I.values
    assert np.array_equal(alone, corpus.sir(**kw, analyzers=Counter()).run().results.sir.I.values), 'An analyzer changed the draws'
    two_sirs = lambda *names: ss.Sim([corpus.make_sir(name=name) for name in names], stop=100, verbose=0, **kw) # Two SIRs that share no states
    both = two_sirs('sir', 'other').perf_profile(kind='loop', plot=False).sim.results.sir.I.values # Profiling makes the run step from Python
    assert np.array_equal(alone, both), f'Another disease, or stepping, changed the draws by up to {np.abs(alone - both).max():n}'
    ab = two_sirs('ab', 'ba').run().results # Two identical diseases whose names are anagrams have their own streams
    assert not np.array_equal(ab.ab.I.values, ab.ba.I.values), 'Two modules got the same stream'
    a, b = corpus.sir(**kw, rand_seed=1), corpus.sir(**kw, rand_seed=2) # Interleaving two sims changes neither
    a.run(until=50)
    b.run(until=50)
    assert np.array_equal(a.run().results.sir.I.values, corpus.sir(**kw, rand_seed=1).run().results.sir.I.values)
    if solver == 'abm':
        return alone

    plain, counted = [corpus.outcomes(solver=solver, rand_seed=5, **k).run().results.sir for k in [{}, dict(analyzers=Counter())]] # Nor the draws of outcomes, drawn after the states
    for key in ['cases', 'hosp', 'deaths']:
        assert np.array_equal(plain[key].values, counted[key].values), f'An analyzer changed the draws of the outcome "{key}"'
    runs = ss.parallel(corpus.outcomes(solver=solver, rand_seed=5), n_runs=3).results.sir.cases.values # Each run of several draws from its own seed, as on its own
    for i in range(3):
        assert np.array_equal(runs[i], corpus.outcomes(solver=solver, rand_seed=5 + i).run().results.sir.cases.values), f'Run {i} of three differs from the same run on its own'

    def seasonal(solver):
        S, I, R, V = ss.states('S I R V')
        sir = ss.Disease(infection=S >> I @ ss.seasonal(mean=0.421, amplitude=0.46, peak=0.92) * I, recovery=I >> R @ ss.perday(0.0909), vax=S >> V @ ss.perday(0.011), init_n=dict(I=25), name='sir')
        return ss.Sim(sir, total_pop=100_000, stop=60, unit='day', solver=ss.solvers.binomial(substeps=1) if solver == 'binomial' else solver, rand_seed=587, verbose=0) # With 2 substeps, a whole binomial run and a stepped one differ
    for s in [solver, 'ode']:
        whole, stepped = [seasonal(s).run().results.sir.vax.values, seasonal(s).perf_profile(kind='loop', plot=False).sim.results.sir.vax.values]
        assert np.allclose(whole, stepped, rtol=1e-12, atol=0), f'{s}: stepping changed the vaccinated by up to {np.abs(whole - stepped).max():n}' # Equal, but for rounding of about 1e-16 in the ODE's substeps
    return alone


# %% The solvers agree, and keep their people

fine = dict(dt=ss.days(0.25), total_pop=10_000) # Agents and stepped solvers differ from the ODE by an amount of order dt, so compare at a fine step; and at a size that runs quickly event by event
skip_solver = dict(screening=dict(gillespie='rounds of a delivery need a step'))
loose = dict( # Wider tolerances, with the reason
    sir         = 0.07, # Agents are about 3% above the ODE in S and below it in R, and the mean of 5 runs varies by about 2%
    custom      = 0.10, # Few people are hospitalized, so the counts are noisy
    ages        = 0.10, # Each of 5,000 agents is 20 people, so the few who start infected make the timing of the epidemic noisy
    age_vaccine = 0.12, # And a force of infection near 0.7 per day, so the difference of order dt is about 6% even at a quarter-day step
    patches     = 0.10,
)
closed = ['sir', 'seir', 'erlang', 'lognormal', 'ages', 'patches', 'vaccination', 'screening', 'flu_rsv', 'age_vaccine', 'patch_matrix'] # No births, deaths, or flows to ss.dead
simple = ['sir', 'seir', 'erlang', 'branching', 'ages'] # The simplest models of the corpus; the others are checked only with -m slow


def people(sim):
    """ The number of people in each declared module, at each time point """
    return [sum(mod.results[state.name].values for state in mod.states.values() if state.people) for mod in sim.declared]


@pytest.mark.parametrize('name', simple + [pytest.param(name, marks=pytest.mark.slow) for name in corpus.compartmental if name not in simple])
def test_solvers_agree(name):
    """ Every solver agrees with the ODE on the mean of each state, and a closed model keeps its population under every solver """
    make = corpus.models[name]
    sims = {'ode': [make(solver='ode', **fine).run()]}
    for solver in stochastic:
        if solver not in skip_solver.get(name, {}):
            sims[solver] = [make(solver=solver, rand_seed=seed, **fine).run() for seed in range(5 if solver == 'abm' else 3)] # Agents vary more between runs
    means = lambda sim: {f'{mod.name}.{key}':res.values.mean() for mod in sim.declared for key,res in mod.results.items() if key in mod.graph.states and isinstance(res, ss.Result) and np.ndim(res.values) == 1}
    ode = means(sims['ode'][0])
    floor = 0.1*max(ode.values()) # A small state is compared with a tenth of the largest, so its noise does not fail it
    for solver, runs in sims.items():
        for key,val in ode.items():
            mean = np.mean([means(sim)[key] for sim in runs])
            assert abs(mean - val) <= loose.get(name, 0.05)*max(val, floor), f'{name} under {solver}: {key} averages {mean:n}, against {val:n} for the ODE'
        if name in closed:
            for total in people(runs[0]):
                assert np.allclose(total, total[0], rtol=1e-9), f'{name} under {solver}: the population goes from {total[0]:n} to {total[-1]:n}'
    return ode


def test_binomial_short_stay():
    """ The chain binomial follows the ODE when a stay is one step """
    def make(solver, seed=0):
        S, I, R = ss.states('S I R')
        dis = ss.Disease(S, I, R, infection=S >> I @ ss.permonth(2)*I.frac, recovery=I >> R @ ss.months(1), init=dict(I=0.001), name='sir')
        return ss.Sim(dis, solver=solver, unit='month', stop=24, total_pop=1e6, rand_seed=seed, verbose=0).run().results.sir
    ode = make('ode')
    runs = [make('binomial', seed) for seed in range(3)]
    assert np.isclose(np.mean([res.R[-1] for res in runs]), ode.R[-1], rtol=0.05) # 950,577 against 797,129 with whole steps
    assert np.abs(np.mean([res.I.values for res in runs], axis=0) - ode.I.values).max() < 0.2*ode.I.max()
    return


@pytest.mark.parametrize('solver', ['ode', 'binomial', 'gillespie', 'tau', 'abm'])
def test_exits(solver):
    """ The order of competing exits does not matter; a fixed delay is a conveyor: everyone leaves exactly then """
    fracs = []
    for order in [('a', 'b', 'c'), ('c', 'a', 'b')]:
        I = ss.State(name='I')
        dests = {key:ss.State(name=key.upper()) for key in 'abc'}
        exits = {key:I >> dests[key] @ ss.perday(1.0) for key in order} # Three equal exits, fast compared with the step
        sim = ss.Sim(ss.Disease(I, *dests.values(), **exits, default_state='I', name='exits'), unit='day', stop=5, solver=solver, verbose=0, **size(solver, 20_000)).run()
        left = np.array([sim.results.exits[k.upper()][-1] for k in 'abc'])
        fracs.append(left/left.sum())
    assert np.allclose(fracs, 1/3, atol=0.02), f'Expected a third by each exit, not {fracs}'
    E, I = ss.states(2)
    cohort = ss.Disease(onset=E >> I @ ss.constant(ss.days(10)), default_state='E', name='cohort')
    n_E = ss.Sim(cohort, unit='day', stop=20, solver=solver, verbose=0, **size(solver, 1000)).run().results.cohort.E.values
    assert n_E[10] == 1000 and n_E[11] == 0, f'{n_E[10]:n} in E on day 10 and {n_E[11]:n} on day 11, not 1000 and 0'
    return fracs


def test_dwell_forms():
    """ Agents agree with the ODE at a fine step for every form of dwell time; one too variable for Erlang stages runs as agents under auto """
    forms = dict(
        exponential  = lambda I, R, D: dict(out=I >> R @ ss.days(4)),
        rate         = lambda I, R, D: dict(out=ss.Transition(I, R, rate=ss.perday(0.25))),
        fixed        = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.constant(ss.days(4)))),
        erlang       = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.days(4), stages=3)),
        lognormal    = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.lognormal(ss.days(4), ss.days(2)))),
        branches     = lambda I, R, D: dict(a=ss.Transition(I, R, p=0.7, dur=ss.days(4)), b=ss.Transition(I, D, p=0.3, dur=ss.days(2))), # Decided on entry
        destinations = lambda I, R, D: dict(out=ss.Transition(I, {R: 0.7, D: 0.3}, dur=ss.days(4))), # Decided at exit
        competing    = lambda I, R, D: dict(out=ss.Transition(I, R, dur=ss.constant(ss.days(4))), die=ss.Transition(I, D, rate=ss.perday(0.1))),
        after        = lambda I, R, D: dict(out=ss.Transition(I, R, rate=ss.perday(0.5), min_stay=ss.days(2))), # A minimum stay, then a rate
    )
    def run(form, solver):
        I, R, D = ss.states('I R D')
        dis = ss.Disease(I, R, D, **forms[form](I, R, D), default_state='I', name='cohort')
        res = ss.Sim(dis, unit='day', dt=0.05, stop=15, solver=solver, rand_seed=1, verbose=0, **size(solver, 20_000)).run().results.cohort
        return np.array([res.I.values.mean(), res.R[-1], res.D[-1]])/20_000 # Time infected, and where people end up
    for form in forms:
        ode, abm = run(form, 'ode'), run(form, 'abm')
        assert np.allclose(abm, ode, rtol=0.03, atol=0.005), f'{form}: agents {abm}, ODE {ode}'

    def wide(c, solver='auto'):
        A, B = ss.states('A B')
        dis = ss.Disease(A, B, go=A >> B @ ss.peryear(1), back=ss.Transition(B, A, dur=ss.weibull(shape=c, scale=ss.years(1))), default_state='A', name='m')
        return ss.Sim(dis, total_pop=1000, stop=2010, solver=solver, verbose=0).init()
    assert isinstance(wide(0.5).solver, ss.solvers.ode) and isinstance(wide(0.3).solver, ss.solvers.abm) # Coefficients of variation of 2.2 and 5.4
    assert any('coefficient of variation' in note for note in wide(0.3, 'ode').notes) # Its long tail is lost, which the run notes
    return


@pytest.mark.parametrize('dwell', ['fixed', 'after'])
def test_joint_states(dwell):
    """ Diseases tracked jointly keep the same people in each under every solver, with a conveyor or a minimum stay; as a joint model written by hand; within a budget """
    dwells = dict(
        fixed = lambda P, NP: ss.Transition(P, NP, dur=ss.constant(ss.months(9))), # A conveyor: the ODE lost 9% of people
        after = lambda P, NP: ss.Transition(P, NP, rate=ss.peryear(2), min_stay=ss.months(3)),
    )
    for solver in ['ode', 'binomial', 'abm']:
        M = ss.states('M0 M1 M2')
        m = ss.Disease(*M, **{f'go{i}':M[i] >> M[(i+1) % 3] @ ss.peryear(1.0) for i in range(3)}, default_state='M0', name='m')
        NP, P = ss.states('NP P')
        p = ss.Disease(NP, P, conception=NP >> P @ ss.peryear(0.25), birth=dwells[dwell](P, NP), default_state='NP', name='p')
        mod = ss.modify(p.conception, multiply=0.1, who=M[1]) # Crosses the two
        sim = ss.Sim([m, p], connectors=mod, start=2000, stop=2010, dt=ss.months(1), solver=solver, verbose=0, **size(solver, 2000)).run()
        assert solver == 'abm' or sim.joint_states
        assert np.allclose(people(sim), 2000, rtol=1e-9), f'{solver}: the population changes'
    if dwell == 'after':
        return

    def pair(solver='auto'): # Dying of one disease leaves the other too, though neither refers to the other
        S, I = ss.states(2)
        a = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.5)*I, death=ss.Transition(I, ss.dead, dur=2), init_prev=0.1, name='a')
        S, I = ss.states(2)
        b = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.3)*I, recovery=ss.Transition(I, S, dur=3), init_prev=0.1, name='b')
        return ss.Sim([a, b], total_pop=20_000, stop=2030, solver=solver, verbose=0)
    res = pair('binomial').run().results
    assert np.allclose(res.a.S + res.a.I, res.b.S + res.b.I)
    SS, SI, IS, II = ss.states('SS SI IS II') # The same as the joint model written as one disease, whose states are the combinations
    ba, bb = ss.peryear(0.5), ss.peryear(0.3)
    joint = ss.Disease(SS, SI, IS, II, a1=SS >> IS @ ba*(IS + II), a2=SI >> II @ ba*(IS + II), b1=SS >> SI @ bb*(SI + II), b2=IS >> II @ bb*(SI + II),
                       r1=SI >> SS @ ss.years(3), r2=II >> IS @ ss.years(3), d1=ss.Transition(IS, ss.dead, dur=2), d2=ss.Transition(II, ss.dead, dur=2),
                       init=dict(SS=0.81, SI=0.09, IS=0.09, II=0.01), default_state='SS', name='ab')
    j = ss.Sim(joint, total_pop=20_000, stop=2030, verbose=0).run().results.ab
    res = pair().run().results
    assert np.allclose(res.a.I.values, j.IS.values + j.II.values, rtol=1e-9) and np.allclose(res.b.I.values, j.SI.values + j.II.values, rtol=1e-9)
    with ss.options.context(max_joint=3): # Above the budget, auto runs agents, and a compartmental solver that was asked for stops
        assert isinstance(pair().init().solver, ss.solvers.abm)
        with pytest.raises(ss.CapabilityError):
            pair('ode').init()
    return j


def test_empty_place():
    """ A place with no one in it changes nothing, under every solver that runs by group """
    def make(solver, places=True):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.perday(0.2)*I.frac + ss.perday(1e-5)*I.n + ss.perday(1e-6)*I.total), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        if places:
            sir.stratify(place=ss.levels({'a': 1.0, 'b': 0.0}, kind='space'))
        return ss.Sim(sir, solver=solver, total_pop=10_000, unit='day', stop=60, verbose=0)
    one = make('ode', places=False).run().results.sir.R.values
    for solver in ['ode', 'jax', 'binomial', 'tau', 'gillespie']:
        sim = make('ode' if solver == 'jax' else solver)
        with ss.options.context(backend='jax' if solver == 'jax' else ss.options.backend):
            res = sim.run().results.sir
        assert np.all(np.isfinite(res.I.values)) and res.I.select(place='b').values.max() == 0, solver
        if solver in ['ode', 'jax']:
            assert np.allclose(res.R.values, one, rtol=1e-9), f'{solver}: an empty place changes the epidemic'
    return


def test_acquisition():
    """ An infection by a bite rate is an acquisition: the bednet idiom, a vaccine, and a cure act on the Ross-Macdonald model, whose R0() is the analytic value """
    R0 = np.sqrt(0.3*0.5*50_000/9_900*100 * 0.3*0.5*0.1/(0.1 + 0.1)*10) # √(abM/(Nr) · acσ/((σ + μ)μ)) = 7.54, with M = 50,000 mosquitoes at equilibrium and N = 9,900 people, the 1% infected at the start left out
    assert np.isclose(corpus.malaria(solver='ode').init().diseases.malaria.R0(), R0)
    for solver in ['ode', 'abm']:
        infections = {}
        for key,intv in dict(none=None, bednet=ss.modify('malaria.rel_sus', multiply=0.0), vaccine=ss.vaccinate(ss.Vaccine(efficacy=1.0), when=[0], coverage=1.0),
                             cure=ss.treat(ss.Treatment(cure=1.0), when=[100], coverage=0.9, name='mda')).items():
            sim = corpus.malaria(solver=solver, interventions=intv, **size(solver, 5000)).run()
            infections[key] = sim.results.malaria.infection.values.sum()
        assert infections['none'] > 5000 and infections['bednet'] == 0 and infections['vaccine'] == 0
        assert sim.results.mda.delivered.values.sum() > 1000 # Those infected by bites count as infected
    return


def behavior(solver, seed=1, n=2000):
    """ A choice of method: a regression over two boolean dimensions and a counter, and a switching row by age, one entry per destination """
    from scipy.special import expit
    none, pill, iud, P = ss.states('none pill iud P')
    p_use = expit(-1.0 + 0.5*ss.person.urban + 0.3*ss.person.edu + 0.4*ss.person.parity) # Parity is a counter, so is exact as compartments
    mix = ss.par(ss.by(age=[0.7, 0.4])) # The share choosing the pill by age (the rest choose the IUD)
    contra = ss.Disease(none, pill, iud, P,
        start    = ss.Transition(none, {pill: mix, iud: None}, rate=ss.peryear(0.3)*p_use),
        stop     = ss.Transition(pill, none, rate=ss.peryear(0.2)),
        remove   = ss.Transition(iud, none, rate=ss.peryear(0.1)),
        conceive = ss.Transition(none, P, rate=ss.peryear(0.2)),
        deliver  = ss.Transition(P, none, dur=ss.months(9), increment='parity'),
        mix=mix, default_state='none', name='contra')
    sim = ss.Sim(contra, solver=solver, rand_seed=seed, start=2000, stop=2006, dt=ss.months(1), aging=False, verbose=0, **size(solver, n))
    sim.stratify(age=[0, 30], urban={False: 0.4, True: 0.6}, edu={False: 0.5, True: 0.5}, parity=[0, 1, 2, 3])
    return sim


def test_behavior_solvers():
    """ A module of behavior (a regression, a counter, and a switching row by age) has the same shares by method and age under the binomial solver and as agents """
    def fracs(sim): # The share in each state by age group, over the last year
        res = sim.results.contra
        n = np.array([res[s].by('age').values[-12:].mean(axis=0) for s in ['none', 'pill', 'iud', 'P']])
        return n/n.sum(axis=0)
    ode = fracs(behavior('ode').run())
    sim = behavior('binomial', n=1_000_000)
    sim.pars.solver = ss.solvers.binomial(substeps=10) # Whole steps of the chain binomial would lengthen the 9 months pregnant by half a step (6% more pregnant)
    assert np.allclose(fracs(sim.run()), ode, rtol=0.01)
    runs = np.array([fracs(behavior('abm', seed=seed).run()) for seed in range(20)]) # Within 3 standard errors
    mean, se = runs.mean(axis=0), runs.std(axis=0, ddof=1)/np.sqrt(len(runs))
    assert np.all(np.abs(mean - ode) < 3*se), f'Agents {mean} against the ODE {ode}, z = {(mean - ode)/se}'
    return ode


if __name__ == '__main__':
    T = sc.timer()
    for name in corpus.compartmental:
        test_solvers_agree(name)
    test_behavior_solvers()
    T.toc()
