"""
Test the distributions (distributions.py and internals/dists.py), and the random numbers: seeds, common random numbers, and independence
"""

# %% Imports and settings
from unittest import mock
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

n = 1_000_000 # For testing statistical properties of distributions
m = 5 # For just testing that drawing works
sc.options(interactive=False)


def make_sim():
    """ Make a tiny sim for initializing the distributions """
    return ss.Sim(n_agents=100).init()


# %% The Dist object

@sc.timer()
def test_dist():
    sc.heading('Testing the Dist object: drawing every distribution, and reset')
    for name in ss.distributions.dist_list:
        rvs = getattr(ss.distributions, name)(name='test', strict=False).init().rvs(1000)
        assert len(rvs) == 1000, name

    # Reset to the most recent state, or the initial one
    dist = ss.random(seed=533, strict=False).init()
    r1, r2 = dist.rvs(m), dist.rvs(m)
    assert all(r1 != r2)
    dist.reset(-1)
    assert all(dist.rvs(m) == r2)
    dist.reset(0)
    assert all(dist.rvs(m) == r1)
    return


@sc.timer()
def test_dists():
    sc.heading('Testing the Dists container: seeds, reset and jump, and the order of drawing')
    def make():
        obj = sc.prettyobj()
        obj.a = sc.obj(mylist=[ss.random(), ss.Dist(distname='uniform', low=2, high=3)])
        obj.b = dict(d3=ss.weibull(shape=2), d4=ss.constant(value=0.3))
        return ss.dists.Dists(obj).init(sim=make_sim())
    sample = lambda dists: np.concatenate([dist(m) for dist in dists.dists.values()])
    dists = make()
    seeds = [d.seed for d in dists.dists.values()]
    assert len(set(seeds)) == len(seeds), 'Each distribution has its own seed'
    before = sample(dists)
    assert np.array_equal(before, sample(make())), 'Newly initialized objects should match'
    dists.reset() # Return to step 0, via the container
    assert np.array_equal(before, sample(dists))
    dists.jump(to=10)
    assert not np.array_equal(before, sample(dists)), 'After jumping, values should be different'
    return


@sc.timer()
def test_callable():
    sc.heading('Testing callable and array parameters, and repeated slots')
    sim = ss.debugtools.mock_sim(n_agents=10)
    def custom_loc(module, sim, uids):
        return sim.people.age[uids]
    uids = np.array([1, 3, 7, 9])
    for dist in [ss.normal, ss.lognormal]:
        draws = dist(name='callable', mean=custom_loc, std=1).init(sim=sim).rvs(uids)
        assert np.abs(sim.people.age[uids] - draws).mean() < 3, 'Outputs should match ages'
    d3 = ss.normal(name='callable2', mean=custom_loc, std=lambda uids: np.zeros(len(uids))).init(sim=sim)
    assert np.allclose(d3.rvs(uids), sim.people.age[uids]), 'Functions with different signatures, and scale=0, should work'

    # An array parameter is indexed by slot; repeated slots draw the same random numbers
    slots = np.array([4, 2, 3, 2, 2, 3])
    low = np.arange(len(slots))
    draws = ss.uniform(low=low, high=low + 1, strict=False).init(slots=slots).rvs(np.arange(len(slots)))
    frac, integ = np.modf(draws)
    assert np.allclose(integ, low), 'Integral part should match the low parameter'
    for s in np.unique(slots):
        assert np.allclose(frac[slots == s], frac[slots == s][0]), 'Same random numbers, so should be same fractional part'
    for dist, expected in [(ss.normal(mean=[3.5, 4.5], std=[0.9, 0.8]), [3.5, 4.5]), (ss.gamma(mean=[1, 2], std=[1, 1]), [1, 2])]: # Arrays given by moments, for a size
        assert np.allclose(dist.mock().rvs((20_000, 2)).mean(axis=0), expected, rtol=0.05), dist
    return


def test_choice():
    sc.heading('Testing ss.choice with a row of probabilities per agent')
    # A row per agent: each agent's choice independent of the others'
    n, k = 20_000, 4
    rng = np.random.default_rng(1)
    P = rng.random((n, k)) # Unnormalized: each row is normalized
    P2 = P.copy()
    P2[1::2] = rng.random((n//2, k)) # Change every other agent's row
    uids = ss.uids(np.arange(n))
    make = lambda Q: ss.choice(a=k, p=lambda uids: Q[uids], strict=False).init(slots=np.arange(n))
    vals = make(P).rvs(uids)
    assert np.array_equal(make(P).rvs(uids[::3]), vals[::3]), 'Drawing for fewer agents changed their choices'
    assert np.array_equal(make(P2).rvs(uids)[::2], vals[::2]), "Changing other agents' rows changed these agents' choices"
    Q = np.where((np.arange(n) % 2)[:, None], [0, 1, 1, 2], [3, 0, 0, 1]) # As weights
    vals = make(Q).rvs(uids)
    assert np.all(vals[0::2] != 1) and np.all(vals[1::2] != 0), 'A choice with probability 0 was chosen'
    assert np.isclose((vals[1::2] == 3).mean(), 0.5, atol=0.02) and np.isclose((vals[0::2] == 0).mean(), 0.75, atol=0.02), 'Frequencies do not match the rows'
    return


@sc.timer()
def test_choose_n():
    sc.heading('Testing ss.choose_n, with and without weights, and its common random numbers')
    uids = ss.uids(np.arange(1000))
    choose = lambda n, weights=None, die=False: ss.choose_n(n, weights=weights, die=die, strict=False).init(slots=uids).filter(uids)
    chosen = choose(20)
    assert len(chosen) == 20 and np.all(np.diff(chosen) > 0) # Exactly n, without replacement, and sorted
    assert len(choose(2000)) == 1000 # Can't choose more than all of them
    assert np.array_equal(choose(20, np.ones(1000)), chosen) # Equal weights are the same as no weights
    assert np.array_equal(choose(20, uids < 10), uids[:10]) # Zero weights are never chosen
    assert (choose(200, np.where(uids < 500, 9, 1)) < 500).mean() > 0.8 # Higher weights are chosen more often (expected 0.9)

    # Mostly the same agents when the candidates change slightly
    n = 10_000
    uids1 = ss.uids(np.arange(n))
    uids2 = uids1[np.random.default_rng(1).random(n) > 0.01] # Remove 1% of candidates
    weights = np.random.default_rng(2).random(n)
    for w1,w2 in [(None, None), (weights, weights[uids2])]:
        c1 = ss.choose_n(500, weights=w1, strict=False).init(slots=uids1).filter(uids1)
        c2 = ss.choose_n(500, weights=w2, strict=False).init(slots=uids1).filter(uids2)
        assert len(np.intersect1d(c1, c2)) > 480 # Should be roughly 495, vs. roughly 25 for an independent draw
    return


# %% Distributions and time

@sc.timer()
def test_timepar_dists():
    sc.heading('Testing distributions with durations and rates as their parameters')
    n = 100_000
    rtol = 0.1
    mods = [ss.debugtools.mock_module(dt=dt) for dt in [ss.years(1), ss.months(1), ss.weeks(1)]]
    year = mods[0]

    # Distributions that scale linearly with time, by the parameter given
    v = sc.obj(base=30.0, dur=ss.years(30), freq=ss.per(30, 'year'))
    for name,par in dict(constant='value', uniform='high', normal='mean', lognormal='mean', expon='mean', weibull='scale', gamma='scale').items():
        for module in mods:
            rvs = sc.obj({key:getattr(ss, name)(**{par:val}, name=key, module=module, strict=False).init().rvs(n) for key,val in v.items()})
            ratio = ss.years(1)/module.t.dt
            assert np.isclose(rvs.base.mean()*ratio, rvs.dur.mean(), rtol=rtol), f'Duration not close for {name}'
            assert np.isclose(rvs.base.mean()/ratio, rvs.freq.mean(), rtol=rtol), f'Rate not close for {name}'

    # A Poisson count of events in a step
    for module in mods:
        for lam in [ss.per(1, 'year'), ss.perday(1)]:
            mean = ss.poisson(mean=lam, module=module, strict=False).init().rvs(n).mean()
            assert np.isclose(mean, lam.to_events(module.t.dt), rtol=rtol), f'Poisson values do not match for {lam}' # The number in a step
    # The unit on the parameters, or as unit=
    for d in [ss.normal(ss.years(10), ss.years(0.1)), ss.normal(10, 0.1, unit=ss.years)]:
        assert np.isclose(d.mock(dt=ss.days(1)).rvs(n).mean(), ss.time.factors.years.days*10, rtol=1e-3)

    return


@sc.timer()
def test_round():
    sc.heading('Testing rounding')
    d = ss.normal(5.3, 2, round='nearest', strict=False)
    assert d.rvs(10).dtype == ss.dtypes.int # Rounded at creation
    assert d.rvs(10, round=False).dtype != ss.dtypes.int # Not rounded if overridden
    d = ss.constant(2.3, round='random', strict=False).init(slots=np.arange(10_000))
    assert np.isclose(d.rvs(ss.uids(np.arange(10_000))).mean(), 2.3, atol=0.03) # Stochastic rounding is unbiased
    assert np.all(d.rvs(ss.uids(np.arange(10)), round=True) == 2) # True is to the nearest integer
    steps = ss.lognormal(ss.days(4.5), ss.days(1.5), round='nearest').mock(dt=ss.days(0.5)).rvs(ss.uids(np.arange(10)))
    assert steps.dtype == ss.dtypes.int and steps.mean() > 6 # Durations are rounded to whole timesteps of half a day
    return


@sc.timer()
def test_shape_scale():
    sc.heading('Testing shape, scale, and mean for gamma, Weibull, and negative binomial, ')
    assert ss.gamma(2, 3).pars.scale == 3 and ss.weibull(2, 5).pars.scale == 5 # The second number is the scale; loc is by keyword
    vals = ss.gamma(mean=1800, std=400, strict=False).rvs(n)
    assert np.isclose(vals.mean(), 1800, rtol=0.01) and np.isclose(vals.std(), 400, rtol=0.02)

    # The negative binomial can be given by its mean, with n as the dispersion: the variance is mean + mean²/n
    counts = ss.nbinom(mean=20, n=0.5, strict=False).rvs(n)
    assert np.isclose(counts.mean(), 20, rtol=0.05) and np.isclose(counts.var(), 20 + 20**2/0.5, rtol=0.1)
    same = ss.nbinom(n=0.5, p=0.5/20.5, strict=False, seed=1).rvs(100)
    assert np.array_equal(ss.nbinom(mean=20, n=0.5, strict=False, seed=1).rvs(100), same) # The same as the equivalent p

    # loc= is a shift, as SciPy's loc: with mean=, the mean of the whole distribution
    g = ss.gamma(mean=10, std=3, loc=2, strict=False, seed=1).rvs(n)
    assert np.isclose(g.mean(), 10, rtol=0.02) and np.isclose(g.std(), 3, rtol=0.05) and g.min() >= 2
    assert np.isclose(ss.erlang(k=3, mean=10, loc=4, strict=False, seed=1).rvs(n).mean(), 10, rtol=0.02)
    return


@sc.timer()
def test_own_unit():
    sc.heading('Testing that a distribution with a unit draws in it outside a sim')
    dist = ss.erlang(k=4, mean=ss.days(10))
    vals = dist.rvs(n) # In days, with no sim and no strict=False
    assert np.isclose(vals.mean(), 10, rtol=0.05) and np.isclose(vals.std(), 5, rtol=0.1)
    assert np.isclose(ss.normal(3, 0.1, unit='weeks').rvs(n).mean(), 3, rtol=0.01)

    # Its mean, spread, and quantiles, in the units it draws in, with no SciPy needed
    assert np.isclose(dist.mean(), 10) and np.isclose(dist.std(), 5)
    lo, hi = dist.ppf([0.025, 0.975])
    assert np.isclose(dist.cdf(lo), 0.025) and np.isclose(dist.cdf(hi), 0.975)
    assert np.isclose(ss.uniform(1, 3, strict=False).mean(), 2) and ss.constant(ss.days(3)).std() == 0 # Drawn by NumPy, or not at all
    assert np.isclose(ss.normal(3, 0.1, unit='weeks').mock(dt=ss.days(1)).mean(), 21) # In steps, in a sim
    return


def test_engine_copy():
    sc.heading('Testing that the copy of a distribution a declared model draws from (in the sim\'s unit) draws what the distribution does')
    year = ss.debugtools.mock_module(dt=ss.years(1)) # Typed values are drawn in steps of a year, the unit the copy is in
    dists = [ss.random(), ss.uniform(1, 3), ss.normal(5, 1), ss.truncnorm(5, 2, low=0, high=10), ss.lognormal(log_mean=1, log_std=0.5), ss.lognormal(5, 2),
             ss.expon(4), ss.poisson(3), ss.beta_dist(2, 3), ss.nbinom(3, 0.5), ss.binom(10, 0.3), ss.randint(0, 10), ss.weibull(2, 3), ss.gamma(2, 3),
             ss.constant(7), ss.bernoulli(0.3), ss.choice([1, 5, 9], p=[0.2, 0.3, 0.5]), ss.histogram(values=[0.6, 0.3, 0.1], bins=[0, 6, 12, 18]),
             ss.lognormal(ss.days(50), ss.days(20)), ss.normal(5, 1, unit='month'), ss.histogram(values=[0.6, 0.4], bins=[0, 6, 12], unit='month')]
    for dist in dists:
        copy = ss.transitions.numeric_dist(dist, lambda v: ss.solvers.convert(v, 'year'))
        dist.init(seed=1, module=year, force=True)
        copy.init(seed=1, module=year, force=True)
        a, b = np.asarray(dist.rvs(m), dtype=float), np.asarray(copy.rvs(m), dtype=float)
        assert np.allclose(a, b), f'{dist}: the engine draws {b}, the distribution {a}'

    # A histogram for each group, as a dwell and as a trait, draws from each person's own (it drew uniform(0, 1))
    h0, h1 = ss.histogram(values=[1], bins=[0, 1], unit='year'), ss.histogram(values=[1], bins=[10, 11], unit='year')
    A, B = ss.states('A B')
    d = ss.Disease(A, B, go=ss.Transition(A, B, dur=ss.by(urban={False: h0, True: h1})), obj=ss.attr(ss.by(urban={False: h0, True: h1})), name='m', default_state='A')
    ppl = ss.People(1000, extra_states=[ss.BoolState('urban', default=ss.bernoulli(p=0.5))])
    sim = ss.Sim(d, people=ppl, start=2000, stop=2003, verbose=0).run()
    u = sim.people.auids
    urban, obj, went = sim.people.urban[u], sim.diseases.m.obj[u], sim.diseases.m.B[u]
    assert np.all(obj[urban] >= 10) and np.all(obj[~urban] <= 1)
    assert went[~urban].all() and not went[urban].any()
    return


# %% Random numbers in sims: common random numbers and independence

class CountInf(ss.Intervention):
    """ Store every infection state in a timepoints x people array """
    def init_pre(self, sim):
        super().init_pre(sim)
        self.n_agents = len(sim.people)
        self.arr = np.zeros((len(sim), self.n_agents))
        return

    def step(self):
        self.arr[self.sim.ti, :] = np.array(self.sim.diseases.sir.infected)[:self.n_agents]
        return


class OneMore(ss.Intervention):
    """ Add one additional agent and infection """
    def step(self):
        sim = self.sim
        if sim.ti == 10:
            new_uids = sim.people.grow(1, parent=ss.uids(0), age=0) # The 0th agent is the "mother"
            sim.people.age[new_uids] = -100 # Set to a very low number to never reach debut age
            sir = sim.diseases.sir
            sir.infect(new_uids)
            sir.ti_recovered[new_uids] = sim.ti + 1 # Infected, and recovered at the next timestep
            for dist in [sir.pars.dur_inf, sir.pars.p_death]:
                dist.jump_dt(force=True) # Already has correct dt value, but we need to force going back in time
        return


@sc.timer()
def test_worlds():
    sc.heading('Testing that one extra birth leads to one extra infection, and nothing else changes')
    pars = dict(
        start = 2000,
        stop = 2100,
        n_agents = 200,
        verbose = 0,
        diseases = ssl.AgentSIR(name='sir', init_prev=0.1, beta=1.0, dur_inf=20, p_death=0), # Written as code, since OneMore sets its prognoses directly
        networks = ss.RandomNet(), # Edges reform each step (< dur_inf), allowing SIR transmission
    )
    s1 = ss.Sim(pars, interventions=CountInf()).run()
    s2 = ss.Sim(pars, interventions=[CountInf(), OneMore()]).run()
    assert len(s2.people) == len(s1.people) + 1
    assert s2.summarize().value.sir.infection == s1.summarize().value.sir.infection + 1
    assert np.array_equal(s1.interventions.countinf.arr, s2.interventions.countinf.arr)
    return


@sc.timer()
def test_independence(thresh=0.1):
    sc.heading('Testing that variables are uncorrelated when created')
    sim = ss.Sim(
        n_agents = 1000,
        diseases = [
            dict(type='sir', init_prev=0.1, beta=dict(random=0.1, sexual=ss.peract(0.01))), # A probability per act on ss.SexualNet
            dict(type='sis', init_prev=0.1, beta=dict(random=0.05, sexual=ss.peract(0.01))),
            ssl.diseases.HIV(init_prev=0.1),
        ],
        networks = [
            dict(type='random', n_contacts=ss.poisson(8)),
            dict(type='sexual', debut=ss.constant(0), participation=0.5), # To avoid age correlations
        ]
    ).init()
    st = sim.people.states
    arrs = [st.age.values, st.female.values]
    for disease in sim.diseases.values():
        arrs.append((disease.infected if isinstance(disease, ss.AgentDisease) else disease.I).values) # Written as code, or declared
    for network in sim.networks.values():
        arrs.append(np.bincount(np.concatenate([network.edges.p1, network.edges.p2]), minlength=len(sim.people)))
    corr = np.corrcoef(np.array(arrs, dtype=float))
    max_corr = np.abs(corr - np.eye(len(arrs))).max()
    assert max_corr < thresh, f'The maximum correlation between variables was {max_corr}, above the threshold {thresh}'

    # A declared disease's random numbers do not depend on which other declared diseases are in the sim
    kw = dict(n_agents=1000, networks='random', dur=ss.years(10), verbose=0)
    alone = ss.Sim(diseases=ss.SIS(), **kw).run()
    both = ss.Sim(diseases=[ss.SIR(p_death=0), ss.SIS()], **kw).run()
    assert np.array_equal(alone.results.sis.I, both.results.sis.I) # The same people infected with SIS
    return


@sc.timer()
def test_crn_option():
    sc.heading('Testing the non-CRN fast path for transmission (ss.options.crn=False)')
    def run(crn, seed=1):
        with ss.options.context(crn=crn):
            sim = ss.Sim(diseases=ssl.AgentSIS(name='sis', beta=0.1, init_prev=0.05), networks=ss.RandomNet(), # The transmission of ss.AgentDisease
                         n_agents=2000, dur=ss.years(30), rand_seed=seed, verbose=0).run()
        return sim.results['sis']['infected'].values
    assert ss.options.crn is True, 'CRN should be the default'
    crn, fast = run(True), run(False)
    assert not np.array_equal(fast, crn), 'Non-CRN path should differ from the CRN path'
    assert np.isclose(crn[-5:].mean(), fast[-5:].mean(), rtol=0.1), 'Epidemic sizes should be similar'
    return


@sc.timer()
def test_streams():
    sc.heading('Testing dist.streams(): one stream of uniforms per agent, for compiled code, the same for an agent whoever else draws')
    @ss.jit
    def take(states, counts): # A varying number of uniforms for each agent, as compiled code draws them
        out = np.zeros((len(states), counts.max()))
        for i in range(len(states)):
            st = states[i]
            for k in range(counts[i]):
                u, st = ss.draws.next_u(st)
                out[i, k] = u
        return out
    every = ss.uids(np.arange(10_000))
    some = every[np.random.default_rng(1).random(len(every)) < 0.1] # A tenth of them
    make = lambda: ss.random(seed=3, strict=False).init(slots=every)
    d1, d2 = make(), make()
    all_states, some_states = d1.streams(every), d2.streams(some)
    assert all_states.dtype == np.uint64 and np.array_equal(all_states[some], some_states) # Each agent's stream is its own
    counts = np.random.default_rng(2).integers(1, 20, len(every))
    draws_all = take(all_states, counts)
    assert np.array_equal(draws_all[some], take(some_states, counts[some])) # The same sequences, whoever else draws
    u, st = ss.draws.next_u(all_states[0]) # The same in Python as compiled
    assert u == draws_all[0, 0] and 0 < u < 1
    assert not np.array_equal(d1.streams(every), all_states) # Another call, another stream
    return


@sc.timer()
def test_repeated_uids():
    sc.heading('Testing draws for UIDs that repeat in a call')
    class M(ss.Module):
        def __init__(self):
            super().__init__(name='m')
            self.r = ss.random(name='r')
            self.ch = ss.choice(a=1000, name='ch')
        def step(self):
            if self.ti == 1:
                uids = ss.uids(np.array([0, 0, 1, 0]))
                self.out = [self.r.rvs(uids), self.r.rvs(uids, repeats=True), self.ch.rvs(uids, repeats=True)]

    sims = [ss.Sim(M(), n_agents=10, dt=ss.days(1), start='2010-01-01', stop='2010-01-03', verbose=0).run() for i in range(2)]
    same, apart, chosen = sims[0].m.out
    assert same[0] == same[1] == same[3] # By default, as its slot
    assert len(set(apart[[0, 1, 3]])) == 3 # Each repeat its own
    assert len(set(chosen[[0, 1, 3]])) == 3
    assert np.array_equal(apart, sims[1].m.out[1]) # And still reproducible
    return


@sc.timer()
def test_crn_undriven():
    sc.heading('Testing that declaring a transition nobody takes, or one with a rate of 0, leaves every draw as it was, with and without strains')
    def make(extra=None, strains=False):
        S, L, I, R, C = ss.states('S L I R C')
        kw = dict(strains=ss.Strains(['DS', 'RIF'], fractions=[0.8, 0.2], rel_trans=[1, 0.6])) if strains else {}
        trs = dict(infection=S >> L @ ss.peryear(4)*I, activation=L >> I @ ss.peryear(0.5), recovery=I >> R @ ss.years(1), waning=R >> S @ ss.years(2))
        if extra == 'undriven':
            trs['clear'] = ss.Transition(L, C, rate=0) # Taken only by a product's cure=
        elif extra == 'zero':
            trs['death'] = I >> C @ ss.peryear(0) # A rate of 0, alongside a duration
        d = ss.Disease(S, L, I, R, C, **trs, name='d', init=dict(L=0.2, I=0.02), **kw)
        return ss.Sim(d, n_agents=1000, start=2000, stop=2003, dt=ss.months(1), verbose=0).run().results.d.infection.values
    for strains in [False, True]:
        base = make(strains=strains)
        for extra in ['undriven', 'zero']:
            assert np.array_equal(base, make(extra, strains)), f'Declaring {extra} changed the draws (strains={strains})'
    return


@sc.timer()
def test_crn_one_agent():
    sc.heading('Testing that making one agent immune leaves everyone else as they were, whichever moves the agent engine takes')
    def make(seed, immune=None, crn=True):
        S, E, I, H, R = ss.states('S E I H R')
        d = ss.Disease(S, E, I, H, R,
            infection = S >> E @ ss.perday(0.1)*(I + H),
            onset     = ss.Transition(E, {I: 0.8, R: None}, dur=ss.lognormal(ss.days(3), ss.days(1))), # A split after a dwell time
            severe    = ss.Transition(I, H, p=0.2, dur=ss.weibull(shape=2, scale=ss.days(4))), # Branches chosen on entry
            mild      = ss.Transition(I, {R: 0.9, ss.stay: None}, dur=ss.uniform(ss.days(3), ss.days(9)), p=None),
            leave     = ss.Transition(H, R, dur=ss.normal(ss.by(sex=dict(female=ss.days(5), male=ss.days(8))), ss.days(1))),
            die       = H >> ss.dead @ ss.perday(0.03), # A competing rate beside a dwell time
            init_prev=0.02, name='d')
        vx = ss.vaccinate(efficacy=1.0, who=lambda sim: ss.uids([immune]), when=[0], coverage=1.0, name='vx') if immune is not None else None
        with ss.options.context(crn=crn):
            sim = ss.Sim(d, networks=ss.RandomNet(n_contacts=4, method='safe'), interventions=vx, n_agents=1000, unit='day', stop=40, rand_seed=seed, verbose=0).run()
        return {k: v.raw.copy() for k, v in sim.diseases.d.__dict__.items() if isinstance(v, ss.Arr)}

    def changed(seed, crn=True):
        """ The arrays in which anyone but the immune agent differs """
        base = make(seed)
        uid = np.nonzero(base['E'])[0][0] # Exposed at the end, and so never infectious (nothing wanes): they infected no one
        other = make(seed, immune=uid, crn=crn)
        assert base['E'][uid] and not other['E'][uid], 'The immune agent should not have been infected'
        rest = np.arange(len(base['E'])) != uid
        return [k for k, v in base.items() if not np.array_equal(v[rest], other[k][rest], equal_nan=True)]

    for seed in [0, 1]:
        assert not (diff := changed(seed)), f'Making one agent immune changed other agents\' {diff} (seed {seed})'
        with mock.patch.object(ss.solvers.agents_compiled, 'compiled_reasons', lambda engine: ['off']): # The agent engine's own moves, not the compiled ones
            assert not (diff := changed(seed)), f'Making one agent immune changed other agents\' {diff} with the agent engine\'s own moves (seed {seed})'
    assert changed(0, crn=False), 'Without CRN, making one agent immune should change other agents' # So the test can see CRN being lost
    return


@sc.timer()
def test_crn_code_moves():
    sc.heading('Testing that moves from code leave the draws of the agents they do not move as they were')
    moved = ss.uids(np.arange(0, 400, 7))
    def make(move):
        S, I, R, W = ss.states('S I R W')
        sir = ss.Disease(S >> I @ 0.1*I, ss.Transition(I, R, dur=ss.expon(10)), R >> W @ 0.0, name='sir', init_prev=0.02)
        class Mover(ss.Intervention):
            def step(self):
                if move:
                    ss.move(self.sim.diseases.sir.R >> self.sim.diseases.sir.W, uids=moved)
        sim = ss.Sim(sir, interventions=Mover(name='mover'), networks='random', n_agents=2000, start=0, stop=60, rand_seed=3, solver=ss.solvers.abm(keep_times=True), verbose=0).run()
        return sim.diseases.sir

    a, b = make(False), make(True)
    assert b.W.n > 0, 'The intervention should have moved someone'
    others = np.setdiff1d(np.arange(2000), moved)
    for key in ['ti_I', 'ti_R']:
        assert np.array_equal(a[key][others], b[key][others], equal_nan=True), f'Moves from code changed {key} of agents they did not move'
    return


def test_names_keep_crn():
    sc.heading('Testing that a module\'s random numbers come from its name (given, else its default numbered within its kind), never its variable')
    kw = dict(start=2000, stop=2005, verbose=0)
    def infections(form, solver='abm'):
        pop = dict(n_agents=500) if solver == 'abm' else dict(total_pop=500)
        if form == 'sir':
            sir = ss.SIR()
            sim = ss.Sim(diseases=sir, networks='random' if solver == 'abm' else None, solver=solver, **pop, **kw)
        else:
            measles = ss.SIR()
            mynet = ss.RandomNet()
            sim = ss.Sim(diseases=measles, networks=mynet if solver == 'abm' else None, solver=solver, **pop, **kw)
        return sim.run().summarize(f'{form}.infection')
    for solver in ['abm', 'binomial']:
        assert infections('sir', solver) == infections('measles', solver), f'Naming the variables should not change a {solver} run'
    sim = ss.Sim(diseases=[ss.SIR(name='a'), ss.SIR(), ss.SIR(), ss.SIS()], networks='random', n_agents=100, **kw).init()
    assert ss.dists.seed_names(sim) == dict(randomnet='randomnet', a='a', sir='sir', sir_1='sir2', sis='sis') # A given name, else numbered within each kind in the order given
    return


if __name__ == '__main__':
    T = sc.timer()
    test_dist()
    test_dists()
    test_callable()
    test_choice()
    test_choose_n()
    test_timepar_dists()
    test_round()
    test_shape_scale()
    test_own_unit()
    test_engine_copy()
    test_worlds()
    test_independence()
    test_crn_option()
    test_streams()
    test_repeated_uids()
    test_crn_undriven()
    test_crn_one_agent()
    test_crn_code_moves()
    test_names_keep_crn()
    T.toc()
