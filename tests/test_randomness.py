"""
Test the Dists object in internals/dists.py
"""

# %% Imports and settings
from unittest import mock
import pytest
import numpy as np
import sciris as sc
import scipy.stats as sps
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl

n = 5 # Default number of samples

def make_dists():
    """ Make a Dists object with two distributions in it """
    sim = ss.Sim(n_agents=100).init() # Need an empty sim to initialize properly
    dists = ss.dists.Dists([ss.random(name='test'), ss.random(name='test')])
    dists.init(sim=sim)
    return dists


# %% Define the tests

@sc.timer()
def test_seed():
    """ Test assignment of seeds """
    sc.heading('Testing assignment of seeds')

    # Create and initialize two distributions
    dists = make_dists()
    dist0, dist1 = dists.dists.values()

    print(f'Dists dist0 and dist1 were assigned seeds {dist0.seed} and {dist1.seed}, respectively')
    assert dist0.seed != dist1.seed

    # Re-initializing the sim's dists should not change their names or seeds
    sim = ss.Sim(n_agents=100, diseases='sis', networks='random', demographics=True).init()
    seeds = {k:d.seed for k,d in sim.dists.dists.items()}
    sim._init_dists()
    assert seeds == {k:d.seed for k,d in sim.dists.dists.items()}
    return dist0, dist1


@sc.timer()
def test_reset_jump(n=n):
    """ Sample, reset or jump, sample """
    sc.heading('Testing sample, reset, sample; and sample, jump, sample')
    out = []
    for how in ['reset', 'jump']:
        dists = make_dists()
        sample = lambda: np.concatenate([dist(n) for dist in dists.dists.values()])
        before = sample()
        if how == 'reset':
            dists.reset() # Return to step 0, via the container
        else:
            dists.jump(to=10) # Jump to the 10th step
        after = sample()
        assert np.array_equal(before, after) == (how == 'reset')
        out.append((before, after))
    return out


@sc.timer()
def test_order(n=n):
    """ Ensure sampling from one RNG doesn't affect another """
    sc.heading('Testing from multiple random number generators to test if sampling order matters')
    dists = make_dists()
    d0, d1 = dists.dists.values()

    # Sample d0, d1
    before = d0(n)
    _ = d1(n)

    dists.reset()

    # Sample d1, d0
    _ = d1(n)
    after = d0(n)

    print(f'When sampling rng0 before rng1: {before}')
    print(f'When sampling rng0 after rng1: {after}')
    assert np.array_equal(before, after)

    return before, after


# %% Test a minimally different world

class CountInf(ss.Intervention):
    """ Store every infection state in a timepoints x people array """
    def init_pre(self, sim):
        super().init_pre(sim)
        n_agents = len(sim.people)
        self.arr = np.zeros((len(sim), n_agents))
        self.n_agents = n_agents
        return

    def step(self):
        self.arr[self.sim.ti, :] = np.array(self.sim.diseases.sir.infected)[:self.n_agents]
        return


class OneMore(ss.Intervention):
    """ Add one additional agent and infection """
    def __init__(self, ti_apply=10):
        super().__init__()
        self.ti_apply = ti_apply
        return

    def step(self):
        """ Create an extra agent """
        sim = self.sim
        if sim.ti == self.ti_apply:
            new_uids = sim.people.grow(1, parent=ss.uids(0), age=0) # The 0th agent is the "mother"
            sim.people.age[new_uids] = -100 # Set to a very low number to never reach debut age

            # Infect that agent and immediately recover
            sir = sim.diseases.sir
            sir.infect(new_uids)
            sir.ti_recovered[new_uids] = sim.ti + 1 # Reset recovery time to next timestep

            # Reset the random states
            p = sir.pars
            for dist in [p.dur_inf, p.p_death]:
                dist.jump_dt(force=True) # Already has correct dt value, but we need to force going back in time

        return


def plot_infs(s1, s2):
    """ Compare infection arrays from two sims """
    a1 = s1.interventions.countinf.arr
    a2 = s2.interventions.countinf.arr

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.pcolormesh(a1.T)
    plt.xlabel('Timestep')
    plt.ylabel('Person')
    plt.title('Baseline')

    plt.subplot(1,3,2)
    plt.pcolormesh(a2.T)
    plt.title('OneMore')

    plt.subplot(1,3,3)
    plt.pcolormesh(a2.T - a1.T)
    plt.title('Difference')

    sc.figlayout()
    return fig


@sc.timer()
def test_worlds(do_plot=False):
    """ Test that one extra birth leads to one extra infection """
    sc.heading('Testing worlds...')

    res = sc.obj()

    pars = dict(
        start = 2000,
        stop = 2100,
        n_agents = 200,
        verbose = 1,
        diseases = ssl.AgentSIR( # Written as code, since OneMore sets its prognoses directly
            name = 'sir',
            init_prev = 0.1,
            beta = 1.0,
            dur_inf = 20,
            p_death = 0, # Here since analyzer can't handle variable numbers of people
        ),
        networks = ss.RandomNet()  # Edges reform each step (< dur_inf), allowing SIR transmission
    )
    s1 = ss.Sim(pars, interventions=CountInf())
    s2 = ss.Sim(pars, interventions=[CountInf(), OneMore()])

    s1.run()
    s2.run()

    sum1 = s1.summarize()
    sum2 = s2.summarize()
    res.sum1 = sum1
    res.sum2 = sum2

    if do_plot:
        s1.plot()
        plot_infs(s1, s2)
        plt.show()

    l1 = len(s1.people)
    l2 = len(s2.people)
    i1 = sum1.value.sir_infection
    i2 = sum2.value.sir_infection
    a1 = s1.interventions.countinf.arr
    a2 = s2.interventions.countinf.arr
    assert l2 == l1 + 1, f'Expected one more person in s2 ({l2}) than s1 ({l1})'
    assert i2 == i1 + 1, f'Expected one more infection in s2 ({i2}) than s1 ({i1})'
    assert (a1 != a2).sum() == 0, f'Expected infection arrays to match:\n{a1}\n{a2}'

    return res


@sc.timer()
def test_independence(do_plot=False, thresh=0.1):
    """ Test that when variables are created, they are uncorrelated """
    sc.heading('Testing independence...')

    # Create the sim and initialize (do not run)
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
    )
    sim.init()

    # Assemble measures
    st = sim.people.states
    arrs = sc.obj()
    arrs.age = st.age.values
    arrs.sex = st.female.values
    for key,disease in sim.diseases.items():
        arrs[f'{key}_init'] = (disease.infected if isinstance(disease, ss.AgentDisease) else disease.I).values # Written as code, or declared
    for key,network in sim.networks.items():
        data = np.zeros(len(sim.people))
        for p in ['p1', 'p2']:
            for uid in network.edges[p]:
                data[uid] += 1 # Could also use a histogram
        arrs[key] = data

    # Compute the correlations
    n = len(arrs)
    stats = np.zeros((n,n))
    for i,arr1 in arrs.enumvals():
        for j,arr2 in arrs.enumvals():
            if i != j:
                stats[i,j] = np.corrcoef(arr1, arr2)[0,1]

    # Optionally plot
    if do_plot:
        plt.figure()
        plt.imshow(stats)
        ticks = np.arange(n)
        labels = arrs.keys()
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)
        plt.xticks(rotation=15)
        plt.colorbar()
        sc.figlayout()
        plt.show()

    # Test that everything is independent
    max_corr = abs(stats).max()
    assert max_corr < thresh, f'The maximum correlation between variables was {max_corr}, above the threshold {thresh}'

    # A declared disease's random numbers do not depend on which other declared diseases are in the sim
    kw = dict(n_agents=1000, networks='random', dur=ss.years(10), verbose=0)
    alone = ss.Sim(diseases=ss.SIS(), **kw).run()
    both = ss.Sim(diseases=[ss.SIR(p_death=0), ss.SIS()], **kw).run()
    assert np.array_equal(alone.results.sis.I, both.results.sis.I) # The same people infected with SIS

    return sim


@sc.timer()
def test_combine_rands(do_plot=False):
    n = int(1e6)
    atol = 1e-3
    target = 0.5
    np.random.seed(2)
    a = np.random.randint(np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
    b = np.random.randint(np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
    c = ss.distributions.combine_rands(a, b)
    if do_plot:
        plt.figure()
        for i,k,v in sc.obj(a=a,b=b,combined=c).enumitems():
            plt.subplot(3,1,i+1)
            plt.hist(v)
            plt.title(k)
        sc.figlayout()
        plt.show()

    mean = c.mean()
    assert np.isclose(mean, target, atol=atol), f'Expected value to be 0.5±{atol}, not {mean}'
    ks = sps.kstest(c, sps.uniform(0,1).cdf)
    assert ks.pvalue > 0.05, f'Distribution does not seem to be uniform, p={ks.pvalue}<0.05'

    # Signed input would give values in [-0.5, 0.5] rather than [0, 1], so it should be refused
    # rather than cast, since casting would silently copy the array
    with pytest.raises(TypeError):
        ss.distributions.combine_rands(a.view(np.int64), b.view(np.int64))

    return c


@sc.timer()
def test_crn_option():
    """ The opt-in non-CRN fast path (ss.options.crn=False) for transmission """
    def run(crn, seed=1):
        with ss.options.context(crn=crn):
            sim = ss.Sim(diseases=ssl.AgentSIS(name='sis', beta=0.1, init_prev=0.05), networks=ss.RandomNet(), # The transmission of ss.AgentDisease
                         n_agents=2000, dur=ss.years(30), rand_seed=seed, verbose=0)
            sim.run()
        return sim.results['sis']['infected'].values

    assert ss.options.crn is True, 'CRN should be the default'

    # CRN is reproducible, and is the default (not changed by toggling the option back)
    crn = run(True)
    assert np.array_equal(crn, run(True)), 'CRN path should be reproducible'

    # Non-CRN is reproducible given the same seed, but differs from CRN (it is a different RNG path)
    fast = run(False)
    assert np.array_equal(fast, run(False)), 'Non-CRN path should be reproducible for a fixed seed'
    assert not np.array_equal(fast, crn), 'Non-CRN path should differ from the CRN path'

    # Statistically equivalent: similar epidemic size at the end
    crn_eq = crn[-5:].mean()
    fast_eq = fast[-5:].mean()
    assert np.isclose(crn_eq, fast_eq, rtol=0.1), f'Epidemic sizes should be similar: {crn_eq} vs {fast_eq}'

    # Per-instance override works independently of the global option
    assert ss.multi_random('a', 'b', crn=False).crn is False
    return fast


@sc.timer()
def test_multi_random_length():
    """
    Confirm multi_random output length

    Tests a regression occurring in 3.5.1 due to dtypes with numba functions
    """
    sim = ss.Sim(n_agents=100).init()

    def make_multi(n_dists):
        mr = ss.multi_random('placeholder')
        mr.dists = [ss.random(name=f'mr_d{i}') for i in range(n_dists)]
        mr.init(sim=sim)
        return mr

    for n_dists in [2, 3]: # 2 dists uses combine2_rvs; 3 dists uses combine_rvs
        mr = make_multi(n_dists)
        for size in [1, 5]: # A singleton and a multi-element case are enough to catch 2*n doubling
            uid_lists = [ss.uids(np.arange(i*size, (i+1)*size)) for i in range(n_dists)]
            rvs = mr.rvs(*uid_lists)
            assert len(rvs) == size, f'Expected len {size} but got {len(rvs)} (n_dists={n_dists}); rvs() is returning double-length arrays'
            assert np.issubdtype(np.asarray(rvs).dtype, np.floating), f'Expected a float dtype, got {np.asarray(rvs).dtype}'
            assert rvs.min() >= 0 and rvs.max() <= 1, f'Values should be uniforms in [0, 1], got [{rvs.min()}, {rvs.max()}]'

    return


@sc.timer()
def test_choose_n_crn():
    """ ss.choose_n() should choose mostly the same agents when the candidates change slightly """
    sc.heading('Testing CRN for ss.choose_n')
    n = 10_000
    uids1 = ss.uids(np.arange(n))
    uids2 = uids1[np.random.default_rng(1).random(n) > 0.01] # Remove 1% of candidates
    weights = np.random.default_rng(2).random(n)
    for w1,w2 in [(None, None), (weights, weights[uids2])]:
        c1 = ss.choose_n(500, weights=w1, strict=False).init(slots=uids1).filter(uids1)
        c2 = ss.choose_n(500, weights=w2, strict=False).init(slots=uids1).filter(uids2)
        overlap = len(np.intersect1d(c1, c2))
        print(f'Overlap after removing 1% of candidates: {overlap}/500')
        assert overlap > 480 # Should be roughly 495, vs. roughly 25 for an independent draw
    return overlap


@sc.timer()
def test_streams():
    """ dist.streams(): one stream of uniforms per agent, for compiled code, the same for an agent whoever else draws """
    sc.heading('Testing per-agent streams for compiled code')
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
    assert ss.draws.streams([0.25])[0] == ss.draws.stream(0.25) # The same seeds as the rejection samplers' streams
    vals = draws_all[draws_all > 0]
    assert abs(vals.mean() - 0.5) < 0.01 and abs(np.corrcoef(draws_all[:, 0], draws_all[:, 1])[0, 1]) < 0.05
    return draws_all


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

    sims = [ss.Sim(n_agents=10, modules=M(), dt=ss.days(1), start='2010-01-01', stop='2010-01-03', verbose=0).run() for i in range(2)]
    same, apart, chosen = sims[0].m.out
    assert same[0] == same[1] == same[3] # By default, as its slot
    assert len(set(apart[[0, 1, 3]])) == 3 # Each repeat its own
    assert len(set(chosen[[0, 1, 3]])) == 3
    assert np.array_equal(apart, sims[1].m.out[1]) # And still reproducible
    return sims


@sc.timer()
def test_crn_undriven():
    """ Test that declaring a transition nobody takes, or one with a rate of 0, leaves every draw of the run as it was, with and without strains """
    sc.heading('Testing that an undriven transition leaves the draws as they were')
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
    return base


@sc.timer()
def test_crn_one_agent():
    """ With common random numbers, making immune one agent who would have been infected too late to infect anyone leaves every other agent exactly as they were, whichever moves the agent engine takes """
    sc.heading('Testing that making one agent immune leaves everyone else as they were')
    def make(seed, immune=None, crn=True):
        S, E, I, H, R = ss.states('S E I H R')
        d = ss.Disease(S, E, I, H, R,
            infection = S >> E @ ss.perday(0.1)*(I + H),
            onset     = ss.Transition(E, {I: 0.8, R: None}, dur=ss.lognormal(ss.days(3), ss.days(1))), # A split after a dwell time
            severe    = ss.Transition(I, H, p=0.2, dur=ss.weibull(shape=2, scale=ss.days(4))), # Branches chosen on entry
            mild      = ss.Transition(I, {R: 0.9, ss.stay: None}, dur=ss.uniform(ss.days(3), ss.days(9))),
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
    """ With common random numbers, an intervention that moves some agents from code, between states that transmission does not read, leaves every other agent's draws as they were """
    sc.heading('Testing that moves from code leave the draws of the agents they do not move as they were')
    moved = ss.uids(np.arange(0, 400, 7))
    def make(move):
        S, I, R, W = ss.states('S I R W')
        sir = ss.Disease(S >> I @ 0.1*I, ss.Transition(I, R, dur=ss.expon(10)), R >> W @ 0.0, name='sir', init_prev=0.02)
        class Mover(ss.Intervention):
            def step(self):
                if move:
                    ss.move(self.sim.diseases.sir.R >> self.sim.diseases.sir.W, uids=moved)
        sim = ss.Sim(sir, interventions=Mover(name='mover'), networks='random', n_agents=2000, start=0, stop=60, rand_seed=3, keep_times=True, verbose=0).run()
        return sim.diseases.sir

    def check():
        a, b = make(False), make(True)
        assert b.W.n > 0, 'The intervention should have moved someone'
        others = np.setdiff1d(np.arange(2000), moved)
        for key in ['ti_I', 'ti_R']:
            assert np.array_equal(a[key][others], b[key][others], equal_nan=True), f'Moves from code changed {key} of agents they did not move'

    check()
    with mock.patch.object(ss.solvers.agents_compiled, 'compiled_reasons', lambda engine: ['off']): # The agent engine's own moves, not the compiled ones
        check()
    return


# %% Run as a script
def test_offset():
    """ A distribution with offset= draws the same numbers wherever it is in the sim, and two with the same offset are refused """
    class Holder(ss.Analyzer):
        def __init__(self, offset='my_stream', **kwargs):
            super().__init__(**kwargs)
            self.my_dist = ss.random(offset=offset)
    def draws(name, offset='my_stream'):
        sim = ss.Sim(diseases='sis', analyzers=Holder(offset=offset, name=name), n_agents=100, start=2000, stop=2001, dt=ss.years(1), verbose=0).init()
        return sim.analyzers[name].my_dist(sim.people.auids)
    for offset in ['my_stream', 123]:
        assert np.array_equal(draws('holder1', offset), draws('holder2', offset)), f'offset={offset!r} should not depend on where the distribution is'
    assert not np.array_equal(draws('holder1', 'my_stream'), draws('holder1', 'other')), 'Different offsets should draw differently'
    with pytest.raises(ss.errors.DistSeedRepeatError):
        ss.Sim(diseases='sis', analyzers=[Holder(name='a'), Holder(name='b')], n_agents=100, start=2000, stop=2001, dt=ss.years(1), verbose=0).init()


if __name__ == '__main__':
    T = sc.timer()
    do_plot = False

    o1 = test_seed()
    o2 = test_reset_jump(n)
    o4 = test_order(n)
    o5 = test_worlds(do_plot=do_plot)
    o6 = test_independence(do_plot=do_plot)
    o7 = test_combine_rands(do_plot=do_plot)
    o8 = test_crn_option()
    o9 = test_multi_random_length()
    o10 = test_choose_n_crn()
    o11 = test_streams()
    o12 = test_repeated_uids()
    o13 = test_crn_undriven()
    o14 = test_crn_one_agent()
    o15 = test_crn_code_moves()

    T.toc()

