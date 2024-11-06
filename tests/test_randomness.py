"""
Test the Dists object from distributions.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import scipy.stats as sps
import matplotlib.pyplot as plt
import starsim as ss

n = 5 # Default number of samples

def make_dist(name='test', **kwargs):
    """ Make a default Dist for testing """
    dist = ss.random(name=name, **kwargs)
    return dist

def make_dists(**kwargs):
    """ Make a Dists object with two distributions in it """
    sim = ss.Sim(n_agents=100).init() # Need an empty sim to initialize properly
    distlist = [make_dist(), make_dist()]
    dists = ss.Dists(distlist)
    dists.init(sim=sim)
    return dists


# %% Define the tests

def test_seed():
    """ Test assignment of seeds """
    sc.heading('Testing assignment of seeds')

    # Create and initialize two distributions
    dists = make_dists()
    dist0, dist1 = dists.dists.values()

    print(f'Dists dist0 and dist1 were assigned seeds {dist0.seed} and {dist1.seed}, respectively')
    assert dist0.seed != dist1.seed
    return dist0, dist1


def test_reset(n=n):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    dists = make_dists()
    distlist = dists.dists.values()

    # Reset via the container, but only
    before = sc.autolist()
    after = sc.autolist()
    for dist in distlist:
        before += list(dist(n))
    dists.reset() # Return to step 0
    for dist in distlist:
        after += list(dist(n))

    print(f'Initial sample:\n{before}')
    print(f'After reset sample:\n{after}')
    assert np.array_equal(before, after)

    return before, after


def test_jump(n=n):
    """ Sample, jump, sample """
    sc.heading('Testing sample, jump, sample')
    dists = make_dists()
    distlist = dists.dists.values()

    # Jump via the contianer
    before = sc.autolist()
    after = sc.autolist()
    for dist in distlist:
        before += list(dist(n))
    dists.jump(to=10) # Jump to 10th step
    for dist in distlist:
        after += list(dist(n))

    print(f'Initial sample:\n{before}')
    print(f'After jump sample:\n{after}')
    assert not np.array_equal(before, after)

    return before, after


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

    def init_pre(self, sim):
        super().init_pre(sim)
        one_birth = ss.Pregnancy(name='one_birth', rel_fertility=0) # Ensure no default births
        one_birth.init_pre(sim)
        self.one_birth = one_birth
        return

    def step(self):
        """ Create an extra agent """
        sim = self.sim
        if sim.ti == self.ti_apply:
            new_uids = self.one_birth.make_embryos(ss.uids(0)) # Assign 0th agent to be the "mother"
            sim.people.age[new_uids] = -100 # Set to a very low number to never reach debut age

            # Infect that agent and immediately recover
            sir = sim.diseases.sir
            sir.set_prognoses(new_uids)
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


def test_worlds(do_plot=False):
    """ Test that one extra birth leads to one extra infection """
    sc.heading('Testing worlds...')

    res = sc.objdict()

    pars = dict(
        start = 2000,
        stop = 2100,
        n_agents = 200,
        verbose = 0.05,
        diseases = dict(
            type = 'sir',
            init_prev = 0.1,
            beta = 1.0,
            dur_inf = 20,
            p_death = 0, # Here since analyzer can't handle variable numbers of people
        ),
        networks = dict(
            type = 'embedding',
            duration = 5, # Must be shorter than dur_inf for SIR transmission to occur
        ),
    )
    s1 = ss.Sim(pars=pars, interventions=CountInf())
    s2 = ss.Sim(pars=pars, interventions=[CountInf(), OneMore()])

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
    i1 = sum1.sir_cum_infections
    i2 = sum2.sir_cum_infections
    a1 = s1.interventions.countinf.arr
    a2 = s2.interventions.countinf.arr
    assert l2 == l1 + 1, f'Expected one more person in s2 ({l2}) than s1 ({l1})'
    assert i2 == i1 + 1, f'Expected one more infection in s2 ({i2}) than s1 ({i1})'
    assert (a1 != a2).sum() == 0, f'Expected infection arrays to match:\n{a1}\n{a2}'

    return res


def test_independence(do_plot=False, thresh=0.1):
    """ Test that when variables are created, they are uncorrelated """
    sc.heading('Testing independence...')

    # Create the sim and initialize (do not run)
    sim = ss.Sim(
        n_agents = 1000,
        diseases = [
            dict(type='sir', init_prev=0.1),
            dict(type='sis', init_prev=0.1),
            dict(type='hiv', init_prev=0.1),
        ],
        networks = [
            dict(type='random', n_contacts=ss.poisson(8)),
            dict(type='mf', debut=ss.constant(0), participation=0.5), # To avoid age correlations
        ]
    )
    sim.init()

    # Assemble measures
    st = sim.people.states
    arrs = sc.objdict()
    arrs.age = st.age.values
    arrs.sex = st.female.values
    for key,disease in sim.diseases.items():
        arrs[f'{key}_init'] = disease.infected.values
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

    return sim


def test_combine_rands(do_plot=False):
    n = int(1e6)
    atol = 1e-3
    target = 0.5
    np.random.seed(2)
    a = np.random.randint(np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
    b = np.random.randint(np.iinfo(np.uint64).max, size=n, dtype=np.uint64)
    c = ss.utils.combine_rands(a, b)
    if do_plot:
        plt.figure()
        for i,k,v in sc.objdict(a=a,b=b,combined=c).enumitems():
            plt.subplot(3,1,i+1)
            plt.hist(v)
            plt.title(k)
        sc.figlayout()
        plt.show()

    mean = c.mean()
    assert np.isclose(mean, target, atol=atol), f'Expected value to be 0.5±{atol}, not {mean}'
    ks = sps.kstest(c, sps.uniform(0,1).cdf)
    assert ks.pvalue > 0.05, f'Distribution does not seem to be uniform, p={ks.pvalue}<0.05'
    return c


# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()
    do_plot = True

    o1 = test_seed()
    o2 = test_reset(n)
    o3 = test_jump(n)
    o4 = test_order(n)
    o5 = test_worlds(do_plot=do_plot)
    o6 = test_independence(do_plot=do_plot)
    o7 = test_combine_rands(do_plot=do_plot)

    T.toc()

