"""
Test the Dist object from distributions.py
"""

import numpy as np
import sciris as sc
import scipy.stats as sps
import starsim as ss
import matplotlib.pyplot as plt
import pytest

n = 1_000_000 # For testing statistical properties and performance of distributions
m = 5 # For just testing that drawing works
dpy = 365.25 # Number of days per year
sc.options(interactive=False)


def plot_rvs(rvs, times=None, nrows=None):
    fig = plt.figure(figsize=(12,12))
    nrows, ncols = sc.getrowscols(len(rvs), nrows=nrows)
    for i,name,r in rvs.enumitems():
        plt.subplot(nrows, ncols, i+1)
        plt.hist(r.astype(float))
        title = times[name] if times else name
        plt.title(title)
        sc.commaticks()
    sc.figlayout()
    return fig


def make_sim():
    """ Make a tiny sim for initializing the distributions """
    sim = ss.Sim(n_agents=100).init() # Need an empty sim to initialize properly
    return sim


# %% Define the tests
def test_dist(m=m):
    """ Test the Dist class """
    sc.heading('Testing the basic Dist call')
    dist = ss.Dist(distname='random', name='test', strict=False)
    dist.init()
    rvs = dist(m)
    print(rvs)
    assert 0 < rvs.min() < 1, 'Values should be between 0 and 1'

    # Test other options
    dist.show_state()
    dist.plot_hist()
    return rvs


def test_custom_dists(n=n, do_plot=False):
    """ Test all custom dists """
    sc.heading('Testing all custom distributions')

    o     = sc.objdict()
    dists = sc.objdict()
    rvs   = sc.objdict()
    times = sc.objdict()
    for name in ss.dist_list:
        dist_class = getattr(ss, name)
        dist = dist_class(name='test', strict=False)
        dist.init()
        dists[name] = dist
        sc.tic()
        rvs[name] = dist.rvs(n)
        times[name] = sc.toc(name, unit='ms', output='message')
        print(f'{name:10s}: mean = {rvs[name].mean():n}')

    if do_plot:
        plot_rvs(rvs, times=times, nrows=5)

    o.dists = dists
    o.rvs = rvs
    o.times = times
    return o


def test_dists(n=n, do_plot=False):
    """ Test the Dists container """
    sc.heading('Testing Dists container')
    testvals = np.zeros((2,2))

    # Create the objects twice
    for i in range(2):

        # Create a complex object containing various distributions
        obj = sc.prettyobj()
        obj.a = sc.objdict()
        obj.a.mylist = [ss.random(), ss.Dist(distname='uniform', low=2, high=3)]
        obj.b = dict(d3=ss.weibull(c=2), d4=ss.constant(v=0.3))
        dists = ss.Dists(obj).init(sim=make_sim())

        # Call each distribution twice
        for j in range(2):
            rvs = sc.objdict()
            for key,dist in dists.dists.items():
                rvs[str(dist)] = dist(n)
                dist.jump() # Reset
            testvals[i,j] = rvs[0][283] # Pick one of the random values and store it

    # Check that results are as expected
    print(testvals)
    assert np.all(testvals[0,:] == testvals[1,:]), 'Newly initialized objects should match'
    assert np.all(testvals[:,0] != testvals[:,1]), 'After jumping, values should be different'

    if do_plot:
        plot_rvs(rvs)

    o = sc.objdict()
    o.dists = dists
    o.rvs = rvs
    return dists


def test_scipy(m=m):
    """ Test that SciPy distributions also work """
    sc.heading('Testing SciPy distributions')

    # Make SciPy distributions in two different ways
    dist1 = ss.Dist(dist=sps.expon, name='scipy', scale=2, strict=False).init() # Version 1: callable
    dist2 = ss.Dist(dist=sps.expon(scale=2), name='scipy', strict=False).init() # Version 2: frozen
    rvs1 = dist1(m)
    rvs2 = dist2(m)

    # Check that they match
    print(rvs1)
    assert np.array_equal(rvs1, rvs2), 'Arrays should match'

    return dist1, dist2


def test_exceptions(m=m):
    """ Check that exceptions are being appropriately raised """
    sc.heading('Testing exceptions and strict')

    # Create a strict distribution
    dist = ss.random(strict=True, auto=False)
    with pytest.raises(ss.distributions.DistNotInitializedError):
        dist(m) # Check that we can't call an uninitialized

    # Initialize and check we can't call repeatedly
    dist.init(trace='test', sim=make_sim())
    rvs = dist(m)
    with pytest.raises(ss.distributions.DistNotReadyError):
        dist(m) # Check that we can't call an already-used distribution

    # Check that we can with a non-strict Dist
    dist2 = ss.random(strict=False)
    dist2.init(trace='test')
    rvs2 = sc.autolist()
    for i in range(2):
        rvs2 += dist2(m) # We should be able to call multiple times with no problem

    print(rvs)
    print(rvs2)
    assert np.array_equal(rvs, rvs2[0]), 'Separate dists should match'
    assert not np.array_equal(rvs2[0], rvs2[1]), 'Multiple calls to the same dist should not match'

    return dist, dist2


def test_reset(m=m):
    """ Check that reset works as expected """
    sc.heading('Testing reset')

    # Create and draw two sets of random numbers
    dist = ss.random(seed=533, strict=False).init()
    r1 = dist.rvs(m)
    r2 = dist.rvs(m)
    assert all(r1 != r2)

    # Reset to the most recent state
    dist.reset(-1)
    r3 = dist.rvs(m)
    assert all(r3 == r2)

    # Reset to the initial state
    dist.reset(0)
    r4 = dist.rvs(m)
    assert all(r4 == r1)

    for r in [r1, r2, r3, r4]:
        print(r)

    return dist


def make_fake_sim(n=10):
    """ Define a fake sim object for testing slots """
    sim = sc.prettyobj()
    sim.n = n
    sim.people = sc.prettyobj()
    sim.people.uid = np.arange(sim.n)
    sim.people.slot = np.arange(sim.n)
    sim.people.age = np.random.uniform(0, 100, size=sim.n)
    return sim


def test_callable(n=n):
    """ Test callable parameters """
    sc.heading('Testing a uniform distribution with callable parameters')

    sim = make_fake_sim()

    # Define a parameter as a function
    def custom_loc(module, sim, uids):
        out = sim.people.age[uids]
        return out

    scale = 1
    d1 = ss.normal(name='callable', loc=custom_loc, scale=scale).init(sim=sim)
    d2 = ss.lognorm_ex(name='callable', mean=custom_loc, std=scale).init(sim=sim)

    uids = np.array([1, 3, 7, 9])
    draws1 = d1.rvs(uids)
    draws2 = d2.rvs(uids)
    print(f'Input ages were: {sim.people.age[uids]}')
    print(f'Output samples were: {draws1}, {draws2}')

    for draws in [draws1, draws2]:
        meandiff = np.abs(sim.people.age[uids] - draws).mean()
        assert meandiff < scale*3, 'Outputs should match ages'

    return d1


def test_array(n=n):
    """ Test array parameters """
    sc.heading('Testing uniform with a array parameters')

    uids = np.array([1, 3])
    low  = np.array([1, 100]) # Low
    high = np.array([3, 125]) # High

    d = ss.uniform(low=low, high=high, strict=False).init(slots=np.arange(uids.max()+1))
    draws = d.rvs(uids)
    print(f'Uniform sample for uids {uids} returned {draws}')

    assert len(draws) == len(uids)
    for i in range(len(uids)):
        assert low[i] < draws[i] < low[i] + high[i], 'Invalid value'
    return draws


def test_repeat_slot():
    """ Test behavior of repeated slots """
    sc.heading('Test behavior of repeated slots')

    # Initialize parameters
    slots = np.array([4,2,3,2,2,3])
    n = len(slots)
    uids = np.arange(n)
    low = np.arange(n)
    high = low + 1

    # Draw values
    d = ss.uniform(low=low, high=high, strict=False).init(slots=slots)
    draws = d.rvs(uids)

    # Print and test
    print(f'Uniform sample for slots {slots} returned {draws}')
    assert len(draws) == len(slots)

    unique_slots = np.unique(slots)
    for s in unique_slots:
        inds = np.where(slots==s)[0]
        frac, integ = np.modf(draws[inds])
        assert np.allclose(integ, low[inds]), 'Integral part should match the low parameter'
        assert np.allclose(frac, frac[0]), 'Same random numbers, so should be same fractional part'
    return draws


def test_timepar_dists():
    """ Test interaction of distributions and timepars """
    sc.heading('Test interaction of distributions and timepars')

    # Set parameters
    n = int(10e3)
    u1 = 'day'
    u2 = 'year'
    dt_dur = 0.1
    dt_rate = 20.0
    ratio_dur  = ss.time_ratio(u1, 1.0, u2, dt_dur)
    ratio_rate = ss.time_ratio(u2, 1.0, u1, dt_rate)

    # Create time parameters
    v = sc.objdict()
    v.base = 30.0
    v.dur = ss.dur(v.base, unit=u1, parent_unit=u2, parent_dt=dt_dur).init()
    v.rate = ss.rate(v.base, unit=u2, parent_unit=u1, parent_dt=dt_rate).init() # Swap units

    # Check distributions that scale linearly with time, with the parameter we'll set
    print('Testing linear distributions ...')
    linear_dists = dict(
        constant   = 'v',
        uniform    = 'high',
        normal     = 'loc',
        lognorm_ex = 'mean',
        expon      = 'scale',
        weibull    = 'scale',
        gamma      = 'scale',
    )

    for name,par in linear_dists.items():
        dist_class = getattr(ss, name)

        # Create the dists, the first parameter of which should have time units
        dists = sc.objdict()
        for key,val in v.items():
            pardict = {par:val} # Convert to a tiny dictionary to insert the correct name
            dists[key] = dist_class(**pardict, name=key, strict=False).init()

        # Create th erandom variates
        rvs = sc.objdict()
        for k,dist in dists.items():
            rvs[k] = dist.rvs(n)

        # Check that the distributions match
        rtol = 0.1 # Be somewhat generous with the uncertainty
        expected = rvs.base.mean()
        expected_dur = expected*ratio_dur
        expected_rate = expected/ratio_rate
        actual_dur = rvs.dur.mean()
        actual_rate = rvs.rate.mean()
        assert np.isclose(expected_dur, actual_dur, rtol=rtol), f'Duration not close for {name}: {expected_dur:n} ≠ {actual_dur:n}'
        assert np.isclose(expected_rate, actual_rate, rtol=rtol), f'Rate not close for {name}: {expected_rate:n} ≠ {actual_rate:n}'
        sc.printgreen(f'✓ {name} passed: {expected_dur:n} ≈ {actual_dur:n}')

    # Check that unitless distributions fail
    print('Testing unitless distributions ...')
    par = ss.dur(10)
    unitless_dists = ['lognorm_im', 'randint', 'choice']
    for name in unitless_dists:
        dist_class = getattr(ss, name)
        with pytest.raises(NotImplementedError):
            dist = dist_class(par, name='notimplemented', strict=False).init()
            dist.rvs(n)
        sc.printgreen(f'✓ {name} passed: raised appropriate error')

    # Check special distributions
    print('Testing Poisson distribution ...')
    lam1 = ss.rate(dpy, 'year', parent_unit='year').init()
    lam2 = ss.rate(1,   'day',  parent_unit='year').init()
    poi1 = ss.poisson(lam=lam1, strict=False).init()
    poi2 = ss.poisson(lam=lam2, strict=False).init()
    mean1 = poi1.rvs(n).mean()
    mean2 = poi2.rvs(n).mean()
    assert np.isclose(mean1, mean2, rtol=rtol), f'Poisson values do not match for {lam1} and {lam2}: {mean1:n} ≠ {mean2:n}'
    sc.printgreen(f'✓ poisson passed: {mean1:n} ≈ {mean2:n}')

    print('Testing Bernoulli distribution ...')
    p1 = 0.1
    p2 = ss.time_prob(0.01, parent_dt=10).init()
    ber1 = ss.bernoulli(p=p1, strict=False).init()
    ber2 = ss.bernoulli(p=p2, strict=False).init()
    mean1 = ber1.rvs(n).mean()
    mean2 = ber2.rvs(n).mean()
    assert np.isclose(mean1, mean2, rtol=rtol), f'Bernoulli values do not match for {lam1} and {lam2}: {mean1:n} ≠ {mean2:n}'
    sc.printgreen(f'✓ bernoulli passed: {mean1:n} ≈ {mean2:n}')

    return ber2


def test_timepar_callable():
    sc.heading('Test that timepars work with (some) callable functions')

    print('Testing callable parameters with regular-scaling distributions')
    sim = make_fake_sim(n=10)
    uids = np.array([1, 3, 7, 9])

    def age_year(module, sim, uids):
        """ Extract age as a year """
        out = sim.people.age[uids].copy()
        return out

    def age_day(module, sim, uids):
        """ Extract age as a day """
        out = sim.people.age[uids].copy()
        out *= dpy # Convert to days manually
        return out

    scale = 1e-3 # Set to a very small but nonzero scale
    loc  = ss.dur(age_year, unit='year', parent_unit='day').init(update_values=False)
    mean = ss.dur(age_day,  unit='day',  parent_unit='day').init(update_values=False)
    d3 = ss.normal(name='callable', loc=loc, scale=scale).init(sim=sim)
    d4 = ss.lognorm_ex(name='callable', mean=mean, std=scale).init(sim=sim)
    draws3 = d3.rvs(uids)
    draws4 = d4.rvs(uids)
    age_in_days = sim.people.age[uids]*dpy
    draw_diff = np.abs(draws3 - draws4).mean()
    age_diff = np.abs(age_in_days - draws3).mean()
    print(f'Input ages were:\n{sim.people.age[uids]}')
    print(f'Output samples were:\n{sc.sigfiground(draws3)}\n{sc.sigfiground(draws4)}')
    assert draw_diff < 1, 'Day and year outputs should match to the nearest day'
    assert age_diff < 1, 'Distribution outputs should match ages to the nearest day'

    print('Testing callable parameters with Bernoulli distributions')
    n = 100_000
    sim = make_fake_sim(n=n)
    uids = np.random.choice(n, size=n//2, replace=False)
    age = sim.people.age[uids]
    mean = age.mean()
    young = sc.findinds(age<=mean)
    old = sc.findinds(age>mean)
    p_young = 0.001
    p_old = 0.010

    def age_prob(module, sim, uids):
        out = np.zeros_like(age)
        out[young] = p_young
        out[old]   = p_old
        return out

    parent_dt = 10
    p1 = age_prob
    p2 = ss.time_prob(age_prob, parent_dt=parent_dt).init(update_values=False)
    ber1 = ss.bernoulli(name='base', p=p1).init(sim=sim)
    ber2 = ss.bernoulli(name='time', p=p2).init(sim=sim)
    brvs1 = ber1.rvs(uids)
    brvs2 = ber2.rvs(uids)

    rtol = 0.5 # We're dealing with order-of-magnitude differences but small numbers, so be generous to avoid random failures
    sum1 = brvs1.sum()*parent_dt
    sum2 = brvs2.sum()
    assert np.isclose(sum1, sum2, rtol=rtol), f'Callable Bernoulli sums did not match: {sum1}  ≠  {sum2}'
    sc.printgreen(f'✓ Callable Bernoulli sums matched: {sum1:n} ≈ {sum2:n}')
    for key,expected,inds in zip(['young', 'old'], [p_young, p_old], [young, old]):
        m1 = brvs1[inds].mean()
        m2 = brvs2[inds].mean()/parent_dt
        assert np.allclose([expected, m1], [expected, m2], rtol=rtol), f'Callable Bernoulli proportions did not match: {expected:n}  ≠  {m1:n}  ≠  {m2:n}'
        sc.printgreen(f'✓ Callable Bernoulli proportions matched: {expected:n}  ≈  {m1:n}  ≈  {m2:n}')

    return


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    T = sc.timer()

    o1 = test_dist()
    o2 = test_custom_dists(do_plot=do_plot)
    o3 = test_dists(do_plot=do_plot)
    o4 = test_scipy()
    o5 = test_exceptions()
    o6 = test_reset()
    o7 = test_callable()
    o8 = test_array()
    o9 = test_repeat_slot()
    o10 = test_timepar_dists()
    o10 = test_timepar_callable()

    T.toc()
