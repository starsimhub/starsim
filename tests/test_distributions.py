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
@sc.timer()
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


@sc.timer()
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


@sc.timer()
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


@sc.timer()
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


@sc.timer()
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


@sc.timer()
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


@sc.timer()
def test_callable(n=n):
    """ Test callable parameters """
    sc.heading('Testing a uniform distribution with callable parameters')

    sim = ss.mock_sim(n_agents=10)

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


@sc.timer()
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


@sc.timer()
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


def test_poisson_ppf():
    """ The fast Poisson inverse CDF must match SciPy exactly (guards the searchsorted optimization) """
    sc.heading('Test fast Poisson inverse CDF')
    import scipy.stats as sps
    u = np.linspace(1e-4, 1-1e-4, 5000) # Deterministic grid of uniforms spanning (0, 1)
    for lam in [0.5, 3.7, 10, 50, 200]:
        d = ss.poisson(lam=lam, strict=False).init(slots=np.arange(10))
        d.process_pars() # Populate _pars so ppf() can read lam
        fast = d.ppf(u)
        ref = sps.poisson.ppf(u, mu=lam)
        assert np.array_equal(fast, ref), f'Fast Poisson ppf does not match SciPy for lam={lam}'
        sc.printgreen(f'✓ Poisson ppf matches SciPy for lam={lam}')
    return fast


def test_choice_slots():
    """ ss.choice over UIDs must stay correct and CRN-safe regardless of slot magnitude (no slots.max()+1 blowup) """
    sc.heading('Test ss.choice slot handling')
    a = np.array([10, 20, 30])
    p = np.array([0.2, 0.3, 0.5])

    # Repeated + very large slots (as for agents born mid-sim with slot_scale=100):
    # equal slots must yield equal values (CRN), and large slots must not allocate slots.max()+1.
    slots = np.array([1_000_000, 5, 1_000_000, 5, 999_999])
    uids = ss.uids(np.arange(len(slots)))
    d = ss.choice(a=a, p=p, strict=False).init(slots=slots)
    vals = d.rvs(uids)
    assert set(np.unique(vals)).issubset(set(a)), 'choice returned values outside a'
    assert vals[0] == vals[2] and vals[1] == vals[3], 'Equal slots must give equal values (CRN)'

    # Frequencies match p, with slots spread across a large range
    n = 20000
    big_slots = np.arange(n) + 1_000_000
    d2 = ss.choice(a=a, p=p, strict=False).init(slots=big_slots)
    freq = d2.rvs(ss.uids(np.arange(n)))
    for val, pexp in zip(a, p):
        assert np.isclose((freq == val).mean(), pexp, atol=0.02), f'choice frequency off for {val}'
    sc.printgreen('✓ ss.choice correct and CRN-safe with large slots')
    return vals


def make_mock_modules():
    """ Create mock modules for the tests to use """
    mod = sc.objdict()
    mod.year  = ss.mock_module(dt=ss.year)
    mod.month = ss.mock_module(dt=ss.month)
    mod.week  = ss.mock_module(dt=ss.week)
    return mod


@sc.timer()
def test_timepar_dists():
    """ Test interaction of distributions and timepars """
    sc.heading('Test interaction of distributions and timepars')

    # Set parameters
    n = int(10e4)

    # Mock modules
    mock_mods = make_mock_modules()

    # Create time parameters
    v = sc.objdict()
    v.base = 30.0
    v.dur = ss.years(30)
    v.freq = ss.freqperyear(30)

    rtol = 0.1  # Be somewhat generous with the uncertainty
    check_which = ['linear', 'unitless', 'special']

    if 'linear' in check_which:
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

            for module in mock_mods.values():

                # Create the dists, the first parameter of which should have time units
                dists = sc.objdict()
                for key,val in v.items():
                    pardict = {par:val} # Convert to a tiny dictionary to insert the correct name
                    dists[key] = dist_class(**pardict, name=key, module=module, strict=False).init()

                # Create the random variates
                rvs = sc.objdict()
                for k,dist in dists.items():
                    rvs[k] = dist.rvs(n)

                # Check that the distributions match
                ratio = ss.years(1)/module.t.dt
                expected = rvs.base.mean()
                expected_dur = expected*ratio
                expected_rate = expected/ratio
                actual_dur = rvs.dur.mean()
                actual_rate = rvs.freq.mean()
                assert np.isclose(expected_dur, actual_dur, rtol=rtol), f'Duration not close for {name}: {expected_dur:n} ≠ {actual_dur:n}'
                assert np.isclose(expected_rate, actual_rate, rtol=rtol), f'Rate not close for {name}: {expected_rate:n} ≠ {actual_rate:n}'
                sc.printgreen(f'✓ {name} passed: {expected_dur:n} ≈ {actual_dur:n} with dt={module.t.dt}')

    if 'unitless' in check_which:
        # Check that unitless distributions fail
        print('Testing unitless distributions ...')
        par = ss.dur(10, 'years')
        unitless_dists = ['lognorm_im', 'randint']
        for name in unitless_dists:
            dist_class = getattr(ss, name)
            with pytest.raises(NotImplementedError):
                dist = dist_class(par, name='notimplemented', module=mock_mods.year, strict=False).init()
                dist.rvs(n)
            sc.printgreen(f'✓ {name} passed: raised appropriate error')

    if 'special' in check_which:
        # Check special distributions
        print('Testing choice distribution ...')
        a = ss.days(np.array([10, 20, 30]))
        p = np.array([0.2, 0.3, 0.5])
        for module in mock_mods.values():
            choice = ss.choice(a=a, p=p, module=module, strict=False).init()
            mean = choice.rvs(n).mean()
            expected = (a*p/module.t.dt).sum()
            assert np.isclose(mean, expected, rtol=rtol), f'Choice values do not match: {mean:n} ≠ {expected:n}'
            sc.printgreen(f'✓ Choice passed: {expected:n} ≈ {mean:n} with dt={module.t.dt}')

        print('Testing Poisson distribution ...')
        lam1 = ss.freqperyear(1)
        lam2 = ss.perday(1)
        for module in mock_mods.values():
            poi1 = ss.poisson(lam=lam1, module=module, strict=False).init()
            poi2 = ss.poisson(lam=lam2, module=module, strict=False).init()
            mean1 = poi1.rvs(n).mean()
            mean2 = poi2.rvs(n).mean()
            expected1 = lam1*module.t.dt
            expected2 = lam2*module.t.dt
            assert np.isclose(mean1, expected1, rtol=rtol), f'Poisson values do not match for {lam1}: {mean1:n} ≠ {expected1:n}'
            assert np.isclose(mean2, expected2, rtol=rtol), f'Poisson values do not match for {lam2}: {mean2:n} ≠ {expected2:n}'
            sc.printgreen(f'✓ Poisson passed: {lam1} {expected1:n} ≈ {mean1:n} with dt={module.t.dt}')
            sc.printgreen(f'✓ Poisson passed: {lam2} {expected2:n} ≈ {mean2:n} with dt={module.t.dt}')

        print('Testing Bernoulli distribution ...')
        p1 = ss.probperday(0.1/365)
        p2 = ss.probperyear(0.1)
        ber1 = ss.bernoulli(p=p1, module=mock_mods.year, strict=False).init()
        ber2 = ss.bernoulli(p=p2, module=mock_mods.year, strict=False).init()
        mean1 = ber1.rvs(n).mean()
        mean2 = ber2.rvs(n).mean()
        assert np.isclose(mean1, mean2, rtol=rtol), f'Bernoulli values do not match for {ber1} and {ber2}: {mean1:n} ≠ {mean2:n}'
        sc.printgreen(f'✓ bernoulli passed: {mean1:n} ≈ {mean2:n}')

    print('Testing different syntaxes for adding timepars')
    kw = dict(dt=ss.days(1))
    mean = 10
    std = 0.1

    d = sc.objdict()
    d.a = ss.normal(ss.years(mean), ss.years(std)).mock(**kw)
    d.b = ss.normal(mean, std, unit=ss.years).mock(**kw)
    d.c = ss.years(ss.normal(mean, std)).mock(**kw)

    n = int(1e6)
    rvs = sc.objdict()
    for key,dist in d.items():
        rvs.key = dist.rvs(n)

    factor = ss.time.factors.years.days
    expected = factor*mean

    means = [rvs.mean() for rvs in rvs.values()]
    assert all([np.isclose(expected, m) for m in means]), f'Expecting timepar scaling to match {expected = } and {means = }'

    return ber2


@sc.timer()
def test_timepar_callable():
    sc.heading('Test that timepars work with (some) callable functions')

    # In this case, the callable returns a time par, which then gets further converted
    print('Testing callable parameters with regular-scaling distributions')

    n = 10_000
    rtol = 0.05

    mock_mods = make_mock_modules()

    def call_scalar(module, sim, uids):
        return ss.years(5.0)

    for module in mock_mods.values():
        d = ss.normal(call_scalar, ss.days(1), module=module, strict=False)
        d.init()
        expected = ss.years(5.0)/module.dt
        actual = d.rvs(n).mean()
        assert np.isclose(actual, expected, rtol=rtol)

    print('Testing callable parameters with Bernoulli distributions')
    n = 1000
    sim = ss.mock_sim(n_agents=n)
    uids = np.random.choice(n, size=n//2, replace=False)
    age = sim.people.age[uids]
    mean = age.mean()
    young = sc.findinds(age<=mean)
    old = sc.findinds(age>mean)
    p_young = 0.1
    p_old = 0.2

    def age_prob(module, sim, uids):
        dt = module.t.dt
        out = np.zeros_like(uids)
        out[young] = ss.peryear(p_young)*dt
        out[old]   = ss.peryear(p_old)*dt
        return out

    sim = ss.mock_sim(n_agents=100_000)
    uids = np.random.choice(n, size=n//2, replace=False)
    age = sim.people.age[uids]
    mean = age.mean()
    young = sc.findinds(age<=mean)
    old = sc.findinds(age>mean)

    ber2 = ss.bernoulli(age_prob, module=mock_mods.year, strict=False).init(sim=sim)
    a = ber2.rvs(uids).mean()

    ber3 = ss.bernoulli(age_prob, module=mock_mods.month, strict=False).init(sim=sim)
    b = ber3.rvs(uids).mean()

    rtol = 0.2  # We're dealing with order-of-magnitude differences but small numbers, so be generous to avoid random failures
    assert np.isclose(a, b*12, rtol=rtol), f'Callable Bernoulli sums did not match: {a} ≠ {b*12}'

    return


@sc.timer()
def test_hist_plotting():
    """ Test that histogram plotting works """
    sc.heading('Test plotting of distribution histograms')

    # Based on the advanced distributions user guide
    sir = ss.SIR(
        init_prev = ss.bernoulli(p=0.15),
        dur_inf = ss.weibull(c=2, loc=1, scale=2)
    )
    sim = ss.Sim(n_agents=100, diseases=sir, dur=15, networks=ss.RandomNet())
    sim.run()

    # Plot the distribution
    rvs = sim.diseases.sir.pars.dur_inf.plot_hist()
    return


@sc.timer()
def test_dtype_consistency():
    """
    The two draw paths must return the same dtype, and float dists must honor ss.dtypes.float.

    A distribution can be sampled two ways: by count (`dist.rvs(n)`, the native path) or by UID
    (`dist.rvs(uids)`, the common-random-number hash path). These must agree. Regression (3.5.1):
    the CRN hash engine (`hash_uniforms`) returned float64 regardless of the declared float dtype,
    so `ss.random`/`ss.uniform` returned float32 by count but float64 by UID. That mismatch also
    made `ss.multi_random.rvs()` return double-length arrays (the XOR-combine views the float bytes
    as uint32, and an 8-byte float viewed as a 4-byte uint doubles the length). See CHANGELOG 3.5.2.
    """
    sc.heading('Testing dtype consistency across draw paths')
    sim = make_sim()
    uids = ss.uids(np.arange(20))
    ft = ss.dtypes.float

    def path_dtypes(dist):
        dist.init(sim=sim)
        native = np.asarray(dist.rvs(20)).dtype    # by count -> native path
        crn    = np.asarray(dist.rvs(uids)).dtype   # by UID   -> CRN/hash path
        return native, crn

    # Float dists (pass-through / linear PPF) must honor ss.dtypes.float on BOTH paths
    for dist in [ss.random(name='r'), ss.uniform(low=0, high=5, name='u')]:
        native, crn = path_dtypes(dist)
        assert native == crn == ft, f'{dist.distname}: by-count={native}, by-UID={crn}, expected {ft}'

    # ss.poisson yields integer counts, so both paths must return ss.dtypes.int. Regression: the CRN
    # ppf cast counts to float64 (`searchsorted(...).astype(float)`) while the native path was int64.
    native, crn = path_dtypes(ss.poisson(lam=3, name='p'))
    assert native == crn == ss.dtypes.int, f'poisson: by-count={native}, by-UID={crn}, expected {ss.dtypes.int}'

    # Other dists aren't independently vulnerable to this issue, but the two paths must still agree
    for dist in [ss.normal(name='n'), ss.expon(name='e'), ss.lognorm_ex(name='l'),
                 ss.constant(v=1.5, name='c'), ss.bernoulli(p=0.3, name='b')]:
        native, crn = path_dtypes(dist)
        assert native == crn, f'{dist.distname}: by-count path {native} != by-UID path {crn}'
    return


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    # ss.options.warnings = 'error' # Turn warnings into exceptions for debugging

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
    o10 = test_poisson_ppf()
    o11 = test_choice_slots()
    o12 = test_timepar_dists()
    o13 = test_timepar_callable()
    o14 = test_hist_plotting()
    o15 = test_dtype_consistency()

    T.toc()
