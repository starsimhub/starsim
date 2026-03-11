"""
Test Starsim features not covered by other test files
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim_examples as sse
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

# %% Define the tests

@sc.timer()
def test_microsim(do_plot=False):
    sc.heading('Test small HIV simulation')

    # Make HIV module
    hiv = sse.HIV()
    # Set beta. The first entry represents transmission risk from infected p1 -> susceptible p2
    # Need to be careful to get the ordering right. The set-up here assumes that in the simple
    # sexual  network, p1 is male and p2 is female. In the maternal network, p1=mothers, p2=babies.
    hiv.pars['beta'] = {'mf': [0.15, 0.10], 'maternal': [0.2, 0]}

    sim = ss.Sim(
        people=ss.People(small),
        networks=[ss.MFNet(), ss.MaternalNet()],
        demographics=ss.Pregnancy(),
        diseases=hiv,
        copy_inputs = False, # So we can reuse hiv
    )
    sim.init()
    sim.run()

    if do_plot:
        plt.figure()
        plt.plot(hiv.timevec, hiv.results.n_infected)
        plt.title('HIV number of infections')

    return sim


@sc.timer()
def test_results():
    sc.heading('Testing results export and plotting')

    # Make a sim with 2 SIS models with varying units and dt
    d1 = ss.SIS(dt=ss.years(1/12), name='sis1')
    d2 = ss.SIS(dt=ss.years(0.5), name='sis2')
    sim = ss.Sim(n_agents=medium, diseases=[d1, d2], networks='random')

    # Run sim and pull out disease results
    sim.run()
    rs1 = sim.results.sis1
    rs2 = sim.results.sis2

    # Export a single result to a series or dataframe
    res = rs1.new_infections
    res_df = res.to_df()
    res_series = res.to_series()
    assert res[-1] == res_df.iloc[-1].value == res_series.iloc[-1]

    # Test resampling (always returns a Result)
    resampled = rs1.new_infections.resample(new_unit='year')
    assert isinstance(resampled, ss.Result)
    assert len(resampled.values) < len(rs1.new_infections.values)

    # Test that resample and annualize produce the same values
    annualized = rs1.new_infections.annualize()
    assert np.allclose(resampled.values, annualized.values, rtol=1e-10)

    # Test Results.annualize() (annualizes all results in the group)
    rs1_annual = rs1.annualize()
    assert isinstance(rs1_annual, ss.Results)
    assert len(rs1_annual.new_infections.values) == len(annualized.values)
    assert np.allclose(rs1_annual.new_infections.values, annualized.values, rtol=1e-10)

    # Export results of a whole module to a dataframe
    dfs = rs1.to_df()
    assert res_df.value.sum() == dfs.cum_infections.values[-1] == sim.summary.sis1_cum_infections

    # Export resampled summary of results to dataframe
    dfy1 = rs1.to_df(resample='year')
    dfy2 = rs2.to_df(resample='5YE')
    assert dfs.new_infections.iloc[:12].sum() == dfy1.new_infections.iloc[0]
    assert rs2.n_susceptible[:2].mean() == dfy2.n_susceptible.iloc[0]  # Entries 0 and 1 represent 2000
    assert rs2.n_susceptible[2:12].mean() == dfy2.n_susceptible.iloc[1]  # Entries 2-12 correspond to 2001-2005

    # Export whole sim to unified annualized dataframe
    sim_df = sim.to_df(resample='year', use_years=True)
    assert sim_df.sis1_n_infected.values[0] == rs1.n_infected[:12].mean()
    assert sim_df.sis2_n_infected.values[0] == rs2.n_infected[:2].mean()

    # Plot
    res.plot()
    sim.results.sis1.plot()
    sim.results.sis2.plot()

    return sim


@sc.timer()
def test_arrs():
    sc.heading('Testing Arr objects')
    o = sc.objdict()

    # Create a sim with only births
    pars = dict(n_agents=medium, diseases='sis', networks='random')
    p1 = sc.mergedicts(pars, birth_rate=ss.freqperyear(10))
    p2 = sc.mergedicts(pars, death_rate=ss.freqperyear(10))
    s1 = ss.Sim(pars=p1).run()
    s2 = ss.Sim(pars=p2).run()

    # Tests
    assert len(s1.people.auids) > len(s2.people.auids), 'Should have more people with births'
    assert np.array_equal(s1.people.age, s1.people.age.raw[s1.people.auids]), 'Different ways of indexing age should match'
    assert np.array_equal(s1.people.alive.uids, s1.people.auids), 'Active/alive agents should match'
    assert np.all(s2.people.alive), 'All agents should be alive when indexed like this'
    assert not np.all(s2.people.alive.raw), 'Some agents should not be alive when indexed like this'
    assert np.array_equal(~s1.people.female, s1.people.male), 'Definition of men does not match'
    assert isinstance(s1.people.age < 5, ss.BoolArr), 'Performing logical operations should return a BoolArr'
    assert np.array_equal(s1.people.age[s1.people.age < 5],s1.people.age[s1.people.age.values < 5])

    # Test BoolArr & uids operations
    test_bool = s1.people.age > 20
    test_uids = test_bool.uids

    # Standard operations
    assert np.all(test_bool == test_uids)
    assert np.all(~(test_bool != test_uids))
    assert np.array_equal(s1.people.female & test_bool, s1.people.female & test_uids)
    assert np.array_equal(s1.people.female | test_bool, s1.people.female | test_uids)
    assert np.array_equal(s1.people.female ^ test_bool, s1.people.female ^ test_uids)
    assert np.array_equal(test_uids & s1.people.female,  test_uids & s1.people.female.uids)
    assert np.array_equal(test_uids | s1.people.female,  test_uids | s1.people.female.uids)
    assert np.array_equal(test_uids ^ s1.people.female,  test_uids ^ s1.people.female.uids)

    # Inplace BoolArr operations
    a = s1.people.female & test_bool
    b = sc.dcp(s1.people.female)
    original_id = id(b)
    original_id_raw = id(b.raw)
    b &= test_bool
    assert np.array_equal(a, b)
    assert id(b) == original_id
    assert id(b.raw) == original_id_raw

    a = s1.people.female & test_uids
    b = sc.dcp(s1.people.female)
    original_id = id(b)
    original_id_raw = id(b.raw)
    b &= test_uids
    assert np.array_equal(a, b)
    assert id(b) == original_id
    assert id(b.raw) == original_id_raw

    a = s1.people.female | test_bool
    b = sc.dcp(s1.people.female)
    original_id = id(b)
    original_id_raw = id(b.raw)
    b |= test_bool
    assert np.array_equal(a, b)
    assert id(b) == original_id
    assert id(b.raw) == original_id_raw

    a = s1.people.female ^ test_uids
    b = sc.dcp(s1.people.female)
    original_id = id(b)
    original_id_raw = id(b.raw)
    b ^= test_uids
    assert np.array_equal(a, b)
    assert id(b) == original_id
    assert id(b.raw) == original_id_raw

    o.s1 = s1
    o.s2 = s2

    return o


@sc.timer()
def test_arr_inplace():
    # Note that the test function above focusses on BoolArr specifially which has its own version of some of the operators
    sc.heading('Testing Arr in-place vs non-in-place arithmetic operators')

    # Create two synthetic FloatArrs without needing a full sim
    a = ss.FloatArr('a', default=3.0, mock=10)
    b = ss.FloatArr('b', default=2.0, mock=10)

    # --- In-place: += must update values without replacing the Arr or .raw array ---
    original_id     = id(a)
    original_id_raw = id(a.raw)
    expected        = a.raw.copy() + b.raw.copy()
    a += b
    assert np.allclose(a.raw, expected),       'In-place += must produce correct values'
    assert id(a)     == original_id,           'In-place += must not replace the Arr object'
    assert id(a.raw) == original_id_raw,       'In-place += must not replace the .raw array'

    # --- Non-in-place: + must return an independent copy without modifying the original ---
    original_raw = a.raw.copy()
    c = a + b
    assert id(c)     != id(a),                 'Non-in-place + must return a new Arr object'
    assert id(c.raw) != id(a.raw),             'Non-in-place + must use a new .raw array'
    assert np.allclose(a.raw, original_raw),   'Non-in-place + must not modify the original Arr'
    assert np.allclose(c.raw, original_raw + b.raw), 'Non-in-place + must produce correct values'

    masked_auids = ss.uids(np.array([0, 1, 2, 4, 5, 6, 7, 8, 9], dtype=int))
    inactive_uid = 3

    # --- In-place masked Arr += Arr ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    b = ss.FloatArr('b', raw=np.arange(10, dtype=float), mock=10)
    a.people.auids = masked_auids
    b.people.auids = masked_auids

    original_id = id(a)
    original_id_raw = id(a.raw)
    expected = a.raw.copy()
    expected[masked_auids] += b.raw[masked_auids]

    a += b
    assert np.allclose(a.raw, expected), 'In-place += must update raw values for active UIDs only'
    assert a.raw[inactive_uid] == 3.0,   'In-place += must not modify inactive UIDs'
    assert id(a) == original_id,         'In-place += must not replace the Arr object'
    assert id(a.raw) == original_id_raw, 'In-place += must not replace the .raw array'

    # --- In-place masked Arr -= Arr ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    b = ss.FloatArr('b', raw=np.arange(10, dtype=float), mock=10)
    a.people.auids = masked_auids
    b.people.auids = masked_auids
    expected = a.raw.copy()
    expected[masked_auids] -= b.raw[masked_auids]
    a -= b
    assert np.allclose(a.raw, expected), 'In-place -= must update raw values for active UIDs only'
    assert a.raw[inactive_uid] == 3.0,   'In-place -= must not modify inactive UIDs'

    # --- In-place masked Arr +/- scalar ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    a.people.auids = masked_auids
    expected = a.raw.copy()
    expected[masked_auids] += 2
    a += 2
    assert np.allclose(a.raw, expected), 'In-place += scalar must update active UIDs only'
    expected[masked_auids] -= 1
    a -= 1
    assert np.allclose(a.raw, expected), 'In-place -= scalar must update active UIDs only'
    assert a.raw[inactive_uid] == 3.0,   'Scalar in-place arithmetic must not modify inactive UIDs'

    # --- In-place masked Arr += ndarray aligned to active UIDs ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    a.people.auids = masked_auids
    rhs = np.arange(len(masked_auids), dtype=float)
    expected = a.raw.copy()
    expected[masked_auids] += rhs
    a += rhs
    assert np.allclose(a.raw, expected), 'In-place += ndarray must treat the ndarray as aligned to active UIDs'

    # --- In-place masked Arr += ndarray with mismatched length ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    a.people.auids = masked_auids
    try:
        a += np.arange(len(masked_auids) + 1, dtype=float)
        raise AssertionError('In-place += with a mismatched ndarray length must raise an error')
    except ValueError:
        pass

    # --- In-place masked Arr += Arr with different active masks ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    b = ss.FloatArr('b', raw=np.arange(10, dtype=float), mock=10)
    other_masked_auids = ss.uids(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int))
    a.people.auids = masked_auids
    b.people.auids = other_masked_auids
    expected = a.raw.copy()
    expected[masked_auids] += b.raw[masked_auids]
    a += b
    assert np.allclose(a.raw, expected), 'Arr-to-Arr arithmetic must read RHS values from other.raw[self.auids]'
    assert a.raw[inactive_uid] == 3.0,   'Mismatched RHS masks must not modify inactive UIDs'

    # --- Non-in-place masked Arr + Arr ---
    a = ss.FloatArr('a', default=3.0, mock=10)
    b = ss.FloatArr('b', raw=np.arange(10, dtype=float), mock=10)
    a.people.auids = masked_auids
    b.people.auids = masked_auids
    original_raw = a.raw.copy()
    c = a + b
    expected = original_raw.copy()
    expected[masked_auids] += b.raw[masked_auids]
    assert id(c) != id(a),               'Masked non-in-place + must return a new Arr object'
    assert id(c.raw) != id(a.raw),       'Masked non-in-place + must use a new .raw array'
    assert np.allclose(a.raw, original_raw), 'Masked non-in-place + must not modify the original Arr'
    assert np.allclose(c.raw, expected), 'Masked non-in-place + must preserve inactive raw values and update active UIDs'
    assert c.raw[inactive_uid] == original_raw[inactive_uid], 'Masked non-in-place + must preserve inactive raw values'

    return a, b


@sc.timer()
def test_arr_type_conversion():
    sc.heading('Testing Arr type conversion and mixed bool/float arithmetic')

    a = np.array([True, False], dtype=bool)
    b = np.array([3, 4], dtype=float)
    c = ss.BoolArr('c', raw=np.array([True, False], dtype=bool), mock=2)
    d = ss.FloatArr('d', raw=np.array([3, 4], dtype=float), mock=2)

    ab = a * b
    ba = b * a
    assert ab.dtype == float
    assert ba.dtype == float
    assert np.array_equal(ab, np.array([3.0, 0.0]))
    assert np.array_equal(ba, np.array([3.0, 0.0]))

    with pytest.raises(TypeError):
        a_copy = a.copy()
        a_copy *= b

    b_copy = b.copy()
    b_copy *= a
    assert b_copy.dtype == float
    assert np.array_equal(b_copy, np.array([3.0, 0.0]))

    cd = c * d
    dc = d * c
    assert isinstance(cd, ss.FloatArr)
    assert isinstance(dc, ss.FloatArr)
    assert cd.raw.dtype == float
    assert dc.raw.dtype == float
    assert np.array_equal(cd.raw, np.array([3.0, 0.0]))
    assert np.array_equal(dc.raw, np.array([3.0, 0.0]))

    with pytest.raises(TypeError):
        c_copy = ss.BoolArr('c_copy', raw=np.array([True, False], dtype=bool), mock=2)
        c_copy *= d

    d_copy = ss.FloatArr('d_copy', raw=np.array([3, 4], dtype=float), mock=2)
    d_copy *= c
    assert d_copy.raw.dtype == float
    assert np.array_equal(d_copy.raw, np.array([3.0, 0.0]))

    return dict(ab=ab, ba=ba, cd=cd, dc=dc)


@sc.timer()
def test_deepcopy():
    sc.heading('Testing deepcopy')
    s1 = ss.Sim(pars=dict(diseases='sir', networks=sse.EmbeddingNet()), n_agents=small)
    s1.init()

    s2 = sc.dcp(s1)

    s1.run()
    s2.run()
    s1.plot()
    s2.plot()

    ss.diff_sims(s1, s2, full=True)
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


@sc.timer()
def test_deepcopy_until():
    sc.heading('Testing deepcopy with until')
    s1 = ss.Sim(diseases=ss.SIR(init_prev=0.1), networks='random', n_agents=small)
    s1.init()

    s1.run(until=ss.date(2005))

    s2 = sc.dcp(s1)

    s1.run()
    s2.run()
    s1.plot()
    s2.plot()

    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


@sc.timer()
def test_custom_imports():
    sc.heading('Testing custom imports')
    o = sc.objdict()

    class MyDisease(ss.SIS):
        pass

    class MyNetwork(ss.RandomNet):
        pass

    my_modules = [MyDisease, MyNetwork]

    # Make both starsim_examples and custom modules searchable
    ss.register_modules(sse, my_modules)
    sim = ss.Sim(n_agents=1000, diseases=['hiv', 'mydisease'], networks='mynetwork')
    sim.run()

    return o


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1  = test_microsim(do_plot)
    sim2  = test_results()
    sims  = test_arrs()
    arrs  = test_arr_inplace()
    vals  = test_arr_type_conversion()
    sims2 = test_deepcopy()
    sims3 = test_deepcopy_until()
    mods  = test_custom_imports()

    sc.toc(T)
    plt.show()
    print('Done.')
