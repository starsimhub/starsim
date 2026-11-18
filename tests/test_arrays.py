"""
Test Starsim arrays (ss.Arr, ss.FloatArr, etc) and UIDs
"""
import numpy as np
import sciris as sc
import starsim as ss
import pytest

sc.options.interactive = False # Assume not running interactively


def test_arrs():
    """ Indexing by active agents, with births and deaths; BoolArr and UID operations, in place and not """
    pars = dict(n_agents=1000, diseases=ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)), networks='random', verbose=0) # In years, for births and deaths over 50 years
    s1 = ss.Sim(pars, demographics=ss.birth(rate=ss.peryear(0.01))).run()
    s2 = ss.Sim(pars, demographics=ss.death(rate=ss.peryear(0.01))).run()
    ppl = s1.people
    assert len(ppl.auids) > len(s2.people.auids), 'Should have more people with births'
    assert np.array_equal(ppl.age, ppl.age.raw[ppl.auids]) and np.array_equal(ppl.alive.uids, ppl.auids)
    assert np.all(s2.people.alive) and not np.all(s2.people.alive.raw), 'Dead agents are in the raw array only'
    assert np.array_equal(~ppl.female, ppl.male) and isinstance(ppl.age < 5, ss.BoolArr)
    assert np.array_equal(ppl.age[ppl.age < 5], ppl.age[ppl.age.values < 5])

    test_bool = ppl.age > 20
    test_uids = test_bool.uids
    assert np.all(test_bool == test_uids) and np.all(~(test_bool != test_uids))
    for op in ['__and__', '__or__', '__xor__']: # A BoolArr with a BoolArr or with UIDs, and UIDs with either
        assert np.array_equal(getattr(ppl.female, op)(test_bool), getattr(ppl.female, op)(test_uids))
        assert np.array_equal(getattr(test_uids, op)(ppl.female), getattr(test_uids, op)(ppl.female.uids))
    for op, iop in [('__and__', '__iand__'), ('__or__', '__ior__'), ('__xor__', '__ixor__')]: # In place: the same values, in the same object and array
        for other in [test_bool, test_uids]:
            b = sc.dcp(ppl.female)
            ids = id(b), id(b.raw)
            expected = getattr(ppl.female, op)(other)
            b = getattr(b, iop)(other)
            assert np.array_equal(expected, b) and (id(b), id(b.raw)) == ids
    return s1


def test_arr_arithmetic():
    """ Arithmetic in place and not, on active agents only, with Arrs, scalars, and arrays; bool times float """
    masked_auids = ss.uids([0, 1, 2, 4, 5, 6, 7, 8, 9])
    def masked(b_auids=masked_auids):
        """ An Arr of 3s and one of 0 to 9, with agent 3 inactive """
        a = ss.FloatArr('a', default=3.0, mock=10)
        b = ss.FloatArr('b', raw=np.arange(10, dtype=float), mock=10)
        a.people.auids = masked_auids
        b.people.auids = b_auids
        return a, b

    a, b = masked()
    ids = id(a), id(a.raw)
    a += b
    a -= 1
    a += np.arange(len(masked_auids), dtype=float) # An array is aligned to the active UIDs
    expected = 3.0 + np.arange(10.0) - 1
    expected[masked_auids] += np.arange(len(masked_auids))
    assert np.allclose(a.values, expected[masked_auids]) and (id(a), id(a.raw)) == ids, 'In place: the active values, in the same object and array'
    with pytest.raises(ValueError): # Neither one value per active agent nor one per raw place
        a += np.arange(11, dtype=float)
    a, b = masked(b_auids=ss.uids(np.arange(9))) # Another active mask: the right side is read at the same agents
    a += b
    assert np.allclose(a.values, 3 + masked_auids)
    a, b = masked()
    raw = a.raw.copy()
    c = a + b
    assert id(c) != id(a) and id(c.raw) != id(a.raw) and np.allclose(a.raw, raw) and np.allclose(c.values, 3 + masked_auids), 'Not in place: a new Arr, the original unchanged'

    c = ss.BoolArr('c', raw=np.array([True, False]), mock=2)
    d = ss.FloatArr('d', raw=np.array([3.0, 4.0]), mock=2)
    for prod in [c*d, d*c]:
        assert isinstance(prod, ss.FloatArr) and np.array_equal(prod.raw, [3.0, 0.0])
    with pytest.raises(TypeError): # As for NumPy arrays: a bool can't hold a float in place
        c *= d
    d *= c
    assert np.array_equal(d.raw, [3.0, 0.0])
    return a


def test_arr_2d():
    """ 2D arrays: by UIDs and columns, named columns, and in a sim with births and deaths """
    imm = ss.FloatArr('imm', default=0, columns=3, mock=5)
    two = ss.uids([0, 1])
    imm[two, 2] = 0.8
    assert imm[two].shape == (2, 3) and np.array_equal((imm.col(2) > 0.5).uids, two) and np.array_equal((imm > 0.5).count(axis=0), [0, 0, 2])
    with pytest.raises(ValueError):
        (imm > 0.5).uids # UIDs are not defined for 2D arrays
    named = ss.FloatArr('imm', default=0, columns=['wildtype', 'mutant'], mock=5)
    named[two, 'mutant'] = 0.8
    assert np.array_equal(named[:, 'mutant'], named[:, 1]) and np.array_equal(named.mutant, named.col('mutant')) and np.array_equal(named['mutant'], named.col('mutant'))
    with pytest.raises(ValueError):
        ss.FloatArr('imm', columns=['raw', 'mean'], mock=5) # Column names can't clash with attributes

    class VarSIS(ss.SIS):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.FloatArr('sus_imm', default=0.5, columns=3))
        def step(self):
            out = super().step()
            self.sus_imm[self.I.uids, 1] += 0.1
            return out
    sim = ss.Sim(n_agents=1000, diseases=VarSIS(), networks='random', demographics=True, dur=ss.years(10), verbose=0).run()
    imm = sim.diseases.varsis.sus_imm
    assert imm.raw.shape == (sim.people.uid.len_tot, 3) and imm.values.shape == (len(sim.people), 3) # Grows along the first axis; values are active agents only
    assert imm[:, 1].mean() > imm[:, 0].mean() == 0.5 and 'sus_imm' not in sim.results.varsis # No automatic results
    return sim


def test_uids():
    """ UIDs: + concatenates UIDs and adds numbers; set operations keep integers; reductions that make no sense raise; construction accepts only integers """
    a, b = ss.uids([1, 2, 3]), ss.uids([4, 5])
    c = a + b
    assert isinstance(c, ss.uids) and list(c) == [1, 2, 3, 4, 5] and list(a) == [1, 2, 3] and list(a + a) == [1, 2, 3, 1, 2, 3] # Duplicates kept, unlike |
    assert list(a + 10) == list(10 + a) == [11, 12, 13] and list(a + np.array([10, 20, 30])) == [11, 22, 33] and isinstance(10 + a, ss.uids)
    ref = a
    a += b # A new object, since the size changes
    assert list(ref) == [1, 2, 3] and list(a) == [1, 2, 3, 4, 5]
    ids = id(a)
    a += 10
    assert id(a) == ids # In place
    x = ss.uids([1, 2, 3])
    for result in [x | [], x & [], x - [], x ^ [], x | ss.uids([3, 4]), x.union([]), x.intersect([]), x.remove([]), x.xor([]), x | [4.0, 5.0]]:
        assert isinstance(result, ss.uids) and result.dtype == ss.dtypes.int
    for bad in [[1.5], [np.nan]]:
        with pytest.raises(TypeError):
            x | bad

    assert np.array_equal(ss.uids.concatenate([None, x, b]), [1, 2, 3, 4, 5]) and len(ss.uids.concatenate()) == 0

    assert type(x.max()) is int and x.max() + 1 == 4 # Plain ints, not uids (+ would concatenate)
    for op in ['sum', 'mean', 'std', 'cumsum']:
        with pytest.raises(TypeError):
            getattr(x, op)()
    with pytest.raises(TypeError): # Rejected, not silently truncated
        ss.uids([1.5])
    return x


def test_catarr():
    """ Categorical states, by label or integer category """
    loc = ss.CatArr('loc', categories=['urban', 'rural'], default='rural', mock=5)
    loc[ss.uids([0, 1])] = 'urban'
    loc[ss.uids([2])] = ['urban']
    assert list((loc == 'urban').uids) == [0, 1, 2] and (loc != 'urban').count() == 2 and loc.isin(['urban', 'rural']).count() == 5
    assert list(loc.labels()) == ['urban']*3 + ['rural']*2 and list(loc.index()) == [0, 0, 0, 1, 1]
    loc[ss.uids([3])] = loc[ss.uids([0])] # Codes, as read, are kept as they are
    assert loc.labels(ss.uids([3]))[0] == 'urban'
    with pytest.raises(ValueError):
        loc == 'suburban'
    quint = ss.CatArr('quintile', categories=[1, 2, 3, 4, 5], default=3, mock=4) # Integer categories are stored as they are
    quint[ss.uids([0])] = 5
    assert (quint == 5).count() == 1 and list(quint.labels()) == [5, 3, 3, 3] and list(quint.index()) == [4, 2, 2, 2]
    with pytest.raises(ValueError):
        ss.CatArr('mixed', categories=[1, 'two']) # An integer could then be a category or a code
    return loc


def test_boolstate_result():
    """ A BoolState is counted as a result of its module, and a BoolArr is not """
    class Flags(ss.Module):
        counted = ss.BoolState(default=True)
        hidden = ss.BoolArr(default=True)
        grid = ss.BoolArr(columns=2)
        def step(self): pass
    sim = ss.Sim(Flags(), n_agents=50, dur=2, verbose=0).run()
    assert 'counted' in sim.results.flags and 'hidden' not in sim.results.flags and 'grid' not in sim.results.flags
    assert sim.results.flags.counted[-1] == sim.results.alive[-1]
    with pytest.raises(ValueError): # Several columns can make no result
        ss.BoolState('x', columns=2)
    return sim


def test_derived_arrays():
    """ Arrays made from others (comparisons, astype) take the NaN and default of their own type; in-place ops work on shared arrays; time in state as the engine's clock """
    times = []
    def time_in(sim): # During the run, when some have just been infected
        times.append(sim.diseases.sir.I.time_in.value)
    sim = ss.Sim(ss.SIR(), networks='random', n_agents=500, demographics=True, solver='abm', unit='day', dur=30, analyzers=time_in, verbose=0).run()
    ppl = sim.people
    age = ppl.age
    inactive = np.setdiff1d(np.arange(len(age.raw)), ppl.auids)
    assert len(inactive)
    older = age._boolmath('>', age.values + 1) # Values for the active agents only: False elsewhere, not left as they were in memory
    assert not older.raw[inactive].any()
    b = age > 3
    assert b.nan is False and b.default is None
    b.set_nan(ss.uids([0]))
    assert b.raw[0] == False
    with np.errstate(invalid='ignore'): # The NaN of the room for births
        assert age.astype(ss.IntArr).nan == ss.dtypes.int_nan
    with pytest.raises(TypeError): # Needs its categories
        age.astype(ss.CatArr)
    f = ss.BoolArr('f', shared=True, mock=5) # No memory of its own until written
    f |= ss.uids([1])
    f &= (f == True)
    f ^= ss.uids([2])
    assert f.uids.tolist() == [1, 2]
    with pytest.raises(TypeError):
        ~ss.uids([1])
    t = np.concatenate(times) # As the engine's clock, those infected in this step have spent none of it there (not -1 steps)
    assert np.isfinite(t).any() and np.nanmin(t) == 0
    with pytest.raises(TypeError): # Not an AttributeError, which would be hidden
        ppl.alive.time_in
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_arrs()
    test_arr_arithmetic()
    test_arr_2d()
    test_uids()
    test_catarr()
    test_boolstate_result()
    test_derived_arrays()
    T.toc()
