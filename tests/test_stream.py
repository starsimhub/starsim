"""
Test the Stream object from streams.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.streams import NotResetException, Stream


def make_rng(multistream=False, base_seed=1, name='Test'):
    """ Create and initialize a stream """
    streams = ss.Streams()
    streams.initialize(base_seed=base_seed)
    rng = Stream(multistream)(name)
    rng.initialize(streams)

    return rng


# %% Define the tests

def test_sample(multistream=True, n=5):
    """ Simple sample """
    sc.heading('Testing stream object')
    rng = make_rng(multistream)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws}')

    return len(draws) == len(uids)


def test_reset(multistream=True, n=5):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    rng = make_rng(multistream)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    print(f'\nRESET')
    rng.reset()

    draws2 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws2}')

    return np.all(np.equal(draws1, draws2))


def test_step(multistream=True, n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample, step, sample')
    rng = make_rng(multistream)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    print(f'\nSTEP(1) - sample should change')
    rng.step(1)

    draws2 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws2}')
    
    return np.all(np.equal(draws1, draws2))


def test_seed(multistream=True, n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample with seeds 0 and 1')
    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = make_rng(multistream, base_seed=0)
    draws0 = rng0.random(uids)
    print(f'\nSAMPLE({n}): {draws0}')

    rng1 = make_rng(multistream, base_seed=1)
    draws1 = rng1.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    return np.all(np.equal(draws0, draws1))


def test_repeat(multistream=True, n=5):
    """ Sample, sample - should raise and exception"""
    sc.heading('Testing sample, sample - should raise an exception')
    rng = make_rng(multistream)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')
    
    print(f'\nSAMPLE({n}): [should raise an exception as neither reset() nor step() have been called]')
    try:
        rng.random(uids)
        return False # Should not get here!
    except NotResetException as e:
        print(f'YAY! Got exception: {e}')
    return True


def test_boolmask(multistream=True, n=5):
    """ Simple sample with a boolean mask"""
    sc.heading('Testing stream object')
    rng = make_rng(multistream)
    uids = np.arange(0,n,2) # every other to make it interesting
    mask = np.full(n, False)
    mask[uids] = True

    draws_bool = rng.random(mask)
    print(f'\nSAMPLE({n}): {draws_bool}')

    rng.reset()

    draws_uids = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws_uids}')

    return np.all(np.equal(draws_bool, draws_uids))


def test_empty(multistream=True):
    """ Simple sample with a boolean mask"""
    sc.heading('Testing empty draw')
    rng = make_rng(multistream)
    uids = np.array([]) # EMPTY
    draws = rng.random(uids)
    print(f'\nSAMPLE: {draws}')

    return len(draws) == 0


def test_drawsize():
    """ Testing the draw_size function directly """
    sc.heading('Testing draw size')

    rng = make_rng(multistream)

    x = ss.states.FusedArray(values=np.array([1,3,9]), uid=np.array([0,1,2]))
    ds_FA = rng.draw_size(x) == 10 # Should be 10 because max value of x is 9

    x = ss.states.DynamicView(int, fill_value=0)
    x.initialize(3)
    x[0] = 9
    ds_DV = rng.draw_size(x) == 10 # Should be 10 because max value (representing uid) is 9

    x = np.full(10, fill_value=True)
    ds_bool = rng.draw_size(x) == 10 # Should be 10 because 10 objects

    x = np.array([9])
    ds_array = rng.draw_size(x) == 10 # Should be 10 because 10 objects

    return np.all([ds_FA, ds_DV, ds_bool, ds_array])


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    n=5

    for multistream in [True, False]:
        print('Testing with multistream set to', multistream)

        # Run tests
        print(test_sample(multistream, n))
        print(test_reset(multistream, n))
        print(test_step(multistream, n))
        print(test_seed(multistream, n))
        print(test_repeat(multistream, n))
        print(test_boolmask(multistream, n))
        print(test_empty(multistream))
        print(test_drawsize())

    sc.toc(T)
    print('Done.')