"""
Test the Stream object from streams.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.streams import NotResetException


def make_rng(n, base_seed=1, name='Test'):
    """ Create and initialize a stream """
    streams = ss.Streams()
    streams.initialize(base_seed=base_seed)
    rng = ss.Stream(name)
    bso = np.arange(n)
    rng.initialize(streams, bso)

    return rng


# %% Define the tests

def test_sample(n=5):
    """ Simple sample """
    sc.heading('Testing stream object')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws}')

    return draws


def test_reset(n=5):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    print(f'\nRESET')
    rng.reset()

    draws2 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws2}')

    return np.all(draws2-draws1 == 0)


def test_step(n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample, step, sample')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    print(f'\nSTEP(1) - sample should change')
    rng.step(1)

    draws2 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws2}')
    
    return np.all(draws2-draws1 != 0)


def test_seed(n=5):
    """ Sample, step, sample """
    sc.heading('Testing sample with seeds 0 and 1')
    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = make_rng(n, based_seed=0)
    draws0 = rng0.random(uids)
    print(f'\nSAMPLE({n}): {draws0}')

    rng1 = make_rng(n, based_seed=1)
    draws1 = rng1.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    return np.all(draws1-draws0 != 0)


def test_repeat(n=5):
    """ Sample, sample - should raise and exception"""
    sc.heading('Testing sample, sample - should raise an exception')
    rng = make_rng(n)
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


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    # Run tests
    test_sample()
    assert test_reset()
    assert test_step()
    assert test_seed()
    assert test_repeat()

    sc.toc(T)
    print('Done.')