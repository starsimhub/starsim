"""
Test the Stream object from streams.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.streams import NotResetException

def make_rng(name='Test'):
    """ Create and initialize a stream """
    sim = ss.Sim()
    sim.initialize()
    rng = ss.Stream('Test')
    rng.initialize(sim)

    return rng


# %% Define the tests

def test_sample(n=5):
    """ Simple sample """
    sc.heading('Testing stream object')
    rng = make_rng(n)
    uids = np.arange(n)

    draws = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws}')

    return draws


def test_reset(n=5):
    """ Sample, reset, sample """
    sc.heading('Testing sample, reset, sample')
    rng = make_rng(n)
    uids = np.arange(n)

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
    uids = np.arange(n)

    draws1 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    print(f'\nSTEP(1) - sample should change')
    rng.step(1)

    draws2 = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws2}')
    
    return np.all(draws2-draws1 != 0)


def test_repeat(n=5):
    """ Sample, sample - should raise and exception"""
    sc.heading('Testing sample, sample - should raise an exception')
    rng = make_rng(n)
    uids = np.arange(n)

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
    assert test_repeat()

    sc.toc(T)
    print('Done.')