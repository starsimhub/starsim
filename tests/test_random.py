"""
Test the RNG object from random.py
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import stisim as ss
from stisim.random import NotResetException, RNG


def make_rng(slots, base_seed=1, name='Test'):
    """ Create and initialize a random number generator within a container """
    rng_container = ss.RNGContainer()
    rng_container.initialize(base_seed=base_seed)
    rng = RNG(name)
    rng.initialize(rng_container, slots=slots)

    return rng


# %% Define the tests

def test_sample(n=5):
    """ Simple sample """
    sc.heading('Testing RNG object')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting

    draws = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws}')

    return len(draws) == len(uids)


def test_neg_binomial(n=5):
    """ Negative Binomial """
    sc.heading('Testing negative binomial')
    rng = make_rng(n)

    # Can call directly:
    #rng.negative_binomial(n=40, p=1/3, size=5)

    # Or through a Distribution
    nb = ss.neg_binomial(mean=80, dispersion=40)
    nb.set_rng(rng)
    draws = nb.sample(size=n)

    rng.step(1) # Prepare to call again
    # Now try calling with UIDs instead of n
    uids = np.arange(0,n,2) # every other to make it interesting
    draws_u = nb.sample(uids)

    print(f'\nSAMPLE({n}): {draws}')
    return len(draws) == n and len(draws_u) == len(uids)


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

    return np.all(np.equal(draws1, draws2))


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
    
    return ~np.all(np.equal(draws1, draws2))


def test_seed(n=5):
    """ Changing seeds """
    sc.heading('Testing sample with seeds 0 and 1')
    uids = np.arange(0,n,2) # every other to make it interesting

    rng0 = make_rng(n, base_seed=0)
    draws0 = rng0.random(uids)
    print(f'\nSAMPLE({n}): {draws0}')

    rng1 = make_rng(n, base_seed=1)
    draws1 = rng1.random(uids)
    print(f'\nSAMPLE({n}): {draws1}')

    return ~np.all(np.equal(draws0, draws1))


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


def test_boolmask(n=5):
    """ Simple sample with a boolean mask"""
    sc.heading('Testing RNG object with a boolean mask')
    rng = make_rng(n)
    uids = np.arange(0,n,2) # every other to make it interesting
    mask = np.full(n, False)
    mask[uids] = True

    draws_bool = rng.random(mask)
    print(f'\nSAMPLE({n}): {draws_bool}')

    rng.reset()

    draws_uids = rng.random(uids)
    print(f'\nSAMPLE({n}): {draws_uids}')

    return np.all(np.equal(draws_bool, draws_uids))


def test_empty():
    """ Simple sample with a boolean mask"""
    sc.heading('Testing empty draw')
    rng = make_rng(5) # Number of slots is arbitrary
    uids = np.array([]) # EMPTY
    draws = rng.random(uids)
    print(f'\nSAMPLE: {draws}')

    return len(draws) == 0


# %% Run as a script
if __name__ == '__main__':
    # Start timing
    T = sc.tic()

    n=5

    for multirng in [True, False]:
        ss.options(multirng=multirng)
        sc.heading('TESTING WITH MULTIRNG SET TO', multirng)

        # Run tests - some will only pass if multirng is True
        print(test_sample(n))
        print(test_neg_binomial(n))
        print(test_reset(n))
        print(test_step(n))
        print(test_seed(n))
        print(test_repeat(n))
        print(test_boolmask(n))
        print(test_empty())

    sc.toc(T)
    print('Done.')