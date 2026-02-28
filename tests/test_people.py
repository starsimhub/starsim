"""
Test Starsim features not covered by other test files
"""
import numpy as np
import sciris as sc
import starsim as ss
import starsim_examples as sse
import matplotlib.pyplot as plt
import pytest

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

# %% Define the tests

@sc.timer()
def test_people():
    sc.heading('Testing people object')

    # Base people contains only the states defined in base.base_states
    ppl = ss.People(small)
    del ppl

    # Possible to initialize people with extra states, e.g. a geolocation
    def geo_func(n):
        locs = [1,2,3]
        return np.random.choice(locs, n)
    extra_states = [
        ss.FloatArr('geolocation', default=geo_func),
    ]
    ppl = ss.People(small, extra_states=extra_states)

    # Possible to add a module to people outside a sim (not typical workflow)
    ppl.add_module(sse.HIV())

    return ppl


@sc.timer()
def test_filtering():
    """ Test people filtering """
    sim = ss.Sim(n_agents=medium, dur=10, networks='random', diseases='sir', verbose=0)
    sim.run()
    ppl = sim.people

    # Traditional filtering
    with sc.timer('Array filtering'):
        f1 = ppl.female == True
        f2 = f1 * (ppl.age>5)
        f3 = f2 * (~ppl.sir.infected)
        af_res = f3.uids.to_numpy()

    # Equivalent using filter
    with sc.timer('Custom filtering'):
        f1 = ppl.filter('female')
        f2 = f1('age')>5
        f3 = ~f2('sir.infected')
        cf_res = f3.uids.to_numpy()

    assert np.array_equal(af_res, cf_res), 'Filtered arrays do not match'
    return f3


@sc.timer()
def test_ppl_construction(do_plot=False):
    sc.heading('Test making people and providing them to a sim')

    def init_debut(module, sim, uids):
        # Test setting the mean debut age by sex, 16 for men and 21 for women.
        loc = np.full(len(uids), 16)
        loc[sim.people.female[uids]] = 21
        return loc

    mf_pars = {
        'debut': ss.normal(loc=init_debut, scale=2),  # Age of debut can vary by using callable parameter values
    }
    sim_pars = {'networks': [ss.MFNet(**mf_pars)], 'n_agents': small}
    gon_pars = {'beta': {'mf': [0.08, 0.04]}}
    gon = sse.Gonorrhea(**gon_pars)

    sim = ss.Sim(pars=sim_pars, diseases=[gon])
    sim.init()
    sim.run()
    if do_plot:
        plt.figure()
        plt.plot(sim.timevec, sim.results.gonorrhea.n_infected)
        plt.title('Number of gonorrhea infections')

    return sim


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    ppl   = test_people()
    filt  = test_filtering()
    sim   = test_ppl_construction(do_plot)

    sc.toc(T)
    plt.show()
    print('Done.')
