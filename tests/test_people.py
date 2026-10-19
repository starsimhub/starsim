"""
Test the People object
"""
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
import matplotlib.pyplot as plt

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
    ppl.add_module(ssl.diseases.HIV())

    # Dead agents are removed from other arrays, but people.dead covers all agents
    sim = ss.Sim(n_agents=small, diseases=dict(type='sir', init_prev=0.2, p_death=0.5), networks='random', verbose=0).run()
    dead = sim.people.dead
    assert len(dead) == sim.people.n_uids > len(sim.people)
    assert dead.sum() == sim.results.cum_deaths[-1] > 0
    assert not np.isin(dead.uids, sim.people.auids).any()

    # Agents that stand for fewer people (people.scale, as HPVsim's finer agents) count for less in the results (before, every agent counted as one)
    S, I = ss.states('S I')
    sis = ss.Disease(S, I, infection=S >> I @ ss.peryear(0.0)*I, recovery=I >> S @ ss.years(1000), init=dict(I=0.5), name='sis')
    sim = ss.Sim(sis, n_agents=1000, start=2000, stop=2001, verbose=0).init()
    sim.people.scale[sim.people.auids[:500]] = 0.01
    sim.run()
    weights = sim.people.scale.raw[sim.people.auids]
    assert sim.results.n_alive[-1] == weights.sum() == 505
    assert np.isclose(sim.results.sis.n_I[-1], weights[sim.diseases.sis.I.raw[sim.people.auids]].sum())

    # And in a declared model's rates: here infected agents who stand for 0.01 people each make I.frac 0.01, not 0.5
    sis = ss.Disease(S, I, infection=S >> I @ ss.peryear(2)*I.frac, recovery=I >> S @ ss.years(1000), init=dict(I=0.5), name='sis')
    sim = ss.Sim(sis, n_agents=1000, start=2000, stop=2001, verbose=0).init()
    sim.people.scale[sim.diseases.sis.I.uids] = 0.01
    assert sim.run().results.sis.infection[-1] < 50 # About 10 (unweighted, about 316)

    # Memory: a People not yet in a sim (as sim.inputs keeps for sim.new()) has no arrays per agent; arrays nothing sets take none (the UIDs are made when read, slot reads them; parent, scale, rel_sus, and rel_trans are their defaults); UIDs are 32-bit and times of death 16-bit
    sim = ss.Sim(ss.SIR(), people=ss.People(small), networks='random', verbose=0)
    assert 'auids' not in sim.inputs['people'].__dict__
    sim.run()
    ppl, sir = sim.people, sim.diseases.sir
    assert ppl.slot.shared is ppl.uid and ppl.parent.shared and ppl.scale.shared and sir.rel_sus.shared # No memory of their own
    assert ppl.auids.dtype == np.int32 and ppl.ti_dead.raw.dtype == np.int16
    assert '_raw' not in ppl.uid.__dict__ and np.array_equal(ppl.uid.raw[:ppl.n_uids], np.arange(ppl.n_uids)) and ppl.slot.raw[5] == 5 # people.uid is the index, with no array
    assert ppl.age[ppl.auids[0]] == ppl.age.raw[0] and len(ppl.age[ppl.auids.view(np.ndarray)]) == len(ppl) # One UID, or a plain array of them, indexes as before
    assert np.array_equal(sim.new().run().results.sir.n_I, sim.results.sir.n_I) # sim.new() rebuilds the same sim
    ppl.scale[ppl.auids[:10]] = 2 # Code that sets one (as a Starsim 3 module does) gives it its own values
    assert ppl.scale.weighted and ppl.scale.shared is None and ppl.scale[ppl.auids[:11]].sum() == 21

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
        f3 = f2 * (~ppl.sir.I)
        af_res = f3.uids.to_numpy()

    # Equivalent using filter
    with sc.timer('Custom filtering'):
        f1 = ppl.filter('female')
        f2 = f1('age')>5
        f3 = ~f2('sir.I')
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
    gon = ssl.diseases.HIV(**gon_pars)

    sim = ss.Sim(pars=sim_pars, diseases=[gon])
    sim.init()
    sim.run()
    if do_plot:
        plt.figure()
        plt.plot(sim.timevec, sim.results.hiv.n_infected)
        plt.title('Number of HIV infections')

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
