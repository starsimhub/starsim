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
    gon = ssl.diseases.HIV(**gon_pars)

    sim = ss.Sim(pars=sim_pars, diseases=[gon])
    sim.init()
    sim.run()
    if do_plot:
        plt.figure()
        plt.plot(sim.timevec, sim.results.hiv.n_infected)
        plt.title('Number of HIV infections')

    return sim


@sc.timer()
def test_people_results_optout():
    """ people_results=False skips People-level result collection without changing disease dynamics. """
    sc.heading('Testing People results opt-out')

    kw = dict(diseases='sis', networks='random', n_agents=500, dur=10, rand_seed=3, verbose=0)
    s_on = ss.Sim(**kw).run()
    s_off = ss.Sim(people_results=False, **kw).run()

    # Disease dynamics are identical: People results do not feed back into the modules
    assert s_on.summary.sis_cum_infections == s_off.summary.sis_cum_infections, 'Opting out of People results changed disease dynamics'

    # People-level results are present when enabled and absent when disabled
    assert 'n_alive' in s_on.results, 'People results should be present by default'
    assert 'n_alive' not in s_off.results, 'People results should be absent when opted out'

    # The loop should not schedule People.update_results when opted out
    def labels(sim):
        return [f"{f['module']}.{f['func_name']}" for f in sim.loop.funcs]
    assert 'people.update_results' in labels(s_on)
    assert 'people.update_results' not in labels(s_off)

    return s_off


@sc.timer()
def test_cum_deaths_invariant():
    """
    cum_deaths must equal cumsum(new_deaths), including deaths on the first and last
    timesteps (the previous implementation was lagged by one step and undercounted).
    """
    sc.heading('Testing cum_deaths cumulative-sum invariant')

    class Killer(ss.Intervention):
        """ Kill a couple of agents on the first and last timesteps """
        def step(self):
            sim = self.sim
            last = sim.t.npts - 1
            if sim.ti in (0, last):
                alive = sim.people.alive.uids
                if len(alive) >= 2:
                    sim.people.request_death(alive[:2])
            return

    sim = ss.Sim(n_agents=100, dur=5, interventions=Killer(), verbose=0).run()
    res = sim.results
    new = np.array(res.new_deaths)
    cum = np.array(res.cum_deaths)

    assert np.array_equal(cum, np.cumsum(new)), f'cum_deaths != cumsum(new_deaths):\n{cum}\nvs\n{np.cumsum(new)}'
    assert new[0] > 0, 'Expected deaths on the first timestep (ti==0)'
    assert new[-1] > 0, 'Expected deaths on the final timestep'
    assert cum[-1] == new.sum(), 'Final cum_deaths must include deaths on the final timestep'

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
    s_off = test_people_results_optout()
    scd   = test_cum_deaths_invariant()

    sc.toc(T)
    plt.show()
    print('Done.')
