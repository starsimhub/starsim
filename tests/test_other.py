"""
Test Starsim features not covered by other test files
"""
import pytest
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from starsim.migrate import migrate_text
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
    hiv = ssl.diseases.HIV()
    # Set beta. The first entry represents transmission risk from infected p1 -> susceptible p2
    # Need to be careful to get the ordering right. The set-up here assumes that in the simple
    # sexual  network, p1 is male and p2 is female. In the prenatal network, p1=mothers, p2=babies.
    hiv.pars['beta'] = {'mf': [0.15, 0.10], 'prenatal': [0.2, 0]}

    sim = ss.Sim(
        people=ss.People(small),
        networks=[ss.MFNet(), ss.PrenatalNet()],
        demographics=ss.LegacyPregnancy(),
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
def test_deepcopy():
    sc.heading('Testing deepcopy')
    s1 = ss.Sim(pars=dict(diseases='sir', networks=ss.RandomNet()), n_agents=small)
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
    s1 = ss.Sim(diseases=ss.SIR(init_prev=0.1), networks='random', n_agents=small, start=2000)
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

    # Make both library and custom modules searchable
    ss.register_modules(ssl.diseases, my_modules)
    sim = ss.Sim(n_agents=1000, diseases=['hiv', 'mydisease'], networks='mynetwork')
    sim.run()

    # A method of a disease written as code, on a declared one, would never be called (before, it was silently ignored)
    with pytest.raises(ss.CapabilityError):
        class Hook(ss.SIS):
            def set_prognoses(self, uids, sources=None):
                pass

    return o


@sc.timer()
def test_migrate():
    """ Starsim 3 code rewritten by the migration script, and the warning about what it rewrites """
    sc.heading('Testing the migration script')
    old = '''
import starsim as ss

class Counter(ss.Module):
    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('new_events', dtype=int, scale=False), ss.Result('seen', dtype=int, scale=False))

    def step(self):
        self.results.new_events[self.ti] += 5 # Starsim 3: the step's index

    def update_results(self):
        super().update_results()
        self.results.seen[self.ti] = self.results.new_events[self.ti]
'''
    new, changes, todo = migrate_text(old)
    assert 'new_events[self.ti + 1]' in new # A result written in step() goes at the end of the step

    # Run both: the Starsim 3 code silently sees nothing, with a warning; the migrated code sees every event
    for text, seen in [(old, 0), (new, 50)]:
        scope = {}
        exec(text, scope)
        sim = ss.Sim(n_agents=100, modules=scope['Counter'](), start=0, stop=10, verbose=0)
        if seen:
            sim.run()
        else:
            with pytest.warns(RuntimeWarning, match='time index'):
                sim.run()
        assert sim.results.counter.seen.sum() == seen

    # Renames, a fractional verbose, and Starsim 3's diseases written as code (a subclass of ss.Disease is unchanged)
    text, changes, todo = migrate_text("p = ss.time_prob(0.1); pars = dict(verbose=1/12)\nclass MySIS(ss.SIS): pass\nclass MyNCD(ss.Disease): pass")
    assert text == "p = ss.probperyear(0.1); pars = dict(verbose=1)\nclass MySIS(ssl.AgentSIS): pass\nclass MyNCD(ss.Disease): pass"
    assert migrate_text('ss.link_dists(obj, sim)')[0] == 'ss.distributions.link_dists(obj, sim)' # Internal helpers are in their submodules
    assert 'ssl.AgentSIR' in migrate_text('sir = ss.SIR(beta=0.1)')[2][0][1] # A use of ss.SIR is flagged, not rewritten

    # An override of set_prognoses() that skips counting the infections, as before Starsim 3.6
    class Quiet(ssl.AgentSIS):
        def set_prognoses(self, uids, sources=None):
            self.susceptible[uids] = False
            self.infected[uids] = True
            self.ti_infected[uids] = self.ti
            self.ti_recovered[uids] = self.ti + 5
    with pytest.warns(RuntimeWarning, match='new_infections'):
        ss.Sim(n_agents=200, diseases=Quiet(beta=ss.peryear(5)), networks='random', verbose=0).run()
    return new


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1  = test_microsim(do_plot)
    sims1 = test_deepcopy()
    sims2 = test_deepcopy_until()
    mods  = test_custom_imports()
    mig   = test_migrate()

    sc.toc(T)
    plt.show()
    print('Done.')
