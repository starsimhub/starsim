"""
Test Sim API
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import sys

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def test_single_defaults(do_plot=do_plot):
    """ Test a single MixingPool using defaults """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp = ss.MixingPool()
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mp, label=test_name) # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


def test_single_age(do_plot=do_plot):
    """ Test a single MixingPool by age """
    # Incidence must decline because 0-15 --> 15+ transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': ss.AgeGroup(0, 15),
        'dst': ss.AgeGroup(15, None),
        'beta': ss.beta(0.15),
        'contacts': ss.poisson(lam=5),
    }
    mp = ss.MixingPool(mp_pars)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mp, label=test_name) # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


def test_single_sex(do_plot=do_plot):
    """ Test a single MixingPool by sex """
    # Incidence must decline because M --> F transmission only
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mp_pars = {
        'src': lambda sim: sim.people.female, # female to male transmission
        'dst': lambda sim: sim.people.male,
        'beta': ss.beta(0.2),
        'contacts': ss.poisson(lam=4),
    }
    mp = ss.MixingPool(mp_pars)

    sir = ss.SIR(init_prev=ss.bernoulli(0.8))
    sim = ss.Sim(diseases=sir, interventions=mp, label=test_name)
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    assert(sim.people.male[sim.diseases.sir.ti_infected>0].all()) # All new infections should be in men
    return sim


def test_multi_defaults(do_plot=do_plot):
    """ Test MixingPools using defaults """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')
    mps = ss.MixingPools()
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mps, label=test_name) # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim

def test_multi(do_plot=do_plot):
    """ Test MixingPools """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    groups = [
        lambda sim: sim.people.female,
        lambda sim: sim.people.male,
    ]

    mps_pars = dict(
        contact_matrix = np.array([[1.4, 0.5], [1.2, 0.7]]),
        beta = ss.beta(0.2),
        src = groups,
        dst = groups,
    )
    mps = ss.MixingPools(mps_pars)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mps, label=test_name) # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim

def test_multi_ses(do_plot=do_plot):
    """ Test MixingPools SES """
    test_name = sys._getframe().f_code.co_name
    sc.heading(f'Testing {test_name}...')

    from enum import IntEnum
    class SES(IntEnum):
        LOW = 0
        MID = 1
        HIGH = 2

    ses = ss.FloatArr('SES', default=ss.choice(a=[SES.LOW, SES.MID, SES.HIGH], p=[0.5, 0.3, 0.2]), label='SES')
    ppl = ss.People(n_agents=11_000, extra_states=ses)

    mps_pars = dict(
        src = [lambda sim, s=s: ss.uids(sim.people.SES == s) for s in [SES.LOW, SES.MID, SES.HIGH]],
        dst = [lambda sim, s=s: ss.uids(sim.people.SES == s) for s in [SES.LOW, SES.MID]],

        # src on rows (1st dimension), dst on cols (2nd dimension)
        contact_matrix = np.array([
            [2.50, 0.00], # LOW->LOW,  LOW->MID
            [0.5, 2.50], # MID->LOW,  MID->MID
            [0.00, 0.5], # HIGH->LOW, HIGH->MID
        ]),

        beta = ss.beta(0.1),
    )
    mps = ss.MixingPools(mps_pars)
    
    def seeding(self, sim, uids):
        p = np.zeros(len(uids))
        high_ses = ss.uids(sim.people.SES == SES.HIGH)
        p[high_ses] = 0.1 # 10% of SES HIGH
        return p

    class AzSES(ss.Analyzer):
        def init_results(self):
            self.new_cases = np.zeros((self.npts, 3))

        def step(self):
            ti = self.ti
            new_inf = self.sim.diseases.sir.ti_infected == ti
            if not new_inf.any():
                return

            for ses in [SES.LOW, SES.MID, SES.HIGH]:
                self.new_cases[ti, ses] = np.count_nonzero(new_inf & (self.sim.people.SES==ses))

    az = AzSES()

    sir = ss.SIR(init_prev = ss.bernoulli(p=seeding))
    sim = ss.Sim(people=ppl, diseases=sir, interventions=mps, analyzers=az, label=test_name) # One week time step
    sim.run()

    if do_plot:
        sim.plot()
        fig, ax = plt.subplots()
        new_cases = sim.analyzers[0].new_cases
        for ses in [SES.LOW, SES.MID, SES.HIGH]:
            ax.plot(sim.results.timevec, new_cases[:,ses], label=ses.name)
        ax.legend()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sim0 = test_single_defaults(do_plot)
    sim1 = test_single_age(do_plot)
    sim2 = test_single_sex(do_plot)

    sim3 = test_multi_defaults(do_plot)
    sim4 = test_multi(do_plot)
    sim5 = test_multi_ses(do_plot)

    T.toc()

    if do_plot:
        plt.show()
