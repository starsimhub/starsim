"""
Test Sim API
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def test_single_defaults(do_plot=do_plot):
    """ Test a single MixingPool using defaults """
    sc.heading('Testing single...')
    mp = ss.MixingPool()
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mp, label='Defaults') # One week time step
    sim.run()

    if do_plot:
        sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections

    return sim


def test_single_age(do_plot=do_plot):
    """ Test a single MixingPool by age """
    sc.heading('Testing age...')
    mp_pars = {
        'src': ss.AgeGroup(0, 15),
        'dst': ss.AgeGroup(15, None),
        'beta': ss.beta(0.4)
    }
    mp = ss.MixingPool(mp_pars)
    mp.eff_contacts.default = ss.poisson(lam=5)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mp, label='Age') # One week time step
    sim.run()

    if do_plot:
        sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections

    return sim


def test_single_sex(do_plot=do_plot):
    """ Test a single MixingPool by sex """
    sc.heading('Testing sex...')
    mp_pars = {
        'src': lambda sim: sim.people.female, # female to male transmission
        'dst': lambda sim: sim.people.male,
        'beta': ss.beta(0.4)
    }
    mp = ss.MixingPool(mp_pars)
    mp.eff_contacts.default.set(v=5)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mp, label='Sex') # One week time step
    sim.run()

    if do_plot:
        sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    assert(sim.people.male[sim.diseases.sir.ti_infected>0].all()) # All new infections should be in men

    return sim


def test_multi_defaults(do_plot=do_plot):
    """ Test MixingPools using defaults """
    sc.heading('Testing single...')
    mps = ss.MixingPools()
    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mps, label='Multi Defaults') # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim

def test_multi(do_plot=do_plot):
    """ Test MixingPools """
    sc.heading('Testing multi...')

    groups = [
        lambda sim: sim.people.female,
        lambda sim: sim.people.male,
    ]

    mps_pars = dict(
        beta_matrix = np.array([[0.04, 0.01], [0.02, 0.03]]),
        src = groups, # female to male transmission
        dst = groups,
    )
    mps = ss.MixingPools(mps_pars)
    mps.eff_contacts.default = ss.poisson(lam=3)

    sir = ss.SIR()
    sim = ss.Sim(diseases=sir, interventions=mps, label='Multi') # One week time step
    sim.run()

    if do_plot: sim.plot()

    assert(sim.results.sir['cum_infections'][-1] > sim.results.sir['cum_infections'][0]) # There were infections
    return sim

def test_multi_ses(do_plot=do_plot):
    """ Test MixingPools """
    sc.heading('Testing multi SES...')

    from enum import IntEnum
    class SES(IntEnum):
        LOW = 0
        MID = 1
        HIGH = 2

    ses = ss.FloatArr('SES', default=ss.choice(a=[SES.LOW, SES.MID, SES.HIGH], p=[0.5, 0.3, 0.2]), label='SES')
    ppl = ss.People(n_agents=5_000, extra_states=ses)

    mps_pars = dict(
        src = [lambda sim: ss.uids(sim.people.SES == s) for s in [SES.LOW, SES.MID, SES.HIGH]],
        dst = [lambda sim: ss.uids(sim.people.SES == s) for s in [SES.LOW, SES.MID]],

        # src on rows (1st dimension), dst on cols (2nd dimension)
        beta_matrix = np.array([
            [0.04, 0.00], # LOW->LOW,  LOW->MID
            [0.01, 0.04], # MID->LOW,  MID->MID
            [0.00, 0.01], # HIGH->LOW, HIGH->MID
        ])
    )
    mps = ss.MixingPools(mps_pars)
    mps.eff_contacts.default = ss.poisson(lam=3)

    def seeding(self, sim, uids):
        p = np.zeros(len(uids))
        high_ses = ss.uids(sim.people.SES == SES.HIGH)
        p[high_ses] = 0.5 # 50% of SES HIGH
        return p

    sir_pars = dict(
        init_prev = ss.bernoulli(p=seeding)
    )

    sir = ss.SIR(sir_pars)
    sim = ss.Sim(people=ppl, diseases=sir, interventions=mps, label='Multi') # One week time step
    sim.run()

    if do_plot: sim.plot()

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
