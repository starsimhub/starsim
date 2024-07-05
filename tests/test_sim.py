"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pytest

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively

def make_sim_pars():
    pars = sc.objdict(
        n_agents = n_agents,
        birth_rate = 20,
        death_rate = 15,
        networks = sc.objdict(
            type = 'randomnet',
            n_contacts = 4,
        ),
        diseases = sc.objdict(
            type = 'sir',
            dur_inf = 10,
            beta = 0.1,
        )
    )
    return pars

def make_sim_pars2():
    pars = sc.objdict(
        n_agents = n_agents*10,
        birth_rate = 10,
        death_rate = 10,
        networks = sc.objdict(
            type = 'randomnet',
            n_contacts = 2,
        ),
        diseases = sc.objdict(
            type = 'sir',
            dur_inf = 20,
            beta = 0.2,
        )
    )
    return pars


def test_demo(do_plot=do_plot):
    """ Test Starsim's demo run """
    sc.heading('Testing demo...')
    s1 = ss.demo(plot=do_plot)
    
    # Test explicit demo
    s2 = ss.Sim(diseases='sir', networks='random').run()
    
    # Test explicit 
    sir = ss.SIR()
    net = ss.RandomNet()
    s3 = ss.Sim(diseases=sir, networks=net).run()
    assert ss.check_sims_match(s1, s2, s3), 'Sims should match'
    return s1


def test_default(do_plot=do_plot):
    """ Create, run, and plot a sim with default settings """
    sc.heading('Testing default...')
    sim = ss.Sim(n_agents=n_agents).run()
    if do_plot:
        sim.plot()
    return sim


def test_simple(do_plot=do_plot):
    """ Create, run, and plot a sim by passing a parameters dictionary """
    sc.heading('Testing simple run...')
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    sim.run()
    if do_plot:
        sim.plot()
    return sim


def test_api():
    """ Test all different ways of creating a sim """
    sc.heading('Testing sim API...')
    
    # Check different ways of specifying a sim
    s1 = ss.Sim(n_agents=n_agents, diseases='sir', networks='random').run() # Supply strings directly
    s2 = ss.Sim(pars=dict(n_agents=n_agents, diseases='sir', networks='random')).run() # Supply as parameters
    s3 = ss.Sim(n_agents=n_agents, diseases=ss.SIR(), networks=ss.RandomNet()).run() # Supply as objects
    ss.check_sims_match(s1, s2, s3), 'Sims should match'
    
    # Check different ways of setting a distribution
    kw = dict(n_agents=n_agents, networks='random')
    d1 = ss.lognorm_ex(10) # Create a distribution with an argument
    d2 = ss.lognorm_ex(mean=10, stdev=2) # Create a distribution with kwargs
    d3 = ss.normal(loc=10) # Create a different type of distribution
    
    # Check specifying dist with a scalar
    s4 = ss.Sim(diseases=dict(type='sir', dur_inf=10), **kw).run() # Supply values as a scalar
    s5 = ss.Sim(diseases=dict(type='sir', dur_inf=d1), **kw).run() # Supply as a distribution
    ss.check_sims_match(s4, s5), 'Sims should match'
    
    # Check specifying dist with a list and dict
    s6 = ss.Sim(diseases=dict(type='sir', dur_inf=[10,2]), **kw).run() # Supply values as a list
    s7 = ss.Sim(diseases=dict(type='sir', dur_inf=dict(mean=10, stdev=2)), **kw).run() # Supply values as a dict
    s8 = ss.Sim(diseases=dict(type='sir', dur_inf=d2), **kw).run() # Supply as a distribution
    ss.check_sims_match(s6, s7, s8), 'Sims should match'
    
    # Check changing dist type
    s9  = ss.Sim(diseases=dict(type='sir', dur_inf=dict(type='normal', loc=10)), **kw).run() # Supply values as a dict
    s10 = ss.Sim(diseases=dict(type='sir', dur_inf=d3), **kw).run() # Supply values as a distribution
    ss.check_sims_match(s9, s10), 'Sims should match'
    
    # Check that Bernoulli distributions can't be changed
    with pytest.raises(TypeError):
        ss.Sim(diseases=dict(type='sir', init_prev=dict(type='normal', loc=10)), **kw).initialize()
    
    return s1


def test_complex_api():
    """ Test that complex inputs can be parsed correctly """
    sc.heading('Testing complex API...')
    
    def jump_age(sim):
        """ Arbitrary intervention to reduce people's ages in the simulation """
        if sim.ti == 20:
            sim.people.age[:] = sim.people.age[:] + 1000
    
    # Specify parameters as a dictionary
    p = dict(
        n_agents = 1000,
        label = 'v1',
        verbose = 'brief',
        end = 2020,
        networks = [
            ss.RandomNet(name='random1', n_contacts=6),
            dict(type='random', name='random2', n_contacts=4)
        ],
        diseases = [
            dict(type='sir',  dur_inf=dict(type='expon', scale=6.0)),
            dict(type='sis', beta=0.07, init_prev=0.1),
        ],
        demographics = [
            ss.Births(birth_rate=20),
            dict(type='deaths', death_rate=20)
        ],
        interventions = jump_age,
    )
    
    # Test with explicit initialization
    pars = ss.SimPars(n_agents=1000, label='v1', verbose='brief', end=2020)
    
    net1 = ss.RandomNet(name='random1', n_contacts=6)
    net2 = ss.RandomNet(name='random2', n_contacts=4)
    networks = ss.ndict(net1, net2)
    
    dis1 = ss.SIR(dur_inf=ss.expon(scale=6.0))
    dis2 = ss.SIS(beta=0.07, init_prev=ss.bernoulli(0.1))
    diseases = ss.ndict(dis1, dis2)
    
    dem1 = ss.Births(birth_rate=20)
    dem2 = ss.Deaths(death_rate=20)
    demographics = ss.ndict(dem1, dem2)
    
    int1 = ss.Intervention.from_func(jump_age)
    interventions = ss.ndict(int1)

    # Assemble
    s1 = ss.Sim(p)
    s2 = ss.Sim(pars=pars, networks=networks, diseases=diseases, demographics=demographics, interventions=interventions)
    
    # Run
    s1.run()
    s2.run()
    
    assert ss.check_sims_match(s1, s2), 'Sims should match'
    
    return s1


def test_simple_vax(do_plot=do_plot):
    """ Create and run a sim with vaccination """
    sc.heading('Testing simple vaccination...')
    ss.set_seed(1)
    pars = make_sim_pars()
    sim_base = ss.Sim(pars=pars)
    sim_base.run()

    my_vax = ss.sir_vaccine(pars=dict(efficacy=0.5))
    intv = ss.routine_vx(start_year=2015, prob=0.2, product=my_vax)
    sim_intv = ss.Sim(pars=pars, interventions=intv)
    sim_intv.run()
    
    assert sim_intv.summary.sir_cum_infections < sim_base.summary.sir_cum_infections, 'Vaccine should avert infections'

    # Check plots
    if do_plot:
        plt.figure()
        plt.plot(sim_base.yearvec, sim_base.results.sir.prevalence, label='Baseline')
        plt.plot(sim_intv.yearvec, sim_intv.results.sir.prevalence, label='Vax')
        plt.axvline(x=2015, color='k', ls='--')
        plt.title('Prevalence')
        plt.legend()

    return sim_base, sim_intv


def test_shared_product(do_plot=do_plot):
    """ Check that multiple interventions can use the same product """
    sc.heading('Testing sharing a product across interventions...')
    
    # Make interventions
    vax = ss.sir_vaccine(pars=dict(efficacy=0.5))
    routine1 = ss.routine_vx(name='early-small', start_year=2010, prob=0.05, product=vax)
    routine2 = ss.routine_vx(name='late-big', start_year=2020, prob=0.5, product=vax)
    
    # Make and run sims
    pars = make_sim_pars()
    s1 = ss.Sim(pars, label='Baseline')
    s2 = ss.Sim(pars, interventions=routine1, label='One vaccine campaign')
    s3 = ss.Sim(pars=pars, interventions=[routine1, routine2], label='Two vaccine campaigns')
    s1.run()
    s2.run()
    s3.run()
    
    if do_plot:
        plt.figure()
        for sim in [s1, s2, s3]:
            plt.plot(sim.yearvec, sim.results.sir.cum_infections, label=sim.label)
        plt.legend()
        plt.title('Impact of vaccination')
        plt.xlabel('Year')
        plt.ylabel('Cumulative infections')
        
    assert s2.summary.sir_cum_infections < s1.summary.sir_cum_infections, 'Vaccine should avert infections'
    assert s3.summary.sir_cum_infections < s2.summary.sir_cum_infections, 'Vaccine should avert infections'
    return s3


def test_components(do_plot=do_plot):
    """ Create, run, and plot a sim by assembling components """
    sc.heading('Testing components...')
    people = ss.People(n_agents=n_agents)
    network = ss.RandomNet(pars=dict(n_contacts=4))
    sir = ss.SIR(pars=dict(dur_inf=10, beta=0.1))
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    if do_plot:
        sim.plot()
    return sim


def test_parallel():
    """ Test running two identical sims in parallel """
    sc.heading('Testing parallel...')
    pars = make_sim_pars()
    pars2 = make_sim_pars2()

    # Check that two identical sims match
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars2, label='Sim2')])
    sims.init_sims()
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    #assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    sims._has_orig_sim
    sims.reset()
    sims.shrink()
    sims.reduce()
    sims.mean()
    sims.median()
    
    # Check that two non-identical sims don't match
    pars2 = sc.dcp(pars)
    pars2.diseases.beta *= 2
    sims = ss.MultiSim([ss.Sim(pars, label='Sim1'), ss.Sim(pars2, label='Sim2')])
    sims.run(keep_people=True)
    s1, s2 = sims.sims
    assert not np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


def test_settings():
    """ Create, run, and plot a sim by passing a parameters dictionary """
    sc.heading('Testing Settings Options...')
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    
    ss.dtypes   
    ss.options.get_orig_options()
    ss.options.values
    ss.options.dict_keys
    ss.options.dict_items
    ss.options.filter
    ss.options.index
    ss.options.items
    ss.options.enumitems
    ss.options.enumkeys
    ss.options.enumvals
    ss.options.enumvalues
    ss.options.disp()
    ss.options.get_default('verbose')
    ss.options.get_default('precision')
    ss.options.changed('verbose')
    ss.options.set_precision
    ss.options.set(verbose=0)
    ss.options.changed('verbose')
    ss.options.changed('precision')
    ss.options.set(warnings='error')
    ss.options.changed('warnings')
    ss.options.context(verbose=1)
    ss.options.context(warnings='error')
    ss.options.changed('warnings')
    ss.options.changed('verbose')
    
    sim.run()
    sim.step
    sim.created
    sim.dists
    sim.dt
    sim.filename
    sim.npts
    sim.ti
    sim.tivec
    sim.summary
    sim.finalize
    sim.complete
    sim.summarize
    sim.disp()
    sim.shrink()
    sim.export_df()
    sim.export_pars()
    ss.options.clear

    if do_plot:
        sim.plot()
    return sim

def test_run():
    """ Test single_run and multi_run """
    sc.heading('Testing SingleRun and MultiRun')
    pars=dict(n_agents=n_agents, diseases='sir', networks='random',verbose=0.5)
    # Check different ways of specifying a sim
    s1 = ss.Sim(n_agents=n_agents, diseases='sir', networks='random',verbose=0.5) # Supply strings directly
    s2 = ss.Sim(pars=pars) # Supply as parameters
    
    ss.run.single_run(s1)
    ss.run.single_run(s2) 
    assert ss.check_sims_match(s1, s2), 'Sims should match'    
    
    #Run multi_run
    ss.run.multi_run([ss.Sim(pars, label='Sim1'), ss.Sim(pars, label='Sim2')])
    
    #Run multi_run
    ss.run.parallel([ss.Sim(pars, label='SimA'), ss.Sim(pars, label='SimB')], n_runs=4)

    
    return s1,s2


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    
    sim0 = test_demo(do_plot=do_plot)
    sim1 = test_default(do_plot=do_plot)
    sim2 = test_simple(do_plot=do_plot)
    sim3 = test_api()
    sim4 = test_complex_api()
    sim5b, sim5i = test_simple_vax(do_plot=do_plot)
    sim6 = test_shared_product(do_plot=do_plot)
    sim7 = test_components(do_plot=do_plot)
    sim8a, sim8b = test_parallel()
    sim10 = test_nullnet()
    sim11 = test_staticnet()
    sim12 = test_settings()
    sim13=test_run()

    T.toc()
    
    if do_plot:
        plt.show()
