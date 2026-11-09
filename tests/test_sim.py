"""
Test Sim API
"""

import pytest
import warnings
import numpy as np
import starsim as ss
import starsim.library as ssl
import sciris as sc
import matplotlib.pyplot as plt

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def make_sim_pars():
    pars = sc.obj(
        n_agents = n_agents,
        start = 2000, # Calendar years, for the interventions below
        demographics = [ss.birth(rate=ss.peryear(0.02)), ss.death(rate=ss.peryear(0.015))],
        networks = sc.obj(
            type = 'randomnet',
            n_contacts = 4,
        ),
        diseases = sc.obj(
            type = 'sir',
            dur_inf = ss.years(10), # In years, as the rest of the sim
            beta = ss.peryear(0.1),
        )
    )
    return pars


def test_capabilities_two_modules():
    """ Choosing the solver for a module with a non-exponential dwell time, then one with a function as a rate (the first's list of dwell times shadowed the types the second was checked against) """
    sc.heading('Testing solver capabilities over several modules')
    class A(ss.Disease):
        S, I = ss.states('S I')
        recover = ss.Transition(I, S, dur=ss.lognormal(ss.years(1), ss.years(1)))
        catch = ss.Transition(S, I, rate=ss.peryear(0.1))
    class B(ss.Disease):
        X, Y = ss.states('X Y')
        default_state = 'X'
        h = ss.par(None)
        go = ss.Transition(X, Y, rate=h)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.pars.update(h=lambda sim: ss.peryear(0.1))
    sim = ss.Sim(diseases=[A(name='a'), B(name='b')], n_agents=100, start=2000, stop=2001, verbose=0).run()
    assert sim.results.b.go.sum() >= 0
    assert all(cap.reason for cap in sim.capabilities) # Only what changes the choice is recorded

    # People given to a compartmental solver are where the population size came from, not a guess
    sim2 = ss.Sim(diseases=A(name='a'), people=ss.People(100), solver='ode', stop=2001, verbose=0).init()
    assert sim2.pop_from == 'people'
    assert '1 group;' in str(sim2.people) # Printing the people of an unstratified compartmental run (which has no ages) works
    return sim


def test_one_layout():
    """ The layout made to choose the solver is the one the run uses, so an SIR is laid out once; and one made for a run as agents says nothing """
    sc.heading('Testing that the compartments are laid out once')
    calls = []
    cls = ss.solvers.Compartmental
    original = cls.build_layout
    def tracked(self):
        calls.append(self.mod.name)
        return original(self)
    cls.build_layout = tracked
    try:
        sim = ss.Sim(ss.SIR(), stop=2, verbose=0).init()
        assert isinstance(sim.solver, ss.solvers.ode) and calls == ['sir'] # Was ['sir', 'sir']: once to choose the solver, and again for the run
        calls.clear()
        dur = ss.gamma(shape=2.5, scale=ss.days(3)) # Approximated by Erlang stages in a compartmental run, which is noted
        sims = [ss.Sim(ss.SIR(dur_inf=dur), solver=solver, unit='day', stop=10, verbose=0).init() for solver in ['auto', 'abm']]
        assert calls == ['sir', 'sir'] # Once each: for the run, and to say what auto would choose
    finally:
        cls.build_layout = original
    noted = ['Erlang' in note for note in sims[0].notes], ['Erlang' in note for note in sims[1].notes]
    assert any(noted[0]) and not any(noted[1]) # Said only for the run that uses the layout
    return sim


@sc.timer()
def test_demo(do_plot=do_plot):
    """ Test Starsim's demo run """
    sc.heading('Testing demo...')
    s1 = ss.demo(plot=do_plot)

    # Test explicit demo: its SIR has beta per contact, for an R0 of 3 on the network
    s2 = ss.Sim(diseases=dict(type='sir', beta=0.03), networks='random').run()

    # Test explicit
    sir = ss.SIR(beta=ss.perday(0.03))
    net = ss.RandomNet()
    s3 = ss.Sim(diseases=sir, networks=net).run()
    assert ss.debugtools.check_sims_match(s1, s2, s3), 'Sims should match'
    return s1


@sc.timer()
def test_default(do_plot=do_plot):
    """ Create, run, and plot a sim with default settings """
    sc.heading('Testing default...')
    sim = ss.Sim(n_agents=n_agents).run()
    if do_plot:
        sim.plot()
    with pytest.raises(ValueError): # It was a ZeroDivisionError
        ss.Sim(n_agents=0, total_pop=1, diseases='sir').init()
    assert not isinstance(ss.Sim(diseases='sir', n_agents=None).run().solver, ss.solvers.abm) # Not given, as if left out
    return sim


@sc.timer()
def test_simple(do_plot=do_plot):
    """ Create, run, and plot a sim by passing a parameters dictionary """
    sc.heading('Testing simple run...')
    pars = make_sim_pars()
    sim = ss.Sim(pars)
    sim.run()
    if do_plot:
        sim.plot()

    # A sim of modules written as code can be built again with other values, as a declared one can
    agentsim = ss.Sim(n_agents=n_agents, diseases=ssl.AgentSIS(name='sis'), networks='random', start=2000, stop=2010, verbose=0)
    again = agentsim.new(rand_seed=2).run()
    assert again.summarize().value.sis_infection == ss.Sim(n_agents=n_agents, diseases=ssl.AgentSIS(name='sis'), networks='random', start=2000, stop=2010, rand_seed=2, verbose=0).run().summarize().value.sis_infection

    # So can a subclass that adds its modules after Sim.__init__() (as FPsim's does), and settings changed after the sim was made are kept
    class MySim(ss.Sim):
        def __init__(self, beta=0.1, **kwargs):
            super().__init__(**kwargs)
            self.pars.diseases = ss.SIS(beta=ss.peryear(beta), dur_inf=ss.years(10), name='sis')
            self.pars.networks = 'random'
    mysim = MySim(beta=0.2, n_agents=n_agents, start=2000, stop=2010, verbose=0)
    mysim.pars.stop = 2005
    new = mysim.new()
    assert type(new) is MySim and new.pars.stop == 2005 and isinstance(new.pars.diseases, ss.SIS) # Was a plain ss.Sim with no modules, to 2010
    new = ss.parallel(mysim.run(), n_runs=2, workers=1).sims[1] # Built again with new(), since it has run
    assert type(new) is MySim and new.t.npts == 6 and 'sis' in new.results
    agentsim.pars.stop = 2005
    assert agentsim.new().pars.stop == 2005 # And for ss.Sim itself

    # new() copies the sim as it was before init(), so an edit in place is kept too (before, one that repr() did not show was lost); an edit after init() belongs to that run only
    class Values(ss.Module):
        def __init__(self):
            super().__init__(name='vals') # Not "values", a method of the results
            self.define_pars(weights=np.ones(2000))
    valsim = ss.Sim(Values(), stop=2, verbose=0)
    valsim.vals.pars.weights[1000] = 7
    assert valsim.new().vals.pars.weights[1000] == 7
    valsim.init().vals.pars.weights[1000] = 9
    assert valsim.new().vals.pars.weights[1000] == 7
    return sim


@sc.timer()
def test_api():
    """ Test all different ways of creating a sim """
    sc.heading('Testing sim API...')

    # Check different ways of specifying a sim
    s1 = ss.Sim(n_agents=n_agents, diseases='sir', networks='random').run() # Supply strings directly
    s2 = ss.Sim(dict(n_agents=n_agents, diseases='sir', networks='random')).run() # Supply as parameters
    s3 = ss.Sim(n_agents=n_agents, diseases=ss.SIR(), networks=ss.RandomNet()).run() # Supply as objects
    assert ss.debugtools.check_sims_match(s1, s2, s3), 'Sims should match'

    # Check different ways of setting a distribution
    kw = dict(n_agents=n_agents, networks='random')

    # Check specifying a duration with a scalar, which is in the model's unit, days here
    s4 = ss.Sim(diseases=dict(type='sir', dur_inf=10), **kw).run() # Supply values as a scalar
    s5 = ss.Sim(diseases=dict(type='sir', dur_inf=ss.days(10)), **kw).run() # Supply as a duration
    assert ss.debugtools.check_sims_match(s4, s5), 'Sims should match'

    # Check that Bernoulli distributions can't be changed
    with pytest.raises(TypeError):
        ss.Sim(diseases=dict(type='sir', init_prev=dict(type='normal', mean=10)), **kw).init()

    # A keyword given outright wins over the same key in a dict of parameters
    sim = ss.Sim(dict(diseases='sir', networks='random', n_agents=100), diseases=ss.SIS(), verbose=0).init()
    assert list(sim.diseases.keys()) == ['sis']

    return s1


@sc.timer()
def test_complex_api():
    """ Test that complex inputs can be parsed correctly """
    sc.heading('Testing complex API...')

    def jump_age(sim):
        """ Arbitrary intervention to reduce people's ages in the simulation """
        if sim.ti == 20:
            sim.people.age[:] = sim.people.age[:] + 1000

    # Specify parameters as a dictionary
    pars1 = dict(
        n_agents = 1000,
        label = 'v1',
        verbose = 0,
        stop = 2020,
        networks = [
            ss.RandomNet(name='random1', n_contacts=6),
            dict(type='random', name='random2', n_contacts=4)
        ],
        diseases = [
            dict(type='sir',  dur_inf=ss.expon(mean=ss.years(6.0))),
            dict(type='sis', beta=ss.prob(0.07, 'year'), init_prev=0.1),
        ],
        demographics = [
            ss.Births(birth_rate=ss.peryear(0.02)),
            dict(type='deaths', death_rate=ss.peryear(0.02))
        ],
        interventions = jump_age,
    )

    # Test with explicit initialization
    pars2 = ss.SimPars(n_agents=1000, label='v1', verbose=0, stop=2020)

    net1 = ss.RandomNet(name='random1', n_contacts=6)
    net2 = ss.RandomNet(name='random2', n_contacts=4)
    networks = ss.ndict(net1, net2)

    dis1 = ss.SIR(dur_inf=ss.expon(mean=ss.years(6.0)))
    dis2 = ss.SIS(beta=ss.prob(0.07, 'year'), init_prev=0.1)
    diseases = ss.ndict(dis1, dis2)

    dem1 = ss.Births(birth_rate=ss.peryear(0.02))
    dem2 = ss.Deaths(death_rate=ss.peryear(0.02))
    demographics = ss.ndict(dem1, dem2)

    int1 = ss.Intervention.from_func(jump_age)
    interventions = ss.ndict(int1)

    # Assemble and run
    s1 = ss.Sim(pars1).run()
    s2 = ss.Sim(pars2, networks=networks, diseases=diseases, demographics=demographics, interventions=interventions).run()
    assert ss.debugtools.check_sims_match(s1, s2), 'Sims should match'
    s3 = ss.Sim(pars=pars2, networks=networks, diseases=diseases, demographics=demographics, interventions=interventions).run() # A whole SimPars as pars=, as in Starsim 3
    assert ss.debugtools.check_sims_match(s1, s3), 'Sims should match'
    sir = ss.SIR()
    s4 = ss.Sim(sir, networks='random', pars=dict(start=2000, dur=2, n_agents=200, verbose=0, beta=0.2), n_agents=300) # Settings and the model's parameters in pars=, told apart by name; a keyword wins
    assert s4.pars.n_agents == 300 and s4.pars.dur == 2 and s4.par_values == dict(beta=0.2)
    with pytest.raises(TypeError, match='in pars='):
        ss.Sim(sir, beta=0.1) # A parameter of the model as a keyword
    with pytest.raises(sc.KeyNotFoundError):
        ss.Sim(sir, pars=dict(bta=0.1), verbose=0).init() # Neither a setting nor a parameter
    with sc.capture() as out: # A fraction: progress every quarter of the run (to a file; in a terminal or a notebook, a bar redrawn in place)
        ss.Sim(n_agents=100, start=2000, stop=2020, verbose=0.25).run()
    assert out.count('%') == 4, out
    with pytest.raises(ValueError):
        ss.Sim(n_agents=100, verbose=-1).init()
    assert len(ss.Sim(n_agents=100, networks=[['random', 'sexual'], 'prenatal'], demographics='pregnancy', verbose=0).init().networks) == 3 # One level of nested lists, e.g. from a helper that returns several modules

    return s1


@sc.timer()
def test_shared_product(do_plot=do_plot):
    """ Check that multiple interventions can use the same product """
    sc.heading('Testing sharing a product across interventions...')

    # Make interventions
    vax = ss.Vaccine(efficacy=0.5)
    routine1 = ss.vaccinate(vax, name='early-small', when=ss.schedule(every=ss.years(1), start=2010), coverage=0.05)
    routine2 = ss.vaccinate(vax, name='late-big', when=ss.schedule(every=ss.years(1), start=2020), coverage=0.5)

    # Make and run sims
    pars = make_sim_pars()
    s1 = ss.Sim(pars, label='Baseline').run()
    s2 = ss.Sim(pars, interventions=routine1, label='One vaccine campaign').run()
    s3 = ss.Sim(pars, interventions=[routine1, routine2], label='Two vaccine campaigns').run()

    if do_plot:
        plt.figure()
        for sim in [s1, s2, s3]:
            plt.plot(sim.timevec, sim.results.sir.infection.cum, label=sim.label)
        plt.legend()
        plt.title('Impact of vaccination')
        plt.xlabel('Year')
        plt.ylabel('Cumulative infections')

    assert s2.summarize().value.sir_infection < s1.summarize().value.sir_infection, 'Vaccine should avert infections'
    assert s3.summarize().value.sir_infection < s2.summarize().value.sir_infection, 'Vaccine should avert infections'
    return s3


@sc.timer()
def test_components(do_plot=do_plot):
    """ Create, run, and plot a sim by assembling components """
    sc.heading('Testing components...')
    people = ss.People(n_agents=n_agents)
    network = ss.RandomNet(n_contacts=4)
    sir = ss.SIR(dur_inf=10, beta=0.1)
    sim = ss.Sim(diseases=sir, people=people, networks=network)
    sim.run()
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_creation_syntax():
    """ Demonstrate different ways of creating a sim """
    sc.heading('Testing sim creation syntax...')
    
    class FirstClass(ss.Module):
        def step(self):
            self.sim.first = True
    
    class SecondClass(ss.Module):
        def step(self):
            assert self.sim.first # Will error if FirstClass is not run first
    
    # Create different modules
    mod1 = FirstClass()
    mod2 = SecondClass()
    dem = ss.Births()
    net = ss.RandomNet()
    dis = ss.SIS()
    
    # Test 1: check that order is respected when specified
    pars = dict(n_agents=100, dur=ss.years(10))
    sim = ss.Sim(
        **pars,
        modules = [mod1, dem],
        demographics = 'deaths',
        connectors = mod2,
        diseases = dis,
        networks = net,
    )
    sim.run()
    
    assert isinstance(sim.custom[0], FirstClass)
    assert len(sim.demographics) == 2 and all(isinstance(mod, ss.Demographics) for mod in sim.demographics())
    assert isinstance(sim.connectors[0], SecondClass)
    assert isinstance(sim.diseases[0], ss.Disease)
    assert isinstance(sim.networks[0], ss.Network)
    with pytest.raises(TypeError, match='Was expecting'):
        ss.Sim(diseases=ss.RandomNet()).init() # A module of another type
    
    # Test 2: check that the order is intuited when not specified
    sim2 = ss.Sim(
        **pars,
        modules = [dis, mod1, net, mod2, dem]
    )
    sim2.run()
    
    assert isinstance(sim2.custom[0], FirstClass)
    assert isinstance(sim2.custom[1], SecondClass)
    assert isinstance(sim2.demographics[0], ss.Demographics)
    assert isinstance(sim2.networks[0], ss.Network)
    assert isinstance(sim2.diseases[0], ss.Disease)

    # Test 3: transitions given to the sim are one disease, named from its states, whose results are also at the top
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    recovery = I >> R @ 0.1
    sims = [ss.Sim(infection, recovery), ss.Sim([infection, recovery]), ss.Sim(diseases=[infection, recovery])]
    for sim in sims:
        sim.run(verbose=0)
        assert list(sim.diseases.keys()) == ['sir'] and sim.sir is sim.diseases.sir # A shortcut to each module
        assert sim.results.I is sim.results.sir.I and sim.results.infection is sim.results.sir.infection
    assert np.array_equal(sims[0].results.I, sims[2].results.I)
    with pytest.warns(RuntimeWarning, match='sim.label is not'): # A module named like an attribute of the sim has no shortcut
        sim = ss.Sim(diseases=ss.SIR(name='label'), n_agents=100).init()
    assert not isinstance(sim.label, ss.Module)

    return sim2


@sc.timer()
def test_add_module():
    """ Test adding a module to a sim that has been partly run """
    sc.heading('Testing sim.add_module()...')
    kw = dict(n_agents=n_agents, diseases='sis', networks='random', start='2000-01-01') # The default run of 100 days
    base = ss.Sim(**kw).run()
    sim = ss.Sim(**kw)
    sim.run(until='2000-02-01')
    ti = sim.ti
    vx = sim.add_module(ss.modify('sis.pars.beta', multiply=0.2, name='masks'))
    with pytest.raises(ss.CapabilityError):
        sim.add_module(ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5)) # A new product adds a dimension (has_vaccine), which is laid out at init()
    sim.run()

    vaccine = ss.Vaccine(efficacy=0.9) # A product the sim already has can be delivered again
    vsim = ss.Sim(**kw, interventions=ss.vaccinate(vaccine, when=['2000-01-15'], coverage=0.2))
    vsim.run(until='2000-02-01')
    vsim.add_module(ss.vaccinate(vaccine, when=['2000-02-15'], coverage=0.5, name='second'))
    vsim.run()
    assert vsim.interventions.second.product is vsim.products()[0]

    inf = sim.results.sis.I
    base_inf = base.results.sis.I
    assert sim.connectors.masks is vx and vx.ti == sim.ti # It was added, and kept in step with the sim
    assert np.array_equal(inf[:ti], base_inf[:ti]) # No change before the module was added...
    assert inf[-1] < base_inf[-1] # ...but fewer infections after
    with pytest.raises(ss.AlreadyRunError):
        sim.add_module(ss.Intervention()) # Can't add a module to a completed sim
    return sim


@sc.timer()
def test_locked_pars():
    """ Test that parameters that don't exist can't be set after initialization """
    sc.heading('Testing locked parameters...')
    sim = ss.Sim(n_agents=n_agents, diseases='sis', networks='random').init()
    with pytest.raises(sc.KeyNotFoundError):
        sim.pars['rand_sed'] = 2 # Typo in a sim parameter
    with pytest.raises(sc.KeyNotFoundError):
        sim.diseases.sis.pars.bta = 0.1 # Typo in a module parameter
    sim.diseases.sis.pars.beta = 0.1 # Existing parameters of a module can be set
    sim.diseases.sis.beta = 0.2 # Directly too
    assert sim.diseases.sis.pars.beta.value == 0.2
    assert ss.SIS(beta=0.3).beta.value == 0.3 # Each module has its own, not the class's
    with pytest.raises(RuntimeError, match='sim.new'):
        sim.pars.rand_seed = 2 # But not the sim's settings, which are its run's: sim.new(rand_seed=2)
    sim.pars.update(my_par=3, create=True) # New parameters can be added explicitly
    assert sim.copy().pars.my_par == 3 # Locked parameters can be copied
    return sim


@sc.timer()
def test_new_after_init():
    """ What a sim takes once it is initialized (values and a seed from restart(pars=, rand_seed=), data, jax) is kept by running it again and by sim.new() """
    sc.heading('Testing sim.new() after init')
    import pandas as pd
    def make(beta=0.3):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=S >> I @ ss.par(beta, bounds=[0.1, 0.6], name='beta')*I, recovery=I >> R @ 0.1, init_n=dict(I=10), name='sir')
        return ss.Sim(sir, unit='day', stop=60, total_pop=1000, verbose=0, solver='binomial')
    ref = make(0.5).new(rand_seed=3).run().results.sir.I[-1]
    sim = make().init()
    assert sim.restart(pars=dict(beta=0.5), rand_seed=3).run().results.sir.I[-1] == ref
    assert sim.run().results.sir.I[-1] == ref and sim.new().run().results.sir.I[-1] == ref, 'Run again, or new, with the values of the restart'
    df = pd.DataFrame({'time':[10, 20], 'sir.I':[50, 100]})
    sim.data = df # Not settings of the run, so they can be set after init
    sim.jax = True
    new = sim.new()
    assert new.data is not None and new.jax is True
    return


@sc.timer()
def test_update_pars():
    """ A single value given for a distribution sets its main parameter, as in Starsim 3; a distribution replaces it """
    sc.heading('Testing parameter updates...')
    with pytest.warns(RuntimeWarning, match=r'set only the mean .*=ss\.constant\(6\)'): # Keeps the std, but says so (only for dur_inf: a probability is the whole of a Bernoulli)
        sir = ssl.AgentSIR(init_prev=0.05, dur_inf=6, p_death=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ssl.AgentSIR(init_prev=0.05, dur_inf=ss.constant(6))
    assert isinstance(sir.pars.init_prev, ss.bernoulli) and sir.pars.init_prev.pars.p == 0.05 # A probability
    assert isinstance(sir.pars.dur_inf, ss.lognormal) and sir.pars.dur_inf.pars.mean == ss.years(6) # Its mean, in the unit of the default
    assert ss.SexualNet(dur=ss.days(15)).pars.dur == ss.days(15) and isinstance(ss.SexualNet(dur=ss.constant(ss.days(15))).pars.dur, ss.constant) # A single duration on a network of partnerships is kept as given, the mean of partnerships ending at a constant rate; a fixed value is a constant
    assert isinstance(ss.RandomNet(n_contacts=dict(type='poisson', mean=4)).pars.n_contacts, ss.poisson) # A distribution given by its type
    for value in [[6, 2], dict(mean=6)]: # The distribution's arguments, or values? Neither: use its own method
        with pytest.raises(TypeError, match='set'):
            ssl.AgentSIR(dur_inf=value)
    sir.pars.dur_inf = ss.lognormal(mean=ss.years(6), std=ss.years(1))
    sir.pars.dur_inf.set(mean=8) # Keeps the standard deviation, and the unit
    assert sir.pars.dur_inf.pars.mean == ss.years(8) and sir.pars.dur_inf.pars.std == ss.years(1)

    # A sim parameter set before init() is kept, as if given (e.g. by a subclass of ss.Sim, as STIsim's does)
    sim = ss.Sim(n_agents=100, start=2000, stop=2002, verbose=0)
    sim.pars.aging = True
    sim.init()
    age0 = sim.people.age.raw[:100].copy()
    sim.run()
    assert sim.pars.aging and np.allclose(sim.people.age.raw[:100] - age0, 3) # A step from each of 2000, 2001, and 2002
    return


@sc.timer()
def test_save():
    """ Test save and export """
    sc.heading('Testing save and export...')
    f = sc.obj() # Filenames
    f.sim  = 'temp.sim'
    f.json = 'sim.json'
    sim = ss.Sim(n_agents=n_agents, diseases=ss.SIS(dur_inf=ss.lognormal(mean=ss.years(10))), networks='random').run()

    # Run the methods
    sim.save(filename=f.sim)
    sim.to_json(filename=f.json)
    json = sim.to_json(strkeys=True) # To match the loaded file, which converts keys to strings

    # Run methods
    s2 = sc.load(f.sim)
    json2 = sc.loadjson(f.json)

    # Run tests
    assert sim.summarize().equals(s2.summarize()), 'Sims do not match'
    assert sim.pars.n_agents == json2['pars']['n_agents'], 'Parameters do not match'
    assert json == json2, 'Outputs do not match'

    # Shrinking removes large objects from the distributions
    sim.shrink()
    assert isinstance(sim.diseases.sis.pars.dur_inf.dist, ss.utils.Shrunk)

    # Delete files
    sc.rmpath(f.values())

    return sim


@sc.timer()
def test_shrink_declared():
    """ Test that shrinking a sim with a declared disease and a treatment that cures by transition leaves its modules small, and its results readable and plottable """
    sc.heading('Testing shrinking a declared disease and a treatment')
    S, L, I, R, C = ss.states('S L I R C')
    tb = ss.Disease(S, L, I, R, C, infection=S >> L @ ss.peryear(2)*I, activation=L >> I @ ss.peryear(0.3), recovery=I >> R @ ss.years(1), clear=ss.Transition(L, C, rate=0), name='tb', init=dict(L=0.2, I=0.02))
    tx = ss.treat(ss.Treatment(cure={tb.clear: 0.7}), who=tb.L, when=ss.schedule(start=2005), coverage=ss.peryear(0.5), name='tpt')
    sim = ss.Sim(tb, interventions=tx, n_agents=2000, start=2000, stop=2020, dt=ss.months(1), verbose=0).run()
    sim.shrink(size_limit=0.3) # Raises if a module is still big; the delivery reaches the disease (through its moves), whose results and arrays are not its own
    assert sim.results.tb.I.values.sum() > 0 and sim.results.tpt.delivered.values.sum() > 0
    sim.plot()
    plt.close('all')
    with pytest.raises(RuntimeError, match='shrunk'):
        sim.tb.current_state()
    with pytest.raises(RuntimeError, match='copy'):
        tb.current_state() # The module given, not the one that ran
    return sim


@sc.timer()
def test_save_mixed_dt():
    """ Test that export works when modules span different time axes (issue #1411) """
    sc.heading('Testing export with heterogeneous timelines...')
    fn = 'sim_mixed.json'
    births = ss.Births(birth_rate=20, dt=ss.years(1)) # An annual module in a sub-annual sim
    sim = ss.Sim(n_agents=n_agents, dur=ss.years(3), dt=0.5, diseases='sis', networks='random', demographics=births).run()

    # With heterogeneous timelines, to_df() returns one dataframe per module rather than a single dataframe
    df = sim.to_df()
    assert not isinstance(df, sc.dataframe), 'Expected a dict of dataframes for a mixed-timeline sim'

    json = sim.to_json(filename=fn, strkeys=True)
    json2 = sc.loadjson(fn)
    assert set(json['results'].keys()) == set(df.keys()), 'Exported results should have one entry per module'
    assert json == json2, 'Outputs do not match'
    sc.rmpath(fn)

    return sim


@sc.timer()
def test_rescale():
    """ Test dynamic rescaling, and that duplicate modules get unique names """
    sc.heading('Testing dynamic rescaling')
    def make_sim(init_inf, **kwargs):
        sir = ss.SIR(beta=ss.perday(0.025), dur_inf=ss.days(10), init_n=dict(I=init_inf), p_death=0)
        return ss.Sim(n_agents=5e3, total_pop=5e4, diseases=sir, networks=ss.RandomNet(n_contacts=10), dt=ss.days(1), dur=ss.days(150), verbose=0, **kwargs)
    fixed = make_sim(init_inf=50).run() # Enough seeds, 5 agents, that the epidemic seldom dies out (with 2, a run in 6 does)
    rescaled = make_sim(init_inf=50, rescale=True).run()
    scale = rescaled.results.pop_scale
    assert scale[0] == 1 and scale[-1] == 10 == rescaled.pop_scale # Scale starts at 1 and ends at pop_scale
    assert np.array_equal(make_sim(init_inf=50, rescale=dict(threshold=0.05, factor=1.2)).run().results.pop_scale, scale) # The defaults, given
    assert not np.array_equal(make_sim(init_inf=50, rescale=dict(threshold=0.2)).run().results.pop_scale, scale)
    with pytest.raises(TypeError, match='threshold'): # An option it does not have
        make_sim(init_inf=50, rescale=dict(thresh=0.2)).init()
    assert make_sim(init_inf=20, rescale=True).current_scale == 1 # Available before the sim is initialized
    cum_fixed = fixed.results.sir.infection.cum[-1]
    cum_rescaled = rescaled.results.sir.infection.cum[-1]
    assert abs(cum_rescaled/cum_fixed - 1) < 0.1 # Rescaling gives about the same epidemic
    S, I, R = ss.states('S I R')
    flu = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_n=dict(I=10))
    seeded = lambda **kw: ss.Sim(flu, total_pop=1e6, n_agents=5000, dur=ss.days(2), verbose=0, **kw).init().diseases[0].I.count()
    assert seeded(rescale=True) == 10 # A number of people is a number of agents at the start scale (it was 0.05 agents, so nobody)
    assert seeded() == 1 # Too few to be an agent: one, with a note

    # A declared disease counts the people no agent stands for yet in its first state, and so do alive and the ratios
    flu = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_n=dict(I=10), prevalence=I/(S+I+R), name='flu')
    res = ss.Sim(flu, total_pop=1e6, n_agents=2000, dur=ss.days(60), rescale=True, verbose=0).run().results
    assert np.allclose(res.flu.S + res.flu.I + res.flu.R, 1e6) and np.allclose(res.alive, 1e6)
    assert res.flu.I[0] == 10 and np.isclose(res.flu.prevalence[0], 1e-5)
    deadly = ss.SIR(beta=ss.perday(0.3), dur_inf=ss.days(10), p_death=0.5, init_prev=0.002) # An agent's death is scale people, not those no agent stands for yet
    res = ss.Sim(diseases=deadly, networks='random', n_agents=2000, total_pop=2e6, rescale=True, stop=40, unit='day', verbose=0).run().results
    assert res.died.cum[-1] > 0 and np.allclose(res.alive, 2e6 - (res.died.cum - res.died)) and np.allclose(res.sir.S + res.sir.I + res.sir.R, res.alive) # Those alive at each time point: less the deaths of the steps before it
    preg = lambda rescale: ss.Sim(demographics=ss.Pregnancy(), n_agents=2000, total_pop=1e6, rescale=rescale, stop=5, unit='day', verbose=0).run().results.pregnancy
    assert np.allclose(preg(True).P, preg(False).P, rtol=1e-3) # In a module that is not a disease, those no agent stands for are in each state as the agents are (rather than all not pregnant); a birth changes the agents' shares a little

    sim = ss.Sim(n_agents=100, diseases=['sis', 'sis'], networks='random').init()
    assert list(sim.diseases.keys()) == ['sis', 'sis_1'] # Default names are made unique, with no warning
    return rescaled


@sc.timer()
def test_copied_modules():
    """ Test that a sim runs copies of the modules it is given, so they can be given to other sims, and functions read them from the sim """
    sc.heading('Testing copied modules')
    class HS(ss.Intervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.served_now = ss.uids()
        def step(self):
            self.served_now = self.sim.diseases.toy.I.uids[:5]

    S, I = ss.states('S I')
    toy = ss.Disease(S, I, inf=S >> I @ ss.peryear(2.0), rec=I >> S @ ss.days(90), name='toy')
    hs = HS(name='hs')
    deliver = lambda sim: sim.interventions.hs.served_now # The sim's copy, which runs
    sims = [ss.Sim(toy, interventions=[hs, ss.treat(ss.Treatment(cure=1.0), who=deliver, when=ss.schedule(every=ss.days(1)), name='tx')], n_agents=2000, dur=ss.days(100), dt=ss.days(1), verbose=0).run() for _ in range(2)]
    assert sims[0].results.tx.delivered.values.sum() > 400
    assert sims[0].results.tx.delivered.values.sum() == sims[1].results.tx.delivered.values.sum() # The same modules, given to two sims
    assert not toy.initialized and sims[0].diseases.toy is not toy # Neither ran the module itself
    return


@sc.timer()
def test_init_order():
    """ Test that a custom module is initialized after the modules it may read """
    sc.heading('Testing the order modules are initialized in')
    class Demography(ss.Demographics):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.FloatArr('district', default=-1))
        def init_post(self):
            super().init_post()
            self.district[self.sim.people.auids] = np.arange(len(self.sim.people.auids)) % 32
    class Lifestyle(ss.Module):
        def init_post(self):
            super().init_post()
            self.unset = np.mean(self.sim.demographics[0].district[self.sim.people.auids] < 0)
    sim = ss.Sim(n_agents=1000, demographics=Demography(), modules=Lifestyle(name='life'), start=2010, stop=2011, verbose=0).init()
    assert sim.custom.life.unset == 0 # Was 1: the custom module came first

    # Whatever reads the disease in init_post() sees who starts infected
    seen = {}
    class Reader(ss.Connector):
        def init_post(self):
            super().init_post()
            seen[self.name] = self.sim.diseases.sir.I.n
        def step(self):
            pass
    class Custom(ss.Module): # Of no standard type, so kept in sim.custom
        def init_post(self):
            super().init_post()
            seen[self.name] = self.sim.diseases.sir.I.n
    sir = ss.SIR(init_n=dict(I=10), name='sir')
    ss.Sim(n_agents=1000, diseases=sir, connectors=Reader(name='conn'), modules=Custom(name='cust'), stop=1, verbose=0).init()
    assert seen == dict(conn=10, cust=10) # Was 0 from the connector
    return sim


@sc.timer()
def test_step_order():
    """ Test that modules step in the order they were given with step_order='given', and by kind otherwise """
    sc.heading('Testing the order modules step in')
    log = []
    class Mark(ss.Module):
        def step(self):
            log.append(self.name)
    class Net(ss.RandomNet):
        def step(self):
            log.append(self.name)
            return super().step()
    make = lambda order: ss.Sim(modules=[Mark(name='first'), Net(name='net'), ss.SIR(name='sir'), Mark(name='last')], step_order=order, n_agents=200, stop=2001, verbose=0)
    make('type').run()
    assert log[:3] == ['first', 'last', 'net'] # Modules of no standard kind first, then the networks
    log.clear()
    sim = make('given').run()
    assert log[:3] == ['first', 'net', 'last']
    assert len(sim.custom) == 2 and len(sim.networks) == 1 # Still found by kind
    with pytest.raises(sc.KeyNotFoundError): # Modules of no standard kind are given with modules=
        ss.Sim(custom=Mark(), n_agents=100)
    return sim


def test_engine_rules():
    """ A module of the user's takes the route the built-in ones do; an engine says what it does not define """
    class Clinic(ss.Intervention):
        def agent_reasons(self, sim):
            return ['the clinic keeps a list of names']
    class Survey(ss.Analyzer):
        def agent_reasons(self, sim):
            return [ss.modules.Preference('the survey samples people')]
    kw = dict(diseases=ss.SIR(), stop=2002, verbose=0)
    sim = ss.Sim(interventions=Clinic(), **kw).init()
    assert isinstance(sim.solver, ss.solvers.abm) and sim.solver_reason == 'the clinic keeps a list of names'
    with pytest.raises(ss.CapabilityError, match='keeps a list'):
        ss.Sim(interventions=Clinic(), solver='ode', **sc.dcp(kw)).init()
    assert isinstance(ss.Sim(analyzers=Survey(), **sc.dcp(kw)).init().solver, ss.solvers.abm)
    assert isinstance(ss.Sim(analyzers=Survey(), solver='ode', **sc.dcp(kw)).init().solver, ss.solvers.ode)

    assert ss.solvers.why_not(ss.Sim(**sc.dcp(kw)).init(), 'kernels') is None
    table = ss.solvers.solver_table()
    assert len(table) == len(ss.solvers.rules) and set(table.compartments) == {'yes', 'no', 'if asked'}
    with pytest.raises(ss.CapabilityError, match='does not define count'):
        object.__new__(ss.solvers.Engine).count([])
    return


@sc.timer()
def test_given_modules():
    sc.heading('Testing modules given to a sim: copied, or not (copy_modules=False), and sim.<name> while initializing')

    class Status(ss.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.IntArr('status', default=0))
        def step(self):
            pass

    class Reader(ss.Module):
        def init_post(self):
            super().init_post()
            self.other = self.sim.status # Another module, by name, while the sim is initializing
        def step(self):
            pass

    mod = Status(name='status')
    sim = ss.Sim(modules=[mod, Reader(name='reader')], n_agents=10, start=2000, stop=2002, verbose=0).run()
    assert sim.reader.other is sim.status
    with pytest.raises(ValueError, match='copy_modules'): # The one given is not the one that ran
        (mod.status == 0).uids

    mod = Status(name='status')
    sim = ss.Sim(modules=mod, n_agents=10, start=2000, stop=2002, verbose=0, copy_modules=False).run()
    assert sim.status is mod and len((mod.status == 0).uids) == 10
    with pytest.raises(ValueError, match='copy_modules'):
        sim.new()
    with pytest.raises(ValueError, match='copy_modules'):
        ss.Sim(modules=mod, copy_modules=False)

    # The model's modules are sorted into their kinds when the sim is made, after those given by keyword, as they are once it is initialized
    sim = ss.Sim(ss.SIR(), [ss.RandomNet(), Status(name='status')], diseases=ss.SIS(), n_agents=10, verbose=0)
    assert [d.name for d in sim.pars.diseases] == ['sis', 'sir'] and len(sim.pars.networks) == 1 and len(sim.pars.modules) == 1 # Of no standard kind: in sim.custom once initialized
    assert list(sim.init().diseases.keys()) == ['sis', 'sir']

    # Whole days are ints, as for pandas
    days = (ss.date('2015-01-01') - ss.date('2010-01-01')).days
    assert days == 1826 and isinstance(days, int)

    # A death requested after the step's deaths were resolved warns
    class Late(ss.Module):
        def finish_step(self):
            super().finish_step()
            if self.ti in [1, 2]:
                self.sim.people.request_death(ss.uids([self.ti]), cause='late')
        def step(self):
            pass
    with pytest.warns(RuntimeWarning, match='next step'):
        ss.Sim(n_agents=10, modules=Late, dt=ss.days(1), start='2010-01-01', stop='2010-01-05', verbose=0).run()
    return sim


@sc.timer()
def test_printed_output():
    sc.heading('Testing what a run prints and describes: in proportions, no notes on correct models, and plain errors')
    S, I, R = ss.states('S I R')
    kw = dict(unit='day', dur=ss.days(70), verbose=0)

    # In proportions (total_pop=1), shares and not counts, and no note about chance
    sim = ss.Sim(ss.Disease(S >> I @ 1.4*I, I >> R @ ss.perday(0.14), init=dict(I=1e-6), name='sir'), total_pop=1, **kw).run()
    assert 'in proportions' in sim._start_lines()[0] and 'at most 0.66' in sim._end_line()
    assert not any('chance' in note for note in sim.notes) and 'in proportions' in sim.describe()

    # No notes on models written as the docs write them: density dependence, a chain binomial in whole steps, and typed rates
    sims = [ss.Sim(ss.Disease(S >> I @ ss.perday(0.1)*I.n, I >> R @ ss.perday(1), init_n=dict(I=1)), total_pop=20, solver='gillespie', **kw),
            ss.Sim(ss.Disease(S >> I @ ss.perday(0.2)*I, I >> R @ ss.perday(0.1), init_n=dict(S=1000, I=10)), total_pop=1010, solver=ss.solvers.binomial(substeps=1), **kw),
            ss.Sim(ss.Disease(ss.birth(S, rate=ss.peryear(1.4)), ss.death(rate=ss.peryear(1.05)), init_n=dict(S=100)), total_pop=100, unit='year', stop=1, verbose=0)]
    for sim in sims:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert not sim.init().notes, sim.notes
    A = ss.State(name='A')
    ss.Sim(ss.Disease(A, ss.death(rate=ss.perday(0.05)*A.n/150), init_n=dict(A=100)), total_pop=100, unit='day', stop=10, verbose=0).run() # The deaths of everyone at a rate on counts (EMULSION's density dependence)
    meant =ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I.n, I >> R @ ss.perday(0.1), init_n=dict(I=10)), total_pop=10_000, **kw).init() # But a beta meant for a share, with a count
    assert any('density-dependent' in note and '.frac' in note for note in meant.notes)

    # Where the rest start is not guessed when the initial values account for everyone; a count is not people; values by group are rounded
    sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.weeks(10), init_n=dict(S=9999, I=1)), total_pop=10_000, unit='week', stop=10, verbose=0).init()
    assert 'guessed: R' not in sim.describe()
    Sv, Iv = ss.states('Sv Iv', people=False)
    d = ss.Disease(S >> I @ ss.perday(0.1)*Iv.frac, Sv >> Iv @ ss.perday(0.1)*I.frac, Iv >> Sv @ ss.days(10), ss.unborn >> S @ ss.by(risk=[2/3, 1/3]), init_n=dict(I=1, Sv=100), name='d')
    d.stratify(risk=dict(low=0.1, high=0.9))
    text = ss.Sim(d, total_pop=1000, solver='ode', **kw).init().describe()
    assert 'Sv = 100' in text and 'Sv = 100 people' not in text and 'by(risk=[0.666667, 0.333333])' in text

    # After a run, why the kernels did not take it in one call is still the real reason
    class Counter(ss.Intervention):
        def step(self):
            self.n = getattr(self, 'n', 0) + 1
    sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.01)), interventions=Counter(), total_pop=1000, **kw)
    assert 'a step of its own' in sim.init()._kernel_reason() and 'a step of its own' in sim.run()._kernel_reason()

    # Numbers by group that disagree with the split say so
    d = ss.Disease(S >> I @ ss.perweek(0.1)*I, init_n=dict(S=ss.by(risk=[999, 9000]), I=ss.by(risk=[1, 0])), name='d')
    d.stratify(risk=dict(low=0.5, high=0.5))
    with pytest.raises(ValueError, match='puts 9,000 people at risk=high, but the split of the stratification puts 5,000 there'):
        ss.Sim(d, total_pop=10_000, solver='ode', **kw).init()
    return sim


@sc.timer()
def test_small_fixes():
    sc.heading('Testing small fixes: rel_trans adjusted for a count of infectiousness, until=False, a module named after a method of the results, and times from whole steps')
    S, I = ss.states('S I') # A count of infectiousness (I.n_infectious) with rel_trans adjusted by level (a garbled error)
    SV, IV = ss.State(people=False), ss.State(people=False)
    d = ss.Disease(infection=S >> I @ 0*I.n, bites=SV >> IV @ ss.perday(0.01)*I.n_infectious/(S + I).n, init_n=dict(I=100, SV=1000), name='d')
    d.stratify(species=dict(a=0.5, b=0.5))
    res = ss.Sim(d, connectors=ss.modify(d.rel_trans, multiply=ss.by(species=[2.0, 2.0])), total_pop=1000, unit='day', stop=10, verbose=0).run().results.d
    assert np.isclose(res.IV[-1], 1000*(1 - np.exp(-0.02*0.1*10))) # Twice the infectiousness
    S, I, R = ss.states('S I R')
    sim = ss.Sim(ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init=dict(I=0.01)), total_pop=1000, dur=ss.days(20), verbose=0)
    with pytest.raises(TypeError, match='neither a time nor a condition'): # It ran one step and gave zeros
        sim.run(until=False)
    with pytest.raises(ValueError, match='ss.modify'): # A state's rel_trans by group (it crashed the ODE); ss.modify(d.rel_trans, multiply=ss.by(...)) gives one
        ss.State('I', rel_trans=ss.by(risk=[2.0, 0.5]))
    class Walk(ss.Connector):
        def step(self): pass
    with pytest.raises(ValueError, match='method of the results'): # sim.results.walk was the method
        ss.Sim(n_agents=100, diseases='sis', connectors=Walk(name='walk'), verbose=0).init()
    timevec = ss.Sim(n_agents=10, start=0, stop=2500, dt=ss.years(0.25), unit='year', verbose=0).init().timevec
    assert timevec[366] == 91.5 and timevec[906] == 226.5 # They were 91.50000004 and 226.50000009
    I0 = lambda ip: ss.Sim(diseases=ss.SIR(init_prev=ip), n_agents=1000, stop=2001, verbose=0, rand_seed=1).run().results.sir.I[0]
    assert I0(ss.bernoulli(p=lambda sim: 0.1)) == I0(lambda sim: 0.1) and I0(ss.bernoulli(0.1)) == I0(0.1) # A declared disease raised a TypeError
    S, I, R = ss.states('S I R')
    trs = dict(death=ss.death(rate=ss.peryear(0.01)), die=I >> ss.dead @ 0.1, split=ss.Transition(I, {R: 0.9, ss.dead: None}, dur=5), leave=ss.emigration(rate=ss.peryear(0.1)), recover=I >> R @ 0.1)
    assert [(tr.dies, tr.may_die) for tr in trs.values()] == [(True, False), (True, False), (False, True), (False, False), (False, False)]
    res = ss.Sim(demographics=[ss.emigration(rate=ss.peryear(0.1))], diseases='sir', total_pop=1000, stop=2005, solver='ode', verbose=0).run().results
    assert np.isclose(res.emigrated[:-1].sum(), 1000 - res.alive[-1]) # Up to the last stock, at the start of the last step; a compartmental run counted no one as having left
    rerun = ss.Sim(diseases='sir', networks='random', n_agents=500, stop=2005, verbose=0).run()
    rerun.restart(pars=dict(beta=0))
    assert not rerun.started
    rerun.run(until=1955) # Its first step (it starts 50 years before the stop)
    assert rerun.started and rerun.ti == 0 # One step taken: sim.ti + sim.started
    assert rerun.run().results.sir.infection.sum() == 0
    return sim

@sc.timer()
def test_solver_options():
    """ A solver's options in its helper (ss.solvers.ode(), ss.solvers.binomial(), ss.solvers.abm()), the old spellings refused, settings for agents refused by compartments, and pars= holding only the model's parameters """
    sc.heading('Testing solver options, and settings given to the wrong solver')
    S, I, R = ss.states(3)
    make = lambda: ss.Disease(S >> I @ ss.par(0.3, name='beta')*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    kw = dict(total_pop=1000, unit='day', stop=30, verbose=0)
    euler = ss.Sim(make(), solver=ss.solvers.ode(integrator='euler', substeps=2), **kw).run()
    assert isinstance(euler.solver, ss.solvers.ode) and euler.solver.integrator == 'euler' and euler.solver.substeps == 2 and euler.solver == euler.pars.solver # What ran, with its options
    assert ss.Sim(make(), solver='ode', **kw).init().solver == ss.solvers.ode() != euler.solver
    assert ss.Sim(make(), solver='gillespie', **kw).init().solver == ss.solvers.gillespie() and ss.solvers.ode() != 'ode' # A solver is always a solver after init, not its name
    assert repr(euler.solver) == "ss.solvers.ode(integrator='euler', substeps=2)" and repr(ss.solvers.ode()) == 'ss.solvers.ode()' and str(euler.solver) == 'ode'
    assert sc.loadobj(sc.saveobj(None, euler.solver)) == euler.solver # Pickled with its options
    assert ss.Sim(make(), solver=ss.solvers.binomial(substeps=1), **kw).init().solver.substeps == 1
    for old, new in [({'integrator': 'euler'}, 'ss.solvers.ode'), ({'substeps': 1}, 'ss.solvers.binomial'), ({'rescale_threshold': 0.1}, 'rescale=dict'), ({'pop_scale': 10}, 'total_pop')]: # The old keywords
        with pytest.raises(TypeError, match=new.replace('.', r'\.')):
            ss.Sim(make(), **old, **kw)
    for old in ['ide', 'abm-events', 'abm-wholestep', 'abm-midstep']:
        with pytest.raises(ValueError, match='replaced'):
            ss.Sim(make(), solver=old, **kw).init()
    with pytest.raises(TypeError): # An option of another solver
        ss.solvers.binomial(integrator='euler')
    with pytest.raises(ValueError):
        ss.solvers.abm(step='event')
    class Whole(ss.SIR): # A class can suggest how it runs as agents, which a sim's step and a module's own solver override
        suggested_solver = ss.solvers.abm(step='wholestep')
    steps = lambda dis, solver: ss.solvers.agent_solver((s := ss.Sim(dis, networks='random', n_agents=200, solver=solver, unit='day', stop=5, verbose=0).init()).diseases[0], s, why=True)
    assert steps(Whole(), 'abm') == ('wholestep', 'suggested') and steps(Whole(), ss.solvers.abm(step='midstep')) == ('midstep', 'sim')
    assert steps(Whole(solver=ss.solvers.abm(step='events')), 'abm') == ('events', 'own') and steps(ss.SIR(), 'abm') == ('midstep', 'default')
    Whole.suggested_solver = 'abm-step'
    with pytest.raises(ValueError, match='suggested_solver'):
        steps(Whole(), 'abm')
    for key, val in [('rescale', True), ('even_draws', True)]: # For agents only
        with pytest.raises(ValueError, match='for agents'):
            ss.Sim(make(), solver='ode', **{key:val}, **kw).init()
    with pytest.raises(ValueError, match='total_pop'): # The number of agents is not taken as the number of people
        ss.Sim(make(), solver='ode', n_agents=1000, unit='day', stop=30, verbose=0).init()
    with pytest.raises(TypeError, match=r'pars=dict\(beta'): # A parameter of the model as a keyword
        ss.Sim(make(), beta=0.2, **kw)
    assert ss.Sim(pars=dict(diseases='sir', n_agents=100)).pars.n_agents == 100 # The sim's modules and settings in pars=, as in Starsim 3
    agents = ss.Sim(make(), solver='abm', n_agents=500, total_pop=5000, unit='day', stop=5, verbose=0).init()
    assert agents.pop_scale == 10 and not hasattr(agents.pars, 'pop_scale') and ss.Sim(make(), solver='ode', **kw).init().pop_scale == 1
    return euler


@sc.timer()
def test_agent_settings():
    sc.heading('Settings for agents only: dropped for what is worked out as an ODE, and a module\'s own solver chooses agents')
    for kw in [dict(rescale=True), dict(even_draws=True)]: # R0() and Rt() run the same model as an ODE
        sim = ss.Sim(diseases=ss.SIR(beta=0.1), networks='random', n_agents=500, total_pop=1e4, stop=10, verbose=0, **kw).run()
        assert sim.diseases.sir.R0() > 0 and len(sim.diseases.sir.Rt())
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, solver=ss.solvers.abm(step='wholestep'), name='sir')
    sim = ss.Sim(sir, unit='day', stop=5, verbose=0).init()
    assert isinstance(sim.solver, ss.solvers.abm) and 'solver of its own' in sim.solver_reason
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    caps = test_capabilities_two_modules()
    layout = test_one_layout()
    sim0 = test_demo(do_plot=do_plot)
    sim1 = test_default(do_plot=do_plot)
    sim2 = test_simple(do_plot=do_plot)
    sim3 = test_api()
    sim4 = test_complex_api()
    sim6 = test_shared_product(do_plot=do_plot)
    sim7 = test_components(do_plot=do_plot)
    sim8 = test_creation_syntax()
    sim9 = test_save()
    sim10 = test_save_mixed_dt()
    sim10b = test_shrink_declared()
    sim11 = test_rescale()
    sim11b = test_solver_options()
    sim12 = test_add_module()
    sim13 = test_locked_pars()
    update = test_update_pars()
    test_copied_modules()
    test_init_order()
    test_step_order()
    test_printed_output()
    test_small_fixes()
    test_agent_settings()

    T.toc()

    if do_plot:
        plt.show()
