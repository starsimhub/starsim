"""
Test the sim: making it, its modules and their names, settings, saving, and the loop that runs it
"""
import gc
import weakref
import pytest
import warnings
import numpy as np
import starsim as ss
import starsim.library as ssl
import sciris as sc
import matplotlib.pyplot as plt

n_agents = 1_000
sc.options(interactive=False) # Assume not running interactively


#%% Making a sim

def test_api():
    """ The ways of making the same sim give the same sim, with simple or complex inputs """
    s1 = ss.demo(plot=False)
    s2 = ss.Sim(diseases=dict(type='sir', beta=0.03), networks='random').run() # Strings, as a dict of parameters
    s3 = ss.Sim(dict(diseases=ss.SIR(beta=ss.perday(0.03)), networks=ss.RandomNet())).run() # Objects (a module's name seeds its random numbers, so the network is not held in a variable, which would name it)
    assert ss.debugtools.check_sims_match(s1, s2, s3), 'Sims should match'
    kw = dict(n_agents=n_agents, networks='random')
    s4 = ss.Sim(diseases=dict(type='sir', dur_inf=10), **kw).run() # A scalar duration is in the model's unit, days here
    s5 = ss.Sim(diseases=dict(type='sir', dur_inf=ss.days(10)), **kw).run()
    assert ss.debugtools.check_sims_match(s4, s5), 'Sims should match'
    sim = ss.Sim(dict(diseases='sir', networks='random', n_agents=100), diseases=ss.SIS(), verbose=0).init() # A keyword wins over a dict of parameters
    assert list(sim.diseases.keys()) == ['sis']

    def jump_age(sim):
        if sim.ti == 20:
            sim.people.age[:] = sim.people.age[:] + 1000
    pars1 = dict(
        n_agents = 1000,
        verbose = 0,
        stop = 2020,
        dt = ss.months(1),
        networks = [ss.RandomNet(name='random1', n_contacts=6), dict(type='random', name='random2', n_contacts=4)],
        diseases = [dict(type='sir', dur_inf=ss.expon(mean=ss.years(6.0))), dict(type='sis', beta=ss.prob(0.07, 'year'), init_prev=0.1)],
        demographics = [ss.Births(birth_rate=ss.peryear(0.02)), dict(type='deaths', death_rate=ss.peryear(0.02))],
        interventions = jump_age,
    )
    s1 = ss.Sim(pars1).run()
    s2 = ss.Sim(ss.SimPars(n_agents=1000, verbose=0, stop=2020, dt=ss.months(1)),
        networks = ss.ndict(ss.RandomNet(name='random1', n_contacts=6), ss.RandomNet(name='random2', n_contacts=4)),
        diseases = ss.ndict(ss.SIR(dur_inf=ss.expon(mean=ss.years(6.0))), ss.SIS(beta=ss.prob(0.07, 'year'), init_prev=0.1)),
        demographics = ss.ndict(ss.Births(birth_rate=ss.peryear(0.02)), ss.Deaths(death_rate=ss.peryear(0.02))),
        interventions = ss.ndict(ss.Intervention.from_func(jump_age)),
    ).run()
    assert ss.debugtools.check_sims_match(s1, s2), 'Sims should match'

    sir = ss.SIR()
    s4 = ss.Sim(sir, networks='random', pars=dict(start=2000, dur=2, n_agents=200, verbose=0, beta=0.2), n_agents=300) # Settings and the model's parameters in pars=, told apart by name; a keyword wins
    assert s4.pars.n_agents == 300 and s4.pars.dur == 2 and s4._par_values == dict(beta=0.2)
    assert ss.Sim(sir, beta=0.1)._par_values == dict(beta=0.1) # A parameter of the model as a keyword, as in pars=
    assert len(ss.Sim(n_agents=100, networks=[['random', 'sexual'], 'prenatal'], demographics='pregnancy', verbose=0).init().networks) == 3 # One level of nested lists
    return s1


def test_creation_syntax():
    """ Modules given as arguments are sorted by kind, in the order given; transitions given to the sim are one disease """
    class FirstClass(ss.Module):
        def step(self):
            self.sim.first = True

    class SecondClass(ss.Module):
        def step(self):
            assert self.sim.first # Will error if FirstClass is not run first

    mod1, mod2, dem, net, dis = FirstClass(), SecondClass(), ss.Births(), ss.RandomNet(), ss.SIS()
    pars = dict(n_agents=100, dur=ss.years(10))
    sim = ss.Sim([mod1, dem], **pars, demographics='deaths', connectors=mod2, diseases=dis, networks=net).run()
    assert isinstance(sim.custom[0], FirstClass) and isinstance(sim.connectors[0], SecondClass)
    assert len(sim.demographics) == 2 and all(isinstance(mod, ss.Demographics) for mod in sim.demographics())
    sim2 = ss.Sim([dis, mod1, net, mod2, dem], **pars).run() # The order intuited
    assert isinstance(sim2.custom[1], SecondClass) and isinstance(sim2.demographics[0], ss.Demographics) and isinstance(sim2.networks[0], ss.Network) and isinstance(sim2.diseases[0], ss.Disease)
    with pytest.raises(TypeError):
        ss.Sim(diseases=ss.RandomNet()).init() # A module of another type

    S, I, R = ss.states(3)
    infection = S >> I @ 0.3*I
    recovery = I >> R @ 0.1
    sims = [ss.Sim(infection, recovery), ss.Sim([infection, recovery]), ss.Sim(diseases=[infection, recovery])]
    for sim in sims:
        sim.run(verbose=0)
        assert list(sim.diseases.keys()) == ['sir'] and sim.sir is sim.diseases.sir # A shortcut to each module
        assert sim.results.I is sim.results.sir.I and sim.results.infection is sim.results.sir.infection
    assert np.array_equal(sims[0].results.I, sims[2].results.I)
    return sim2


def test_new():
    """ sim.new() builds the sim again, with what was set before init(), for a coded module or a subclass too; what init() took is kept """
    agentsim = ss.Sim(n_agents=n_agents, diseases=ssl.AgentSIS(name='sis'), networks='random', start=2000, stop=2010, verbose=0)
    again = agentsim.new(rand_seed=2).run()
    assert again.summarize().value.sis.infection == ss.Sim(n_agents=n_agents, diseases=ssl.AgentSIS(name='sis'), networks='random', start=2000, stop=2010, rand_seed=2, verbose=0).run().summarize().value.sis.infection

    class MySim(ss.Sim): # Adds its modules after Sim.__init__(), as FPsim's does
        def __init__(self, beta=0.1, **kwargs):
            super().__init__(**kwargs)
            self.pars.diseases = ss.SIS(beta=ss.peryear(beta), dur_inf=ss.years(10), name='sis')
            self.pars.networks = 'random'
    mysim = MySim(beta=0.2, n_agents=n_agents, start=2000, stop=2010, verbose=0)
    mysim.pars.stop = 2005
    new = mysim.new()
    assert type(new) is MySim and new.pars.stop == 2005 and isinstance(new.pars.diseases, ss.SIS)

    class Values(ss.Module): # An edit in place before init() is kept; one after belongs to that run only
        def __init__(self):
            super().__init__(name='vals')
            self.define_pars(weights=np.ones(2000))
    valsim = ss.Sim(Values(), stop=2, verbose=0)
    valsim.vals.pars.weights[1000] = 7
    valsim.init().vals.pars.weights[1000] = 9
    assert valsim.new().vals.pars.weights[1000] == 7

    S, I, R = ss.states('S I R') # The values and seed of restart() are kept by running again and by new()
    sir = ss.Disease(infection=S >> I @ ss.par(0.3, bounds=[0.1, 0.6], name='beta')*I, recovery=I >> R @ 0.1, init_n=dict(I=10), name='sir')
    make = lambda: ss.Sim(sir, unit='day', stop=60, total_pop=1000, verbose=0, solver='binomial')
    ref = make().new(pars=dict(beta=0.5), rand_seed=3).run().results.sir.I[-1]
    sim = make().init()
    assert sim.restart(pars=dict(beta=0.5), rand_seed=3).run().results.sir.I[-1] == ref
    assert sim.run().results.sir.I[-1] == ref and sim.new().run().results.sir.I[-1] == ref
    return new


def test_add_module():
    """ Adding a module to a sim that has been partly run """
    kw = dict(n_agents=n_agents, diseases='sis', networks='random', start='2000-01-01')
    base = ss.Sim(**kw).run()
    sim = ss.Sim(**kw)
    sim.run(until='2000-02-01')
    ti = sim.ti
    vx = sim.add_module(ss.modify('sis.pars.beta', multiply=0.2, name='masks'))
    with pytest.raises(ss.CapabilityError):
        sim.add_module(ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5)) # A new product adds a dimension (has_vaccine), which is laid out at init()
    with pytest.raises(ValueError, match='already exists in diseases'): # The kind of the module whose name it is
        sim.add_module(ss.Analyzer(name='sis'))
    sim.run()
    assert sim.connectors.masks is vx and vx.ti == sim.ti # It was added, and kept in step with the sim
    assert np.array_equal(sim.results.sis.I[:ti], base.results.sis.I[:ti]) and sim.results.sis.I[-1] < base.results.sis.I[-1]
    with pytest.raises(ss.AlreadyRunError):
        sim.add_module(ss.Intervention())
    return sim


def test_pars():
    """ Parameters that don't exist can't be set, nor the run's settings; a single value for a distribution sets its main parameter, as in Starsim 3 """
    sim = ss.Sim(n_agents=n_agents, diseases='sis', networks='random').init()
    with pytest.raises(sc.KeyNotFoundError):
        sim.diseases.sis.pars.bta = 0.1
    sim.diseases.sis.beta = 0.2 # Directly
    assert sim.diseases.sis.pars.beta.value == 0.2 and ss.SIS(beta=0.3).beta.value == 0.3 # Each module has its own, not the class's
    with pytest.raises(RuntimeError):
        sim.pars.rand_seed = 2 # The sim's settings are its run's: sim.new(rand_seed=2)

    with pytest.warns(RuntimeWarning): # Keeps the std, but says so
        sir = ssl.AgentSIR(init_prev=0.05, dur_inf=6, p_death=0)
    assert isinstance(sir.pars.init_prev, ss.bernoulli) and sir.pars.init_prev.pars.p == 0.05
    assert isinstance(sir.pars.dur_inf, ss.lognormal) and sir.pars.dur_inf.pars.mean == ss.years(6) # In the unit of the default
    assert ss.SexualNet(dur=ss.days(15)).pars.dur == ss.days(15) and isinstance(ss.RandomNet(n_contacts=dict(type='poisson', mean=4)).pars.n_contacts, ss.poisson)
    sir.pars.dur_inf.set(mean=8) # Keeps the standard deviation, and the unit
    assert sir.pars.dur_inf.pars.mean == ss.years(8)

    sim = ss.Sim(n_agents=100, start=2000, stop=2002, verbose=0) # A sim parameter set before init() is kept, as if given
    sim.pars.aging = True
    age0 = sim.init().people.age.raw[:100].copy()
    assert np.allclose(sim.run().people.age.raw[:100] - age0, 3)
    return


def test_save(tmp_path):
    """ Save, export, and shrink, also with a declared disease and a treatment that cures by transition """
    sim = ss.Sim(n_agents=n_agents, diseases=ss.SIS(dur_inf=ss.lognormal(mean=ss.years(10))), networks='random').run()
    sim.save(filename=tmp_path/'test.sim')
    sim.to_json(filename=tmp_path/'test.json')
    assert sim.summarize().equals(sc.load(tmp_path/'test.sim').summarize())
    assert sim.to_json(strkeys=True) == sc.loadjson(tmp_path/'test.json') # Keys as strings, as loaded
    sim.shrink()
    assert isinstance(sim.diseases.sis.pars.dur_inf.dist, ss.utils.Shrunk)

    S, L, I, R, C = ss.states('S L I R C')
    tb = ss.Disease(S, L, I, R, C, infection=S >> L @ ss.peryear(2)*I, activation=L >> I @ ss.peryear(0.3), recovery=I >> R @ ss.years(1), clear=ss.Transition(L, C, rate=0), name='tb', init=dict(L=0.2, I=0.02))
    tx = ss.treat(ss.Treatment(cure={tb.clear: 0.7}), who=tb.L, when=ss.schedule(start=2005), rate=ss.peryear(0.5), name='tpt')
    sim = ss.Sim(tb, interventions=tx, n_agents=2000, start=2000, stop=2020, dt=ss.months(1), verbose=0).run()
    sim.shrink(size_limit=0.3) # Raises if a module is still big
    assert sim.results.tb.I.values.sum() > 0 and sim.results.tpt.delivered.values.sum() > 0
    with pytest.raises(RuntimeError):
        sim.tb.current_state()
    return sim


def test_rescale():
    """ Dynamic rescaling of the agents: about the same epidemic; people no agent stands for yet are in the first state; an agent's death is scale people """
    def make_sim(**kwargs):
        sir = ss.SIR(beta=ss.perday(0.025), dur_inf=ss.days(10), init_n=dict(I=50), p_death=0)
        return ss.Sim(n_agents=5e3, total_pop=5e4, diseases=sir, networks=ss.RandomNet(n_contacts=10), dt=ss.days(1), dur=ss.days(150), verbose=0, **kwargs)
    fixed = make_sim().run()
    rescaled = make_sim(solver=ss.solvers.abm(rescale=True)).run()
    scale = rescaled.results.pop_scale
    assert scale[0] == 1 and scale[-1] == 10 == rescaled.pop_scale
    assert abs(rescaled.results.sir.infection.cum[-1]/fixed.results.sir.infection.cum[-1] - 1) < 0.1

    S, I, R = ss.states('S I R')
    flu = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_n=dict(I=10), prevalence=I/(S+I+R), name='flu')
    res = ss.Sim(flu, total_pop=1e6, n_agents=2000, dur=ss.days(60), solver=ss.solvers.abm(rescale=True), verbose=0).run().results
    assert res.flu.I[0] == 10 # A number of people is a number of agents at the start scale
    assert np.allclose(res.flu.S + res.flu.I + res.flu.R, 1e6) and np.allclose(res.alive, 1e6) and np.isclose(res.flu.prevalence[0], 1e-5)
    deadly = ss.SIR(beta=ss.perday(0.3), dur_inf=ss.days(10), p_death=0.5, init_prev=0.002)
    res = ss.Sim(diseases=deadly, networks='random', n_agents=2000, total_pop=2e6, solver=ss.solvers.abm(rescale=True), stop=40, unit='day', verbose=0).run().results
    assert res.death.cum[-1] > 0 and np.allclose(res.alive, 2e6 - (res.death.cum - res.death)) and np.allclose(res.sir.S + res.sir.I + res.sir.R, res.alive)
    return rescaled


#%% Modules in the sim

def test_module_order():
    """ Custom modules are initialized after the modules they may read; modules step by kind, unless moved with sim.loop.insert() """
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
    sim = ss.Sim(Lifestyle(name='life'), n_agents=1000, demographics=Demography(), start=2010, stop=2011, verbose=0).init()
    assert sim.custom.life.unset == 0

    seen = {} # Whatever reads the disease in init_post() sees who starts infected
    class Reader(ss.Connector):
        def init_post(self):
            super().init_post()
            seen[self.name] = self.sim.diseases.sir.I.n
        def step(self):
            pass
    class Custom(ss.Module):
        def init_post(self):
            super().init_post()
            seen[self.name] = self.sim.diseases.sir.I.n
    ss.Sim(Custom(name='cust'), n_agents=1000, diseases=ss.SIR(init_n=dict(I=10), name='sir'), connectors=Reader(name='conn'), stop=1, verbose=0).init()
    assert seen == dict(conn=10, cust=10)

    log = []
    class Mark(ss.Module):
        def step(self):
            log.append(self.name)
    class Net(ss.RandomNet):
        def step(self):
            log.append(self.name)
            return super().step()
    make = lambda: ss.Sim([Mark(name='first'), Net(name='net'), ss.SIR(name='sir'), Mark(name='last')], n_agents=200, stop=2001, verbose=0)
    make().run()
    assert log[:3] == ['first', 'last', 'net'] # Modules of no standard kind first, then the networks
    log.clear()
    sim = make().init()
    sim.loop.insert(sim.networks.net, label='first.step') # Moved to just after the first module's step
    sim.run()
    assert log[:3] == ['first', 'net', 'last'] and len(sim.custom) == 2 and len(sim.networks) == 1 # Still found by kind
    sim = ss.Sim(Mark(name='a'), custom=[Mark(name='b'), ss.SIR()], n_agents=100, verbose=0).init() # custom= keeps them where they are given
    assert list(sim.custom.keys()) == ['a', 'b', 'sir'] and not len(sim.diseases)
    return sim


def test_given_modules():
    """ Modules given to a sim are copied, or not (copy_inputs=False), and sorted by kind when the sim is made; sim.<name> while initializing """
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
    sim = ss.Sim([mod, Reader(name='reader')], n_agents=10, start=2000, stop=2002, verbose=0).run()
    assert sim.reader.other is sim.status
    with pytest.raises(ValueError): # The one given is not the one that ran
        (mod.status == 0).uids
    mod = Status(name='status')
    sim = ss.Sim(mod, n_agents=10, start=2000, stop=2002, verbose=0, copy_inputs=False).run()
    assert sim.status is mod and len((mod.status == 0).uids) == 10

    sim = ss.Sim(ss.SIR(), [ss.RandomNet(), Status(name='status')], diseases=ss.SIS(), n_agents=10, verbose=0) # After those given by keyword
    assert [d.name for d in sim.pars.diseases] == ['sis', 'sir'] and len(sim.pars.networks) == 1 and len(sim.pars.modules) == 1
    assert list(sim.init().diseases.keys()) == ['sis', 'sir']
    return sim


def test_module_names():
    """ One rule for every name: the variable, else the default for the class or kind; lookups and paths by the default name, if unambiguous """
    kw = dict(n_agents=200, start=2000, stop=2002, verbose=0)
    def make_sim():
        m = ss.SIR()
        net = ss.RandomNet()
        return ss.Sim(diseases=m, networks=net, analyzers=ss.infection_log(), **kw)
    sim = make_sim().init()
    assert list(sim.diseases.keys()) == ['m'] and list(sim.networks.keys()) == ['net'] and list(sim.analyzers.keys()) == ['infection_log']
    assert sim.diseases.sir is sim.diseases['sir'] is sim.sir is sim.diseases.m and 'sir' not in sim.diseases # By the default name, without changing what is stored
    assert sim.networks.randomnet is sim.networks.net and sim.results.sir is sim.results.m and sim.results['sir.I'] is sim.results.m.I
    a = ss.SIR() # An ambiguous default name
    b = ss.SIR()
    two = ss.Sim(diseases=[a, b], networks='random', **kw).init()
    with pytest.raises(AttributeError):
        two.sir
    assert list(ss.Sim(n_agents=100, diseases=['sis', 'sis'], networks='random').init().diseases.keys()) == ['sis', 'sis_1'] # Default names are made unique

    steps = [] # A variable that would name several modules (here, in a loop) names none of them; a name given (or a key) always wins
    for year in [2000, 2001]:
        step = ss.vaccinate(efficacy=0.9, coverage=0.5, when=year)
        steps.append(step)
    flu = ss.SIR(name='flu')
    sim = ss.Sim(diseases=flu, networks=dict(main=ss.RandomNet()), interventions=steps, **kw).init()
    assert list(sim.interventions.keys()) == ['vaccinate', 'vaccinate_1'] and list(sim.diseases.keys()) == ['flu'] and list(sim.networks.keys()) == ['main']
    S, I, R = ss.states(3) # An outflow is named as written
    dis = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(8), I >> ss.dead @ ss.perday(0.01), ss.unborn >> S @ ss.peryear(0.02))
    assert list(dis.graph.transitions) == ['infection', 'I_to_R', 'I_to_dead', 'unborn_to_S']

    kw = dict(n_agents=300, dur=5, verbose=0) # Paths by the default name, for an ss.SIR called measles
    def make(**kwargs):
        measles = ss.SIR(beta=0.1)
        return ss.Sim(diseases=measles, networks='random', **kw, **kwargs)
    sim = make(pars={'sir.beta':0.2}).init()
    assert sim.diseases.measles.pars.beta.raw == 0.2 and sim.pars.sir is sim.pars.measles
    assert ss.parallel(make(), pars={'sir.beta':[0.1, 0.2]}, workers=1, verbose=0).runs['sir.beta'].tolist() == [0.1, 0.2]
    assert make(connectors=ss.modify('sir.pars.beta', multiply=0.5, who='sir.I')).run().results.sir.infection.sum() > 0
    assert make(interventions=ss.vaccinate(efficacy=0.9, disease='sir', who='sir.S', coverage=0.5, when=2)).init().interventions[0].disease == 'measles'

    for name in ['module', 'sim', 'pars', 'results']: # A variable whose name the sim uses itself names nothing
        ns = dict(ss=ss, kw=kw)
        exec(f'{name} = ss.SIR(beta=0.1)\nout = ss.Sim(diseases={name}, networks="random", **kw).run()', ns)
        assert ns['out'].diseases[0].name == 'sir', name
    run = make().run()
    extra = {key for key in vars(run) if not key.startswith('_')} - ss.Sim._attrs - set(dir(ss.Sim)) - set(ss.SimPars()) - ss.People.taken_names()
    assert not extra, f'New attributes of the sim, which a module must not take from its variable: add {extra} to ss.Sim._attrs'
    class Tally(ss.Module): # sim.custom (those of no standard kind) and sim.modules (all of them)
        def step(self): pass
    sim = ss.Sim(Tally(), diseases='sis', networks='random', n_agents=200, verbose=0).init()
    assert len(sim.modules) == 3 and list(sim.custom.keys()) == ['tally'] and sim.custom.tally is sim.tally
    return sim


#%% Solvers and what a run says

def test_ode_map():
    """ The ODE's map integrator: each step whole, 1 - exp(-rate*dt) of each compartment leaving it, shared by rate, the rates as at the start of the step; compiled """
    S, I, R, D = ss.states(4)
    beta, gamma, mu, dt = 0.4, 0.1, 0.02, 2.0
    dis = ss.Disease(S >> I @ ss.perday(beta)*I.frac, I >> R @ ss.perday(gamma), I >> D @ ss.perday(mu), init=dict(S=0.99, I=0.01), name='x')
    sim = ss.Sim(dis, total_pop=1.0, unit='day', dt=ss.days(dt), stop=ss.days(60), verbose=0, solver=ss.solvers.ode(integrator='map'))
    sim.init()
    assert sim._compiled_reason() is None and sim.solver.substeps == 1
    sim.run()
    s, i, r, d = 0.99, 0.01, 0.0, 0.0
    expect = [s]
    for t in range(30): # The discrete map, by hand
        inf = s*-np.expm1(-beta*i*dt)
        left = i*-np.expm1(-(gamma + mu)*dt)
        s, i, r, d = s - inf, i + inf - left, r + left*gamma/(gamma + mu), d + left*mu/(gamma + mu)
        expect.append(s)
    assert np.allclose(sim.results.x.S.values, expect, rtol=1e-12, atol=0)
    assert np.isclose(sim.results.x.D.values[-1], d, rtol=1e-12)
    run = lambda step, n: ss.Sim(dis, total_pop=1.0, unit='day', dt=ss.days(step), stop=ss.days(60), verbose=0, solver=ss.solvers.ode(integrator='map', substeps=n)).run().results.x.S.values
    assert np.allclose(run(dt, 2), run(dt/2, 1)[::2], rtol=1e-12, atol=0) # With substeps, each substep is taken as the map (exponential Euler)
    ode = ss.Sim(dis, total_pop=1.0, unit='day', dt=ss.days(dt), stop=ss.days(60), verbose=0, solver='ode').run().results.x.S.values
    assert np.abs(run(dt, 50) - ode).max() < 0.1*np.abs(run(dt, 1) - ode).max() # More substeps approach the ODE
    return sim



def test_solver_options():
    """ A solver's options in its helper; a module's own solver chooses agents; the scale of agents """
    S, I, R = ss.states(3)
    make = lambda: ss.Disease(S >> I @ ss.par(0.3, name='beta')*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    kw = dict(total_pop=1000, unit='day', stop=30, verbose=0)
    euler = ss.Sim(make(), solver=ss.solvers.ode(integrator='euler', substeps=2), **kw).run()
    assert isinstance(euler.solver, ss.solvers.ode) and euler.solver.integrator == 'euler' and euler.solver.substeps == 2 and euler.solver == euler.pars.solver
    assert ss.Sim(make(), solver='ode', **kw).init().solver == ss.solvers.ode() != euler.solver # A solver after init, not its name
    with pytest.raises(TypeError): # An option of another solver
        ss.solvers.binomial(integrator='euler')
    for make_solver in [lambda: ss.solvers.gillespie(substeps=2), lambda: ss.solvers.tau(foo=1)]: # An option a solver has none of is not dropped
        with pytest.raises(TypeError):
            make_solver()
    assert ss.solvers.ode(integrator='map').substeps == 1 and ss.solvers.ode(integrator='map', substeps=3).substeps == 3 # The map takes each step whole unless given substeps
    named = ss.solvers.abm(step='midstep') # A step named as the default still overrides a module's suggested step, so it is not the same as abm()
    assert named != ss.solvers.abm() and repr(named) == "ss.solvers.abm(step='midstep')" and ss.solvers.abm(step='events') == ss.solvers.abm(step='events')
    agents = ss.Sim(make(), solver='abm', n_agents=500, total_pop=5000, unit='day', stop=5, verbose=0).init()
    assert agents.pop_scale == 10 and ss.Sim(make(), solver='ode', **kw).init().pop_scale == 1
    sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, solver=ss.solvers.abm(step='wholestep'), name='sir')
    assert isinstance(ss.Sim(sir, unit='day', stop=5, verbose=0).init().solver, ss.solvers.abm)
    sim = ss.Sim(diseases=ss.SIR(beta=0.1), networks='random', n_agents=500, total_pop=1e4, stop=10, verbose=0, solver=ss.solvers.abm(rescale=True)).run() # Settings for agents only are dropped for what R0() and Rt() work out as an ODE
    assert sim.diseases.sir.R0() > 0 and len(sim.diseases.sir.Rt())
    return euler


def test_engine_rules():
    """ A module of the user's takes the route the built-in ones do, also over several modules; people given to a compartmental solver; agents read from one """
    class Clinic(ss.Intervention):
        def agent_reasons(self, sim):
            return ['the clinic keeps a list of names']
    class Survey(ss.Analyzer):
        def agent_reasons(self, sim):
            return [ss.modules.Preference('the survey samples people')]
    kw = dict(diseases=ss.SIR(), stop=2002, verbose=0)
    sim = ss.Sim(interventions=Clinic(), **kw).init()
    assert isinstance(sim.solver, ss.solvers.abm) and sim.solver_reason == 'the clinic keeps a list of names'
    with pytest.raises(ss.CapabilityError):
        ss.Sim(interventions=Clinic(), solver='ode', **sc.dcp(kw)).init()
    assert isinstance(ss.Sim(analyzers=Survey(), **sc.dcp(kw)).init().solver, ss.solvers.abm)
    assert isinstance(ss.Sim(analyzers=Survey(), solver='ode', **sc.dcp(kw)).init().solver, ss.solvers.ode)
    class Level(ss.Module): # A plain module that keeps a value for each person needs agents
        def __init__(self):
            super().__init__()
            self.define_states(ss.FloatArr('level', default=0.0))
    assert isinstance(ss.Sim(connectors=Level(), **sc.dcp(kw)).init().solver, ss.solvers.abm)

    class A(ss.Disease): # A non-exponential dwell time, then a function as a rate
        S, I = ss.states('S I')
        recover = ss.Transition(I, S, dur=ss.lognormal(ss.years(1), ss.years(1)))
        catch = ss.Transition(S, I, rate=ss.peryear(0.1))
    class B(ss.Disease):
        X, Y = ss.states('X Y')
        default_state = 'X'
        go = ss.Transition(X, Y, rate=lambda sim: ss.peryear(0.1))
    sim = ss.Sim(diseases=[A(name='a'), B(name='b')], n_agents=100, start=2000, stop=2001, verbose=0).run()
    assert sim.results.b.go.sum() >= 0 and all(cap.reason for cap in sim.capabilities) # Only what changes the choice is recorded
    sim2 = ss.Sim(diseases=A(name='a'), people=ss.People(100), solver='ode', stop=2001, verbose=0).init()
    assert sim2.pop_from == 'people' # Where the population size came from, not a guess
    class Peek(ss.Analyzer): # Reading agents from a compartmental run raises
        def step(self):
            self.sim.diseases.a.I.uids
    with pytest.raises(ss.CapabilityError):
        ss.Sim(diseases=A(name='a'), analyzers=Peek(), solver='ode', stop=2001, verbose=0).run()
    return sim


def test_notes():
    """ No notes on models written as the docs write them, but on a beta meant for a share given with a count; a sim in proportions; why a run steps """
    S, I, R = ss.states('S I R')
    kw = dict(unit='day', dur=ss.days(70), verbose=0)
    sims = [ss.Sim(ss.Disease(S >> I @ ss.perday(0.1)*I.n, I >> R @ ss.perday(1), init_n=dict(I=1)), total_pop=20, solver='gillespie', **kw), # Density dependence
            ss.Sim(ss.Disease(S >> I @ ss.perday(0.2)*I, I >> R @ ss.perday(0.1), init_n=dict(S=1000, I=10)), total_pop=1010, solver=ss.solvers.binomial(substeps=1), **kw), # A chain binomial in whole steps
            ss.Sim(ss.Disease(ss.birth(S, rate=ss.peryear(1.4)), ss.death(rate=ss.peryear(1.05), rate_per=1000), init_n=dict(S=100)), total_pop=100, unit='year', stop=1, verbose=0)] # Typed rates
    for sim in sims:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert not sim.init().notes, sim.notes
    meant = ss.Sim(ss.Disease(S >> I @ ss.perday(0.3)*I.n, I >> R @ ss.perday(0.1), init_n=dict(I=10)), total_pop=10_000, **kw).init()
    assert any('density-dependent' in note for note in meant.notes)
    sim = ss.Sim(ss.Disease(S >> I @ 1.4*I, I >> R @ ss.perday(0.14), init=dict(I=1e-6), name='sir'), total_pop=1, **kw).run() # In proportions: shares, and no note about chance
    assert not any('chance' in note for note in sim.notes) and 'in proportions' in sim.describe()
    rerun = ss.Sim(diseases='sir', networks='random', n_agents=500, stop=2005, verbose=0).run() # A restarted run takes its first step as a run would
    rerun.restart(pars=dict(beta=0))
    rerun.run(until=1955) # Its first step (it starts 50 years before the stop)
    assert rerun.started and rerun.ti == 0 and rerun.run().results.sir.infection.sum() == 0
    return sim


#%% The loop

def test_loop():
    """ A manual run matches an automatic one, and a run split in two; every module sees the step's time index; insertions run once per step; profiling """
    pars = dict(dur=ss.years(5), dt=ss.weeks(1), n_agents=100, diseases='sis', networks='random', rand_seed=1, verbose=0)
    s1 = ss.Sim(pars).run()
    s2 = ss.Sim(pars).init()
    while s2.loop.index < len(s2.loop):
        s2.run_one_step()
    s2.finalize()
    s3 = ss.Sim(pars)
    s3.run(until=1)
    s3.run()
    assert s1.summarize().equals(s2.summarize()) and s1.summarize().equals(s3.summarize())

    seen = sc.obj(rate=[], intv=[])
    def rate(sim):
        seen.rate.append(sim.ti)
        return 0.1
    def intv(sim):
        seen.intv.append(sim.ti)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, rate=rate), init_prev=0.1, name='sir')
    sim = ss.Sim(sir, interventions=intv, solver='abm', n_agents=100, unit='day', stop=4, verbose=0).run()
    assert seen.intv == [0, 1, 2, 3, 4] and set(seen.rate) == {0, 1, 2, 3, 4} and sim.ti == 4

    calls = []
    sim = ss.Sim(pars).init()
    sim.loop.insert(lambda sim: calls.append(sim.ti), label='sim.finish_step', before=True)
    sim.run()
    assert len(calls) == sim.t.npts
    prof = ss.Sim(pars).perf_profile(kind='loop', plot=False) # As on the user guide page on profiling
    assert len(prof.sim.loop.cpu_time)
    prof.sim.loop.plot_cpu()
    plt.close('all')
    return sim


def test_memory_cleanup():
    """ Sims are collectable without shrink(), partly run, complete, or failed; caches of callable distributions are cleared and made again """
    def custom_duration(module, sim, uids):
        return 5 + sim.people.age[uids] * 0
    pars = dict(dur=ss.years(5), n_agents=100, diseases=ss.SIR(init_prev=0.2, dur_inf=ss.normal(mean=custom_duration, std=0.1)), networks='random', rand_seed=2)
    def collectable(make):
        ref = weakref.ref(make()) # Held nowhere else
        gc.collect()
        return ref() is None

    s1 = ss.Sim(pars).run()
    s2 = ss.Sim(pars)
    s2.run(until=1)
    assert all(getattr(dist, '_callable_args', None) is None for dist in s2.dists.dists.values())
    s2.run()
    assert s1.summarize().equals(s2.summarize()), 'Callable cache regeneration changed results'
    assert collectable(lambda: ss.Sim(pars).run(until=1)), 'Partially run sim was not collectable'
    assert collectable(lambda: ss.Sim(pars).run()), 'Completed sim was not collectable'

    sim = ss.Sim(pars).init()
    def fail_after_cache(sim): # The cleanup runs even when the loop raises
        dist = next(iter(sim.dists.dists.values()))
        dist._callable_args = {'sim': sim}
        raise RuntimeError('intentional loop failure')
    sim.loop.insert(fail_after_cache, label='sim.start_step')
    with pytest.raises(RuntimeError):
        sim.run()
    assert all(getattr(dist, '_callable_args', None) is None for dist in sim.dists.dists.values())
    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None, 'Exception-path sim was not collectable'
    return


def test_coded_alone():
    """ A module written as code runs with no agents, and the loop calls only what does something """
    class Decay(ss.Module): # An ODE of its own, with nothing per person
        def __init__(self):
            super().__init__()
            self.y = 1.0
        def step(self):
            self.y *= np.exp(-0.1)
    sim = ss.Sim(Decay(), stop=10, verbose=0).run()
    assert not isinstance(sim.solver, ss.solvers.abm) and len(sim.people.uid) == 1 and sim.people.n == sim.pars.total_pop == 10_000
    assert np.isclose(sim.custom.decay.y, np.exp(-1.1)) # A step from each of the 11 time points
    assert set(sim.loop.to_df().label) == {'sim.advance', 'sim.start_step', 'decay.step', 'sim.finish_step'}
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_api()
    test_creation_syntax()
    test_new()
    test_add_module()
    test_pars()
    test_rescale()
    test_module_order()
    test_given_modules()
    test_module_names()
    test_solver_options()
    test_engine_rules()
    test_notes()
    test_loop()
    test_memory_cleanup()
    test_coded_alone()
    T.toc()
