"""
Test networks and mixing pools
"""

# %% Imports and settings
import pytest
import warnings
from unittest import mock
import sciris as sc
import numpy as np
import networkx as nx
import starsim as ss
import starsim.library as ssl
from scipy.integrate import solve_ivp

sc.options.interactive = False # Assume not running interactively

small = 100
medium = 1000


def degree(nw, n):
    """ The number of edges of each agent """
    return np.bincount(np.concatenate([np.asarray(nw.p1), np.asarray(nw.p2)]), minlength=n)


def make_dhs_data(n=100, include_sexes=False):
    """ Synthetic household data, as from the DHS: the ages (and sexes, 1=male, 2=female) of the members of each household """
    ages, sexes = [], []
    for size in np.random.randint(1, 6, n):
        ages.append(sc.strjoin(np.random.randint(0, 80, size)))
        sexes.append(sc.strjoin(np.random.choice([1, 2], size)))
    data = sc.dataframe(hh_id=np.arange(n), ages=ages)
    if include_sexes:
        data['sexes'] = sexes
    return data


# %% Networks of agents

def test_random():
    sc.heading('Testing random networks')
    nw1 = ss.Sim(n_agents=small, networks=ss.RandomNet()).init().networks[0]
    nw2 = ss.Sim(n_agents=small, networks='random').init().networks[0]
    nw3 = ss.Sim(n_agents=small, networks=dict(type='random', n_contacts=20)).init().networks[0]
    assert np.array_equal(nw1.p2, nw2.p2), 'Implicit and explicit creation should give the same network'
    assert len(nw3) == len(nw2)*2, 'Doubling n_contacts should produce twice as many contacts'

    # The default (fast) method has the mean degree of method='exact', but a variable (Poisson-like) degree
    make = lambda **kw: ss.Sim(n_agents=medium, networks=ss.RandomNet(n_contacts=10, **kw), rand_seed=1).init().networks[0]
    exact, fast = make(method='exact'), make()
    deg = degree(fast, medium)
    assert len(exact) == len(fast) and np.isclose(deg.mean(), 10, rtol=0.05)
    assert degree(exact, medium).std() == 0 and deg.std() > 1
    assert np.array_equal(fast.p2, make().p2), 'RandomNet should be reproducible for a fixed seed'

    # With uniform targets, the variance in degree is about n_contacts/2 from the sources plus n_contacts/2 from the targets (as in Covasim v3)
    n = 10_000
    deg4 = degree(ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=ss.poisson(20), method='uniform'), rand_seed=1).init().networks[0], n)
    assert np.isclose(deg4.mean(), 20, rtol=0.05) and np.isclose(deg4.var(), 15, rtol=0.1), (deg4.mean(), deg4.var())


def test_randomsafe():
    sc.heading('Testing the RandomSafe network')
    pars = dict(n_agents=small, dur=ss.years(5), verbose=False)
    births = ss.Births(birth_rate=ss.peryear(0.005))
    for method, similar in [(None, False), ('safe', True)]: # With a few births, ordinary random networks diverge; safe ones don't
        net = lambda: ss.RandomNet(method=method) if method else ss.RandomNet()
        n1 = ss.Sim(pars, networks=net()).run().networks[0].to_edgelist()
        n2 = ss.Sim(pars, networks=net(), demographics=births).run().networks[0].to_edgelist()
        assert (sc.similarity(n1, n2) > 0.5) == similar, f'method={method}: similarity {sc.similarity(n1, n2):%}'


def test_library_networks():
    sc.heading('Testing the disk, null, and cluster networks')
    sim = ss.Sim(n_agents=medium, networks=ssl.networks.DiskNet(r=0.15, v=ss.per(0.05, unit=ss.years(1))), diseases='sir').run()
    assert len(sim.networks[0]) > 0 and sim.results.sir.infection.cum[-1] > 0, 'Expected edges, and infections through them'

    sim = ss.Sim(diseases=ss.SIR(dur_inf=10, beta=0.1), n_agents=small, networks=ssl.networks.NullNet()).run()
    net = sim.networks.nullnet
    assert len(net) == small and np.all(net.edges.weight == 0) and (net.edges.p1 == net.edges.p2).all(), 'Expected one zero-weight self-edge per agent'

    net = ssl.ClusterNet(cluster_size=ss.poisson(5), who=(ss.person.age >= 18) & (ss.person.age < 65))
    sim = ss.Sim(n_agents=medium, networks=net).init()
    net = sim.networks[0]
    ages = sim.people.age
    assert np.array_equal(net.cluster[net.p1], net.cluster[net.p2]) # Edges are only within clusters
    assert np.all(ages[net.members] >= 18) and np.all(ages[net.members] < 65) # Only within the age range
    sizes = np.bincount(net.cluster[net.cluster.notnan].astype(int))
    assert len(net) == (sizes*(sizes-1)/2).sum() # Clusters are fully connected


def test_household():
    sc.heading('Testing HouseholdNet...')
    household = ssl.networks.HouseholdNet(dhs_data=make_dhs_data(n=small, include_sexes=True), dynamic=False)
    sim = ss.Sim(n_agents=small, diseases='sis', networks=household).run()
    assert np.all(np.isfinite(sim.networks[0].household_ids[:]))

    # Dynamic: all delivered agents have household IDs. Agents born in the sim with age >= dt have been through at least one full step
    # after delivery; those with age < dt may be undelivered embryos whose age crossed zero only due to the end-of-step aging
    household = ssl.networks.HouseholdNet(dhs_data=make_dhs_data(n=medium), dynamic=True)
    sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=ss.Pregnancy()).run()
    ppl = sim.people
    delivered = ~ppl.parent.notnan | (ppl.age >= sim.t.dt_year)
    assert np.all(np.isfinite(sim.networks[0].household_ids[delivered.uids])), 'Some delivered agents are missing household IDs'


def test_hybrid():
    sc.heading('Testing hybrid networks')
    sis = ss.SIS(beta=dict(h=0.1, s=0.05, w=0.05, c=0.02)) # Per-network beta
    sim = ss.Sim(n_agents=medium, diseases=sis, networks=ssl.HybridNet(n_contacts=dict(c=10, s=0)), verbose=0).run()
    nets = sim.networks
    assert list(nets.keys()) == ['h', 's', 'w', 'c'] # Expanded into four networks
    assert nets.c.pars.n_contacts.pars.value == 10 and isinstance(nets.w, ssl.ClusterNet) # Partial update of n_contacts; clusters by default
    assert len(nets.s) == 0 and len(nets.w) > 0 # No contacts is an empty network

    # Covasim's transmissibility on each layer, as a disease's beta by network, with random schools and workplaces
    sim3 = ss.Sim(n_agents=small, diseases=ss.SIR(beta=ssl.HybridNet.betas()), networks=ssl.HybridNet(clustered=False, method='uniform'), unit='day', stop=30, verbose=0).run()
    assert sim3.results.sir.infection.sum() > 0 and np.all(sim3.people.age[sim3.networks.s.members] < 22) # Schools are age-restricted


def test_edge_ops():
    sc.heading('Testing network edge operations, a sexual network between men, and a maternal network')
    df = sc.dataframe(p1=[0, 1, 2, 3], p2=[1, 2, 3, 4])
    net = ss.Network(name='mynet').from_df(df) # Create from a dataframe
    assert net.name == 'mynet' and len(net) == 4 and net.weight.sum() == 4 # The weight defaults to 1
    net.from_df(dict(p1=[0, 1], p2=[1, 2], weight=[1, 2])) # Replace the edges with a dict of arrays
    assert net['weight'][1] == 2
    net.remove_edges(net.p1 == 0) # Remove edges by mask
    assert len(net) == 1 and net.p1[0] == 1
    net.remove_edges([0]) # ...or by index
    assert len(net) == 0
    assert 'foo' in ss.Network(p1=[0], p2=[1], edges=dict(foo=[1])).edges # Other columns go in edges=
    with pytest.raises(ValueError):
        ss.RandomNet(n_contact=5) # A typo is not an edge column

    sim = ss.Sim(diseases=dict(type='sis', beta=0.5), networks=ss.SexualNet(between='mm', participation=0.3)).run()
    msm = sim.networks[0]
    assert len(msm) and sim.people.male.raw[msm.edges.p1].all() and sim.people.male.raw[msm.edges.p2].all()

    sim = ss.Sim(n_agents=medium, demographics='pregnancy').init()
    nw = ss.PrenatalNet()
    nw.init_pre(sim)
    nw.add_pairs(mother_uids=[1, 2, 3], unborn_uids=[100, 101, 102])
    assert len(nw) == 3


def test_one_way():
    sc.heading('Testing a network that transmits one way (one_way=True)')
    reached = {}
    for one_way in [False, True]:
        net = ss.Network(p1=np.arange(99), p2=np.arange(1, 100), weight=np.ones(99), one_way=one_way, name='chain') # A chain, each person linked to the next
        sim = ss.Sim(n_agents=100, diseases=ss.SIR(beta=ss.perday(5), init_prev=0, dur_inf=ss.days(1000)), networks=net, dt=ss.days(1), dur=ss.days(20), verbose=0)
        sim.init()
        sim.diseases.sir.infect(ss.uids([50]))
        sim.run()
        reached[one_way] = np.flatnonzero(~sim.diseases.sir.S.values)
    assert reached[False].min() < 50 < reached[False].max() # Both ways from person 50
    assert reached[True].min() == 50 < reached[True].max() # Only from p1 to p2


def test_dynamic():
    sc.heading('Testing static (dynamic=False) networks')
    kw = dict(networks=ss.RandomNet(dynamic=False), n_agents=small, diseases='sis', stop=2002, verbose=0)
    sim = ss.Sim(**kw).init()
    p1 = sim.networks[0].p1.copy()
    sim.run()
    assert np.array_equal(p1, sim.networks[0].p1) # Static edges don't change
    sim = ss.Sim(demographics=ss.Births(birth_rate=ss.peryear(0.05)), **kw).run()
    assert sim.people.uid.max() >= small and sim.networks[0].members.max() < small # Agents born during the sim are not added


def test_bare_units():
    sc.heading("Testing that a bare number in a network's dur= is in the sim's unit, and a single duration a mean")
    sim = lambda net: ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.peract(0.05), dur_inf=ss.days(10)), networks=net, dt=ss.days(1), dur=ss.days(10), verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error') # Nothing to warn of: one rule
        nets = [sim(net).init().networks[0] for net in [ss.SexualNet(dur=5), ss.SexualNet(dur=ss.days(5)), ss.DegreeNet(dur=5, acts=ss.perday(1))]]
    assert all(net.mean_dur == 5 for net in nets) # Each 5 days at daily steps, ending at a constant rate, as for ss.DegreeNet
    s = ss.Sim(n_agents=200, diseases=ss.SIS(beta=ss.perday(0.05)), networks=ss.RandomNet(n_contacts=4, dur=5), dt=ss.days(1), dur=ss.days(10), verbose=0).init()
    net = s.networks[0]
    before = set(zip(net.p1.tolist(), net.p2.tolist()))
    s.run_one_step()
    assert before == set(zip(net.p1.tolist(), net.p2.tolist())) and np.all(net.edges.dur == 4) # Random contacts that last, a bare number in the sim's unit too


def test_network_args():
    """ A subclass that wraps step() keeps its parameters, and one with a step() of its own forms its own partnerships """
    sc.heading('Testing network subclasses')
    class Wrapped(ss.SexualNet):
        def step(self):
            return super().step()
    class Own(ss.SexualNet):
        def step(self):
            self.end_pairs()
            if not len(self):
                n = 10
                self.append(p1=np.arange(n), p2=np.arange(n, 2*n), weight=np.ones(n), dur=np.full(n, 100.0), acts=np.ones(n))
    kw = dict(diseases=ss.SIS(beta=ss.peract(0.01)), n_agents=300, verbose=0)
    for cls, n_start in [(Own, 0), (Wrapped, 0), (ss.SexualNet, None)]:
        sim = ss.Sim(networks=cls(n_contacts=0.5, dur=ss.years(2)), **kw, stop=2001).init()
        assert (len(sim.networks[0]) == 0) == (n_start == 0), f'{cls.__name__} should start with partnerships only from DegreeNet.step(), if it calls it'
        sim.run()
        net = sim.networks[0]
        if cls is Own:
            assert len(net) == 10 and set(net.p1) == set(range(10)), 'Only its own partnerships'
        else:
            assert np.isclose(net.results.mean_degree[0], 0.5, rtol=0.25), f'{cls.__name__} starts at the target'

    # A network among some people (who=), or with a step() of its own, has no compartmental form
    for net in [Own(n_contacts=0.5, dur=ss.years(2)), ss.RandomNet(who=ss.person.age < 30)]:
        with pytest.raises(ss.CapabilityError):
            ss.Sim(networks=net, solver='binomial', diseases=ss.SIS(beta=ss.peract(0.01)), total_pop=1000, verbose=0).init()


# %% Transmission over networks

class Couples(ss.SexualNet):
    """ Every man and woman in a couple, for life, with a fixed number of acts per step (acts=) """
    def __init__(self, acts=2, **kwargs):
        super().__init__(**kwargs)
        self.n_acts = acts
    def add_pairs(self):
        self.participant[self.sim.people.auids] = True
        m, f = [self.available(self.sim.people, sex) for sex in ['male', 'female']]
        n = min(len(m), len(f))
        self.append(p1=m[:n], p2=f[:n], weight=np.ones(n), dur=np.full(n, 10.0), acts=np.full(n, float(self.n_acts)))
        return


def test_certain_transmission():
    sc.heading('A probability of 1 per contact (an infinite rate) transmits to every contact, and a bare beta on a network is a rate')
    def infected(beta):
        S, I = ss.states('S I')
        si = ss.Disease(infection=S >> I @ beta*I, init_prev=0.3, name='si')
        sim = ss.Sim(si, networks=ss.RandomNet(n_contacts=2, dynamic=False), n_agents=5000, unit='week', dt=ss.weeks(1), stop=1, rand_seed=1, verbose=0).init()
        before = sim.diseases.si.I.raw.copy()
        e = sim.networks[0].edges
        k = np.bincount(e.p2[before[e.p1]], minlength=len(before)) + np.bincount(e.p1[before[e.p2]], minlength=len(before)) # Infectious contacts of each person
        at_risk = (k > 0) & ~before
        sim.run(until=sim.t.timevec[0]) # The first step only
        return sim.diseases.si.I.raw[at_risk].mean(), k[at_risk]
    assert infected(ss.probperweek(1))[0] == 1
    frac, k = infected(0.3)
    assert np.isclose(frac, np.mean(1 - np.exp(-0.3*k)), atol=0.03) # A rate per contact, not a probability of 0.3


def test_beta_readings():
    sc.heading('Testing how a beta is read on networks with and without acts, and their directions')
    kw = dict(n_agents=500, start=2000, stop=2002, verbose=0)
    def sis(beta):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.1, name='sis')
    assert ss.SexualNet().directions == ['m2f', 'f2m'] and ss.PrenatalNet().directions == ['mother2child'] and ss.RandomNet().directions == []
    assert Couples().directions == ['m2f', 'f2m'] and ss.SexualNet(between='mm').directions == [] # A sexual network of its own is between men and women unless it says otherwise; men only: one beta
    sim = ss.Sim(sis(ss.peract(m2f=0.2, f2m=0.1)), networks=Couples(name='couples', acts=2), **(kw | dict(n_agents=2000))).run()
    assert sim.results.sis.infection.sum() > 0
    assert 'per act' in sim.diseases.sis.describe()
    assert ss.peract({('m', 'f'): 0.3, ('f', 'm'): 0.15}) == ss.peract(m2f=0.3, f2m=0.15) # A direction as a pair of sides, or by name
    with pytest.raises(TypeError): # A rate per unit time times acts per unit time is not a hazard
        ss.Sim(sis(ss.permonth(0.01)), networks=ss.SexualNet(), **kw).init()
    with pytest.raises(ValueError): # Every direction needs a value
        ss.Sim(sis(ss.peract(m2f=0.01)), networks=ss.SexualNet(), **kw).init()
    for acts in [ss.freqperyear(80), ss.peryear(80)]: # A frequency is a number of acts per unit time, and a rate is taken as one
        net = ss.Sim(sis(ss.peract(0.01)), networks=ss.DegreeNet(dur=ss.years(1), acts=acts), **kw).init().networks[0]
        assert np.isclose(net.acts_per_step, 80*net.t.dt.years)

    # Mother-to-child networks transmit one way, so one value is enough; a library disease names routes the sim may not have, which take the default
    m2f = ss.par(0.06, name='m2f')
    S, I = ss.states('S I')
    rate = dict(sexualnet=ss.peract(m2f=m2f, f2m=0.5*m2f), prenatalnet=ss.peryear(0.5), msm=ss.peract(0.1), default=0)
    hiv = ss.Disease(infection=ss.Transition(S, I, rate=rate * I), init_prev=0.1, name='hiv')
    sim = ss.Sim(hiv, networks=[ss.SexualNet(), ss.PrenatalNet()], demographics=ss.Pregnancy(), **kw).run()
    assert sim.results.hiv.I[-1] > 0 and any('msm' in note for note in sim.notes)

    # With no networks (homogeneous mixing), the default is the beta; a disease written as code reads the default too
    S, I = ss.states('S I')
    d = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=ss.peract(0.01), default=ss.peryear(2)) * I), init_prev=0.1, name='d')
    assert ss.Sim(d, **kw).run().results.d.I[-1] > 0
    sim = ss.Sim(diseases=ssl.AgentSIS(beta=dict(random=0.05, msm=0.1, default=0)), networks=['random', ss.SexualNet()], **kw).run()
    assert sim.results.agentsis.prevalence[-1] > 0 and sim.diseases.agentsis.engine.code_betas()['sexual'] == [0, 0]


def test_route_par():
    sc.heading('Testing a modifier of a parameter on one route, which acts on every transmission that the parameter scales')
    def run(mods):
        S, I, R = ss.states('S I R')
        beta = ss.par(ss.peryear(3.0))
        sir = ss.Disease(infection=S >> I @ beta*I, reinfection=R >> I @ 0.5*beta*I, recovery=I >> R @ ss.years(0.5), init_prev=0.05, name='sir')
        nets = [ss.RandomNet(name='a', n_contacts=3), ss.RandomNet(name='b', n_contacts=3)]
        sim = ss.Sim(diseases=sir, networks=nets, connectors=mods, n_agents=1000, start=2000, stop=2003, rand_seed=1, verbose=0).run()
        return sim.results.sir.infection.sum(), sim.results.sir.reinfection.sum()
    by_par = run(ss.modify('sir.pars.beta', multiply=0.2, network='b'))
    assert by_par == run([ss.modify(f'sir.{tr}', multiply=0.2, network='b') for tr in ['infection', 'reinfection']])
    assert by_par[0] < run(None)[0]


def test_beta_by_person():
    sc.heading('Testing a beta per act that differs between the people who would be infected, and who infected whom')
    def male_infections(beta):
        S, I = ss.states('S I')
        d = ss.Disease(infection=S >> I @ beta * I, recovery=I >> S @ ss.years(1), init_prev=0.2, name='d')
        sim = ss.Sim(d, networks=ss.SexualNet(), n_agents=2000, stop=3, rand_seed=1, verbose=0).run()
        d = sim.diseases.d
        return np.count_nonzero(d.I & sim.people.male & (d.ti_I > 0)) # Infected during the run, not seeded

    # By the sex of the susceptible, as by direction: men are infected only by f2m, so with f2m = 0 never
    m2f = ss.par(0.2, name='m2f')
    assert male_infections(m2f * ss.by(sex=dict(female=1.0, male=0.0))) == 0
    assert male_infections(ss.peract(m2f=0.2, f2m=0.1)) == male_infections(m2f * ss.by(sex=dict(female=1.0, male=0.5)))

    # Each infection over a network has its source (engine.sources()), whichever transmission it was by
    found = []
    def check(sim):
        d = sim.diseases.d
        new = d.infection.uids | d.mtct.uids
        sources, nets = d.engine.sources(new)
        found.append((len(new), np.count_nonzero(~np.isnan(sources))))
    S, I = ss.states('S I')
    d = ss.Disease(infection=S >> I @ dict(sexualnet=ss.peract(0.2), default=0) * I, mtct=S >> I @ dict(prenatalnet=ss.peryear(2), default=0) * I, recovery=I >> S @ ss.years(1), init_prev=0.2, name='d')
    ss.Sim(d, networks=[ss.SexualNet(), ss.PrenatalNet()], demographics=ss.Pregnancy(fertility_rate=ss.peryear(0.1)), n_agents=2000, stop=3, analyzers=check, rand_seed=1, verbose=0).run()
    n_new, n_found = np.sum(found, axis=0)
    assert n_new > 0 and n_found == n_new


def test_edge_index():
    sc.heading('Testing that a static network transmits and traces by the edges of each agent, as by a pass over all its edges, and makes a random graph as NetworkX does')
    p1, p2 = ssl.networks.theoretical.random_graph(500, 0.02, np.random.default_rng(2))
    edges = np.array(nx.fast_gnp_random_graph(500, 0.02, seed=np.random.default_rng(2)).edges())
    assert np.array_equal(np.column_stack([p1, p2]), edges), 'Expected the same random graph as NetworkX makes from the same random numbers'
    S, I, X = ss.states('S I X')
    sis = ss.Disease(infection=S >> I @ 0.1*I, recovery=I >> S @ 0.2, quarantine=I >> X @ 0.05, init_n=dict(I=20), name='sis')
    tracing = ss.Delivery(I >> X, when=sis.quarantine, coverage=0.5, trace='net')
    sim = ss.Sim(sis, networks=ssl.StaticNet(n_contacts=8, name='net'), interventions=tracing, n_agents=2000, unit='day', stop=30, verbose=0)
    runs = []
    for frac in [0, 1]: # By a pass over the edges, then by the edges of each agent wherever it can
        with mock.patch.object(ss.solvers.numba.passes, 'index_frac', frac):
            runs.append(sim.new().run())
    assert runs[1].networks.net.agent_edges.lists is not None, 'Expected a static network to make the edges of each agent'
    assert all(np.array_equal(runs[0].results.sis[k], runs[1].results.sis[k]) for k in 'SIX'), 'Expected the same results by the edges of each agent'


def test_network_saturation():
    sc.heading('Testing that an ODE on a sparse network matches the agents, whose edges each transmit at most once in a step')
    K, b, dur, dt = 2, 3.0, 0.25, 1/12 # Few contacts, each with a large chance of transmitting in a month
    sis = ss.SIS(beta=ss.peryear(b), dur_inf=ss.years(dur), init_prev=0.1)
    sim = ss.Sim(diseases=sis, networks=ss.RandomNet(n_contacts=K), total_pop=5000, start=2000, stop=2020, dt=ss.months(1), solver='ode', verbose=0).run()
    ode = sim.results.sis.prevalence[-120:].mean()
    expected = 1 - 1/(K*(-np.expm1(-b*dt)/dt)*dur) # Mass action with the transmission the agents' edges give: K·(1 − e^(−b·dt))/dt
    assert np.isclose(ode, expected, rtol=0.01), f'{ode = :.4f}, but the per-step transmission over each edge gives {expected:.4f}'

    # Each flow saturates by its own rate, and each source state by its own coefficient: a reinfection at a quarter of the rate, from a state half as infectious.
    # The agents on a sparse network differ from the ODE as they do on a dense one (where mass action holds), apart from saturation
    def make(solver, K, b, n=5_000):
        S, A, I, R = ss.states('S A I R')
        beta = ss.peryear(b)
        dis = ss.Disease(infection=ss.Transition(S, A, rate=beta*(0.5*A + I)), reinfection=ss.Transition(R, A, rate=0.25*beta*(0.5*A + I)),
                         progression=ss.Transition(A, I, rate=ss.peryear(4)), recovery=ss.Transition(I, R, rate=ss.peryear(4)), init_prev=0.1, name='d')
        sim = ss.Sim(diseases=dis, networks=ss.RandomNet(n_contacts=K), n_agents=n, total_pop=n, start=2000, stop=2030, dt=ss.months(1), solver=solver, verbose=0).run()
        return sim.results.d.prevalence[-120:].mean()
    sparse = make('abm', 3, 6.0)/make('ode', 3, 6.0)
    dense = make('abm', 60, 6.0*3/60)/make('ode', 60, 6.0*3/60)
    assert abs(sparse - dense) < 0.08, f'agents ÷ ODE is {sparse:.3f} on the sparse network but {dense:.3f} on the dense one'


# %% Mixing pools and contact matrices

def test_mixing_pools():
    """ A MixingPool with defaults and by sex; MixingPools by group; and a disease written as code against a declared one """
    sc.heading('Testing mixing pools...')
    sim = ss.Sim(n_agents=medium, diseases='sir', networks=ss.MixingPool()).run()
    assert sim.results.sir.infection.sum() > 0
    mp = ss.MixingPool(source=lambda sim: sim.people.female, target=lambda sim: sim.people.male, n_contacts=ss.poisson(mean=4)) # Female to male (only)
    sim = ss.Sim(n_agents=medium, diseases=ss.SIR(init_prev=lambda sim, uids: 0.05*sim.people.female[uids]), networks=mp).run()
    new = sim.diseases.sir.ti_I > 0
    assert new.any() and sim.people.male[new].all() # All new infections should be in men
    groups = dict(Female=lambda sim: sim.people.female, Male=lambda sim: sim.people.male)
    sim = ss.Sim(n_agents=medium, diseases='sir', networks=ss.networks.MixingPools(n_contacts=np.array([[1.4, 0.5], [1.2, 0.7]]), source=groups, target=groups)).run()
    assert sim.results.sir.infection.sum() > 0
    with pytest.raises(TypeError): # A mixing pool takes no beta=, which is the disease's to say
        ss.MixingPool(beta=1.0)

    # The hazard is multiplied by the share infectious before it is a probability, for a disease written as code as for a declared one
    S, I, R = ss.states('S I R')
    pars = dict(networks=ss.MixingPool(n_contacts=1), n_agents=5000, dur=ss.days(150), verbose=0)
    coded = ss.Sim(ssl.AgentSIR(beta=ss.perday(0.4), dur_inf=ss.expon(mean=ss.days(7)), p_death=0, init_prev=0.01, name='sir'), **pars).run()
    declared = ss.Sim(ss.Disease(S >> I @ 0.4*I, I >> R @ ss.days(7), init_prev=0.01, name='sir'), **pars).run()
    sizes = coded.results.sir.recovered[-1], declared.results.sir.R[-1]
    assert np.isclose(*sizes, rtol=0.03), sizes


def test_layers():
    sc.heading('Testing contact layers with a beta for each, under both engines')
    def run(beta, solver, layers=('home', 'school')):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        nets = [ss.contacts('Kenya', layer=layer) for layer in layers]
        sim = ss.Sim(sir, networks=nets, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100, solver=solver, total_pop=10_000, verbose=0, **(dict(n_agents=10_000) if solver == 'abm' else {}))
        return sim.run().results.sir.infection.cum[-1]
    ode = run(dict(home=0.03, school=0.02), 'ode')
    assert np.isclose(run(dict(home=0.03, school=0.02), 'abm'), ode, rtol=0.1)
    assert np.isclose(run(dict(home=0.03, school=0.0), 'ode'), run(0.03, 'ode', layers=['home'])) < 0.9*ode # No transmission at school is home contacts only
    with pytest.raises(ValueError): # A route with no key
        run(dict(home=0.03), 'abm')


def test_matrix():
    """ A matrix of numbers, or one distribution of matrices, makes no pool for each pair of groups, and gives the epidemic those pools would """
    sc.heading('Testing MixingPools with a matrix...')
    groups = dict(young=ss.person.age < 30, old=ss.person.age >= 30)
    run = lambda n: ss.Sim(ss.SIS(init_prev=0.02, beta=ss.perday(0.05)), networks=ss.networks.MixingPools(source=groups, target=groups, n_contacts=n), n_agents=5000, unit='day', stop=40, solver='abm', verbose=0).run()
    matrix = np.array([[3.0, 1], [1, 2]])
    sims = [run(matrix), run(np.vectorize(lambda v: ss.uniform(v, v), otypes=[object])(matrix))] # A distribution makes a pool for each pair, here with the same contacts for everyone
    assert len(sims[0].networks[0].pools) == 0 and len(sims[1].networks[0].pools) == 4
    assert np.isclose(*[sim.results.sis.infection.sum() for sim in sims], rtol=0.05)

    matrix = np.array([[2.4, 0.9], [0.5, 1.6]])
    sims = [run(ss.poisson(matrix)), run(np.vectorize(ss.poisson, otypes=[object])(matrix))]
    net = sims[0].networks[0]
    assert len(net.pools) == 0 and len(sims[1].networks[0].pools) == 4
    auids = sims[0].people.auids
    contacts = net.contacts_0.raw[auids][sims[0].people.age.raw[auids] < 30] # Each young person's contacts with the young: a Poisson of mean 2.4
    assert np.isclose(contacts.mean(), 2.4, rtol=0.1) and np.isclose(contacts.var(), 2.4, rtol=0.2)
    assert np.isclose(*[sim.results.sis.infection.sum() for sim in sims], rtol=0.1)

    # Through ss.contacts(), from a country's table of contacts a day, in a sim by week
    sim = ss.Sim(ss.SIS(beta=ss.perday(0.02)), networks=ss.contacts(ss.poisson(ss.data.contacts('Kenya'))), n_agents=2000, unit='week', stop=4, verbose=0).run()
    assert sim.networks[0].contact_scale == 7 and len(sim.networks[0].pools) == 0

    # A matrix given by position alone is over the stratification with as many levels, as if given by name; with none or several, it is an error
    C = [[10, 3, 1], [3, 5, 2], [1, 2, 4]]
    def run_strat(net, **strata):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ 0.05*I, I >> R @ ss.days(10), init_prev=0.01, name='sir').stratify(**strata)
        return ss.Sim(sir, networks=net, unit='day', stop=30, verbose=0).run().results.sir.I[:]
    assert np.allclose(run_strat(ss.contacts(C), age=[0, 15, 65], patch=['n', 's']), run_strat(ss.contacts(age=C), age=[0, 15, 65], patch=['n', 's']))
    with pytest.raises(ValueError, match='each have 2 levels'):
        run_strat(ss.contacts([[1, 2], [2, 1]]), sex=['f', 'm'], patch=['n', 's'])

    # A disease whose beta on a pool is 0 does not spread there
    S, I = ss.states('S I')
    make = lambda name, beta: ss.Disease(S >> I @ dict(pool=beta)*I, I >> S @ ss.days(10), init_prev=0.05, name=name)
    sim = ss.Sim(diseases=[make('a', 0.1), make('b', 0)], networks=ss.MixingPool(n_contacts=4, name='pool'), n_agents=1000, unit='day', stop=20, solver='abm', verbose=0).run()
    assert sim.results.a.infection.sum() > 10 and sim.results.b.infection.sum() == 0


def test_counts():
    """ A contact matrix acts on what the rate says: the share of each group (beta*I), the number in each (beta*I.n), or that over everyone (beta*I.frac), as compartments and as agents """
    sc.heading('Testing what a contact matrix acts on...')
    beta, split, pop = np.array([[10, 0.1], [0.1, 1]]), np.array([0.2, 0.8]), 10_000 # Keeling & Rohani's Program 3.1, on shares of everyone
    def make(rate, solver='ode', **kwargs):
        S, I = ss.states('S I')
        sis = ss.Disease(infection=S >> I @ rate(S, I), recovery=I >> S @ ss.peryear(1), init=dict(I=0.01), name='sis').stratify(risk={'high':split[0], 'low':split[1]})
        sim = ss.Sim(sis, networks=ss.contacts(risk=beta), total_pop=pop, stop=15, unit='year', dt=0.05, solver=solver, verbose=0, **kwargs).run()
        return np.array([sim.results.sis.I.select(risk=risk).values for risk in ['high', 'low']])/pop
    t = np.arange(0, 15.01, 0.05)
    solve = lambda w: solve_ivp(lambda t, I: (beta @ (w*I))*(split - I) - I, [0, 15], 0.01*split, t_eval=t, rtol=1e-10, atol=1e-12).y
    frac = make(lambda S, I: I.frac)
    assert np.abs(frac - solve(1)).max() < 1e-4
    assert np.abs(make(lambda S, I: (ss.by(risk=[2, 1])*I).n/pop) - solve(np.array([2, 1]))).max() < 1e-4 # A weighted count, also acted on by the matrix
    assert np.allclose(make(lambda S, I: I.n/(S + I).total), frac, rtol=1e-9) # The same, as counts over the population
    assert make(lambda S, I: I)[:, -1].sum() > 2*frac[:, -1].sum() # On the share of each group, which is more
    agents = make(lambda S, I: I.n/pop, solver='abm', n_agents=pop)
    assert np.allclose(agents[:, -1], frac[:, -1], rtol=0.1)

    # A count inside a function of the rate modifies it by the total, and is not met: three groups mixing evenly are one group
    def capped(groups):
        S, I, R = ss.states('S I R')
        slowdown = 1 - 0.5*np.clip(I.n/2000, 0, 1) # As MEmilio's capacity to test and trace
        d = ss.Disease(infection=S >> I @ ss.perday(0.3)*slowdown*I, recovery=I >> R @ ss.perday(0.1), init_n=dict(I=30), name='d')
        nets = None
        if groups:
            d.stratify(age=['0', '1', '2'])
            nets = ss.contacts(age=np.full((3, 3), 1/3))
        return ss.Sim(d, networks=nets, total_pop=10_000, unit='day', stop=100, verbose=0).run().results.d.R[-1]
    assert np.isclose(capped(True), capped(False))


if __name__ == '__main__':
    T = sc.timer()
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
    T.toc()
