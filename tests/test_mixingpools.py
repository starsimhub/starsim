"""
Test mixing pools and contact layers
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import starsim as ss
import starsim.library as ssl
import pytest

n_agents = 1_000
do_plot = False
sc.options(interactive=False) # Assume not running interactively


def run(networks, sir=None, do_plot=False):
    """ Run an SIR over the given mixing pools, and check there were infections """
    sim = ss.Sim(n_agents=n_agents, diseases=sir or ss.SIR(), networks=networks)
    sim.run()
    if do_plot:
        sim.plot()
    cum = sim.results.sir.infection.cum
    assert cum[-1] > cum[0], 'There were no new infections in the simulation'
    return sim


@sc.timer()
def test_single(do_plot=do_plot):
    """ Test a single MixingPool: with defaults, by UIDs, by age, and by sex """
    sc.heading('Testing single mixing pools...')
    sims = sc.obj()
    sims.defaults = run(ss.MixingPool(), do_plot=do_plot)

    # In this arrangement, src transmits to dst, but can never go the other way, so the epidemic dies out once everyone in src recovers
    k = n_agents // 2
    sims.uids = run(ss.MixingPool(source=ss.uids(np.arange(k)), target=ss.uids(np.arange(k, n_agents))), do_plot=do_plot)

    # Incidence must decline because 0-15 --> 15+ transmission only
    mp = ss.MixingPool(source=ss.person.age < 15, target=ss.person.age >= 15, beta=1.0, n_contacts=ss.poisson(lam=5))
    sims.age = run(mp, ss.SIR(init_prev=0.1), do_plot=do_plot) # Enough seeds under 15 that some infect

    # Female to male (only) transmission
    mp = ss.MixingPool(source=lambda sim: sim.people.female, target=lambda sim: sim.people.male, beta=1.0, n_contacts=ss.poisson(lam=4))
    sir = ss.SIR(init_prev=lambda sim, uids: 0.05*sim.people.female[uids]) # Seed 5% of the female population
    sims.sex = sim = run(mp, sir, do_plot=do_plot)
    assert sim.people.male[sim.diseases.sir.ti_I>0].all() # All new infections should be in men
    return sims


@sc.timer()
def test_single_errors():
    """ Test a single MixingPool with an NCD, and with a missing disease """
    sc.heading('Testing mixing pool errors...')
    mp_pars = dict(source=ss.person.age < 15, target=ss.person.age >= 15, beta=1.0, n_contacts=ss.poisson(lam=5))
    for disease, pool_disease in [(ssl.NCD(), 'ncd'), (ss.SIR(), 'hiv')]:
        sim = ss.Sim(n_agents=n_agents, diseases=disease, networks=ss.MixingPool(diseases=pool_disease, **mp_pars))
        with pytest.raises(Exception):
            sim.run()
    return sim


@sc.timer()
def test_multi(do_plot=do_plot):
    """ Test MixingPools, with defaults and by sex """
    sc.heading('Testing multiple mixing pools...')
    sim1 = run(ss.networks.MixingPools(source={'all':None}, target={'all':None}, n_contacts=[[1]]), do_plot=do_plot)
    groups = {
        'Female': lambda sim: sim.people.female,
        'Male': lambda sim: sim.people.male,
    }
    mps = ss.networks.MixingPools(n_contacts=np.array([[1.4, 0.5], [1.2, 0.7]]), beta=1.0, source=groups, target=groups)
    sim2 = run(mps, do_plot=do_plot)
    return sim1, sim2


@sc.timer()
def test_coded_declared():
    sc.heading('Testing that a disease written as code gives the same epidemic on a mixing pool as a declared one')
    S, I, R = ss.states('S I R')
    pars = dict(networks=ss.MixingPool(n_contacts=1), n_agents=5000, stop=ss.days(150), verbose=0)
    coded = ss.Sim(ssl.AgentSIR(beta=ss.perday(0.4), dur_inf=ss.expon(scale=ss.days(7)), p_death=0, init_prev=0.01, name='sir'), **pars).run()
    declared = ss.Sim(ss.Disease(S >> I @ 0.4*I, I >> R @ ss.days(7), init_prev=0.01, name='sir'), **pars).run()
    sizes = coded.results.sir.recovered[-1], declared.results.sir.R[-1]
    assert np.isclose(*sizes, rtol=0.03), sizes # The hazard is multiplied by the share infectious before it is a probability (when it was not, the coded epidemic was about 5% smaller, and its peak a fifth lower)
    return sizes


@sc.timer()
def test_layers():
    sc.heading('Testing contact layers with a beta for each, under both engines')
    def run(beta, solver, layers=('home', 'school')):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
        nets = [ss.contacts('Kenya', layer=layer) for layer in layers]
        sim = ss.Sim(sir, networks=nets, stratify=dict(age=[0, 5, 18, 65]), unit='day', stop=100, solver=solver, total_pop=10_000, verbose=0, **(dict(n_agents=10_000) if solver == 'abm' else {}))
        return sim.run().results.sir.infection.cum[-1]
    ode = run(dict(home=0.03, school=0.02), 'ode') # A layered example
    assert np.isclose(run(dict(home=0.03, school=0.02), 'abm'), ode, rtol=0.1)
    assert np.isclose(run(dict(home=0.03, school=0.0), 'ode'), run(0.03, 'ode', layers=['home'])) < 0.9*ode # No transmission at school is home contacts only
    for bad in [dict(home=0.03, school=0.02, wrok=0.1), dict(home=0.03)]: # A key with no route, and a route with no key
        with pytest.raises(ValueError):
            run(bad, 'abm')
    return ode


@sc.timer()
def test_contacts_coded():
    """ A disease written as code goes through ss.contacts() from the counts in each band, as through a pool for each pair of bands """
    sc.heading('Testing ss.contacts() with a disease written as code...')
    ages = [0, 15, 65]
    matrix = np.array([[6.0, 2, 1], [2, 8, 2], [1, 3, 4]])
    groups = {str(low):(ss.person.age >= low) & (ss.person.age < high) if high is not None else ss.person.age >= low for low,high in zip(ages, [15, 65, None])}
    pools = ss.networks.MixingPools(source=groups, target=sc.dcp(groups), n_contacts=np.vectorize(lambda v: ss.constant(float(v)), otypes=[object])(matrix.T))
    cases = [ss.Sim(diseases=ssl.AgentSIR(init_prev=0.02, beta=ss.perday(0.05), dur_inf=ss.days(6)), networks=net, n_agents=n_agents, unit='day', stop=60, verbose=0).run().results.agentsir.infection.sum() for net in [ss.contacts(matrix, ages=ages), pools]]
    assert cases[0] > 0.5*n_agents and np.isclose(*cases, rtol=0.05)
    return


@sc.timer()
def test_matrix():
    """ A matrix of numbers makes no pool for each pair of groups, and gives the epidemic those pools would """
    sc.heading('Testing MixingPools with a matrix of numbers...')
    matrix = np.array([[3.0, 1], [1, 2]])
    groups = dict(young=ss.person.age < 30, old=ss.person.age >= 30)
    pools = np.vectorize(lambda v: ss.uniform(v, v), otypes=[object])(matrix) # A distribution makes a pool for each pair, here with the same contacts for everyone
    sims = [ss.Sim(ss.SIS(init_prev=0.02, beta=ss.perday(0.05)), networks=ss.networks.MixingPools(source=groups, target=groups, n_contacts=n), n_agents=n_agents, unit='day', stop=40, solver='abm', verbose=0).run() for n in [matrix, pools]]
    assert len(sims[0].networks[0].pools) == 0 and len(sims[1].networks[0].pools) == 4
    assert np.isclose(*[sim.results.sis.infection.sum() for sim in sims], rtol=0.05)

    # Groups can overlap, and be people who die
    uids = ss.uids(np.arange(100))
    sim = ss.Sim(ss.SIS(init_prev=0.1, beta=ss.perday(0.05)), networks=ss.networks.MixingPools(source=dict(a=uids, b=None), target=dict(a=uids, b=None), n_contacts=[[1, 1], [1, 1]]), demographics=ss.Deaths(), n_agents=1000, unit='year', stop=30, solver='abm', verbose=0).run()
    assert len(sim.networks[0].pars.source.a) < 100

    # A pool transmits only the diseases it is for, written as code or declared
    def make(name, cls=None):
        S, I = ss.states('S I')
        return cls(init_prev=0.05, beta=ss.perday(0.1), name=name) if cls else ss.Disease(S >> I @ 0.1*I, I >> S @ ss.days(10), init_prev=0.05, name=name)
    for cls in [None, ssl.AgentSIR]:
        for net in [ss.MixingPool(diseases='a', n_contacts=4), ss.networks.MixingPools(diseases='a', source=dict(x=None), target=dict(x=None), n_contacts=[[4]])]:
            sim = ss.Sim(diseases=[make('a', cls), make('b', cls)], networks=net, n_agents=1000, unit='day', stop=20, solver='abm', verbose=0).run()
            assert sim.results.a.infection.sum() > 10*sim.results.b.infection.sum()
    return


@sc.timer()
def test_counts():
    """ A contact matrix acts on what the rate says: the share of each group (beta*I), the number in each (beta*I.n), or that over everyone (beta*I.frac), as compartments and as agents """
    sc.heading('Testing what a contact matrix acts on...')
    from scipy.integrate import solve_ivp
    beta, split, pop = np.array([[10, 0.1], [0.1, 1]]), np.array([0.2, 0.8]), 10_000 # Keeling & Rohani's Program 3.1, on shares of everyone
    def make(rate, solver='ode', **kwargs):
        S, I = ss.states('S I')
        sis = ss.Disease(infection=S >> I @ rate(S, I), recovery=I >> S @ ss.peryear(1), init=dict(I=0.01), name='sis').stratify(risk={'high':split[0], 'low':split[1]})
        sim = ss.Sim(sis, networks=ss.contacts(beta, dims='risk'), total_pop=pop, stop=15, unit='year', dt=0.05, solver=solver, verbose=0, **kwargs).run()
        return np.array([sim.results.sis.I.select(risk=risk).values for risk in ['high', 'low']])/pop
    t = np.arange(0, 15.01, 0.05)
    solve = lambda w: solve_ivp(lambda t, I: (beta @ (w*I))*(split - I) - I, [0, 15], 0.01*split, t_eval=t, rtol=1e-10, atol=1e-12).y
    ref = solve(1)
    frac = make(lambda S, I: I.frac)
    assert np.abs(frac - ref).max() < 1e-4
    assert np.abs(make(lambda S, I: (ss.by(risk=[2, 1])*I).n/pop) - solve(np.array([2, 1]))).max() < 1e-4 # A weighted count, also acted on by the matrix (it was an error)
    for rate in [lambda S, I: I.n/pop, lambda S, I: I.n/(S + I).total]: # The same, as counts over the population
        assert np.allclose(make(rate), frac, rtol=1e-9)
    shares = make(lambda S, I: I) # On the share of each group, which is more
    assert shares[:, -1].sum() > 2*frac[:, -1].sum()
    agents = make(lambda S, I: I.n/pop, solver='abm', n_agents=pop)
    assert np.allclose(agents[:, -1], frac[:, -1], rtol=0.1)

    # A matrix with no terms between places still acts on a count (it was ignored)
    def spread(networks):
        S, I = ss.states('S I')
        d = ss.Disease(S >> I @ (3*I.n), I >> S @ ss.perday(1), init=dict(I=ss.by(patch=[0.01, 0, 0])), name='d').stratify(patch=['a', 'b', 'c'], kind='space')
        return ss.Sim(d, networks=networks, total_pop=3, stop=20, solver='ode', verbose=0).run().results.d.I.by('patch').values[-1]
    assert np.allclose(spread(ss.contacts(np.ones((3, 3)), dims='patch')), 8/9)
    assert np.allclose(spread(ss.contacts(np.eye(3), dims='patch')), spread(None))

    # A count inside a function of the rate modifies it by the total, and is not met: three groups mixing evenly are one group
    def capped(groups):
        S, I, R = ss.states('S I R')
        slowdown = 1 - 0.5*np.clip(I.n/2000, 0, 1) # As MEmilio's capacity to test and trace
        d = ss.Disease(infection=S >> I @ ss.perday(0.3)*slowdown*I, recovery=I >> R @ ss.perday(0.1), init_n=dict(I=30), name='d')
        nets = None
        if groups:
            d.stratify(age=['0', '1', '2'])
            nets = ss.contacts(np.full((3, 3), 1/3))
        return ss.Sim(d, networks=nets, total_pop=10_000, unit='day', stop=100, verbose=0).run().results.d.R[-1]
    assert np.isclose(capped(True), capped(False)) # Before, the count in the function was met, as I/3
    return


def test_keep_memory(monkeypatch):
    """ Older glibc refuses a threshold above 32 MB, which should not stop the rest of the setting """
    memory = ss.solvers.backends.memory
    class Lib:
        calls = []
        def mallopt(self, key, value):
            self.calls.append((key, value))
            return int(key != memory.M_MMAP_THRESHOLD or value <= 2**25)
    monkeypatch.setattr(memory, 'libc', Lib)
    assert memory.keep()
    assert Lib.calls[-2:] == [(memory.M_MMAP_THRESHOLD, 2**25), (memory.M_TRIM_THRESHOLD, 2**31 - 1)]
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    sims = test_single(do_plot)
    sim = test_single_errors()
    sims2 = test_multi(do_plot)
    layers = test_layers()

    T.toc()

    if do_plot:
        plt.show()
