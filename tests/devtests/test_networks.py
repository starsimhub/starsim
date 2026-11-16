"""
Test networks: lasting partnerships, transmission per act, and the partner-change model
"""

# %% Imports and settings
import pytest
import sciris as sc
import numpy as np
import starsim as ss

sc.options.interactive = False # Assume not running interactively
medium = 1000


def test_partners():
    sc.heading('Testing lasting partnerships, and a target number of partners at a time')
    S, I = ss.states(2)
    sis = ss.Disease(S >> I @ ss.prob(0.64), I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')

    # The targets of EpiModel's introductory network, from the library
    net = ss.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)
    sim = ss.Sim(sis, networks=net, n_agents=medium, stop=400, verbose=0).run()
    net = sim.networks[0]
    res = net.results
    assert np.isclose(net.dist_degree @ np.arange(4), 0.7) and np.isclose(net.dist_degree[2:].sum(), 0.22) # The targets
    assert np.isclose(res.mean_degree[200:].mean(), 0.7, rtol=0.1) # ... are what the network has
    assert np.isclose(res.concurrency[200:].mean(), 0.22, rtol=0.1)
    assert net.degrees(sim.people.alive.uids).max() <= 3
    assert sim.results.sis.I[-1] > 100 # Infection spreads through the partnerships
    table = net.check(n_runs=2, output=True)
    assert np.isclose(table.realized.dur, 50, rtol=0.15) # How long partnerships last, from how many there are and how many form

    # Acts in each partnership (acts=, per unit time): beta is then per act, so two acts a day at 0.4 transmit as one partnership at 1 - 0.6**2 a day
    def prev(acts, beta):
        S, I = ss.states(2)
        sis2 = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')
        return ss.Sim(sis2, networks=ss.DegreeNet(n_contacts=0.7, dur=50, acts=acts), n_agents=medium, stop=200, rand_seed=1, verbose=0).run().results.sis.I.values
    assert np.allclose(prev(ss.perday(2), ss.peract(0.4)), prev(None, ss.prob(1 - 0.6**2)))

    # Most partnerships between people at the same level of risk (EpiModel's nodematch)
    S, I = ss.states('S I')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), name='sir').stratify(risk={'low': 0.8, 'high': 0.2})
    net = ss.DegreeNet(n_contacts=0.7, dur=50, match=dict(risk=0.9), name='net')
    sim = ss.Sim(sir, networks=net, n_agents=2000, stop=300, verbose=0).run()
    net = sim.networks.net
    level = sim.groups.level_raw(sim.groups.dim_index('risk'))
    assert np.isclose(np.mean(level[net.edges.p1] == level[net.edges.p2]), 0.9, atol=0.03) # By chance, 0.8² + 0.2² = 0.68
    assert np.isclose(net.results.mean_degree[150:].mean(), 0.7, rtol=0.05)


def chance(beta, acts=ss.constant(ss.peryear(1)), dt=ss.years(1), steps=1, connectors=None, net=None, n=40_000, f2m=False, seed=1):
    """ The chance that a partnership with an infectious partner transmits over some steps, measured over many lasting partnerships (from men to women, or with f2m=True, women to men) """
    S, I = ss.states('S I')
    si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.5, name='si')
    net = net or ss.SexualNet
    net = net(dur=ss.constant(ss.years(100)), participation=ss.bernoulli(p=1.0), debut=ss.constant(0), acts=acts)
    sim = ss.Sim(si, networks=net, connectors=connectors, n_agents=n, start=2000, stop=2000 + (steps - 1)*dt.years, dt=dt, rand_seed=seed, verbose=0).init() # A step from each time point, the last included
    e = sim.networks[0].edges
    female = sim.people.female.raw
    men, women = np.where(female[e.p1], e.p2, e.p1), np.where(female[e.p1], e.p1, e.p2)
    src, trg = (women, men) if f2m else (men, women)
    before = sim.diseases.si.I.raw.copy()
    sim.run()
    at_risk = before[src] & ~before[trg]
    return sim.diseases.si.I.raw[trg[at_risk]].mean()


def test_per_act():
    sc.heading('Testing a probability per act on a network with acts, which keeps its meaning at any step')

    # A partnership with 80 acts a year transmits with 1 - 0.99**80 = 0.552 a year at per-act 0.01, whatever the step
    for dt in [ss.years(1/12), ss.weeks(1)]:
        p = chance(ss.peract(m2f=0.01, f2m=0), acts=ss.constant(ss.freqperyear(80)), dt=dt, steps=round(1/dt.years))
        assert np.isclose(p, 1 - 0.99**80, atol=0.025), f'{dt}: {p:.3f}'

    # Each direction has its own probability per act: from women to men at 0.15 per act, 4 acts in a step give 1 - 0.85**4 = 0.478
    p = chance(ss.peract(m2f=0.3, f2m=0.15), acts=ss.constant(ss.peryear(4)), f2m=True)
    assert np.isclose(p, 1 - 0.85**4, atol=0.02), p

    # rel_trans and modifiers multiply the hazard per act: beta 0.03 with rel_trans 10 is 1 - 0.97**10 = 0.263 per act
    p10 = chance(ss.peract(m2f=0.03, f2m=0), connectors=[ss.modify('si.rel_trans', multiply=10)])
    assert np.isclose(p10, 1 - 0.97**10, atol=0.015), p10
    assert p10 == chance(ss.peract(m2f=0.03, f2m=0), connectors=[ss.modify('si.rel_trans', multiply=2), ss.modify('si.pars.beta', multiply=5)])


def test_route_modifiers():
    sc.heading('Testing modifiers of a transmission on one route: condoms, needle programs')

    # A condom of efficacy 0.9 in half the acts multiplies the hazard by 1 - 0.5*0.9: p=0.1, 10 acts gives 1 - exp(-10*0.1054*0.55) = 0.440
    condoms = lambda **kw: [ss.modify('si.infection', network='sexualnet', **kw)]
    p = chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.peryear(10)), connectors=condoms(efficacy=0.9, frac=0.5), n=100_000)
    assert np.isclose(p, 1 - np.exp(10*np.log(0.9)*0.55), atol=0.01), p
    assert p == chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.peryear(10)), connectors=condoms(multiply=0.1, frac=0.5), n=100_000) # efficacy= is multiply=1 - efficacy



def test_partnership_approx():
    sc.heading('Testing the partner-change model: lasting partnerships under the ODE, binomial, and Gillespie solvers')
    def run(solver, net, beta=ss.peract(m2f=0.004, f2m=0.002), stratify=dict(sex=['female', 'male']), seed=1, n=5000):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.05, name='d') # SI with no recovery
        sim = ss.Sim(si, networks=net, n_agents=n, total_pop=n, start=2000, stop=2020, dt=ss.months(1), solver=solver, rand_seed=seed, verbose=0)
        return sim.stratify(**stratify)

    # With partnerships that end at a constant rate and no way out of the infectious state, the chance over a partnership is H/(H + 1/D̄)
    sim = run('ode', ss.DegreeNet(name='pairs', n_contacts=1, dur=ss.years(2), acts=ss.peryear(10)), beta=ss.peract(0.05), stratify=dict(risk=['low', 'high'])).init()
    H = 10*-np.log(0.95)
    assert 'partner-change model (approx)' in sim.diseases.d.describe() # Which also works out partnership_info
    assert np.isclose(sim.diseases.d.engine.partnership_info[('infection', 'I', 'pairs', 0)][0], H/(H + 1/2)), 'H/(H + 1/D)'

    # Long monogamous partnerships: the ODE is somewhat too high against agents (20 runs of 10,000 agents gave ratios of 1.105 and 1.247 at 10 and 20 years), since it spreads infection that agents keep within pairs
    mf = lambda: ss.SexualNet(participation=ss.bernoulli(p=1.0), debut=ss.constant(0), dur=ss.lognormal(mean=ss.years(5), std=ss.years(3)), acts=ss.peryear(80))
    prev = lambda solver, seed=1: run(solver, mf(), seed=seed).run().results.d.I.values[[120, -1]]
    ode = prev('ode')
    ratio = ode/np.mean([prev('abm', seed) for seed in range(4)], axis=0)
    assert 1.0 < ratio[0] < 1.25 and 1.1 < ratio[1] < 1.4, ratio
    for solver in ['binomial', 'gillespie']: # The other compartmental solvers use the same force of infection
        assert np.isclose(prev(solver)[-1], ode[-1], rtol=0.1), solver
    with pytest.raises(ss.CapabilityError), ss.options.context(approximations='error'):
        run('ode', mf()).init()
    assert isinstance(run('auto', mf()).init().solver, ss.solvers.abm)


def test_between():
    sc.heading('Testing ss.SexualNet between two groups, with targets by group, and risk matching')
    kw = dict(n_agents=5000, start=2000, stop=2015, dt=ss.months(1), verbose=0)
    everyone = dict(debut=ss.constant(0), participation=ss.bernoulli(1.0))

    # Sex workers and clients, each with their own number of partners at a time
    sw = ss.SexualNet(name='sexwork', between=('fsw', 'client'), n_contacts=dict(fsw=4, client=0.8), dur=ss.months(6), **everyone)
    assert sw.directions == ['fsw2client', 'client2fsw']
    sim = ss.Sim(diseases=ss.SIS(beta=dict(sexwork=ss.peract(fsw2client=0.02, client2fsw=0.01))), networks=sw, **kw)
    sim.stratify(work={'none': 0.88, 'fsw': 0.02, 'client': 0.1}).run()
    net = sim.networks.sexwork
    level = sim.groups.level_raw([s.name for s in sim.groups.strats].index('work'))
    uids = sim.people.alive.uids
    degree = net.degrees(uids)
    assert (level[net.edges.p1] == 1).all() and (level[net.edges.p2] == 2).all() # Each partnership is a sex worker and a client
    assert np.isclose(degree[level[uids] == 1].mean(), 4, rtol=0.15) and np.isclose(degree[level[uids] == 2].mean(), 0.8, rtol=0.15)

    # Risk matching between men and women, with concurrency
    net = ss.SexualNet(n_contacts=ss.poisson(0.8), concurrency=0.1, dur=ss.years(2), match=dict(risk=0.8), **everyone)
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=net, **kw).stratify(risk={'low': 0.7, 'high': 0.3}).run()
    res = sim.networks[0].results
    assert np.isclose(res.match[-24:].mean(), 0.8, atol=0.05) and np.isclose(res.mean_degree[-24:].mean(), 0.8, rtol=0.1), (res.match[-24:].mean(), res.mean_degree[-1])
    assert np.isclose(res.concurrency[-24:].mean(), 0.1, rtol=0.15), res.concurrency[-24:].mean()


if __name__ == '__main__':
    T = sc.timer()
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
    T.toc()
