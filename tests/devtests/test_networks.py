"""
Test networks
"""

# %% Imports and settings
import pytest
import warnings
import sciris as sc
import numpy as np
import networkx as nx
import starsim as ss
import starsim.library as ssl
import scipy.stats as sps
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively
# ss.options.warnings = 'error' # For additional debugging

small = 100
medium = 1000

# %% Define the tests


def make_dhs_data(n=100, include_sexes=False):
    """ Helper to construct synthetic household data """
    age_strings = []
    sex_strings = []
    for i in range(n):
        household_size = np.random.randint(1, 6)
        ages = np.random.randint(0, 80, household_size)
        age_strings.append(sc.strjoin(ages))
        if include_sexes:
            sexes = np.random.choice([1, 2], household_size)  # DHS convention: 1=male, 2=female
            sex_strings.append(sc.strjoin(sexes))
    if include_sexes:
        return sc.dataframe(hh_id=np.arange(n), ages=age_strings, sexes=sex_strings)
    return sc.dataframe(hh_id=np.arange(n), ages=age_strings)


@sc.timer()
def test_household_coarse_pregnancy_dt():
    """ Regression test: a Pregnancy module running on a coarser dt than the sim must not corrupt
    HouseholdNet move-out timing.

    Move-out is evaluated once per pregnancy, so the resulting number of households should be
    ~invariant to the pregnancy module's dt. Previously, HouseholdNet compared
    pregnancy.ti_exit_P (an index in the pregnancy module's own timeline) against sim.ti; when
    pregnancy ran on a coarser dt (e.g. ss.Pregnancy(dt=ss.months(3))), pregnant non-heads became
    re-eligible far too often, greatly over-fragmenting households. """
    sc.heading('Testing HouseholdNet with a coarse-dt Pregnancy module...')
    dhs_data = make_dhs_data(n=medium)

    def run(preg_dt):
        household = ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=True, prob_move_out=0.5)
        preg = ss.Pregnancy(dt=preg_dt)
        sim = ss.Sim(n_agents=medium, diseases='sis', networks=household, demographics=preg,
                     start='2000-01-01', stop='2020-12-31', dt=ss.months(1), rand_seed=1)
        sim.run()
        return sim.networks[0].n_households

    n_month   = run(ss.months(1))
    n_quarter = run(ss.months(3))
    rel = abs(n_quarter - n_month) / n_month
    assert rel < 0.15, f'Coarse-dt pregnancy over-fragments households: monthly={n_month}, quarterly={n_quarter} (rel diff {rel:.1%})'

    return n_month, n_quarter


@sc.timer()
def test_partners():
    sc.heading('Testing lasting partnerships, and a target number of partners at a time')

    # Partnerships that form at a constant rate: a Poisson number of partners at a time, with the mean given
    S, I = ss.states(2)
    sis = ss.Disease(S >> I @ ss.prob(0.64), I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')
    sim = ss.Sim(sis, networks=ss.DegreeNet(n_contacts=0.7, dur=50), n_agents=2000, stop=300, verbose=0).run()
    e = sim.networks[0].edges
    degree = np.bincount(np.concatenate([e.p1, e.p2]), minlength=2000)
    assert np.isclose(degree.mean(), 0.7, rtol=0.1) and np.isclose(degree.var(), degree.mean(), rtol=0.15)
    assert sim.results.sis.I[-1] > 100 # Infection spreads through the partnerships

    # The targets of EpiModel's introductory network, from the library
    net = ss.DegreeNet(n_contacts=0.7, concurrency=0.22, max_contacts=3, dur=50)
    sim = ss.Sim(sis, networks=net, n_agents=medium, stop=400, verbose=0).run()
    net = sim.networks[0]
    res = net.results
    k = np.arange(4)
    assert np.isclose(net.dist_degree @ k, 0.7) and np.isclose(net.dist_degree[2:].sum(), 0.22) # The targets
    assert np.isclose(res.mean_degree[200:].mean(), 0.7, rtol=0.1) # ... are what the network has (partnerships last 50 steps, so 200 steps of 1,000 agents vary by about 5% between seeds)
    assert np.isclose(res.concurrency[200:].mean(), 0.22, rtol=0.1)
    assert net.degrees(sim.people.alive.uids).max() <= 3
    assert sim.results.sis.I[-1] > 100 # Infection spreads through the partnerships
    table = net.check(n_runs=2, output=True)
    assert list(table.index) == ['mean_degree', 'concurrency', 'max_contacts', 'dur']
    assert np.isclose(table.realized.dur, 50, rtol=0.15) # How long partnerships last, from how many there are and how many form

    # Acts in each partnership (acts=, per unit time): beta is then per act, so two acts a day at 0.4 transmit as one partnership at 1 - 0.6**2 a day
    def prev(acts, beta):
        S, I = ss.states(2)
        sis2 = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), recovery=I >> S @ ss.prob(0.05), init_n=dict(I=10), name='sis')
        return ss.Sim(sis2, networks=ss.DegreeNet(n_contacts=0.7, dur=50, acts=acts), n_agents=medium, stop=200, rand_seed=1, verbose=0).run().results.sis.I.values
    assert np.allclose(prev(ss.freqperday(2), ss.peract(0.4)), prev(None, ss.prob(1 - 0.6**2)))
    with pytest.raises(ValueError, match='per unit time'): # A bare number would be acts in a step, which depends on the step
        prev(2, ss.peract(0.4))

    # A number of partners for each level of a property, as EpiModel's nodefactor
    byrisk = ss.DegreeNet(n_contacts=ss.by(risk=[0.4, 1.6]), dur=50)
    sim2 = ss.Sim(sis, networks=byrisk, n_agents=medium, stop=200, verbose=0).stratify(risk=['low', 'high'], split=[0.8, 0.2]).run()
    level = np.asarray(sim2.networks[0].levels('risk'))[sim2.people.alive.uids]
    degree = sim2.networks[0].degrees(sim2.people.alive.uids)
    assert np.isclose(degree[level == 0].mean(), 0.4, rtol=0.25) and np.isclose(degree[level == 1].mean(), 1.6, rtol=0.25)
    table = sim2.networks[0].check(n_runs=2, output=True) # A row for each level, by name, and the target for everyone that of the mix of levels
    assert {'mean_degree[low]', 'mean_degree[high]'} <= set(table.index)
    assert np.isclose(table.target.mean_degree, 0.8*0.4 + 0.2*1.6, rtol=0.1)

    # Targets that cannot be met together are an error; partnerships are approximate under an ODE (see test_partnership_approx()), and a static network is approximated by the mean degree of its graph
    with pytest.raises(ValueError):
        ss.DegreeNet.degree_dist(0.7, concurrency=0.9, max_contacts=3)
    with pytest.raises(ss.CapabilityError, match='partner-change model'):
        ss.Sim(sis, networks=ss.DegreeNet(n_contacts=0.7, dur=50), total_pop=medium, solver='ode', stop=100, approx=False, verbose=0).run()
    ring = ss.Sim(sis, networks=ssl.StaticNet(graph=nx.cycle_graph(1000)), total_pop=1000, solver='ode', stop=10, verbose=0).init()
    assert ring.mixing.terms[0].contacts == 2 # Not n_contacts=10, whatever the graph

    # With births and deaths, partnerships also end when a partner dies, which is allowed for so the mean holds
    net = ss.DegreeNet(n_contacts=0.5, dur=50, name='main')
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')
    demog = [ss.Births(birth_rate=ss.peryear(50)), ss.Deaths(death_rate=ss.peryear(50))]
    vital = ss.Sim(sir, networks=net, demographics=demog, n_agents=2000, dt=ss.weeks(1), start=2000, stop=2030, verbose=0).run()
    assert abs(vital.networks.main.results.mean_degree[-500:].mean() - 0.5) < 0.03

    # Several networks of the same kind take their names from the keys of a dict
    S, I, R = ss.states('S I R')
    two = ss.Disease(infection=ss.Transition(S, I, rate=dict(main=0.1, casual=0.05) * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.05, name='two')
    keyed = ss.Sim(two, networks=dict(main=ss.SexualNet(dur=50), casual=ss.SexualNet(dur=5)), n_agents=500, stop=5, verbose=0).init()
    assert list(keyed.networks.keys()) == ['main', 'casual']
    return sim


def test_partner_targets():
    sc.heading('Testing other targets: partnerships within a level, and targets between men and women')

    # Most partnerships between people at the same level of risk
    S, I = ss.states('S I')
    sir = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), name='sir')
    sir.stratify(risk=['low', 'high'], split=[0.8, 0.2])
    for share in [None, 0.9]:
        net = ss.DegreeNet(n_contacts=0.7, dur=50, match=None if share is None else dict(risk=share), name='net')
        sim = ss.Sim(sir, networks=net, n_agents=5000, stop=300, verbose=0).run()
        net = sim.networks.net
        level = sim.groups.level_raw(sim.groups.dim_index('risk'))
        within = np.mean(level[net.edges.p1] == level[net.edges.p2])
        assert np.isclose(net.results.mean_degree[150:].mean(), 0.7, rtol=0.05)
        assert np.isclose(within, 0.68, atol=0.03) if share is None else np.isclose(within, share, atol=0.02) # By chance, 0.8² + 0.2²
    assert 'match' in net.check(n_runs=2, output=True).index

    # Within the levels of several properties at once, each with its own target (EpiModel's nodematch on each)
    sir.stratify(region=['a', 'b', 'c'], split=[0.5, 0.3, 0.2])
    net = ss.DegreeNet(n_contacts=0.7, dur=50, match=dict(risk=0.9, region=0.7), name='net')
    res = ss.Sim(sir, networks=net, n_agents=5000, stop=300, verbose=0).run().networks.net.results
    assert np.isclose(res.match_risk[150:].mean(), 0.9, atol=0.02) and np.isclose(res.match_region[150:].mean(), 0.7, atol=0.02) # By chance, 0.68 and 0.38

    # A target number of partners at a time between men and women, with concurrency
    duration = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))
    mf = ss.SexualNet(n_contacts=ss.poisson(0.7), concurrency=0.1, dur=duration, name='mf')
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=mf, n_agents=5000, start=2000, stop=2030, dt=ss.months(1), verbose=0).run()
    res = sim.networks.mf.results
    assert np.isclose(res.mean_degree[60:].mean(), 0.7, rtol=0.03) and np.isclose(res.concurrency[60:].mean(), 0.1, rtol=0.1)
    with pytest.raises(ValueError, match='n_contacts'):
        ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=ss.SexualNet(concurrency=0.1), n_agents=500, verbose=0).init()

    # Exactly one partner each: people with none always look (before, nobody did, and almost nobody had a partner)
    mf = ss.SexualNet(n_contacts=ss.constant(1), participation=1.0, debut=ss.constant(0), dur=ss.months(6), name='mf')
    sim = ss.Sim(networks=mf, n_agents=2000, start=2000, stop=2005, dt=ss.months(1), verbose=0).run()
    assert sim.networks.mf.results.mean_degree[-1] > 0.95

    # In an open population with long partnerships, newcomers join and the target holds (before, 0.65 against 0.87, and nobody born in the run took part)
    mf = ss.SexualNet(n_contacts=0.87, dur=ss.lognorm_ex(mean=ss.years(20), std=ss.years(5)), participation=ss.bernoulli(p=lambda self, sim, uids: np.where(sim.people.age[uids] < 10, 0, 0.9)), name='mf')
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=mf, demographics=[ss.Births(birth_rate=35), ss.Deaths(death_rate=15)], n_agents=5000, start=2000, stop=2040, verbose=0).run()
    ppl, mf = sim.people, sim.networks.mf
    born = (ppl.alive & (ppl.age >= 18) & (ppl.age < 40)).uids # All born during the run, when the probability at birth was 0
    assert np.isclose(mf.participant[born].mean(), 0.9, atol=0.04)
    assert np.isclose(mf.results.mean_degree[-10:].mean(), 0.87, rtol=0.05)
    return sim


def chance(beta, acts=ss.constant(ss.freqperyear(1)), dt=ss.years(1), steps=1, connectors=None, net=None, n=40_000, f2m=False, seed=1):
    """ The chance that a partnership with an infectious partner transmits over some steps, measured over many lasting partnerships (from men to women, or with f2m=True, women to men) """
    S, I = ss.states('S I')
    si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.5, name='si')
    net = net or ss.SexualNet
    net = net(dur=ss.constant(ss.years(100)), participation=ss.bernoulli(p=1.0), debut=ss.constant(0), acts=acts)
    sim = ss.Sim(si, networks=net, connectors=connectors, n_agents=n, start=2000, stop=2000 + steps*dt.years, dt=dt, rand_seed=seed, verbose=0).init()
    e = sim.networks[0].edges
    female = sim.people.female.raw
    men, women = np.where(female[e.p1], e.p2, e.p1), np.where(female[e.p1], e.p1, e.p2)
    src, trg = (women, men) if f2m else (men, women)
    before = sim.diseases.si.I.raw.copy()
    sim.run()
    at_risk = before[src] & ~before[trg]
    return sim.diseases.si.I.raw[trg[at_risk]].mean()


@sc.timer()
def test_per_act():
    sc.heading('Testing a probability per act on a network with acts, which keeps its meaning at any step')

    # A partnership with 80 acts a year transmits with 1 - 0.99**80 = 0.552 a year at per-act 0.01, whatever the step (before, 0.531 monthly and 0.465 weekly with whole acts per step, which a rate no longer gives)
    for dt in [ss.years(1/12), ss.weeks(1)]:
        p = chance(ss.peract(m2f=0.01, f2m=0), acts=ss.constant(ss.freqperyear(80)), dt=dt, steps=round(1/dt.years))
        assert np.isclose(p, 1 - 0.99**80, atol=0.025), f'{dt}: {p:.3f}'

    # And an epidemic over 10 years is the same at monthly and weekly steps (it was 38% against 12% with ss.probpermonth(0.01), and 37% against 29% with -log(1 - p) as a rate per month)
    def prev(dt, beta=ss.peract(0.01)):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=beta * I), init_prev=0.05, name='si')
        sims = [ss.Sim(si, networks=ss.SexualNet(dur=ss.lognorm_ex(mean=ss.years(2), std=ss.years(1))), n_agents=5000, start=2000, stop=2010, dt=dt, rand_seed=seed, verbose=0) for seed in [1, 2]]
        return np.mean([sim.run().results.si.I[-1]/5000 for sim in sims])
    monthly, weekly = prev(ss.years(1/12)), prev(ss.weeks(1))
    assert np.isclose(monthly, weekly, atol=0.03), f'{monthly:.3f} against {weekly:.3f}'
    assert prev(ss.years(1/12), 0.01) == monthly # A bare number on a network with acts is per act too (as in Starsim 3)

    # Each direction has its own probability per act: from women to men at 0.15 per act, 2, 4, and 8 acts in a step give Starsim 3's 0.278, 0.478, and 0.728 (v4 had 0.232, 0.346, and 0.429)
    for k,target in zip([2, 4, 8], [0.278, 0.478, 0.728]):
        p = chance(ss.peract(m2f=0.3, f2m=0.15), acts=ss.constant(ss.freqperyear(k)), f2m=True)
        assert np.isclose(p, target, atol=0.02), f'{k} acts: {p:.3f} against {target}'

    # The directions are named, so the order of a network's partners does not matter (before, [b, 0] infected only women on ss.SexualNet, and only men on HPVsim's network, which puts women in p1)
    class WomenFirst(ss.SexualNet):
        def append(self, edges=None, **kwargs):
            kwargs['p1'], kwargs['p2'] = kwargs['p2'], kwargs['p1']
            return super().append(edges, **kwargs)
    for net in [ss.SexualNet, WomenFirst]:
        assert chance(ss.peract(m2f=0.5, f2m=0), net=net, n=4000) > 0.3
        assert chance(ss.peract(m2f=0.5, f2m=0), net=net, n=4000, f2m=True) == 0
    assert ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=WomenFirst(), n_agents=200, verbose=0).init().networks[0].edges.p1[:10].tolist() # Women in p1, and it runs

    # rel_trans, rel_sus, and modifiers multiply the hazard per act: beta 0.03 with rel_trans 10 is 1 - 0.97**10 = 0.263 per act
    tenfold = lambda b: [ss.modify('si.rel_trans', multiply=10)] if b is None else [ss.modify('si.rel_trans', multiply=2), ss.modify('si.pars.beta', multiply=5)]
    p10 = chance(ss.peract(m2f=0.03, f2m=0), connectors=tenfold(None))
    assert np.isclose(p10, 1 - 0.97**10, atol=0.015), p10
    assert p10 == chance(ss.peract(m2f=0.03, f2m=0), connectors=tenfold(2)) # rel_trans 2 and beta times 5 is the same as rel_trans 10

    # A network written as code with Starsim 3's net_beta() keeps Starsim 3's numbers: the chance for the partnership in the step, which rel_sus then scales (0.44 at p=0.3, 6 acts, rel_sus=0.5, against 0.66 per act)
    class V3Net(ss.SexualNet):
        def net_beta(self, disease_beta=None, inds=None, disease=None): # As Starsim 3 calls it: for every edge (STIsim's takes uids= instead of inds=)
            return self.edges.beta*(1 - (1 - disease_beta)**self.edges.acts)
    half = lambda: [ss.modify('si.rel_sus', multiply=0.5)]
    core = chance(ss.peract(0.3), acts=ss.constant(ss.freqperyear(6)), connectors=half())
    coded = chance(ss.peract(0.3), acts=ss.constant(ss.freqperyear(6)), connectors=half(), net=V3Net)
    assert np.isclose(core, 1 - 0.7**3, atol=0.015) and np.isclose(coded, 0.5*(1 - 0.7**6), atol=0.015), (core, coded)
    return


@sc.timer()
def test_route_modifiers():
    sc.heading('Testing modifiers of a transmission on one route: condoms, needle programs')

    # A condom of efficacy 0.9 in half the acts multiplies the hazard by 1 - 0.5*0.9: p=0.1, 10 acts gives 1 - exp(-10*0.1054*0.55) = 0.440 (STIsim's split of the acts gives 0.438)
    condoms = lambda **kw: [ss.modify('si.infection', route='sexualnet', **kw)]
    p = chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.freqperyear(10)), connectors=condoms(efficacy=0.9, share=0.5), n=100_000)
    assert np.isclose(p, 1 - np.exp(10*np.log(0.9)*0.55), atol=0.01), p
    assert p == chance(ss.peract(m2f=0.1, f2m=0), acts=ss.constant(ss.freqperyear(10)), connectors=condoms(multiply=0.1, share=0.5), n=100_000) # efficacy= is multiply=1 - efficacy
    assert chance(ss.peract(m2f=0.1, f2m=0), connectors=condoms(multiply=0, share=1)) == 0 # Nothing on that route
    with pytest.raises(ValueError, match='efficacy'):
        ss.modify('si.infection', efficacy=0.9, multiply=0.1)

    # ... and leaves the other routes alone
    def run(mods=None, beta=ss.peryear(0.5)):
        S, I = ss.states('S I')
        si = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=ss.peract(0.05), randomnet=beta) * I), init_prev=0.05, name='si')
        return ss.Sim(si, networks=[ss.SexualNet(), ss.RandomNet(n_contacts=2)], connectors=mods, n_agents=2000, start=2000, stop=2005, verbose=0).run().results.si.I[-1]
    assert run(condoms(multiply=0, share=1)) > run(condoms(multiply=0, share=1), beta=0) > 0 # Infections only over the random network
    with pytest.raises(ValueError, match='not a route'):
        run(condoms(multiply=0, share=1) + [ss.modify('si.infection', multiply=0.5, route='needles')])
    with pytest.raises(ValueError, match='transmission'):
        run([ss.modify('si.rel_sus', multiply=0.5, route='sexualnet')])

    # Under an ODE, over a mixing pool, it is a multiplier on that route's hazard, by group with a share by group, as agents
    def pool(solver, mods, seed=1):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=dict(pool=ss.perday(0.03), home=ss.perday(0.02)) * I), recovery=ss.Transition(I, R, dur=ss.days(10)), init_prev=0.01, name='sir')
        nets = [ss.MixingPool(n_contacts=4, name='pool'), ss.MixingPool(n_contacts=2, name='home')]
        sim = ss.Sim(sir, networks=nets, interventions=mods, stratify=dict(risk=['low', 'high'], split=[0.7, 0.3]), total_pop=10_000, n_agents=10_000, solver=solver, unit='day', stop=100, rand_seed=seed, verbose=0)
        return sim.run().results.sir.R.values
    share = lambda: [ss.modify('sir.infection', multiply=0.2, share=ss.by(risk=[0.3, 0.9]), route='pool')]
    ode = pool('ode', share())
    assert np.allclose(ode, pool('ode', [ss.modify('sir.infection', multiply=ss.by(risk=[1 - 0.3*0.8, 1 - 0.9*0.8]), route='pool')])) # The same as its multiplier, 1 - share*(1 - multiply)
    ratio = lambda solver, seeds: np.mean([pool(solver, share(), seed)[-1] for seed in seeds])/np.mean([pool(solver, None, seed)[-1] for seed in seeds])
    ode_ratio, abm_ratio = ratio('ode', [1]), ratio('abm', range(12)) # 12 seeds, since near its threshold one run's epidemic varies a lot (seeds 0-3 alone gave 0.22 here); each solver's own epidemic (agents have about 8% fewer infections here either way), cut by the same share: 0.26
    assert ode_ratio < 0.5 and np.isclose(abm_ratio, ode_ratio, rtol=0.1), (abm_ratio, ode_ratio)
    return


@sc.timer()
def test_needles():
    sc.heading('Testing a needle-sharing network beside a sexual one (injecting drug use)')
    def run(mf_beta, needles_beta=ss.peract(0.006), nsp=False):
        needles = ss.DegreeNet(name='needles', where=ss.ppl.pwid, n_contacts=2, dur=ss.months(6), acts=ss.freqpermonth(20)) # Among people who inject only; no sides, so one value
        S, I = ss.states('S I')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=dict(sexualnet=mf_beta, needles=needles_beta) * I), init_prev=0.05, name='hiv')
        intv = ss.modify(hiv.infection, multiply=0.2, share=0.5, route='needles', start=2005) if nsp else None # Needle and syringe programs: half of injections with a clean needle, 80% less risk
        sim = ss.Sim(hiv, networks=[ss.SexualNet(), needles], interventions=intv, n_agents=5000, start=2000, stop=2010, dt=ss.months(1), verbose=0)
        sim.stratify(pwid=[False, True], split=[0.9, 0.1]).init()
        sim.before = sim.diseases.hiv.I.raw.copy()
        return sim.run()
    sim = run(0)
    pwid = sim.people.pwid.raw.astype(bool)
    net = sim.networks.needles
    assert pwid[net.edges.p1].all() and pwid[net.edges.p2].all() and np.isclose(2*len(net)/pwid[sim.people.alive.uids].sum(), 2, rtol=0.1) # Only people who inject, with 2 partners each on average
    new = sim.diseases.hiv.I.raw & ~sim.before
    assert not np.any(new & ~pwid) and np.any(new & pwid) # Nobody outside them is infected with beta 0 on sexualnet
    low = ss.peract(0.0005)
    assert run(ss.peract(m2f=0.001, f2m=0.0005), low, nsp=True).results.hiv.infection.cum[-1] < run(ss.peract(m2f=0.001, f2m=0.0005), low).results.hiv.infection.cum[-1]
    return


@sc.timer()
def test_partnership_approx():
    sc.heading('Testing the partner-change model: lasting partnerships under the ODE, binomial, and Gillespie solvers')
    def sis(beta, dur=None):
        S, I = ss.states('S I')
        return ss.Disease(infection=ss.Transition(S, I, rate=beta * I), **(dict(recovery=ss.Transition(I, S, dur=dur)) if dur else {}), init_prev=0.05, name='d') # SI with no dur
    def run(solver, disease, net, dt=ss.months(1), stratify=None, seed=1, n=5000, **kw):
        sim = ss.Sim(disease, networks=net, n_agents=n, total_pop=n, start=2000, stop=2020, dt=dt, solver=solver, rand_seed=seed, verbose=0, **kw)
        return sim.stratify(**(stratify or dict(sex=['female', 'male'])))

    # With partnerships that end at a constant rate and no way out of the infectious state, the chance over a partnership is H/(H + 1/D̄)
    sim = run('ode', sis(ss.peract(0.05)), ss.DegreeNet(name='pairs', n_contacts=1, dur=ss.years(2), acts=ss.freqperyear(10)), stratify=dict(risk=['low', 'high'])).init()
    H = 10*-np.log(0.95)
    assert sim.diseases.d.describe(output=True).count('partner-change model (approx)') == 1
    assert np.isclose(sim.diseases.d.engine.partnership_info[('infection', 'I', 'pairs', 0)][0], H/(H + 1/2)), 'H/(H + 1/D)'

    # Prevalence at 10 and 20 years under the ODE against agents (4 runs here, 1 for gono; 20 runs of 10,000 agents gave the ratios in the comments): it is close
    # for risk groups with short partnerships, too high for long monogamous partnerships with a long infection, and too low with concurrency
    mf = lambda **kw: ss.SexualNet(participation=ss.bernoulli(p=1.0), debut=ss.constant(0), **kw)
    cases = dict(
        risk = ([0.9, 1.1], [0.9, 1.1], dict(disease=sis(ss.peract(0.002)), stratify=dict(risk=['low', 'medium', 'high'], split=[0.8, 0.15, 0.05]), # 0.999, 0.995
                 net=ss.DegreeNet(n_contacts=ss.by(risk=[0.5, 1.5, 4]), match=dict(risk=0.7), dur=ss.months(6), acts=ss.freqperyear(80)))),
        gono = ([0.08, 0.4], [0.06, 0.4], dict(disease=sis(ss.peract(m2f=0.05, f2m=0.03), ss.months(4)), dt=ss.weeks(1), # 0.19, 0.13: concurrency (Poisson partners), which the approximation leaves out
                 net=ss.SexualNet(participation=ss.bernoulli(p=1.0), debut=ss.constant(0), n_contacts=1, dur=ss.lognorm_ex(mean=ss.weeks(4), std=ss.weeks(2)), acts=ss.freqperweek(2)))),
        hiv  = ([1.0, 1.25], [1.1, 1.4], dict(disease=sis(ss.peract(m2f=0.004, f2m=0.002)), # 1.105, 1.247: infection kept within pairs, which the approximation spreads
                 net=mf(dur=ss.lognorm_ex(mean=ss.years(5), std=ss.years(3)), acts=ss.freqperyear(80)))),
    )
    for name,(at10,at20,kw) in cases.items():
        prev = lambda solver, seed=1: run(solver, seed=seed, **sc.dcp(kw)).run().results.d.I.values[[120, -1] if name != 'gono' else [522, -1]]/5000
        ratio = prev('ode')/np.mean([prev('abm', seed) for seed in range(1 if name == 'gono' else 4)], axis=0) # One run is enough for the weekly gono, whose ratios are far from its bounds (0.18-0.26 and 0.11-0.13 over 8 seeds)
        assert at10[0] < ratio[0] < at10[1] and at20[0] < ratio[1] < at20[1], f'{name}: {ratio}'

    # The other compartmental solvers use the same force of infection; approx=False raises, sides need a dimension, and 'auto' keeps agents
    hiv = cases['hiv'][2]
    ode = run('ode', **sc.dcp(hiv)).run().results.d.I[-1]
    for solver in ['binomial', 'gillespie']: # Stochastic, in 5,000 people
        assert np.isclose(run(solver, **sc.dcp(hiv)).run().results.d.I[-1], ode, rtol=0.1), solver
    with pytest.raises(ss.CapabilityError, match='abm'):
        run('ode', approx=False, **sc.dcp(hiv)).init()
    with pytest.raises(ss.CapabilityError, match=r'stratify\(sex='):
        ss.Sim(sis(0.01), networks=ss.SexualNet(), solver='ode', verbose=0).init()
    assert run('auto', **sc.dcp(hiv)).init().solver == 'abm'
    return


@sc.timer()
def test_between():
    sc.heading('Testing ss.SexualNet between two groups, with targets by group and risk matching, and subclasses that pair their own')
    kw = dict(n_agents=5000, start=2000, stop=2015, dt=ss.months(1), verbose=0)
    everyone = dict(debut=ss.constant(0), participation=ss.bernoulli(1.0))

    # Sex workers and clients, each with their own number of partners at a time
    sw = ss.SexualNet(name='sexwork', between=('fsw', 'client'), n_contacts=dict(fsw=4, client=0.8), dur=ss.months(6), **everyone)
    assert sw.directions == ['fsw2client', 'client2fsw']
    sim = ss.Sim(diseases=ss.SIS(beta=dict(sexwork=ss.peract(fsw2client=0.02, client2fsw=0.01))), networks=sw, **kw)
    sim.stratify(work=['none', 'fsw', 'client'], split=[0.88, 0.02, 0.1]).run()
    net = sim.networks.sexwork
    level = sim.groups.level_raw([s.name for s in sim.groups.strats].index('work'))
    uids = sim.people.alive.uids
    degree = net.degrees(uids)
    assert (level[net.edges.p1] == 1).all() and (level[net.edges.p2] == 2).all() # Each partnership is a sex worker and a client
    assert np.isclose(degree[level[uids] == 1].mean(), 4, rtol=0.15) and np.isclose(degree[level[uids] == 2].mean(), 0.8, rtol=0.15), (degree[level[uids] == 1].mean(), degree[level[uids] == 2].mean())
    with pytest.raises(ValueError, match='each group of between'):
        ss.Sim(diseases='sis', networks=ss.SexualNet(n_contacts=dict(male=1, fsw=2), dur=ss.years(1)), n_agents=200, verbose=0).init()

    # Risk matching between men and women
    net = ss.SexualNet(n_contacts=0.8, dur=ss.years(2), match=dict(risk=0.8), **everyone)
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=net, **kw).stratify(risk=['low', 'high'], split=[0.7, 0.3]).run()
    res = sim.networks[0].results
    assert np.isclose(res.match[-24:].mean(), 0.8, atol=0.05) and np.isclose(res.mean_degree[-1], 0.8, rtol=0.1), (res.match[-24:].mean(), res.mean_degree[-1])

    # A subclass that forms its own partnerships gets none from the base, and only between= of its parameters; between men, its error suggests between='mm'
    class OwnMSM(ss.SexualNet):
        def step(self):
            self.end_pairs()
            if not len(self):
                men = self.sim.people.male.uids
                n = len(men)//2
                self.append(p1=men[:n], p2=men[n:2*n], beta=np.ones(n), dur=np.full(n, 100.0), acts=np.ones(n))
    assert list(OwnMSM().pars.keys()) == ['memory', 'between'] # And memory=, of every dynamic network
    sim = ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm'), n_agents=500, verbose=0).init()
    assert len(sim.networks.msm) == 0
    with pytest.raises(ValueError, match="between='mm'"):
        ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm'), n_agents=500, verbose=0).run()
    assert len(ss.Sim(diseases=ss.SIS(beta=ss.peract(0.01)), networks=OwnMSM(name='msm', between='mm'), n_agents=500, stop=2002, verbose=0).run().networks.msm)
    return


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Run tests
    betw = test_between()
    ehh2 = test_household_coarse_pregnancy_dt()
    part = test_partners()
    targ = test_partner_targets()
    peract = test_per_act()
    route = test_route_modifiers()
    needles = test_needles()
    pa = test_partnership_approx()

    T.toc()
