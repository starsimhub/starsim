"""
Switching demo for the LASER seat: 3.3M agents across 149 places with life tables and seeding.

Passes when: within 2 to 3x of LASER threaded, at about 50 bytes per agent.
Blocked today by: N25.
Scaled down to 100,000 and 400,000 agents over 149 synthetic places; speed and memory are compared per agent-step and per agent, which do not depend on the size.
"""
import tracemalloc
import numpy as np
import pytest
import sciris as sc
import starsim as ss

ref_ns_per_agent_step = 1.1 # LASER on 20 threads, England and Wales within 150 km of Norwich (3.3M agents, 149 places); switching/laser.md, "The second week" §1
ref_bytes_per_agent   = 30  # LASER's peak memory per agent; same section
target_bytes          = 50  # The realistic target for v4 (int32 UIDs, int16 day timers, exit arrays only where needed); "The second week" §2
n_places = 149


def make_places(total, seed=1):
    """ A synthetic stand-in for the places within 150 km of Norwich: lognormal populations and random positions """
    rng = np.random.default_rng(seed)
    pop = 0.5 + rng.lognormal(0, 0.8, n_places)
    x, y = rng.uniform(0, 300, (2, n_places))
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    with np.errstate(divide='ignore'):
        gravity = pop**0.25/dist**1.56 # LASER's gravity model
    np.fill_diagonal(gravity, 0)
    gravity *= 0.045/gravity.sum(axis=1).mean()
    gravity *= np.minimum(1, 0.2/gravity.sum(axis=1))[:, None] # The mean share exported is 0.045, and at most 0.2
    return sc.objdict(names=[f'p{i}' for i in range(n_places)], popsize=np.round(pop*total/pop.sum()), dist=dist, gravity=gravity)


def make_sim(total, days=30, declared=False):
    """ The England and Wales measles model; with declared=True, with the life table and seeding the demo declares """
    d = make_places(total)
    ages = np.arange(89)
    pyramid = np.exp(-0.017*ages) # The stable age pyramid for the birth rate
    deaths = dict(deaths=ss.death(lifetable=ss.lifetable(ages, pyramid*0.017))) if declared else {} # Without them, no deaths: LASER draws each age at death once, which costs nothing per step
    measles = ss.Disease(
        infection = ss.transmission('S -> E', beta=ss.perday(3.9), source='I'),
        onset     = ss.progression('E -> I', dur=ss.erlang(k=40, mean=ss.days(10))),
        recovery  = ss.progression('I -> R', dur=ss.normal(7, 2, unit='days')),
        births    = ss.birth('-> S', rate=ss.peryear(0.017)),
        init      = dict(I=ss.by(patch=[3]*n_places), R=0.95),
        name      = 'measles',
        **deaths,
    )
    measles.stratify(patch=ss.geo(names=d.names, pop=d.popsize, distances=d.dist), mobility=ss.Mobility(matrix=d.gravity, kind='force'))
    seeding = None
    if declared: # Three people seeded every 30 days; and age groups, for the life table as compartments
        measles.stratify(age=[0, 5, 15, 30, 50, 70])
        seeding = ss.Delivery(move='S -> I', capacity=3, start=ss.days(30), interval=ss.days(30), name='seeding')
    people = ss.People(int(d.popsize.sum()), age_data=np.array([ages, pyramid]).T)
    return ss.Sim(measles, people=people, interventions=seeding, unit='day', stop=days, verbose=0)


def test_laser_declared():
    sc.heading('LASER: life-table deaths and seeding, declared, as agents and as compartments')
    agents = make_sim(100_000, days=365, declared=True).run()
    comps = make_sim(100_000, days=365, declared=True).new(solver='binomial').run()
    seeded = agents.results.seeding.delivered.values
    assert np.isclose(seeded.sum(), 3*12), f'Expected 3 people seeded every 30 days, not {seeded.sum()} in a year'
    for sim in [agents, comps]:
        assert sim.results.measles.deaths.values.sum() > 0, 'The life table should give deaths'
    inf = [sim.results.measles.infection.values.sum() for sim in [agents, comps]]
    assert np.isclose(*inf, rtol=0.2), f'The same text as agents and as compartments should agree: {inf[0]:,.0f} against {inf[1]:,.0f} infections' # One run of each, seeded by 3 people a month, so chance and the order-dt difference (2,686 against 3,027)
    return agents, comps


def peak_memory(total):
    sim = make_sim(total)
    tracemalloc.start()
    sim.init().run()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


@pytest.mark.xfail(reason='N25: about 114 bytes per agent at the peak here (142 before 16-bit step numbers, 240 before the fused passes and 32-bit UIDs): the People arrays (42 bytes: 64-bit auids, uid, slot, parent, age, ti_dead, ti_removed, scale, patch), the disease (26: rel_sus, rel_trans, four states, their 16-bit times, the scheduled exits), the groups kept (12), all with a fifth more room for births, and the temporaries of a step', strict=False)
def test_laser_memory():
    sc.heading('LASER: memory per agent')
    n1, n2 = 50_000, 200_000
    per_agent = (peak_memory(n2) - peak_memory(n1))/(n2 - n1) # The extra memory per extra agent, which leaves out fixed costs
    assert per_agent < target_bytes, f'{per_agent:.0f} bytes per agent, against LASER\'s {ref_bytes_per_agent} and the target of {target_bytes}'
    return per_agent


@pytest.mark.xfail(reason='N25: about 5 ns per agent-step here (23 before the threaded passes), of which about 3 is growing the arrays for the first births, which a 20-day run does not spread out; the full England and Wales model (3.3M agents, 2 years) takes 7 to 8.5 s, against 2.75 s for LASER', strict=False)
def test_laser_speed():
    sc.heading('LASER: time per agent-step')
    days, times = 20, {}
    make_sim(300_000, days=2).run() # Warm up, at the larger size, whose passes over the agents are threaded
    for total in [100_000, 300_000]:
        for rep in range(3): # The best of three, since the machine may be busy
            sim = make_sim(total, days=days).init()
            T = sc.timer()
            sim.run()
            times[total] = min(times.get(total, np.inf), T.toc(output=True))
    ns = (times[300_000] - times[100_000])/(200_000*days)*1e9 # The marginal cost, which leaves out the per-step Python overhead
    assert ns < 3*ref_ns_per_agent_step, f'{ns:.1f} ns per agent-step, more than 3x LASER threaded ({ref_ns_per_agent_step} ns)'
    return ns


if __name__ == '__main__':
    sims = test_laser_declared()
    mem  = test_laser_memory()
    ns   = test_laser_speed()
