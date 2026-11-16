"""
Test interventions on declared models: modifiers, products, and their delivery (the slower and more exhaustive tests)
"""

# %% Imports and settings
import numpy as np
import sciris as sc
import starsim as ss
from corpus import make_sir, make_sis, check_close, check_agents
import corpus

kw = dict(verbose=0, unit='day', total_pop=50_000)
solvers = ['ode', 'binomial', 'abm']


def run_each(make, stop=150, **kwargs):
    """ Run the sim that `make` returns with each solver """
    return corpus.run_each(make, stop=stop, **kw, **kwargs)


# %% Modifiers

def test_branch_modifiers():
    sc.heading('Testing modifiers on a branch chosen with p=: they change its probability, and the other branches take the rest')
    def make(solver, intv=None):
        S, I, H, R = ss.states('S I H R')
        dis = ss.Disease(
            infection = ss.Transition(S, I, rate=0.3 * I),
            severe    = ss.Transition(I, H, p=0.2, dur=5),
            mild      = ss.Transition(I, R, p=0.8, dur=10),
            discharge = ss.Transition(H, R, dur=7),
            init_prev = 0.01, name='dis',
        )
        pars = dict(n_agents=20_000) if solver == 'abm' else {}
        return ss.Sim(dis, interventions=intv(dis) if intv is not None else None, solver=solver, stop=200, **kw, **pars).run().results.dis
    frac = lambda res: res.severe.sum()/(res.severe.sum() + res.mild.sum())
    halve = lambda dis: ss.modify(dis.severe, multiply=0.5, name='halve')
    vaccine = lambda dis: ss.vaccinate(ss.Vaccine(efficacy=dict(severe=0.75)), when=0, name='vx')
    for solver in solvers:
        assert np.isclose(frac(make(solver)), 0.2, rtol=0.05), solver
        assert np.isclose(frac(make(solver, halve)), 0.1, rtol=0.05), solver
        assert np.isclose(frac(make(solver, vaccine)), 0.05, rtol=0.1), solver

    # An efficacy on one destination of a transition with several halves its probability, without changing the dwell time
    for solver in ['ode', 'abm']:
        P, C, R = ss.states('P C R')
        hpv = ss.Disease(P, C, R, precin=ss.Transition(P, {C: 0.2, R: 0.8}, dur=ss.years(2)), default_state='P', name='hpv')
        txvx = ss.vaccinate(ss.Vaccine(efficacy={'precin.C': 0.5}), when=[2000], coverage=0.5, name='txvx')
        size = dict(n_agents=20_000) if solver == 'abm' else dict(total_pop=20_000)
        res = ss.Sim(hpv, interventions=txvx, start=2000, stop=2012, dt=ss.years(0.25), solver=solver, verbose=0, **size).run().results.hpv
        assert np.isclose(res.C[-1]/20_000, 0.15, atol=0.01) # Half at 0.1, half at 0.2
        assert np.isclose(res.P[8]/20_000, np.exp(-1), rtol=0.1) # Left P after two years on average, as without it
    return res


# %% Products

def test_vaccinate():
    sc.heading('Testing vaccination: a campaign with a cost, and waning and rounds')
    base = ss.Sim(make_sir(), solver='ode', stop=150, **kw).run()
    def campaign(**kwargs):
        rounds = ss.vaccinate(ss.Vaccine(efficacy=0.9, cost=2.0), when=[5], coverage=0.6)
        return ss.Sim(make_sir(), interventions=rounds, **kwargs)
    for sim in run_each(campaign).values():
        res = sim.results.rounds
        assert np.isclose(res.delivered.sum(), 30_000, rtol=0.02) and np.isclose(res.coverage[5], 0.6, atol=0.02)
        assert np.isclose(res.cost.sum(), 2*res.delivered.sum())
        assert sim.results.sir.infection.cum[-1] < 0.5*base.results.sir.infection.cum[-1]
        assert [strat.name for strat in sim.groups.strats] == ['has_vaccine'] # Holding the vaccine is a property of people

    # Protection that is lost, and given again in the next round; with efficacy against two things
    def waning(**kwargs):
        vaccine = ss.Vaccine(efficacy=dict(acquisition=0.8, transmission=0.5), waning=ss.days(30))
        rounds = ss.vaccinate(vaccine, when=ss.schedule(every=ss.days(50), start=20), coverage=0.8)
        return ss.Sim(make_sis(), interventions=rounds, **kwargs)
    sims = run_each(waning, stop=200)
    for sim in sims.values():
        delivered = sim.results.rounds.delivered
        assert list(np.flatnonzero(delivered)) == [20, 70, 120, 170] # The steps that start at 20, 70, ...
        expected = 0.8*(10_000 + 40_000*(1 - np.exp(-50/30))) # Those missed in the last round, and those who have lost their protection since
        assert np.isclose(delivered[70], expected, rtol=0.05)
    check_close(sims, lambda sim: sim.results.sis.I[100])

    return sims


def test_treat():
    sc.heading('Testing treatment: changing rates, and the states a cure takes people out of')
    # A treatment that changes rates
    def make_sird():
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), death=ss.death(I, rate=0.02), init_prev=0.01, name='sird')
    def rates(**kwargs):
        sird = make_sird()
        program = ss.treat(ss.Treatment(efficacy=dict(transmission=0.5, mortality=0.9)), who=sird.I, coverage=0.8)
        return ss.Sim(sird, interventions=program, **kwargs)
    untreated = ss.Sim(make_sird(), solver='ode', stop=150, **kw).run()
    sims = run_each(rates)
    for sim in sims.values():
        assert sim.results.sird.death.sum() < 0.6*untreated.results.sird.death.sum()
    check_close(sims, lambda sim: sim.results.program.delivered.sum()/sim.results.sird.infection.cum[-1]) # Most of the people infected, whenever that was

    # A cure takes people out of every infected state, not only the sources of the transition named
    for solver in ['ode', 'abm']:
        S, A, Y = ss.State(), ss.State(init=0.5), ss.State(init=0.5)
        sti = ss.Disease(S=S, A=A, Y=Y, infection=S >> A @ ss.perday(1e-9)*(A + Y), symp=A >> Y @ ss.days(1e5), clear_a=A >> S @ ss.days(1000), clear_y=Y >> S @ ss.days(1000), name='sti')
        res = ss.Sim(sti, interventions=ss.treat(ss.Treatment(), when=[5], name='tx'), unit='day', stop=10, solver=solver, n_agents=2000, total_pop=2000, verbose=0).run().results.sti
        assert res.A[-1] == 0 and res.Y[-1] == 0
    return sims


# %% Tests and the steps they trigger

def test_screen():
    sc.heading('Testing screening in rounds, with several steps on its results, in any order, under every solver')
    def rounds(**kwargs):
        screening = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.95), when=ss.schedule(every=7, start=30), coverage=0.4)
        treatment = ss.treat(ss.Treatment(cure=0.95), when=screening.positive, coverage=0.8)
        v1 = ss.vaccinate(ss.Vaccine(efficacy=0.5, name='vx1'), when=screening.positive)
        v2 = ss.vaccinate(ss.Vaccine(efficacy=0.5, name='vx2'), when=screening.positive)
        return ss.Sim(make_sis(), interventions=[treatment, v1, screening, v2], **kwargs) # In any order: the follow-ups run after the test
    sims = run_each(rounds)
    for sim in sims.values():
        res = sim.results
        ti = 37 # The step from time 37
        n_I = res.sis.I[ti] # At the time, the start of the step
        pos = res.screening.positive[ti]
        assert np.isclose(res.screening.delivered[ti], 20_000, rtol=0.03)
        assert np.isclose(pos, 0.4*(0.9*n_I + 0.05*(50_000 - n_I)), rtol=0.05) # True and false positives: under every solver the screening sees the people at the start of the step
        assert np.isclose(res.treatment.delivered[ti], 0.8*pos, rtol=0.05) # The follow-up is a step of its own, on the step of the test
        assert res.v1.delivered[ti] > 0 and np.isclose(res.v1.delivered[ti], res.v2.delivered[ti], rtol=0.01) # Each finds the people where the others moved them: not fewer for the later step
        assert np.isclose(pos + res.screening.negative[ti], res.screening.delivered[ti])
        assert np.isclose(res.treatment.cured[ti] + res.treatment.failed[ti] + res.treatment.unneeded[ti], res.treatment.delivered[ti])
        assert np.isclose(sim.summarize('screening_coverage'), 0.4, rtol=0.03) # Over the rounds, not averaged with the steps between them
    check_close(sims, lambda sim: sim.results.screening.positive.sum())
    return sims


def test_triggered():
    sc.heading('Testing steps triggered by events as agents: a delayed result, and a union and an intersection of events')
    def make(**kwargs):
        slow = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.9, result_delay=ss.days(3)), when=[40], coverage=0.5)
        cure = ss.treat(ss.Treatment(cure=1.0), when=slow.positive)
        protect = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=slow.negative)
        a = ss.screen(ss.Diagnostic(sensitivity=1.0, specificity=1.0, name='dx_a'), when=[20], coverage=0.5, name='a')
        b = ss.screen(ss.Diagnostic(sensitivity=1.0, specificity=1.0, name='dx_b'), when=[20], coverage=0.5, name='b')
        either = ss.vaccinate(ss.Vaccine(efficacy=0.0, name='v_either'), when=a.positive | b.positive, name='either')
        both = ss.vaccinate(ss.Vaccine(efficacy=0.0, name='v_both'), when=a.positive & b.positive, name='both')
        return ss.Sim(make_sis(), interventions=[either, both, slow, cure, protect, a, b], **kwargs)
    check_agents(make, stop=60, **kw)
    res = make(solver='abm', stop=60, **kw).run().results
    assert list(np.flatnonzero(res.slow.delivered)) == [40] and list(np.flatnonzero(res.slow.positive)) == [43] # The follow-ups happen when the result arrives
    assert np.isclose(res.cure.delivered[43], res.slow.positive[43])
    assert np.isclose(res.slow.positive[43] + res.protect.delivered[43], res.slow.delivered[40], rtol=0.01) # Everyone tested has one result or the other
    assert np.isclose(res.either.delivered[20]/res.sis.I[20], 0.75, atol=0.03) # Half of the infected tested by each, independently: 75% by either, once each
    assert np.isclose(res.both.delivered[20]/res.sis.I[20], 0.25, atol=0.03)
    return res


def test_coinfection_compartments():
    sc.heading('Testing two diseases that modify each other, as compartments')
    def make(solver='ode', coinfect=True):
        S, I = ss.states('S I')
        hiv = ss.Disease(infection=ss.Transition(S, I, rate=0.15 * I), death=ss.death(I, rate=0.05), init_prev=0.05)
        S, I = ss.states('S I')
        syph = ss.Disease(infection=ss.Transition(S, I, rate=1.0 * I), cure=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.05)
        coinfection = [
            ss.modify(hiv.rel_trans, multiply=3.0, who=syph.I, name='hiv_trans'), # Both diseases have a state called I, so each condition remembers its own
            ss.modify(syph.rel_sus, multiply=2.0, who=hiv.I, name='syph_sus'),
        ]
        return ss.Sim(diseases=[hiv, syph], connectors=coinfection if coinfect else None, solver=solver, start=2000, stop=2030, dt=ss.years(0.1), total_pop=20_000, verbose=0)
    sims = sc.obj({solver:make(solver).run() for solver in solvers})
    base = make(coinfect=False).run()

    # The states of the two diseases are tracked jointly, so they have the same people, and the results of each can be split by the other
    res = sims.ode.results
    assert list(sims.ode.diseases.keys()) == ['hiv', 'syph'] # Named from their variables
    assert np.allclose(res.hiv.S + res.hiv.I, res.syph.S + res.syph.I) # Deaths from HIV are deaths of people with syphilis too
    with_syph = res.hiv.I.by('syph').values[-1] / (res.hiv.S.by('syph').values[-1] + res.hiv.I.by('syph').values[-1])
    assert with_syph[1] > with_syph[0] # More HIV among people with syphilis
    assert res.hiv.I[-1] > 1.2*base.results.hiv.I[-1] and res.syph.I[-1] > 2*base.results.syph.I[-1]
    for solver in ['binomial', 'abm']:
        for key in ['hiv', 'syph']:
            assert np.isclose(sims[solver].results[key].I[-1], res[key].I[-1], rtol=0.15) # As agents, syphilis is about 5% above the ODE on average, and varies by about 3% between runs
    stepped = make('binomial').run(until=2030) # The stochastic run takes the diseases together, by one program, whole or step by step
    assert sims.binomial._compiled_reason() is None and np.array_equal(stepped.results.syph.I, sims.binomial.results.syph.I)

    return sims


# %% Waning

def test_waning_functions():
    sc.heading('Testing waning: ss.boxcar() under agents and the ODE, and an effect while in some states')
    def make(solver):
        S, I = ss.states('S I')
        mal = ss.Disease(infection=S >> I @ ss.peryear(2.0), recovery=I >> S @ ss.days(100), name='mal')
        vx = ss.Vaccine(efficacy=0.8, waning=box, name='vx')
        size = dict(n_agents=10_000) if solver == 'abm' else dict(total_pop=10_000)
        return ss.Sim(mal, interventions=ss.vaccinate(vx, when=[ss.days(20), ss.days(400)], coverage=0.8, name='rounds'), unit='day', stop=600, solver=solver, verbose=0, **size).run()
    box = ss.boxcar(plateau=ss.days(180), decay=ss.days(200))
    t = np.array([100, 300, 450, 600])
    sims = {solver:make(solver) for solver in ['abm', 'ode']}
    prev = {solver:sim.results.mal.I[t]/10_000 for solver,sim in sims.items()}
    assert np.allclose(prev['abm'], prev['ode'], atol=0.03), prev
    ode = sims['ode']
    assert isinstance(ode.solver, ss.solvers.ode) and len(ode.groups.strats[ode.groups.dim_index('has_vx')].labels) == 12 # False, 10 stages of the plateau, decaying
    assert np.allclose(box(ss.days(np.array([0, 179, 380]))), [1, 1, np.exp(-1)])

    # An effect only while in some states: held until people leave them
    S, I, R = ss.states('S I R')
    sir = ss.Disease(infection=S >> I @ ss.perday(0.25)*I, recovery=I >> R @ ss.days(10), waning=R >> S @ ss.days(20), init_prev=0.01, name='sir')
    drug = ss.Treatment(efficacy=dict(transmission=0.9), during=sir.I, name='drug')
    sim = ss.Sim(sir, interventions=ss.treat(drug, who=sir.I, rate=ss.perday(0.2), name='tx'), unit='day', stop=50, solver='abm', n_agents=5000, verbose=0).run()
    held = sim.people['has_drug'].raw > 0
    left = held & ~sim.diseases.sir.I.raw[:len(held)]
    assert held.sum() > 0 and left.sum() <= sim.results.sir.recovery[-1] # Only those who left I in the last step still hold it
    return sim


# %% A whole run in the kernels

def test_whole_run():
    sc.heading('Testing deliveries in a whole run in the kernels: the same as step by step, in many runs at once, and in JAX')
    def make(**kwargs):
        sir = make_sir().stratify(age=[0, 15, 65], protected={False: 1, True: 0})
        halved = ss.modify(sir.infection, multiply=ss.by(protected=[1.0, 0.5]))
        routine = ss.Vaccine(efficacy=0.9, waning=ss.days(60))
        offer = ss.vaccinate(routine, who=sir.S, when=ss.schedule(start=10), coverage=ss.by(age=[0.6, 0.3, 0.0])) # Each person offered it once
        rounds = ss.vaccinate(ss.Vaccine(efficacy=0.7), when=[20, 40], coverage=0.5)
        rollout = ss.Event(dict(protected=True), when=[30], p=0.1) # Along a stratification
        seed = ss.Event(sir.S >> sir.I, when=[25], p=0.01, name='seed') # Between states
        return ss.Sim(sir, interventions=[offer, rounds, rollout, seed], connectors=halved, stop=100, **kw, **kwargs)
    whole = make(solver='ode').init()
    assert whole._compiled_reason() is None, whole._compiled_reason()
    whole.run()
    stepped = make(solver='ode').init()
    stepped._compiled_reason = lambda: 'stepped' # As the loop takes it, one step at a time
    stepped.run()
    for mod,key in [('sir', 'S'), ('sir', 'I'), ('sir', 'R'), ('offer', 'delivered'), ('offer', 'eligible'), ('rounds', 'coverage'), ('rollout', 'delivered'), ('seed', 'delivered')]:
        a, b = whole.results[mod][key], stepped.results[mod][key]
        assert np.allclose(a.values, b.values, rtol=1e-9, atol=1e-9), f'{mod}.{key} differs'
        if getattr(b, 'strata', None) is not None: # By group too: routine delivery starts a new spell for those no longer eligible in a whole run as in the loop
            assert np.allclose(a.strata, b.strata, rtol=1e-9, atol=1e-9), f'{mod}.{key} by group differs'

    # Many runs in one call: the same as each run on its own, deliveries included
    msim = ss.parallel(make(solver='binomial'), n_runs=4)
    single = make(solver='binomial', rand_seed=3).run()
    assert np.array_equal(msim.results.offer.delivered.values[2], single.results.offer.delivered.values)
    assert np.array_equal(msim.results.sir.I.values[2], single.results.sir.I.values)
    # And in JAX
    jax = make(solver='ode')
    with ss.options.context(backend='jax'):
        jax.run()
    assert np.allclose(jax.results.sir.I.values, whole.results.sir.I.values, rtol=1e-10)
    assert np.allclose(jax.results.offer.delivered.values, whole.results.offer.delivered.values, rtol=1e-10)
    return whole


if __name__ == '__main__':
    sc.options(interactive=False)
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
