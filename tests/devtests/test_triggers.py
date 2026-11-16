"""
Test triggered moves (ss.on), the rest staying (ss.stay), lazy handles (ss.person, ss.ref), and distributions by group
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)


@sc.timer()
def test_bv_debut():
    sc.heading('Testing BV debut as soon as women turn 15, as agents')
    adult_f = ss.person.female & (ss.person.age >= 15)
    def make():
        U, CST1, CST4 = ss.states('U CST1 CST4')
        return ss.Disease(U, CST1, CST4,
            debut = ss.Transition(U, {CST1: 0.8, CST4: None}, when=adult_f), # Was rate=ss.perday(1), where=adult_f
            onset = CST1 >> CST4 @ ss.peryear(0.2),
            clear = CST4 >> CST1 @ ss.months(6),
            default_state='U', name='bv')
    checked = sc.obj(n=0)
    def check(sim): # After every module has stepped, nobody who qualifies is still waiting
        bv, ppl = sim.diseases.bv, sim.people
        u = bv.U.uids
        assert not np.any(ppl.female[u] & (ppl.age[u] >= 15)), f'step {sim.ti}: women 15+ still in U'
        checked.n += 1
    demog = lambda: [ss.Births(birth_rate=0.02), ss.Deaths(death_rate=0.01)]
    sim = ss.Sim(make(), analyzers=check, demographics=demog(), n_agents=5000, start=2000, stop=2010, dt=ss.months(1), unit='year', **kw).run()
    res = sim.results.bv
    assert checked.n >= len(res.timevec) - 1 and res.debut[1:].sum() > 100
    assert res.U[0] < 0.7*5000 # Those 15+ at the start debuted before it

    return sim


@sc.timer()
def test_on_event():
    sc.heading('Testing ss.on() with an event: edge-triggered, in the same step, whatever the module order, as agents and as compartments')
    for order in [True, False]:
        sim = make_on_event(order)
        res = sim.results
        assert res.b.flag.sum() > 100
        assert res.b.flag.sum() <= res.a.sa.sum() # Only those infected (and in S of b at the time)
        assert np.isclose(res.b.flag[1:].sum(), res.a.sa[1:].sum(), rtol=0.25)
        r = make_on_event(order, solver='ode').results
        assert np.isclose(r.b.flag.sum(), res.b.flag.sum(), rtol=0.05), f'{r.b.flag.sum():.0f} as an ODE, {res.b.flag.sum():.0f} as agents'
        assert np.isclose(r.b.flag[:-1].sum(), r.b.unflag[:-1].sum() + r.b.F[-1], rtol=1e-6) # The flow is the move (F[-1] is before the last step's flows)

    # An event of its own module, as compartments: those it happened to move at the end of the step
    S, F, G = ss.states('S F G')
    a = ss.Disease(S, F, G, sa=(sa := S >> F @ ss.peryear(0.5)), flag=F >> G @ ss.on(sa), name='a')
    res = ss.Sim(a, solver='ode', **kw).run().results.a
    assert res.flag.sum() > 0 and np.isclose(res.flag.sum(), res.sa.sum(), rtol=0.01) and res.F.max() < 1e-6*res.S[0]
    return sim


def make_a():
    S, I = ss.states('S I')
    return ss.Disease(sa=ss.Transition(S, I, rate=ss.peryear(0.5)), ia=ss.Transition(I, S, dur=ss.years(1)), name='a')


def make_on_event(order, solver='abm'):
    a = make_a()
    S, F, G = ss.states('S F G')
    b = ss.Disease(S, F, G, flag=S >> F @ ss.on(a.sa), unflag=F >> S @ ss.years(0.2), name='b') # Flagged when infected with a; level semantics would flag them again on return
    mods = [a, b] if order else [b, a]
    size = dict(n_agents=3000) if solver == 'abm' else dict(total_pop=3000)
    return ss.Sim(mods, solver=solver, start=2000, stop=2005, dt=ss.months(1), unit='year', **size, **kw).run()


@pytest.mark.parametrize('form', ['since', 'ever', 'count', 'recorded'])
def test_memory_slots(form):
    sc.heading(f'Testing the memory of an event ({form}) in a rate, a where=, and a modifier')
    def run(slot):
        A, B = ss.states('A B')
        back = ss.Transition(B, A, rate=ss.peryear(1))
        preg = ss.Pregnancy(fertility_rate=ss.peryear(0.1)) if form == 'recorded' else None # An event it records that is not a transition
        cond = dict(since=back.since < ss.years(1), ever=back.ever, count=back.n_times >= 1, recorded=preg.live_birth.since < ss.years(1) if preg else None)[form]
        go = ss.Transition(A, B, rate=ss.peryear(2)*cond if slot == 'rate' else ss.peryear(2), who=cond if slot == 'where' else None)
        d = ss.Disease(A, B, go=go, back=back, init=dict(B=0.5), default_state='A', name='d')
        mods = ss.modify(d.go, multiply=cond) if slot == 'modifier' else None
        sim = ss.Sim(d, demographics=preg, connectors=mods, n_agents=2000, start=2000, stop=2005, dt=ss.months(1), rand_seed=1, **kw).run()
        return sim.results.d.go.values.sum()
    out = [run(slot) for slot in ['rate', 'where', 'modifier']]
    assert out[0] > 0 and out[0] == out[1] == out[2], out # The same people move, whichever slot reads it
    return out


if __name__ == '__main__':
    sc.options(interactive=False)
    test_bv_debut()
    test_on_event()
    for form in ['since', 'ever', 'count', 'recorded']:
        test_memory_slots(form)
