"""
Test connectors and modifiers, and code in plain NumPy composed with Starsim
"""

import sciris as sc
import numpy as np
import pytest
import starsim as ss
import starsim.library as ssl

sc.options(interactive=False) # Assume not running interactively


def test_connectors():
    sc.heading('Testing modifiers between the HIV written as code and a declared SIS')
    def run(*connectors):
        hiv = ssl.diseases.HIV(beta={'sexual': [0.008, 0.004]}, init_prev=0.2)
        sis = ss.SIS(beta={'sexual': ss.peract(m2f=0.01, f2m=0.005)}, dur_inf=ss.years(1), init_prev=0.05)
        mods = [make(hiv, sis) for make in connectors]
        return ss.Sim(n_agents=2000, networks=ss.SexualNet(), diseases=[hiv, sis], connectors=mods, start=2000, stop=2020, dt=ss.months(1), verbose=0).run().results
    base = run()
    stop_hiv = run(lambda hiv, sis: ss.modify(hiv.rel_trans, multiply=0, when=ss.schedule(start=2010))) # A modifier of the rel_trans of a disease written as code
    no_sis = run(lambda hiv, sis: ss.modify(sis.rel_sus, multiply=0, who=hiv.cd4 < 1e6)) # A condition on an array of a disease written as code (everyone's CD4 is below 1e6)
    after = stop_hiv.timevec > 2010
    assert base.hiv.infection[after].sum() > 0 and stop_hiv.hiv.infection[after].sum() == 0 # No HIV transmission after 2010
    assert base.sis.infection.sum() > 0 and no_sis.sis.infection.sum() == 0


def test_lazy_arrays():
    """ A lazy condition on another module's array (not a declared state) is resolved in that module """
    sc.heading('Testing lazy conditions on module arrays')
    class HIV(ss.Disease):
        S = ss.State()
        I = ss.State()
        infection = S >> I @ 0.0 * I
        def __init__(self, **kwargs):
            super().__init__(init_prev=0.2, **kwargs)
            self.define_states(ss.FloatArr('cd4', default=ss.choice([300, 800])), ss.BoolArr('flag', default=ss.bernoulli(0.5)))
            return

    class Syph(ss.Disease):
        S = ss.State()
        I = ss.State()
        infection = S >> I @ 0.2 * I

    class Net(ss.RandomNet):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.BoolArr('core', default=ss.bernoulli(0.5)))
            return

    for which in ['cd4', 'flag', 'core']:
        hiv, syph, net = HIV(name='hiv'), Syph(name='syph', init_prev=0.05), Net(name='net')
        where = dict(cd4=hiv.cd4 >= 500, flag=~hiv.flag, core=~net.core)[which]
        protect = ss.modify(syph.rel_sus, multiply=0, who=where)
        sim = ss.Sim([hiv, syph], connectors=protect, networks=net, n_agents=1000, stop=2005, verbose=0, rand_seed=1).init()
        syph = sim.diseases.syph
        seeds = syph.I.uids
        sim.run()
        mods = dict(cd4=sim.diseases.hiv, flag=sim.diseases.hiv, core=sim.networks.net)
        new = syph.I.uids.remove(seeds) # Infected during the run
        arr = mods[which][which]
        protected = (arr[new] >= 500) if which == 'cd4' else ~arr[new]
        assert len(new) and not protected.any(), f'{which}: {protected.sum()} of {len(new)} infected were protected'


def test_where_count():
    sc.heading('A condition on a count in when= is worked out at each step, under every solver; in who=, it raises')
    def make():
        S, I, R = ss.states(3)
        return ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='d')
    def peak(solver, thresh):
        d = make()
        npi = ss.modify(d.infection, multiply=0.5, when=d.I.n > thresh)
        return ss.Sim(d, connectors=npi, n_agents=2000, total_pop=2000, stop=100, solver=solver, rand_seed=1, verbose=0).run().results.d.I.max()
    for solver in ['ode', 'binomial', 'tau', 'gillespie', 'abm']:
        never, once = peak(solver, 2000), peak(solver, 200)
        assert 200 < once < 0.6*never, f'Under {solver}, the peak is {once:n} with a threshold of 200, and {never:n} without one reached'
    d = make()
    vaccinate = ss.vaccinate(ss.Vaccine(efficacy=0.9), when=d.I.n > 200, coverage=0.1) # A delivery takes one too
    assert ss.Sim(d, interventions=vaccinate, n_agents=2000, stop=100, verbose=0).run().results.d.I.max() > 200
    with pytest.raises(TypeError): # A count says when, for everyone at once, so it is not who=
        ss.modify(d.infection, multiply=0.5, who=d.I.n > 200)
    with pytest.raises(TypeError): # ...and a condition on people is not when=
        ss.modify(d.pars.beta, multiply=0.5, when=d.I)


def test_count_is_when():
    sc.heading('A count in when= of a transition says when its rate applies; an intervention written as code takes a count and a schedule')
    S, I, Q = ss.states('S I Q')
    assert ss.Transition(I, Q, rate=ss.peryear(5), when=Q.n < 20)._trigger is None # With a rate, the count gates it
    assert ss.Transition(I, Q, when=Q.n < 20)._trigger is not None # With no rate, it moves everyone at once
    for solver in ['ode', 'abm']:
        S, I, Q = ss.states('S I Q')
        d = ss.Disease(S, I, Q, infection=S >> I @ ss.peryear(2)*I, q=ss.Transition(I, Q, rate=ss.peryear(5), when=Q.n < 20), init_prev=0.1, name='d')
        size = dict(n_agents=1000) if solver == 'abm' else dict(total_pop=1000)
        assert ss.Sim(d, solver=solver, start=2000, stop=2003, dt=ss.years(0.25), verbose=0, **size).run().results.d.Q[-1] < 100 # Without the gate, about 800

    class Count(ss.Intervention):
        def init_pre(self, sim):
            super().init_pre(sim)
            self.n = []
        def step(self):
            self.n.append(len(self.check_eligibility()))
    sir = ss.SIR(beta=ss.perday(0.1), init_prev=0.01, name='sir')
    count = Count(who=ss.person.age < 30, when=ss.schedule(every=ss.days(10)) & (sir.I.n > 100), name='count')
    sim = ss.Sim(diseases=sir, networks='random', interventions=count, n_agents=1000, unit='day', dt=1, stop=60, rand_seed=1, verbose=0).run()
    I, n = sim.results.sir.I, np.array(sim.interventions.count.n)
    due = np.arange(len(n)) % 10 == 0
    assert np.all(n[~due] == 0) and np.all((n[due] > 0) == (I[:len(n)][due] > 100)) and (n > 0).sum() >= 2


def test_when_joined():
    sc.heading('when= joins a schedule, or an event, to a condition on the population with & (both) or | (either)')
    d = ss.SIR(name='d')
    assert repr(ss.schedule(start=2025) & (d.I.n > 1000)) == 'ss.schedule(start=2025) & (I.n > 1000)'
    with pytest.raises(TypeError):
        ss.schedule(start=2025) & d.I.n > 1000 # Not in parentheses

    # Rounds every 10 days, each only if the count holds then, as compartments and as agents
    for solver in ['ode', 'abm']:
        d = ss.SIR(beta=ss.perday(0.1), init_prev=0.02, name='d')
        ev = ss.Event(d.S >> d.R, p=0.5, when=ss.schedule(every=ss.days(10)) & (d.I.n > 50), name='ev')
        size = dict(n_agents=2000, networks='random') if solver == 'abm' else dict(total_pop=10_000)
        sim = ss.Sim(diseases=d, interventions=ev, solver=solver, unit='day', dt=1, stop=60, rand_seed=1, verbose=0, **size).run()
        I, reached = sim.results.d.I, sim.results.ev.delivered
        ran = reached > 0
        assert np.all(~ran[np.arange(len(ran)) % 10 != 0]) # Only on the schedule
        assert np.any(ran) and np.all(I[ran] > 50) and not np.all(ran[::10]) # Only while the count holds, which it stops doing

    # After an event and a delay, the count is checked when it acts
    for op, acts in [('<', False), ('>', True)]:
        d = ss.SIR(beta=ss.perday(0.3), init_prev=0.01, name='d')
        count = (d.I.n < 300) if op == '<' else (d.I.n > 300)
        ev = ss.Event(when=ss.on(d.infection, delay=ss.days(10)) & count, name='ev')
        sim = ss.Sim(diseases=d, networks='random', interventions=ev, n_agents=2000, unit='day', dt=1, stop=20, rand_seed=1, verbose=0).run()
        assert (sim.results.ev.delivered[10:].sum() > 0) == acts # Infected while fewer than 300 were infectious, and acted on when more were

    # Joined with |: a round on the schedule's steps, and everyone eligible at each step the count holds; or, with an event, those it happens to
    def run(when):
        d = ss.SIR(beta=ss.perday(0.2), init_prev=0.01, name='d')
        sim = ss.Sim(diseases=d, interventions=ss.Event(when=when(d), p=0.1, name='ev'), n_agents=2000, networks='random', unit='day', dt=1, stop=50, rand_seed=1, verbose=0).run()
        return np.asarray(sim.results.ev.delivered), np.asarray(sim.results.d.I)
    rounds = np.arange(51) % 10 == 0
    event, I = run(lambda d: d.infection) # The event changes nothing, so the epidemic is the same in each run
    thresh = I.max()/2 # Held for some steps, not all
    either, I = run(lambda d: ss.schedule(every=ss.days(10)) | (d.I.n > thresh))
    held = I > thresh
    assert held.any() and not held.all() and np.array_equal(either > 0, rounds | held)
    either, I = run(lambda d: d.infection | (d.I.n > thresh))
    held = I > thresh
    assert np.allclose(either[~held], event[~held]) and np.all(either[held] > event[held]) # The event, and everyone while the count holds

    # A modifier on a schedule, or while a count holds
    d = ss.SIR(beta=ss.perday(0.3), init_prev=0.001, name='d')
    npi = ss.modify(d.pars.beta, multiply=0.0, when=ss.schedule(every=ss.days(10)) | (d.I.n > 100))
    res = ss.Sim(diseases=d, connectors=npi, solver='ode', total_pop=10_000, unit='day', dt=1, stop=60, verbose=0).run().results.d
    off = (np.arange(61) % 10 == 0) | (np.asarray(res.I) > 100)
    inf = np.asarray(res.infection)
    assert np.all(inf[off] == 0) and np.all(inf[~off][1:] > 0) and off.sum() > 6 # Off on the schedule and while the count holds (the first step has no infections yet)


def test_member():
    sc.heading('ss.modify() as a member of a disease: the same as given to the sim')
    kw = dict(networks='random', n_agents=500, unit='day', stop=40, solver='abm', verbose=0)

    class Fading(ss.Disease):
        beta = ss.par(ss.perday(0.1))
        S, E, I, R = ss.states('S E I R')
        infection = S >> E @ beta*I
        onset = E >> I @ ss.days(3)
        recovery = I >> R @ ss.days(8)
        fading = ss.modify('rel_trans', multiply=0.5**(I.time_in/ss.days(3)), who=I)
        distancing = ss.modify(beta, multiply=0.5, when=ss.schedule(start=20))
        slower = ss.modify(recovery, multiply=2, when=ss.schedule(start=30))

    held = ss.Sim(Fading(init_prev=0.1, name='a'), **kw).run()
    assert list(held.connectors.keys()) == ['a_fading', 'a_distancing', 'a_slower']
    S, E, I, R = ss.states('S E I R')
    beta = ss.par(ss.perday(0.1))
    d = ss.Disease(S, E, I, R, beta=beta, infection=S >> E @ beta*I, onset=E >> I @ ss.days(3), recovery=I >> R @ ss.days(8), init_prev=0.1, name='a')
    given = [ss.modify(d.rel_trans, multiply=0.5**(d.I.time_in/ss.days(3)), who=d.I), ss.modify(d.pars.beta, multiply=0.5, when=ss.schedule(start=20)), ss.modify(d.recovery, multiply=2, when=ss.schedule(start=30))]
    assert np.array_equal(held.results.a.I, ss.Sim(d, connectors=given, **kw).run().results.a.I)

    # A member whose target is in another module, by its path: the same as given to the sim
    S, I, R = ss.states('S I R')
    a = lambda: ss.Disease(S, I, R, infection=S >> I @ 0*I, recovery=I >> R @ ss.days(5), init_prev=0.2, name='a')
    U, V = ss.states('U V')
    b = lambda **kw: ss.Disease(U, V, infection=U >> V @ ss.perday(0.1), init_prev=0.5, name='b', **kw)
    kw2 = dict(n_agents=500, unit='day', stop=10, verbose=0)
    member = ss.Sim(diseases=[a(), b(slower=ss.modify('a.recovery', multiply=0.25, who=V))], **kw2).run()
    given = ss.Sim(diseases=[a(), b()], connectors=ss.modify('a.recovery', multiply=0.25, who='b.V'), **kw2).run()
    assert np.array_equal(member.results.a.R, given.results.a.R)
    assert member.results.a.R[-1] < ss.Sim(diseases=[a(), b()], **kw2).run().results.a.R[-1] # It slows a's recovery


def test_stop_due():
    sc.heading('A modifier that stops an exit stops those due in the step it starts, as the ODE does')
    S, I, R = ss.states('S I R')
    for solver,kw in [('abm', dict(n_agents=2000)), ('ode', dict(total_pop=2000))]:
        d = ss.Disease(S, I, R, infection=S >> I @ 0*I, recovery=I >> R @ ss.days(10), init_prev=0.9, name='a')
        stopped = ss.Sim(diseases=d, connectors=ss.modify(d.recovery, multiply=0, start=5), unit='day', stop=10, solver=solver, verbose=0, **kw).run().results.a.I
        assert stopped[4] > stopped[5] and np.all(stopped[5:] == stopped[5]), f'{solver}: {stopped}' # None leave from the step it starts


def test_coded():
    sc.heading('ss.modify() reaches the parameters of any module written as code, which are back to their base values after the run')
    class Wealth(ss.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(rate=1.0)
            self.seen = []
        def step(self):
            self.seen.append(self.pars.rate)
    class Link(Wealth, ss.Connector):
        pass
    for solver in ['abm', 'ode']:
        mods = [ss.modify('wealth.pars.rate', multiply=2, when=ss.schedule(start=2)), ss.modify('wealth.pars.rate', add=1, when=ss.schedule(start=4)), ss.modify('link.pars.rate', set=ss.timeseries({1: 3, 3: 4}), when=ss.schedule(start=1))]
        sim = ss.Sim(Wealth(name='wealth'), diseases=ss.SIR(beta=ss.perday(0.1)), connectors=[Link(name='link'), *mods], n_agents=100, total_pop=100, unit='year', stop=5, solver=solver, verbose=0).run()
        wealth, link = sim.get_module('wealth'), sim.get_module('link')
        assert wealth.seen == [1, 1, 2, 2, 3, 3] and link.seen == [1, 3, 3, 4, 4, 4], f'{solver}: {wealth.seen} and {link.seen}' # A step from each year 0 to 5
        assert wealth.pars.rate == 1


def test_modifier_tables():
    sc.heading('Modifiers that depend only on time are worked out before a compartmental run, span by span, and kept for a rerun until something they depend on changes')
    def make(start=40, cut=0.5, beta=0.3, **kwargs):
        S, I, R = ss.states(3)
        season = ss.seasonal(amplitude=0.5, peak=ss.days(50), period=ss.days(180))
        d = ss.Disease(S, I, R, infection=S >> I @ ss.par(beta, name='beta')*season*I, recovery=I >> R @ ss.days(10), init=dict(I=0.001), name='d')
        d.stratify(age={0: 0.3, 20: 0.5, 60: 0.2})
        mods = [ss.modify(d.rel_sus, multiply=ss.by(age=[0.5, 1, 0.8]), when=ss.schedule(start=start, stop=150)),
                ss.modify(d.rel_trans, multiply=ss.par(cut, name='cut'), when=ss.schedule(stop=100)),
                ss.modify(d.recovery, multiply=ss.timeseries({0: 1, 60: 2}), when=ss.schedule(start=20))]
        return ss.Sim(d, interventions=mods, unit='day', stop=200, verbose=0, **kwargs)
    I = lambda sim: sim.results.d.I.values.copy()
    sim = make().run()
    ref = I(sim)
    assert sim._compiled_reason() is None
    assert np.array_equal(I(sim.run()), ref), 'A rerun gives the same'
    assert np.allclose(I(make().run(until=200)), ref, rtol=1e-12), 'A run a step at a time gives the same'
    assert np.array_equal(I(sim.restart(pars=dict(cut=0.2)).run()), I(make(cut=0.2).run())), 'A new value of a parameter in a modifier'
    sim = make().run()
    sim.interventions()[0].start = 60
    assert np.array_equal(I(sim.run()), I(make(start=60).run())), 'A modifier changed in place'


def test_sim_diagram():
    sc.heading('A sim\'s diagram and equations show every disease, and how they act on each other')
    hiv, syph = ss.SIS(name='hiv'), ss.SIR(name='syph')
    sim = ss.Sim(diseases=[hiv, syph], connectors=ss.modify(hiv.rel_trans, multiply=3, who=syph.I), verbose=0)
    assert 'subgraph syph' in sim.diagram('text') and 'syph.I → hiv.infection' in sim.equations()
    assert len(sim.diagram(do_show=False).axes[0].patches) > 8


# %% Code in plain NumPy, composed with Starsim

class NumpySIR(ss.Module):
    """ An SIR ODE in plain NumPy, with Euler steps; it reads nothing of the sim but its step """
    def __init__(self, beta=0.3, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(beta=beta, gamma=gamma)
        self.update_pars(**kwargs)

    def init_post(self):
        super().init_post()
        self.y = np.array([0.99, 0.01, 0.0])

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('I', label='Infectious', kind='stock'))

    def step(self):
        S, I, R = self.y
        b, g, dt = self.pars.beta, self.pars.gamma, self.t.dt.days
        self.y = np.array([S - b*S*I*dt, I + (b*S*I - g*I)*dt, R + g*I*dt])

    def record_states(self):
        super().record_states()
        self.results.I[self.ti] = self.y[1]


class Reservoir(ss.Module):
    """ Pathogen in the environment, in plain NumPy: shed by the infectious, decaying, and raising the declared disease's beta """
    def __init__(self, shed=2.0, decay=0.1, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(shed=shed, decay=decay)
        self.update_pars(**kwargs)
        self.W = 0.0

    def step(self):
        sir = self.sim.diseases.sir
        prev = sir.I.n/(sir.S.n + sir.I.n + sir.R.n) # An aggregate value, which the declared disease publishes under any solver
        self.W += (self.pars.shed*prev - self.pars.decay*self.W)*self.t.dt.days
        sir.pars.beta = 0.1*(1 + self.W) # And the value it publishes back


class Mosquitoes(ss.Module):
    """ Agents of its own, in plain NumPy, with a random number generator of its own """
    def init_post(self):
        super().init_post()
        self.rng = np.random.default_rng(1)
        self.age = np.zeros(1000)

    def step(self):
        self.age += self.t.dt.days
        dead = self.rng.random(len(self.age)) < 0.1
        self.age[dead] = 0 # Each replaced by a newly hatched one


def test_numpy_beside():
    sc.heading('Testing an ODE in NumPy beside a declared disease, coupled through aggregate values; and agents in NumPy beside a standard module')
    def declared_sir():
        S, I, R = ss.states(3)
        return ss.Disease(S >> I @ ss.par(0.1, name='beta') * I, I >> R @ ss.days(10), init=dict(I=0.01), name='sir')
    for solver, size in [('abm', dict(n_agents=5000, rand_seed=2)), ('ode', dict(total_pop=5000))]: # Chosen from the size, whatever the NumPy module
        kw = dict(unit='day', stop=100, verbose=0, **size)
        alone = ss.Sim(declared_sir(), **kw).run()
        coupled = ss.Sim(declared_sir(), connectors=Reservoir(), **kw).run()
        assert coupled.solver.name == alone.solver.name == solver and coupled.connectors.reservoir.W > 0
        assert coupled.results.sir.infection.sum() > 1.2*alone.results.sir.infection.sum() # The environment adds to transmission

    kw = dict(diseases='sir', networks='random', n_agents=2000, start=2000, stop=2010, verbose=0)
    both = ss.Sim(connectors=Mosquitoes(), **kw).run()
    assert both.connectors.mosquitoes.age.max() > 0
    assert np.array_equal(both.results.sir.I.values, ss.Sim(**kw).run().results.sir.I.values) # Its own random numbers leave those of the other modules alone

    # An intervention given as a function, changing a parameter of a NumPy module
    def distance(sim):
        if sim.ti == 20:
            sim.connectors.numpysir.pars.beta = 0.05
    kw = dict(unit='day', dt=1, start=0, stop=100, verbose=0)
    I0 = ss.Sim(connectors=NumpySIR(), **kw).run().results.numpysir.I.values
    I1 = ss.Sim(connectors=NumpySIR(), interventions=distance, **kw).run().results.numpysir.I.values
    assert np.array_equal(I0[:21], I1[:21]) and I1[40] < 0.5*I0[40]


def test_coded_and_declared():
    sc.heading('Testing code that writes the states of a declared disease run as agents')
    class Mover(ss.Intervention): # Moves ten people from S to I at step 3: by indexing, and with an in-place operator
        def step(self):
            sir = self.sim.diseases.sir
            if self.ti == 3:
                sir.S[sir.S.uids[:10]] = False
                sir.I |= ~(sir.S | sir.R)
    class Reader(ss.Intervention): # Reads the number infected later in the same step
        def step(self):
            if self.ti == 3:
                self.n = np.sum(self.sim.diseases.sir.I.n)
    S, I, R = ss.states('S I R')
    sir = ss.Disease(S >> I @ 0*I, I >> R @ ss.days(1000), init_prev=0, name='sir')
    sim = ss.Sim(sir, interventions=[Mover(), Reader()], unit='day', stop=6, n_agents=100, solver='abm', verbose=0).run()
    res = sim.results.sir
    assert sim.interventions.reader.n == 10 and res.I[4] == 10 and res.S[4] == 90 # Seen in the same step, and in the results


if __name__ == '__main__':
    T = sc.timer()
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
    T.toc()
