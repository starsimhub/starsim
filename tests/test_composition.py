"""
Test that code written in plain NumPy composes with Starsim: an ODE of its own beside a declared
disease, agents of its own beside a standard module, an intervention that changes it, and a fit
of it.
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl

sc.options(interactive=False)


class NumpySIR(ss.Module):
    """ An SIR ODE in plain NumPy, with Euler steps; it reads nothing of the sim but its step """
    def __init__(self, beta=0.3, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(beta=beta, gamma=gamma)
        self.update_pars(**kwargs)
        return

    def init_post(self):
        super().init_post()
        self.y = np.array([0.99, 0.01, 0.0])
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('I', label='Infectious'))
        return

    def step(self):
        S, I, R = self.y
        b, g = self.pars.beta, self.pars.gamma
        dt = self.t.dt.days
        self.y = np.array([S - b*S*I*dt, I + (b*S*I - g*I)*dt, R + g*I*dt])
        return

    def update_results(self):
        super().update_results()
        self.results.I[self.ti] = self.y[1]
        return


class Reservoir(ss.Module):
    """ Pathogen in the environment, in plain NumPy: shed by the infectious, decaying, and raising the declared disease's beta """
    def __init__(self, shed=2.0, decay=0.1, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(shed=shed, decay=decay)
        self.update_pars(**kwargs)
        self.W = 0.0
        return

    def step(self):
        sir = self.sim.diseases.sir
        prev = sir.I.n/(sir.S.n + sir.I.n + sir.R.n) # An aggregate value, which the declared disease publishes under any solver
        self.W += (self.pars.shed*prev - self.pars.decay*self.W)*self.t.dt.days
        sir.pars.beta = 0.1*(1 + self.W) # And the value it publishes back
        return


class Mosquitoes(ss.Module):
    """ Agents of its own, in plain NumPy, with a random number generator of its own """
    def init_post(self):
        super().init_post()
        self.rng = np.random.default_rng(1)
        self.age = np.zeros(1000)
        return

    def step(self):
        self.age += self.t.dt.days
        dead = self.rng.random(len(self.age)) < 0.1
        self.age[dead] = 0 # Each replaced by a newly hatched one
        return


def declared_sir():
    S, I, R = ss.states(3)
    return ss.Disease(S >> I @ ss.par(0.1, name='beta') * I, I >> R @ ss.days(10), init=dict(I=0.01), name='sir')


@sc.timer()
def test_numpy_ode_beside():
    sc.heading('Testing an ODE in NumPy beside a declared disease run as agents, coupled through aggregate values')
    kw = dict(unit='day', stop=100, n_agents=5000, verbose=0, rand_seed=2)
    alone = ss.Sim(declared_sir(), **kw).run()
    coupled = ss.Sim(declared_sir(), connectors=Reservoir(), **kw).run()
    assert coupled.solver == alone.solver == 'abm'
    assert coupled.connectors.reservoir.W > 0
    assert coupled.results.sir.S_to_I.sum() > 1.2*alone.results.sir.S_to_I.sum() # The environment adds to transmission
    ode_kw = dict(unit='day', stop=100, total_pop=5000, verbose=0) # The same, with the declared disease as an ODE
    ode = ss.Sim(declared_sir(), connectors=Reservoir(), **ode_kw).run()
    assert ode.solver == 'ode' and ode.connectors.reservoir.W > 0
    assert ode.results.sir.S_to_I.sum() > 1.2*ss.Sim(declared_sir(), **ode_kw).run().results.sir.S_to_I.sum()
    return coupled


@sc.timer()
def test_numpy_agents_beside():
    sc.heading('Testing agents in NumPy, with their own random numbers, beside a standard module')
    kw = dict(diseases='sir', networks='random', n_agents=2000, start=2000, stop=2010, verbose=0)
    alone = ss.Sim(**kw).run()
    both = ss.Sim(connectors=Mosquitoes(), **kw).run()
    assert both.connectors.mosquitoes.age.max() > 0
    assert np.array_equal(both.results.sir.I.values, alone.results.sir.I.values) # Its own random numbers leave those of the other modules alone
    return both


@sc.timer()
def test_callback_changes_component():
    sc.heading('Testing an intervention given as a function, changing a parameter of a NumPy module')
    def distance(sim): # No products or modifiers: a function of the sim
        if sim.ti == 20:
            sim.connectors.numpysir.pars.beta = 0.05
        return
    kw = dict(unit='day', dt=1, start=0, stop=100, verbose=0)
    base = ss.Sim(connectors=NumpySIR(), **kw).run()
    changed = ss.Sim(connectors=NumpySIR(), interventions=distance, **kw).run()
    I0, I1 = base.results.numpysir.I.values, changed.results.numpysir.I.values
    assert np.array_equal(I0[:21], I1[:21]) and I1[40] < 0.5*I0[40]
    return changed


@sc.timer()
def test_coded_and_declared():
    sc.heading('Testing code that writes the states of a declared disease run as agents, and a declared intervention on a disease written as code')
    class Mover(ss.Intervention): # Moves ten people from S to I at step 3: by indexing, and with an in-place operator
        def __init__(self, raw=False, **kwargs):
            super().__init__(**kwargs)
            self.raw = raw
            return
        def step(self):
            sir = self.sim.diseases.sir
            if self.ti == 3:
                uids = sir.S.uids[:10]
                if self.raw: # Through the NumPy array behind each state, which is not tracked
                    sir.S.raw[uids] = False
                    sir.I.raw[uids] = True
                else:
                    sir.S[uids] = False
                    sir.I |= ~(sir.S | sir.R)
            return
    class Reader(ss.Intervention): # Reads the number infected later in the same step
        def step(self):
            if self.ti == 3:
                self.n = np.sum(self.sim.diseases.sir.I.n)
            return
    def make(raw=False):
        S, I, R = ss.states('S I R')
        sir = ss.Disease(S >> I @ 0*I, I >> R @ ss.days(1000), init_prev=0, name='sir')
        return ss.Sim(sir, interventions=[Mover(raw=raw), Reader()], unit='day', stop=6, n_agents=100, solver='abm', verbose=0).run()
    sim = make()
    res = sim.results.sir
    assert sim.interventions.reader.n == 10 and res.I[4] == 10 and res.S[4] == 90 # Seen in the same step, and in the results
    raw = make(raw=True) # Not tracked: the counts and results do not see it, which STARSIM_CHECK_COUNTS=1 checks at each step
    assert raw.diseases.sir.I.count() == 10 and raw.results.sir.I[4] == 0
    with pytest.raises(RuntimeError, match='differs from a new count'):
        raw.diseases.sir.engine.check_counts()

    # A disease written as code takes a declared vaccine
    kw = dict(networks='random', n_agents=5000, stop=100, rand_seed=2, verbose=0)
    infections = [ss.Sim(ssl.AgentSIR(beta=0.1, init_prev=0.02), interventions=vx, **kw).run().results.agentsir.cum_infections[-1] for vx in [None, ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5, at=0)]]
    assert infections[1] < 0.8*infections[0]
    return sim


if __name__ == '__main__':
    T = sc.timer()
    s1 = test_numpy_ode_beside()
    s2 = test_numpy_agents_beside()
    s3 = test_callback_changes_component()
    s5 = test_coded_and_declared()
    T.toc()
