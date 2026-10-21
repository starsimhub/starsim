"""
Test that code written in plain NumPy composes with Starsim: an ODE of its own beside a declared
disease, agents of its own beside a standard module, an intervention that changes it, and a fit
of it. (A loop of NumPy functions with no sim, and with a timeline, is in test_loop.py.)
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss

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
    assert np.array_equal(both.results.sir.n_I.values, alone.results.sir.n_I.values) # Its own random numbers leave those of the other modules alone
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
def test_fit_numpy():
    sc.heading('Testing a fit of a NumPy module through a function that makes the sim, with no gradient')
    def make(beta=0.3, data=None):
        return ss.Sim(connectors=NumpySIR(beta=beta, name='numpy'), unit='day', dt=1, start=0, stop=100, verbose=0, data=data)
    I = make(0.25).run().results.numpy.I
    data = pd.DataFrame({'numpy.I': I.values[::10]}, index=I.timevec[::10])
    fit = make(beta=ss.par(0.3, bounds=[0.1, 0.5]), data=data).calibrate(method='optuna', trials=30, verbose=0)
    assert np.isclose(fit.pars.beta, 0.25, rtol=0.02)
    with pytest.raises(ss.CapabilityError, match='written as code'): # No gradient, which it says
        make(beta=ss.par(0.3, bounds=[0.1, 0.5]), data=data).gradient()
    return fit


if __name__ == '__main__':
    T = sc.timer()
    s1 = test_numpy_ode_beside()
    s2 = test_numpy_agents_beside()
    s3 = test_callback_changes_component()
    s4 = test_fit_numpy()
    T.toc()
