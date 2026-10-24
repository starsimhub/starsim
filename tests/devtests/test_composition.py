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


@sc.timer()
def test_fit_numpy():
    sc.heading('Testing a fit of a NumPy module through a function that makes the sim, with no gradient')
    def make(beta=0.3, data=None):
        return ss.Sim(connectors=NumpySIR(beta=beta, name='numpy'), unit='day', dt=1, start=0, stop=100, verbose=0, data=data)
    I = make(0.25).run().results.numpy.I
    data = pd.DataFrame({'numpy.I': I.values[::10]}, index=I.timevec[::10])
    fit = make(beta=ss.par(0.3, bounds=[0.1, 0.5]), data=data).fit(method='optuna', trials=30, verbose=0)
    assert np.isclose(fit.pars.beta, 0.25, rtol=0.02)
    with pytest.raises(ss.CapabilityError, match='written as code'): # No gradient, which it says
        make(beta=ss.par(0.3, bounds=[0.1, 0.5]), data=data).gradient()
    return fit


if __name__ == '__main__':
    T = sc.timer()
    s4 = test_fit_numpy()
    T.toc()
