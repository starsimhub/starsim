"""
Test undriven transitions and treatments that move people along them
"""

# %% Imports and settings
import warnings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day')


def make_sis(cure=None):
    S, I = ss.states(2)
    return ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> S @ ss.days(20), cure=cure(S, I), init_prev=0.05, name='sis')


def w209(sim):
    """ The warnings raised when a sim is initialized that a transition has nothing to drive it """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        sim.init()
    return [str(x.message) for x in w if 'so only custom code can move people along it' in str(x.message)]


# %% Define the tests

def test_undriven():
    sc.heading('Testing undriven transitions')
    forms = dict(
        op  = lambda S, I: I >> S,
        tr  = lambda S, I: ss.Transition(I, S),
    )
    for key,form in forms.items():
        sis = make_sis(form)
        assert sis.cure.undriven
        assert 'undriven: moved by products, deliveries, or code' in sis.describe(output=True)
        assert len(w209(ss.Sim(sis, stop=10, **kw))) == 1, f'{key}: expected W209 for an undriven transition nothing names'

    # Named by a product, or by on=, it is not warned about; a rate of 0 is not undriven
    sis = make_sis(forms['op'])
    sim = ss.Sim(sis, interventions=ss.treat(ss.Treatment(transition='cure'), coverage=0.2, interval=5, name='tx'), stop=10, **kw)
    assert not w209(sim)
    assert 'undriven: moved by treatment' in sim.diseases.sis.describe(output=True)
    sis = make_sis(forms['op'])
    assert not w209(ss.Sim(sis, interventions=ss.screen(ss.Diagnostic(), on=sis.cure), stop=10, **kw))
    sis = make_sis(lambda S, I: I >> S @ 0)
    assert not sis.cure.undriven and not w209(ss.Sim(sis, stop=10, **kw))

    # Moved by a treatment, as agents and compartments
    for solver,pars in [('ode', dict(total_pop=10_000)), ('abm', dict(n_agents=10_000))]:
        sis = make_sis(forms['op'])
        tx = ss.treat(ss.Treatment(cure=0.9, transition='cure'), coverage=0.3, at=[ss.days(20)])
        sim = ss.Sim(sis, interventions=tx, solver=solver, stop=40, **kw, **pars).run()
        res = sim.results
        assert np.isclose(res.sis.cure.sum(), 0.9*res.tx.delivered.sum(), rtol=0.05)
        assert res.sis.cure.sum() > 0

    with pytest.raises(ValueError, match='has no rate. Give it one'): # An inflow still needs a rate
        S, I = ss.states('S I')
        ss.Disease(births=ss.Transition(ss.null, S), infection=ss.Transition(S, I, rate=0.3 * I))
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    undriven = test_undriven()
