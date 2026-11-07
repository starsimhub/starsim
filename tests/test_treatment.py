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
        assert 'undriven: moved by products, deliveries, or code' in sis.describe()
        with warnings.catch_warnings():
            warnings.simplefilter('error') # Nothing names it, which is a note, not a warning: it exists for products to drive (e.g. in only some scenarios)
            sim = ss.Sim(sis, stop=10, **kw).init()
        assert 'undriven: moved by nothing' in sim.diseases.sis.describe()
        assert any('"cure" of "sis"' in note and 'no rate or duration' in note for note in sim.notes) # Said at init

    # Named by a product; a rate of 0 is not undriven
    sis = make_sis(forms['op'])
    sim = ss.Sim(sis, interventions=ss.treat(ss.Treatment(cure={'cure': 1.0}), when=ss.schedule(every=5), coverage=0.2, name='tx'), stop=10, **kw).init()
    assert 'undriven: moved by treatment' in sim.diseases.sis.describe()
    assert not any('no rate or duration' in note for note in sim.notes) # Not when a product moves people along it
    sis = make_sis(lambda S, I: I >> S @ 0)
    assert not sis.cure.undriven

    # Moved by a treatment, as agents and compartments
    for solver,pars in [('ode', dict(total_pop=10_000)), ('abm', dict(n_agents=10_000))]:
        sis = make_sis(forms['op'])
        tx = ss.treat(ss.Treatment(cure={'cure': 0.9}), when=[ss.days(20)], coverage=0.3)
        sim = ss.Sim(sis, interventions=tx, solver=solver, stop=40, **kw, **pars).run()
        res = sim.results
        assert np.isclose(res.sis.cure.sum(), 0.9*res.tx.delivered.sum(), rtol=0.05)
        assert res.sis.cure.sum() > 0

    with pytest.raises(ValueError, match='has no rate. Give it one'): # An inflow still needs a rate
        S, I = ss.states('S I')
        ss.Disease(births=ss.Transition(ss.unborn, S), infection=ss.Transition(S, I, rate=0.3 * I))
    return sim


def test_cure_forms():
    sc.heading('Testing the forms of cure=: a number, by state, by transition, by name, and a move written with the states')
    S, I = ss.states(2)
    forms = dict(
        number = lambda sis: 0.9,
        state  = lambda sis: dict(I=0.9),
        trans  = lambda sis: {sis.cure: 0.9},
        name   = lambda sis: dict(cure=0.9),
        move   = lambda sis: {sis.I >> sis.S: 0.9},
    )
    cured = {}
    for key,form in forms.items():
        sis = make_sis(lambda S, I: I >> S)
        tx = ss.treat(ss.Treatment(cure=form(sis)), when=[ss.days(20)], name='tx')
        sim = ss.Sim(sis, interventions=tx, solver='ode', stop=40, **kw).run()
        cured[key] = sim.results.tx.cured.sum()
    assert np.allclose(list(cured.values()), cured['number'], rtol=1e-6), cured
    with pytest.raises(TypeError, match='cure='): # The removed keyword names its replacement
        ss.Treatment(**{'moves': I >> S})
    with pytest.raises(TypeError, match='as a key'): # A transition alone is not a probability
        ss.Treatment(cure=I >> S)
    return cured


if __name__ == '__main__':
    sc.options(interactive=False)
    undriven = test_undriven()
    cured = test_cure_forms()
