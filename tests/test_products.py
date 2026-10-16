"""
Test products over several diseases and multi-outcome tests (proposal 02 §2.2-2.3)
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, start=2000, stop=2010, dt=ss.months(3))


def make_geno(name, beta=1.2):
    """ A genotype as a disease of its own, S -> I -> S """
    S, I = ss.states('S I')
    return ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(beta)), clearance=I >> S @ ss.years(1), init_prev=0.1, name=name)


def run_both(make, pop=10_000, n_agents=10_000, **kwargs):
    return sc.objdict(ode=make(solver='ode', total_pop=pop, **kw, **kwargs).run(),
                      abm=make(solver='abm', total_pop=pop, n_agents=n_agents, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.objdict({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


# %% Define the tests

def test_vaccine_diseases():
    sc.heading('Testing a vaccine over several diseases: efficacy keyed by disease, one has_ per person, a modifier per disease and quantity')
    def make(eff, **kwargs):
        g16, g18 = make_geno('hpv16'), make_geno('hpv18')
        vx = ss.Vaccine(efficacy=eff(g16, g18), name='vx')
        prog = ss.vaccinate(vx, coverage=0.8, at=[2001])
        cross = ss.modify(g18.clearance, multiply=2, where=vx.held, name='cross') # vx.held is a condition
        return ss.Sim([g16, g18], interventions=prog, connectors=cross, **kwargs)
    base = run_both(lambda **k: make(lambda a, b: {a: 0.0}, **k))
    sims = run_both(lambda **k: make(lambda a, b: {a: 0.95, 'hpv18': dict(acquisition=0.2)}, **k)) # A disease or its name; a number or a dict of effects
    for solver,sim in sims.items():
        r, r0 = sim.results, base[solver].results
        assert r.hpv16.n_I[-2] < 0.5*r0.hpv16.n_I[-2] and r.hpv18.n_I[-2] < r0.hpv18.n_I[-2], solver
        assert r.prog.delivered.sum() == pytest.approx(8000, rel=0.02)
        assert len(sim.modifiers[('hpv16', 'rel_sus', None)]) == len(sim.modifiers[('hpv18', 'rel_sus', None)]) == 1
        assert [s.name for s in sim.groups.strats].count('has_vx') <= 1
    check_close(sims, lambda sim: sim.results.hpv18.infection.sum(), rtol=0.15)
    assert sims.abm.people.has_vx.sum() > 0 and 'has_vx' not in [s.name for s in sims.abm.diseases.hpv16.graph.strats]

    with pytest.raises(ss.UnknownEffect): # Keyed by a disease and by something else
        make(lambda a, b: {a: 0.9, 'acquisition': 0.5}, solver='abm', **kw).init()
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    vaccine = test_vaccine_diseases()
