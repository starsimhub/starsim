"""
Test products over several diseases, and tests with several results
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
    return ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(beta) * I), clearance=I >> S @ ss.years(1), init_prev=0.1, name=name)


def make_hpv(name, beta=1.5):
    """ A genotype with a lesion and cancer: S -> I -> CIN -> CA, cleared to S, and excised (undriven) back to I """
    S, I, CIN, CA = ss.states('S I CIN CA')
    return ss.Disease(
        infection = S >> I @ ss.peryear(beta) * (I + CIN),
        clearance = I >> S @ ss.years(1),
        dysplasia = I >> CIN @ ss.years(2),
        regress   = CIN >> S @ ss.years(3),
        invasion  = CIN >> CA @ ss.years(5),
        excised   = CIN >> I,
        init_prev=0.1, name=name,
    )


def run_both(make, pop=10_000, n_agents=10_000, **kwargs):
    return sc.obj(ode=make(solver='ode', total_pop=pop, **kw, **kwargs).run(),
                      abm=make(solver='abm', total_pop=pop, n_agents=n_agents, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.obj({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def check_agents(make, pop=10_000, **kwargs):
    """ Check that a model is agents only: an explicit ODE refuses it before the run, and solver='auto' runs it as agents """
    with pytest.raises(ss.CapabilityError):
        make(solver='ode', total_pop=pop, **kw, **kwargs).init()
    assert make(total_pop=pop, **kw, **kwargs).init().solver == 'abm'
    return


def run_abm(make, pop=10_000, n_agents=10_000, **kwargs):
    return make(solver='abm', total_pop=pop, n_agents=n_agents, **kw, **kwargs).run()


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
        assert r.hpv16.I[-2] < 0.5*r0.hpv16.I[-2] and r.hpv18.I[-2] < r0.hpv18.I[-2], solver
        assert r.prog.delivered.sum() == pytest.approx(8000, rel=0.02)
        assert len(sim.modifiers[('hpv16', 'rel_sus', None)]) == len(sim.modifiers[('hpv18', 'rel_sus', None)]) == 1
        assert [s.name for s in sim.groups.strats].count('has_vx') <= 1
    check_close(sims, lambda sim: sim.results.hpv18.infection.sum(), rtol=0.15)
    assert sims.abm.people.has_vx.sum() > 0 and 'has_vx' not in [s.name for s in sims.abm.diseases.hpv16.graph.strats]

    with pytest.raises(KeyError, match='is keyed by diseases and by acquisition'): # Keyed by a disease and by something else
        make(lambda a, b: {a: 0.9, 'acquisition': 0.5}, solver='abm', **kw).init()
    return sims


def test_any():
    sc.heading('Testing a condition across diseases in detects= and eligible=, with the disease taken from the conditions')
    seen = sc.obj()
    def make(**kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        any_cin, any_ca = genos[0].CIN | genos[1].CIN, genos[0].CA | genos[1].CA
        def look(sim):
            if sim.solver == 'abm':
                d, u = sim.diseases, sim.people.auids
                ca = d.hpv16.CA.raw[u] | d.hpv18.CA.raw[u]
                seen.update(eligible=np.count_nonzero(~ca), found=0.9*np.count_nonzero((d.hpv16.CIN.raw[u] | d.hpv18.CIN.raw[u]) & ~ca))
            return True
        prog = ss.screen(ss.Diagnostic(sensitivity=0.9, detects=any_cin), who=~any_ca, at=[2005], when=look) # No disease=
        return ss.Sim(genos, interventions=prog, **kwargs)
    sims = run_both(make)
    for solver,sim in sims.items():
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        exp = seen
        if solver == 'ode': # Independent genotypes: from the marginals
            f = {x:[r[g][x][k-1]/10_000 for g in ['hpv16', 'hpv18']] for x in ['CIN', 'CA']} # At the start of the step it was delivered in
            no_ca = np.prod([1 - p for p in f['CA']])
            exp = sc.obj(eligible=10_000*no_ca, found=0.9*10_000*(no_ca - np.prod([1 - a - b for a,b in zip(f['CA'], f['CIN'])])))
        assert np.isclose(r.prog.eligible[k], exp.eligible) and np.isclose(r.prog.positive[k], exp.found, rtol=1e-6 if solver == 'ode' else 0.05), solver
    assert any('tracked jointly' in note for note in sims.ode.notes)
    return sims


# %% Acceptance: hpvsim_interventions/port.py, with genotypes as diseases


def test_own_product():
    sc.heading('Testing a product written as code: administer() is given the UIDs reached, its outcomes trigger a step, and delivery.reached() is who it reached')

    class Swab(ss.Product):
        """ A test of our own, which finds the infected exactly """
        def outcome_names(self):
            return ['positive']

        def administer(self, uids):
            return dict(positive=uids[self.sim.diseases.flu.I[uids]])

    class Reached(ss.Analyzer):
        """ The number each step reached, read after it has run """
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('swabbed', dtype=float), ss.Result('positive', dtype=float))

        def step(self):
            screening = self.sim.interventions.screening
            self.results.swabbed[self.ti] = len(screening.reached())
            self.results.positive[self.ti] = len(screening.reached('positive'))

    def make(**kwargs):
        screening = ss.Delivery(Swab(), coverage=0.2, interval=ss.months(3), name='screening')
        treatment = ss.treat(ss.Treatment(cure=1.0), who=screening.positive, name='treatment')
        return ss.Sim(make_geno('flu'), interventions=[screening, treatment], analyzers=Reached(), **kwargs)
    check_agents(make) # Its own code acts on agents
    sim = run_abm(make)
    r = sim.results
    assert r.screening.delivered.sum() > 0 and r.screening.positive.sum() > 0
    assert np.array_equal(r.reached.swabbed, r.screening.delivered) # An analyzer reads the step just taken, at the same index
    assert np.array_equal(r.reached.positive, r.screening.positive)
    assert np.array_equal(r.treatment.cured, r.screening.positive) # Every one found is treated, and cured
    with pytest.raises(AttributeError):
        sim.interventions.screening.reached('negative')
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    vaccine = test_vaccine_diseases()
    any_ = test_any()
    own    = test_own_product()
