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


def make_hpv(name, beta=1.5):
    """ A genotype with a lesion: S -> I -> CIN, cleared to S, and excised (undriven) back to I """
    S, I, CIN = ss.states('S I CIN')
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=ss.peryear(beta), source=I + CIN),
        clearance = I >> S @ ss.years(1),
        dysplasia = I >> CIN @ ss.years(2),
        regress   = CIN >> S @ ss.years(3),
        excised   = CIN >> I,
        init_prev=0.1, name=name,
    )


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


def test_test_diseases():
    sc.heading('Testing a test keyed by states of several diseases: independent detection, or the highest sensitivity')
    s16, s18 = [0, 0.6, 0.9], [0, 0.5, 0.8] # By S, I, CIN
    seen = sc.objdict()
    def make(combine='worst', **kwargs):
        g16, g18 = make_hpv('hpv16'), make_hpv('hpv18')
        dna = ss.Test(sensitivity={g16.I: 0.6, g16.CIN: 0.9, 'hpv18.I': 0.5, g18.CIN: 0.8}, specificity=0.95, combine=combine)
        def look(sim): # As agents: the expected positives, from each agent's states when tested
            if sim.solver == 'abm':
                d, u = sim.diseases, sim.people.auids
                p16, p18 = [np.select([d[g].I.raw[u], d[g].CIN.raw[u]], s[1:], 0) for g,s in [('hpv16', s16), ('hpv18', s18)]]
                pos = 1 - (1 - p16)*(1 - p18) if combine == 'worst' else np.maximum(p16, p18)
                seen[combine] = np.where((p16 == 0) & (p18 == 0), 0.05, pos).sum()
            return True
        prog = ss.screen(dna, at=[2005], when=look)
        return ss.Sim([g16, g18], interventions=prog, **kwargs)
    def expected(sim, combine):
        """ As an ODE: the diseases do not interact, so each person's states are independent, and the expected positives come from the marginals """
        if sim.solver == 'abm':
            return seen[combine]
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        n = {g:np.array([r[g][f'n_{x}'][k] for x in ['S', 'I', 'CIN']]) for g in ['hpv16', 'hpv18']}
        N = n['hpv16'].sum()
        p = 0.0
        for x in range(3):
            for y in range(3):
                pos = 0.05 if x == y == 0 else 1 - (1 - s16[x])*(1 - s18[y]) if combine == 'worst' else max(s16[x], s18[y])
                p += n['hpv16'][x]/N*n['hpv18'][y]/N*pos
        return p*N
    for combine in ['worst', 'max']:
        sims = run_both(make, combine=combine)
        for solver,sim in sims.items():
            exp, got = expected(sim, combine), sim.results.prog.diagnosed.sum()
            assert np.isclose(got, exp, rtol=1e-5 if solver == 'ode' else 0.03), f'{solver} {combine}: {got:.1f} vs {exp:.1f}' # v4's joint stepping moves the marginals by about 1e-6
        assert any('tracked jointly: 3 × 3 = 9 joint states per group' in note for note in sims.ode.notes)
        if combine == 'worst':
            worst = sims.ode.results.prog.diagnosed.sum()
    assert sims.ode.results.prog.diagnosed.sum() < worst

    g16, g18 = make_hpv('hpv16'), make_hpv('hpv18') # A state the disease does not have: the message names it, and that disease's states
    bad = ss.screen(ss.Test(sensitivity={g16.I: 0.6, 'hpv18.X': 0.5}), at=[2005])
    with pytest.raises(ss.UnknownEffect, match='"hpv18.X", but "hpv18" has no state "X"; its states are S, I, CIN'):
        ss.Sim([g16, g18], interventions=bad, solver='abm', **kw).run()
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    vaccine = test_vaccine_diseases()
    test = test_test_diseases()
