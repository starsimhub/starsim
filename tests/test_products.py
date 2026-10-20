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
    return ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(beta)), clearance=I >> S @ ss.years(1), init_prev=0.1, name=name)


def make_hpv(name, beta=1.5):
    """ A genotype with a lesion and cancer: S -> I -> CIN -> CA, cleared to S, and excised (undriven) back to I """
    S, I, CIN, CA = ss.states('S I CIN CA')
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=ss.peryear(beta), source=I + CIN),
        clearance = I >> S @ ss.years(1),
        dysplasia = I >> CIN @ ss.years(2),
        regress   = CIN >> S @ ss.years(3),
        invasion  = CIN >> CA @ ss.years(5),
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
    s16, s18 = [0, 0.6, 0.9, 0], [0, 0.5, 0.8, 0] # By S, I, CIN, CA
    seen = sc.objdict()
    def make(combine='worst', **kwargs):
        g16, g18 = make_hpv('hpv16'), make_hpv('hpv18')
        dna = ss.Diagnostic(sensitivity={g16.I: 0.6, g16.CIN: 0.9, 'hpv18.I': 0.5, g18.CIN: 0.8}, specificity=0.95, combine=combine)
        def look(sim): # As agents: the expected positives, from each agent's states when tested
            if sim.solver == 'abm':
                d, u = sim.diseases, sim.people.auids
                p16, p18 = [np.select([d[g].I.raw[u], d[g].CIN.raw[u]], s[1:3], 0) for g,s in [('hpv16', s16), ('hpv18', s18)]]
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
        n = {g:np.array([r[g][f'n_{x}'][k] for x in ['S', 'I', 'CIN', 'CA']]) for g in ['hpv16', 'hpv18']}
        N = n['hpv16'].sum()
        p = 0.0
        for x in range(4):
            for y in range(4):
                pos = 0.05 if x in [0, 3] and y in [0, 3] else 1 - (1 - s16[x])*(1 - s18[y]) if combine == 'worst' else max(s16[x], s18[y])
                p += n['hpv16'][x]/N*n['hpv18'][y]/N*pos
        return p*N
    for combine in ['worst', 'max']:
        sims = run_both(make, combine=combine)
        for solver,sim in sims.items():
            exp, got = expected(sim, combine), sim.results.prog.positive.sum()
            assert np.isclose(got, exp, rtol=1e-5 if solver == 'ode' else 0.03), f'{solver} {combine}: {got:.1f} vs {exp:.1f}' # v4's joint stepping moves the marginals by about 1e-6
        assert any('tracked jointly: 4 × 4 = 16 joint states per group' in note for note in sims.ode.notes)
        if combine == 'worst':
            worst = sims.ode.results.prog.positive.sum()
    assert sims.ode.results.prog.positive.sum() < worst

    g16, g18 = make_hpv('hpv16'), make_hpv('hpv18') # A state the disease does not have: the message names it, and that disease's states
    bad = ss.screen(ss.Diagnostic(sensitivity={g16.I: 0.6, 'hpv18.X': 0.5}), at=[2005])
    with pytest.raises(ss.UnknownEffect, match='for "X", which is not a state of "hpv18". Its states are S, I, CIN, CA'):
        ss.Sim([g16, g18], interventions=bad, solver='abm', **kw).run()
    return sims


def test_outcomes():
    sc.heading('Testing a test with several results: colposcopy, the worst over genotypes, a follow-up for some results, and a delay for one')
    table = dict(lsil=dict(I=0.15, CIN=0.2), hsil=dict(CIN=0.7, CA=0.1), cancer=dict(CA=0.85)) # P(result | state), the same for each genotype
    seen = sc.objdict()
    def make(delay=None, **kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        results = {res:{g[x]:p for g in genos for x,p in probs.items()} for res,probs in reversed(table.items())} | dict(normal=None) # Most severe first
        colpo = ss.Diagnostic(results=results, combine='worst', delay=delay)
        excise = ss.Treatment(moves={g.excised: 0.9 for g in genos}) # Across genotypes, with no fan-out
        def look(sim): # As agents: each result expected, from each agent's states when tested
            if sim.solver == 'abm':
                u = sim.people.auids
                tails = 1.0
                for g in genos:
                    d = sim.diseases[g.name]
                    p = np.array([[sum(table[r].get(x, 0) for r in list(table)[k:]) for k in range(3)] for x in ['S', 'I', 'CIN', 'CA']])
                    state = np.select([d[x].raw[u] for x in ['S', 'I', 'CIN', 'CA']], [0, 1, 2, 3])
                    tails = tails*(1 - p[state])
                tails = np.column_stack([np.ones(len(u)), 1 - tails, np.zeros(len(u))])
                seen.expected = (tails[:, :-1] - tails[:, 1:]).sum(axis=0)
            return True
        prog = ss.screen(colpo, at=[2005], when=look)
        prog_hsil = ss.treat(excise, who=prog.hsil) # A follow-up of some results
        prog_cancer = ss.screen(ss.Diagnostic(name='biopsy'), who=prog.cancer)
        return ss.Sim(genos, interventions=[prog, prog_hsil, prog_cancer], **kwargs)
    def expected(sim):
        """ As an ODE: from the marginals of independent genotypes, P(worst ≥ k) = 1 − ∏(1 − P_g(≥ k)) """
        if sim.solver == 'abm':
            return seen.expected
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        p = np.array([[sum(table[res].get(x, 0) for res in list(table)[j:]) for j in range(3)] for x in ['S', 'I', 'CIN', 'CA']])
        miss = 1.0
        for g in ['hpv16', 'hpv18']:
            n = np.array([r[g][f'n_{x}'][k] for x in ['S', 'I', 'CIN', 'CA']])
            N = n.sum()
            miss = miss*(1 - n/N @ p)
        tails = np.concatenate([[1], 1 - miss, [0]])
        return (tails[:-1] - tails[1:])*N
    sims = run_both(make)
    for solver,sim in sims.items():
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        got = np.array([r.prog[name][k] for name in ['normal', 'lsil', 'hsil', 'cancer']])
        assert np.allclose(got, expected(sim), rtol=1e-6 if solver == 'ode' else 0.08), f'{solver}: {got} vs {expected(sim)}' # As agents, about 3% sampling noise for the smallest result (seeds 1-6)
        assert np.isclose(got.sum(), r.prog.delivered[k])
        assert np.isclose(r.prog_hsil.delivered[k], got[2]) and np.isclose(r.prog_cancer.delivered[k], got[3]) # The follow-ups
        cured, failed = r.prog_hsil.cured.sum(), r.prog_hsil.failed.sum()
        assert np.isclose(cured, 0.9*(cured + failed), rtol=1e-6 if solver == 'ode' else 0.05) and r.prog_hsil.unneeded.sum() > 0 # Some with HSIL from cancer
    check_close(sims, lambda sim: sum(sim.results.prog[name].sum() for name in ['lsil', 'hsil', 'cancer']), rtol=0.25) # At 3-month steps, agents have about 20% fewer people in CIN than the ODE (v4's own agents too)

    sims = run_both(make, delay=dict(hsil=ss.months(3))) # A delay for one result: its follow-up a step later
    for solver,sim in sims.items():
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        assert list(np.flatnonzero(r.prog_hsil.delivered)) == [k + 1] and list(np.flatnonzero(r.prog_cancer.delivered)) == [k], solver
        assert np.isclose(r.prog_hsil.delivered[k + 1], r.prog.hsil[k + 1], rtol=0.02 if solver == 'ode' else 0.05) # The result is counted when it comes
    assert [s.labels for s in sims.ode.groups.strats if s.name == 'pending_prog'] == [['none', 'hsil 1']]

    # A follow-up of any result but the last; a result's name has to be one; probabilities above 1 are an error
    g = make_hpv('hpv')
    colpo = ss.Diagnostic(results=dict(cancer={g.CA: 0.9}, abnormal={g.CIN: 0.9}, normal=None))
    screening = ss.screen(colpo, at=[2005])
    refer = ss.screen(ss.Diagnostic(name='biopsy'), who=screening.cancer | screening.abnormal)
    sim = ss.Sim(g, interventions=[screening, refer], solver='ode', **kw).run()
    assert np.isclose(sim.results.refer.delivered.sum(), sim.results.screening.cancer.sum() + sim.results.screening.abnormal.sum())
    with pytest.raises(AttributeError, match='its outcomes .events. are cancer, abnormal, normal'):
        screening.hsil
    with pytest.raises(ValueError, match='sum to at most 1'):
        ss.Sim(g, interventions=ss.screen(ss.Diagnostic(results=dict(c={g.CIN: 0.6}, b={g.CIN: 0.6}, a=None))), solver='abm', **kw).init().run()
    return sims


def test_any():
    sc.heading('Testing ss.any() across diseases in detects= and eligible=, with the disease taken from the conditions')
    seen = sc.objdict()
    def make(**kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        any_cin, any_ca = ss.any(g.CIN for g in genos), ss.any(g.CA for g in genos)
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
            f = {x:[r[g][f'n_{x}'][k]/10_000 for g in ['hpv16', 'hpv18']] for x in ['CIN', 'CA']}
            no_ca = np.prod([1 - p for p in f['CA']])
            exp = sc.objdict(eligible=10_000*no_ca, found=0.9*10_000*(no_ca - np.prod([1 - a - b for a,b in zip(f['CA'], f['CIN'])])))
        assert np.isclose(r.prog.eligible[k], exp.eligible) and np.isclose(r.prog.positive[k], exp.found, rtol=1e-6 if solver == 'ode' else 0.05), solver
    assert any('tracked jointly' in note for note in sims.ode.notes)
    return sims


def test_strain_efficacy():
    sc.heading('Testing an efficacy by a strain that susceptibles do not carry: an error, pointing to efficacy by disease')
    def make(eff, **kwargs):
        S, I = ss.states('S I')
        hpv = ss.Disease(infection=ss.transmission('S -> I', beta=ss.peryear(1.2)), clearance=I >> S @ ss.years(1), init_prev=0.1, name='hpv')
        hpv.stratify(genotype=['hpv16', 'hpv18'], kind='strain') # In v4 a name only suggests a kind
        return ss.Sim(hpv, interventions=ss.vaccinate(ss.Vaccine(efficacy=eff), at=[2005]), **kwargs)
    for solver in ['abm', 'ode']:
        with pytest.raises(ss.InvalidAdjustment, match='"genotype", but people in S of "hpv" have no level of it.*efficacy=dict.hpv16=0.95'):
            make(ss.by(genotype=[0.95, 0.0]), solver=solver, **kw).init()
    sim = make(dict(transmission=ss.by(genotype=[0.5, 0.0])), solver='abm', **kw).run() # Infectious people do carry a genotype
    assert sim.results.vaccinate.delivered.sum() > 0
    return sim


def test_disease_list():
    sc.heading('Testing disease= as a list or an ss.Strains (HP7): a bivalent vaccine, a test and a cure of each, and their counts')
    def make(dis=None, vx=True, **kwargs):
        g16, g18 = make_geno('hpv16'), make_geno('hpv18')
        diseases = ss.Strains('hpv', [g16, g18]) if dis == 'strains' else [g16, g18]
        given = diseases if dis == 'strains' else [g16, g18]
        ivs = [ss.vaccinate(ss.Vaccine(efficacy=0.95 if vx else 0.0), coverage=0.8, at=[2001], disease=given, name='vx')]
        if kwargs.get('solver') == 'abm':
            dna = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=1.0), coverage=0.5, at=[2005], disease=given, name='dna')
            ivs += [dna, ss.treat(ss.Treatment(cure=1.0), who=dna.positive, disease=given, name='tx')]
        return ss.Sim(diseases, interventions=ivs, **kwargs)
    base = run_both(lambda **k: make(vx=False, **k))
    sims = run_both(make)
    for solver,sim in sims.items(): # The vaccine protects against both
        r, r0 = sim.results, base[solver].results
        assert r.hpv16.n_I[8] < 0.5*r0.hpv16.n_I[8] and r.hpv18.n_I[8] < 0.5*r0.hpv18.n_I[8], solver
    r = sims.abm.results
    assert r.dna.positive.sum() > 0 and r.tx.cured.select(disease='hpv16').sum() > 0 and r.tx.cured.select(disease='hpv18').sum() > 0 # Detected and cured in each
    assert sims.abm.results.hpv18.n_I[21] < sims.abm.results.hpv18.n_I[20]
    family = make('strains', solver='abm', n_agents=10_000, **kw).run().results # The same as the list
    assert np.array_equal(family.hpv18.n_I, r.hpv18.n_I) and family.tx.cured.sum() == r.tx.cured.sum()

    # A scalar cure is the dict over every infected state; a test's count and cost are one per person tested
    def cure(c, **kwargs):
        g = make_hpv('hpv16')
        cyto = ss.Diagnostic(results=dict(high=dict(CIN=0.9, CA=1.0), low=dict(I=0.5), normal=None), cost=10)
        screening = ss.screen(cyto, at=[2005], name='screening')
        return ss.Sim(g, interventions=[screening, ss.treat(ss.Treatment(cure=c, to='S'), who=screening.high, name='tx')], solver='abm', n_agents=20_000, **kw, **kwargs).run().results
    scalar, bystate = cure(0.936), cure(dict(I=0.936, CIN=0.936)) # Its infected states
    assert np.array_equal(scalar.hpv16.n_I, bystate.hpv16.n_I) and scalar.tx.cured.sum() == bystate.tx.cured.sum() > 0
    assert scalar.screening.delivered.sum() == 20_000 and scalar.screening.cost.sum() == 200_000

    # One product for one disease, delivered for another, says to give both
    g16, g18 = make_geno('hpv16'), make_geno('hpv18')
    test = ss.Diagnostic(sensitivity=0.9)
    with pytest.raises(ss.UnknownDisease, match=r"disease=\['hpv16', 'hpv18'\]"):
        ss.Sim([g16, g18], interventions=[ss.screen(test, at=[2005], disease=g16, name='a'), ss.screen(test, at=[2006], disease=g18, name='b')], solver='abm', **kw).init()
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    vaccine = test_vaccine_diseases()
    tests = test_test_diseases()
    outcomes = test_outcomes()
    any_ = test_any()
    strain = test_strain_efficacy()
    dlist  = test_disease_list()
