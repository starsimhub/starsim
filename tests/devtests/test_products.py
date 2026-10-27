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
    return sc.objdict(ode=make(solver='ode', total_pop=pop, **kw, **kwargs).run(),
                      abm=make(solver='abm', total_pop=pop, n_agents=n_agents, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.objdict({solver:func(sim) for solver,sim in sims.items()})
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


def test_test_diseases():
    sc.heading('Testing a test keyed by states of several diseases: each detected independently')
    s16, s18 = [0, 0.6, 0.9, 0], [0, 0.5, 0.8, 0] # By S, I, CIN, CA
    seen = sc.objdict()
    def make(**kwargs):
        g16, g18 = make_hpv('hpv16'), make_hpv('hpv18')
        dna = ss.Diagnostic(sensitivity={g16.I: 0.6, g16.CIN: 0.9, 'hpv18.I': 0.5, g18.CIN: 0.8}, specificity=0.95)
        def look(sim): # As agents: the expected positives, from each agent's states when tested
            if sim.solver == 'abm':
                d, u = sim.diseases, sim.people.auids
                p16, p18 = [np.select([d[g].I.raw[u], d[g].CIN.raw[u]], s[1:3], 0) for g,s in [('hpv16', s16), ('hpv18', s18)]]
                pos = 1 - (1 - p16)*(1 - p18)
                seen.pos = np.where((p16 == 0) & (p18 == 0), 0.05, pos).sum()
            return True
        prog = ss.screen(dna, at=[2005], when=look)
        return ss.Sim([g16, g18], interventions=prog, **kwargs)
    def expected(sim):
        """ As an ODE: the diseases do not interact, so each person's states are independent, and the expected positives come from the marginals """
        if sim.solver == 'abm':
            return seen.pos
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        n = {g:np.array([r[g][x][k-1] for x in ['S', 'I', 'CIN', 'CA']]) for g in ['hpv16', 'hpv18']} # At the start of the step it was delivered in
        N = n['hpv16'].sum()
        p = 0.0
        for x in range(4):
            for y in range(4):
                pos = 0.05 if x in [0, 3] and y in [0, 3] else 1 - (1 - s16[x])*(1 - s18[y])
                p += n['hpv16'][x]/N*n['hpv18'][y]/N*pos
        return p*N
    sims = run_both(make)
    for solver,sim in sims.items():
        exp, got = expected(sim), sim.results.prog.positive.sum()
        assert np.isclose(got, exp, rtol=1e-5 if solver == 'ode' else 0.03), f'{solver}: {got:.1f} vs {exp:.1f}' # v4's joint stepping moves the marginals by about 1e-6
    assert any('tracked jointly: 4 × 4 = 16 joint states per group' in note for note in sims.ode.notes)

    g16, g18 = make_hpv('hpv16'), make_hpv('hpv18') # A state the disease does not have: the message names it, and that disease's states
    bad = ss.screen(ss.Diagnostic(sensitivity={g16.I: 0.6, 'hpv18.X': 0.5}), at=[2005])
    with pytest.raises(ss.errors.UnknownEffect, match='for "X", which is not a state of "hpv18". Its states are S, I, CIN, CA'):
        ss.Sim([g16, g18], interventions=bad, solver='abm', **kw).run()
    return sims


def test_outcomes():
    sc.heading('Testing a test with several results: colposcopy, the worst over genotypes, a follow-up for some results, and a delay for one')
    table = dict(lsil=dict(I=0.15, CIN=0.2), hsil=dict(CIN=0.7, CA=0.1), cancer=dict(CA=0.85)) # P(result | state), the same for each genotype
    seen = sc.objdict()
    def make(delay=None, **kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        results = {res:{g[x]:p for g in genos for x,p in probs.items()} for res,probs in reversed(table.items())} | dict(normal=None) # Most severe first
        colpo = ss.Diagnostic(results=results, delay=delay)
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
            n = np.array([r[g][x][k-1] for x in ['S', 'I', 'CIN', 'CA']]) # At the start of the step it was delivered in
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

    check_agents(make, delay=dict(hsil=ss.months(3))) # A delay for one result (agents only): its follow-up a step later
    r = run_abm(make, delay=dict(hsil=ss.months(3))).results
    k = int(np.flatnonzero(r.prog.delivered)[0])
    assert list(np.flatnonzero(r.prog_hsil.delivered)) == [k + 1] and list(np.flatnonzero(r.prog_cancer.delivered)) == [k]
    assert np.isclose(r.prog_hsil.delivered[k + 1], r.prog.hsil[k + 1], rtol=0.05) # The result is counted when it comes

    # A follow-up of any result but the last (positive); a result's name has to be one, and not positive or negative; probabilities above 1 are an error
    g = make_hpv('hpv')
    colpo = ss.Diagnostic(results=dict(cancer={g.CA: 0.9}, abnormal={g.CIN: 0.9}, normal=None))
    screening = ss.screen(colpo, at=[2005])
    refer = ss.screen(ss.Diagnostic(name='biopsy'), who=screening.positive)
    for solver in ['ode', 'abm']:
        r = ss.Sim(g, interventions=[screening, refer], solver=solver, **kw).run().results
        assert np.isclose(r.refer.delivered.sum(), r.screening.cancer.sum() + r.screening.abnormal.sum()) and r.refer.delivered.sum() > 0, solver
        assert np.isclose(r.screening.negative.sum(), r.screening.normal.sum()) and np.isclose(r.screening.positive.sum() + r.screening.negative.sum(), r.screening.delivered.sum()), solver
    with pytest.raises(AttributeError, match='its outcomes .events. are cancer, abnormal, normal, positive'):
        screening.hsil
    with pytest.raises(ValueError, match='cannot be called positive'):
        ss.Diagnostic(results=dict(positive={g.CIN: 0.9}, normal=None))
    with pytest.raises(ValueError, match='sum to at most 1'):
        ss.Sim(g, interventions=ss.screen(ss.Diagnostic(results=dict(c={g.CIN: 0.6}, b={g.CIN: 0.6}, a=None))), solver='abm', **kw).init().run()
    return sims


def test_result_none():
    sc.heading('Testing the None key of results=: people in none of the states keyed, once however many diseases, unlike a result by state for each')
    seen = sc.objdict()
    def run(abn):
        g16, g18 = make_hpv('hpv16'), make_hpv('hpv18')
        keyed = [x for x in abn if x is not None]
        def look(sim): # The expected abnormal results, from each agent's states when tested
            d, u = sim.diseases, sim.people.auids
            p = [sum(d[g][x].raw[u]*abn[x] for x in keyed) for g in ['hpv16', 'hpv18']]
            none = np.prod([~np.any([d[g][x].raw[u] for x in keyed], axis=0) for g in ['hpv16', 'hpv18']], axis=0)
            seen.expected = (1 - (1 - p[0])*(1 - p[1]) + none*abn.get(None, 0)).sum()
            seen.n_sus = (d.hpv16.S.raw[u] & d.hpv18.S.raw[u]).sum() # Susceptible to both
            return True
        prog = ss.screen(ss.Diagnostic(results=dict(abnormal=abn, normal=None)), disease=[g16, g18], at=[2005], when=look, name='prog')
        r = ss.Sim([g16, g18], interventions=prog, solver='abm', n_agents=10_000, **kw).run().results.prog
        k = int(np.flatnonzero(r.delivered)[0])
        assert np.isclose(r.abnormal[k], seen.expected, rtol=0.05), f'{r.abnormal[k]:.0f} abnormal, not {seen.expected:.0f}'
        return seen.expected, seen.n_sus
    by_state, n_sus = run(dict(CIN=0.9, S=0.1, I=0.1)) # 0.1 for each disease: about 0.19 for someone susceptible to both
    by_none, _ = run({'CIN': 0.9, 'CA': 0.0, None: 0.1}) # 0.1 for someone in neither CIN nor CA of either
    assert by_state - by_none > 0.05*n_sus
    return


def test_strain_efficacy():
    sc.heading('Testing an efficacy by strain: against acquiring each strain, as agents and compartments; and against transmission by the strain carried')
    def make(eff, **kwargs):
        S, I = ss.states('S I')
        hpv = ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(1.2) * I), clearance=I >> S @ ss.years(1), init_prev=0.1, name='hpv', strains=ss.Strains(['hpv16', 'hpv18'], name='genotype'))
        return ss.Sim(hpv, interventions=ss.vaccinate(ss.Vaccine(efficacy=eff), at=[2005]), **kwargs)
    for solver in ['abm', 'ode']:
        res = [make(eff, solver=solver, **kw).run().results.hpv.infection.by('genotype').values[-1] for eff in [0.0, ss.by(genotype=[0.95, 0.0])]]
        assert res[1][0] < 0.5*res[0][0] and res[1][1] > 0.9*res[0][1], f'{solver}: {res}' # Against hpv16 only
    sim = make(dict(transmission=ss.by(genotype=[0.5, 0.0])), solver='abm', **kw).run() # Infectious people do carry a genotype
    assert sim.results.vaccinate.delivered.sum() > 0
    return sim


def test_disease_list():
    sc.heading('Testing disease= as a list or strains with courses of their own (agents only): a bivalent vaccine, a test and a cure of each, and their counts')
    def make(dis=None, vx=True, **kwargs):
        g16, g18 = make_geno('hpv16'), make_geno('hpv18')
        given = ss.Disease(name='hpv', strains=ss.Strains([g16, g18])) if dis == 'strains' else [g16, g18]
        ivs = [ss.vaccinate(ss.Vaccine(efficacy=0.95 if vx else 0.0), coverage=0.8, at=[2001], disease=given, name='vx')]
        if kwargs.get('solver') == 'abm':
            dna = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=1.0), coverage=0.5, at=[2005], disease=given, name='dna')
            ivs += [dna, ss.treat(ss.Treatment(cure=1.0), who=dna.positive, disease=given, name='tx')]
        return ss.Sim(given, interventions=ivs, **kwargs)
    check_agents(make) # A product keyed by the diseases has a compartmental form (see test_vaccine_diseases())
    base, sim = run_abm(make, vx=False), run_abm(make)
    r, r0 = sim.results, base.results
    assert r.hpv16.I[8] < 0.5*r0.hpv16.I[8] and r.hpv18.I[8] < 0.5*r0.hpv18.I[8] # The vaccine protects against both
    assert r.dna.positive.sum() > 0 and r.tx.cured.select(disease='hpv16').sum() > 0 and r.tx.cured.select(disease='hpv18').sum() > 0 # Detected and cured in each
    assert r.hpv18.I[21] < r.hpv18.I[20]
    family = make('strains', solver='abm', n_agents=10_000, **kw).run().results # The same as the list
    assert np.array_equal(family.hpv18.I, r.hpv18.I) and family.tx.cured.sum() == r.tx.cured.sum()

    # A scalar cure is the dict over every infected state; a test's count and cost are one per person tested
    def cure(c, **kwargs):
        g = make_hpv('hpv16')
        cyto = ss.Diagnostic(results=dict(high=dict(CIN=0.9, CA=1.0), low=dict(I=0.5), normal=None), cost=10)
        screening = ss.screen(cyto, at=[2005], name='screening')
        return ss.Sim(g, interventions=[screening, ss.treat(ss.Treatment(cure=c, to='S'), who=screening.high, name='tx')], solver='abm', n_agents=20_000, **kw, **kwargs).run().results
    scalar, bystate = cure(0.936), cure(dict(I=0.936, CIN=0.936)) # Its infected states
    assert np.array_equal(scalar.hpv16.I, bystate.hpv16.I) and scalar.tx.cured.sum() == bystate.tx.cured.sum() > 0
    assert scalar.screening.delivered.sum() == 20_000 and scalar.screening.cost.sum() == 200_000

    # One product for one disease, delivered for another, says to give both
    g16, g18 = make_geno('hpv16'), make_geno('hpv18')
    test = ss.Diagnostic(sensitivity=0.9)
    with pytest.raises(ss.errors.UnknownDisease, match=r"disease=\['hpv16', 'hpv18'\]"):
        ss.Sim([g16, g18], interventions=[ss.screen(test, at=[2005], disease=g16, name='a'), ss.screen(test, at=[2006], disease=g18, name='b')], solver='abm', **kw).init()
    return sim


# %% Acceptance: hpvsim_interventions/port.py, with genotypes as diseases

def hpv_vaccine(name, genos):
    """ port.py vaccine(): one product over the genotype diseases, efficacy by genotype (HPVsim's products_vx.csv) times the dose """
    rel = dict(bivalent=dict(hpv16=1.0, hpv18=1.0, hi5=0.5), nonavalent=dict(hpv16=1.0, hpv18=1.0, hi5=1.0))[name]
    return ss.Vaccine(efficacy={g:0.95*rel[g.name] for g in genos}, doses=[0.85, 1.0], name=name)


def test_hpv_vaccines():
    sc.heading('Acceptance: bivalent against nonavalent over genotype diseases, two doses 6 months apart (port.py vaccination()), as agents')
    def make(product=None, **kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.3), make_hpv('hi5', beta=1.1)]
        progs = []
        if product:
            vx = hpv_vaccine(product, genos) # The only change between the scenarios
            progs = [ss.vaccinate(vx, who=~vx.ever, coverage=ss.probperyear(0.5), start=2002, name='dose1'),
                     ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.months(6)), coverage=0.9, name='dose2')]
        return ss.Sim(genos, interventions=progs, **kwargs)
    inf = lambda sim, g: sim.results[g].infection[sim.t.yearvec >= 2002].sum()
    check_agents(make, product='bivalent') # The second dose reads vx.since, which only agents keep
    out = sc.objdict({key:run_abm(make, product=None if key == 'base' else key) for key in ['base', 'bivalent', 'nonavalent']})
    base, bi, nona = out.values()
    drop = sc.objdict({k:np.array([1 - inf(sim, g)/inf(base, g) for g in ['hpv16', 'hpv18', 'hi5']]) for k,sim in dict(bi=bi, nona=nona).items()})
    assert drop.bi[2] < 0.75*drop.nona[2] and abs(drop.bi[0] - drop.nona[0]) < 0.05, drop # Less against hi5, the same against hpv16
    r = bi.results
    start = int(np.flatnonzero(r.dose1.delivered)[0])
    assert r.dose2.delivered[:start + 2].sum() == 0 and r.dose2.delivered[start + 2] > 0 # 6 months at quarterly steps
    return out


def test_hpv_screen_treat():
    sc.heading('Acceptance: HPV DNA across genotypes, then a treatment visit with a capacity and a queue, then ablation or excision across genotypes (port.py algo_a), as agents')
    def make(**kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.3)]
        tx = lambda name, eff: ss.Treatment(moves={g.excised: eff for g in genos}, name=name) # port.py treat_all_genotypes(), with no fan-out
        hpv_dna = ss.Diagnostic(sensitivity={g[x]:p for g in genos for x,p in dict(I=0.9, CIN=0.95).items()}, specificity=0.97)
        any_ca = genos[0].CA | genos[1].CA
        primary = ss.screen(hpv_dna, who=hpv_dna.since >= ss.years(5), coverage=ss.probperyear(0.3), start=2005, name='primary')
        tx_visit = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.9, detects=~any_ca), who=primary.positive, coverage=0.8, capacity=ss.perweek(10), queue=ss.months(6), name='tx_visit')
        ablate = ss.treat(tx('ablation', 0.936), who=tx_visit.positive, name='ablate')
        excise = ss.treat(tx('excision', 0.81), who=tx_visit.negative, name='excise')
        return ss.Sim(genos, interventions=[primary, tx_visit, ablate, excise], **kwargs)
    check_agents(make) # A queue, and hpv_dna.since
    sim = run_abm(make)
    cap = 10*365.25/4/7 # 10 a week, about 13 weeks a quarter
    r = sim.results
    q = r.tx_visit
    assert np.all(q.delivered <= cap + 1.1) and q.waiting.max() > 0 and q.never_ran.sum() > 0
    assert np.isclose(r.ablate.delivered.sum() + r.excise.delivered.sum(), q.delivered.sum(), rtol=0.01)
    cured = r.ablate.cured.by('disease').values.sum(axis=0)
    assert np.all(cured > 0) and r.ablate.unneeded.sum() > 0 # Ablation cures CIN of each genotype; HPV+ without CIN needs none
    assert r.primary.delivered[:-1].sum() < 2*10_000 # Nobody screened twice in 5 years
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    tests = test_test_diseases()
    outcomes = test_outcomes()
    test_result_none()
    strain = test_strain_efficacy()
    dlist  = test_disease_list()
    hpv_vx = test_hpv_vaccines()
    hpv_st = test_hpv_screen_treat()
