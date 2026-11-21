"""
Test products over several diseases, tests with several results, and programs: numbers are people, priority, routine delivery, and doses (the slower and more exhaustive tests)
"""
import numpy as np
import sciris as sc
import starsim as ss
from corpus import make_hpv, run_each, check_close

hpv_kw = dict(verbose=0, start=2000, stop=2010, dt=ss.months(3), total_pop=10_000) # For the models of HPV genotypes


# %% Products over several diseases

def test_outcomes():
    sc.heading('Testing a test with several results: colposcopy, the worst over genotypes, and a follow-up for some results')
    table = dict(lsil=dict(I=0.15, CIN=0.2), hsil=dict(CIN=0.7, CA=0.1), cancer=dict(CA=0.85)) # P(result | state), the same for each genotype
    p = np.array([[sum(table[r].get(x, 0) for r in list(table)[k:]) for k in range(3)] for x in ['S', 'I', 'CIN', 'CA']]) # P(result at least as severe as each), by state
    seen = sc.obj()
    def make(**kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        results = {res:{g[x]:prob for g in genos for x,prob in probs.items()} for res,probs in reversed(table.items())} | dict(normal=None) # Most severe first
        colpo = ss.Diagnostic(outcomes=results)
        excise = ss.Treatment(cure={g.excised: 0.9 for g in genos}) # Across genotypes, with no fan-out
        def look(sim): # As agents: each result expected, from each agent's states when tested
            if isinstance(sim.solver, ss.solvers.abm):
                u = sim.people.auids
                tails = 1.0
                for g in genos:
                    d = sim.diseases[g.name]
                    state = np.select([d[x].raw[u] for x in ['S', 'I', 'CIN', 'CA']], [0, 1, 2, 3])
                    tails = tails*(1 - p[state])
                tails = np.column_stack([np.ones(len(u)), 1 - tails, np.zeros(len(u))])
                seen.expected = (tails[:, :-1] - tails[:, 1:]).sum(axis=0)
            return True
        prog = ss.screen(colpo, who=look, when=[2005])
        prog_hsil = ss.treat(excise, when=prog.hsil) # A follow-up of some results
        prog_cancer = ss.screen(ss.Diagnostic(name='biopsy'), when=prog.cancer)
        return ss.Sim(genos, interventions=[prog, prog_hsil, prog_cancer], **hpv_kw, **kwargs)
    def expected(sim):
        """ As an ODE: from the marginals of independent genotypes, P(worst ≥ k) = 1 − ∏(1 − P_g(≥ k)) """
        if isinstance(sim.solver, ss.solvers.abm):
            return seen.expected
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        miss = 1.0
        for g in ['hpv16', 'hpv18']:
            n = np.array([r[g][x][k] for x in ['S', 'I', 'CIN', 'CA']]) # At the start of the step it was delivered in
            miss = miss*(1 - n/n.sum() @ p)
        tails = np.concatenate([[1], 1 - miss, [0]])
        return (tails[:-1] - tails[1:])*n.sum()
    sims = run_each(make, solvers=['ode', 'abm'], n_agents=10_000)
    for solver,sim in sims.items():
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        got = np.array([r.prog[name][k] for name in ['normal', 'lsil', 'hsil', 'cancer']])
        assert np.allclose(got, expected(sim), rtol=1e-6 if solver == 'ode' else 0.08), f'{solver}: {got} vs {expected(sim)}' # As agents, about 3% sampling noise for the smallest result (seeds 1-6)
        assert np.isclose(got.sum(), r.prog.delivered[k])
        assert np.isclose(r.prog_hsil.delivered[k], got[2]) and np.isclose(r.prog_cancer.delivered[k], got[3]) # The follow-ups
        cured, failed = r.prog_hsil.cured.sum(), r.prog_hsil.failed.sum()
        assert np.isclose(cured, 0.9*(cured + failed), rtol=1e-6 if solver == 'ode' else 0.05) and r.prog_hsil.unneeded.sum() > 0 # Some with HSIL from cancer
    assert any('tracked jointly' in note for note in sims.ode.notes)

    return sims


# %% Programs

def make_sis(name='sis', beta=0.25):
    """ An SIS with a cure that only a delivery moves people along """
    S, I = ss.states(2)
    return ss.Disease(infection=S >> I @ beta*I, recovery=I >> S @ ss.days(20), cure=I >> S @ 0, init_prev=0.01, name=name)


def run_both(make, stop=100, pop=50_000, n_agents=10_000, **kwargs):
    """ Run the sim that `make` returns as an ODE and as agents (each agent standing for several people) """
    return run_each(make, solvers=['ode', 'abm'], total_pop=pop, n_agents=n_agents, stop=stop, verbose=0, unit='day', **kwargs)


def test_program():
    sc.heading('Testing a program run as an ODE and as agents: numbers are people, priority by tier and by score, routine delivery once per spell, and doses')
    seen = sc.obj()
    def make(**kwargs):
        sis = make_sis(beta=0.15).stratify(risk={'low': 0.8, 'high': 0.2}).stratify(age=[0, 15, 65])
        def look(sim): # Results read during the run are people, with 5 people to an agent
            if sim.ti == 30:
                seen[sim.solver.name] = sc.obj(I=sim.results.sis.I[sim.ti], delivered=sim.results.tiers.delivered[sim.ti - 1]) # The state at the start of the step, and the deliveries of the last
            return True
        tiers = ss.vaccinate(ss.Vaccine(efficacy=0.0, name='v_tiers'), who=look, capacity=ss.capacity(ss.perday(500), priority=[sis.risk == 'high']), name='tiers') # All of the high-risk first (10,000 people, 20 days)
        oldest = ss.vaccinate(ss.Vaccine(efficacy=0.0, name='v_oldest'), capacity=ss.capacity(ss.perday(500), priority=ss.person.age), name='oldest')
        drug = ss.Treatment(cure={sis.cure: 1.0})
        clinic = ss.treat(drug, who=sis.I, coverage=0.3, name='clinic') # Routine delivery is once per spell of being eligible: each infection is offered treatment once
        vx = ss.Vaccine(efficacy=0.8, doses=[0.5, 1.0])
        first = ss.vaccinate(vx, when=[5], coverage=0.6, name='first')
        second = ss.vaccinate(vx, who=vx.ever, when=[35], coverage=0.5, name='second') # The second dose in a round of its own (every solver)
        held = ss.vaccinate(ss.Vaccine(efficacy=0.0, name='other'), who=vx.held, when=[60], name='held')
        return ss.Sim(sis, interventions=[tiers, oldest, clinic, first, second, held], **kwargs)
    sims = run_both(make, stop=80)
    for solver,sim in sims.items():
        res = sim.results
        by = res.tiers.delivered.by('risk').values
        assert np.allclose(res.tiers.delivered[:79], 500, rtol=0.02) and np.allclose(by[0:20, 1], 500) and np.allclose(by[0:20, 0], 0), f'{solver}: {by[:22]}'
        assert seen[solver].I == res.sis.I[30] and seen[solver].delivered == res.tiers.delivered[29] # Not scaled again at the end
        by = res.oldest.delivered.by('age').values
        first = int((res.sis.S.by('age').values[0, 2] + res.sis.I.by('age').values[0, 2])//500) - 1 # Oldest first
        assert np.allclose(by[0:first, 2], 500) and by[0:first, :2].sum() == 0, f'{solver}: {by[:first+3]}'
        assert np.isclose(res.clinic.delivered[1:].sum(), 0.3*res.sis.infection[:-1].sum(), rtol=0.1), solver # As agents, those whose dwell rounds to no steps recover in the step they are infected (a few %)
        assert np.isclose(res.first.delivered[5], 30_000, rtol=0.02)
        assert res.second.delivered[:35].sum() == 0 and np.isclose(res.second.delivered[35], 15_000, rtol=0.03) # Half of those who had the first
        assert np.isclose(res.held.delivered[60], 30_000, rtol=0.02) # Everyone who had a dose
    assert sims.abm.pop_scale == 5
    check_close(sims, lambda sim: seen[sim.solver.name].I, rtol=0.15)
    check_close(sims, lambda sim: sim.results.clinic.delivered.sum(), rtol=0.15)
    assert [s.labels for s in sims.ode.groups.strats if s.name == 'has_vx'][0] == [False, 'dose 1', 'dose 2']
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
