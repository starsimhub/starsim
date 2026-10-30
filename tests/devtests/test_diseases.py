"""
Run tests of disease models
"""

# %% Imports and settings
import pytest
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import starsim.library as ssl

n_agents = 2_000
do_plot = False
sc.options(interactive=do_plot) # Assume not running interactively


@sc.timer()
def test_fetal_health(do_plot=do_plot):
    """
    Test the FetalHealth module across three scenarios:
      1. Baseline: no disease — birth weights ~3400g at term
      2. Disease: SIR infection worsens fetal outcomes via a connector
      3. Treatment: treating infected pregnant women partially reverses damage
    """
    sc.heading('Testing fetal health module')

    sim_kw = dict(
        n_agents=5_000,
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
    )
    def demog():
        return [
            ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), burnin=True),
            ss.Deaths(death_rate=ss.time.freq(10/1010*1000, 'year')),
        ]
    sir_kw = dict(dur_inf=ss.lognorm_ex(mean=ss.years(6), std=ss.years(1)), beta=0.1, init_prev=0.1)

    def mean_bw(sim):
        bw = sim.results.fetal_health.mean_birth_weight
        return np.nanmean(bw) # NaN on steps with no births

    # -- 1. Baseline (no disease) --
    sim_baseline = ss.Sim(
        demographics=demog(),
        custom=ssl.mnch.FetalHealth(),
        networks=ss.PrenatalNet(),
        **sim_kw,
    )
    sim_baseline.run()

    # Basic sanity checks
    res_fh = sim_baseline.results.fetal_health
    total_births = res_fh.n_births.sum()
    assert total_births > 0, 'No births recorded'
    print(f'Total births: {total_births}')

    bw_baseline = mean_bw(sim_baseline)
    print(f'Mean birth weight (baseline): {bw_baseline:.0f}g')
    assert 2500 < bw_baseline < 4000, f'Mean birth weight {bw_baseline:.0f}g outside expected range'

    ga = res_fh.mean_ga_at_birth
    mean_ga = np.nanmean(ga)
    print(f'Mean GA at birth: {mean_ga:.1f} weeks')
    assert 36 < mean_ga < 44, f'Mean GA {mean_ga:.1f} weeks outside expected range'
    none = res_fh.n_births.values == 0
    assert np.isnan(res_fh.mean_birth_weight.values[none]).all() and not np.isnan(res_fh.mean_birth_weight.values[~none]).any() # Undefined on a step with no births
    assert not np.isnan(res_fh.mean_birth_weight.resample('year').values).any() # And left out of a mean over time

    # -- 2. Disease only (no treatment) --
    sim_disease = ss.Sim(
        demographics=demog(),
        diseases=ssl.AgentSIR(name='sir', **sir_kw),
        connectors=ssl.mnch.fetal_infection(),
        custom=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_disease.run()

    bw_disease = mean_bw(sim_disease)
    print(f'Mean birth weight (disease): {bw_disease:.0f}g')
    assert bw_disease < bw_baseline, f'Disease ({bw_disease:.0f}g) should be lower than baseline ({bw_baseline:.0f}g)'

    ptb_disease = sim_disease.results.pregnancy.n_preterm.sum()
    ptb_baseline = sim_baseline.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — baseline: {ptb_baseline}, disease: {ptb_disease}')
    assert ptb_disease > 3*ptb_baseline, f'Disease scenario should have several times the preterm births ({ptb_disease} against {ptb_baseline})' # About 5×; about 1.7× if the infection's shift of delivery is lost

    # -- 3. Disease + treatment --
    sim_treated = ss.Sim(
        demographics=demog(),
        diseases=ssl.AgentSIR(name='sir', **sir_kw),
        connectors=ssl.mnch.fetal_infection(),
        interventions=ssl.mnch.treat_pregnant(disease='sir'),
        custom=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_treated.run()

    bw_treated = mean_bw(sim_treated)
    print(f'Mean birth weight (treated): {bw_treated:.0f}g')
    rtol = 0.05 # Generous tolerance for stochastic comparison
    assert bw_treated > bw_disease * (1 - rtol), f'Treated ({bw_treated:.0f}g) should be at least as high as disease-only ({bw_disease:.0f}g)'

    ptb_treated = sim_treated.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — disease: {ptb_disease}, treated: {ptb_treated}')

    print(f'Summary — baseline: {bw_baseline:.0f}g, disease: {bw_disease:.0f}g, treated: {bw_treated:.0f}g')

    if do_plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

        for sim, label in [(sim_baseline, 'Baseline'), (sim_disease, 'Disease'), (sim_treated, 'Treated')]:
            fh = sim.custom['fetal_health']
            born = fh.birth_weight > 0
            ax0.hist(np.asarray(fh.birth_weight[born]), bins=30, alpha=0.4, label=label, edgecolor='k')
        ax0.axvline(2500, color='r', ls='--', label='LBW threshold')
        ax0.set_xlabel('Birth weight (g)')
        ax0.set_ylabel('Count')
        ax0.set_title('Birth weight: baseline vs disease vs treated')
        ax0.legend()

        tvec = sim_baseline.timevec
        ax1.plot(tvec, sim_baseline.results.pregnancy.preterm_rate, label='Baseline')
        ax1.plot(tvec, sim_disease.results.pregnancy.preterm_rate, label='Disease')
        ax1.plot(tvec, sim_treated.results.pregnancy.preterm_rate, label='Treated')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Preterm rate')
        ax1.set_title('Preterm birth rate')
        ax1.legend()

        fig.tight_layout()

    return sim_baseline, sim_disease, sim_treated


def family(members, name='hpv', init_prev=None, **kwargs):
    """ A disease whose strains are diseases of their own (ss.Strains of diseases), each with its own course """
    return ss.Disease(name=name, init_prev=init_prev, strains=ss.Strains(members, **kwargs))


def make_genotype(name, beta=3.0, **kwargs):
    """ One genotype of a toy HPV: infection, clearance with immunity that wanes, and cancer (for strains with courses of their own) """
    S, I, R, Ca = ss.states('S I R Ca')
    return ss.Disease(
        infection   = S >> I @ ss.peryear(beta)*I,
        clearance   = I >> R @ ss.years(1),
        reinfection = R >> I @ ss.peryear(beta)*I,
        waning      = R >> S @ ss.years(5),
        cancer      = I >> Ca @ ss.years(20),
        death       = Ca >> ss.null @ ss.years(5),
        X           = ss.State(people=True), # Where someone with cancer from another genotype goes
        infected    = I + Ca,
        name        = name,
        **kwargs,
    )


@sc.timer()
def test_strains_results():
    sc.heading('Testing a family of diseases: union results and conditions, and one cancer per person')
    names = ['hpv16', 'hpv18', 'hpv31']
    hpv = family([make_genotype('hpv16'), make_genotype('hpv18', 2.5), make_genotype('hpv31', 2.0)],
        immunity  = ss.immunity(due_to='R', rel_sus=[[0.9, 0.3, 0.2], [0.3, 0.9, 0.2], [0.1, 0.1, 0.5]], combine='sum'),
        init_prev = 0.1,
        shared    = 'Ca',
    )
    counts = []
    def count(sim): # The union and the co-infected, by hand, at every time point
        held = np.array([sim.diseases[name].I.raw | sim.diseases[name].Ca.raw for name in names])
        cancers = np.array([sim.diseases[name].Ca.raw for name in names])
        counts.append((held.any(0).sum(), (held.sum(0) >= 2).sum(), (cancers.sum(0) >= 2).sum()))
    sim = ss.Sim(diseases=hpv, analyzers=count, n_agents=5000, start=2000, stop=2030, dt=0.25, verbose=0).run()
    res = sim.results
    union, coinfected, two_cancers = np.array(counts).T
    assert sim.solver == 'abm' and list(sim.diseases.keys()) == names + ['hpv']
    assert np.array_equal(res.hpv.infected, union) and coinfected.max() > 0 # A union, not a sum: the co-infected are counted once
    assert np.array_equal(res.hpv.cancer, sum(res[name].cancer for name in names)) # A flow is a sum
    assert np.array_equal(res.hpv.infected.by('strain')[:, 1], res.hpv18.infected) # Each member's own
    assert two_cancers.max() == 0 and res.hpv.X[-1] > 0 # One cancer per person: the others move to X
    assert 'Strains of hpv: 3 with courses of their own' in sim.describe(output=True)

    # A union is a condition, with a count of members
    sim = ss.Sim(diseases=hpv, n_agents=500, start=2000, stop=2001, verbose=0).init()
    for name in names[:2]:
        d = sim.diseases[name]
        ss.move(d, d.I, uids=ss.uids([0, 1, 2]))
    selector = sim.diseases.hpv31.engine.selector
    assert np.array_equal(selector.select(sim.diseases.hpv.infected), np.array([sim.diseases[name].I.raw | sim.diseases[name].Ca.raw for name in names]).any(0))
    assert np.flatnonzero(selector.select(sim.diseases.hpv.infected.count() >= 2)).tolist() == [0, 1, 2]
    return sim


@sc.timer()
def test_strains_immunity():
    sc.heading('Testing protection between the members of a family: the combination rules, a vaccine on top, and shorter stays')
    def make(name, beta=1.0):
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=S >> I @ ss.peryear(beta)*I, clearance=I >> R @ ss.years(1), reinfection=R >> I @ ss.peryear(beta)*I, name=name)
    hpv16, hpv18 = make('hpv16'), make('hpv18')
    matrix = [[0.2, 0.5], [0.1, 0.4]] # Rows from, columns against
    def rel_sus(combine):
        """ rel_sus of hpv16 for someone who has had both, someone who has had hpv16 only, and someone who has had neither; all with a vaccine of efficacy 0.5 """
        hpv = family([hpv16, hpv18], immunity=ss.immunity(due_to='R', rel_sus=matrix, combine=combine))
        vaccine = ss.modify(hpv16.rel_sus, multiply=0.5, where=ss.ppl.age >= 0)
        sim = ss.Sim(diseases=hpv, connectors=vaccine, n_agents=100, verbose=0).init()
        d16, d18 = sim.diseases.hpv16, sim.diseases.hpv18
        ss.move(d16, d16.R, uids=ss.uids([0, 1]))
        ss.move(d18, d18.R, uids=ss.uids([0]))
        sim.diseases.hpv.start_step()
        return sim.diseases.hpv16.engine.mult_raw('rel_sus')[:3]
    assert np.allclose(rel_sus('sum'), [(1 - (0.2 + 0.1))*0.5, 0.4, 0.5]) # (1 − min(1, Σ))·(1 − vaccine) = 0.35, as HPVsim
    assert np.allclose(rel_sus('sum')[0], 0.35)
    assert np.allclose(rel_sus('max'), [0.4, 0.4, 0.5])
    assert np.allclose(rel_sus('multiply'), [(1 - (1 - 0.8*0.9))*0.5, 0.4, 0.5])
    assert np.allclose(rel_sus(lambda p: np.minimum(1, p.max(0) + 0.5*(p.sum(0) - p.max(0)))), [0.375, 0.4, 0.5]) # The strongest plus half the rest

    # Time in a state: protection 0.5 halves the stay, so after a year exp(-2) are left rather than exp(-1)
    left = []
    for dur in [0.0, 0.5]:
        a, b = make('a', 0.0), make('b', 0.0)
        hpv = family([a, b], immunity=ss.immunity(due_to='R', dur=dict(I=[[0, 0], [dur, 0]])))
        sim = ss.Sim(diseases=hpv, n_agents=20_000, start=2000, stop=2002, dt=1/52, verbose=0).init()
        ss.move(sim.diseases.b, sim.diseases.b.R) # Everyone
        ss.move(sim.diseases.a, sim.diseases.a.I)
        left.append(sim.run().results.a.I[52]/20_000)
    assert np.allclose(left, [np.exp(-1), np.exp(-2)], atol=0.01)
    return sim


def make_nab_genotype(name, beta=1.0, init_prev=0.1, nab_init=0):
    """ One genotype whose protection is an antibody level drawn at clearance (an attribute), as HPVsim's """
    S, I = ss.states('S I')
    clearance = I >> S @ ss.years(1)
    nab = ss.attr(ss.uniform(0.5, 0.95), on=clearance, combine='max', init=nab_init)
    return ss.Disease(S, I, infection=S >> I @ ss.peryear(beta)*I, clearance=clearance, nab=nab, init_prev=init_prev, name=name)


@sc.timer()
def test_strains_immunity_attr():
    sc.heading('Testing protection from an attribute of each member (antibodies): m[k, g]·nab_k, combined over k')
    m = np.array([[0.8, 0.5], [0.3, 0.9]]) # Rows from, columns against

    # Antibodies fixed at known values: rel_sus and rel_trans of each genotype, by the rule
    def mults(combine, nab_init=0):
        a, b = make_nab_genotype('a', init_prev=0, nab_init=nab_init), make_nab_genotype('b', init_prev=0, nab_init=nab_init)
        hpv = family([a, b], immunity=ss.immunity(due_to='nab', rel_sus=m, rel_trans=m, combine=combine))
        sim = ss.Sim(diseases=hpv, n_agents=100, verbose=0).init()
        a, b = sim.diseases.a, sim.diseases.b
        a.nab.raw[:3] = [0.6, 0.6, 0.0] if nab_init == 0 else [0.6, 0.6, np.nan] # The third has none
        b.nab.raw[:3] = [0.5, 0.0, 0.0] if nab_init == 0 else [0.5, np.nan, np.nan]
        sim.diseases.hpv.start_step()
        return [[mod.engine.mult_raw(key)[:3] for key in ['rel_sus', 'rel_trans']] for mod in [a, b]]
    nab = np.array([[0.6, 0.6, 0.0], [0.5, 0.0, 0.0]]) # By genotype (rows) and person
    for combine in ['max', 'sum', 'multiply']:
        expected = [1 - ss.strains.combine_protection(m[:, g][:, None]*nab, combine) for g in range(2)]
        for got in [mults(combine), mults(combine, nab_init=None)]: # A NaN (no antibodies yet) gives none
            for g in range(2):
                assert np.allclose(got[g][0], expected[g]) and np.allclose(got[g][1], expected[g]), f'{combine}, genotype {g}: {got[g]} vs {expected[g]}'
    assert np.allclose(mults('sum')[0][0], [1 - (0.8*0.6 + 0.3*0.5), 1 - 0.8*0.6, 1])

    # Time in a state: antibodies of 0.625 to b with a protection of 0.8 halve the stay in a's I, so after a year exp(-2) are left
    a, b = make_nab_genotype('a', beta=0, init_prev=0), make_nab_genotype('b', beta=0, init_prev=0)
    hpv = family([a, b], immunity=ss.immunity(due_to='nab', dur=dict(I=[[0, 0], [0.8, 0]])))
    sim = ss.Sim(diseases=hpv, n_agents=20_000, start=2000, stop=2002, dt=1/52, verbose=0).init()
    sim.diseases.b.nab.raw[:] = 0.625
    ss.move(sim.diseases.a, sim.diseases.a.I) # Everyone
    assert np.isclose(sim.run().results.a.I[52]/20_000, np.exp(-2), atol=0.01)

    # The same as a modifier on each genotype, as HPVsim writes it outside a family (the family reads every member's antibodies once a step, and a modifier when its genotype's rel_sus is worked out, so they differ within noise)
    pars = dict(n_agents=2000, start=2000, stop=2010, dt=0.25, verbose=0)
    def run(how, seed=1):
        gens = [make_nab_genotype('hpv16', 2.0), make_nab_genotype('hpv18', 1.6)]
        if how == 'family':
            return ss.Sim(diseases=family(gens, immunity=ss.immunity(due_to='nab', rel_sus=m, combine='sum')), rand_seed=seed, **pars).run()
        mods = [ss.modify(h.rel_sus, multiply=1 - np.clip(sum(m[k, g]*gens[k].nab for k in range(2)), 0, 1), name=f'nab_{h.name}') for g,h in enumerate(gens)] if how == 'modify' else []
        return ss.Sim(diseases=gens, connectors=mods, rand_seed=seed, **pars).run()
    out = sc.objdict()
    for how in ['family', 'modify', 'none']:
        sim = run(how)
        out[how] = np.array([sim.results[name].infection.sum() for name in ['hpv16', 'hpv18']])
    assert np.allclose(out.family, out.modify, rtol=0.15), out # One run each at 2,000 agents: a run's SD is about 8% (8 seeds: hpv18 1,249 ± 34 vs 1,178 ± 30)
    assert np.all(out.family < 0.8*out.none), out # It protects

    # As compartments: attributes have none yet, so auto runs agents, and a compartmental solver refuses
    make_family = lambda: family([make_nab_genotype('a'), make_nab_genotype('b')], immunity=ss.immunity(due_to='nab', rel_sus=m, combine='sum'))
    assert ss.Sim(diseases=make_family(), approx=True, stop=2001, verbose=0).init().solver == 'abm'
    for solver in ['ode', 'binomial']:
        with pytest.raises(ss.CapabilityError):
            ss.Sim(diseases=make_family(), solver=solver, approx=True, stop=2001, verbose=0).init()

    # A name that is a state in one member and an attribute in another
    S, I = ss.states('S I')
    other = ss.Disease(S, I, infection=S >> I @ ss.peryear(1)*I, clearance=I >> S @ ss.years(1), nab=ss.State(people=True), name='b')
    with pytest.raises(ValueError, match='attribute'):
        ss.Sim(diseases=family([make_nab_genotype('a'), other], immunity=ss.immunity(due_to='nab', rel_sus=0.5)), n_agents=100, stop=2001, verbose=0).init()
    return out


@sc.timer()
def test_strains_dengue():
    sc.heading('Testing a family that is not HPV-shaped: two dengue serotypes with cross-protection, tested once per person')
    def serotype(name):
        S, I, R = ss.states('S I R')
        return ss.Disease(S, I, R, infection=S >> I @ ss.perday(0.3)*I, recovery=I >> R @ ss.days(7), name=name)
    def dengue(cross=None):
        imm = ss.immunity(due_to='R', rel_sus=[[1, cross], [cross, 1]]) if cross is not None else None
        return family([serotype('den1'), serotype('den2')], name='dengue', init_prev=0.01, immunity=imm)
    pars = dict(n_agents=5000, unit='day', stop=200, verbose=0)
    total = lambda cross: ss.Sim(diseases=dengue(cross), **pars).run().results.dengue.infection.sum()
    assert total(0.9) < 0.8*total(None) # Recovering from one serotype protects against the other
    test = ss.Diagnostic(results=dict(positive_ns1={'I': 1.0}, unclear={'S': 0.1}, negative_ns1=None), name='ns1')
    scr = ss.screen(test, at=[1], coverage=1.0, disease='dengue', name='survey')
    res = ss.Sim(diseases=dengue(), interventions=scr, **pars).run().results.survey
    assert np.isclose(res.unclear.sum()/res.delivered.sum(), 0.1, atol=0.02) # Susceptible to both, but tested once (not 1 − 0.9² = 0.19)
    return res


@sc.timer()
def test_strains_compartmental():
    sc.heading('Testing a family under the compartmental solvers: independent members side by side; members that affect each other need agents')
    def make(name, beta, **kwargs):
        S, I, R, Ca = ss.states('S I R Ca')
        return ss.Disease(infection=S >> I @ ss.peryear(beta)*I, clearance=I >> R @ ss.years(1), reinfection=R >> I @ ss.peryear(beta)*I, waning=R >> S @ ss.years(2),
                          cancer=I >> Ca @ ss.years(50), X=ss.State(people=True), infected=I + Ca, name=name, **kwargs)
    pars = dict(start=2000, stop=2020, dt=1/12, verbose=0)

    # With nothing between them, the members run as they would alone, and the union is from their shares
    alone = ss.Sim(diseases=[make('a', 3, init_prev=0.05), make('b', 2, init_prev=0.05)], solver='ode', total_pop=10_000, **pars).run()
    sim = ss.Sim(diseases=family([make('a', 3, init_prev=0.05), make('b', 2, init_prev=0.05)]), total_pop=10_000, **pars).run()
    res = sim.results
    assert sim.solver == 'ode' and np.array_equal(res.a.I, alone.results.a.I) and np.array_equal(res.b.I, alone.results.b.I) # Bit for bit
    assert np.allclose(res.hpv.I, 10_000*(1 - (1 - res.a.I/10_000)*(1 - res.b.I/10_000)))

    # Strains that affect each other (immunity=, shared states) need agents: auto runs them as agents, and a compartmental solver refuses, whatever approx=
    families = [lambda: family([make('a', 3), make('b', 2.5)], immunity=ss.immunity(due_to='R', rel_sus=0.3), init_prev=0.1),
                lambda: family([make('a', 3), make('b', 2.5)], init_prev=0.1, shared='Ca')]
    for make_family in families:
        sim = ss.Sim(diseases=make_family(), total_pop=10_000, approx=True, **pars).init()
        assert sim.solver == 'abm' and 'compartmental solvers: none' in sim.describe(output=True)
        for solver in ['ode', 'binomial']:
            with pytest.raises(ss.CapabilityError, match='only agents follow'):
                ss.Sim(diseases=make_family(), solver=solver, total_pop=10_000, approx=True, **pars).init()
    return res


@sc.timer()
def test_strains_branch_flows():
    sc.heading('Testing the flow to one destination of a split, in a family and as an outcome, as agents and compartments')
    def make(name, beta):
        S, P, CIN, CA = ss.states('S P CIN CA')
        cin = ss.Transition(CIN, [CA, S], p=[0.3, 0.7], dur=ss.years(2)) # A split, as HPVsim's CIN that clears or becomes cancer
        return ss.Disease(infection=S >> P @ ss.peryear(beta)*P, progression=P >> CIN @ ss.years(1), cin=cin,
                          cancers=ss.outcome(cin.to(CA)), init_prev=0.1, name=name)
    pars = dict(start=2000, stop=2015, dt=0.25, total_pop=10_000, verbose=0)
    out = sc.objdict()
    for solver in ['abm', 'ode']:
        sim = ss.Sim(diseases=family([make('a', 3.0), make('b', 2.5)]), solver=solver, n_agents=10_000, rand_seed=1, **pars).run()
        res = sim.results
        assert res.a.cin.CA.sum() > 0 and np.allclose(res.a.cancers, res.a.cin.CA) and np.allclose(res.b.cancers, res.b.cin.CA) # The outcome is the flow to CA
        assert np.allclose(res.hpv.cin.CA, res.a.cin.CA + res.b.cin.CA) and np.allclose(res.hpv.cin.S, res.a.cin.S + res.b.cin.S) # Summed across members
        assert np.allclose(res.hpv.cin.CA + res.hpv.cin.S, res.hpv.cin) and np.allclose(res.hpv.cancers, res.hpv.cin.CA)
        out[solver] = [res.hpv.cin.CA.sum(), res.hpv.cin.CA.sum()/res.hpv.cin.sum()]
    assert np.isclose(out.abm[0], out.ode[0], rtol=0.1) and np.allclose([out.abm[1], out.ode[1]], 0.3, rtol=0.05), out
    return out


def test_condition_by_sex():
    """ A condition acquired at a rate by sex whose values are parameters, with its initial prevalence from a function of the agents (no transmission), given in pars= """
    class Condition(ss.Disease):
        p_female = ss.par(ss.probperyear(0.05))
        p_male = ss.par(ss.probperyear(0.02))
        susceptible, affected = ss.states('susceptible affected')
        onset = ss.Transition(susceptible, affected, rate=ss.by(sex=dict(female=p_female, male=p_male)))

    cond = Condition(pars=dict(init_prev=ss.bernoulli(p=lambda sim, uids: np.where(sim.people.female[uids], 0.3, 0.1))), p_male=ss.par(ss.probperyear(0.01), bounds=[0.001, 0.1]))
    assert {'p_female', 'p_male'} <= set(cond.pars.keys()) and cond.pars.p_male.bounds == [0.001, 0.1] # The parameters inside ss.by are the disease's, and one given replaces the class's
    sim = ss.Sim(diseases=cond, n_agents=10_000, start=2000, stop=2010, dt=1, verbose=0).run()
    first = sim.results.condition.affected[0]/10_000
    assert np.isclose(first, 0.2, atol=0.02), first # Seeded by sex through the acquisition
    assert sim.results.condition.onset.sum() > 0
    return sim



@sc.timer()
def test_coded_seeds():
    sc.heading('Testing that the initial infections of a disease written as code are dated at the start, as a declared disease\'s are, so they are infectious for as long as later cases')
    sim = ss.Sim(diseases=ssl.AgentSIR(dur_inf=ss.constant(ss.years(2)), p_death=0), networks='random', n_agents=1000, start=2000, stop=2005, verbose=0).init()
    sir = sim.diseases.agentsir
    seeds = sir.infected.uids
    assert len(seeds) and np.all(sir.ti_infected[seeds] == 0) and np.all(sir.ti_recovered[seeds] == 3) # Infectious in steps 1 and 2, as a case infected in step 1 is in steps 2 and 3
    sim.run()
    assert np.all(sir.recovered[seeds]) and np.all(sim.results.agentsir.infected[1:3] >= len(seeds))
    return sim


@sc.timer()
def test_coded_par_modifiers():
    sc.heading('Testing modifiers of the parameters of a disease written as code')

    class Coded(ss.Infection):
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(beta=0.1, init_prev=0.1, level=0.4, rate=ss.peryear(0.2), risk=ss.probperyear(0.3), spread=ss.normal(1, 1))
            self.update_pars(**kwargs)
            self.seen = []
            return
        def step_state(self):
            self.seen.append([self.pars.level, self.pars.rate.to('year').value, self.pars.risk.value])
            return

    class PerPerson(Coded):
        allow_modifiers = ('rate',)
        def step_state(self):
            self.rates = self.current_value('rate', self.sim.people.auids)
            return

    pars = dict(n_agents=200, start=2000, stop=2004, dt=ss.years(1), networks='random', verbose=0)
    make = lambda **kwargs: ss.modify(kwargs.pop('target', 'coded.pars.rate'), **kwargs)

    # For everyone, code reading self.pars sees the modified value while it applies, and the base value is back after the run
    mods = [make(target='coded.pars.level', multiply=0.5, start=2002), make(multiply=2), make(target='coded.pars.risk', set=ss.probperyear(0.6))]
    dis = ss.Sim(diseases=Coded(), connectors=mods, **pars).run().diseases.coded
    level, rate, risk = np.array(dis.seen).T
    assert np.allclose(level, [0.4, 0.4, 0.2, 0.2])
    assert np.allclose(rate, 0.4)
    assert np.allclose(risk, 0.6)
    assert dis.pars.level == 0.4 and dis.pars.rate == ss.peryear(0.2) and dis.pars.risk == ss.probperyear(0.3)
    assert np.isclose(dis.current_value('level'), 0.2)

    # For some people, code reads a value for each of them, and says so; otherwise it is refused
    where = lambda sim: sim.people.female
    dis = ss.Sim(diseases=PerPerson(), connectors=make(target='perperson.pars.rate', set=ss.peryear(1.0), where=where), **pars).run().diseases.perperson
    ppl = dis.sim.people
    rates = dis.rates.to('year').value
    assert np.allclose(rates, np.where(ppl.female[ppl.auids], 1.0, 0.2))
    with pytest.raises(ValueError, match='allow_modifiers'):
        ss.Sim(diseases=Coded(), connectors=make(multiply=0.5, where=where), **pars).init()
    with pytest.raises(ss.CapabilityError, match='allow_modifiers'):
        ss.Sim(diseases=Coded(), connectors=make(multiply=lambda sim: np.full(len(sim.people), 0.5)), **pars).run()
    with pytest.raises(ValueError, match='normal'):
        ss.Sim(diseases=Coded(), connectors=make(target='coded.pars.spread', multiply=0.5), **pars).init()

    # The same call on a declared disease, as agents and as compartments
    for solver in ['abm', 'ode']:
        sir = ss.SIR()
        kw = dict(networks='random') if solver == 'abm' else dict()
        sim = ss.Sim(diseases=sir, connectors=ss.modify(sir.pars.beta, multiply=0.5), solver=solver, n_agents=200, start=2000, stop=2004, dt=ss.years(1), verbose=0, **kw).run()
        beta = sim.diseases.sir
        assert np.isclose(beta.current_value('beta').to('year').value, beta.pars.beta.value.to('year').value*0.5)
    return dis


@sc.timer()
def test_coded_modifiers():
    sc.heading('Testing modifiers of rel_sus on a disease written as code that reads them (MG1), a property a module adds to the people (MG3), and the cause of death by hazard (MG8)')

    class Condition(ss.Disease):
        allow_modifiers = ('rel_sus',)
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(p_onset=ss.probperyear(0.2))
            self.define_states(ss.BoolState('affected', label='Affected'), ss.FloatArr('rel_sus', default=1.0))
            self.update_pars(**kwargs)
        def init_pre(self, sim):
            super().init_pre(sim)
            sim.people.add_state(ss.BoolArr('housing', default=ss.bernoulli(0.5))) # Idempotent: each module that reads it adds it
            sim.people.add_state(ss.BoolArr('housing'))
            return
        def step(self):
            uids = (~self.affected).uids
            p = -np.expm1(-self.pars.p_onset.rate*self.t.dt_year*self.current_value('rel_sus', uids))
            self.affected[uids[self.sim.people.housing[uids] | (np.random.default_rng(self.ti).random(len(uids)) < p)]] = True
            return

    cond = Condition(name='cond')
    sim = ss.Sim(diseases=cond, interventions=ss.modify(cond.rel_sus, multiply=0.0, where=ss.ppl.female), n_agents=2000, start=2000, stop=2005, verbose=0).run()
    ppl, d = sim.people, sim.diseases.cond
    assert np.all(d.affected[ppl.female.uids] == ppl.housing[ppl.female.uids]) # Only those housed: the onset of women is modified away
    assert (d.affected & ppl.male & ~ppl.housing).sum() > 100
    assert np.isclose(ppl.housing.mean(), 0.5, atol=0.05)
    Condition.allow_modifiers = ()
    with pytest.raises(ValueError, match='allow_modifiers'):
        cond = Condition(name='cond')
        ss.Sim(diseases=cond, interventions=ss.modify(cond.rel_sus, multiply=0.0), n_agents=100, start=2000, stop=2001, verbose=0).init()

    # A property nobody added has the default given; added after the sim is made, it has values at once
    S, D = ss.states('S D')
    dis = ss.Disease(S, D, onset=S >> D @ ss.peryear(0.2)*(1 + 4*ss.ppl.get('housing', 0)), name='dep')
    sim = ss.Sim(diseases=dis, n_agents=1000, start=2000, stop=2001, verbose=0).run()
    assert 'housing' not in sim.people and sim.results.dep.D[-1] > 0
    assert len(sim.people.add_state(ss.FloatArr('adherence', default=ss.uniform())).values) == len(sim.people)

    # Two causes of death in the same step, weighted by their hazards; and a declared transition's hazard, with its modifiers
    class Killer(ss.Module):
        def __init__(self, hazard, **kwargs):
            super().__init__(**kwargs)
            self.hazard = hazard
        def step(self):
            if self.ti == 2:
                self.sim.people.request_death(self.sim.people.auids, cause=self, hazard=self.hazard)
    sim = ss.Sim(connectors=[Killer(9, name='a'), Killer(1, name='b')], n_agents=5000, start=2000, stop=2003, verbose=0).run()
    codes = sim.people.cause_of_death.raw
    share = np.mean(codes[codes >= 0] == sim.people.causes.index('a'))
    assert np.isclose(share, 0.9, atol=0.02), share
    S, I = ss.states('S I')
    dis = ss.Disease(S, I, infection=S >> I @ ss.peryear(1), death=I >> ss.null @ ss.peryear(0.3), name='d')
    sim = ss.Sim(diseases=dis, interventions=ss.modify(dis.death, multiply=0.5, where=ss.ppl.female), n_agents=500, start=2000, stop=2001, verbose=0).run()
    uids = sim.people.auids
    assert np.allclose(sim.diseases.d.death.hazard(uids), np.where(sim.people.female[uids], 0.15, 0.3))
    return sim



if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    fetal = test_fetal_health(do_plot=do_plot)
    strains = test_strains_results()
    immune = test_strains_immunity()
    nabs   = test_strains_immunity_attr()
    compart = test_strains_compartmental()
    branch  = test_strains_branch_flows()
    cond    = test_condition_by_sex()
    seeds   = test_coded_seeds()
    par_mods = test_coded_par_modifiers()
    mods    = test_coded_modifiers()
