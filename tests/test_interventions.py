"""
Test interventions: products, delivery by year and at a rate, interventions and products written as code, and products over several diseases
"""
from unittest import mock
import sciris as sc
import numpy as np
import starsim as ss
import pytest
import corpus
from corpus import make_geno, make_hpv, check_agents

hpv_kw = dict(verbose=0, start=2000, stop=2010, dt=ss.months(3), total_pop=10_000) # For the models of HPV genotypes


def make_pars(n_agents=2_000, start=2000, stop=2020):
    """ Lightweight SIS sim params shared across delivery tests """
    return sc.obj(
        n_agents = n_agents,
        start    = start,
        stop     = stop,
        diseases = ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)), # In years, as the deliveries are
        networks = 'random',
        verbose  = 0,
    )


def make_dx():
    """ Minimal diagnostic, as Starsim 3's table of results by state """
    return ss.Diagnostic(sensitivity=0.95, specificity=0.99)


#%% Products and their delivery

def test_products():
    sc.heading('Testing products: a vaccine, a test with several results, treatment with a queue, and one product delivered by two deliveries')
    vaccine = ss.Vaccine(efficacy=0.9)
    vx1 = ss.vaccinate(vaccine, who=~vaccine.ever, when=ss.schedule(every=ss.years(1), start=2005), coverage=0.5)
    vx2 = ss.vaccinate(vaccine, who=~vaccine.ever, when=ss.schedule(every=ss.years(1), start=2005), coverage=0.3)
    cytology = ss.Diagnostic(outcomes=dict(high=dict(I=0.9), low={'I': 0.05, None: 0.02}, normal=None)) # Three results, most severe first (Starsim 3's ss.Dx table)
    screening = ss.screen(cytology, when=ss.schedule(every=ss.years(1), start=2010), coverage=0.9)
    sim = ss.Sim(make_pars(n_agents=5e3), interventions=[screening, vx1, vx2]).run()

    dxres, y = sim.results.screening, sim.t.yearvec
    assert dxres.delivered[y < 2010].sum() == 0 and dxres.delivered[y > 2010].sum() > 0
    assert dxres.positive.sum() == dxres.high.sum() + dxres.low.sum() > 0, 'Expected positives to be every result but the last'
    assert dxres.high.sum() + dxres.low.sum() + dxres.normal.sum() == dxres.delivered.sum()
    assert sim.results.sis.infection[y < 2005].mean() > sim.results.sis.infection[y > 2005].mean(), 'Expected vaccine to reduce infections'
    assert len(sim.products()) == 2
    assert sim.results.vx1.delivered.sum() + sim.results.vx2.delivered.sum() == sim.interventions.vx1.product.ti_delivered.notnan.sum() > 0, 'Expected each person to be vaccinated by one delivery, once'

    # Treatment of a fixed number each step (Starsim 3's treat_num): those not served wait, and leave the queue if no longer infected
    sis = ss.SIS(beta=ss.peryear(0.5), dur_inf=ss.years(10))
    treatment = ss.treat(ss.Treatment(cure=0.8), when=sis.infection, capacity=ss.capacity(50, queue=True))
    res = ss.Sim(make_pars() | dict(diseases=sis), interventions=treatment).run().results.treatment
    assert res.delivered.max() == 50 and res.waiting.max() > 0, 'Expected at most 50 treated each step, and a queue'

    return sim


def test_delivery_years():
    sc.heading('Testing delivery by year: a window, a coverage that changes by year, a campaign of one year and of several, years outside the sim, and a probability per year')
    years = [2005, 2010, 2015]
    intvs = [
        ss.screen(make_dx(), when=ss.schedule(every=ss.years(1), start=2008, stop=2013), coverage=1.0, name='window'), # The stop is not included, so Starsim 3's end_year=2012 is stop=2013
        ss.screen(make_dx(), when=ss.schedule(every=ss.years(1)), coverage=ss.timeseries(np.arange(2005, 2016), np.linspace(0.05, 0.95, 11), interpolation='linear'), name='scaleup'),
        ss.screen(make_dx(), when=[2010], coverage=0.5, name='once'),
        ss.screen(make_dx(), when=years, coverage=ss.timeseries(years, [0.3, 0.5, 0.7]), name='rounds'),
        ss.screen(make_dx(), when=[2030], coverage=0.5, name='late'), # After the sim ends: never runs, so a sim can be run to an earlier stop
        ss.screen(make_dx(), start=2030, rate=ss.peryear(0.5), name='late_start'),
        ss.screen(make_dx(), rate=ss.probperyear(0.5), name='always'), # No window: every step, with a probability per year
    ]
    sim = ss.Sim(make_pars(), dt=0.5, interventions=intvs).run()
    r, y = sim.results, sim.t.yearvec # A step's results are in the row of the year it starts
    assert r.window.delivered[(y < 2008) | (y >= 2013)].sum() == 0 and r.window.delivered[np.isin(y, range(2008, 2013))].min() > 0
    assert r.scaleup.delivered[(y >= 2013) & (y <= 2015)].mean() > 3*r.scaleup.delivered[(y >= 2005) & (y <= 2007)].mean()
    on = np.isclose(y, 2010)
    assert r.once.delivered[on].sum() > 0 and r.once.delivered[~on].sum() == 0
    on = np.isin(y, years)
    assert r.rounds.delivered[~on].sum() == 0 and np.all(np.diff(r.rounds.delivered[on]) > 0), 'Expected each round to reach its own coverage'
    assert r.late.delivered.sum() == r.late_start.delivered.sum() == 0
    assert r.always.delivered.min() > 0 and np.isclose(r.always.delivered[0]/2000, 1 - 0.5**0.5, atol=0.05), 'Expected the probability per year to be per step'
    with pytest.raises(ValueError, match='outside the sim'): # Before the sim starts is an error
        ss.Sim(make_pars(), interventions=ss.screen(make_dx(), when=[1950], coverage=0.5)).init()
    return sim


def test_rate():
    sc.heading('rate= reaches those eligible at a rate, as agents and as compartments, and by group')
    def make(solver, **kw):
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ ss.perday(0.1)*I, recovery=I >> R @ ss.days(10), init_prev=0.01, name='sir')
        return ss.Sim(sir, interventions=ss.vaccinate(efficacy=1.0, name='vx', **kw), n_agents=2000, total_pop=2000, unit='day', stop=50, solver=solver, verbose=0)
    for solver in ['ode', 'abm']:
        frac = make(solver, rate=ss.peryear(2)).run().results.vx.delivered.sum()/2000
        assert np.isclose(frac, 1 - np.exp(-2*50/365), rtol=0.15), (solver, frac) # Nearly everyone stays eligible: few are infected
    with pytest.raises(TypeError): # A rate needs a unit, however it is given
        make('abm', rate=lambda sim: 0.5).run()

    # A rate by group, one for each level
    S, V = ss.states('S V')
    sim = ss.Sim(ss.Disease(S, V, name='d'), interventions=ss.Event(S >> V, rate=ss.by(risk=[ss.peryear(0.1), ss.peryear(2)]), name='ev'), start=2000, stop=2001, n_agents=2000, verbose=0)
    sim.stratify(risk=ss.levels({'lo':0.5, 'hi':0.5}))
    reached = sim.run().results.d.V.strata[-1]/1000
    assert np.allclose(reached, 1 - np.exp(-np.array([0.1, 2])), atol=0.06), reached
    return


def test_modify_fast_paths():
    sc.heading('Dated modifiers, and modifiers on a comparison of counts, stay in the kernels, with the same results')
    kw = dict(total_pop=1e6, start='2026-01-01', stop='2026-12-31', dt=ss.days(1), solver='ode', verbose=0)
    def make_sir():
        beta = ss.par(0.3, name='beta')
        S, I, R = ss.states('S I R')
        return ss.Disease(infection=S >> I @ beta*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir')
    def run(**mod):
        sir = make_sir()
        return ss.Sim(sir, interventions=ss.modify(sir.pars.beta, **{'multiply':0.75, **mod}) if mod else None, **kw).run()

    # Dates, a step function, and a condition on time alone are worked out where they change, with the results of working them out at every step
    mods = [
        lambda: dict(when=ss.schedule(start='2026-03-01', stop='2026-05-02')),
        lambda: dict(multiply=ss.timeseries({20: 0.7, 40: 0.4, 90: 1.0}), when=ss.schedule(start='2026-01-15')),
        lambda: dict(when=lambda now: now.day in [1, 15]),
    ]
    for mod in mods:
        fast = run(**mod())
        with mock.patch.object(ss.solvers.Compartmental, 'changing_steps', lambda *args: None):
            slow = run(**mod())
        assert fast._compiled_reason() is None and np.array_equal(fast.results.sir.I.values, slow.results.sir.I.values)
    assert not np.array_equal(fast.results.sir.I.values, run().results.sir.I.values) # They had an effect

    # A comparison of counts is part of the rate: the same as writing it there; with ss.options.require_compiled, what needs memory raises an error instead of falling back to Python
    sir = make_sir()
    with ss.options.context(require_compiled=True):
        sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, when=sir.I.n >= 20_000), **kw).run()
        with pytest.raises(ss.CapabilityError):
            ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.75, when=sir.I.n >= 20_000, until=sir.I.n <= 10_000), **kw).run()
    beta2 = ss.par(0.3, name='beta')
    S, I, R = ss.states('S I R')
    written = ss.Sim(ss.Disease(infection=S >> I @ (beta2*(1 - 0.25*(I.n >= 20_000)))*I, recovery=I >> R @ ss.days(10), init_n=dict(I=100), name='sir'), **kw).run()
    assert sim.interventions()[0].inlined and np.allclose(sim.results.sir.I.values, written.results.sir.I.values, rtol=1e-12)
    return sim


#%% Interventions and products written as code

def test_coded_interventions():
    sc.heading('Testing interventions written as code: their states and flows, check_eligibility() from who=, and ss.modify() of their parameters')
    class Program(ss.Intervention):
        """ Its states are counted at the start of each step, and its own flows at the end, so the first row has the first step's """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.BoolState('never_reached', default=True))
            self.define_aliases(reached=lambda mod: ~mod.never_reached)

        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('new_reached', dtype=int))

        def step(self):
            self.new_reached = 1

        def record_events(self):
            super().record_events()
            self.results.new_reached[self.ti] = self.new_reached

    class Count(ss.Intervention):
        """ Who it can reach, read from who= """
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_eligible', dtype=int))

        def step(self):
            self.results.n_eligible[self.ti] = len(self.check_eligibility())

    class Tx(ss.Intervention):
        """ A parameter that ss.modify() can change: one value in self.pars, or, with who=, each person's with current_value() """
        allow_modifiers = ('p_treat',)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(p_treat=0.5)
            self.seen = []

        def step(self):
            self.seen.append(self.pars.p_treat)
            self.by_person = self.current_value('p_treat', self.sim.people.auids)

    sis = ss.SIS()
    whos = dict(everyone=None, cond=ss.person.age < 15, fn=lambda sim: sim.people.age < 15, uids=lambda self, sim: (sim.people.age < 15).uids, path='sis.I', aged=ss.at_age(5))
    counts = [Count(who=who, name=k) for k,who in whos.items()]
    mods = [ss.modify('tx.pars.p_treat', multiply=0.2, start=2005), ss.modify('tx.pars.p_treat', multiply=0.5, who=sis.I, name='inf')]
    sim = ss.Sim(diseases=sis, networks='random', interventions=[Program(), Tx(name='tx'), *counts, *mods], start=2000, stop=2010, dt=ss.years(1), n_agents=500, verbose=0).run()
    res = sim.results.program
    assert np.all(res.never_reached == 500) and np.all(res.reached == 0) and np.all(res.new_reached == 1) # The first step's at the first time point
    n = {k: sim.results[k].n_eligible[0] for k in whos}
    assert n['everyone'] == 500 and n['cond'] == n['fn'] == n['uids'] > 0, n
    assert 0 < n['path'] == sim.results.sis.I[0] < 500, n
    assert 0 < n['aged'] < 50 # Those who reached the age on the step (a year)
    tx = sim.interventions.tx
    assert tx.seen[0] == 0.5 and np.isclose(tx.seen[-1], 0.5), 'p_treat in self.pars is the base value, since the class reads it with current_value()'
    vals = np.asarray(tx.by_person)
    inf = sim.diseases.sis.I[sim.people.auids]
    assert np.allclose(vals[inf], 0.05) and np.allclose(vals[~inf], 0.1), f'Expected 0.05 for the infected and 0.1 otherwise, not {np.unique(vals)}'
    return sim


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
            self.define_results(ss.Result('swabbed', dtype=float, kind='flow'), ss.Result('positive', dtype=float, kind='flow'))

        def step(self):
            screening = self.sim.interventions.screening
            self.results.swabbed[self.ti] = len(screening.reached())
            self.results.positive[self.ti] = len(screening.reached('positive'))

    screening = ss.Delivery(Swab(), when=ss.schedule(every=ss.months(3)), coverage=0.2, name='screening')
    treatment = ss.treat(ss.Treatment(cure=1.0), when=screening.positive, name='treatment')
    sim = ss.Sim(make_geno('flu'), interventions=[screening, treatment], analyzers=Reached(), n_agents=10_000, **hpv_kw).run()
    r = sim.results
    assert isinstance(sim.solver, ss.solvers.abm) # Its own code acts on agents
    assert r.screening.delivered.sum() > 0 and r.screening.positive.sum() > 0
    assert np.array_equal(r.reached.swabbed, r.screening.delivered) # An analyzer reads the step just taken, at the same index
    assert np.array_equal(r.reached.positive, r.screening.positive)
    assert np.array_equal(r.treatment.cured, r.screening.positive) # Every one found is treated, and cured
    return sim


#%% Products over several diseases

def test_vaccine_diseases():
    sc.heading('Testing a vaccine over several diseases: efficacy keyed by disease, one has_ per person, a modifier per disease and quantity')
    def make(eff, **kwargs):
        g16, g18 = make_geno('hpv16'), make_geno('hpv18')
        vx = ss.Vaccine(efficacy=eff(g16, g18), name='vx')
        prog = ss.vaccinate(vx, when=[2001], coverage=0.8)
        cross = ss.modify(g18.clearance, multiply=2, who=vx.held, name='cross') # vx.held is a condition
        return ss.Sim([g16, g18], interventions=prog, connectors=cross, n_agents=10_000, **hpv_kw, **kwargs)
    base = make(lambda a, b: {a: 0.0}, solver='ode').run().results
    sims = corpus.run_each(lambda **k: make(lambda a, b: {a: 0.95, 'hpv18': dict(acquisition=0.2)}, **k), solvers=['ode', 'abm']) # A disease or its name; a number or a dict of effects
    r = sims.ode.results
    assert r.hpv16.I[-2] < 0.5*base.hpv16.I[-2] and r.hpv18.I[-2] < base.hpv18.I[-2]
    for sim in sims.values():
        assert sim.results.prog.delivered.sum() == pytest.approx(8000, rel=0.02)
        assert len(sim.modifiers[('hpv16', 'rel_sus', None)]) == len(sim.modifiers[('hpv18', 'rel_sus', None)]) == 1
        assert [s.name for s in sim.groups.strats].count('has_vx') <= 1
    corpus.check_close(sims, lambda sim: sim.results.hpv18.infection.sum(), rtol=0.15)
    assert sims.abm.people.has_vx.sum() > 0 and 'has_vx' not in [s.name for s in sims.abm.diseases.hpv16.graph.strats]
    with pytest.raises(KeyError): # Keyed by a disease and by something else
        make(lambda a, b: {a: 0.9, 'acquisition': 0.5}, solver='abm').init()
    return sims


def test_any():
    sc.heading('Testing a condition across diseases in detects= and eligible=, with the disease taken from the conditions')
    seen = sc.obj()
    def make(**kwargs):
        genos = [make_hpv('hpv16'), make_hpv('hpv18', beta=1.2)]
        any_cin, any_ca = genos[0].CIN | genos[1].CIN, genos[0].CA | genos[1].CA
        def look(sim):
            if isinstance(sim.solver, ss.solvers.abm):
                d, u = sim.diseases, sim.people.auids
                ca = d.hpv16.CA.raw[u] | d.hpv18.CA.raw[u]
                seen.update(eligible=np.count_nonzero(~ca), found=0.9*np.count_nonzero((d.hpv16.CIN.raw[u] | d.hpv18.CIN.raw[u]) & ~ca))
            return True
        prog = ss.screen(ss.Diagnostic(sensitivity=0.9, detects=any_cin), who=~any_ca & look, when=[2005]) # No disease=
        return ss.Sim(genos, interventions=prog, **hpv_kw, **kwargs)
    sims = corpus.run_each(make, solvers=['ode', 'abm'], n_agents=10_000)
    for solver,sim in sims.items():
        r = sim.results
        k = int(np.flatnonzero(r.prog.delivered)[0])
        exp = seen
        if solver == 'ode': # Independent genotypes: from the marginals
            f = {x:[r[g][x][k]/10_000 for g in ['hpv16', 'hpv18']] for x in ['CIN', 'CA']} # At the start of the step it was delivered in
            no_ca = np.prod([1 - p for p in f['CA']])
            exp = sc.obj(eligible=10_000*no_ca, found=0.9*10_000*(no_ca - np.prod([1 - a - b for a,b in zip(f['CA'], f['CIN'])])))
        assert np.isclose(r.prog.eligible[k], exp.eligible) and np.isclose(r.prog.positive[k], exp.found, rtol=1e-6 if solver == 'ode' else 0.05), solver
    assert any('tracked jointly' in note for note in sims.ode.notes)
    return sims


def test_doses_and_holding():
    sc.heading('Testing a two-dose vaccine, the second at least 6 months after the first; and a product that fades held until none of its protection is left, and not given again within a time')
    sis = corpus.make_sis(beta=ss.peryear(1.5), dur=ss.years(1)).stratify(age=[0, 9, 15])
    vx = ss.Vaccine(efficacy=0.9, doses=[0.6, 1.0])
    girls = (ss.person.age >= 9) & (ss.person.age < 15)
    first = ss.vaccinate(vx, who=girls & ~vx.ever, when=ss.schedule(start=2020), rate=ss.probperyear(0.5))
    second = ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.months(6)), coverage=0.9) # The next dose, for those who had the first
    sim = ss.Sim(sis, interventions=[first, second], start=2019, stop=2026, dt=ss.months(1), aging=False, total_pop=50_000, n_agents=25_000, verbose=0).run()
    res = sim.results
    assert isinstance(sim.solver, ss.solvers.abm) # vx.since is each person's own
    d1, d2 = res.first.delivered, res.second.delivered
    start = int(np.flatnonzero(d1)[0])
    assert d2[:start + 6].sum() == 0 and d2[start + 6] > 0 # No second dose within 6 months of the first
    assert np.isclose(d2[start + 6:].sum(), 0.9*d1[:-6].sum(), rtol=0.05)
    assert np.isclose(res.first.delivered.by('age').values[:, [0, 2]].sum(), 0)

    # Held while any protection is left, then not: it acts only while held; monthly rounds for a year reach each person at most twice, 6 months apart
    S, L = ss.states('S L')
    tpt = ss.Vaccine(efficacy=dict(acquisition=0.0), waning=ss.boxcar(ss.months(3)), name='tpt')
    tb = ss.Disease(S, L, infection=S >> L @ ss.peryear(0.5), cure=ss.Transition(L, S, rate=ss.peryear(20), who=tpt.held), init=dict(L=0.5), name='tb')
    give = ss.vaccinate(tpt, when=ss.schedule(every=ss.months(1), start=2001, stop=2002), coverage=0.2, skip_reached_within=ss.months(6), name='give')
    sim = ss.Sim(tb, interventions=give, start=2000, stop=2004, dt=ss.months(1), n_agents=500, verbose=0).run()
    assert sim.results.tb.cure[12:].sum() > 0 and sim.results.tb.cure[36:].sum() == 0 and not sim.people.has_tpt.values.any()
    give = sim.interventions.give
    assert len((give.n_times == 2).uids) and not len((give.n_times > 2).uids)
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
