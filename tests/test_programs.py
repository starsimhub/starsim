"""
Test programs: numbers are people, priority, queues, shared resources, stock targets, memory and doses, and steps at the same visit
"""

# %% Imports and settings
import pytest
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0, unit='day')


def make_sir():
    S, I, R = ss.states('S I R')
    return ss.Disease(infection=ss.Transition(S, I, rate=0.3 * I), recovery=ss.Transition(I, R, dur=10), init_prev=0.01, name='sir')


# %% Acceptance tests: the worked examples of proposal 02, small


def test_hpv_algorithm():
    sc.heading('Acceptance: HPV screen, then a treatment visit with a capacity of 10 a week and a queue (hpvsim_interventions/port.py algo_a, one genotype), as agents')
    def make(**kwargs):
        S, I, CIN, CA = ss.states('S I CIN CA')
        hpv = ss.Disease(
            infection = S >> I @ 0.6*(I + CIN),
            clearance = I >> S @ ss.years(1.5),
            dysplasia = I >> CIN @ ss.years(5),
            regress   = CIN >> I @ ss.years(4),
            invasion  = CIN >> CA @ ss.years(15),
            excised   = CIN >> I, # Undriven: clears the lesion, not the infection
            init=dict(I=0.1, CIN=0.03), name='hpv',
        ).stratify(age=[0, 30, 50])
        women = (ss.ppl.age >= 30) & (ss.ppl.age < 50)
        hpv_dna = ss.Diagnostic(sensitivity=dict(I=0.9, CIN=0.95), specificity=0.97)
        primary = ss.screen(hpv_dna, who=women & (hpv_dna.since >= ss.years(5)), coverage=ss.probperyear(0.3), start=2020)
        tx_visit = ss.screen(ss.Diagnostic(sensitivity=0.9, specificity=0.9, detects=~hpv.CA), who=primary.positive, coverage=0.8, capacity=ss.perweek(10), queue=ss.months(6))
        ablate = ss.treat(ss.Treatment(moves={hpv.excised: 0.85}, name='ablation'), who=tx_visit.positive)
        excise = ss.treat(ss.Treatment(moves={hpv.excised: 0.95}, name='excision'), who=tx_visit.negative)
        return ss.Sim(hpv, interventions=[primary, tx_visit, ablate, excise], start=2015, stop=2030, dt=ss.months(3), use_aging=False, **kwargs)
    with pytest.raises(ss.CapabilityError): # The queue, and hpv_dna.since
        make(solver='ode', total_pop=100_000, verbose=0).init()
    sim = make(total_pop=100_000, n_agents=20_000, verbose=0).run()
    res = sim.results
    q = res.tx_visit
    n = sc.obj({k:q[k].sum() for k in ['eligible', 'delivered', 'never_ran', 'left']})
    print(f'{n}, waiting at the end {q.waiting[-1]:.0f}')
    cap = 10*365.25/4/7 # 10 a week, about 13 weeks a quarter
    full = q.waiting[26:] > 1 # Steps with people queueing, when the capacity is all used
    assert np.all(q.delivered <= cap + 5.1) and full.sum() > 5 and np.isclose(q.delivered[26:][full].mean(), cap, rtol=0.02) # Within an agent, as agents
    assert q.waiting.max() > 0 and n.never_ran > 0 and q.waited.max() <= 6 + 1e-6, q.waited.max() # In the sim's unit, months
    assert np.isclose(0.8*n.eligible, n.delivered + n.never_ran + n.left + q.waiting[-1], rtol=0.05) # Of those who come (coverage 0.8; as agents, a few % low, the known coverage bias)
    assert np.isclose(res.ablate.delivered.sum() + res.excise.delivered.sum(), n.delivered, rtol=0.01)
    assert res.ablate.unneeded.sum() > 0 and res.ablate.cured.sum() > 0
    women = res.primary.eligible[21] # Women 30-49, none screened yet at the start of 2020 (the step that starts then)
    assert res.primary.delivered.sum() < 2*women # Nobody is screened twice in 5 years, so at most twice in 2020-2030
    return sim


def test_rescreen():
    sc.heading('Acceptance: rescreening every 5 years at most, at 30% a year, as agents')
    def make(**kwargs):
        sir = make_sir()
        test = ss.Diagnostic()
        primary = ss.screen(test, who=test.since >= ss.years(5), coverage=ss.probperyear(0.3))
        return ss.Sim(sir, interventions=primary, start=2020, stop=2040, dt=ss.months(3), **kwargs)
    with pytest.raises(ss.CapabilityError): # test.since
        make(solver='ode', total_pop=10_000, verbose=0).init()
    sim = make(total_pop=10_000, n_agents=10_000, verbose=0).run()
    d, e = sim.results.primary.delivered, sim.results.primary.eligible
    assert np.isclose(d[1], 10_000*(1 - 0.7**0.25), rtol=0.05)
    assert np.isclose(e[20], 10_000 - d[1:20].sum()) and np.isclose(e[21], 10_000 - d[2:21].sum()) # Those screened first are due again after exactly 5 years
    return sim


def test_two_doses():
    sc.heading('Acceptance: a two-dose vaccine, the second at least 6 months after the first (hpvsim_interventions/port.py), as agents')
    def make(**kwargs):
        S, I = ss.states('S I')
        sir = ss.Disease(infection=ss.Transition(S, I, rate=ss.peryear(1.5) * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.05, name='sis').stratify(age=[0, 9, 15])
        vx = ss.Vaccine(efficacy=0.9, doses=[0.6, 1.0])
        girls = (ss.ppl.age >= 9) & (ss.ppl.age < 15)
        first = ss.vaccinate(vx, who=girls & ~vx.ever, coverage=ss.probperyear(0.5), start=2020)
        second = ss.vaccinate(vx, who=vx.ever & (vx.since >= ss.months(6)), coverage=0.9) # The next dose, for those who had the first
        return ss.Sim(sir, interventions=[first, second], start=2019, stop=2026, dt=ss.months(1), use_aging=False, **kwargs)
    with pytest.raises(ss.CapabilityError): # vx.since
        make(solver='ode', total_pop=50_000, verbose=0).init()
    sim = make(total_pop=50_000, n_agents=25_000, verbose=0).run()
    res = sim.results
    d1, d2 = res.first.delivered, res.second.delivered
    start = int(np.flatnonzero(d1)[0])
    assert d2[:start + 6].sum() == 0 and d2[start + 6] > 0 # No second dose within 6 months of the first
    assert np.isclose(d2[start + 6:].sum(), 0.9*d1[:-6].sum(), rtol=0.05)
    assert np.isclose(res.first.delivered.by('age').values[:, [0, 2]].sum(), 0)
    return sim


if __name__ == '__main__':
    sc.options(interactive=False)
    hpv = test_hpv_algorithm()
    rescreen = test_rescreen()
    doses = test_two_doses()
