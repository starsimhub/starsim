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


def w209(sim):
    """ The warnings raised when a sim is initialized that a transition has nothing to drive it """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        sim.init()
    return [str(x.message) for x in w if 'so only custom code can move people along it' in str(x.message)]


# %% Define the tests


def run_both(make, stop=200, pop=20_000, **kwargs):
    """ Run the sim that `make` returns as an ODE and as agents """
    return sc.obj(ode=make(solver='ode', total_pop=pop, stop=stop, **kw, **kwargs).run(), abm=make(solver='abm', n_agents=pop, stop=stop, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.obj({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def make_sti(name='ng', beta=0.015, fail=None, init_prev=0.1, p_fail=None):
    """ A bacterial STI as in STIsim (stisim_bacterial/port.py), with treatment as an undriven transition and no treated state; `fail` is the event of a treatment failing """
    S, E, A, P, Y, C = ss.states('S E A P Y C')
    trs = dict(
        infection = S >> E @ beta*(A + P + Y + C),
        latency   = ss.Transition(E, [P, A], p=0.4, dur=ss.days(5)),
        onset     = P >> Y @ ss.days(10),
        seek      = Y >> C @ ss.days(20),
        clear_a   = A >> S @ ss.days(150),
        clear_y   = (Y + C) >> S @ ss.days(200),
        cure      = (E + A + P + Y + C) >> S, # Undriven: moved only by treatment
    )
    if fail is not None: # Failures lose their symptoms and stop seeking care
        trs['fail'] = ss.Transition(P + Y + C, A, where=fail, p=p_fail)
    return ss.Disease(**trs, init_prev=init_prev, name=name)


def test_sti_failure():
    sc.heading('Testing treatment with failure and no treated state: cured, failed, and unneeded')
    def make(**kwargs):
        ng = make_sti()
        drug = ss.Treatment(moves={ng.cure: 0.9})
        clinic = ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic')
        mass = ss.treat(drug, where=ng.S | ng.A, coverage=0.05, interval=ss.days(30), name='mass') # Most of them uninfected: unneeded
        return ss.Sim(ng, interventions=[clinic, mass], **kwargs)
    sims = run_both(make)
    for solver,sim in sims.items():
        res = sim.results
        for key in ['clinic', 'mass']:
            r = res[key]
            n = sc.obj({k:r[k].sum() for k in ['delivered', 'cured', 'failed', 'unneeded']})
            print(f'{solver} {key}: {n}')
            assert np.isclose(n.cured + n.failed + n.unneeded, n.delivered)
            assert np.isclose(n.cured/(n.cured + n.failed), 0.9, atol=0.03)
        assert res.clinic.unneeded.sum() == 0 and res.mass.unneeded.sum() > res.mass.cured.sum()
        assert np.isclose(res.ng.cure.sum(), res.clinic.cured.sum() + res.mass.cured.sum()) # The disease's flow is the same people
    check_close(sims, lambda sim: sim.results.clinic.cured.sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.mass.unneeded.sum(), rtol=0.05)

    # Failure along another transition, triggered by the outcome, and a modifier of the undriven transition, which multiplies the efficacy
    def make_fail(**kwargs):
        drug = ss.Treatment(moves='ng.cure', cure=0.8)
        ng = make_sti(fail=drug.failed)
        return ss.Sim(ng, interventions=ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic'),
                      connectors=ss.modify(ng.cure, multiply=0.5, start=ss.days(100)), **kwargs)
    sims = run_both(make_fail)
    for solver,sim in sims.items():
        res = sim.results
        early, late = slice(0, 100), slice(101, None)
        share = lambda s: res.clinic.cured[s].sum()/(res.clinic.cured[s].sum() + res.clinic.failed[s].sum())
        assert np.isclose(share(early), 0.8, atol=0.04) and np.isclose(share(late), 0.4, atol=0.05), f'{solver}: {share(early):.2f}, {share(late):.2f}'
        assert np.isclose(res.ng.fail.sum(), res.clinic.failed.sum()) # Every failure (in C) moved to A
    check_close(sims, lambda sim: sim.results.clinic.failed.sum(), rtol=0.15)

    # With p=, that share of the failures move (resistance at failure)
    def make_resist(**kwargs):
        drug = ss.Treatment(moves='ng.cure', cure=0.8)
        ng = make_sti(fail=drug.failed, p_fail=0.05)
        return ss.Sim(ng, interventions=ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic'), **kwargs)
    for solver,sim in run_both(make_resist).items():
        res = sim.results
        assert np.isclose(res.ng.fail.sum(), 0.05*res.clinic.failed.sum(), rtol=0.01 if solver == 'ode' else 0.5), f'{solver}: {res.ng.fail.sum()} of {res.clinic.failed.sum()} failures' # As agents, about 25 of 500 failures: a relative sd of 0.2

    # A transfer for those who failed, as a step of its own: resistance acquired
    def make_amr(**kwargs):
        ng = make_sti().stratify(amr=['sens', 'res'], split=[1, 0])
        drug = ss.Treatment(moves={ng.cure: 0.8})
        clinic = ss.treat(drug, where=ng.C, interval=ss.days(30))
        clinic_fail = ss.Delivery(move=dict(amr=('sens', 'res')), who=clinic.failed, coverage=0.5)
        return ss.Sim(ng, interventions=[clinic, clinic_fail], **kwargs)
    sims = run_both(make_amr)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.clinic_fail.delivered.sum(), 0.5*res.clinic.failed.select(amr='sens').sum(), rtol=0.15)
        assert res.clinic.failed.select(amr='res').sum() > 0 # Those who became resistant fail again
    check_close(sims, lambda sim: sim.results.clinic_fail.delivered.sum(), rtol=0.2)
    return sims


def test_crossed_outcomes():
    """ The outcomes of a treatment of two diseases tracked jointly are events as compartments too: steps on them, and a failure triggered by one """
    sc.heading('Testing the outcomes of a treatment of two diseases tracked jointly')
    def make(**kwargs):
        drug = ss.Treatment(moves={'ng.cure': 0.8, 'ct.cure': 0.8}, name='drug')
        ng = make_sti('ng', fail=drug.failed)
        ct = make_sti('ct')
        clinic = ss.treat(drug, where=ng.C | ct.C, interval=ss.days(30), name='clinic')
        fail_step = ss.Delivery(who=clinic.failed, name='fail_step')
        cure_step = ss.Delivery(who=clinic.cured, name='cure_step')
        return ss.Sim([ng, ct], interventions=[clinic, fail_step, cure_step], **kwargs)
    sims = run_both(make)
    for solver,sim in sims.items():
        r = sim.results
        assert sim.solver == solver and r.fail_step.delivered.sum() > 0 and r.cure_step.delivered.sum() > 0, solver # As compartments, it reached nobody
        assert r.fail_step.delivered.sum() <= 1.02*r.clinic.failed.sum() and r.ng.fail.sum() > 0, solver # The results count each disease; the event, each person
    for key in ['fail_step', 'cure_step']:
        check_close(sims, lambda sim: sim.results[key].delivered.sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.ng.fail.sum(), rtol=0.15)

    # One disease: a step on the outcome finds the failures that a transition triggered by it has moved (as compartments it found none)
    def make_one(**kwargs):
        drug = ss.Treatment(moves='ng.cure', cure=0.8, name='drug')
        ng = make_sti('ng', fail=drug.failed)
        clinic = ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic')
        return ss.Sim(ng, interventions=[clinic, ss.Delivery(who=clinic.failed, name='fail_step')], **kwargs)
    for solver,sim in run_both(make_one).items():
        r = sim.results
        assert np.isclose(r.fail_step.delivered.sum(), r.clinic.failed.sum(), rtol=0.01), f'{solver}: {r.fail_step.delivered.sum()} of {r.clinic.failed.sum()} failures'
    return sims


def test_efficacy_forms():
    sc.heading('Testing efficacy as anything a parameter accepts, and a modifier of the undriven transition for some people')
    for eff,expected in [(ss.by(amr=[0.9, 0.3]), [0.9, 0.15]), (ss.par(0.7), [0.7, 0.35]), (lambda sim: 0.6, [0.6, 0.3])]:
        def make(**kwargs):
            ng = make_sti().stratify(amr=['sens', 'res'], split=[0.5, 0.5])
            drug = ss.Treatment(moves={ng.cure: eff})
            return ss.Sim(ng, interventions=ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic'),
                          connectors=ss.modify(ng.cure, multiply=0.5, where=ng.amr == 'res'), **kwargs)
        for solver,sim in run_both(make).items():
            r = sim.results.clinic
            shares = [r.cured.select(amr=lvl).sum()/(r.cured.select(amr=lvl).sum() + r.failed.select(amr=lvl).sum()) for lvl in ['sens', 'res']]
            assert np.allclose(shares, expected, atol=0.04), f'{solver}: {shares} for {expected}'
    return sim


def test_syphilis_stages():
    sc.heading('Testing a treatment with a destination for each stage: early and late syphilis')
    def make_syph():
        S, Sp, Spl, E, P, S2, EL, LL = ss.states('S Sp Spl E P S2 EL LL')
        return ss.Disease(
            infection  = (S + Sp + Spl) >> E @ 0.01*(P + S2 + EL),
            primary    = E >> P @ ss.days(30),
            secondary  = P >> S2 @ ss.days(45),
            latent     = S2 >> EL @ ss.days(100),
            late       = EL >> LL @ ss.days(300),
            cure_early = (E + P + S2 + EL) >> Sp, # Treated in an early stage: treponemal positive
            cure_late  = LL >> Spl,                # Treated late: a different serological state
            init=dict(S2=0.05, EL=0.05, LL=0.05), name='syph',
        )
    def make(**kwargs):
        syph = make_syph()
        bpg = ss.Treatment(moves={syph.cure_early: 0.98, syph.cure_late: 0.8})
        return ss.Sim(syph, interventions=ss.treat(bpg, coverage=0.1, interval=ss.days(30), name='tx'), **kwargs)
    sims = run_both(make)
    for solver,sim in sims.items():
        res = sim.results
        cured = res.tx.cured
        early, late = [cured.select(edge=e).sum() for e in ['cure_early', 'cure_late']]
        assert np.isclose(early + late, cured.sum()) and early > 0 and late > 0
        assert np.isclose(early, res.syph.cure_early.sum()) and np.isclose(late, res.syph.cure_late.sum()) # Each to its own destination
        for edge,eff in [('cure_early', 0.98), ('cure_late', 0.8)]:
            c, f = cured.select(edge=edge).sum(), res.tx.failed.select(edge=edge).sum()
            assert np.isclose(c/(c + f), eff, atol=0.04), f'{solver} {edge}: {c/(c + f):.2f}'
        assert np.allclose(cured.by('edge').values.sum(axis=0), [early, late])
        assert np.isclose(res.tx.unneeded.sum(), 0) # By default, only people in the sources are eligible
    check_close(sims, lambda sim: sim.results.tx.cured.select(edge='cure_late').sum(), rtol=0.15)

    # A person who could match two transitions of one disease is an error
    syph = make_syph()
    with pytest.raises(ValueError):
        ss.Sim(syph, interventions=ss.treat(ss.Treatment(moves=[syph.cure_early, syph.late])), **kw).init()
    return sims


def test_syndromic():
    sc.heading('Testing one treatment for two diseases: the same people are treated for both')
    def make(**kwargs):
        kit = ss.Treatment(moves={'ng.cure': 0.95, 'ct.cure': 0.9}, name='kit')
        ng, ct = make_sti('ng', beta=0.015, fail=kit.failed), make_sti('ct', beta=0.012) # Its outcomes are events, for a transition or a step
        syndromic = ss.treat(kit, where=ng.C | ct.C, coverage=0.5, interval=ss.days(30)) # No disease=: the product names them
        second = ss.treat(ss.Treatment(moves={'ng.cure': 1.0, 'ct.cure': 1.0}), where=syndromic.failed, name='second')
        return ss.Sim([ng, ct], interventions=[syndromic, second], **kwargs)
    sims = run_both(make, pop=10_000)
    for solver,sim in sims.items():
        r = sim.results.syndromic
        delivered = r.delivered.sum()
        for dis in ['ng', 'ct']:
            n = [r[key].select(disease=dis).sum() for key in ['cured', 'failed', 'unneeded']]
            print(f'{solver} {dis}: delivered {delivered:.0f}, cured/failed/unneeded {np.round(n)}')
            assert np.isclose(sum(n), delivered) # Each disease sees every recipient: chosen once, before anyone moves
            assert np.isclose(n[0] + sim.results.second.cured.select(disease=dis).sum(), sim.results[dis].cure.sum())
        assert r.unneeded.select(disease='ct').sum() > 0 # Treated for CT because of NG symptoms, and the reverse
        assert np.isclose(r.cured.sum(), r.cured.by('disease').values.sum())
        assert np.isclose(sim.results.second.delivered.sum(), r.failed.sum(), rtol=0.05) and sim.results.ng.fail.sum() > 0 # Few fail for both
    check_close(sims, lambda sim: sim.results.ng.fail.sum(), rtol=0.2)
    check_close(sims, lambda sim: sim.results.syndromic.unneeded.select(disease='ct').sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.syndromic.cured.select(disease='ng').sum(), rtol=0.15)
    assert sims.ode.crossings # The diseases are tracked jointly, since the treatment acts on both
    return sims


def test_bv():
    sc.heading('Testing BV treatment along an undriven transition (it was tx = ... @ 0)')
    def make_bv():
        CST1, CST3, CST4a, CST4p, CST4s, P = ss.states('CST1 CST3 CST4a CST4p CST4s P')
        return ss.Disease(
            shift13  = CST1 >> CST3 @ ss.probpermonth(0.10),
            onset    = ss.Transition(CST3, [CST4p, CST4a], rate=ss.probpermonth(0.05), p=0.7),
            revert   = CST3 >> CST1 @ ss.weeks(10),
            symptoms = CST4p >> CST4s @ ss.weeks(2),
            clear_s  = CST4s >> CST1 @ ss.weeks(25),
            clear_a  = CST4a >> CST1 @ ss.weeks(25),
            tx       = (CST4a + CST4p + CST4s) >> P, # Moved only by treatment
            recur    = ss.Transition(P, [CST4p, CST4a], rate=ss.permonth(0.1), p=0.7),
            settle   = P >> CST1 @ ss.months(6),
            init=dict(CST3=0.3, CST4a=0.1), first='CST1', name='bv',
        )
    def make(**kwargs):
        bv = make_bv()
        mtz = ss.Treatment(moves=bv.tx, cure=0.9)
        return ss.Sim(bv, interventions=ss.treat(mtz, where=bv.symptoms, coverage=0.9, name='treat_bv'), **kwargs)
    sims = run_both(make, stop=365)
    for solver,sim in sims.items():
        res = sim.results
        assert not w209(make(solver=solver, stop=10, **kw))
        assert 'undriven: moved by mtz' in sim.diseases.bv.describe(output=True)
        assert np.isclose(res.treat_bv.delivered.sum(), 0.9*res.bv.symptoms.sum(), rtol=0.05)
        assert np.isclose(res.bv.tx.sum(), res.treat_bv.cured.sum()) and np.isclose(res.treat_bv.cured.sum(), 0.9*res.treat_bv.delivered.sum(), rtol=0.05)
        assert res.bv.settle.sum() > 0 # People moved into P leave it again
    check_close(sims, lambda sim: sim.results.bv.tx.sum(), rtol=0.1)
    return sims


if __name__ == '__main__':
    sc.options(interactive=False)
    failure  = test_sti_failure()
    forms    = test_efficacy_forms()
    syph     = test_syphilis_stages()
    synd     = test_syndromic()
    bv       = test_bv()
