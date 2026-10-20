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


def make_sis(cure=None):
    S, I = ss.states(2)
    return ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> S @ ss.days(20), cure=cure(S, I), init_prev=0.05, name='sis')


def w209(sim):
    """ The ss.UndrivenUnused warnings raised when a sim is initialized """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        sim.init()
    return [str(x.message) for x in w if issubclass(x.category, ss.UndrivenUnused)]


# %% Define the tests

def test_undriven():
    sc.heading('Testing undriven transitions')
    forms = dict(
        op  = lambda S, I: I >> S,
        tr  = lambda S, I: ss.Transition(I, S),
        arr = lambda S, I: ss.progression('I -> S'),
    )
    for key,form in forms.items():
        sis = make_sis(form)
        assert sis.cure.undriven
        assert 'undriven: moved by products, deliveries, or code' in sis.describe(output=True)
        assert len(w209(ss.Sim(sis, stop=10, **kw))) == 1, f'{key}: expected W209 for an undriven transition nothing names'

    # Named by a product, or by on=, it is not warned about; a rate of 0 is not undriven
    sis = make_sis(forms['op'])
    sim = ss.Sim(sis, interventions=ss.treat(ss.Treatment(transition='cure'), coverage=0.2, interval=5, name='tx'), stop=10, **kw)
    assert not w209(sim)
    assert 'undriven: moved by treatment' in sim.diseases.sis.describe(output=True)
    sis = make_sis(forms['op'])
    assert not w209(ss.Sim(sis, interventions=ss.screen(ss.Test(), on=sis.cure), stop=10, **kw))
    sis = make_sis(lambda S, I: I >> S @ 0)
    assert not sis.cure.undriven and not w209(ss.Sim(sis, stop=10, **kw))

    # Moved by a treatment, as agents and compartments
    for solver,pars in [('ode', dict(total_pop=10_000)), ('abm', dict(n_agents=10_000))]:
        sis = make_sis(forms['op'])
        tx = ss.treat(ss.Treatment(cure=0.9, transition='cure'), coverage=0.3, at=[ss.days(20)])
        sim = ss.Sim(sis, interventions=tx, solver=solver, stop=40, **kw, **pars).run()
        res = sim.results
        assert np.isclose(res.sis.cure.sum(), 0.9*res.tx.delivered.sum(), rtol=0.05)
        assert res.sis.cure.sum() > 0

    with pytest.raises(ss.UndrivenTransition): # An inflow still needs a rate
        ss.Disease(births=ss.Transition(ss.null, 'S'), infection=ss.transmission('S -> I', beta=0.3))
    return sim


def run_both(make, stop=200, pop=20_000, **kwargs):
    """ Run the sim that `make` returns as an ODE and as agents """
    return sc.objdict(ode=make(solver='ode', total_pop=pop, stop=stop, **kw, **kwargs).run(), abm=make(solver='abm', n_agents=pop, stop=stop, **kw, **kwargs).run())


def check_close(sims, func, rtol=0.1):
    vals = sc.objdict({solver:func(sim) for solver,sim in sims.items()})
    assert np.allclose(vals[:], vals.ode, rtol=rtol), f'Expected the solvers to agree, not {vals}'
    return vals


def make_sti(name='ng', beta=0.015, fail=False, init_prev=0.1):
    """ A bacterial STI as in STIsim (stisim_bacterial/port.py), with treatment as an undriven transition and no treated state """
    S, E, A, P, Y, C = ss.states('S E A P Y C')
    trs = dict(
        infection = S >> E @ beta*(A + P + Y + C),
        latency   = ss.progression('E -> P | A', p=0.4, dur=ss.days(5)),
        onset     = P >> Y @ ss.days(10),
        seek      = Y >> C @ ss.days(20),
        clear_a   = A >> S @ ss.days(150),
        clear_y   = (Y + C) >> S @ ss.days(200),
        cure      = (E + A + P + Y + C) >> S, # Undriven: moved only by treatment
    )
    if fail: # Failures lose their symptoms and stop seeking care
        trs['fail'] = (P + Y + C) >> A
    return ss.Disease(**trs, init_prev=init_prev, name=name)


def test_sti_failure():
    sc.heading('Testing treatment with failure and no treated state: cured, failed, and overtreated')
    def make(**kwargs):
        ng = make_sti()
        drug = ss.Treatment(moves={ng.cure: 0.9})
        clinic = ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic')
        mass = ss.treat(drug, where=ng.S | ng.A, coverage=0.05, interval=ss.days(30), name='mass') # Most of them uninfected: overtreated
        return ss.Sim(ng, interventions=[clinic, mass], **kwargs)
    sims = run_both(make)
    for solver,sim in sims.items():
        res = sim.results
        for key in ['clinic', 'mass']:
            r = res[key]
            n = sc.objdict({k:r[k].sum() for k in ['delivered', 'cured', 'failed', 'overtreated']})
            print(f'{solver} {key}: {n}')
            assert np.isclose(n.cured + n.failed + n.overtreated, n.delivered)
            assert np.isclose(n.cured/(n.cured + n.failed), 0.9, atol=0.03)
        assert res.clinic.overtreated.sum() == 0 and res.mass.overtreated.sum() > res.mass.cured.sum()
        assert np.isclose(res.ng.cure.sum(), res.clinic.cured.sum() + res.mass.cured.sum()) # The disease's flow is the same people
    check_close(sims, lambda sim: sim.results.clinic.cured.sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.mass.overtreated.sum(), rtol=0.05)

    # Failure along another transition, and a modifier of the undriven transition, which multiplies the efficacy
    def make_fail(**kwargs):
        ng = make_sti(fail=True)
        drug = ss.Treatment(moves=ng.cure, cure=0.8, on_fail=ng.fail)
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

    # A transfer for those who failed: resistance acquired
    def make_amr(**kwargs):
        ng = make_sti().stratify(amr=['sens', 'res'], split=[1, 0])
        drug = ss.Treatment(moves={ng.cure: 0.8}, on_fail=ss.Delivery(move=dict(amr='sens -> res'), coverage=0.5))
        return ss.Sim(ng, interventions=ss.treat(drug, where=ng.C, interval=ss.days(30), name='clinic'), **kwargs)
    sims = run_both(make_amr)
    for solver,sim in sims.items():
        res = sim.results
        assert np.isclose(res.clinic_fail.delivered.sum(), 0.5*res.clinic.failed.select(amr='sens').sum(), rtol=0.15)
        assert res.clinic.failed.select(amr='res').sum() > 0 # Those who became resistant fail again
    check_close(sims, lambda sim: sim.results.clinic_fail.delivered.sum(), rtol=0.2)
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
        assert np.isclose(res.tx.overtreated.sum(), 0) # By default, only people in the sources are eligible
    check_close(sims, lambda sim: sim.results.tx.cured.select(edge='cure_late').sum(), rtol=0.15)

    # A person who could match two transitions of one disease is an error
    syph = make_syph()
    with pytest.raises(ss.OverlappingSources):
        ss.Sim(syph, interventions=ss.treat(ss.Treatment(moves=[syph.cure_early, syph.late])), **kw).init()
    return sims


def test_syndromic():
    sc.heading('Testing one treatment for two diseases: the same people are treated for both')
    def make(**kwargs):
        ng, ct = make_sti('ng', beta=0.015), make_sti('ct', beta=0.012)
        kit = ss.Treatment(moves={ng.cure: 0.95, ct.cure: 0.9}, name='kit')
        syndromic = ss.treat(kit, where=ng.C | ct.C, coverage=0.5, interval=ss.days(30)) # No disease=: the product names them
        return ss.Sim([ng, ct], interventions=syndromic, **kwargs)
    sims = run_both(make, pop=10_000)
    for solver,sim in sims.items():
        r = sim.results.syndromic
        delivered = r.delivered.sum()
        for dis in ['ng', 'ct']:
            n = [r[key].select(disease=dis).sum() for key in ['cured', 'failed', 'overtreated']]
            print(f'{solver} {dis}: delivered {delivered:.0f}, cured/failed/overtreated {np.round(n)}')
            assert np.isclose(sum(n), delivered) # Each disease sees every recipient: chosen once, before anyone moves
            assert np.isclose(n[0], sim.results[dis].cure.sum())
        assert r.overtreated.select(disease='ct').sum() > 0 # Treated for CT because of NG symptoms, and the reverse
        assert np.isclose(r.cured.sum(), r.cured.by('disease').values.sum())
    check_close(sims, lambda sim: sim.results.syndromic.overtreated.select(disease='ct').sum(), rtol=0.15)
    check_close(sims, lambda sim: sim.results.syndromic.cured.select(disease='ng').sum(), rtol=0.15)
    assert sims.ode.crossings # The diseases are tracked jointly, since the treatment acts on both
    return sims


def test_bv():
    sc.heading('Testing BV treatment along an undriven transition (it was tx = ... @ 0)')
    def make_bv():
        CST1, CST3, CST4a, CST4p, CST4s, P = ss.states('CST1 CST3 CST4a CST4p CST4s P')
        return ss.Disease(
            shift13  = CST1 >> CST3 @ ss.probpermonth(0.10),
            onset    = ss.progression('CST3 -> CST4p | CST4a', rate=ss.probpermonth(0.05), p=0.7),
            revert   = CST3 >> CST1 @ ss.weeks(10),
            symptoms = CST4p >> CST4s @ ss.weeks(2),
            clear_s  = CST4s >> CST1 @ ss.weeks(25),
            clear_a  = CST4a >> CST1 @ ss.weeks(25),
            tx       = (CST4a + CST4p + CST4s) >> P, # Moved only by treatment
            recur    = ss.progression('P -> CST4p | CST4a', rate=ss.permonth(0.1), p=0.7),
            settle   = P >> CST1 @ ss.months(6),
            init=dict(CST3=0.3, CST4a=0.1), name='bv',
        )
    def make(**kwargs):
        bv = make_bv()
        mtz = ss.Treatment(moves=bv.tx, cure=0.9)
        return ss.Sim(bv, interventions=ss.treat(mtz, on=bv.symptoms, coverage=0.9, name='treat_bv'), **kwargs)
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
    undriven = test_undriven()
    failure  = test_sti_failure()
    forms    = test_efficacy_forms()
    syph     = test_syphilis_stages()
    synd     = test_syndromic()
    bv       = test_bv()
