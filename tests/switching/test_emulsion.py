"""
Switching demo for the EMULSION seat: the published herd model with minimum stays, offspring, and births between stages.

Passes when: matches EMULSION's published output.
Blocked today by: `after=`, `offspring=`, N11.
The published model is EMULSION's quickstart (100 animals, 105 weekly steps); the means are compared over 10 runs rather than EMULSION's 200.
"""
import sys
import tempfile
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.interop
sys.path.insert(0, str(sc.thispath().parent)) # For tests/paths.py when run as a script (conftest.py does this under pytest)
import paths

dt, adult_age, n_runs = 7, 30, 10 # Days, days (the YAML's second value, as EMULSION uses), and runs to average
ref_quickstart = { # Mean of 200 EMULSION 1.1.1 runs after 53 and 105 steps: LFD examples/emulsion/quickstart/reference/results.csv (switching/emulsion.md, "The first day", gives the 66 at the end)
    53:  dict(total=76.83, J=15.83, M=21.48, I=18.84),
    105: dict(total=65.89, J=6.16, M=10.98, I=18.43),
}
ref_metapop = dict(total=1499, J=764, A=735, I=36.8, R=66.6) # switching/emulsion.md, "The second week" §2: compart_SIR_metapop.yaml, mean of 100 EMULSION runs at day 100
ref_proba_left = 0.9**7 # switching/emulsion.md §1: EMULSION's share still infectious after one 7-day step with proba 0.1 (47.7 of 100 in its run)


def check_quickstart(msim, age='growth', health='health'):
    """ Compare the mean counts over runs with EMULSION's, within the noise of 10 runs """
    mean = lambda mod, *states: np.sum([np.asarray(msim.results[mod][f'n_{s}'].values) for s in states], axis=0).mean(axis=0)
    for k,ref in ref_quickstart.items():
        ours = dict(total=mean(age, 'J', 'A')[k], J=mean(age, 'J')[k], M=mean(health, 'M')[k], I=mean(health, 'I')[k])
        for key,val in ref.items():
            assert abs(ours[key] - val) < 0.1*ref_quickstart[105]['total'], f'After {k} steps, {key} is {ours[key]:.1f} against EMULSION\'s {val}'


@pytest.mark.skipif(not paths.lfd.exists(), reason=paths.no_lfd)
@pytest.mark.xfail(reason='The import lists the untranslated parts (cond, when, on_cross, statevars, the calendar; minimum stays are now after=), and warns and hands them to the import-model skill when run (15 animals at the end against 66); a complete import needs after=/before= and offspring=, as test_emulsion_quickstart', strict=False)
def test_emulsion_import():
    sc.heading('Testing importing EMULSION\'s quickstart and matching its published output')
    sim = ss.interop.load(paths.lfd/'examples/emulsion/quickstart/original/quickstart.yaml', format='emulsion', verbose=False)
    assert not sim.imported.incomplete, f'{len(sim.imported.incomplete)} parts of the model were not translated'
    check_quickstart(sim.new(verbose=0).run(n_runs=n_runs), age='age_group', health='health_state')


def test_emulsion_proba(tmp_path):
    sc.heading('Testing that an EMULSION proba is per time unit, as EMULSION reads it')
    yaml = '''
time_info: {time_unit: days, delta_t: 7, origin: 'January 1', total_duration: 14}
levels: {population: {aggregation_type: compartment}}
processes: {population: [health_state]}
state_machines:
  health_state:
    states: [{S: {}}, {I: {}}, {R: {}}]
    transitions: [{from: S, to: I, rate: 0}, {from: I, to: R, proba: 0.1}]
prototypes: {population: [{infected: {health_state: I}}]}
initial_conditions: {population: [{prototype: infected, amount: 1000}]}
'''
    sc.savetext(tmp_path/'proba.yaml', yaml)
    sim = ss.interop.load(tmp_path/'proba.yaml', format='emulsion', verbose=False).new(solver='ode', verbose=0).run()
    left = sim.results[sim.diseases[0].name].n_I.values[1]/1000
    assert np.isclose(left, ref_proba_left, rtol=1e-3), f'{left:.1%} still infectious after one step, against {ref_proba_left:.1%} in EMULSION'


@pytest.mark.xfail(reason='offspring= by prototype (a newborn in a state of each of several diseases, with probabilities from the mother\'s state) does not exist; offspring= takes one state and a number (after= and before= exist)', strict=False)
def test_emulsion_quickstart():
    sc.heading('Testing the quickstart written in v4 with minimum stays and offspring declared')
    ppl = ss.People(100, age_data=ss.uniform(low=dt/365, high=(adult_age + dt)/365))
    J, A = ss.states('J A')
    growth = ss.Disease(maturity=ss.Transition(J, A, rate=ss.perday(1e9), where=ppl.age >= adult_age/365))
    S, E, I, R, Q, M = ss.states('S E I R Q M')
    quarantine = ss.par(ss.timeseries({'2021-01-01': 0, '2021-04-01': 1, '2021-09-02': 0, '2022-04-01': 1, '2022-09-02': 0}))
    health = ss.Disease(
        infection   = S >> E @ ss.perday(0.1)*I.n/(S + E + I + R + Q + M).n,
        onset       = ss.Transition(E, I, rate=ss.perday(0.3), after=ss.days(10)), # EMULSION's duration: a minimum stay
        recovery    = I >> R @ ss.perday(0.01),
        detection   = I >> Q @ ss.perday(0.01)*quarantine*(Q.n < 20),
        release     = Q >> R @ ss.constant(ss.days(30)),
        waning      = R >> S @ ss.perday(0.01),
        loss        = ss.Transition(M, S, rate=ss.perday(0.04), after=ss.days(adult_age)),
        deaths      = S + R >> ss.null @ ss.perday(0.002),
        sick_deaths = I + Q >> ss.null @ ss.perday(0.004),
        M_deaths    = ss.Transition(M, ss.null, rate=ss.perday(0.002), before=ss.days(adult_age)), # EMULSION's escape
        E_deaths    = ss.Transition(E, ss.null, rate=ss.perday(0.004), before=ss.days(10)),
        init_n      = dict(I=10),
    )
    NG, G = ss.states('NG G')
    life = ss.Disease(
        conception = ss.Transition(NG, G, rate=ss.perday(0.05), after=ss.days(90), where=growth.A),
        farrowing  = ss.Transition(G, NG, dur=ss.constant(ss.days(240)), offspring={(J, NG, I): 0.3*(E | I | Q), (J, NG, M): 1 - 0.3*(E | I | Q)}), # EMULSION's produce_offspring, by prototype
        abortion   = ss.Transition(G, NG, rate=ss.perday(-np.log(0.8)/240), where=health.I),
    )
    sim = ss.Sim([growth, life, health], people=ppl, unit='day', dt=dt, start='2021-01-01', stop=ss.days(735), use_aging=True, solver='abm', verbose=0)
    check_quickstart(sim.run(n_runs=n_runs))


def make_metapop(staged=False):
    """ EMULSION's compart_SIR_metapop: an SIR crossed with juvenile and adult, in 10 populations of 100 """
    b, K = ss.perday(0.05), 150
    if staged: # As an EMULSION user writes it: the stage as a dimension, with births from adults
        S, I, R = ss.states(3)
        N = S + I + R
        sir = ss.Disease(infection=S >> I @ ss.perday(0.5)*I.n/N.n, recovery=I >> R @ ss.perday(0.1), births=ss.birth(S, rate=ss.by(stage=[0, 2*b])),
                         deaths=ss.death(rate=b*N.n/K), maturation=ss.transfer(stage=('J', 'A'), rate=ss.perday(1/20)), init_n=dict(I=ss.by(pop=[10] + [0]*9)), name='sir')
        sir.stratify(stage=['J', 'A'], split=[1, 0])
    else: # The two machines crossed by hand into six states
        SJ, IJ, RJ, SA, IA, RA = ss.states(6)
        N, I = SJ + IJ + RJ + SA + IA + RA, IJ + IA
        sir = ss.Disease(infection_J=SJ >> IJ @ ss.perday(0.5)*I.n/N.n, infection_A=SA >> IA @ ss.perday(0.5)*I.n/N.n, recovery_J=IJ >> RJ @ ss.perday(0.1),
                         recovery_A=IA >> RA @ ss.perday(0.1), mature_S=SJ >> SA @ ss.perday(1/20), mature_I=IJ >> IA @ ss.perday(1/20), mature_R=RJ >> RA @ ss.perday(1/20),
                         births=ss.null >> SJ @ 2*b*(SA + IA + RA).n, deaths=ss.death(rate=b*N.n/K), init_n=dict(IJ=ss.by(pop=[10] + [0]*9)), first='SJ', name='sir')
    sir.stratify(pop=[f'p{i}' for i in range(10)], kind='space')
    return ss.Sim(sir, solver='binomial', unit='day', start='2018-01-01', stop=ss.days(100), total_pop=1000, verbose=0).run(n_runs=20)


def check_metapop(counts):
    for key,val in ref_metapop.items():
        assert np.isclose(counts[key], val, rtol=0.1 if key in 'IR' else 0.05), f'At day 100, {key} is {counts[key]:.1f} against EMULSION\'s {val}'


def test_emulsion_metapop_crossed():
    sc.heading('Testing EMULSION\'s metapopulation example with the two machines crossed by hand')
    r = make_metapop().results.sir
    end = lambda *states: sum(np.asarray(r[f'n_{s}'].values).mean(axis=0)[-1] for s in states)
    check_metapop(dict(total=end('SJ', 'IJ', 'RJ', 'SA', 'IA', 'RA'), J=end('SJ', 'IJ', 'RJ'), A=end('SA', 'IA', 'RA'), I=end('IJ', 'IA'), R=end('RJ', 'RA')))


@pytest.mark.xfail(reason='A birth has no where=, so births from one stratum into another cannot be declared (phase 7); ss.by(stage=...) on a birth rate raises (N11)', strict=False)
def test_emulsion_metapop_stages():
    sc.heading('Testing EMULSION\'s metapopulation example with births from adults into juveniles')
    r = make_metapop(staged=True).results.sir
    end = lambda s, **sel: np.asarray((r[f'n_{s}'].select(**sel) if sel else r[f'n_{s}']).values).mean(axis=0)[-1]
    total = sum(end(s) for s in 'SIR')
    check_metapop(dict(total=total, J=sum(end(s, stage='J') for s in 'SIR'), A=sum(end(s, stage='A') for s in 'SIR'), I=end('I'), R=end('R')))


if __name__ == '__main__':
    for test in [test_emulsion_import, lambda: test_emulsion_proba(sc.path(tempfile.mkdtemp())), test_emulsion_quickstart, test_emulsion_metapop_crossed, test_emulsion_metapop_stages]:
        try:
            test()
        except Exception as e: # Most are expected to fail today: see each test's xfail reason
            print(f'Failed: {type(e).__name__}: {e}')
