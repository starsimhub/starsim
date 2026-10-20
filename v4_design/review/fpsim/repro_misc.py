"""Small v4 issues hit by the FPsim port, one check each. Run: conda run -n starsim python repro_misc.py"""
import numpy as np
import starsim as ss

kw = dict(start=2000, stop=2001, dt=ss.months(1), unit='year', verbose=0)

def check(label, fn):
    try:
        print(f'{label}: {fn()}')
    except Exception as e:
        print(f'{label}: {type(e).__name__}: {str(e)[:160]}')

# 1. A dwell distribution whose parameter is a function (v3's age-dependent duration of use)
def callable_dist():
    A, B = ss.states('A B')
    d = ss.Disease(A, B, ab=ss.Transition(A, B, dur=ss.lognorm_ex(mean=lambda self, sim, uids: np.full(len(uids), 6.0), std=2)), init=dict(A=1.0), name='d')
    return ss.Sim(d, n_agents=100, **kw).run().results.d.ab.sum()
check('1 Dist with a callable parameter as dur=', callable_dist)

# 2. Typed rates inside ss.by
check('2a ss.permonth(ss.by(...))', lambda: ss.permonth(ss.by(age=[1, 2])))
check('2b ss.by(2-D table of ss.permonth)', lambda: ss.by(ss.permonth(np.ones((2, 2))), dims=['age', 'parity']))

# 3. Branches kept to one level each by move= still have to sum to 1 across levels (E205)
def e205():
    A, B = ss.states('A B')
    d = ss.Disease(A, B, b0=ss.Transition(A, B, p=0.9, dur=ss.months(1), move=dict(k='0 -> 1')), b1=ss.Transition(A, B, p=0.9, dur=ss.months(1), move=dict(k='1 -> 1')),
                   c0=ss.Transition(A, B, p=0.1, dur=ss.months(1), move=dict(k='0 -> 1')), c1=ss.Transition(A, B, p=0.1, dur=ss.months(1), move=dict(k='1 -> 1')), name='d')
    return d.stratify(k=['0', '1'])
check('3 p= per parity level (each sums to 1 within its level)', e205)

# 4. A state reached only from ss.null (an inflow split) is guessed to be a count, and the run crashes
def inflow_state():
    S, X = ss.states('S X')
    d = ss.Disease(S, X, births=ss.null >> (0.9*S + 0.1*X) @ ss.peryear(10), name='d')
    return ss.Sim(d, n_agents=100, **kw).run().results.d.n_X[-1]
check('4 inflow into a weighted destination', inflow_state)

# 5. init= by (age, a disease dimension) is silently zero
def init_joint():
    A, B = ss.states('A B')
    d = ss.Disease(A, B, ab=A >> B @ ss.peryear(1e-6), init=dict(B=ss.by(np.array([[0, 0.5], [0, 0.5]]), dims=['age', 'ever'])), name='d')
    d.stratify(ever=[False, True], split=[1, 0])
    return f'n_B at start = {ss.Sim(d, n_agents=1000, stratify=dict(age=[0, 30]), **kw).run().results.d.n_B[0]} (expected ~500)'
check('5 init=ss.by(..., dims=[age, ever])', init_joint)

# 6. A split by (age, level) given as shares within each age overrides the age split
def split_joint():
    A = ss.State('A')
    d = ss.Disease(A, stay=A >> A @ ss.peryear(0), name='d')
    d.stratify(k=['x', 'y'], split=ss.by(np.array([[0.5, 0.5], [0.5, 0.5]]), dims=['age', 'k']))
    sim = ss.Sim(d, n_agents=1000, stratify=dict(age=[0, 30], split=[0.9, 0.1]), **kw).init()
    return f'agents by age group: {np.bincount((sim.people.age.values >= 30).astype(int))} (age split given: 0.9/0.1)'
check('6 split=ss.by(..., dims=[age, k]) with rows summing to 1', split_joint)

# 7. A delivered transfer between two levels
def transfer_arrow():
    S = ss.State('S')
    d = ss.Disease(S, stay=S >> S @ ss.peryear(0), name='d')
    d.stratify(method=['none', 'impl'])
    prog = ss.transfer(method='none -> impl', coverage=0.5, years=[2000.5], disease='d')
    return ss.Sim(d, n_agents=100, interventions=prog, **kw).run().results.d.n_S.by('method').values[-1]
check('7 ss.transfer(method="none -> impl", coverage=...)', transfer_arrow)

# 8. An outcome of transitions that were scaled before being given to the module (t*0.8, passed by keyword)
def flow_sum(scaled):
    S, I, R = ss.states('S I R')
    tr = dict(a=S >> I @ ss.peryear(1), b=S >> R @ ss.peryear(1))
    if scaled:
        tr = {k: v*0.8 for k,v in tr.items()}
    d = ss.Disease(S, I, R, **tr, both=ss.outcome(0.5*(tr['a'] + tr['b'])), name='d')
    return ss.Sim(d, n_agents=100, **kw).run().results.d.both.sum()
check('8a ss.outcome(0.5*(a + b))', lambda: flow_sum(False))
check('8b the same, with a and b scaled first', lambda: flow_sum(True))

# 9. set= and add= on a p= branch do nothing; multiply= works but cannot lift a zero branch (v3 add_method: a method that starts mid-run)
def branch_mod(p0, how):
    none, pill, new = ss.states('none pill new')
    d = ss.Disease(none, pill, new, stay=ss.Transition(none, none, p=0.5, dur=ss.months(6)), to_pill=ss.Transition(none, pill, p=0.5 - p0, dur=ss.months(6)),
                   to_new=ss.Transition(none, new, p=p0, dur=ss.months(6)), back=ss.Transition([pill, new], none, dur=ss.months(12)), init=dict(none=1.0), name='c')
    amount = 0.2/p0 if how == 'multiply' else 0.2
    sim = ss.Sim(d, n_agents=2000, connectors=ss.modify(d.to_new, **{how: amount}, start=2001), **{**kw, 'stop': 2003}).run()
    return f'to_new after 2001: {sim.results.c.to_new[12:].sum():.0f}'
for p0 in [1e-6]:
    for how in ['set', 'add', 'multiply']:
        check(f'9 {how}= on a branch with p={p0}', lambda: branch_mod(p0, how))
