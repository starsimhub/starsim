"""Names from variables: where it works and where it does not"""
import os
os.environ['MPLBACKEND'] = 'agg'
import starsim as ss

def show(label, mod):
    print(f'{label:40s} transitions={list(mod.graph.transitions.keys())} states={list(mod.graph.states.keys())} pars={list(mod.graph.pars.keys())} guessed={dict(mod.graph.guessed)} trguess={ {k: dict(t.guessed) for k, t in mod.graph.transitions.items() if t.guessed} }')

# 1. bound to variables at module level
S, I, R = ss.states(3)
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)
show('1 module-level variables', ss.Disease(infection, recovery))

# 2. in a list
S, I, R = ss.states(3)
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)
show('2 transitions=[...] list', ss.Disease(transitions=[infection, recovery]))

# 3. inline
S, I, R = ss.states(3)
show('3 inline', ss.Disease(S >> I @ 0.05 * I, I >> R @ ss.days(10)))

# 4. in a loop (chain of stages)
sts = ss.states('S E1 E2 E3 I R')
trs = [sts[i] >> sts[i+1] @ ss.days(2) for i in range(1, 4)]
inf = sts[0] >> sts[1] @ 0.3 * sts[4]
rec = sts[4] >> sts[5] @ ss.days(5)
show('4 loop-built', ss.Disease(inf, rec, *trs))

# 5. inside a function (local variables)
def make():
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)
    return ss.Disease(infection, recovery)
show('5 inside a function', make())

# 6. built in a helper and constructed in another function (frame mismatch)
def ingredients():
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)
    return [infection, recovery]
def build(trs):
    return ss.Disease(*trs)
try:
    show('6 made in one fn, Disease in another', build(ingredients()))
except Exception as E:
    print('6 made in one fn, Disease in another   ERROR:', type(E).__name__, str(E)[:300])

# 7. two names for one object
S, I, R = ss.states(3)
infection = S >> I @ 0.05 * I
inf2 = infection
recovery  = I >> R @ ss.days(10)
show('7 two names for one transition', ss.Disease(infection, recovery))

# 8. two names for one state
S, I, R = ss.states(3)
Inf = I
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)
try:
    show('8 two names for one state', ss.Disease(infection, recovery))
except Exception as E:
    print('8 two names for one state   ERROR:', type(E).__name__, str(E)[:300])

# 9. class body
class SIR(ss.Disease):
    beta = ss.par(0.05, bounds=[0.01, 0.5])
    S = ss.State(); I = ss.State(); R = ss.State()
    infection = S >> I @ beta * I
    recovery  = I >> R @ ss.days(10)
show('9 class body', SIR())
a, b = SIR(), SIR()
print('   class instances share par object?', a.pars.beta is b.pars.beta, ' share state?', a.S is b.S)

# 10. class body with ss.states(3)
class SIR2(ss.Disease):
    S, I, R = ss.states(3)
    infection = S >> I @ 0.05 * I
    recovery  = I >> R @ ss.days(10)
show('10 class body ss.states(3)', SIR2())

# 11. exec / REPL-like
ns = {}
exec('''
import starsim as ss
S, I, R = ss.states(3)
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)
sir = ss.Disease(infection, recovery)
''', ns)
show('11 exec namespace', ns['sir'])

# 12. a par named from variable vs a literal in a slot
S, I, R = ss.states(3)
beta = ss.par(0.05)
gamma = 0.1
show('12 par var + bare number var', ss.Disease(S >> I @ beta * I, I >> R @ gamma))

# 13. anonymous pars: two literal rates -> named rate_<tr>
show('13 two literal rates', ss.Disease(ss.progression('S -> I', rate=0.1), ss.progression('I -> R', rate=0.2)))
show('13b beta literal + dur literal', ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10)))

# 14. module name from the variable passed to Sim
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
measles = sir
sim = ss.Sim(sir, stop=10).init()
print('14 module name from variable at ss.Sim:', list(sim.diseases.keys()))
mysir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10))
sim = ss.Sim(mysir, stop=10).init()
print('14b single-bound variable:', list(sim.diseases.keys()))
sim = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10)), stop=10).init()
print('14c inline Disease:', list(sim.diseases.keys()))

# 15. same transition object passed to two Diseases
S, I, R = ss.states(3)
infection = S >> I @ 0.05 * I
recovery  = I >> R @ ss.days(10)
d1 = ss.Disease(infection, recovery)
try:
    d2 = ss.Disease(infection, recovery)
    print('15 reuse transitions in two Diseases: ok', d2.graph.states)
except Exception as E:
    print('15 reuse transitions in two Diseases ERROR:', type(E).__name__, str(E)[:300])

# 16. reserved names and collision with Module attrs
for name in ['step', 'pars', 't', 'dt', 'results', 'name', 'plot', 'init', 'sim', 'infected', 'susceptible']:
    try:
        ss.Disease(**{name: ss.transmission('S -> I', beta=0.05)}, recovery=ss.progression('I -> R', dur=10))
        print(f'16 keyword {name!r}: accepted')
    except Exception as E:
        print(f'16 keyword {name!r}: {type(E).__name__} {str(E)[:90]}')
