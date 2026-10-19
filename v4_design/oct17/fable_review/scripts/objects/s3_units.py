"""Bare numbers and the sim's unit"""
import os
os.environ['MPLBACKEND'] = 'agg'
import starsim as ss

def sir(**kw):
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', **(kw or dict(dur=10))))

def unit(label, sim):
    try:
        sim.init()
        print(f'{label:55s} unit={sim.unit!r:8} source={sim.unit_source!r:35} dt={sim.pars.dt!r:12} start={sim.pars.start!r} stop={sim.pars.stop!r} npts={len(sim.t)} disagree={sim.units_disagree}')
    except Exception as E:
        print(f'{label:55s} ERROR {type(E).__name__}: {str(E)[:200]}')

ss.options.verbose = 0
unit('a stop=100', ss.Sim(sir(), stop=100))
unit("b unit='day', stop=100", ss.Sim(sir(), unit='day', stop=100))
unit("c dt=ss.weeks(1), stop=100", ss.Sim(sir(), dt=ss.weeks(1), stop=100))
unit("d dates", ss.Sim(sir(), start='2020-01-01', stop='2020-04-01'))
unit("e start=2020, stop=2023", ss.Sim(sir(), start=2020, stop=2023))
unit("f start=0, stop=100", ss.Sim(sir(), start=0, stop=100))
unit("g stop=ss.weeks(12), per-day rate (known issue)", ss.Sim(sir(rate=ss.perday(0.1)), stop=ss.weeks(12)))
unit("h dur=ss.days(10), stop=100", ss.Sim(sir(dur=ss.days(10)), stop=100))
unit("i dur=ss.days(10) + ss.peryear elsewhere", ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=ss.days(10)), death=ss.death('I ->', rate=ss.peryear(0.02))), stop=100))
unit("j module dt=ss.days(1)", ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), dt=ss.days(1)), stop=100))
unit("k start=1900", ss.Sim(sir(), start=1900, stop=1950))
unit("l start=1899, stop=1950", ss.Sim(sir(), start=1899, stop=1950))
unit("m unit='day', start=2020, stop=2030", ss.Sim(sir(), unit='day', start=2020, stop=2030))
unit("n unit='year', start=2020, stop=2030", ss.Sim(sir(), unit='year', start=2020, stop=2030))
unit("o dt=0.1, stop=20 (summer2)", ss.Sim(sir(), dt=0.1, stop=20))
unit("p no declared module: ss.SIR()", ss.Sim(ss.SIR(), stop=10))
unit("q ss.SIR + declared, stop=100", ss.Sim([ss.SIR(), sir()], stop=100))
unit("r unit='days' (plural)", ss.Sim(sir(), unit='days', stop=10))
unit("s dt='week'", ss.Sim(sir(), dt='week', stop=10))
unit("t dates + dur=10 bare: what is 10?", ss.Sim(sir(dur=10), start='2020-01-01', stop='2020-02-01'))

print()
print('--- @ bare number is a rate; dur= is a duration ---')
S, I, R = ss.states('S I R')
d = ss.Disease(S >> I @ 0.05 * I, I >> R @ 10)
sim = ss.Sim(d, stop=20).init()
d.describe()
print()
print('--- same model with a typed duration, run in a dated sim: value conversions ---')
sim = ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=ss.perday(0.3)), recovery=ss.progression('I -> R', dur=ss.weeks(2))), start='2020-01-01', stop='2020-03-01', dt=ss.weeks(1))
sim.describe()

print()
print('--- known issue: a bare number for a module written as code keeps the default unit ---')
ss.options.verbose = 0.1
sim = ss.Sim([ss.SIS(dur_inf=10), sir()], start='2020-01-01', stop='2020-02-01')
sim.init()
print('SIS dur_inf =', sim.diseases.sis.pars.dur_inf, '| declared recovery dur =', sim.diseases[1].graph.transitions.recovery.dur.raw, 'in unit', sim.unit)
