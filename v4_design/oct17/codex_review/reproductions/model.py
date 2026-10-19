"""Small observations used in evidence/model.txt; no implementation edits."""
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import starsim as ss


def disease(beta=0.1, init=None):
    return ss.Disease(
        infection=ss.transmission('S -> I', beta=beta),
        recovery=ss.progression('I -> R', dur=10),
        init={'I': 0.01} if init is None else init,
        name='sir',
    )


for label, beta in [
    ('scalar', 0.1),
    ('function', lambda sim: 0.1),
    ('timeseries', ss.timeseries({0: 0.1, 1: 0.1})),
]:
    model = disease(beta)
    sim = ss.Sim(model, interventions=ss.modify(model.pars.beta, multiply=0),
                 solver='ode', total_pop=100, start=0, stop=1, unit='day', verbose=0).run()
    print('Zero transmission modifier:', label, sim.results.sir.n_I.values[-1])

sim = ss.Sim(disease(init={'I': -0.1}), solver='ode', total_pop=100,
             start=0, stop=1, unit='day', verbose=0).init()
print('Negative initial population:', sim.diseases.sir.engine.y.tolist())

for label, kwargs in [
    ('negative split', dict(levels=['a', 'b'], split=[-1, 2])),
    ('zero split', dict(levels=['a', 'b'], split=[0, 0])),
    ('unordered edges', dict(levels=[20, 0, 10])),
]:
    strat = ss.transitions.Strat('age', **kwargs)
    print('Stratification:', label, strat.split)

model = disease(ss.par(0.8, bounds=[0.01, 0.1]))
sim = ss.Sim(model, solver='ode', total_pop=100, start=0, stop=1,
             unit='day', verbose=0).init()
print('Out-of-bounds beta accepted:', sim.diseases.sir.pars.beta.raw)
print('Truncated grouped arithmetic:', (ss.by(age=[1, 2, 3]) + ss.by(age=[10, 20])).values)

people = ss.People(10)
try:
    print('Uninitialized age arithmetic:', people.age + 1)
except Exception as exc:
    print('Uninitialized age arithmetic:', type(exc).__name__, str(exc))
membership = people.age.isin([1, 2])
print('Uninitialized membership:', type(membership).__name__, membership)
