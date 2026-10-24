"""The full task end to end, the v4 way."""
import traceback, starsim as ss, numpy as np, pandas as pd, pylab as pl, sciris as sc

def attempt(label, fn):
    print(f'\n######## {label}')
    try: out = fn(); print('[ok]'); return out
    except Exception: print('\n'.join(traceback.format_exc().splitlines()[-5:])); print('[FAIL]')

def make():
    S, E, I, R, D = ss.states(5)
    infection = S >> E @ 0.3 * I
    latent    = E >> I @ ss.days(5)
    outcome   = I >> (0.9*R + 0.1*D) @ ss.days(10)
    cases     = ss.outcome(0.5 * latent, every=ss.weeks(1))
    seir = ss.Disease(infection, latent, outcome, cases, init_prev=0.001)
    return seir

def branches():
    sim = ss.Sim(make(), start='2020-01-01', stop='2020-06-01', total_pop=1e6, verbose=0).run()
    print(list(sim.results.seir.keys()))
attempt('what results does a branching transition give?', branches)

# synthetic data: weekly cases, with columns 'cases' and also try 'deaths'
dates = pd.date_range('2020-01-01', '2021-12-31', freq='7D')
t = np.arange(len(dates))
df = pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'), cases=(30000*np.exp(-((t-10)/4)**2)).round()))
df.to_csv('seir_cases.csv', index=False)

ppl = ss.People(1)
def build(**kw):
    seir = make()
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.8), coverage=0.8, years=[2021], eligible=ppl.age > 65)
    return ss.Sim(seir, interventions=vx, data='seir_cases.csv', start='2020-01-01', stop='2022-01-01', total_pop=1e6, stratify=dict(age=[0, 65]), **kw)

def ode():
    sim = build()
    sim.run()
    print('--- results.vx:', list(sim.results.vx.keys()))
    print('--- results.vaccine:', list(sim.results.vaccine.keys()))
    for k in sim.results.vx.keys():
        r = sim.results.vx[k]
        try: print(f'   {k}: max {np.max(r.values):.3g}, last {r.values[-1]:.3g}')
        except Exception as e: print('  ', k, type(r), e)
    print('--- n_S by age and vaccine at end:')
    try: print(sim.results.seir.n_S.by('age', 'has_vaccine'))
    except Exception: print('\n'.join(traceback.format_exc().splitlines()[-2:]))
    sim.plot(); pl.savefig('s8_ode.png')
    return sim
sim_ode = attempt('ODE run', ode)

def abm():
    sim = build(solver='abm', n_agents=2000)
    sim.run()
    sim.plot(); pl.savefig('s8_abm.png')
    return sim
sim_abm = attempt('ABM run', abm)

def compare():
    a, b = sim_ode.results.seir, sim_abm.results.seir
    print('peak I ode', a.n_I.values.max(), 'abm', b.n_I.values.max())
    print('deaths ode', a.n_D.values[-1], 'abm', b.n_D.values[-1])
    print('over-65 vaccinated (abm):', sim_abm.results.vx.keys())
    ss.diff_sims(sim_ode, sim_abm) if hasattr(ss, 'diff_sims') else None
attempt('compare', compare)

def overlay_both():
    fig = sim_ode.plot('seir.cases')
    pl.savefig('s8_cases_ode.png')
attempt('plot one result', overlay_both)

def split_check():
    seir = make()
    seir.stratify(vaccinated=[False, True], adjust={'infection': [1.0, 0.2]})
    sim = ss.Sim(seir, start='2020-01-01', stop='2020-02-01', total_pop=1e6, verbose=0).run()
    print('n_S at t=0 by vaccinated:', sim.results.seir.n_S.by('vaccinated').iloc[0].to_dict())
attempt('V3 initial split of a vaccinated stratification', split_check)
