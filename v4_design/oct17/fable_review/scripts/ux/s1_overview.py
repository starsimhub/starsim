import traceback, sys, time
import numpy as np, pandas as pd
import starsim as ss

# synthetic weekly cases
dates = pd.date_range('2020-01-01', '2021-01-01', freq='7D')
t = np.arange(len(dates))
cases = (50e6*0.5*0.01*np.exp(-((t-20)/8)**2)*0.1 + 200).round()
pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'), cases=cases)).to_csv('weekly_cases.csv', index=False)

def step(label, fn):
    print(f'\n######## {label}')
    t0 = time.time()
    try:
        out = fn()
        print(f'[ok {time.time()-t0:.1f}s]')
        return out
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        print('\n'.join(tb[-8:]))
        print(f'[FAIL {time.time()-t0:.1f}s]')
        return None

G = {}
def s1():
    G['beta'] = ss.par(0.05, bounds=[0.01, 0.5])
    G['dur']  = ss.days(10)
    G['S'], G['I'], G['R'] = ss.states(3)
    print(G['beta'], G['dur'], G['S'], G['I'], G['R'])
step('ingredients', s1)

def s2():
    S, I, R, beta, dur = G['S'], G['I'], G['R'], G['beta'], G['dur']
    infection = S >> I @ beta * I
    recovery  = I >> R @ dur
    cases     = ss.outcome(0.5 * infection, delay=ss.days(5), every=ss.weeks(1))
    sir = ss.Disease(infection, recovery, cases, init_prev=0.01)
    print(infection); print(recovery); print(cases); print(sir)
    G['sir'] = sir
step('disease', s2)

def s3():
    kw  = dict(stratify=dict(age=[0, 15, 65]), networks=ss.contacts('Kenya'), total_pop=50e6, start='2020-01-01', stop='2021-01-01')
    G['kw'] = kw
    sim = ss.Sim(G['sir'], data='weekly_cases.csv', **kw)
    G['sim'] = sim
step('sim', s3)

def s4():
    G['sim'].describe()
step('describe', s4)

def s5():
    G['sim'].run()
step('run', s5)

def s6():
    G['sim'].plot()
    import pylab as pl; pl.savefig('s1_plot.png')
step('plot', s6)

def s7():
    fit = G['sim'].calibrate()
    G['fit'] = fit
    print(fit)
step('calibrate', s7)

def s8():
    sir, kw, fit = G['sir'], G['kw'], G['fit']
    masks = ss.modify(sir.pars.beta, multiply=0.6, start='2020-03-15')
    sims  = ss.parallel(
        baseline = fit.sim,
        masks    = ss.Sim(sir, pars=fit.pars, interventions=masks, **kw),
    )
    sims.plot()
    import pylab as pl; pl.savefig('s1_scen.png')
step('scenarios', s8)

def s9():
    msim = ss.parallel(ss.Sim(G['sir'], solver='stochastic', **G['kw']), n_runs=3)
    msim.plot()
    import pylab as pl; pl.savefig('s1_msim.png')
step('msim (n_runs=3)', s9)
