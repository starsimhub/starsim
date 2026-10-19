import traceback, starsim as ss, numpy as np, pandas as pd, pylab as pl
def attempt(label, fn):
    print(f'\n######## {label}')
    try: out = fn(); print('[ok]'); return out
    except Exception: print('\n'.join(traceback.format_exc().splitlines()[-3:])); print('[FAIL]')

def decl():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')

def c1():
    sim = ss.Sim(decl(), networks=ss.RandomNet(n_contacts=10), start='2020-01-01', stop='2020-06-01', n_agents=1000, verbose=0).init()
    print('declared disease + RandomNet -> solver', sim.solver)
attempt('declared disease with a v3 network', c1)

def c2():
    vx = ss.routine_vx(product=ss.Vx(efficacy=0.8), prob=0.8, start_year=2020, eligibility=lambda sim: sim.people.age > 65)
    sim = ss.Sim(decl(), interventions=vx, start='2020-01-01', stop='2020-06-01', n_agents=1000, verbose=0).run()
    print('declared disease + routine_vx -> solver', sim.solver, list(sim.results.keys()))
attempt('declared disease with v3 routine_vx', c2)

def c3():
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.8), coverage=0.8, years=[2020], eligible=lambda sim: sim.people.age > 65)
    sim = ss.Sim(diseases=ss.SIR(beta=ss.perday(0.05), init_prev=0.01), interventions=vx, networks=ss.RandomNet(), start='2020-01-01', stop='2020-06-01', dt=ss.days(1), n_agents=1000, verbose=0).run()
    print('v3 SIR + v4 vaccinate ->', sim.solver, list(sim.results.keys()))
attempt('v3 disease with v4 vaccinate', c3)

def c4():
    sir = decl()
    sim = ss.Sim(sir, interventions=ss.change(sir.pars.beta, at=['2020-03-01'], by=[0.5]), start='2020-01-01', stop='2020-06-01', total_pop=1e5, verbose=0).run()
    print('ss.change ok, n_I max', sim.results.sir.n_I.values.max())
    sir = decl()
    sim = ss.Sim(sir, interventions=ss.modify(sir.pars.beta, multiply=0.5, start='2020-03-01'), start='2020-01-01', stop='2020-06-01', total_pop=1e5, verbose=0).run()
    print('ss.modify ok, n_I max', sim.results.sir.n_I.values.max())
attempt('ss.change vs ss.modify', c4)

def c5():
    sir = decl()
    sim = ss.Sim(sir, start='2020-01-01', stop='2020-06-01', total_pop=1e5, verbose=0).run()
    base = sim.new()
    msim = ss.parallel(baseline=base, half=sim.new(pars=dict(beta=0.025)), verbose=0)
    msim.plot(); pl.savefig('s11_scen.png')
    print('scenarios via sim.new() ok;', msim)
attempt('fix for the overview: sim.new()', c5)

def c6():
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ 0.1)
attempt('E210 unnamed state? (states bound, transitions inline)', c6)

def c7():
    states = ss.states(3)
    sir = ss.Disease(states[0] >> states[1] @ 0.3*states[1], states[1] >> states[2] @ 0.1)
    ss.Sim(sir, stop=10, verbose=0).init()
attempt('E210 unnamed state (list of states)', c7)

def c8():
    pd.DataFrame(dict(date=['2020-02-01','2020-03-01'], casez=[1,2])).to_csv('bad.csv', index=False)
    ss.Sim(decl(), data='bad.csv', start='2020-01-01', stop='2020-06-01', total_pop=1e5, verbose=0).init()
attempt('E705 unknown data column', c8)

def c9():
    sir = decl()
    sim = ss.Sim(sir, start='2020-01-01', stop='2020-06-01', total_pop=1e5)
    print('--- fresh run() prints:'); sim.run()
    print('--- run() again prints:')
    try: sim.run()
    except Exception as e: print(type(e).__name__, e)
attempt('what run() prints', c9)

def c10():
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=ss.par(0.05, bounds=[0.01, 0.5])), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    pd.DataFrame(dict(date=pd.date_range('2020-01-08','2020-05-01',freq='7D').strftime('%Y-%m-%d'), new_infections=np.linspace(100,1000,17).round())).to_csv('fitdata.csv', index=False)
    sim = ss.Sim(sir, data='fitdata.csv', start='2020-01-01', stop='2020-06-01', total_pop=1e5, verbose=0)
    fit = sim.calibrate()
    print('--- fit.summarize():'); fit.summarize()
    print('--- fit.sim is complete:', fit.sim.complete, '| original complete:', sim.complete)
    fit.plot(); pl.savefig('s11_fit.png')
attempt('calibrate + fit.summarize', c10)
