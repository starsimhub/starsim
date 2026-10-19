import os; os.environ['MPLBACKEND']='agg'
import numpy as np, pandas as pd, sciris as sc, starsim as ss, sys
import warnings

def make(beta=0.3, solver='auto', n_agents=None, bounds=True, **kw):
    b = ss.par(beta, bounds=[0.05, 1.0] if bounds else None, name='beta')
    S, I, R = ss.states(3)
    infection = S >> I @ b * I
    recovery = I >> R @ ss.days(10)
    cases = ss.outcome(0.5 * infection, delay=ss.days(3), every=ss.weeks(1))
    sir = ss.Disease(infection, recovery, cases, init_prev=0.01, name='sir')
    args = dict(total_pop=10_000, start='2020-01-01', stop='2020-05-01', unit='day', solver=solver, verbose=0)
    if n_agents: args['n_agents'] = n_agents
    return ss.Sim(sir, **sc.mergedicts(args, kw))

truth = make(0.3).run()
res = truth.results.sir
df = sc.dataframe(date=truth.t.timevec[7::7], cases=res.cases.values[7::7])
print('data rows', len(df), 'cases mean', df.cases.mean())

section = sys.argv[1] if len(sys.argv) > 1 else 'ode'

if section == 'ode':
    print('\n=== ODE default ===')
    T = sc.timer()
    sim = make(0.15, data=df)
    fit = sim.calibrate(verbose=True)
    T.toc('ODE default calibrate')
    fit.summarize()
    print('method:', fit.method, 'evals:', len(fit.trials), 'pars:', fit.pars, type(fit.pars.beta))
    # Round trip
    T = sc.timer()
    refit = ss.Sim(make(0.15).inputs if False else None) if False else None
    new = make(0.15, pars=fit.pars, data=df).run()
    print('round trip max diff n_I:', np.abs(new.results.sir.n_I.values - fit.sim.results.sir.n_I.values).max())
    print('fit.sim beta:', fit.sim.diseases.sir.pars.beta, '| original beta unchanged:', sim.new().init().diseases.sir.pars.beta)

    print('\n=== ODE nelder-mead ===')
    T = sc.timer()
    fit2 = make(0.15, data=df).calibrate(method='nelder-mead', verbose=True)
    T.toc('ODE nelder-mead calibrate')
    print('evals:', len(fit2.trials), 'best', fit2.pars)

    print('\n=== gof="foo" (unknown name) ===')
    try:
        fit3 = make(0.15, data=df).calibrate(gof='foo', method='nelder-mead')
        print('no error! mismatch', fit3.mismatch, 'equals nae?', np.isclose(fit3.mismatch, ss.mismatch(fit3.sim)))
    except Exception as e:
        print('error:', type(e).__name__, e)

    print('\n=== method=list (staged) ===')
    try:
        make(0.15, data=df).calibrate(method=['nelder-mead', 'optuna'])
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:200])

    print('\n=== pars=[beta*contacts] product ===')
    try:
        b = ss.par(0.3, bounds=[0.05, 1], name='beta'); c = ss.par(4, bounds=[1, 10], name='c')
        make(0.15, data=df).calibrate(pars=[b*c])
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:200])

elif section == 'stoch':
    for solver in ['sde', 'binomial']:
        print(f'\n=== {solver} default, no likelihood ===')
        T = sc.timer()
        fit = make(0.15, data=df, solver=solver).calibrate(verbose=True, trials=30)
        T.toc(f'{solver} calibrate (30 trials)')
        fit.summarize()
        print('method', fit.method, 'evals', len(fit.trials))
    print('\n=== sde, with likelihood: what is chosen (do not run) ===')
    sim = make(0.15, data=df, solver='sde').init()
    print(ss.calibration.default_method(sim, use_lik=True))
    print(ss.calibration.default_method(sim, use_lik=False))
    sim = make(0.15, data=df, solver='ode').init()
    print('ode', ss.calibration.default_method(sim, use_lik=True), ss.calibration.default_method(sim, use_lik=False))
    # Does n_runs do anything?
    print('\n=== sde n_runs=3 ===')
    T = sc.timer()
    fit = make(0.15, data=df, solver='sde').calibrate(verbose=True, trials=10, n_runs=3)
    T.toc('sde 10 trials n_runs=3')
    print('evals', len(fit.trials))

elif section == 'abm':
    print('\n=== ABM (2000 agents) default ===')
    T = sc.timer()
    sim = make(0.15, data=df, solver='abm', n_agents=2000)
    t1 = sc.timer(); sim.new().run(); t1.toc('one ABM run')
    fit = sim.calibrate(verbose=True, trials=30)
    T.toc('ABM calibrate (30 trials)')
    fit.summarize()
    print('method', fit.method, 'evals', len(fit.trials), 'pars', fit.pars)
    print('\n=== ABM auto (networks given) ===')
    sim = make(0.15, data=df, networks=ss.RandomNet(n_contacts=4), n_agents=2000)
    sim.init(); print('solver chosen:', sim.solver)

elif section == 's3':
    print('\n=== Starsim 3 sim .calibrate() ===')
    sir = ss.SIR(beta=ss.perday(0.075), init_prev=ss.bernoulli(0.02))
    sim = ss.Sim(n_agents=1000, start='2020-01-01', stop='2020-02-12', dt=ss.days(1), diseases=sir, networks=ss.RandomNet(n_contacts=ss.poisson(4)), verbose=0)
    try:
        sim.calibrate()
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:300])
    print('\n=== Starsim 3 sim with data= ===')
    try:
        sim = ss.Sim(n_agents=1000, start='2020-01-01', stop='2020-02-12', dt=ss.days(1), diseases=ss.SIR(), networks=ss.RandomNet(), data=df, verbose=0)
        sim.calibrate()
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:300])
    print('\n=== v4 declared sim through ss.Calibration ===')
    try:
        sim = make(0.15)
        calib = ss.Calibration(sim, calib_pars=dict(beta=dict(low=0.05, high=1, guess=0.15)), total_trials=4, n_workers=1)
        calib.calibrate()
    except Exception as e:
        print('error:', type(e).__name__, str(e)[:300])
