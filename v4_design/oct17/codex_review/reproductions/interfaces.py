import starsim as ss
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
kw=dict(verbose=0, unit='day', total_pop=100, stop=10)
def disease(beta=0.0, prev=1.0):
    return ss.Disease(infection=ss.transmission('S -> I', beta=beta), recovery=ss.progression('I -> R',dur=1000000), init_prev=prev,name='sir')
for solver in ['abm','ode']:
    screening=ss.screen(ss.Test(sensitivity=1.0,specificity=1.0,delay=ss.days(3)),interval=1,coverage=1.0,name='screening')
    sim=ss.Sim(disease(),interventions=screening,solver=solver,n_agents=100,**kw).run()
    print('SCREEN',solver,'delivered',sim.results.screening.delivered.values.tolist(),'diagnosed',sim.results.screening.diagnosed.values.tolist(),flush=True)
r=ss.Result(name='infection',values=np.arange(14.0),timevec=pd.date_range('2020-01-01',periods=14),summarize_by='sum')
r.strata=np.tile(np.arange(14.0)[:,None],(1,2)); r.denom=np.ones((14,2))
out=r.resample('week')
print('RESAMPLE values',out.values.shape,'strata',out.strata.shape,'denom',out.denom.shape,flush=True)
sim=ss.Sim(disease(prev=.1),data=pd.DataFrame({'t':[1,2],'n_I':[10,10]}),**kw)
sim.init(); spec=sim.to_spec(); restored=ss.from_spec(spec)
print('SPEC data',spec.get('data'),'RESTORED data',restored.data,flush=True)
spec['schema']='lingua-franca/99.0'
try:
    ss.from_spec(spec)
    print('SCHEMA accepted 99.0',flush=True)
except Exception as e: print('SCHEMA',type(e).__name__,str(e),flush=True)
try:
    fit=ss.Sim(disease(beta=ss.par(.1,bounds=[.01,.5]),prev=.1),data=pd.DataFrame({'t':[1,2],'n_I':[20,30]}),**kw).calibrate(method='nelder-mead',maxiter=1,scan=0,verbose=False)
    print('FIT',vars(fit).keys(),'trials',len(fit.trials),flush=True)
except Exception as e: print('FIT ERROR',type(e).__name__,str(e),flush=True)
try:
    fit=ss.Sim(disease(prev=ss.par(.1,bounds=[.01,.5])),data=pd.DataFrame({'t':[1,2],'n_I':[20,30]}),**kw).calibrate(method='nelder-mead',maxiter=1,scan=0,verbose=False)
    print('INIT FIT',fit.pars,flush=True)
except Exception as e: print('INIT FIT ERROR',type(e).__name__,str(e),flush=True)
