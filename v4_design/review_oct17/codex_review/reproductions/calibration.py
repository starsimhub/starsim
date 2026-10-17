import starsim as ss
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
def make():
    sir=ss.Disease(infection=ss.transmission('S -> I',beta=ss.par(.1,bounds=[.01,.5])),recovery=ss.progression('I -> R',dur=10),init_prev=.1,name='sir')
    return ss.Sim(sir,solver='abm',n_agents=100,total_pop=100,unit='day',stop=10,verbose=0,rand_seed=9,data=pd.DataFrame({'t':[1,2,3,4,5],'n_I':[20,25,30,35,40]}))
f=make().calibrate(method='nelder-mead',maxiter=1,scan=0,n_runs=3,verbose=False)
print('RUNS fit mismatch',f.mismatch,'recorded evaluated mismatch range',f.trials.mismatch.tolist(),'best pars',f.pars)
sir=ss.Disease(infection=ss.transmission('S -> I',beta=.1),recovery=ss.progression('I -> R',dur=10),init_prev=.1,name='sir')
try:
    sim=ss.Sim(sir,networks=ss.RandomNet(n_contacts=ss.par(3,bounds=[1,6])),solver='abm',n_agents=100,total_pop=100,unit='day',stop=10,verbose=0,data=pd.DataFrame({'t':[1,2],'n_I':[20,30]}))
    sim.calibrate(method='nelder-mead',maxiter=1,scan=0,verbose=False)
except Exception as e: print('NETWORK bounded',type(e).__name__,str(e))
