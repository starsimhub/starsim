import sys, time
t0 = time.time()
import starsim as ss
t1 = time.time()
print(f'import starsim: {t1-t0:.2f} s')
for mod in ['jax', 'blackjax', 'optuna', 'seaborn', 'scipy.optimize']:
    print(f'{mod:16s} in sys.modules after import starsim: {mod in sys.modules}')
# Exposed names
names = [n for n in dir(ss) if n in ['Calibration','CalibComponent','BetaBinomial','Binomial','DirichletMultinomial','GammaPoisson','Normal','linear_interp','linear_accum','step_containing','data','compare','mismatch','loglik','ParticleFilter','Fit','fit','ModelComparison','compare_models']]
print('calibration names at top level:', names)
