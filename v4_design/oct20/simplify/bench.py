"""
Baseline benchmarks for the simplification plan (simplify_plan.md, S0.3 and S8.1)

Each workload runs in its own process, so "cold" is its first run in a fresh
process (including Numba compilation, from the on-disk cache where there is one)
and "warm" is the best of three runs after that.

    /software/conda/envs/v4/bin/python bench.py           # All workloads
    /software/conda/envs/v4/bin/python bench.py ode abm   # Some of them
"""
import os
import sys
import subprocess
import time

workloads = {}

def workload(func):
    workloads[func.__name__] = func
    return func


@workload
def ode():
    """ A declared SIR as an ODE, a year in days """
    import starsim as ss
    def make():
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.days(10), init=dict(I=0.001), name='sir')
        return ss.Sim(sir, unit='day', start=0, stop=365, solver='ode', total_pop=1e6, verbose=0)
    return make


@workload
def abm():
    """ A coded SIR as agents over a random network, 20k agents, 20 years in months """
    import starsim as ss
    return lambda: ss.Sim(diseases='sir', networks='random', n_agents=20_000, start=2000, stop=2020, dt=ss.months(1), verbose=0)


@workload
def multi():
    """ Two coded diseases as agents with demographics, 10k agents, 30 years in months """
    import starsim as ss
    return lambda: ss.Sim(diseases=['sis', 'sir'], networks='random', demographics=True, n_agents=10_000, start=2000, stop=2030, dt=ss.months(1), verbose=0)


@workload
def spatial():
    """ A declared SIR over 50 places with commuting, binomial, a year in days """
    import numpy as np
    import starsim as ss
    n = 50
    rng = np.random.default_rng(1)
    names = [f'p{i}' for i in range(n)]
    pop = rng.integers(10_000, 100_000, n)
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    def make():
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.days(10), init_n=dict(I=ss.by(patch=[100] + [0]*(n-1))), name='sir')
        sir.stratify(patch=ss.geo.location(names=names, pop=pop, x=x, y=y), mobility=ss.geo.Mobility(ss.geo.gravity(), away=0.05))
        return ss.Sim(sir, unit='day', start=0, stop=365, solver='binomial', total_pop=int(pop.sum()), verbose=0, rand_seed=1)
    return make


@workload
def ensemble():
    """ 100 binomial runs of a declared SIR, a year in days """
    import starsim as ss
    def make():
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.days(10), init=dict(I=0.001), name='sir')
        sim = ss.Sim(sir, unit='day', start=0, stop=365, solver='binomial', total_pop=1e5, verbose=0)
        return ss.MultiSim(sim, n_runs=100)
    return make


@workload
def calib():
    """ Fit beta of a binomial SIR to Poisson cases with Optuna, 20 trials """
    import numpy as np
    import pandas as pd
    import starsim as ss
    def sim(beta=0.5, data=None):
        beta = ss.par(beta, bounds=[0.1, 1], name='beta')
        S, I, R = ss.states(3)
        infection = S >> I @ beta * I
        cases = ss.outcome(0.5 * infection, likelihood=ss.poisson())
        sir = ss.Disease(infection, I >> R @ ss.days(6), cases, init=dict(I=0.002), name='sir')
        return ss.Sim(sir, unit='day', start='2020-01-01', stop='2020-04-01', solver='binomial', total_pop=10_000, data=data, verbose=0, rand_seed=3)
    truth = sim().run().results.sir
    data = pd.DataFrame(dict(cases=np.random.default_rng(1).poisson(truth.cases.values[1::7])), index=truth.timevec[1::7])
    class Fit:
        def run(self):
            sim(beta=0.3, data=data).calibrate(method='optuna', trials=20, rand_seed=1)
    return Fit


def tests_model(name, crn=True):
    """ A model from tests/benchmark_models.py (B0 of the October 22 plan: kinds of model the ones above miss) """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'tests'))
    import starsim as ss
    import benchmark_models
    ss.options(crn=crn)
    return getattr(benchmark_models, f'make_{name}')


@workload
def fseis10():
    """ malariasim's fseis10, ported: coded SEIS humans as agents and coded ODE mosquitoes, 10k, 2 years in days, crn=False """
    return tests_model('fseis', crn=False)


@workload
def counts():
    """ A declared malaria model with mosquitoes as counts, humans as 10k agents, a year in days """
    return tests_model('counts')


@workload
def coded():
    """ A disease written as code (ssl.AgentSIS) on a random network with demographics, 20k agents, 20 years in months """
    return tests_model('coded')


@workload
def by_age():
    """ A declared Covasim-style model with ss.by age tables, and deaths by age and sex, 20k agents, 120 days """
    return tests_model('by_age')


def run_one(name):
    """ Time one workload in this process: cold, then the best of three warm runs """
    t0 = time.perf_counter()
    import starsim # noqa: F401
    t_import = time.perf_counter() - t0
    make = workloads[name]()
    times = []
    for i in range(4):
        t0 = time.perf_counter()
        make().run()
        times.append(time.perf_counter() - t0)
    print(f'{name:10s} import {t_import:5.2f} s  cold {times[0]:6.2f} s  warm {min(times[1:]):6.3f} s')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--one':
        run_one(sys.argv[2])
    else:
        names = sys.argv[1:] or list(workloads)
        env = dict(os.environ, OMP_NUM_THREADS='1', NUMBA_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', SCIRIS_BACKEND='agg')
        for name in names:
            subprocess.run([sys.executable, __file__, '--one', name], env=env, check=False)
