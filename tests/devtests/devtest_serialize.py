"""
Test simple APIs
"""

# %% Imports and settings
import starsim as ss
import scipy.stats as sps
import sciris as sc
import multiprocessing as mp

def test_saveobj():
    d = sps.bernoulli(p=0.6)
    sc.saveobj('dist.obj', d, method='pickle')

    x = sc.loadobj('dist.obj')
    r = x.rvs(size=10)
    print(r)
    return r

def test_saveobjSD():
    d = ss.ScipyDistribution(sps.bernoulli(p=0.2))
    sc.saveobj('distSD.obj', d, method='pickle')

    x = sc.loadobj('distSD.obj')
    r = x.rvs(size=10)
    print(r)
    return r

def run(x=0.5):
    dist = ss.ScipyDistribution(ss.bernoulli(p=x))
    return dist

def test_cliff1():
    o = run()
    return 0

def test_cliff2():
    vals = [0.5, 0.7]
    # Doesn't work
    with mp.Pool(2) as pool:
        out = pool.map(run, vals)
    return out

#obj = (0,0,(True, [run(x=0.6)]))
#<class 'multiprocessing.reduction.ForkingPickler'>
#_ForkingPickler.dumps(obj)

if __name__ == '__main__':
    test_saveobj()
    test_saveobjSD()
    test_cliff1()
    test_cliff2()
