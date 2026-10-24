import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/starsim-review-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/starsim-review-numba')
import numpy as np
import networkx as nx
import starsim as ss

def make():
    return ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')

def attempt(label, fn):
    try:
        print(label, fn())
    except Exception as e:
        print(label, type(e).__name__, str(e).splitlines()[0])

attempt('density', lambda: ss.MixingPool(dependence='density'))
attempt('sexual ODE', lambda: ss.Sim(make(), networks=ss.MFNet(n_contacts=.7), solver='ode', unit='day', stop=1, verbose=0).init().solver)
for graph in [nx.empty_graph(100), nx.complete_graph(100)]:
    attempt(f'static ODE edges={len(graph.edges)}', lambda: ss.Sim(make(), networks=ss.StaticNet(graph=graph), solver='ode', total_pop=100, unit='day', stop=1, verbose=0).init().mixing.terms)

def matrix(split):
    dis=make().stratify(age=[0, 20], split=split)
    sim=ss.Sim(dis, networks=ss.contacts([[10,2],[3,4]], age=[0,10]), unit='day', stop=1, verbose=0).init()
    return sim.mixing.terms
attempt('age aggregation, population 90% young', lambda: matrix([.9, .1]))
attempt('age aggregation, population 10% young', lambda: matrix([.1, .9]))
attempt('negative contact matrix', lambda: ss.contacts([[-1,0],[0,1]], age=[0,20]).pars.n_contacts)
attempt('unordered contact ages', lambda: ss.contacts([[1,0],[0,1]], age=[20,0]).pars.src)
