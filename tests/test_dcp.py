import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

sc.options.set(interactive=False) # Assume not running interactively

def test_dcp():
    s = ss.Sim(pars=dict(diseases='sir', networks='embedding'))
    s.initialize()
    s2 = sc.dcp(s)
    s.run()
    s2.run()
    s.plot()
    s2.plot()

    for k in s.results.keys():
        assert np.allclose(s.results[k], s2.results[k])

    return s, s2

if __name__ == '__main__':
    s1 = test_dcp()
    plt.show()