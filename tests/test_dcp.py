import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

sc.options(interactive=False) # Assume not running interactively

n_agents = 250


def test_dcp():
    s1 = ss.Sim(pars=dict(diseases='sir', networks='embedding'), n_agents=n_agents)
    s1.initialize()
    
    s2 = sc.dcp(s1)

    s1.run()
    s2.run()
    s1.plot()
    s2.plot()
    
    ss.diff_sims(s1, s2, full=True)
    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


def test_dcp_until():
    s1 = ss.Sim(pars=dict(diseases='sir', networks='embedding'), n_agents=n_agents)
    s1.initialize()

    s1.run(until=5)

    s2 = sc.dcp(s1)

    s1.run()
    s2.run()
    s1.plot()
    s2.plot()

    assert np.allclose(s1.summary[:], s2.summary[:], rtol=0, atol=0, equal_nan=True)

    return s1, s2


if __name__ == '__main__':
    sc.options(interactive=True)
    s1 = test_dcp()
    s1 = test_dcp_until()
    plt.show()