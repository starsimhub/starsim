""" Test minimally different worlds """

import sciris as sc
import numpy as np
import starsim as ss
import pylab as pl


class CountInf(ss.Intervention):
    """ Store every infection state in a timepoints x people array """
    def initialize(self, sim):
        n_agents = len(sim.people)
        self.arr = np.zeros((sim.npts, n_agents))
        self.n_agents = n_agents
        return
    
    def apply(self, sim):
        self.arr[sim.ti, :] = np.array(sim.diseases.sir.infected)[:self.n_agents]
        return


class OneMore(ss.Intervention):
    """ Add one additional agent and infection """
    def apply(self, sim):
        if sim.ti == 0:
            # Create an extra agent
            preg = ss.Pregnancy(rel_fertility=0) # Ensure no default births
            preg.initialize(sim)
            new_uids = np.array([len(sim.people)]) # Hack since make_embryos doesn't return UIDs
            preg.make_embryos(sim, np.array([0])) # Assign 0th agent to be the "mother"
            assert len(new_uids) == 1
            sim.people.age[new_uids] = -100 # Set to a very low number to never reach debut age
            
            # Infect that agent
            sir = sim.diseases.sir
            sir.set_prognoses(sim, new_uids)
            sir.ti_recovered[new_uids] = sim.ti + 1 # Reset recovery time to next timestep
            
            # Reset the random states
            p = sir.pars
            for dist in [p.dur_inf, p.p_death]:
                dist.jump(sim.ti+1)

        return


def plot_infs(s1, s2):
    """ Compare infection arrays from two sims """
    a1 = s1.interventions.countinf.arr
    a2 = s2.interventions.countinf.arr
    
    fig = pl.figure()
    pl.subplot(1,3,1)
    pl.pcolormesh(a1.T)
    pl.xlabel('Timestep')
    pl.ylabel('Person')
    pl.title('Baseline')
    
    pl.subplot(1,3,2)
    pl.pcolormesh(a2.T)
    pl.title('OneMore')
    
    pl.subplot(1,3,3)
    pl.pcolormesh(a2.T - a1.T)
    pl.title('Difference')
    
    sc.figlayout()
    return fig


def test_worlds(do_plot=False):
    
    res = sc.objdict()
    
    pars = dict(
        start = 2000,
        end = 2100,
        n_agents = 200,
        verbose = 0.05,
        diseases = dict(
            type = 'sir',
            init_prev = 0.1,
            beta = 1.0,
            dur_inf = 20,
            p_death = 0, # Here since analyzer can't handle variable numbers of people
        ),
        networks = dict(
            type = 'embedding',
            duration = 5, # Must be shorter than dur_inf for SIR transmission to occur
        ),
    )
    s1 = ss.Sim(pars=pars, interventions=CountInf())
    s2 = ss.Sim(pars=pars, interventions=[CountInf(), OneMore()])
    
    s1.run()
    s2.run()
    
    sum1 = s1.summarize()
    sum2 = s2.summarize()
    res.sum1 = sum1
    res.sum2 = sum2
    
    if do_plot:
        s1.plot()
        plot_infs(s1, s2)
        pl.show()
    
    assert len(s2.people) == len(s1.people) + 1
    assert sum2.sir_cum_infections == sum1.sir_cum_infections + 1
    assert (s1.interventions.countinf.arr != s2.interventions.countinf.arr).sum() == 0
        
    return res
    

if __name__ == '__main__':
    T = sc.timer()
    do_plot = True
    
    res = test_worlds(do_plot=do_plot)
    
    T.toc()