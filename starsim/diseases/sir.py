"""
Define example disease modules
"""

import pylab as pl
import starsim as ss

__all__ = ['SIR']

class SIR(ss.Infection):
    """
    Example SIR model

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omergeleft(pars,
            dur_inf = 6,
            init_prev = 0.01,
            p_death = 0.01,
            beta = 0.5,
        )

        par_dists = ss.omergeleft(par_dists,
            dur_inf   = ss.lognorm_o,
            init_prev = ss.bernoulli,
            p_death   = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            ss.State('recovered', bool, False),
            ss.State('ti_recovered', int, ss.INT_NAN),
            ss.State('ti_dead', int, ss.INT_NAN),
        )
        return

    def update_pre(self, sim):
        # Progress infectious -> recovered
        recovered = ss.true(self.infected & (self.ti_recovered <= sim.ti))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.ti_dead <= sim.year)
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses """
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = sim.ti + dur_inf[will_die] / sim.dt # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = sim.ti + dur_inf[~will_die] / sim.dt

        return

    def update_death(self, sim, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return

    def plot(self):
        """ Default plot for SIR model """
        fig = pl.figure()
        for rkey in ['susceptible', 'infected', 'recovered']:
            pl.plot(self.results['n_'+rkey], label=rkey.title())
        pl.legend()
        return fig
    

# %% Interventions
__all__ += ['sir_vaccine']


class sir_vaccine(ss.Vx):
    """
    Create a vaccine product that changes susceptible people to recovered (i.e., perfect immunity)
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        pars = ss.omerge({
            'efficacy': 0.9,
        }, pars)

        super().__init__(pars=pars, *args, **kwargs)

        return

    def administer(self, people, uids):
        people.sir.rel_sus[uids] *= 1-self.pars.efficacy
        return
