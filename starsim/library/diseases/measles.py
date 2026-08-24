"""
Measles, as an SEIR model.

Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import starsim as ss


class Measles(ss.SIR):
    """
    Measles, as an SEIR model.

    Extends `ss.SIR` with an exposed (latent, non-infectious) state. Exposed
    agents become infectious after `dur_exp`, then either die (with probability
    `p_death`) or recover after `dur_inf`. Natural history parameters are from
    the US CDC.

    Args:
        beta (prob):        per-contact transmission probability
        init_prev (Dist):   initial prevalence
        dur_exp (Dist):     duration from exposure to infectiousness
        dur_inf (Dist):     duration of infectiousness
        p_death (Dist):     probability of death among infected agents

    Note that `infected` covers both the exposed and infectious compartments, so
    `n_infected` is E+I, while only `infectious` agents transmit; `exposed` is derived
    as `infected & ~infectious`. `ti_infected` is the time of infection; `ti_infectious`
    is the time of becoming infectious.

    Attributes:
        infectious (BoolState):     infectious
        exposed (derived):          infected but not yet infectious
        ti_infectious (FloatArr):   timestep of becoming infectious

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Measles(), networks='random')
        sim.run()
        sim.plot()
        ```
    """
    def __init__(self, pars=None, **kwargs):
        """ Initialize with parameters """
        super().__init__()
        self.define_pars(
            # Initial conditions and beta
            beta = 1.0, # Placeholder value
            init_prev = ss.bernoulli(p=0.005),

            # Natural history parameters, all specified in days
            dur_exp = ss.normal(loc=ss.days(8)),        # (days) - source: US CDC
            dur_inf = ss.normal(loc=ss.days(11)),       # (days) - source: US CDC
            p_death = ss.bernoulli(p=0.005), # Probability of death
        )
        self.update_pars(pars, **kwargs)

        # SIR states are added automatically; here we split the infected period into E and I
        self.define_states(
            ss.BoolState('infectious', label='Infectious'),
            ss.FloatArr('ti_infectious', label='Time of becoming infectious'),
            exposed = lambda self: self.infected & ~self.infectious, # Infected, but not yet infectious
            ti_exposed = 'ti_infected', # Exposure and infection are the same event
        )
        return

    def step_state(self):
        # Progress exposed -> infectious
        becoming = (self.exposed & (self.ti_infectious <= self.ti)).uids
        self.infectious[becoming] = True

        # Progress infectious -> recovered, and trigger deaths
        super().step_state()
        self.infectious[~self.infected] = False # Recovering or dying also ends infectiousness
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(uids, sources) # Infects the agents and schedules recovery/death

        # Delay the onset of infectiousness, and push recovery and death back to match
        dur_exp = self.pars.dur_exp.rvs(uids)
        self.ti_infectious[uids] = self.ti + dur_exp
        self.ti_recovered[uids] += dur_exp
        self.ti_dead[uids] += dur_exp
        return

    def step_die(self, uids):
        # Reset infected/recovered flags for dead agents
        super().step_die(uids)
        self.infectious[uids] = False
        return

