"""
Measles, as an SEIR model.

Adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import starsim as ss


class Measles(ss.SEIR):
    """
    Measles, as an SEIR model.

    Configures `ss.SEIR` with measles natural-history parameters: exposed agents
    become infectious after `dur_exp`, then either die (with probability
    `p_death`) or recover after `dur_inf`. Natural history parameters are from
    the US CDC.

    Args:
        beta (prob):        per-contact transmission probability
        init_prev (Dist):   initial prevalence
        dur_exp (Dist):     duration from exposure to infectiousness
        dur_inf (Dist):     duration of infectiousness
        p_death (Dist):     probability of death among infected agents

    Attributes:
        exposed (BoolState):    infected but not yet infectious
        ti_exposed (FloatArr):  timestep of exposure

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

        return
