"""
Example neonatal sepsis disease.

A simple SIR-like disease that kills a fraction of infected newborns shortly
after birth, producing neonatal deaths that the Pregnancy module passively
detects and classifies.
"""

import numpy as np
import starsim as ss

__all__ = ['NeonatalSepsis']


class NeonatalSepsis(ss.SIR):
    """
    Minimal neonatal sepsis model.

    Infects newborns at birth with probability init_prev, then kills a fraction
    (p_death) within a short window (dur_inf). Useful for testing passive
    neonatal death detection in the Pregnancy module.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta     = 0,                          # No ongoing transmission — infection is set at birth
            init_prev = ss.bernoulli(p=0.3),       # 30% of newborns infected at birth
            dur_inf  = ss.lognorm_ex(mean=ss.days(7), std=ss.days(3)),
            p_death  = ss.bernoulli(p=0.5),        # 50% case fatality
        )
        self.update_pars(pars, **kwargs)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses — only affect newborns (age < 28 days) """
        super().set_prognoses(uids, sources)

        # Only apply lethal prognoses to neonates
        age_days = ss.years(self.sim.people.age[uids]).days
        neonates = uids[age_days <= 28]
        if len(neonates):
            will_die = self.pars.p_death.filter(neonates)
            dur = self.pars.dur_inf.rvs(will_die)
            self.ti_dead[will_die] = self.ti + dur
        return
