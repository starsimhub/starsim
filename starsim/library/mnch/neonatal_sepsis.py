"""
Example neonatal sepsis disease.

A simple SIR-like disease that infects newborns at birth and kills a fraction
shortly after, producing neonatal deaths that the Pregnancy module passively
detects and classifies.

Neonatal death detection is passive: any death of an agent aged 0-28 days (disease, congenital outcomes, background
mortality) is classified as a neonatal death by the Pregnancy module, with no registration or callback.

Usage::

    import starsim as ss
    import starsim.library as ssl

    sim = ss.Sim(
        diseases=ssl.mnch.NeonatalSepsis(),
        demographics=[ss.Pregnancy(fertility_rate=ss.peryear(0.03)), ss.Deaths()],
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

    # NND results are on the Pregnancy module, not the disease
    print('Neonatal deaths:', sim.results.pregnancy.neonatal_death.sum())
    print('Total births:', sim.results.pregnancy.birth.sum())

Used with: agents; coded modules
"""

import numpy as np
import starsim as ss
from ..diseases.agents import AgentSIR

__all__ = ['NeonatalSepsis']


class NeonatalSepsis(AgentSIR):
    """
    Minimal neonatal sepsis model.

    Infects newborns at birth with probability `p_at_birth`, then kills a
    fraction (`p_death`) within a short window (`dur_inf`). Useful for
    testing passive neonatal death detection in the Pregnancy module.

    Note: CRN safe.

    Args:
        beta (float):       transmission rate — set to 0 since infection is only at birth
        p_at_birth (Dist):  fraction of newborns infected at birth
        init_prev (Dist):   fraction of the people at the start who are infected (none by default)
        dur_inf (Dist):     time from infection to death (for those who die)
        p_death (Dist):     case fatality rate among infected neonates
        pars (dict): the same parameters, as a dict
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta      = 0,                                         # No ongoing transmission — infection is set at birth
            p_at_birth = ss.bernoulli(p=0.3),                      # 30% of newborns infected at birth
            init_prev = ss.bernoulli(p=0),                         # Nobody at the start: infection is at birth
            dur_inf   = ss.lognormal(mean=ss.days(7), std=ss.days(3)),  # ~1 week illness
            p_death   = ss.bernoulli(p=0.5),                       # 50% case fatality
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.BoolArr('screened', label='Screened for sepsis at birth'),
        )
        return

    def init_post(self):
        """ Exempt the initial population from being treated as newborns """
        out = super().init_post()
        self.screened[self.sim.people.auids] = True
        return out

    def step(self):
        """ Infect each newborn with `p_at_birth`, screening each agent once, the first step after they are born """
        out = super().step()
        newborns = (~self.screened & (self.sim.people.age >= 0)).uids
        if len(newborns):
            self.screened[newborns] = True
            self.infect(self.pars.p_at_birth.filter(newborns)) # With no source (-1)
        return out

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses: only neonates (aged up to 28 days) can die, and others (e.g. infected at the start) recover

        Args:
            uids (array): the agents infected
            sources (array): the agents who infected them
        """
        super().set_prognoses(uids, sources) # Schedules death for a fraction p_death, and recovery for everyone else

        age_days = ss.years(self.sim.people.age[uids]).days
        non_neonates = uids[age_days > 28]
        self.ti_recovered[non_neonates] = np.fmin(self.ti_recovered[non_neonates], self.ti_dead[non_neonates])
        self.ti_dead[non_neonates] = np.nan
        return
