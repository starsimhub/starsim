"""
Example neonatal sepsis disease.

A simple SIR-like disease that infects newborns at birth and kills a fraction
shortly after, producing neonatal deaths that the Pregnancy module passively
detects and classifies.

This demonstrates:
    - How a disease can cause neonatal deaths via ``request_death`` on newborns
    - How the Pregnancy module passively detects deaths of agents aged 0-28 days
      and classifies them as neonatal deaths (no special hookup needed)
    - How to use ``set_prognoses`` to apply lethal outcomes only to neonates

The key insight is that neonatal death detection is passive: any mechanism that
kills an agent aged 0-28 days (disease, congenital outcomes, background
mortality) is automatically classified as a neonatal death by the Pregnancy
module. No registration or callback is needed.

Usage::

    import starsim as ss
    import starsim_examples as sse

    sim = ss.Sim(
        diseases=sse.NeonatalSepsis(),
        demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths()],
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

    # NND results are on the Pregnancy module, not the disease
    print('Neonatal deaths:', sim.results.pregnancy.nnds.sum())
    print('Total births:', sim.results.pregnancy.births.sum())
"""

import starsim as ss

__all__ = ['NeonatalSepsis']


class NeonatalSepsis(ss.SIR):
    """
    Minimal neonatal sepsis model.

    Infects newborns at birth with probability ``init_prev``, then kills a
    fraction (``p_death``) within a short window (``dur_inf``). Useful for
    testing passive neonatal death detection in the Pregnancy module.

    Pars:
        beta (float):       transmission rate — set to 0 since infection is only at birth
        init_prev (Dist):   fraction of newborns infected at birth
        dur_inf (Dist):     time from infection to death (for those who die)
        p_death (Dist):     case fatality rate among infected neonates
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            beta      = 0,                                         # No ongoing transmission — infection is set at birth
            init_prev = ss.bernoulli(p=0.3),                       # 30% of newborns infected at birth
            dur_inf   = ss.lognorm_ex(mean=ss.days(7), std=ss.days(3)),  # ~1 week illness
            p_death   = ss.bernoulli(p=0.5),                       # 50% case fatality
        )
        self.update_pars(pars, **kwargs)
        return

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses — only apply lethal outcomes to neonates (age < 28 days).

        Non-neonates who get infected (via init_prev at sim start) just recover
        normally via the base SIR logic.
        """
        super().set_prognoses(uids, sources)

        # Only neonates can die from this disease
        age_days = ss.years(self.sim.people.age[uids]).days
        neonates = uids[age_days <= 28]
        if len(neonates):
            will_die = self.pars.p_death.filter(neonates)
            dur = self.pars.dur_inf.rvs(will_die)
            self.ti_dead[will_die] = self.ti + dur
        return
