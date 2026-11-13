"""
Measles, as an SEIR model.

Originally adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py
Original version by @alina-muellenmeister, @domdelport, and @RomeshA
"""

import starsim as ss

__all__ = ['Measles']


class Measles(ss.Disease):
    """
    Measles, as an SEIR model with deaths

    People infected are latent (`E`) for `dur_exp`, then infectious (`I`) for `dur_inf` (each a normal, with a spread of a
    day or two), after which they recover
    with lifelong immunity (`R`) or, with probability `p_death`, die. It is declared, so it runs under any solver:
    as compartments mixing homogeneously, or as agents on a network (where `beta` is the rate per contact).

    Sources: the latent and infectious periods are from the US CDC (about 8 days from infection to being infectious,
    and infectious for about 11 days, from 4 days before the rash to 4 days after it, rounded up; the standard
    deviations, 2 days and 1.5 days, are illustrative, since the CDC gives ranges: 7 to 21 days from exposure to the rash); `beta` gives an R0
    of about 15 (`beta` times `dur_inf`), within the range of 12 to 18 usually quoted (Guerra FM et al., "The basic
    reproduction number (R0) of measles: a systematic review", *Lancet Infect Dis* 2017;17(12):e420-e428,
    https://doi.org/10.1016/S1473-3099(17)30307-9). The probability of death is illustrative of a setting with good care.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (per contact, if there are networks)
        dur_exp (`ss.Duration`/`ss.Dist`): the time from infection to being infectious (a single duration given is the mean of an exponential)
        dur_inf (`ss.Duration`/`ss.Dist`): the time infectious
        p_death (float): the probability of death, at the end of the infectious period
        kwargs (dict): passed to `ss.Disease`, e.g. `init_prev`, `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Measles(init=dict(R=0.9)), total_pop=100_000, start='2025-01-01', stop='2026-01-01')
        sim.run()
        sim.measles.plot()
        ```
    """
    beta      = ss.par(ss.perday(1.4), doc='Transmission rate, giving R0 of about 15', source='Guerra et al. (2017)')
    dur_exp   = ss.par(ss.normal(mean=ss.days(8), std=ss.days(2)),    doc='Time from infection to being infectious', source='US CDC (mean); the spread is illustrative')
    dur_inf   = ss.par(ss.normal(mean=ss.days(11), std=ss.days(1.5)), doc='Time infectious', source='US CDC (mean); the spread is illustrative')
    p_death   = ss.par(0.005, doc='Probability of death', source='Illustrative')
    init_prev = 0.005

    S    = ss.State()
    E, I = ss.states('E I', infected=True) # Counted in the results infected and prevalence
    R    = ss.State()

    infection = S >> E @ beta*I
    onset     = E >> I @ dur_exp
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)
