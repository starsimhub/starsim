"""
Ebola, with severe disease and transmission from unburied bodies.

Originally adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/ebola.py
Original version by @domdelport and @RomeshA
"""

import starsim as ss

__all__ = ['Ebola']


class Ebola(ss.Disease):
    """
    Ebola, including severe disease and transmission from unburied bodies

    People infected are latent (`E`) for `dur_exp`, then symptomatic (`I`). A fraction `p_sev` become severely ill
    (`Sev`), and the rest recover; of the severe, a fraction `p_death` die, and the rest recover. Severe cases are more
    infectious (`sev_factor`), and so are the bodies of those who have died (`D`), until they are buried
    (`unburied_factor`): a fraction `p_safe_bury` are buried safely, at once, and the rest after `dur_dead2buried`.
    Bodies are people until they are buried, when they leave the population; the outcome `burial` counts both
    kinds of burial (`safe_burial` and `unsafe_burial`), and `deaths` every death. It is declared, so it runs under any solver.

    Sources: the structure follows the model of Legrand J et al., "Understanding the dynamics of Ebola epidemics",
    *Epidemiol Infect* 2007;135(4):610-621 (https://doi.org/10.1017/S0950268806007217), with funeral transmission;
    the durations and probabilities are illustrative of the 2014 West African epidemic (WHO Ebola Response Team,
    *N Engl J Med* 2014;371:1481-1495, https://doi.org/10.1056/NEJMoa1411100), and `beta` gives an R0 of about 1.7,
    within its estimates of 1.5 to 2.0.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate from a symptomatic person (per contact, if there are networks)
        sev_factor (float): the relative transmissibility of severe cases
        unburied_factor (float): the relative transmissibility of unburied bodies
        dur_exp (`ss.Duration`/`ss.Dist`): the time from infection to symptoms (and being infectious); each duration is a lognormal by default, and a single duration given is the mean of an exponential
        dur_symp2sev (`ss.Duration`/`ss.Dist`): the time from symptoms to severe disease, for those who become severe
        dur_symp2rec (`ss.Duration`/`ss.Dist`): the time from symptoms to recovery, for those who do not
        dur_sev2dead (`ss.Duration`/`ss.Dist`): the time from severe disease to death, for those who die
        dur_sev2rec (`ss.Duration`/`ss.Dist`): the time from severe disease to recovery, for those who recover
        dur_dead2buried (`ss.Duration`/`ss.Dist`): the time from death to an unsafe burial
        p_sev (float): the probability of severe disease
        p_death (float): the probability of death, of the severe
        p_safe_bury (float): the probability of a safe burial, at once
        kwargs (dict): passed to `ss.Disease`, e.g. `init_prev`, `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Ebola(), total_pop=10_000, start='2025-01-01', stop='2026-01-01')
        sim.run()
        sim.ebola.plot()
        ```
    """
    beta            = ss.par(ss.perday(0.1), doc='Transmission rate from a symptomatic person, giving R0 of about 1.7', source='WHO Ebola Response Team (2014)')
    sev_factor      = ss.par(2.2, doc='Relative transmissibility of severe cases', source='Illustrative')
    unburied_factor = ss.par(2.1, doc='Relative transmissibility of unburied bodies', source='Illustrative')
    dur_exp         = ss.par(ss.lognormal(mean=ss.days(12.7), std=ss.days(1)), doc='Time from infection to symptoms', source='Illustrative')
    dur_symp2sev    = ss.par(ss.lognormal(mean=ss.days(6),    std=ss.days(1)), doc='Time from symptoms to severe disease', source='Illustrative')
    dur_symp2rec    = ss.par(ss.lognormal(mean=ss.days(10),   std=ss.days(1)), doc='Time from symptoms to recovery, if not severe', source='Illustrative')
    dur_sev2dead    = ss.par(ss.lognormal(mean=ss.days(1.5),  std=ss.days(1)), doc='Time from severe disease to death', source='Illustrative')
    dur_sev2rec     = ss.par(ss.lognormal(mean=ss.days(10.4), std=ss.days(1)), doc='Time from severe disease to recovery', source='Illustrative')
    dur_dead2buried = ss.par(ss.lognormal(mean=ss.days(2),    std=ss.days(1)), doc='Time from death to an unsafe burial', source='Illustrative')
    p_sev           = ss.par(0.7,  doc='Probability of severe disease', source='Illustrative')
    p_death         = ss.par(0.55, doc='Probability of death, of the severe', source='Illustrative')
    p_safe_bury     = ss.par(0.25, doc='Probability of a safe burial', source='Illustrative')
    init_prev       = 0.005

    S   = ss.State()
    E   = ss.State('Exposed', infected=True)
    I   = ss.State('Symptomatic', infected=True)
    Sev = ss.State('Severe', infected=True)
    R   = ss.State()
    D   = ss.State('Dead, not yet buried') # Still people, until they are buried

    infection    = S >> E @ beta*(I + sev_factor*Sev + unburied_factor*D)
    onset        = E >> I @ dur_exp
    severe       = ss.Transition(I, Sev, dur=dur_symp2sev, p=p_sev)
    recovery     = ss.Transition(I, R, dur=dur_symp2rec, p=1 - p_sev)
    death        = ss.Transition(Sev, D, dur=dur_sev2dead, p=p_death*(1 - p_safe_bury)) # Dies, and is buried later, unsafely
    safe_burial  = ss.Transition(Sev, ss.dead, dur=dur_sev2dead, p=p_death*p_safe_bury) # Dies, and is buried at once, safely
    recovery_sev = ss.Transition(Sev, R, dur=dur_sev2rec, p=1 - p_death)
    unsafe_burial = D >> ss.dead @ dur_dead2buried
    burial       = safe_burial + unsafe_burial # Both kinds of burial
    deaths       = death + safe_burial # Every death
