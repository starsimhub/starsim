"""
Cholera, with direct and environmental (waterborne) transmission.

Originally adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/cholera.py
Original version by Dom Delport 2024
"""
# Authorship: human drafted, AI refactored, human reviewed

import starsim as ss

__all__ = ['Cholera']


class Cholera(ss.Disease):
    """
    Cholera, with both direct and environmental (waterborne) transmission

    People infected are latent (`E`) for `dur_exp`, then a fraction `p_symp` become symptomatic (`Is`), and the rest are
    asymptomatic (`Ia`), shedding far less (`asymp_trans`). Asymptomatic people recover; symptomatic people recover,
    or die (with probability `p_death`, sooner). Besides transmission from person to person (`beta`), infectious
    people shed bacteria into a well-mixed reservoir of water (`W`, a count of bacteria, not people) at
    `shedding_rate`, the bacteria die off at `decay_rate`, and susceptible people are infected from the water at a
    rate that rises with the concentration, `beta_env` times `env_conc` = W/(W + `half_sat_rate`), as in the model of
    Codeço (2001). It is declared, so it runs under any solver.

    Sources: Codeço CT, "Endemic and epidemic dynamics of cholera: the role of the aquatic reservoir", *BMC Infect Dis*
    2001;1:1 (https://doi.org/10.1186/1471-2334-1-1), for the structure; the latent period is from Azman et al.
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3677557/), a lognormal; asymptomatic shedding of 1 to 10 days (uniform) from the
    WHO fact sheet; 5 days of symptoms (a lognormal fit to the range of 2.9 to 14 days) from Fung (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3926264/); a rapid death
    of severe cases (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5767916/); half of infections symptomatic, mid range
    of Jackson et al. and Nelson et al.; asymptomatic carriers shedding 100 times less; the dose-response, shedding,
    and decay from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/). `beta` (from person to
    person) and `beta_env` are illustrative. Since the bacteria are a count, shed by everyone infectious into one
    reservoir, the water carries more of the epidemic in a larger population: with these values, it sustains an
    outbreak in a town of 10,000, and transmission from person to person alone does not.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the rate of transmission from person to person (per contact, if there are networks), for a symptomatic person
        asymp_trans (float): the relative transmissibility and shedding of asymptomatic people
        beta_env (`ss.per`): the rate of infection from the water, when it is saturated with bacteria
        half_sat_rate (float): the amount of bacteria at which infection from the water is half its maximum
        shedding_rate (`ss.freq`): the bacteria each symptomatic person sheds into the water, per unit time
        decay_rate (`ss.per`): the rate at which bacteria in the water die off
        dur_exp (`ss.Duration`/`ss.Dist`): the time from infection to being infectious (a duration is the mean of an exponential)
        dur_asymp2rec (`ss.Duration`/`ss.Dist`): the time asymptomatic people are infectious
        dur_symp2rec (`ss.Duration`/`ss.Dist`): the time symptomatic people are ill, for those who recover
        dur_symp2dead (`ss.Duration`/`ss.Dist`): the time from symptoms to death, for those who die
        p_symp (float): the probability that an infection is symptomatic
        p_death (float): the probability that a symptomatic person dies
        kwargs (dict): passed to `ss.Disease`, e.g. `init_prev`, `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Cholera(), total_pop=10_000, start='2025-01-01', stop='2026-01-01')
        sim.run()
        sim.cholera.plot()
        ```
    """
    beta          = ss.par(ss.perday(0.05), doc='Transmission rate from person to person, for a symptomatic person', source='Illustrative')
    asymp_trans   = ss.par(0.01, doc='Relative transmissibility and shedding of asymptomatic people', source='Asymptomatic carriers shed 100-1000 times less (Nelson et al.)')
    beta_env      = ss.par(ss.perday(0.5), doc='Rate of infection from water saturated with bacteria', source='Illustrative, as in Codeço (2001)')
    half_sat_rate = ss.par(1_000_000, doc='Amount of bacteria at which infection from the water is half its maximum', source='Mukandavire et al.')
    shedding_rate = ss.par(ss.freqperday(10), doc='Bacteria shed into the water by each symptomatic person', source='Mukandavire et al.')
    decay_rate    = ss.par(ss.perday(0.033), doc='Rate at which bacteria in the water die off', source='Mukandavire et al.')
    dur_exp       = ss.par(ss.lognormal(mean=ss.days(2.772), std=ss.days(4.737)), doc='Time from infection to being infectious', source='Calculated from Azman et al.')
    dur_asymp2rec = ss.par(ss.uniform(low=ss.days(1), high=ss.days(10)), doc='Time asymptomatic people shed', source='WHO cholera fact sheet (1-10 days)')
    dur_symp2rec  = ss.par(ss.lognormal(mean=ss.days(5), std=ss.days(1.8)), doc='Time ill, for those who recover', source='Fung (5 days in most models; 2.9-14 days found)')
    dur_symp2dead = ss.par(ss.lognormal(mean=ss.days(1), std=ss.days(0.5)), doc='Time from symptoms to death, for those who die', source='Death from severe disease is rapid')
    p_symp        = ss.par(0.5, doc='Probability that an infection is symptomatic', source='Jackson et al.; Nelson et al.')
    p_death       = ss.par(0.005, doc='Probability that a symptomatic person dies (with treatment)', source='Illustrative')
    init_prev     = 0.005

    S  = ss.State()
    E  = ss.State('Exposed', infected=True)
    Ia = ss.State('Asymptomatic', infected=True)
    Is = ss.State('Symptomatic', infected=True)
    R  = ss.State()
    W  = ss.State('Bacteria in the water', people=False) # A count, not people

    env_conc = W.n/(W.n + half_sat_rate) # The concentration of bacteria, as the fraction of the maximum rate of infection from the water

    infection     = S >> E @ beta*(Is + asymp_trans*Ia) + beta_env*env_conc
    onset         = E >> {Is: p_symp, Ia: None} @ dur_exp
    recovery_asymp = Ia >> R @ dur_asymp2rec
    recovery      = ss.Transition(Is, R, dur=dur_symp2rec, p=1 - p_death)
    death         = ss.Transition(Is, ss.dead, dur=dur_symp2dead, p=p_death)
    shedding      = ss.outside >> W @ shedding_rate*(Is.n + asymp_trans*Ia.n)
    decay         = W >> ss.dead @ decay_rate
