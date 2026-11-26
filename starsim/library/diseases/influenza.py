"""
Seasonal influenza by age, with a yearly vaccination campaign (`Influenza`)
"""
# Authorship: AI drafted, human reviewed and edited

import numpy as np
import starsim as ss

__all__ = ['Influenza']


age_groups = [0, 1, 5, 15, 25, 45, 65] # Baguelin et al.'s age groups: under 1, 1-4, 5-14, 15-24, 25-44, 45-64, and 65+
by_age = lambda *values: ss.by(age=list(values))
baguelin = 'Baguelin et al. (2013)'
cdc = 'CDC burden estimates for 2018-19, by age (0-4, 5-17, 18-49, 50-64, 65+)'


class Influenza(ss.Disease):
    """
    Seasonal influenza, as an SEIR model by age with or without symptoms, waning immunity, and a yearly vaccination campaign

    People who are infected are exposed (`E`), then infectious with symptoms (`I`) or without (`A`, less
    infectious), and then recover (`R`). Some symptomatic cases are hospitalized (`H`), and some of those die,
    most of all among the over-65s. Immunity wanes (`R` to `S`) within a year or two, as the virus drifts away
    from it, and transmission follows a yearly cycle (`ss.seasonal`), highest in January in the northern
    hemisphere; together these bring influenza back each winter. Since an epidemic this fast all but dies out
    by summer, a few people are infected from outside the population each year (`import_rate`), to seed the
    next, and 30% of people start immune, as after earlier seasons. For the southern hemisphere, move the cycle's
    peak: `ssl.Influenza(beta=ss.seasonal(mean=ss.perday(0.87), amplitude=0.3, peak='July'))`, and the campaign
    with it (`campaign=`, below).

    Vaccination is a yearly campaign before the season, in the tenth and eleventh months of each year of the sim
    (October and November, when it starts in January): `cov_vax` of each age group is vaccinated, and the vaccine
    protects `eff_vax` of them fully (all or nothing, as in Baguelin et al.), moving them into `V`, from which
    they become susceptible again in about 6 months. Only susceptible people are vaccinated (vaccinating the
    immune would change nothing here), and `vaccination` counts those protected, not doses. Coverage is 0 by
    default, the season with no vaccination.

    Without contacts given, `beta` gives an R0 of 1.3 on average over the year, the median estimate for seasonal
    influenza; `R0()` reports 1.7, since it reads the cycle at the start of the run, its January peak. Each season
    then infects about a fifth of people. With a contact matrix, `beta` is per contact, and 0.063 gives the same
    R0 with the United Kingdom's contacts: `ssl.Influenza(beta=ss.seasonal(mean=0.063, amplitude=0.3,
    peak='January'))` with `networks=ss.contacts('United Kingdom')` (Baguelin et al.'s chance of infection per
    contact, 0.17, goes with susceptibility by age that they fitted, which is left out here).

    Left out: Baguelin et al.'s risk groups, two stages each of `E` and `I`, susceptibility by age, the
    separate strains, and fitting each season; here immunity carries over from season to season, waning at a
    single rate. The amplitude of the cycle, the durations of immunity and of protection by the vaccine, the time
    in hospital, the infectiousness of asymptomatic infections, the importations, and the immunity at the start are
    illustrative. Its stages are shorter than a day, so as agents it needs short steps (`sim.check_dt()` says how
    much they matter).

    Sources:

    - Baguelin M et al., "Assessing optimal target populations for influenza vaccination programmes: an evidence
      synthesis and modelling study", *PLoS Med* 2013;10(10):e1001527, https://doi.org/10.1371/journal.pmed.1001527.
      The structure: seven age groups, latent and infectious periods of 0.8 and 1.8 days, and all-or-nothing
      vaccination with an efficacy of 70% under 65 and 46% over 65 (in a year when the vaccine matches the strains).
    - Carrat F et al., "Time lines of infection and disease in human influenza: a review of volunteer challenge
      studies", *Am J Epidemiol* 2008;167(7):775-785, https://doi.org/10.1093/aje/kwm375. Two-thirds of
      infections with symptoms (66.9%).
    - US CDC, "Estimated flu-related illnesses, medical visits, hospitalizations, and deaths in the United States,
      2018-2019 flu season", https://archive.cdc.gov/www_cdc_gov/flu/about/burden/2018-2019.html. The fractions of
      symptomatic cases hospitalized, and of those hospitalized who die, as ratios of its estimates by age, with
      every death counted as one in hospital: 9% of symptomatic cases over 65 are hospitalized, and 10% of those die.
    - Biggerstaff M et al., "Estimates of the reproduction number for seasonal, pandemic, and zoonotic influenza: a
      systematic review of the literature", *BMC Infect Dis* 2014;14:480, https://doi.org/10.1186/1471-2334-14-480.
      A median R0 of 1.28 for seasonal influenza.

    Note: CRN safe as agents.

    Args:
        beta (`ss.seasonal`/`ss.per`): the transmission rate over the year, as a cycle around its mean (per contact, with a contact matrix)
        rel_asym (float): the infectiousness of asymptomatic infections, relative to symptomatic ones
        p_sym (float): the fraction of infections with symptoms
        p_hosp (float/`ss.by`): the fraction of symptomatic cases who are hospitalized, by age
        p_death (float/`ss.by`): the fraction of those hospitalized who die, by age
        dur_exp (`ss.Duration`): the mean time from infection to being infectious
        dur_inf (`ss.Duration`): the mean time infectious
        dur_hosp (`ss.Duration`): the mean time in hospital
        dur_imm (`ss.Duration`): the mean time immune after recovering, until drift makes people susceptible again
        cov_vax (float/`ss.by`): the fraction of each age group vaccinated in each yearly campaign
        eff_vax (float/`ss.by`): the fraction of those vaccinated whom the vaccine protects, by age
        dur_vax (`ss.Duration`): the mean time protected by the vaccine
        campaign (`ss.timeseries`): when the campaign runs each year, as a rate during it that adds up to 1 over the campaign, and 0 otherwise
        import_rate (`ss.per`): the rate at which susceptible people are infected from outside the population
        birth_rate (`ss.per`): the birth rate, which supplies new susceptible people
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Influenza(), total_pop=100_000, start='2000-01-01', stop='2010-01-01', dt=ss.days(1))
        sim.run()
        sim.influenza.plot()
        sim.results.influenza.outcome.select(to='dead').by('age').plot() # Deaths by age

        # Vaccinating 70% of the over-65s and 50% of children each autumn, with the UK's contacts
        flu = ssl.Influenza(beta=ss.seasonal(mean=0.063, amplitude=0.3, peak='January'), cov_vax=ss.by(age=[0, 0.5, 0.5, 0, 0, 0, 0.7]))
        sim = ss.Sim(diseases=flu, networks=ss.contacts('United Kingdom'), total_pop=100_000, start='2000-01-01', stop='2010-01-01', dt=ss.days(1))
        ```
    """
    beta     = ss.par(ss.seasonal(mean=ss.perday(0.87), amplitude=0.3, peak='January'),
                      doc='Transmission rate over the year, as a cycle around a mean giving R0 of 1.3', source='Biggerstaff et al. (2014); the cycle is illustrative')
    rel_asym = ss.par(0.5,   doc='Infectiousness of asymptomatic infections, relative to symptomatic ones', source='Illustrative')
    p_sym    = ss.par(0.669, doc='Fraction of infections with symptoms', source='Carrat et al. (2008)')
    p_hosp   = ss.par(by_age(0.0070, 0.0070, 0.0027, 0.0056, 0.0056, 0.0106, 0.091), doc='Fraction of symptomatic cases hospitalized', source=cdc)
    p_death  = ss.par(by_age(0.010, 0.010, 0.0086, 0.029, 0.029, 0.057, 0.104),     doc='Fraction of those hospitalized who die', source=cdc)
    dur_exp  = ss.par(ss.days(0.8), doc='Mean time from infection to being infectious', source=baguelin)
    dur_inf  = ss.par(ss.days(1.8), doc='Mean time infectious', source=baguelin)
    dur_hosp = ss.par(ss.days(5),   doc='Mean time in hospital', source='Illustrative')
    dur_imm  = ss.par(ss.years(2),  doc='Mean time immune after recovering', source='Illustrative (antigenic drift)')
    cov_vax  = ss.par(0.0, doc='Fraction of each age group vaccinated each year', source='None by default')
    eff_vax  = ss.par(by_age(0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.46), doc='Fraction of those vaccinated who are protected', source=f'{baguelin}, for a well-matched vaccine')
    dur_vax  = ss.par(ss.months(6), doc='Mean time protected by the vaccine', source='Illustrative (protection wanes within the season)')
    campaign = ss.par(ss.timeseries({ss.months(0): 0, ss.months(9): ss.permonth(1/2), ss.months(11): 0}, period=ss.years(1)),
                      doc='When the campaign runs each year (a rate adding up to 1 over it)', source='Illustrative (October and November)')
    import_rate  = ss.par(ss.peryear(0.005), doc='Rate at which susceptible people are infected from outside the population', source='Illustrative')
    birth_rate   = ss.par(ss.peryear(0.02), doc='Birth rate',                   source='Illustrative')
    death_rate   = ss.par(ss.peryear(0.02), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S = ss.State()
    E = ss.State(infected=True) # Counted in the results infected and prevalence
    I = ss.State(init=0.001, label='Symptomatic', infected=True)
    A = ss.State(label='Asymptomatic', infected=True)
    H = ss.State(label='Hospitalized', infected=True)
    R = ss.State(init=0.3)
    V = ss.State(label='Vaccinated')

    infection   = S >> E @ beta*(I + rel_asym*A)
    importation = S >> E @ import_rate
    onset       = E >> {I: p_sym, A: None} @ dur_exp
    hospital    = I >> {H: p_hosp, R: None} @ dur_inf
    recovery    = A >> R @ dur_inf
    outcome     = H >> {ss.dead: p_death, R: None} @ dur_hosp
    waning      = R >> S @ dur_imm
    vaccination = S >> V @ -np.log(1 - eff_vax*cov_vax)*campaign # Protects eff_vax*cov_vax of the susceptible over the campaign
    waning_vax  = V >> S @ dur_vax
    births      = ss.birth(S, rate=birth_rate)
    deaths      = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify(age=age_groups)
        return
