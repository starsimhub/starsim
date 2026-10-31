"""
Seasonal and waning respiratory infections: influenza (`Influenza`), RSV (`RSV`), and pertussis (`Pertussis`)

All three are declared, and all three keep going through the supply of susceptible people that births and
waning immunity provide, rather than burning out as a single outbreak does.
"""
import starsim as ss

__all__ = ['Influenza', 'RSV', 'Pertussis']


class Influenza(ss.Disease):
    """
    Seasonal influenza, as a SEIRS model with a yearly cycle in transmission

    Transmission follows a smooth yearly cycle (`ss.seasonal`), highest in January in the northern hemisphere, and
    immunity wanes within a year or so, as the virus drifts. Together these keep influenza returning each winter.
    For the southern hemisphere, give the cycle another peak:
    `ssl.Influenza(beta=ss.seasonal(mean=ss.perday(0.26), amplitude=0.3, peak='July'))`.

    Sources: the natural history is from Carrat F et al., "Time lines of infection and disease in human influenza:
    a review of volunteer challenge studies", *Am J Epidemiol* 2008;167(7):775-785
    (https://doi.org/10.1093/aje/kwm375): about 2 days from infection to shedding, and about 5 days of shedding.
    `beta` gives an R0 of 1.3 on average over the year, the median estimate for seasonal influenza in the review of
    Biggerstaff M et al., *BMC Infect Dis* 2014;14:480 (https://doi.org/10.1186/1471-2334-14-480); `R0()` reports
    1.8, since it reads the cycle at the start of the run, which is its January peak. The amplitude of the cycle
    and the duration of immunity are illustrative, and since everyone here becomes fully susceptible again, the
    epidemics infect more people each year than real seasons do.

    Note: CRN safe as agents.

    Args:
        beta (`ss.seasonal`/`ss.per`): the transmission rate over the year, as a cycle around its mean
        dur_exp (`ss.dur`): the mean time from infection to being infectious
        dur_inf (`ss.dur`): the mean time infectious
        dur_imm (`ss.dur`): the mean time immune after recovering
        birth_rate (`ss.per`): the birth and death rate, which supplies new susceptible people
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Influenza(), total_pop=100_000, start='2000-01-01', stop='2010-01-01', dt=ss.days(1))
        sim.run()
        sim.plot('influenza')
        ```
    """
    beta       = ss.par(ss.seasonal(mean=ss.perday(0.26), amplitude=0.3, peak='January'),
                        doc='Transmission rate over the year, as a cycle around a mean giving R0 of 1.3', source='Biggerstaff et al. (2014); the cycle is illustrative')
    dur_exp    = ss.par(ss.days(2),  doc='Mean time from infection to being infectious', source='Carrat et al. (2008)')
    dur_inf    = ss.par(ss.days(5),  doc='Mean time infectious',                         source='Carrat et al. (2008)')
    dur_imm    = ss.par(ss.years(1), doc='Mean time immune after recovering',            source='Illustrative (antigenic drift)')
    birth_rate = ss.par(ss.peryear(0.02), doc='Birth and death rate',                    source='Illustrative')

    S, E = ss.states('S E')
    I = ss.State(init=0.0001, label='Infectious')
    R = ss.State()

    infection = S >> E @ beta*I
    onset     = E >> I @ dur_exp
    recovery  = I >> R @ dur_inf
    waning    = R >> S @ dur_imm
    births    = ss.birth(S, rate=birth_rate)
    deaths    = ss.death(rate=birth_rate)

    prevalence = (E + I)/(S + E + I + R)


class RSV(ss.Disease):
    """
    Respiratory syncytial virus (RSV), as a seasonal MSEIRS model

    Babies are born with maternal antibodies (`M`), which protect them for a few months; everyone else is
    susceptible, exposed, infectious, or immune, and immunity wanes within about a year, so people are infected
    again and again. Transmission follows a yearly cycle, as RSV does in temperate countries.

    Sources: the infectious period of 10 days and the cosine seasonal form are from Weber A, Weber M, Milligan P,
    "Modeling epidemics caused by respiratory syncytial virus (RSV)", *Math Biosci* 2001;172(2):95-113
    (https://doi.org/10.1016/S0025-5564(01)00066-9), whose estimates of R0 for a SIRS model of RSV, 1.2 to 2.1,
    `beta` is set within (R0 of 2 on average over the year). The latent period of 5 days is the median of
    Crowcroft NS et al., as used by Hogan AB et al., *BMC Infect Dis* 2014. The duration of maternal protection,
    the duration of immunity, and the amplitude of the cycle are illustrative.

    Note: CRN safe as agents.

    Args:
        beta (`ss.seasonal`/`ss.per`): the transmission rate over the year, as a cycle around its mean
        dur_mat (`ss.dur`): the mean time babies are protected by maternal antibodies
        dur_exp (`ss.dur`): the mean time from infection to being infectious
        dur_inf (`ss.dur`): the mean time infectious
        dur_imm (`ss.dur`): the mean time immune after recovering
        birth_rate (`ss.per`): the birth and death rate; babies are born into `M`
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.RSV(), total_pop=100_000, start='2000-01-01', stop='2006-01-01', dt=ss.days(1))
        sim.run()
        sim.plot('rsv')
        ```
    """
    beta       = ss.par(ss.seasonal(mean=ss.perday(0.2), amplitude=0.35, peak='January'),
                        doc='Transmission rate over the year, as a cycle around a mean giving R0 of 2', source='Weber et al. (2001); the cycle is illustrative')
    dur_mat    = ss.par(ss.months(4), doc='Mean time protected by maternal antibodies', source='Illustrative')
    dur_exp    = ss.par(ss.days(5),   doc='Mean time from infection to being infectious', source='Crowcroft et al., as used by Hogan et al. (2014)')
    dur_inf    = ss.par(ss.days(10),  doc='Mean time infectious',                         source='Weber et al. (2001)')
    dur_imm    = ss.par(ss.years(1),  doc='Mean time immune after recovering',            source='Illustrative')
    birth_rate = ss.par(ss.peryear(0.02), doc='Birth and death rate', source='Illustrative')

    M, S, E = ss.states('M S E')
    I = ss.State(init=0.001, label='Infectious')
    R = ss.State()

    maternal  = M >> S @ dur_mat
    infection = S >> E @ beta*I
    onset     = E >> I @ dur_exp
    recovery  = I >> R @ dur_inf
    waning    = R >> S @ dur_imm
    births    = ss.birth(M, rate=birth_rate) # Babies are born protected
    deaths    = ss.death(rate=birth_rate)

    prevalence = (E + I)/(M + S + E + I + R)


class Pertussis(ss.Disease):
    """
    Pertussis (whooping cough), as a SEIRS model with slowly waning immunity

    Pertussis is highly transmissible, and immunity after infection, while long, is not lifelong, so people are
    infected again decades later. That waning, together with births, keeps pertussis circulating and gives the
    epidemics that return every few years, as in Keeling and Rohani's treatment of SEIRS models with demography.

    Sources: the natural history and the duration of immunity are from Wearing HJ, Rohani P, "Estimating the
    duration of pertussis immunity using epidemiological signatures", *PLoS Pathogens* 2009;5(10):e1000647
    (https://doi.org/10.1371/journal.ppat.1000647): a latent period of 8 days, an infectious period of 15 days,
    natural immunity of at least 30 years on average, and a pre-vaccine birth rate of 0.012 to 0.028 per year.
    `beta` gives an R0 of 13, within their pre-vaccine estimate of 11 to 15. The structure is that of Keeling MJ,
    Rohani P, *Modeling Infectious Diseases in Humans and Animals*, Princeton University Press, 2008, chapter 2.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (R0 of 13 with the default infectious period)
        dur_exp (`ss.dur`): the mean latent period
        dur_inf (`ss.dur`): the mean infectious period
        dur_imm (`ss.dur`): the mean time immune after recovering
        birth_rate (`ss.per`): the birth and death rate
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Pertussis(), total_pop=500_000, start='1920-01-01', stop='1960-01-01', dt=ss.days(2))
        sim.run()
        sim.plot('pertussis')
        ```
    """
    beta       = ss.par(ss.perday(13/15), doc='Transmission rate (R0 of 13)', source='Wearing and Rohani (2009): R0 of 11 to 15')
    dur_exp    = ss.par(ss.days(8),   doc='Mean latent period',                   source='Wearing and Rohani (2009)')
    dur_inf    = ss.par(ss.days(15),  doc='Mean infectious period',               source='Wearing and Rohani (2009)')
    dur_imm    = ss.par(ss.years(30), doc='Mean time immune after recovering',    source='Wearing and Rohani (2009)')
    birth_rate = ss.par(ss.peryear(0.02), doc='Birth and death rate',             source='Wearing and Rohani (2009): 0.012 to 0.028 per year')

    S, E = ss.states('S E')
    I = ss.State(init=0.0001, label='Infectious')
    R = ss.State()

    infection = S >> E @ beta*I
    onset     = E >> I @ dur_exp
    recovery  = I >> R @ dur_inf
    waning    = R >> S @ dur_imm
    births    = ss.birth(S, rate=birth_rate)
    deaths    = ss.death(rate=birth_rate)

    prevalence = (E + I)/(S + E + I + R)
