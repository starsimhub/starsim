"""
Malaria: the Ross-Macdonald model of humans and mosquitoes (`Malaria`), and the Garki model of malaria with immunity (`MalariaImmunity`)

Both are declared, so they run as compartments or as agents, and neither needs a network: malaria passes
through mosquitoes, which are counts rather than people.
"""
import numpy as np
import starsim as ss

__all__ = ['Malaria', 'MalariaImmunity']


class Malaria(ss.Disease):
    """
    Malaria as the Ross-Macdonald model of humans and mosquitoes

    People are susceptible, infectious, or recovered, and are born and die; mosquitoes are susceptible or
    infectious, and stay infectious until they die. Mosquitoes are a count, not people (`Sv`, `Iv`), and start
    at their density of `m` per person, all susceptible.

    Source: the Ross-Macdonald model, in the form of Keeling MJ, Rohani P, *Modeling Infectious Diseases in
    Humans and Animals*, Princeton University Press, 2008, section 4.4. Every default here is from its Program 4.4:
    the biting rate `a` (its `r` of 0.5 per 1,000 humans), the transmission probabilities `b` and `c` (its
    transmission matrix), `gamma`, `mu_h`, `mu_v`, and `m` (its 10,091 mosquitoes per 1,000 people). The model is
    originally that of Ross R, *The Prevention of Malaria*, Murray, 1911, and Macdonald G, *The Epidemiology and
    Control of Malaria*, Oxford University Press, 1957.

    `R0()` is 14.6 with these values, which is the square root of the classical Ross-Macdonald value,
    m·a²·b·c/(γ·μ_v) = 213, since each passage through a mosquito and back is two generations.

    Note: CRN safe as agents.

    Args:
        a (`ss.per`): bites on people per mosquito per day
        b (float): the chance that a bite by an infectious mosquito infects a person
        c (float): the chance that a bite on an infectious person infects a mosquito
        m (float): mosquitoes per person
        gamma (`ss.per`): the rate at which people recover
        mu_h (`ss.per`): the birth and death rate of people
        mu_v (`ss.per`): the death rate of mosquitoes, which emerge at the same rate
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Malaria(), total_pop=10_000, unit='day', stop=1000)
        sim.run()
        sim.diseases.malaria.plot('Ih', 'Iv')
        ```
    """
    a     = ss.par(ss.perday(0.5),    doc='Bites on people per mosquito per day',                    source='Keeling and Rohani (2008), Program 4.4')
    b     = ss.par(0.5,               doc='Chance that a bite by an infectious mosquito infects a person', source='Keeling and Rohani (2008), Program 4.4')
    c     = ss.par(0.8,               doc='Chance that a bite on an infectious person infects a mosquito',  source='Keeling and Rohani (2008), Program 4.4')
    m     = ss.par(10.09,             doc='Mosquitoes per person',                                   source='Keeling and Rohani (2008), Program 4.4')
    gamma = ss.par(ss.perday(0.033),  doc='Rate at which people recover',                            source='Keeling and Rohani (2008), Program 4.4')
    mu_h  = ss.par(ss.perday(5.5e-5), doc='Birth and death rate of people',                          source='Keeling and Rohani (2008), Program 4.4')
    mu_v  = ss.par(ss.perday(0.143),  doc='Death rate of mosquitoes',                                source='Keeling and Rohani (2008), Program 4.4')

    Sh = ss.State(label='Susceptible people')
    Ih = ss.State(init=0.001, label='Infectious people')
    Rh = ss.State(label='Recovered people')
    Sv = ss.State(people=False, label='Susceptible mosquitoes')
    Iv = ss.State(people=False, label='Infectious mosquitoes')
    Nh = (Sh + Ih + Rh).n
    init = dict(Sv=m*Nh)

    infection  = Sh >> Ih @ a*b*Iv.n/Nh
    recovery   = Ih >> Rh @ gamma
    biting     = Sv >> Iv @ a*c*Ih.n/Nh
    emergence  = ss.null >> Sv @ m*mu_v*Nh
    mortality  = Sv + Iv >> ss.null @ mu_v
    births     = ss.birth(Sh, rate=mu_h)
    deaths     = ss.death(rate=mu_h)
    prevalence = Ih/(Sh + Ih + Rh)


class MalariaImmunity(ss.Disease):
    """
    Malaria with immunity, as the Garki model

    The states keep the source's names: `x1` (nonimmune and negative), `x2` (nonimmune, incubating), `y1`
    (positive and infectious), `y2` (positive, not infectious, recovering slowly), `y3` (immune positive,
    recovering fast), `x3` (immune and negative), and `x4` (immune, incubating). People lose infectivity before
    they gain a fast recovery rate, and are detected less often once immune (`q3`). Everything about the
    mosquitoes is in one number, the vectorial capacity `C`: the potentially infective contacts each person makes
    per day through them. Inoculation saturates as `C` rises (h = g(1 − exp(−C·y1))), and superinfection slows
    recovery (R(h) = h/(exp(h/r) − 1)).

    Source: the Garki model of Dietz K, Molineaux L, Thomas A, "A malaria model tested in the African savannah",
    *Bull World Health Organ* 1974;50(3-4):347-357, in the form given by Molineaux L, Gramiccia G, *The Garki
    Project*, WHO, 1980, chapter 10. The defaults are its Table 31: `g`, `alpha2`, and `r1` were fitted to the
    Garki data, and `alpha1`, `delta`, `dur_inc`, `ratio` (r2/r1), and `q3` were assumed. `C` varied by season and
    village from below 0.01 to above 2 (its Fig. 77), so the default here is a constant chosen for illustration;
    give an `ss.seasonal()` for a seasonal one.

    Malaria dies out below the critical vectorial capacity, (alpha1 + delta)/g, which is 0.022 per day with
    these values, as the source gives.

    Note: CRN safe as agents.

    Args:
        C (float): the vectorial capacity: potentially infective contacts per person per day
        g (float): the chance that an infection results, given at least one inoculation
        r1 (`ss.per`): the recovery rate of nonimmune people, before superinfection
        ratio (float): the recovery rate of immune people, relative to `r1`
        alpha1 (`ss.per`): the rate of losing infectivity
        alpha2 (`ss.per`): the rate of acquiring a high recovery rate (immunity)
        delta (`ss.per`): the birth and death rate
        dur_inc (`ss.dur`): the incubation period in people
        q3 (float): the chance that an immune positive person is detected by a blood examination
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.MalariaImmunity(C=0.1), total_pop=10_000, unit='day', stop=365*20)
        sim.run()
        sim.plot('malariaimmunity')
        ```
    """
    C       = ss.par(0.1,                doc='Vectorial capacity: potentially infective contacts per person per day', source='Illustrative; the Garki villages ranged from below 0.01 to above 2')
    g       = ss.par(0.097,              doc='Chance that an infection results, given at least one inoculation', source='Molineaux and Gramiccia (1980), Table 31 (fitted)')
    r1      = ss.par(ss.perday(0.0023),  doc='Recovery rate of nonimmune people, before superinfection',        source='Molineaux and Gramiccia (1980), Table 31 (fitted)')
    ratio   = ss.par(10.0,               doc='Recovery rate of immune people, relative to r1',                  source='Molineaux and Gramiccia (1980), Table 31 (assumed)')
    alpha1  = ss.par(ss.perday(0.002),   doc='Rate of losing infectivity',                                      source='Molineaux and Gramiccia (1980), Table 31 (assumed)')
    alpha2  = ss.par(ss.perday(0.00019), doc='Rate of acquiring a high recovery rate',                          source='Molineaux and Gramiccia (1980), Table 31 (fitted)')
    delta   = ss.par(ss.perday(1e-4),    doc='Birth and death rate (36.5 per 1,000 per year)',                  source='Molineaux and Gramiccia (1980), Table 31 (assumed)')
    dur_inc = ss.par(ss.days(15),        doc='Incubation period in people',                                     source='Molineaux and Gramiccia (1980), Table 31 (assumed)')
    q3      = ss.par(0.7,                doc='Chance that an immune positive person is detected',               source='Molineaux and Gramiccia (1980), Table 31 (assumed)')

    x1, x2, x3, x4, y2, y3 = ss.states('x1 x2 x3 x4 y2 y3')
    y1 = ss.State(init=0.01, label='Positive and infectious')

    # The inoculation rate saturates with the vectorial capacity, and superinfection slows recovery; the floor keeps R(h) finite when nobody is infectious
    h  = np.maximum(g*(1 - np.exp(-C*y1.frac)), 1e-12)
    R1 = ss.perday(1)*h/(np.exp(h/(r1*ss.days(1))) - 1)
    R2 = ss.perday(1)*h/(np.exp(h/(ratio*r1*ss.days(1))) - 1)

    inoculation   = x1 >> x2 @ ss.perday(1)*h
    onset         = x2 >> y1 @ ss.constant(dur_inc)
    losing_infect = y1 >> y2 @ alpha1
    recovery      = y2 >> x1 @ R1
    immunity      = y2 >> y3 @ alpha2
    reinoculation = x3 >> x4 @ ss.perday(1)*h
    reonset       = x4 >> y3 @ ss.constant(dur_inc)
    recovery_imm  = y3 >> x3 @ R2
    births        = ss.birth(x1, rate=delta)
    deaths        = ss.death(rate=delta)

    prevalence = (y1 + y2 + y3)/(x1 + x2 + x3 + x4 + y1 + y2 + y3) # The true parasite rate
    detected   = (y1 + y2 + q3*y3)/(x1 + x2 + x3 + x4 + y1 + y2 + y3) # What a blood examination finds
