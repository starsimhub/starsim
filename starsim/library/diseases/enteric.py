"""
Enteric infections: typhoid with chronic carriers (`Typhoid`), and polio with a paralytic fraction (`Polio`)
"""
import starsim as ss

__all__ = ['Typhoid', 'Polio']


class Typhoid(ss.Disease):
    """
    Typhoid fever, with chronic carriers

    Most people recover from typhoid, but a small share become chronic carriers, who shed for life and are much
    less infectious than an acute case (`rel_carrier`), while contributing to transmission for decades. Immunity
    after infection wanes slowly, if at all.

    Sources: the values of the carriers are from Pitzer VE et al., "Predicting the impact of vaccination on the
    transmission dynamics of typhoid in South Asia", *PLoS Negl Trop Dis* 2014;8(1):e2642
    (https://doi.org/10.1371/journal.pntd.0002642): a relative infectiousness of carriers of 0.01, a share
    becoming carriers within their 0.3% to 10% by age, and a rate of waning immunity estimated as close to zero,
    so immunity is lifelong by default. `beta` is set to give their estimated R0 of 2.8 for Kolkata and Vellore.
    The duration of acute infection, 4 weeks, is a standard value.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (R0 of 2.8 with the defaults)
        rel_carrier (float): the infectiousness of a chronic carrier, relative to an acute case
        p_carrier (float): the share of cases who become chronic carriers
        dur_inf (`ss.dur`): the mean duration of acute infection
        dur_imm (`ss.dur`): the mean time immune after recovering (lifelong by default)
        birth_rate (`ss.per`): the birth and death rate
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Typhoid(), total_pop=100_000, start=1980, stop=2020, dt=ss.weeks(2))
        sim.run()
        sim.plot('typhoid')
        ```
    """
    beta        = ss.par(ss.perday(0.072), doc='Transmission rate (R0 of 2.8)',                source='Pitzer et al. (2014)')
    rel_carrier = ss.par(0.01, doc='Infectiousness of a chronic carrier, relative to an acute case', source='Pitzer et al. (2014)')
    p_carrier   = ss.par(0.05, doc='Share of cases who become chronic carriers',                     source='Pitzer et al. (2014): 0.3% to 10% by age')
    dur_inf     = ss.par(ss.weeks(4), doc='Mean duration of acute infection',                        source='Standard value')
    dur_imm     = ss.par(ss.years(1000), doc='Mean time immune after recovering (lifelong by default)', source='Pitzer et al. (2014): waning estimated as close to zero')
    birth_rate  = ss.par(ss.peryear(1/60), doc='Birth and death rate', source='Illustrative')

    S = ss.State()
    I = ss.State(init=0.001, label='Acute infection')
    C = ss.State(label='Chronic carrier')
    R = ss.State()

    infection = S >> I @ beta*(I + rel_carrier*C)
    outcome   = ss.Transition(I, [C, R], dur=dur_inf, p=[p_carrier, None])
    waning    = R >> S @ dur_imm
    births    = ss.birth(S, rate=birth_rate)
    deaths    = ss.death(rate=birth_rate)

    prevalence = (I + C)/(S + I + C + R)


class Polio(ss.Disease):
    """
    Poliovirus, with a paralytic fraction

    Nearly every infection with wild poliovirus passes without paralysis, so the states are those of the
    infection, and paralysis is a rare outcome of it: a share `p_paralysis` of infections end in `P` rather than
    `R`, and the result `outcome.P` counts the paralytic cases, which are the ones surveillance sees.

    Sources: about 1 in 200 infections with wild poliovirus type 1 causes paralysis in a fully susceptible person
    (US CDC, *Epidemiology and Prevention of Vaccine-Preventable Diseases*, the "Pink Book", chapter on
    poliomyelitis). `beta` gives an R0 of 6, within the usual range of 5 to 7 for wild poliovirus in settings with
    poor sanitation (Fine PEM, Carneiro IAM, "Transmissibility and persistence of oral polio vaccine viruses",
    *Am J Epidemiol* 1999;150(10):1001-1021). The latent period of 3 days and 20 days of shedding are standard
    values; shedding lasts longer in young children.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (R0 of 6 with the defaults)
        dur_exp (`ss.dur`): the mean time from infection to shedding
        dur_inf (`ss.dur`): the mean time shedding
        p_paralysis (float): the share of infections that cause paralysis
        birth_rate (`ss.per`): the birth and death rate, which supplies the susceptible children
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Polio(), total_pop=100_000, start=1980, stop=2000, dt=ss.days(2))
        sim.run()
        print('Paralytic cases:', sim.results.polio['outcome.P'].sum())
        ```
    """
    beta        = ss.par(ss.perday(0.3), doc='Transmission rate (R0 of 6)',       source='Fine and Carneiro (1999): R0 of 5 to 7')
    dur_exp     = ss.par(ss.days(3),  doc='Mean time from infection to shedding', source='Standard value')
    dur_inf     = ss.par(ss.days(20), doc='Mean time shedding',                   source='Standard value')
    p_paralysis = ss.par(1/200, doc='Share of infections that cause paralysis',   source='US CDC Pink Book: about 1 in 200 for wild type 1')
    birth_rate  = ss.par(ss.peryear(0.03), doc='Birth and death rate',            source='Illustrative')

    S, E = ss.states('S E')
    I = ss.State(init=0.001, label='Shedding')
    R = ss.State()
    P = ss.State(label='Paralysed')

    infection = S >> E @ beta*I
    onset     = E >> I @ dur_exp
    outcome   = ss.Transition(I, [P, R], dur=dur_inf, p=[p_paralysis, None])
    births    = ss.birth(S, rate=birth_rate)
    deaths    = ss.death(rate=birth_rate)

    prevalence = (E + I)/(S + E + I + R + P)
