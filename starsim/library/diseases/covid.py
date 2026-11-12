"""
COVID-19 as a SECIR model (`COVID`)
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['COVID']


ages = [0, 5, 15, 35, 60, 80] # MEmilio's age groups: 0-4, 5-14, 15-34, 35-59, 60-79, and 80+
by_age = lambda *values: ss.by(age=list(values))
days = lambda *values: by_age(*[ss.days(v) for v in values])
source = 'MEmilio ode_secir_parameter_study_graph (midpoint of its range)'


class COVID(ss.Disease):
    """
    COVID-19, as a SECIR model by age: a chain of increasingly severe states, each of which some people leave by recovering

    People who are infected are exposed (`E`), then infected without symptoms (`INS`), and then either recover or
    develop symptoms (`ISy`); of those, some become severe (`ISev`, in hospital), then critical (`ICr`, in
    intensive care), and some of those die. People without symptoms and with symptoms are both infectious, at
    `rel_ins` and `rel_sym` of the transmission rate.

    The disease is stratified by age into MEmilio's six groups (0-4, 5-14, 15-34, 35-59, 60-79, and 80+), since
    the severity of COVID-19 rises steeply with age: in the oldest group, a symptomatic case is 30 times as likely
    to become severe as in a child, and a critical one 12 times as likely to die. The fraction of people in each group
    is the world's, unless the sim's population says otherwise (e.g. `people=ss.population('Germany')`).

    Source: the values for Germany of MEmilio (Kühn MJ et al., *Mathematical Biosciences* 2021,
    https://doi.org/10.1016/j.mbs.2021.108648; `examples/ode_secir_parameter_study_graph.cpp`), which gives each as a
    range, by age; each default here is the middle of its range. MEmilio's transmission is a contact rate times a
    chance of infection per contact, which is 0.06 for adults and from 0.03 (under 5) to 0.175 (80 and over), here
    the relative susceptibility `rel_sus` (1 for adults). Without contacts given, `beta` is 1 a day, which with
    the world's ages gives an R0 of about 3.2, as for the first strains of COVID-19. With a contact matrix,
    `beta` is per contact, and MEmilio's is 0.06: `ssl.COVID(beta=0.06)` with `networks=ss.contacts('Germany')`.

    With `ages=False`, the model has one group, with the values of MEmilio's simple example (`ode_secir_simple.cpp`),
    whose `beta` of 1 a day is one contact a day, each certain to infect; with people infectious at `rel_ins` for 2
    days and at `rel_sym` for 6, this gives an R0 of about 2.7.

    Note: CRN safe as agents.

    Args:
        ages (bool): if True (the default), stratify by age, with the values for each age group; if False, one group
        beta (`ss.per`): the transmission rate: the rate at which someone fully infectious infects others, as contacts a day times the chance that each infects (per contact, with a network)
        rel_sus (float/`ss.by`): the susceptibility of each age group, relative to adults'
        rel_ins (float): the infectiousness of people without symptoms, relative to `beta`
        rel_sym (float): the infectiousness of people with symptoms, relative to `beta`
        dur_exp (`ss.Duration`): the mean time from infection to being infectious
        dur_ins (`ss.Duration`): the mean time without symptoms
        dur_sym (`ss.Duration`/`ss.by`): the mean time with symptoms
        dur_sev (`ss.Duration`/`ss.by`): the mean time with severe disease
        dur_crit (`ss.Duration`/`ss.by`): the mean time with critical disease
        p_sym (float/`ss.by`): the fraction of people without symptoms who go on to have symptoms
        p_sev (float/`ss.by`): the fraction of people with symptoms whose disease becomes severe
        p_crit (float/`ss.by`): the fraction of severe cases that become critical
        p_death (float/`ss.by`): the fraction of critical cases who die
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.COVID(), total_pop=83_000, unit='day', stop=100)
        sim.run()
        sim.covid.plot()
        sim.results.covid.outcome.select(to='dead').by('age').plot() # Deaths by age

        # With Germany's ages and contacts, and MEmilio's chance of infection per contact
        sim = ss.Sim(diseases=ssl.COVID(beta=0.06), networks=ss.contacts('Germany'), people=ss.population('Germany'), total_pop=83_000, unit='day', stop=200)
        ```
    """
    beta     = ss.par(ss.perday(1.0), doc='Transmission rate', source=source)
    rel_sus  = ss.par(by_age(0.5, 1, 1, 1, 1.5, 2.9), doc='Susceptibility, relative to adults', source=source)
    rel_ins  = ss.par(1.0, doc='Infectiousness of people without symptoms, relative to the transmission rate', source=source)
    rel_sym  = ss.par(0.2, doc='Infectiousness of people with symptoms, relative to the transmission rate', source=source)
    dur_exp  = ss.par(ss.days(3.3), doc='Mean time from infection to being infectious', source=source)
    dur_ins  = ss.par(ss.days(1.9), doc='Mean time without symptoms', source=source)
    dur_sym  = ss.par(days(7.0, 7.0, 7.1, 6.9, 6.8, 6.8),    doc='Mean time with symptoms',        source=source)
    dur_sev  = ss.par(days(5.0, 5.0, 5.9, 7.6, 8.5, 11.0),   doc='Mean time with severe disease',  source=source)
    dur_crit = ss.par(days(7.0, 7.0, 6.9, 17.4, 17.1, 11.6), doc='Mean time with critical disease', source=source)
    p_sym    = ss.par(by_age(0.75, 0.75, 0.8, 0.8, 0.8, 0.8),               doc='Fraction of people without symptoms who go on to have symptoms', source=source)
    p_sev    = ss.par(by_age(0.0075, 0.0075, 0.019, 0.0615, 0.165, 0.225),  doc='Fraction of symptomatic cases that become severe',               source=source)
    p_crit   = ss.par(by_age(0.075, 0.075, 0.075, 0.15, 0.30, 0.40),        doc='Fraction of severe cases that become critical',                  source=source)
    p_death  = ss.par(by_age(0.05, 0.05, 0.14, 0.14, 0.40, 0.60),           doc='Fraction of critical cases who die',                             source=source)

    S, E, INS, ISy, ISev, ICr, R = ss.states(7)
    infected = E + INS + ISy + ISev + ICr # Severe and critical cases don't transmit here, so they are infected only if said; prevalence is the fraction of everyone in these

    infection = S >> E @ beta*rel_sus*(rel_ins*INS + rel_sym*ISy)
    onset     = E >> INS @ dur_exp
    symptoms  = INS >> {ISy: p_sym, R: None} @ dur_ins
    severe    = ISy >> {ISev: p_sev, R: None} @ dur_sym
    critical  = ISev >> {ICr: p_crit, R: None} @ dur_sev
    outcome   = ICr >> {ss.dead: p_death, R: None} @ dur_crit

    def __init__(self, ages=True, **kwargs):
        if not ages: # One group, with the values of MEmilio's simple example
            defaults = dict(rel_sus=1.0, rel_ins=0.67, rel_sym=0.25, dur_exp=ss.days(3.2), dur_ins=ss.days(2), dur_sym=ss.days(6), dur_sev=ss.days(12),
                            dur_crit=ss.days(8), p_sym=0.91, p_sev=0.20, p_crit=0.25, p_death=0.30)
            kwargs = defaults | kwargs
        super().__init__(**kwargs)
        if ages:
            self.stratify(age=globals()['ages'])
        return
