"""
COVID-19 as a SECIR model (`COVID`)
"""
import starsim as ss

__all__ = ['COVID']


class COVID(ss.Disease):
    """
    COVID-19, as a SECIR model: a chain of increasingly severe states, each of which some people leave by recovering

    People who are infected are exposed (`E`), then infected without symptoms (`INS`), and then either recover or
    develop symptoms (`ISy`); of those, some become severe (`ISev`, in hospital), then critical (`ICr`, in
    intensive care), and some of those die. People without symptoms and with symptoms are both infectious, at
    `rel_ins` and `rel_sym` of the contact rate.

    Source: the simple SECIR example of MEmilio (Kühn MJ et al., *Mathematical Biosciences* 2021,
    https://doi.org/10.1016/j.mbs.2021.108648; `examples/ode_secir_simple.cpp`), from which every default here is
    taken: the mean time in each state, the share who recover at each step, and the relative infectiousness of
    people without symptoms and with symptoms.

    This version is not age-structured. Since the severity of COVID-19 rises steeply with age, a real model would
    be: `covid.stratify(age=[0, 20, 60], split=[0.25, 0.5, 0.25], adjust=dict(severe=[0.2, 1, 4]))`, with a
    contact matrix given to the sim (see the user guide on stratification).

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the contact rate (per contact, with a network)
        rel_ins (float): the infectiousness of people without symptoms, relative to the contact rate
        rel_sym (float): the infectiousness of people with symptoms, relative to the contact rate
        dur_exp (`ss.dur`): the mean time from infection to being infectious
        dur_ins (`ss.dur`): the mean time without symptoms
        dur_sym (`ss.dur`): the mean time with symptoms
        dur_sev (`ss.dur`): the mean time with severe disease
        dur_crit (`ss.dur`): the mean time with critical disease
        p_sym (float): the share of people without symptoms who go on to have symptoms
        p_sev (float): the share of people with symptoms whose disease becomes severe
        p_crit (float): the share of severe cases that become critical
        p_death (float): the share of critical cases who die
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.COVID(), total_pop=83_000, unit='day', stop=100)
        sim.run()
        sim.plot('covid')
        ```
    """
    beta     = ss.par(ss.perday(1.0), doc='Contact rate',                                source='MEmilio ode_secir_simple')
    rel_ins  = ss.par(0.67, doc='Infectiousness of people without symptoms, relative to the contact rate', source='MEmilio ode_secir_simple')
    rel_sym  = ss.par(0.25, doc='Infectiousness of people with symptoms, relative to the contact rate',    source='MEmilio ode_secir_simple')
    dur_exp  = ss.par(ss.days(3.2), doc='Mean time from infection to being infectious', source='MEmilio ode_secir_simple')
    dur_ins  = ss.par(ss.days(2),   doc='Mean time without symptoms',                   source='MEmilio ode_secir_simple')
    dur_sym  = ss.par(ss.days(6),   doc='Mean time with symptoms',                      source='MEmilio ode_secir_simple')
    dur_sev  = ss.par(ss.days(12),  doc='Mean time with severe disease',                source='MEmilio ode_secir_simple')
    dur_crit = ss.par(ss.days(8),   doc='Mean time with critical disease',              source='MEmilio ode_secir_simple')
    p_sym    = ss.par(0.91, doc='Share of people without symptoms who go on to have symptoms', source='MEmilio ode_secir_simple')
    p_sev    = ss.par(0.20, doc='Share of symptomatic cases that become severe',               source='MEmilio ode_secir_simple')
    p_crit   = ss.par(0.25, doc='Share of severe cases that become critical',                  source='MEmilio ode_secir_simple')
    p_death  = ss.par(0.30, doc='Share of critical cases who die',                             source='MEmilio ode_secir_simple')

    S, E, INS, ISy, ISev, ICr, R = ss.states('S E INS ISy ISev ICr R')

    infection = S >> E @ beta*(rel_ins*INS + rel_sym*ISy)
    onset     = E >> INS @ dur_exp
    symptoms  = ss.Transition(INS, [ISy, R], dur=dur_ins, p=[p_sym, None])
    severe    = ss.Transition(ISy, [ISev, R], dur=dur_sym, p=[p_sev, None])
    critical  = ss.Transition(ISev, [ICr, R], dur=dur_sev, p=[p_crit, None])
    outcome   = ss.Transition(ICr, [ss.null, R], dur=dur_crit, p=[p_death, None])

    prevalence = (E + INS + ISy + ISev + ICr)/(S + E + INS + ISy + ISev + ICr + R)
