"""
Dengue, with several serotypes and a mosquito vector (`Dengue`, a set of `DengueSerotype`)
"""
# Authorship: AI written, human reviewed

import starsim as ss

__all__ = ['DengueSerotype', 'Dengue']


class DengueSerotype(ss.Disease):
    """
    One dengue serotype: people are susceptible, infectious, or recovered, with mosquitoes of its own

    The host and vector structure is that of `ssl.Malaria`, plus the extrinsic incubation period in the mosquito.
    Having recovered from any of the other serotypes (`others=`, by name) protects in part (`cross`), a modifier of
    this serotype's `rel_sus`. `ssl.Dengue` makes the serotypes together, each protecting against the others.

    Note: CRN safe as agents.

    Args:
        others (list): the names of the other serotypes, recovery from which protects in part
        cross (float): the protection against this serotype after recovering from one of the others (1 is complete)
        a (`ss.per`): bites on people per mosquito per day
        b (float): the chance that a bite by an infectious mosquito infects a person
        c (float): the chance that a bite on an infectious person infects a mosquito
        m (float): mosquitoes per person
        dur_inf (`ss.Duration`): the mean time a person is infectious (viremic)
        eip (`ss.Duration`): the extrinsic incubation period: the mean time from a mosquito being infected to being infectious
        mu_v (`ss.per`): the death rate of mosquitoes, which emerge at the same rate
        kwargs (dict): passed to `ss.Disease`, e.g. `init_prev` or `name`
    """
    a       = ss.par(ss.perday(0.5), doc='Bites on people per mosquito per day')
    b       = ss.par(0.5,            doc='Chance that a bite by an infectious mosquito infects a person')
    c       = ss.par(0.5,            doc='Chance that a bite on an infectious person infects a mosquito')
    m       = ss.par(2.0,            doc='Mosquitoes per person')
    dur_inf = ss.par(ss.days(6),     doc='Mean time a person is infectious (viremic)')
    eip     = ss.par(ss.days(10),    doc='Extrinsic incubation period')
    mu_v    = ss.par(ss.perday(0.1), doc='Death rate of mosquitoes, which emerge at the same rate')

    S, R = ss.states('S R')
    I  = ss.State(label='Infectious')
    Sv = ss.State(people=False, label='Susceptible mosquitoes')
    Ev = ss.State(people=False, label='Infected mosquitoes, incubating')
    Iv = ss.State(people=False, label='Infectious mosquitoes')

    infection  = S >> I @ a*b*Iv.n/(S + I + R).n
    recovery   = I >> R @ dur_inf
    biting     = Sv >> Ev @ a*c*I.n/(S + I + R).n
    incubation = Ev >> Iv @ eip
    emergence  = ss.unborn >> Sv @ m*mu_v*(S + I + R).n
    mortality  = Sv + Ev + Iv >> ss.dead @ mu_v

    def __init__(self, others=(), cross=0.3, **kwargs):
        if others: # The strongest protection from any serotype had, which is the same one from each
            had = ss.ref(f'{others[0]}.R')
            for other in others[1:]:
                had = had | ss.ref(f'{other}.R')
            kwargs['cross_protection'] = ss.modify('rel_sus', multiply=1 - cross, who=had)
        super().__init__(**kwargs)
        return


class Dengue(ss.ndict):
    """
    Dengue: several serotypes, each passed on by mosquitoes, with partial cross-protection between them

    Each serotype is a disease of its own (`ssl.DengueSerotype`), `den1` to `den4`, so this is a set of diseases,
    given to a sim as one (`ss.Sim(diseases=ssl.Dengue())`), and the results are `sim.results.den1` and so on.
    Having had one serotype gives complete protection against that one and partial protection against the others
    (`cross`), a modifier of each serotype's `rel_sus`.

    Sources: the host-vector structure is the Ross-Macdonald model as given by Keeling MJ, Rohani P, *Modeling
    Infectious Diseases in Humans and Animals*, Princeton University Press, 2008, section 4.4 (see `ssl.Malaria`).
    The dengue values are indicative ones from the literature rather than one source's fitted set: about 6 days of
    viremia in people and an extrinsic incubation period of about 10 days in *Aedes aegypti* at tropical
    temperatures, with a mosquito lifetime of about 10 days. That infection with one serotype gives lasting
    protection against it and temporary, partial protection against the others is the finding of Reich NG et al.,
    "Interactions between serotypes of dengue highlight epidemiological impact of cross-immunity", *J R Soc
    Interface* 2013;10(86):20130414 (https://doi.org/10.1098/rsif.2013.0414); here that cross-protection does not
    wane, which is a simplification, and its size (`cross`) is illustrative.

    Note: CRN safe as agents.

    Args:
        n_serotypes (int): how many serotypes (up to 4)
        cross (float): the protection against the other serotypes after recovering from one (1 is complete)
        init_prev (float): the share of people who start infected (infectious) with each serotype
        kwargs (dict): the parameters of each serotype (see `ssl.DengueSerotype`), e.g. `m=3`, and anything else for `ss.Disease`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Dengue(), demographics=True, n_agents=5000, unit='day', stop=365*5)
        sim.run()
        sim.den1.plot()
        ```

    Without births, each serotype burns through the people susceptible to it and stops; `demographics=True` gives
    the sim births and deaths, which keep the serotypes returning.
    """
    def __init__(self, n_serotypes=4, cross=0.3, init_prev=0.001, **kwargs):
        if not 1 <= n_serotypes <= 4:
            errormsg = f'Dengue has 1 to 4 serotypes, not {n_serotypes}'
            raise ValueError(errormsg)
        names = [f'den{i + 1}' for i in range(n_serotypes)]
        super().__init__(*[DengueSerotype(others=[other for other in names if other != name], cross=cross, init_prev=init_prev, name=name, **kwargs) for name in names])
        return
