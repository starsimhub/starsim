"""
Dengue, with several serotypes and a mosquito vector (`Dengue`)
"""
import numpy as np
import starsim as ss

__all__ = ['Dengue']


def serotype(name, pars):
    """ One dengue serotype: people are susceptible, infectious, or recovered, with mosquitoes of its own """
    p = pars
    S, R = ss.states('S R')
    I = ss.State('I', init=p['init'], label='Infectious')
    Sv = ss.State('Sv', people=False, label='Susceptible mosquitoes')
    Ev = ss.State('Ev', people=False, label='Infected mosquitoes, incubating')
    Iv = ss.State('Iv', people=False, label='Infectious mosquitoes')
    N = (S + I + R).n
    return ss.Disease(S, I, R, Sv, Ev, Iv,
        infection  = S >> I @ p['a']*p['b']*Iv.n/N,
        recovery   = I >> R @ p['dur_inf'],
        biting     = Sv >> Ev @ p['a']*p['c']*I.n/N,
        incubation = Ev >> Iv @ p['eip'],
        emergence  = ss.null >> Sv @ p['m']*p['mu_v']*N,
        mortality  = Sv + Ev + Iv >> ss.null @ p['mu_v'],
        name = name,
    )


class Dengue(ss.Disease):
    """
    Dengue: several serotypes, each passed on by mosquitoes, with partial cross-protection between them

    Each serotype is a disease of its own (`ss.Strains` of members), with the host and vector structure of
    `ssl.Malaria`, plus the extrinsic incubation period in the mosquito. Having had one serotype gives complete
    protection against that one and partial protection against the others (`cross`). Because the serotypes
    protect each other, this runs as agents.

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
        a (`ss.per`): bites on people per mosquito per day
        b (float): the chance that a bite by an infectious mosquito infects a person
        c (float): the chance that a bite on an infectious person infects a mosquito
        m (float): mosquitoes per person
        dur_inf (`ss.dur`): the mean time a person is infectious (viremic)
        eip (`ss.dur`): the extrinsic incubation period: the mean time from a mosquito being infected to being infectious
        mu_v (`ss.per`): the death rate of mosquitoes, which emerge at the same rate
        init (float): the share of people who start infectious with each serotype
        kwargs (dict): passed to `ss.Disease`, e.g. `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Dengue(), demographics=True, n_agents=5000, unit='day', stop=365*5)
        sim.run()
        sim.plot('dengue')
        ```

    Without births, each serotype burns through the people susceptible to it and stops; `demographics=True` gives
    the sim births and deaths, which keep the serotypes returning.
    """
    def __init__(self, n_serotypes=4, cross=0.3, a=ss.perday(0.5), b=0.5, c=0.5, m=2.0,
                 dur_inf=ss.days(6), eip=ss.days(10), mu_v=ss.perday(0.1), init=0.001, name='dengue', **kwargs):
        pars = dict(a=a, b=b, c=c, m=m, dur_inf=dur_inf, eip=eip, mu_v=mu_v, init=init)
        members = [serotype(f'den{i + 1}', pars) for i in range(n_serotypes)]
        protection = np.full((n_serotypes, n_serotypes), cross)
        np.fill_diagonal(protection, 1.0) # Complete protection against the serotype already had
        strains = ss.Strains(members, immunity=ss.immunity(due_to='R', rel_sus=protection))
        super().__init__(name=name, strains=strains, **kwargs)
        return
