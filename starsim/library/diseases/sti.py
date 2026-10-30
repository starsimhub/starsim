"""
Sexually transmitted infections: syphilis (`Syphilis`) and gonorrhea (`Gonorrhea`)

Both pass through a sexual network (`ss.SexualNet`), so they take a transmission probability per act, by
direction (`ss.peract`).
"""
import starsim as ss

__all__ = ['Syphilis', 'Gonorrhea']


class Syphilis(ss.Disease):
    """
    Syphilis, as a staged infection: declared stages, with the infectiousness of latent syphilis written as code

    People who are infected incubate (`E`), then have a primary chancre (`P`), then secondary syphilis (`Se`),
    and then early latent syphilis (`El`), from which some relapse to secondary syphilis and the rest pass to late
    latent syphilis (`Ll`). A share of those develop tertiary syphilis (`T`), which can kill; the rest are no
    longer infectious (`R`).

    The stages are declared, but how infectious a person with latent syphilis is cannot be: it falls with the time
    since their latent stage began, which is a quantity each person carries. That one part is written as code, in
    `step()`, which sets each latent person's `rel_trans`; everything else runs from the declaration. Since it acts
    on each person, this disease runs as agents.

    Source: the natural history and its parameters are those of the syphilis module of STIsim
    (https://github.com/starsimhub/stisim, `stisim/diseases/syphilis.py`), whose own sources are cited there: the
    incubation period, primary and secondary stages from Garnett GP et al. (1997), the definition of early latent
    syphilis as up to two years after infection from WHO Europe, and the share developing tertiary syphilis. The
    half-life of latent infectiousness is STIsim's.

    Nobody is treated here, and the latent and tertiary stages last for years and count towards `prevalence`, so
    the level it settles at is well above what a setting with treatment and screening would show; add
    `ss.screen()` and `ss.treat()` to the sim for that. How far it spreads also depends on the network: the
    partnerships, how long they last, and the acts in them.

    Note: CRN safe as agents.

    Args:
        beta (`ss.peract`): the transmission probability per act, from men to women and from women to men
        dur_exp (`ss.dur`): the mean incubation period
        dur_primary (`ss.dur`): the mean duration of the primary stage
        dur_secondary (`ss.dur`): the mean duration of the secondary stage
        dur_early (`ss.dur`): the mean duration of early latent syphilis
        dur_late (`ss.dur`): the mean time in late latent syphilis before tertiary syphilis, for those who develop it
        p_reactivate (float): the share of people with early latent syphilis who relapse to secondary syphilis
        p_tertiary (float): the share who develop tertiary syphilis
        death_rate (`ss.per`): the rate of dying of tertiary syphilis
        latent_half_life (`ss.dur`): the half-life of infectiousness during latent syphilis
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        net = ss.SexualNet(dur=ss.years(2), n_contacts=ss.poisson(1))
        sim = ss.Sim(diseases=ssl.Syphilis(), networks=net, n_agents=5000, start=2000, stop=2030, dt=ss.months(1))
        sim.run()
        sim.plot('syphilis')
        ```
    """
    beta             = ss.par(ss.peract(m2f=0.1, f2m=0.05), doc='Transmission probability per act, by direction', source='STIsim')
    dur_exp          = ss.par(ss.days(50),    doc='Mean incubation period',                        source='STIsim (Garnett et al. 1997)')
    dur_primary      = ss.par(ss.weeks(6),    doc='Mean duration of the primary stage',            source='STIsim (Garnett et al. 1997)')
    dur_secondary    = ss.par(ss.months(3.6), doc='Mean duration of the secondary stage',          source='STIsim (Garnett et al. 1997)')
    dur_early        = ss.par(ss.months(23),  doc='Mean duration of early latent syphilis',        source='STIsim (WHO Europe: up to two years after infection)')
    dur_late         = ss.par(ss.years(10),   doc='Mean time to tertiary syphilis, for those who develop it', source='STIsim')
    p_reactivate     = ss.par(0.35, doc='Share with early latent syphilis who relapse to the secondary stage', source='STIsim')
    p_tertiary       = ss.par(0.35, doc='Share who develop tertiary syphilis',                     source='STIsim')
    death_rate       = ss.par(ss.peryear(0.05), doc='Rate of dying of tertiary syphilis',          source='STIsim')
    latent_half_life = ss.par(ss.years(1), doc='Half-life of infectiousness during latent syphilis', source='STIsim')

    S, E = ss.states('S E')
    P = ss.State(init=0.01, label='Primary syphilis')
    Se, El, Ll, T, R = ss.states('Se El Ll T R')

    infection = S >> E @ beta*(P + Se + El + Ll) # Latent infectiousness falls with time, in step()
    onset     = E >> P @ dur_exp
    secondary = P >> Se @ dur_primary
    latency   = Se >> El @ dur_secondary
    relapse   = ss.Transition(El, [Se, Ll], dur=dur_early, p=[p_reactivate, None])
    tertiary  = ss.Transition(Ll, [T, R], dur=dur_late, p=[p_tertiary, None])
    death     = ss.death(T, rate=death_rate)

    prevalence = (E + P + Se + El + Ll + T)/(S + E + P + Se + El + Ll + T + R)

    def step(self):
        """ Run the declared transitions, then let each latent person's infectiousness fall with time since their latent stage began """
        out = super().step()
        for state in [self.El, self.Ll]:
            uids = state.uids
            if len(uids):
                self.rel_trans[uids] = 0.5**(state.time[uids]/self.pars.latent_half_life)
        return out


class Gonorrhea(ss.Disease):
    """
    Gonorrhea, as an SIS model on a sexual network

    Gonorrhea leaves no lasting immunity: people clear the infection, by treatment or on their own, and are
    susceptible again at once. Most infections in women go without symptoms and so last much longer than those in
    men, which is why a fuller model gives the duration and the transmission probability by sex.

    Source: the parameters are those of the gonorrhea module of STIsim
    (https://github.com/starsimhub/stisim, `stisim/diseases/gonorrhea.py`), whose own sources are cited there:
    a transmission probability per act from men to women of 0.06, and durations to clearance of a few months to
    about a year, from which `dur_inf` here is a single average.

    Nobody is treated here, so the level gonorrhea settles at is above what a setting with treatment would show,
    and it depends strongly on the network: with few partners and few acts it dies out instead.

    Note: CRN safe as agents.

    Args:
        beta (`ss.peract`): the transmission probability per act, from men to women and from women to men
        dur_inf (`ss.dur`): the mean time to clearance
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        net = ss.SexualNet(dur=ss.years(2), n_contacts=ss.poisson(1))
        sim = ss.Sim(diseases=ssl.Gonorrhea(), networks=net, n_agents=5000, start=2000, stop=2030, dt=ss.months(1))
        sim.run()
        sim.plot('gonorrhea')
        ```
    """
    beta    = ss.par(ss.peract(m2f=0.06, f2m=0.03), doc='Transmission probability per act, by direction', source='STIsim')
    dur_inf = ss.par(ss.months(6), doc='Mean time to clearance', source='STIsim: a few months in men, up to about a year in women')

    S = ss.State()
    I = ss.State(init=0.02, label='Infected')

    infection = S >> I @ beta*I
    clearance = I >> S @ dur_inf

    prevalence = I/(S + I)
