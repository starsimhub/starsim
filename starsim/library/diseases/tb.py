"""
Tuberculosis as the LSHTM spectrum-of-disease natural history (`TB`)
"""
import starsim as ss

__all__ = ['TB']


class TB(ss.Disease):
    """
    Tuberculosis, as the LSHTM model of the spectrum of TB disease

    People who are infected either clear the infection, or develop disease that moves back and forth between
    non-infectious, asymptomatic (infectious, but at `trans_asymp` of a symptomatic person), and symptomatic.
    People who cleared or recovered can be infected again, at their own relative rate. Only symptomatic people die
    of TB; every death is replaced by a birth, so the population stays the same size, as in the source.

    Source: the LSHTM natural history of Schwalb A et al., *PLOS Glob Public Health* 2025, 5(1):e0005050
    (https://doi.org/10.1371/journal.pgph.0005050), in the form of the reference ODE of TBsim
    (`tbsim/compartmental/lshtm_ode.R`), from which every default here is taken: `beta`, `trans_asymp`, the
    reinfection multipliers, the rates between states, and the background death rate. Treatment is off in that
    reference, and is left out here; TBsim adds treatment, drug resistance, and preventive therapy.

    With the reference's `beta` of 9, TB declines slowly over centuries (`R0()` is 0.23, since most infections are
    cleared and few reach disease), reproducing that reference exactly; `beta` is what a model of a particular
    place fits to its data.

    Rates are per year, so a step of a year or a month suits it.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): contacts per person per year (per contact, with a network)
        trans_asymp (float): infectiousness of asymptomatic TB, relative to symptomatic
        rr_reinfect_cleared (float): the rate of reinfection of people who cleared a latent infection, relative to a first infection
        rr_reinfect_rec (float): the same, for people who recovered from disease
        inf_cle, inf_non, inf_asy (`ss.per`): the rates of leaving latent infection, to cleared, non-infectious, and asymptomatic
        non_rec, non_asy (`ss.per`): the rates of leaving non-infectious disease, to recovered and asymptomatic
        asy_non, asy_sym (`ss.per`): the rates of leaving asymptomatic disease, to non-infectious and symptomatic
        sym_asy (`ss.per`): the rate of returning from symptomatic to asymptomatic disease
        sym_dead (`ss.per`): the rate of dying of symptomatic TB
        mu (`ss.per`): the background birth and death rate
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.TB(), total_pop=100_000, start=1900, stop=2000)
        sim.run()
        sim.plot('tb')
        ```
    """
    beta        = ss.par(ss.peryear(9.0), doc='Contacts per person per year', source='TBsim lshtm_ode.R')
    trans_asymp = ss.par(0.82, doc='Infectiousness of asymptomatic TB, relative to symptomatic', source='TBsim lshtm_ode.R')
    rr_reinfect_cleared = ss.par(1.0,  doc='Rate of reinfection after clearing a latent infection, relative to a first infection', source='TBsim lshtm_ode.R')
    rr_reinfect_rec     = ss.par(0.21, doc='Rate of reinfection after recovering from disease, relative to a first infection', source='TBsim lshtm_ode.R')
    inf_cle  = ss.par(ss.peryear(1.90), doc='Rate from latent infection to cleared',           source='TBsim lshtm_ode.R')
    inf_non  = ss.par(ss.peryear(0.16), doc='Rate from latent infection to non-infectious TB', source='TBsim lshtm_ode.R')
    inf_asy  = ss.par(ss.peryear(0.06), doc='Rate from latent infection to asymptomatic TB',   source='TBsim lshtm_ode.R')
    non_rec  = ss.par(ss.peryear(0.18), doc='Rate from non-infectious TB to recovered',        source='TBsim lshtm_ode.R')
    non_asy  = ss.par(ss.peryear(0.25), doc='Rate from non-infectious to asymptomatic TB',     source='TBsim lshtm_ode.R')
    asy_non  = ss.par(ss.peryear(1.66), doc='Rate from asymptomatic to non-infectious TB',     source='TBsim lshtm_ode.R')
    asy_sym  = ss.par(ss.peryear(0.88), doc='Rate from asymptomatic to symptomatic TB',        source='TBsim lshtm_ode.R')
    sym_asy  = ss.par(ss.peryear(0.54), doc='Rate from symptomatic back to asymptomatic TB',   source='TBsim lshtm_ode.R')
    sym_dead = ss.par(ss.peryear(0.34), doc='Rate of dying of symptomatic TB',                 source='TBsim lshtm_ode.R')
    mu       = ss.par(ss.peryear(1/70), doc='Background birth and death rate',                 source='TBsim lshtm_ode.R')

    S, L, C, R, N, A = ss.states('S L C R N A')
    Y = ss.State(init=0.01, label='Symptomatic TB')
    foi = beta*(trans_asymp*A + Y)

    infection    = S >> L @ foi
    reinfect_C   = C >> L @ rr_reinfect_cleared*foi
    reinfect_R   = R >> L @ rr_reinfect_rec*foi
    clearance    = L >> C @ inf_cle
    latent_non   = L >> N @ inf_non
    latent_asy   = L >> A @ inf_asy
    recovery     = N >> R @ non_rec
    worsening    = N >> A @ non_asy
    improvement  = A >> N @ asy_non
    symptoms     = A >> Y @ asy_sym
    remission    = Y >> A @ sym_asy
    tb_death     = ss.death(Y, rate=sym_dead)
    births       = ss.birth(S, rate=mu)
    replacement  = ss.null >> S @ sym_dead*Y.n # Births that replace the TB deaths, so the population is constant
    deaths       = ss.death(rate=mu)

    prevalence = (A + Y)/(S + L + C + R + N + A + Y) # Prevalence of TB disease
