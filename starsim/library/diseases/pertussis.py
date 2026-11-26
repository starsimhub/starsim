"""
Pertussis, or whooping cough, as an SEIRWS model with immune boosting (`Pertussis`)
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['Pertussis']


age_groups = [0, 2/12, 4/12, 6/12, 1, 5, 10, 15, 20, 40] # Fine under a year, where deaths are and the DTP series is given
by_age = lambda *values: ss.by(age=list(values))
wr = 'Wearing and Rohani (2009), Table 1'
lavine = 'Lavine, King, and Bjørnstad (2011), Table 1'


class Pertussis(ss.Disease):
    """
    Pertussis (whooping cough), as an SEIRWS model by age: immunity wanes, and is boosted by exposure before it is lost

    People who are infected are exposed (`E`), then infectious (`I`), then immune (`R`). Immunity wanes in two
    stages of equal length: from `R` to `W` (waned: still protected, but boostable), and from `W` back to
    susceptible (`S`). People in `W` who are exposed are boosted back to `R` without being infected, at `kappa`
    times the force of infection, so immunity lasts longer where pertussis circulates more. This is the SIRWS
    model of Lavine et al. (2011), with the latent and infectious periods, R0, and background force of infection
    (`import_rate`, which keeps pertussis from dying out between epidemics) of Wearing and Rohani (2009).

    The disease is stratified by age, finely under a year (0, 2, 4, and 6 months, 1, 5, 10, 15, 20, and 40
    years), since deaths are concentrated in young infants: `p_death` is the fraction of infections that are
    fatal, by age, and these deaths are `outcome.dead`, by age.

    The primary DTP series (doses at about 2, 4, and 6 months) is given as babies leave `S0`, where they are
    born, susceptible and not yet vaccinated. They leave it at a mean age of `age_dtp`, 6 months, though spread
    exponentially about it (as compartments and as agents alike), so that some leave much sooner and some later:
    `cov_dtp*eff_dtp` of them to the vaccinated (`V`), and the rest to the susceptible (`S`). The vaccinated wane
    into `W` sooner than natural immunity does, as with acellular vaccine, and can then be boosted. `cov_dtp` is 0
    by default, the pre-vaccine disease. Run stochastically (e.g. `solver='binomial'`), it has epidemics every 3
    to 5 years; as an ODE, these damp towards an equilibrium. With vaccination, there are fewer infections and
    deaths, the epidemics are further apart, and infections move to older children and adults.

    Left out: the lower infectiousness of reinfections that Wearing and Rohani explore (1 in their basic model,
    as here), seasonal forcing, contacts by age (people mix at random), maternal antibodies and vaccination in
    pregnancy, boosters after the primary series, protection from incomplete series, and the gamma-distributed
    periods of both papers. Infections here are all infections, most of them in older children and adults mild
    and unreported.

    Sources:

    - Lavine JS et al., "Natural immune boosting in pertussis dynamics and the potential for long-term vaccine
      failure", *PNAS* 2011;108(17):7259-7264, https://doi.org/10.1073/pnas.1014394108. The SIRWS
      structure, and a boosting coefficient `kappa` of 20 (with a mean duration of immunity of 10 years without
      boosting, where this model has the 30 of Wearing and Rohani).
    - Rozhnova G, Nunes A, "Modelling the long-term dynamics of pre-vaccination pertussis", *J R Soc Interface*
      2012;9(76):2959-2970, https://doi.org/10.1098/rsif.2012.0432. The waning rates of the SIRWS model, from
      R to W and from W to S, each twice the rate of loss of immunity.
    - Wearing HJ, Rohani P, "Estimating the duration of pertussis immunity using epidemiological signatures",
      *PLoS Pathog* 2009;5(10):e1000647, https://doi.org/10.1371/journal.ppat.1000647. A latent period of 8
      days, an infectious period of 15 days, a pre-vaccine birth rate of 0.012 to 0.028 per year, a
      background force of infection of 50 per million per year, an R0 of 11 to 15, and a mean duration of
      natural immunity of 30 to 80 years in their models with boosting or milder reinfections.
    - Yeung KHT et al., "An update of the global burden of pertussis in children younger than 5 years: a
      modelling study", *Lancet Infect Dis* 2017;17(9):974-980, https://doi.org/10.1016/S1473-3099(17)30390-0.
      85,900 deaths in 5.1 million cases in infants (1.7%), and 74,800 in 19.0 million at 1 to 4 years (0.4%).
    - Zhang L et al., "Acellular vaccines for preventing whooping cough in children", *Cochrane Database Syst
      Rev* 2014;(9):CD001478, https://doi.org/10.1002/14651858.CD001478.pub6. An
      efficacy of 84% to 85% against typical whooping cough for multi-component acellular vaccines.
    - Wendelboe AM et al., "Duration of immunity against pertussis after natural infection or vaccination",
      *Pediatr Infect Dis J* 2005;24(5 Suppl):S58-S61, https://doi.org/10.1097/01.inf.0000160914.59160.41.
      Immunity after infection wanes after 4 to 20 years, and after vaccination after 4 to 12.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (R0 of 13 with the default infectious period)
        kappa (float): the boosting coefficient: the rate at which people in `W` are boosted, relative to the force of infection
        dur_exp (`ss.Duration`): the mean latent period
        dur_inf (`ss.Duration`): the mean infectious period
        dur_imm (`ss.Duration`): the mean duration of natural immunity without boosting, half in `R` and half in `W`
        dur_dtp (`ss.Duration`): the mean time protected by vaccine before waning into `W`
        p_death (float/`ss.by`): the fraction of infections that are fatal, by age
        cov_dtp (float): the coverage of the primary DTP series
        eff_dtp (float): the fraction of those vaccinated who are protected
        age_dtp (`ss.Duration`): the mean age at which the DTP series is completed
        import_rate (`ss.per`): the background force of infection, from outside the population, which keeps pertussis from dying out between epidemics
        birth_rate (`ss.per`): the birth rate
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        # Pre-vaccine pertussis, with epidemics every few years
        sim = ss.Sim(diseases=ssl.Pertussis(), total_pop=500_000, solver='binomial', start='1920-01-01', stop='1960-01-01', dt=ss.days(5))
        sim.run()
        sim.results.pertussis.infection.plot()

        # With 80% coverage of the DTP series: fewer cases, and older
        sim = ss.Sim(diseases=ssl.Pertussis(cov_dtp=0.8), total_pop=500_000, solver='binomial', start='1920-01-01', stop='1960-01-01', dt=ss.days(5))
        sim.run()
        sim.results.pertussis.infection.by('age').plot()
        sim.results.pertussis.outcome.select(to='dead').by('age').plot() # Deaths by age
        ```
    """
    beta       = ss.par(ss.perday(13/15), doc='Transmission rate (R0 of 13)', source='Wearing and Rohani (2009): R0 of 11 to 15')
    kappa      = ss.par(20.0, doc='Boosting coefficient: the rate of boosting relative to the force of infection', source=lavine)
    dur_exp    = ss.par(ss.days(8),   doc='Mean latent period',     source=wr)
    dur_inf    = ss.par(ss.days(15),  doc='Mean infectious period', source=wr)
    dur_imm    = ss.par(ss.years(30), doc='Mean duration of natural immunity without boosting, half in R and half in W', source='Wearing and Rohani (2009): 30 to 80 years, with boosting or milder reinfections; in two equal stages, as Rozhnova and Nunes (2012)')
    dur_dtp    = ss.par(ss.years(8),  doc='Mean time protected by vaccine before waning into W', source='Illustrative: within the 4 to 12 years of protection after vaccination of Wendelboe et al. (2005), and shorter than the 15 years natural immunity lasts before W')
    p_death    = ss.par(by_age(0.05, 0.02, 0.01, 0.005, 0.004, 0, 0, 0, 0, 0), doc='Fraction of infections that are fatal', source='Illustrative under a year, concentrated in young infants and averaging about 1.7% (Yeung et al. 2017: 1.7% under a year); 0.4% at 1 to 4 years (Yeung et al. 2017); 0 from 5 years')
    cov_dtp    = ss.par(0.0,  doc='Coverage of the primary DTP series', source='0 for the pre-vaccine disease')
    eff_dtp    = ss.par(0.85, doc='Fraction of those vaccinated who are protected', source='Zhang et al. (2014): 84% to 85% for multi-component acellular vaccines')
    age_dtp    = ss.par(ss.years(0.5), doc='Mean age at which the DTP series is completed', source='Doses at about 2, 4, and 6 months')
    import_rate = ss.par(ss.peryear(50e-6), doc='Background force of infection, from outside the population', source='Wearing and Rohani (2009): 50 per million per year')
    birth_rate = ss.par(ss.peryear(0.02), doc='Birth rate',                   source='Wearing and Rohani (2009): 0.012 to 0.028 per year')
    death_rate = ss.par(ss.peryear(0.02), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S  = ss.State()
    S0 = ss.State(label='Susceptible infant, before the DTP series') # Where babies are born
    E  = ss.State(infected=True) # Counted in the results infected and prevalence
    I  = ss.State(init=0.0001, label='Infectious', infected=True)
    R  = ss.State(label='Immune')
    W  = ss.State(label='Waned (boostable)')
    V  = ss.State(label='Vaccinated')

    infection   = (S0 + S) >> E @ beta*I
    importation = (S0 + S) >> E @ import_rate
    onset       = E >> I @ dur_exp
    outcome     = I >> {ss.dead: p_death, R: None} @ dur_inf
    waning      = R >> W @ (2/dur_imm) # Two stages of equal length
    loss        = W >> S @ (2/dur_imm)
    boosting    = W >> R @ kappa*beta*I
    routine     = S0 >> {V: cov_dtp*eff_dtp, S: None} @ age_dtp
    waning_dtp  = V >> W @ dur_dtp
    births      = ss.birth(S0, rate=birth_rate)
    deaths      = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify(age=age_groups)
        return
