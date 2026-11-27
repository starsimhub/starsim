"""
Polio, with waning gut immunity and routine OPV and IPV (`Polio`)
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['Polio']


age_groups = [0, 0.5, 1, 2, 5, 15, 40] # 0-5 months, 6-11 months, 1, 2-4, 5-14, 15-39, and 40+ years
by_age = lambda *values: ss.by(age=list(values))
famulare = 'Famulare et al. (2018)'
kalkowska = 'Kalkowska et al. (2021), the DEB model for northeast Nigeria'


class Polio(ss.Disease):
    """
    Poliovirus (type 1 by default), with lasting protection from paralysis and gut immunity that wanes

    Immunity to polio has two parts. Protection from paralysis (serum antibodies) lasts, and comes from infection or
    either vaccine. Gut (intestinal) immunity, which stops infection and shedding, comes from infection or oral
    vaccine (OPV), and wanes over years (Famulare et al. 2018). So the states are: never infected (`S`, and `S0` as
    babies, before routine vaccination); shedding, after a first infection (`I`); immune in the gut (`R`); with gut
    immunity waned (`W`), who can be infected again (`Iw`), less often, shedding less and for less long, and without
    paralysis; paralysed (`P`); protected from paralysis by inactivated vaccine (IPV) alone (`V`), whose gut is as
    naive as the never infected; and shedding vaccine virus after OPV (`O`).

    Paralysis comes only from a first infection of someone never immune: a fraction `p_paralysis` of them, 1 in 190
    for type 1. Babies under 6 months are mostly protected from paralysis, though not from infection, by their
    mothers' antibodies (`rel_paralysis`, a tenth as likely). Young children are also the most exposed: adults
    receive about a quarter of the dose that children under 5 do (`rel_sus`). So where transmission is intense, as
    with the R0 of 8 here, polio is endemic, most children are infected by the age of 5, and nearly 90% of
    paralysis is under 5 ("infantile paralysis"). The states start near this endemic state, by age. Polio needs a
    large population to persist: as agents, it dies out in a few thousand.

    Vaccination is routine, and is off by default (`cov_opv` and `cov_ipv` are 0), for polio before vaccines.
    Babies are born into `S0`, never infected and not yet vaccinated, which they leave at a mean age of `age_vax`
    (spread exponentially about it): a fraction `cov_opv*eff_opv` immunized by OPV, then a fraction
    `cov_ipv*eff_ipv` of the rest protected by IPV, and the rest never infected (`S`). OPV is a live virus: the
    vaccinated shed it (`O`) and then have the immunity infection gives (`R`); while shedding, they pass it to the
    never infected, at `rel_inf_opv` of wild virus's infectiousness, which immunizes them too. IPV protects from
    paralysis only (`V`).

    For type 2 or 3, give their ratios of infections to paralytic cases, 1,886 and 1,149 (Nathanson and Kew 2010),
    and their lower R0s, 7.2 and 6.0 (Kalkowska et al. 2021), as `beta` in proportion:
    `ssl.Polio(p_paralysis=1/1886, beta=ss.perday(0.45))` and `ssl.Polio(p_paralysis=1/1149, beta=ss.perday(0.375))`.

    Sources:

    - Famulare M et al., "Assessing the stability of polio eradication after the withdrawal of oral polio
      vaccine", *PLoS Biol* 2018;16(4):e2002468, https://doi.org/10.1371/journal.pbio.2002468. The median
      shedding of wild virus, 43 days in the naive and 8 at the highest immunity, and of Sabin virus, 30 days in
      the naive; gut immunity falling from its peak to that of two doses in about 5 months and 4 years; Sabin
      strains 4-10 times less infectious than wild; and adults receiving 26% of the dose of children under 5.
    - Kalkowska DA et al., "Modeling poliovirus transmission in Borno and Yobe, northeast Nigeria", *Risk Anal*
      2021;41(2):289-302, https://doi.org/10.1111/risa.13485. The differential-equation (DEB) model of Duintjer
      Tebbens, Thompson, et al., with an R0 of 8.0, 7.2, and 6.0 for types 1, 2, and 3, and a take of 0.45 for
      each dose of trivalent OPV against type 1.
    - Nathanson N, Kew OM, "From emergence to eradication: the epidemiology of poliomyelitis deconstructed",
      *Am J Epidemiol* 2010;172(11):1213-1229, https://doi.org/10.1093/aje/kwq320. One paralytic case in 190
      infections with type 1 (Table 1); maternal antibodies do not stop infection but prevent paralysis.
    - World Health Organization, "Polio vaccines: WHO position paper – June 2022", *Wkly Epidemiol Rec*
      2022;97(25):277-300. OPV at 6, 10, and 14 weeks, and IPV at 14 weeks, of age.
    - US CDC, *Epidemiology and Prevention of Vaccine-Preventable Diseases* (the "Pink Book"), chapter on
      poliomyelitis. 90% or more of those given 2 doses of IPV have antibodies.

    Left out: the three serotypes together (one is modeled), the gradual waning (one step, rather than the power law
    of Famulare et al. or the five stages of the DEB model), paralysis by vaccine virus (about 1 in 2-3 million
    doses) and its evolution into circulating vaccine-derived virus, IPV's boost to gut immunity in people
    immunized before, campaigns, seasonality, mixing by age, and deaths from polio. Paralysis is counted at
    infection, and the paralysed don't shed.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate of wild virus (R0 of 8 with the defaults)
        rel_sus (`ss.by`): the susceptibility of each age group, relative to children under 5
        dur_inf (`ss.Duration`): the mean time shedding, after a first infection
        dur_gut (`ss.Duration`): the mean time until gut immunity wanes
        rel_sus_w (float): the susceptibility of people whose gut immunity has waned, relative to the never infected
        rel_inf_w (float): the infectiousness of people infected again, relative to a first infection
        dur_inf_w (`ss.Duration`): the mean time shedding, when infected again
        p_paralysis (float): the fraction of first infections of the never immune that cause paralysis
        rel_paralysis (`ss.by`): the chance of paralysis for each age group, relative to `p_paralysis` (lower under 6 months, from maternal antibodies)
        cov_opv (float): the coverage of routine OPV
        eff_opv (float): the fraction of children given OPV whom it immunizes
        dur_opv (`ss.Duration`): the mean time shedding vaccine virus
        rel_inf_opv (float): the infectiousness of vaccine virus, relative to wild virus
        cov_ipv (float): the coverage of routine IPV
        eff_ipv (float): the fraction of children given IPV whom it protects from paralysis
        age_vax (`ss.Duration`): the mean age at routine vaccination, with OPV and IPV
        birth_rate (`ss.per`): the birth rate, which supplies the never infected
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        pars = dict(total_pop=1_000_000, start=1950, stop=1980)
        sim = ss.Sim(diseases=ssl.Polio(), **pars)
        sim.run()
        sim.results.polio.infection.P.by('age').plot() # Paralysis, mostly under 5

        # With routine OPV for 80% of babies
        vax = ss.Sim(diseases=ssl.Polio(cov_opv=0.8), **pars)
        vax.run()
        print('Paralytic cases:', sim.results.polio.infection.P.sum(), 'without OPV;', vax.results.polio.infection.P.sum(), 'with')
        ```
    """
    dt = ss.weeks(1) # The default step (a week, short beside its infectious periods of two weeks or more), used if the sim is given no dt=

    beta          = ss.par(ss.perday(0.5), doc='Transmission rate of wild virus (R0 of 8)', source=f'{kalkowska}: R0 of 8 for type 1')
    rel_sus       = ss.par(by_age(1, 1, 1, 1, 0.6, 0.26, 0.26), doc='Susceptibility, relative to children under 5', source=f'{famulare}: adults receive 26% of the dose of children under 5; 5-14 illustrative')
    dur_inf       = ss.par(ss.days(43), doc='Mean time shedding, after a first infection', source=f'{famulare}: median of 43 days for wild virus')
    dur_gut       = ss.par(ss.years(5), doc='Mean time until gut immunity wanes', source=f'{famulare}: to two-dose immunity in about 5 months and 4 years')
    rel_sus_w     = ss.par(0.5, doc='Susceptibility once gut immunity has waned', source=f'{famulare}: below half at two-dose immunity')
    rel_inf_w     = ss.par(0.1, doc='Infectiousness when infected again', source=f'Illustrative; {famulare}: 10 times less virus shed for each 8-fold rise in immunity')
    dur_inf_w     = ss.par(ss.days(15), doc='Mean time shedding, when infected again', source=f'Illustrative, between the medians of {famulare}, 43 days (naive) and 8 (most immune)')
    p_paralysis   = ss.par(1/190, doc='Fraction of first infections that cause paralysis', source='Nathanson and Kew (2010): 1 in 190 for type 1')
    rel_paralysis = ss.par(by_age(0.1, 1, 1, 1, 1, 1, 1), doc='Chance of paralysis, relative to p_paralysis', source='Illustrative; Nathanson and Kew (2010): maternal antibodies prevent paralysis')
    cov_opv       = ss.par(0.0, doc='Coverage of routine OPV', source='0 for polio before vaccines')
    eff_opv       = ss.par(0.83, doc='Fraction of children given OPV whom it immunizes', source=f'{kalkowska}: 0.45 a dose, over 3 doses')
    dur_opv       = ss.par(ss.days(30), doc='Mean time shedding vaccine virus', source=f'{famulare}: median of 30 days for Sabin virus')
    rel_inf_opv   = ss.par(0.15, doc='Infectiousness of vaccine virus, relative to wild', source=f'{famulare}: 4 to 10 times less infectious')
    cov_ipv       = ss.par(0.0, doc='Coverage of routine IPV', source='0 for polio before vaccines')
    eff_ipv       = ss.par(0.9, doc='Fraction of children given IPV whom it protects', source='US CDC Pink Book: 90% after 2 doses')
    age_vax       = ss.par(ss.days(100), doc='Mean age at routine vaccination (about 14 weeks)', source='WHO (2022): OPV at 6, 10, and 14 weeks, and IPV at 14 weeks')
    birth_rate    = ss.par(ss.peryear(0.03), doc='Birth rate', source='Illustrative')
    death_rate    = ss.par(ss.peryear(0.03), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S  = ss.State('Never infected')
    S0 = ss.State('Never infected, before routine vaccination') # Where babies are born
    I  = ss.State('Shedding', init=by_age(0.05, 0.05, 0.03, 0.01, 0.002, 0, 0), infected=True) # Starting near the endemic state
    R  = ss.State('Gut immune', init=by_age(0.2, 0.35, 0.5, 0.6, 0.55, 0.35, 0.3))
    W  = ss.State('Gut immunity waned', init=by_age(0, 0.05, 0.1, 0.25, 0.4, 0.65, 0.7))
    Iw = ss.State('Shedding again', infected=True)
    P  = ss.State('Paralysed')
    V  = ss.State('IPV only')
    O  = ss.State('Shedding OPV')

    wild = beta*rel_sus*(I + rel_inf_w*Iw)
    infection   = (S0 + S) >> {P: p_paralysis*rel_paralysis, I: None} @ wild
    infection_v = V >> I @ wild # Protected from paralysis
    reinfection = W >> Iw @ rel_sus_w*wild
    recovery    = I >> R @ dur_inf
    recovery_w  = Iw >> R @ dur_inf_w
    waning      = R >> W @ dur_gut
    routine     = S0 >> {O: cov_opv*eff_opv, V: (1 - cov_opv*eff_opv)*cov_ipv*eff_ipv, S: None} @ age_vax
    opv_spread  = (S0 + S) >> O @ beta*rel_sus*rel_inf_opv*O
    opv_immune  = O >> R @ dur_opv
    births      = ss.birth(S0, rate=birth_rate)
    deaths      = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify(age=age_groups)
        return
