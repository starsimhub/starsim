"""
Respiratory syncytial virus (`RSV`), as Pitzer et al.'s model of repeated infections
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['RSV']


age_groups = [0, 2/12, 4/12, 6/12, 1, 2, 5, 20, 40, 65] # 0-2, 2-4, 4-6, and 6-12 months; 1, 2-4, 5-19, 20-39, 40-64, and 65+ years
by_age = lambda *values: ss.by(age=list(values))
pitzer = 'Pitzer et al. (2015), Table 2'


class RSV(ss.Disease):
    """
    RSV, by age: maternal protection, then repeated infections, each less likely, shorter, and less infectious than the last

    Babies are born protected by maternal antibodies (`M`), which wane within a few months. Then they are
    susceptible (`S0`) to a first infection (`I1`), after which they are partly immune (`S1`), and so on: each
    infection leaves people less susceptible to the next, which is shorter and less infectious, up to a third
    infection (`S3`), after which every infection (`I4`) leaves them as they were (`S3`). Transmission follows a
    yearly cycle, peaking in winter. This is the structure of Pitzer et al.'s model of RSV in the United States,
    with their values; most infections are mild reinfections.

    First infections are the severe ones, and most happen in the first year of life: a fraction of each age group
    is hospitalized (`H1`), most of all babies under 2 months. Later in life, reinfections are mild, except in
    older adults, a fraction of whose infections (`I4`) lead to hospital (`H4`). Some of those in hospital die
    (results by age, e.g. `sim.results.rsv.recovery1.select(to='H1').by('age')` and
    `sim.results.rsv.discharge4.select(to='dead').by('age')`).

    Babies can be immunized, neither by default: by a maternal vaccine (`cov_matvax`, e.g. RSVpreF, given in
    pregnancy) or by nirsevimab (`cov_mab`, a long-acting monoclonal antibody given at birth). A baby gets one or
    the other, so the two coverages add up to at most 1. Immunized babies (`Vmat`, `Vmab`) are protected as other
    babies are by maternal antibodies, but lose that protection at only `1 - eff` the rate, until the immunization
    wanes (after about `dur_matvax` or `dur_mab`); so the efficacy is relative to babies who were not immunized,
    as in the trials. Since this only delays first infections, to ages when they are less severe, it prevents
    fewer of the hospitalizations of the first year than the efficacy: about a third of those under 6 months, with
    every baby given nirsevimab.

    With these values, about 90% of children are infected by 2 years of age, the hospitalizations of children
    are near those that Hall et al. (2013) found, and those of older adults near those of Surie et al. (2023).
    Everyone mixes at random, so adults, who are mostly in `S3`, are reinfected about once a year; with a contact
    matrix (`networks=ss.contacts(...)`), transmission is more among children, as in Pitzer et al. As agents,
    RSV fades out between seasons in a small population (under about 20,000 people).

    Left out: Pitzer et al.'s contact matrix, their monthly age groups under 1 year, the lower respiratory tract
    disease of second infections, the climate drivers of the seasonal cycle, giving nirsevimab before the season
    rather than at birth, and vaccines for older adults.

    Sources:

    - Pitzer VE et al., "Environmental drivers of the spatiotemporal dynamics of respiratory syncytial virus in the
      United States", *PLoS Pathog* 2015;11(1):e1004591, https://doi.org/10.1371/journal.ppat.1004591. The
      structure; the duration of maternal immunity (16 weeks); the infectious periods (10, 7, and 5 days); the
      relative susceptibility after one, two, and three infections (0.76, 0.6, 0.4); the relative infectiousness of
      second and later infections (0.75, 0.51); and an R0 of about 9 (here with random mixing). Their fraction of
      first infections hospitalized (the fraction with lower respiratory tract disease, 0.5 under 6 months to 0.1
      at 2 years and over, times the 3-7% of those hospitalized) is about the same as `p_hosp1` here, which falls
      more steeply over the first 6 months.
    - Hall CB et al., "Respiratory syncytial virus-associated hospitalizations among children less than 24 months
      of age", *Pediatrics* 2013;132(2):e341-e348, https://doi.org/10.1542/peds.2013-0303. Hospitalizations of
      17.9 per 1,000 children a year at 2 months of age and under, and 5.2 under 2 years, which `p_hosp1` is
      chosen to give.
    - Weber A et al., "Modeling epidemics caused by respiratory syncytial virus (RSV)", *Math Biosci*
      2001;172(2):95-113, https://doi.org/10.1016/S0025-5564(01)00066-9. The cosine form of the seasonal cycle
      (also Pitzer et al.'s); its amplitude and peak here are illustrative.
    - Kampmann B et al., "Bivalent prefusion F vaccine in pregnancy to prevent RSV illness in infants", *N Engl J
      Med* 2023;388(16):1451-1464, https://doi.org/10.1056/NEJMoa2216480. Efficacy against severe medically
      attended disease of 81.8% within 90 days of birth and 69.4% within 180 days.
    - Drysdale SB et al., "Nirsevimab for prevention of hospitalizations due to RSV in infants", *N Engl J Med*
      2023;389(26):2425-2435, https://doi.org/10.1056/NEJMoa2309189. Efficacy against hospitalization of 83.2%
      over the first season.
    - Surie D et al., "Disease severity of respiratory syncytial virus compared with COVID-19 and influenza among
      hospitalized adults aged ≥60 years", *MMWR* 2023;72(40):1083-1088, https://doi.org/10.15585/mmwr.mm7240a2.
      In the United States, 60,000-160,000 hospitalizations and 6,000-10,000 deaths a year from RSV among adults
      65 and over (about 1-3 hospitalizations per 1,000 a year, and 6-10% of those hospitalized dying).

    Note: CRN safe as agents.

    Args:
        beta (`ss.seasonal`/`ss.per`): the transmission rate over the year (for a first infection), as a cycle around its mean
        rel_sus1 (float): the susceptibility after one infection, relative to before
        rel_sus2 (float): the susceptibility after two infections
        rel_sus3 (float): the susceptibility after three or more infections
        rel_inf2 (float): the infectiousness of second infections, relative to first infections
        rel_inf3 (float): the infectiousness of third and later infections
        dur_mat (`ss.Duration`): the mean time babies are protected by maternal antibodies
        dur_inf1 (`ss.Duration`): the mean time infectious in a first infection
        dur_inf2 (`ss.Duration`): the mean time infectious in a second infection
        dur_inf3 (`ss.Duration`): the mean time infectious in a third or later infection
        p_hosp1 (`ss.by`): the fraction of first infections hospitalized, by age
        p_hosp4 (`ss.by`): the fraction of fourth and later infections hospitalized, by age
        dur_hosp (`ss.Duration`): the mean time in hospital
        p_death (`ss.by`): the fraction of those in hospital who die, by age
        cov_matvax (float): the fraction of babies whose mothers were vaccinated in pregnancy
        eff_matvax (float): the efficacy of the maternal vaccine: how much more slowly those babies lose their protection
        dur_matvax (`ss.Duration`/`ss.Dist`): the time the maternal vaccine protects
        cov_mab (float): the fraction of babies given nirsevimab at birth
        eff_mab (float): the efficacy of nirsevimab: how much more slowly those babies lose their protection
        dur_mab (`ss.Duration`/`ss.Dist`): the time nirsevimab protects
        birth_rate (`ss.per`): the birth rate
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.RSV(), total_pop=100_000, start='2000-01-01', stop='2015-01-01')
        sim.run()
        sim.rsv.plot()
        sim.results.rsv.recovery1.select(to='H1').by('age').plot() # Hospitalizations with a first infection, by age

        # Half of babies given nirsevimab, and a quarter of mothers vaccinated in pregnancy
        immunized = ss.Sim(diseases=ssl.RSV(cov_mab=0.5, cov_matvax=0.25), total_pop=100_000, start='2000-01-01', stop='2015-01-01')
        immunized.run()
        ```
    """
    dt = ss.days(1) # The default step (a day, for an infectious period of under a week), used if the sim is given no dt=

    beta     = ss.par(ss.seasonal(mean=ss.perday(0.9), amplitude=0.1, peak='November'),
                      doc='Transmission rate over the year, as a cycle around a mean giving R0 of 9', source=f'{pitzer}, for R0; Weber et al. (2001), for the form; the amplitude and peak are illustrative, for epidemics that peak in January')
    rel_sus1 = ss.par(0.76, doc='Susceptibility after one infection',            source=pitzer)
    rel_sus2 = ss.par(0.6,  doc='Susceptibility after two infections',           source=pitzer)
    rel_sus3 = ss.par(0.4,  doc='Susceptibility after three or more infections', source=pitzer)
    rel_inf2 = ss.par(0.75, doc='Infectiousness of second infections',           source=pitzer)
    rel_inf3 = ss.par(0.51, doc='Infectiousness of third and later infections',  source=pitzer)
    dur_mat  = ss.par(ss.days(112), doc='Mean time protected by maternal antibodies (16 weeks)', source=pitzer)
    dur_inf1 = ss.par(ss.days(10),  doc='Mean time infectious in a first infection',          source=pitzer)
    dur_inf2 = ss.par(ss.days(7),   doc='Mean time infectious in a second infection',         source=pitzer)
    dur_inf3 = ss.par(ss.days(5),   doc='Mean time infectious in a third or later infection', source=pitzer)
    p_hosp1  = ss.par(by_age(0.04, 0.02, 0.01, 0.007, 0.007, 0.005, 0.005, 0.005, 0.005, 0.005),
                      doc='Fraction of first infections hospitalized', source='Chosen to give the hospitalizations by age of Hall et al. (2013), near those of Pitzer et al. (2015)')
    p_hosp4  = ss.par(by_age(0, 0, 0, 0, 0, 0, 0, 0, 0.0002, 0.002),
                      doc='Fraction of fourth and later infections hospitalized', source='Chosen to give about the hospitalizations of older adults of Surie et al. (2023); illustrative for those 40-64')
    dur_hosp = ss.par(ss.days(5), doc='Mean time in hospital', source='Illustrative')
    p_death  = ss.par(by_age(0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.01, 0.03, 0.08),
                      doc='Fraction of those in hospital who die', source='Surie et al. (2023), for those 65 and over; illustrative for the rest')
    cov_matvax   = ss.par(0.0, doc='Fraction of babies whose mothers were vaccinated in pregnancy', source='None by default, for RSV before immunization')
    eff_matvax   = ss.par(0.7, doc='Efficacy of the maternal vaccine', source='Kampmann et al. (2023), within 180 days of birth')
    dur_matvax   = ss.par(ss.erlang(k=4, mean=ss.days(180)), doc='Time the maternal vaccine protects', source='Kampmann et al. (2023), who followed babies for 180 days')
    cov_mab      = ss.par(0.0, doc='Fraction of babies given nirsevimab at birth', source='None by default, for RSV before immunization')
    eff_mab      = ss.par(0.8, doc='Efficacy of nirsevimab', source='Drysdale et al. (2023), against hospitalization')
    dur_mab      = ss.par(ss.erlang(k=4, mean=ss.days(150)), doc='Time nirsevimab protects', source='Drysdale et al. (2023), over a season')
    birth_rate   = ss.par(ss.peryear(0.02), doc='Birth rate',                   source='Illustrative')
    death_rate   = ss.par(ss.peryear(0.02), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S0, I1, H1, S1, I2, S2, I3, S3, I4, H4, M, Vmat, Vmab = ss.states(13)
    infected = I1 + I2 + I3 + I4
    init = dict(M=by_age(1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0), S3=by_age(0, 0, 0, 0, 0, 0.5, 0.99, 0.99, 0.99, 0.99), I4=by_age(0, 0, 0, 0, 0, 0, 0.01, 0.01, 0.01, 0.01)) # Near the endemic state

    _foi       = beta*(I1 + rel_inf2*I2 + rel_inf3*(I3 + I4)) # The force of infection (with an underscore, not a result)
    infection1 = S0 >> I1 @ _foi
    infection2 = S1 >> I2 @ rel_sus1*_foi
    infection3 = S2 >> I3 @ rel_sus2*_foi
    infection4 = S3 >> I4 @ rel_sus3*_foi
    recovery1  = I1 >> {H1: p_hosp1, S1: None} @ dur_inf1
    recovery2  = I2 >> S2 @ dur_inf2
    recovery3  = I3 >> S3 @ dur_inf3
    recovery4  = I4 >> {H4: p_hosp4, S3: None} @ dur_inf3
    discharge1 = H1 >> {ss.dead: p_death, S1: None} @ dur_hosp
    discharge4 = H4 >> {ss.dead: p_death, S3: None} @ dur_hosp
    maternal   = M >> S0 @ dur_mat
    breakthrough_matvax = Vmat >> S0 @ (1 - eff_matvax)/dur_mat # Immunized babies lose their protection more slowly
    breakthrough_mab    = Vmab >> S0 @ (1 - eff_mab)/dur_mat
    waning_matvax = Vmat >> S0 @ dur_matvax
    waning_mab    = Vmab >> S0 @ dur_mab
    births     = ss.birth(M, rate=birth_rate*(1 - cov_matvax - cov_mab))
    births_mat = ss.birth(Vmat, rate=birth_rate*cov_matvax)
    births_mab = ss.birth(Vmab, rate=birth_rate*cov_mab)
    deaths     = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cov_matvax, cov_mab = self.pars.cov_matvax, self.pars.cov_mab
        if cov_matvax + cov_mab > 1:
            raise ValueError(f'A baby gets one immunization or the other, so cov_matvax + cov_mab must be at most 1, not {cov_matvax.raw + cov_mab.raw:g}')
        self.stratify(age=age_groups)
        return
