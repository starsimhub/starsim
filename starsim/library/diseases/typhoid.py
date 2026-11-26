"""
Typhoid, after Pitzer et al. (2014), with typhoid conjugate vaccine (`Typhoid`)
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['Typhoid']


age_groups = [0, 2, 5, 10, 15, 25, 35, 45]
by_age = lambda *values: ss.by(age=list(values))
source = 'Pitzer et al. (2014), Table 1'


class Typhoid(ss.Disease):
    """
    Typhoid fever, after Pitzer et al. (2014): clinical and subclinical infections, chronic carriers, and water-borne spread

    People are fully susceptible (`S1`; babies are in `S0` until routine vaccination, below). A first infection
    (`I1`) is the one that can be clinical typhoid (about 1 in 10 are). After 4 weeks, a few people die, a fraction
    that rises with age become chronic carriers (`C`), who shed for life, and the rest are immune (`R`). Full
    immunity lasts about 2 years; then people are partially susceptible (`S2`): immune to clinical disease, but
    they can be infected again without symptoms (`I2`), after which they are immune again or become carriers.
    Subclinical infections and carriers are a hundredth as infectious as a first infection (`rel_carrier`).

    Transmission is from person to person ("short cycle", `beta`), and through the water supply ("long cycle",
    `beta_w`): everyone infectious sheds into a reservoir (`W`, a level of contamination, not people), which
    decays in 3 weeks, and infects at a rate that peaks in May each year. The disease is stratified by age
    (0-1, 2-4, 5-9, 10-14, 15-24, 25-34, 35-44, and 45+), since the fraction who become carriers rises with age.

    Typhoid conjugate vaccine (TCV) is given routinely to infants at about 9 months of age: babies are born into
    `S0`, which they leave at a mean age of `age_tcv`, when a fraction `cov_tcv` are vaccinated, and the vaccine
    protects a fraction `eff_tcv` of them, fully (`V`, "all or nothing", as Pitzer et al. model the Vi vaccines),
    until it wanes, back to `S1`. Coverage is 0 by default, for typhoid without vaccination.

    The structure and default values are those of Pitzer et al. (2014), fitted to typhoid in Vellore, India.
    `beta` and `beta_w` give their estimated R0 of 2.49 from person to person and 0.28 through the water, 2.8 in
    all, on average over the year (`R0()`, at the start of the year, when the water carries less, gives 2.7). The
    fraction of infections that become carriers rises with age, from 0.3% to 10.1%. The birth rate is within
    their range for Tamil Nadu.

    With these values, at equilibrium (after about 50 years), about 137 in 100,000 a year have clinical typhoid
    (taking 1 in 10 first infections as clinical; Pitzer et al. predict 146), at a mean age of about 18 (17.6
    among their patients), and about 2% of people aged 25 to 44 are carriers (Pitzer et al.: about 2% of those
    aged 30 to 40). Since carriage is lifelong and deaths from other causes are at one rate at every age,
    carriers build up in the oldest group, to about 1 in 6 of those 45 and over.

    Left out: the waning of immunity to clinical disease (from `S2` back to `S1`), which Pitzer et al. estimate as
    about zero; vaccinating people in `S2` and `R`, and catch-up campaigns (which `ss.vaccinate()` can add, as a
    vaccine that lowers susceptibility); and treatment and antimicrobial resistance.

    Sources:

    - Pitzer VE et al., "Predicting the impact of vaccination on the transmission dynamics of typhoid in South
      Asia: a mathematical modeling study", *PLoS Negl Trop Dis* 2014;8(1):e2642,
      https://doi.org/10.1371/journal.pntd.0002642. The structure; the R0s from person to person and through the
      water; the seasonal amplitude of 0.69 and peak at 18.2 weeks; the relative infectiousness of 0.01; the 4
      weeks of infectiousness; the 104 weeks of full immunity; the disease-induced mortality of 0.001 (applied here
      to first infections only, so about 1% of clinical cases); the decay of the water's contamination (1/3 a
      week); the range of 0.3% to 10.1% of infections that become carriers, by age; and the birth rates of Tamil
      Nadu (Table 1).
    - Ames WR, Robins M, "Age and sex as factors in the development of the typhoid carrier state, and a method for
      estimating carrier prevalence", *Am J Public Health* 1943;33(3):221-230,
      https://doi.org/10.2105/AJPH.33.3.221. Patients over 30 becoming carriers 9 times as often as younger ones,
      the basis of the illustrative values between Pitzer et al.'s lowest and highest.
    - Shakya M et al., "Efficacy of typhoid conjugate vaccine in Nepal: final results of a phase 3, randomised,
      controlled trial", *Lancet Glob Health* 2021;9(11):e1561-e1568, https://doi.org/10.1016/S2214-109X(21)00346-6.
      The efficacy of TCV, 79%.
    - Bilcke J et al., "Cost-effectiveness of routine and campaign use of typhoid Vi-conjugate vaccine in
      Gavi-eligible countries: a modelling study", *Lancet Infect Dis* 2019;19(7):728-739,
      https://doi.org/10.1016/S1473-3099(18)30804-1. The mean duration of protection by TCV, 15 years (as
      corrected in 2020: waning at 0.0672 a year), and routine vaccination at 9 months.

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the rate of transmission from person to person, for a first infection
        beta_w (`ss.per`/`ss.seasonal`): the rate of infection from the water, per unit of contamination; seasonal
        rel_carrier (float): the infectiousness of subclinical infections and chronic carriers, relative to a first infection
        shedding (`ss.per`): the rate at which the infectious contaminate the water, per person infectious (as a fraction of everyone)
        decay (`ss.per`): the rate at which the water's contamination decays
        dur_inf (`ss.Duration`): the mean duration of an infection, first or subclinical
        dur_imm (`ss.Duration`): the mean time fully immune, before people can be infected again without symptoms
        p_carrier (float/`ss.by`): the fraction of infections, by age, after which people become chronic carriers
        p_death (float): the fraction of first infections that die
        cov_tcv (float): the coverage of routine TCV (0 by default)
        eff_tcv (float): the efficacy of TCV: the fraction of those vaccinated who are protected
        dur_tcv (`ss.Duration`): the mean duration of protection from TCV
        age_tcv (`ss.Duration`): the mean age at routine TCV
        birth_rate (`ss.per`): the birth rate
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        kwargs (dict): passed to `ss.Disease`, e.g. `init` or `name`

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sim = ss.Sim(diseases=ssl.Typhoid(), total_pop=100_000, start=1960, stop=2020, dt=ss.weeks(1))
        sim.run()
        sim.typhoid.plot()
        sim.results.typhoid.recovery.select(to='dead').by('age').plot() # Deaths by age

        # With routine TCV at 9 months, at 80% coverage
        sim = ss.Sim(diseases=ssl.Typhoid(cov_tcv=0.8), total_pop=100_000, start=1960, stop=2020, dt=ss.weeks(1))
        ```
    """
    beta        = ss.par(ss.perweek(0.47), doc='Rate of transmission from person to person, for a first infection (R0 of 2.49)', source=f'{source}: R0 of 2.49 from person to person')
    beta_w      = ss.par(ss.seasonal(ss.perweek(0.0175), amplitude=0.69, peak=ss.weeks(18.2)), doc='Rate of infection from the water (R0 of 0.28), seasonal', source=f'{source}: R0 of 0.28 through the water; amplitude 0.69, peak at 18.2 weeks')
    rel_carrier = ss.par(0.01, doc='Infectiousness of subclinical infections and carriers, relative to a first infection', source=source)
    shedding    = ss.par(ss.perweek(1), doc='Rate of contamination of the water by the infectious', source=f'{source} (inseparable from beta_w)')
    decay       = ss.par(ss.perweek(1/3), doc="Rate of decay of the water's contamination", source=source)
    dur_inf     = ss.par(ss.weeks(4), doc='Mean duration of infection', source=source)
    dur_imm     = ss.par(ss.weeks(104), doc='Mean time fully immune', source=source)
    p_carrier   = ss.par(by_age(0.003, 0.003, 0.003, 0.003, 0.01, 0.02, 0.04, 0.101), doc='Fraction of infections that lead to chronic carriage, by age', source=f'{source}: 0.003 to 0.101 by age; values between illustrative, after Ames & Robins (1943)')
    p_death     = ss.par(0.001, doc='Fraction of first infections that die', source=f'{source}: disease-induced mortality')
    cov_tcv     = ss.par(0.0, doc='Coverage of routine TCV', source='No vaccination by default')
    eff_tcv     = ss.par(0.79, doc='Efficacy of TCV', source='Shakya et al. (2021): 79% over 2 years in Nepal')
    dur_tcv     = ss.par(ss.weeks(780), doc='Mean duration of protection from TCV (15 years)', source='Bilcke et al. (2019), corrected: waning of 0.0672 a year')
    age_tcv     = ss.par(ss.weeks(39), doc='Mean age at routine TCV (9 months)', source='Bilcke et al. (2019): routine vaccination at 9 months')
    birth_rate  = ss.par(ss.peryear(0.017), doc='Birth rate', source=f'{source}: 14.9 to 19.2 births per 1,000 in Tamil Nadu')
    death_rate  = ss.par(ss.peryear(0.017), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S1 = ss.State(label='Fully susceptible') # The first state, which people start in
    S0 = ss.State(label='Susceptible infant, before routine vaccination') # Where babies are born
    I1 = ss.State(init=0.001, label='First infection', infected=True)
    C  = ss.State(label='Chronic carrier', infected=True)
    R  = ss.State(label='Immune')
    S2 = ss.State(label='Partially susceptible')
    I2 = ss.State(label='Subclinical infection', infected=True)
    V  = ss.State(label='Protected by vaccine')
    W  = ss.State(label='Contamination of the water', people=False) # A level, not people

    force = beta*(I1 + rel_carrier*(I2 + C)) + beta_w*W.n # Short cycle (from person to person) and long cycle (through the water)

    infection   = (S0 + S1) >> I1 @ force
    recovery    = I1 >> {ss.dead: p_death, C: p_carrier, R: None} @ dur_inf
    waning      = R >> S2 @ dur_imm
    reinfection = S2 >> I2 @ force
    recovery2   = I2 >> {C: p_carrier, R: None} @ dur_inf
    routine     = S0 >> {V: cov_tcv*eff_tcv, S1: None} @ age_tcv
    waning_tcv  = V >> S1 @ dur_tcv
    shed        = ss.outside >> W @ shedding*(I1.frac + rel_carrier*(I2.frac + C.frac))
    clearance   = W >> ss.dead @ decay
    births      = ss.birth(S0, rate=birth_rate)
    deaths      = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify(age=age_groups)
        return
