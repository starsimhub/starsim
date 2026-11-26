"""
Measles by age, with maternal immunity and routine vaccination (`Measles`)

The first version was adapted from https://github.com/optimamodel/gavi-outbreaks/blob/main/stisim/gavi/measles.py,
by @alina-muellenmeister, @domdelport, and @RomeshA
"""
# Authorship: AI drafted, human reviewed and edited

import starsim as ss

__all__ = ['Measles']


age_groups = [0, 0.5, 1, 2, 5, 10, 15] # Years: 0-5 months, 6-11 months, 1, 2-4, 5-9, 10-14, and 15+
dynamice = 'Fu et al. (2021), DynaMICE'


class Measles(ss.Disease):
    """
    Measles by age, as an SEIR model with maternal immunity, deaths by age, and two routine doses of vaccine

    Babies are born protected by their mothers' antibodies (`M`), which wane after about 6 months (`dur_mat`).
    Susceptible people who are infected are latent (`E`) for `dur_exp`, then infectious (`I`) for `dur_inf`, after
    which they recover with lifelong immunity (`R`) or die, with a probability by age (`p_death`) that is highest
    in infants. It is declared, so it runs as compartments mixing homogeneously, or as agents on a network (where
    `beta` is the rate per contact), and it is stratified by age (0-5 and 6-11 months, 1, 2-4, 5-9, 10-14, and
    15+ years).

    Two routine doses of vaccine each protect a share of children for life (`V`), all or nothing: the first (MCV1)
    at a mean age of 9 months (`age_mcv1`), the second (MCV2) at a mean of 72 weeks (`age_mcv2`). To give the same
    coverage as compartments and as agents, each dose is the exit from a holding state: a child whose maternal
    antibodies have waned is in `S0` until the first dose, when a fraction `cov_mcv1*eff_mcv1` are protected and
    the rest move to `S1`, which they leave at the second dose, a fraction `cov_mcv2*eff_mcv2` protected and the
    rest to `S`, where everyone is at the start. Children in `S0` and `S1` can be infected like those in `S`.
    The first dose protects fewer (84%) than a dose at a year or more (92.5%), partly because maternal antibodies
    still present at 9 months stop it from taking; that is in `eff_mcv1`, so the model gives the dose after `M`.
    Coverage is 0 by default: measles before vaccines.

    With the defaults, R0 is 15.4, endemic measles has damped cycles of about 2.3 years (and, with seasonal
    transmission, e.g. `beta=ss.seasonal(ss.perday(1.4), amplitude=0.2)`, the lasting two-year cycles measles is
    known for), about two thirds of infections and four fifths of deaths are in children under 5, and 0.3% of
    those infected die.

    Sources: the structure is that of DynaMICE, an age-structured model of measles and its vaccination (Verguet et
    al., 2015; with maternal immunity, Fu et al., 2021), with the latent period of the age-structured SEIR models of
    Keeling and Rohani (2008). From DynaMICE: maternal immunity for a mean of 6 months, doses at 39 and
    72 weeks, vaccines that are all or nothing with lifelong protection, and an R0 of 16 (here `beta` gives 15.4,
    `beta` times `dur_inf`; 12 to 18 is the range usually quoted: Guerra et al., 2017). The latent and infectious
    periods are near those of the US CDC, which gives a rash about 14 days after exposure (7 to 21), and
    infectiousness from 4 days before the rash to 4 days after it: here 8 days latent and 11 infectious, as in
    Starsim's earlier model (DynaMICE, with no latent period, has 14 days infectious); the spreads are
    illustrative. The vaccine's effectiveness is the median of field studies with verified vaccination
    and confirmed cases: 84% for a dose at 9 to 11 months, and 92.5% at 12 months or older (Uzicanin and Zimmerman,
    2011; DynaMICE has 85% and 95%); with both doses, 99% are protected (94% in the same review, 98% in DynaMICE).
    The probability of death by age is a fifth of that of Sbarra et al. (2023) for community cases in low- and
    middle-income countries in 2019, which is 3.03% under 1, 1.63% from 1 to 4, 0.84% from 5 to 9, and 0.67% from
    10 to 14 (held for older ages here); 1.3% over all ages, from 0.18% in central and eastern Europe and central
    Asia to 1.9% in sub-Saharan Africa. A fifth is illustrative of good care (0.3% of those infected in endemic
    measles, near the 1 to 3 in 1,000 children that the US CDC gives); for a high-burden setting, give the values
    of Sbarra et al., `p_death=ss.by(age=[0.0303, 0.0303, 0.0163, 0.0163, 0.0084, 0.0067, 0.0067])` (1.5% of those
    infected), or more.

    Left out: seasonal transmission (give `beta` as an `ss.seasonal()`); contacts by age (give
    `networks=ss.contacts(...)`); infants born to susceptible mothers, who have no maternal antibodies; the spread
    of ages at each dose, which is wider here (a mean age, not a fixed one); second doses given only to children
    who had the first (here the two are independent); waning of the vaccine's protection; supplementary campaigns
    (which an `ss.Event()` adds: see the examples); and deaths after measles from its damage to immune memory.

    References:
        - Verguet S, Johri M, Morris SK, et al. Controlling measles using supplemental immunization activities: a
          mathematical model to inform optimal policy. *Vaccine* 2015;33(10):1291-1296.
          https://doi.org/10.1016/j.vaccine.2014.11.050
        - Fu H, Abbas K, Klepac P, et al. Effect of evidence updates on key determinants of measles vaccination
          impact: a DynaMICE modelling study in ten high-burden countries. *BMC Med* 2021;19:281.
          https://doi.org/10.1186/s12916-021-02157-4
        - Keeling MJ, Rohani P. *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press, 2008.
        - Guerra FM, Bolotin S, Lim G, et al. The basic reproduction number (R0) of measles: a systematic review.
          *Lancet Infect Dis* 2017;17(12):e420-e428. https://doi.org/10.1016/S1473-3099(17)30307-9
        - Uzicanin A, Zimmerman L. Field effectiveness of live attenuated measles-containing vaccines: a review of
          published literature. *J Infect Dis* 2011;204 Suppl 1:S133-S148. https://doi.org/10.1093/infdis/jir102
        - Sbarra AN, Mosser JF, Jit M, et al. Estimating national-level measles case-fatality ratios in low-income
          and middle-income countries: an updated systematic review and modelling study. *Lancet Glob Health*
          2023;11(4):e516-e524. https://doi.org/10.1016/S2214-109X(23)00043-8

    Note: CRN safe as agents.

    Args:
        beta (`ss.per`): the transmission rate (per contact, if there are networks)
        dur_mat (`ss.Duration`): the mean time protected by maternal antibodies after birth
        dur_exp (`ss.Duration`/`ss.Dist`): the time from infection to being infectious (a single duration given is the mean of an exponential)
        dur_inf (`ss.Duration`/`ss.Dist`): the time infectious
        p_death (float/`ss.by`): the probability of death, by age, at the end of the infectious period
        cov_mcv1 (float): the fraction of children given the first dose (0 by default)
        eff_mcv1 (float): the fraction of those given the first dose who are protected
        age_mcv1 (`ss.Duration`): the mean age at the first dose
        cov_mcv2 (float): the fraction of children given the second dose (0 by default)
        eff_mcv2 (float): the fraction of those still susceptible when given the second dose who are protected
        age_mcv2 (`ss.Duration`): the mean age at the second dose
        birth_rate (`ss.per`): the birth rate; babies are born into `M`
        death_rate (`ss.per`): the death rate from other causes; equal to `birth_rate` by default, which keeps the population size constant
        args (list): extra states, transitions, and outcomes, beside the class's
        kwargs (dict): passed to `ss.Disease`, e.g. `init`, `init_prev` or `name`; or extra states, transitions, and outcomes by name

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        # An outbreak in a town where 90% are immune
        sim = ss.Sim(diseases=ssl.Measles(init=dict(R=0.9)), total_pop=100_000, start='2025-01-01', stop='2026-01-01')
        sim.run()
        sim.measles.plot()
        sim.results.measles.death.by('age').plot() # Deaths by age

        # Endemic measles in a high-burden setting, with routine vaccination
        p_death = ss.by(age=[0.0303, 0.0303, 0.0163, 0.0163, 0.0084, 0.0067, 0.0067])
        measles = ssl.Measles(cov_mcv1=0.8, cov_mcv2=0.5, p_death=p_death, init=dict(I=0.0001, R=0.93))
        sim = ss.Sim(diseases=measles, total_pop=1_000_000, start='2000-01-01', stop='2030-01-01', dt=ss.days(2))
        sim.run()
        sim.measles.plot()

        # A campaign that reaches 90% of children under 5, to give to the sim as interventions=campaign
        campaign = ss.Event((measles.S0 + measles.S1 + measles.S) >> measles.V, when='2010-01-01', who=ss.person.age < 5, p=0.9*0.925)
        ```
    """
    beta       = ss.par(ss.perday(1.4), doc='Transmission rate, giving R0 of about 15', source='Guerra et al. (2017): 12 to 18 usually quoted; 16 in DynaMICE')
    dur_mat    = ss.par(ss.days(182), doc='Mean time protected by maternal antibodies (6 months)', source=dynamice)
    dur_exp    = ss.par(ss.erlang(k=8, mean=ss.days(8)),  doc='Time from infection to being infectious (a spread of 3 days)', source='Illustrative, near the US CDC')
    dur_inf    = ss.par(ss.erlang(k=8, mean=ss.days(11)), doc='Time infectious (a spread of 4 days)', source='Illustrative, near the US CDC')
    p_death    = ss.par(ss.by(age=[0.006, 0.006, 0.0033, 0.0033, 0.0017, 0.0013, 0.0013]), doc='Probability of death, by age', source='Illustrative of good care: a fifth of Sbarra et al. (2023), for low- and middle-income countries')
    cov_mcv1   = ss.par(0.0,   doc='Coverage of the first dose', source='None before vaccination')
    eff_mcv1   = ss.par(0.84,  doc='Fraction protected by the first dose', source='Uzicanin and Zimmerman (2011): median at 9 to 11 months')
    age_mcv1   = ss.par(ss.days(273), doc='Mean age at the first dose (39 weeks, 9 months)', source=dynamice + ', after the schedule of the WHO')
    cov_mcv2   = ss.par(0.0,   doc='Coverage of the second dose', source='None before vaccination')
    eff_mcv2   = ss.par(0.925, doc='Fraction of those still susceptible protected by the second dose', source='Uzicanin and Zimmerman (2011): median at 12 months or older')
    age_mcv2   = ss.par(ss.days(504), doc='Mean age at the second dose (72 weeks)', source=dynamice + ': the middle of 15 to 18 months')
    birth_rate = ss.par(ss.peryear(0.02), doc='Birth rate',                   source='Illustrative')
    death_rate = ss.par(ss.peryear(0.02), doc='Death rate from other causes', source='Equal to the birth rate, for a constant population')

    S  = ss.State() # First, so the people at the start are susceptible (unless init= says otherwise)
    M  = ss.State(label='Maternal immunity') # Where babies are born
    S0 = ss.State(label='Susceptible infant, before the first dose')
    S1 = ss.State(label='Susceptible child, before the second dose')
    E  = ss.State(infected=True) # Counted in the results infected and prevalence
    I  = ss.State(init=0.005, label='Infectious', infected=True)
    R  = ss.State()
    V  = ss.State(label='Protected by vaccine')

    births    = ss.birth(M, rate=birth_rate)
    waning    = M >> S0 @ dur_mat
    mcv1      = ss.Transition(S0, {V: cov_mcv1*eff_mcv1, S1: None}, dur=age_mcv1 - dur_mat)
    mcv2      = ss.Transition(S1, {V: cov_mcv2*eff_mcv2, S: None}, dur=age_mcv2 - age_mcv1)
    infection = (S0 + S1 + S) >> E @ beta*I
    onset     = E >> I @ dur_exp
    recovery  = ss.Transition(I, R, dur=dur_inf, p=1 - p_death)
    death     = ss.Transition(I, ss.dead, dur=dur_inf, p=p_death)
    deaths    = ss.death(rate=death_rate)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify(age=age_groups)
        return
