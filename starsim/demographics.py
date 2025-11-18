"""
Define pregnancy, deaths, migration, etc.
"""
import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

ss_float = ss.dtypes.float
ss_int = ss.dtypes.int
_ = None

__all__ = ['Demographics', 'Births', 'Deaths', 'PregnancyPars', 'Pregnancy']


class Demographics(ss.Module):
    """
    A demographic module typically handles births/deaths/migration and takes
    place at the start of the timestep, before networks are updated and before
    any disease modules are executed.
    """
    pass


class Births(Demographics):
    """
    Create births based on rates, rather than based on pregnancy.

    Births are generated using population-level birth rates that can vary
    by year. The number of births per timestep is determined by applying
    the birth rate to the current population size.

    Args:
        birth_rate (float/rate/dataframe): value for birth rate, or birth rate data
        rel_birth (float): constant used to scale all birth rates
        rate_units (float): units for birth rates (default assumes per 1000)
        metadata (dict): dict with data column mappings for birth rate data (if birth_rate is a dataframe)
    """
    def __init__(self, pars=None, birth_rate=_, rel_birth=_, rate_units=_, metadata=None, **kwargs):
        
        super().__init__()
        self.define_pars(
            birth_rate = ss.peryear(20),
            rel_birth = 1,
            rate_units = 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Year', value='CBR')),
            metadata,
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.n_births_this_step = 0 # For results tracking
        self.dist = ss.bernoulli(p=0) # Used to generate random numbers; set to use the birth rate above
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        if isinstance(self.pars.birth_rate, pd.DataFrame):
            br_year = self.pars.birth_rate[self.metadata.data_cols['year']]
            br_val = self.pars.birth_rate[self.metadata.data_cols['cbr']]
            all_birth_rates = np.interp(self.timevec, br_year, br_val) # This assumes a year timestep -- probably ok?
            self.pars.birth_rate = all_birth_rates
        return

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        if isinstance(birth_rate, (pd.Series, pd.DataFrame)):
            return birth_rate.xs(0,level='age')
        if sc.isnumber(birth_rate):
            # If the user has provided a bare number, assume it is per year
            msg = f'Birth rate was specified as a number rather than a rate - assuming it is {birth_rate} per year'
            ss.warn(msg)
            return ss.peryear(birth_rate)
        return birth_rate

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,  summarize_by='sum',  label='New births'),
            ss.Result('cumulative', dtype=int,   scale=True,  summarize_by='last', label='Cumulative births'),
            ss.Result('cbr',        dtype=float, scale=False, summarize_by='mean', label='Crude birth rate'),
        )
        return

    def get_births(self):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        sim = self.sim
        p = self.pars

        if isinstance(p.birth_rate, (pd.Series, pd.DataFrame)):
            available_years = p.birth_rate.index
            year_ind = sc.findnearest(available_years, sim.t.now('year'))
            nearest_year = available_years[year_ind]
            this_birth_rate = p.birth_rate.loc[nearest_year]
            scaled_birth_prob = ss.prob.array_to_prob(this_birth_rate, self.t.dt, v=p.rate_units * p.rel_birth)
        else:
            this_birth_rate = p.birth_rate
            if isinstance(this_birth_rate, ss.Rate):
                this_birth_rate = ss.prob(this_birth_rate.value * self.pars.rate_units * self.pars.rel_birth, this_birth_rate.unit).to_prob(ss.years(1))
            else: # number
                this_birth_rate = this_birth_rate * self.pars.rate_units * self.pars.rel_birth

            scaled_birth_prob = ss.prob.array_to_prob(np.array([this_birth_rate]), self.t.dt)[0]

        scaled_birth_prob = np.clip(scaled_birth_prob, a_min=0, a_max=1)
        self.dist.set(p=scaled_birth_prob) # Update the distribution with the probabilities for this timestep
        birth_uids = self.dist.filter()
        # n_new = np.random.binomial(n=sim.people.alive.count(), p=scaled_birth_prob) # Not CRN safe, see issue #404
        return birth_uids

    def step(self):
        new_uids = self.add_births()
        self.n_births_this_step = len(new_uids)
        return new_uids

    def add_births(self):
        """ Add n_new births to each state in the sim """
        people = self.sim.people
        birth_uids = self.get_births()
        new_uids = people.grow(len(birth_uids))
        people.age[new_uids] = 0
        people.parent[new_uids] = birth_uids
        return new_uids

    def update_results(self):
        """ Calculate new births and crude birth rate """
        # New births -- already calculated
        self.results.new[self.ti] = self.n_births_this_step

        # Calculate crude birth rate (CBR)
        inv_rate_units = 1.0/self.pars.rate_units
        births_per_year = self.n_births_this_step/self.sim.t.dt_year
        denom = self.sim.people.alive.sum()
        self.results.cbr[self.ti] = inv_rate_units*births_per_year/denom
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        return


class Deaths(Demographics):
    """
    Configure disease-independent "background" deaths.

    The probability of death for each agent on each timestep is determined
    by the `death_rate` parameter and the time step. The default value of
    this parameter is 0.01, indicating that all agents will
    face a 1% chance of death per year.

    However, this function can be made more realistic by using a dataframe
    for the `death_rate` parameter, to allow it to vary by year, sex, and
    age.  The separate 'metadata' argument can be used to configure the
    details of the input datafile.

    Alternatively, it is possible to override the `death_rate` parameter
    with a bernoulli distribution containing a constant value of function of
    your own design.

    Args:
        pars: dict with arguments including:
            rel_death: constant used to scale all death rates
            death_rate: float, dict, or pandas dataframe/series containing mortality data
            rate_units: units for death rates (see in-line comment on par dict below)

        metadata: data about the data contained within the data input.
            "data_cols" is is a dictionary mapping standard keys, like "year" to the
            corresponding column name in data. Similar for "sex_keys". 
    """
    def __init__(self, pars=None, rel_death=_, death_rate=_, rate_units=_, metadata=None, **kwargs):
        super().__init__()
        self.define_pars(
            rel_death = 1,
            death_rate = ss.peryear(10),  # Default = a fixed rate of 2%/year, overwritten if data provided
            rate_units = 1e-3,  # assumes death rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx'),
                sex_keys = {'Female':'f', 'Male':'m'},
            ),
            metadata
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data() # TODO: refactor
        self._p_death = ss.bernoulli(p=0)  # Placeholder, populated by make_p_death
        self.n_deaths = 0 # For results tracking
        return

    def standardize_death_data(self):
        """ Standardize/validate death rates - handled in an external file due to shared functionality """
        death_rate = ss.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        if isinstance(death_rate, (pd.Series, pd.DataFrame)):
            death_rate = death_rate.unstack(level='age')
            assert not death_rate.isna().any(axis=None) # For efficiency, we assume that the age bins are the same for all years in the input dataset
        if sc.isnumber(death_rate):
            # If the user has provided a bare number, assume it is per year
            msg = f'Death rate was specified as a number rather than a rate - assuming it is {death_rate} per year'
            ss.warn(msg)
            return ss.peryear(death_rate)
        return death_rate

    def make_p_death(self):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """
        sim = self.sim
        drd = self.death_rate_data

        # We are going to set up death_rate as a prob with unit=ss.years(1). That is the probability of death in 1 year
        if sc.isnumber(drd):
            death_rate = np.array([drd * self.pars.rate_units * self.pars.rel_death])  # Later gets interpreted as a prob with unit=ss.years(1) by ss.prob.array_to_prob
        elif isinstance(drd, ss.Rate):
            if drd.unit == 1: # drd is already in years so don't need to translate
                death_rate = np.array([drd.value * self.pars.rate_units * self.pars.rel_death])
            else:
                # Convert from prob per drd.unit to prob per year, ss.prob.array_to_prob will convert to prob per timestep later
                death_rate = np.array([ss.prob(drd.value * self.pars.rate_units * self.pars.rel_death, drd.unit).to_prob(ss.years(1))])

        # Process data
        else:
            ppl = sim.people
            uids = ppl.auids  # Get the UIDs of all alive people

            available_years = drd.index.get_level_values('year')
            year_ind = sc.findnearest(available_years, sim.t.now('year'))  # TODO: make work with different timesteps
            nearest_year = available_years[year_ind]

            death_rate = np.empty(uids.shape, dtype=ss_float)

            if 'sex' in drd.index.names:
                s = drd.loc[nearest_year, 'f']
                binned_ages = np.digitize(ppl.age[ppl.female], s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.female] = s.values[binned_ages]
                s = drd.loc[nearest_year, 'm']
                binned_ages = np.digitize(ppl.age[ppl.male], s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[ppl.male] = s.values[binned_ages]
            else:
                s = drd.loc[nearest_year]
                binned_ages = np.digitize(ppl.age, s.index)-1 # Negative ages will be in the first bin - do *not* subtract 1 so that this bin is 0
                death_rate[:] = s.values[binned_ages]
            death_rate *= self.pars.rate_units * self.pars.rel_death

        # Scale from rate to probability
        death_rate = ss.peryear(death_rate)
        p_death = death_rate.to_prob(self.t.dt)  # Convert to probability per timestep

        if sc.isnumber(drd) or isinstance(drd, ss.Rate):
            p_death = p_death[0] # TODO: what???
        return p_death

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new',        dtype=int,   scale=True,  summarize_by='sum',  label='Deaths', auto_plot=False), # Use sim deaths instead
            ss.Result('cumulative', dtype=int,   scale=True,  summarize_by='last', label='Cumulative deaths', auto_plot=False),
            ss.Result('cmr',        dtype=float, scale=False, summarize_by='mean', label='Crude mortality rate'),
        )
        return

    def step(self):
        """ Select people to die """
        p_death = self.make_p_death()  # Get the probability of death for each agent
        self._p_death.set(p=p_death)  # Update the distribution with the probabilities for this timestep
        death_uids = self._p_death.filter()
        self.sim.people.request_death(death_uids)
        self.n_deaths = len(death_uids)
        return self.n_deaths

    def update_results(self):
        self.results['new'][self.ti] = self.n_deaths
        return

    def finalize(self):
        super().finalize()
        self.results.cumulative[:] = np.cumsum(self.results.new)
        units = self.pars.rate_units*self.sim.t.dt_year
        inds = self.match_time_inds()
        n_alive = self.sim.results.n_alive[inds]
        deaths = np.divide(self.results.new, n_alive, where=n_alive>0)
        self.results.cmr[:] = deaths/units
        return


class PregnancyPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Parameters related to probability of getting pregnant
        self.fertility_rate=ss.peryear(100)  # Can be a number or a Pandas DataFrame
        self.rel_fertility=1  # Constant to scale all fertility rates, useful if a dataframe is used
        self.p_infertile=ss.bernoulli(p=0)  # Primary infertility
        self.min_age=15  # Minimum age to become pregnant
        self.max_age=50  # Maximum age to become pregnant
        self.rate_units=1e-3  # Assumes fertility rates are per 1000. If using percentages, switch this to 1

        # Parameters related to pregnancy duration
        self.dur_pregnancy=ss.choice(a=ss.weeks(np.arange(32, 43)), p=np.array([0.001, 0.002, 0.005, 0.012, 0.026, 0.05, 0.087, 0.134, 0.188, 0.226, 0.269])) # Quantiles for looking up fertility rates at delivery time
        self.dur_postpartum=ss.months(6)  # Duration of postpartum period. Not used for anything within this module

        # Parameters related to breastfeeding
        self.dur_breastfeed=ss.lognorm_ex(mean=ss.years(0.75), std=ss.years(0.5))  # Only relevant if postnatal transmission used...
        self.p_breastfeed=ss.bernoulli(p=1)  # Probability of breastfeeding, set to 1 for consistency

        # Pregnancy outcome parameters - PTBs, deaths
        self.rr_ptb=ss.normal(loc=1, scale=0.1)  # Base risk of pre-term birth due to factors other than maternal age
        self.rr_ptb_age= np.array([[18, 35, 1000], [1.2, 1, 1.2]]) # Relative risk of pre-term birth by maternal age
        self.p_maternal_death=ss.bernoulli(0)
        self.p_survive_maternal_death=ss.bernoulli(0)

        # Parameters related to newborn agents
        self.sex_ratio=ss.bernoulli(0.5)  # Ratio of babies born female
        self.slot_scale=5 # Random slots will be assigned to newborn agents between min=n_agents and max=slot_scale*n_agents
        self.min_slots=100  # Minimum number of slots, useful if the population size is very small

        # Settings
        self.burnin=True # Should we seed pregnancies that would have happened before the start of the simulation?
        self.trimesters=[ss.weeks(13), ss.weeks(26)]
        self.update(kwargs)
        return


class Pregnancy(Demographics):
    """
    Create births via pregnancies for each agent.

    This module models conception, pregnancy, and birth at the individual level using
    age-specific fertility rates. Supports prenatal and postnatal transmission
    networks, maternal and neonatal mortality, and burn-in of existing
    pregnancies at simulation start.

    Args:
        fertility_rate (float/dataframe): value or dataframe with age-specific fertility rates
        rel_fertility (float): constant used to scale all fertility rates
        p_infertile (bernoulli): probability of primary infertility (default 0)
        min_age (float): minimum age for pregnancy (default 15)
        max_age (float): maximum age for pregnancy (default 50)
        rate_units (float): units for fertility rates (default assumes per 1000)
        dur_pregnancy (float/dur): duration of pregnancy (drawn from choice distribution by default)
        dur_postpartum (float/dur): duration of postpartum period for postnatal transmission (default 6 months)
        dur_breastfeed (float/dur): duration of breastfeeding (default lognormal with mean 9 months, std 6 months)
        p_breastfeed (float): probability of breastfeeding (default 1)
        rr_ptb (float): base relative risk of pre-term birth (default normal with mean 1, std 0.1)
        rr_ptb_age (array): relative risk of pre-term birth by maternal age (default [[18,35,100],[1.2,1,1.2]])
        p_maternal_death (float): probability of maternal death during pregnancy (default 0.0)
        p_survive_maternal_death (float): probability that an unborn agent will survive death of the mother (default 0)
        sex_ratio (float): probability of female births (default 0.5)
        burnin (bool): whether to seed pregnancies from before simulation start (default true)
        slot_scale (float): scale factor for assigning slots to newborn agents (default 5)
        min_slots (int): minimum number of slots for newborn agents (default 100)
        trimesters (list): list of durations defining the end of each trimester
        metadata (dict): data column mappings for fertility rate data if a dataframe is supplied
    """
    def __init__(self, pars=None, fertility_rate=_, rel_fertility=_, p_infertile=_, min_age=_, max_age=_,
                 rate_units=_, dur_pregnancy=_, dur_postpartum=_, dur_breastfeed=_, p_breastfeed=_, rr_ptb=_,
                 rr_ptb_age=_, p_maternal_death=_, p_survive_maternal_death=_, sex_ratio=_, burnin=_, slot_scale=_,
                 min_slots=_, trimesters=_, metadata=None, **kwargs):
        super().__init__()
        default_pars = PregnancyPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        # Distributions: binary outcomes
        self._p_miscarriage = ss.bernoulli(p=0)  # Probability of miscarriage - placeholder, not used
        self._p_conceive = ss.bernoulli(p=0)   # Placeholder, see make_p_conceive
        self._p_stillbirth = ss.bernoulli(p=0)  # Probability of stillbirth - placeholder, not used
        self._p_twins = ss.bernoulli(p=0.5)  # Probability of twins - placeholder, not used

        # States
        self.define_states(
            # Pregnancy and fertility
            ss.BoolState('pregnant', label='Pregnant'),  # Currently pregnant
            ss.BoolState('infertile', label='Infertile (primary infertility)'),
            ss.FloatArr('rel_sus', default=0),  # Susceptibility to pregnancy, set to 1 for non-pregnant women
            ss.FloatArr('dur_pregnancy', label='Pregnancy duration'),  # Duration of pregnancy
            ss.FloatArr('parity', label='Parity', default=0),  # Number of births (may include live + still)
            ss.FloatArr('n_pregnancies', label='Number of pregnancies', default=0),  # Number of pregnancies
            ss.FloatArr('child_uid', label='UID of children, from embryo to birth'),
            ss.FloatArr('gestation', label='Number of weeks into pregnancy'),  # Gestational clock
            ss.FloatArr('gestation_at_birth', label='Gestational age in weeks'),  # Gestational age at birth, NA if not born during the sim
            ss.FloatArr('ti_pregnant', label='Time of pregnancy'),  # Time pregnancy begins
            ss.FloatArr('ti_delivery', label='Timestep of delivery'),  # Timestep of delivery (integer)
            ss.FloatArr('date_delivery', label='Date of delivery'),  # Date of delivery (ss.DateArray)
            ss.FloatArr('ti_dead', label='Time of maternal death'),  # Maternal mortality

            # Pre-term birth
            ss.FloatArr('rel_ptb_base', label='Relative risk of pre-term birth at base', default=1),  # Baseline relative risk of pre-term birth - constant throughout lifetime of mother
            ss.FloatArr('rel_ptb', label='Relative risk of pre-term birth', default=1),  # Relative risk of pre-term birth - resets every timestep

            # Breastfeeding
            ss.BoolState('breastfeeding', label='Breastfeeding'),  # Currently breastfeeding
            ss.FloatArr('dur_breastfeed', label='Duration of breastfeeding'),  # Duration of breastfeeding
            ss.FloatArr('ti_stop_breastfeed', label='Time breastfeeding stops'),  # Time breastfeeding stops
            ss.BoolState('breastfed', label='Breastfed'),  # Property of newborn indicating whether they were breastfed
        )

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Time', age='AgeGrp', value='ASFR')),
            metadata,
        )
        self.choose_slots = None  # Distribution for choosing slots; set in self.init()
        self.fertility_rate_data = None  # Processed data; set in self.init_pre() if fertility rate data is in pars

        # For results tracking
        self.n_pregnancies_this_step = 0
        self.n_births_this_step = 0
        self.derived_results = None  # Updated in init_results

        # Define ASFR
        self.asfr_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
        self.asfr_width = self.asfr_bins[1]-self.asfr_bins[0]
        self.asfr = None  # Storing this separately from results as it has a different format

        return

    @property
    def fecund(self):
        """ Defined as women of childbearing age """
        ppl = self.sim.people
        pars = self.pars
        return (ppl.age >= pars.min_age) & (ppl.age <= pars.max_age) & ppl.female

    @property
    def fertile(self):
        """ Defined as women of childbearing age who are not infertile. Includes women who may be pregnant """
        return self.fecund & (~self.infertile)

    @property
    def susceptible(self):
        """ Defined as fertile women of childbearing age who are not pregnant, and so are susceptible to conception """
        return self.fertile & (~self.pregnant)

    @property
    def nulliparous(self):
        """ Women of childbearing age who have never been pregnant """
        return (self.parity == 0) & self.fecund

    @property
    def postpartum(self):
        """
        Within a pre-defined postpartum window, as specified by the dur_postpartum par
        This does not directly affect any other functionality within this module, but is
        provided for convenience for modules that need to know which women are X timesteps
        postpartum.
        """
        timesteps_since_birth = self.ti - self.ti_delivery
        pp_timesteps = int(self.pars.dur_postpartum/self.t.dt)
        pp_bools = ~self.pregnant & (timesteps_since_birth >= 0) & (timesteps_since_birth <= pp_timesteps)
        return pp_bools

    @property
    def dur_gestation(self):
        """ Return duration of gestation for currently-pregnant women in years """
        return ss.years((self.ti - self.ti_pregnant[self.pregnant])*self.t.dt_year)

    @property
    def dur_gestation_at_birth(self):
        """ Return duration of gestation at birth for agents born during the simulation """
        born_during_sim = self.sim.people.parent.notnan & (self.sim.people.age > self.t.dt.years)
        parents_during_sim = self.sim.people.parent[born_during_sim]
        return ss.years(self.ti - self.ti_delivery[parents_during_sim]*self.t.dt_year)

    @property
    def tri1_uids(self):
        """ Return UIDs of those in their first trimester """
        return self.pregnant.uids[self.dur_gestation < self.pars.trimesters[0]]

    @property
    def tri2_uids(self):
        """ Return UIDs of those in their second trimester """
        return self.pregnant.uids[(self.dur_gestation >= self.pars.trimesters[0]) & (self.dur_gestation < self.pars.trimesters[1])]

    @property
    def tri3_uids(self):
        """ Return UIDs of those in their third trimester """
        return self.pregnant.uids[self.dur_gestation >= self.trimesters[1]]

    def make_p_conceive(self, filter_uids=None):
        """ Take in the module, sim, and uids, and return the conception probability for each UID on this timestep """
        ppl = self.sim.people

        # Apply filter UIDS and get ages
        uids = self.fecund
        if filter_uids is not None: uids = filter_uids & uids
        age = ppl.age[uids]

        # Get data, check it's in the right form
        frd = self.fertility_rate_data
        if sc.isnumber(frd):
            raise TypeError('Fertility rate should be specified as a rate (or with time-varying data)')

        # Initialize rate as an array the length of raw UIDs, so we can index it with UIDs
        fertility_rate = np.zeros(len(ppl.uid.raw), dtype=ss_float)

        if isinstance(frd, ss.Rate):
            fertility_rate[uids] = ss.prob(frd.value * (self.pars.rate_units * self.pars.rel_fertility), frd.unit).to_prob(ss.years(1))
        else:
            # Get the time of birth for pregnancies conceived now and pull out the birth rates for that year.
            # We adjust the birth rates to account for the fact that some women are already pregnant, which
            # is why we make a copy.
            birth_year = self.t.year + self.pars.dur_pregnancy.pars.a.years
            birth_year_inds = sc.findnearest(frd.index, birth_year)
            nearest_year = frd.index[birth_year_inds][0]  # 0 because the corner case of spanning 2 years can be ignored
            new_rate = self.fertility_rate_data.loc[nearest_year].values.copy()  # Initialize array with new rates

            # Assign agents to age bins
            age_bins = self.fertility_rate_data.columns.values
            age_bin_all = np.digitize(age, age_bins) - 1

            if self.pregnant.any():
                # Scale the new rate to convert the denominator from all women to non-pregnant women
                v, c = np.unique(age_bin_all, return_counts=True)
                age_counts = np.zeros_like(new_rate)
                age_counts[v] = c

                age_bin_pregnant = np.digitize(ppl.age[self.pregnant], age_bins) - 1
                v, c = np.unique(age_bin_pregnant, return_counts=True)
                pregnant_age_counts = np.zeros_like(new_rate)
                pregnant_age_counts[v] = c

                num_to_make = new_rate * age_counts  # Number that we need to make pregnant
                new_denom = age_counts - pregnant_age_counts  # New denominator for rates
                np.divide(num_to_make, new_denom, where=new_denom>0, out=new_rate)

            # Overall fertility rate
            fertility_rate[uids] = new_rate[age_bin_all] * (self.pars.rate_units * self.pars.rel_fertility)  # Prob per year

        # Scale from rate to probability
        fertility_rate[self.pregnant.uids] = 0  # Currently pregnant women cannot become pregnant
        fertility_rate = ss.peryear(fertility_rate[filter_uids])  # Only return rates for requested UIDs
        p_conceive = fertility_rate.to_prob(self.t.dt)  # Convert to probability per timestep
        return p_conceive

    def standardize_fertility_data(self):
        """
        Standardize/validate fertility rates
        """
        fertility_rate = ss.standardize_data(data=self.pars.fertility_rate, metadata=self.metadata)
        if isinstance(fertility_rate, (pd.Series, pd.DataFrame)):
            fertility_rate = fertility_rate.unstack()
            # Interpolate to 1 year increments
            fertility_rate = fertility_rate.reindex(np.arange(fertility_rate.index.min(), fertility_rate.index.max() + 1)).interpolate()
            max_age = fertility_rate.columns.max()
            fertility_rate[max_age + 1] = 0
            assert not fertility_rate.isna().any(axis=None) # For efficiency, we assume that the age bins are the same for all years in the input dataset
        if sc.isnumber(fertility_rate):
            msg = f'Fertility rate was specified as a number rather than a rate - assuming it is {fertility_rate} per year'
            ss.warn(msg)
            return ss.peryear(fertility_rate)
        return fertility_rate

    def init_pre(self, sim):
        super().init_pre(sim)

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        if self.pars.fertility_rate is not None:
            self.fertility_rate_data = self.standardize_fertility_data()

        low = sim.pars.n_agents + 1
        high = int(self.pars.slot_scale*sim.pars.n_agents)
        high = np.maximum(high, self.pars.min_slots) # Make sure there are at least min_slots slots to avoid artifacts related to small populations
        self.choose_slots = ss.randint(low=low, high=high, sim=sim, module=self)

        return

    def init_post(self):
        super().init_post()
        self.updates_pre()
        return

    def init_results(self):
        """
        Initialize results. By default, this includes:
        - pregnancies: number of new pregnancies on each timestep
        - births: number of new births on each timestep
        - cbr: crude birth rate on each timestep
        - tfr: total fertility rate on each timestep
        - The number of people who are fecund, (in)fertile, susceptible, postpartum, pregnant, infertile, breastfeeding
        """
        super().init_results()

        scaling_kw = dict(dtype=int, scale=True)
        nonscaling_kw = dict(dtype=float, scale=False)
        self.derived_results = ['n_fecund', 'n_fertile', 'n_susceptible', 'n_postpartum']

        # Define results
        results = sc.autolist()
        results += [
            ss.Result('pregnancies', **scaling_kw, label='New pregnancies', summarize_by='sum'),
            ss.Result('births', **scaling_kw, label='New births', summarize_by='sum'),
            ss.Result('maternal_deaths', **scaling_kw, label='Maternal deaths', summarize_by='sum'),
            ss.Result('mmr', **nonscaling_kw, summarize_by='mean', label='Maternal mortality rate'),
            ss.Result('cbr', **nonscaling_kw, summarize_by='mean', label='Crude birth rate'),
            ss.Result('tfr', **nonscaling_kw, summarize_by='sum',  label='Total fertility rate'),
        ]
        for res in self.derived_results:
            results += ss.Result(res, **scaling_kw)
        self.define_results(*results)

        # Extra results
        self.asfr = np.zeros((len(self.asfr_bins)-1, self.t.npts))

        return

    # Helper methods
    def _get_uids(self, upper_age=None, female_only=True):
        """ Get the UIDs of people, usually only females, who are below a certain age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age <= upper_age
        if female_only:
            f_uids = (within_age & people.female).uids
            return f_uids
        else:
            uids = within_age.uids
            return uids

    # Methods for setting states
    def set_ptb(self):
        """ Update relative risk of pre-term birth """
        self.rel_ptb[:] = self.rel_ptb_base[:]  # Reset to baseline
        rr_ptb_age_bins = self.pars.rr_ptb_age[0].astype(int)
        age_ind = np.searchsorted(rr_ptb_age_bins, self.sim.people.age, side="left")
        self.rel_ptb[:] *= self.pars.rr_ptb_age[1][age_ind]
        return

    def set_rel_sus(self):
        """
        Set relative susceptibility to pregnancy. Note that rel_sus isn't used in this module,
        but it's a key ingredient for derived modules that compute pregnancies based on exposure.
        """
        self.rel_sus[self.susceptible] = 1.0
        self.rel_sus[~self.susceptible] = 0.0
        return

    def updates_pre(self, uids=None, upper_age=None):
        """
        This runs prior at the beginning of each timestep, prior to calculating pregnancy exposure,
        advancing pregnancies, adding new pregnancies, or determing delivery outcomes. Here we make
        any updates that affect the risk of pregnancy or pre-term birth on this timestep. We also
        set the baseline values for newborn agents.
        """
        if uids is None: uids = self._get_uids(upper_age=upper_age)
        self.infertile[uids] = self.pars.p_infertile.rvs(uids)  # Infertility
        self.rel_ptb_base[uids] = self.pars.rr_ptb.rvs(uids)  # Baseline relative risk of pre-term birth
        self.set_ptb()  # Update pre-term birth relative risk

        # Check if anyone stops breastfeeding
        if self.breastfeeding.any():
            self.update_breastfeeding(self.breastfeeding.uids)

        return uids

    def update_breastfeeding(self, uids):
        stopping = uids[self.ti >= self.ti_stop_breastfeed[uids]]
        if np.any(stopping):
            self.breastfeeding[stopping] = False
        return stopping

    def process_delivery(self, uids, newborn_uids):
        """
        Handle delivery by updating all states for the mother. This method also transfers
        recorded information stored with the mother to the newborn. During pregnancy,
        gestational age is tracked with the mother; at birth, this value is transferred
        to the newborn agent before being reset for the mother. Likewise, during pregnancy,
        the child UID is stored with the mother; at birth, this value is removed from the mother,
        although the newborn agent can still be linked to the mother via the parent state.
        """
        self.gestation_at_birth[newborn_uids] = self.gestation[uids]  # Transfer to newborn before it gets reset
        self.child_uid[uids]  # Remove child UIDs for women once they are no longer pregnant
        self.pregnant[uids] = False
        self.ti_delivery[uids] = self.ti  # Record timestep of delivery as timestep, not fractional time
        self.gestation[uids] = np.nan  # No longer pregnant so remove gestational clock
        self.parity[uids] += 1  # Increment parity for the mothers

        # Set maternal death outcomes
        dead = self.pars.p_maternal_death.rvs(uids)
        if np.any(dead):  # NB: 100x faster than np.sum(), 10x faster than np.count_nonzero()
            self.ti_dead[uids[dead]] = self.ti
        return uids, newborn_uids

    def process_newborns(self, uids):
        """ Set states for newborn agents """
        return

    def process_postpartum(self, uids):
        """
        Handle postpartum updates for new mothers
        Examples of what could go in here include:
         - adjusting susceptibility to pregnancy via lactational amenorrhea (LAM),
           postpartum anovulation, and postpartum sexual abstinence or reduced activity
         - adjusting care-seeking behavior
         - adjusting susceptibility to other infections
        """
        return

    def set_breastfeeding(self, newborn_uids):
        """
        Set breastfeeding durations for new mothers. Thie method could be extended to
        store duration of exclusive breastfeeding, partial breastfeeding, etc, and these
        properties could be stored with the infant for tracking other health outcomes.
        """
        breastfed_bools = self.pars.p_breastfeed.rvs(newborn_uids)
        gets_breastfed = newborn_uids[breastfed_bools]
        will_breastfeed = self.sim.people.parent[gets_breastfed]
        self.breastfeeding[will_breastfeed] = True  # For the mother
        self.dur_breastfeed[will_breastfeed] = self.pars.dur_breastfeed.rvs(will_breastfeed)
        self.ti_stop_breastfeed[will_breastfeed] = self.ti + self.dur_breastfeed[will_breastfeed]
        self.breastfed[gets_breastfed] = True  # For the infant
        return

    def update_prenatal_network(self, conceive_uids, new_uids):
        """ Add connections to any pre-natal networks """

        # Add connections to any prenatal transmission layers
        for lkey, layer in self.sim.networks.items():
            if layer.prenatal:
                durs = self.dur_pregnancy[conceive_uids]
                start = np.full(len(new_uids), fill_value=self.ti)
                layer.add_pairs(conceive_uids, new_uids, dur=durs, start=start)

    def update_breastfeeding_network(self, delivery_uids):
        """ Add connections to any post-natal networks """
        for lkey, layer in self.sim.networks.items():
            if layer.postnatal and self.n_births_this_step:

                # Add postnatal connections by finding the prenatal contacts
                # Validation of the networks is done during initialization to ensure that 1 prenatal netwrok is present
                prenatalnet = [nw for nw in self.sim.networks.values() if nw.prenatal][0]

                # Find the prenatal connections that are ending
                prenatal_ending = (prenatalnet.edges.end > self.ti) & (prenatalnet.edges.end < (self.ti + 1))
                new_mother_uids = prenatalnet.edges.p1[prenatal_ending]
                new_infant_uids = prenatalnet.edges.p2[prenatal_ending]

                # Validation
                if not set(new_mother_uids) == set(delivery_uids):  # Not sure why sometimes out of order
                    errormsg = 'IDs of new mothers do not match IDs of new deliveries'
                    raise ValueError(errormsg)

                # Create durations and start dates, and add connections
                durs = self.dur_breastfeed[new_mother_uids]
                start = np.full(self.n_births_this_step, fill_value=self.ti)

                # # Remove pairs from prenatal network and add to postnatal
                prenatalnet.end_pairs()
                layer.add_pairs(new_mother_uids, new_infant_uids, dur=durs, start=start)

        return

    def progress_pregnancies(self):
        """
        Update pregnant women. The method can be enhanced by derived classes that add logic
        for miscarriage, termination, maternal death, etc.
        """
        # Update gestational clock for ongoing pregnancies that aren't going to deliver on this timestep
        if self.pregnant.any():
            will_deliver = (self.ti_delivery > self.ti) & (self.ti_delivery < (self.ti+1)) & self.pregnant
            self.gestation[self.pregnant] = self.dur_gestation.weeks
            self.gestation[will_deliver] = (self.t.dt*self.dur_pregnancy[will_deliver]).weeks  # Set to full term for those delivering this timestep

        # Check that gestational clock has values for all currently-pregnant women
        if np.any(self.pregnant & np.isnan(self.gestation)):
            which_uids = ss.uids(self.pregnant & np.isnan(self.gestation))
            errormsg = f'Gestational clock has NaN values for {len(which_uids)} pregnant agent(s) at timestep {self.ti}.'
            raise ValueError(errormsg)

        return

    def update_maternal_deaths(self):
        """
        Check for maternal deaths
        """
        maternal_deaths = (self.ti_dead <= self.ti).uids
        self.sim.people.request_death(maternal_deaths)
        self.results['maternal_deaths'][self.ti] = len(maternal_deaths)
        return

    def select_conceivers(self, uids=None):
        """ Select people to make pregnant """
        # People eligible to become pregnant. We don't remove pregnant people here, these
        # are instead handled in the fertility_dist logic as the rates need to be adjusted
        if uids is None: uids = self.sim.people.female.uids
        p_conceive = self.make_p_conceive(uids)
        self._p_conceive.set(p_conceive)
        conceive_uids = self._p_conceive.filter(uids)

        if len(conceive_uids) == 0:
            return ss.uids()

        # Validation
        if np.any(self.pregnant[conceive_uids]):
            which_uids = conceive_uids[self.pregnant[conceive_uids]]
            errormsg = f'New conceptions registered in {len(which_uids)} pregnant agent(s) at timestep {self.ti}.'
            raise ValueError(errormsg)

        return conceive_uids

    def _make_newborn_uids(self, conceive_uids):
        """ Helper method to link embryos to mothers """
        # Choose slots for the unborn agents
        new_slots = self.choose_slots.rvs(conceive_uids)
        new_uids = self.sim.people.grow(len(new_slots), new_slots)
        return new_uids, new_slots

    def _set_embryo_states(self, conceive_uids, new_uids, new_slots):
        """ Set states for the just-conceived """
        people = self.sim.people
        gest_years = self.dur_pregnancy[conceive_uids] * self.t.dt_year
        people.age[new_uids] = -gest_years
        people.slot[new_uids] = new_slots  # Before sampling female_dist
        people.female[new_uids] = self.pars.sex_ratio.rvs(conceive_uids)
        people.parent[new_uids] = conceive_uids
        return

    def make_embryos(self, conceive_uids):
        """
        Make newly-conceived agents. This method calls two helper methods, which grow the population
        and set the states for the newborn agents.
        """
        people = self.sim.people
        n_unborn = len(conceive_uids)
        if n_unborn == 0:
            new_uids = ss.uids()
        else:
            new_uids, new_slots = self._make_newborn_uids(conceive_uids)
            self._set_embryo_states(conceive_uids, new_uids, new_slots)
            self.child_uid[conceive_uids] = new_uids  # Stored for the duration of pregnancy then removed

        if self.ti < 0:
            people.age[new_uids] += -self.ti * self.sim.t.dt_year  # Age to ti=0

        return new_uids

    def make_pregnancies(self, uids):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """

        # Change states for the newly pregnant woman
        ti = self.ti
        self.pregnant[uids] = True
        self.ti_pregnant[uids] = ti
        self.gestation[uids] = 0
        self.n_pregnancies[uids] += 1

        # Use rel_ptb to assign pregnancy durations
        rel_ptb = self.rel_ptb[uids]
        sorted_uids = uids[np.argsort(-rel_ptb)]
        dur_preg = self.pars.dur_pregnancy.rvs(uids)
        self.dur_pregnancy[sorted_uids] = np.sort(dur_preg)
        self.ti_delivery[sorted_uids] = ti + self.dur_pregnancy[sorted_uids]
        self.date_delivery[sorted_uids] = self.t.now() + ss.DateArray(self.dur_pregnancy[sorted_uids]*self.t.dt)

        # Check that all pregnant women have a delivery time set
        missing_delivery = self.pregnant[uids] & np.isnan(self.ti_delivery[uids])
        if np.any(missing_delivery):
            which_uids = uids[missing_delivery]
            errormsg = f'Delivery time has NaN values for {len(which_uids)} pregnant agent(s) at timestep {self.ti}.'
            raise ValueError(errormsg)

        return

    def step(self):
        if self.ti == 0 and self.pars.burnin:  # TODO: refactor
            dtis = np.arange(np.ceil(-1 * self.pars.dur_pregnancy.rvs()), 0, 1).astype(int)
            for dti in dtis:
                self.t.ti = dti
                self.do_step()
            self.t.ti = 0
        self.do_step()
        return

    def do_step(self):
        """ Perform all updates except for deaths, which are handled in finish_step """

        # Set base states
        self.updates_pre(upper_age=self.t.dt_year)  # Set base states for new agents

        # Update ongoing pregnancies
        self.progress_pregnancies()

        # Process deliveries and births
        mothers = (self.pregnant & (self.ti_delivery >= self.ti) & (self.ti_delivery < (self.ti + 1))).uids
        if len(mothers):
            newborns = ss.uids(self.child_uid[mothers])
            mothers, newborns = self.process_delivery(mothers, newborns)    # Resets maternal states & transfers data to child
            self.n_births_this_step += len(newborns)    # += to handle burn-in
            self.process_newborns(newborns)             # Process newborns
            self.set_breastfeeding(newborns)            # Set breastfeeding states
            self.update_breastfeeding_network(mothers)  # Update transmission networks

        # Make any postpartum updates
        if self.postpartum.any():
            self.process_postpartum(self.postpartum.uids)

        # Figure out who conceives, set prognoses, and make embryos
        self.set_rel_sus()                              # Update rel_sus
        conceivers = self.select_conceivers()           # Get the UIDs of women who are going to conceive this timestep
        self.n_pregnancies_this_step += len(conceivers) # += to handle burn-in
        if len(conceivers):
            self.make_pregnancies(conceivers)           # Set prognoses for new pregnancies
            new_uids = self.make_embryos(conceivers)    # Create unborn agents
            self.update_prenatal_network(conceivers, new_uids)    # Update networks with new pregnancies

        # Other updates - maternal deaths
        self.update_maternal_deaths()                   # Handle maternal deaths

        return

    def step_die(self, uids):
        """ Wipe dates and states following death """
        self.pregnant[uids] = False
        self.dur_pregnancy[uids] = np.nan
        self.gestation[uids] = np.nan
        self.child_uid[uids] = np.nan
        self.ti_delivery[uids] = np.nan
        self.date_delivery[uids] = np.nan
        return

    def finish_step(self):
        super().finish_step()
        death_uids = ss.uids(self.sim.people.ti_dead <= self.ti)
        if len(death_uids) == 0:
            return

        # Any pregnant? Consider death of the neonate. Default probability is set to 1
        # meaning we assume that unborn children do not survive.
        mother_death_uids = death_uids[self.pregnant[death_uids]]
        if len(mother_death_uids):
            unborn_uids = ss.uids(self.child_uid[mother_death_uids])
            unborn_survival = self.pars.p_survive_maternal_death.rvs(unborn_uids)
            unborn_death_uids = unborn_uids[~unborn_survival]
            if len(unborn_death_uids):
                self.sim.people.request_death(unborn_death_uids)
            self.step_die(mother_death_uids)

        # Any prenatal? Handle changes to pregnancy
        is_prenatal = self.sim.people.age[death_uids] < 0
        prenatal_death_uids = death_uids[is_prenatal]
        if len(prenatal_death_uids):
            mother_uids = self.sim.people.parent[prenatal_death_uids]
            self.step_die(mother_uids)

        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results['pregnancies'][ti] = self.n_pregnancies_this_step
        self.results['births'][ti] = self.n_births_this_step

        for res in self.derived_results:
            state = getattr(self, res.replace('n_', ''))
            self.results[res][self.ti] = state.sum()

        # Reset for the next step
        self.n_pregnancies_this_step = 0
        self.n_births_this_step = 0

        # Update ASFR, TFR, and MMR
        res = self.results
        self.compute_asfr()
        self.results.tfr[ti] = sum(self.asfr[:, ti])*self.asfr_width/1000
        self.results.mmr[ti] = sc.safedivide(res.maternal_deaths[ti], res.births[ti]) * 100e3

        return

    def compute_asfr(self):
        """
        Computes age-specific fertility rates (ASFR). Since this is calculated each timestep,
        the annualized results should compute the sum.
        """
        new_mother_uids = (self.ti_delivery == self.ti).uids
        new_mother_ages = self.sim.people.age[new_mother_uids]
        births_by_age, _ = np.histogram(new_mother_ages, bins=self.asfr_bins)
        women_by_age, _ = np.histogram(self.sim.people.age[self.sim.people.female], bins=self.asfr_bins)
        self.asfr[:, self.ti] = sc.safedivide(births_by_age, women_by_age) * 1000
        return

    def finalize(self):
        super().finalize()
        units = self.pars.rate_units*self.sim.t.dt_year
        inds = self.match_time_inds()
        n_alive = self.sim.results.n_alive[inds]
        births = np.divide(self.results['births'], n_alive, where=n_alive>0)
        self.results['cbr'][:] = births/units

        # Aggregate the ASFR results, taking rolling annual sums
        asfr = np.zeros((len(self.asfr_bins)-1, self.t.npts))
        tdim = int(1/self.t.dt_year)
        for i in range(len(self.asfr_bins)-1):
            asfr[i, (tdim-1):] = np.convolve(self.asfr[i, :], np.ones(tdim), mode='valid')
        self.asfr = asfr

        return
