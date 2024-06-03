"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

__all__ = ['Demographics', 'Births', 'Deaths', 'Pregnancy']


class Demographics(ss.Module):
    """
    A demographic module typically handles births/deaths/migration and takes
    place at the start of the timestep, before networks are updated and before
    any disease modules are executed.
    """
    def init_pre(self, sim):
        super().init_pre(sim)
        self.init_results()
        return

    def init_results(self):
        pass

    def update(self):
        pass

    def update_results(self):
        pass


class Births(Demographics):
    """ Create births based on rates, rather than based on pregnancy """
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.default_pars(
            birth_rate = 30,
            rel_birth = 1,
            units = 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Year', cbr='CBR')),
            metadata,
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        self.n_births = 0 # For results tracking
        return

    def init_pre(self, sim):
        """ Initialize with sim information """
        super().init_pre(sim)
        if isinstance(self.pars.birth_rate, pd.DataFrame):
            br_year = self.pars.birth_rate[self.metadata.data_cols['year']]
            br_val = self.pars.birth_rate[self.metadata.data_cols['cbr']]
            all_birth_rates = np.interp(sim.yearvec, br_year, br_val)
            self.pars.birth_rate = all_birth_rates
        return

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        return birth_rate

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new', npts, dtype=int, scale=True),
            ss.Result(self.name, 'cumulative', npts, dtype=int, scale=True),
            ss.Result(self.name, 'cbr', npts, dtype=int, scale=False),
        ]
        return

    def update(self):
        new_uids = self.add_births()
        self.n_births = len(new_uids)
        return new_uids

    def get_births(self):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        sim = self.sim
        p = self.pars
        if sc.isnumber(p.birth_rate):
            this_birth_rate = p.birth_rate
        elif sc.checktype(p.birth_rate, 'arraylike'):
            this_birth_rate = p.birth_rate[sim.ti]

        scaled_birth_prob = this_birth_rate * p.units * p.rel_birth * sim.pars.dt
        scaled_birth_prob = np.clip(scaled_birth_prob, a_min=0, a_max=1)
        n_new = int(np.floor(sim.people.alive.count() * scaled_birth_prob))
        return n_new

    def add_births(self):
        """ Add n_new births to each state in the sim """
        people = self.sim.people
        n_new = self.get_births()
        new_uids = people.grow(n_new)
        people.age[new_uids] = 0
        return new_uids

    def update_results(self):
        self.results['new'][self.sim.ti] = self.n_births
        return

    def finalize(self):
        super().finalize()
        res = self.sim.results
        self.results.cumulative = np.cumsum(self.results.new)
        self.results.cbr = 1/self.pars.units*np.divide(self.results.new/self.sim.dt, res.n_alive, where=res.n_alive>0)
        return


class Deaths(Demographics):
    def __init__(self, pars=None, metadata=None, **kwargs):
        """
        Configure disease-independent "background" deaths.

        The probability of death for each agent on each timestep is determined
        by the `death_rate` parameter and the time step. The default value of
        this parameter is 0.02, indicating that all agents will
        face a 2% chance of death per year.

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
                units: units for death rates (see in-line comment on par dict below)

            metadata: data about the data contained within the data input.
                "data_cols" is is a dictionary mapping standard keys, like "year" to the
                corresponding column name in data. Similar for "sex_keys". Finally,
        """
        super().__init__()
        self.default_pars(
            rel_death = 1,
            death_rate = 20,  # Default = a fixed rate of 2%/year, overwritten if data provided
            units = 1e-3,  # assumes death rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(
                data_cols = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx'),
                sex_keys = dict(f='Female', m='Male'),
            ),
            metadata
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data() # TODO: refactor
        self.pars.death_rate = ss.bernoulli(p=self.make_death_prob_fn)
        self.n_deaths = 0 # For results tracking
        return
    
    def standardize_death_data(self):
        """ Standardize/validate death rates - handled in an external file due to shared functionality """
        death_rate = ss.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        return death_rate

    @staticmethod # Needs to be static since called externally, although it sure looks like a class method!
    def make_death_prob_fn(self, sim, uids):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """

        drd = self.death_rate_data
        if sc.isnumber(drd) or isinstance(drd, ss.Dist):
            death_rate = drd

        else:
            ppl = sim.people
            data_cols = sc.objdict(self.metadata.data_cols)
            year_label = data_cols.year
            age_label  = data_cols.age
            sex_label  = data_cols.sex
            val_label  = data_cols.value
            sex_keys = self.metadata.sex_keys

            available_years = self.death_rate_data[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]

            df = self.death_rate_data.loc[self.death_rate_data[year_label] == nearest_year]
            age_bins = df[age_label].unique()

            f_arr = df[val_label].loc[df[sex_label] == sex_keys['f']].values
            m_arr = df[val_label].loc[df[sex_label] == sex_keys['m']].values

            # Initialize
            death_rate_df = pd.Series(index=uids)
            f_uids = uids.intersect(ppl.female.uids) # TODO: reduce duplication
            m_uids = uids.intersect(ppl.male.uids)
            f_age_inds = np.digitize(ppl.age[f_uids], age_bins) - 1
            m_age_inds = np.digitize(ppl.age[m_uids], age_bins) - 1
            death_rate_df[f_uids] = f_arr[f_age_inds]
            death_rate_df[m_uids] = m_arr[m_age_inds]
            unborn_inds = uids.intersect((sim.people.age < 0).uids)
            death_rate_df[unborn_inds] = 0  # Don't use background death rates for unborn babies

            death_rate = death_rate_df.values

        # Scale from rate to probability. Consider an exponential here.
        death_prob = death_rate * (self.pars.units * self.pars.rel_death * sim.pars.dt)
        death_prob = np.clip(death_prob, a_min=0, a_max=1)

        return death_prob

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new', npts, dtype=int, scale=True),
            ss.Result(self.name, 'cumulative', npts, dtype=int, scale=True),
            ss.Result(self.name, 'cmr', npts, dtype=int, scale=False),
        ]
        return

    def update(self):
        self.n_deaths = self.apply_deaths()
        return

    def apply_deaths(self):
        """ Select people to die """
        death_uids = self.pars.death_rate.filter()
        self.sim.people.request_death(death_uids)
        return len(death_uids)

    def update_results(self):
        self.results['new'][self.sim.ti] = self.n_deaths
        return

    def finalize(self):
        super().finalize()
        n_alive = self.sim.results.n_alive
        self.results.cumulative = np.cumsum(self.results.new)
        self.results.cmr = 1/self.pars.units*np.divide(self.results.new / self.sim.dt, n_alive, where=n_alive>0)
        return


class Pregnancy(Demographics):
    """ Create births via pregnancies """
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_pregnancy = 0.75,   # Duration for pre-natal transmission
            dur_postpartum = ss.lognorm_ex(0.5, 0.5),   # Duration for post-natal transmission (e.g. via breastfeeding)
            fertility_rate = 0,     # See make_fertility_prob_function
            rel_fertility = 1,
            maternal_death_prob = ss.bernoulli(0),
            sex_ratio = ss.bernoulli(0.5), # Ratio of babies born female
            min_age = 15, # Minimum age to become pregnant
            max_age = 50, # Maximum age to become pregnant
            units = 1e-3, # Assumes fertility rates are per 1000. If using percentages, switch this to 1
        )
        self.update_pars(pars, **kwargs)
        
        # Other, e.g. postpartum, on contraception...
        self.add_states(
            ss.BoolArr('infertile'),  # Applies to girls and women outside the fertility window
            ss.BoolArr('fecund', default=True),  # Applies to girls and women inside the fertility window
            ss.BoolArr('pregnant'),  # Currently pregnant
            ss.BoolArr('postpartum'),  # Currently post-partum
            ss.FloatArr('dur_postpartum'),  # Duration of postpartum phase
            ss.FloatArr('ti_pregnant'),  # Time pregnancy begins
            ss.FloatArr('ti_delivery'),  # Time of delivery
            ss.FloatArr('ti_postpartum'),  # Time postpartum ends
            ss.FloatArr('ti_dead'),  # Maternal mortality
        )

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = sc.mergedicts(
            sc.objdict(data_cols=dict(year='Time', age='AgeGrp', value='ASFR')),
            metadata,
        )
        self.choose_slots = None # Distribution for choosing slots; set in self.initialize()

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.fertility_rate_data = self.standardize_fertility_data()
        self.pars.fertility_rate = ss.bernoulli(self.make_fertility_prob_fn)

        # For results tracking
        self.n_pregnancies = 0
        self.n_births = 0
        return

    @staticmethod
    def make_fertility_prob_fn(self, sim, uids):
        """ Take in the module, sim, and uids, and return the conception probability for each UID on this timestep """

        if sc.isnumber(self.fertility_rate_data):
            fertility_rate = pd.Series(index=uids, data=self.fertility_rate_data)

        else:
            # Abbreviate key variables
            data_cols = sc.objdict(self.metadata.data_cols)
            year_label = data_cols.year
            age_label  = data_cols.age
            val_label  = data_cols.value

            available_years = self.fertility_rate_data[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year-self.pars.dur_pregnancy)
            nearest_year = available_years[year_ind]

            df = self.fertility_rate_data.loc[self.fertility_rate_data[year_label] == nearest_year]
            df_arr = df[val_label].values  # Pull out dataframe values
            df_arr = np.append(df_arr, 0)  # Add zeros for those outside data range

            # Process age data
            age_bins = df[age_label].unique()
            age_bins = np.append(age_bins, age_bins[-1]+1) # WARNING: Assumes one year age bins! TODO: More robust handling.
            age_inds = np.digitize(sim.people.age[uids], age_bins) - 1
            age_inds[age_inds == len(age_bins)-1] = -1  # This ensures women outside the data range will get a value of 0

            # Adjust rates: rates are based on the entire population, but we need to remove
            # anyone already pregnant and then inflate the rates for the remainder
            pregnant_uids = self.pregnant.uids # Find agents who are already pregnant
            pregnant_age_counts, _ = np.histogram(sim.people.age[pregnant_uids], age_bins)  # Count them by age
            age_counts, _ = np.histogram(sim.people.age[uids], age_bins)  # Count overall number per age bin
            new_denom = age_counts - pregnant_age_counts  # New denominator for rates
            num_to_make = df_arr[:-1]*age_counts  # Number that we need to make pregnant
            new_percent = sc.dcp(df_arr)  # Initialize array with new rates
            inds_to_rescale = new_denom > 0  # Rescale any non-zero age bins
            new_percent[:-1][inds_to_rescale] = num_to_make[inds_to_rescale] / new_denom[inds_to_rescale]  # New rates

            # Make array of fertility rates
            fertility_rate = pd.Series(index=uids)
            fertility_rate[uids] = new_percent[age_inds]

        # Scale from rate to probability
        age = self.sim.people.age[uids]
        invalid_age = (age < self.pars.min_age) | (age > self.pars.max_age)
        fertility_prob = fertility_rate * (self.pars.units * self.pars.rel_fertility * sim.pars.dt)
        fertility_prob[self.pregnant.uids] = 0 # Currently pregnant women cannot become pregnant again
        fertility_prob[uids[invalid_age]] = 0 # Women too young or old cannot become pregnant
        fertility_prob = np.clip(fertility_prob, a_min=0, a_max=1)

        return fertility_prob

    def standardize_fertility_data(self):
        """ Standardize/validate fertility rates - handled in an external file due to shared functionality """
        fertility_rate = ss.standardize_data(data=self.pars.fertility_rate, metadata=self.metadata)
        return fertility_rate

    def init_pre(self, sim):
        super().init_pre(sim)
        low = sim.pars.n_agents + 1
        high = int(sim.pars.slot_scale*sim.pars.n_agents)
        self.choose_slots = ss.randint(low=low, high=high, sim=sim, module=self)
        return

    def init_results(self):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'pregnancies', npts, dtype=int, scale=True),
            ss.Result(self.name, 'births', npts, dtype=int, scale=True),
            ss.Result(self.name, 'cbr', npts, dtype=int, scale=False),
        ]
        return

    def update(self):
        """ Perform all updates """
        self.update_states()
        conceive_uids = self.make_pregnancies()
        self.n_pregnancies = len(conceive_uids)
        self.make_embryos(conceive_uids)
        return

    def update_states(self):
        """ Update states """
        # Check for new deliveries
        ti = self.sim.ti
        deliveries = self.pregnant & (self.ti_delivery <= ti)
        self.n_births = np.count_nonzero(deliveries)
        self.pregnant[deliveries] = False
        self.postpartum[deliveries] = True
        self.fecund[deliveries] = False

        # Add connections to any postnatal transmission layers
        for lkey, layer in self.sim.networks.items():
            if layer.postnatal and self.n_births:

                # Add postnatal connections by finding the prenatal contacts
                # Validation of the networks is done during initialization to ensure that 1 prenatal netwrok is present
                prenatalnet = [nw for nw in self.sim.networks.values() if nw.prenatal][0]

                # Find the prenatal connections that are ending
                prenatal_ending = prenatalnet.edges.end<=self.sim.ti
                new_mother_uids = prenatalnet.edges.p1[prenatal_ending]
                new_infant_uids = prenatalnet.edges.p2[prenatal_ending]

                # Validation
                if not np.array_equal(new_mother_uids, deliveries.uids):
                    errormsg = f'IDs of new mothers do not match IDs of new deliveries.'
                    raise ValueError(errormsg)

                # Create durations and start dates, and add connections
                durs = self.dur_postpartum[new_mother_uids]
                start = np.full(self.n_births, fill_value=self.sim.ti)

                # # Remove pairs from prenatal network and add to postnatal
                prenatalnet.end_pairs()
                layer.add_pairs(new_mother_uids, new_infant_uids, dur=durs, start=start)

        # Check for new women emerging from post-partum
        postpartum = ~self.pregnant & (self.ti_postpartum <= ti)
        self.postpartum[postpartum] = False
        self.fecund[postpartum] = True

        # Maternal deaths
        maternal_deaths = (self.ti_dead <= ti).uids
        self.sim.people.request_death(maternal_deaths)
        return

    def make_pregnancies(self):
        """ Select people to make pregnant using incidence data """
        # People eligible to become pregnant. We don't remove pregnant people here, these
        # are instead handled in the fertility_dist logic as the rates need to be adjusted
        eligible_uids = self.sim.people.female.uids
        conceive_uids = self.pars.fertility_rate.filter(eligible_uids)

        # Validation
        if np.any(self.pregnant[conceive_uids]):
            which_uids = conceive_uids[self.pregnant[conceive_uids]]
            errormsg = f'New conceptions registered in {len(which_uids)} pregnant agent(s) at timestep {self.sim.ti}.'
            raise ValueError(errormsg)

        # Set prognoses for the pregnancies
        if len(conceive_uids) > 0:
            self.set_prognoses(conceive_uids)
        return conceive_uids

    def make_embryos(self, conceive_uids):
        """ Add properties for the just-conceived """
        people = self.sim.people
        n_unborn = len(conceive_uids)
        if n_unborn == 0:
            new_uids = ss.uids()
        else:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(conceive_uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = people.grow(len(new_slots), new_slots)
            people.age[new_uids] = -self.pars.dur_pregnancy
            people.slot[new_uids] = new_slots  # Before sampling female_dist
            people.female[new_uids] = self.pars.sex_ratio.rvs(conceive_uids)

            # Add connections to any prenatal transmission layers
            for lkey, layer in self.sim.networks.items():
                if layer.prenatal:
                    durs = np.full(n_unborn, fill_value=self.pars.dur_pregnancy)
                    start = np.full(n_unborn, fill_value=self.sim.ti)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs, start=start)

        return new_uids

    def set_prognoses(self, uids):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """

        # Change states for the newly pregnant woman
        ti = self.sim.ti
        dt = self.sim.dt
        self.fecund[uids] = False
        self.pregnant[uids] = True
        self.ti_pregnant[uids] = ti

        # Outcomes for pregnancies
        dur_preg = np.full(len(uids), self.pars.dur_pregnancy)  # Duration in years
        dur_postpartum = self.pars.dur_postpartum.rvs(uids)
        dead = self.pars.maternal_death_prob.rvs(uids)
        self.ti_delivery[uids] = ti + dur_preg/dt # Currently assumes maternal deaths still result in a live baby
        self.ti_postpartum[uids] = self.ti_delivery[uids] + dur_postpartum/dt
        self.dur_postpartum[uids] = dur_postpartum

        if np.any(dead): # NB: 100x faster than np.sum(), 10x faster than np.count_nonzero()
            self.ti_dead[uids[dead]] = ti + dur_preg[dead]
        return

    def update_results(self):
        ti = self.sim.ti
        self.results['pregnancies'][ti] = self.n_pregnancies
        self.results['births'][ti] = self.n_births
        return

    def finalize(self):
        super().finalize()
        n_alive = self.sim.results.n_alive
        self.results['cbr'] = 1/self.pars.units * np.divide(self.results['births'] / self.sim.dt, n_alive, where=n_alive>0)
        return
