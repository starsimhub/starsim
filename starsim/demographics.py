"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import starsim as ss
import sciris as sc
import pandas as pd

__all__ = ['Demographics', 'Births', 'Deaths', 'Pregnancy']


class Demographics(ss.Module):
    # A demographic module typically handles births/deaths/migration and takes
    # place at the start of the timestep, before networks are updated and before
    # any disease modules are executed

    def initialize(self, sim):
        super().initialize(sim)
        self.init_results(sim)
        return

    def init_results(self, sim):
        pass

    def update(self, sim):
        # Note that for demographic modules, any result updates should be
        # carried out inside this function
        pass


class Births(Demographics):
    def __init__(self, pars=None, metadata=None, **kwargs):
        super().__init__(pars, **kwargs)

        # Set defaults
        self.pars = ss.omergeleft(self.pars,
            birth_rate = 0,
            rel_birth = 1,
            units = 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
        )

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omergeleft(metadata,
            data_cols = dict(year='Year', cbr='CBR'),
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.birth_rate = self.standardize_birth_data()
        return

    def initialize(self, sim):
        """ Initialize with sim information """
        super().initialize(sim)
        if isinstance(self.pars.birth_rate, pd.DataFrame):
            br_year = self.pars.birth_rate[self.metadata.data_cols['year']]
            br_val = self.pars.birth_rate[self.metadata.data_cols['cbr']]
            all_birth_rates = np.interp(sim.yearvec, br_year, br_val)
            self.pars.birth_rate = all_birth_rates

    def standardize_birth_data(self):
        """ Standardize/validate birth rates - handled in an external file due to shared functionality """
        birth_rate = ss.standardize_data(data=self.pars.birth_rate, metadata=self.metadata)
        return birth_rate

    def init_results(self, sim):
        self.results += [
            ss.Result(self.name, 'new', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cumulative', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cbr', sim.npts, dtype=int, scale=False),
        ]
        return

    def update(self, sim):
        new_uids = self.add_births(sim)
        self.update_results(len(new_uids), sim)
        return new_uids

    def get_births(self, sim):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        """
        p = self.pars
        if sc.isnumber(p.birth_rate):
            this_birth_rate = p.birth_rate
        elif sc.checktype(p.birth_rate, 'arraylike'):
            this_birth_rate = p.birth_rate[sim.ti]

        scaled_birth_prob = this_birth_rate * p.units * p.rel_birth * sim.pars.dt
        scaled_birth_prob = np.clip(scaled_birth_prob, a_min=0, a_max=1)
        n_new = int(np.floor(sim.people.alive.count() * scaled_birth_prob))
        return n_new

    def add_births(self, sim):
        # Add n_new births to each state in the sim
        n_new = self.get_births(sim)
        new_uids = sim.people.grow(n_new)
        sim.people.age[new_uids] = 0
        return new_uids

    def update_results(self, n_new, sim):
        self.results['new'][sim.ti] = n_new

    def finalize(self, sim):
        super().finalize(sim)
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cbr'] = 1/self.pars.units*np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class Deaths(Demographics):
    def __init__(self, pars=None, par_dists=None, metadata=None, **kwargs):
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

            par_dists: dict

            metadata: data about the data contained within the data input.
                "data_cols" is is a dictionary mapping standard keys, like "year" to the
                corresponding column name in data. Similar for "sex_keys". Finally,
        """
        super().__init__(pars, **kwargs)

        self.pars = ss.omergeleft(self.pars,
            rel_death = 1,
            death_rate = 20,  # Default = a fixed rate of 2%/year, overwritten if data provided
            units = 1e-3,  # assumes death rates are per 1000. If using percentages, switch this to 1
        )

        self.par_dists = ss.omergeleft(par_dists,
            death_rate = ss.bernoulli
        )

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omergeleft(metadata,
            data_cols = dict(year='Time', sex='Sex', age='AgeGrpStart', value='mx'),
            sex_keys = dict(f='Female', m='Male'),
        )

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.death_rate_data = self.standardize_death_data()
        self.pars.death_rate = self.make_death_prob_fn

        return

    @staticmethod
    def make_death_prob_fn(module, sim, uids):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """

        if sc.isnumber(module.death_rate_data):
            death_rate = module.death_rate_data

        else:
            data_cols = sc.objdict(module.metadata.data_cols)
            year_label = data_cols.year
            age_label  = data_cols.age
            sex_label  = data_cols.sex
            val_label  = data_cols.value
            sex_keys = module.metadata.sex_keys

            available_years = module.death_rate_data[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]

            df = module.death_rate_data.loc[module.death_rate_data[year_label] == nearest_year]
            age_bins = df[age_label].unique()
            age_inds = np.digitize(sim.people.age[uids], age_bins) - 1

            f_arr = df[val_label].loc[df[sex_label] == sex_keys['f']].values
            m_arr = df[val_label].loc[df[sex_label] == sex_keys['m']].values

            # Initialize
            death_rate_df = pd.Series(index=uids)
            death_rate_df[uids[sim.people.female[uids]]] = f_arr[age_inds[sim.people.female[uids]]] # TODO: avoid double indexing
            death_rate_df[uids[sim.people.male[uids]]] = m_arr[age_inds[sim.people.male[uids]]]
            death_rate_df[uids[sim.people.age[uids] < 0]] = 0  # Don't use background death rates for unborn babies

            death_rate = death_rate_df.values

        # Scale from rate to probability. Consider an exponential here.
        death_prob = death_rate * (module.pars.units * module.pars.rel_death * sim.pars.dt)
        death_prob = np.clip(death_prob, a_min=0, a_max=1)

        return death_prob

    def standardize_death_data(self):
        """ Standardize/validate death rates - handled in an external file due to shared functionality """
        death_rate = ss.standardize_data(data=self.pars.death_rate, metadata=self.metadata)
        return death_rate

    def init_results(self, sim):
        self.results += [
            ss.Result(self.name, 'new', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cumulative', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cmr', sim.npts, dtype=int, scale=False),
        ]
        return

    def update(self, sim):
        n_deaths = self.apply_deaths(sim)
        self.update_results(n_deaths, sim)
        return

    def apply_deaths(self, sim):
        """ Select people to die """
        alive_uids = ss.true(sim.people.alive)
        death_uids = self.pars.death_rate.filter(alive_uids)
        sim.people.request_death(death_uids)
        return len(death_uids)

    def update_results(self, n_deaths, sim):
        self.results['new'][sim.ti] = n_deaths
        return

    def finalize(self, sim):
        super().finalize(sim)
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cmr'] = 1/self.pars.units*np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)
        return


class Pregnancy(Demographics):

    def __init__(self, pars=None, par_dists=None, metadata=None, **kwargs):
        super().__init__(pars, **kwargs)

        # Other, e.g. postpartum, on contraception...
        self.add_states(
            ss.State('infertile', bool, False),  # Applies to girls and women outside the fertility window
            ss.State('fecund', bool, True),  # Applies to girls and women inside the fertility window
            ss.State('pregnant', bool, False),  # Currently pregnant
            ss.State('postpartum', bool, False),  # Currently post-partum
            ss.State('ti_pregnant', int, ss.INT_NAN),  # Time pregnancy begins
            ss.State('ti_delivery', int, ss.INT_NAN),  # Time of delivery
            ss.State('ti_postpartum', int, ss.INT_NAN),  # Time postpartum ends
            ss.State('ti_dead', int, ss.INT_NAN),  # Maternal mortality
        )

        self.pars = ss.omergeleft(self.pars,
            dur_pregnancy = 0.75,
            dur_postpartum = 0.5,
            fertility_rate = 0,    # Usually this will be provided in CSV format
            rel_fertility = 1,
            maternal_death_rate = 0,
            sex_ratio = 0.5,       # Ratio of babies born female
            units = 1e-3,          # Assumes fertility rates are per 1000. If using percentages, switch this to 1
        )

        self.par_dists = ss.omergeleft(par_dists,
            fertility_rate = ss.bernoulli,
            maternal_death_rate = ss.bernoulli,
            sex_ratio = ss.bernoulli
        )

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omergeleft(metadata,
            data_cols = dict(year='Time', age='AgeGrp', value='ASFR'),
        )

        self.choose_slots = ss.randint() # Low and high will be reset upon initialization

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.fertility_rate_data = self.standardize_fertility_data()
        self.pars.fertility_rate = self.make_fertility_prob_fn

        return

    @staticmethod
    def make_fertility_prob_fn(module, sim, uids):
        """ Take in the module, sim, and uids, and return the conception probability for each UID on this timestep """

        if sc.isnumber(module.fertility_rate_data):
            fertility_rate = module.fertility_rate_data

        else:
            # Abbreviate key variables
            data_cols = sc.objdict(module.metadata.data_cols)
            year_label = data_cols.year
            age_label  = data_cols.age
            val_label  = data_cols.value

            available_years = module.fertility_rate_data[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year-module.pars.dur_pregnancy)
            nearest_year = available_years[year_ind]

            df = module.fertility_rate_data.loc[module.fertility_rate_data[year_label] == nearest_year]
            df_arr = df[val_label].values  # Pull out dataframe values
            df_arr = np.append(df_arr, 0)  # Add zeros for those outside data range

            # Process age data
            age_bins = df[age_label].unique()
            age_bins = np.append(age_bins, age_bins[-1]+1) # WARNING: Assumes one year age bins! TODO: More robust handling.
            age_inds = np.digitize(sim.people.age[uids], age_bins) - 1
            age_inds[age_inds == len(age_bins)-1] = -1  # This ensures women outside the data range will get a value of 0

            # Adjust rates: rates are based on the entire population, but we need to remove
            # anyone already pregnant and then inflate the rates for the remainder
            pregnant_uids = ss.true(module.pregnant[uids])  # Find agents who are already pregnant
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
            fertility_rate[pregnant_uids] = 0

        # Scale from rate to probability. Consider an exponential here.
        fertility_prob = fertility_rate * (module.pars.units * module.pars.rel_fertility * sim.pars.dt)
        fertility_prob = np.clip(fertility_prob, a_min=0, a_max=1)

        return fertility_prob

    def standardize_fertility_data(self):
        """ Standardize/validate fertility rates - handled in an external file due to shared functionality """
        fertility_rate = ss.standardize_data(data=self.pars.fertility_rate, metadata=self.metadata)
        return fertility_rate

    def initialize(self, sim):
        super().initialize(sim)
        low = sim.pars.n_agents + 1
        high = int(sim.pars.slot_scale*sim.pars.n_agents)
        self.choose_slots.set(low=low, high=high)
        return

    def init_results(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        self.results += [
            ss.Result(self.name, 'pregnancies', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'births', sim.npts, dtype=int, scale=True),
            ss.Result(self.name, 'cbr', sim.npts, dtype=int, scale=False),
        ]
        return

    def update(self, sim):
        """
        Perform all updates
        """
        self.update_states(sim)
        conceive_uids = self.make_pregnancies(sim)
        self.make_embryos(sim, conceive_uids)
        self.update_results(sim)
        return

    def update_states(self, sim):
        """
        Update states
        """

        # Check for new deliveries
        deliveries = self.pregnant & (self.ti_delivery <= sim.ti)
        self.pregnant[deliveries] = False
        self.postpartum[deliveries] = True
        self.fecund[deliveries] = False

        # Check for new women emerging from post-partum
        postpartum = ~self.pregnant & (self.ti_postpartum <= sim.ti)
        self.postpartum[postpartum] = False
        self.fecund[postpartum] = True

        # Maternal deaths
        maternal_deaths = ss.true(self.ti_dead <= sim.ti)
        sim.people.request_death(maternal_deaths)

        return

    def make_pregnancies(self, sim):
        """
        Select people to make pregnant using incidence data
        """
        # Abbreviate
        ppl = sim.people

        # People eligible to become pregnant. We don't remove pregnant people here, these
        # are instead handled in the fertility_dist logic as the rates need to be adjusted
        denom_conds = ppl.female & ppl.alive
        inds_to_choose_from = ss.true(denom_conds)
        conceive_uids = self.pars.fertility_rate.filter(inds_to_choose_from)

        # Set prognoses for the pregnancies
        if len(conceive_uids) > 0:
            self.set_prognoses(sim, conceive_uids)

        return conceive_uids

    def make_embryos(self, sim, conceive_uids):
        """ Add properties for the just-conceived """
        n_unborn_agents = len(conceive_uids)
        if n_unborn_agents > 0:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(conceive_uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = sim.people.grow(len(new_slots), new_slots)
            sim.people.age[new_uids] = -self.pars.dur_pregnancy
            sim.people.slot[new_uids] = new_slots  # Before sampling female_dist
            sim.people.female[new_uids] = self.pars.sex_ratio.rvs(new_uids)

            # Add connections to any vertical transmission layers
            # Placeholder code to be moved / refactored. The maternal network may need to be
            # handled separately to the sexual networks, TBC how to handle this most elegantly
            for lkey, layer in sim.networks.items():
                if layer.vertical:  # What happens if there's more than one vertical layer?
                    durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy + self.pars.dur_postpartum)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs)

        return

    def set_prognoses(self, sim, uids):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """

        # Change states for the newly pregnant woman
        self.fecund[uids] = False
        self.pregnant[uids] = True
        self.ti_pregnant[uids] = sim.ti

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.ti + self.pars.dur_pregnancy / sim.dt)
        dead = self.pars.maternal_death_rate.rvs(uids)
        self.ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(uids), dur + self.pars.dur_postpartum / sim.dt)
        self.ti_postpartum[uids] = dur_post_partum

        if np.any(dead): # NB: 100x faster than np.sum(), 10x faster than np.count_nonzero()
            self.ti_dead[uids[dead]] = dur[dead]
        return

    def update_results(self, sim):
        self.results['pregnancies'][sim.ti] = np.count_nonzero(self.ti_pregnant == sim.ti)
        self.results['births'][sim.ti] = np.count_nonzero(self.ti_delivery == sim.ti)
        return

    def finalize(self, sim):
        super().finalize(sim)
        self.results['cbr'] = 1/self.pars.units * np.divide(self.results['births'], sim.results['n_alive'], where=sim.results['n_alive']>0)
