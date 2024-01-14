"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import stisim as ss
import sciris as sc
import pandas as pd
import scipy.stats as sps

__all__ = ['DemographicModule', 'births', 'background_deaths', 'Pregnancy']


class DemographicModule(ss.Module):
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

class births(DemographicModule):
    def __init__(self, pars=None, metadata=None):
        super().__init__(pars)

        # Set defaults
        self.pars = ss.omerge({
            'birth_rate': 0,
            'rel_birth': 1,
            'units': 1e-3  # assumes birth rates are per 1000. If using percentages, switch this to 1
        }, self.pars)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omerge({
            'data_cols': {'year': 'Year', 'cbr': 'CBR'},
        }, metadata)

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
        self.results += ss.Result(self.name, 'new', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cumulative', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cbr', sim.npts, dtype=int, scale=False)
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
        n_new = int(np.floor(np.count_nonzero(sim.people.alive) * scaled_birth_prob))
        return n_new

    def add_births(self, sim):
        # Add n_new births to each state in the sim
        n_new = self.get_births(sim)
        new_uids = sim.people.grow(n_new)
        return new_uids

    def update_results(self, n_new, sim):
        self.results['new'][sim.ti] = n_new

    def finalize(self, sim):
        super().finalize(sim)
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cbr'] = np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class background_deaths(DemographicModule):
    def __init__(self, pars=None, metadata=None):
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
        
        :param pars: dict with arguments including:
            rel_death: constant used to scale all death rates
            death_rate: float, dict, or pandas dataframe/series containing mortality data
            units: units for death rates (see in-line comment on par dict below)

        :param metadata: data about the data contained within the data input.
            "data_cols" is is a dictionary mapping standard keys, like "year" to the
            corresponding column name in data. Similar for "sex_keys". Finally,
        """
        super().__init__(pars)

        self.pars = ss.omerge({
            'rel_death': 1,
            'death_rate': 0.02,  # Default = a fixed rate of 2%/year, overwritten if data provided
            'units': 1,  # units for death rates. If using percentages, leave as 1. If using a CMR (e.g. 12 deaths per 1000), change to 1/1000
        }, self.pars)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omerge({
            'data_cols': {'year': 'Time', 'sex': 'Sex', 'age': 'AgeGrpStart', 'value': 'mx'},
            'sex_keys': {'f': 'Female', 'm': 'Male'},
        }, metadata)

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.death_rate = self.standardize_death_data()

        # Create death_prob_fn, a function which returns a probability of death for each requested uid
        self.death_prob_fn = self.make_death_prob_fn
        self.death_dist = sps.bernoulli(p=self.death_prob_fn)

        return

    @staticmethod
    def make_death_prob_fn(module, sim, uids):
        """ Take in the module, sim, and uids, and return the probability of death for each UID on this timestep """

        if sc.isnumber(module.pars.death_rate):
            death_rate = module.pars.death_rate

        else:
            year_label = module.metadata.data_cols['year']
            age_label = module.metadata.data_cols['age']
            sex_label = module.metadata.data_cols['sex']
            val_label = module.metadata.data_cols['value']
            sex_keys = module.metadata.sex_keys

            available_years = module.pars.death_rate[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]

            df = module.pars.death_rate.loc[module.pars.death_rate[year_label] == nearest_year]
            age_bins = df[age_label].unique()
            age_inds = np.digitize(sim.people.age[uids], age_bins) - 1

            f_arr = df[val_label].loc[df[sex_label] == sex_keys['f']].values
            m_arr = df[val_label].loc[df[sex_label] == sex_keys['m']].values

            # Initialize
            death_rate_df = pd.Series(index=uids)
            death_rate_df[uids[sim.people.female[uids]]] = f_arr[age_inds[sim.people.female[uids]]]
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
        self.results += ss.Result(self.name, 'new', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cumulative', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cmr', sim.npts, dtype=int, scale=False)
        return

    def update(self, sim):
        n_deaths = self.apply_deaths(sim)
        self.update_results(n_deaths, sim)
        return

    def apply_deaths(self, sim):
        """ Select people to die """
        alive_uids = ss.true(sim.people.alive)
        death_uids = self.death_dist.filter(alive_uids)
        sim.people.request_death(death_uids)
        return len(death_uids)

    def update_results(self, n_deaths, sim):
        self.results['new'][sim.ti] = n_deaths

    def finalize(self, sim):
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cmr'] = np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class Pregnancy(DemographicModule):

    def __init__(self, pars=None, metadata=None):
        super().__init__(pars)

        # Other, e.g. postpartum, on contraception...
        self.infertile = ss.State('infertile', bool, False)  # Applies to girls and women outside the fertility window
        self.susceptible = ss.State('susceptible', bool, True)  # Applies to girls and women inside the fertility window - needs renaming
        self.pregnant = ss.State('pregnant', bool, False)  # Currently pregnant
        self.postpartum = ss.State('postpartum', bool, False)  # Currently post-partum
        self.ti_pregnant = ss.State('ti_pregnant', int, ss.INT_NAN)  # Time pregnancy begins
        self.ti_delivery = ss.State('ti_delivery', int, ss.INT_NAN)  # Time of delivery
        self.ti_postpartum = ss.State('ti_postpartum', int, ss.INT_NAN)  # Time postpartum ends
        self.ti_dead = ss.State('ti_dead', int, ss.INT_NAN)  # Maternal mortality

        self.pars = ss.omerge({
            'dur_pregnancy': 0.75,  # Make this a distribution?
            'dur_postpartum': 0.5,  # Make this a distribution?
            'fertility_rate': 0,    # Usually this will be provided in CSV format
            'rel_fertility': 1,
            'maternal_death_rate': 0,
            'sex_ratio': 0.5,       # Ratio of babies born female
            'units': 1e-3,             # ???
        }, self.pars)

        # Process metadata. Defaults here are the labels used by UN data
        self.metadata = ss.omerge({
            'data_cols': {'year': 'Time', 'age': 'AgeGrp', 'value': 'ASFR'},
        }, metadata)

        self.choose_slots = sps.randint(low=0, high=1) # Low and high will be reset upon initialization

        # Process data, which may be provided as a number, dict, dataframe, or series
        # If it's a number it's left as-is; otherwise it's converted to a dataframe
        self.pars.fertility_rate = self.standardize_fertility_data()

        # Create fertility_prob_fn, a function which returns a probability of death for each requested uid
        self.fertility_prob_fn = self.make_fertility_prob_fn
        self.fertility_dist = sps.bernoulli(p=self.fertility_prob_fn)

        # Add other sampling functions
        self.sex_dist = sps.bernoulli(p=self.pars.sex_ratio)
        self.death_dist = sps.bernoulli(p=self.pars.maternal_death_rate)

        return

    @staticmethod
    def make_fertility_prob_fn(module, sim, uids):
        """ Take in the module, sim, and uids, and return the conception probability for each UID on this timestep """

        if sc.isnumber(module.pars.fertility_rate):
            fertility_rate = module.pars.fertility_rate

        else:
            # Abbreviate key variables
            year_label = module.metadata.data_cols['year']
            age_label = module.metadata.data_cols['age']
            val_label = module.metadata.data_cols['value']

            available_years = module.pars.fertility_rate[year_label].unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]

            df = module.pars.fertility_rate.loc[module.pars.fertility_rate[year_label] == nearest_year]
            df_arr = df[val_label].values  # Pull out dataframe values
            df_arr = np.append(df_arr, 0)  # Add zeros for those outside data range

            # # Number of births over a given period classified by age group of mother
            # # We turn that into a probability of each woman of a given age conceiving
            # # First, we need to calculate how many women there are of each age
            # counts, bins1 = np.histogram(sim.people.age[(sim.people.female)], np.append(age_bins, 100))  # Append 100
            # scaled_counts = counts * sim.pars['pop_scale']

            # Process age data
            age_bins = df[age_label].unique()
            age_bins = np.append(age_bins, 50)
            age_inds = np.digitize(sim.people.age[uids], age_bins) - 1
            age_inds[age_inds>=max(age_inds)] = -1  # This ensures women outside the data range will get a value of 0

            # Make array of fertility rates
            fertility_rate = pd.Series(index=uids)
            fertility_rate[uids] = df_arr[age_inds]
            # fertility_rate[uids[sim.people.male[uids]]] = 0
            # fertility_rate[uids[(sim.people.age < 0)[uids]]] = 0

            # if sim.ti==1:
            #     import traceback;
            #     traceback.print_exc();
            #     import pdb;
            #     pdb.set_trace()

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
        self.choose_slots.kwds['low'] = sim.pars['n_agents']+1
        self.choose_slots.kwds['high'] = int(sim.pars['slot_scale']*sim.pars['n_agents'])
        return

    def init_results(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        self.results += ss.Result(self.name, 'pregnancies', sim.npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'births', sim.npts, dtype=int, scale=True)
        return

    def update(self, sim):
        """
        Perform all updates
        """
        self.make_pregnancies(sim)
        self.update_states(sim)
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
        self.susceptible[deliveries] = False
        self.ti_delivery[deliveries] = sim.ti

        # Check for new women emerging from post-partum
        postpartum = ~self.pregnant & (self.ti_postpartum == sim.ti)
        self.postpartum[postpartum] = False
        self.susceptible[postpartum] = True
        self.ti_postpartum[postpartum] = sim.ti

        # Maternal deaths
        maternal_deaths = ss.true(self.ti_dead <= sim.ti)
        sim.people.request_death(maternal_deaths)

        return

    def make_pregnancies(self, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate
        ppl = sim.people

        denom_conds = ppl.female & self.susceptible & ppl.alive
        inds_to_choose_from = ss.true(denom_conds)
        uids = self.fertility_dist.filter(inds_to_choose_from)
        #
        # import traceback;
        # traceback.print_exc();
        # import pdb;
        # pdb.set_trace()

        # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
        n_unborn_agents = len(uids)
        if n_unborn_agents > 0:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = sim.people.grow(len(new_slots))

            sim.people.age[new_uids] = -self.pars.dur_pregnancy
            sim.people.slot[new_uids] = new_slots # Before sampling female_dist
            sim.people.female[new_uids] = self.sex_dist.rvs(uids)

            # Add connections to any vertical transmission layers
            # Placeholder code to be moved / refactored. The maternal network may need to be
            # handled separately to the sexual networks, TBC how to handle this most elegantly
            for lkey, layer in sim.people.networks.items():
                if layer.vertical:  # What happens if there's more than one vertical layer?
                    durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy + self.pars.dur_postpartum)
                    layer.add_pairs(uids, new_uids, dur=durs)

            # Set prognoses for the pregnancies
            self.set_prognoses(sim, uids) # Could set from_uids to network partners?

        return

    def set_prognoses(self, sim, uids, from_uids=None):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """

        # Change states for the newly pregnant woman
        self.susceptible[uids] = False
        self.pregnant[uids] = True
        self.ti_pregnant[uids] = sim.ti

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.ti + self.pars.dur_pregnancy / sim.dt)
        dead = self.death_dist.rvs(uids)
        self.ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(uids), dur + self.pars.dur_postpartum / sim.dt)
        self.ti_postpartum[uids] = dur_post_partum

        if np.count_nonzero(dead):
            self.ti_dead[uids[dead]] = dur[dead]
        return

    def update_results(self, sim):
        self.results['pregnancies'][sim.ti] = np.count_nonzero(self.ti_pregnant == sim.ti)
        self.results['births'][sim.ti] = np.count_nonzero(self.ti_delivery == sim.ti)
        return
