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
    def __init__(self, pars=None):
        super().__init__(pars)

        # Set defaults
        self.pars = ss.omerge({
            'birth_rates': 0,
            'rel_birth': 1,
            'data_cols': {'year': 'Year', 'cbr': 'CBR'},
            'units_per_100': 1e-3  # assumes birth rates are per 1000. If using percentages, switch this to 1
        }, self.pars)

        # Validate birth rate inputs
        self.set_birth_rates(pars['birth_rates'])

    def set_birth_rates(self, birth_rates):
        """ Determine format that birth rates have been provided and standardize/validate """
        if sc.checktype(birth_rates, pd.DataFrame):
            if not set(self.pars.data_cols.values()).issubset(birth_rates.columns):
                errormsg = 'Please ensure the columns of the birth rate data match the values in pars.data_cols.'
                raise ValueError(errormsg)
        if sc.checktype(birth_rates, dict):
            if not set(self.pars.data_cols.values()).issubset(birth_rates.keys()):
                errormsg = 'Please ensure the keys of the birth rate data dict match the values in pars.data_cols.'
                raise ValueError(errormsg)
            birth_rates = pd.DataFrame(birth_rates)
        if sc.isnumber(birth_rates):
            birth_rates = pd.DataFrame({self.pars.data_cols['year']: [2000], self.pars.data_cols['cbr']: [birth_rates]})

        self.pars.birth_rates = birth_rates
        return


    def init_results(self, sim):
        self.results += ss.Result(self.name, 'new', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cumulative', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cbr', sim.npts, dtype=int)
        return

    def update(self, sim):
        new_uids = self.add_births(sim)
        self.update_results(len(new_uids), sim)
        return new_uids

    def get_birth_rate(self, sim):
        """
        Extract the right birth rates to use and translate it into a number of people to add.
        Eventually this might also process time series data.
        """
        p = self.pars
        br_year = p.birth_rates[p.data_cols['year']]
        br_val = p.birth_rates[p.data_cols['cbr']]
        this_birth_rate = np.interp(sim.year, br_year, br_val) * p.units_per_100 * p.rel_birth * sim.dt
        n_new = int(np.floor(np.count_nonzero(sim.people.alive) * this_birth_rate))
        return n_new

    def add_births(self, sim):
        # Add n_new births to each state in the sim
        n_new = self.get_birth_rate(sim)
        new_uids = sim.people.grow(n_new)
        return new_uids

    def update_results(self, n_new, sim):
        self.results['new'][sim.ti] = n_new

    def finalize(self, sim):
        super().finalize(sim)
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cbr'] = np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class background_deaths(DemographicModule):
    def __init__(self, pars=None):
        super().__init__(pars)

        self.pars = ss.omerge({
            'death_rates': 0,
            'rel_death': 1,
            'data_cols': {'year': 'Time', 'sex': 'Sex', 'age': 'AgeGrpStart', 'value': 'mx'},
            'sex_keys': {'f': 'Female', 'm': 'Male'},
            'units_per_100': 1e-3  # assumes birth rates are per 1000. If using percentages, switch this to 1
        }, self.pars)

        # Validate death rate inputs
        self.set_death_rates(self.pars['death_rates'])

        # Set death probs
        self.death_probs = ss.State('death_probs', float, 0)

    def set_death_rates(self, death_rates):
        """Standardize/validate death rates"""

        if sc.checktype(death_rates, pd.DataFrame):
            if not set(self.pars.data_cols.values()).issubset(death_rates.columns):
                errormsg = 'Please ensure the columns of the death rate data match the values in pars.data_cols.'
                raise ValueError(errormsg)
            df = death_rates

        elif sc.checktype(death_rates, pd.Series):
            if (death_rates.index < 120).all():  # Assume index is age bins
                df = pd.DataFrame({
                    self.pars.data_cols['year']: 2000,
                    self.pars.data_cols['age']: death_rates.index.values,
                    self.pars.data_cols['value']: death_rates.values,
                })
            elif (death_rates.index > 1900).all():  # Assume index year
                df = pd.DataFrame({
                    self.pars.data_cols['year']: death_rates.index.values,
                    self.pars.data_cols['age']: 0,
                    self.pars.data_cols['value']: death_rates.values,

                })
            else:
                errormsg = 'Could not understand index of death rate series: should be age or year.'
                raise ValueError(errormsg)

            df = pd.concat([df, df])
            df[self.pars.data_cols['sex']] = np.repeat(list(self.pars.sex_keys.values()), len(death_rates))

        elif sc.checktype(death_rates, dict):
            if not set(self.pars.data_cols.values()).issubset(death_rates.keys()):
                errormsg = 'Please ensure the keys of the death rate data dict match the values in pars.data_cols.'
                raise ValueError(errormsg)
            df = pd.DataFrame(death_rates)

        elif sc.isnumber(death_rates):
            df = pd.DataFrame({
                self.pars.data_cols['year']: [2000, 2000],
                self.pars.data_cols['age']: [0, 0],
                self.pars.data_cols['sex']: self.pars.sex_keys.values(),
                self.pars.data_cols['value']: [death_rates, death_rates],
            })

        else:
            errormsg = f'Death rate data type {type(death_rates)} not understood.'
            raise ValueError(errormsg)

        self.pars.death_rates = df

        return

    def init_results(self, sim):
        self.results += ss.Result(name='new', shape=sim.npts, dtype=int, scale=True)
        self.results += ss.Result(name='cumulative', shape=sim.npts, dtype=int, scale=True)
        self.results += ss.Result(name='mortality_rate', shape=sim.npts, dtype=int, scale=False)
        return

    def update(self, sim):
        n_deaths = self.apply_deaths(sim)
        self.update_results(n_deaths, sim)
        return

    def apply_deaths(self, sim):
        """ Select people to die """

        p = self.pars
        year_label = p.data_cols['year']
        age_label = p.data_cols['age']
        sex_label = p.data_cols['sex']
        val_label = p.data_cols['value']
        sex_keys = p.sex_keys

        available_years = p.death_rates[year_label].unique()
        year_ind = sc.findnearest(available_years, sim.year)
        nearest_year = available_years[year_ind]

        df = p.death_rates.loc[p.death_rates[year_label] == nearest_year]
        age_bins = df[age_label].unique()
        age_inds = np.digitize(sim.people.age, age_bins) - 1

        f_arr = df[val_label].loc[df[sex_label] == sex_keys['f']].values
        m_arr = df[val_label].loc[df[sex_label] == sex_keys['m']].values
        self.death_probs[sim.people.female] = f_arr[age_inds[sim.people.female]]
        self.death_probs[sim.people.male] = m_arr[age_inds[sim.people.male]]
        self.death_probs[sim.people.age < 0] = 0  # Don't use background death rates for unborn babies
        self.death_probs *= p.rel_death * sim.dt  # Adjust overall death probabilities by rel rates and dt

        # Get indices of people who die of other causes
        death_uids = ss.true(ss.binomial_arr(self.death_probs))
        death_uids = ss.true(sim.people.alive[death_uids])
        sim.people.request_death(death_uids)

        return len(death_uids)

    def update_results(self, n_deaths, sim):
        self.results['new'][sim.ti] = n_deaths

    def finalize(self, sim):
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['mortality_rate'] = np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class Pregnancy(DemographicModule):

    def __init__(self, pars=None):
        super().__init__(pars)

        # Other, e.g. postpartum, on contraception...
        self.infertile = ss.State('infertile', bool, False)  # Applies to girls and women outside the fertility window
        self.susceptible = ss.State('fecund', bool, True)  # Applies to girls and women inside the fertility window
        self.pregnant = ss.State('pregnant', bool, False)  # Currently pregnant
        self.postpartum = ss.State('postpartum', bool, False)  # Currently post-partum
        self.ti_pregnant = ss.State('ti_pregnant', int, ss.INT_NAN)  # Time pregnancy begins
        self.ti_delivery = ss.State('ti_delivery', int, ss.INT_NAN)  # Time of delivery
        self.ti_postpartum = ss.State('ti_postpartum', int, ss.INT_NAN)  # Time postpartum ends
        self.ti_dead = ss.State('ti_dead', int, ss.INT_NAN)  # Maternal mortality
        self.conception_probs = ss.State('conception_probs', float, 0)

        self.pars = ss.omerge({
            'dur_pregnancy': 0.75,  # Make this a distribution?
            'dur_postpartum': 0.5,  # Make this a distribution?
            'fertility_rates': 0,
            'data_cols': {'year': 'Time', 'age': 'AgeGrp', 'value': 'ASFR'},
            'units_per_100': 1e-3,  # assumes fertility rates are per 1000. If using percentages, switch this to 1
            'p_death': sps.bernoulli(p=0),  # Probability of maternal death. Question, should this be linked to age and/or duration?
        }, self.pars)

        # Validate fertility rate inputs
        self.set_fertility_rates(self.pars['fertility_rates'])

        return

    def set_fertility_rates(self, fertility_rates):
        """ Validate fertility rates """
        if sc.checktype(fertility_rates, pd.DataFrame):
            if not set(self.pars.data_cols.values()).issubset(fertility_rates.columns):
                errormsg = 'Please ensure the columns of the fertility rate data match the values in pars.data_cols.'
                raise ValueError(errormsg)
            df = fertility_rates

        elif sc.checktype(fertility_rates, pd.Series):
            if (fertility_rates.index < 120).all():  # Assume index is age bins
                df = pd.DataFrame({
                    self.pars.data_cols['year']: 2000,
                    self.pars.data_cols['age']: fertility_rates.index.values,
                    self.pars.data_cols['value']: fertility_rates.values,
                })
            elif (fertility_rates.index > 1900).all():  # Assume index year
                df = pd.DataFrame({
                    self.pars.data_cols['year']: fertility_rates.index.values,
                    self.pars.data_cols['age']: 0,
                    self.pars.data_cols['value']: fertility_rates.values,

                })
            else:
                errormsg = 'Could not understand index of fertility rate series: should be age or year.'
                raise ValueError(errormsg)

        elif sc.checktype(fertility_rates, dict):
            if not set(self.pars.data_cols.values()).issubset(fertility_rates.keys()):
                errormsg = 'Please ensure the keys of the fertility rate data dict match the values in pars.data_cols.'
                raise ValueError(errormsg)
            df = pd.DataFrame(fertility_rates)

        elif sc.isnumber(fertility_rates):
            df = pd.DataFrame({
                self.pars.data_cols['year']: [2000],
                self.pars.data_cols['age']: [15],
                self.pars.data_cols['value']: fertility_rates,
            })

        else:
            errormsg = f'Fertility rate data type {type(fertility_rates)} not understood.'
            raise ValueError(errormsg)

        self.pars.fertility_rates = df

        self.choose_slots = sps.randint(low=0, high=1) # Low and high will be reset upon initialization

        return

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
        self.results += ss.Result(name='pregnancies', shape=sim.npts, dtype=int, scale=True)
        self.results += ss.Result(name='births', shape=sim.npts, dtype=int, scale=True)
        return

    def update(self, sim):
        """
        Perform all updates
        """
        self.update_states(sim)
        self.make_pregnancies(sim)
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
        postpartum = ~self.pregnant & (self.ti_postpartum <= sim.ti)
        self.postpartum[postpartum] = False
        self.susceptible[postpartum] = True
        self.ti_postpartum[postpartum] = sim.ti

        # Maternal deaths
        maternal_deaths = ss.true(self.ti_dead <= sim.ti)
        sim.people.request_death(maternal_deaths)

        return

    def make_pregnancies(self, sim):
        """
        Select people to make pregnancy using fertility rates
        """

        # Abbreviate key variables
        p = self.pars
        year_label = p.data_cols['year']
        age_label = p.data_cols['age']
        val_label = p.data_cols['value']

        available_years = p.fertility_rates[year_label].unique()
        year_ind = sc.findnearest(available_years, sim.year)
        nearest_year = available_years[year_ind]

        df = p.fertility_rates.loc[p.fertility_rates[year_label] == nearest_year]
        age_bins = df[age_label].unique()
        age_bins = np.append(age_bins, 50)  # Add 50, and specify no births after 50
        age_inds = np.digitize(sim.people.age, age_bins) - 1

        try:
            conception_eligible = sim.people.female & (age_inds >= 0) & (age_inds < max(age_inds))
        except:
            import traceback;
            traceback.print_exc();
            import pdb;
            pdb.set_trace()
        conception_probs = df[val_label].values * p.units_per_100 * sim.dt
        self.conception_probs[conception_eligible] = conception_probs[age_inds[conception_eligible]]
        conceive_uids = ss.true(ss.binomial_arr(self.conception_probs))


        # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
        n_unborn_agents = len(conceive_uids)
        if n_unborn_agents > 0:
            # Grow the arrays and set properties for the unborn agents
            new_uids = sim.people.grow(n_unborn_agents)
            sim.people.age[new_uids] = -self.pars.dur_pregnancy

            # Add connections to any vertical transmission layers
            # Placeholder code to be moved / refactored. The maternal network may need to be
            # handled separately to the sexual networks, TBC how to handle this most elegantly
            for lkey, layer in sim.people.networks.items():
                if layer.vertical:  # What happens if there's more than one vertical layer?
                    durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy + self.pars.dur_postpartum)
                    layer.add_pairs(conceive_uids, new_uids, dur=durs)

            # Set prognoses for the pregnancies
            self.set_prognoses(sim, conceive_uids)

        return

    def set_prognoses(self, sim, uids):
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
        dead = self.pars.p_death.rvs(uids)
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
