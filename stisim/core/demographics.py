"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import stisim as ss
import sciris as sc
import pandas as pd
import scipy.stats as sps

from . import Module
__all__ = ['DemographicModule', 'births', 'background_deaths', 'Pregnancy']


class DemographicModule(Module):
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
        this_birth_rate = np.interp(sim.year, br_year, br_val) * p.units_per_100 * p.rel_birth * sim.pars.dt_demog
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
            #'rel_death': 1,
            #'units_per_100': 1e-3,  # assumes birth rates are per 1000. If using percentages, switch this to 1
            'death_prob_func': self.death_prob,
        }, self.pars)
        self.death_prob_dist = sps.bernoulli(p=self.pars['death_prob_func'])
        return

    @staticmethod
    def death_prob(self, sim, uids):
        #p = self.pars
        #year = p.death_rates[p.data_cols['year']]
        #val = p.death_rates[p.data_cols['cbr']]
        #this_death_rate = np.interp(sim.year, year, val) * p.units_per_100 * p.rel_death * sim.pars.dt_demog
        #n_new = int(np.floor(np.count_nonzero(sim.people.alive) * this_death_rate))
        return 0.01

    def init_results(self, sim):
        self.results += ss.Result(self.name, 'new', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cumulative', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cmr', sim.npts, dtype=int)
        return

    def update(self, sim):
        n_deaths = self.apply_deaths(sim)
        self.update_results(n_deaths, sim)
        return

    def apply_deaths(self, sim):
        """ Select people to die """
        alive_uids = ss.true(sim.people.alive)
        death_uids = self.death_prob_dist.filter(alive_uids)
        sim.people.request_death(death_uids)

        return len(death_uids)

    def update_results(self, n_deaths, sim):
        self.results['new'][sim.ti] = n_deaths

    def finalize(self, sim):
        self.results['cumulative'] = np.cumsum(self.results['new'])
        self.results['cmr'] = np.divide(self.results['new'], sim.results['n_alive'], where=sim.results['n_alive']>0)


class Pregnancy(DemographicModule):

    def __init__(self, pars=None):
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
            'pregnancy_prob_per_dt': sps.bernoulli(p=0.3),  # Probabilty of acquiring pregnancy on each time step. Can replace with callable parameters for age-specific rates, etc. NOTE: You will manually need to adjust for the simulation timestep dt!
            'p_death': sps.bernoulli(p=0.15),  # Probability of maternal death.
            'p_female': sps.bernoulli(p=0.5),
            'init_prev': 0.3,  # Number of women initially pregnant # TODO: Default value
        }, self.pars)

        self.choose_slots = sps.randint(low=0, high=1) # Low and high will be reset upon initialization
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.choose_slots.kwds['low'] = sim.pars['n_agents']+1
        self.choose_slots.kwds['high'] = int(sim.pars['slot_scale']*sim.pars['n_agents'])
        sim.pars['birth_rates'] = None  # This turns off birth rate pars so births only come from this module
        return

    def init_results(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        self.results += ss.Result(self.name, 'pregnancies', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'births', sim.npts, dtype=int)
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
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate key variables
        ppl = sim.people

        # If incidence of pregnancy is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        denom_conds = ppl.female & ppl.active & self.susceptible
        inds_to_choose_from = ss.true(denom_conds)
        uids = self.pars['pregnancy_prob_per_dt'].filter(inds_to_choose_from)

        # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
        n_unborn_agents = len(uids)
        if n_unborn_agents > 0:

            # Choose slots for the unborn agents
            new_slots = self.choose_slots.rvs(uids)

            # Grow the arrays and set properties for the unborn agents
            new_uids = sim.people.grow(len(new_slots))

            sim.people.age[new_uids] = -self.pars.dur_pregnancy
            sim.people.slot[new_uids] = new_slots # Before sampling female_dist
            sim.people.female[new_uids] = self.pars['p_female'].rvs(uids)

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
