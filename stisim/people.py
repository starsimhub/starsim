'''
Defines the People class and functions associated with making people and handling
the transitions between states (e.g., from susceptible to infected).
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as ssu
from . import defaults as ssd
from . import base as ssb
from . import population as sspop


__all__ = ['People']

class People(ssb.BasePeople):
    '''
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people, while
    ``ss.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        ppl1 = ss.People(2000)
        sim = ss.Sim()
        ppl2 = ss.People(sim.pars)
    '''
    
    #%% Basic methods

    def __init__(self, pars, strict=True, pop_trend=None, pop_age_trend=None, **kwargs):
        '''
        Initialize with pars
        '''
        super().__init__(pars)
        self.pop_trend = pop_trend
        self.pop_age_trend = pop_age_trend
        self.age_bin_edges = self.pars['age_bin_edges'] # Age bins for age results
        self.contacts = sc.objdict() # Create
        self._modules = []
        self.module_states = sc.objdict()
        self.state_names = sc.autolist([state.name for state in self.meta.states_to_set])

        if strict:
            self.lock() # If strict is true, stop further keys from being set (does not affect attributes)

        self.initialized = False
        self.kwargs = kwargs
        return

    def add_module(self, module, force=False):
        # Initialize all of the states associated with a module
        # This is implemented as People.add_module rather than
        # Module.add_to_people(people) or similar because its primary
        # role is to modify the People object
        if hasattr(self, module.name)and not force:
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())
        for state in module.states.values():
            combined_name = module.name + '.' + state.name
            self.state_names += combined_name
            self.module_states[combined_name] = state
            self._data[combined_name] = state.new(self.pars, self._n, module=module.name)
        self._map_arrays() # TODO: do not remap all arrays every time a module is added
        return


    def init_flows(self):
        ''' Initialize flows to be zero '''
        df = ssd.default_float
        dem_keys = ['births', 'other_deaths', 'migration']
        by_sex_keys = ['other_deaths_by_sex']
        self.demographic_flows = {f'{key}': 0 for key in dem_keys}
        self.sex_flows         = {f'{key}': np.zeros(2, dtype=df) for key in by_sex_keys}
        return

    def scale_flows(self, inds):
        '''
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        '''
        return self.scale[inds].sum()

    def increment_age(self):
        ''' Let people age by one timestep '''
        self.age[self.alive] += self.dt
        return


    def initialize(self, sim_pars=None):
        ''' Perform initializations '''
        super().initialize() # Initialize states
        kwargs = self.kwargs

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if self._lock:
                self.set(key, value)
            elif key in self._data:
                self[key][:] = value
            else:
                self[key] = value
        
        # Set the scale factor
        self.scale[:] = sim_pars['pop_scale']

        # Additional validation
        self.validate(sim_pars=sim_pars) # First, check that essential-to-match parameters match
        self.initialized = True

        return


    def update_states(self, t, sim):
        ''' Perform all state updates at the current timestep '''
        # Initialize
        self.init_flows()
        self.t = t
        self.dt = self.pars['dt']

        for module in sim.modules.values():
            module.update_states(sim)

        # Perform network updates
        for lkey, layer in self.contacts.items():
            layer.update(self)

        return

    def update_demography(self, t, year=None):

        self.increment_age()

        update_freq = max(1, int(self.pars['dt_demog'] / self.pars['dt']))  # Ensure it's an integer not smaller than 1
        if t % update_freq == 0:
            # Apply death rates from other causes
            other_deaths, deaths_female, deaths_male = self.apply_death_rates(year=year)
            self.demographic_flows['other_deaths'] = other_deaths
            self.sex_flows['other_deaths_by_sex'][0] = deaths_female
            self.sex_flows['other_deaths_by_sex'][1] = deaths_male

            # Add births if pregnancy not in modules
            new_births = self.add_people(year=year)
            self.demographic_flows['births'] = new_births

            # Check migration
            migration = self.check_migration(year=year)
            self.demographic_flows['migration'] = migration


    #%% Methods for updating state
    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = ssu.false(current)
        else:
            not_current = ssu.ifalsei(current, filter_inds)
        has_date = ssu.idefinedi(date, not_current)
        inds     = ssu.itrue(self.t >= date[has_date], has_date)
        return inds


    def check_inds_true(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is true and which meet the date criterion '''
        if filter_inds is None:
            current_inds = ssu.true(current)
        else:
            current_inds = ssu.itruei(current, filter_inds)
        has_date = ssu.idefinedi(date, current_inds)
        inds     = ssu.itrue(self.t >= date[has_date], has_date)
        return inds


    def apply_death_rates(self, year=None):
        '''
        Apply death rates to remove people from the population
        NB people are not actually removed to avoid issues with indices
        '''

        death_pars = self.pars['death_rates']
        if death_pars:
            all_years = np.array(list(death_pars.keys()))
            base_year = all_years[0]
            age_bins = death_pars[base_year]['m'][:,0]
            age_inds = np.digitize(self.age, age_bins)-1
            death_probs = np.empty(len(self), dtype=ssd.default_float)
            year_ind = sc.findnearest(all_years, year)
            nearest_year = all_years[year_ind]
            mx_f = death_pars[nearest_year]['f'][:,1]*self.pars['dt_demog']
            mx_m = death_pars[nearest_year]['m'][:,1]*self.pars['dt_demog']
    
            death_probs[self.is_female] = mx_f[age_inds[self.is_female]]
            death_probs[self.is_male] = mx_m[age_inds[self.is_male]]
            death_probs[self.age>100] = 1 # Just remove anyone >100
            death_probs[~self.alive] = 0
            death_probs *= self.pars['rel_death'] # Adjust overall death probabilities
    
            # Get indices of people who die of other causes
            death_inds = ssu.true(ssu.binomial_arr(death_probs))
        else:
            death_inds = np.array([], dtype=int)
            
        deaths_female = self.scale_flows(ssu.true(self.is_female[death_inds]))
        deaths_male = self.scale_flows(ssu.true(self.is_male[death_inds]))
        other_deaths = self.remove_people(death_inds, cause='other') # Apply deaths

        return other_deaths, deaths_female, deaths_male


    def add_people(self, year=None, new_people=None, ages=0):
        '''
        Add more people to the population

        Specify either the year from which to retrieve the birth rate, or the absolute number
        of new people to add. Must specify one or the other. People are added in-place to the
        current `People` instance.
        '''

        assert (year is None) != (new_people is None), 'Must set either year or n_births, not both'

        if new_people is None:
            if self.pars['birth_rates'] is None: # Births taken care of via pregnancy module
                new_people = 0
            else:
                years = self.pars['birth_rates'][0]
                rates = self.pars['birth_rates'][1]
                this_birth_rate = self.pars['rel_birth']*np.interp(year, years, rates)*self.pars['dt_demog']/1e3
                new_people = sc.randround(this_birth_rate*self.n_alive) # Crude births per 1000

        if new_people>0:
            # Generate other characteristics of the new people
            uids, sexes, debuts = sspop.set_static_demog(new_n=new_people, existing_n=len(self), pars=self.pars)
            # Grow the arrays`
            new_inds = self._grow(new_people)
            self.uid[new_inds]          = uids
            self.age[new_inds]          = ages
            self.sex[new_inds]          = sexes
            self.debut[new_inds]        = debuts


        return new_people*self.pars['pop_scale']


    def check_migration(self, year=None):
        """
        Check if people need to immigrate/emigrate in order to make the population
        size correct.
        """

        if self.pars['use_migration'] and self.pop_trend is not None:

            # Pull things out
            sim_start = self.pars['start']
            sim_pop0 = self.pars['n_agents']
            data_years = self.pop_trend.year.values
            data_pop = self.pop_trend.pop_size.values
            data_min = data_years[0]
            data_max = data_years[-1]
            age_dist_data = self.pop_age_trend[self.pop_age_trend.year == int(year)]

            # No migration if outside the range of the data
            if year < data_min:
                return 0
            elif year > data_max:
                return 0
            if sim_start < data_min: # Figure this out later, can't use n_agents then
                errormsg = 'Starting the sim earlier than the data is not hard, but has not been done yet'
                raise NotImplementedError(errormsg)

            # Do basic calculations
            data_pop0 = np.interp(sim_start, data_years, data_pop)
            scale = sim_pop0 / data_pop0 # Scale factor
            alive_inds = ssu.true(self.alive)
            alive_ages = self.age[alive_inds].astype(int) # Return ages for everyone level 0 and alive
            count_ages = np.bincount(alive_ages, minlength=age_dist_data.shape[0]) # Bin and count them
            expected = age_dist_data['PopTotal'].values*scale # Compute how many of each age we would expect in population
            difference = (expected-count_ages).astype(int) # Compute difference between expected and simulated for each age
            n_migrate = np.sum(difference) # Compute total migrations (in and out)
            ages_to_remove = ssu.true(difference<0) # Ages where we have too many, need to apply emigration
            n_to_remove = difference[ages_to_remove] # Determine number of agents to remove for each age
            ages_to_add = ssu.true(difference>0) # Ages where we have too few, need to apply imigration
            n_to_add = difference[ages_to_add] # Determine number of agents to add for each age
            ages_to_add_list = np.repeat(ages_to_add, n_to_add)
            self.add_people(new_people=len(ages_to_add_list), ages=np.array(ages_to_add_list))

            # Remove people
            remove_frac = n_to_remove / count_ages[ages_to_remove]
            remove_probs = np.zeros(len(self))
            for ind,rf in enumerate(remove_frac):
                age = ages_to_remove[ind]
                inds_this_age = ssu.true((self.age>=age) * (self.age<age+1) * self.alive)
                remove_probs[inds_this_age] = -rf
            migrate_inds = ssu.choose_w(remove_probs, -n_to_remove.sum())
            self.remove_people(migrate_inds, cause='emigration')  # Remove people

        else:
            n_migrate = 0

        return n_migrate*self.pars['pop_scale'] # These are not indices, so they scale differently


    #%% Methods to make events occur (death, others TBC)

    def remove_people(self, inds, cause=None):
        ''' Remove people - used for death and migration '''

        if cause == 'other':
            self.date_dead_other[inds] = self.t
            self.dead_other[inds] = True
        elif cause == 'emigration':
            self.emigrated[inds] = True
        else:
            errormsg = f'Cause of death must be one of "other", or "emigration", not {cause}.'
            raise ValueError(errormsg)

        # Set states to false
        self.alive[inds] = False

        return self.scale_flows(inds)

