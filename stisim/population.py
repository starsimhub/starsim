'''
Defines functions for making the population.
'''

#%% Imports
import numpy as np
import sciris as sc
from . import utils as ssu
from . import misc as ssm
from . import data as ssdata
from . import defaults as ssd
from . import people as ssppl
from . import base as ssb


# Specify all externally visible functions this file defines
__all__ = ['make_people']


def make_people(sim, popdict=None, reset=False, verbose=None, use_age_data=True,
                sex_ratio=0.5, dt_round_age=True, **kwargs):
    '''
    Make the people for the simulation.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict or People object
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        verbose  (bool) : level of detail to print
        use_age_data (bool):
        sex_ratio (bool):
        dt_round_age (bool): whether to round people's ages to the nearest timestep (default true)

    Returns:
        people (People): people
    '''

    # Set inputs and defaults
    n_agents = int(sim['n_agents']) # Shorten
    total_pop = None # Optionally created but always returned
    pop_trend = None # Populated later if location is specified
    pop_age_trend = None

    if verbose is None:
        verbose = sim['verbose']
    dt = sim['dt'] # Timestep

    # If a people object or popdict is supplied, use it
    if sim.people and not reset:
        sim.people.initialize(sim_pars=sim.pars)
        return sim.people, total_pop # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict # Use stored one
        sim.popdict = None # Once loaded, remove

    if popdict is None:

        n_agents = int(sim['n_agents']) # Number of people
        total_pop = None

        # Load age data by country if available, or use defaults.
        # Other demographic data like mortality and fertility are also available by
        # country, but these are loaded directly into the sim since they are not
        # stored as part of the people.
        location = sim['location']
        if sim['verbose']:
            print(f'Loading location-specific data for "{location}"')
        if use_age_data:
            try:
                age_data = ssdata.get_age_distribution(location, year=sim['start'])
                pop_trend = ssdata.get_total_pop(location)
                total_pop = sum(age_data[:, 2])  # Return the total population
                pop_age_trend = ssdata.get_age_distribution_over_time(location)
            except ValueError as E:
                warnmsg = f'Could not load age data for requested location "{location}" ({str(E)})'
                ssm.warn(warnmsg, die=True)

        uids, sexes, debut = set_static_demog(n_agents, pars=sim.pars, sex_ratio=sex_ratio)

        # Set ages, rounding to nearest timestep if requested
        age_data_min   = age_data[:,0]
        age_data_max   = age_data[:,1]
        age_data_range = age_data_max - age_data_min
        age_data_prob   = age_data[:,2]
        age_data_prob   /= age_data_prob.sum() # Ensure it sums to 1
        age_bins        = ssu.n_multinomial(age_data_prob, n_agents) # Choose age bins
        if dt_round_age:
            ages = age_data_min[age_bins] + np.random.randint(age_data_range[age_bins]/dt)*dt # Uniformly distribute within this age bin
        else:
            ages = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(n_agents) # Uniformly distribute within this age bin

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['age'] = ages
        popdict['sex'] = sexes
        popdict['debut'] = debut

    else:
        ages = popdict['age']

    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    people = ssppl.People(sim.pars, pop_trend=pop_trend, pop_age_trend=pop_age_trend, **popdict) # List for storing the people

    sc.printv(f'Created {n_agents} agents, average age {ages.mean():0.2f} years', 2, verbose)

    return people, total_pop


def set_static_demog(new_n, existing_n=0, pars=None, sex_ratio=0.5):
    '''
    Set static population characteristics that do not change over time.
    Can be used when adding new births, in which case the existing popsize can be given.
    '''
    uid             = np.arange(existing_n, existing_n+new_n, dtype=ssd.default_int)
    sex             = np.random.binomial(1, sex_ratio, new_n)
    debut           = np.full(new_n, np.nan, dtype=ssd.default_float)
    debut[sex==1]   = ssu.sample(**pars['debut']['m'], size=sum(sex))
    debut[sex==0]   = ssu.sample(**pars['debut']['f'], size=new_n-sum(sex))

    return uid, sex, debut


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys() # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or hpv.People object, but instead is {type(popdict)}'
        raise TypeError(errormsg) from E

    # Check keys and lengths
    required_keys = ['uid', 'age', 'sex', 'debut']
    popdict_keys = popdict.keys()
    n_agents = pars['n_agents']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != n_agents:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {n_agents}'
            raise ValueError(errormsg)

        isnan = np.isnan(popdict[key]).sum()
        if isnan:
            errormsg = f'Population not fully created: {isnan:,} NaNs found in {key}.'
            raise ValueError(errormsg)

    return


class DynamicSexualLayer(ssb.Layer):
    def __init__(self, pars=None):
        super().__init__()

        # Define default parameters
        self.pars = dict()
        self.pars['cross_layer'] = 0.05  # Proportion of agents who have concurrent cross-layer relationships
        self.pars['partners'] = dict(dist='poisson', par1=0.01)  # The number of concurrent sexual partners
        self.pars['acts'] = dict(dist='neg_binomial', par1=80, par2=40)  # The number of sexual acts per year
        self.pars['age_act_pars'] = dict(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1) # Parameters describing changes in coital frequency over agent lifespans
        self.pars['condoms'] = 0.2  # The proportion of acts in which condoms are used
        self.pars['dur_pship'] = dict(dist='normal_pos', par1=1, par2=1)  # Duration of partnerships
        self.pars['participation'] = None  # Incidence of partnership formation by age
        self.pars['mixing'] = None  # Mixing matrices for storing age differences in partnerships

        self.update_pars(pars)
        self.get_layer_probs()

    def initialize(self, people):
        self.add_partnerships(people)

    def update_pars(self, pars):
        if pars is not None:
            for k,v in pars.items():
                self.pars[k] = v
        return

    def get_layer_probs(self):

        defaults = {}
        mixing = np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ])

        participation = np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,    55,    60,    65,    70,    75],
                [ 0,  0,  0.10,   0.7,  0.8,  0.6,  0.6,  0.4,   0.1,  0.05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], # Share of females of each age newly having casual relationships
                [ 0,  0,  0.05,   0.7,  0.8,  0.6,  0.6,  0.4,   0.4,   0.3,   0.1,  0.05,  0.01,  0.01, 0.001, 0.001]], # Share of males of each age newly having casual relationships
            )

        defaults['mixing'] = mixing
        defaults['participation'] = participation

        for pkey, pval in defaults.items():
            if self.pars[pkey] is None:
                self.pars[pkey] = pval

        return

    def add_partnerships(self, people):

        is_female = people.is_female
        is_active = people.is_active
        f_active = is_female & is_active
        m_active = ~is_female & is_active

        # Compute number of partners
        f_partnered_inds, f_partnered_counts = np.unique(self['p1'], return_counts=True)
        m_partnered_inds, m_partnered_counts = np.unique(self['p2'], return_counts=True)
        current_partners = np.zeros((len(people)))
        current_partners[f_partnered_inds] = f_partnered_counts
        current_partners[m_partnered_inds] = m_partnered_counts
        partners = ssu.sample(**self.pars['partners'], size=len(people)) + 1
        underpartnered = current_partners < partners  # Indices of underpartnered people
        f_eligible = f_active & underpartnered
        m_eligible = m_active & underpartnered

        # Bin the agents by age
        bins = self.pars['participation'][0, :]  # Extract age bins

        # Try randomly select females for pairing
        f_eligible_inds = ssu.true(f_eligible)  # Inds of all eligible females
        age_bins_f = np.digitize(people.age[f_eligible_inds], bins=bins) - 1  # Age bins of selected females
        bin_range_f = np.unique(age_bins_f)  # Range of bins
        f = []  # Initialize the female partners
        for ab in bin_range_f:  # Loop over age bins
            these_f_contacts = ssu.binomial_filter(self.pars['participation'][1][ab], f_eligible_inds[
                age_bins_f == ab])  # Select females according to their participation rate in this layer
            f += these_f_contacts.tolist()

        # Select males according to their participation rate in this layer
        m_eligible_inds = ssu.true(m_eligible)
        age_bins_m = np.digitize(people.age[m_eligible_inds], bins=bins) - 1
        bin_range_m = np.unique(age_bins_m)  # Range of bins
        m = []  # Initialize the male partners
        for ab in bin_range_m:
            these_m_contacts = ssu.binomial_filter(self.pars['participation'][2][ab], m_eligible_inds[
                age_bins_m == ab])  # Select males according to their participation rate in this layer
            m += these_m_contacts.tolist()

        # Create preference matrix between eligible females and males that combines age and geo mixing
        age_bins_f = np.digitize(people.age[f], bins=bins) - 1  # Age bins of females that are entering new relationships
        age_bins_m = np.digitize(people.age[m], bins=bins) - 1  # Age bins of active and participating males
        age_f, age_m = np.meshgrid(age_bins_f, age_bins_m)
        pair_probs = self.pars['mixing'][age_m, age_f + 1]

        f_to_remove = pair_probs.max(axis=0) == 0  # list of female inds to remove if no male partners are found for her
        f = [i for i, flag in zip(f, f_to_remove) if ~flag]  # remove the inds who don't get paired on this timestep
        selected_males = []
        if len(f):
            pair_probs = pair_probs[:, np.invert(f_to_remove)]
            choices = []
            fems = np.arange(len(f))
            f_paired_bools = np.full(len(fems), True, dtype=bool)
            np.random.shuffle(fems)
            for fem in fems:
                m_col = pair_probs[:, fem]
                if m_col.sum() > 0:
                    m_col_norm = m_col / m_col.sum()
                    choice = np.random.choice(len(m_col_norm), 1, replace=False, p=m_col_norm)
                    choices.append(choice)
                    pair_probs[choice, :] = 0  # Once male partner is assigned, remove from eligible pool
                else:
                    f_paired_bools[fem] = False
            selected_males = np.array(m)[np.array(choices).flatten()]
            f = np.array(f)[f_paired_bools]

        p1 = np.array(f)
        p2 = selected_males
        n_partnerships = len(p1)
        dur = ssu.sample(**self.pars['dur_pship'], size=n_partnerships)
        acts = ssu.sample(**self.pars['acts'], size=n_partnerships)
        age_p1 = people.age[p1]
        age_p2 = people.age[p2]

        age_debut_p1 = people.debut[p1]
        age_debut_p2 = people.debut[p2]

        # For each couple, get the average age they are now and the average age of debut
        avg_age = np.array([age_p1, age_p2]).mean(axis=0)
        avg_debut = np.array([age_debut_p1, age_debut_p2]).mean(axis=0)

        # Shorten parameter names
        dr = self.pars['age_act_pars']['debut_ratio']
        peak = self.pars['age_act_pars']['peak']
        rr = self.pars['age_act_pars']['retirement_ratio']
        retire = self.pars['age_act_pars']['retirement']

        # Get indices of people at different stages
        below_peak_inds = avg_age <= self.pars['age_act_pars']['peak']
        above_peak_inds = (avg_age > self.pars['age_act_pars']['peak']) & (avg_age < self.pars['age_act_pars']['retirement'])
        retired_inds = avg_age > self.pars['age_act_pars']['retirement']

        # Set values by linearly scaling the number of acts for each partnership according to
        # the age of the couple at the commencement of the relationship
        below_peak_vals = acts[below_peak_inds] * (dr + (1 - dr) / (peak - avg_debut[below_peak_inds]) * (
                    avg_age[below_peak_inds] - avg_debut[below_peak_inds]))
        above_peak_vals = acts[above_peak_inds] * (
                    rr + (1 - rr) / (peak - retire) * (avg_age[above_peak_inds] - retire))
        retired_vals = 0

        # Set values and return
        scaled_acts = np.full(len(acts), np.nan, dtype=ssd.default_float)
        scaled_acts[below_peak_inds] = below_peak_vals
        scaled_acts[above_peak_inds] = above_peak_vals
        scaled_acts[retired_inds] = retired_vals
        start = np.array([people.t] * n_partnerships, dtype=ssd.default_float)
        end = start + dur/people.pars['dt']

        new_contacts = dict(
            p1=p1,
            p2=p2,
            dur=dur,
            acts=acts,
            start=start,
            end=end
        )
        self.append(new_contacts)
        return

    def update(self, people):
        # First remove any relationships due to end
        self['dur'] = self['dur'] - people.dt
        active = self['dur'] > 0
        for key in self.meta.keys():
            self[key] = self[key][active]

        # Then add new relationships
        self.add_partnerships(people)
        return