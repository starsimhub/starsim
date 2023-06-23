'''
Set parameters
'''

import numpy as np
import sciris as sc
import pandas as pd
from .settings import options as sso # For setting global options
from . import misc as ssm
from . import utils as ssu
from . import defaults as ssd
from . import population as sspop
from .data import loaders as ssdata

__all__ = ['make_pars']


def make_pars(**kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        version       (str):  if supplied, use parameters from this version
        kwargs        (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = sc.objdict()

    # Population parameters
    pars['n_agents']        = 10e3      # Number of agents
    pars['total_pop']       = None      # If defined, used for calculating the scale factor
    pars['pop_scale']       = None      # How much to scale the population
    pars['location']        = 'nigeria' # What location to load data from -- default Nigeria
    pars['lx']              = None      # Proportion of people alive at the beginning of age interval x
    pars['birth_rates']     = None      # Birth rates, loaded below
    pars['death_rates']     = None      # Death rates, loaded below
    pars['rel_birth']       = 1.0       # Birth rate scale factor
    pars['rel_death']       = 1.0       # Death rate scale factor

    # Simulation parameters
    pars['start']           = 1995.         # Start of the simulation
    pars['end']             = None          # End of the simulation
    pars['n_years']         = 35            # Number of years to run, if end isn't specified. Note that this includes burn-in
    pars['burnin']          = 25            # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
    pars['dt']              = 1.0           # Timestep (in years)
    pars['dt_demog']        = 1.0           # Timestep for demographic updates (in years)
    pars['rand_seed']       = 1             # Random seed, if None, don't reset
    pars['verbose']         = sso.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
    pars['use_migration']   = True          # Whether to estimate migration rates to correct the total population size

    # Events and interventions
    pars['connectors']      = sc.autolist()
    pars['interventions']   = sc.autolist() # The interventions present in this simulation; populated by the user
    pars['analyzers']       = sc.autolist() # The functions present in this simulation; populated by the user
    pars['timelimit']       = None # Time limit for the simulation (seconds)
    pars['stopping_func']   = None # A function to call to stop the sim partway through

    # Population distribution of the World Standard Population, used to calculate age-standardised rates (ASR)
    pars['age_bin_edges']        = np.array( [  0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,  75,    80,    85, 100])
    pars['standard_pop']    = np.array([pars['age_bin_edges'],
                                        [.12, .10, .09, .09, .08, .08, .06, .06, .06, .06, .05, .04, .04, .03, .02, .01, 0.005, 0.005,   0]])

    # Network parameters, generally initialized after the population has been constructed
    pars['network_pars']    = sc.autolist() # Network types and parameters
    pars['geostructure']    = 1     # Defines how many geographic clusters there should be in the simulated population
    pars['debut']           = dict(f=dict(dist='normal', par1=15.0, par2=2.1), # Location-specific data should be used here if possible
                                   m=dict(dist='normal', par1=17.6, par2=1.8))

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)

    return pars


def get_births_deaths(location, verbose=1, by_sex=True, overall=False, die=True):
    '''
    Get mortality and fertility data by location if provided, or use default

    Args:
        location (str):  location
        verbose (bool):  whether to print progress
        by_sex   (bool): whether to get sex-specific death rates (default true)
        overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

    Returns:
        lx (dict): dictionary keyed by sex, storing arrays of lx - the number of people who survive to age x
        birth_rates (arr): array of crude birth rates by year
    '''

    if verbose:
        print(f'Loading location-specific demographic data for "{location}"')
    try:
        death_rates = ssdata.get_death_rates(location=location, by_sex=by_sex, overall=overall)
        birth_rates = ssdata.get_birth_rates(location=location)
        return birth_rates, death_rates
    except ValueError as E:
        warnmsg = f'Could not load demographic data for requested location "{location}" ({str(E)})'
        ssm.warn(warnmsg, die=die)