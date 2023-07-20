"""
Default parameters to run a simulation, could also be loaded from a json file
"""

import sciris as sc
from .settings import options as sso  # For setting global options
from .parameters import BaseParameter, ParameterSet

__all__ = ['make_default_pars', 'build_default_pars', 'make_default_parset', 'default_pars_dict',
           'get_default_parameter', 'build_pars']


def build_default_pars(**kwargs):
    """
    Build a dictionary with BaseParameter instances using default parameters from default_pars_dict
    """
    pars = build_pars(default_pars_dict())
    return pars


def make_default_parset():
    """
    NOTE: return the default parameter dictionary as a ParameterSet.
    """
    return ParameterSet(build_pars(default_pars_dict()))


def default_pars_dict():
    """
    TODO: Provisional way of defining default parameters, need to shorten or
    provide it as a default ParameterSet
    """
    default_parameters = {
        'n_agents': {
            'name': 'n_agents',
            'dtype': int,
            'default_value': 10e3,
            'ptype': 'required',
            'valid_range': (2, None),
            'category': ["simulation", "people", "networks"],
            'validator': None,
            'label': None,
            'description': 'Number of agents.',
            'units': 'agents',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'total_pop': {
            'name': 'total_pop',
            'dtype': int,
            'default_value': 10e3,
            'ptype': 'optional',
            'valid_range': None,
            'category': ["simulation", "people", "networks"],
            'validator': None,
            'label': None,
            'description': 'Number of people in the real-world population.',
            'units': 'people',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'pop_scale': {
            'name': 'pop_scale',
            'dtype': float,
            'default_value': None,
            'ptype': 'derived',
            'valid_range': None,
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'Ratio total_pop/n_agents. Ratio person:agent, it defines how much to scale the population '
                           'of agents.',
            'units': 'people',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'location': {
            'name': 'location',
            'dtype': None,
            'default_value': None,
            'ptype': 'required',
            'valid_range': None,
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'What demographics to use - NOT CURRENTLY FUNCTIONAL',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'birth_rates': {
            'name': 'birth_rates',
            'dtype': None,
            'default_value': None,
            'ptype': 'derived',
            'valid_range': None,
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'Annual birth rates, loaded below',
            'units': 'number of births/year',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'death_rates': {
            'name': 'death_rates',
            'dtype': None,
            'default_value': None,
            'ptype': 'derived',
            'valid_range': None,
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'Annual death rates, loaded below',
            'units': 'number of deaths/year',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'rel_birth': {
            'name': 'rel_birth',
            'dtype': float,
            'default_value': 1.0,
            'ptype': 'required',
            'valid_range': (0, None),
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'Relative birth rates. A factor to scale (default) birth rate values.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'rel_death': {
            'name': 'rel_death',
            'dtype': float,
            'default_value': 1.0,
            'ptype': 'required',
            'valid_range': (0, None),
            'category': ["people"],
            'validator': None,
            'label': None,
            'description': 'Relative death rates. A factor to scale (default) death rate values.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'debut': {
            'name': 'debut',
            'dtype': dict,
            'default_value': dict(f=dict(dist='normal', par1=15.0, par2=2.0),
                                  m=dict(dist='normal', par1=17.5, par2=2.0)),
            'ptype': 'required',
            'valid_range': None,
            'category': ["people", "networks"],
            'validator': None,
            'label': None,
            'description': 'Distributions of demographic sexual debut ages for males and females.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'use_migration': {
            'name': 'use_migration',
            'dtype': bool,
            'default_value': True,
            'ptype': 'required',
            'valid_range': [True, False],
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Whether to estimate migration rates to correct the total population size.',
            'units': 'dimensionless',  # Not sure what this unit should be
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'start': {
            'name': 'start',
            'dtype': float,
            'default_value': 1995.0,
            'ptype': 'required',
            'valid_range': (1900.0, None),
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Start date of the simulation.',
            'units': 'years',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'end': {
            'name': 'end',
            'dtype': float,
            'default_value': None,
            'ptype': 'optional',
            'valid_range': (None, 2100.0),
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'End date of the simulation.',
            'units': 'years',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'n_years': {
            'name': 'n_years',
            'dtype': int,
            'default_value': 35,
            'ptype': 'required',
            'valid_range': (1, None),
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Number of years to simulate, if "end" is not specified. Note that this includes burn-in, '
                           'if end date is not specified.',
            'units': 'years',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'burnin': {
            'name': 'burnin',
            'dtype': int,
            'default_value': 25,
            'ptype': 'required',
            'valid_range': (0, None),
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Initial period of time (in years) during which the simulation is allowed to reach a '
                           'stable state. Any transient dynamics due to imperfect initial conditions is discared, '
                           'ensuring subsequent analysis focuses on meaningful and unbiased results. '
                           'If start and end dates are provided, then these burnin years are added to the total '
                           'simulation duration, and are removed from plots.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'dt': {
            'name': 'dt',
            'dtype': float,
            'default_value': 1.0,
            'ptype': 'required',
            'valid_range': (2 ** -3, 1.0),
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Simulation time step size in years.',
            'units': 'years',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'dt_demog': {
            'name': 'dt_demog',
            'dtype': float,
            'default_value': 1.0,
            'ptype': 'required',
            'valid_range': (2 ** -3, 1.0),
            'category': ["simulation", "people"],
            'validator': None,
            'label': None,
            'description': 'Time step size for demographic dynamics changes (in years)',
            'units': 'years',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'rand_seed': {
            'name': 'rand_seed',
            'dtype': int,
            'default_value': 1,
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Random seed to use, if None, don\'t reset random state.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'connectors': {
            'name': 'connectors',
            'dtype': list,
            'default_value': sc.autolist(),
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation", "connectors"],
            'validator': None,
            'label': None,
            'description': 'The connectors present in this simulation; populated by the user',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'interventions': {
            'name': 'interventions',
            'dtype': list,
            'default_value': sc.autolist(),
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation", "interventions"],
            'validator': None,
            'label': None,
            'description': 'The interventions present in this simulation; populated by the user.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'analyzers': {
            'name': 'analyzers',
            'dtype': list,
            'default_value': sc.autolist(),
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation", "analyzers"],
            'validator': None,
            'label': None,
            'description': 'The analyzers present in this simulation; populated by the user.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'timelimit': {
            'name': 'timelimit',
            'dtype': float,
            'default_value': None,
            'ptype': 'optional',
            'valid_range': None,
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'Time limit for the simulation (seconds).',
            'units': 'seconds',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'stopping_func': {
            'name': 'stopping_func',
            'dtype': callable,
            'default_value': None,
            'ptype': 'optional',
            'valid_range': None,
            'category': ["simulation"],
            'validator': None,
            'label': None,
            'description': 'A function to call to stop the simulation partway through.',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'networks': {
            'name': 'networks',
            'dtype': list,
            'default_value': sc.autolist(),
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation", "network"],
            'validator': None,
            'label': None,
            'description': 'Network types and parameters',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'verbose': {
            'name': 'verbose',
            'dtype': (float, str, int),
            'default_value': sso.verbose,
            'ptype': 'required',
            'valid_range': [-1, 0, 0.1, 1, 2],
            'category': ["other"],
            'validator': validator_verbose,
            'label': None,
            'description': 'Whether or not to display information during the run -- options are 0 (silent), '
                           '0.1 (some; default), 1 (default), 2 (everything)',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        }
    }
    return default_parameters


def default_parset_dict():
    pass


def get_default_parameter(default_pars, parameter_name):
    return default_pars.get(parameter_name, {})


def build_pars(input_pars):
    """
    Build the parameter structure needed for a simulation

    input_pars = use_default_pars()
    pars   = build_pars(inputs)
    """
    pars = sc.objdict()
    for parameter_name, parameter_data in input_pars.items():
        # TODO: parsing parameters will have more complexity and require a Parser class
        pars[parameter_name] = BaseParameter(**parameter_data)
    return pars


def make_default_pars(**kwargs):
    """
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    #NOTE: current pars is acting as a more general inputs structure to the simulation, rather than
    as model parameters -- though it may be a matter of semantics what we categorise as parameters.
    Parameters are inputs but not all inputs are parameters?

    Args:
        kwargs        (dict): any additional kwargs are interpreted as parameter names
    Returns:
        pars (dict): the parameters of the simulation
    """
    pars = sc.objdict()

    # Population parameters
    pars['n_agents'] = 10e3  # Number of agents
    pars['total_pop'] = 10e3  # If defined, used for calculating the scale factor
    pars['pop_scale'] = None  # How much to scale the population
    pars['location'] = None  # What demographics to use - NOT CURRENTLY FUNCTIONAL
    pars['birth_rates'] = None  # Birth rates, loaded below
    pars['death_rates'] = None  # Death rates, loaded below
    pars['rel_birth'] = 1.0  # Birth rate scale factor
    pars['rel_death'] = 1.0  # Death rate scale factor

    # Simulation parameters
    pars['start'] = 1995.  # Start of the simulation
    pars['end'] = None  # End of the simulation
    pars['n_years'] = 35  # Number of years to run, if end isn't specified. Note that this includes burn-in
    pars[
        'burnin'] = 25  # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
    pars['dt'] = 1.0  # Timestep (in years)
    pars['dt_demog'] = 1.0  # Timestep for demographic updates (in years)
    pars['rand_seed'] = 1  # Random seed, if None, don't reset
    pars[
        'verbose'] = sso.verbose  # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
    pars['use_migration'] = True  # Whether to estimate migration rates to correct the total population size

    # Events and interventions
    pars['connectors'] = sc.autolist()
    pars['interventions'] = sc.autolist()  # The interventions present in this simulation; populated by the user
    pars['analyzers'] = sc.autolist()  # The functions present in this simulation; populated by the user
    pars['timelimit'] = None  # Time limit for the simulation (seconds)
    pars['stopping_func'] = None  # A function to call to stop the sim partway through

    # Network parameters, generally initialized after the population has been constructed
    pars['networks'] = sc.autolist()  # Network types and parameters
    pars['debut'] = dict(f=dict(dist='normal', par1=15.0, par2=2.0),
                         m=dict(dist='normal', par1=17.5, par2=2.0))

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)

    return pars


# def get_births_deaths(location, verbose=1, by_sex=True, overall=False, die=True):
#     """
#     Get mortality and fertility data by location if provided, or use default
#
#     Args:
#         location (str):  location
#         verbose (bool):  whether to print progress
#         by_sex   (bool): whether to get sex-specific death rates (default true)
#         overall  (bool): whether to get overall values ie not disaggregated by sex (default false)
#
#     Returns:
#         lx (dict): dictionary keyed by sex, storing arrays of lx - the number of people who survive to age x
#         birth_rates (arr): array of crude birth rates by year
#     """
#
#     if verbose:
#         print(f'Loading location-specific demographic data for "{location}"')
#     try:
#         death_rates = ssdata.get_death_rates(location=location, by_sex=by_sex, overall=overall)
#         birth_rates = ssdata.get_birth_rates(location=location)
#         return birth_rates, death_rates
#     except ValueError as E:
#         warnmsg = f'Could not load demographic data for requested location "{location}" ({str(E)})'
#         ssm.warn(warnmsg, die=die)


def validator_verbose(value):
    if not sc.isnumber(value):  # pragma: no cover
        errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(value)} "{value}"'
        raise ValueError(errormsg)
