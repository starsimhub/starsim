"""
Default parameters to run a simulation, could also be loaded from a json file
"""

import sciris as sc
from .settings import options as sso  # For setting global options
from .parameters import BaseParameter

__all__ = ['make_default_pars', 'default_pars_dict', 'get_default_parameter', 'build_pars']


def make_default_pars(**kwargs):
    pars = build_pars(default_pars_dict())
    return pars


def default_pars_dict():
    default_parameters = {
        'n_agents': {
            'name': 'n_agents',
            'dtype': int,
            'default_value': 10e3,
            'ptype': 'required',
            'valid_range': None,
            'category': ["simulation", "people", "network"],
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
            'category': ["simulation", "people", "network"],
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
            'default_value':dict(f=dict(dist='normal', par1=15.0, par2=2.0),
                                 m=dict(dist='normal', par1=17.5, par2=2.0)),
            'ptype': 'required',
            'valid_range': None,
            'category': ["people", "network"],
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
            'units': 'dimensionless', # Not sure what this unit should be
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
                           'This is does not affect the start and end dates of the simulation, but it is possible '
                           'remove these years from plots',
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
            'valid_range': (2**-3, 1.0),
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
            'valid_range': (2**-3, 1.0),
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
            'category': None,
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
            'category': None,
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
            'category': None,
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
            'category': None,
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
            'category': None,
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
            'category': None,
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
            'dtype': float,
            'default_value': sso.verbose,
            'ptype': 'required',
            'valid_range': [0, 0.1, 1, 2],
            'category': ["other"],
            'validator': None,
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