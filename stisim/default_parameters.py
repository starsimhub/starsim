"""
Default parameters to run a simulation, could also be loaded from a json file
"""

import sciris as sc
from .parameters import BaseParameter

__all__ = ['use_default_parameters', 'get_default_parameter']


def use_default_parameters():
    default_parameters = {
        'n_agents': {
            'name': 'n_agents',
            'dtype': float,
            'default_value': 10e3,
            'ptype': 'required',
            'valid_range': None,
            'category': None,
            'validator': None,
            'label': None,
            'description': '#TODO Document me',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
        'total_pop': {
            'name': 'total_pop',
            'dtype': float,
            'default_value': 10e3,
            'ptype': 'required',
            'valid_range': None,
            'category': None,
            'validator': None,
            'label': None,
            'description': '#TODO Document me',
            'units': 'dimensionless',
            'has_been_validated': False,
            'nondefault': False,
            'enabled': True
        },
    }

    return default_parameters


def get_default_parameter(default_pars_dict, parameter_name):
    return default_pars_dict.get(parameter_name, {})


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
