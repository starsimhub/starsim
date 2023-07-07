"""
Default parameters to run a simulation, could also be loaded from a json file
"""

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
