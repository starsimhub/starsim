"""
Set parameters
"""

import sciris as sc
from .base import ParsObj

__all__ = ['BaseParameter', 'ParameterSet']


class ParameterSet(ParsObj):
    """
    A derived class of ParsObj where __getitem__ returns pars[key].value
                                     __setitem__ sets    pars[key].update(value)
    """

    # def __init__(self, pars, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.pars = pars
    #     self.update_pars(pars, create=True)
    #     return

    def __getitem__(self, key):
        """ Return the value of pars[key] """
        try:
            return self.pars[key].value
        except KeyError:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)

    def __setitem__(self, key, value):
        """ Ditto """
        if key in self.pars:
            self.pars[key].update(value)
        else:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return


class BaseParameter(sc.prettyobj):
    def __init__(self, name, dtype, default_value=0, value=None, ptype="required", valid_range=None, category=None, validator=None,
                 label=None, description="#TODO Document me", units="dimensionless",
                 has_been_validated=False, nondefault=False, enabled=True):
        """
        Args:
            name: (str) name of the parameter
            dtype: (type) datatype
            ptype: (str) parameter type, three values "required", "optional", "derived"
            category: (str) what component or module this parameter belongs to (ie, sim, people, network) -- may not be necessary/used in the end, atm using it as a guide to organise parameters
            valid_range (list, tuple, dict?): the range of validity (numerical), or the valid set (categorical)
            validator: (callable) function that validates the parameter value
            label: (str) text used to construct labels for the result for displaying on plots and other outputs
            description: (str) human-readbale text describing what this parameter is about, maybe bibliographic references.
            default_value:  default value for this state upon initialization
            value: curent value of this instance
            has_been_validated: (bool) whether the parameter has passed validation
            enabled: (bool) whether the parameter is not available (ie, because a module/disease/ is not available)
            nondefault: (bool) whether user has modified from default parameter
        """
        self.name = name
        self.dtype = dtype
        self.ptype = ptype
        self.category = category
        self.valid_range = valid_range
        self.validator = validator
        self.has_been_validated = has_been_validated
        self.nondefault = nondefault
        self.enabled = enabled
        self.label = label or name
        self.description = description
        self.units = units
        self.value = value or default_value # If value is not specified at instantiation, use default
        self.default_value = default_value

    def validate(self):
        """
        Method to validate parameter values

        Returns
        -------
        bool or raise a value error if validation fails

        """
        # Perform parameter specific validation defined in self.validator
        if self.validator is not None:
            if not callable(self.validator):
                raise ValueError("Validator is not callable.")
            if not self.validator.__call__(self.value):
                raise ValueError(f"Parameter failed validation.")
            self.has_been_validated = True
        else:
            wrnmsg = f"No validator provided. Will try to perform basic validation."
            print(wrnmsg)
        # Perform basic validation
        if self.valid_range is None:
            # TODO: maybe we should say something if there's no valid_range
            pass
        elif isinstance(self.valid_range, tuple) and len(self.valid_range) == 2:
            vmin, vmax = self.valid_range
            if vmin is not None and self.value < vmin:
                errmsg = f"Value {self.value} is below the minimum valid value {vmin}."
                raise ValueError(errmsg)
            if vmax is not None and self.value > vmax:
                errmsg = f"Value {self.value} is above the maximum valid value {vmax}."
                raise ValueError(errmsg)
            self.has_been_validated = True
        elif isinstance(self.valid_range, list):  # Works for numerical and categorical sets
            if self.value not in self.valid_range:
                errmsg = f"Value {self.value} is not in the allowed list [{self.valid_range}]."
                raise ValueError(errmsg)
            self.has_been_validated = True
        else:
            raise ValueError("Bad valid_range specification.")
        return True

    def update(self, new_value):
        """
        Update parameter value with new_value
        """
        self.value = new_value
        self.validate()
        self.compare_to_default()

    def compare_to_default(self):
        """
        Check if current value is identical to default_value for this parameter.
        Useful to compare how a model deviates from default parameters.
        """
        if not self.value == self.default_value:
            self.nondefault = True


class ParameterMapper():
    """
    Class to map a dictionary
    pars = {'n_agents':10, 'rand_seed': 42}

    to a parameter set

    pars = {'n_agents': BaseParemeter(value=42), 'rand_seed': BaseParameter(value=42)}

    Merge default and user defined dictionaries, give priority to user defined values
    """
    pass

class ParameterInt(BaseParameter):
    pass


class ParameterFloat():

    def to(self, new_unit):
        # Implement unit conversion logic here
        pass

    pass


class ParameterRange():
    # valid_range
    pass


class ParameterCategorical():
    # allowed_values
    pass
