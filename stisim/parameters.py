"""
Set parameters
"""

import numpy as np
import sciris as sc
from .settings import options as sso  # For setting global options

# from . import misc as ssm
# from .data import loaders as ssdata

__all__ = ['BaseParameter']


class BaseParameter(sc.prettyobj):
    def __init__(self, name, dtype, default_value=0, ptype="required", valid_range=None, category=None, validator=None,
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
        self.value = default_value
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
        else:
            wrnmsg = f"No validator provided. Will try to perform basic validation."
            print(wrnmsg)
        # Perform basic
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
        elif isinstance(self.valid_range, list):  # Works for numerical and categorical sets
            if self.value not in self.valid_range:
                errmsg = f"Value {self.value} is not in the allowed list [{self.valid_range}]."
                raise ValueError(errmsg)
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
        Check if current value is default_value for this parameter.
        Useful to compare how a model deviates from default parameters.
        """
        if not self.value == self.default_value:
            self.nondefault = True


class ParameterInt(BaseParameter):
    def __init__(self, name, dtype, default_value=0, ptype="required", valid_range=None, category=None, validator=None,
                 label=None, description="#TODO Document me", units="dimensionless", has_been_validated=False,
                 nondefault=False, enabled=True):
        super().__init__(name, dtype, default_value, ptype, valid_range, category, validator, label, description,
                         units, has_been_validated, nondefault, enabled)

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


# def make_default_pars(**kwargs):
#     """
#     Create the parameters for the simulation. Typically, this function is used
#     internally rather than called by the user; e.g. typical use would be to do
#     sim = ss.Sim() and then inspect sim.pars, rather than calling this function
#     directly.
#
#     #NOTE: current pars is acting as a more general inputs structure to the simulation, rather than
#     as model parameters -- though it may be a matter of semantics what we categorise as parameters.
#     Parameters are inputs but not all inputs are parameters?
#
#     Args:
#         kwargs        (dict): any additional kwargs are interpreted as parameter names
#     Returns:
#         pars (dict): the parameters of the simulation
#     """
#     pars = sc.objdict()
#
#     # Population parameters
#     pars['n_agents'] = 10e3  # Number of agents
#     pars['total_pop'] = 10e3  # If defined, used for calculating the scale factor
#     pars['pop_scale'] = None  # How much to scale the population
#     pars['location'] = None  # What demographics to use - NOT CURRENTLY FUNCTIONAL
#     pars['birth_rates'] = None  # Birth rates, loaded below
#     pars['death_rates'] = None  # Death rates, loaded below
#     pars['rel_birth'] = 1.0  # Birth rate scale factor
#     pars['rel_death'] = 1.0  # Death rate scale factor
#
#     # Simulation parameters
#     pars['start'] = 1995.  # Start of the simulation
#     pars['end'] = None  # End of the simulation
#     pars['n_years'] = 35  # Number of years to run, if end isn't specified. Note that this includes burn-in
#     pars[
#         'burnin'] = 25  # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
#     pars['dt'] = 1.0  # Timestep (in years)
#     pars['dt_demog'] = 1.0  # Timestep for demographic updates (in years)
#     pars['rand_seed'] = 1  # Random seed, if None, don't reset
#     pars[
#         'verbose'] = sso.verbose  # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
#     pars['use_migration'] = True  # Whether to estimate migration rates to correct the total population size
#
#     # Events and interventions
#     pars['connectors'] = sc.autolist()
#     pars['interventions'] = sc.autolist()  # The interventions present in this simulation; populated by the user
#     pars['analyzers'] = sc.autolist()  # The functions present in this simulation; populated by the user
#     pars['timelimit'] = None  # Time limit for the simulation (seconds)
#     pars['stopping_func'] = None  # A function to call to stop the sim partway through
#
#     # Network parameters, generally initialized after the population has been constructed
#     pars['networks'] = sc.autolist()  # Network types and parameters
#     pars['debut'] = dict(f=dict(dist='normal', par1=15.0, par2=2.0),
#                          m=dict(dist='normal', par1=17.5, par2=2.0))
#
#     # Update with any supplied parameter values and generate things that need to be generated
#     pars.update(kwargs)
#
#     return pars

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
