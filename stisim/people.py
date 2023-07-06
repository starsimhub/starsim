"""
Defines the People class and functions associated with making people
"""

# %% Imports
import sciris as sc
import numpy as np
from . import base as ssb
from . import misc as ssm
from . import utils as ssu
from . import settings as sss
from .data import loaders as ssdata


__all__ = ['People', 'make_popdict']


# %% Main people class


class People(ssb.BasePeople):
    """
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
        strict (bool): whether to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::
        ppl = ss.People(2000)
    """

    # %% Basic methods

    def __init__(self, n, states=None, strict=True, **kwargs):
        """
        Initialize
        """
        super().__init__(n, states=states)
        if strict: self.lock()  # If strict is true, stop further keys from being set (does not affect attributes)
        self.kwargs = kwargs
        return

    def initialize(self, popdict=None):
        """ Perform initializations """
        super().initialize(popdict=popdict)  # Initialize states
        return

    def add_module(self, module, force=False):
        # Initialize all the states associated with a module
        # This is implemented as People.add_module rather than
        # Module.add_to_people(people) or similar because its primary
        # role is to modify the People object
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())

        for state_name, state in module.states.items():
            combined_name = module.name + '.' + state_name
            self._data[combined_name] = state.new(self._n)
            self._map_arrays(keys=combined_name)
            self.states[combined_name] = state

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def update_demographics(self, dt, ti):
        """ Perform vital dynamic updates at the current timestep """

        self.age[~self.dead] += dt
        self.dead[self.ti_dead <= ti] = True


# %% Helper functions to create popdicts

def set_static(new_n, existing_n=0, f_ratio=0.5):
    """
    Set static population characteristics that do not change over time.
    This function can be used when adding new births, in which case the existing popsize can be given as `existing_n`.

    Arguments:
        new_n (int): Number of new individuals to add to the population.
        existing_n (int, optional): Number of existing individuals in the population. Default is 0.
        pars (dict, optional): Dictionary of parameters. Default is None.
        f_ratio (float, optional): Female ratio in the population. Default is 0.5.

    Returns:
        uid (ndarray, int): unique identifiers for the individuals.
        female (ndarray, bool): array indicating whether an individual is female or not.
        debut (ndarray, bool): array indicating the debut value for each individual.
    """
    uid = np.arange(existing_n, existing_n+new_n, dtype=sss.default_int)
    female = np.random.binomial(1, f_ratio, new_n)
    return uid, female


def make_popdict(n=None, location=None, year=None, verbose=None, f_ratio=0.5, dt_round_age=True, dt=None):
    """ Create a location-specific population dictionary """

    # Initialize total pop size
    total_pop = None

    # Load age data for this location if available
    if verbose:
        print(f'Loading location-specific data for "{location}"')
    try:
        age_data = ssdata.get_age_distribution(location, year=year)
        total_pop = sum(age_data[:, 2])  # Return the total population
    except ValueError as E:
        warnmsg = f'Could not load age data for requested location "{location}" ({str(E)})'
        ssm.warn(warnmsg, die=True)

    uid, female = set_static(n, f_ratio=f_ratio)

    # Set ages, rounding to nearest timestep if requested
    age_data_min = age_data[:, 0]
    age_data_max = age_data[:, 1]
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:, 2]
    age_data_prob /= age_data_prob.sum()  # Ensure it sums to 1
    age_bins = ssu.n_multinomial(age_data_prob, n)  # Choose age bins
    if dt_round_age:
        ages = age_data_min[age_bins] + np.random.randint(
            age_data_range[age_bins] / dt) * dt  # Uniformly distribute within this age bin
    else:
        ages = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(n)  # Uniform  within this age bin

    # Store output
    popdict = dict(
        uid=uid,
        age=ages,
        female=female
    )

    return total_pop, popdict
