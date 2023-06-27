"""
Defines the People class and functions associated with making people
"""

# %% Imports
import numpy as np
import sciris as sc
from . import utils as ssu
from . import settings as sss
from . import base as ssb
from . import population as sspop

__all__ = ['People']


# %% Define all properties of people


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
        self.initialized = False
        self.kwargs = kwargs
        return

    def initialize(self):
        """ Perform initializations """
        super().initialize()  # Initialize states
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

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def increment_age(self):
        """ Let people age by one timestep """
        self.age[self.alive] += self.dt
        return

    def update_states(self, t, sim):
        """ Perform all state updates at the current timestep """

        for module in sim.modules.values():
            module.update_states(sim)

        # Perform network updates
        for lkey, layer in self.contacts.items():
            layer.update(self)

        return
