import pandas as pd
import numpy as np
import sciris as sc
import numba as nb
from stisim.utils.ndict import INT_NAN
from stisim.core.distributions import ScipyDistribution
from stisim.utils.ndict import *
from stisim.utils.actions import *
from stisim.states.fussedarray import FusedArray
from numpy.lib.mixins import NDArrayOperatorsMixin  # Inherit from this to automatically gain operators like +, -, ==, <, etc.
from scipy.stats._distn_infrastructure import rv_frozen
from .dinamicview import DynamicView

__all__ = ['State']

class State(FusedArray):

    def __init__(self, name, dtype, fill_value=None, label=None):
        """

        :param name: A string name for the state
        :param dtype: The dtype to use for this instance
        :param fill_value: Specify default value for new agents. This can be
            - A scalar with the same dtype (or castable to the same dtype) as the State
            - A callable, with a single argument for the number of values to produce
            - An ss.ScipyDistribution instance
        :param label:
        """

        super().__init__(values=None, uid=None, uid_map=None)  # Call the FusedArray constructor

        self.fill_value = fill_value

        self._data = DynamicView(dtype=dtype)
        self.name = name
        self.label = label or name
        self.values = self._data._view
        self._initialized = False

    def __repr__(self):
        if not self._initialized:
            return f'<State {self.name} (uninitialized)>'
        else:
            return FusedArray.__repr__(self)

    def _new_vals(self, uids):
        if isinstance(self.fill_value, ScipyDistribution):
            new_vals = self.fill_value.rvs(uids)
        elif callable(self.fill_value):
            new_vals = self.fill_value(len(uids))
        else:
            new_vals = self.fill_value
        return new_vals

    def initialize(self, sim=None, people=None):
        if self._initialized:
            return

        if sim is not None and people is None:
            people = sim.people

        sim_still_needed = False
        if isinstance(self.fill_value, rv_frozen):
            if sim is not None:
                self.fill_value = ScipyDistribution(self.fill_value, f'{self.__class__.__name__}_{self.label}')
                self.fill_value.initialize(sim, self)
            else:
                sim_still_needed = True

        people.add_state(self, die=False) # CK: should not be needed
        if not sim_still_needed:
            self._uid_map = people._uid_map
            self.uid = people.uid
            self._data.grow(len(self.uid))
            self._data[:len(self.uid)] = self._new_vals(self.uid)
            self.values = self._data._view
            self._initialized = True
        return

    def grow(self, uids):
        """
        Add state for new agents

        This method is normally only called via `People.grow()`.

        :param uids: Numpy array of UIDs for the new agents being added This array should have length n
        """

        n = len(uids)
        self._data.grow(n)
        self.values = self._data._view
        self._data[-n:] = self._new_vals(uids)
        return

    def _trim(self, inds):
        # Trim arrays to remove agents - should only be called via `People.remove()`
        self._data._trim(inds)
        self.values = self._data._view
        return
