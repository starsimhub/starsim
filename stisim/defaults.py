'''
Set the defaults
'''

import numpy as np
import sciris as sc
import pylab as pl
from .settings import options as sso # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. hpv.defaults.default_int
__all__ = ['datadir', 'default_float', 'default_int']

# Define paths
datadir = sc.path(sc.thisdir(__file__)) / 'data'

#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if sso.precision == 32:
    default_float = np.float32
    default_int   = np.int32
elif sso.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {sso.precision}')


#%% Define all properties of people

class State(sc.prettyobj):
    def __init__(self, name, dtype, fill_value=0, shape=None, label=None, color=None):
        '''
        Args:
            name: name of the result as used in the model
            dtype: datatype
            fill_value: default value for this state upon model initialization
            shape: If not none, set to match a string in `pars` containing the dimensionality
            label: text used to construct labels for the result for displaying on plots and other outputs
            color: color (used for plotting stocks)
        '''
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value
        self.shape = shape
        self.label = label or name
        self.color = color
        return
    
    @property
    def ndim(self):
        return len(sc.tolist(self.shape))+1
    
    def new(self, pars, n, module=None):
        shape = sc.tolist(self.shape)
        if module is not None:
            if len(shape) and shape[0] in pars[module].keys():
                pars = pars[module]
        shape = [pars[s] for s in shape]
        shape.append(n) # We always want to have shape n
        return np.full(shape, dtype=self.dtype, fill_value=self.fill_value)


class PeopleMeta(sc.prettyobj):
    ''' For storing all the keys relating to a person and people '''

    # (attribute, nrows, dtype, default value)
    # If the default value is None, then the array will not be initialized - this is faster and can
    # be used for variables where the People object explicitly supplies the values e.g. age

    def __init__(self):

        # Set the properties of a person
        self.person = [
            State('uid',    default_int),            # Int
            State('age',    default_float,  np.nan), # Float
            State('sex',    default_float,  np.nan), # Float
            State('debut',  default_float,  np.nan), # Float
            State('scale',  default_float,  1.0),    # Float
        ]

        ###### The following section consists of all the boolean states

        # The following three groupings are all mutually exclusive and collectively exhaustive.
        self.alive_states = [
            # States related to whether or not the person is alive or dead
            State('alive',          bool,   True,   label='Population'),    # Save this as a state so we can record population sizes
            State('dead_other',     bool,   False,  label='Cumulative deaths from other causes'),   # Dead from all other causes
            State('emigrated',      bool,   False,  label='Emigrated'),  # Emigrated
        ]

    # Collection of states for which we store associated dates
    @property
    def date_states(self):
        return [state for state in self.alive_states if not state.fill_value]

    # Set dates
    @property
    def dates(self):
        dates = [State(f'date_{state.name}', default_float, np.nan, shape=state.shape) for state in self.date_states]
        dates += [
            State('date_dead', default_float, np.nan)
        ]
        return dates

    # States to set when creating people # TODO -- can probably remove?
    @property
    def states_to_set(self):
        return self.person + self.alive_states + self.dates

    def validate(self):
        '''
        TODO check that states are valid
        '''
        return


#%% Default result settings

# Flows
class Flow():
    def __init__(self, name, label=None, color=None):
        self.name = name
        self.label = label or name
        self.color = color
