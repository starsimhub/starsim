import sciris as sc
import numpy as np
from .results import Result
import functools

obj_set = object.__setattr__


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class State:
    # Simplified version of hpvsim.State()
    def __init__(self, name, dtype, fill_value=0):
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value

    def new(self, n):
        return np.full(n, dtype=self.dtype, fill_value=self.fill_value)


class People(sc.prettyobj):
    # TODO - cater to use cases of
    #   - Initial contact networks are independent of modules and we want to pre-generate agents and their contact layers
    #   - Initial contact networks depend on modules, and we need to add the modules before adding subsequent contact layers

    # Define states that every People instance has, regardless of which modules are enabled
    base_states = [
        State('uid', int),  # TODO: will we support removing agents? It could make indexing much more complicated...
        State('age', float),
        State('female', bool, False),
        State('pregnant', bool, False),
        State('dead', bool, False),
        State('ti_dead', float, np.nan),
        # Time index for death - defaults to natural causes but gets overwritten if they die of something first
    ]

    state_names = sc.autolist([state.name for state in base_states])

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    @property
    def male(self):
        return ~self.female

    @property
    def n(self):
        return len(self)

    @property
    def indices(self):
        return self.uid

    # noinspection PyUnresolvedReferences
    def __init__(self, n):
        # TODO - where is the right place to change how the initial population is generated?

        # Private variables relaying to dynamic allocation
        self._data = dict()
        self._n = n  # Number of agents (initial)
        self._s = self._n  # Underlying array sizes

        # Initialize base states
        for state in self.base_states:
            self._data[state.name] = state.new(n)
        self._map_arrays()
        self.uid[:] = np.arange(n)
        self.age[:] = np.random.random(n)
        self.female[:] = np.random.randint(0, 2, n)

        self.contacts = sc.odict()  # Dict containing Layer instances
        self._modules = []
        self.module_states = sc.objdict()

    def add_module(self, module):
        # Initialize all the states associated with a module
        # This is implemented as People.add_module rather than
        # Module.add_to_people(people) or similar because its primary
        # role is to modify the people attributes
        if hasattr(self, module.name):
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())
        for state in module.states:
            combined_name = module.name + '.' + state.name
            self.state_names += combined_name
            self.module_states[combined_name] = state
            self._data[combined_name] = state.new(self.n)
            self._map_arrays()

    # These methods provide a single function to call that handles
    # autonomous updates of the people, so it simplifies Sim.run()
    # However it's a little hacky that People and Module both define
    # these methods. Would it make sense for the base variables like
    # UID, age etc. to be in a Module? Or are they so heavily baked into
    # People that we don't want to separate them out?
    def update_states(self, sim):
        self.dead[self.ti_dead <= sim.ti] = True
        self.age += sim.dt  # ??

        for module in sim.modules:
            module.update_states(sim)

    def initialize(self, sim):
        sim.results['n_people'] = Result(None, 'n_people', sim.npts, dtype=int)
        sim.results['new_deaths'] = Result(None, 'new_deaths', sim.npts, dtype=int)

    def update_results(self, sim):
        sim.results['new_deaths'] = np.count_nonzero(sim.people.ti_dead == sim.ti)

        for module in sim.modules:
            module.update_results(sim)

    def finalize_results(self, sim):
        pass

    def _grow(self, n):
        """
        Increase the number of agents stored
        Automatically reallocate underlying arrays if required
        Args:
            n (int): Number of new agents to add
        """
        orig_n = self._n
        new_total = orig_n + n
        if new_total > self._s:
            n_new = max(n, int(self._s / 2))  # Minimum 50% growth
            for state in self.base_states:
                self._data[state.name] = np.concatenate([self._data[state.name], state.new(n_new)],
                                                        axis=self._data[state.name].ndim - 1)
            for state_name, state in self.module_states.items():
                self._data[state_name] = np.concatenate([self._data[state_name], state.new(n_new)],
                                                        axis=self._data[state_name].ndim - 1)
            self._s += n_new
        self._n += n
        self._map_arrays()
        new_inds = np.arange(orig_n, self._n)
        return new_inds

    def _map_arrays(self):
        """
        Set main simulation attributes to be views of the underlying data
        This method should be called whenever the number of agents required changes
        (regardless of whether the underlying arrays have been resized)
        """
        row_inds = slice(None, self._n)

        for k in self.state_names:
            arr = self._data[k]
            if arr.ndim == 1:
                rsetattr(self, k, arr[row_inds])
            elif arr.ndim == 2:
                rsetattr(self, k, arr[:, row_inds])
            else:
                errormsg = 'Can only operate on 1D or 2D arrays'
                raise TypeError(errormsg)

        return


def make_people(pars):
    # A generic function to handle making people via parameters during sim.initialize()
    # Users could also call this function to pre-generate people using the same routine beforehand
    # Or otherwise, they can write a custom equivalent of make_people and pass it to the sim directly
    raise NotImplementedError
