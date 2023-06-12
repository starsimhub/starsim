import sciris as sc
import numpy as np
from .results import Result


class State():
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
        State('uid', int), # TODO: will we support removing agents? It could make indexing much more complicated...
        State('age', float),
        State('female', bool, False),
        State('dead', bool, False),
        State('ti_dead', float, np.nan), # Time index for death - defaults to natural causes but gets overwritten if they die of something first
    ]

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

    def __init__(self, n):
        # TODO - where is the right place to change how the initial population is generated?
        for state in self.base_states:
            self.__setattr__(state.name, state.new(n))
        self.uid[:] = np.arange(n)
        self.age[:] = np.random.random(n)
        self.female[:] = np.random.randint(0,2,n)
        self.contacts = sc.odict()  # Dict containing Layer instances
        self._modules = []

    def add_module(self, module):
        # Initialize all of the states associated with a module
        # This is implemented as People.add_module rather than
        # Module.add_to_people(people) or similar because its primary
        # role is to modify the
        if hasattr(self,module.name):
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())
        for state in module.states:
            self[module.name][state.name] = state.new(self.n)

    # These methods provide a single function to call that handles
    # autonomous updates of the people, so it simplifies Sim.run()
    # However it's a little hacky that People and Module both define
    # these methods. Would it make sense for the base variables like
    # UID, age etc. to be in a Module? Or are they so heavily baked into
    # People that we don't want to separate them out?
    def update_states_pre(self, sim):
        self.dead[self.ti_dead <= sim.ti] = True

        for module in sim.modules:
            module.update_states_pre(sim)

    def initialize(self, sim):
        sim.results['n_people'] = Result(None, 'n_people', sim.npts, dtype=int)
        sim.results['new_deaths'] = Result(None, 'new_deaths', sim.npts, dtype=int)

    def update_results(self, sim):
        sim.results['new_deaths'] = np.count_nonzero(sim.people.ti_dead == sim.ti)

        for module in sim.modules:
            module.update_results(sim)

    def finalize_results(self, sim):
        pass


def make_people(pars):
    # A generic function to handle making people via parameters during sim.initialize()
    # Users could also call this function to pre-generate people using the same routine beforehand
    # Or otherwise, they can write a custom equivalent of make_people and pass it to the sim directly
    raise NotImplementedError