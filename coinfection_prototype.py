# TOY MODEL
# This toy model has HIV and gonorrhea with coinfection. The diseases interact
# such that gonorrhea susceptibility depends on the HIV CD4 value, and the HIV
# module includes ART as a state which decreases the impact of HIV coinfection by
# raising the CD4 value.

# Some challenges
# - How would you make it so that people infected with gonorrhea reduce sexual activity
#   until their infection is cleared?

import numpy as np
import sciris as sc


class State():
    # Simplified version of hpvsim.State()
    def __init__(self, name, dtype, fill_value=0):
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value

    def new(self, n):
        return np.full(n, dtype=self.dtype, fill_value=self.fill_value)


class People():
    # TODO - cater to use cases of
    #   - Initial contact networks are independent of modules and we want to pre-generate agents and their contact layers
    #   - Initial contact networks depend on modules, and we need to add the modules before adding subsequent contact layers

    # Define states that every People instance has, regardless of which modules are enabled
    base_states = [
        State('uid', int),
        State('age', float),
        State('female', bool, False),
        State('dead', bool, False),
        State('ti_dead', float, np.nan), # Time index for death by natural causes
    ]

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr')
            If the key is an integer, alias `people.person()` to return a `Person` instance
        '''
        if isinstance(key, int):
            return self.person(key)
        else:
            return self.__getattribute__(key)

    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__: # pragma: no cover
            errormsg = f'Key "{key}" is not a current attribute of people, and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        return self.__setattr__(key, value)

    def __setattr__(self, attr, value):
        ''' Ditto '''
        if hasattr(self, '_data') and attr in self._data:
            # Prevent accidentally overwriting a view with an actual array - if this happens, the updated values will
            # be lost the next time the arrays are resized
            raise Exception('Cannot assign directly to a dynamic array view - must index into the view instead e.g. `people.uid[:]=`')
        else:   # If not initialized, rely on the default behavior
            obj_set(self, attr, value)
        return

    @property
    def male(self):
        return ~self.female

    @property
    def n(self):
        return len(self)

    def __init__(self, n):
        # TODO - where is the right place to change how the initial population is generated?
        for state in self.states:
            self.__setattr__(state.name, state.new(n))
        self.uid[:] = np.arange(n)
        self.age[:] = np.random.random(n)
        self.female[:] = np.random.randint(0,2,n)
        self.contacts = sc.odict()  # Dict containing Layer instances

    def add_module(self, module):
        # Initialize all of the states associated with a module
        self.__setattr__(module.name, sc.objdict)
        for state in module.states:
            self.__setattr__(state.name, state.new(self.n))

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

    def init_results(self, sim):
        for module in sim.modules:
            module.init_results(sim)

    def update_results(self, sim):
        for module in sim.modules:
            module.update_results(sim)

    def finalize_results(self, sim):
        for module in sim.modules:
            module.finalize_results(sim)

class Layer(sc.FlexDict):
    def __init__(self, people: People):
        self.p1 = np.array([], dtype=int)
        self.p2 = np.array([], dtype=int)
        self.beta = np.array([], dtype=float)

    @property
    def members(self):
        return set(self.p1).union(set(self.p2))

    def update(self):
        pass


# TODO - modify network to have long-term and casual partnerships with different transmission risk per contact
class RandomDynamicSexualLayer(Layer):
    # Randomly pair males and females with variable relationship durations
    def __init__(self, people, mean_dur=5):
        super().__init__(self, people)
        self.mean_dur = mean_dur
        self.dur = np.array([], dtype=float)
        self.update()

    def update(self, people):
        # Decrement time step
        self.dur = self.dur - people.dt

        # Remove any partnerships with a duration of 0
        active = self.dur > 0
        self.p1 = self.p1[active]
        self.p2 = self.p2[active]
        self.beta = self.beta[active]

        # Find unpartnered males and females - could in principle check other contact layers too
        # by having the People object passed in here
        available_m = np.setdiff(people.indices[people.male], self.members)
        available_f = np.setdiff(people.indices[~people.male], self.members)

        if len(available_m) <= len(available_f):
            p1 = available_m
            p2 = np.random.sample(available_f, len(p1), replace=False)
        else:
            p2 = available_f
            p1 = np.random.sample(available_m, len(p2), replace=False)

        beta = np.ones_like(p1)
        dur = np.random.randn(len(p1))*self.mean_dur
        self.p1 = np.concatenate([self.p1, p1])
        self.p2 = np.concatenate([self.p2, p2])
        self.beta = np.concatenate([self.beta, beta])
        self.dur = np.concatenate([self.dur, dur])


def make_people(n):
    people = People(n)
    people.contacts['random'] = RandomDynamicSexualLayer(people)


class Module():
    # Base module contains states/attributes that all modules have
    default_pars = {}

    def __init__(self, pars):
        self.pars = sc.odict()
        for k,v in self.default_pars:
            if k in pars:
                self.pars[k] = pars[k]
            else:
                self.pars[k] = v

    def init_people(self, people):
        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        people.add_module(self)

    def update_states_pre(self, people):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    def init_results(self):
        pass

    def update_results(self):
        pass

    def finalize_results(self):
        pass

    @property
    def name(self):
        # The module name is a lower-case version of its class name
        return self.__class__.name.lower()


class HIV(Module):
    states = [
        State('susceptible', bool, True),
        State('infected', bool, False),
        State('on_art', bool, False),
        State("cd4", float, np.nan),
    ]

    default_pars = {
        'cd4_min' = 100,
        'cd4_max' = 500,
        'cd4_rate' = 2,
    }


    def __init__(self, pars):
        super().__init__()


    def update_states_pre(self):
        # TODO - What if something in here should depend on another module?
        pass

    def init_results( sim):
        pass



class Result(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, module, name, n, dtype):
        self.module = module.name if module else None
        self.name = name
        self.values = np.zeros(n, dtype=dtype)

    def __getitem__(self, *args, **kwargs):  return self.values.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs): return self.values.__setitem__(*args, **kwargs)
    def __len__(self): return len(self.values)

    # These methods allow automatic use of functions like np.sum, np.exp, etc.
    # with higher performance in some cases
    @property
    def __array_interface__(self): return self.values.__array_interface__
    def __array__(self): return self.values
    def __array_ufunc__(self, *args, **kwargs):
        args = [(x if x is not self else self.values) for x in args]
        kwargs = {k:v if v is not self else self.values for k,v in kwargs.items()}
        return self.values.__array_ufunc__(*args, **kwargs)



class Gonorrhea(Module):
    states = [
        State('susceptible', bool, True),
        State('infected', bool, False),
        State('ti_infected', float, 0),
        State('ti_recovered', float, 0),
        State('ti_dead', float, 0), # Death due to gonorrhea
    ]

    default_pars = {
        'dur_inf': 0.1, # not modelling diagnosis or treatment explicitly here
    }

    def __init__(self, pars):
        super().__init__(pars)


    def update_states_pre(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        sim.people.dead[sim.people.gonorrhea.ti_dead <= sim.ti] = True



    def init_results(self, sim):
        # TODO - not a huge fan of having the result key duplicate the Result name
        sim.results[self.name,'n_susceptible'] = Result(self.name, 'n_susceptible', sim.n, dtype=int)
        sim.results[self.name,'n_infected'] = Result(self.name, 'n_infected', sim.n, dtype=int)
        sim.results[self.name,'prevalence'] = Result(self.name, 'prevalence', sim.n, dtype=float)
        sim.results[self.name,'new_infections'] = Result(self.name, 'n_infected', sim.n, dtype=int)

    def update_results(self, sim):
        # TODO sim.results.gonorrhea.susceptible or sim.results['gonorrhea.susceptible']?
        sim.results[self.name,'n_susceptible'][sim.ti] = np.count_nonzero(sim.people.gonorrhea.susceptible)
        sim.results[self.name,'n_infected'][sim.ti] = np.count_nonzero(sim.people.gonorrhea.infected)
        sim.results[self.name,'prevalence'][sim.ti] = sim.results.gonorrhea.infected[sim.ti] / sim.people.n
        sim.results[self.name,'new_infections'] = sim.people.gonorrhea.ti_infected == sim.ti

class Analyzer():
    # TODO - what if the analyzer depends on a specific variable? How does the analyzer know which modules are required?
    def initialize(self, sim):
        pass
    def update(self, sim):
        pass
    def finalize(self, sim):
        pass



class DefaultOutputs(Analyzer):
    def initialize(self):
        class Intervention():

        class ART(Intervention):




# Sanitize modules
# A module is pre-loaded with parameters EXCLUDING person-generation pars

class Sim():
    def __init__(self, people, modules = None, pars=None, interventions=None, analyzers=None):
        # TODO - clearer options for time units?
        #   ti - index self.ti = 0
        #    t - floating point year e.g., 2020.5
        #   dt - simulation primary step size (years per step)  # TODO - use-case for model-specific dt?
        # date - '20200601'

        self.ti = 0
        self.dt = 1 # could get this from pars of course

        self.people = people
        self.modules = sc.promotetolist(modules)
        self.pars = pars if pars is not None else sc.objdict()  # What pars are these? Are they by module? Are the parameters consumed when the module is created?
        self.interventions = sc.promotetolist(interventions)
        self.analyzers = sc.promotetolist(analyzers)
        self.results = {}


    def initialize(self):
        for module in self.modules:
            module.init(self.people, self.pars) # Do this?

    # @property
    # def all_results(self):
    #     for module, module_results in self.results.items():
    #         for

    def run(self):

        self.results = {}

        for i in range(self.pars.npts):
            self.step()




        res = self.people.update(t)
        self.results.append(res)

    def step(self):
        self.t += self.pars.dt

        self.people.update_states_pre(sim)

        for module in self.modules:
            module.infect(self.people)

        self.people.update_results(sim)




