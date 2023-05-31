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
from collections import defaultdict
import matplotlib.pyplot as plt

#### PEOPLE/POPULATIONS

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



class Layer(sc.odict):
    def __init__(self):
        super().__init__()
        self.p1 = np.array([], dtype=int)
        self.p2 = np.array([], dtype=int)
        self.beta = np.array([], dtype=float)

    def __repr__(self):
        return f'<{self.__class__.__name__}, {len(self.members)} members, {len(self.p1)} contacts>'

    @property
    def members(self):
        return set(self.p1).union(set(self.p2))

    def update(self):
        pass

# TODO - modify network to have long-term and casual partnerships with different transmission risk per contact
class RandomDynamicSexualLayer(Layer):
    # Randomly pair males and females with variable relationship durations
    def __init__(self, people, mean_dur=5):
        super().__init__()
        self.mean_dur = mean_dur
        self.dur = np.array([], dtype=float)
        self.add_partnerships(people)

    def add_partnerships(self, people):
        # Find unpartnered males and females - could in principle check other contact layers too
        # by having the People object passed in here

        available_m = np.setdiff1d(people.indices[people.male], self.members)
        available_f = np.setdiff1d(people.indices[~people.male], self.members)

        if len(available_m) <= len(available_f):
            p1 = available_m
            p2 = np.random.choice(available_f, len(p1), replace=False)
        else:
            p2 = available_f
            p1 = np.random.choice(available_m, len(p2), replace=False)

        beta = np.ones_like(p1)
        dur = np.random.randn(len(p1))*self.mean_dur
        self.p1 = np.concatenate([self.p1, p1])
        self.p2 = np.concatenate([self.p2, p2])
        self.beta = np.concatenate([self.beta, beta])
        self.dur = np.concatenate([self.dur, dur])

    def update(self, people):
        # First remove any relationships due to end
        self.dur = self.dur - people.dt
        active = self.dur > 0
        self.p1 = self.p1[active]
        self.p2 = self.p2[active]
        self.beta = self.beta[active]

        # Then add new relationships for unpartnered people
        self.add_partnerships(people)

class StaticLayer(Layer):
    # Randomly make some partnerships that don't change over time
    def __init__(self, people, sex='mf'):
        super().__init__()

        if sex[0]=='m':
            p1 = people.indices[people.male]
        else:
            p1 = people.indices[~people.male]

        if sex[1]=='f':
            p2 = people.indices[~people.male]
        else:
            p2 = people.indices[people.male]

        self.p1 = p1
        self.p2 = np.random.choice(p2, len(p1), replace=True)
        self.beta = np.ones_like(p1)


def make_people(n):
    people = People(n)
    people.contacts['random'] = RandomDynamicSexualLayer(people)
    people.contacts['msm'] = StaticLayer(people,'mm')

    # relationship types, homosexual/heterosexual, births?
    # do we model pregnancies? track parents? households?
    return people

##### DISEASE MODULES

# timestep concerns - how to


class Module():
    # Base module contains states/attributes that all modules have
    default_pars = {}

    def __init__(self):
        raise Exception('Modules should not be instantiated but their classes should be used directly')

    @classmethod
    def initialize(cls, sim):
        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        # and initializing any outputs that are required
        sim.people.add_module(cls)

        # Merge parameters
        if cls.name not in sim.pars:
            sim.pars[cls.name] = sc.objdict(sc.dcp(cls.default_pars))
        else:
            if ~isinstance(sim.pars[cls.name], sc.objdict):
                sim.pars[cls.name] = sc.objdict(sim.pars[cls.name])
            for k,v in cls.default_pars.items():
                if k not in sim.pars[cls.name]:
                    sim.pars[cls.name][k] = v

        sim.results[cls.name] = sc.objdict()

    @classmethod
    def update_states_pre(cld, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    @classmethod
    def update_results(cls, sim):
        pass

    @classmethod
    def finalize_results(cls, sim):
        pass

    @classmethod
    @property
    def name(cls):
        # The module name is a lower-case version of its class name
        return cls.__name__.lower()


class HIV(Module):
    states = [
        State('susceptible', bool, True),
        State('infected', bool, False),
        State('on_art', bool, False),
        State("cd4", float, np.nan),
    ]

    default_pars = {
        'cd4_min': 100,
        'cd4_max': 500,
        'cd4_rate': 2,
    }


    @classmethod
    def update_states_pre(cls, sim):
        # TODO - What if something in here should depend on another module?
        pass

    @classmethod
    def init_results(cls, sim):
        pass


class Result(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, module, name, n, dtype):
        self.module = module if module else None
        self.name = name
        self.values = np.zeros(n, dtype=dtype)

    def __getitem__(self, *args, **kwargs):  return self.values.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs): return self.values.__setitem__(*args, **kwargs)
    def __len__(self): return len(self.values)
    def __repr__(self): return f'Result({self.module},{self.name}): {self.values.__repr__()}'

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
        State('ti_dead', float, np.nan), # Death due to gonorrhea
    ]

    default_pars = {
        'dur_inf': 3, # not modelling diagnosis or treatment explicitly here
        'p_death': 0.2,
        'initial': 3,
    }

    def __init__(self, pars):
        super().__init__(pars)

    @classmethod
    def update_states_pre(cls, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.ti
        sim.people.dead[gonorrhea_deaths] = True
        sim.people.ti_dead[gonorrhea_deaths] = sim.ti

    @classmethod
    def initialize(cls, sim):
        # Do any steps in this method depend on what other modules are going to be added? We can inspect
        # them via sim.modules at this point
        super(Gonorrhea, cls).initialize(sim)

        # Pick some initially infected agents
        cls.infect(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

        if 'beta' not in sim.pars[cls.name]:
            sim.pars[cls.name].beta = sc.objdict({k:1 for k in sim.people.contacts})

        # TODO - not a huge fan of having the result key duplicate the Result name
        sim.results[cls.name]['n_susceptible'] = Result(cls.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[cls.name]['n_infected'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['prevalence'] = Result(cls.name, 'prevalence', sim.npts, dtype=float)
        sim.results[cls.name]['new_infections'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)


    @classmethod
    def update_results(cls, sim):
        sim.results[cls.name]['n_susceptible'][sim.ti] = np.count_nonzero(sim.people.gonorrhea.susceptible)
        sim.results[cls.name]['n_infected'][sim.ti] = np.count_nonzero(sim.people.gonorrhea.infected)
        sim.results[cls.name]['prevalence'][sim.ti] = sim.results[cls.name].n_infected[sim.ti] / sim.people.n
        sim.results[cls.name]['new_infections'] = np.count_nonzero(sim.people[cls.name].ti_infected == sim.ti)

    @classmethod
    def transmit(cls, sim):
        for k, layer in sim.people.contacts.items():
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people.gonorrhea.infected & ~sim.people.dead).astype(float)
                rel_sus = (sim.people.gonorrhea.susceptible & ~sim.people.dead).astype(float)
                for a,b in [[layer.p1,layer.p2],[layer.p2,layer.p1]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a]*rel_sus[b]*layer.beta*sim.pars[cls.name]['beta'][k]
                    cls.infect(sim, b[np.random.random(len(a))<p_transmit])

    @classmethod
    def infect(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.ti

        dur = sim.ti+np.random.poisson(sim.pars[cls.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids))<sim.pars[cls.name].p_death
        sim.people[cls.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]

        # if HIV in sim.modules:
        #     # check CD4 count
        #     # increase susceptibility where relevant


# class MyComplexHIVModel(HIV):
#     ...

class Analyzer():
    # TODO - what if the analyzer depends on a specific variable? How does the analyzer know which modules are required?
    def initialize(self, sim):
        pass
    def update(self, sim):
        pass
    def finalize(self, sim):
        pass



# class DefaultOutputs(Analyzer):
#     def initialize(self):


# class Intervention():


# class ART(Intervention):


class Sim():

    default_pars = sc.objdict(
        n=1000,
        npts=30,
        dt=1
    )

    def __init__(self, people=None, modules=None, pars=None, interventions=None, analyzers=None):
        # TODO - clearer options for time units?
        #   ti - index self.ti = 0
        #    t - floating point year e.g., 2020.5
        #   dt - simulation primary step size (years per step)  # TODO - use-case for model-specific dt?
        # date - '20200601'

        self.ti = 0

        self.pars = sc.dcp(self.default_pars)
        if pars is not None:
            self.pars.update(pars)
        self.people = people if people is not None else make_people(self.pars['n'])
        self.modules = sc.promotetolist(modules)
        self.interventions = sc.promotetolist(interventions)
        self.analyzers = sc.promotetolist(analyzers)
        self.results = sc.objdict()

    @property
    def dt(self):
        return self.pars['dt']

    @property
    def t(self):
        return self.ti*self.dt

    @property
    def n(self):
        return len(self.people)

    @property
    def tvec(self):
        return np.arange(0,self.npts)*self.dt

    @property
    def npts(self):
        return self.pars['npts']

    def initialize(self):

        self.people.initialize(sim)
        for module in self.modules:
            module.initialize(sim)


    def run(self):
        self.initialize()
        for i in range(self.pars.npts-1):
            self.step()
        self.people.finalize_results(sim)

        for module in self.modules:
            module.finalize_results(sim)

    def step(self):
        self.ti += self.pars.dt

        self.people.update_states_pre(sim)

        for module in self.modules:
            module.transmit(self)

        self.people.update_results(sim)


#######

# TODO - should the module be stateful or stateless?

people = make_people(100)
pars = defaultdict(dict)
pars['gonorrhea']['beta'] = {'random':0.3,'msm':0.5}
sim = Sim(people, [Gonorrhea], pars=pars)
sim.run()
plt.figure()
plt.plot(sim.tvec, sim.results.gonorrhea.n_infected)


# # Custom module by user
# class Gonorrhea_DR(Gonorrhea):
#     default_pars = sc.dcp(Gonorrhea.default_pars)
#     default_pars['p_death'] = 0.3
#
# How should coinfection with these two modules be prevented?
#
#
# people = make_people(100)
# pars = defaultdict(dict)
# pars['gonorrhea']['beta'] = {'random':0.1,'msm':0.15}
# pars['gonorrhea_dr']['beta'] = {'random':0.05,'msm':0.1}
# sim = Sim(people, [Gonorrhea, Gonorrhea_DR], pars=pars)
# sim.run()
# plt.figure()
# plt.plot(sim.tvec, sim.results.gonorrhea.prevalence)
# plt.plot(sim.tvec, sim.results.gonorrhea_dr.prevalence, color='r')
# plt.show()