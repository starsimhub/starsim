import sciris as sc
import numpy as np
from .people import State
from .results import Result

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
