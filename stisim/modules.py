import sciris as sc
import numpy as np
from .defaults import State
from .results import Result
from . import defaults as ssd
from . import misc as ssm

class Module():
    # Base module contains states/attributes that all modules have
    default_pars = {
        'connectors': []
    }
    states = [
        State('rel_sus', float, 1),
        State('rel_sev', float, 1),
        State('rel_trans', float, 1),
    ]

    def __init__(self):
        raise Exception('Modules should not be instantiated but their classes should be used directly')

    @classmethod
    def initialize(cls, sim):
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

        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        # and initializing any outputs that are required
        sim.people.add_module(cls)
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
    def update_connectors(cls, sim):
        if len(sim.modules)>1:
            connectors = sim.pars[cls.name]['connectors']
            if len(connectors)>0:
                for connector in connectors:
                    if callable(connector):
                        connector(sim)
                    else:
                        warnmsg = f'Connector must be a callable function'
                        ssm.warn(warnmsg, die=True)
            elif sim.t==0: # only raise warning on first timestep
                warnmsg = f'No connector for {cls.name}'
                ssm.warn(warnmsg, die=False)
            else:
                return
        return

    @classmethod
    @property
    def name(cls):
        # The module name is a lower-case version of its class name
        return cls.__name__.lower()


class HIV(Module):
    states_to_set = [
        State('susceptible', bool, True),
        State('infected', bool, False),
        State('ti_infected', float, 0),
        State('on_art', bool, False),
        State('cd4', float, 500),
    ]

    default_pars = {
        'cd4_min': 100,
        'cd4_max': 500,
        'cd4_rate': 5,
        'initial': 30,
        'eff_condoms': 0.7,
        'connectors': []
    }

    @classmethod
    def update_states_pre(cls, sim):
        # Update CD4
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art] += (sim.pars.hiv.cd4_max - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art] += (sim.pars.hiv.cd4_min - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate

    @classmethod
    def initialize(cls, sim):
        super(HIV, cls).initialize(sim)

        # Pick some initially infected agents
        cls.infect(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

        if 'beta' not in sim.pars[cls.name]:
            sim.pars[cls.name].beta = sc.objdict({k:1 for k in sim.people.contacts})

        sim.results[cls.name]['n_susceptible'] = Result(cls.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[cls.name]['n_infected'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['prevalence'] = Result(cls.name, 'prevalence', sim.npts, dtype=float)
        sim.results[cls.name]['new_infections'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['n_art'] = Result(cls.name, 'n_art', sim.npts, dtype=int)

    @classmethod
    def update_results(cls, sim):
        sim.results[cls.name]['n_susceptible'][sim.t] = np.count_nonzero(sim.people.hiv.susceptible)
        sim.results[cls.name]['n_infected'][sim.t] = np.count_nonzero(sim.people.hiv.infected)
        sim.results[cls.name]['prevalence'][sim.t] = sim.results[cls.name].n_infected[sim.t] / sim.people._n
        sim.results[cls.name]['new_infections'] = np.count_nonzero(sim.people[cls.name].ti_infected == sim.t)
        sim.results[cls.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[cls.name].on_art)

    @classmethod
    def transmit(cls, sim):
        eff_condoms = sim.pars[cls.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            condoms = layer.pars['condoms']
            effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a]*sim.people[cls.name].rel_trans[a]*rel_sus[b]*sim.people[cls.name].rel_sus[b]*sim.pars[cls.name]['beta'][k]*(1-effective_condoms)
                    cls.infect(sim, b[np.random.random(len(a))<p_transmit])

    @classmethod
    def infect(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.t

class Gonorrhea(Module):
    states_to_set = [
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
        'eff_condoms': 0.7,
        'connectors': []
    }

    def __init__(self, pars):
        super().__init__(pars)

    @classmethod
    def update_states_pre(cls, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.t
        sim.people.alive[gonorrhea_deaths] = False
        sim.people.date_dead[gonorrhea_deaths] = sim.t

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
        sim.results[cls.name]['n_susceptible'][sim.t] = np.count_nonzero(sim.people.gonorrhea.susceptible)
        sim.results[cls.name]['n_infected'][sim.t] = np.count_nonzero(sim.people.gonorrhea.infected)
        sim.results[cls.name]['prevalence'][sim.t] = sim.results[cls.name].n_infected[sim.t] / sim.people._n
        sim.results[cls.name]['new_infections'] = np.count_nonzero(sim.people[cls.name].ti_infected == sim.t)

    @classmethod
    def transmit(cls, sim):
        eff_condoms = sim.pars[cls.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            condoms = layer.pars['condoms']
            effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a]*sim.people[cls.name].rel_trans[a]*rel_sus[b]*sim.people[cls.name].rel_sus[b]*sim.pars[cls.name]['beta'][k]*(1-effective_condoms)
                    cls.infect(sim, b[np.random.random(len(a))<p_transmit])

    @classmethod
    def infect(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.t

        dur = sim.t+np.random.poisson(sim.pars[cls.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids))<sim.pars[cls.name].p_death
        sim.people[cls.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]

