import sciris as sc
import numpy as np
from .defaults import State
from .results import Result
from . import defaults as ssd


class Module(sc.prettyobj):
    # Base module contains states/attributes that all modules have
    default_pars = {}
    states = [
        State('rel_sus', float, 1),
        State('rel_sev', float, 1),
        State('rel_trans', float, 1),
    ]
    
    def __init__(self):
        pass
    
    def initialize(self, sim):
        # Merge parameters
        if self.name not in sim.pars:
            sim.pars[self.name] = sc.objdict(sc.dcp(self.default_pars))
        else:
            if ~isinstance(sim.pars[self.name], sc.objdict):
                sim.pars[self.name] = sc.objdict(sim.pars[self.name])
            for k,v in self.default_pars.items():
                if k not in sim.pars[self.name]:
                    sim.pars[self.name][k] = v

        sim.results[self.name] = sc.objdict()

        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        # and initializing any outputs that are required
        sim.people.add_module(self)
    
    def update_states_pre(self, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    
    def update_results(self, sim):
        pass

    
    def finalize_results(self, sim):
        pass

    
    @property
    def name(self):
        # The module name is a lower-case version of its class name
        return self.__class__.__name__.lower()


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
    }

    
    def update_states_pre(self, sim):
        # Update CD4
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art] += (sim.pars.hiv.cd4_max - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art] += (sim.pars.hiv.cd4_min - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate

    
    def initialize(self, sim):
        super(HIV, self).initialize(sim)

        # Pick some initially infected agents
        self.infect(sim, np.random.choice(sim.people.uid, sim.pars[self.name]['initial']))

        if 'beta' not in sim.pars[self.name]:
            sim.pars[self.name].beta = sc.objdict({k:1 for k in sim.people.contacts})

        sim.results[self.name]['n_susceptible'] = Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[self.name]['n_infected'] = Result(self.name, 'n_infected', sim.npts, dtype=int)
        sim.results[self.name]['prevalence'] = Result(self.name, 'prevalence', sim.npts, dtype=float)
        sim.results[self.name]['new_infections'] = Result(self.name, 'n_infected', sim.npts, dtype=int)
        sim.results[self.name]['n_art'] = Result(self.name, 'n_art', sim.npts, dtype=int)

    
    def update_results(self, sim):
        sim.results[self.name]['n_susceptible'][sim.t] = np.count_nonzero(sim.people.hiv.susceptible)
        sim.results[self.name]['n_infected'][sim.t] = np.count_nonzero(sim.people.hiv.infected)
        sim.results[self.name]['prevalence'][sim.t] = sim.results[self.name].n_infected[sim.t] / sim.people._n
        sim.results[self.name]['new_infections'] = np.count_nonzero(sim.people[self.name].ti_infected == sim.t)
        sim.results[self.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[self.name].on_art)

    
    def transmit(self, sim):
        eff_condoms = sim.pars[self.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            condoms = layer.pars['condoms']
            effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[self.name]['beta']:
                rel_trans = (sim.people[self.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[self.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a]*sim.people[self.name].rel_trans[a]*rel_sus[b]*sim.people[self.name].rel_sus[b]*sim.pars[self.name]['beta'][k]*(1-effective_condoms)
                    self.infect(sim, b[np.random.random(len(a))<p_transmit])

    
    def infect(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.t

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
    }

    # def __init__(self, pars=None):
    #     super().__init__(pars)

    
    def update_states_pre(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.t
        sim.people.alive[gonorrhea_deaths] = False
        sim.people.date_dead[gonorrhea_deaths] = sim.t

    
    def initialize(self, sim):
        # Do any steps in this method depend on what other modules are going to be added? We can inspect
        # them via sim.modules at this point
        super(Gonorrhea, self).initialize(sim)

        # Pick some initially infected agents
        self.infect(sim, np.random.choice(sim.people.uid, sim.pars[self.name]['initial']))

        if 'beta' not in sim.pars[self.name]:
            sim.pars[self.name].beta = sc.objdict({k:1 for k in sim.people.contacts})

        # TODO - not a huge fan of having the result key duplicate the Result name
        sim.results[self.name]['n_susceptible'] = Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[self.name]['n_infected'] = Result(self.name, 'n_infected', sim.npts, dtype=int)
        sim.results[self.name]['prevalence'] = Result(self.name, 'prevalence', sim.npts, dtype=float)
        sim.results[self.name]['new_infections'] = Result(self.name, 'n_infected', sim.npts, dtype=int)


    
    def update_results(self, sim):
        sim.results[self.name]['n_susceptible'][sim.t] = np.count_nonzero(sim.people.gonorrhea.susceptible)
        sim.results[self.name]['n_infected'][sim.t] = np.count_nonzero(sim.people.gonorrhea.infected)
        sim.results[self.name]['prevalence'][sim.t] = sim.results[self.name].n_infected[sim.t] / sim.people._n
        sim.results[self.name]['new_infections'] = np.count_nonzero(sim.people[self.name].ti_infected == sim.t)

    
    def transmit(self, sim):
        eff_condoms = sim.pars[self.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            condoms = layer.pars['condoms']
            effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[self.name]['beta']:
                rel_trans = (sim.people[self.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[self.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a]*sim.people[self.name].rel_trans[a]*rel_sus[b]*sim.people[self.name].rel_sus[b]*sim.pars[self.name]['beta'][k]*(1-effective_condoms)
                    self.infect(sim, b[np.random.random(len(a))<p_transmit])

    
    def infect(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.t

        dur = sim.t+np.random.poisson(sim.pars[self.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids))<sim.pars[self.name].p_death
        sim.people[self.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[self.name].ti_dead[uids[dead]] = dur[dead]

