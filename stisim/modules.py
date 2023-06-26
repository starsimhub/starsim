import sciris as sc
import numpy as np
from .defaults import State
from .results import Result
from . import defaults as ssd
from . import utils as ssu
from . import population as sspop


class Module(sc.prettyobj):
    # Base module contains states/attributes that all modules have
    default_pars = {}
    states = [
        State('rel_sus', float, 1),
        State('rel_sev', float, 1),
        State('rel_trans', float, 1),
    ]
    
    def __init__(self, pars=None):
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
    
    def update_states(self, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    def make_new_cases(self, sim):
        # Add new cases of module, through transmission, incidence, etc.
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

    
    def update_states(self, sim):
        # Update CD4
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art] += (sim.pars.hiv.cd4_max - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art] += (sim.pars.hiv.cd4_min - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate

    
    def initialize(self, sim):
        super(HIV, self).initialize(sim)

        # Pick some initially infected agents
        self.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[self.name]['initial']))

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

    
    def make_new_cases(self, sim):
        eff_condoms = sim.pars[self.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            if layer.transmission == 'vertical':
                effective_condoms = 1
            else:
                condoms = layer.pars['condoms']
                effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[self.name]['beta']:
                rel_trans = (sim.people[self.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[self.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = layer['beta']*rel_trans[a]*sim.people[self.name].rel_trans[a]*rel_sus[b]*sim.people[self.name].rel_sus[b]*sim.pars[self.name]['beta'][k]*(1-effective_condoms)
                    self.set_prognoses(sim, b[np.random.random(len(a))<p_transmit])

    
    def set_prognoses(self, sim, uids):
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

    
    def update_states(self, sim):
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
        self.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[self.name]['initial']))

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

    
    def make_new_cases(self, sim):
        eff_condoms = sim.pars[self.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            if layer.transmission == 'vertical':
                effective_condoms = 1
            else:
                condoms = layer.pars['condoms']
                effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[self.name]['beta']:
                rel_trans = (sim.people[self.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[self.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = layer['beta']*rel_trans[a]*sim.people[self.name].rel_trans[a]*rel_sus[b]*sim.people[self.name].rel_sus[b]*sim.pars[self.name]['beta'][k]*(1-effective_condoms)
                    self.set_prognoses(sim, b[np.random.random(len(a))<p_transmit])

    
    def set_prognoses(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.t

        dur = sim.t+np.random.poisson(sim.pars[self.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids))<sim.pars[self.name].p_death
        sim.people[self.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[self.name].ti_dead[uids[dead]] = dur[dead]


class Pregnancy(Module):
    # Other, e.g. postpartum, on contraception...
    states_to_set = [
        State('infertile', bool, False),  # Applies to girls and women outside the fertility window
        State('susceptible', bool, True),  # Applies to girls and women inside the fertility window - needs renaming
        State('pregnant', bool, False),  # Currently pregnant
        State('postpartum', bool, False),  # Currently post-partum
        State('ti_pregnant', float, np.nan),  # Time pregnancy begins
        State('ti_delivery', float, np.nan),  # Time of delivery
        State('ti_postpartum', float, np.nan),  # Time postpartum ends
        State('ti_dead', float, np.nan),  # Maternal mortality
    ]

    default_pars = {
        'dur_pregnancy': 0.75,  # Make this a distribution?
        'dur_postpartum': 0.5,  # Make this a distribution?
        'inci': 0.03,  # Replace this with age-specific rates
        'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
        'initial': 3,  # Number of women initially pregnant
    }

    # def __init__(self, pars=None):
    #     super().__init__(pars)


    def initialize(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        super(Pregnancy, self).initialize(sim)
        sim.results[self.name]['pregnancies'] = Result(self.name, 'pregnancies', sim.npts, dtype=int)
        sim.results[self.name]['births'] = Result(self.name, 'births', sim.npts, dtype=int)
        sim['birth_rates'] = None # This turns off birth rates so births only come from this module
        return


    def update_states(self, sim):

        cpars = sim.pars[self.name]

        # Check for new deliveries
        deliveries = sim.people[self.name].pregnant & (sim.people[self.name].ti_delivery <= sim.t)
        sim.people[self.name].pregnant[deliveries] = False
        sim.people[self.name].postpartum[deliveries] = True
        sim.people[self.name].susceptible[deliveries] = False
        sim.people[self.name].ti_delivery[deliveries] = sim.t

        # Check for new kids emerging from post-partum
        postpartum = ~sim.people[self.name].pregnant & (sim.people[self.name].ti_postpartum <= sim.t)
        sim.people[self.name].postpartum[postpartum] = False
        sim.people[self.name].susceptible[postpartum] = True
        sim.people[self.name].ti_postpartum[postpartum] = sim.t

        delivery_inds = ssu.true(deliveries)
        if len(delivery_inds):
            for _, layer in sim.people.contacts.items():
                if isinstance(layer, sspop.Maternal):
                    new_birth_inds = layer.find_contacts(delivery_inds)
                    new_births = len(new_birth_inds) * sim['pop_scale']
                    sim.people.demographic_flows['births'] = new_births

        # Maternal deaths
        maternal_deaths = ssu.true(sim.people[self.name].ti_dead <= sim.t)
        if len(maternal_deaths):
            sim.people.alive[maternal_deaths] = False
            sim.people.date_dead[maternal_deaths] = sim.t

        return


    def make_new_cases(self, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate key variables
        cpars = sim.pars[self.name]
        ppl = sim.people
        this_inci = cpars['inci']

        # If incidence of pregnancy is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        if this_inci > 0:
            demon_conds = ppl.is_female & ppl.is_active & ppl[self.name].susceptible
            inds_to_choose_from = ssu.true(demon_conds)
            uids = ssu.binomial_filter(this_inci, inds_to_choose_from)

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                # Grow the arrays and set properties for the unborn agents
                new_inds = sim.people._grow(n_unborn_agents)
                sim.people.uid[new_inds] = new_inds
                sim.people.age[new_inds] = -cpars['dur_pregnancy']
                sex = np.random.binomial(1, 0.5, n_unborn_agents)
                debut = np.full(n_unborn_agents, np.nan, dtype=ssd.default_float)
                debut[sex == 1] = ssu.sample(**sim.pars['debut']['m'], size=sum(sex))
                debut[sex == 0] = ssu.sample(**sim.pars['debut']['f'], size=n_unborn_agents - sum(sex))
                sim.people.debut[new_inds] = debut

                # Add connections to any vertical transmission layers
                # Placeholder code to be moved / refactored. The maternal contact network may need to be
                # handled separately to the sexual networks, TBC how to handle this most elegantly
                for lkey, layer in sim.people.contacts.items():
                    if layer.transmission == 'vertical':  # What happens if there's more than one vertical layer?
                        durs = np.full((n_unborn_agents), fill_value=cpars['dur_pregnancy']+cpars['dur_postpartum'])
                        layer.add_connections(uids, new_inds, dur=durs)

                # Set prognoses for the pregnancies
                self.set_prognoses(sim, uids)

        return


    def set_prognoses(self, sim, uids):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """
        cpars = sim.pars[self.name]

        # Change states for the newly pregnant woman
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].pregnant[uids] = True
        sim.people[self.name].ti_pregnant[uids] = sim.t

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.t + cpars['dur_pregnancy'] / sim.pars.dt)
        dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
        sim.people[self.name].ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(uids), dur + cpars['dur_postpartum'] / sim.pars.dt)
        sim.people[self.name].ti_postpartum[uids] = dur_post_partum

        if len(ssu.true(dead)):
            sim.people[self.name].ti_dead[uids[dead]] = dur[dead]
        return


    def update_results(self, sim):
        mppl = sim.people[self.name]
        sim.results[self.name]['pregnancies'][sim.t] = np.count_nonzero(mppl.ti_pregnant == sim.t)
        sim.results[self.name]['births'][sim.t] = np.count_nonzero(mppl.ti_delivery == sim.t)
