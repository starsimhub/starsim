import sciris as sc
import numpy as np
from .base import State
from .results import Result
from . import utils as ssu
from . import networks as ssnet


class Module(sc.prettyobj):
    # Base module contains states/attributes that all modules have
    
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pars = ssu.omerge(pars)
        self.states = ssu.named_dict(
            State('rel_sus', float, 1),
            State('rel_sev', float, 1),
            State('rel_trans', float, 1),
        )
        self.results = sc.objdict()
        return

    def initialize(self, sim):
        # Merge parameters
        sim.pars[self.name] = self.pars
        sim.results[self.name] = self.results

        # Add this module to a People instance. This would always involve calling People.add_module
        # but subsequently modules could have their own logic for initializing the default values
        # and initializing any outputs that are required
        sim.people.add_module(self)

        # Pick some initially infected agents
        self.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[self.name]['initial']))

        # Validate pars
        if 'beta' not in sim.pars[self.name]:
            sim.pars[self.name].beta = sc.objdict({k: [1, 1] for k in sim.people.networks})

        # Initialize results
        sim.results[self.name]['n_susceptible'] = Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[self.name]['n_infected'] = Result(self.name, 'n_infected', sim.npts, dtype=int)
        sim.results[self.name]['prevalence'] = Result(self.name, 'prevalence', sim.npts, dtype=float)
        sim.results[self.name]['new_infections'] = Result(self.name, 'n_infected', sim.npts, dtype=int)

    def update_states(self, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        pars = sim.pars[self.name]
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                rel_trans = (sim.people[self.name].infected & ~sim.people.dead).astype(float)
                rel_sus = (sim.people[self.name].susceptible & ~sim.people.dead).astype(float)
                for a, b, beta in [[layer['p1'], layer['p2'], pars['beta'][k][0]], [layer['p2'], layer['p1'], pars['beta'][k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer['beta'] * beta
                    new_cases = np.random.random(len(a)) < p_transmit
                    if new_cases.any():
                        self.set_prognoses(sim, b[new_cases])

    def set_prognoses(self, sim, uids):
        pass

    def update_results(self, sim):
        sim.results[self.name]['n_susceptible'][sim.ti] = np.count_nonzero(sim.people[self.name].susceptible)
        sim.results[self.name]['n_infected'][sim.ti] = np.count_nonzero(sim.people[self.name].infected)
        sim.results[self.name]['prevalence'][sim.ti] = sim.results[self.name].n_infected[sim.ti] / sim.people.n
        sim.results[self.name]['new_infections'] = np.count_nonzero(sim.people[self.name].ti_infected == sim.ti)

    def finalize_results(self, sim):
        pass

    @property
    def name(self):
        # The module name is a lower-case version of its class name
        return self.__class__.__name__.lower()


class HIV(Module):
    
    def __init__(self, pars=None):
        super().__init__(pars)
        self.states = ssu.omerge(ssu.named_dict(
            State('susceptible', bool, True),
            State('infected', bool, False),
            State('ti_infected', float, 0),
            State('on_art', bool, False),
            State('cd4', float, 500),
        ), self.states)
    
        self.pars = ssu.omerge({
            'cd4_min': 100,
            'cd4_max': 500,
            'cd4_rate': 5,
            'initial': 30,
            'eff_condoms': 0.7,
        }, self.pars)
        return

    def update_states(self, sim):
        # Update CD4
        hivppl = sim.people.hiv
        hivppl.cd4[sim.people.alive & hivppl.infected & hivppl.on_art] += (sim.pars.hiv.cd4_max - hivppl.cd4[sim.people.alive & hivppl.infected & hivppl.on_art])/sim.pars.hiv.cd4_rate
        hivppl.cd4[sim.people.alive & hivppl.infected & ~hivppl.on_art] += (sim.pars.hiv.cd4_min - hivppl.cd4[sim.people.alive & hivppl.infected & ~hivppl.on_art])/sim.pars.hiv.cd4_rate
        return

    def initialize(self, sim):
        Module.initialize(self, sim)
        sim.results[self.name]['n_art'] = Result('n_art', self.name, sim.npts, dtype=int)
    
    def update_results(self, sim):
        super(HIV, self).update_results(sim)
        sim.results[self.name]['n_art'] = np.count_nonzero(sim.people.alive & sim.people[self.name].on_art)

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
    
    def set_prognoses(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti


class Gonorrhea(Module):

    def __init__(self, pars=None):
        super().__init__(pars)
        self.states = ssu.omerge(ssu.named_dict(
            State('susceptible', bool, True),
            State('infected', bool, False),
            State('ti_infected', float, 0),
            State('ti_recovered', float, 0),
            State('ti_dead', float, np.nan), # Death due to gonorrhea
        ), self.states)

        self.pars = ssu.omerge({
            'dur_inf': 3, # not modelling diagnosis or treatment explicitly here
            'p_death': 0.2,
            'initial': 3,
            'eff_condoms': 0.7,
        }, self.pars)
        return
    
    def update_states(self, sim):
        # What if something in here should depend on another module?
        # I guess we could just check for it e.g., 'if HIV in sim.modules' or
        # 'if 'hiv' in sim.people' or something
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.ti
        sim.people.alive[gonorrhea_deaths] = False
        sim.people.ti_dead[gonorrhea_deaths] = sim.ti
        return

    def initialize(self, sim):
        # Do any steps in this method depend on what other modules are going to be added? We can inspect
        # them via sim.modules at this point
        Module.initialize(self, sim)
    
    def update_results(self, sim):
        super(Gonorrhea, self).update_results(sim)
    
    def make_new_cases(self, sim):
        super(Gonorrhea, self).make_new_cases(sim)

    def set_prognoses(self, sim, uids):
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti

        dur = sim.ti + np.random.poisson(sim.pars[self.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
        sim.people[self.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[self.name].ti_dead[uids[dead]] = dur[dead]


class Pregnancy(Module):

    def __init__(self, pars=None):
        super().__init__(pars)


        # Other, e.g. postpartum, on contraception...
        self.states = ssu.omerge(ssu.named_dict(
            State('infertile', bool, False),  # Applies to girls and women outside the fertility window
            State('susceptible', bool, True),  # Applies to girls and women inside the fertility window - needs renaming
            State('pregnant', bool, False),  # Currently pregnant
            State('postpartum', bool, False),  # Currently post-partum
            State('ti_pregnant', float, np.nan),  # Time pregnancy begins
            State('ti_delivery', float, np.nan),  # Time of delivery
            State('ti_postpartum', float, np.nan),  # Time postpartum ends
            State('ti_dead', float, np.nan),  # Maternal mortality
        ), self.states)

        self.pars = ssu.omerge({
            'dur_pregnancy': 0.75,  # Make this a distribution?
            'dur_postpartum': 0.5,  # Make this a distribution?
            'inci': 0.03,  # Replace this with age-specific rates
            'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
            'initial': 3,  # Number of women initially pregnant
        }, self.pars)

        return

    def initialize(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        Module.initialize(self, sim)
        sim.results[self.name]['pregnancies'] = Result('pregnancies', self.name, sim.npts, dtype=int)
        sim.results[self.name]['births'] = Result('births', self.name, sim.npts, dtype=int)
        sim['birth_rates'] = None  # This turns off birth rate pars so births only come from this module
        return

    def update_states(self, sim):

        # Check for new deliveries
        deliveries = sim.people[self.name].pregnant & (sim.people[self.name].ti_delivery <= sim.ti)
        sim.people[self.name].pregnant[deliveries] = False
        sim.people[self.name].postpartum[deliveries] = True
        sim.people[self.name].susceptible[deliveries] = False
        sim.people[self.name].ti_delivery[deliveries] = sim.ti

        # Check for new kids emerging from post-partum
        postpartum = ~sim.people[self.name].pregnant & (sim.people[self.name].ti_postpartum <= sim.ti)
        sim.people[self.name].postpartum[postpartum] = False
        sim.people[self.name].susceptible[postpartum] = True
        sim.people[self.name].ti_postpartum[postpartum] = sim.ti

        delivery_inds = ssu.true(deliveries)
        if len(delivery_inds):
            for _, layer in sim.people.networks.items():
                if isinstance(layer, ssnet.maternal):
                    # new_birth_inds = layer.find_contacts(delivery_inds)  # Don't think we need this?
                    new_births = len(delivery_inds) * sim['pop_scale']
                    # sim.people.demographic_flows['births'] = new_births

        # Maternal deaths
        maternal_deaths = ssu.true(sim.people[self.name].ti_dead <= sim.ti)
        if len(maternal_deaths):
            sim.people.alive[maternal_deaths] = False
            sim.people.ti_dead[maternal_deaths] = sim.ti

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
            demon_conds = ppl.female & ppl.active & ppl[self.name].susceptible
            inds_to_choose_from = ssu.true(demon_conds)
            uids = ssu.binomial_filter(this_inci, inds_to_choose_from)

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                # Grow the arrays and set properties for the unborn agents
                new_inds = sim.people._grow(n_unborn_agents)
                sim.people.uid[new_inds] = new_inds
                sim.people.age[new_inds] = -cpars['dur_pregnancy']
                sim.people.female[new_inds] = np.random.choice([True, False], size=n_unborn_agents)

                # Add connections to any vertical transmission layers
                # Placeholder code to be moved / refactored. The maternal network may need to be
                # handled separately to the sexual networks, TBC how to handle this most elegantly
                for lkey, layer in sim.people.networks.items():
                    if layer.transmission == 'vertical':  # What happens if there's more than one vertical layer?
                        durs = np.full(n_unborn_agents, fill_value=cpars['dur_pregnancy']+cpars['dur_postpartum'])
                        layer.add_pairs(uids, new_inds, dur=durs)

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
        pars = sim.pars[self.name]

        # Change states for the newly pregnant woman
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].pregnant[uids] = True
        sim.people[self.name].ti_pregnant[uids] = sim.ti

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.ti + pars['dur_pregnancy'] / sim.pars.dt)
        dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
        sim.people[self.name].ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(uids), dur + pars['dur_postpartum'] / sim.dt)
        sim.people[self.name].ti_postpartum[uids] = dur_post_partum

        if len(ssu.true(dead)):
            sim.people[self.name].ti_dead[uids[dead]] = dur[dead]
        return

    def update_results(self, sim):
        mppl = sim.people[self.name]
        sim.results[self.name]['pregnancies'][sim.ti] = np.count_nonzero(mppl.ti_pregnant == sim.ti)
        sim.results[self.name]['births'][sim.ti] = np.count_nonzero(mppl.ti_delivery == sim.ti)
