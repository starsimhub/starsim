import sciris as sc
import numpy as np
from .defaults import State
from .results import Result
from . import defaults as ssd
from . import misc as ssm
from . import utils as ssu

class Module():
    # Base module contains states/attributes that all modules have
    default_pars = {}
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
    def update_states(cld, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    @classmethod
    def make_new_cases(cls, sim):
        # Add new cases of module, through transmission, incidence, etc.
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

    @classmethod
    def update_states(cls, sim):
        # Update CD4
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art] += (sim.pars.hiv.cd4_max - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate
        sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art] += (sim.pars.hiv.cd4_min - sim.people.hiv.cd4[sim.people.alive & sim.people.hiv.infected & ~sim.people.hiv.on_art])/sim.pars.hiv.cd4_rate

    @classmethod
    def initialize(cls, sim):
        super(HIV, cls).initialize(sim)

        # Pick some initially infected agents
        cls.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

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
    def make_new_cases(cls, sim):
        eff_condoms = sim.pars[cls.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            if layer.transmission == 'vertical':
                effective_condoms = 1
            else:
                condoms = layer.pars['condoms']
                effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = layer['beta']*rel_trans[a]*sim.people[cls.name].rel_trans[a]*rel_sus[b]*sim.people[cls.name].rel_sus[b]*sim.pars[cls.name]['beta'][k]*(1-effective_condoms)
                    cls.set_prognoses(sim, b[np.random.random(len(a))<p_transmit])

    @classmethod
    def set_prognoses(cls, sim, uids):
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
    }

    def __init__(self, pars):
        super().__init__(pars)

    @classmethod
    def update_states(cls, sim):
        gonorrhea_deaths = sim.people.gonorrhea.ti_dead <= sim.t
        sim.people.alive[gonorrhea_deaths] = False
        sim.people.date_dead[gonorrhea_deaths] = sim.t

    @classmethod
    def initialize(cls, sim):
        super(Gonorrhea, cls).initialize(sim)

        # Pick some initially infected agents
        cls.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

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
    def make_new_cases(cls, sim):
        eff_condoms = sim.pars[cls.name]['eff_condoms']

        for k, layer in sim.people.contacts.items():
            if layer.transmission == 'vertical':
                effective_condoms = 1
            else:
                condoms = layer.pars['condoms']
                effective_condoms = ssd.default_float(condoms * eff_condoms)
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & sim.people.alive).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & sim.people.alive).astype(float)
                for a,b in [[layer['p1'],layer['p2']],[layer['p2'],layer['p1']]]:
                    # probability of a->b transmission
                    p_transmit = layer['beta']*rel_trans[a]*sim.people[cls.name].rel_trans[a]*rel_sus[b]*sim.people[cls.name].rel_sus[b]*sim.pars[cls.name]['beta'][k]*(1-effective_condoms)
                    cls.set_prognoses(sim, b[np.random.random(len(a))<p_transmit])

    @classmethod
    def set_prognoses(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.t

        dur = sim.t+np.random.poisson(sim.pars[cls.name]['dur_inf']/sim.pars.dt, len(uids))
        dead = np.random.random(len(uids))<sim.pars[cls.name].p_death
        sim.people[cls.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]


class Pregnancy(Module):
    # Other, e.g. postpartum, on contraception...
    states_to_set = [
        State('infertile', bool, False),  # Applies to girls and women outside the fertility window
        State('susceptible', bool, True),  # Applies to girls and women inside the fertility window - needs renaming
        State('pregnant', bool, False),  # Currently pregnant
        State('ti_pregnant', float, np.nan),  # Time pregnancy begins
        State('ti_delivery', float, np.nan),  # Time of delivery
        State('ti_dead', float, np.nan),  # Maternal mortality
    ]

    default_pars = {
        'dur_pregnancy': 0.75,  # Make this a distribution?
        'inci': 0.03,  # Replace this with age-specific rates
        'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
        'initial': 3,  # Number of women initially pregnant
    }

    def __init__(self, pars):
        super().__init__(pars)

    @classmethod
    def initialize(cls, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        super(Pregnancy, cls).initialize(sim)
        sim.results[cls.name]['pregnancies'] = Result(cls.name, 'pregnancies', sim.npts, dtype=int)
        sim.results[cls.name]['births'] = Result(cls.name, 'births', sim.npts, dtype=int)
        sim['birth_rates'] = None
        return

    @classmethod
    def update_states(cls, sim):

        cpars = sim.pars[cls.name]

        # Deliveries
        deliveries = sim.people[cls.name].pregnant & (sim.people[cls.name].ti_delivery <= sim.t)
        sim.people[cls.name].pregnant[deliveries] = False
        sim.people[cls.name].susceptible[deliveries] = True  # Currently assuming no postpartum window
        sim.people[cls.name].ti_delivery[deliveries] = sim.t

        # Maternal deaths
        maternal_deaths = ssu.true(sim.people.pregnancy.ti_dead <= sim.t)
        if len(maternal_deaths):
            sim.people.alive[maternal_deaths] = False
            sim.people.date_dead[maternal_deaths] = sim.t

        return

    @classmethod
    def make_new_cases(cls, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate key variables
        cpars = sim.pars[cls.name]
        ppl = sim.people
        this_inci = cpars['inci']

        # If incidence of pregnancy is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        if this_inci > 0:
            demon_conds = ppl.is_female & (ppl.alive) & ppl[cls.name].susceptible
            inds_to_choose_from = ssu.true(demon_conds)
            uids = ssu.binomial_filter(this_inci, inds_to_choose_from)

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                # Grow the arrays and set properties for the unborn agents
                # noinspection PyProtectedMember
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
                        layer.add_connections(uids, new_inds)

                # Set prognoses for the pregnancies
                cls.set_prognoses(sim, uids)

        return

    @classmethod
    def set_prognoses(cls, sim, uids):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """
        cpars = sim.pars[cls.name]

        # Change states for the newly pregnant woman
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].pregnant[uids] = True  # bad to have a module directly modify a people attribute?
        sim.people[cls.name].ti_pregnant[uids] = sim.t

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.t + cpars['dur_pregnancy'] / sim.pars.dt)
        dead = np.random.random(len(uids)) < sim.pars[cls.name].p_death
        sim.people[cls.name].ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        if len(ssu.true(dead)):
            sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]
        return

    @classmethod
    def update_results(cls, sim):
        mppl = sim.people[cls.name]
        sim.results[cls.name]['pregnancies'][sim.t] = np.count_nonzero(mppl.ti_pregnant == sim.t)
        sim.results[cls.name]['births'][sim.t] = np.count_nonzero(mppl.ti_delivery == sim.t)