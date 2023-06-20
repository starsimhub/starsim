import sciris as sc
import numpy as np
from .people import State
from .results import Result
from . import utils as ssu


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
            for k, v in cls.default_pars.items():
                if k not in sim.pars[cls.name]:
                    sim.pars[cls.name][k] = v

        sim.results[cls.name] = sc.objdict()

    @classmethod
    def update_states_pre(cls, sim):
        # Carry out any autonomous state changes at the start of the timestep
        pass

    @classmethod
    def make_new_cases(cls, sim):
        """ Create new cases in the population, using either incidence or network-based transmission """

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
        State('ti_infected', float, 0),
        State('on_art', bool, False),
        State("cd4", float, 500),
    ]

    default_pars = {
        'cd4_min': 100,
        'cd4_max': 500,
        'cd4_rate': 5,
        'initial': 3,
    }

    @classmethod
    def update_states_pre(cls, sim):
        # Update CD4
        sim.people.hiv.cd4[~sim.people.dead & sim.people.hiv.infected & sim.people.hiv.on_art] += (
                                                                                                              sim.pars.hiv.cd4_max -
                                                                                                              sim.people.hiv.cd4[
                                                                                                                  ~sim.people.dead & sim.people.hiv.infected & sim.people.hiv.on_art]) / sim.pars.hiv.cd4_rate
        sim.people.hiv.cd4[~sim.people.dead & sim.people.hiv.infected & ~sim.people.hiv.on_art] += (
                                                                                                               sim.pars.hiv.cd4_min -
                                                                                                               sim.people.hiv.cd4[
                                                                                                                   ~sim.people.dead & sim.people.hiv.infected & ~sim.people.hiv.on_art]) / sim.pars.hiv.cd4_rate

    @classmethod
    def initialize(cls, sim):
        super(HIV, cls).initialize(sim)

        # Pick some initially infected agents
        cls.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

        if 'beta' not in sim.pars[cls.name]:
            sim.pars[cls.name].beta = sc.objdict({k: 1 for k in sim.people.contacts})

        sim.results[cls.name]['n_susceptible'] = Result(cls.name, 'n_susceptible', sim.npts, dtype=int)
        sim.results[cls.name]['n_infected'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['prevalence'] = Result(cls.name, 'prevalence', sim.npts, dtype=float)
        sim.results[cls.name]['new_infections'] = Result(cls.name, 'n_infected', sim.npts, dtype=int)
        sim.results[cls.name]['n_art'] = Result(cls.name, 'n_art', sim.npts, dtype=int)

    @classmethod
    def update_results(cls, sim):
        sim.results[cls.name]['n_susceptible'][sim.ti] = np.count_nonzero(sim.people.hiv.susceptible)
        sim.results[cls.name]['n_infected'][sim.ti] = np.count_nonzero(sim.people.hiv.infected)
        sim.results[cls.name]['prevalence'][sim.ti] = sim.results[cls.name].n_infected[sim.ti] / sim.people.n
        sim.results[cls.name]['new_infections'] = np.count_nonzero(sim.people[cls.name].ti_infected == sim.ti)
        sim.results[cls.name]['n_art'] = np.count_nonzero(~sim.people.dead & sim.people[cls.name].on_art)

    @classmethod
    def make_new_cases(cls, sim):
        for k, layer in sim.people.contacts.items():
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & ~sim.people.dead).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & ~sim.people.dead).astype(float)
                for a, b in [[layer.p1, layer.p2], [layer.p2, layer.p1]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.beta * sim.pars[cls.name]['beta'][k]
                    cls.set_prognoses(sim, b[np.random.random(len(a)) < p_transmit])

    @classmethod
    def set_prognoses(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.ti

        # Check for MTCT - here or in pregnancy module?



class Gonorrhea(Module):
    states = [
        State('susceptible', bool, True),
        State('infected', bool, False),
        State('ti_infected', float, 0),
        State('ti_recovered', float, 0),
        State('ti_dead', float, np.nan),  # Death due to gonorrhea
    ]

    default_pars = {
        'dur_inf': 3,  # not modelling diagnosis or treatment explicitly here
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
        cls.set_prognoses(sim, np.random.choice(sim.people.uid, sim.pars[cls.name]['initial']))

        if 'beta' not in sim.pars[cls.name]:
            sim.pars[cls.name].beta = sc.objdict({k: 1 for k in sim.people.contacts})

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
    def make_new_cases(cls, sim):
        # Sexual transmission
        for k, layer in sim.people.contacts.items():
            if k in sim.pars[cls.name]['beta']:
                rel_trans = (sim.people[cls.name].infected & ~sim.people.dead).astype(float)
                rel_sus = (sim.people[cls.name].susceptible & ~sim.people.dead).astype(float)
                for a, b in [[layer.p1, layer.p2], [layer.p2, layer.p1]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.beta * sim.pars[cls.name]['beta'][k]
                    cls.set_prognoses(sim, b[np.random.random(len(a)) < p_transmit])

    # Vertical transmission



    @classmethod
    def set_prognoses(cls, sim, uids):
        sim.people[cls.name].susceptible[uids] = False
        sim.people[cls.name].infected[uids] = True
        sim.people[cls.name].ti_infected[uids] = sim.ti

        dur = sim.ti + np.random.poisson(sim.pars[cls.name]['dur_inf'] / sim.pars.dt, len(uids))
        dead = np.random.random(len(uids)) < sim.pars[cls.name].p_death
        sim.people[cls.name].ti_recovered[uids[~dead]] = dur[~dead]
        sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]

        # if HIV in sim.modules:
        #     # check CD4 count
        #     # increase susceptibility where relevant


class Pregnancy(Module):

    # Other, e.g. postpartum, on contraception...
    states = [
        State('infertile', bool, False),  # Applies to girls and women outside the fertility window
        State('susceptible', bool, True),  # Applies to girls and women inside the fertility window - needs renaming
        State('pregnant', bool, False),  # Currently pregnant
        State('ti_pregnant', float, np.nan),  # Time pregnancy begins
        State('ti_delivery', float, np.nan),  # Time of delivery
        State('ti_dead', float, np.nan),  # Maternal mortality
    ]

    default_pars = {
        'dur_pregnancy': 0.75,  # Make this a distribution?
        'inci': 0.01,  # Replace this with age-specific rates
        'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
        'p_live_birth': 0.98,  # Probability of a live birth
    }

    def __init__(self, pars):
        super().__init__(pars)

    @classmethod
    def update_states_pre(cls, sim):

        cpars = sim.pars[cls.name]

        # Deliveries
        deliveries = sim.people[cls.name].ti_delivery <= sim.ti
        new_mothers = ssu.true(deliveries)
        sim.people[cls.name].pregnant[deliveries] = False
        sim.people[cls.name].susceptible[deliveries] = True  # Currently assuming no postpartum window

        # Births
        births = ssu.binomial_filter(cpars['p_live_birth'], ssu.true(deliveries))
        n_births = len(births)
        if n_births > 0:
            # noinspection PyProtectedMember
            new_inds = sim.people._grow(n_births, bp=True)
            sim.people.uid[new_inds] = new_inds
            sim.people.age[new_inds] = 0
            sim.people.female[new_inds] = np.random.randint(0, 2, n_births)

        # Maternal deaths
        maternal_deaths = ssu.true(sim.people.pregnancy.ti_dead <= sim.ti)
        if len(maternal_deaths):
            sim.people.dead[maternal_deaths] = True
            sim.people.ti_dead[maternal_deaths] = sim.ti

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

        # If incidence is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        if this_inci > 0:
            demon_conds = ppl.female & (~ppl.dead) & ppl[cls.name].susceptible
            inds_to_choose_from = ssu.true(demon_conds)
            uids = ssu.binomial_filter(this_inci, inds_to_choose_from)
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

        sim.people[cls.name].susceptible[uids] = False
        sim.people.pregnant[uids] = True  # bad to have a module directly modify a people attribute?
        sim.people[cls.name].ti_pregnant[uids] = sim.ti

        # Outcomes for pregnancies
        dur = np.full(len(uids), sim.ti + cpars['dur_pregnancy'] / sim.pars.dt)
        dead = np.random.random(len(uids)) < sim.pars[cls.name].p_death
        sim.people[cls.name].ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        if len(ssu.true(dead)):
            sim.people[cls.name].ti_dead[uids[dead]] = dur[dead]
        return

    @classmethod
    def update_results(cls, sim):
        return
