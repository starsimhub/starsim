import sciris as sc
import numpy as np
from .base import State
from .results import Result
from . import utils as ssu



class Module(sc.prettyobj):
    # Base module contains states/attributes
    
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pars = ssu.omerge(pars)
        self.states = sc.objdict()
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


    def update_module(self, sim):
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

    def update_module(self, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Check for new deliveries
        deliveries = sim.people[self.name].pregnant & (sim.people[self.name].ti_delivery <= sim.ti)
        sim.people[self.name].pregnant[deliveries] = False
        sim.people[self.name].postpartum[deliveries] = True
        sim.people[self.name].susceptible[deliveries] = False
        sim.people[self.name].ti_delivery[deliveries] = sim.ti

        # Check for new women emerging from post-partum
        postpartum = ~sim.people[self.name].pregnant & (sim.people[self.name].ti_postpartum <= sim.ti)
        sim.people[self.name].postpartum[postpartum] = False
        sim.people[self.name].susceptible[postpartum] = True
        sim.people[self.name].ti_postpartum[postpartum] = sim.ti

        # Maternal deaths
        maternal_deaths = ssu.true(sim.people[self.name].ti_dead <= sim.ti)
        if len(maternal_deaths):
            sim.people.alive[maternal_deaths] = False
            sim.people.ti_dead[maternal_deaths] = sim.ti

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

                # Change states for the newly pregnant woman
                sim.people[self.name].susceptible[uids] = False
                sim.people[self.name].pregnant[uids] = True
                sim.people[self.name].ti_pregnant[uids] = sim.ti

                # Outcomes for pregnancies
                dur = np.full(len(uids), sim.ti + cpars['dur_pregnancy'] / sim.dt)
                dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
                sim.people[self.name].ti_delivery[
                            uids] = dur  # Currently assumes maternal deaths still result in a live baby
                dur_post_partum = np.full(len(uids), dur + cpars['dur_postpartum'] / sim.dt)
                sim.people[self.name].ti_postpartum[uids] = dur_post_partum

                if len(ssu.true(dead)):
                    sim.people[self.name].ti_dead[uids[dead]] = dur[dead]


        return



    def update_results(self, sim):
        mppl = sim.people[self.name]
        sim.results[self.name]['pregnancies'][sim.ti] = np.count_nonzero(mppl.ti_pregnant == sim.ti)
        sim.results[self.name]['births'][sim.ti] = np.count_nonzero(mppl.ti_delivery == sim.ti)
