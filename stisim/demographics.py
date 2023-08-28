"""
Define pregnancy, deaths, migration, etc.
"""

import numpy as np
import stisim as ss


__all__ = ['Pregnancy']


class Pregnancy(ss.Module):

    def __init__(self, pars=None):
        super().__init__(pars)

        # Other, e.g. postpartum, on contraception...
        self.states = ss.ndict(
            ss.State('infertile', bool, False),  # Applies to girls and women outside the fertility window
            ss.State('susceptible', bool, True),  # Applies to girls and women inside the fertility window - needs renaming
            ss.State('pregnant', bool, False),  # Currently pregnant
            ss.State('postpartum', bool, False),  # Currently post-partum
            ss.State('ti_pregnant', float, np.nan),  # Time pregnancy begins
            ss.State('ti_delivery', float, np.nan),  # Time of delivery
            ss.State('ti_postpartum', float, np.nan),  # Time postpartum ends
            ss.State('ti_dead', float, np.nan),  # Maternal mortality
            self.states,
        )

        self.pars = ss.omerge({
            'dur_pregnancy': 0.75,  # Make this a distribution?
            'dur_postpartum': 0.5,  # Make this a distribution?
            'inci': 0.03,  # Replace this with age-specific rates
            'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
            'initial': 3,  # Number of women initially pregnant
        }, self.pars)

        return


    def initialize(self, sim):
        super().initialize(sim)
        self.init_results(sim)
        sim.pars['birth_rates'] = None  # This turns off birth rate pars so births only come from this module
        return
    

    def init_results(self, sim):
        """
        Results could include a range of birth outcomes e.g. LGA, stillbirths, etc.
        Still unclear whether this logic should live in the pregnancy module, the
        individual disease modules, the connectors, or the sim.
        """
        self.results['pregnancies'] = ss.Result('pregnancies', self.name, sim.npts, dtype=int)
        self.results['births']      = ss.Result('births', self.name, sim.npts, dtype=int)
        return
    
    
    def update(self, sim):
        """
        Perform all updates
        """
        self.update_states(sim)
        self.make_pregnancies(sim)
        self.update_results(sim)
        return


    def update_states(self, sim):
        """
        Update states
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
        maternal_deaths = ss.true(sim.people[self.name].ti_dead <= sim.ti)
        if len(maternal_deaths):
            sim.people.alive[maternal_deaths] = False
            sim.people.ti_dead[maternal_deaths] = sim.ti

        return


    def make_pregnancies(self, sim):
        """
        Select people to make pregnancy using incidence data
        This should use ASFR data from https://population.un.org/wpp/Download/Standard/Fertility/
        """
        # Abbreviate key variables
        ppl = sim.people

        # If incidence of pregnancy is non-zero, make some cases
        # Think about how to deal with age/time-varying fertility
        if self.pars.inci > 0:
            denom_conds = ppl.female & ppl.active & ppl[self.name].susceptible
            inds_to_choose_from = ss.true(denom_conds)
            uids = ss.binomial_filter(self.pars.inci, inds_to_choose_from)

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                # Grow the arrays and set properties for the unborn agents
                new_uids = sim.people.grow(n_unborn_agents)
                sim.people.age[new_uids] = -self.pars.dur_pregnancy
                sim.people.female[new_uids] = np.random.choice([True, False], size=n_unborn_agents)

                # Add connections to any vertical transmission layers
                # Placeholder code to be moved / refactored. The maternal network may need to be
                # handled separately to the sexual networks, TBC how to handle this most elegantly
                for lkey, layer in sim.people.networks.items():
                    if layer.vertical:  # What happens if there's more than one vertical layer?
                        durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy+self.pars.dur_postpartum)
                        layer.add_pairs(uids, new_uids, dur=durs)

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
        dur = np.full(len(uids), sim.ti + pars['dur_pregnancy'] / sim.dt)
        dead = np.random.random(len(uids)) < sim.pars[self.name].p_death
        sim.people[self.name].ti_delivery[uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(uids), dur + pars['dur_postpartum'] / sim.dt)
        sim.people[self.name].ti_postpartum[uids] = dur_post_partum

        if np.count_nonzero(dead):
            sim.people[self.name].ti_dead[uids[dead]] = dur[dead]
        return

    def update_results(self, sim):
        mppl = sim.people[self.name] # TODO: refactor with states owning their own data
        self.results['pregnancies'][sim.ti] = np.count_nonzero(mppl.ti_pregnant == sim.ti)
        self.results['births'][sim.ti] = np.count_nonzero(mppl.ti_delivery == sim.ti)
        return