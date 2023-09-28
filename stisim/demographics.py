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
        self.infertile = ss.State('infertile', bool, False)   # Applies to girls and women outside the fertility window
        self.susceptible = ss.State('susceptible', bool, True)   # Applies to girls and women inside the fertility window - needs renaming
        self.pregnant = ss.State('pregnant', bool, False)   # Currently pregnant
        self.postpartum = ss.State('postpartum', bool, False)   # Currently post-partum
        self.ti_pregnant = ss.State('ti_pregnant', float, np.nan)   # Time pregnancy begins
        self.ti_delivery = ss.State('ti_delivery', float, np.nan)   # Time of delivery
        self.ti_postpartum = ss.State('ti_postpartum', float, np.nan)   # Time postpartum ends
        self.ti_dead = ss.State('ti_dead', float, np.nan)   # Maternal mortality

        self.pars = ss.omerge({
            'dur_pregnancy': 0.75,  # Make this a distribution?
            'dur_postpartum': 0.5,  # Make this a distribution?
            'inci': 0.03,  # Replace this with age-specific rates
            'p_death': 0.02,  # Probability of maternal death. Question, should this be linked to age and/or duration?
            'initial': 3,  # Number of women initially pregnant
        }, self.pars)

        self.rng_sex = ss.Stream(self.multistream)('sex_at_birth')
        self.rng_conception = ss.Stream(self.multistream)('conception')

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
        self.results += ss.Result(self.name, 'pregnancies', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'births', sim.npts, dtype=int)
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
        deliveries = self.pregnant & (self.ti_delivery <= sim.ti)
        self.pregnant[deliveries] = False
        self.postpartum[deliveries] = True
        self.susceptible[deliveries] = False
        self.ti_delivery[deliveries] = sim.ti

        # Check for new women emerging from post-partum
        postpartum = ~self.pregnant & (self.ti_postpartum <= sim.ti)
        self.postpartum[postpartum] = False
        self.susceptible[postpartum] = True
        self.ti_postpartum[postpartum] = sim.ti

        # Maternal deaths
        maternal_deaths = ss.true(self.ti_dead <= sim.ti)
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
            denom_conds = ppl.female & ppl.active & self.susceptible
            inds_to_choose_from = ss.true(denom_conds)
            uids = self.rng_conception.bernoulli_filter(arr=inds_to_choose_from, prob=self.pars.inci)

            # Add UIDs for the as-yet-unborn agents so that we can track prognoses and transmission patterns
            n_unborn_agents = len(uids)
            if n_unborn_agents > 0:
                # Grow the arrays and set properties for the unborn agents
                new_uids = sim.people.grow(n_unborn_agents)
                sim.people.age[new_uids] = -self.pars.dur_pregnancy
                sim.people.female[new_uids] = self.rng_sex.bernoulli(arr=uids, prob=0.5) # Replace 0.5 with sex ratio at birth

                # Add connections to any vertical transmission layers
                # Placeholder code to be moved / refactored. The maternal network may need to be
                # handled separately to the sexual networks, TBC how to handle this most elegantly
                for lkey, layer in sim.people.networks.items():
                    if layer.vertical:  # What happens if there's more than one vertical layer?
                        durs = np.full(n_unborn_agents, fill_value=self.pars.dur_pregnancy+self.pars.dur_postpartum)
                        layer.add_pairs(uids, new_uids, dur=durs)

                # Set prognoses for the pregnancies
                self.set_prognoses(sim, uids) # Could set from_uids to network partners?

        return

    def set_prognoses(self, sim, to_uids, from_uids=None):
        """
        Make pregnancies
        Add miscarriage/termination logic here
        Also reconciliation with birth rates
        Q, is this also a good place to check for other conditions and set prognoses for the fetus?
        """

        # Change states for the newly pregnant woman
        self.susceptible[to_uids] = False
        self.pregnant[to_uids] = True
        self.ti_pregnant[to_uids] = sim.ti

        # Outcomes for pregnancies
        dur = np.full(len(to_uids), sim.ti + self.pars.dur_pregnancy / sim.dt)
        dead = np.random.random(len(to_uids)) < self.pars.p_death
        self.ti_delivery[to_uids] = dur  # Currently assumes maternal deaths still result in a live baby
        dur_post_partum = np.full(len(to_uids), dur + self.pars.dur_postpartum / sim.dt)
        self.ti_postpartum[to_uids] = dur_post_partum

        if np.count_nonzero(dead):
            self.ti_dead[to_uids[dead]] = dur[dead]
        return

    def update_results(self, sim):
        self.results['pregnancies'][sim.ti] = np.count_nonzero(self.ti_pregnant == sim.ti)
        self.results['births'][sim.ti] = np.count_nonzero(self.ti_delivery == sim.ti)
        return
