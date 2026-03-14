"""
Example connector and intervention for fetal health testing.

Demonstrates how a disease (SIR) can damage fetal outcomes via a connector,
and how treatment can partially reverse that damage.
"""

import starsim as ss

__all__ = ['fetal_infection', 'fetal_treat']


class fetal_treat(ss.Intervention):
    """ Treat infected pregnant women each timestep """

    def __init__(self, disease='sir', **kwargs):
        super().__init__(**kwargs)
        self.disease_name = disease
        self.define_pars(
            p_treat=ss.bernoulli(p=0.9),
        )
        self.define_states(
            ss.FloatArr('ti_treated', label='Time of treatment'),
        )
        return

    def step(self):
        preg = self.sim.people.pregnancy
        disease = self.sim.diseases[self.disease_name]
        eligible = preg.pregnant & disease.infected
        treated = self.pars.p_treat.filter(eligible)
        if len(treated):
            disease.infected[treated] = False
            disease.recovered[treated] = True
            self.ti_treated[treated] = self.ti
        return


class fetal_infection(ss.Connector):
    """ Connector: infection damages fetal health, treatment partially reverses it """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            timing_shift       = ss.lognorm_ex(mean=3.0, std=1.0),
            growth_penalty     = 0.15,
            tx_growth_reversal = 0.7,
            tx_timing_reversal = 0.7,
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        fh = sim.custom['fetal_health']
        fh.add_conception_callback(self._on_conception)
        return

    def _on_conception(self, uids):
        infected = self.sim.diseases.sir.infected[uids]
        infected_uids = uids[infected]
        if len(infected_uids):
            self._apply_damage(infected_uids)
        return

    def _apply_damage(self, uids):
        fh = self.sim.custom['fetal_health']
        shifts = self.pars.timing_shift.rvs(uids)
        fh.apply_timing_shift(uids, shifts)
        fh.apply_growth_restriction(uids, self.pars.growth_penalty)
        return

    def _apply_treatment_reversal(self, uids):
        fh = self.sim.custom['fetal_health']
        reversible = self.pars.growth_penalty * self.pars.tx_growth_reversal
        fh.reverse_growth_restriction(uids, reversible)
        fh.reverse_timing_shift(uids, self.pars.tx_timing_reversal)
        return

    def step(self):
        sim = self.sim
        preg = sim.people.pregnancy
        if not preg.pregnant.any():
            return

        pregnant_uids = preg.pregnant.uids

        # New infections in pregnant women
        newly_infected = sim.diseases.sir.ti_infected == self.ti
        affected = pregnant_uids[newly_infected[pregnant_uids]]
        if len(affected):
            self._apply_damage(affected)

        # Newly treated pregnant women
        if 'tx' in sim.interventions:
            intv = sim.interventions['tx']
            just_treated = intv.ti_treated == self.ti
            treated_pregnant = pregnant_uids[just_treated[pregnant_uids]]
            if len(treated_pregnant):
                self._apply_treatment_reversal(treated_pregnant)
        return
