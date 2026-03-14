"""
Fetal health connector and treatment intervention examples.

These modules demonstrate how to connect a disease to fetal health outcomes
and how to implement treatment that reverses fetal damage. They are designed
to be extended for real applications.

Architecture:
    - ``fetal_infection`` (Connector): watches for infections in pregnant women
      and applies fetal damage (preterm risk via timing shifts, low birth weight
      via growth restriction). Damage is applied both at conception (if already
      infected) and during pregnancy (if newly infected).
    - ``fetal_treat`` (Intervention): treats infected pregnant women within a
      specified year range and partially reverses fetal damage.

Both modules require ``ss.FetalHealth()`` in the sim's ``custom`` modules and
``ss.Pregnancy()`` in demographics. The connector also requires the target
disease (default: SIR) to be present.

Usage::

    import starsim as ss
    import starsim_examples as sse

    sim = ss.Sim(
        diseases=ss.SIR(beta=0.1),
        demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths()],
        connectors=sse.fetal_infection(),
        interventions=sse.fetal_treat(disease='sir', start_year=2025),
        custom=ss.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

To extend for a different disease, subclass ``fetal_infection`` and override
``_apply_damage`` with disease-specific logic (e.g. stage-dependent penalties).
To extend ``fetal_treat``, subclass and override ``step()`` with custom
eligibility criteria or reversal logic.
"""

import starsim as ss

__all__ = ['fetal_infection', 'fetal_treat']


class fetal_treat(ss.Intervention):
    """
    Treat infected pregnant women and partially reverse fetal damage.

    Each timestep, identifies pregnant women infected with the target disease,
    treats a fraction of them (curing infection), and reverses a portion of
    the accumulated fetal damage (growth restriction and timing shift).

    Treatment only applies between ``start_year`` and ``end_year``. If not
    specified, defaults to the full simulation period.

    Args:
        disease (str):          name of the target disease (default: 'sir')
        start_year (float):     first year treatment is available (default: sim start)
        end_year (float):       last year treatment is available (default: sim end)

    Pars:
        p_treat (Dist):             probability of treating an eligible woman per timestep
        tx_growth_reversal (float): fraction of growth restriction to reverse (0-1)
        tx_timing_reversal (float): fraction of timing shift to reverse (0-1)

    States:
        ti_treated (FloatArr):  timestep when each agent was treated

    Example — treatment starting in 2025 with 50% coverage::

        sse.fetal_treat(disease='sir', start_year=2025, p_treat=ss.bernoulli(p=0.5))
    """

    def __init__(self, disease='sir', start_year=None, end_year=None, **kwargs):
        super().__init__(**kwargs)
        self.disease_name = disease
        self.start_year = start_year
        self.end_year   = end_year
        self.define_pars(
            p_treat            = ss.bernoulli(p=0.9),
            tx_growth_reversal = 0.7,
            tx_timing_reversal = 0.7,
        )
        self.define_states(
            ss.FloatArr('ti_treated', label='Time of treatment'),
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate that the required disease and FetalHealth modules exist
        if self.disease_name not in sim.diseases:
            raise ValueError(f'fetal_treat requires disease "{self.disease_name}" but it was not found in the sim.')
        if 'fetal_health' not in sim.custom:
            raise ValueError('fetal_treat requires a FetalHealth module. Add ss.FetalHealth() to custom.')

        # Default year bounds to the full sim period
        if self.start_year is None: self.start_year = sim.t.start
        if self.end_year is None:   self.end_year = sim.t.stop
        return

    def step(self):
        # Only apply treatment within the active year range
        year = self.t.now('year')
        if year < self.start_year or year > self.end_year:
            return

        sim     = self.sim
        preg    = sim.people.pregnancy
        disease = sim.diseases[self.disease_name]
        fh      = sim.custom['fetal_health']

        # Find pregnant women infected with the target disease
        eligible = preg.pregnant & disease.infected
        treated = self.pars.p_treat.filter(eligible)
        if len(treated):
            # Cure infection
            disease.infected[treated]  = False
            disease.recovered[treated] = True
            self.ti_treated[treated]   = self.ti

            # Partially reverse fetal damage from the infection
            fh.reverse_growth_restriction(treated, self.pars.tx_growth_reversal)
            fh.reverse_timing_shift(treated, self.pars.tx_timing_reversal)
        return


class fetal_infection(ss.Connector):
    """
    Connect a disease to fetal health outcomes during pregnancy.

    Monitors for infections in pregnant women and applies fetal damage:
    - **Timing shift**: brings delivery forward (increases preterm birth risk).
      Drawn from a lognormal distribution per affected pregnancy.
    - **Growth restriction**: reduces birth weight by a fixed fractional penalty.

    Damage is applied at two points:
    1. At conception, if the mother is already infected (via a conception callback
       registered with FetalHealth).
    2. During pregnancy, when a new infection occurs (detected in ``step()``
       by checking ``ti_infected == self.ti``).

    Requires ``ss.FetalHealth()`` in ``custom`` and an SIR disease in ``diseases``.

    Pars:
        timing_shift (Dist):    weeks to shift delivery forward per infection (default: lognormal mean=3, std=1)
        growth_penalty (float): fractional birth weight reduction per infection (default: 0.15 = 15%)

    To adapt for a different disease::

        class my_fetal_connector(fetal_infection):
            def init_pre(self, sim):
                # Register with FetalHealth, but check for your disease instead
                ...

            def _apply_damage(self, uids):
                # Custom damage logic, e.g. stage-dependent penalties
                fh = self.sim.custom['fetal_health']
                disease = self.sim.diseases.my_disease
                severe = disease.severe[uids]
                mild_uids   = uids[~severe]
                severe_uids = uids[severe]
                fh.apply_growth_restriction(mild_uids, 0.05)
                fh.apply_growth_restriction(severe_uids, 0.25)
                ...
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            timing_shift   = ss.lognorm_ex(mean=3.0, std=1.0),
            growth_penalty = 0.15,
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate that the required modules exist
        if 'fetal_health' not in sim.custom:
            raise ValueError('fetal_infection requires a FetalHealth module. Add ss.FetalHealth() to custom.')
        if 'sir' not in sim.diseases:
            raise ValueError('fetal_infection requires an SIR disease module.')

        # Register a callback so we can apply damage at conception for
        # women who are already infected when they become pregnant
        fh = sim.custom['fetal_health']
        fh.add_conception_callback(self._on_conception)
        return

    def _on_conception(self, uids):
        """ Called by FetalHealth when new pregnancies begin — damage if already infected """
        infected = self.sim.diseases.sir.infected[uids]
        infected_uids = uids[infected]
        if len(infected_uids):
            self._apply_damage(infected_uids)
        return

    def _apply_damage(self, uids):
        """
        Apply fetal damage to pregnancies of infected women.

        Override this method to customize the damage logic for a different
        disease (e.g. stage-dependent growth penalties, trimester-dependent
        timing shifts).
        """
        fh = self.sim.custom['fetal_health']
        shifts = self.pars.timing_shift.rvs(uids)
        fh.apply_timing_shift(uids, shifts)
        fh.apply_growth_restriction(uids, self.pars.growth_penalty)
        return

    def step(self):
        """ Each timestep, check for new infections in pregnant women and apply damage """
        sim  = self.sim
        preg = sim.people.pregnancy
        if not preg.pregnant.any():
            return

        # Find pregnant women newly infected this timestep
        pregnant_uids  = preg.pregnant.uids
        newly_infected = sim.diseases.sir.ti_infected == self.ti
        affected = pregnant_uids[newly_infected[pregnant_uids]]
        if len(affected):
            self._apply_damage(affected)
        return
