"""
Fetal health connector and treatment intervention examples.

These modules demonstrate how to connect a disease to fetal health outcomes
and how to implement treatment that reverses fetal damage. They are designed
to be extended for real applications.

Architecture:
    - `fetal_infection` (Connector): watches for infections in pregnant women
      and applies fetal damage (preterm risk via timing shifts, low birth weight
      via growth restriction). Damage is applied both at conception (if already
      infected) and during pregnancy (if newly infected).
    - `treat_pregnant` (Intervention): treats infected pregnant women within a
      specified year range and partially reverses fetal damage.

Both modules require `ssl.mnch.FetalHealth()` in the sim's `custom` modules and
`ss.Pregnancy()` in demographics. The connector also requires the target
disease (default: SIR) to be present.

Usage::

    import starsim as ss
    import starsim.library as ssl

    sim = ss.Sim(
        diseases=ss.SIR(beta=0.1),
        demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30)), ss.Deaths()],
        connectors=ssl.mnch.fetal_infection(),
        interventions=ssl.mnch.treat_pregnant(disease='sir', start_year=2025),
        custom=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

To extend for a different disease, subclass `fetal_infection` and override
`_apply_damage` with disease-specific logic (e.g. stage-dependent penalties).
To extend `treat_pregnant`, subclass and override `step()` with custom
eligibility criteria or reversal logic.
"""

import numpy as np
import starsim as ss


class FetalHealth(ss.Module):
    """
    Track fetal health outcomes during pregnancy.

    Works alongside the Pregnancy module to model birth weight outcomes (low
    birth weight, small for gestational age) based on fetal growth restriction.
    Disease-agnostic by design: external modules (connectors, interventions)
    modify fetal health via the public API methods.

    Preterm classification is handled by the Pregnancy module (based on
    gestational age at birth). This module focuses on the weight/growth axis.

    Integrates with Pregnancy via callbacks: Pregnancy calls `on_conception`
    when new pregnancies begin and `on_delivery` when births occur. External
    modules (e.g. disease connectors) can register their own callbacks via
    `add_conception_callback` to act on new pregnancies after baseline
    initialization.

    During pregnancy, `weight_percentile`, `growth_restriction`,
    `timing_shift`, and `n_exposures` are tracked on the mother. At
    delivery, `birth_weight`, `lbw`, `vlbw`, and `sga` are stored
    on the newborn.

    Each pregnancy gets a baseline weight percentile drawn at conception.
    Two modification levers are available:

        1. **Delivery timing**: bring `ti_delivery` forward (preterm birth risk)
        2. **Growth restriction**: accumulate fractional weight reduction

    At delivery: `birth_weight = baseline_for_GA × percentile × (1 - restriction)`

    Args:
        weight_by_ga (array):           Nx2 array of [gestational_age_weeks, weight_grams]
        interp_fn (callable):           interpolation function with signature (x, xp, fp) -> array (default np.interp)
        sga_ratio (float):              fraction of GA-appropriate weight below which SGA is declared
        lbw_threshold (float):          birth weight in grams below which LBW is declared
        vlbw_threshold (float):         birth weight in grams below which VLBW is declared
        min_ga (dur):                   floor for timing shifts (delivery can't be brought before this GA)
        percentile_dist (Dist):         distribution for baseline fetal weight percentile

    Examples:
        ```python
        import starsim as ss

        sim = ss.Sim(
            demographics=[ss.Pregnancy(fertility_rate=10), ss.Deaths(death_rate=10)],
            modules=ssl.mnch.FetalHealth(),
            networks=ss.PrenatalNet(),
        )
        sim.run()
        ```
    """

    # Approximate 50th percentile fetal weight (grams) by gestational age (weeks)
    # Source: Hadlock 1991 / INTERGROWTH-21st
    default_weight_by_ga = np.array([
        [24, 600],  [25, 700],  [26, 800],  [27, 900],  [28, 1000],
        [29, 1150], [30, 1300], [31, 1500], [32, 1700], [33, 1900],
        [34, 2100], [35, 2400], [36, 2600], [37, 2850], [38, 3050],
        [39, 3250], [40, 3400], [41, 3500], [42, 3550],
    ], dtype=float)

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='fetal_health')
        self.define_pars(
            weight_by_ga=self.default_weight_by_ga,
            interp_fn=np.interp,  # Interpolation function for mapping gestational age to reference birth weight
            sga_ratio=0.87,  # SGA if birth_weight < baseline_for_ga * sga_ratio; ~10% baseline rate given percentile_dist=N(1,0.1)
            lbw_threshold=2500,
            vlbw_threshold=1500,
            min_ga=ss.weeks(24),  # Earliest possible delivery GA; timing shifts cannot bring delivery before this
            percentile_dist=ss.normal(loc=1.0, scale=0.1),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            # Pregnancy-time tracking (stored on mothers)
            ss.FloatArr('weight_percentile', label='Fetal weight percentile'),
            ss.FloatArr('growth_restriction', label='Cumulative growth restriction'),
            ss.FloatArr('timing_shift',      label='Accumulated delivery shift (weeks)'),
            ss.FloatArr('n_exposures',       label='Disease exposures during pregnancy'),

            # Birth outcomes (stored on newborns at delivery)
            ss.FloatArr('birth_weight', label='Birth weight (grams)'),
            ss.BoolArr('lbw',  label='Low birth weight'),
            ss.BoolArr('vlbw', label='Very low birth weight'),
            ss.BoolArr('sga',  label='Small for gestational age'),
            ss.BoolArr('svn',  label='Small vulnerable newborn'),
        )

        self._conception_callbacks = []
        return

    def init_pre(self, sim):
        """ Register callbacks with the Pregnancy module """
        super().init_pre(sim)
        if not hasattr(sim.demographics, 'pregnancy'):
            raise ValueError('FetalHealth requires a Pregnancy module. Add ss.Pregnancy() to demographics.')
        preg = sim.demographics.pregnancy
        preg.add_conception_callback(self.on_conception)
        preg.add_delivery_callback(self.on_delivery)
        return

    def add_conception_callback(self, fn):
        """
        Register a function to be called when new pregnancies are detected.
        The function receives `(uids,)` after baseline initialization.
        """
        self._conception_callbacks.append(fn)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_births',          dtype=int, label='Births'),
            ss.Result('n_lbw',             dtype=int, label='Low birth weight'),
            ss.Result('n_vlbw',            dtype=int, label='Very low birth weight'),
            ss.Result('n_sga',             dtype=int, label='Small for gestational age'),
            ss.Result('n_svn',             dtype=int, label='Small vulnerable newborns'),
            ss.Result('mean_birth_weight', scale=False, label='Mean birth weight (g)'),
            ss.Result('mean_ga_at_birth',  scale=False, label='Mean GA at birth (weeks)'),
            ss.Result('mean_exposures',    scale=False, label='Mean exposures per pregnancy'),
            ss.Result('lbw_rate',          scale=False, label='LBW rate'),
            ss.Result('sga_rate',          scale=False, label='SGA rate'),
            ss.Result('svn_rate',          scale=False, label='SVN rate'),
        )
        return

    def on_conception(self, uids):
        """ Initialize fetal health for new pregnancies (called by Pregnancy) """
        self.weight_percentile[uids] = self.pars.percentile_dist.rvs(uids)
        self.growth_restriction[uids] = 0.0
        self.timing_shift[uids] = 0.0
        self.n_exposures[uids] = 0
        for cb in self._conception_callbacks:
            cb(uids)
        return

    def on_delivery(self, mother_uids, newborn_uids):
        """
        Classify birth outcomes (called by Pregnancy).

        Birth weight is computed from the mother's pregnancy-time states
        (weight_percentile, growth_restriction). Outcomes (birth_weight,
        lbw, vlbw, sga) are stored on the newborn agents.
        """
        if not len(newborn_uids):
            return

        # Compute birth weight using mother states (handles twins via parent lookup)
        parents = self.sim.people.parent[newborn_uids]
        birth_weights, ga_wk = self.compute_birth_weight(parents)

        # Store outcomes on newborns
        self.birth_weight[newborn_uids] = birth_weights

        ref = self.pars.weight_by_ga
        sga_threshold = self.pars.interp_fn(ga_wk, ref[:, 0], ref[:, 1]) * self.pars.sga_ratio
        lbw  = birth_weights < self.pars.lbw_threshold
        vlbw = birth_weights < self.pars.vlbw_threshold
        sga  = birth_weights < sga_threshold

        self.lbw[newborn_uids]  = lbw
        self.vlbw[newborn_uids] = vlbw
        self.sga[newborn_uids]  = sga

        # SVN: small vulnerable newborn = preterm | lbw | sga
        preterm = self.sim.demographics.pregnancy.preterm[newborn_uids]
        svn = preterm | lbw | sga
        self.svn[newborn_uids] = svn

        # Results
        n  = len(newborn_uids)
        ti = self.ti
        self.results['n_births'][ti]          += n
        self.results['n_lbw'][ti]             += lbw.sum()
        self.results['n_vlbw'][ti]            += vlbw.sum()
        self.results['n_sga'][ti]             += sga.sum()
        self.results['n_svn'][ti]             += svn.sum()
        self.results['mean_birth_weight'][ti]  = birth_weights.mean() if n else 0
        self.results['mean_ga_at_birth'][ti]   = ga_wk.mean() if n else 0
        self.results['mean_exposures'][ti]     = self.n_exposures[parents].mean() if n else 0
        self.results['lbw_rate'][ti]           = lbw.mean() if n else 0
        self.results['sga_rate'][ti]           = sga.mean() if n else 0
        self.results['svn_rate'][ti]           = svn.mean() if n else 0
        return

    def apply_timing_shift(self, uids, shift_weeks):
        """
        Bring delivery forward for pregnant women.

        Uses a one-way ratchet: delivery can only be brought forward, never
        pushed back. The actual shift applied is tracked in `timing_shift`.

        Args:
            uids: UIDs of pregnant women
            shift_weeks (float/array): shift in weeks; positive = earlier delivery
        """
        if not len(uids):
            return

        preg = self.sim.people.pregnancy
        weeks_per_ts = self.dt.weeks
        shifts_ts = shift_weeks / weeks_per_ts

        min_ga_ts = self.pars.min_ga.weeks / weeks_per_ts
        current_delivery = preg.ti_delivery[uids]
        new_delivery = current_delivery - shifts_ts
        min_delivery = preg.ti_pregnant[uids] + min_ga_ts
        new_delivery = np.maximum(new_delivery, min_delivery)

        actually_shifted_ts = np.maximum(0, current_delivery - new_delivery)
        preg.ti_delivery[uids] = np.minimum(current_delivery, new_delivery)
        self.timing_shift[uids] += actually_shifted_ts * weeks_per_ts
        return

    def apply_growth_restriction(self, uids, penalty):
        """
        Apply fractional growth restriction (cumulative, diminishing).

        Positive penalties use diminishing returns: `current + (1-current) * penalty`.
        Negative penalties (growth boost, e.g. GDM macrosomia) are additive.

        Args:
            uids: UIDs of pregnant women
            penalty (float): fractional weight reduction; negative = growth boost
        """
        if not len(uids):
            return

        current = self.growth_restriction[uids]
        positive = penalty >= 0
        new_val = np.where(positive, current + (1 - current) * penalty, current + penalty)
        self.growth_restriction[uids] = new_val
        return

    def reverse_timing_shift(self, uids, fraction):
        """
        Recover a fraction of the accumulated delivery timing shift.

        Args:
            uids: UIDs of pregnant women
            fraction (float/array): fraction to recover (0-1)
        """
        if not len(uids):
            return

        preg = self.sim.people.pregnancy
        current_shift = self.timing_shift[uids]
        recover_weeks = current_shift * fraction

        has_shift = recover_weeks > 0
        if not has_shift.any():
            return

        recover_uids = uids[has_shift]
        recover_wk   = recover_weeks[has_shift]
        recover_ts   = recover_wk / self.dt.weeks

        current_delivery = preg.ti_delivery[recover_uids]
        preg.ti_delivery[recover_uids] = current_delivery + recover_ts
        self.timing_shift[recover_uids] -= recover_wk
        return

    def reverse_growth_restriction(self, uids, amount):
        """
        Reverse a specific amount of growth restriction.

        Args:
            uids: UIDs of pregnant women
            amount (float/array): amount to reverse
        """
        if not len(uids):
            return

        current = self.growth_restriction[uids]
        self.growth_restriction[uids] = np.maximum(0, current - amount)
        return

    def compute_birth_weight(self, uids):
        """
        Compute birth weight at delivery.

        Override this method to customize the birth weight formula. The
        interpolation function can be swapped via `pars.interp_fn`.

        Returns:
            tuple: (birth_weights, ga_weeks) arrays
        """
        preg = self.sim.people.pregnancy
        ga_wk = (preg.ti_delivery[uids] - preg.ti_pregnant[uids]) * self.dt.weeks

        ref = self.pars.weight_by_ga
        baseline    = self.pars.interp_fn(ga_wk, ref[:, 0], ref[:, 1])
        percentile  = self.weight_percentile[uids]
        restriction = self.growth_restriction[uids]

        return baseline * percentile * (1 - restriction), ga_wk

    def step(self):
        pass  # All logic is driven by Pregnancy callbacks


class treat_pregnant(ss.Intervention):
    """
    Treat infected pregnant women and partially reverse fetal damage.

    Each timestep, identifies pregnant women infected with the target disease,
    treats a fraction of them (curing infection), and reverses a portion of
    the accumulated fetal damage (growth restriction and timing shift).

    Treatment only applies between `start_year` and `end_year`. If not
    specified, defaults to the full simulation period.

    Args:
        disease (str):              name of the target disease (default: 'sir')
        start_year (float):         first year treatment is available (default: sim start)
        end_year (float):           last year treatment is available (default: sim end)
        p_treat (Dist):             par: probability of treating an eligible woman per timestep
        tx_growth_reversal (float): par: fraction of growth restriction to reverse (0-1)
        tx_timing_reversal (float): par: fraction of timing shift to reverse (0-1)

    Attributes:
        ti_treated (FloatArr):  timestep when each agent was treated

    Example — treatment starting in 2025 with 50% coverage::

        ssl.mnch.treat_pregnant(disease='sir', start_year=2025, p_treat=ss.bernoulli(p=0.5))
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
            raise ValueError(f'treat_pregnant requires disease "{self.disease_name}" but it was not found in the sim.')
        if 'fetal_health' not in sim.custom:
            raise ValueError('treat_pregnant requires a FetalHealth module. Add ssl.mnch.FetalHealth() to custom.')

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
    2. During pregnancy, when a new infection occurs (detected in `step()`
       by checking `ti_infected == self.ti`).

    Requires `ssl.mnch.FetalHealth()` in `custom` and an SIR disease in `diseases`.

    Args:
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
            raise ValueError('fetal_infection requires a FetalHealth module. Add ssl.mnch.FetalHealth() to custom.')
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
