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
    - `treat_pregnant` (Delivery): treats infected pregnant women, as `ss.treat()`
      delivers a treatment, and partially reverses fetal damage.

Both modules require `ssl.mnch.FetalHealth()` in the sim's `modules` and
`ss.Pregnancy()` in demographics. The connector also requires the target
disease (default: an `ssl.AgentSIR` named "sir") to be present.

Usage::

    import starsim as ss
    import starsim.library as ssl

    sim = ss.Sim(
        diseases=ssl.AgentSIR(name='sir', beta=0.1), # Written as code, since treat_pregnant() sets its states directly
        demographics=[ss.Pregnancy(fertility_rate=ss.peryear(0.03)), ss.Deaths()],
        connectors=ssl.mnch.fetal_infection(),
        interventions=ssl.mnch.treat_pregnant(disease='sir', when=ss.schedule(start=2025)),
        modules=ssl.mnch.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

To extend for a different disease, subclass `fetal_infection` and override
`_apply_damage` with disease-specific logic (e.g. stage-dependent penalties).
To extend `treat_pregnant`, give `who=` for other eligibility criteria, or
subclass `PregnancyTreatment` and override `administer()` for other reversal logic.

Used with: agents; coded modules
"""

import numpy as np
import starsim as ss

__all__ = ['FetalHealth', 'treat_pregnant', 'PregnancyTreatment', 'fetal_infection']


class FetalHealth(ss.Module):
    """
    Track fetal health outcomes during pregnancy.

    Works alongside the Pregnancy module to model birth weight outcomes (low
    birth weight, small for gestational age) based on fetal growth restriction.
    Disease-agnostic by design: external modules (connectors, interventions)
    modify fetal health via the public API methods.

    Preterm classification is handled by the Pregnancy module (based on
    gestational age at birth). This module focuses on the weight/growth axis.

    Integrates with Pregnancy via its callbacks (`add_callback()`): Pregnancy
    calls `on_conception` when new pregnancies begin and `on_delivery` when
    births occur. External modules (e.g. disease connectors) can register their
    own on `preg.conception`, after this module's, to act on new pregnancies
    after baseline initialization.

    During pregnancy, `weight_percentile`, `growth_restriction`,
    `timing_shift`, and `n_exposures` are tracked on the mother. At
    delivery, `birth_weight`, `lbw`, `vlbw`, and `sga` are stored
    on the newborn.

    Each pregnancy gets a baseline weight percentile drawn at conception.
    Two modification levers are available:

        1. **Delivery timing**: bring the delivery (`ti_exit_P`) forward (preterm birth risk)
        2. **Growth restriction**: accumulate fractional weight reduction

    At delivery: `birth_weight = baseline_for_GA × percentile × (1 - restriction)`

    The results are counts of births and outcomes on each step, and means and rates over a step's births
    (`mean_birth_weight`, `lbw_rate`, ...), which are NaN on a step with no births, and are left out of a
    mean over time (e.g. `res.mean_birth_weight.resample('year')`).

    Note: CRN safe.

    Args:
        weight_by_ga (array):           Nx2 array of [gestational_age_weeks, weight_grams]
        interp_fn (callable):           interpolation function with signature (x, xp, fp) -> array (default np.interp)
        sga_ratio (float):              fraction of GA-appropriate weight below which SGA is declared
        lbw_threshold (float):          birth weight in grams below which LBW is declared
        vlbw_threshold (float):         birth weight in grams below which VLBW is declared
        min_ga (dur):                   floor for timing shifts (delivery can't be brought before this GA)
        percentile_dist (Dist):         distribution for baseline fetal weight percentile
        pars (dict): the same parameters, as a dict

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

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
    means = ['mean_birth_weight', 'mean_ga_at_birth', 'mean_exposures', 'lbw_rate', 'sga_rate', 'svn_rate'] # The results over the births of a step, which are NaN on a step with none

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='fetal_health')
        self.define_pars(
            weight_by_ga=self.default_weight_by_ga,
            interp_fn=np.interp,  # Interpolation function for mapping gestational age to reference birth weight
            sga_ratio=0.87,  # SGA if birth_weight < baseline_for_ga * sga_ratio; ~10% baseline rate given percentile_dist=N(1,0.1)
            lbw_threshold=2500,
            vlbw_threshold=1500,
            min_ga=ss.weeks(24),  # Earliest possible delivery GA; timing shifts cannot bring delivery before this
            percentile_dist=ss.normal(mean=1.0, std=0.1),
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
        return

    def init_pre(self, sim):
        """ Register callbacks with the Pregnancy module

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)
        preg = sim.get_module(ss.Pregnancy, die=False) # By type, so a subclass with its own name (e.g. FPsim's fp) is found
        if preg is None:
            raise ValueError('FetalHealth requires a Pregnancy module. Add ss.Pregnancy() to demographics.')
        preg.add_callback(preg.conception, self.on_conception)
        preg.add_callback(preg.birth, self.on_delivery)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('birth',          dtype=int, label='Births', kind='flow'),
            ss.Result('lbw',             dtype=int, label='Low birth weight', kind='flow'),
            ss.Result('vlbw',            dtype=int, label='Very low birth weight', kind='flow'),
            ss.Result('sga',             dtype=int, label='Small for gestational age', kind='flow'),
            ss.Result('svn',             dtype=int, label='Small vulnerable newborns', kind='flow'),
            ss.Result('mean_birth_weight', kind='ratio', label='Mean birth weight (g)'),
            ss.Result('mean_ga_at_birth',  label='Mean GA at birth (weeks)', kind='ratio'),
            ss.Result('mean_exposures',    label='Mean exposures per pregnancy', kind='ratio'),
            ss.Result('lbw_rate',          label='LBW rate', kind='ratio'),
            ss.Result('sga_rate',          label='SGA rate', kind='ratio'),
            ss.Result('svn_rate',          label='SVN rate', kind='ratio'),
        )
        for key in self.means:
            self.results[key][:] = np.nan # Undefined on a step with no births, so that it is left out of a mean over time
        return

    def on_conception(self, uids):
        """ Initialize fetal health for new pregnancies (called by Pregnancy)

        Args:
            uids (array): the women who conceived
        """
        self.weight_percentile[uids] = self.pars.percentile_dist.rvs(uids)
        self.growth_restriction[uids] = 0.0
        self.timing_shift[uids] = 0.0
        self.n_exposures[uids] = 0
        return

    def on_delivery(self, newborn_uids):
        """
        Classify birth outcomes (called by Pregnancy).

        Birth weight is computed from the mother's pregnancy-time states
        (weight_percentile, growth_restriction). Outcomes (birth_weight,
        lbw, vlbw, sga) are stored on the newborn agents.

        Args:
            newborn_uids (array): the newborns
        """
        if not len(newborn_uids):
            return

        # Compute birth weight using mother states (handles twins via parent lookup)
        parents = self.sim.people.parent[newborn_uids]
        unset = np.unique(parents[np.isnan(self.weight_percentile[parents])])
        if len(unset): # Pregnant at the start (ss.Pregnancy's burnin=), so on_conception() was not called for them
            self.on_conception(ss.uids(unset))
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
        preterm = self.sim.get_module(ss.Pregnancy).preterm[newborn_uids]
        svn = preterm | lbw | sga
        self.svn[newborn_uids] = svn

        # Results: the means are over the births of the step, which may come in more than one call
        res = self.results
        ti = self.ti
        n, before = len(newborn_uids), res['birth'][ti]
        values = dict(mean_birth_weight=birth_weights, mean_ga_at_birth=ga_wk, mean_exposures=self.n_exposures[parents], lbw_rate=lbw, sga_rate=sga, svn_rate=svn)
        for key in self.means:
            total = values[key].sum() + (res[key][ti]*before if before else 0)
            res[key][ti] = total/(before + n)
        res['birth'][ti] += n
        res['lbw'][ti]    += lbw.sum()
        res['vlbw'][ti]   += vlbw.sum()
        res['sga'][ti]    += sga.sum()
        res['svn'][ti]    += svn.sum()
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

        preg = self.sim.get_module(ss.Pregnancy)
        weeks_per_ts = self.dt.weeks
        shifts_ts = shift_weeks / weeks_per_ts

        min_ga_ts = self.pars.min_ga.weeks / weeks_per_ts
        current_delivery = np.ceil(preg.ti_exit_P[uids]) # The step each delivery is due in
        new_delivery = current_delivery - shifts_ts
        min_delivery = preg.ti_P[uids] + min_ga_ts
        new_delivery = np.maximum(new_delivery, min_delivery)

        actually_shifted_ts = np.maximum(0, current_delivery - new_delivery)
        preg.set_delivery(uids, np.minimum(current_delivery, new_delivery))
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

        preg = self.sim.get_module(ss.Pregnancy)
        current_shift = self.timing_shift[uids]
        recover_weeks = current_shift * fraction

        has_shift = recover_weeks > 0
        if not has_shift.any():
            return

        recover_uids = uids[has_shift]
        recover_wk   = recover_weeks[has_shift]
        recover_ts   = recover_wk / self.dt.weeks

        current_delivery = np.ceil(preg.ti_exit_P[recover_uids])
        preg.set_delivery(recover_uids, current_delivery + recover_ts)
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

        Args:
            uids (array): the mother of each newborn

        Returns:
            tuple: (birth_weights, ga_weeks) arrays
        """
        preg = self.sim.get_module(ss.Pregnancy)
        ga_wk = (preg.ti_delivered[uids] - preg.ti_P[uids]) * self.dt.weeks # At delivery, which has just happened

        ref = self.pars.weight_by_ga
        baseline    = self.pars.interp_fn(ga_wk, ref[:, 0], ref[:, 1])
        percentile  = self.weight_percentile[uids]
        restriction = self.growth_restriction[uids]

        return baseline * percentile * (1 - restriction), ga_wk

    def step(self):
        pass  # All logic is driven by Pregnancy callbacks


class treat_pregnant(ss.Delivery):
    """
    Treat infected pregnant women and partially reverse fetal damage.

    A delivery, as `ss.treat()` is: by default each pregnant woman infected with the
    target disease is offered treatment once, with the probability `coverage`. Treatment
    cures the infection, and reverses a share of the fetal damage accumulated so far
    (growth restriction and timing shift). `who=` restricts who among them is offered it,
    and `when=` when, as for any delivery (e.g. `when=ss.schedule(start=2025)`).

    Note: CRN safe.

    Args:
        disease (str):           name of the target disease (default: 'sir'), written as code (e.g. `ssl.AgentSIR`)
        coverage (float):        the probability that each woman eligible is treated
        who (Cond/func):         who among the infected pregnant women is offered it
        when (various):          when it is offered, as for any delivery (default: on every step, once to each woman when she becomes eligible)
        growth_reversal (float): the share of the growth restriction reversed (0-1)
        timing_reversal (float): the share of the timing shift reversed (0-1)
        kwargs (dict):           passed to `ss.Delivery` (e.g. `capacity=`, `name=`)

    Attributes:
        ti_treated (FloatArr):  timestep when each agent was treated

    Example — treatment from 2025, of half the women eligible::

        ssl.mnch.treat_pregnant(disease='sir', start=2025, coverage=0.5)
    """

    def __init__(self, disease='sir', coverage=0.9, who=None, when=None, growth_reversal=0.7, timing_reversal=0.7, name=None, **kwargs):
        ss.v3_deprecations.check(type(self), '__init__', kwargs)
        self.disease_name = disease
        self.growth_reversal = growth_reversal
        self.timing_reversal = timing_reversal
        eligible = self.eligible if who is None else ss.conditions.Cond('&', who, self.eligible)
        super().__init__(PregnancyTreatment(self, name=f'{name or "treat_pregnant"}_drug'), who=eligible, when=when, coverage=coverage, disease=disease, name=name, **kwargs)
        self.define_states(
            ss.FloatArr('ti_treated', label='Time of treatment'),
        )
        return

    def init_pre(self, sim):
        """ Check that the sim has the disease and `ssl.mnch.FetalHealth`

        Args:
            sim (Sim): the sim being initialized
        """
        if self.disease_name not in sim.diseases:
            raise ValueError(f'treat_pregnant requires disease "{self.disease_name}" but it was not found in the sim.')
        if 'fetal_health' not in sim.custom:
            raise ValueError('treat_pregnant requires a FetalHealth module. Add it with modules=ssl.mnch.FetalHealth().')
        super().init_pre(sim)
        return

    def eligible(self, sim):
        """ The pregnant women infected with the disease """
        preg = sim.get_module(ss.Pregnancy)
        return preg.P & sim.diseases[self.disease_name].infected


class PregnancyTreatment(ss.Product):
    """ What `treat_pregnant` delivers: a cure of the infection, and a partial reversal of the fetal damage """
    def __init__(self, delivery, **kwargs):
        super().__init__(**kwargs)
        self.delivery = delivery
        return

    def administer(self, uids):
        tx      = self.delivery
        sim     = self.sim
        disease = sim.diseases[tx.disease_name]
        fh      = sim.custom['fetal_health']
        uids    = uids[disease.infected[uids]]

        # Cure infection
        disease.infected[uids]  = False
        disease.recovered[uids] = True
        disease.ti_dead[uids]   = np.nan # Cancel any scheduled death from the disease
        tx.ti_treated[uids]     = sim.ti

        # Partially reverse fetal damage from the infection
        fh.reverse_growth_restriction(uids, fh.growth_restriction[uids] * tx.growth_reversal)
        fh.reverse_timing_shift(uids, tx.timing_reversal)
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
       registered with the pregnancy, after FetalHealth's).
    2. During pregnancy, when a new infection occurs (detected in `step()`
       by checking `ti_infected == self.ti - 1`, since transmission happens after connectors).

    Requires `ssl.mnch.FetalHealth()` in `modules` and an SIR disease written as code (`ssl.AgentSIR(name="sir")`) in `diseases`.

    Note: CRN safe.

    Args:
        timing_shift (Dist):    weeks to shift delivery forward per infection (default: lognormal mean=3, std=1)
        growth_penalty (float): fractional birth weight reduction per infection (default: 0.15 = 15%)
        pars (dict): the same parameters, as a dict

    To adapt for a different disease::

        class my_fetal_connector(fetal_infection):
            def init_pre(self, sim):
                # Register with FetalHealth, but check for your disease instead
                ...

            def _apply_damage(self, uids):
                # Custom damage logic, e.g. stage-dependent penalties
                fh = self.fh
                disease = self.sim.diseases.my_disease
                severe = disease.severe[uids]
                mild_uids   = uids[~severe]
                severe_uids = uids[severe]
                fh.apply_growth_restriction(mild_uids, 0.05)
                fh.apply_growth_restriction(severe_uids, 0.25)
                ...
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            timing_shift   = ss.lognormal(mean=3.0, std=1.0),
            growth_penalty = 0.15,
        )
        self.update_pars(pars, **kwargs)
        return

    @property
    def fh(self):
        """ Shortcut to the FetalHealth module """
        return self.sim.custom['fetal_health']

    def init_pre(self, sim):
        """ Check that the sim has what the connector needs

        Args:
            sim (Sim): the sim being initialized
        """
        super().init_pre(sim)

        # Validate that the required modules exist
        if 'fetal_health' not in sim.custom:
            raise ValueError('fetal_infection requires a FetalHealth module. Add it with modules=ssl.mnch.FetalHealth().')
        if 'sir' not in sim.diseases:
            raise ValueError('fetal_infection requires an SIR disease module.')
        return

    def init_post(self):
        """ Register a callback so we can apply damage at conception for women who are already infected when they become pregnant: after FetalHealth's (registered in its init_pre()), which sets the baseline """
        super().init_post()
        preg = self.sim.get_module(ss.Pregnancy)
        preg.add_callback(preg.conception, self._on_conception)
        return

    def _on_conception(self, uids):
        """ Called by Pregnancy when new pregnancies begin, after FetalHealth's baseline — damage if already infected """
        infected = self.sim.diseases.sir.infected[uids]
        infected_uids = uids[infected]
        if len(infected_uids):
            self.fh.n_exposures[infected_uids] += 1
            self._apply_damage(infected_uids)
        return

    def _apply_damage(self, uids):
        """
        Apply fetal damage to pregnancies of infected women.

        Override this method to customize the damage logic for a different
        disease (e.g. stage-dependent growth penalties, trimester-dependent
        timing shifts).
        """
        fh = self.fh
        shifts = self.pars.timing_shift.rvs(uids)
        fh.apply_timing_shift(uids, shifts)
        fh.apply_growth_restriction(uids, self.pars.growth_penalty)
        return

    def step(self):
        """ Each timestep, check for new infections in pregnant women and apply damage """
        sim  = self.sim
        preg = sim.get_module(ss.Pregnancy)
        if not preg.P.any():
            return

        # Find pregnant women infected on the previous timestep (since transmission happens after connectors), excluding new pregnancies (handled at conception)
        pregnant_uids  = preg.P.uids
        newly_infected = (sim.diseases.sir.ti_infected == self.ti - 1) & (preg.ti_P < self.ti)
        affected = pregnant_uids[newly_infected[pregnant_uids]]
        if len(affected):
            self.fh.n_exposures[affected] += 1
            self._apply_damage(affected)
        return
