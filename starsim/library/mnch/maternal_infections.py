"""
Example maternal infection with congenital outcomes.

`CongenitalDisease` is a simple SIR that assigns birth outcomes (stillborn,
congenital infection, normal) to unborn agents infected via `PrenatalNet`.

How it works:
    1. Mother-to-child transmission happens via `PrenatalNet`: `ss.Infection`
       calls `set_congenital()` when the target is an unborn agent (age <= 0),
       which by default does nothing.
    2. `CongenitalDisease.set_congenital()` samples an outcome from
       `birth_outcomes` and schedules it by setting `ti_<outcome>`. Most
       outcomes are scheduled for the delivery timestep, but fetal losses
       ('stillborn' and 'miscarriage') are scheduled one timestep earlier so
       that they occur before `Pregnancy` delivers the fetus -- otherwise they
       would be counted as a live birth followed by a neonatal death.
    3. `step_congenital()` (called each timestep from `step_state()`)
       checks whether any scheduled outcomes are due. Death outcomes
       ('stillborn', 'miscarriage', 'neonatal_deaths') call `request_death()`;
       non-lethal outcomes set a BoolArr state (e.g. `congenital = True`).

To use it in your own disease, subclass `CongenitalDisease` (or copy its
`set_congenital()` and `step_congenital()`), and:
    1. Define `birth_outcome_keys` and `birth_outcomes` in pars
    2. Define matching `ti_<key>` FloatArr states for each outcome
    3. Define BoolArr states for non-lethal outcomes (same name as the key)
    4. Optionally define `cs_outcome` FloatArr to store outcome indices

For diseases with state- or gestational-age-dependent outcome probabilities
(e.g. syphilis, where outcomes differ by infection stage), override
`_assign_congenital_outcomes()` and provide multiple keyed distributions
in `birth_outcomes`.

Usage::

    import starsim as ss
    import starsim.library as ssl

    sim = ss.Sim(
        diseases=ssl.mnch.CongenitalDisease(beta=0.2, init_prev=0.2),
        demographics=[ss.Pregnancy(fertility_rate=ss.time.freq(30, 'year'), burnin=True), ss.Deaths()],
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

    # Check results; note that stillbirths are recorded by the Pregnancy module
    print('Stillbirths:', sim.results.pregnancy.stillbirth.sum())
    print('Congenital infections:', sim.diseases.congenitaldisease.congenital.sum())
"""

import numpy as np
import sciris as sc
import starsim as ss
from ..diseases.agents import AgentSIR


class CongenitalDisease(AgentSIR):
    """
    Simple disease with congenital outcomes, scheduled around delivery

    Infected mothers transmit to their unborn via PrenatalNet. At transmission,
    `set_congenital()` samples an outcome (stillborn, congenital,
    normal) and schedules it for delivery time. `step_congenital()`
    in `step_state()` executes those scheduled events.

    Note: CRN safe.

    Args:
        birth_outcome_keys (list): outcome names — each needs a `ti_<key>` state
        birth_outcomes (objdict):  `ss.choice` distributions keyed by category;
            use 'default' for a single distribution applied to all infections

    Attributes:
        ti_stillborn (FloatArr):  timestep when stillbirth fires (triggers request_death)
        ti_congenital (FloatArr): timestep when congenital infection fires (sets BoolArr)
        ti_normal (FloatArr):     timestep when normal outcome fires (no effect)
        congenital (BoolArr):     True if agent has congenital infection
        cs_outcome (FloatArr):    index into birth_outcome_keys for each agent
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            # Keys define the possible outcomes; probabilities sum to 1
            birth_outcome_keys = ['stillborn', 'congenital', 'normal'],
            birth_outcomes     = sc.objdict(
                default=ss.choice(a=3, p=np.array([0.3, 0.4, 0.3])),
            ),
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('ti_stillborn'),   # Scheduled timestep for stillbirth
            ss.FloatArr('ti_congenital'),  # Scheduled timestep for congenital infection
            ss.FloatArr('ti_normal'),      # Scheduled timestep for normal outcome
            ss.BoolArr('congenital'),      # Whether agent has congenital infection
            ss.FloatArr('cs_outcome'),     # Outcome index (0=stillborn, 1=congenital, 2=normal)
        )
        return

    def step_state(self):
        super().step_state()
        self.step_congenital()  # Process any outcomes scheduled for this timestep
        return

    # Birth outcomes that trigger request_death rather than setting a bool state.
    # Prenatal outcomes are fetal losses and must fire while the agent is still unborn;
    # postnatal outcomes happen to a live newborn. See set_congenital() for why this matters.
    prenatal_death_keys = {'miscarriage', 'stillborn'}
    postnatal_death_keys = {'neonatal_deaths'}
    congenital_death_keys = prenatal_death_keys | postnatal_death_keys

    def set_congenital(self, target_uids, source_uids=None):
        """
        Assign congenital outcomes during in-utero infection (called when
        transmission occurs via PrenatalNet).

        Does nothing unless the disease defines `birth_outcome_keys` and
        `birth_outcomes` in its pars. Diseases that need fully custom logic
        (e.g. syphilis, which has stage-dependent outcomes) can override this
        method entirely.

        To use it, define in the disease's `__init__`:

            self.define_pars(
                birth_outcome_keys = ['stillborn', 'congenital', 'normal'],
                birth_outcomes     = sc.objdict(default=ss.choice(a=3, p=[0.3, 0.4, 0.3])),  # Illustrative placeholder
            )

        Each outcome name needs a matching `ti_<name>` FloatArr state; non-lethal
        outcomes also need a BoolArr of the same name. Death outcomes ('miscarriage',
        'neonatal_deaths', 'stillborn') fire via `request_death`; others set a bool state.

        Outcomes are scheduled for the mother's delivery timestep, except for fetal
        losses ('miscarriage', 'stillborn'), which are scheduled one timestep earlier so
        that they occur before the fetus is delivered and are classified as fetal losses
        rather than neonatal deaths. A fetus infected within one timestep of delivery is
        therefore born alive, and a lethal outcome for it counts as a neonatal death.

        For state- or GA-dependent probabilities, provide multiple keyed
        distributions in `birth_outcomes` and override
        `_assign_congenital_outcomes`.

        Call `step_congenital` from the disease's `step_state()` to
        execute the scheduled events each timestep.
        """
        if 'birth_outcomes' not in self.pars or self.pars.birth_outcomes is None:
            return

        # Prevent repeated MTC transmission each timestep of pregnancy
        self.susceptible[target_uids] = False

        # Assign outcomes
        outcomes = self.pars.birth_outcomes
        if len(outcomes) == 1 and 'default' in outcomes:
            assigned = outcomes['default'].rvs(target_uids)
        else:
            assigned = self._assign_congenital_outcomes(target_uids, source_uids)

        # Store outcome index if the disease tracks it
        if hasattr(self, 'cs_outcome'):
            self.cs_outcome[target_uids] = assigned

        # Schedule events at delivery time
        preg = self.sim.get_module(ss.Pregnancy)
        dt_delivery = np.ceil(preg.ti_exit_P.raw) - preg.engine.ti  # The steps until each pregnancy is due to end in delivery (dated by its engine: see ss.solvers.Engine.ti); full array, indexed by UID below
        for oi, key in enumerate(self.pars.birth_outcome_keys):
            o_uids = target_uids[assigned == oi]
            s_uids = source_uids[assigned == oi]
            if len(o_uids):
                ti_key = f'ti_{key}'
                if hasattr(self, ti_key):
                    ti_event = self.ti + dt_delivery[s_uids]
                    if key in self.prenatal_death_keys:
                        # Fetal losses must fire while the agent is still unborn. Pregnancy.step()
                        # runs before diseases' step_state() in the integration loop, so an event
                        # scheduled at the delivery would kill an already-delivered newborn: it would
                        # be counted as a live birth and then as a neonatal death, rather than as a
                        # fetal loss. Firing a timestep early also matches the biology, since a
                        # stillborn fetus dies in utero before labor.
                        ti_event = np.maximum(ti_event - 1, self.ti)
                    getattr(self, ti_key)[o_uids] = ti_event
        return

    def _assign_congenital_outcomes(self, target_uids, source_uids):
        """
        Override point for diseases with state- or GA-dependent outcome probabilities.
        Must return an integer array of outcome indices (one per target_uid),
        corresponding to `self.pars.birth_outcome_keys`.
        """
        errormsg = 'Subclass must implement _assign_congenital_outcomes or use a single "default" distribution in birth_outcomes'
        raise NotImplementedError(errormsg)

    def step_congenital(self):
        """
        Execute scheduled congenital events whose `ti_<key>` has arrived.

        Does nothing unless the disease defines `birth_outcome_keys` in its
        pars. Call from the disease's `step_state()`; see `set_congenital`
        for setup details.
        """
        if 'birth_outcome_keys' not in self.pars:
            return
        death_keys = self.congenital_death_keys
        for key in self.pars.birth_outcome_keys:
            ti_key = f'ti_{key}'
            if not hasattr(self, ti_key):
                continue
            vals = getattr(self, ti_key)
            due = (vals.notnan & (vals <= self.ti)).uids
            if len(due):
                if key in death_keys:
                    self.sim.people.request_death(due, cause=self)
                elif hasattr(self, key):
                    getattr(self, key)[due] = True
                vals[due] = np.nan  # Clear after firing
        return
