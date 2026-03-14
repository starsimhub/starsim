"""
Example maternal infection with congenital outcomes.

Demonstrates the generic congenital outcome framework in the base ``Infection``
class. ``CongenitalDisease`` is a simple SIR that assigns birth outcomes
(stillborn, congenital infection, normal) to unborn agents infected via
``PrenatalNet``.

How the congenital framework works:
    1. Mother-to-child transmission happens via ``PrenatalNet`` — the base
       ``Infection.infect()`` calls ``set_congenital()`` when the target is
       an unborn agent (age < 0).
    2. ``set_congenital()`` samples an outcome from ``birth_outcomes`` and
       schedules it at delivery time by setting ``ti_<outcome>``.
    3. ``fire_congenital_outcomes()`` (called each timestep from ``step_state()``)
       checks whether any scheduled outcomes are due. Death outcomes
       ('stillborn', 'nnd', 'miscarriage') call ``request_death()``;
       non-lethal outcomes set a BoolArr state (e.g. ``congenital = True``).

To use the framework in your own disease:
    1. Define ``birth_outcome_keys`` and ``birth_outcomes`` in pars
    2. Define matching ``ti_<key>`` FloatArr states for each outcome
    3. Define BoolArr states for non-lethal outcomes (same name as the key)
    4. Optionally define ``cs_outcome`` FloatArr to store outcome indices
    5. Call ``self.fire_congenital_outcomes()`` from ``step_state()``

For diseases with state- or gestational-age-dependent outcome probabilities
(e.g. syphilis, where outcomes differ by infection stage), override
``_assign_congenital_outcomes()`` and provide multiple keyed distributions
in ``birth_outcomes``.

Usage::

    import starsim as ss
    import starsim_examples as sse

    sim = ss.Sim(
        diseases=sse.CongenitalDisease(beta=0.2, init_prev=0.2),
        demographics=[ss.Pregnancy(fertility_rate=ss.freqperyear(30), burnin=True), ss.Deaths()],
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

    # Check results
    print('Stillbirths:', sim.results.pregnancy.stillbirths.sum())
    print('Congenital infections:', sim.diseases.congenitaldisease.congenital.sum())
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['CongenitalDisease']


class CongenitalDisease(ss.SIR):
    """
    Simple disease with congenital outcomes via the generic framework.

    Infected mothers transmit to their unborn via PrenatalNet. At transmission,
    the base ``set_congenital()`` samples an outcome (stillborn, congenital,
    normal) and schedules it for delivery time. ``fire_congenital_outcomes()``
    in ``step_state()`` executes those scheduled events.

    Pars:
        birth_outcome_keys (list): outcome names — each needs a ``ti_<key>`` state
        birth_outcomes (objdict):  ``ss.choice`` distributions keyed by category;
            use 'default' for a single distribution applied to all infections

    States:
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
        self.fire_congenital_outcomes()  # Process any outcomes scheduled for this timestep
        return
