"""
Example maternal infections with congenital outcomes.

Demonstrates the generic congenital outcome framework in the base Infection
class. CongenitalDisease is a simple SIR that assigns birth outcomes
(stillborn, congenital infection, normal) to infected mothers at delivery.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['CongenitalDisease']


class CongenitalDisease(ss.SIR):
    """
    Simple disease with congenital outcomes via the generic framework.

    Infected mothers transmit to their unborn via PrenatalNet. At transmission,
    the base set_congenital() samples an outcome (stillborn, congenital, normal)
    and schedules it for delivery time. fire_congenital_outcomes() in step_state()
    executes those scheduled events.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            birth_outcome_keys = ['stillborn', 'congenital', 'normal'],
            birth_outcomes     = sc.objdict(
                default=ss.choice(a=3, p=np.array([0.3, 0.4, 0.3])),
            ),
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('ti_stillborn'),
            ss.FloatArr('ti_congenital'),
            ss.FloatArr('ti_normal'),
            ss.BoolArr('congenital'),
            ss.FloatArr('cs_outcome'),
        )
        return

    def step_state(self):
        super().step_state()
        self.fire_congenital_outcomes()
        return
