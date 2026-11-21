"""
Health economics: the costs of a sim, disability-adjusted life years, discounting, and incremental cost-effectiveness ratios.

Used with: agents and compartments; declared and coded modules

Costs are not part of every sim, so these live in the library rather than in core
Starsim: `ssl.Costs` adds a sim's costs up as results, `ssl.DALYs` does the same for
disability-adjusted life years, `ssl.discount()` discounts a result over time, and
`ssl.ICER` compares the costs and effects of scenarios, after they have run or as a sim runs.
"""
from .costs import Costs, discount
from .dalys import DALYs, ICER
