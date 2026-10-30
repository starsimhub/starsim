"""
Health economics: the costs of a sim, discounting, and incremental cost-effectiveness ratios.

Costs are not part of every sim, so these live in the library rather than in core
Starsim: `ssl.Costs` adds a sim's costs up as results, `ssl.discount()` discounts a
result over time, and `ssl.icer()` compares the costs and effects of scenarios.
"""
from .costs import Costs, discount, icer
