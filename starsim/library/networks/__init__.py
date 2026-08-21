"""
Additional network types: household, spatial, and theoretical.

`HouseholdNet` builds households from DHS-style survey data; `DiskNet` connects
agents that are spatially close; `ErdosRenyiNet` and `NullNet` are useful for
theoretical work and debugging. Core network classes (`ss.RandomNet`,
`ss.MFNet`, `ss.StaticNet`, etc.) live in `starsim.networks`.
"""
from .household   import HouseholdNet
from .spatial     import DiskNet
from .theoretical import ErdosRenyiNet, NullNet
