"""
Additional network types: household, cluster, spatial, and theoretical.

`HouseholdNet` builds households from DHS-style survey data; `ClusterNet` divides
agents into fully connected clusters, and `HybridNet` is Covasim's population of
households, schools, workplaces, and community; `DiskNet` connects agents that are spatially close, `GeoNet` connects them across real places, and `GridNet` puts them on a lattice; `ErdosRenyiNet` and `NullNet` are
useful for theoretical work and debugging, `SmallWorldNet` and `ScaleFreeNet`
are the classic graphs, and `StaticNet` turns a NetworkX graph into a network. Core
network classes (`ss.RandomNet`, `ss.DegreeNet`, `ss.SexualNet`, etc.) live in
`starsim.networks`.

Most of these networks run as agents only. Those a compartmental run can approximate by a mean number of contacts are marked in their files.

Used with: agents and compartments; declared and coded modules
"""
from .household   import HouseholdNet
from .clusters    import ClusterNet, HybridNet
from .theoretical import ErdosRenyiNet, NullNet, StaticNet, SmallWorldNet, ScaleFreeNet
from .spatial     import DiskNet, GeoNet, GridNet
