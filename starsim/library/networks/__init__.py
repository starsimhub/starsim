"""
Additional network types: household, cluster, sexual, partnership targets, spatial, and theoretical.

`HouseholdNet` builds households from DHS-style survey data; `ClusterNet` divides
agents into fully connected clusters, and `HybridNet` is Covasim's population of
households, schools, workplaces, and community; `MSMNet` pairs men; `DegreeNet` has
a target number of partners at a time, as EpiModel's networks do;
`DiskNet` connects agents that are spatially close; `ErdosRenyiNet` and `NullNet` are
useful for theoretical work and debugging, `StaticNet` turns a NetworkX graph into a
network, and `RandomSafeNet` is a random network that is random-number safe. Core
network classes (`ss.RandomNet`, `ss.MFNet`, `ss.PartnerNet`, etc.) live in
`starsim.networks`.
"""
from .household   import HouseholdNet
from .clusters    import ClusterNet, HybridNet
from .sexual      import MSMNet
from .degree      import DegreeNet
from .spatial     import DiskNet
from .theoretical import ErdosRenyiNet, NullNet, RandomSafeNet, StaticNet
