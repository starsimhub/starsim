"""
Solvers: how a module's declared states and transitions are run.

The same `ss.graph.Graph` can be run as compartments (`ode`, `binomial`, `gillespie`,
`tau`) or as agents (`abm`). Each module with a graph gets one engine, which is called from
the sim's loop like any other module code.

The modules: `engine` (what the engines share), `mixing` (who meets whom in a
compartmental run), `compartmental` and `agents` (the two engines; `agents_expressions`, `agents_moving`, `agents_stepping`, and `agents_branching` are parts of the second), `crossing`
(diseases tracked jointly), `choose` (which solver runs a model, and why),
`deliveries` (what a delivery does differently in a compartmental run), and
`particles` (many runs of a program stepped together, for a particle filter).
Their names are all found here, e.g. `ss.solvers.Compartmental`. The kernels they run
are in `backends` (`ss.solvers.backends`, and `ss.kernels`), with a copy in Rust in `rust/`.

Used with: agents and compartments; declared and coded modules
"""
from .engine import *
from .mixing import *
from .compartmental import *
from .crossing import *
from .agents import *
from .choose import *
from .deliveries import *
from .particles import *
