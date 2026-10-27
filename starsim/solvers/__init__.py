"""
Solvers: how a module's declared states and transitions are run.

The same `ss.transitions.Graph` can be run as compartments (`ode`, `binomial`, `gillespie`,
`tau`) or as agents (`abm`). Each module with a graph gets one engine, which is called from
the sim's loop like any other module code.

The modules: `engine` (what the engines share), `mixing` (who meets whom in a
compartmental run), `compartmental` and `agents` (the two engines), `lowered`
(the rounds of moves of a disease as agents, compiled where they can be), `crossing`
(diseases tracked jointly), `choose` (which solver runs a model, and why),
`deliveries` (what a delivery does differently in a compartmental run), and
`particles` (many runs of a program stepped together, for a particle filter).
Their names are all found here, e.g. `ss.solvers.Compartmental`.
"""
from .engine import *
from .mixing import *
from .compartmental import *
from .crossing import *
from .agents import *
from .lowered import *
from .choose import *
from .deliveries import *
from .particles import *
