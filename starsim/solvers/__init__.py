"""
Solvers: how a module's declared states and transitions are run.

The same `ss.graph.Graph` can be run as compartments (`ode`, with its dwell times as stages or as they are, `binomial`, `gillespie`,
`tau`) or as agents (`abm`: by steps, with arrivals during a step facing the rest of it; rates over whole steps only; or event by event: see `ss.solvers.abm()` and `agent_solver()`). Each module with a graph gets one engine, which is called from
the sim's loop like any other module code.

The modules: `engine` (what the engines share), `contact_rates` (who meets whom in a
compartmental run), `compartmental` and `agents` (the two engines; `agents_expressions`, `agents_moving`, `agents_stepping`, `agents_transmission`, and `agents_branching` are parts of the second;
its variants are `agents_compiled`, whose `Compiled` is the default engine of `solver='abm'` and takes the moves of a step as one compiled call, and `agents_events`, which takes it event by event), `joint_states`
(diseases tracked jointly), `choose` (which solver runs a model, and why), `build` (the steps that build a declared model),
`deliveries` (what a delivery does differently in a compartmental run), and
`particles` (many runs of a program stepped together, for a particle filter).
Their names are all found here, e.g. `ss.solvers.Compartmental`. The kernels they run
are in `numba` (`ss.solvers.numba`), with a copy in Rust in `rust/`, and the machinery around them (swapping in Rust's, compiling, `ss.jit`) in
`backends` (`ss.solvers.backends`). A *solver* is the method (e.g. `ode`), and a
*backend* how its numbers are computed (`ss.options.backend`: Rust, Numba, plain Python, or JAX); an *engine* is the
object that runs one module under the solver, which a modeler never needs.

Used with: agents and compartments; declared and coded modules
"""
from .engine import *
from .contact_rates import *
from .compartmental import *
from .joint_states import *
from .agents import *
from .agents_events import *
from .agents_compiled import *
from .choose import *
from .build import *
from .deliveries import *
from .particles import *
from . import numba, backends
from .backends.compiling import compile_status # Whether the kernels are compiled, and with which backend
