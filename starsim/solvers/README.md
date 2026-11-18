# Solvers

How a module's declared states and transitions are run. The same declared model can be run as compartments (`solver='ode'`, or `ss.solvers.ode(dwells='exact')`, `'binomial'`, `'gillespie'`, or `'tau'`) or as agents (`'abm'`, or `ss.solvers.abm(step='events')`, event by event in continuous time); each module with declared states gets one engine, which the sim's loop calls like any other module's code. Their names are all in `ss.solvers`, e.g. `ss.solvers.Compartmental`.

- `choose.py`: the solvers to choose from (`ss.solvers.ode()` ... `abm()`), which solver runs a model, and why (`ss.solvers.choose_solver()`, by the rules at its end), and the engine each module gets (`make_engine()`)
- `build.py`: the steps that build a declared model, before the solver is chosen (`init_model()`) and once it is
- `engine.py`: what the engines share, and the methods each defines (in the `Engine` docstring)
- `compartmental.py`: the compartmental engine, which runs a model as groups of people, through the compartmental kernels
- `agents.py`: the agent engine, `Agents`, which runs a model as individual people; its parts are `agents_expressions.py`, `agents_moving.py`, `agents_stepping.py`, and `agents_branching.py`
- `agents_compiled.py`: `Compiled`, the default engine of `solver='abm'`: the agent engine with the moves of each step taken in one compiled call
- `agents_events.py`: `Events`, the agent engine taken event by event within each step (`solver=ss.solvers.abm(step='events')`), for a module or for the whole sim
- `contact_rates.py`: the contact rates between groups that the networks become (who meets whom in a compartmental run; also the places and mixing pools of an agent run)
- `joint_states.py`: diseases whose states are tracked jointly in a compartmental run
- `deliveries.py`: what an intervention's delivery does differently in a compartmental run
- `particles.py`: many runs stepped forward together, for a particle filter
- `numba/`: the kernels the engines run, in Numba, with a table of each file and its Rust copy (see its README)
- `rust/`: the same kernels in Rust, built from source (see its README); not part of the installed package
- `backends/`: the machinery around the kernels: swapping in Rust's, compiling Numba's, `ss.jit` (see its README)
