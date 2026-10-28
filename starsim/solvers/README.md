# Solvers

How a module's declared states and transitions are run. The same declared model can be run as compartments (`solver='ode'`, `'binomial'`, `'gillespie'`, or `'tau'`) or as agents (`'abm'`); each module with declared states gets one engine, which the sim's loop calls like any other module's code. Their names are all in `ss.solvers`, e.g. `ss.solvers.Compartmental`.

- `choose.py`: which solver runs a model, and why (`ss.solvers.choose_solver()`), and the steps that build a declared model before it is run
- `engine.py`: what the engines share
- `compartmental.py`: the compartmental engine, which runs a model as groups of people
- `agents.py`: the agent engine, which runs a model as individual people
- `mixing.py`: who meets whom in a compartmental run (the contact rates between groups that the networks become)
- `crossing.py`: diseases whose states are tracked jointly in a compartmental run
- `deliveries.py`: what an intervention's delivery does differently in a compartmental run
- `particles.py`: many runs stepped forward together, for a particle filter
- `backends/`: the compiled kernels the engines run, in Numba and JAX (see its README)
- `rust/`: the same kernels in Rust, built from source (see its README); not part of the installed package
