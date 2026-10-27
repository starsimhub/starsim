"""
Many runs of a compartmental program stepped forward together: the particles of a particle filter
"""
import numpy as np
import numba as nb
import starsim as ss
from .crossing import Crossing

__all__ = ['Particles']


class Particles:
    """
    Many runs of an engine's program, from its start, each with its own compartments and random numbers, stepped forward together; for `ss.fit.ParticleFilter`

    The runs start from the engine as it is (restarted), and the engine is not changed until `restore()`.
    `advance()` takes every run to a later time point, and `select()` keeps some of the runs (each as
    often as it is picked), as a filter does when it draws them again.

    Args:
        engine (`ss.solvers.Compartmental`): the engine of a declared disease, restarted; or an `ss.solvers.Crossing`, for diseases tracked jointly, whose program's compartments are then the states
        n (int): the number of runs
        rng (`np.random.Generator`): from which the random numbers of each run are spawned
        by_group (bool): whether to keep the stocks and flows by group (else their total, as one group)

    Attributes:
        stocks (array): the number in each state at each time point of each run (runs × time points × states × groups)
        flows (array): the flow along each transition and mover over the step to each time point, likewise
        done (int): the time point the runs have reached
        npts (int): the time points of a whole run
        per_step (int): the rows of the tables for each step (see `ss.solvers.Compartmental.per_step()`)
    """
    def __init__(self, engine, n, rng, by_group=False):
        args, tables = engine.run_args()
        self.per_step = engine.per_step()
        self.engines = engine.engines
        if isinstance(engine, Crossing):
            state_of = np.arange(engine.n_rows)
            n_states, n_groups = engine.n_rows, engine.n_cols
        else:
            states = list(engine.graph.states)
            state_of = np.array([states.index(comp.state.name) for comp in engine.comps])
            n_states, n_groups = len(states), engine.n_groups
        y0, self.program, self.dt, self.npts, self.solver, self.n_sub = args
        p = self.program
        self.state_of, self.register, self.shifts = state_of, engine.register, engine.shifts
        self.tables = [table[None] for table in tables] # The same values for every run
        self.own_share = p.sh_val.copy()
        self.shares = self.own_share[None] # The same time share between places for every run
        nk = n_groups if by_group else 1
        self.stocks = np.zeros((n, self.npts, n_states, nk))
        self.flows = np.zeros((n, self.npts, p.n_trans + p.n_movers, nk))
        for c,state in enumerate(state_of):
            self.stocks[:, 0, state] += y0[c] if by_group else y0[c].sum()
        self.ys = np.ascontiguousarray(np.repeat(y0[None], n, axis=0))
        self.rngs = nb.typed.List([np.random.Generator(bits) for bits in rng.bit_generator.spawn(n)]) # The random numbers of each run (of each place among the runs, which a run drawn again takes over)
        self.n_threads = min(ss.options.threads or nb.get_num_threads(), n)
        if self.n_threads > 1: # A copy of the program for each thread, since a step changes the parts of it that vary in time, and the threads would contend for the counts of references to shared arrays
            self.progs = nb.typed.List([ss.kernels.Program(*[part.copy() if isinstance(part, np.ndarray) else part for part in p]) for i in range(self.n_threads)])
        self.done = 0
        return

    def advance(self, end, tables=None, shares=None, row0=0):
        """
        Take every run to time point `end`, if it is not there already

        Args:
            end (int): the time point
            tables (list): the parts of the program that vary in time, as `ss.kernels.advance()` takes them, if not the engine's own: one entry for every run, or one for each (as each particle of IF2 has its own values)
            shares (array): the time share between places of every run (runs by entries), if not the engine's own
            row0 (int): the row of a whole run's tables that the first row of `tables` is
        """
        if end <= self.done:
            return
        tables = self.tables if tables is None else tables
        shares = self.shares if shares is None else shares
        rest = (self.solver, self.n_sub, *tables, shares, self.register, self.shifts, self.state_of, self.stocks, self.flows, self.per_step, row0)
        n_sub = ss.kernels.substeps(np.append(self.program.fixed[self.program.src >= 0], self.program.mov_rate), self.dt, self.n_sub, self.solver) # The rates of the step last refreshed
        kn = ss.kernels.choose(len(self.ys)*ss.kernels.plain_seconds(self.program, self.ys[0], end - self.done + 1, self.solver, n_sub))
        if self.n_threads > 1:
            kn.advance_threads(self.rngs, self.progs, self.ys, self.dt, self.done, end, *rest)
        else:
            kn.advance(self.rngs, self.program, self.ys, 0, len(self.ys), self.dt, self.done, end, *rest)
        self.done = end
        return

    def select(self, picks, keep=0):
        """ Keep the runs picked, each as often as it is picked, with their stocks and flows from time point `keep` on (those before it are left as they were) """
        self.ys = self.ys[picks]
        if keep <= self.done:
            self.stocks[:, keep:self.done+1] = self.stocks[picks, keep:self.done+1]
            self.flows[:, keep:self.done+1] = self.flows[picks, keep:self.done+1]
        return

    def restore(self):
        """ Give the engine back its program as it was (a run with a time share of its own writes it there), without what the runs left in its cache """
        self.program.sh_val[:] = self.own_share
        for engine in self.engines:
            engine.cache.clear()
        return
