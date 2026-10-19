"""
The ODE run in JAX: gradients of results with respect to parameters, and many runs at once

A compartmental engine flattens its flows into a `ss.kernels.Program`. The Numba
kernels step through it; `run()` here takes the same program and runs the same
fourth-order Runge-Kutta (or Euler) steps in JAX, so that the run can be
differentiated (`gradient()`) and many runs with different values can be taken in
one call (`run_sets()`). JAX is imported only when one of these is used.

What JAX differentiates is the run: the program's numbers, and the compartments at
the start, to the compartments and flows at every step. The steps either side of it
are Python (working out those numbers from the parameters, and the results from
the run, including outcomes and the goodness of fit), and are differentiated by
central differences along the direction the run is differentiated in. They are cheap
and smooth, so this is accurate to about 1e-9 of the value.
"""
import os
import sys
import itertools
import importlib.util
import numpy as np
import sciris as sc
import starsim as ss

__all__ = [] # Used as ss.gradients

# The parts of the program that are numbers, and so can be differentiated (the rest says how the flows are put together)
diff_fields = ['inflow', 'contacts', 'mat', 'sh_val', 'tgt_w', 'crd_w', 'maps', 'mov_rate']
table_fields = ['fixed', 'mult', 'consts', 'rel_trans'] # As tables over the steps: see ss.solvers.Compartmental.tables()


def jax():
    """
    Import JAX, or say that it is needed

    If JAX has not been imported yet, it is given a device for each core of the CPU (as
    `numpyro.set_host_device_count()` does), so that the chains of a posterior run on
    cores of their own; everything else runs on the first, as before.
    """
    if 'jax' not in sys.modules and 'xla_force_host_platform_device_count' not in os.environ.get('XLA_FLAGS', ''):
        os.environ['XLA_FLAGS'] = f'{os.environ.get("XLA_FLAGS", "")} --xla_force_host_platform_device_count={os.cpu_count() or 1}'.strip()
    try:
        import jax
    except ImportError as e:
        errormsg = 'This needs JAX, which is not installed: pip install jax'
        raise ImportError(errormsg) from e
    if ss.options.jax_cache and jax.config.jax_compilation_cache_dir is None: # Keep what is compiled for later sessions, unless told otherwise
        jax.config.update('jax_compilation_cache_dir', ss.options.jax_cache)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.25)
    return jax


def available():
    """ Whether JAX is installed (without importing it) """
    return importlib.util.find_spec('jax') is not None



def reason(sim):
    """ Why the sim cannot be run in JAX, in a few words, or None if it can """
    if not available():
        return 'JAX is not installed'
    elif sim.solver != 'ode':
        return f'solver="{sim.solver}" is stochastic (gradients are for solver="ode" only)'
    elif not sim.kernel_only():
        return sim.kernel_reason()
    for mod in sim.declared:
        engine = mod.engine
        if engine.register is not None and len(engine.register.src):
            return f'"{mod.name}" has a movement register'
    return None


#%% The run

class Run:
    """
    One engine's run, as a function of the numbers in it

    Args:
        program (Program): the engine's program, for how its flows are put together
        dt (float): the timestep
        npts (int): the number of steps
        solver (int): 0 for RK4, 1 for Euler (see `ss.kernels.solvers`)
        n_sub (int): the number of substeps a step, or 0 for enough that no compartment turns over, or grows through its inflows, by more than a quarter in one
        per_step (int): the rows of the tables for each step (see `ss.solvers.Compartmental.per_step()`)
        n_max (int): the most substeps a step can take; a run that needs more is taken again with more
    """
    def __init__(self, program, dt, npts, solver, n_sub, per_step, n_max):
        self.p = program
        self.dt = dt
        self.npts = npts
        self.solver = solver
        self.n_sub = n_sub
        self.per_step = per_step
        self.n_max = n_sub if n_sub else n_max
        self.fixed = False # Whether a clamped step always takes n_max substeps (those past the ones it needs change nothing), so the run can be differentiated in reverse mode
        self.compiled = dict()
        self.layout()
        return

    @staticmethod
    def key(program, dt, npts, solver, n_sub, per_step):
        """ What a compiled run depends on, apart from the numbers in it """
        static = [getattr(program, name) for name in program._fields if name not in diff_fields + table_fields]
        return (dt, npts, solver, n_sub, per_step) + tuple(np.asarray(v).tobytes() + str(np.shape(v)).encode() for v in static)

    def layout(self):
        """ Work out, once, the matrices that put the flows together """
        p = self.p
        nf = len(p.src)
        ng = len(p.places)
        self.nf, self.ng = nf, ng
        self.nflows = p.n_trans + p.n_movers
        self.dynamic = [f for f in range(nf) if p.starts[f] < p.starts[f+1]]
        self.leaves = np.flatnonzero(p.src >= 0) # The flows that leave a compartment
        self.place = np.zeros((p.n_places, ng)) # Which place each group is in
        self.place[p.places, np.arange(ng)] = 1
        self.sets = [p.set_comps[p.set_starts[s]:p.set_starts[s+1]] for s in range(len(p.set_starts) - 1)]
        self.tgt_flow = np.repeat(np.arange(nf), np.diff(p.tgt_starts)) # The flow each target belongs to
        self.crd_flow = np.repeat(np.arange(nf), np.diff(p.crd_starts)) # ... and each transition a flow is also counted as

        # Where each flow takes people, and what it is counted as, as matrices (which XLA does much faster than adding into arrays at indices)
        named = np.flatnonzero(p.named)
        self.plain = named[p.record[named] < 0]
        self.mapped = named[p.record[named] >= 0]
        self.count_mat = np.zeros((self.nflows, nf)) # Transitions by flows: 1 where a flow is counted as its transition
        self.count_mat[p.index[self.plain], self.plain] = 1
        nt = len(p.tgt_comp)
        self.tgt_plain = np.flatnonzero(p.tgt_map < 0) if nt else np.zeros(0, dtype=int)
        self.tgt_mapped = np.flatnonzero(p.tgt_map >= 0) if nt else np.zeros(0, dtype=int)
        self.gather_tgt = np.zeros((nt, nf)) # Targets by flows: which flow each target takes people from
        self.gather_tgt[np.arange(nt), self.tgt_flow] = 1
        ncr = len(p.crd_index)
        self.credit_mat = np.zeros((self.nflows, ncr)) # Transitions by credits
        self.credit_mat[p.crd_index, np.arange(ncr)] = 1
        self.gather_crd = np.zeros((ncr, nf))
        self.gather_crd[np.arange(ncr), self.crd_flow] = 1
        self.simple = p.n_places == 1 and ng == 1 # One group: no places or cells to go through
        extra = [(p.starts[i], p.starts[i+1]) for i in range(nf, len(p.starts) - 1)] # The conveyors and updates, after the flows' operations (see ss.kernels.convey() and update_values())
        conveyors = [(a, b) for a,b in extra if p.ops[a] == ss.kernels.CONVEY]
        self.updates = [(a, b) for a,b in extra if p.ops[a] != ss.kernels.CONVEY] # The operations of each (the last sets its compartment)
        self.up_comp = np.array([p.args[b-1] for a,b in self.updates], dtype=int)
        self.cv_flow = np.array([p.args[a] for a,b in conveyors], dtype=int)
        slots = [p.args[a+1:b] for a,b in conveyors] # The slots of each conveyor: its first, last, and each that people move along from and to
        self.cv_first = np.array([s[0] for s in slots], dtype=int)
        self.cv_last = np.array([s[-1] for s in slots], dtype=int)
        self.cv_prev = np.array([c for s in slots for c in s[:-1]], dtype=int)
        self.cv_next = np.array([c for s in slots for c in s[1:]], dtype=int)
        self.cv_exits = (p.src[None, :] == self.cv_last[:, None]).astype(float) # The other exits of the last slot of each conveyor, which act on the people there until they leave (see ss.kernels.release_rates())
        self.comp_mats = {}
        return

    def comp_matrices(self, nc):
        """ For nc compartments: the compartments by flows (-1 where a flow leaves) and by targets (1 where a target not mapped between groups takes people) """
        if nc not in self.comp_mats:
            p = self.p
            leave = np.zeros((nc, self.nf))
            leave[p.src[self.leaves], self.leaves] = -1
            arrive = np.zeros((nc, len(p.tgt_comp)))
            arrive[p.tgt_comp[self.tgt_plain], self.tgt_plain] = 1
            self.comp_mats[nc] = (leave, arrive)
        return self.comp_mats[nc]

    def rates(self, ys, vals):
        """ The rate of every flow, by group, with the compartments at ys and the program's values in vals: as `ss.kernels.step()` works them out """
        jnp = jax().numpy
        p = self.p
        fixed, mult, consts, rel_trans, X = vals
        if not self.dynamic:
            return fixed
        n = ys[p.people].sum(axis=0) # The people in each group
        local_n = n if self.simple else self.place.T @ (self.place @ n) # ... and in the same place as each group
        exposures = []
        for e in range(len(p.exp_set)): # Exposure to each infectious state through each mixing term
            s = p.exp_set[e]
            if s < 0: # The sum of the -s exposures before it, in order
                expo = exposures[e+s]
                for x in range(e+s+1, e):
                    expo = expo + exposures[x]
                exposures.append(expo)
                continue
            infectious = (ys[self.sets[p.exp_set[e]]]*rel_trans[e]).sum(axis=0)
            expo = 0.0
            if self.simple and not p.use[0].any() and not p.use[2].any(): # One group, mixing with itself
                for t in range(len(p.sizes)):
                    expo = expo + X['contacts'][min(e, len(p.contacts) - 1), t]*jnp.where(n > 0, infectious/jnp.where(n > 0, n, 1.0), 0.0)
                exposures.append(expo)
                continue
            for t in range(len(p.sizes)):
                size = int(p.sizes[t])
                cells = np.zeros((size, self.ng))
                cells[p.levels[t], np.arange(self.ng)] = 1
                num = cells @ infectious
                den = cells @ n
                block = int(p.block[t])
                if p.use[0, t]: # Move people to where they spend their time
                    num = self.share(num, block, X, True)
                    den = self.share(den, block, X, True)
                prev = jnp.where(den > 0, num/jnp.where(den > 0, den, 1.0), 0.0)
                if p.use[1, t]: # Mix between the levels in each place
                    prev = (prev.reshape(-1, block) @ X['mat'][t, :block, :block].T).ravel()
                if p.use[2, t]: # Mix between places
                    prev = self.share(prev, block, X, False)
                expo = expo + X['contacts'][min(e, len(p.contacts) - 1), t]*prev[p.levels[t]]
            exposures.append(expo)

        rows = [fixed[f] for f in range(self.nf)]
        for f in self.dynamic: # The operations on a stack, as in the kernels
            rows[f] = jnp.broadcast_to(self.evaluate(ys, p.starts[f], p.starts[f+1], consts, exposures, local_n), (self.ng,))*mult[f]
        return jnp.stack(rows)

    def evaluate(self, ys, a, b, consts, exposures, local_n):
        """ The value of the operations a to b-1 of the program, by group, with the compartments at ys (as `ss.kernels.step()` works out a rate) """
        jnp = jax().numpy
        p = self.p
        kn = ss.kernels
        stack = []
        for k in range(a, b):
            op, arg = p.ops[k], p.args[k]
            if op == kn.CONST:
                stack.append(consts[arg])
            elif op == kn.EXPOSURE:
                stack.append(exposures[arg])
            elif op == kn.TOTAL: # In the whole population, every place
                stack.append(jnp.broadcast_to(ys[self.sets[arg]].sum(), (self.ng,)))
            elif op in [kn.COUNT, kn.FRAC]:
                local = ys[self.sets[arg]].sum(axis=0)
                if not self.simple:
                    local = self.place.T @ (self.place @ local)
                stack.append(local/local_n if op == kn.FRAC else local)
            elif op >= kn.EXP:
                x = stack.pop()
                func = {kn.EXP:jnp.exp, kn.LOG:jnp.log, kn.SQRT:jnp.sqrt, kn.ABS:jnp.abs, kn.SIN:jnp.sin, kn.COS:jnp.cos}[op]
                stack.append(func(x))
            else:
                y = stack.pop()
                x = stack.pop()
                if   op == kn.ADD: stack.append(x + y)
                elif op == kn.SUB: stack.append(x - y)
                elif op == kn.MUL: stack.append(x * y)
                elif op == kn.DIV: stack.append(jnp.where(x == 0, 0.0, x/jnp.where(x == 0, 1.0, y))) # As in the kernels: nothing divided by anything is nothing
                elif op == kn.POW: stack.append(x ** y)
                elif op == kn.MIN: stack.append(jnp.minimum(x, y))
                elif op == kn.MAX: stack.append(jnp.maximum(x, y))
                elif op == kn.LT:  stack.append(jnp.where(x < y, 1.0, 0.0))
                elif op == kn.LE:  stack.append(jnp.where(x <= y, 1.0, 0.0))
                elif op == kn.GT:  stack.append(jnp.where(x > y, 1.0, 0.0))
                else:              stack.append(jnp.where(x >= y, 1.0, 0.0))
        return stack[0]

    def update_values(self, ys, consts):
        """ The values that counts are set to at the end of a step (`ss.update()`), from the compartments at its start (as `ss.kernels.update_values()`) """
        jnp = jax().numpy
        p = self.p
        n = ys[p.people].sum(axis=0)
        local_n = n if self.simple else self.place.T @ (self.place @ n)
        local_n = jnp.where(local_n > 0, local_n, jnp.inf) # A share of nobody is 0, as in the kernels
        return jnp.stack([jnp.broadcast_to(self.evaluate(ys, a, b-1, consts, [], local_n), (self.ng,)) for a,b in self.updates])

    def share(self, v, block, X, by_columns):
        """ The time share between places times values by place and level (as `ss.kernels.share()`) """
        jnp = jax().numpy
        p = self.p
        v = v.reshape(-1, block)
        frm, to = (p.sh_pair[0], p.sh_pair[1]) if by_columns else (p.sh_pair[1], p.sh_pair[0]) # Taken from the entry's row and added to its column, or the reverse
        return jnp.zeros_like(v).at[to].add(X['sh_val'][:, None]*v[frm]).ravel()

    def derivs(self, ys, rates, X, h):
        """ The rate of change of each compartment, and of each transition and mover """
        jnp = jax().numpy
        p = self.p
        nc = ys.shape[0]
        dy = jnp.zeros_like(ys)
        df = jnp.zeros((self.nflows, self.ng))
        for m in range(p.n_movers): # Movement between groups
            ent = slice(p.mov_starts[1, m], p.mov_starts[1, m+1])
            frm, to, rate = p.mov_pair[0, ent], p.mov_pair[1, ent], X['mov_rate'][ent]
            comps = p.mov_comps[p.mov_starts[0, m]:p.mov_starts[0, m+1]]
            moved = ys[comps][:, frm]*rate
            dy = dy.at[comps].add(jnp.zeros((len(comps), self.ng)).at[:, to].add(moved).at[:, frm].add(-moved))
            df = df.at[p.n_trans + m].add(jnp.zeros(self.ng).at[frm].add(moved.sum(axis=0)))
        src = np.maximum(p.src, 0)
        leave_mat, arrive_mat = self.comp_matrices(nc)
        amount = jnp.where((p.src >= 0)[:, None], rates*ys[src], rates*X['inflow'])
        if self.solver == 1: # Euler: outflows that would take more than everyone in a compartment are scaled down together
            total = -leave_mat @ (rates*(p.src >= 0)[:, None]*h)
            scale = jnp.where(total > 1, 1/jnp.where(total > 1, total, 1.0), 1.0)
            amount = jnp.where((p.src >= 0)[:, None], amount*scale[src], amount)
        dy = dy + leave_mat @ amount
        df = df + self.count_mat @ amount
        mapped = self.mapped
        if len(mapped):
            df = df.at[p.index[mapped]].add(jnp.einsum('kg,kgh->kh', amount[mapped], X['maps'][p.record[mapped]]))
        if len(p.tgt_comp):
            moved = (self.gather_tgt @ amount)*X['tgt_w']
            dy = dy + arrive_mat @ moved
            mapped = self.tgt_mapped
            if len(mapped):
                dy = dy.at[p.tgt_comp[mapped]].add(jnp.einsum('kg,kgh->kh', moved[mapped], X['maps'][p.tgt_map[mapped]]))
        if len(p.crd_index):
            df = df + self.credit_mat @ ((self.gather_crd @ amount)*X['crd_w'])
        return dy, df

    def stepper(self, X, clamp=False):
        """
        One step of the run, as a function of the compartments at its start and its number: returns the compartments at its end, the flows over it, and the substeps it needed

        Args:
            X (dict): the numbers in the run (see `capture()`)
            clamp (bool): if True, a step takes at most `n_max` substeps, so that the run is the same function everywhere (for sampling, where it cannot be taken again with more)
        """
        jx = jax()
        jnp = jx.numpy
        lax = jx.lax
        dt, per_step, n_max = self.dt, self.per_step, self.n_max
        rk4 = self.solver == 0

        def at(table, ti, th):
            """ A table's value at a fraction th of the way through step ti (see ss.kernels.interp()) """
            if table.shape[0] == 1:
                return table[0]
            row = ti*per_step
            if per_step == 3:
                a = (2*th - 1)*(th - 1)
                b = 4*th*(1 - th)
                c = th*(2*th - 1)
                return a*table[row] + b*table[row+1] + c*table[row+2]
            return table[row]

        def vals(ti, th):
            return [at(X[name], ti, th) for name in table_fields] + [X]

        def step(y, ti):
            due = y[self.cv_last] # The people due to leave each conveyor, at a constant rate over the step (see ss.kernels.convey())
            if len(self.updates): # Counts set to a value at the end of the step, worked out from the start
                values = self.update_values(y, at(X['consts'], ti, 0.0))

            if len(self.cv_flow): # The people due to leave each conveyor leave at a constant rate over the step, as the inflow of its flow (whose last target is its last slot, with a share of -1), lowered so that the slot is empty at its end with its other exits (see ss.kernels.release_rates())
                x = self.cv_exits @ jnp.maximum(self.rates(y, vals(ti, 0.0)), 0.0)*dt
                small = x < 1e-12
                share = jnp.where(small, 1.0, jnp.where(small, 1.0, x)/jnp.expm1(jnp.where(small, 1.0, x)))
                Xs = dict(X, inflow=X['inflow'].at[self.cv_flow].set(due/dt*share))
            else:
                Xs = X
            if self.n_sub:
                n = self.n_sub
            else: # Enough substeps that no compartment turns over by more than a quarter in one, nor grows by more than a quarter through its inflows (see ss.kernels.step())
                rates0 = self.rates(y, vals(ti, 0.0))
                fastest = jnp.max(rates0[self.leaves]) if len(self.leaves) else 0.0
                if len(self.p.mov_rate):
                    fastest = jnp.maximum(fastest, jnp.max(X['mov_rate']))
                dy0 = self.derivs(y, rates0, Xs, dt/jnp.maximum(jnp.ceil(fastest*dt/0.25), 1))[0]
                filling = (dy0 > 0) & (y > 1e-12*y.sum(axis=0)) # Not a compartment that is only starting to fill
                growing = jnp.max(jnp.where(filling, dy0/jnp.where(filling, y, 1.0), 0.0))
                n = jnp.maximum(jnp.ceil(lax.stop_gradient(jnp.maximum(fastest, growing))*dt/0.25), 1)
                if clamp:
                    n = jnp.minimum(n, n_max)
            h = dt/n

            def derivs(ys, th):
                return self.derivs(ys, self.rates(ys, vals(ti, th)), Xs, h)

            def sub(s, carry):
                y, flows = carry
                th = lambda frac: (s + frac)/n
                dy1, df1 = derivs(y, th(0.0))
                if rk4:
                    dy2, df2 = derivs(y + h/2*dy1, th(0.5))
                    dy3, df3 = derivs(y + h/2*dy2, th(0.5))
                    dy4, df4 = derivs(y + h*dy3, th(1.0))
                    new_y = y + h/6*(dy1 + 2*dy2 + 2*dy3 + dy4)
                    new_f = flows + h/6*(df1 + 2*df2 + 2*df3 + df4)
                else:
                    new_y = y + h*dy1
                    new_f = flows + h*df1
                active = s < n
                return jnp.where(active, new_y, y), jnp.where(active, new_f, flows)

            flows = jnp.zeros((self.nflows, self.ng))
            if self.n_sub: # A given number of substeps: written out, which XLA runs faster than a loop inside the loop over steps
                carry = (y, flows)
                for s in range(self.n_sub):
                    carry = sub(s, carry)
                y_end, flows = carry
            elif clamp and self.fixed: # A loop with a fixed count, which reverse mode needs
                y_end, flows = lax.fori_loop(0, n_max, sub, (y, flows))
            else: # As many substeps as the step needs (a loop with a fixed count would take n_max every time)
                def body(carry):
                    s, y, flows = carry
                    y, flows = sub(s, (y, flows))
                    return s + 1, y, flows
                _, y_end, flows = lax.while_loop(lambda carry: carry[0] < n, body, (0, y, flows))
            if len(self.cv_flow): # Everyone in a conveyor moves along one slot
                y_end = y_end.at[self.cv_next].set(y_end[self.cv_prev]).at[self.cv_first].set(0.0)
            if len(self.updates):
                y_end = y_end.at[self.up_comp].set(values)
            return jnp.maximum(y_end, 0.0), flows, n # The compartments after the step, as the kernels record them

        return step

    @staticmethod
    def deliver(y, X, ti):
        """ The deliveries at time point ti, at the start of the step from it, in order (see `ss.kernels.apply_shifts()`): each moves some of the people in each group to another """
        for k in range(X['sh_due'].shape[0]):
            reached = y*X['sh_sel'][k]*X['sh_due'][k, ti]
            a = reached*X['sh_cov'][k]
            b = reached - a
            y = y - reached + a @ X['sh_to'][k] + b @ X['sh_rest'][k]
        return y

    def function(self, clamp=False):
        """
        The run, as a function of a dict of its numbers (see `capture()`): returns the compartments at the end, at the end of each step, the flows over each step, and the most substeps a step needed

        Args:
            clamp (bool): see `stepper()`
        """
        jx = jax()
        jnp = jx.numpy

        def run(X):
            step = self.stepper(X, clamp)
            def body(y, ti):
                y = self.deliver(y, X, ti)
                y_end, flows, n = step(y, ti)
                return y_end, (y_end, flows, n)
            y_end, (stocks, flows, n) = jx.lax.scan(body, X['y'], jnp.arange(self.npts - 1)) # Step ti+1 goes from time point ti to ti+1, with the values at the start of it
            stocks = jnp.concatenate([X['y'][None], stocks]) # The start, then the end of each step, as the kernels record them
            flows = jnp.concatenate([jnp.zeros((1,) + flows.shape[1:]), flows])
            return y_end, stocks, flows, jnp.max(n)

        return run

    def get(self, kind='run'):
        """ The compiled run: 'run' for one, 'many' for a batch, 'jvp' for its derivative along a batch of directions """
        if kind not in self.compiled:
            self.compiled[kind] = compile_run(self.function(), kind)
        return self.compiled[kind]


def compile_run(run, kind):
    """ A run compiled for one use: 'run' for one, 'many' for a batch, 'jvp' for its derivative along a batch of directions """
    jx = jax()
    if kind == 'run':
        return jx.jit(run)
    elif kind == 'many':
        return jx.jit(jx.vmap(run))
    return jx.jit(lambda X, dX: jx.vmap(lambda d: jx.jvp(run, (X,), (d,))[1][:3])(dX)) # The substeps a step needs have no derivative


class CrossRun:
    """
    The diseases of a crossing, run together by the crossing's program (see `ss.solvers.Crossing`)

    Called with a tuple of the numbers of each disease's run: they are put together into the numbers of the
    program (`ss.solvers.Crossing.numbers()`), which is run as any other, and its output is split between the diseases
    again. Returns what `Run.function()` does, with a tuple over the diseases for each of the first three.

    Args:
        cross (`ss.solvers.Crossing`): the crossing
        run (Run): the run of its program
        reps (list): for each disease, the rows of the program's tables for each row of its own
    """
    def __init__(self, cross, run, reps):
        self.cross = cross
        self.run = run
        self.runs = [run]
        self.reps = reps
        self.npts = run.npts
        self.n_sub = run.n_sub
        self.compiled = dict()
        return

    @property
    def n_max(self):
        return self.run.n_max

    @n_max.setter
    def n_max(self, value):
        self.run.n_max = value
        return

    def function(self, clamp=False):
        jnp = jax().numpy
        inner = self.run.function(clamp)
        cross = self.cross

        def run(Xs):
            y_end, stocks, flows, n = inner(cross.numbers(list(Xs), xp=jnp, reps=self.reps))
            parts = [cross.split(k, y_end, stocks, flows, xp=jnp) for k in range(len(Xs))]
            return tuple(part[0] for part in parts), tuple(part[1] for part in parts), tuple(part[2] for part in parts), n

        return run

    def get(self, kind='run'):
        if kind not in self.compiled:
            self.compiled[kind] = compile_run(self.function(), kind)
        return self.compiled[kind]


def capture(engine):
    """ The numbers in an engine's run, as it is set up now (with its current parameter values): the program, how to run it, and the numbers """
    args, tables = engine.run_args()
    y, program, dt, npts, solver, n_sub = args
    X = {name:np.array(getattr(program, name), dtype=float) for name in diff_fields}
    X.update({name:np.array(table, dtype=float) for name,table in zip(table_fields, tables)})
    X['y'] = np.array(y, dtype=float)
    sh = engine.shifts # The deliveries, with where each group's people go as matrices
    onehot = lambda to: np.eye(engine.n_groups)[to] if len(to) else np.zeros((0, engine.n_groups, engine.n_groups))
    X.update(sh_due=sh.due.astype(float), sh_sel=sh.sel, sh_cov=sh.cov, sh_to=onehot(sh.to), sh_rest=onehot(sh.rest))
    return program, (dt, npts, solver, n_sub, engine.per_step()), X


def runner(engine, program, how):
    """ The engine's `Run`, made again only if how its flows are put together has changed """
    kept = engine.jax_run
    if kept is not None and kept[1] is program and kept[2] == how: # The same program: nothing can have changed
        return kept[3]
    key = Run.key(program, *how)
    if kept is None or kept[0] != key:
        run = Run(program, *how, n_max=kept[3].n_max if kept is not None else 4)
    else:
        run = kept[3]
    engine.jax_run = (key, program, how, run)
    return run


def joint_runner(cross, captured, n_max=None):
    """ The `CrossRun` of a crossing, from the program and how to run each of its engines; made again only if how the program's flows are put together has changed (or, with `n_max`, always) """
    programs = [program for program,how in captured]
    if cross.program is None or cross.built != [id(p) for p in programs]:
        cross.build(programs)
    per_step, reps = cross.per_step()
    how = captured[0][1][:4] + (per_step,)
    if n_max is not None:
        return CrossRun(cross, Run(cross.program, *how, n_max=n_max), reps)
    key = Run.key(cross.program, *how)
    kept = cross.jax_run
    if kept is not None and kept[0] == key:
        return kept[1]
    run = CrossRun(cross, Run(cross.program, *how, n_max=kept[1].n_max if kept is not None else 4), reps)
    cross.jax_run = (key, run)
    return run


def units(sim):
    """ The declared modules that are run together, by their positions in `sim.declared`: the diseases of each crossing (in the order they step), and each other one on its own """
    names = [mod.name for mod in sim.declared]
    out = [[names.index(name) for name in cross.names] for cross in sim.crossings]
    crossed = {j for unit in out for j in unit}
    out += [[j] for j in range(len(names)) if j not in crossed]
    return out


def run_crossing(cross):
    """ Take a whole run of the diseases of a crossing in JAX: returns, for each, what `ss.kernels.run()` does """
    captured = [capture(engine) for engine in cross.engines]
    run = joint_runner(cross, [c[:2] for c in captured])
    ys, stocks, flows, n = call(run, 'run', tuple(c[2] for c in captured))
    return [(np.asarray(ys[k]), np.asarray(stocks[k]), np.asarray(flows[k]), 0.0) for k in range(len(captured))]


def call(run, kind, *args):
    """ Call a compiled run, taking it again with more substeps if a step needed more than it allowed """
    jx = jax()
    with jx.enable_x64(True):
        while True:
            out = run.get(kind)(*args)
            if kind == 'jvp' or run.n_sub:
                return out
            need = int(np.max(out[3]))
            if need <= run.n_max:
                return out
            run.n_max = int(2**np.ceil(np.log2(need))) # Rounded up, so that it is compiled again only a few times
            run.compiled.clear()


def run(engine):
    """ Take a whole run of an engine in JAX: returns what `ss.kernels.run()` does """
    program, how, X = capture(engine)
    y, stocks, flows, n = call(runner(engine, program, how), 'run', X)
    return np.asarray(y), np.asarray(stocks), np.asarray(flows), 0.0


#%% Many runs, and gradients

def stack(Xs):
    return {key:np.stack([X[key] for X in Xs]) for key in Xs[0]}


def check_same(programs):
    """ Runs can only be taken together if their flows are put together in the same way """
    first, how = programs[0]
    static = [name for name in first._fields if name not in diff_fields + table_fields]
    if all(h == how and all(getattr(p, name) is getattr(first, name) for name in static) for p,h in programs): # The same arrays, so nothing to compare
        return
    keys = {Run.key(program, *how) for program,how in programs}
    if len(keys) > 1:
        errormsg = 'The parameter values change how the model is laid out (e.g. the number of stages of a dwell time), so the runs cannot be taken together'
        raise ValueError(errormsg)
    return


def outputs(sim, Xs, kind='run', dX=None):
    """ Take the runs of each engine: Xs is, for each engine, its program, how to run it, and its numbers (a list of them, for kind='many') """
    out = [None]*len(sim.declared)
    for unit in units(sim):
        if kind == 'many':
            programs = [[(program, how) for program,how,X in Xs[j]] for j in unit]
            for progs in programs:
                check_same(progs)
            args = [tuple(stack([X for program,how,X in Xs[j]]) for j in unit)]
        else:
            programs = [[Xs[j][:2]] for j in unit]
            args = [tuple(Xs[j][2] for j in unit)] + ([tuple(stack(dX[j]) for j in unit)] if kind == 'jvp' else [])
        if len(unit) == 1:
            run = runner(sim.declared[unit[0]].engine, *programs[0][0])
            out[unit[0]] = [np.asarray(v) for v in call(run, kind, *[arg[0] for arg in args])]
        else:
            cross = sim.declared[unit[0]].engine.cross
            res = call(joint_runner(cross, [progs[0] for progs in programs]), kind, *args)
            for k,j in enumerate(unit):
                out[j] = [np.asarray(v[k]) for v in res[:3]] + [np.asarray(v) for v in res[3:]]
    return out


def give(sim, outs, shift=None):
    """ Give each engine the output of its run, so that `sim.run()` uses it instead of running the kernels (see `ss.solvers.Compartmental.run_all()`) """
    for j,mod in enumerate(sim.declared):
        y, stocks, flows = outs[j][:3]
        if shift is not None:
            dy, dstocks, dflows = shift[j]
            y, stocks, flows = y + dy, stocks + dstocks, flows + dflows
        mod.engine.given = (y, stocks, flows, 0.0)
    return


def run_sets(sim, par_sets):
    """
    Run a sim with many sets of parameter values, taking the runs in one call to JAX

    Each set is a dict of values by name, as for `sim.restart(pars=...)`. Only for a
    sim that `reason()` allows: an ODE that the kernels run from start to finish.

    Args:
        sim (`ss.Sim`): an initialized sim
        par_sets (list): the values of each run

    Returns, for each declared module, its results over the runs, as `ss.solvers.Compartmental.run_many()` does.
    """
    mods = sim.declared
    captured = [[] for mod in mods]
    for pars in par_sets:
        sim.restart(pars=pars)
        for j,mod in enumerate(mods):
            captured[j].append(capture(mod.engine))
    outs = outputs(sim, captured, kind='many')
    base = sim.par_values
    results = sc.objdict()
    for j,mod in enumerate(mods): # The number in each state, by group
        engine = mod.engine
        states = list(engine.graph.states)
        member = np.array([[comp.state.name == name for comp in engine.comps] for name in states], dtype=float)
        stocks = np.einsum('rtcg,sc->rtsg', outs[j][1], member)
        flows = outs[j][2]
        npts = engine.mod.t.npts
        if not (sim.groups.strats != [] and len(par_sets)*npts*(stocks.shape[2] + flows.shape[2])*engine.n_groups*8 <= engine.max_bytes): # As in run_many()
            stocks, flows = stocks.sum(axis=-1, keepdims=True), flows.sum(axis=-1, keepdims=True)
        runs = []
        for i,pars in enumerate(par_sets): # Results worked out from the values of the parameters in each run (an outcome can depend on them)
            sim.par_values = sc.mergedicts(base, pars)
            sim.set_par_values()
            engine.cache.clear()
            runs.append(engine.collect(stocks[i:i+1], flows[i:i+1]))
        results[mod.name] = sc.objdict({key:(sc.objdict({k:np.concatenate([run[key][k] for run in runs]) for k in runs[0][key]}) if key in ['groups', 'denoms'] else np.concatenate([run[key] for run in runs])) for key in runs[0] if key != 'group_ti'})
        results[mod.name].group_ti = None # The values by group, if kept, are at every time point
    sim.par_values = base
    sim.set_par_values()
    return results


def tangent(Xp, Xm, h):
    return {key:(Xp[key] - Xm[key])/(2*h) for key in Xp}


def gradient(sim, set_pars, x0, read, rel_step=1e-5):
    """
    The value of something read from a sim's run, and its derivative with respect to each of a set of values

    The run is differentiated by JAX (forward mode, one direction per value); see the
    module docstring for the rest.

    Args:
        sim (`ss.Sim`): an initialized sim that `reason()` allows, which is left as its run at x0
        set_pars (func): called with an array of values; restarts the sim with them (e.g. `lambda x: sim.restart(pars=dict(beta=x[0]))`)
        x0 (array): the values to work out the derivative at
        read (func): called with the sim after its run; returns a number or an array (or an `sc.objdict` of arrays)
        rel_step (float): the step for the central differences, relative to each value (or absolute, for a value of 0)

    Returns:
        value, derivative: what `read()` returned at x0, and its derivative with respect to each value (a list, one per value, of the same type as the value)
    """
    x0 = np.asarray(x0, dtype=float)
    hs = rel_step*np.where(x0 != 0, np.abs(x0), 1.0)
    mods = sim.declared
    set_pars(x0)
    base = [capture(mod.engine) for mod in mods]
    dirs = [[] for mod in mods] # For each engine, the direction of each value
    for k,h in enumerate(hs):
        sides = []
        for sign in [1, -1]:
            x = x0.copy()
            x[k] += sign*h
            set_pars(x)
            sides.append([capture(mod.engine) for mod in mods])
        for j in range(len(mods)):
            check_same([sides[0][j][:2], sides[1][j][:2], base[j][:2]])
            dirs[j].append(tangent(sides[0][j][2], sides[1][j][2], h))
    set_pars(x0)
    outs = outputs(sim, base)
    slopes = outputs(sim, base, kind='jvp', dX=dirs)
    give(sim, outs)
    value = read(sim.run())
    derivs = []
    for k,h in enumerate(hs):
        sides = []
        for sign in [1, -1]:
            x = x0.copy()
            x[k] += sign*h
            set_pars(x)
            give(sim, outs, shift=[[sign*h*v[k] for v in slope[:3]] for slope in slopes])
            sides.append(read(sim.run()))
        derivs.append(difference(sides[0], sides[1], h))
    set_pars(x0)
    give(sim, outs)
    sim.run() # Leave the sim as its run at x0
    return value, derivs


def difference(a, b, h):
    """ (a - b)/2h, for numbers, arrays, or dicts of them """
    if isinstance(a, dict):
        return sc.objdict({key:difference(a[key], b[key], h) for key in a})
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float))/(2*h)


def find_pars(sim, names=None):
    """ The parameters named (by name, or 'module.name' where needed), or every one that has bounds: an `sc.objdict` of them by name """
    out = sc.objdict()
    for mod in sim.declared:
        for name,p in mod.graph.pars.items():
            if (names is None and p.bounds is not None) or (names is not None and (name in names or f'{mod.name}.{name}' in names)):
                if not any(p is q for q in out.values()): # A parameter shared by several modules is counted once
                    same = [m.graph.pars[name] for m in sim.declared if name in m.graph.pars]
                    out[name if all(q is same[0] for q in same) else f'{mod.name}.{name}'] = p
    missing = [name for name in sc.tolist(names) if not any(name == key or key.endswith(f'.{name}') or name.endswith(f'.{key}') for key in out)] if names is not None else [] # By name, or 'module.name'
    if missing or not out:
        options = [f'{mod.name}.{k}' for mod in sim.declared for k in mod.graph.pars]
        what = f'no parameter named {sc.strjoin(missing)}' if missing else 'no parameter with bounds'
        errormsg = f'Found {what}; parameters are {sc.strjoin(options)}'
        raise ValueError(errormsg)
    for key,p in out.items():
        val = p.raw.value if isinstance(p.raw, ss.TimePar) else p.raw
        if not sc.isnumber(val):
            errormsg = f'Only a parameter that is a single number can be differentiated with respect to, not "{key}" ({p.raw})'
            raise ValueError(errormsg)
    return out


def of_results(sim, results=None, pars=None):
    """ The derivative of results with respect to parameters: see `ss.Sim.gradient()` """
    why = reason(sim)
    if why:
        errormsg = f'Gradients need a sim that runs in JAX, and this one cannot, since {why}'
        raise ValueError(errormsg)
    found = find_pars(sim, None if pars is None else sc.tolist(pars))
    types = [type(p.raw) if isinstance(p.raw, ss.TimePar) else None for p in found.values()]
    x0 = np.array([p.raw.value if cls is not None else p.raw for p,cls in zip(found.values(), types)], dtype=float)
    base = sim.par_values

    def set_pars(x):
        sim.restart(pars=sc.mergedicts(base, {key:(cls(val) if cls is not None else float(val)) for key,val,cls in zip(found.keys(), x, types)}))

    def read(sim):
        out = sc.objdict()
        for mod in sim.declared:
            for key,res in mod.results.items():
                if isinstance(res, ss.Result) and np.ndim(res.values) == 1 and (results is None or key in sc.tolist(results) or f'{mod.name}.{key}' in sc.tolist(results)):
                    out.setdefault(mod.name, sc.objdict())[key] = np.array(res.values, dtype=float)
        return out

    value, derivs = gradient(sim, set_pars, x0, read)
    if not value:
        errormsg = f'Found no result named {sc.strjoin(sc.tolist(results))}'
        raise ValueError(errormsg)
    return sc.objdict(zip(found.keys(), derivs))


#%% The log-density in JAX

class Unsupported(Exception):
    """ Something in the model that the log-density in JAX does not cover yet """
    pass


class Context:
    """ What expressions are evaluated with in JAX: the fitted parameters as JAX values, the others as the engine has them, and (for results) the number in each state """
    def __init__(self, density, engine, x, n=None, series=False):
        self.density = density
        self.engine = engine
        self.x = x
        self.n = n
        self.series = series # Whether a value that varies in time is wanted at every row of the tables, rather than now
        self.xp = jax().numpy
        return

    def value(self, p):
        i = self.density.index.get(id(p))
        if i is not None:
            return self.density.convert(i, self.x[i])
        kept = self.engine.series.get(id(p)) if self.series else None
        if kept is not None: # Rows by groups, as in the engine's tables
            return kept[1]
        return self.engine.value(p)

    def stock(self, state):
        return self.n[state.name]

    def count(self, states):
        return sum(self.stock(s) for s in states)

    def frac(self, states):
        jnp = self.xp
        n = self.count(self.engine.graph.people_states)
        return jnp.where(n > 0, self.count(states)/jnp.where(n > 0, n, 1.0), 0.0)

    local_count = count
    local_frac = frac

    def total_count(self, states):
        return self.xp.sum(self.count(states))

    def exposure(self, state):
        if self.n is None:
            raise Unsupported(f'a rate that depends on "{state.name}" where only parameters were expected')
        return self.stock(state)


class Density:
    """
    The log of the posterior density of a sim's fitted parameters, as one function in JAX, so that it can be compiled and differentiated whole (see `ss.fit()` with method='nuts')

    The numbers in the run are worked out from the parameters in JAX: the values in the rates
    by evaluating the model's own expressions, and the rest (e.g. the compartments at the start)
    from the change in them as each parameter changes, which has to be linear (`ss.fit()` checks
    the result against runs of the sim). Then the run, the results that have data, and the
    likelihood and priors. If anything is not covered, `Unsupported` says what.

    Args:
        sim (`ss.Sim`): an initialized sim that `reason()` allows
        fitted (dict): the fitted parameters, by name
        x0 (array): their values to start from
        set_pars (func): restarts the sim with an array of values
        likelihood (dict): the likelihood of each data series, as for `ss.loglik()`
        weights (dict): a multiplier on each series' log-likelihood
        external (list): for each parameter, whether it belongs to a likelihood given at calibration time (rather than to the model)
        bounds (tuple): the lowest and highest value of each parameter, over which a step is allowed as many substeps as the run needs (default: each parameter's `bounds`)
    """
    def __init__(self, sim, fitted, x0, set_pars, likelihood=None, weights=None, external=None, bounds=None):
        self.sim = sim
        self.fitted = list(fitted.values())
        self.names = list(fitted.keys())
        self.x0 = np.asarray(x0, dtype=float)
        self.set_pars = set_pars
        self.likelihood = likelihood
        self.weights = weights or {}
        self.external = external or [False]*len(x0)
        self.bounds = bounds
        self.mods = list(sim.declared)
        self.engines = [mod.engine for mod in self.mods]
        self.units = units(sim)
        self.index = self.par_index()
        self.init_program()
        self.init_outputs()
        return

    def par_index(self):
        """ The position among the fitted parameters of each parameter object that stands for one """
        out = {}
        for i,(name,p) in enumerate(zip(self.names, self.fitted)):
            out[id(p)] = i
            if not self.external[i]:
                modname, _, key = name.rpartition('.')
                for mod in self.mods:
                    if modname and mod.name != modname:
                        continue
                    graph_par = mod.graph.pars.get(key)
                    if graph_par is not None:
                        out[id(graph_par)] = i
                    if self.sim.modifiers.get((mod.name, 'par', key)):
                        raise Unsupported(f'a modifier of the fitted parameter "{name}"')
        return out

    def convert(self, i, val):
        """ A fitted value as the engine uses it: in the sim's unit """
        jnp = jax().numpy
        raw = self.fitted[i].raw
        if isinstance(raw, ss.prob):
            period = 1.0 if raw.unit is None else raw.unit.to(self.sim.unit_base).value
            return -jnp.log1p(-val)/period
        elif isinstance(raw, ss.TimePar):
            return val*ss.solvers.convert(type(raw)(1.0), self.sim.unit_base)
        elif self.external[i] or sc.isnumber(raw):
            return val
        raise Unsupported(f'the fitted parameter "{self.names[i]}" is a {type(raw).__name__}')

    def corners(self):
        """ The points where the substeps a step needs are worked out: the corners of the bounds (or with more than 4 parameters, each parameter at each of its bounds, the others at their initial values) """
        if self.bounds is not None:
            lows, highs = [np.asarray(b, dtype=float) for b in self.bounds]
        else:
            bound = ss.calibration.bound_value
            lims = [[bound(b, p.raw) for b in p.bounds] if p.bounds is not None else [x, x] for p,x in zip(self.fitted, self.x0)]
            lows, highs = np.array(lims, dtype=float).T
        if len(self.x0) <= 4:
            return [np.array(corner) for corner in itertools.product(*zip(lows, highs))]
        points = []
        for i in range(len(self.x0)):
            for lim in [lows, highs]:
                x = self.x0.copy()
                x[i] = lim[i]
                points.append(x)
        return points

    #%% The numbers in the run

    def capture(self, x):
        """ The numbers in the run of each engine with some parameter values, worked out in Python """
        self.set_pars(x)
        return [capture(engine) for engine in self.engines]

    def init_program(self):
        """ Work out which numbers in the run depend on which parameters, and how """
        captured = self.capture(self.x0)
        self.programs = [c[0] for c in captured]
        self.hows = [c[1] for c in captured]
        self.X0 = [c[2] for c in captured]
        keys = [Run.key(program, *how) for program,how in zip(self.programs, self.hows)]
        corners = [] # The numbers at the corners of the bounds, where the run is laid out as at the start
        for x in self.corners():
            captured = self.capture(x)
            if all(Run.key(program, *how) == key for (program,how,X),key in zip(captured, keys)):
                corners.append([c[2] for c in captured])

        # The run of each set of engines that are run together, with room to spare for the substeps
        self.runs = []
        for unit in self.units:
            if len(unit) == 1:
                j = unit[0]
                base = runner(self.engines[j], self.programs[j], self.hows[j])
                args = self.X0[j]
            else:
                cross = self.engines[unit[0]].cross
                base = joint_runner(cross, [(self.programs[j], self.hows[j]) for j in unit])
                args = tuple(self.X0[j] for j in unit)
            if not base.n_sub: # The most a step needs at the start and at the corners of the bounds, with room to spare, so that a draw anywhere within them gets the substeps it needs
                need = np.asarray(call(base, 'run', args)[3]).max() # As NumPy, since a JAX array's max() outside x64 warns that it is truncated
                for Xs in corners:
                    need = max(need, np.asarray(call(base, 'run', Xs[unit[0]] if len(unit) == 1 else tuple(Xs[j] for j in unit))[3]).max())
                n_max = int(2**np.ceil(np.log2(max(2*need, 1))))
            else:
                n_max = base.n_sub
            if len(unit) == 1:
                self.runs.append(Run(self.programs[unit[0]], *self.hows[unit[0]], n_max=n_max))
            else:
                self.runs.append(joint_runner(cross, [(self.programs[j], self.hows[j]) for j in unit], n_max=n_max))

        # The values in the rates that are fitted, and the flows whose rate is worked out from fitted parameters alone
        self.consts = []
        self.fixed = []
        for engine in self.engines:
            self.consts.append({k:self.index[id(p)] for k,p in enumerate(engine.prog_pars) if id(p) in self.index})
            fixed = []
            for f,flow in enumerate(engine.flows):
                if flow.dynamic or flow.kind == 'conveyor': # The flow out of a conveyor has no rate of its own (see ss.kernels.convey())
                    continue
                dw = engine.dwells.get(flow.tr.name) if flow.expr is None else None
                leaves = flow.expr.leaves if flow.expr is not None else [dw.par] if dw is not None and dw.dist is None else []
                if dw is not None and dw.dist is not None and any(id(leaf) in self.index for leaf in dw.par.leaves):
                    raise Unsupported(f'a fitted parameter of the distribution of a dwell time ("{flow.tr.name}")')
                if any(id(leaf) in self.index for leaf in leaves):
                    mult = np.broadcast_to(engine.flow_mult(flow), (engine.n_groups,)).astype(float)
                    fixed.append((f, flow, dw, mult))
            self.fixed.append(fixed)

        # The rest, by how they change with each parameter
        h = 1e-4*np.where(self.x0 != 0, np.abs(self.x0), 1.0)
        self.slopes = [{} for engine in self.engines]
        for i in range(len(self.x0)):
            sides = []
            for sign in [1, -1]:
                x = self.x0.copy()
                x[i] += sign*h[i]
                captured = self.capture(x)
                if any(Run.key(program, *how) != key for (program,how,X),key in zip(captured, keys)):
                    raise Unsupported(f'the fitted parameter "{self.names[i]}" changes how the model is laid out')
                sides.append([c[2] for c in captured])
            for j,X0 in enumerate(self.X0):
                for key in X0:
                    slope = (sides[0][j][key] - sides[1][j][key])/(2*h[i])
                    if np.any(slope != 0):
                        self.slopes[j].setdefault(key, {})[i] = slope
        self.set_pars(self.x0)
        return

    def numbers(self, x):
        """ The numbers in the run of each engine, as JAX arrays, at some parameter values """
        jnp = jax().numpy
        out = []
        for j,engine in enumerate(self.engines):
            X = {}
            for key,X0 in self.X0[j].items():
                val = jnp.asarray(X0)
                for i,slope in self.slopes[j].get(key, {}).items(): # Linear in each parameter (checked in ss.fit()); the values in the rates are replaced below by what they exactly are
                    val = val + slope*(x[i] - self.x0[i])
                X[key] = val
            for k,i in self.consts[j].items():
                X['consts'] = X['consts'].at[:, k, :].set(self.convert(i, x[i]))
            ctx = Context(self, engine, x, series=True)
            for f,flow,dw,mult in self.fixed[j]:
                if flow.expr is None: # A dwell time: the stages over the mean
                    val = ctx.value(dw.par)
                    mean = 1/val if dw.is_rate else val
                    base = (flow.stages if flow.kind == 'stage' else 1)/mean
                else:
                    base = ss.transitions.evaluate(flow.expr, ctx)
                X['fixed'] = X['fixed'].at[:, f, :].set(base*mult)
            out.append(X)
        return out

    #%% What is compared with the data

    def init_outputs(self):
        """ Which result each data series is compared with, and the matrix that lines it up with the data """
        sim = self.sim
        liks = ss.calibration.likelihoods(sim, likelihood=self.likelihood)
        lined = ss.calibration.lineup(sim)
        self.member = [np.array([[comp.state.name == name for comp in engine.comps] for name in engine.graph.states], dtype=float) for engine in self.engines] # Which compartments each state is
        self.series = []
        for col in sim.data.columns:
            base, dims = ss.results.parse_column(col)
            res = ss.results.match_name(sim, base)
            owners = [j for j,mod in enumerate(self.mods) if mod.results.get(res.name) is res] # By the result itself, since its module is a label
            keep = None
            if dims: # The groups of a stratum
                keep = np.ones(sim.groups.n_groups)
                for dim,level in ss.results.column_levels(sim, dims).items():
                    d = sim.groups.dim_index(dim)
                    keep = keep*(sim.groups.group_levels[:, d] == sim.groups.strats[d].index(level))
            if not owners:
                raise Unsupported(f'the data series "{col}", which is not a result of a declared disease')
            j = owners[0]
            engine = self.engines[j]
            graph = engine.graph
            name = res.name
            staying = name.startswith('n_') and name[2:] in engine.outcomes # The number in an outcome that people stay in
            if not (name.startswith('n_') and name[2:] in graph.states) and not staying and name not in graph.transitions and name not in graph.outcomes and name not in graph.exprs:
                raise Unsupported(f'the data series "{col}" is the result "{name}", which is not a state, a transition, an outcome, or an expression')
            if (name in graph.outcomes and name not in engine.outcomes) or (staying and engine.outcomes[name[2:]].stay is None and engine.outcomes[name[2:]].p_leave is None):
                raise Unsupported(f'the outcome "{name}"')
            W, expected, rows = lined[col]
            self.series.append(sc.objdict(col=col, j=j, name=name, keep=keep, W=W, expected=expected, lik=liks[col], weight=ss.calibration.series_weight(self.weights, col)))
        return

    def outcomes(self, engine, ctx, new, npts):
        """ The outcomes, as `ss.solvers.Compartmental.derived_outcomes()` works them out """
        jnp = jax().numpy
        out = {}
        for name,info in engine.outcomes.items():
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.transitions.Expr):
                    weight = ss.transitions.evaluate(weight, ctx)
                if np.ndim(weight) > 0:
                    raise Unsupported(f'the outcome "{name}", whose weight differs by group')
                amount = amount + flow*weight
            kernel = info.kernel
            now = amount*kernel[0] if len(kernel) == 1 else jnp.convolve(amount, jnp.asarray(kernel))[:npts]
            if info.stay is not None: # Each arrival leaves after a stay drawn for it
                leaving = jnp.convolve(now, jnp.asarray(info.stay))[:npts]
                out[f'n_{name}'] = jnp.cumsum(now - leaving)
            elif info.p_leave is not None: # A share leaves at each step: the arrivals of each step, decaying since
                decay = (1 - np.asarray(info.p_leave, dtype=float))**np.arange(npts)
                out[f'n_{name}'] = jnp.convolve(now, jnp.asarray(decay))[:npts]
            out[name] = now
        return out

    def results(self, x, stocks, flows):
        """ The results that have data, from the run of each engine: a dict by data column """
        out = {}
        for j,engine in enumerate(self.engines):
            graph = engine.graph
            for keep in {None if s.keep is None else tuple(s.keep) for s in self.series if s.j == j}: # Everyone, or the groups of each stratum with data
                series = [s for s in self.series if s.j == j and (None if s.keep is None else tuple(s.keep)) == keep]
                weight = 1.0 if keep is None else np.array(keep)
                counts = (stocks[j]*weight).sum(axis=-1) @ self.member[j].T # Steps by states
                totals = (flows[j]*weight).sum(axis=-1)
                n = {name:counts[:, s] for s,name in enumerate(graph.states)}
                new = {name:totals[:, ind] for name,ind in engine.tr_index.items()}
                ctx = Context(self, engine, x, n=n)
                outcomes = self.outcomes(engine, ctx, new, counts.shape[0]) if any(s.name in graph.outcomes or s.name[2:] in graph.outcomes for s in series) else {}
                for s in series:
                    name = s.name
                    if name.startswith('n_') and name[2:] in n:
                        val = n[name[2:]]
                    elif name in new:
                        val = new[name]
                    elif name in outcomes:
                        val = outcomes[name]
                    else:
                        val = ss.transitions.evaluate(graph.exprs[name], ctx)
                    out[s.col] = val
        return out

    def loglik(self, dist, observed, model, x):
        """ As `ss.calibration.logpdf()`, in JAX, with the fitted parameters of the likelihood at their values in x """
        def par(val):
            i = self.index.get(id(val)) if isinstance(val, ss.par) else None
            return x[i] if i is not None else (val.raw if isinstance(val, ss.par) else val)
        return ss.calibration.logpdf(dist, observed, model, xp=jax().numpy, pars={key:par(val) for key,val in dist.pars.items()})

    #%% The whole of it

    def reverse(self):
        """ Make the runs differentiable in reverse mode (each step takes its most substeps), for `jax.grad`; forward mode, as NUTS uses, is faster without """
        for run in self.runs:
            for one in getattr(run, 'runs', [run]):
                one.fixed = True
        return self

    def states(self, x):
        """ The number in each state at each time point, at some parameter values: a dict of JAX arrays by 'module.n_state' """
        Xs = self.numbers(x)
        out = {}
        for unit,run in zip(self.units, self.runs):
            args = Xs[unit[0]] if len(unit) == 1 else tuple(Xs[j] for j in unit)
            stocks = run.function(clamp=True)(args)[1]
            for k,j in enumerate(unit):
                st = stocks if len(unit) == 1 else stocks[k]
                total = st.sum(axis=-1) # Over the groups
                for i,name in enumerate(self.engines[j].graph.states):
                    out[f'{self.mods[j].name}.n_{name}'] = total @ self.member[j][i]
        return out

    def __call__(self, x):
        """ The log of the posterior density at some parameter values (a JAX array), up to a constant """
        Xs = self.numbers(x)
        stocks = [None]*len(self.engines)
        flows = [None]*len(self.engines)
        for unit,run in zip(self.units, self.runs):
            if len(unit) == 1:
                y_end, stocks[unit[0]], flows[unit[0]], n = run.function(clamp=True)(Xs[unit[0]])
            else:
                y_end, unit_stocks, unit_flows, n = run.function(clamp=True)(tuple(Xs[j] for j in unit))
                for k,j in enumerate(unit):
                    stocks[j], flows[j] = unit_stocks[k], unit_flows[k]
        results = self.results(x, stocks, flows)
        total = 0.0
        for s in self.series:
            model = jax().numpy.asarray(s.W) @ results[s.col]
            total = total + s.weight*self.loglik(s.lik, s.expected, model, x).sum()
        for i in range(len(self.fitted)):
            total = total + ss.calibration.logprior(self.fitted[i], x[i], xp=jax().numpy)
        return total


class LogDensity:
    """
    The log of the posterior density of a sim's fitted parameters as a function of their values, in JAX, for use outside `sim.calibrate()`: see `ss.Sim.log_density()`

    Attributes:
        names (list): the parameters, in the order of the values
        x0 (array): their current values
        fn (func): the function itself, of a JAX array of values, which can be composed with JAX (`jax.grad`, `jax.vmap`, ...)
    """
    def __init__(self, sim, pars=None, likelihood=None):
        why = reason(sim)
        if why:
            errormsg = f'The log-density in JAX needs a sim that runs in JAX, and this one cannot, since {why}'
            raise ValueError(errormsg)
        fitted = find_pars(sim, None if pars is None else sc.tolist(pars))
        self.names = list(fitted.keys())
        self.x0 = np.array([p.raw.value if isinstance(p.raw, ss.TimePar) else p.raw for p in fitted.values()], dtype=float)
        types = [type(p.raw) if isinstance(p.raw, ss.TimePar) else None for p in fitted.values()]
        def set_pars(x):
            sim.restart(pars={key:(cls(val) if cls is not None else float(val)) for key,val,cls in zip(self.names, x, types)})
        self.density = Density(sim, fitted, self.x0, set_pars, likelihood=likelihood).reverse()
        self.fn = self.density
        jx = jax()
        with jx.enable_x64(True):
            self._value = jx.jit(self.density)
            self._grad = jx.jit(jx.grad(self.density))
            self._states = jx.jit(self.density.states)
        return

    def __call__(self, x):
        """ The log-density at some values """
        with jax().enable_x64(True):
            return float(self._value(jax().numpy.asarray(x, dtype=float)))

    def grad(self, x):
        """ Its gradient, by reverse-mode automatic differentiation (one pass, whatever the number of parameters) """
        with jax().enable_x64(True):
            return np.asarray(self._grad(jax().numpy.asarray(x, dtype=float)))

    def run(self, x):
        """ The number in each state at each time point, at some values: a dict of arrays by 'module.n_state' """
        with jax().enable_x64(True):
            return {key:np.asarray(val) for key,val in self._states(jax().numpy.asarray(x, dtype=float)).items()}

