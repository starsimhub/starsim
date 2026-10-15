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
import importlib.util
import numpy as np
import sciris as sc
import starsim as ss

__all__ = [] # Used as ss.gradients

# The parts of the program that are numbers, and so can be differentiated (the rest says how the flows are put together)
diff_fields = ['inflow', 'contacts', 'pre', 'post', 'tgt_w', 'crd_w', 'maps', 'mov_mat']
table_fields = ['fixed', 'mult', 'consts', 'rel_trans'] # As tables over the steps: see ss.Compartmental.tables()
methods = ['rk4', 'euler', 'jax'] # The integrators that the JAX run can take the place of ('jax' is RK4)


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
    elif sim.integrator not in methods:
        return f'integrator="{sim.integrator}" is not RK4 or Euler'
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
        n_sub (int): the number of substeps a step, or 0 for enough that no compartment turns over by more than a quarter in one
        per_step (int): the rows of the tables for each step (see `ss.Compartmental.per_step()`)
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
        self.nflows = p.n_trans + len(p.mov_mat)
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
            infectious = (ys[self.sets[p.exp_set[e]]]*rel_trans[e]).sum(axis=0)
            expo = 0.0
            if self.simple and not p.use_pre.any() and not p.use_post.any(): # One group, mixing with itself
                for t in range(len(p.sizes)):
                    expo = expo + X['contacts'][t]*jnp.where(n > 0, infectious/jnp.where(n > 0, n, 1.0), 0.0)
                exposures.append(expo)
                continue
            for t in range(len(p.sizes)):
                size = int(p.sizes[t])
                cells = np.zeros((size, self.ng))
                cells[p.levels[t], np.arange(self.ng)] = 1
                num = cells @ infectious
                den = cells @ n
                if p.use_pre[t]: # Move people to where they spend their time
                    pre = X['pre'][t, :size, :size]
                    num = num @ pre
                    den = den @ pre
                prev = jnp.where(den > 0, num/jnp.where(den > 0, den, 1.0), 0.0)
                if p.use_post[t]: # Mix between cells
                    prev = X['post'][t, :size, :size] @ prev
                expo = expo + X['contacts'][t]*prev[p.levels[t]]
            exposures.append(expo)

        rows = [fixed[f] for f in range(self.nf)]
        for f in self.dynamic: # The operations on a stack, as in the kernels
            stack = []
            for k in range(p.starts[f], p.starts[f+1]):
                op, arg = p.ops[k], p.args[k]
                kn = ss.kernels
                if op == kn.CONST:
                    stack.append(consts[arg])
                elif op == kn.EXPOSURE:
                    stack.append(exposures[arg])
                elif op in [kn.COUNT, kn.FRAC]:
                    local = ys[self.sets[arg]].sum(axis=0)
                    if not self.simple:
                        local = self.place.T @ (self.place @ local)
                    stack.append(local/local_n if op == kn.FRAC else local)
                elif op >= kn.EXP:
                    a = stack.pop()
                    func = {kn.EXP:jnp.exp, kn.LOG:jnp.log, kn.SQRT:jnp.sqrt, kn.ABS:jnp.abs, kn.SIN:jnp.sin, kn.COS:jnp.cos}[op]
                    stack.append(func(a))
                else:
                    b = stack.pop()
                    a = stack.pop()
                    if   op == kn.ADD: stack.append(a + b)
                    elif op == kn.SUB: stack.append(a - b)
                    elif op == kn.MUL: stack.append(a * b)
                    elif op == kn.DIV: stack.append(a / b)
                    elif op == kn.POW: stack.append(a ** b)
                    elif op == kn.MIN: stack.append(jnp.minimum(a, b))
                    elif op == kn.MAX: stack.append(jnp.maximum(a, b))
                    elif op == kn.LT:  stack.append(jnp.where(a < b, 1.0, 0.0))
                    elif op == kn.LE:  stack.append(jnp.where(a <= b, 1.0, 0.0))
                    elif op == kn.GT:  stack.append(jnp.where(a > b, 1.0, 0.0))
                    else:              stack.append(jnp.where(a >= b, 1.0, 0.0))
            rows[f] = jnp.broadcast_to(stack[0], (self.ng,))*mult[f]
        return jnp.stack(rows)

    def derivs(self, ys, rates, X, h):
        """ The rate of change of each compartment, and of each transition and mover """
        jnp = jax().numpy
        p = self.p
        nc = ys.shape[0]
        dy = jnp.zeros_like(ys)
        df = jnp.zeros((self.nflows, self.ng))
        for m in range(len(p.mov_mat)): # Movement between groups
            mat = X['mov_mat'][m]
            comps = p.mov_comps[p.mov_starts[m]:p.mov_starts[m+1]]
            y = ys[comps]
            leaving = y*mat.sum(axis=1)
            dy = dy.at[comps].add(y @ mat - leaving)
            df = df.at[p.n_trans + m].add(leaving.sum(axis=0))
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

    def function(self, clamp=False):
        """
        The run, as a function of a dict of its numbers (see `capture()`): returns the compartments at the end, at the end of each step, the flows over each step, and the most substeps a step needed

        Args:
            clamp (bool): if True, a step takes at most `n_max` substeps, so that the run is the same function everywhere (for sampling, where it cannot be taken again with more)
        """
        jx = jax()
        jnp = jx.numpy
        lax = jx.lax
        dt, npts, per_step, n_max = self.dt, self.npts, self.per_step, self.n_max
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

        def run(X):
            def vals(ti, th):
                return [at(X[name], ti, th) for name in table_fields] + [X]

            def step(y, ti):
                if self.n_sub:
                    n = self.n_sub
                else: # Enough substeps that no compartment turns over by more than a quarter in one
                    rates0 = self.rates(y, vals(ti, 0.0))
                    fastest = jnp.max(rates0[self.leaves]) if len(self.leaves) else 0.0
                    if len(self.p.mov_mat):
                        fastest = jnp.maximum(fastest, jnp.max(X['mov_mat']))
                    n = jnp.maximum(jnp.ceil(lax.stop_gradient(fastest)*dt/0.25), 1)
                    if clamp:
                        n = jnp.minimum(n, n_max)
                h = dt/n

                def sub(s, carry):
                    y, flows = carry
                    th = lambda frac: (s + frac)/n
                    dy1, df1 = self.derivs(y, self.rates(y, vals(ti, th(0.0))), X, h)
                    if rk4:
                        dy2, df2 = self.derivs(y + h/2*dy1, self.rates(y + h/2*dy1, vals(ti, th(0.5))), X, h)
                        dy3, df3 = self.derivs(y + h/2*dy2, self.rates(y + h/2*dy2, vals(ti, th(0.5))), X, h)
                        dy4, df4 = self.derivs(y + h*dy3, self.rates(y + h*dy3, vals(ti, th(1.0))), X, h)
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
                else:
                    y_end, flows = lax.fori_loop(0, n_max, sub, (y, flows))
                y_end = jnp.maximum(y_end, 0.0)
                return y_end, (y_end, flows, n) # The compartments after the step, as the kernels record them

            y_end, (stocks, flows, n) = lax.scan(step, X['y'], jnp.arange(npts - 1)) # Step ti+1 goes from time point ti to ti+1, with the values at the start of it
            stocks = jnp.concatenate([X['y'][None], stocks]) # The start, then the end of each step, as the kernels record them
            flows = jnp.concatenate([jnp.zeros((1,) + flows.shape[1:]), flows])
            return y_end, stocks, flows, jnp.max(n)

        return run

    def get(self, kind='run'):
        """ The compiled run: 'run' for one, 'many' for a batch, 'jvp' for its derivative along a batch of directions """
        if kind not in self.compiled:
            jx = jax()
            run = self.function()
            if kind == 'run':
                fn = jx.jit(run)
            elif kind == 'many':
                fn = jx.jit(jx.vmap(run))
            else:
                fn = jx.jit(lambda X, dX: jx.vmap(lambda d: jx.jvp(run, (X,), (d,))[1][:3])(dX)) # The substeps a step needs have no derivative
            self.compiled[kind] = fn
        return self.compiled[kind]


def capture(engine):
    """ The numbers in an engine's run, as it is set up now (with its current parameter values): the program, how to run it, and the numbers """
    args, tables = engine.run_args()
    y, program, dt, npts, solver, n_sub = args
    X = {name:np.array(getattr(program, name), dtype=float) for name in diff_fields}
    X.update({name:np.array(table, dtype=float) for name,table in zip(table_fields, tables)})
    X['y'] = np.array(y, dtype=float)
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
    out = []
    for j,mod in enumerate(sim.declared):
        engine = mod.engine
        if kind == 'many':
            programs = [(program, how) for program,how,X in Xs[j]]
            check_same(programs)
            args = [stack([X for program,how,X in Xs[j]])]
        else:
            programs = [Xs[j][:2]]
            args = [Xs[j][2]] + ([stack(dX[j])] if kind == 'jvp' else [])
        run = runner(engine, *programs[0])
        out.append([np.asarray(v) for v in call(run, kind, *args)])
    return out


def give(sim, outs, shift=None):
    """ Give each engine the output of its run, so that `sim.run()` uses it instead of running the kernels (see `ss.Compartmental.run_all()`) """
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

    Returns, for each declared module, its results over the runs, as `ss.Compartmental.run_many()` does.
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
        if not (sim.strats != [] and len(par_sets)*npts*(stocks.shape[2] + flows.shape[2])*engine.n_groups*8 <= engine.max_bytes): # As in run_many()
            stocks, flows = stocks.sum(axis=-1, keepdims=True), flows.sum(axis=-1, keepdims=True)
        runs = []
        for i,pars in enumerate(par_sets): # Results worked out from the values of the parameters in each run (an outcome can depend on them)
            sim.par_values = sc.mergedicts(base, pars)
            sim.set_par_values()
            engine.cache.clear()
            runs.append(engine.collect(stocks[i:i+1], flows[i:i+1]))
        results[mod.name] = sc.objdict({key:(sc.objdict({k:np.concatenate([run[key][k] for run in runs]) for k in runs[0][key]}) if key == 'groups' else np.concatenate([run[key] for run in runs])) for key in runs[0]})
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
    missing = [name for name in sc.tolist(names) if name not in out and not any(key.endswith(f'.{name}') for key in out)] if names is not None else []
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
    def __init__(self, density, engine, x, n=None):
        self.density = density
        self.engine = engine
        self.x = x
        self.n = n
        self.xp = jax().numpy
        return

    def value(self, p):
        i = self.density.index.get(id(p))
        if i is not None:
            return self.density.convert(i, self.x[i])
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
    """
    def __init__(self, sim, fitted, x0, set_pars, likelihood=None, weights=None, external=None):
        self.sim = sim
        self.fitted = list(fitted.values())
        self.names = list(fitted.keys())
        self.x0 = np.asarray(x0, dtype=float)
        self.set_pars = set_pars
        self.likelihood = likelihood
        self.weights = weights or {}
        self.external = external or [False]*len(x0)
        if len(sim.declared) != 1:
            raise Unsupported('a sim with more than one declared disease')
        self.engine = sim.declared[0].engine
        self.mod = self.engine.mod
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
                key = name.rpartition('.')[2]
                graph_par = self.mod.graph.pars.get(key)
                if graph_par is not None:
                    out[id(graph_par)] = i
                if self.sim.modifiers.get((self.mod.name, 'par', key)):
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

    #%% The numbers in the run

    def capture(self, x):
        """ The numbers in the run with some parameter values, worked out in Python """
        self.set_pars(x)
        program, how, X = capture(self.engine)
        return program, how, X

    def init_program(self):
        """ Work out which numbers in the run depend on which parameters, and how """
        engine = self.engine
        self.program, self.how, self.X0 = self.capture(self.x0)
        if self.how[3]: # Given substeps: as many as that
            n_max = self.how[3]
        else: # The most a step needs at the start, with room to spare
            out = call(runner(engine, self.program, self.how), 'run', self.X0)
            n_max = int(2**np.ceil(np.log2(max(2*np.max(out[3]), 1))))
        self.run = Run(self.program, *self.how, n_max=n_max)
        self.consts = {k:self.index[id(p)] for k,p in enumerate(engine.prog_pars) if id(p) in self.index} # The values in the rates that are fitted
        self.fixed = [] # The flows whose rate is worked out from fitted parameters alone
        for f,flow in enumerate(engine.flows):
            if flow.dynamic:
                continue
            dw = engine.dwells.get(flow.tr.name) if flow.expr is None else None
            leaves = flow.expr.leaves if flow.expr is not None else [dw.par] if dw is not None and dw.dist is None else []
            if dw is not None and dw.dist is not None and any(id(leaf) in self.index for leaf in dw.par.leaves):
                raise Unsupported(f'a fitted parameter of the distribution of a dwell time ("{flow.tr.name}")')
            if any(id(leaf) in self.index for leaf in leaves):
                mult = np.broadcast_to(engine.flow_mult(flow), (engine.n_groups,)).astype(float)
                self.fixed.append((f, flow, dw, mult))

        # The rest, by how they change with each parameter
        h = 1e-4*np.where(self.x0 != 0, np.abs(self.x0), 1.0)
        self.slopes = {}
        for i in range(len(self.x0)):
            sides = []
            for sign in [1, -1]:
                x = self.x0.copy()
                x[i] += sign*h[i]
                program, how, X = self.capture(x)
                if Run.key(program, *how) != Run.key(self.program, *self.how):
                    raise Unsupported(f'the fitted parameter "{self.names[i]}" changes how the model is laid out')
                sides.append(X)
            for key in self.X0:
                slope = (sides[0][key] - sides[1][key])/(2*h[i])
                if np.any(slope != 0):
                    self.slopes.setdefault(key, {})[i] = slope
        self.set_pars(self.x0)
        return

    def numbers(self, x):
        """ The numbers in the run, as JAX arrays, at some parameter values """
        jnp = jax().numpy
        X = {}
        for key,X0 in self.X0.items():
            val = jnp.asarray(X0)
            for i,slope in self.slopes.get(key, {}).items(): # Linear in each parameter (checked in check()); the values in the rates are replaced below by what they exactly are
                val = val + slope*(x[i] - self.x0[i])
            X[key] = val
        for k,i in self.consts.items():
            X['consts'] = X['consts'].at[:, k, :].set(self.convert(i, x[i]))
        ctx = Context(self, self.engine, x)
        for f,flow,dw,mult in self.fixed:
            if flow.expr is None: # A dwell time: the stages over the mean
                val = ctx.value(dw.par)
                mean = 1/val if dw.is_rate else val
                base = (flow.stages if flow.kind == 'stage' else 1)/mean
            else:
                base = ss.transitions.evaluate(flow.expr, ctx)
            X['fixed'] = X['fixed'].at[:, f, :].set(base*mult)
        return X

    #%% What is compared with the data

    def init_outputs(self):
        """ Which result each data series is compared with, and the matrix that lines it up with the data """
        sim = self.sim
        engine = self.engine
        graph = engine.graph
        sim_t, data_t = ss.calibration.data_times(sim, sim.data)
        eps = np.median(np.diff(sim_t))/2 if len(sim_t) > 1 else 1e-9
        inside = (data_t >= sim_t.min() - eps) & (data_t <= sim_t.max() + eps)
        liks = ss.calibration.likelihoods(sim, likelihood=self.likelihood)
        self.member = np.array([[comp.state.name == name for comp in engine.comps] for name in graph.states], dtype=float) # Which compartments each state is
        self.series = []
        for col in sim.data.columns:
            res = ss.data.match(sim, col)
            if res.module != self.mod.name:
                raise Unsupported(f'the data series "{col}" is of "{res.module}", which is not the fitted disease')
            name = res.name
            if not (name.startswith('n_') and name[2:] in graph.states) and name not in graph.transitions and name not in graph.outcomes and name not in graph.exprs:
                raise Unsupported(f'the data series "{col}" is the result "{name}", which is not a state, a transition, an outcome, or an expression')
            if name in graph.outcomes and name not in engine.outcomes:
                raise Unsupported(f'the outcome "{name}"')
            expected = np.asarray(sim.data[col], dtype=float)
            ok = inside & np.isfinite(expected)
            npts = len(sim_t)
            if res.summarize_by == 'sum':
                spacing = np.median(np.diff(data_t)) if len(data_t) > 1 else np.inf
                starts = np.append(data_t[0] - spacing, data_t[:-1])
                W = np.array([((sim_t > lo + eps) & (sim_t <= hi + eps)).astype(float) for lo,hi in zip(starts, data_t)])
            else:
                W = np.array([np.interp(data_t, sim_t, row) for row in np.eye(npts)]).T
            self.series.append(sc.objdict(col=col, name=name, W=W[ok], expected=expected[ok], lik=liks[col], weight=self.weights.get(col, 1.0)))
        return

    def outcomes(self, ctx, new, npts):
        """ The outcomes, as `ss.Compartmental.derived_outcomes()` works them out """
        jnp = jax().numpy
        engine = self.engine
        out = {}
        for name,info in engine.outcomes.items():
            if info.stay is not None or info.p_leave is not None:
                raise Unsupported(f'the outcome "{name}", which people stay in')
            amount = 0.0
            for key,weight in info.sources:
                flow = new[key] if key in new else out[key]
                if isinstance(weight, ss.Expr):
                    weight = ss.transitions.evaluate(weight, ctx)
                if np.ndim(weight) > 0:
                    raise Unsupported(f'the outcome "{name}", whose weight differs by group')
                amount = amount + flow*weight
            kernel = info.kernel
            now = amount*kernel[0] if len(kernel) == 1 else jnp.convolve(amount, jnp.asarray(kernel))[:npts]
            if info.every is not None: # Totaled over each reporting period
                ends = np.arange(info.every, npts, info.every)
                totals = jnp.diff(jnp.cumsum(now)[ends], prepend=0.0)
                now = jnp.zeros(npts).at[ends].set(totals)
            out[name] = now
        return out

    def results(self, x, stocks, flows):
        """ The results that have data, from the run """
        engine = self.engine
        graph = engine.graph
        stocks = stocks.sum(axis=-1) @ self.member.T # Steps by states
        flows = flows.sum(axis=-1)
        n = {name:stocks[:, s] for s,name in enumerate(graph.states)}
        new = {name:flows[:, ind] for name,ind in engine.tr_index.items()}
        ctx = Context(self, engine, x, n=n)
        outcomes = self.outcomes(ctx, new, stocks.shape[0]) if any(s.name in graph.outcomes for s in self.series) else {}
        out = {}
        for s in self.series:
            name = s.name
            if name.startswith('n_') and name[2:] in n:
                out[name] = n[name[2:]]
            elif name in new:
                out[name] = new[name]
            elif name in outcomes:
                out[name] = outcomes[name]
            else:
                out[name] = ss.transitions.evaluate(graph.exprs[name], ctx)
        return out

    def loglik(self, dist, observed, model, x):
        """ As `ss.calibration.logpdf()`, in JAX """
        jnp = jax().numpy
        stats = jax().scipy.stats
        def par(val):
            i = self.index.get(id(val)) if isinstance(val, ss.par) else None
            return x[i] if i is not None else (val.raw if isinstance(val, ss.par) else val)
        pars = {key:par(val) for key,val in dist.pars.items()}
        mean = jnp.maximum(model, 1e-9)
        if isinstance(dist, ss.poisson):
            return stats.poisson.logpmf(observed, mean)
        elif isinstance(dist, ss.nbinom):
            n = pars['n']
            return stats.nbinom.logpmf(observed, n, n/(n + mean))
        elif isinstance(dist, ss.normal):
            return stats.norm.logpdf(observed, model, pars['scale'])
        elif isinstance(dist, ss.lognorm_ex):
            sigma2 = jnp.log1p(pars['std']**2/mean**2)
            return stats.norm.logpdf(jnp.log(observed), jnp.log(mean) - sigma2/2, jnp.sqrt(sigma2)) - jnp.log(observed)
        raise Unsupported(f'the likelihood {type(dist).__name__}')

    def logprior(self, i, val):
        """ As `ss.calibration.logprior()`, in JAX """
        jnp = jax().numpy
        stats = jax().scipy.stats
        p = self.fitted[i]
        prior = p.prior
        low, high = p.bounds
        if prior is None:
            return -np.log(high - low)
        elif isinstance(prior, ss.uniform):
            return stats.uniform.logpdf(val, prior.pars.low, prior.pars.high - prior.pars.low)
        frozen = sc.dcp(prior).init(force=True).dist
        name = getattr(getattr(frozen, 'dist', None), 'name', None)
        kw = dict(loc=0.0, scale=1.0) | frozen.kwds
        args = frozen.args
        if name == 'norm':
            return stats.norm.logpdf(val, kw['loc'], kw['scale'])
        elif name == 'lognorm':
            s = args[0] if args else kw['s']
            return stats.norm.logpdf(jnp.log(val - kw['loc']), jnp.log(kw['scale']), s) - jnp.log(val - kw['loc'])
        elif name == 'gamma':
            a = args[0] if args else kw['a']
            return stats.gamma.logpdf(val, a, kw['loc'], kw['scale'])
        elif name == 'beta':
            a, b = args if args else (kw['a'], kw['b'])
            return stats.beta.logpdf(val, a, b, kw['loc'], kw['scale'])
        elif name == 'expon':
            return stats.expon.logpdf(val, kw['loc'], kw['scale'])
        raise Unsupported(f'the prior of "{self.names[i]}" ({type(prior).__name__})')

    #%% The whole of it

    def __call__(self, x):
        """ The log of the posterior density at some parameter values (a JAX array), up to a constant """
        X = self.numbers(x)
        y_end, stocks, flows, n = self.run.function(clamp=True)(X)
        results = self.results(x, stocks, flows)
        total = 0.0
        for s in self.series:
            model = jax().numpy.asarray(s.W) @ results[s.name]
            total = total + s.weight*self.loglik(s.lik, s.expected, model, x).sum()
        for i in range(len(self.fitted)):
            total = total + self.logprior(i, x[i])
        return total
