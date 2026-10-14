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
import numpy as np
import sciris as sc
import starsim as ss

__all__ = [] # Used as ss.gradients

# The parts of the program that are numbers, and so can be differentiated (the rest says how the flows are put together)
diff_fields = ['inflow', 'contacts', 'pre', 'post', 'tgt_w', 'crd_w', 'maps', 'mov_mat']
table_fields = ['fixed', 'mult', 'consts', 'rel_trans'] # As tables over the steps: see ss.Compartmental.tables()
methods = ['rk4', 'euler', 'jax'] # The integrators that the JAX run can take the place of ('jax' is RK4)


def jax():
    """ Import JAX, or say that it is needed """
    try:
        import jax
    except ImportError as e:
        errormsg = 'This needs JAX, which is not installed: pip install jax'
        raise ImportError(errormsg) from e
    return jax


def available():
    """ Whether JAX is installed """
    try:
        import jax # noqa: F401
        return True
    except ImportError:
        return False


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
        return

    def rates(self, ys, vals):
        """ The rate of every flow, by group, with the compartments at ys and the program's values in vals: as `ss.kernels.step()` works them out """
        jnp = jax().numpy
        p = self.p
        fixed, mult, consts, rel_trans, X = vals
        if not self.dynamic:
            return fixed
        n = ys[p.people].sum(axis=0) # The people in each group
        local_n = self.place.T @ (self.place @ n) # ... and in the same place as each group
        exposures = []
        for e in range(len(p.exp_set)): # Exposure to each infectious state through each mixing term
            infectious = (ys[self.sets[p.exp_set[e]]]*rel_trans[e]).sum(axis=0)
            expo = 0.0
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
                    local = self.place.T @ (self.place @ ys[self.sets[arg]].sum(axis=0))
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
        amount = jnp.where((p.src >= 0)[:, None], rates*ys[src], rates*X['inflow'])
        if self.solver == 1: # Euler: outflows that would take more than everyone in a compartment are scaled down together
            total = jnp.zeros((nc, self.ng)).at[p.src[self.leaves]].add(rates[self.leaves]*h)
            scale = jnp.where(total > 1, 1/jnp.where(total > 1, total, 1.0), 1.0)
            amount = jnp.where((p.src >= 0)[:, None], amount*scale[src], amount)
        dy = dy.at[p.src[self.leaves]].add(-amount[self.leaves])
        named = np.flatnonzero(p.named)
        plain = named[p.record[named] < 0]
        mapped = named[p.record[named] >= 0]
        df = df.at[p.index[plain]].add(amount[plain])
        if len(mapped):
            df = df.at[p.index[mapped]].add(jnp.einsum('kg,kgh->kh', amount[mapped], X['maps'][p.record[mapped]]))
        if len(p.tgt_comp):
            moved = amount[self.tgt_flow]*X['tgt_w']
            plain = np.flatnonzero(p.tgt_map < 0)
            mapped = np.flatnonzero(p.tgt_map >= 0)
            dy = dy.at[p.tgt_comp[plain]].add(moved[plain])
            if len(mapped):
                dy = dy.at[p.tgt_comp[mapped]].add(jnp.einsum('kg,kgh->kh', moved[mapped], X['maps'][p.tgt_map[mapped]]))
        if len(p.crd_index):
            df = df.at[p.crd_index].add(amount[self.crd_flow]*X['crd_w'])
        return dy, df

    def function(self):
        """ The run, as a function of a dict of its numbers (see `capture()`): returns the compartments at the end, at the start of each step, the flows over each step, and the most substeps a step needed """
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
                rates0 = self.rates(y, vals(ti, 0.0))
                if self.n_sub:
                    n = self.n_sub
                else: # Enough substeps that no compartment turns over by more than a quarter in one
                    fastest = jnp.max(rates0[self.leaves]) if len(self.leaves) else 0.0
                    if len(self.p.mov_mat):
                        fastest = jnp.maximum(fastest, jnp.max(X['mov_mat']))
                    n = jnp.maximum(jnp.ceil(lax.stop_gradient(fastest)*dt/0.25), 1)
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
                y_end, flows = lax.fori_loop(0, n_max, sub, (y, flows))
                return jnp.maximum(y_end, 0.0), (y, flows, n)

            y_end, (stocks, flows, n) = lax.scan(step, X['y'], jnp.arange(npts))
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
