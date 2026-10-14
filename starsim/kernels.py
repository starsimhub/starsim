"""
Numba kernels for the compartmental solvers

A compartmental engine flattens its flows into a `Program`: a set of arrays that
say where each flow comes from, where it goes, and how its rate is worked out.
The kernels here then take a whole solver step in one call, instead of one NumPy
call per flow, per term, and per stage of the integrator.

A rate that depends on the state of the model within a step (a force of infection,
or anything that uses `I.n`) is stored as a short list of operations on a stack, one
value per group. Everything else is a number per group that Python sets once a step.
"""

import collections
import numpy as np
import numba as nb

__all__ = ['Program', 'opcodes']

# The operations a rate can be made of
CONST, COUNT, FRAC, EXPOSURE, ADD, SUB, MUL, DIV, POW = range(9)
opcodes = {'+':ADD, '-':SUB, '*':MUL, '/':DIV, '**':POW}

fields = [
    # The flows
    'n_trans',    # The number of named transitions
    'src',        # The compartment each flow leaves (-1 for an inflow)
    'named',      # Whether each flow is counted as its transition
    'index',      # The transition each flow belongs to
    'record',     # The map (in maps) to the groups each flow is recorded in, or -1
    'inflow',     # For an inflow, the share that arrives in each group
    'fixed',      # The rate of each flow that does not change within a step, by group
    'mult',       # The multiplier on each flow whose rate does, by group
    # The rates that change within a step
    'depth',      # The largest stack any of them needs
    'starts',     # Where each flow's operations start and end (the same, if it has none)
    'ops',        # The operations
    'args',       # What each one acts on
    'consts',     # The parameter values, by group
    'set_starts', # Where each set of compartments (the ones counted together) starts and ends
    'set_comps',  # Their compartments
    'people',     # The compartments that hold people
    'places',     # The place of each group (all zero if the model has none)
    'n_places',
    # Exposure to each infectious state
    'exp_set',    # The set of compartments that are the state
    'rel_trans',  # Its infectiousness, by group
    'levels',     # For each mixing term, the cell of its matrix that each group is in
    'sizes',      # The number of cells in each term
    'contacts',   # The contacts per unit time of each term
    'use_pre',    # Whether people are moved to where they spend their time before prevalence is worked out
    'pre',
    'use_post',   # Whether prevalence is mixed between cells
    'post',
    # Where the flows go
    'tgt_starts', # Where each flow's targets start and end
    'tgt_comp',   # The compartment
    'tgt_w',      # The share that goes there, by group
    'tgt_map',    # The map to the groups they land in, or -1
    'crd_starts', # The same for the transitions that a flow is also counted as
    'crd_index',
    'crd_w',
    'maps',       # Groups by groups: where people land when they change group as they change state
    # Movement between groups
    'mov_starts', # Where each mover's compartments start and end
    'mov_comps',
    'mov_mat',    # Its rates, from group to group
    # For the stochastic solvers: the flows in the order they are drawn, grouped by the compartment they leave
    'src_starts',
    'src_flows',
]

Program = collections.namedtuple('Program', fields)

jit = nb.jit(nopython=True, cache=True, error_model='numpy')


@jit
def workspace(y, p):
    """ The arrays that the kernels work in, made once so that nothing is allocated inside a step """
    nc, ng = y.shape
    nflows = p.n_trans + len(p.mov_mat)
    size = max(p.pre.shape[1], p.n_places)
    rates     = np.empty(p.fixed.shape)         # The rate of each flow, by group
    exposures = np.empty((len(p.exp_set), ng))  # Exposure to each infectious state
    stack     = np.empty((p.depth, ng))         # For working out a rate
    groups    = np.empty((6, ng))               # Scratch, by group
    cells     = np.empty((5, size))             # Scratch, by cell of a mixing term or by place
    dy        = np.empty((4, nc, ng))           # The stages of the integrator
    df        = np.empty((4, nflows, ng))
    yt        = np.empty((nc, ng))
    return rates, exposures, stack, groups, cells, dy, df, yt


@jit
def binomial(n, prob):
    """ The number of n people that something with a probability happens to """
    if n <= 0 or prob <= 0:
        return 0.0
    elif prob >= 1:
        return float(n)
    return float(np.random.binomial(int(n), prob))


@jit
def land(n, m, maps, out):
    """ Add some arrivals to the groups they land in: a draw, where they are shared between groups """
    ng = len(n)
    if m < 0:
        for g in range(ng):
            out[g] += n[g]
        return
    for g in range(ng):
        left = n[g]
        rest = 1.0
        for h in range(ng):
            w = maps[m, g, h]
            if left <= 0:
                break
            elif w >= rest - 1e-12: # The last place they can go
                out[h] += left
                left = 0.0
            elif w > 0:
                k = binomial(left, w/rest)
                out[h] += k
                left -= k
                rest -= w
    return


@jit
def split(n, a, b, weights, credit, out, comps, which, maps, scratch):
    """ Share n whole events between the targets of a flow (or the transitions it is also counted as), at random """
    ng = len(n)
    total, left, drawn = scratch[0], scratch[1], scratch[2]
    if b - a == 1 and weights[a].min() >= 1 - 1e-12:
        land(n, -1 if credit else which[a], maps, out[comps[a]])
    elif b > a:
        whole = True
        for g in range(ng):
            total[g] = 0
            left[g] = n[g]
            for k in range(a, b):
                total[g] += weights[k, g]
            whole = whole and abs(total[g] - 1) < 1e-9
        for k in range(a, b): # One multinomial draw, as a chain of conditional binomials
            for g in range(ng):
                drawn[g] = binomial(left[g], weights[k, g]/total[g]) if total[g] > 0 else 0.0
                left[g] -= drawn[g]
                total[g] -= weights[k, g]
            land(drawn, -1 if credit else which[k], maps, out[comps[k]])
        if not credit and whole and left.max() > 0: # Rounding only
            land(left, which[b-1], maps, out[comps[b-1]])
    return


@jit
def step_binomial(y, p, w, dt, flows):
    """
    One timestep of the chain-binomial solver: one multinomial draw per compartment over the flows that leave it

    Called by `step()`, which has worked out the rates.
    """
    ng = y.shape[1]
    rates, groups, new = w[0], w[3], w[7]
    total, left, n = groups[3], groups[4], groups[5]
    new[:] = y
    flows[:] = 0
    for s in range(len(p.src_starts) - 1):
        a, b = p.src_starts[s], p.src_starts[s+1]
        src = p.src[p.src_flows[a]]
        if src >= 0:
            for g in range(ng):
                total[g] = 0
                for k in range(a, b):
                    total[g] += max(rates[p.src_flows[k], g], 0.0)
                left[g] = binomial(y[src, g], -np.expm1(-total[g]*dt)) # Everyone who leaves
                new[src, g] -= left[g]
        for k in range(a, b):
            f = p.src_flows[k]
            for g in range(ng):
                if src < 0: # An inflow
                    n[g] = np.random.poisson(max(rates[f, g]*p.inflow[f, g]*dt, 0.0))
                else:
                    hazard = max(rates[f, g], 0.0)
                    n[g] = binomial(left[g], hazard/total[g]) if total[g] > 0 else 0.0
                    total[g] -= hazard
                    left[g] -= n[g]
            if p.named[f]:
                land(n, p.record[f], p.maps, flows[p.index[f]])
            split(n, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups)
            split(n, p.crd_starts[f], p.crd_starts[f+1], p.crd_w, True, flows, p.crd_index, p.tgt_map, p.maps, groups)
    for m in range(len(p.mov_mat)): # Movement between groups
        for g in range(ng):
            for h in range(ng):
                if p.mov_mat[m, g, h] != 0:
                    prob = -np.expm1(-p.mov_mat[m, g, h]*dt)
                    for k in range(p.mov_starts[m], p.mov_starts[m+1]):
                        c = p.mov_comps[k]
                        moved = binomial(new[c, g], prob)
                        new[c, g] -= moved
                        new[c, h] += moved
                        flows[p.n_trans + m, g] += moved
    y[:] = new
    return


@jit
def step(y, p, w, dt, n_sub, solver, flows):
    """
    Take one timestep

    The ODE is stepped by Euler's method or fourth-order Runge-Kutta. Everything is in
    this one function, and written as loops over single numbers, because that is many
    times faster for a small model: a call that is passed the program costs more than
    working out the rates of a few compartments does.

    Args:
        y (array): the compartments, by group; changed in place
        p (Program): the flows
        w (tuple): the arrays to work in: see `workspace()`
        dt (float): the timestep
        n_sub (int): for the ODE, the number of substeps (if 0, enough that no compartment turns over by more than a quarter in one)
        solver (int): which of `solvers` to use
        flows (array): filled in with the total of each transition and mover over the step
    """
    rates, exposures, stack, groups, cells, dy, df, yt = w
    n, local_n, vals, infectious = groups[0], groups[1], groups[2], groups[3]
    num, den, num2, den2, prev = cells[0], cells[1], cells[2], cells[3], cells[4]
    nc, ng = y.shape
    nf = len(p.src)
    nflows = p.n_trans + len(p.mov_mat)
    stages = 4 if solver == 0 else 1
    h = dt
    sub = 0
    while True:
        for stage in range(stages):
            ys = y if stage == 0 else yt # Where the rates are worked out

            # The rate of every flow, by group: per member of its source compartment, or in total for an inflow
            for f in range(nf):
                for g in range(ng):
                    rates[f, g] = p.fixed[f, g]
            if len(p.ops):
                for g in range(ng): # The people in each group, and in the same place as each group
                    n[g] = 0
                    for c in p.people:
                        n[g] += ys[c, g]
                for i in range(p.n_places):
                    num[i] = 0
                for g in range(ng):
                    num[p.places[g]] += n[g]
                for g in range(ng):
                    local_n[g] = num[p.places[g]]

                for e in range(len(p.exp_set)): # Exposure to each infectious state through each mixing term: see ss.solvers.Mixing.exposure()
                    s = p.exp_set[e]
                    for g in range(ng):
                        exposures[e, g] = 0
                        infectious[g] = 0
                        for k in range(p.set_starts[s], p.set_starts[s+1]):
                            infectious[g] += ys[p.set_comps[k], g]*p.rel_trans[e, g]
                    for t in range(len(p.sizes)):
                        size = p.sizes[t]
                        for i in range(size):
                            num[i] = 0
                            den[i] = 0
                        for g in range(ng):
                            num[p.levels[t, g]] += infectious[g]
                            den[p.levels[t, g]] += n[g]
                        if p.use_pre[t]: # Move people to where they spend their time
                            for j in range(size):
                                num2[j] = 0
                                den2[j] = 0
                                for i in range(size):
                                    num2[j] += num[i]*p.pre[t, i, j]
                                    den2[j] += den[i]*p.pre[t, i, j]
                            for i in range(size):
                                num[i] = num2[i]
                                den[i] = den2[i]
                        for i in range(size):
                            prev[i] = num[i]/den[i] if den[i] > 0 else 0.0
                        if p.use_post[t]: # Mix between cells
                            for i in range(size):
                                num[i] = 0
                                for j in range(size):
                                    num[i] += p.post[t, i, j]*prev[j]
                            for i in range(size):
                                prev[i] = num[i]
                        for g in range(ng):
                            exposures[e, g] += p.contacts[t]*prev[p.levels[t, g]]

                for f in range(nf): # The rates that are worked out from these, as operations on a stack
                    top = -1
                    for k in range(p.starts[f], p.starts[f+1]):
                        op = p.ops[k]
                        arg = p.args[k]
                        if op == CONST:
                            top += 1
                            for g in range(ng):
                                stack[top, g] = p.consts[arg, g]
                        elif op == EXPOSURE:
                            top += 1
                            for g in range(ng):
                                stack[top, g] = exposures[arg, g]
                        elif op == COUNT or op == FRAC: # The number in some compartments, in the same place
                            top += 1
                            for i in range(p.n_places):
                                num[i] = 0
                            for g in range(ng):
                                vals[g] = 0
                                for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    vals[g] += ys[p.set_comps[i], g]
                                num[p.places[g]] += vals[g]
                            for g in range(ng):
                                stack[top, g] = num[p.places[g]]/local_n[g] if op == FRAC else num[p.places[g]]
                        else:
                            top -= 1
                            for g in range(ng):
                                a = stack[top, g]
                                b = stack[top+1, g]
                                if   op == ADD: stack[top, g] = a + b
                                elif op == SUB: stack[top, g] = a - b
                                elif op == MUL: stack[top, g] = a * b
                                elif op == DIV: stack[top, g] = a / b
                                else:           stack[top, g] = a ** b
                    if top >= 0:
                        for g in range(ng):
                            rates[f, g] = stack[0, g]*p.mult[f, g]

            if sub == 0 and stage == 0:
                if solver == 2: # The stochastic solver takes it from here
                    step_binomial(y, p, w, dt, flows)
                    return
                if n_sub <= 0: # Enough substeps that no compartment turns over by more than a quarter in one
                    fastest = 0.0
                    for f in range(nf):
                        if p.src[f] >= 0:
                            for g in range(ng):
                                fastest = max(fastest, rates[f, g])
                    for m in range(len(p.mov_mat)):
                        for g in range(ng):
                            for k in range(ng):
                                fastest = max(fastest, p.mov_mat[m, g, k])
                    n_sub = max(int(np.ceil(fastest*dt/0.25)), 1)
                h = dt/n_sub
                for f in range(nflows):
                    for g in range(ng):
                        flows[f, g] = 0

            # The rate of change of each compartment, and the rate of each transition and mover
            for c in range(nc):
                for g in range(ng):
                    dy[stage, c, g] = 0
            for f in range(nflows):
                for g in range(ng):
                    df[stage, f, g] = 0
            for m in range(len(p.mov_mat)):
                for k in range(p.mov_starts[m], p.mov_starts[m+1]):
                    c = p.mov_comps[k]
                    for g in range(ng):
                        for j in range(ng):
                            r = p.mov_mat[m, g, j]
                            if r != 0:
                                dy[stage, c, g] -= ys[c, g]*r
                                dy[stage, c, j] += ys[c, g]*r
                                df[stage, p.n_trans + m, g] += ys[c, g]*r
            for f in range(nf):
                s = p.src[f]
                for g in range(ng):
                    if s >= 0:
                        amount = rates[f, g]*ys[s, g]
                        dy[stage, s, g] -= amount
                    else:
                        amount = rates[f, g]*p.inflow[f, g]
                    if amount != 0:
                        if p.named[f]:
                            if p.record[f] < 0:
                                df[stage, p.index[f], g] += amount
                            else:
                                for j in range(ng):
                                    df[stage, p.index[f], j] += amount*p.maps[p.record[f], g, j]
                        for k in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                            if p.tgt_map[k] < 0:
                                dy[stage, p.tgt_comp[k], g] += amount*p.tgt_w[k, g]
                            else:
                                for j in range(ng):
                                    dy[stage, p.tgt_comp[k], j] += amount*p.tgt_w[k, g]*p.maps[p.tgt_map[k], g, j]
                        for k in range(p.crd_starts[f], p.crd_starts[f+1]):
                            df[stage, p.crd_index[k], g] += amount*p.crd_w[k, g]

            if stage < stages - 1: # Where the next stage is worked out
                ahead = h if stage == 2 else h/2
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = y[c, g] + ahead*dy[stage, c, g]

        if stages == 1:
            for c in range(nc):
                for g in range(ng):
                    y[c, g] += h*dy[0, c, g]
            for f in range(nflows):
                for g in range(ng):
                    flows[f, g] += h*df[0, f, g]
        else:
            for c in range(nc):
                for g in range(ng):
                    y[c, g] += h/6*(dy[0, c, g] + 2*dy[1, c, g] + 2*dy[2, c, g] + dy[3, c, g])
            for f in range(nflows):
                for g in range(ng):
                    flows[f, g] += h/6*(df[0, f, g] + 2*df[1, f, g] + 2*df[2, f, g] + df[3, f, g])
        sub += 1
        if sub >= n_sub:
            break
    for c in range(nc):
        for g in range(ng):
            if y[c, g] < 0:
                y[c, g] = 0
    return


@jit
def seed(value):
    """ Seed the random numbers that the kernels draw """
    np.random.seed(value)
    return


#%% A whole run

solvers = dict(rk4=0, euler=1, binomial=2)

@jit
def set_step(p, ti, fixed, mult, consts, rel_trans):
    """ Set the parts of the program that can change over time to their values at a step (each has one entry per step, or a single entry) """
    if ti < len(fixed):     p.fixed[:] = fixed[ti]
    if ti < len(mult):      p.mult[:] = mult[ti]
    if ti < len(consts):    p.consts[:] = consts[ti]
    if ti < len(rel_trans): p.rel_trans[:] = rel_trans[ti]
    return


@jit
def run(y, p, dt, npts, solver, n_sub, seed, fixed, mult, consts, rel_trans):
    """
    Take every step of a run

    Args:
        y (array): the compartments at the start, by group
        p (Program): the flows
        dt (float): the timestep
        npts (int): the number of steps
        solver (int): which of `solvers` to use
        n_sub (int): for the ODE, the number of substeps (see `step()`)
        seed (int): the seed for the random numbers
        fixed, mult, consts, rel_trans (arrays): the values of these parts of the program at each step, or a single
            entry if they are the same throughout

    Returns the compartments at the end, the compartments at the start of each step, and the total of each
    transition and mover over each step.
    """
    np.random.seed(seed)
    y = y.copy()
    w = workspace(y, p)
    stocks = np.empty((npts, y.shape[0], y.shape[1]))
    flows = np.empty((npts, p.n_trans + len(p.mov_mat), y.shape[1]))
    for ti in range(npts):
        set_step(p, ti, fixed, mult, consts, rel_trans)
        stocks[ti] = y
        step(y, p, w, dt, n_sub, solver, flows[ti])
    return y, stocks, flows


@jit
def run_many(y0, p, dt, npts, solver, n_sub, seeds, fixed, mult, consts, rel_trans, state_of, n_states):
    """
    Take every step of many runs that differ only in their seed, keeping the totals over groups

    Args:
        seeds (array): the seed of each run
        state_of (array): the state that each compartment is part of
        n_states (int): the number of states
        (the others are as for `run()`)

    Returns the number in each state at the start of each step, and the total of each transition and
    mover over each step, both as runs by steps by states (or transitions).
    """
    nc, ng = y0.shape
    nflows = p.n_trans + len(p.mov_mat)
    stocks = np.zeros((len(seeds), npts, n_states))
    flows = np.zeros((len(seeds), npts, nflows))
    y = np.empty_like(y0)
    w = workspace(y, p)
    now = np.empty((nflows, ng))
    for r in range(len(seeds)):
        np.random.seed(seeds[r])
        y[:] = y0
        for ti in range(npts):
            set_step(p, ti, fixed, mult, consts, rel_trans)
            for c in range(nc):
                for g in range(ng):
                    stocks[r, ti, state_of[c]] += y[c, g]
            step(y, p, w, dt, n_sub, solver, now)
            for f in range(nflows):
                for g in range(ng):
                    flows[r, ti, f] += now[f, g]
    return stocks, flows


@jit
def one_step(y, p, dt, solver, n_sub):
    """ Take one step, and return the compartments after it and the total of each transition and mover over it: see `run()` """
    y = y.copy()
    flows = np.empty((p.n_trans + len(p.mov_mat), y.shape[1]))
    step(y, p, workspace(y, p), dt, n_sub, solver, flows)
    return y, flows


#%% For agents

@jit
def levels(raw):
    """ The level of each agent in a dimension, as whole numbers (0 for an agent with none) """
    out = np.zeros(len(raw), dtype=np.int64)
    for i in range(len(raw)):
        if raw[i] == raw[i]: # Not NaN
            out[i] = int(raw[i])
    return out
