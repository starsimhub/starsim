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

__all__ = ['Program', 'Register', 'opcodes']

# The operations a rate can be made of
CONST, COUNT, FRAC, EXPOSURE, ADD, SUB, MUL, DIV, POW, MIN, MAX, LT, LE, GT, GE, EXP, LOG, SQRT, ABS, SIN, COS = range(21)
opcodes = {'+':ADD, '-':SUB, '*':MUL, '/':DIV, '**':POW, 'minimum':MIN, 'maximum':MAX, 'exp':EXP, 'log':LOG, 'sqrt':SQRT, 'absolute':ABS, 'sin':SIN, 'cos':COS,
           'less':LT, 'less_equal':LE, 'greater':GT, 'greater_equal':GE}

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
    'has_det',    # Whether there are any
    'det',        # Whether each flow is of a state that is not people (a reservoir), so is taken as its expected value by the stochastic solvers
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
    # For the exact solver when places do not affect each other within a step: see gillespie_places()
    'local',      # Whether they do not
    'pl_starts',  # Where the groups in each place start and end
    'pl_groups',
]

Program = collections.namedtuple('Program', fields)

# A movement register (see ss.Mobility), as the kernels apply it
register_fields = [
    'starts',     # Where each step's events start and end
    'src',        # The place each event takes people from (-1 if it adds them)
    'dst',        # The place it takes them to (-1 if they leave)
    'n',          # The number of people
    'sel',        # The set of states it chooses them from, or adds them to
    'sel_starts', # For each set: where the compartments that people are chosen from start and end
    'sel_comps',
    'sel_to',     # The compartment that the people chosen from each of those go to (the same, unless the event moves them to other states)
    'ent_starts', # For each set: where the compartments that people are added to (the first of each state) start and end
    'ent_comps',
    'ent_w',      # For each of those: the share of the people added to a place who go into each of its groups (by group)
    'pg_starts',  # Where the groups in each place start and end
    'pg_groups',
    'step',       # How far apart adjacent places are, in groups
]

Register = collections.namedtuple('Register', register_fields)

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
def draw(y, p, w, dt, flows, tau):
    """
    Draw whole numbers of events over a time dt, and add them to the flows

    For the chain-binomial solver, one multinomial draw per compartment over the flows
    that leave it. For tau-leaping, a Poisson number of events for each flow. Called
    by `step()`, which has worked out the rates.
    """
    ng = y.shape[1]
    rates, groups, new = w[0], w[3], w[7]
    total, left, n = groups[3], groups[4], groups[5]
    new[:] = y
    for s in range(len(p.src_starts) - 1):
        a, b = p.src_starts[s], p.src_starts[s+1]
        src = p.src[p.src_flows[a]]
        if src >= 0 and p.det[p.src_flows[a]]: # A state that is not people: see settle()
            continue
        if src >= 0 and not tau:
            for g in range(ng):
                total[g] = 0
                for k in range(a, b):
                    total[g] += max(rates[p.src_flows[k], g], 0.0)
                left[g] = binomial(y[src, g], -np.expm1(-total[g]*dt)) # Everyone who leaves
                new[src, g] -= left[g]
        for k in range(a, b):
            f = p.src_flows[k]
            if p.det[f]:
                continue
            for g in range(ng):
                if src < 0: # An inflow
                    n[g] = np.random.poisson(max(rates[f, g]*p.inflow[f, g]*dt, 0.0))
                elif tau: # Cannot take more than are there
                    n[g] = min(np.random.poisson(max(rates[f, g]*y[src, g]*dt, 0.0)), new[src, g])
                    new[src, g] -= n[g]
                else:
                    hazard = max(rates[f, g], 0.0)
                    n[g] = binomial(left[g], hazard/total[g]) if total[g] > 0 else 0.0
                    total[g] -= hazard
                    left[g] -= n[g]
            if p.named[f]:
                land(n, p.record[f], p.maps, flows[p.index[f]])
            split(n, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups)
            split(n, p.crd_starts[f], p.crd_starts[f+1], p.crd_w, True, flows, p.crd_index, p.tgt_map, p.maps, groups)
    for m in range(len(p.mov_mat)): # Movement between groups: as move() does, but written out, since a call here (once a leap) slows tau-leaping by a few percent
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
def place_rates(ys, p, a, b, rates, stack, expo):
    """
    The rate of every flow for the groups of one place (p.pl_groups[a:b]), when people mix only within their place

    The same arithmetic as in `step()`, for one place: exposure is the prevalence in the place, and
    a count is the number in the place.
    """
    nf = len(p.src)
    n = 0.0
    for j in range(a, b):
        g = p.pl_groups[j]
        for c in p.people:
            n += ys[c, g]
    for e in range(len(p.exp_set)):
        s = p.exp_set[e]
        num = 0.0
        for j in range(a, b):
            g = p.pl_groups[j]
            for k in range(p.set_starts[s], p.set_starts[s+1]):
                num += ys[p.set_comps[k], g]*p.rel_trans[e, g]
        prev = num/n if n > 0 else 0.0
        expo[e] = 0.0
        for t in range(len(p.sizes)):
            expo[e] += p.contacts[t]*prev
    for f in range(nf):
        for j in range(a, b):
            g = p.pl_groups[j]
            rates[f, g] = p.fixed[f, g]
            if p.starts[f] == p.starts[f+1]:
                continue
            top = -1
            for k in range(p.starts[f], p.starts[f+1]):
                op = p.ops[k]
                arg = p.args[k]
                if op == CONST:
                    top += 1
                    stack[top, 0] = p.consts[arg, g]
                elif op == EXPOSURE:
                    top += 1
                    stack[top, 0] = expo[arg]
                elif op == COUNT or op == FRAC:
                    top += 1
                    v = 0.0
                    for jj in range(a, b):
                        for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                            v += ys[p.set_comps[i], p.pl_groups[jj]]
                    stack[top, 0] = v/n if (op == FRAC and n > 0) else (0.0 if op == FRAC else v)
                elif op >= EXP:
                    x = stack[top, 0]
                    if   op == EXP:  stack[top, 0] = np.exp(x)
                    elif op == LOG:  stack[top, 0] = np.log(x)
                    elif op == SQRT: stack[top, 0] = np.sqrt(x)
                    elif op == ABS:  stack[top, 0] = abs(x)
                    elif op == SIN:  stack[top, 0] = np.sin(x)
                    else:            stack[top, 0] = np.cos(x)
                else:
                    top -= 1
                    x = stack[top, 0]
                    z = stack[top+1, 0]
                    if   op == ADD: stack[top, 0] = x + z
                    elif op == SUB: stack[top, 0] = x - z
                    elif op == MUL: stack[top, 0] = x * z
                    elif op == DIV: stack[top, 0] = x / z
                    elif op == POW: stack[top, 0] = x ** z
                    elif op == MIN: stack[top, 0] = min(x, z)
                    elif op == MAX: stack[top, 0] = max(x, z)
                    elif op == LT:  stack[top, 0] = 1.0 if x < z else 0.0
                    elif op == LE:  stack[top, 0] = 1.0 if x <= z else 0.0
                    elif op == GT:  stack[top, 0] = 1.0 if x > z else 0.0
                    else:           stack[top, 0] = 1.0 if x >= z else 0.0
            if top >= 0:
                rates[f, g] = stack[0, 0]*p.mult[f, g]
    return


@jit
def gillespie_places(y, p, w, dt, flows):
    """
    The exact solver over a time dt, one place at a time, when places do not affect each other within a step (as SimInf runs each node on its own)

    Each place has its own events, and after each one only the rates of that place are worked
    out again, so the cost of an event does not grow with the number of places.
    """
    rates, stack = w[0], w[2]
    nf = len(p.src)
    expo = np.zeros(max(len(p.exp_set), 1))
    for i in range(len(p.pl_starts) - 1):
        a, b = p.pl_starts[i], p.pl_starts[i+1]
        t = 0.0
        while True:
            place_rates(y, p, a, b, rates, stack, expo)
            total = 0.0
            for f in range(nf):
                for j in range(a, b):
                    total += propensity(y, p, rates, f, p.pl_groups[j])
            if total <= 0:
                break
            t += np.random.exponential(1/total)
            if t > dt:
                break
            target = np.random.random()*total
            cum = 0.0
            which, where = -1, -1
            for f in range(nf):
                for j in range(a, b):
                    g = p.pl_groups[j]
                    r = propensity(y, p, rates, f, g)
                    if r > 0:
                        cum += r
                        which, where = f, g
                        if cum > target:
                            break
                if cum > target:
                    break
            if p.src[which] >= 0:
                y[p.src[which], where] -= 1
            if p.named[which]:
                flows[p.index[which], where] += 1
            ka, kb = p.tgt_starts[which], p.tgt_starts[which+1] # Where it goes: one of its targets, in proportion to their shares
            u = np.random.random()
            tot = 0.0
            for k in range(ka, kb):
                tot += p.tgt_w[k, where]
            cum = 0.0
            for k in range(ka, kb):
                cum += p.tgt_w[k, where]
                if u*tot < cum or k == kb - 1:
                    y[p.tgt_comp[k], where] += 1
                    break
            u = np.random.random() # The transitions it also counts as (immediate branches)
            cum = 0.0
            for k in range(p.crd_starts[which], p.crd_starts[which+1]):
                cum += p.crd_w[k, where]
                if u < cum:
                    flows[p.crd_index[k], where] += 1
                    break
    return


@jit
def settle_places(y, p, dt, flows):
    """
    For the exact solver by place: the counts settled over the step, with the people as they are at its end, after the events of a movement register

    As SimInf updates its continuous state after each day's scheduled events, so that people
    who move count towards the place they arrive at; see `gillespie_places()` and `settle()`.
    """
    w = workspace(y, p)
    expo = np.zeros(max(len(p.exp_set), 1))
    for i in range(len(p.pl_starts) - 1):
        place_rates(y, p, p.pl_starts[i], p.pl_starts[i+1], w[0], w[2], expo)
    settle(y, p, w[0], dt, flows)
    return


@jit
def settle(y, p, rates, dt, flows):
    """
    For the solvers that draw whole events: the flows of states that are not people (a reservoir), as expected values over a time dt

    A count is a real number, not a number of people, so it is never drawn. Its flows are
    integrated over dt by RK4, at the current rates and with the people as they are, so a count
    that is filled at a steady rate and decays settles where it would in continuous time.
    """
    nc, ng = y.shape
    nf = len(p.src)
    nflows = flows.shape[0]
    k = np.zeros((4, nc, ng))
    kf = np.zeros((4, nflows, ng))
    y0 = y.copy()
    yt = y.copy()
    for stage in range(4):
        for f in range(nf):
            if not p.det[f]:
                continue
            s = p.src[f]
            for g in range(ng):
                amount = max(rates[f, g]*(yt[s, g] if s >= 0 else p.inflow[f, g]), 0.0)
                if amount == 0:
                    continue
                if s >= 0:
                    k[stage, s, g] -= amount
                if p.named[f]:
                    if p.record[f] < 0:
                        kf[stage, p.index[f], g] += amount
                    else:
                        for j in range(ng):
                            kf[stage, p.index[f], j] += amount*p.maps[p.record[f], g, j]
                for t in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                    if p.tgt_map[t] < 0:
                        k[stage, p.tgt_comp[t], g] += amount*p.tgt_w[t, g]
                    else:
                        for j in range(ng):
                            k[stage, p.tgt_comp[t], j] += amount*p.tgt_w[t, g]*p.maps[p.tgt_map[t], g, j]
                for t in range(p.crd_starts[f], p.crd_starts[f+1]):
                    kf[stage, p.crd_index[t], g] += amount*p.crd_w[t, g]
        if stage < 3:
            ahead = dt if stage == 2 else dt/2
            for c in range(nc):
                for g in range(ng):
                    yt[c, g] = max(y0[c, g] + ahead*k[stage, c, g], 0.0)
    for c in range(nc):
        for g in range(ng):
            y[c, g] = max(y0[c, g] + dt/6*(k[0, c, g] + 2*k[1, c, g] + 2*k[2, c, g] + k[3, c, g]), 0.0)
    for f in range(nflows):
        for g in range(ng):
            flows[f, g] += dt/6*(kf[0, f, g] + 2*kf[1, f, g] + 2*kf[2, f, g] + kf[3, f, g])
    return


@jit
def move(y, mov_mat, mov_starts, mov_comps, n_trans, dt, flows, whole):
    """ Move people between groups over a time dt: as whole numbers drawn at random, or as the expected share (for the exact solver and the diffusion approximation, once a step) """
    ng = y.shape[1]
    for m in range(len(mov_mat)):
        for g in range(ng):
            for h in range(ng):
                if mov_mat[m, g, h] != 0:
                    prob = -np.expm1(-mov_mat[m, g, h]*dt)
                    for k in range(mov_starts[m], mov_starts[m+1]):
                        c = mov_comps[k]
                        moved = binomial(y[c, g], prob) if whole else y[c, g]*prob
                        y[c, g] -= moved
                        y[c, h] += moved
                        flows[n_trans + m, g] += moved
    return


@jit
def propensity(y, p, rates, f, g):
    """ The rate at which events of a flow happen in a group: per member of its source compartment, or in total for an inflow """
    s = p.src[f]
    if p.det[f]: # Not an event: see settle()
        return 0.0
    return max(rates[f, g]*(y[s, g] if s >= 0 else p.inflow[f, g]), 0.0)


@jit
def event(y, p, w, dt, flows):
    """
    For the exact solver: the time to the next event and, if it comes within dt, the event itself

    Called by `step()`, which has worked out the rates. Returns the time taken, which is
    dt if the next event comes after it.
    """
    nf, ng = len(p.src), y.shape[1]
    rates, groups = w[0], w[3]
    n = groups[5]
    total = 0.0
    for f in range(nf):
        for g in range(ng):
            total += propensity(y, p, rates, f, g)
    if total <= 0:
        return dt
    wait = np.random.exponential(1/total)
    if wait > dt:
        return dt

    # Which event: the last one with any chance, if rounding leaves the draw past the end
    target = np.random.random()*total
    cum = 0.0
    which, where = -1, -1
    for f in range(nf):
        for g in range(ng):
            rate = propensity(y, p, rates, f, g)
            if rate > 0:
                cum += rate
                which, where = f, g
                if cum > target:
                    break
        if cum > target:
            break

    for g in range(ng):
        n[g] = 0.0
    n[where] = 1.0
    if p.src[which] >= 0:
        y[p.src[which], where] -= 1
    if p.named[which]:
        land(n, p.record[which], p.maps, flows[p.index[which]])
    split(n, p.tgt_starts[which], p.tgt_starts[which+1], p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups)
    split(n, p.crd_starts[which], p.crd_starts[which+1], p.crd_w, True, flows, p.crd_index, p.tgt_map, p.maps, groups)
    return wait


@jit
def diffuse(y, p, w, dt, flows):
    """
    For the diffusion approximation (Euler-Maruyama): each flow over a time dt, with Gaussian noise of its Poisson variance

    Called by `step()`, which has worked out the rates.
    """
    ng = y.shape[1]
    rates, new = w[0], w[7]
    new[:] = y
    for f in range(len(p.src)):
        s = p.src[f]
        for g in range(ng):
            if p.det[f]: # A state that is not people: its expected value
                amount = max(rates[f, g]*(y[s, g] if s >= 0 else p.inflow[f, g]), 0.0)*dt
            else:
                mean = propensity(y, p, rates, f, g)*dt
                amount = max(mean + np.sqrt(mean)*np.random.standard_normal(), 0.0)
            if s >= 0: # Cannot take more than are there
                amount = min(amount, new[s, g])
                new[s, g] -= amount
            if amount != 0:
                if p.named[f]:
                    if p.record[f] < 0:
                        flows[p.index[f], g] += amount
                    else:
                        for j in range(ng):
                            flows[p.index[f], j] += amount*p.maps[p.record[f], g, j]
                for k in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                    if p.tgt_map[k] < 0:
                        new[p.tgt_comp[k], g] += amount*p.tgt_w[k, g]
                    else:
                        for j in range(ng):
                            new[p.tgt_comp[k], j] += amount*p.tgt_w[k, g]*p.maps[p.tgt_map[k], g, j]
                for k in range(p.crd_starts[f], p.crd_starts[f+1]):
                    flows[p.crd_index[k], g] += amount*p.crd_w[k, g]
    for c in range(y.shape[0]):
        for g in range(ng):
            y[c, g] = max(new[c, g], 0.0)
    return


@jit
def interp(out, table, row, th):
    """ A part of the program at a fraction th of the way through a step, from its values at the start, middle, and end (rows row to row+2 of its table) """
    a = (2*th - 1)*(th - 1) # Quadratic through the three points, so exact at each stage of a single RK4 step
    b = 4*th*(1 - th)
    c = th*(2*th - 1)
    for i in range(out.shape[0]):
        for g in range(out.shape[1]):
            out[i, g] = a*table[row, i, g] + b*table[row+1, i, g] + c*table[row+2, i, g]
    return


@jit
def step(y, p, w, dt, n_sub, solver, flows, tables, row):
    """
    Take one timestep

    The ODE is stepped by Euler's method or fourth-order Runge-Kutta; the stochastic
    solvers work out the rates here and hand them to `draw()`, `event()`, or `diffuse()`. Everything is in
    this one function, and written as loops over single numbers, because that is many
    times faster for a small model: a call that is passed the program costs more than
    working out the rates of a few compartments does.

    Args:
        y (array): the compartments, by group; changed in place
        p (Program): the flows
        w (tuple): the arrays to work in: see `workspace()`
        dt (float): the timestep
        n_sub (int): the number of substeps; if 0, for the ODE, enough that no compartment turns over by more than a quarter in one (tau-leaping and the exact solver choose their own)
        solver (int): which of `solvers` to use
        flows (array): filled in with the total of each transition and mover over the step
        tables (tuple): the fixed, mult, consts, and rel_trans tables (see `run()`), with three rows per step
        row (int): the row of the start of this step, or -1 if the program holds for the whole step
    """
    rates, exposures, stack, groups, cells, dy, df, yt = w
    n, local_n, vals, infectious = groups[0], groups[1], groups[2], groups[3]
    num, den, num2, den2, prev = cells[0], cells[1], cells[2], cells[3], cells[4]
    nc, ng = y.shape
    nf = len(p.src)
    nflows = p.n_trans + len(p.mov_mat)
    stages = 4 if solver == 0 else 1
    h = dt
    timed = solver == 3 or solver == 4 # Tau-leaping and the exact solver take steps of their own length
    elapsed = 0.0 # For these: how much of the timestep has been taken
    fastest = 0.0
    sub = 0
    while True:
        for stage in range(stages):
            ys = y if stage == 0 else yt # Where the rates are worked out
            if row >= 0 and solver < 2: # Values that vary in time, at the time of this stage
                th = (sub + (0.0 if stage == 0 else (1.0 if stage == 3 else 0.5)))/max(n_sub, 1)
                for k in range(4):
                    if len(tables[k]) > 1:
                        interp((p.fixed, p.mult, p.consts, p.rel_trans)[k], tables[k], row, th)

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
                        elif op >= EXP: # A function of one value
                            for g in range(ng):
                                a = stack[top, g]
                                if   op == EXP:  stack[top, g] = np.exp(a)
                                elif op == LOG:  stack[top, g] = np.log(a)
                                elif op == SQRT: stack[top, g] = np.sqrt(a)
                                elif op == ABS:  stack[top, g] = abs(a)
                                elif op == SIN:  stack[top, g] = np.sin(a)
                                else:            stack[top, g] = np.cos(a)
                        else:
                            top -= 1
                            for g in range(ng):
                                a = stack[top, g]
                                b = stack[top+1, g]
                                if   op == ADD: stack[top, g] = a + b
                                elif op == SUB: stack[top, g] = a - b
                                elif op == MUL: stack[top, g] = a * b
                                elif op == DIV: stack[top, g] = a / b
                                elif op == POW: stack[top, g] = a ** b
                                elif op == MIN: stack[top, g] = min(a, b)
                                elif op == MAX: stack[top, g] = max(a, b)
                                elif op == LT:  stack[top, g] = 1.0 if a < b else 0.0
                                elif op == LE:  stack[top, g] = 1.0 if a <= b else 0.0
                                elif op == GT:  stack[top, g] = 1.0 if a > b else 0.0
                                else:           stack[top, g] = 1.0 if a >= b else 0.0
                    if top >= 0:
                        for g in range(ng):
                            rates[f, g] = stack[0, g]*p.mult[f, g]

            if solver == 3 or (sub == 0 and stage == 0 and n_sub <= 0): # The fastest rate at which any compartment is being left
                fastest = 0.0
                for f in range(nf):
                    if p.src[f] >= 0:
                        for g in range(ng):
                            fastest = max(fastest, rates[f, g])
            if sub == 0 and stage == 0:
                if solver < 2 and n_sub <= 0: # Enough substeps that no compartment turns over by more than a quarter in one
                    for m in range(len(p.mov_mat)):
                        for g in range(ng):
                            for k in range(ng):
                                fastest = max(fastest, p.mov_mat[m, g, k])
                    n_sub = int(np.ceil(fastest*dt/0.25))
                n_sub = max(n_sub, 1)
                h = dt/n_sub
                for f in range(nflows):
                    for g in range(ng):
                        flows[f, g] = 0
            if solver >= 2: # The stochastic solvers take it from here
                if solver == 4 and p.local: # The exact solver, place by place: the whole step at once
                    gillespie_places(y, p, w, dt, flows)
                    h = dt - elapsed
                elif solver == 4: # The exact solver: up to the next event
                    h = event(y, p, w, dt - elapsed, flows)
                elif solver == 5:
                    diffuse(y, p, w, h, flows)
                else:
                    if solver == 3: # Tau-leaping: a leap short enough that the rates change little over it
                        h = min(dt - elapsed, 0.03/max(fastest, 1e-12))
                    draw(y, p, w, h, flows, solver == 3)
                if solver != 5 and p.has_det and not (solver == 4 and p.local): # By place, the counts are settled after the step's events: see settle_places()
                    settle(y, p, rates, h, flows)
                elapsed += h
                continue

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
            if solver == 1: # Euler: a compartment whose outflows would take more than everyone in it in a substep has them scaled down together, as compartmental frameworks with large steps do (yt is free, so holds the scale)
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = 0
                for f in range(nf):
                    if p.src[f] >= 0:
                        for g in range(ng):
                            yt[p.src[f], g] += rates[f, g]*h
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = 1.0/yt[c, g] if yt[c, g] > 1 else 1.0
            for f in range(nf):
                s = p.src[f]
                for g in range(ng):
                    if s >= 0:
                        amount = rates[f, g]*ys[s, g]
                        if solver == 1:
                            amount *= yt[s, g]
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

        if solver >= 2:
            pass
        elif stages == 1:
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
        if (timed and elapsed >= dt - 1e-12) or (not timed and sub >= n_sub):
            break
    if solver == 4 or solver == 5: # These move people between groups once, over the whole step
        move(y, p.mov_mat, p.mov_starts, p.mov_comps, p.n_trans, dt, flows, solver == 4)
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


@jit
def apply_events(y, r, ti, whole):
    """
    Apply the events of a movement register that are dated in a step, in order

    People who move or leave are chosen at random from the chosen compartments of their
    place: as whole people (a multivariate hypergeometric draw, one compartment and group
    at a time), or as the expected share of each. People who are added are shared
    equally between the states of their set, and between the groups of their place as
    births are.

    Args:
        y (array): the compartments, by group; changed in place
        r (Register): the events
        ti (int): the step
        whole (bool): whether the compartments hold whole people

    Returns the number of people that events asked for and could not find.
    """
    short = 0.0
    if ti >= len(r.starts) - 1:
        return short
    for e in range(r.starts[ti], r.starts[ti+1]):
        src, dst, s = r.src[e], r.dst[e], r.sel[e]
        n = r.n[e]
        if src >= 0: # Move people, or remove them
            total = 0.0
            for k in range(r.sel_starts[s], r.sel_starts[s+1]):
                for j in range(r.pg_starts[src], r.pg_starts[src+1]):
                    total += y[r.sel_comps[k], r.pg_groups[j]]
            if n > total:
                short += n - total
                n = total
            left = n
            rest = total
            for k in range(r.sel_starts[s], r.sel_starts[s+1]):
                c = r.sel_comps[k]
                for j in range(r.pg_starts[src], r.pg_starts[src+1]):
                    g = r.pg_groups[j]
                    have = y[c, g]
                    if left <= 0 or rest <= 0:
                        break
                    if have <= 0:
                        continue
                    if whole:
                        take = float(np.random.hypergeometric(int(have), int(rest - have), int(left)))
                    else:
                        take = left*have/rest
                    rest -= have
                    left -= take
                    y[c, g] -= take
                    if dst >= 0:
                        y[r.sel_to[k], g + (dst - src)*r.step] += take
        else: # Add people
            left = n
            for k in range(r.ent_starts[s], r.ent_starts[s+1]):
                remaining = r.ent_starts[s+1] - k # The states not yet given their share
                part = binomial(left, 1/remaining) if whole else left/remaining
                left -= part
                wrest = 0.0
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    wrest += r.ent_w[k, r.pg_groups[j]]
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    g = r.pg_groups[j]
                    w = r.ent_w[k, g]
                    if part <= 0 or wrest <= 0:
                        break
                    take = binomial(part, w/wrest) if whole else part*w/wrest
                    y[r.ent_comps[k], g] += take
                    part -= take
                    wrest -= w
    return short


#%% A whole run

solvers = dict(rk4=0, euler=1, binomial=2, tau=3, gillespie=4, sde=5)

@jit
def set_step(p, ti, fixed, mult, consts, rel_trans):
    """ Set the parts of the program that can change over time to their values at a step (each has one entry per step, or a single entry) """
    if ti < len(fixed):     p.fixed[:] = fixed[ti]
    if ti < len(mult):      p.mult[:] = mult[ti]
    if ti < len(consts):    p.consts[:] = consts[ti]
    if ti < len(rel_trans): p.rel_trans[:] = rel_trans[ti]
    return


@jit
def run(y, p, dt, npts, solver, n_sub, seed, fixed, mult, consts, rel_trans, reg, per_step=1):
    """
    Take every step of a run

    Args:
        y (array): the compartments at the start, by group
        p (Program): the flows
        dt (float): the timestep
        npts (int): the number of time points: the start, and the end of each of the npts-1 steps
        solver (int): which of `solvers` to use
        n_sub (int): for the ODE, the number of substeps (see `step()`)
        seed (int): the seed for the random numbers
        fixed, mult, consts, rel_trans (arrays): the values of these parts of the program at each step, or a single
            entry if they are the same throughout
        reg (Register): the events of a movement register, applied at the end of the step they are dated in (those of the start are already in y)
        per_step (int): 3 if the tables have the values at the start, middle, and end of each step (for the ODE), and otherwise 1

    Returns the compartments at the end, the compartments at each time point (the start, then the end of each
    step), the total of each transition and mover over the step that ends at each time point (0 at the start),
    and the number of people that events of the register could not find.
    """
    np.random.seed(seed)
    y = y.copy()
    w = workspace(y, p)
    whole = solver >= 2 and solver != 5
    stocks = np.empty((npts, y.shape[0], y.shape[1]))
    flows = np.zeros((npts, p.n_trans + len(p.mov_mat), y.shape[1]))
    short = 0.0
    tables = (fixed, mult, consts, rel_trans)
    stocks[0] = y
    for ti in range(1, npts): # Step ti goes from time point ti-1 to ti, with the values at the start of it
        set_step(p, (ti-1)*per_step, fixed, mult, consts, rel_trans)
        step(y, p, w, dt, n_sub, solver, flows[ti], tables, (ti-1)*per_step if per_step == 3 else -1)
        if len(reg.src): # The events dated at its end (not called if there are none, since passing the register costs as much as a small step)
            short += apply_events(y, reg, ti, whole)
        if solver == 4 and p.local and p.has_det:
            settle_places(y, p, dt, flows[ti])
        stocks[ti] = y
    return y, stocks, flows, short


@jit
def run_many(y0, p, dt, npts, solver, n_sub, seeds, fixed, mult, consts, rel_trans, reg, state_of, n_states, by_group, per_step=1):
    """
    Take every step of many runs that differ only in their seed

    Args:
        seeds (array): the seed of each run
        state_of (array): the state that each compartment is part of
        n_states (int): the number of states
        by_group (bool): whether to keep the values of each group, or only their totals
        (the others are as for `run()`)

    Returns the number in each state at each time point, and the total of each transition and mover
    over the step that ends at it, both as runs by time points by states (or transitions) by groups
    (one, the total, unless `by_group`).
    """
    nc, ng = y0.shape
    nflows = p.n_trans + len(p.mov_mat)
    nk = ng if by_group else 1
    stocks = np.zeros((len(seeds), npts, n_states, nk))
    flows = np.zeros((len(seeds), npts, nflows, nk))
    y = np.empty_like(y0)
    w = workspace(y, p)
    whole = solver >= 2 and solver != 5
    now = np.empty((nflows, ng))
    tables = (fixed, mult, consts, rel_trans)
    for r in range(len(seeds)):
        np.random.seed(seeds[r])
        y[:] = y0
        for c in range(nc):
            for g in range(ng):
                stocks[r, 0, state_of[c], g if by_group else 0] += y[c, g]
        for ti in range(1, npts):
            set_step(p, (ti-1)*per_step, fixed, mult, consts, rel_trans)
            step(y, p, w, dt, n_sub, solver, now, tables, (ti-1)*per_step if per_step == 3 else -1)
            if len(reg.src):
                apply_events(y, reg, ti, whole)
            if solver == 4 and p.local and p.has_det:
                settle_places(y, p, dt, now)
            for c in range(nc):
                for g in range(ng):
                    stocks[r, ti, state_of[c], g if by_group else 0] += y[c, g]
            for f in range(nflows):
                for g in range(ng):
                    flows[r, ti, f, g if by_group else 0] += now[f, g]
    return stocks, flows


@jit
def one_step(y, p, dt, solver, n_sub, tables, row):
    """ Take one step, and return the compartments after it and the total of each transition and mover over it: see `run()` and `step()` """
    y = y.copy()
    flows = np.empty((p.n_trans + len(p.mov_mat), y.shape[1]))
    step(y, p, workspace(y, p), dt, n_sub, solver, flows, tables, row)
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


@jit
def infection_probs(susceptible, auids, groups, rate, rel_sus, state_sus, adjust, mult, dt):
    """
    The agents who could be infected this step, and the probability for each: one pass over the agents

    Args:
        susceptible (array): whether each agent is in the state, by UID
        auids (array): the agents to consider
        groups (array): the group of each agent, by UID
        rate (array): the rate of infection for a person of average susceptibility, by group
        rel_sus (array): the susceptibility of each agent, by UID
        state_sus (float): the susceptibility of the state
        adjust (array): the multiplier on the transition, by group
        mult (float): any other multiplier on susceptibility
        dt (float): the timestep

    Returns the UIDs of the agents with some chance of infection, and that chance.
    """
    uids = np.empty(len(auids), dtype=np.int64)
    probs = np.empty(len(auids))
    usual = np.empty(len(rate)) # The probability for someone of average susceptibility, which is most people
    for g in range(len(rate)):
        usual[g] = -np.expm1(-max(rate[g]*1.0*state_sus*adjust[g]*mult, 0.0)*dt)
    n = 0
    for uid in auids: # Written every time and kept only if n advances, since a branch on who is susceptible cannot be predicted
        g = groups[uid]
        if rel_sus[uid] == 1:
            prob = usual[g]
        else:
            prob = -np.expm1(-max(rate[g]*rel_sus[uid]*state_sus*adjust[g]*mult, 0.0)*dt) # In this order, to match the arithmetic of the general case
        uids[n] = uid
        probs[n] = prob
        n += susceptible[uid] & (prob > 0)
    return uids[:n], probs[:n]


@jit
def count_groups(members, auids, groups, collapse, n_groups):
    """ The number of agents in a state, by the group they are counted in: one pass over the agents, with no list of who they are """
    counts = np.zeros((4, n_groups)) # Four running counts, taken in turn: adding to one count each time would wait on the last addition when neighbors are in the same group
    for i in range(len(auids)):
        uid = auids[i]
        if members[uid]:
            counts[i & 3, collapse[groups[uid]]] += 1
    return counts.sum(axis=0)
