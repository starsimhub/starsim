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

import os
import sys
import ctypes
import contextlib
import collections
import multiprocessing as mp
import numpy as np
import numba as nb
import starsim as ss
from .distributions import _GOLDEN, _MIX1, _MIX2, _U30, _U27, _U31, _U11, _U40, _INV53, _INV24

__all__ = ['Program', 'Register', 'Shifts', 'opcodes']

# The operations a rate can be made of
CONST, COUNT, FRAC, EXPOSURE, ADD, SUB, MUL, DIV, POW, MIN, MAX, LT, LE, GT, GE, EXP, LOG, SQRT, ABS, SIN, COS, TOTAL = range(22) # TOTAL is a count in the whole population, every place (I.total)
CONVEY, ASSIGN = 22, 23 # Not in a rate: a conveyor (its flow, then its slots), and the compartment an update sets (see convey() and update_values()); kept out of the program's fields, since every field adds to the cost of each call that is passed it
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
    'starts',     # Where each flow's operations start and end (the same, if it has none); and after the flows', those of each conveyor and each update (see convey() and update_values())
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
    'exp_set',    # The set of compartments that are the state; or, if negative, minus the number of exposures before it whose sum it is (diseases tracked jointly, see ss.solvers.Crossing)
    'rel_trans',  # Its infectiousness, by group
    'levels',     # For each mixing term, the cell that each group is in
    'sizes',      # The number of cells in each term
    'block',      # The number of cells for each place in each term, if its cells are places by levels (a cell is the place times this, plus the level), or else all of them
    'contacts',   # The contacts per unit time of each term, as (1, terms, 1); or for each exposure, term, and group, if they differ (a beta for each route, or a modifier of a network's contacts for some people)
    'use',        # For each term, whether people are moved to where they spend their time (by the time share between places) before prevalence is worked out;
                  # whether prevalence is then mixed between the levels in each place, by the term's matrix; and whether it is then mixed between places, by the time share
    'mat',
    'sh_pair',    # The time share between places: the place each entry is from and to (as row and column)
    'sh_val',     # Its share
    # Where the flows go
    'tgt_starts', # Where each flow's targets start and end
    'tgt_comp',   # The compartment
    'tgt_w',      # The share that goes there, by group
    'tgt_map',    # The map to the groups they land in, or -1
    'tgt_rec',    # The flow to a destination of a transition with several that it is also counted in (in the group they leave), or -1
    'crd_starts', # The same for the transitions that a flow is also counted as
    'crd_index',
    'crd_w',
    'crd_parent', # The credit each is nested in (a transition passed through after another), -1 for none, or -2 for one that overlaps the others
    'maps',       # Groups by groups: where people land when they change group as they change state
    # Movement between groups
    'n_movers',
    'mov_starts', # Where each mover's compartments start and end, and its entries (one for each pair of groups it moves people between)
    'mov_comps',
    'mov_pair',   # The group of each entry that people move from, and to
    'mov_rate',   # Its rate
    # For the stochastic solvers: the flows in the order they are drawn, grouped by the compartment they leave
    'src_starts', # (In the program of a crossing, the flows of each disease are drawn in turn: an empty set of flows ends each disease's; see draw())
    'src_flows',
    'src_chain',  # For the compartments of a dwell time in stages: the number of stages left (this one included), so the chain-binomial solver can move people through several in a step; 0 otherwise
    'src_stage',  # Their flow to the next stage
    'src_last',   # The flow out of the last stage
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

# The deliveries of a whole run (see ss.Delivery.shifts()), as the kernels apply them: each moves some of the people in each compartment and group to another group
shift_fields = [
    'due',  # Whether each delivery happens at each time point (deliveries by time points)
    'sel',  # The share of the people in each compartment and group who are eligible (deliveries by compartments by groups)
    'cov',  # The share of those who are reached
    'to',   # The group that each group's people go to if they are reached (deliveries by groups)
    'rest', # The group they go to if they are eligible but not reached (their own, unless the delivery records who has been offered it)
]

Shifts = collections.namedtuple('Shifts', shift_fields)

jit = nb.jit(nopython=True, cache=True, error_model='numpy')
pieces = 4 # The stochastic solvers that take steps of their own length hold values that vary in time for a quarter of a step at a time: see step()
announced = False # Whether a cold cache has been checked for, so it is said only once
golden = np.uint64(0x9e3779b97f4a7c15) # The step between the counters of a random stream: see uniform()
per_chunk = 256 # The places with events that make it worth a thread for the exact solver by place: see gillespie_places()
sparse_flows = 256 # The flows above which the exact solver by place follows only the compartments that hold anyone: see sparse_events()


def announce(verbose=True):
    """ Say when the kernels are about to be built, since the first use after installing (or changing this file) takes one or two minutes """
    global announced
    if verbose and not announced:
        announced = True
        if not run._cache._cache_file._load_index(): # Empty if there is no cache, or it is out of date
            print('Building the solver kernels for first use (one or two minutes; once per install)')
    return


@jit
def workspace(y, p):
    """ The arrays that the kernels work in, made once so that nothing is allocated inside a step """
    nc, ng = y.shape
    nflows = p.n_trans + p.n_movers
    size = max(p.sizes.max(), p.n_places, len(p.pl_starts) - 1)
    rates     = np.empty(p.fixed.shape)         # The rate of each flow, by group
    exposures = np.empty((len(p.exp_set), ng))  # Exposure to each infectious state
    stack     = np.empty((p.depth, ng))         # For working out a rate
    groups    = np.empty((7 + len(p.starts) - len(p.src) - 1, ng)) # Scratch, by group; and after it, the value of each update (see update_values())
    cells     = np.zeros((6, size))             # Scratch, by cell of a mixing term or by place; the last row is, for the exact solver by place, whether each place can have no event until something changes it (see gillespie_places()), kept here since another array to pass would slow every step
    dy        = np.empty((4, nc, ng))           # The stages of the integrator
    df        = np.empty((4, nflows, ng))
    yt        = np.empty((nc, ng))
    chain     = np.empty((p.src_chain.max() if len(p.src_chain) else 0) + 2) # Where people at a stage of a dwell time end up: see chain_probs()
    credits   = np.empty((2, (np.diff(p.crd_starts).max() if len(p.crd_starts) > 1 else 0) + 1, ng)) # For credit(): the count of each credit of a flow not yet shared among those nested in it, and their weight left
    return rates, exposures, stack, groups, cells, dy, df, yt, chain, credits


@jit
def binomial(rng, n, prob):
    """ The number of n people that something with a probability happens to """
    if n <= 0 or prob <= 0:
        return 0.0
    elif prob >= 1:
        return float(n)
    return float(rng.binomial(int(n), prob))


@jit
def hypergeometric(rng, good, bad, n):
    """ The number of good items in n drawn without replacement from good and bad ones, drawn one at a time (the draws of a register are small) """
    good, bad, n = int(good), int(bad), int(n)
    flip = n > (good + bad)//2 # Draw the ones left behind instead, if that is fewer
    if flip:
        n = good + bad - n
    k = 0
    for i in range(n):
        if rng.random()*(good + bad) < good:
            good -= 1
            k += 1
        else:
            bad -= 1
    return float(good if flip else k)


@jit
def land(rng, n, m, maps, out):
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
                k = binomial(rng, left, w/rest)
                out[h] += k
                left -= k
                rest -= w
    return


@jit
def split(rng, n, a, b, weights, credit, out, comps, which, maps, scratch, rec, counted):
    """ Share n whole events between the targets of a flow (or the transitions it is also counted as), at random; a target's share is also counted in the flow to its destination (rec, in counted), if it has one """
    ng = len(n)
    total, left, drawn = scratch[0], scratch[1], scratch[2]
    if b - a == 1 and weights[a].min() >= 1 - 1e-12:
        land(rng, n, -1 if credit else which[a], maps, out[comps[a]])
        if not credit and rec[a] >= 0:
            for g in range(ng):
                counted[rec[a], g] += n[g]
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
                drawn[g] = binomial(rng, left[g], weights[k, g]/total[g]) if total[g] > 0 else 0.0
                left[g] -= drawn[g]
                total[g] -= weights[k, g]
            land(rng, drawn, -1 if credit else which[k], maps, out[comps[k]])
            if not credit and rec[k] >= 0:
                for g in range(ng):
                    counted[rec[k], g] += drawn[g]
        if not credit and whole and left.max() > 0: # Rounding only
            land(rng, left, which[b-1], maps, out[comps[b-1]])
            if rec[b-1] >= 0:
                for g in range(ng):
                    counted[rec[b-1], g] += left[g]
    return


@jit
def credit(rng, n, a, b, p, scratch, flows):
    """
    Count the events of a flow (n, by group) also in the transitions they pass through with no wait (see
    `ss.solvers.Compartmental.entry`). The credits nested in the same one (or in none) are exclusive, so they are one
    multinomial draw from its count, as a chain of conditional binomials; a credit that overlaps the others is drawn on its own
    """
    if b == a:
        return
    ng = len(n)
    left, mass = scratch[0], scratch[1]
    for g in range(ng):
        left[0, g] = n[g]
        mass[0, g] = 1.0
    for k in range(a, b):
        q = p.crd_parent[k]
        i = k - a + 1
        for g in range(ng):
            w = p.crd_w[k, g]
            if q == -2:
                c = binomial(rng, n[g], w)
            else:
                j = 0 if q < 0 else q - a + 1
                c = binomial(rng, left[j, g], w/mass[j, g]) if mass[j, g] > 0 else 0.0
                left[j, g] -= c
                mass[j, g] -= w
            left[i, g] = c
            mass[i, g] = w
            flows[p.crd_index[k], g] += c
    return


@jit
def credit_one(us, a, b, p, g, flows):
    """ As `credit()`, for one event in group g, from a uniform for each credit (`us`) """
    left = np.zeros(b - a + 1)
    mass = np.zeros(b - a + 1)
    left[0] = 1.0
    mass[0] = 1.0
    for k in range(a, b):
        q = p.crd_parent[k]
        i = k - a + 1
        w = p.crd_w[k, g]
        if q == -2:
            hit = us[i-1] < w
        else:
            j = 0 if q < 0 else q - a + 1
            hit = left[j] > 0 and us[i-1]*mass[j] < w
            if hit:
                left[j] = 0.0
            mass[j] -= w
        left[i] = 1.0 if hit else 0.0
        mass[i] = w
        if hit:
            flows[p.crd_index[k], g] += 1
    return


@jit
def chain_probs(r, h, dt, d, out):
    """
    Where people at a stage of a dwell time are at the end of a step, with d stages left (this one included)

    Each stage is left at rate r, and the state by its other exits at rate h. The number of
    stages passed in the step is Poisson, so out[m] (for m < d) is the chance of being m stages
    on, out[d] of having left by the last stage, and out[d+1] of having left by another exit.
    """
    x = r*dt
    xt = (r + h)*dt
    stay = np.exp(-h*dt)
    term = np.exp(-x)
    term_t = np.exp(-xt)
    cdf_t = 0.0
    total = 0.0
    for m in range(d):
        out[m] = stay*term
        total += out[m]
        cdf_t += term_t
        term *= x/(m + 1)
        term_t *= xt/(m + 1)
    out[d] = (r/(r + h))**d*max(1 - cdf_t, 0.0) if r + h > 0 else 0.0 # The last stage is reached before another exit, and within the step
    out[d+1] = max(1 - total - out[d], 0.0)
    return


@jit
def draw_chain(rng, y, p, w, s, dt, new, flows):
    """
    For the chain-binomial solver: the people at a stage of a dwell time move through as many stages as they pass in the step (see `chain_probs()`)

    Exact for the stages, where one stage a step (or a substep for each stage) is not; the
    rates are held for the step, as for any other flow.
    """
    ng = y.shape[1]
    rates, groups, probs = w[0], w[3], w[8]
    total, left, n, ends = groups[3], groups[4], groups[5], groups[6]
    a, b = p.src_starts[s], p.src_starts[s+1]
    d = p.src_chain[s]
    stage = p.src_stage[s]
    src = p.src[stage]
    stride = p.tgt_comp[p.tgt_starts[stage]] - src # How far apart its stages are: 1, except in the program of a crossing
    for g in range(ng):
        r = max(rates[stage, g], 0.0)
        h = 0.0
        for k in range(a, b):
            f = p.src_flows[k]
            if f != stage:
                h += max(rates[f, g], 0.0)
        chain_probs(r, h, dt, d, probs)
        rest = y[src, g]
        mass = 1.0
        left[g] = binomial(rng, rest, probs[d+1]) # Leave by another exit
        rest -= left[g]
        mass -= probs[d+1]
        ends[g] = binomial(rng, rest, probs[d]/mass) if mass > 0 else rest # Leave by the last stage
        rest -= ends[g]
        mass -= probs[d]
        moved = left[g] + ends[g]
        for m in range(1, d): # Move on some stages; the rest stay
            k = binomial(rng, rest, probs[m]/mass) if mass > 0 else 0.0
            new[src + m*stride, g] += k
            rest -= k
            mass -= probs[m]
            moved += k
        new[src, g] -= moved
        total[g] = h
    for k in range(a, b): # Another exit: as in draw()
        f = p.src_flows[k]
        if f == stage:
            continue
        for g in range(ng):
            hazard = max(rates[f, g], 0.0)
            n[g] = binomial(rng, left[g], hazard/total[g]) if total[g] > 0 else 0.0
            total[g] -= hazard
            left[g] -= n[g]
        if p.named[f]:
            land(rng, n, p.record[f], p.maps, flows[p.index[f]])
        split(rng, n, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows)
        credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p, w[9], flows)
    f = p.src_last[s] # The last stage
    if p.named[f]:
        land(rng, ends, p.record[f], p.maps, flows[p.index[f]])
    split(rng, ends, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows)
    credit(rng, ends, p.crd_starts[f], p.crd_starts[f+1], p, w[9], flows)
    return


@jit
def draw(rng, y, p, w, dt, flows, tau):
    """
    Draw whole numbers of events over a time dt, and add them to the flows

    For each compartment, the number who leave it (binomial for the chain-binomial solver;
    for tau-leaping, Poisson, and no more than are there), then one multinomial draw of
    which flow each leaves by. Called by `step()`, which has worked out the rates.
    """
    ng = y.shape[1]
    rates, groups, new = w[0], w[3], w[7]
    total, left, n = groups[3], groups[4], groups[5]
    new[:] = y
    for s in range(len(p.src_starts) - 1):
        a, b = p.src_starts[s], p.src_starts[s+1]
        if a == b: # In the program of a crossing, the end of one disease's flows: the next disease's are drawn from where these left people (each from the rates at the start of the step), so each disease's moves are drawn as they would be for it alone
            y[:] = new
            continue
        src = p.src[p.src_flows[a]]
        if src >= 0 and p.det[p.src_flows[a]]: # A state that is not people: see settle()
            continue
        if not tau and p.src_chain[s] > 1:
            draw_chain(rng, y, p, w, s, dt, new, flows)
            continue
        if src >= 0:
            for g in range(ng):
                total[g] = 0
                for k in range(a, b):
                    total[g] += max(rates[p.src_flows[k], g], 0.0)
                if tau:
                    left[g] = min(float(rng.poisson(total[g]*y[src, g]*dt)), y[src, g]) # Everyone who leaves, clipped once, so the order of the flows does not matter
                else:
                    left[g] = binomial(rng, y[src, g], -np.expm1(-total[g]*dt))
                new[src, g] -= left[g]
        for k in range(a, b):
            f = p.src_flows[k]
            if p.det[f]:
                continue
            for g in range(ng):
                if src < 0: # An inflow
                    n[g] = rng.poisson(max(rates[f, g]*p.inflow[f, g]*dt, 0.0))
                else:
                    hazard = max(rates[f, g], 0.0)
                    n[g] = binomial(rng, left[g], hazard/total[g]) if total[g] > 0 else 0.0
                    total[g] -= hazard
                    left[g] -= n[g]
            if p.named[f] and p.record[f] < 0: # Counted in the group it happens in: as land() does, but written out (as is the one target below), since the calls cost more than the draws of a small model
                for g in range(ng):
                    flows[p.index[f], g] += n[g]
            elif p.named[f]:
                land(rng, n, p.record[f], p.maps, flows[p.index[f]])
            t = p.tgt_starts[f]
            if p.tgt_starts[f+1] == t + 1 and p.tgt_map[t] < 0 and p.tgt_rec[t] < 0 and p.tgt_w[t].min() >= 1 - 1e-12: # Everyone to one compartment, in the same group
                for g in range(ng):
                    new[p.tgt_comp[t], g] += n[g]
            else:
                split(rng, n, t, p.tgt_starts[f+1], p.tgt_w, False, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows)
            credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p, w[9], flows)
    for m in range(p.n_movers): # Movement between groups: as move() does, but written out, since a call here (once a leap) slows tau-leaping by a few percent
        for e in range(p.mov_starts[1, m], p.mov_starts[1, m+1]):
            if p.mov_rate[e] != 0:
                g, h = p.mov_pair[0, e], p.mov_pair[1, e]
                prob = -np.expm1(-p.mov_rate[e]*dt)
                for k in range(p.mov_starts[0, m], p.mov_starts[0, m+1]):
                    c = p.mov_comps[k]
                    moved = binomial(rng, new[c, g], prob)
                    new[c, g] -= moved
                    new[c, h] += moved
                    flows[p.n_trans + m, g] += moved
    y[:] = new
    return


@nb.jit(nopython=True, cache=True, inline='always')
def sum_exposures(expo, e, s):
    """ The sum of the -s exposures before exposure e, added in order (an exposure whose set is negative, see `Program`) """
    v = expo[e+s]
    for x in range(e+s+1, e):
        v += expo[x]
    return v


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
        if s < 0:
            expo[e] = sum_exposures(expo, e, s)
            continue
        num = 0.0
        for j in range(a, b):
            g = p.pl_groups[j]
            for k in range(p.set_starts[s], p.set_starts[s+1]):
                num += ys[p.set_comps[k], g]*p.rel_trans[e, g]
        prev = num/n if n > 0 else 0.0
        expo[e] = 0.0
        for t in range(len(p.sizes)):
            expo[e] += p.contacts[0, t, 0]*prev
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
                elif op == TOTAL:
                    top += 1
                    v = 0.0
                    for gg in range(ys.shape[1]):
                        for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                            v += ys[p.set_comps[i], gg]
                    stack[top, 0] = v
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
                    elif op == DIV: stack[top, 0] = x / z if x != 0 else 0.0 # Nothing divided by anything is nothing, e.g. the share of an empty group
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


@nb.jit(nopython=True, cache=True, inline='always') # Inlined, as it is called for every event of the exact solver by place
def splitmix(z):
    """ The mixing function of SplitMix64, which turns a counter into random bits (as `ss.distributions.hash_uniforms()` does for agents) """
    z = (z ^ (z >> np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    z = (z ^ (z >> np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return z ^ (z >> np.uint64(31))


@nb.jit(nopython=True, cache=True, inline='always')
def uniform(z):
    """ A number uniform on (0, 1) from the counter of a random stream (SplitMix64), which the caller moves on by `golden` before each number """
    return (float(splitmix(z) >> np.uint64(11)) + 0.5)/9007199254740992.0


@jit
def place_events(y, p, w, active, first, last, dt, flows, seed):
    """ The events over a time dt of the places to step from first to last-1 (in `active`), each drawn from its own random numbers: see `gillespie_places()` """
    rates, idle = w[0], w[4][5]
    nf = len(p.src)
    src, det, inflow, pl_groups = p.src, p.det, p.inflow, p.pl_groups
    stack = np.empty((p.depth, 1)) # Scratch of its own, so that chunks of places can be stepped at once
    expo = np.empty(max(len(p.exp_set), 1))
    for m in range(first, last):
        i = active[m]
        a, b = p.pl_starts[i], p.pl_starts[i+1]
        idle[i] = 0
        t = 0.0
        z = np.uint64(0)
        while True:
            place_rates(y, p, a, b, rates, stack, expo)
            total = 0.0
            for f in range(nf): # As propensity(), written out, since passing it the program for each place costs many times the arithmetic
                if det[f]:
                    continue
                s = src[f]
                for j in range(a, b):
                    g = pl_groups[j]
                    total += max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
            if total <= 0:
                idle[i] = 1
                break
            if t == 0.0: # The first event in this step: the counter of the place's random numbers in this step
                z = splitmix(seed + splitmix(np.uint64(i + 1)*golden))
            z += golden
            t -= np.log(uniform(z))/total
            if t > dt:
                break
            z += golden
            target = uniform(z)*total
            cum = 0.0
            which, where = -1, -1
            for f in range(nf):
                if det[f]:
                    continue
                s = src[f]
                for j in range(a, b):
                    g = pl_groups[j]
                    r = max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
                    if r > 0:
                        cum += r
                        which, where = f, g
                        if cum > target:
                            break
                if cum > target:
                    break
            if src[which] >= 0:
                y[src[which], where] -= 1
            if p.named[which]:
                flows[p.index[which], where] += 1
            ka, kb = p.tgt_starts[which], p.tgt_starts[which+1] # Where it goes: one of its targets, in proportion to their shares
            z += golden
            u = uniform(z)
            tot = 0.0
            for k in range(ka, kb):
                tot += p.tgt_w[k, where]
            cum = 0.0
            for k in range(ka, kb):
                cum += p.tgt_w[k, where]
                if u*tot < cum or k == kb - 1:
                    y[p.tgt_comp[k], where] += 1
                    if p.tgt_rec[k] >= 0:
                        flows[p.tgt_rec[k], where] += 1
                    break
            z += golden
            u = uniform(z) # The transitions it also counts as (immediate branches)
            ca, cb = p.crd_starts[which], p.crd_starts[which+1]
            if cb > ca:
                us = np.empty(cb - ca)
                us[0] = u
                for k in range(1, cb - ca):
                    z += golden
                    us[k] = uniform(z)
                credit_one(us, ca, cb, p, where, flows)
    return


@jit
def sparse_maps(p, nc):
    """
    For `sparse_events()`: the sets of compartments that each compartment is in, the exposures to each
    set, the flows out of each compartment (in order), the inflows, which compartments hold people, and
    whether any rate is a count in every place
    """
    n_sets = len(p.set_starts) - 1
    cs_starts = np.zeros(nc + 1, dtype=np.int64)
    for k in range(len(p.set_comps)):
        cs_starts[p.set_comps[k] + 1] += 1
    cs_starts = np.cumsum(cs_starts)
    cs_sets = np.empty(len(p.set_comps), dtype=np.int64)
    fill = cs_starts[:-1].copy()
    for s in range(n_sets): # In order, so each compartment's sets are too
        for k in range(p.set_starts[s], p.set_starts[s+1]):
            c = p.set_comps[k]
            cs_sets[fill[c]] = s
            fill[c] += 1
    se_starts = np.zeros(n_sets + 1, dtype=np.int64)
    for e in range(len(p.exp_set)):
        if p.exp_set[e] >= 0: # Not a sum of others
            se_starts[p.exp_set[e] + 1] += 1
    se_starts = np.cumsum(se_starts)
    se_exps = np.empty(se_starts[-1], dtype=np.int64)
    fill = se_starts[:-1].copy()
    for e in range(len(p.exp_set)):
        if p.exp_set[e] >= 0:
            se_exps[fill[p.exp_set[e]]] = e
            fill[p.exp_set[e]] += 1
    cf_starts = np.zeros(nc + 1, dtype=np.int64) # The flows that are events (see settle()), by the compartment they leave
    n_in = 0
    for f in range(len(p.src)):
        if p.det[f]:
            continue
        if p.src[f] >= 0:
            cf_starts[p.src[f] + 1] += 1
        else:
            n_in += 1
    cf_starts = np.cumsum(cf_starts)
    cf_flows = np.empty(cf_starts[-1], dtype=np.int64)
    inflows = np.empty(n_in, dtype=np.int64)
    fill = cf_starts[:-1].copy()
    n_in = 0
    for f in range(len(p.src)):
        if p.det[f]:
            continue
        if p.src[f] >= 0:
            cf_flows[fill[p.src[f]]] = f
            fill[p.src[f]] += 1
        else:
            inflows[n_in] = f
            n_in += 1
    people = np.zeros(nc, dtype=np.int64)
    for c in p.people:
        people[c] = 1
    has_total = False # A count in every place, which the kept rates cannot follow
    for k in range(len(p.ops)):
        has_total = has_total or p.ops[k] == TOTAL
    return cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total


@nb.jit(nopython=True, cache=True, inline='always')
def add_key(keys, n, key):
    """ Put a key into the first n of a sorted array, in order; returns the new number of keys """
    m = n
    while m > 0 and keys[m-1] > key:
        keys[m] = keys[m-1]
        m -= 1
    keys[m] = key
    return n + 1


@nb.jit(nopython=True, cache=True, inline='always')
def drop_key(keys, n, key):
    """ Take a key out of the first n of a sorted array; returns the new number of keys """
    m = np.searchsorted(keys[:n], key)
    for k in range(m, n - 1):
        keys[k] = keys[k+1]
    return n - 1


@jit
def sparse_events(y, p, w, active, first, last, dt, flows, seed, maps):
    """
    The events of the places to step, as `place_events()`, for a large program (e.g. diseases tracked jointly,
    whose compartments are the product of theirs), most of whose compartments are empty

    Each place keeps the compartments that hold anyone, and the flows out of them, in order. After each
    event, the counts and exposures are worked out from those compartments only, and only those flows
    are taken (with the inflows), so the cost of an event grows with the compartments that hold anyone
    rather than with all of them. A rate that is worked out from the counts and exposures is kept until
    they change, which most events (a move to the next stage of a dwell time) do not. A flow out of an
    empty compartment has no events, and the flows are taken in the same order, with the same arithmetic
    and random numbers, so the events are the same as `place_events()` gives.
    """
    idle = w[4][5]
    nc = y.shape[0]
    nf = len(p.src)
    src, inflow, pl_groups, starts, ops, args = p.src, p.inflow, p.pl_groups, p.starts, p.ops, p.args
    cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total = maps
    width = 1
    for m in range(first, last):
        width = max(width, p.pl_starts[active[m]+1] - p.pl_starts[active[m]])
    gbits, cbits = 0, 0 # The bits of a key that hold the group in the place, and the compartment
    while (1 << gbits) < width:
        gbits += 1
    while (1 << cbits) < nc:
        cbits += 1
    gmask, cmask = (1 << gbits) - 1, (1 << cbits) - 1
    held = np.empty(nc*width, dtype=np.int64)          # The compartments that hold anyone in the place, as keys of (group, compartment), in order
    keys = np.empty(nf*width, dtype=np.int64)          # The flows out of them and the inflows, as keys of (flow, group), in order
    props = np.empty(nf*width)                         # The rate of each
    kept = np.empty(nf << gbits)                       # By key, the rate of a flow worked out from the counts and exposures,
    fresh = np.zeros(nf << gbits, dtype=np.bool_)      # and whether it is at their current values
    counts = np.empty(len(p.set_starts) - 1)
    num = np.empty(len(p.exp_set))
    before = np.empty(len(counts) + len(num) + 1) # The counts, exposures, and people at the last event
    stack = np.empty((p.depth, 1))
    expo = np.empty(max(len(p.exp_set), 1))
    for m in range(first, last):
        i = active[m]
        a, b = p.pl_starts[i], p.pl_starts[i+1]
        ng = b - a
        idle[i] = 0
        n_held = 0
        for j in range(ng):
            g = pl_groups[a+j]
            for c in range(nc):
                if y[c, g] != 0:
                    held[n_held] = (j << cbits) | c
                    n_held += 1
        n_keys = 0
        for f in inflows:
            for j in range(ng):
                if inflow[f, pl_groups[a+j]] != 0:
                    keys[n_keys] = (f << gbits) | j
                    n_keys += 1
        for h in range(n_held):
            j = held[h] >> cbits
            c = held[h] & cmask
            for k in range(cf_starts[c], cf_starts[c+1]):
                keys[n_keys] = (cf_flows[k] << gbits) | j
                n_keys += 1
        keys[:n_keys].sort()
        t = 0.0
        z = np.uint64(0)
        begun = False
        while True:
            n = 0.0 # The counts and exposures, as place_rates() works them out, from the compartments that hold anyone
            counts[:] = 0.0
            num[:] = 0.0
            for h in range(n_held):
                j = held[h] >> cbits
                c = held[h] & cmask
                g = pl_groups[a+j]
                v = y[c, g]
                if people[c]:
                    n += v
                for k in range(cs_starts[c], cs_starts[c+1]):
                    s = cs_sets[k]
                    counts[s] += v
                    for q in range(se_starts[s], se_starts[s+1]):
                        e = se_exps[q]
                        num[e] += v*p.rel_trans[e, g]
            for e in range(len(p.exp_set)):
                if p.exp_set[e] < 0:
                    expo[e] = sum_exposures(expo, e, p.exp_set[e])
                    continue
                prev = num[e]/n if n > 0 else 0.0
                expo[e] = 0.0
                for tt in range(len(p.sizes)):
                    expo[e] += p.contacts[0, tt, 0]*prev
            same = begun and not has_total and n == before[-1] # Whether the rates kept still hold (the first event of a place works them out)
            for k in range(len(counts)):
                same = same and counts[k] == before[k]
            for e in range(len(num)):
                same = same and expo[e] == before[len(counts) + e]
            before[:len(counts)] = counts
            before[len(counts):-1] = expo[:len(num)]
            before[-1] = n
            begun = True
            total = 0.0
            for q in range(n_keys): # The rate of each flow, as place_rates() works it out
                key = keys[q]
                f = key >> gbits
                g = pl_groups[a + (key & gmask)]
                if starts[f] == starts[f+1]:
                    rate = p.fixed[f, g]
                elif same and fresh[key]:
                    rate = kept[key]
                else:
                    rate = p.fixed[f, g]
                    top = -1
                    for k in range(starts[f], starts[f+1]):
                        op = ops[k]
                        arg = args[k]
                        if op == CONST:
                            top += 1
                            stack[top, 0] = p.consts[arg, g]
                        elif op == EXPOSURE:
                            top += 1
                            stack[top, 0] = expo[arg]
                        elif op == TOTAL:
                            top += 1
                            v = 0.0
                            for gg in range(y.shape[1]):
                                for ii in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    v += y[p.set_comps[ii], gg]
                            stack[top, 0] = v
                        elif op == COUNT or op == FRAC:
                            top += 1
                            v = counts[arg]
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
                            u = stack[top+1, 0]
                            if   op == ADD: stack[top, 0] = x + u
                            elif op == SUB: stack[top, 0] = x - u
                            elif op == MUL: stack[top, 0] = x * u
                            elif op == DIV: stack[top, 0] = x / u if x != 0 else 0.0
                            elif op == POW: stack[top, 0] = x ** u
                            elif op == MIN: stack[top, 0] = min(x, u)
                            elif op == MAX: stack[top, 0] = max(x, u)
                            elif op == LT:  stack[top, 0] = 1.0 if x < u else 0.0
                            elif op == LE:  stack[top, 0] = 1.0 if x <= u else 0.0
                            elif op == GT:  stack[top, 0] = 1.0 if x > u else 0.0
                            else:           stack[top, 0] = 1.0 if x >= u else 0.0
                    if top >= 0:
                        rate = stack[0, 0]*p.mult[f, g]
                    kept[key] = rate
                    fresh[key] = True
                s = src[f]
                props[q] = max(rate*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)
                total += props[q]
            if total <= 0:
                idle[i] = 1
                break
            if t == 0.0: # As in place_events()
                z = splitmix(seed + splitmix(np.uint64(i + 1)*golden))
            z += golden
            t -= np.log(uniform(z))/total
            if t > dt:
                break
            z += golden
            target = uniform(z)*total
            cum = 0.0
            pick = -1
            for q in range(n_keys):
                if props[q] > 0:
                    cum += props[q]
                    pick = q
                    if cum > target:
                        break
            which = keys[pick] >> gbits
            j = keys[pick] & gmask
            where = pl_groups[a+j]
            s = src[which]
            if s >= 0:
                y[s, where] -= 1
                if y[s, where] == 0: # Its compartment is now empty
                    n_held = drop_key(held, n_held, (j << cbits) | s)
                    for k in range(cf_starts[s], cf_starts[s+1]):
                        n_keys = drop_key(keys, n_keys, (cf_flows[k] << gbits) | j)
            if p.named[which]:
                flows[p.index[which], where] += 1
            ka, kb = p.tgt_starts[which], p.tgt_starts[which+1]
            z += golden
            u = uniform(z)
            tot = 0.0
            for k in range(ka, kb):
                tot += p.tgt_w[k, where]
            cum = 0.0
            for k in range(ka, kb):
                cum += p.tgt_w[k, where]
                if u*tot < cum or k == kb - 1:
                    c = p.tgt_comp[k]
                    if y[c, where] == 0: # Its compartment now holds someone
                        n_held = add_key(held, n_held, (j << cbits) | c)
                        for kk in range(cf_starts[c], cf_starts[c+1]):
                            n_keys = add_key(keys, n_keys, (cf_flows[kk] << gbits) | j)
                            fresh[(cf_flows[kk] << gbits) | j] = False
                    y[c, where] += 1
                    if p.tgt_rec[k] >= 0:
                        flows[p.tgt_rec[k], where] += 1
                    break
            z += golden
            u = uniform(z) # The transitions it also counts as (immediate branches)
            ca, cb = p.crd_starts[which], p.crd_starts[which+1]
            if cb > ca:
                us = np.empty(cb - ca)
                us[0] = u
                for k in range(1, cb - ca):
                    z += golden
                    us[k] = uniform(z)
                credit_one(us, ca, cb, p, where, flows)
    return


@nb.jit(nopython=True, cache=True, error_model='numpy', parallel=True)
def threaded_places(y, p, w, active, n_active, chunks, dt, flows, seed, sparse):
    """ The places to step, in chunks on Numba's threads: see `gillespie_places()` """
    if sparse: # The maps are made here, since a tuple passed through object mode breaks Numba's compiler
        maps = sparse_maps(p, y.shape[0])
        for c in nb.prange(chunks):
            sparse_events(y, p, w, active, c*n_active//chunks, (c + 1)*n_active//chunks, dt, flows, seed, maps)
    else:
        for c in nb.prange(chunks):
            place_events(y, p, w, active, c*n_active//chunks, (c + 1)*n_active//chunks, dt, flows, seed)
    return


@jit
def gillespie_places(rng, y, p, w, dt, flows, fresh):
    """
    The exact solver over a time dt, one place at a time, when places do not affect each other within a step (as SimInf runs each node on its own)

    Each place has its own events, and after each one only the rates of that place are worked
    out again, so the cost of an event does not grow with the number of places. A place where
    no event can happen (an SIR with nobody infected) is skipped until something changes it:
    the register's events (see `touch()`), the deliveries, movers, or rates that change over
    time (`fresh`, which steps every place). A large program (e.g. diseases tracked jointly) is
    stepped by the compartments that hold anyone (see `sparse_events()`), with the same events.

    Each place draws from its own random numbers in each step, seeded by one number from the
    engine's generator, so that when many places have events, they are stepped on the threads
    Numba has (`nb.set_num_threads()`), with the same results as on one. The threads are started
    from Python (`threaded_places()`, through object mode), since a cached kernel that calls a
    parallel one segfaults when it is loaded from the cache in a later process (Numba 0.68).
    """
    idle = w[4][5]
    active = np.empty(len(p.pl_starts) - 1, dtype=np.int64) # The places to step
    chunks = 1
    for k in range(len(p.ops)):
        if p.ops[k] == TOTAL: # A count in every place, which any place can change: every place is stepped, in turn
            fresh = True
            chunks = 0
    n_active = 0
    for i in range(len(active)):
        if fresh or not idle[i]:
            active[n_active] = i
            n_active += 1
    if chunks:
        chunks = max(min(n_active//per_chunk, 64), 1) # Each chunk is a call that is passed the program, which costs as much as stepping a few places with nothing to do
    seed = np.uint64(rng.integers(0, 2**62))
    sparse = len(p.src) > sparse_flows # A large program, most of whose compartments are empty: see sparse_events()
    if chunks > 1:
        with nb.objmode():
            threaded_places(y, p, w, active, n_active, chunks, dt, flows, np.uint64(seed), sparse) # As a NumPy number, which Python would make an int
    elif sparse:
        sparse_events(y, p, w, active, 0, n_active, dt, flows, seed, sparse_maps(p, y.shape[0]))
    else:
        place_events(y, p, w, active, 0, n_active, dt, flows, seed)
    return


@jit
def touch(r, ti, places, w):
    """ For the exact solver by place: the places that the events of a movement register dated in a step change, which are stepped again (see `gillespie_places()`) """
    if ti >= len(r.starts) - 1:
        return
    idle = w[4][5]
    for e in range(r.starts[ti], r.starts[ti+1]):
        for place in (r.src[e], r.dst[e]):
            if place >= 0:
                for j in range(r.pg_starts[place], r.pg_starts[place+1]):
                    idle[places[r.pg_groups[j]]] = 0
    return


@jit
def wake(w):
    """ For the exact solver by place: every place is stepped again, after something that could change any of them (see `gillespie_places()`) """
    w[4][5, :] = 0
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
                    if p.tgt_rec[t] >= 0:
                        kf[stage, p.tgt_rec[t], g] += amount*p.tgt_w[t, g]
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
def move(rng, y, mov_starts, mov_comps, mov_pair, mov_rate, n_trans, dt, flows, whole):
    """ Move people between groups over a time dt: as whole numbers drawn at random, or as the expected share (for the exact solver, once a step) """
    for m in range(mov_starts.shape[1] - 1):
        for e in range(mov_starts[1, m], mov_starts[1, m+1]):
            if mov_rate[e] != 0:
                g, h = mov_pair[0, e], mov_pair[1, e]
                prob = -np.expm1(-mov_rate[e]*dt)
                for k in range(mov_starts[0, m], mov_starts[0, m+1]):
                    c = mov_comps[k]
                    moved = binomial(rng, y[c, g], prob) if whole else y[c, g]*prob
                    y[c, g] -= moved
                    y[c, h] += moved
                    flows[n_trans + m, g] += moved
    return


@nb.jit(nopython=True, cache=True, error_model='numpy', inline='always') # Inlined, and given the program's src, det, and inflow rather than the program, since it is called for every flow and group at each event, and passing the program costs (even inlined) far more than the arithmetic
def propensity(y, src, det, inflow, rates, f, g):
    """ The rate at which events of a flow happen in a group: per member of its source compartment, or in total for an inflow """
    s = src[f]
    if det[f]: # Not an event: see settle()
        return 0.0
    return max(rates[f, g]*(y[s, g] if s >= 0 else inflow[f, g]), 0.0)


@jit
def event(rng, y, p, w, dt, flows):
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
            total += propensity(y, p.src, p.det, p.inflow, rates, f, g)
    if total <= 0:
        return dt
    wait = rng.exponential(1/total)
    if wait > dt:
        return dt

    # Which event: the last one with any chance, if rounding leaves the draw past the end
    target = rng.random()*total
    cum = 0.0
    which, where = -1, -1
    for f in range(nf):
        for g in range(ng):
            rate = propensity(y, p.src, p.det, p.inflow, rates, f, g)
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
        land(rng, n, p.record[which], p.maps, flows[p.index[which]])
    split(rng, n, p.tgt_starts[which], p.tgt_starts[which+1], p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows)
    credit(rng, n, p.crd_starts[which], p.crd_starts[which+1], p, w[9], flows)
    return wait


#%% The exact solver for groups that affect each other: see gillespie_groups()

min_coupled = 8 # The fewest groups for which the exact solver follows only what each event changes; with fewer, working out every rate again after each event costs less


@jit
def coupled(p, ng):
    """ Whether the exact solver can step a program by `gillespie_groups()`: enough groups, events that stay in their group, and rates worked out from counts by place and exposures by cell (no reservoir, conveyor, update, count in every place, contacts by group, or sum of exposures) """
    if ng < min_coupled or p.has_det or len(p.starts) - len(p.src) - 1 > 0:
        return False
    if p.contacts.shape[0] > 1 or p.contacts.shape[2] > 1:
        return False
    for k in range(len(p.ops)):
        if p.ops[k] == TOTAL:
            return False
    for f in range(len(p.src)):
        if p.record[f] >= 0:
            return False
        if p.src[f] < 0:
            for g in range(ng):
                if p.inflow[f, g] < 0:
                    return False
    for k in range(len(p.tgt_map)):
        if p.tgt_map[k] >= 0:
            return False
    for e in range(len(p.exp_set)):
        if p.exp_set[e] < 0:
            return False
    return True


@nb.jit(nopython=True, cache=True, inline='always')
def group_order(p, g, h, counts):
    """ -1, 0, or 1 as the numbers that group g's rates are worked out from sort before, are the same as, or sort after group h's: the cells they are in, their place (if a rate has a count in it), and the program's values for them """
    for t in range(len(p.sizes)):
        if p.levels[t, g] != p.levels[t, h]:
            return -1 if p.levels[t, g] < p.levels[t, h] else 1
    if counts and p.places[g] != p.places[h]:
        return -1 if p.places[g] < p.places[h] else 1
    for f in range(len(p.src)):
        if p.fixed[f, g] != p.fixed[f, h]:
            return -1 if p.fixed[f, g] < p.fixed[f, h] else 1
        if p.mult[f, g] != p.mult[f, h]:
            return -1 if p.mult[f, g] < p.mult[f, h] else 1
    for k in range(len(p.consts)):
        if p.consts[k, g] != p.consts[k, h]:
            return -1 if p.consts[k, g] < p.consts[k, h] else 1
    return 0


@jit
def sort_groups(p, ng, counts):
    """ The groups in the order of `group_order()`, keeping their own order where it is the same (a merge sort) """
    order = np.arange(ng)
    tmp = np.empty(ng, dtype=np.int64)
    width = 1
    while width < ng:
        for a in range(0, ng, 2*width):
            m, b = min(a + width, ng), min(a + 2*width, ng)
            i, j, k = a, m, a
            while i < m and j < b:
                if group_order(p, order[j], order[i], counts) < 0:
                    tmp[k] = order[j]
                    j += 1
                else:
                    tmp[k] = order[i]
                    i += 1
                k += 1
            while i < m:
                tmp[k] = order[i]
                i += 1
                k += 1
            while j < b:
                tmp[k] = order[j]
                j += 1
                k += 1
        order, tmp = tmp, order
        width *= 2
    return order


@nb.jit(nopython=True, cache=True, inline='always')
def fen_add(tree, c, i, d):
    """ Add d to entry i of row c of a Fenwick tree, which keeps sums of the entries before each position, each in log steps """
    i += 1
    while i < tree.shape[1]:
        tree[c, i] += d
        i += i & (-i)
    return


@nb.jit(nopython=True, cache=True, inline='always')
def fen_sum(tree, c, i):
    """ The sum of entries 0 to i-1 of row c of a Fenwick tree """
    out = 0.0
    while i > 0:
        out += tree[c, i]
        i -= i & (-i)
    return out


@nb.jit(nopython=True, cache=True, inline='always')
def fen_find(tree, c, v, top):
    """ The first entry of row c of a Fenwick tree at which the sum up to and including it is more than v (`top`: the largest power of 2 not over the number of entries) """
    pos = 0
    while top > 0:
        if pos + top < tree.shape[1] and tree[c, pos + top] <= v:
            pos += top
            v -= tree[c, pos]
        top >>= 1
    return pos


@nb.jit(nopython=True, cache=True, inline='always')
def seg_set(tree, size, i, v):
    """ Set leaf i of a tree of sums (`size` leaves, a power of 2), and the sums above it """
    j = size + i
    tree[j] = v
    j >>= 1
    while j >= 1:
        tree[j] = tree[2*j] + tree[2*j+1]
        j >>= 1
    return


@nb.jit(nopython=True, cache=True, inline='always')
def pick(rng, a, b, weights, g, whole):
    """ Which target (or transition also counted) of a flow one event in group g goes to, as `split()` shares one: -1 for none (`whole`: whether the weights add up to 1 in every group, so rounding goes to the last) """
    total = 0.0
    for k in range(a, b):
        total += weights[k, g]
    for k in range(a, b):
        prob = weights[k, g]/total if total > 0 else 0.0
        if prob >= 1 or (prob > 0 and rng.random() < prob):
            return k
        total -= weights[k, g]
    return b - 1 if whole and b > a else -1


@nb.jit(nopython=True, cache=True, inline='always')
def class_values(p, C, g, expo, rate, post, cnt, npl, stack):
    """ The exposures of class C (whose first group is g), from the exposure of each cell (`post`), and the rate of each flow, as `step()` works them out """
    for e in range(len(p.exp_set)):
        expo[e, C] = 0.0
        for t in range(len(p.sizes)):
            expo[e, C] += p.contacts[0, t, 0]*post[e, t, p.levels[t, g]]
    pl = p.places[g]
    for f in range(len(p.src)):
        rate[f, C] = p.fixed[f, g]
        top = -1
        for k in range(p.starts[f], p.starts[f+1]):
            op = p.ops[k]
            arg = p.args[k]
            if op == CONST:
                top += 1
                stack[top] = p.consts[arg, g]
            elif op == EXPOSURE:
                top += 1
                stack[top] = expo[arg, C]
            elif op == COUNT or op == FRAC:
                top += 1
                stack[top] = cnt[arg, pl]/npl[pl] if op == FRAC else cnt[arg, pl]
            elif op >= EXP:
                a = stack[top]
                if   op == EXP:  stack[top] = np.exp(a)
                elif op == LOG:  stack[top] = np.log(a)
                elif op == SQRT: stack[top] = np.sqrt(a)
                elif op == ABS:  stack[top] = abs(a)
                elif op == SIN:  stack[top] = np.sin(a)
                else:            stack[top] = np.cos(a)
            else:
                top -= 1
                a = stack[top]
                b = stack[top+1]
                if   op == ADD: stack[top] = a + b
                elif op == SUB: stack[top] = a - b
                elif op == MUL: stack[top] = a * b
                elif op == DIV: stack[top] = a / b if a != 0 else 0.0
                elif op == POW: stack[top] = a ** b
                elif op == MIN: stack[top] = min(a, b)
                elif op == MAX: stack[top] = max(a, b)
                elif op == LT:  stack[top] = 1.0 if a < b else 0.0
                elif op == LE:  stack[top] = 1.0 if a <= b else 0.0
                elif op == GT:  stack[top] = 1.0 if a > b else 0.0
                else:           stack[top] = 1.0 if a >= b else 0.0
        if top >= 0:
            rate[f, C] = stack[0]*p.mult[f, g]
    return


@nb.jit(nopython=True, cache=True, inline='always')
def pair_rate(src, f, C, rate, ysum, isum):
    """ The rate at which events of flow f happen in class C: its rate per member of the source compartment times their number in the class, or in total for an inflow """
    r = max(rate[f, C], 0.0)
    return r*ysum[src[f], C] if src[f] >= 0 else r*isum[f, C]


@jit
def gillespie_groups(rng, y, p, dt, flows):
    """
    The exact solver over a time dt, for groups that affect each other (strata that mix, or places that people commute between), following only what each event changes

    Groups whose rates are always the same (they are in the same cells of each mixing term, and the same
    place if a rate has a count in it, with the same values; e.g. strata that differ only in who is in them)
    are a class, whose rates are worked out once, and whose events happen in proportion to the number in
    each flow's source compartment summed over the class. An event changes the people of one group, so the
    exposures are worked out again only for the cells that the mixing takes the change to (through the time
    share between places and the matrix of each place), and the rates only for the classes in those cells.
    The flow and class of each event are drawn from a tree of their rates, and the group from a tree of the
    counts in the source compartment by group, so the cost of an event grows with how far the mixing takes
    it rather than with the number of groups. The events are those of `event()` in distribution, though not
    draw for draw. See `coupled()` for the programs it takes.
    """
    nc, ng = y.shape
    nf, nt, ne = len(p.src), len(p.sizes), len(p.exp_set)
    counts = False
    for k in range(len(p.ops)):
        counts = counts or p.ops[k] == COUNT or p.ops[k] == FRAC

    # The classes, and the groups in each (in a row, in `order`)
    order = sort_groups(p, ng, counts)
    cls = np.empty(ng, dtype=np.int64)
    pos_of = np.empty(ng, dtype=np.int64)
    cstart = np.empty(ng + 1, dtype=np.int64)
    nC = 0
    for i in range(ng):
        if i == 0 or group_order(p, order[i-1], order[i], counts) != 0:
            cstart[nC] = i
            nC += 1
        cls[order[i]] = nC - 1
        pos_of[order[i]] = i
    cstart[nC] = ng
    rep = np.empty(nC, dtype=np.int64)
    for C in range(nC):
        rep[C] = order[cstart[C]]

    # The number in each compartment by class, and by group (a Fenwick tree in class order); the inflows, likewise
    top = 1
    while 2*top <= ng:
        top *= 2
    fen = np.zeros((nc, ng + 1))
    ysum = np.zeros((nc, nC))
    for c in range(nc):
        for i in range(ng):
            ysum[c, cls[order[i]]] += y[c, order[i]]
            fen[c, i+1] = y[c, order[i]]
        for i in range(1, ng + 1): # Each entry adds its sum to the next that covers it
            j = i + (i & (-i))
            if j <= ng:
                fen[c, j] += fen[c, i]
    isum = np.zeros((nf, nC))
    icum = np.zeros((nf, ng + 1))
    whole = np.zeros(nf, dtype=np.bool_)
    for f in range(nf):
        if p.src[f] < 0:
            for i in range(ng):
                isum[f, cls[order[i]]] += p.inflow[f, order[i]]
                icum[f, i+1] = icum[f, i] + p.inflow[f, order[i]]
        whole[f] = True
        for g in range(ng):
            total = 0.0
            for k in range(p.tgt_starts[f], p.tgt_starts[f+1]):
                total += p.tgt_w[k, g]
            whole[f] = whole[f] and abs(total - 1) < 1e-9

    # The counts by place, and the people
    cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total = sparse_maps(p, nc)
    npl = np.zeros(p.n_places)
    cnt = np.zeros((len(p.set_starts) - 1, p.n_places))
    for g in range(ng):
        for c in range(nc):
            if people[c]:
                npl[p.places[g]] += y[c, g]
            for k in range(cs_starts[c], cs_starts[c+1]):
                cnt[cs_sets[k], p.places[g]] += y[c, g]

    # The exposure of each cell of each mixing term, by its stages (see step()): the infectious (and how many, so that none is exactly none) and the people in it, those present after the time share, prevalence, mixed by the matrix, and shared between places
    S = p.sizes.max()
    num, icnt, an, prev, mx, post = np.zeros((ne, nt, S)), np.zeros((ne, nt, S)), np.zeros((ne, nt, S)), np.zeros((ne, nt, S)), np.zeros((ne, nt, S)), np.zeros((ne, nt, S))
    den, ad = np.zeros((nt, S)), np.zeros((nt, S))
    for g in range(ng):
        n = 0.0
        for c in p.people:
            n += y[c, g]
        for t in range(nt):
            den[t, p.levels[t, g]] += n
        for e in range(ne):
            s = p.exp_set[e]
            v, m = 0.0, 0.0
            for k in range(p.set_starts[s], p.set_starts[s+1]):
                v += y[p.set_comps[k], g]*p.rel_trans[e, g]
                m += y[p.set_comps[k], g]
            for t in range(nt):
                num[e, t, p.levels[t, g]] += v
                icnt[e, t, p.levels[t, g]] += m
    for t in range(nt):
        size, block = p.sizes[t], p.block[t]
        if p.use[0, t]:
            share(p.sh_pair, p.sh_val, den[t], size, block, ad[t], True)
        else:
            ad[t, :size] = den[t, :size]
        for e in range(ne):
            if p.use[0, t]:
                share(p.sh_pair, p.sh_val, num[e, t], size, block, an[e, t], True)
            else:
                an[e, t, :size] = num[e, t, :size]
            for i in range(size):
                prev[e, t, i] = an[e, t, i]/ad[t, i] if ad[t, i] > 0 else 0.0
            if p.use[1, t] and block == size:
                mix(p.mat[t], prev[e, t], size, mx[e, t])
            elif p.use[1, t]:
                by_block(p.mat[t], prev[e, t], size, block, mx[e, t])
            else:
                mx[e, t, :size] = prev[e, t, :size]
            if p.use[2, t]:
                share(p.sh_pair, p.sh_val, mx[e, t], size, block, post[e, t], False)
            else:
                post[e, t, :size] = mx[e, t, :size]

    # The entries of the time share by the place they are from (rows) and to (columns), the classes in each cell of each term, and in each place
    nnz = len(p.sh_val)
    n_rows = p.n_places
    for k in range(nnz):
        n_rows = max(n_rows, p.sh_pair[0, k] + 1, p.sh_pair[1, k] + 1)
    rstart, cstart_sh = np.zeros(n_rows + 1, dtype=np.int64), np.zeros(n_rows + 1, dtype=np.int64)
    for k in range(nnz):
        rstart[p.sh_pair[0, k] + 1] += 1
        cstart_sh[p.sh_pair[1, k] + 1] += 1
    rstart, cstart_sh = np.cumsum(rstart), np.cumsum(cstart_sh)
    rk, ck = np.empty(nnz, dtype=np.int64), np.empty(nnz, dtype=np.int64)
    rfill, cfill = rstart[:-1].copy(), cstart_sh[:-1].copy()
    for k in range(nnz):
        rk[rfill[p.sh_pair[0, k]]] = k
        rfill[p.sh_pair[0, k]] += 1
        ck[cfill[p.sh_pair[1, k]]] = k
        cfill[p.sh_pair[1, k]] += 1
    ccs_start = np.zeros((nt, S + 1), dtype=np.int64)
    ccs = np.empty((nt, nC), dtype=np.int64)
    for t in range(nt):
        for C in range(nC):
            ccs_start[t, p.levels[t, rep[C]] + 1] += 1
        for i in range(S):
            ccs_start[t, i+1] += ccs_start[t, i]
        fill = ccs_start[t, :-1].copy()
        for C in range(nC):
            cell = p.levels[t, rep[C]]
            ccs[t, fill[cell]] = C
            fill[cell] += 1
    pcs_start = np.zeros(p.n_places + 1, dtype=np.int64)
    pcs = np.empty(nC, dtype=np.int64)
    for C in range(nC):
        pcs_start[p.places[rep[C]] + 1] += 1
    pcs_start = np.cumsum(pcs_start)
    fill = pcs_start[:-1].copy()
    for C in range(nC):
        pcs[fill[p.places[rep[C]]]] = C
        fill[p.places[rep[C]]] += 1

    # The rates of each class, and the tree of the rate of each flow in each class
    expo = np.zeros((max(ne, 1), nC))
    rate = np.zeros((nf, nC))
    stack = np.empty(max(p.depth, 1))
    for C in range(nC):
        class_values(p, C, rep[C], expo, rate, post, cnt, npl, stack)
    n_pairs = nf*nC
    size = 1
    while size < n_pairs:
        size *= 2
    tree = np.zeros(2*size)
    for f in range(nf):
        for C in range(nC):
            tree[size + f*nC + C] = pair_rate(p.src, f, C, rate, ysum, isum)
    for j in range(size - 1, 0, -1):
        tree[j] = tree[2*j] + tree[2*j+1]

    # The events
    stamp = np.zeros(nC, dtype=np.int64) # The last event that marked each class to be worked out again, and each cell and place
    cell_stamp = np.zeros((nt, S), dtype=np.int64)
    place_stamp = np.zeros((nt, n_rows), dtype=np.int64)
    marked = np.empty(nC, dtype=np.int64)
    d1, d2, d3 = np.empty(S, dtype=np.int64), np.empty(S, dtype=np.int64), np.empty(S, dtype=np.int64)
    chc, chd = np.empty(2, dtype=np.int64), np.empty(2)
    elapsed = 0.0
    tick = 0
    while True:
        total = tree[1]
        if total <= 0:
            break
        elapsed += rng.exponential(1/total)
        if elapsed > dt:
            break
        tick += 1

        # Which flow and class, then which group
        v = rng.random()*total
        j = 1
        while j < size:
            if v < tree[2*j]:
                j = 2*j
            else:
                v -= tree[2*j]
                j = 2*j + 1
        leaf = j - size
        while leaf >= n_pairs or tree[size + leaf] <= 0: # Rounding at the end: the last with any chance
            leaf -= 1
        f, C = leaf//nC, leaf % nC
        s = p.src[f]
        a, b = cstart[C], cstart[C+1]
        if s >= 0:
            i = min(max(fen_find(fen, s, fen_sum(fen, s, a) + rng.random()*ysum[s, C], top), a), b - 1)
            while y[s, order[i]] <= 0 and i > a:
                i -= 1
        else:
            i = min(a + np.searchsorted(icum[f, a+1:b+1], icum[f, a] + rng.random()*isum[f, C], side='right'), b - 1)
            while p.inflow[f, order[i]] <= 0 and i > a:
                i -= 1
        g = order[i]

        # The event, as event() takes it
        nch = 0
        if s >= 0:
            chc[nch], chd[nch] = s, -1.0
            nch += 1
        if p.named[f]:
            flows[p.index[f], g] += 1
        k = pick(rng, p.tgt_starts[f], p.tgt_starts[f+1], p.tgt_w, g, whole[f])
        if k >= 0:
            chc[nch], chd[nch] = p.tgt_comp[k], 1.0
            nch += 1
            if p.tgt_rec[k] >= 0:
                flows[p.tgt_rec[k], g] += 1
        ca, cb = p.crd_starts[f], p.crd_starts[f+1]
        if cb > ca:
            credit_one(np.array([rng.random() for k in range(cb - ca)]), ca, cb, p, g, flows)

        # What it changes: the counts of its group, and its class, place, and cells
        C0, pl = cls[g], p.places[g]
        dn = 0.0
        moved = False
        counted = False
        for q in range(nch):
            c, d = chc[q], chd[q]
            y[c, g] += d
            ysum[c, C0] += d
            fen_add(fen, c, pos_of[g], d)
            if people[c]:
                dn += d
            for kk in range(cs_starts[c], cs_starts[c+1]):
                st = cs_sets[kk]
                cnt[st, pl] += d
                counted = True
                for qq in range(se_starts[st], se_starts[st+1]):
                    e = se_exps[qq]
                    moved = True
                    for t in range(nt):
                        cell = p.levels[t, g]
                        icnt[e, t, cell] += d
                        num[e, t, cell] = num[e, t, cell] + d*p.rel_trans[e, g] if icnt[e, t, cell] != 0 else 0.0
        if dn != 0:
            npl[pl] += dn
            counted = True
            moved = True
            for t in range(nt):
                den[t, p.levels[t, g]] += dn
        nm = 0
        stamp[C0] = tick
        marked[nm] = C0
        nm += 1
        if counted and counts:
            for kk in range(pcs_start[pl], pcs_start[pl+1]):
                if stamp[pcs[kk]] != tick:
                    stamp[pcs[kk]] = tick
                    marked[nm] = pcs[kk]
                    nm += 1
        for t in range(nt if moved else 0):
            size_t, block = p.sizes[t], p.block[t]
            i0 = p.levels[t, g]
            n1 = 0 # The cells where who is present changes
            if p.use[0, t]:
                l = i0 % block
                for kk in range(rstart[i0//block], rstart[i0//block + 1]):
                    pb = p.sh_pair[1, rk[kk]]
                    cell = pb*block + l
                    d1[n1] = cell
                    n1 += 1
                    ad[t, cell] = 0.0
                    for e in range(ne):
                        an[e, t, cell] = 0.0
                    for q in range(cstart_sh[pb], cstart_sh[pb+1]):
                        x = ck[q]
                        src_cell = p.sh_pair[0, x]*block + l
                        ad[t, cell] += p.sh_val[x]*den[t, src_cell]
                        for e in range(ne):
                            an[e, t, cell] += p.sh_val[x]*num[e, t, src_cell]
            else:
                d1[0] = i0
                n1 = 1
                ad[t, i0] = den[t, i0]
                for e in range(ne):
                    an[e, t, i0] = num[e, t, i0]
            for q in range(n1):
                cell = d1[q]
                for e in range(ne):
                    prev[e, t, cell] = an[e, t, cell]/ad[t, cell] if ad[t, cell] > 0 else 0.0
            n2 = 0 # The cells whose mixed prevalence changes
            if p.use[1, t] and block == size_t:
                for e in range(ne):
                    mix(p.mat[t], prev[e, t], size_t, mx[e, t])
                for cell in range(size_t):
                    d2[cell] = cell
                n2 = size_t
            elif p.use[1, t]:
                for q in range(n1):
                    pb = d1[q]//block
                    if place_stamp[t, pb] == tick:
                        continue
                    place_stamp[t, pb] = tick
                    for ii in range(block):
                        for e in range(ne):
                            acc = 0.0
                            for jj in range(block):
                                acc += p.mat[t, ii, jj]*prev[e, t, pb*block + jj]
                            mx[e, t, pb*block + ii] = acc
                        d2[n2] = pb*block + ii
                        n2 += 1
            else:
                for q in range(n1):
                    d2[q] = d1[q]
                    for e in range(ne):
                        mx[e, t, d1[q]] = prev[e, t, d1[q]]
                n2 = n1
            n3 = 0 # The cells whose exposure changes
            if p.use[2, t]:
                for q in range(n2):
                    pb, l = d2[q]//block, d2[q] % block
                    for kk in range(cstart_sh[pb], cstart_sh[pb+1]):
                        pa = p.sh_pair[0, ck[kk]]
                        cell = pa*block + l
                        if cell_stamp[t, cell] == tick:
                            continue
                        cell_stamp[t, cell] = tick
                        for e in range(ne):
                            acc = 0.0
                            for r in range(rstart[pa], rstart[pa+1]):
                                x = rk[r]
                                acc += p.sh_val[x]*mx[e, t, p.sh_pair[1, x]*block + l]
                            post[e, t, cell] = acc
                        d3[n3] = cell
                        n3 += 1
            else:
                for q in range(n2):
                    d3[q] = d2[q]
                    for e in range(ne):
                        post[e, t, d2[q]] = mx[e, t, d2[q]]
                n3 = n2
            for q in range(n3):
                cell = d3[q]
                for kk in range(ccs_start[t, cell], ccs_start[t, cell+1]):
                    if stamp[ccs[t, kk]] != tick:
                        stamp[ccs[t, kk]] = tick
                        marked[nm] = ccs[t, kk]
                        nm += 1

        # The rates of the classes it changes
        for q in range(nm):
            C = marked[q]
            class_values(p, C, rep[C], expo, rate, post, cnt, npl, stack)
            for ff in range(nf):
                seg_set(tree, size, ff*nC + C, pair_rate(p.src, ff, C, rate, ysum, isum))
    return


@jit
def mix(matrix, v, size, out):
    """ The product of the first `size` rows and columns of a mixing matrix with a vector (`matrix @ v`), into out; by BLAS when the whole matrix is used (so is contiguous) and is large enough that the call costs less than the loop """
    if size == matrix.shape[0] and size > 16:
        out[:size] = np.dot(matrix, v[:size])
        return
    for i in range(size):
        acc = 0.0
        for j in range(size):
            acc += matrix[i, j]*v[j]
        out[i] = acc
    return


@jit
def by_block(matrix, v, size, block, out):
    """ The product of the first `block` rows and columns of a mixing matrix with each block of `block` entries of a vector, into out (`matrix @ v` for each place) """
    for a in range(0, size, block):
        for i in range(block):
            acc = 0.0
            for j in range(block):
                acc += matrix[i, j]*v[a + j]
            out[a + i] = acc
    return


@jit
def share(pair, vals, v, size, block, out, by_columns):
    """ The product of the time share between places (its entries: see `Program`) with a vector of places by levels (each level in turn, `block` of them), into out: `v @ share` if `by_columns`, else `share @ v` """
    for i in range(size):
        out[i] = 0.0
    for k in range(len(vals)):
        a, b = pair[0, k], pair[1, k]
        if by_columns: # From place a to place b
            for i in range(block):
                out[b*block + i] += vals[k]*v[a*block + i]
        else:
            for i in range(block):
                out[a*block + i] += vals[k]*v[b*block + i]
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
def convey(rng, y, p, w, flows, release):
    """
    The conveyors of fixed delays and minimum stays: release the people in the last slot of each, or move them all along one slot

    A conveyor of K steps has K+1 slots. People arrive in slot 0 during a step, and at its
    end (`release=False`) everyone moves along one slot, so at each time point slot j holds
    the people who entered j steps ago, and the last those who have been there K steps.
    These leave in the next step: the chain-binomial solver releases them at its start (once
    the step's rates are worked out from the people as they were, as all its rates are), as
    agents leave at the start of the step their dwell time ends in, and Gillespie and
    tau-leaping at its middle (`release=True`); the ODE lets them leave at a constant rate
    over the step (see `step()`), which is exact at each time point for a delay of whole steps.
    The other exits of the state act on every slot, the last until its people leave it (for
    the ODE, see `release_rates()`). A place with people in a conveyor is stepped by the exact
    solver by place, since the people in it move.

    Each conveyor is a set of operations after the flows' (see `Program.starts`), all CONVEY:
    the first one's argument is the flow the last slot leaves by, and the others' are its
    slots, in order. The flow is an inflow in the program, whose last target is the last slot,
    with a share of -1: for the ODE, `step()` sets its inflow to the people due to leave over
    the step, so the loops over flows move them; otherwise its inflow is 0.
    """
    ng = y.shape[1]
    nf = len(p.src)
    groups, idle = w[3], w[4][5]
    n = groups[5]
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] != CONVEY:
            continue
        if release:
            last = p.args[b-1]
            f = p.args[a]
            for g in range(ng):
                n[g] = y[last, g]
                y[last, g] = 0
                if n[g] > 0:
                    idle[p.places[g]] = 0
            if p.named[f]:
                land(rng, n, p.record[f], p.maps, flows[p.index[f]])
            split(rng, n, p.tgt_starts[f], p.tgt_starts[f+1] - 1, p.tgt_w, False, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows) # Not the last target, which is the slot they leave
            credit(rng, n, p.crd_starts[f], p.crd_starts[f+1], p, w[9], flows)
        else:
            for g in range(ng):
                for k in range(b-1, a+1, -1): # The last slot is empty by now (or nearly, by rounding, in the ODE)
                    y[p.args[k], g] = y[p.args[k-1], g]
                    if y[p.args[k], g] > 0:
                        idle[p.places[g]] = 0
                y[p.args[a+1], g] = 0
    return


@jit
def update_values(y, p, stack, groups, cells):
    """
    The values that counts are set to at the end of a step (`ss.update()`), from the compartments at its start, into the rows of `groups` after the scratch

    Worked out as the rates in `step()` are, for the operations an update can have (any but
    exposure, which is contact): called once a step, and only if the program has conveyors or
    updates. Each update is a set of operations after the flows' (see `Program.starts`), the
    last of which is ASSIGN, whose argument is the compartment it sets.
    """
    ng = y.shape[1]
    nf = len(p.src)
    n, local_n, vals = groups[0], groups[1], groups[2]
    num = cells[0]
    for g in range(ng): # The people in each group, and in the same place as each group
        n[g] = 0
        for c in p.people:
            n[g] += y[c, g]
    for i in range(p.n_places):
        num[i] = 0
    for g in range(ng):
        num[p.places[g]] += n[g]
    for g in range(ng):
        local_n[g] = num[p.places[g]]
    for u in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+u], p.starts[nf+u+1]
        if p.ops[b-1] != ASSIGN:
            continue
        top = -1
        for k in range(a, b-1):
            op = p.ops[k]
            arg = p.args[k]
            if op == CONST:
                top += 1
                for g in range(ng):
                    stack[top, g] = p.consts[arg, g]
            elif op == TOTAL or op == COUNT or op == FRAC:
                top += 1
                for i in range(p.n_places):
                    num[i] = 0
                for g in range(ng):
                    vals[g] = 0
                    for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                        vals[g] += y[p.set_comps[i], g]
                    num[p.places[g]] += vals[g]
                total = 0.0
                for i in range(p.n_places):
                    total += num[i]
                for g in range(ng):
                    if op == TOTAL:
                        stack[top, g] = total
                    elif op == FRAC:
                        stack[top, g] = num[p.places[g]]/local_n[g] if local_n[g] > 0 else 0.0
                    else:
                        stack[top, g] = num[p.places[g]]
            elif op >= EXP:
                for g in range(ng):
                    x = stack[top, g]
                    if   op == EXP:  stack[top, g] = np.exp(x)
                    elif op == LOG:  stack[top, g] = np.log(x)
                    elif op == SQRT: stack[top, g] = np.sqrt(x)
                    elif op == ABS:  stack[top, g] = abs(x)
                    elif op == SIN:  stack[top, g] = np.sin(x)
                    else:            stack[top, g] = np.cos(x)
            else:
                top -= 1
                for g in range(ng):
                    x = stack[top, g]
                    z = stack[top+1, g]
                    if   op == ADD: stack[top, g] = x + z
                    elif op == SUB: stack[top, g] = x - z
                    elif op == MUL: stack[top, g] = x * z
                    elif op == DIV: stack[top, g] = x / z if x != 0 else 0.0
                    elif op == POW: stack[top, g] = x ** z
                    elif op == MIN: stack[top, g] = min(x, z)
                    elif op == MAX: stack[top, g] = max(x, z)
                    elif op == LT:  stack[top, g] = 1.0 if x < z else 0.0
                    elif op == LE:  stack[top, g] = 1.0 if x <= z else 0.0
                    elif op == GT:  stack[top, g] = 1.0 if x > z else 0.0
                    else:           stack[top, g] = 1.0 if x >= z else 0.0
        for g in range(ng):
            groups[7+u, g] = stack[0, g]
    return


@jit
def step_start(y, p, w, dt, solver):
    """
    For `step()`, in a program with conveyors or updates: the values of the updates, from the compartments at the start of the step, and for the ODE, the people due to leave each conveyor over it

    Returns whether a stochastic solver has people to release from a conveyor in the step. Kept
    out of `step()` (as is `step_end()`), since code there, even if a model never runs it, slows
    the steps of every model by a few percent.
    """
    nf = len(p.src)
    ng = y.shape[1]
    update_values(y, p, w[2], w[3], w[4])
    pending = False
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] == CONVEY:
            pending = solver >= 2
            if solver < 2: # For the ODE, they leave at a constant rate over the step, as the inflow of the conveyor's flow
                for g in range(ng):
                    p.inflow[p.args[a], g] = y[p.args[b-1], g]/dt
    return pending


@jit
def release_rates(y, p, rates, dt):
    """
    For the ODE, in a program with conveyors: the rate at which the people due to leave each conveyor leave it over the step, once the rates of the step are worked out

    They leave at a constant rate (see `step_start()`), and the other exits of the state act on them
    until they do, so the rate is lowered for the slot to be empty at the end of the step: with a
    total rate h of those exits, to a share ψ(h·dt) = h·dt/(exp(h·dt) - 1) of the people in it. With
    those arriving in the first slot over a step, of whom a share (1 - exp(-h·dt))/(h·dt) are left at
    its end, the exits act over the whole delay, as for agents: a delay of K steps keeps a share
    exp(-h·K·dt), where leaving the last slot by itself kept exp(-h·(K - 1/2)·dt).
    """
    nf = len(p.src)
    ng = y.shape[1]
    for i in range(len(p.starts) - nf - 1):
        a, b = p.starts[nf+i], p.starts[nf+i+1]
        if p.ops[a] != CONVEY:
            continue
        last = p.args[b-1]
        f_out = p.args[a]
        for g in range(ng):
            x = 0.0
            for f in range(nf):
                if p.src[f] == last:
                    x += max(rates[f, g], 0.0)*dt
            if x > 1e-12:
                p.inflow[f_out, g] *= x/np.expm1(x)
    return


@jit
def step_end(rng, y, p, w, flows):
    """ For `step()`, in a program with conveyors or updates: everyone in a conveyor moves along one slot, and the counts that are updated are set (see `step_start()`) """
    nf = len(p.src)
    ng = y.shape[1]
    groups = w[3]
    convey(rng, y, p, w, flows, False)
    for i in range(len(p.starts) - nf - 1):
        b = p.starts[nf+i+1]
        if p.ops[b-1] == ASSIGN:
            for g in range(ng):
                y[p.args[b-1], g] = groups[7+i, g]
    return


@jit
def step(rng, y, p, w, dt, n_sub, solver, flows, tables, row):
    """
    Take one timestep

    The ODE is stepped by Euler's method or fourth-order Runge-Kutta; the stochastic
    solvers work out the rates here and hand them to `draw()` or `event()`. Everything is in
    this one function, and written as loops over single numbers, because that is many
    times faster for a small model: a call that is passed the program costs more than
    working out the rates of a few compartments does.

    Args:
        y (array): the compartments, by group; changed in place
        p (Program): the flows
        w (tuple): the arrays to work in: see `workspace()`
        dt (float): the timestep
        n_sub (int): the number of substeps; if 0, for the ODE, enough that no compartment turns over, or grows through its inflows, by more than a quarter in one, chosen at the start of each step (tau-leaping and the exact solver choose their own)
        solver (int): which of `solvers` to use
        flows (array): filled in with the total of each transition and mover over the step
        tables (tuple): the fixed, mult, consts, and rel_trans tables (see `run()`), with three rows per step
        row (int): the row of the start of this step, or -1 if the program holds for the whole step
    """
    rates, exposures, stack, groups, cells, dy, df, yt, chain, credits = w
    n, local_n, vals, infectious = groups[0], groups[1], groups[2], groups[3]
    num, den, num2, den2, prev = cells[0], cells[1], cells[2], cells[3], cells[4]
    nc, ng = y.shape
    nf = len(p.src)
    nflows = p.n_trans + p.n_movers
    stages = 4 if solver == 0 else 1
    h = dt
    timed = solver == 3 or solver == 4 # Tau-leaping and the exact solver take steps of their own length
    by_groups = solver == 4 and (not p.local or p.n_places == 1) and coupled(p, ng) # The exact solver for groups that affect each other (or are all in one place), following only what each event changes (see gillespie_groups())
    by_place = solver == 4 and p.local and not by_groups # The exact solver by place, where each place works out its own rates (see gillespie_places())
    auto = solver < 2 and n_sub <= 0 # The ODE, with as many substeps as the step needs
    elapsed = 0.0 # For these: how much of the timestep has been taken
    fastest = 0.0
    sub = 0
    extra = len(p.starts) - nf - 1 # Conveyors and updates, whose operations follow the flows'
    pending = step_start(y, p, w, dt, solver) if extra else False # Whether the people due to leave a conveyor have yet to leave
    while True:
        for stage in range(stages):
            ys = y if stage == 0 else yt # Where the rates are worked out
            if row >= 0: # Values that vary in time, at the time of this stage; for the stochastic solvers, at the middle of each substep, or of each quarter of the step for those that take steps of their own length
                if solver < 2:
                    th = (sub + (0.0 if stage == 0 else (1.0 if stage == 3 else 0.5)))/max(n_sub, 1)
                elif timed:
                    th = (min(int((elapsed + 1e-12)*pieces/dt), pieces - 1) + 0.5)/pieces
                else:
                    th = (sub + 0.5)/max(n_sub, 1)
                for k in range(4):
                    if len(tables[k]) > 1:
                        interp((p.fixed, p.mult, p.consts, p.rel_trans)[k], tables[k], row, th)

            # The rate of every flow, by group: per member of its source compartment, or in total for an inflow
            for f in range(nf):
                for g in range(ng):
                    rates[f, g] = p.fixed[f, g]
            if len(p.ops) and not by_place:
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
                    if s < 0: # The sum of the -s exposures before it, in order
                        for g in range(ng):
                            exposures[e, g] = exposures[e+s, g]
                            for x in range(e+s+1, e):
                                exposures[e, g] += exposures[x, g]
                        continue
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
                        block = p.block[t]
                        if p.use[0, t]: # Move people to where they spend their time
                            share(p.sh_pair, p.sh_val, num, size, block, num2, True)
                            share(p.sh_pair, p.sh_val, den, size, block, den2, True)
                            for i in range(size):
                                num[i] = num2[i]
                                den[i] = den2[i]
                        for i in range(size):
                            prev[i] = num[i]/den[i] if den[i] > 0 else 0.0
                        if p.use[1, t]: # Mix between the levels in each place
                            if block == size:
                                mix(p.mat[t], prev, size, num)
                            else:
                                by_block(p.mat[t], prev, size, block, num)
                            for i in range(size):
                                prev[i] = num[i]
                        if p.use[2, t]: # Mix between places
                            share(p.sh_pair, p.sh_val, prev, size, block, num, False)
                            for i in range(size):
                                prev[i] = num[i]
                        if p.contacts.shape[0] > 1 or p.contacts.shape[2] > 1: # Contacts by exposure and group
                            for g in range(ng):
                                exposures[e, g] += p.contacts[e, t, g]*prev[p.levels[t, g]]
                        else:
                            c = p.contacts[0, t, 0]
                            for g in range(ng):
                                exposures[e, g] += c*prev[p.levels[t, g]]

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
                        elif op == TOTAL: # The number in some compartments, in every place
                            top += 1
                            v = 0.0
                            for g in range(ng):
                                for i in range(p.set_starts[arg], p.set_starts[arg+1]):
                                    v += ys[p.set_comps[i], g]
                            for g in range(ng):
                                stack[top, g] = v
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
                                elif op == DIV: stack[top, g] = a / b if a != 0 else 0.0
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

            if solver == 3: # Tau-leaping: the fastest rate at which any compartment is being left, by all its flows
                fastest = 0.0
                for s in range(len(p.src_starts) - 1):
                    if p.src[p.src_flows[p.src_starts[s]]] >= 0:
                        for g in range(ng):
                            leaving = 0.0
                            for k in range(p.src_starts[s], p.src_starts[s+1]):
                                leaving += max(rates[p.src_flows[k], g], 0.0)
                            fastest = max(fastest, leaving)
            elif sub == 0 and stage == 0 and n_sub <= 0: # The fastest rate of any flow out of a compartment
                fastest = 0.0
                for f in range(nf):
                    if p.src[f] >= 0:
                        for g in range(ng):
                            fastest = max(fastest, rates[f, g])
            if sub == 0 and stage == 0:
                if solver < 2 and n_sub <= 0: # Enough substeps that no compartment turns over by more than a quarter in one
                    for e in range(len(p.mov_rate)):
                        fastest = max(fastest, p.mov_rate[e])
                    n_sub = int(np.ceil(fastest*dt/0.25))
                n_sub = max(n_sub, 1)
                h = dt/n_sub
                for f in range(nflows):
                    for g in range(ng):
                        flows[f, g] = 0
                if extra and solver < 2:
                    release_rates(y, p, rates, dt)
            if solver >= 2: # The stochastic solvers take it from here
                end = dt if row < 0 else min(int((elapsed + 1e-12)*pieces/dt) + 1, pieces)*dt/pieces # The end of the piece of the step that values which vary in time are held for; events are exact within it, since the rates are held
                if pending and timed: # The people due to leave a conveyor leave at the middle of the step
                    end = min(end, dt/2)
                if by_place: # The exact solver, place by place: the whole step (or piece) at once
                    varying = row >= 0 or p.has_det # Rates that may have changed since the last step, in every place
                    for k in range(4):
                        varying = varying or len(tables[k]) > 1
                    gillespie_places(rng, y, p, w, end - elapsed, flows, varying)
                    h = end - elapsed
                elif by_groups: # The same for groups that affect each other
                    gillespie_groups(rng, y, p, end - elapsed, flows)
                    h = end - elapsed
                elif solver == 4: # The exact solver: up to the next event
                    h = event(rng, y, p, w, end - elapsed, flows)
                else:
                    if solver == 3: # Tau-leaping: a leap short enough that the rates change little over it
                        h = min(end - elapsed, 0.03/max(fastest, 1e-12))
                    elif pending: # The chain-binomial solver: they leave at the start of the step, once the rates are worked out from the people as they were (as agents' scheduled exits come first, but not in the counts the rates use)
                        convey(rng, y, p, w, flows, True)
                        pending = False
                    draw(rng, y, p, w, h, flows, solver == 3)
                if p.has_det and not by_place: # By place, the counts are settled after the step's events: see settle_places()
                    settle(y, p, rates, h, flows)
                elapsed += h
                if pending and timed and elapsed >= dt/2 - 1e-12:
                    convey(rng, y, p, w, flows, True)
                    pending = False
                continue

            # The rate of change of each compartment, and the rate of each transition and mover
            for c in range(nc):
                for g in range(ng):
                    dy[stage, c, g] = 0
            for f in range(nflows):
                for g in range(ng):
                    df[stage, f, g] = 0
            for m in range(p.n_movers):
                for k in range(p.mov_starts[0, m], p.mov_starts[0, m+1]):
                    c = p.mov_comps[k]
                    for e in range(p.mov_starts[1, m], p.mov_starts[1, m+1]):
                        r = p.mov_rate[e]
                        if r != 0:
                            g, j = p.mov_pair[0, e], p.mov_pair[1, e]
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
                            if p.tgt_rec[k] >= 0:
                                df[stage, p.tgt_rec[k], g] += amount*p.tgt_w[k, g]
                            if p.tgt_map[k] < 0:
                                dy[stage, p.tgt_comp[k], g] += amount*p.tgt_w[k, g]
                            else:
                                for j in range(ng):
                                    dy[stage, p.tgt_comp[k], j] += amount*p.tgt_w[k, g]*p.maps[p.tgt_map[k], g, j]
                        for k in range(p.crd_starts[f], p.crd_starts[f+1]):
                            df[stage, p.crd_index[k], g] += amount*p.crd_w[k, g]

            if auto and sub == 0 and stage == 0: # ... nor grows by more than a quarter in one, through its inflows (an epidemic's infected, growing much faster than they recover)
                growing = 0.0
                for g in range(ng):
                    total = 0.0
                    for c in range(nc):
                        total += y[c, g]
                    for c in range(nc):
                        if dy[0, c, g] > 0 and y[c, g] > 1e-12*total: # Not a compartment that is only starting to fill, whose growth follows another's
                            growing = max(growing, dy[0, c, g]/y[c, g])
                more = int(np.ceil(growing*dt/0.25))
                if more > n_sub:
                    n_sub = more
                    h = dt/n_sub

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
    if solver == 4: # The exact solver moves people between groups once, over the whole step
        move(rng, y, p.mov_starts, p.mov_comps, p.mov_pair, p.mov_rate, p.n_trans, dt, flows, True)
        for e in range(len(p.mov_rate)): # The places they move between are stepped again
            if p.mov_rate[e] != 0:
                cells[5, p.places[p.mov_pair[0, e]]] = 0
                cells[5, p.places[p.mov_pair[1, e]]] = 0
    if extra:
        step_end(rng, y, p, w, flows)
    for c in range(nc):
        for g in range(ng):
            if y[c, g] < 0:
                y[c, g] = 0
    return


@jit
def apply_events(rng, y, r, ti, whole):
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
                        take = hypergeometric(rng, have, rest - have, left)
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
                part = binomial(rng, left, 1/remaining) if whole else left/remaining
                left -= part
                wrest = 0.0
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    wrest += r.ent_w[k, r.pg_groups[j]]
                for j in range(r.pg_starts[dst], r.pg_starts[dst+1]):
                    g = r.pg_groups[j]
                    w = r.ent_w[k, g]
                    if part <= 0 or wrest <= 0:
                        break
                    take = binomial(rng, part, w/wrest) if whole else part*w/wrest
                    y[r.ent_comps[k], g] += take
                    part -= take
                    wrest -= w
    return short


#%% A whole run

solvers = dict(rk4=0, euler=1, binomial=2, tau=3, gillespie=4)

@jit
def apply_shifts(rng, y, sh, ti, whole, tally):
    """
    Apply the deliveries that happen at time point ti, at the start of the step from it, in order

    Args:
        y (array): the compartments, by group; changed in place
        sh (Shifts): the deliveries
        ti (int): the time point
        whole (bool): whether the compartments hold whole people, so the people reached are drawn
        tally (array): the number reached and the number eligible, for each delivery and group (deliveries by 2 by groups); added to
    """
    nc, ng = y.shape
    for k in range(len(sh.to)):
        if not sh.due[k, ti]:
            continue
        for c in range(nc):
            before = y[c].copy() # Each group's people as they were, so nobody moved into a group is moved on again (a counter up a level)
            for g in range(ng):
                e = sh.sel[k, c, g]
                n = before[g]
                if e <= 0 or n <= 0:
                    continue
                p = sh.cov[k, c, g]
                tally[k, 0, sh.rest[k, g]] += n*e*p # The expected number, as a compartmental run counts them, in the group of those offered it (see ss.Delivery.mark_offered())
                tally[k, 1, g] += n*e
                if whole:
                    a = binomial(rng, n, e*p)
                    b = binomial(rng, n - a, e*(1 - p)/(1 - e*p)) if e*p < 1 else 0.0
                else:
                    a, b = n*e*p, n*e*(1 - p)
                to, rest = sh.to[k, g], sh.rest[k, g]
                if a > 0 and to != g:
                    y[c, g] -= a
                    y[c, to] += a
                if b > 0 and rest != g:
                    y[c, g] -= b
                    y[c, rest] += b
    return


@jit
def set_step(parts, ti, tables):
    """
    Set the parts of the program that can change over time to their values at a step

    Args:
        parts (tuple): the program's fixed, mult, consts, and rel_trans (not the program itself, since a call that is passed it costs more than a small step)
        ti (int): the row of the tables
        tables (tuple): the values of each, with one entry per step (or per row: see `run()`), or a single entry
    """
    for k in range(4):
        if ti < len(tables[k]):
            parts[k][:] = tables[k][ti]
    return


@jit
def run(rng, y, p, dt, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, tally, per_step=1, start=0):
    """
    Take every step of a run, or the steps from time point `start` on

    Args:
        y (array): the compartments at the start, by group
        p (Program): the flows
        dt (float): the timestep
        npts (int): the number of time points: the start, and the end of each of the npts-1 steps
        solver (int): which of `solvers` to use
        n_sub (int): for the ODE, the number of substeps (see `step()`)
        rng (Generator): the random numbers to draw from
        fixed, mult, consts, rel_trans (arrays): the values of these parts of the program at each step, or a single
            entry if they are the same throughout
        reg (Register): the events of a movement register, applied at the end of the step they are dated in (those of the start are already in y)
        sh (Shifts): the deliveries, applied at the start of the step from the time point they happen at
        tally (array): filled in with the number each delivery reached and found eligible, by group (time points by deliveries by 2 by groups)
        per_step (int): 3 if the tables have the values at the start, middle, and end of each step (for the ODE), and otherwise 1
        start (int): the time point that y is at; the run is taken from there to time point npts-1, so it can be taken a part at a time
            (the tables then start at the step from `start`, unless they have a single entry)

    Returns the compartments at the end, the compartments at each time point (the start, then the end of each
    step), the total of each transition and mover over the step that ends at each time point (0 at the start),
    and the number of people that events of the register could not find. The time points are those from
    `start` on, as is `tally`.
    """
    y = y.copy()
    w = workspace(y, p)
    whole = solver >= 2
    stocks = np.empty((npts - start, y.shape[0], y.shape[1]))
    flows = np.zeros((npts - start, p.n_trans + p.n_movers, y.shape[1]))
    short = 0.0
    tables = (fixed, mult, consts, rel_trans)
    stocks[0] = y
    for ti in range(start + 1, npts): # Step ti goes from time point ti-1 to ti, with the values at the start of it
        k = ti - start
        if len(sh.to):
            apply_shifts(rng, y, sh, ti-1, whole, tally[k-1])
            if sh.due[:, ti-1].any():
                wake(w)
        set_step((p.fixed, p.mult, p.consts, p.rel_trans), (k-1)*per_step, tables)
        step(rng, y, p, w, dt, n_sub, solver, flows[k], tables, (k-1)*per_step if per_step == 3 else -1)
        if len(reg.src): # The events dated at its end (not called if there are none, since passing the register costs as much as a small step)
            short += apply_events(rng, y, reg, ti, whole)
            if solver == 4 and p.local:
                touch(reg, ti, p.places, w)
        if solver == 4 and p.local and p.has_det:
            settle_places(y, p, dt, flows[k])
        stocks[k] = y
    return y, stocks, flows, short


@jit
def run_many(rngs, ys, p, dt, start, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, stocks, flows, stocks_g, flows_g, tally, tally_g, state_of, rows, grp_state, grp_flow, per_step=1):
    """
    Take the steps from time point `start` to npts-1 of many runs that differ only in their random numbers, filling in their results

    Args:
        rngs (list): the random numbers of each run, as a typed list of generators
        ys (array): the compartments of each run at time point `start`, by group; changed in place
        start (int): the time point to start from, so the runs can be taken a part at a time (the tables then start at the step from it, unless they have a single entry)
        stocks, flows (arrays): filled in with the number in each state at each time point, and the total of each transition and mover over the step that ends at it, as runs by time points by states (or transitions) by 1 (the total)
        stocks_g, flows_g (arrays): those kept by group, as runs by rows by states (or transitions) by groups: a state as it is at the last time point of its row, and a flow added up over them (see `ss.Sim.group_rows()`)
        tally (array): filled in with the total number each delivery reached and found eligible, as runs by time points by deliveries by 2 by 1
        tally_g (array): the same by group, in the rows of `rows` (runs by rows by deliveries by 2 by groups)
        state_of (array): the state that each compartment is part of
        rows (array): the row of each time point in the values kept by group (none, if they are not kept)
        grp_state (array): for each state, its position among those kept by group, or -1
        grp_flow (array): the same for each transition and mover
        (the others are as for `run()`)
    """
    nc, ng = ys.shape[1], ys.shape[2]
    nflows = p.n_trans + p.n_movers
    n_rows = len(stocks_g[0]) if len(rows) else 0
    w = workspace(ys[0], p)
    whole = solver >= 2
    now = np.empty((nflows, ng))
    tables = (fixed, mult, consts, rel_trans)
    now_sh = np.zeros((len(sh.to), 2, ng))
    for r in range(len(rngs)):
        rng = rngs[r]
        y = ys[r]
        wake(w) # Each run steps every place first
        for ti in range(start, npts):
            if ti > start:
                k = ti - start
                if len(sh.to):
                    now_sh[:] = 0
                    apply_shifts(rng, y, sh, ti-1, whole, now_sh)
                    if sh.due[:, ti-1].any():
                        wake(w)
                    for d in range(len(sh.to)):
                        for g in range(ng):
                            tally[r, ti-1, d, 0, 0] += now_sh[d, 0, g]
                            tally[r, ti-1, d, 1, 0] += now_sh[d, 1, g]
                            if n_rows:
                                tally_g[r, rows[ti-1], d, 0, g] += now_sh[d, 0, g]
                                tally_g[r, rows[ti-1], d, 1, g] += now_sh[d, 1, g]
                set_step((p.fixed, p.mult, p.consts, p.rel_trans), (k-1)*per_step, tables)
                step(rng, y, p, w, dt, n_sub, solver, now, tables, (k-1)*per_step if per_step == 3 else -1)
                if len(reg.src):
                    apply_events(rng, y, reg, ti, whole)
                    if solver == 4 and p.local:
                        touch(reg, ti, p.places, w)
                if solver == 4 and p.local and p.has_det:
                    settle_places(y, p, dt, now)
                for f in range(nflows):
                    for g in range(ng):
                        flows[r, ti, f, 0] += now[f, g]
                        if n_rows and grp_flow[f] >= 0:
                            flows_g[r, rows[ti], grp_flow[f], g] += now[f, g]
            elif ti > 0: # The start of a later part, which the part before has recorded
                continue
            end = n_rows > 0 and (ti == len(rows) - 1 or rows[ti+1] != rows[ti]) # The last time point of its row
            for c in range(nc):
                s = state_of[c]
                for g in range(ng):
                    stocks[r, ti, s, 0] += y[c, g]
                    if end and grp_state[s] >= 0:
                        stocks_g[r, rows[ti], grp_state[s], g] += y[c, g]
    return


inline_step = nb.jit(nopython=True, cache=True, error_model='numpy', inline='always')(step.py_func) # step(), compiled into advance() rather than called from it, since a call that is passed the program costs as much as a small step


@jit
def advance(rngs, p, ys, first, last, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step=1, row0=0):
    """
    Take steps start+1 to end of runs first to last-1 (particles of a particle filter), each from its own compartments, with its own random numbers

    Args:
        rngs (list): the random numbers of each run, as a typed list of generators
        p (Program): the flows; the parts of it that vary in time are changed as it runs, so each thread needs its own (see `advance_threads()`)
        ys (array): the compartments of each run, by group; changed in place
        first, last (int): the runs to take
        start, end (int): the time points to step from and to
        fixed, mult, consts, rel_trans (arrays): as for `run()`, with a first axis for the runs: one entry if every run has the same values, or one for each run (as each particle of IF2 has its own parameter values)
        sh_vals (array): the time share between places (`sh_val` of the program) of each run, as runs by entries: one entry if every run has the program's own
        row0 (int): the row of the tables that their first row is (those with more than one), if they do not start at the start of the run
        state_of (array): the state that each compartment is part of
        stocks, flows (arrays): filled in at time points start+1 to end, as `run_many()` returns them (by group, or the
            total in one group if they have only one)
        (the others are as for `run()`)
    """
    nc, ng = ys.shape[1], ys.shape[2]
    nflows = p.n_trans + p.n_movers
    by_group = stocks.shape[3] == ng
    w = workspace(ys[0], p)
    whole = solver >= 2
    now = np.empty((nflows, ng))
    tally = np.zeros((len(sh.to), 2, ng)) # Not kept
    parts = (p.fixed, p.mult, p.consts, p.rel_trans)
    own = ys[first:last].copy() # Copies of its own of these runs and their values, since threads that share an array contend for the count of references to it
    mine = [table[first:last].copy() if len(table) > 1 else table.copy() for table in (fixed, mult, consts, rel_trans)]
    for r in range(first, last):
        rng = rngs[r]
        y = own[r - first]
        tables = (mine[0][r - first if len(fixed) > 1 else 0], mine[1][r - first if len(mult) > 1 else 0], mine[2][r - first if len(consts) > 1 else 0], mine[3][r - first if len(rel_trans) > 1 else 0])
        set_step(parts, 0, tables) # The values of this run, for those that are held for the whole run
        if len(sh_vals) > 1:
            p.sh_val[:] = sh_vals[r]
        wake(w) # Each run steps every place first
        for ti in range(start + 1, end + 1):
            if len(sh.to):
                apply_shifts(rng, y, sh, ti-1, whole, tally)
                if sh.due[:, ti-1].any():
                    wake(w)
            set_step(parts, (ti-1)*per_step - row0, tables)
            inline_step(rng, y, p, w, dt, n_sub, solver, now, tables, (ti-1)*per_step - row0 if per_step == 3 else -1)
            if len(reg.src):
                apply_events(rng, y, reg, ti, whole)
                if solver == 4 and p.local:
                    touch(reg, ti, p.places, w)
            if solver == 4 and p.local and p.has_det:
                settle_places(y, p, dt, now)
            for s in range(stocks.shape[2]):
                for g in range(stocks.shape[3]):
                    stocks[r, ti, s, g] = 0
            for f in range(nflows):
                for g in range(flows.shape[3]):
                    flows[r, ti, f, g] = 0
            for c in range(nc):
                for g in range(ng):
                    stocks[r, ti, state_of[c], g if by_group else 0] += y[c, g]
            for f in range(nflows):
                for g in range(ng):
                    flows[r, ti, f, g if by_group else 0] += now[f, g]
    ys[first:last] = own
    return


@nb.jit(nopython=True, cache=True, error_model='numpy', parallel=True)
def advance_threads(rngs, progs, ys, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step=1, row0=0):
    """ `advance()` for every run, with the runs shared between threads: one program for each thread (progs, a typed list of copies), since a step changes the parts of it that vary in time """
    k = len(progs)
    bounds = np.arange(k + 1)*len(ys)//k # The runs of each thread
    threads = np.arange(k) # Indices as signed integers, which the typed list takes, since a prange index is unsigned
    for i in nb.prange(k):
        c = threads[i]
        advance(rngs, progs[c], ys, bounds[c], bounds[c+1], dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, per_step, row0)
    return


@jit
def one_step(rng, y, p, dt, solver, n_sub, tables, row):
    """ Take one step, and return the compartments after it and the total of each transition and mover over it: see `run()` and `step()` """
    y = y.copy()
    flows = np.empty((p.n_trans + p.n_movers, y.shape[1]))
    step(rng, y, p, workspace(y, p), dt, n_sub, solver, flows, tables, row)
    return y, flows


#%% For agents

@jit
def levels(raw):
    """ The level of each agent in a dimension, as whole numbers (0 for an agent with none) """
    out = np.zeros(len(raw), dtype=np.int32) # 32-bit, as the groups are kept
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


#%% For agents: passes over the whole population, in chunks taken by several threads when it is large
# Each pass has a serial and a threaded driver around the same work on one chunk, so a sim that is one of many
# (in ss.parallel() or a calibration's workers, which run with one thread) never starts Numba's threads. Counts are
# whole numbers and lists of agents are kept in order, so the results do not depend on the number of threads.

pjit = nb.jit(nopython=True, cache=True, error_model='numpy', parallel=True)
min_threaded = 200_000 # The smallest population whose passes are threaded: below it, starting the threads costs more than they save
chunk_size = 65_536 # The fewest agents a thread takes at a time

def n_chunks(n, width=1):
    """ The number of chunks to split a pass over `n` agents into (1, unless there are threads to share it: see `ss.options.threads`), keeping the per-chunk counts (`width` numbers each) small """
    if n < min_threaded:
        return 1
    threads = ss.options.threads
    if threads == 1:
        return 1
    if threads > 1 and threads != nb.get_num_threads():
        nb.set_num_threads(min(threads, nb.config.NUMBA_NUM_THREADS))
    threads = nb.get_num_threads()
    if threads < 2:
        return 1
    return int(max(1, min(2*threads, n//chunk_size, 4_000_000//max(width, 1))))


def one_thread(func, *args, **kwargs):
    """ Run a function with one thread for the kernels (the passes over the agents, see `ss.options.threads`, and the places of a Gillespie run): for the workers of `ss.parallel()` and of a calibration, which share the cores already """
    ss.options.threads = 1
    nb.set_num_threads(1)
    return func(*args, **kwargs)


blas_threads = [('openblas_set_num_threads', 'openblas_get_num_threads'), ('openblas_set_num_threads64_', 'openblas_get_num_threads64_'), ('scipy_openblas_set_num_threads', 'scipy_openblas_get_num_threads'),
                ('scipy_openblas_set_num_threads64_', 'scipy_openblas_get_num_threads64_'), ('MKL_Set_Num_Threads', 'MKL_Get_Max_Threads')] # How each BLAS library sets and gives its number of threads
blas_env = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']

@contextlib.contextmanager
def one_blas_thread():
    """
    One thread for NumPy's BLAS (OpenBLAS or MKL) while the workers of `ss.parallel()` or of a calibration are started, as
    threadpoolctl's `threadpool_limits(1)` would: in the kernels, a dense mixing matrix of more than 16 levels is a matrix
    product, which would otherwise take every core in each worker. It is set here, before the workers are forked (who
    keep it), since setting it in a forked worker starts OpenBLAS's threads again there, which made each run twice as
    long; workers started afresh take it from the environment. The libraries are found among those loaded, on Linux;
    elsewhere (e.g. Accelerate on a Mac) only the environment is set. Restored afterwards.
    """
    libs = []
    if os.path.exists('/proc/self/maps'):
        with open('/proc/self/maps') as f:
            paths = {line.split()[-1] for line in f if 'blas' in line.lower() or 'libmkl_rt' in line.lower()}
        for path in sorted(paths):
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            libs += [(getattr(lib, setter), getattr(lib, getter)) for setter,getter in blas_threads if hasattr(lib, setter) and hasattr(lib, getter)]
    env = {key:os.environ.get(key) for key in blas_env}
    before = [getter() for setter,getter in libs]
    try:
        os.environ.update({key:'1' for key in blas_env})
        for setter,getter in libs:
            setter(1)
        yield
    finally:
        for (setter,getter),n in zip(libs, before):
            setter(n)
        for key,val in env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    return


def start_method():
    """
    How to start the workers of `ss.parallel()` and of a calibration: as the platform does (on Linux, by fork, the
    fastest), unless this process has started threads that a fork can deadlock on (Numba's, for the passes over a
    large population, or JAX's); then by forkserver, whose workers start clean (W17)
    """
    if mp.get_all_start_methods()[0] != 'fork': # The first is the platform's default
        return None
    threads = 'jax' in sys.modules
    try:
        nb.threading_layer() # Only once Numba has started its threads
        threads = True
    except ValueError:
        pass
    return 'forkserver' if threads else None


def compile_drivers(make):
    """ Compile the serial and the threaded version of a pass (`make(threaded)` gives the function), each cached under a name of its own """
    out = []
    for threaded,decorator in [(False, jit), (True, pjit)]:
        func = make(threaded)
        func.__qualname__ = f'{func.__name__}_{"threaded" if threaded else "serial"}' # Otherwise both would be cached as the same function, and either could be loaded for the other
        out.append(decorator(func))
    return out


@jit
def bounds(c, n, nchunks):
    """ The agents that chunk c of n chunks covers """
    size = (n + nchunks - 1)//nchunks
    a = min(c*size, n)
    return a, min(a + size, n)


@jit
def census_chunk(members, alive, groups, a, b, counts):
    """ For agents a to b (by UID): the number in each state by group, and (in the last row) the number alive """
    ns = len(members)
    whole = len(groups) == 0
    for uid in range(a, b):
        if not alive[uid]: # Nobody who has died is in a state: see ss.solvers.Agents.die()
            continue
        g = 0 if whole else groups[uid]
        r = uid & 3 # Four running counts, taken in turn: adding to one each time would wait on the last addition when neighbors are in the same group
        counts[r, ns, g] += 1
        k = 0
        last = 0
        for s in range(ns):
            if members[s][uid]:
                k += 1
                last = s
        if k == 1: # The usual case: in one state
            counts[r, last, g] += 1
        elif k > 1:
            for s in range(ns):
                if members[s][uid]:
                    counts[r, s, g] += 1
    return


@jit
def census_fill(members, want, a, b, out, pos):
    """ List the agents a to b who are in each state whose agents are listed, from the positions given """
    ns = len(members)
    for s in range(ns):
        k = want[s]
        if k >= 0:
            p = list_true(members[s], a, b, out, pos[k])
            pos[k] = p
    return


@jit
def list_true(raw, a, b, out, p):
    """ Write the indices of the true values of a boolean array from a to b into out, from position p; eight at a time where they are all false """
    i = a
    while i < b and i % 8:
        if raw[i]:
            out[p] = i
            p += 1
        i += 1
    end = i + (b - i)//8*8
    if end > i:
        words = raw[i:end].view(np.uint64)
        for w in range(len(words)):
            if words[w]:
                for j in range(i + 8*w, i + 8*w + 8):
                    if raw[j]:
                        out[p] = j
                        p += 1
    for j in range(end if end > i else i, b):
        if raw[j]:
            out[p] = j
            p += 1
    return p


def census_driver(threaded):
    """ Make the serial or threaded census (see census()) """
    loop = nb.prange if threaded else range
    def census(members, alive, groups, n_groups, want, n_want, nchunks):
        ns = len(members)
        n = len(alive)
        counts = np.zeros((nchunks, 4, ns + 1, n_groups), dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            census_chunk(members, alive, groups, a, b, counts[c])
        found = np.zeros((nchunks, max(n_want, 1)), dtype=np.int64) # The number in each state that is listed, in each chunk
        for s in range(ns):
            if want[s] >= 0:
                for c in range(nchunks):
                    found[c, want[s]] = counts[c, :, s, :].sum()
        starts = np.zeros((nchunks, max(n_want, 1)), dtype=np.int64) # Where each chunk's agents in each state go in the output
        offsets = np.zeros(n_want + 1, dtype=np.int64)
        pos = 0
        for k in range(n_want):
            offsets[k] = pos
            for c in range(nchunks):
                starts[c, k] = pos
                pos += found[c, k]
        offsets[n_want] = pos
        out = np.empty(pos, dtype=np.int64)
        if pos:
            for c in loop(nchunks):
                a, b = bounds(c, n, nchunks)
                census_fill(members, want, a, b, out, starts[c])
        total = np.zeros((ns + 1, n_groups), dtype=np.int64)
        for c in range(nchunks):
            for r in range(4):
                total += counts[c, r]
        return total, out, offsets
    return census

census_serial, census_threaded = compile_drivers(census_driver)

def census(members, alive, groups, n_groups, want):
    """
    Count the agents in each of several states by group, and list those in some of them: one pass over the agents for all the states

    Args:
        members (tuple): for each state, whether each agent (by UID) is in it
        alive (array): whether each agent is alive (by UID), which also gives the number of agents
        groups (array): the group of each agent (by UID), or an empty array if there is one group
        n_groups (int): the number of groups
        want (array): for each state, its position among the states whose agents are listed, or -1

    Returns the counts (states plus one, by groups: the last row is everyone alive), the agents in the
    states listed (in order, one state after another), and where each state's agents start and end in them.
    """
    n_want = int((want >= 0).sum())
    nchunks = n_chunks(len(alive), 4*(len(members) + 1)*n_groups)
    func = census_threaded if nchunks > 1 else census_serial
    return func(members, alive, groups, n_groups, want, n_want, nchunks)


@jit
def true_count(raw, a, b):
    """ The number of true values of a boolean array from a to b, eight at a time """
    m = 0
    i = a
    while i < b and i % 8:
        m += raw[i]
        i += 1
    end = i + (b - i)//8*8
    if end > i:
        words = raw[i:end].view(np.uint64)
        for w in range(len(words)):
            v = words[w]
            if v: # Each byte is 0 or 1, so their sum is the top byte of this product
                m += (v*np.uint64(0x0101010101010101)) >> np.uint64(56)
    for j in range(end if end > i else i, b):
        m += raw[j]
    return m


@jit
def true_fill(raw, a, b, out, pos):
    """ Write the indices of the true values from a to b, from a position """
    list_true(raw, a, b, out, pos)
    return


def flatnonzero_driver(threaded):
    loop = nb.prange if threaded else range
    def flatnonzero(raw, nchunks):
        n = len(raw)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = true_count(raw, a, b)
        starts = np.cumsum(found)
        out = np.empty(starts[-1], dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            true_fill(raw, a, b, out, starts[c])
        return out
    return flatnonzero

flatnonzero_serial, flatnonzero_threaded = compile_drivers(flatnonzero_driver)

def flatnonzero(raw):
    """ The indices of the true values of an array, as `np.flatnonzero()`, threaded for a large one """
    nchunks = n_chunks(len(raw))
    if nchunks == 1:
        return np.flatnonzero(raw)
    return flatnonzero_threaded(raw, nchunks)


@jit
def crn_key(seed, ind):
    """ The key that one call's uniforms are hashed with, as in `distributions._hash_uniforms_fill()` """
    x = (np.uint64(seed) ^ (np.uint64(ind)*_GOLDEN)) + _GOLDEN
    x = (x ^ (x >> _U30))*_MIX1
    x = (x ^ (x >> _U27))*_MIX2
    return x ^ (x >> _U31)


@jit
def crn_uniform(key, slot, shift, scale):
    """ An agent's uniform for one call, from its slot: the same number as `hash_uniforms(seed, ind, slots)` gives it (exact as a float64) """
    z = (np.uint64(slot) + key)*_GOLDEN
    z = (z ^ (z >> _U30))*_MIX1
    z = (z ^ (z >> _U27))*_MIX2
    z = z ^ (z >> _U31)
    return max(z >> shift, 0.5)*scale


def crn_bits(dtype):
    """ How many bits of each hash a uniform of a dtype keeps, as the shift and scale that `crn_uniform()` takes """
    return (_U40, _INV24) if np.dtype(dtype).itemsize <= 4 else (_U11, _INV53)


@jit
def infect_chunk(susceptible, groups, usual, rate, rel_sus, state_sus, adjust, mult, dt, slots, key, shift, scale, a, b, hit):
    """ For agents a to b: whether each is infected (see infect()), and how many are """
    sus = susceptible[a:b] # Slices, so the loop's indices are known not to be negative
    gs = groups[a:b]
    rs = rel_sus[a:b]
    sl = slots[a:b]
    by_uid = len(slots) == 0 # The slots are the UIDs
    hs = hit[a:b]
    m = 0
    for i in range(len(sus)):
        got = False
        if sus[i]:
            g = gs[i]
            if rs[i] == 1:
                prob = usual[g]
            else:
                prob = -np.expm1(-max(rate[g]*rs[i]*state_sus*adjust[g]*mult, 0.0)*dt) # As in infection_probs()
            if prob > 0:
                got = crn_uniform(key, a + i if by_uid else sl[i], shift, scale) < prob
        hs[i] = got
        m += got
    return m


def infect_driver(threaded):
    loop = nb.prange if threaded else range
    def infect(susceptible, groups, rate, rel_sus, state_sus, adjust, mult, dt, slots, seed, ind, shift, scale, nchunks):
        n = len(susceptible)
        usual = np.empty(len(rate)) # The probability for someone of average susceptibility, which is most people
        for g in range(len(rate)):
            usual[g] = -np.expm1(-max(rate[g]*1.0*state_sus*adjust[g]*mult, 0.0)*dt)
        key = crn_key(seed, ind)
        hit = np.empty(n, dtype=np.bool_)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = infect_chunk(susceptible, groups, usual, rate, rel_sus, state_sus, adjust, mult, dt, slots, key, shift, scale, a, b, hit)
        starts = np.cumsum(found)
        out = np.empty(starts[-1], dtype=np.int64)
        if starts[-1]:
            for c in loop(nchunks):
                a, b = bounds(c, n, nchunks)
                if found[c+1]:
                    list_true(hit, a, b, out, starts[c])
        return out
    return infect

infect_serial, infect_threaded = compile_drivers(infect_driver)

def infect(susceptible, groups, rate, rel_sus, state_sus, adjust, mult, dt, slots, seed, ind, dtype):
    """
    The agents infected this step under homogeneous mixing, when the rate is the same for everyone in a group: one pass over the agents, which works out each
    one's chance (as `infection_probs()`) and draws whether they are infected from their uniform for the call (as `hash_uniforms()` gives it: see `ss.random.crn_key()`)

    Args:
        susceptible (array): whether each agent is in the state, by UID
        groups (array): the group of each agent, by UID (or an empty array if there is one group)
        rate, rel_sus, state_sus, adjust, mult, dt: as for `infection_probs()`
        slots (array): the slot of each agent, by UID, or None if they are the UIDs (see `ss.People.slot_raw()`)
        seed, ind (int): the seed and index of the distribution's call
        dtype (dtype): the dtype of its uniforms

    Returns the UIDs of the agents infected, in order.
    """
    shift, scale = crn_bits(dtype)
    nchunks = n_chunks(len(susceptible))
    if slots is None:
        slots = np.empty(0, dtype=np.int32)
    func = infect_threaded if nchunks > 1 else infect_serial
    return func(susceptible, groups, rate, rel_sus, state_sus, adjust, mult, dt, slots, seed, ind, shift, scale, nchunks)


#%% For people: the passes over the active agents at the end of each step

@jit
def due_chunk(auids, ti_dead, ti_removed, ti, a, b, hit):
    """ Whether each of the active agents a to b has died or left by step ti, and how many have """
    m = 0
    for i in range(a, b):
        uid = auids[i]
        got = (ti_dead[uid] <= ti) | (ti_removed[uid] <= ti)
        hit[i] = got
        m += got
    return m


def due_driver(threaded):
    loop = nb.prange if threaded else range
    def due(auids, ti_dead, ti_removed, ti, nchunks):
        n = len(auids)
        hit = np.empty(n, dtype=np.bool_)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = due_chunk(auids, ti_dead, ti_removed, ti, a, b, hit)
        starts = np.cumsum(found)
        out = np.empty(starts[-1], dtype=np.int64)
        if starts[-1]:
            for c in loop(nchunks):
                a, b = bounds(c, n, nchunks)
                if found[c+1]:
                    p = starts[c]
                    for i in range(a, b):
                        if hit[i]:
                            out[p] = auids[i]
                            p += 1
        return out
    return due

due_serial, due_threaded = compile_drivers(due_driver)

def due(auids, ti_dead, ti_removed, ti):
    """ The active agents who have died or left by step ti (as `((ti_dead <= ti) | (ti_removed <= ti)).uids`), in one pass """
    nchunks = n_chunks(len(auids))
    func = due_threaded if nchunks > 1 else due_serial
    return func(auids, ti_dead, ti_removed, ti, nchunks)


@jit
def count_chunk(auids, states, alive, ti_dead, ti_removed, ti, a, b, out):
    """ For the active agents a to b: the number in each state, then the number who died and who left (who are no longer alive) """
    ns = len(states)
    for i in range(a, b):
        uid = auids[i]
        for s in range(ns):
            out[s] += states[s][uid] != 0
        if not alive[uid]:
            out[ns] += ti_dead[uid] <= ti
            out[ns+1] += ti_removed[uid] <= ti
    return


def count_driver(threaded):
    loop = nb.prange if threaded else range
    def count_people(auids, states, alive, ti_dead, ti_removed, ti, nchunks):
        n = len(auids)
        out = np.zeros((nchunks, len(states) + 2), dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            count_chunk(auids, states, alive, ti_dead, ti_removed, ti, a, b, out[c])
        return out.sum(axis=0)
    return count_people

count_people_serial, count_people_threaded = compile_drivers(count_driver)

def count_people(auids, states, alive, ti_dead, ti_removed, ti):
    """ The number of active agents in each of some states (as `state.count()`), then the number who died and who left by step ti (as `count_gone()`), in one pass """
    nchunks = n_chunks(len(auids))
    func = count_people_threaded if nchunks > 1 else count_people_serial
    return func(auids, states, alive, ti_dead, ti_removed, ti, nchunks)


@jit
def keep_chunk(auids, alive, a, b):
    """ The number of the active agents a to b who are still alive """
    m = 0
    for i in range(a, b):
        m += alive[auids[i]]
    return m


@jit
def keep_fill(auids, alive, age, dt, aging, a, b, kept, k, gone, g):
    """ List the active agents a to b who are still alive (from position k), and those who are not (from g); the age of those alive goes up by dt """
    for i in range(a, b):
        uid = auids[i]
        if alive[uid]:
            kept[k] = uid
            k += 1
            if aging:
                age[uid] += dt
        else:
            gone[g] = uid
            g += 1
    return


def keep_driver(threaded):
    loop = nb.prange if threaded else range
    def keep_alive(auids, alive, age, dt, aging, kept, nchunks):
        n = len(auids)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = keep_chunk(auids, alive, a, b)
        starts = np.cumsum(found)
        n_kept = starts[-1]
        gone = np.empty(n - n_kept, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            keep_fill(auids, alive, age, dt, aging, a, b, kept, starts[c], gone, a - starts[c])
        return n_kept, gone
    return keep_alive

keep_alive_serial, keep_alive_threaded = compile_drivers(keep_driver)

def keep_alive(auids, alive, age, dt, aging, kept):
    """
    Take the agents who are no longer alive out of the active agents, and age those who are: one pass

    Args:
        auids (array): the active agents
        alive (array): whether each agent is alive, by UID
        age (array): the age of each agent, by UID (changed in place)
        dt (float32): how much they age
        aging (bool): whether they age
        kept (array): where to write the agents still alive, in order (at least as long as `auids`)

    Returns the number still alive, and the agents who are not.
    """
    nchunks = n_chunks(len(auids))
    func = keep_alive_threaded if nchunks > 1 else keep_alive_serial
    return func(auids, alive, age, dt, aging, kept, nchunks)


# The same passes, over every agent by UID rather than over the list of active agents, which People uses when the
# active agents are those alive (and those who died in this step), as they are unless something else changes who is alive

@jit
def due_alive_chunk(alive, ti_dead, ti_removed, ti, a, b, hit):
    """ For agents a to b (by UID): whether each is alive and has died or left by step ti; returns how many have, and how many are alive """
    al = alive[a:b].view(np.uint8) # Slices, so the loop's indices are known not to be negative, and it is vectorized
    td = ti_dead[a:b]
    tr = ti_removed[a:b]
    hs = hit[a:b].view(np.uint8)
    m = 0
    k = 0
    for i in range(len(al)):
        y = al[i]
        x = (np.uint8(td[i] <= ti) | np.uint8(tr[i] <= ti)) & y
        hs[i] = x
        m += x
        k += y
    return m, k


def due_alive_driver(threaded):
    loop = nb.prange if threaded else range
    def due_alive(alive, ti_dead, ti_removed, ti, nchunks):
        n = len(alive)
        hit = np.empty(n, dtype=np.bool_)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        n_alive = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            m, k = due_alive_chunk(alive, ti_dead, ti_removed, ti, a, b, hit)
            found[c+1] = m
            n_alive[c] = k
        starts = np.cumsum(found)
        out = np.empty(starts[-1], dtype=np.int64)
        if starts[-1]:
            for c in loop(nchunks):
                if found[c+1]:
                    a, b = bounds(c, n, nchunks)
                    list_true(hit, a, b, out, starts[c])
        return out, n_alive.sum()
    return due_alive

due_alive_serial, due_alive_threaded = compile_drivers(due_alive_driver)

def due_alive(alive, ti_dead, ti_removed, ti):
    """ The agents alive who have died or left by step ti (as `due()`, but by UID), and the number alive """
    nchunks = n_chunks(len(alive))
    func = due_alive_threaded if nchunks > 1 else due_alive_serial
    return func(alive, ti_dead, ti_removed, ti, nchunks)


@jit
def count_alive_chunk(states, alive, a, b, out):
    """ For agents a to b (by UID): the number of those alive in each state, then the number alive """
    ns = len(states)
    for s in range(ns):
        out[s] += both_count(states[s], alive, a, b)
    out[ns] += true_count(alive, a, b)
    return


@jit
def both_count(x, y, a, b):
    """ The number of agents from a to b for whom two boolean arrays are both true, eight at a time """
    m = 0
    i = a
    while i < b and i % 8:
        m += x[i] & y[i]
        i += 1
    end = i + (b - i)//8*8
    if end > i:
        xw = x[i:end].view(np.uint64)
        yw = y[i:end].view(np.uint64)
        for w in range(len(xw)):
            v = xw[w] & yw[w]
            m += (v*np.uint64(0x0101010101010101)) >> np.uint64(56) # Each byte is 0 or 1: see true_count()
    for j in range(end if end > i else i, b):
        m += x[j] & y[j]
    return m


def count_alive_driver(threaded):
    loop = nb.prange if threaded else range
    def count_alive(states, alive, nchunks):
        n = len(alive)
        out = np.zeros((nchunks, len(states) + 1), dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            count_alive_chunk(states, alive, a, b, out[c])
        return out.sum(axis=0)
    return count_alive

count_alive_serial, count_alive_threaded = compile_drivers(count_alive_driver)

def count_alive(states, alive):
    """ The number of agents alive in each of some states, then the number alive, in one pass by UID """
    nchunks = n_chunks(len(alive))
    func = count_alive_threaded if nchunks > 1 else count_alive_serial
    return func(states, alive, nchunks)


@jit
def keep_alive_fill(alive, age, dt, aging, a, b, kept, k):
    """ List the agents a to b (by UID) who are alive, from position k, and age them by dt """
    al = alive[a:b]
    if aging:
        ages = age[a:b]
        for i in range(len(al)):
            if al[i]:
                kept[k] = a + i
                k += 1
                ages[i] += dt
    else:
        for i in range(len(al)):
            if al[i]:
                kept[k] = a + i
                k += 1
    return


def keep_raw_driver(threaded):
    loop = nb.prange if threaded else range
    def keep_raw(alive, age, dt, aging, kept, expected, nchunks):
        n = len(alive)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = true_count(alive, a, b)
        starts = np.cumsum(found)
        if starts[-1] != expected: # Not the agents expected: nothing is changed
            return -1
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            keep_alive_fill(alive, age, dt, aging, a, b, kept, starts[c])
        return starts[-1]
    return keep_raw

keep_raw_serial, keep_raw_threaded = compile_drivers(keep_raw_driver)

def keep_raw(alive, age, dt, aging, kept, expected):
    """
    As `keep_alive()`, but by UID: list the agents alive (in order), and age them, in one pass, if there are as many as expected

    Returns the number alive, or -1 (having changed nothing) if that is not the number expected.
    """
    nchunks = n_chunks(len(alive))
    func = keep_raw_threaded if nchunks > 1 else keep_raw_serial
    return func(alive, age, dt, aging, kept, expected, nchunks)
