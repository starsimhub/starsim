"""
The chain-binomial and tau-leaping solvers in JAX: a whole run as one compiled function, with all of a step's draws in one native call

The rates come from `ss.gradients.Run.rates()` (the program's rates by group, in JAX, as
for the ODE); the draws are the exact one-uniform draws of `_draws` (Numba), made for a
whole step at once by `step()`, which JAX calls through an XLA FFI custom call: a Numba
`cfunc` that reads XLA's call frame by byte offset (no C++ build). `step()` follows
`ss.kernels.draw()` draw for draw: for each compartment, the number who leave it, how they
split among its flows and their targets, the stages of a dwell time they pass, Poisson
arrivals, and movement between groups; and around it, as `ss.kernels.run()` does, the
deliveries at the start of a step (`deliver()`), the conveyors, and the events of a movement
register at its end (`apply_events()`). Substeps are calls of their own, as are the leaps of
tau-leaping, whose length JAX works out from the rates. Its uniforms are counter-based (a
hash of the run's seed, the step, and the draw), so runs taken together with `jax.vmap` (over
seeds or parameter sets) give the same draws as runs taken one at a time.

`reason()` says what a model has that this does not cover (then the kernels run it).
Everything here is CPU only, and is imported only when used (it imports JAX; Numba caches the
handler, so it compiles once per install, in about 20 s, with a note, or ahead of time with
`ss.compile()`; under `ss.options.compile = 'never'`, the kernels take the run instead).
"""
import os
import re
import ctypes
import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic
from numba.core.caching import FunctionCache
from numba.core.ccallback import CFunc
import starsim as ss
from . import _draws as draws, _cache

_cache.depends(__file__, draws.__file__) # The handler compiles in the draws

__all__ = []

CONVEY = ss.kernels.CONVEY
vec_min = 128 # The groups from which a row of draws is made by the vectorized draws (which pay their fixed cost a row only with many lanes)
fields = ['src_starts', 'src_flows', 'src', 'src_chain', 'src_stage', 'src_last', 'tgt_starts', 'tgt_comp', 'tgt_rec', 'named', 'index', 'mov_comps', 'tgt_map', 'record', 'crd_starts', 'crd_parent', 'crd_index', 'det', 'starts', 'ops', 'args'] # The parts of the program step() needs, packed into one array (see ChainRun.layout())
n_fields = len(fields)
n_head = 6 # Before the offsets: compartments, groups, transitions and movers, movers, vectorized, and the most stages of a dwell time
reg_fields = ['starts', 'src', 'dst', 'sel', 'sel_starts', 'sel_comps', 'sel_to', 'ent_starts', 'ent_comps', 'pg_starts', 'pg_groups'] # The parts of a movement register, packed the same way (see pack_register())
TAU, RELEASE, RELEASE_END, SHIFT, REGISTER, DELIVER = 1, 2, 4, 8, 16, 32 # What a call does: tau-leaping, release the conveyors' last slots before the draws or after them, move the conveyors along, apply the register's events, or only the deliveries
leap_bits = 20 # The step's number, shifted by this many bits, plus the substep or leap, counts the calls (for the key of each one's uniforms)
pieces = ss.kernels.pieces


def reason(sim):
    """ Why the sim's chain-binomial (or tau-leaping) run cannot be taken in JAX, in a few words, or None if it can """
    if sim.solver not in ['binomial', 'tau']:
        return f'solver="{sim.solver}" is neither the chain binomial nor tau-leaping'
    elif not getattr(sim, 'declared', None):
        return 'the model is written as code, with no declared states and transitions'
    elif not sim.kernel_only():
        return sim.kernel_reason()
    elif any(isinstance(mod, ss.Delivery) and mod.move_states is not None for mod in sim.interventions()):
        return f'"{next(mod.name for mod in sim.interventions() if isinstance(mod, ss.Delivery) and mod.move_states is not None)}" moves people between states, which only the kernels draw'
    elif ss.options.compile == 'never':
        return 'its draws are compiled by Numba, and ss.options.compile is "never"'
    for mod in sim.declared:
        name = mod.engine.certain() if hasattr(mod.engine, 'certain') else None
        if name is not None:
            return f'"{mod.name}" has a probability of 1 over a period ({name}), an infinite rate, which the kernels take as everyone moving at once'
    try:
        check()
    except RuntimeError as e:
        return str(e)
    return None


#%% The draws of a step, in Numba

@nb.jit(nopython=True, cache=True, error_model='numpy')
def field(S, i):
    """ Part i of the program, from the packed array """
    return S[S[n_head+i]:S[n_head+i+1]]


@nb.jit(nopython=True, cache=True, error_model='numpy')
def binomials(k, row, n, p, out, w, u, vec):
    """ A row of binomial draws (one per group), from uniforms row*groups, row*groups+1, ... of the step's stream """
    if vec:
        draws.uniforms(k, row*len(n), u)
        draws.binomials(n, p, u, out, w)
    else:
        for g in range(len(n)):
            out[g] = draws.binomial(n[g], p[g], draws.uniform(k, row*len(n) + g))
    return row + 1


@nb.jit(nopython=True, cache=True, error_model='numpy')
def land(k, row, n, m, maps, out):
    """ Add arrivals (n, by group) to the groups they land in, as `ss.kernels.land()` does: shared between groups by map m, if m >= 0, by a chain of binomials for each group """
    ng = len(n)
    if m < 0:
        out += n
        return row
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
                x = draws.binomial(left, w/rest, draws.uniform(k, (row + g)*ng + h))
                out[h] += x
                left -= x
                rest -= w
    return row + ng


@nb.jit(nopython=True, cache=True, error_model='numpy')
def split(k, row, n, f, S, tgt_w, maps, new, flows, scratch, w, u, vec, skip_last=False):
    """ Share n (by group) between the targets of flow f (but its last, with skip_last), as `ss.kernels.split()` does: one multinomial draw, as a chain of conditional binomials """
    tgt_starts, tgt_comp, tgt_rec, tgt_map = field(S, 6), field(S, 7), field(S, 8), field(S, 12)
    a, b = tgt_starts[f], tgt_starts[f+1] - skip_last
    ng = len(n)
    if b - a == 1 and tgt_w[a].min() >= 1 - 1e-12:
        row = land(k, row, n, tgt_map[a], maps, new[tgt_comp[a]])
        if tgt_rec[a] >= 0:
            flows[tgt_rec[a]] += n
        return row
    total, left, drawn, p = scratch[0], scratch[1], scratch[2], scratch[3]
    whole = True
    for g in range(ng):
        total[g] = 0.0
        left[g] = n[g]
        for t in range(a, b):
            total[g] += tgt_w[t, g]
        whole = whole and abs(total[g] - 1) < 1e-9
        if 0 < total[g] < 1 - 1e-9: # Shares that sum below 1 leave the rest nowhere
            total[g] = 1.0
    for t in range(a, b):
        for g in range(ng):
            p[g] = tgt_w[t, g]/total[g] if total[g] > 0 else 0.0
        row = binomials(k, row, left, p, drawn, w, u, vec)
        for g in range(ng):
            left[g] -= drawn[g]
            total[g] -= tgt_w[t, g]
        row = land(k, row, drawn, tgt_map[t], maps, new[tgt_comp[t]])
        if tgt_rec[t] >= 0:
            flows[tgt_rec[t]] += drawn
    if whole and b > a and left.max() > 0: # Rounding only
        row = land(k, row, left, tgt_map[b-1], maps, new[tgt_comp[b-1]])
        if tgt_rec[b-1] >= 0:
            flows[tgt_rec[b-1]] += left
    return row


@nb.jit(nopython=True, cache=True, error_model='numpy')
def credit(k, row, n, f, S, crd_w, flows, w, u, vec):
    """ Count the events of flow f (n, by group) also in the transitions they pass through with no wait, as `ss.kernels.credit()` does: the credits nested in the same one are one multinomial draw, as a chain of binomials """
    crd_starts, crd_parent, crd_index = field(S, 14), field(S, 15), field(S, 16)
    a, b = crd_starts[f], crd_starts[f+1]
    if b == a:
        return row
    ng = len(n)
    left = np.empty((b - a + 1, ng))
    mass = np.empty((b - a + 1, ng))
    c = np.empty(ng)
    p = np.empty(ng)
    left[0] = n
    mass[0] = 1.0
    for j in range(a, b):
        q = crd_parent[j]
        i = j - a + 1
        if q == -2: # Overlaps the others: drawn on its own
            row = binomials(k, row, n, crd_w[j], c, w, u, vec)
        else:
            r = 0 if q < 0 else q - a + 1
            for g in range(ng):
                p[g] = crd_w[j, g]/mass[r, g] if mass[r, g] > 0 else 0.0
            row = binomials(k, row, left[r], p, c, w, u, vec)
            for g in range(ng):
                left[r, g] -= c[g]
                mass[r, g] -= crd_w[j, g]
        left[i] = c
        mass[i] = crd_w[j]
        flows[crd_index[j]] += c
    return row


@nb.jit(nopython=True, cache=True, error_model='numpy')
def settle(y, H, dt, S, tgt_w, crd_w, maps, flows):
    """ The flows of states that are not people (a reservoir), as expected values over the step, integrated by RK4 at the step's rates with the people as they are, as `ss.kernels.settle()` does """
    src_of, tgt_starts, tgt_comp, tgt_rec, named, index = field(S, 2), field(S, 6), field(S, 7), field(S, 8), field(S, 9), field(S, 10)
    tgt_map, record, crd_starts, crd_index, det = field(S, 12), field(S, 13), field(S, 14), field(S, 16), field(S, 17)
    nc, ng = y.shape
    nflows = flows.shape[0]
    fastest = 0.0
    for f in range(len(src_of)):
        if det[f] and src_of[f] >= 0:
            for g in range(ng):
                fastest = max(fastest, H[f, g]/dt)
    n_sub = max(1, int(np.ceil(fastest*dt/0.25)))
    h = dt/n_sub
    k = np.zeros((4, nc, ng))
    kf = np.zeros((4, nflows, ng))
    for sub in range(n_sub):
        k[:] = 0.0
        kf[:] = 0.0
        y0 = y.copy()
        yt = y.copy()
        for stage in range(4):
            for f in range(len(src_of)):
                if not det[f]:
                    continue
                s = src_of[f]
                for g in range(ng):
                    amount = H[f, g]/dt*(yt[s, g] if s >= 0 else 1.0) # H is the rate times dt, or for an inflow its arrivals
                    if amount == 0:
                        continue
                    if s >= 0:
                        k[stage, s, g] -= amount
                    if named[f]:
                        if record[f] < 0:
                            kf[stage, index[f], g] += amount
                        else:
                            for j in range(ng):
                                kf[stage, index[f], j] += amount*maps[record[f], g, j]
                    for t in range(tgt_starts[f], tgt_starts[f+1]):
                        if tgt_rec[t] >= 0:
                            kf[stage, tgt_rec[t], g] += amount*tgt_w[t, g]
                        if tgt_map[t] < 0:
                            k[stage, tgt_comp[t], g] += amount*tgt_w[t, g]
                        else:
                            for j in range(ng):
                                k[stage, tgt_comp[t], j] += amount*tgt_w[t, g]*maps[tgt_map[t], g, j]
                    for t in range(crd_starts[f], crd_starts[f+1]):
                        kf[stage, crd_index[t], g] += amount*crd_w[t, g]
            if stage < 3:
                ahead = h if stage == 2 else h/2
                for c in range(nc):
                    for g in range(ng):
                        yt[c, g] = max(y0[c, g] + ahead*k[stage, c, g], 0.0)
        for c in range(nc):
            for g in range(ng):
                y[c, g] = max(y0[c, g] + h/6*(k[0, c, g] + 2*k[1, c, g] + 2*k[2, c, g] + k[3, c, g]), 0.0)
        for f in range(nflows):
            for g in range(ng):
                flows[f, g] += h/6*(kf[0, f, g] + 2*kf[1, f, g] + 2*kf[2, f, g] + kf[3, f, g])
    return


@nb.jit(nopython=True, cache=True, error_model='numpy')
def release(k, row, new, S, tgt_w, crd_w, maps, flows, scratch, w, u, vec):
    """ Release the people in the last slot of each conveyor, as `ss.kernels.convey()` does with release=True """
    src_of, named, index, record = field(S, 2), field(S, 9), field(S, 10), field(S, 13)
    starts, ops, args = field(S, 18), field(S, 19), field(S, 20)
    n = np.empty(new.shape[1])
    for i in range(len(starts) - len(src_of) - 1): # Conveyors, whose operations follow the flows'
        a, b = starts[len(src_of)+i], starts[len(src_of)+i+1]
        if ops[a] == CONVEY:
            f = args[a]
            n[:] = new[args[b-1]]
            new[args[b-1]] = 0.0
            if named[f]:
                row = land(k, row, n, record[f], maps, flows[index[f]])
            row = split(k, row, n, f, S, tgt_w, maps, new, flows, scratch, w, u, vec, True) # Not the last target, which is the slot they leave
            row = credit(k, row, n, f, S, crd_w, flows, w, u, vec)
    return row


@nb.jit(nopython=True, cache=True, error_model='numpy')
def deliver(k, y, due, sel, cov, dst, tally):
    """ The deliveries due at the start of a step, in order, as whole people, as `ss.kernels.apply_shifts()` does: each moves some of the people in each compartment and group to another group; tally gets the number each reached (expected) and found eligible, by group """
    nc, ng = y.shape
    before = np.empty(ng)
    i = 0
    for d in range(len(due)):
        if due[d] <= 0:
            continue
        for c in range(nc):
            before[:] = y[c] # Each group's people as they were, so nobody moved into a group is moved on again
            for g in range(ng):
                e = sel[d, c, g]
                n = before[g]
                if e <= 0 or n <= 0:
                    continue
                p = cov[d, c, g]
                tally[d, 0, int(dst[d, 1, g])] += n*e*p # In the group of those offered it (see ss.Delivery.mark_offered())
                tally[d, 1, g] += n*e
                a = draws.binomial(n, e*p, draws.uniform(k, i))
                b = draws.binomial(n - a, e*(1 - p)/(1 - e*p), draws.uniform(k, i + 1)) if e*p < 1 else 0.0
                i += 2
                to, rest = int(dst[d, 0, g]), int(dst[d, 1, g])
                if a > 0 and to != g:
                    y[c, g] -= a
                    y[c, to] += a
                if b > 0 and rest != g:
                    y[c, g] -= b
                    y[c, rest] += b
    return


@nb.jit(nopython=True, cache=True, error_model='numpy')
def hypergeometric(k, i, good, bad, n):
    """ The number of good items in n drawn without replacement, one at a time from uniforms i, i+1, ..., as `ss.kernels.hypergeometric()` draws them; returns it and the next uniform """
    good, bad, n = np.floor(good), np.floor(bad), np.floor(n)
    flip = n > (good + bad)//2 # Draw the ones left behind instead, if that is fewer
    if flip:
        n = good + bad - n
    got = 0.0
    for j in range(int(n)):
        if draws.uniform(k, i + j)*(good + bad) < good:
            good -= 1
            got += 1
        else:
            bad -= 1
    return (good if flip else got), i + int(n)


@nb.jit(nopython=True, cache=True, error_model='numpy')
def rfield(R, i):
    """ Part i of the register, from the packed array (see `pack_register()`) """
    return R[R[1+i]:R[2+i]]


@nb.jit(nopython=True, cache=True, error_model='numpy')
def apply_events(k, i, y, R, reg_n, ent_w, ti):
    """ The events of a movement register dated at time point ti, in order, as whole people, as `ss.kernels.apply_events()` applies them; returns the number of people they asked for and could not find """
    starts, src_of, dst_of, sel_of, sel_starts, sel_comps, sel_to = rfield(R, 0), rfield(R, 1), rfield(R, 2), rfield(R, 3), rfield(R, 4), rfield(R, 5), rfield(R, 6)
    ent_starts, ent_comps, pg_starts, pg_groups = rfield(R, 7), rfield(R, 8), rfield(R, 9), rfield(R, 10)
    step = R[0]
    short = 0.0
    if ti >= len(starts) - 1:
        return short
    for e in range(starts[ti], starts[ti+1]):
        src, dst, s = src_of[e], dst_of[e], sel_of[e]
        n = reg_n[e]
        if src >= 0: # Move people, or remove them
            total = 0.0
            for j in range(sel_starts[s], sel_starts[s+1]):
                for q in range(pg_starts[src], pg_starts[src+1]):
                    total += y[sel_comps[j], pg_groups[q]]
            if n > total:
                short += n - total
                n = total
            left = n
            rest = total
            for j in range(sel_starts[s], sel_starts[s+1]):
                c = sel_comps[j]
                for q in range(pg_starts[src], pg_starts[src+1]):
                    g = pg_groups[q]
                    have = y[c, g]
                    if left <= 0 or rest <= 0:
                        break
                    if have <= 0:
                        continue
                    take, i = hypergeometric(k, i, have, rest - have, left)
                    rest -= have
                    left -= take
                    y[c, g] -= take
                    if dst >= 0:
                        y[sel_to[j], g + (dst - src)*step] += take
        else: # Add people
            left = n
            for j in range(ent_starts[s], ent_starts[s+1]):
                part = draws.binomial(left, 1/(ent_starts[s+1] - j), draws.uniform(k, i)) # Shared between the states not yet given theirs
                i += 1
                left -= part
                wrest = 0.0
                for q in range(pg_starts[dst], pg_starts[dst+1]):
                    wrest += ent_w[j, pg_groups[q]]
                for q in range(pg_starts[dst], pg_starts[dst+1]):
                    g = pg_groups[q]
                    wt = ent_w[j, g]
                    if part <= 0 or wrest <= 0:
                        break
                    take = draws.binomial(part, wt/wrest, draws.uniform(k, i))
                    i += 1
                    y[ent_comps[j], g] += take
                    part -= take
                    wrest -= wt
    return short


@nb.jit(nopython=True, cache=True, error_model='numpy')
def step(k, y, H, S, tgt_w, crd_w, maps, mov_prob, dt, flags, sh_due, sh_sel, sh_cov, sh_dst, R, reg_n, ent_w, ti, new, flows, short, tally):
    """
    All of a step's draws (or a substep's, or a leap's): the compartments at its end (new) and the count of each transition and mover over it (flows), as `ss.kernels.draw()` gives them

    Args:
        k (int): the key of the call's stream of uniforms (see `_draws.key()`)
        y (array): the compartments at the start of the step, by group
        H (array): each flow's hazard over the step (its rate times dt) by group, or for an inflow its expected arrivals
        S (array): the parts of the program that say how the flows are put together, packed (see `ChainRun.layout()`)
        tgt_w (array): the share of a flow that goes to each of its targets, by group
        crd_w (array): the share of a flow's events that each of its credits counts, by group
        maps (array): the share of the people (or counts) of a group that land in each group, for each map
        mov_prob (array): for each entry of each mover, the chance of moving in the step
        dt (float): the step
        flags (int): what the call does, as a sum of TAU, RELEASE, ... (see the top of this file)
        sh_due, sh_sel, sh_cov, sh_dst (arrays): the deliveries (with DELIVER, the call applies these alone): whether each is due, and its share eligible, reached, and the groups they go to (see `deliver()`)
        R, reg_n, ent_w (arrays): the movement register (see `pack_register()`), whose events dated at time point ti are applied at the end, with REGISTER
        short (array): the people the register's events asked for and could not find
        tally (array): with DELIVER, the number each delivery reached and found eligible, by group
    """
    new[:] = y
    flows[:] = 0.0
    short[0] = 0.0
    tally[:] = 0.0
    if flags & DELIVER:
        deliver(k, new, sh_due, sh_sel, sh_cov, sh_dst, tally)
        return
    tau = (flags & TAU) > 0
    nc, ng, nflows, n_movers, vec = S[0], S[1], S[2], S[3], S[4] > 0
    src_starts, src_flows, src_of, src_chain, src_stage, src_last = field(S, 0), field(S, 1), field(S, 2), field(S, 3), field(S, 4), field(S, 5)
    named, index, mov_comps, record, det = field(S, 9), field(S, 10), field(S, 11), field(S, 13), field(S, 17)
    tgt_starts, tgt_comp, tgt_rec, tgt_map, crd_starts = field(S, 6), field(S, 7), field(S, 8), field(S, 12), field(S, 14)
    base = S[n_head+n_fields] # Then the movers' starts and pairs, as two rows
    width = (len(S) - base)//2
    buf = np.empty((23 + S[5], ng)) # Scratch, in one piece: every row by group
    w, scratch, P = buf[:9], buf[9:13], buf[21:]
    total, left, n, p, u, ends, rest, mass = buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19], buf[20]
    row = 0
    starts, ops, args = field(S, 18), field(S, 19), field(S, 20)
    n_conv = len(starts) - len(src_of) - 1 # Conveyors, whose operations follow the flows' (see ss.kernels.convey())
    if n_conv and flags & RELEASE: # Release the people in the last slot of each, at the start of the step, once the rates are worked out from the people as they were
        row = release(k, row, new, S, tgt_w, crd_w, maps, flows, scratch, w, u, vec)
        y = new.copy() # Who leaves is drawn from the people as they are now
    for s in range(len(src_starts) - 1):
        a, b = src_starts[s], src_starts[s+1]
        if a == b: # In the program of a crossing, the end of one disease's flows: the next disease's are drawn from where these left people (each from the rates at the start of the step), as ss.kernels.draw() does
            y = new.copy()
            continue
        src = src_of[src_flows[a]]
        d = 1 if tau else src_chain[s] # Tau-leaping takes the stages of a dwell time one at a time, as any other compartment
        if src >= 0 and det[src_flows[a]]: # A state that is not people: see settle()
            continue
        if d > 1: # The stages of a dwell time: as many as people pass in the step (see ss.kernels.draw_chain())
            stage = src_stage[s]
            stride = field(S, 7)[field(S, 6)[stage]] - src
            for g in range(ng):
                x = H[stage, g]
                hx = 0.0
                for j in range(a, b):
                    if src_flows[j] != stage:
                        hx += H[src_flows[j], g]
                total[g] = hx
                stay, term, term_t, cdf_t, tot = np.exp(-hx), np.exp(-x), np.exp(-x - hx), 0.0, 0.0
                for m in range(d):
                    P[m, g] = stay*term
                    tot += P[m, g]
                    cdf_t += term_t
                    term *= x/(m + 1)
                    term_t *= (x + hx)/(m + 1)
                P[d, g] = (x/(x + hx))**d*max(1 - cdf_t, 0.0) if x + hx > 0 else 0.0
                P[d+1, g] = max(1 - tot - P[d, g], 0.0)
            row = binomials(k, row, y[src], P[d+1], left, w, u, vec) # Leave by another exit
            for g in range(ng):
                rest[g] = y[src, g] - left[g]
                mass[g] = 1.0 - P[d+1, g]
                p[g] = P[d, g]/mass[g] if mass[g] > 0 else 1.0
            row = binomials(k, row, rest, p, ends, w, u, vec) # Leave by the last stage
            for g in range(ng):
                rest[g] -= ends[g]
                mass[g] -= P[d, g]
                new[src, g] -= left[g] + ends[g]
            for m in range(1, d): # Move on some stages; the rest stay
                for g in range(ng):
                    p[g] = P[m, g]/mass[g] if mass[g] > 0 else 0.0
                row = binomials(k, row, rest, p, n, w, u, vec)
                for g in range(ng):
                    rest[g] -= n[g]
                    mass[g] -= P[m, g]
                    new[src, g] -= n[g]
                    new[src + m*stride, g] += n[g]
            last = src_last[s]
            if named[last]:
                row = land(k, row, ends, record[last], maps, flows[index[last]])
            row = split(k, row, ends, last, S, tgt_w, maps, new, flows, scratch, w, u, vec)
            row = credit(k, row, ends, last, S, crd_w, flows, w, u, vec)
        elif src >= 0: # Who leaves
            for g in range(ng):
                total[g] = 0.0
                for j in range(a, b):
                    total[g] += H[src_flows[j], g]
                p[g] = total[g]*y[src, g] if tau else -np.expm1(-total[g])
            if tau: # Poisson, and no more than are there (clipped once, so the order of the flows does not matter)
                if vec:
                    draws.uniforms(k, row*ng, u)
                    draws.poissons(p, u, left, w)
                else:
                    for g in range(ng):
                        left[g] = draws.poisson(p[g], draws.uniform(k, row*ng + g))
                row += 1
                for g in range(ng):
                    left[g] = min(left[g], y[src, g])
            else:
                row = binomials(k, row, y[src], p, left, w, u, vec)
            new[src] -= left
        for j in range(a, b): # How they split between the flows (or the arrivals)
            f = src_flows[j]
            if (d > 1 and f == src_stage[s]) or det[f]: # Taken above, or settled below
                continue
            if src < 0:
                if vec:
                    draws.uniforms(k, row*ng, u)
                    draws.poissons(H[f], u, n, w)
                else:
                    for g in range(ng):
                        n[g] = draws.poisson(H[f, g], draws.uniform(k, row*ng + g))
                row += 1
            else:
                for g in range(ng):
                    p[g] = H[f, g]/total[g] if total[g] > 0 else 0.0
                row = binomials(k, row, left, p, n, w, u, vec)
                for g in range(ng):
                    total[g] -= H[f, g]
                    left[g] -= n[g]
            if named[f] and record[f] < 0: # Counted in the group it happens in, and (below) everyone to one compartment in the same group: as land() and split() do, but written out, since the calls cost more than the draws of a small model
                flows[index[f]] += n
            elif named[f]:
                row = land(k, row, n, record[f], maps, flows[index[f]])
            t = tgt_starts[f]
            if tgt_starts[f+1] == t + 1 and tgt_map[t] < 0 and tgt_rec[t] < 0 and tgt_w[t].min() >= 1 - 1e-12:
                new[tgt_comp[t]] += n
            else:
                row = split(k, row, n, f, S, tgt_w, maps, new, flows, scratch, w, u, vec)
            if crd_starts[f+1] > crd_starts[f]:
                row = credit(k, row, n, f, S, crd_w, flows, w, u, vec)
    i = row*ng # Movement between groups, from where people are after the transitions
    for m in range(n_movers):
        for e in range(S[base+width+m], S[base+width+m+1]):
            if mov_prob[e] > 0:
                g, h = S[base+n_movers+1+e], S[base+width+n_movers+1+e]
                for j in range(S[base+m], S[base+m+1]):
                    c = mov_comps[j]
                    moved = draws.binomial(new[c, g], mov_prob[e], draws.uniform(k, i))
                    i += 1
                    new[c, g] -= moved
                    new[c, h] += moved
                    flows[nflows - n_movers + m, g] += moved
    if det.any():
        settle(new, H, dt, S, tgt_w, crd_w, maps, flows)
    row = i//ng + 1
    if n_conv and flags & RELEASE_END: # Tau-leaping releases them at the middle of the step, after the leap that reaches it
        row = release(k, row, new, S, tgt_w, crd_w, maps, flows, scratch, w, u, vec)
    if n_conv and flags & SHIFT: # Everyone in a conveyor moves along one slot
        for i in range(n_conv):
            a, b = starts[len(src_of)+i], starts[len(src_of)+i+1]
            if ops[a] == CONVEY:
                for j in range(b-1, a+1, -1):
                    new[args[j]] = new[args[j-1]]
                new[args[a+1]] = 0.0
    if flags & REGISTER: # The events dated at the step's end
        short[0] = apply_events(k, row*ng, new, R, reg_n, ent_w, ti)
    return


#%% The XLA FFI handler: reads XLA's call frame (the structs of xla/ffi/api/c_api.h) by byte offset

def _pointer(nbtype):
    @intrinsic
    def to_ptr(typingctx, addr):
        sig = types.CPointer(nbtype)(types.int64)
        def codegen(context, builder, signature, args):
            return builder.inttoptr(args[0], context.get_value_type(signature.return_type))
        return sig, codegen
    return to_ptr

i64_ptr, i32_ptr, u32_ptr, f64_ptr = [_pointer(t) for t in [types.int64, types.int32, types.uint32, types.float64]]

in_ranks = (1, 2, 2, 1, 2, 2, 3, 1, 0, 1, 3, 3, 3, 1, 1, 2) # The rank of each argument of one run (see step()): ctl, y, H, S, tgt_w, crd_w, maps, mov_prob, dt, sh_due, sh_sel, sh_cov, sh_dst, R, reg_n, ent_w
out_ranks = (2, 2, 1, 3) # ... and of each output: new, flows, short, tally


@nb.jit(nopython=True, inline='always')
def load(addr):
    return nb.carray(i64_ptr(addr), 1)[0]


@nb.jit(nopython=True, inline='always')
def buffer(addr, rank0, runs):
    """
    XLA_FFI_Buffer: struct_size, extension_start, dtype, data (+24), rank (+32), dims (+40)

    Returns its data and, for an argument of rank rank0 in one run, how far apart the runs are in it (0 if it is
    the same for all of them: under vmap, an argument that is not batched has a batch of 1) and the size of one;
    or with runs=0, the number of runs (for an output, which always has them all).
    """
    rank = load(addr + 32)
    dims = nb.carray(i64_ptr(load(addr + 40)), rank)
    size = 1
    for d in dims:
        size *= d
    batch = dims[0] if rank > rank0 else 1
    if runs == 0:
        return load(addr + 24), batch, size
    one = size//batch if batch > 0 else 0
    return load(addr + 24), one if batch > 1 else 0, one


@nb.jit(nopython=True, cache=True)
def handle(frame, major, minor):
    """ XLA_FFI_Error* handler(XLA_FFI_CallFrame*), as a cached function of the frame's address (see `make_handler()`): the arguments of step() in, new, flows, short, and tally out. Runs batched by vmap have the runs first, in the outputs and in each argument that is batched """
    # XLA_FFI_CallFrame: struct_size, extension_start (+8), api, ctx, stage (+32), args (+40), rets (+80)
    ext = load(frame + 8)
    if ext != 0 and nb.carray(i32_ptr(ext + 8), 1)[0] == 1: # XLA_FFI_Extension_Metadata: report the API version
        meta = load(ext + 24)
        version = nb.carray(i64_ptr(meta + 8), 2) # XLA_FFI_Api_Version: struct_size, extension_start, major, minor
        version[0] = 24
        version[1] = 0
        ints = nb.carray(i32_ptr(meta + 24), 2)
        ints[0] = major
        ints[1] = minor
        nb.carray(u32_ptr(meta + 32), 1)[0] = 0 # Traits
        return 0
    if nb.carray(i32_ptr(frame + 32), 1)[0] != 3: # Only XLA_FFI_ExecutionStage_EXECUTE
        return 0
    args = load(frame + 40 + 32) # XLA_FFI_Args: struct_size, extension_start, size, types, args (+32)
    rets = load(frame + 80 + 32)
    n_in = len(in_ranks)
    data = np.empty(n_in, dtype=np.int64)
    gap = np.empty(n_in, dtype=np.int64) # How far apart the runs are in each argument, in elements
    one = np.empty(n_in, dtype=np.int64) # The size of one run's
    n_data, runs, _ = buffer(load(rets), out_ranks[0], 0)
    f_data, _, _ = buffer(load(rets + 8), out_ranks[1], 0)
    s_data, _, _ = buffer(load(rets + 16), out_ranks[2], 0)
    t_data, _, _ = buffer(load(rets + 24), out_ranks[3], 0)
    for j in range(n_in):
        data[j], gap[j], one[j] = buffer(load(args + 8*j), in_ranks[j], runs)
    for r in range(runs):
        at = data + 8*r*gap # Where each argument of this run is
        ctl = nb.carray(i64_ptr(at[0]), 4) # The seed, the call's count, its flags, and the time point
        S = nb.carray(i64_ptr(at[3]), one[3])
        nc, ng, nflows = S[0], S[1], S[2]
        K = one[9]
        y = nb.carray(f64_ptr(at[1]), (nc, ng))
        H = nb.carray(f64_ptr(at[2]), (one[2]//ng, ng))
        tgt_w = nb.carray(f64_ptr(at[4]), (one[4]//ng, ng))
        crd_w = nb.carray(f64_ptr(at[5]), (one[5]//ng, ng))
        maps = nb.carray(f64_ptr(at[6]), (one[6]//(ng*ng), ng, ng))
        mov_prob = nb.carray(f64_ptr(at[7]), one[7])
        dt = nb.carray(f64_ptr(at[8]), 1)[0]
        sh_due = nb.carray(f64_ptr(at[9]), K)
        sh_sel = nb.carray(f64_ptr(at[10]), (K, nc, ng))
        sh_cov = nb.carray(f64_ptr(at[11]), (K, nc, ng))
        sh_dst = nb.carray(f64_ptr(at[12]), (K, 2, ng))
        R = nb.carray(i64_ptr(at[13]), one[13])
        reg_n = nb.carray(f64_ptr(at[14]), one[14])
        ent_w = nb.carray(f64_ptr(at[15]), (one[15]//ng, ng))
        new = nb.carray(f64_ptr(n_data + 8*r*nc*ng), (nc, ng))
        flows = nb.carray(f64_ptr(f_data + 8*r*nflows*ng), (nflows, ng))
        short = nb.carray(f64_ptr(s_data + 8*r), 1)
        tally = nb.carray(f64_ptr(t_data + 8*r*K*2*ng), (K, 2, ng))
        step(draws.key(ctl[0], ctl[1]), y, H, S, tgt_w, crd_w, maps, mov_prob, dt, ctl[2], sh_due, sh_sel, sh_cov, sh_dst, R, reg_n, ent_w, ctl[3], new, flows, short, tally)
    return 0


def api_version():
    """ The version of XLA's FFI API that the installed JAX has (from its header, read once a session), which the handler reports """
    global version
    if version is None:
        header = open(os.path.join(ss.gradients.jax().ffi.include_dir(), 'xla/ffi/api/c_api.h')).read()
        version = tuple(int(re.search(rf'#define XLA_FFI_API_{part} (\d+)', header)[1]) for part in ['MAJOR', 'MINOR'])
    return version

version = None


def handler_func(major, minor):
    """ The function XLA calls (see `handle()`), reporting API version major.minor, as Python; Numba caches it by the version (the values a closure holds are part of the key of its cache), so a JAX with another version gets a handler of its own """
    def handler(frame):
        return handle(frame, major, minor)
    return handler


def make_handler(major, minor):
    """ The C function XLA calls: `handler_func()`, compiled """
    return nb.cfunc(types.int64(types.int64), cache=True)(handler_func(major, minor))


def is_compiled():
    """ Whether the handler for the installed JAX is compiled already: in this session, or in Numba's cache from an earlier one (as `ss.kernels.is_compiled()`) """
    v = api_version()
    if v in handlers:
        return True
    cache = FunctionCache(handler_func(*v))
    return cache._index_key(types.int64(types.int64), CFunc._targetdescr.target_context.codegen()) in cache._cache_file._load_index()


handlers = {} # The handler registered with XLA, and its name there, by the API version it reports
registered = {} # Every handler registered with XLA this session, by its name there


def register():
    """ Register the handler with XLA, once a session (or again, under another name, for another FFI API version): returns its name """
    v = api_version()
    if v not in handlers:
        jx = ss.gradients.jax()
        if not is_compiled():
            ss.perflib.compiling("the chain binomial's draws for JAX", 'Numba', 10, 30) # 14-23 s measured, with nothing in Numba's cache
        name = f'starsim_chain_step_{v[0]}_{v[1]}'
        if name not in registered: # XLA refuses a second handler under a name, so one dropped by check() is reused
            registered[name] = make_handler(*v)
            jx.ffi.register_ffi_target(name, jx.ffi.pycapsule(registered[name].address), platform='cpu')
        handlers[v] = (registered[name], name)
    return handlers[v][1]


def reported_version(handler):
    """ The API version the handler reports to XLA, by calling it as XLA does when it is registered (with a call frame holding the metadata extension) """
    meta = (ctypes.c_int64*5)() # XLA_FFI_Metadata: struct_size, api_version (struct_size, extension_start, major and minor as int32s), traits
    ext = (ctypes.c_int64*4)(32, 1, 0, ctypes.addressof(meta)) # XLA_FFI_Metadata_Extension: struct_size, type (1: metadata), next, metadata
    frame = (ctypes.c_int64*14)(112, ctypes.addressof(ext)) # XLA_FFI_CallFrame: struct_size, extension_start, and the rest (unread)
    ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(handler.address)(ctypes.addressof(frame))
    ints = (ctypes.c_int32*2).from_buffer(meta, 24)
    return (ints[0], ints[1])


def check():
    """ Check that the registered handler reports the XLA FFI API version of the installed JAX; if not, unregister it (so it is made and registered again next time) and raise an error, so the run falls back to the kernels """
    v = api_version()
    register()
    reported = reported_version(handlers[v][0])
    if reported != v:
        handlers.pop(v)
        errormsg = f'its handler reports XLA FFI API version {reported[0]}.{reported[1]}, but the installed JAX ({ss.gradients.jax().__version__}) has {v[0]}.{v[1]}'
        raise RuntimeError(errormsg)
    return v


#%% The run

def pack_register(reg):
    """ A movement register (`ss.kernels.Register`) as step() takes it: its integers packed into one array (its step, the offset of each part, then the parts: see `rfield()`), its numbers of people, and the shares of those added """
    parts = [np.asarray(getattr(reg, name), dtype=np.int64).ravel() for name in reg_fields]
    offsets = 2 + len(parts) + np.concatenate([[0], np.cumsum([len(v) for v in parts])])
    R = np.concatenate([[reg.step], offsets] + parts).astype(np.int64)
    return dict(R=R, reg_n=np.asarray(reg.n, dtype=float), ent_w=np.asarray(reg.ent_w, dtype=float))


class ChainRun(ss.gradients.Run):
    """
    One engine's chain-binomial (or tau-leaping) run in JAX, as a function of the numbers in it and a seed (see `function()`)

    Args:
        program, dt, npts, per_step: as for `ss.gradients.Run`
        solver (int): `ss.kernels.solvers['binomial']` or `['tau']`
        n_sub (int): the substeps of each step, for the chain binomial
    """
    def __init__(self, program, dt, npts, solver, n_sub, per_step):
        super().__init__(program, dt, npts, solver=solver, n_sub=max(n_sub, 1), per_step=per_step, n_max=1)
        self.tau = solver == ss.kernels.solvers['tau']
        return

    def layout(self):
        """ As `ss.gradients.Run.layout()`, and the parts of the program that `step()` needs, packed into one array of integers: a head, the offset of each part, the parts, then the movers' starts and pairs """
        super().layout()
        p = self.p
        parts = [np.asarray(getattr(p, name), dtype=np.int64).ravel() for name in fields]
        n_off = len(fields) + 1
        offsets = n_head + n_off + np.concatenate([[0], np.cumsum([len(v) for v in parts])])
        head = [0, self.ng, self.nflows, p.n_movers, self.ng >= vec_min, max(p.src_chain.max(initial=0), 1)] # The compartments are filled in by function(); the most stages a dwell time has, for the scratch
        movers = np.concatenate([np.asarray(p.mov_starts, dtype=np.int64), np.asarray(p.mov_pair, dtype=np.int64)], axis=1).ravel()
        self.packed = np.concatenate([head, offsets] + parts + [movers]).astype(np.int64)
        self.inflows = p.src < 0
        self.has_conv = len(self.cv_flow) > 0
        return

    def middle_only(self):
        """ Whether the tables are only needed at the middle of each step (one substep of the chain binomial), so are kept at one row a step: see `tables()` """
        return not self.tau and self.n_sub == 1

    def tables(self, X):
        """ The numbers of a run, with its tables kept at one row a step if only the middle of each is needed (as the kernels take them for the chain binomial) """
        if self.per_step == 1 or not self.middle_only():
            return X
        return dict(X, **{name:(X[name][1::3] if len(X[name]) > 1 else X[name]) for name in ss.gradients.table_fields})

    def vals(self, X, ti, th):
        """ The tables at a fraction th of the way through step ti (see `ss.kernels.interp()`) """
        jnp = ss.gradients.jax().numpy
        def at(table):
            if table.shape[0] == 1:
                return table[0]
            elif self.per_step == 1 or self.middle_only(): # One row a step
                return table[ti]
            row = ti*3
            return (2*th - 1)*(th - 1)*table[row] + 4*th*(1 - th)*table[row+1] + th*(2*th - 1)*table[row+2]
        return [at(jnp.asarray(X[name])) for name in ss.gradients.table_fields] + [X]

    def function(self, reduce=False):
        """
        The run, as a function of a dict of its numbers (see `ss.gradients.capture()`, with the tables as `tables()` gives
        them, the deliveries' groups as `sh_dst`, and the register as `pack_register()` gives it) and a seed

        Returns the compartments at the end, at each time point, the flows over each step, by group, the number
        each delivery reached and found eligible at each time point, by group, and the people that the register's
        events could not find, as `ss.kernels.run()` does. With reduce=True (for many runs), rather than the
        compartments and flows by group: the number in each state and the total of each flow at each time point,
        and some of them by group (`X['state_mat']` gives the states from the compartments, and `X['keep_s']` and
        `X['keep_f']` choose the states and flows to keep by group).
        """
        jx = ss.gradients.jax()
        jnp = jx.numpy
        lax = jx.lax
        dt, n_sub, tau = self.dt, self.n_sub, self.tau

        def run(X, seed):
            X = self.prepare(X)
            nc = X['y'].shape[0]
            S = self.packed.copy()
            S[0] = nc
            K = X['sh_due'].shape[0]
            has_reg = X['reg_n'].shape[0] > 0
            shapes = [(nc, self.ng), (self.nflows, self.ng), (1,), (K, 2, self.ng)]
            draw = jx.ffi.ffi_call(register(), [jx.ShapeDtypeStruct(shape, np.float64) for shape in shapes], vmap_method='expand_dims')
            if reduce:
                keep = lambda y, flows: (X['state_mat'] @ y.sum(axis=-1), flows.sum(axis=-1), X['keep_s'] @ X['state_mat'] @ y, X['keep_f'] @ flows)
            else:
                keep = lambda y, flows: (y, flows)

            def call(y, ti, count, flags, rates=None, h=dt):
                """ One call to step(): the draws of a step, a substep, or a leap of length h, with the rates given, or the deliveries alone """
                ti = jnp.asarray(ti, dtype=np.int64)
                ctl = jnp.stack([jnp.asarray(seed, dtype=np.int64), (ti << leap_bits) + count, jnp.asarray(flags, dtype=np.int64), ti + 1])
                if rates is None:
                    H = jnp.zeros((self.nf, self.ng))
                else:
                    H = jnp.maximum(jnp.where(self.inflows[:, None], rates*X['inflow'], rates), 0.0)*h
                mov_prob = -jnp.expm1(-X['mov_rate']*h)
                return draw(ctl, y, H, S, X['tgt_w'], X['crd_w'], X['maps'], mov_prob, jnp.asarray(h, dtype=np.float64), X['sh_due'][:, ti], X['sh_sel'], X['sh_cov'], X['sh_dst'], X['R'], X['reg_n'], X['ent_w'])

            def body(carry, ti):
                y, short = carry
                tally = jnp.zeros((K, 2, self.ng))
                if K: # The deliveries, at the start of the step
                    y, _, _, tally = call(y, ti, n_sub if not tau else (1 << leap_bits) - 1, DELIVER)
                flows = jnp.zeros((self.nflows, self.ng))
                if tau: # Leaps short enough that the rates change little over each (see ss.kernels.step())
                    def leap(c):
                        y, flows, short, elapsed, count, pending = c
                        piece = jnp.minimum(jnp.floor((elapsed + 1e-12)*pieces/dt), pieces - 1)
                        rates = jnp.maximum(self.rates(y, self.vals(X, ti, (piece + 0.5)/pieces)), 0.0)
                        fastest = jnp.max(jx.ops.segment_sum(rates[self.leaves], self.p.src[self.leaves], nc)) # The fastest rate at which any compartment is being left
                        end = dt if self.per_step == 1 else (piece + 1)*dt/pieces # The end of the piece of the step that values which vary in time are held for
                        end = jnp.where(pending, jnp.minimum(end, dt/2), end) # The people due to leave a conveyor leave at the middle of the step
                        h = jnp.minimum(end - elapsed, 0.03/jnp.maximum(fastest, 1e-12))
                        out = elapsed + h >= dt - 1e-12
                        released = pending & (elapsed + h >= dt/2 - 1e-12)
                        flags = TAU + RELEASE_END*released + (SHIFT*self.has_conv + REGISTER*has_reg)*out
                        y, f, s, _ = call(y, ti, count, flags, rates, h)
                        return y, flows + f, short + s[0], jnp.where(out, dt, elapsed + h), count + 1, pending & ~released
                    y, flows, short = lax.while_loop(lambda c: c[3] < dt - 1e-12, leap, (y, flows, short, 0.0, 0, jnp.asarray(self.has_conv)))[:3]
                else:
                    for sub in range(n_sub): # Written out, which XLA runs faster than a loop inside the loop over steps
                        last = sub == n_sub - 1
                        flags = RELEASE*(sub == 0 and self.has_conv) + SHIFT*(last and self.has_conv) + REGISTER*(last and has_reg)
                        rates = jnp.maximum(self.rates(y, self.vals(X, ti, (sub + 0.5)/n_sub)), 0.0)
                        y, f, s, _ = call(y, ti, sub, flags, rates, dt/n_sub)
                        flows = flows + f
                        short = short + s[0]
                return (y, short), keep(y, flows) + (tally,)

            (y_end, short), outs = lax.scan(body, (X['y'], 0.0), jnp.arange(self.npts - 1))
            first = keep(X['y'], jnp.zeros((self.nflows, self.ng))) + (jnp.zeros((K, 2, self.ng)),) # The start, then the end of each step, as the kernels record them (with no flows, or deliveries, at the start)
            outs = tuple(jnp.concatenate([a[None], b]) for a,b in zip(first, outs))
            return (y_end,) + outs + (short,)

        return run


compiled = {} # The compiled runs, by what they depend on apart from the numbers in them (see `ss.gradients.Run.key()`), so a new sim of the same model is not compiled again


def on_cpu():
    """ Where the runs are taken: on the CPU, since the draws' handler is registered with XLA for the CPU only (see `register()`), also where JAX's default device is a GPU (CV24) """
    jx = ss.gradients.jax()
    return jx.default_device(jx.devices('cpu')[0])


def runner(program, how, many=False):
    """ The `ChainRun` of a program, and its compiled run: for one, or for a batch over seeds (all with the same numbers), reduced (see `ChainRun.function()`) """
    key = (many,) + ss.gradients.Run.key(program, *how)
    if key not in compiled:
        dt, npts, solver, n_sub, per_step = how
        run = ChainRun(program, dt, npts, solver, n_sub, per_step)
        jx = ss.gradients.jax()
        compiled[key] = (run, jx.jit(jx.vmap(run.function(reduce=True), in_axes=(None, 0)) if many else run.function()))
    return compiled[key]


def numbers(engine):
    """ The program of an engine's run (or a crossing's), how to run it, and the numbers in it, as `ChainRun.function()` takes them """
    if isinstance(engine, ss.solvers.Crossing):
        args, tables, per_step = engine.run_args()
        y, program, dt, npts, solver, n_sub = args
        X = {name:np.array(getattr(program, name), dtype=float) for name in ss.gradients.diff_fields}
        X.update({name:np.asarray(table, dtype=float) for name,table in zip(ss.gradients.table_fields, tables)}, y=np.array(y, dtype=float))
        how = (dt, npts, solver, n_sub, per_step)
    else:
        program, how, X = ss.gradients.capture(engine)
    sh = engine.shifts
    X.update(sh_due=sh.due.astype(float), sh_sel=np.asarray(sh.sel, dtype=float), sh_cov=np.asarray(sh.cov, dtype=float), sh_dst=np.stack([sh.to, sh.rest], axis=1).astype(float))
    X.update(pack_register(engine.register))
    for key in ['sh_to', 'sh_rest', 'sh_land', 'sh_moves']: # The ODE's
        X.pop(key, None)
    return program, how, X


def seed_of(engine, seed=None):
    """ The seed of a run: from the engine's random numbers (as the kernels are given theirs; for a crossing, its first engine's) """
    engine = engine.engines[0] if isinstance(engine, ss.solvers.Crossing) else engine
    return int(engine.make_rng(stream=1, **({} if seed is None else dict(seed=seed))).integers(2**62))


def run(engine):
    """ Take a whole chain-binomial (or tau-leaping) run of an engine (or a crossing) in JAX: returns what `ss.kernels.run()` does, and the number each delivery reached and found eligible (time points by deliveries by 2 by groups) """
    jx = ss.gradients.jax()
    program, how, X = numbers(engine)
    chain, func = runner(program, how)
    with jx.enable_x64(True), on_cpu():
        y, stocks, flows, tally, short = func(chain.tables(X), seed_of(engine))
    return np.asarray(y), np.asarray(stocks), np.asarray(flows), float(short), np.asarray(tally)


def run_crossing(cross):
    """ Take a whole run of the diseases of a crossing together in JAX: returns, for each, what `ss.kernels.run()` does """
    y, stocks, flows, short, tally = run(cross)
    cross.y = y
    return [cross.split(k, y, stocks, flows) + (short if k == 0 else 0.0,) for k in range(len(cross.engines))]


def run_many(engine, seeds, state_of, kept_s, kept_f, rows, max_bytes=200e6):
    """
    Take many runs of an engine (or a crossing) that differ only in their seed, in batches of runs taken together (`jax.vmap`)

    Returns what `ss.kernels.run_many()` fills in: the number in each state and the total of each flow at each time
    point, the states and flows kept by group at each row (see `ss.solvers.group_rows()`), and the number each delivery
    reached and found eligible, in total and by group, as runs by time points (or rows) by ... by groups (or 1).

    Args:
        state_of (array): the state of each compartment
        kept_s, kept_f (lists): the states and flows to keep by group
        rows (array): the row of each time point, or None if nothing is kept by group
        max_bytes (float): about the most memory a batch's results may take
    """
    jx = ss.gradients.jax()
    program, how, X = numbers(engine)
    chain, func = runner(program, how, many=True)
    nc, ng, npts, n_states = X['y'].shape[0], chain.ng, how[1], int(np.max(state_of)) + 1
    K = X['sh_due'].shape[0]
    X = chain.tables(X)
    X.update(state_mat=np.eye(n_states)[state_of].T, keep_s=np.eye(n_states)[list(kept_s)].reshape(-1, n_states), keep_f=np.eye(chain.nflows)[list(kept_f)].reshape(-1, chain.nflows))
    per_run = npts*(n_states + chain.nflows + (len(kept_s) + len(kept_f) + 2*K)*ng)*8
    batch = int(min(max(max_bytes//per_run, 1), len(seeds)))
    keys = np.array([seed_of(engine, seed) for seed in seeds])
    keys = np.concatenate([keys, np.zeros(-len(keys) % batch, dtype=keys.dtype)]) # The last batch filled out, so every batch is the same shape (compiled once)
    parts = []
    with jx.enable_x64(True), on_cpu():
        for i in range(0, len(keys), batch):
            out = func(X, keys[i:i+batch])
            parts.append([np.asarray(v) for v in out[1:6]])
    stocks, flows, stocks_g, flows_g, tally = [np.concatenate(v)[:len(seeds)] for v in zip(*parts)]
    tally_t = tally.sum(axis=-1, keepdims=True)
    if rows is None:
        stocks_g, flows_g, tally_g = [np.zeros((len(seeds), 0) + v.shape[2:]) for v in [stocks_g, flows_g, tally]]
    else: # A state as it is at the last time point of its row, and a flow (or a delivery) added up over them
        rows = np.asarray(rows)
        ends = np.flatnonzero(np.r_[rows[1:] != rows[:-1], True])
        starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
        stocks_g = stocks_g[:, ends]
        flows_g, tally_g = [np.add.reduceat(v, starts, axis=1) for v in [flows_g, tally]]
    return stocks[..., None], flows[..., None], stocks_g, flows_g, tally_t, tally_g
