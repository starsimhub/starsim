"""
Compiled passes for agents: over a network's edges (new random edges, the edges that might and that can transmit, those
that end, and copies without some), and over the agents (who leaves a state by competing rates, the women of an age for
births, and who reaches their debut on a sexual network)

Each is one pass (or two) in place of several NumPy passes and their temporaries, in chunks that several threads take for
a large network or population, as the passes over the agents in `ss.kernels` are (see `ss.options.threads`; one thread in
the workers of `ss.parallel()`). Lists are kept in order, and random numbers come from counters, so the results do not
depend on the number of threads. See v4_design/oct22/jax/agents.md for the timings. Called by `ss.Network`, `ss.RandomNet`,
`ss.SexualNet`, `ss.Births` and `ss.solvers.Agents`; not for the user.
"""
import numpy as np
import numba as nb
from . import _cache
from .. import kernels as k

_cache.depends(__file__, k.__file__) # The chunks and uniforms come from ss.kernels


#%% New random edges (ss.RandomNet)

uniform = nb.jit(cache=True, inline='always')(k.crn_uniform.py_func) # Starsim's counter-based uniform, inlined (a call to one loaded from Numba's cache is not)


@k.jit
def sources_chunk(inds, starts, a, b, source):
    """ The first end of each edge, for agents a to b: each agent once for each of its stubs, which start at `starts` """
    ids = inds[a:b] # Slices, so the loops' indices are known not to be negative
    st = starts[a:b+1]
    for i in range(len(ids)):
        u = ids[i]
        for j in range(st[i], st[i+1]):
            source[j] = u
    return


@k.jit
def targets_chunk(inds, source, key, by_agent, a, b, target):
    """ The other end of edges a to b: an agent drawn at random (`by_agent`), or the owner of a stub drawn at random, from the uniform for each edge """
    pool = inds if by_agent else source
    n = len(pool)
    out = target[a:b]
    for j in range(len(out)):
        r = min(np.int64(uniform(key, a + j, k._U11, k._INV53)*n), n - 1) # Uniform on 0 to n-1 (to within n/2^53)
        out[j] = pool[r]
    return


def edges_driver(threaded):
    loop = nb.prange if threaded else range
    def random_edges(inds, starts, seed, by_agent, source, target, nchunks):
        n = len(source)
        for c in loop(nchunks):
            a, b = k.bounds(c, len(inds), nchunks)
            sources_chunk(inds, starts, a, b, source)
        key = k.crn_key(seed, 0)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            targets_chunk(inds, source, key, by_agent, a, b, target)
        return
    return random_edges

edges_drivers = k.compile_drivers(edges_driver)

def random_edges(inds, n_contacts, seed, by_agent):
    """
    The two ends of new random edges: each agent is the first end once for each of its stubs (`n_contacts`, one number each or one for all), and the
    other end of each is an agent drawn at random (`by_agent`, which with the same number of stubs for everyone is the same) or the owner of a stub drawn
    at random from all of them, with replacement; the uniform for edge j is `ss.kernels.crn_uniform()` of j and a key made from `seed`
    """
    inds = np.asarray(inds, dtype=np.int64)
    nc = np.broadcast_to(np.asarray(n_contacts, dtype=np.int64), inds.shape)
    starts = np.zeros(len(inds) + 1, dtype=np.int64)
    np.cumsum(nc, out=starts[1:])
    source = np.empty(starts[-1], dtype=np.int64) # Made here, and filled by the driver
    target = np.empty(starts[-1], dtype=np.int64)
    func, nchunks = threads(edges_drivers, len(source))
    func(inds, starts, seed, by_agent, source, target, nchunks)
    return source, target


#%% The edges that can transmit, that end, or that touch someone removed

@k.jit
def flags_chunk(rel_trans, rel_sus, a, b, flags):
    """ For agents a to b (by UID): 1 if their rel_trans is not 0, plus 2 if their rel_sus is not 0 (so a NaN counts, and is ruled out later) """
    rt = rel_trans[a:b] # As can_chunk()
    rs = rel_sus[a:b]
    out = flags[a:b]
    for i in range(len(out)):
        out[i] = (1 if rt[i] != 0 else 0) + (2 if rs[i] != 0 else 0)
    return


@k.jit
def candidates_chunk(p1, p2, rel_trans, rel_sus, flags, two_way, a, b, out0, out1):
    """
    For edges a to b: the indices of those from someone infectious to someone susceptible, p1 to p2 in `out0` and (if `two_way`) p2 to p1 in
    `out1`, each written from a; how many of each. By `flags` if given (for many edges), else from `rel_trans` and `rel_sus` (for a few)
    """
    q1 = p1[a:b] # As can_chunk()
    q2 = p2[a:b]
    o0 = out0[a:b]
    o1 = out1[a:b] if two_way else out1[:0]
    by_flags = len(flags) > 0
    m0 = 0
    m1 = 0
    for i in range(len(q1)):
        u = q1[i]
        v = q2[i]
        if by_flags:
            f1 = flags[u]
            f2 = flags[v]
        else:
            f1 = (1 if rel_trans[u] != 0 else 0) + (2 if rel_sus[u] != 0 else 0)
            f2 = (1 if rel_trans[v] != 0 else 0) + (2 if rel_sus[v] != 0 else 0)
        o0[m0] = a + i # Branchless, as can_chunk()
        m0 += (f1 & 1) & (f2 >> 1)
        if two_way:
            o1[m1] = a + i
            m1 += (f2 & 1) & (f1 >> 1)
    return m0, m1


@k.jit
def close_gaps(out, found, n, nchunks):
    """ Move the part of `out` that each chunk wrote (`found[c]` from its start) to follow the one before, and return the list as far as written """
    p = found[0]
    for c in range(1, nchunks):
        a = k.bounds(c, n, nchunks)[0]
        for i in range(found[c]):
            out[p + i] = out[a + i]
        p += found[c]
    return out[:p]


def candidates_driver(threaded):
    loop = nb.prange if threaded else range
    def candidates(p1, p2, rel_trans, rel_sus, two_way, flags, out0, out1, nchunks):
        n_agents = len(flags)
        for c in loop(nchunks):
            a, b = k.bounds(c, n_agents, nchunks)
            flags_chunk(rel_trans, rel_sus, a, b, flags)
        n = len(p1)
        found0 = np.zeros(nchunks, dtype=np.int64)
        found1 = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            found0[c], found1[c] = candidates_chunk(p1, p2, rel_trans, rel_sus, flags, two_way, a, b, out0, out1)
        return close_gaps(out0, found0, n, nchunks), close_gaps(out1, found1, n, nchunks)
    return candidates

candidates_drivers = k.compile_drivers(candidates_driver)

def candidates(p1, p2, rel_trans, rel_sus, two_way=True):
    """
    The indices of the edges from someone with `rel_trans` not 0 to someone with `rel_sus` not 0 (by UID), p1 to p2 and (if `two_way`) p2 to p1, in order:
    the edges that might transmit, found for both directions in one pass over the edges (a superset of those that can, which `can_transmit()` then finds among them)
    """
    flags = np.empty(len(rel_trans) if 8*len(p1) > len(rel_trans) else 0, dtype=np.uint8) # A byte for each agent, which the edges read 5-10 times more quickly than two numbers each, unless there are few edges
    func, nchunks = threads(candidates_drivers, max(len(p1), len(flags)))
    out1 = np.empty(len(p1) if two_way else 0, dtype=np.int64)
    found = func(p1, p2, rel_trans, rel_sus, two_way, flags, np.empty(len(p1), dtype=np.int64), out1, nchunks)
    return found if two_way else found[:1]


@k.jit
def can_chunk(src, trg, rel_trans, rel_sus, beta, a, b, out, pos):
    """ For edges a to b: the indices of those with a chance of transmission above 0, written from `pos`, and how many there are """
    s = src[a:b] # Slices, so the loop's indices are known not to be negative
    t = trg[a:b]
    bs = beta[a:b] if len(beta) == len(src) else beta
    step = 1 if len(bs) == len(s) else 0
    m = 0
    for i in range(len(s)):
        hit = rel_trans[s[i]]*rel_sus[t[i]]*bs[i*step] > 0 # As nb_transmit(), with no fastmath, so a NaN is no transmission
        out[pos + m] = a + i # Written every time (branchless) and kept only if m advances
        m += hit
    return m


@k.jit
def ending_chunk(dur, p1, p2, alive, a, b, out, pos, count):
    """ For edges a to b: the number that end (fewer than two steps left, as dur - 1 <= 0, or a partner not alive), or (unless `count`) their indices, written from `pos` """
    d = dur[a:b] # As can_chunk()
    q1 = p1[a:b]
    q2 = p2[a:b]
    m = 0
    for i in range(len(d)):
        if not (d[i] > 1 and alive[q1[i]] and alive[q2[i]]):
            if not count:
                out[pos + m] = a + i
            m += 1
    return m


@k.jit
def touching_chunk(p1, p2, removed, a, b, out, pos, count):
    """ For edges a to b: the number with either end in `removed` (by UID), or (unless `count`) their indices, written from `pos` """
    q1 = p1[a:b] # As can_chunk()
    q2 = p2[a:b]
    m = 0
    for i in range(len(q1)):
        if removed[q1[i]] | removed[q2[i]]:
            if not count:
                out[pos + m] = a + i
            m += 1
    return m


def can_driver(threaded):
    loop = nb.prange if threaded else range
    def can_transmit(src, trg, rel_trans, rel_sus, beta, out, nchunks):
        n = len(src)
        found = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks): # Each chunk writes its own part, without counting first (branchless, since the test is often true)
            a, b = k.bounds(c, n, nchunks)
            found[c] = can_chunk(src, trg, rel_trans, rel_sus, beta, a, b, out, a)
        return close_gaps(out, found, n, nchunks)
    return can_transmit

def ending_driver(threaded):
    loop = nb.prange if threaded else range
    def ending(dur, p1, p2, alive, nchunks):
        n = len(dur)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        none = np.empty(0, dtype=np.int64)
        for c in loop(nchunks): # Counted first, so only what is found is written
            a, b = k.bounds(c, n, nchunks)
            found[c+1] = ending_chunk(dur, p1, p2, alive, a, b, none, 0, True)
        starts = np.cumsum(found)
        if starts[-1] == 0 or starts[-1] == n: # None or all: no list is needed
            return starts[-1], none
        out = np.empty(starts[-1], dtype=np.int64)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            ending_chunk(dur, p1, p2, alive, a, b, out, starts[c], False)
        return starts[-1], out
    return ending

def touching_driver(threaded):
    loop = nb.prange if threaded else range
    def touching(p1, p2, removed, nchunks):
        n = len(p1)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        none = np.empty(0, dtype=np.int64)
        for c in loop(nchunks): # As ending()
            a, b = k.bounds(c, n, nchunks)
            found[c+1] = touching_chunk(p1, p2, removed, a, b, none, 0, True)
        starts = np.cumsum(found)
        if starts[-1] == 0 or starts[-1] == n:
            return starts[-1], none
        out = np.empty(starts[-1], dtype=np.int64)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            touching_chunk(p1, p2, removed, a, b, out, starts[c], False)
        return starts[-1], out
    return touching

can_drivers = k.compile_drivers(can_driver)
ending_drivers = k.compile_drivers(ending_driver)
touching_drivers = k.compile_drivers(touching_driver)

def threads(drivers, n):
    """ The serial or threaded driver of a pass over n edges, and the number of chunks to take it in """
    nchunks = k.n_chunks(n)
    return drivers[nchunks > 1], nchunks


def can_transmit(src, trg, rel_trans, rel_sus, beta):
    """ The indices of the edges with a chance of transmission above 0 (`rel_trans[src]·rel_sus[trg]·beta`, with `beta` one value for each edge or one for all), in order """
    func, nchunks = threads(can_drivers, len(src))
    return func(src, trg, rel_trans, rel_sus, np.atleast_1d(beta), np.empty(len(src), dtype=np.int64), nchunks)


def ending(dur, p1, p2, alive):
    """ The number of partnerships that end in a step (with fewer than two steps left, as `dur` counts them, or a partner not alive), and their indices if not none or all """
    func, nchunks = threads(ending_drivers, len(dur))
    return func(dur, p1, p2, alive, nchunks)


def touching(p1, p2, removed):
    """ The number of edges with either end in `removed` (a mask by UID, covering every UID in the edges), and their indices if not none or all """
    func, nchunks = threads(touching_drivers, len(p1))
    return func(p1, p2, removed, nchunks)


#%% Women of an age (ss.Births)

@k.jit
def women_chunk(auids, female, age, lo, hi, a, b, out):
    """ For agents a to b of `auids`: the UIDs of the women aged at least `lo` and under `hi`, written from a; how many there are """
    us = auids[a:b] # As can_chunk()
    o = out[a:b]
    m = 0
    for i in range(len(us)):
        u = us[i]
        x = age[u]
        o[m] = u # Branchless, as can_chunk()
        m += female[u] & (x >= lo) & (x < hi)
    return m


def women_driver(threaded):
    loop = nb.prange if threaded else range
    def women(auids, female, age, lo, hi, out, nchunks):
        n = len(auids)
        found = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            found[c] = women_chunk(auids, female, age, lo, hi, a, b, out)
        return close_gaps(out, found, n, nchunks)
    return women

women_drivers = k.compile_drivers(women_driver)

def women(auids, female, age, lo, hi):
    """ The UIDs among `auids` of the women (`female`, by UID) aged at least `lo` and under `hi` (`age`, by UID), in order; compared as NumPy would """
    dtype = np.result_type(age, lo, hi) # E.g. float32 for float32 ages and Python numbers
    func, nchunks = threads(women_drivers, len(auids))
    return func(auids, female, age, dtype.type(lo), dtype.type(hi), np.empty(len(auids), dtype=auids.dtype), nchunks)


#%% Debut (ss.SexualNet)

@k.jit
def debut_chunk(auids, alive, decided, age, debut, reached, a, b, out):
    """ For agents a to b of `auids`: the UIDs of those alive with no debut yet (unless `reached`), or (if `reached`) alive, undecided, and at least their debut age, written from a; how many """
    us = auids[a:b] # As can_chunk()
    o = out[a:b]
    m = 0
    for i in range(len(us)):
        u = us[i]
        o[m] = u # Branchless, as can_chunk()
        if reached:
            m += alive[u] & (not decided[u]) & (age[u] >= debut[u])
        else:
            m += alive[u] & np.isnan(debut[u])
    return m


def debut_driver(threaded):
    loop = nb.prange if threaded else range
    def debuts(auids, alive, decided, age, debut, reached, out, nchunks):
        n = len(auids)
        found = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            found[c] = debut_chunk(auids, alive, decided, age, debut, reached, a, b, out)
        return close_gaps(out, found, n, nchunks)
    return debuts

debut_drivers = k.compile_drivers(debut_driver)

def debuts(auids, alive, decided, age, debut, reached):
    """ The UIDs among `auids` (in order) of those alive with no debut age yet, or (if `reached`) alive, undecided, and at least their debut age (the arrays by UID) """
    func, nchunks = threads(debut_drivers, len(auids))
    return func(auids, alive, decided, age, debut, reached, np.empty(len(auids), dtype=auids.dtype), nchunks)


#%% Removing edges

@k.jit
def compress_chunk(arr, gone, a, b, out):
    """ Copy elements a to b of an array, except those at the indices `gone`, to where they go in `out`, in runs between them """
    g = np.searchsorted(gone, a) # The first index removed at or after a
    o = a - g # Where element a goes
    x = a
    while x < b:
        y = min(gone[g], b) if g < len(gone) else b # The end of this run
        run = arr[x:y] # Slices, so the loop's indices are known not to be negative (and it is a copy of memory)
        dst = out[o:o+len(run)]
        for i in range(len(run)):
            dst[i] = run[i]
        o += len(run)
        x = y + 1 # Past the element removed
        g += 1
    return


def compress_driver(threaded):
    loop = nb.prange if threaded else range
    def compress(arr, gone, out, nchunks):
        n = len(arr)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            compress_chunk(arr, gone, a, b, out)
        return
    return compress

compress_drivers = k.compile_drivers(compress_driver)

def compress(arr, gone):
    """ A copy of an array without the elements at the indices `gone` (in order), as `np.delete()` """
    out = np.empty(len(arr) - len(gone), dtype=arr.dtype)
    func, nchunks = threads(compress_drivers, len(arr))
    func(arr, gone, out, nchunks)
    return out


#%% Competing exits from a state (ss.solvers.Agents)

@k.jit
def exits_chunk(uids, slots, key, shift, scale, p_exit, edges, a, b, pos, which):
    """
    For agents a to b of `uids`: those who leave a state (their uniform for the call is below `p_exit`), written at their own positions in
    `pos` (their position in `uids`) and `which` (the exit they take: the first i with uniform < edges[i+1], or -1 if none); how many leave
    """
    us = uids[a:b] # Slices, so the loop's indices are known not to be negative
    ps = pos[a:b]
    ws = which[a:b]
    by_uid = len(slots) == 0 # The slots are the UIDs
    n_exits = len(edges) - 1
    m = 0
    for i in range(len(us)):
        u = uniform(key, us[i] if by_uid else slots[us[i]], shift, scale)
        if u < p_exit:
            t = 0
            while t < n_exits and not u < edges[t+1]:
                t += 1
            ps[m] = a + i
            ws[m] = t if t < n_exits else -1
            m += 1
    return m


def exits_driver(threaded):
    loop = nb.prange if threaded else range
    def exits(uids, slots, seed, ind, shift, scale, p_exit, edges, pos, which, nchunks):
        n = len(uids)
        found = np.zeros(nchunks, dtype=np.int64)
        key = k.crn_key(seed, ind)
        for c in loop(nchunks):
            a, b = k.bounds(c, n, nchunks)
            found[c] = exits_chunk(uids, slots, key, shift, scale, p_exit, edges, a, b, pos, which)
        p = found[0]
        for c in range(1, nchunks): # Close the gaps between the chunks
            a = k.bounds(c, n, nchunks)[0]
            for i in range(found[c]):
                pos[p + i] = pos[a + i]
                which[p + i] = which[a + i]
            p += found[c]
        return p
    return exits

exits_drivers = k.compile_drivers(exits_driver)

def competing_exits(uids, slots, seed, ind, dtype, p_exit, edges):
    """
    Who leaves a state by competing rates that are the same for everyone, in one pass that draws each agent's uniform for the call (as
    `hash_uniforms()` gives it: see `ss.Dist.crn_key()`): the positions in `uids` of those who leave, and the exit each takes (by `edges`,
    the cumulative chances of the exits, from 0; -1 for none, where rounding leaves a gap below `p_exit`), in order
    """
    uids = np.asarray(uids, dtype=np.int64)
    pos = np.empty(len(uids), dtype=np.int64) # Only those who leave are written
    which = np.empty(len(uids), dtype=np.int64)
    func, nchunks = threads(exits_drivers, len(uids))
    m = func(uids, np.empty(0, dtype=np.int64) if slots is None else slots, seed, ind, *k.crn_bits(dtype), float(p_exit), np.asarray(edges, dtype=float), pos, which, nchunks)
    return pos[:m], which[:m]
