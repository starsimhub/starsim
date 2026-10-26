"""
Compiled passes for agents: over the agents (who is infected, the counts of each state, who has died and who is kept alive at the
end of each step, who leaves a state by competing rates, the women of an age for births, and who reaches their debut on a sexual
network), and over a network's edges (new random edges, the edges that might and that can transmit, those that end, and copies
without some)

Each is one pass (or two) in place of several NumPy passes and their temporaries, in chunks that several threads take for
a large network or population (see `ss.options.threads`; one thread in the workers of `ss.parallel()`). Lists are kept in
order, and random numbers come from counters, so the results do not depend on the number of threads. See
v4_design/oct25/backlog/perf_jax.md for the timings. Called by `ss.People`, `ss.Network`, `ss.RandomNet`, `ss.SexualNet`,
`ss.Births`, `ss.solvers.Agents` and the groups of `ss.dimensions`; not for the user. They are kept apart from the
compartmental kernels (`ss.kernels`), so a change to either does not make Numba compile the other again.
"""
import numpy as np
import numba as nb
import starsim as ss
from . import _cache, _draws as draws

_cache.depends(__file__, draws.__file__) # The uniforms come from the draws

jit = nb.jit(nopython=True, cache=True, error_model='numpy') # As ss.kernels compiles its kernels
pjit = nb.jit(nopython=True, cache=True, error_model='numpy', parallel=True)


#%% Passes over the whole population, in chunks taken by several threads when it is large
# Each pass has a serial and a threaded driver around the same work on one chunk, so a sim that is one of many
# (in ss.parallel() or a calibration's workers, which run with one thread) never starts Numba's threads. Counts are
# whole numbers and lists of agents are kept in order, so the results do not depend on the number of threads.

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


#%% Levels (ss.dimensions), and who could be infected (ss.solvers.Agents)

@jit
def levels(raw):
    """ The level of each agent in a dimension, as whole numbers (0 for an agent with none) """
    out = np.zeros(len(raw), dtype=np.int32) # 32-bit, as the groups are kept
    for i in range(len(raw)):
        if raw[i] == raw[i]: # Not NaN
            out[i] = int(raw[i])
    return out



@jit
def crossed(raw, level, lo, hi):
    """ The agents whose value of a binned property is no longer in the bin of their level (`lo[level] <= raw < hi[level]`; NaN, e.g. of an agent who has died, is in the last), in order """
    out = np.empty(len(raw), dtype=np.int64)
    last = len(lo) - 1
    m = 0
    for i in range(len(raw)):
        k = level[i]
        x = raw[i]
        if not ((x >= lo[k] and x < hi[k]) or (x != x and k == last)):
            out[m] = i
            m += 1
    return out[:m]


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


#%% Counts by group, those in a state, and who is infected (ss.solvers.Agents)

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
                got = draws.crn_uniform(key, a + i if by_uid else sl[i], shift, scale) < prob
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
        key = draws.key(seed, ind)
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
    shift, scale = draws.crn_bits(dtype)
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


#%% New random edges (ss.RandomNet)

@jit
def sources_chunk(inds, starts, a, b, source):
    """ The first end of each edge, for agents a to b: each agent once for each of its stubs, which start at `starts` """
    ids = inds[a:b] # Slices, so the loops' indices are known not to be negative
    st = starts[a:b+1]
    for i in range(len(ids)):
        u = ids[i]
        for j in range(st[i], st[i+1]):
            source[j] = u
    return


@jit
def targets_chunk(inds, source, key, by_agent, a, b, target):
    """ The other end of edges a to b: an agent drawn at random (`by_agent`), or the owner of a stub drawn at random, from the uniform for each edge """
    pool = inds if by_agent else source
    n = len(pool)
    out = target[a:b]
    for j in range(len(out)):
        r = min(np.int64(draws.crn_uniform(key, a + j, draws.U11, draws.INV53)*n), n - 1) # Uniform on 0 to n-1 (to within n/2^53)
        out[j] = pool[r]
    return


def edges_driver(threaded):
    loop = nb.prange if threaded else range
    def random_edges(inds, starts, seed, by_agent, source, target, nchunks):
        n = len(source)
        for c in loop(nchunks):
            a, b = bounds(c, len(inds), nchunks)
            sources_chunk(inds, starts, a, b, source)
        key = draws.key(seed, 0)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            targets_chunk(inds, source, key, by_agent, a, b, target)
        return
    return random_edges

edges_drivers = compile_drivers(edges_driver)

def random_edges(inds, n_contacts, seed, by_agent):
    """
    The two ends of new random edges: each agent is the first end once for each of its stubs (`n_contacts`, one number each or one for all), and the
    other end of each is an agent drawn at random (`by_agent`, which with the same number of stubs for everyone is the same) or the owner of a stub drawn
    at random from all of them, with replacement; the uniform for edge j is `draws.crn_uniform()` of j and a key made from `seed`
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

@jit
def flags_chunk(rel_trans, rel_sus, a, b, flags):
    """ For agents a to b (by UID): 1 if their rel_trans is not 0, plus 2 if their rel_sus is not 0 (so a NaN counts, and is ruled out later) """
    rt = rel_trans[a:b] # As can_chunk()
    rs = rel_sus[a:b]
    out = flags[a:b]
    for i in range(len(out)):
        out[i] = (1 if rt[i] != 0 else 0) + (2 if rs[i] != 0 else 0)
    return


@jit
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


@jit
def close_gaps(out, found, n, nchunks):
    """ Move the part of `out` that each chunk wrote (`found[c]` from its start) to follow the one before, and return the list as far as written """
    p = found[0]
    for c in range(1, nchunks):
        a = bounds(c, n, nchunks)[0]
        for i in range(found[c]):
            out[p + i] = out[a + i]
        p += found[c]
    return out[:p]


def candidates_driver(threaded):
    loop = nb.prange if threaded else range
    def candidates(p1, p2, rel_trans, rel_sus, two_way, flags, out0, out1, nchunks):
        n_agents = len(flags)
        for c in loop(nchunks):
            a, b = bounds(c, n_agents, nchunks)
            flags_chunk(rel_trans, rel_sus, a, b, flags)
        n = len(p1)
        found0 = np.zeros(nchunks, dtype=np.int64)
        found1 = np.zeros(nchunks, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found0[c], found1[c] = candidates_chunk(p1, p2, rel_trans, rel_sus, flags, two_way, a, b, out0, out1)
        return close_gaps(out0, found0, n, nchunks), close_gaps(out1, found1, n, nchunks)
    return candidates

candidates_drivers = compile_drivers(candidates_driver)

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


@jit
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


@jit
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


@jit
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
            a, b = bounds(c, n, nchunks)
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
            a, b = bounds(c, n, nchunks)
            found[c+1] = ending_chunk(dur, p1, p2, alive, a, b, none, 0, True)
        starts = np.cumsum(found)
        if starts[-1] == 0 or starts[-1] == n: # None or all: no list is needed
            return starts[-1], none
        out = np.empty(starts[-1], dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
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
            a, b = bounds(c, n, nchunks)
            found[c+1] = touching_chunk(p1, p2, removed, a, b, none, 0, True)
        starts = np.cumsum(found)
        if starts[-1] == 0 or starts[-1] == n:
            return starts[-1], none
        out = np.empty(starts[-1], dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            touching_chunk(p1, p2, removed, a, b, out, starts[c], False)
        return starts[-1], out
    return touching

can_drivers = compile_drivers(can_driver)
ending_drivers = compile_drivers(ending_driver)
touching_drivers = compile_drivers(touching_driver)

def threads(drivers, n):
    """ The serial or threaded driver of a pass over n edges, and the number of chunks to take it in """
    nchunks = n_chunks(n)
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

@jit
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
            a, b = bounds(c, n, nchunks)
            found[c] = women_chunk(auids, female, age, lo, hi, a, b, out)
        return close_gaps(out, found, n, nchunks)
    return women

women_drivers = compile_drivers(women_driver)

def women(auids, female, age, lo, hi):
    """ The UIDs among `auids` of the women (`female`, by UID) aged at least `lo` and under `hi` (`age`, by UID), in order; compared as NumPy would """
    dtype = np.result_type(age, lo, hi) # E.g. float32 for float32 ages and Python numbers
    func, nchunks = threads(women_drivers, len(auids))
    return func(auids, female, age, dtype.type(lo), dtype.type(hi), np.empty(len(auids), dtype=auids.dtype), nchunks)


#%% Debut (ss.SexualNet)

@jit
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
            a, b = bounds(c, n, nchunks)
            found[c] = debut_chunk(auids, alive, decided, age, debut, reached, a, b, out)
        return close_gaps(out, found, n, nchunks)
    return debuts

debut_drivers = compile_drivers(debut_driver)

def debuts(auids, alive, decided, age, debut, reached):
    """ The UIDs among `auids` (in order) of those alive with no debut age yet, or (if `reached`) alive, undecided, and at least their debut age (the arrays by UID) """
    func, nchunks = threads(debut_drivers, len(auids))
    return func(auids, alive, decided, age, debut, reached, np.empty(len(auids), dtype=auids.dtype), nchunks)


#%% Removing edges

@jit
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
            a, b = bounds(c, n, nchunks)
            compress_chunk(arr, gone, a, b, out)
        return
    return compress

compress_drivers = compile_drivers(compress_driver)

def compress(arr, gone):
    """ A copy of an array without the elements at the indices `gone` (in order), as `np.delete()` """
    out = np.empty(len(arr) - len(gone), dtype=arr.dtype)
    func, nchunks = threads(compress_drivers, len(arr))
    func(arr, gone, out, nchunks)
    return out


#%% Competing exits from a state (ss.solvers.Agents)

@jit
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
        u = draws.crn_uniform(key, us[i] if by_uid else slots[us[i]], shift, scale)
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
        key = draws.key(seed, ind)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c] = exits_chunk(uids, slots, key, shift, scale, p_exit, edges, a, b, pos, which)
        p = found[0]
        for c in range(1, nchunks): # Close the gaps between the chunks
            a = bounds(c, n, nchunks)[0]
            for i in range(found[c]):
                pos[p + i] = pos[a + i]
                which[p + i] = which[a + i]
            p += found[c]
        return p
    return exits

exits_drivers = compile_drivers(exits_driver)

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
    m = func(uids, np.empty(0, dtype=np.int64) if slots is None else slots, seed, ind, *draws.crn_bits(dtype), float(p_exit), np.asarray(edges, dtype=float), pos, which, nchunks)
    return pos[:m], which[:m]


@jit
def agent_exits_chunk(uids, slots, key, shift, scale, p_exit, share, hazards, a, b, pos, which):
    """
    As `exits_chunk()`, with chances that differ between agents: agent i leaves if their uniform is below `p_exit[i]`, by the first
    exit t whose edge, the hazards of exits 0 to t added in order, times `share[i]`, is above it (the same numbers as NumPy's cumsum)
    """
    us = uids[a:b]
    ps = pos[a:b]
    ws = which[a:b]
    by_uid = len(slots) == 0
    n_exits = hazards.shape[0]
    m = 0
    for i in range(len(us)):
        j = a + i
        u = draws.crn_uniform(key, us[i] if by_uid else slots[us[i]], shift, scale)
        if u < p_exit[j]:
            t = 0
            edge = 0.0
            while t < n_exits:
                edge += hazards[t, j]
                if u < edge*share[j]:
                    break
                t += 1
            ps[m] = j
            ws[m] = t if t < n_exits else -1
            m += 1
    return m


def agent_exits_driver(threaded):
    loop = nb.prange if threaded else range
    def exits(uids, slots, seed, ind, shift, scale, p_exit, share, hazards, pos, which, nchunks):
        n = len(uids)
        found = np.zeros(nchunks, dtype=np.int64)
        key = draws.key(seed, ind)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c] = agent_exits_chunk(uids, slots, key, shift, scale, p_exit, share, hazards, a, b, pos, which)
        p = found[0]
        for c in range(1, nchunks): # Close the gaps between the chunks
            a = bounds(c, n, nchunks)[0]
            for i in range(found[c]):
                pos[p + i] = pos[a + i]
                which[p + i] = which[a + i]
            p += found[c]
        return p
    return exits

agent_exits_drivers = compile_drivers(agent_exits_driver)

def competing_agent_exits(uids, slots, seed, ind, dtype, p_exit, share, hazards):
    """
    As `competing_exits()`, with chances that differ between agents: each agent's chance of leaving (`p_exit`), and the exits' hazards
    (one row for each exit, and a column for each agent), whose running total times `share` (`p_exit` over the total hazard) are the edges
    """
    uids = np.asarray(uids, dtype=np.int64)
    pos = np.empty(len(uids), dtype=np.int64)
    which = np.empty(len(uids), dtype=np.int64)
    func, nchunks = threads(agent_exits_drivers, len(uids))
    m = func(uids, np.empty(0, dtype=np.int64) if slots is None else slots, seed, ind, *draws.crn_bits(dtype), np.ascontiguousarray(p_exit, dtype=float), np.ascontiguousarray(share, dtype=float), np.ascontiguousarray(hazards, dtype=float), pos, which, nchunks)
    return pos[:m], which[:m]
