"""
Not for the user: compiled passes for agents: over the agents (who is infected, the counts of each state, who has died and who is kept alive at the
end of each step, who leaves a state by competing rates, the women of an age for births, and who reaches their debut on a sexual
network), and over a network's edges (new random edges, the edges that might and that can transmit, those that end, and copies
without some)

Each is one pass (or two) in place of several NumPy passes and their temporaries, in chunks that several threads take for
a large network or population (see `ss.options.threads`; one thread in the workers of `ss.parallel()`). Lists are kept in
order, and random numbers come from counters, so the results do not depend on the number of threads. Called by `ss.People`, `ss.Network`, `ss.RandomNet`, `ss.SexualNet`,
`ss.Births`, `ss.solvers.Agents` and the groups of `ss.dimensions`. They are kept apart from the
compartmental kernels (`ss.solvers.numba.compartmental`), so a change to either does not make Numba compile the other again.

Used with: agents; declared and coded modules
"""
import types
import numpy as np
import numba as nb
import starsim as ss
from . import draws
from ..backends import compiling

compiling.depends(__file__, draws.__file__) # The uniforms come from the draws

jit = nb.jit(nopython=True, cache=True, error_model='numpy') # As ss.solvers.numba.compartmental compiles its kernels
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


def threads(drivers, n, width=1):
    """ The serial or threaded driver of a pass over n agents or edges (see `n_chunks()`), and the number of chunks to take it in """
    nchunks = n_chunks(n, width)
    return drivers[nchunks > 1], nchunks


@jit
def bounds(c, n, nchunks):
    """ The agents that chunk c of n chunks covers """
    size = (n + nchunks - 1)//nchunks
    a = min(c*size, n)
    return a, min(a + size, n)


@jit
def close_gaps(out, found, n, nchunks):
    """ Move the part of `out` that each chunk wrote (`found[c]` from its start) to follow the one before; the length of the list """
    p = found[0]
    for c in range(1, nchunks):
        a = bounds(c, n, nchunks)[0]
        for i in range(found[c]):
            out[p + i] = out[a + i]
        p += found[c]
    return p


#%% The two drivers that most passes share, made for each pass from its work on one chunk
# Each is a template, compiled for each pass as a copy that sees, as globals, the pass's work on one chunk (`chunk`), the loop over the chunks
# (`loop`: `range`, or `nb.prange` for the threaded driver) and the template's option. They are globals rather than a closure's variables
# because Numba keys its cache by a closure's variables, pickled, and a compiled function pickles differently in each session, so a closure
# over one would be compiled again in every session.

def specialize(template, chunk, **options):
    """ The serial and the threaded driver of a pass: the template (`count_fill()` or `in_place()`) for its work on one chunk, with the template's option, cached under the chunk's name """
    def make(threaded):
        names = dict(globals(), chunk=chunk, loop=nb.prange if threaded else range, **options)
        return types.FunctionType(template.__code__, names, f'{chunk.__name__}_{template.__name__}')
    return compile_drivers(make)


def count_fill(*args):
    """
    The template of a pass that counts what each chunk finds, then lists it in order, each chunk writing only what it found (see `specialize()`)

    Its arguments are the pass's inputs (the first being what is taken in chunks), then the number of chunks; `chunk(*inputs, a, b, out, pos, count)` gives
    the number found from a to b if `count`, else writes their indices into `out` from `pos`. With the option `none_or_all`, no list is made if none or all are
    found. Returns the number found, and the list (empty if none or all are found, with `none_or_all`).
    """
    nchunks = args[-1]
    inputs = args[:-1]
    n = len(args[0])
    found = np.zeros(nchunks + 1, dtype=np.int64)
    none = np.empty(0, dtype=np.int64)
    for c in loop(nchunks): # Counted first, so only what is found is written
        a, b = bounds(c, n, nchunks)
        found[c+1] = chunk(*inputs, a, b, none, 0, True)
    starts = np.cumsum(found)
    total = starts[-1]
    if none_or_all and (total == 0 or total == n): # None or all: no list is needed
        return total, none
    out = np.empty(total, dtype=np.int64)
    for c in loop(nchunks):
        a, b = bounds(c, n, nchunks)
        chunk(*inputs, a, b, out, starts[c], False)
    return total, out


def in_place(*args):
    """
    The template of a pass whose chunks each write what they find at the start of their own part of the output, without counting first (see `specialize()`)

    Its arguments are the pass's inputs (the first being what is taken in chunks), whose last is the output (or, with the option `outputs=2`, the last two are
    two outputs, written alike), then the number of chunks; `chunk(*inputs, a, b)` writes what it finds from a to b into the output from a, and gives how many.
    The parts are then joined. Returns the length of the list, at the start of the output (or of each).
    """
    nchunks = args[-1]
    inputs = args[:-1]
    n = len(args[0])
    found = np.zeros(nchunks, dtype=np.int64)
    for c in loop(nchunks):
        a, b = bounds(c, n, nchunks)
        found[c] = chunk(*inputs, a, b)
    if outputs == 2: # The option is a constant, so for one output this is not compiled
        close_gaps(args[-3], found, n, nchunks)
    return close_gaps(args[-2], found, n, nchunks)


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

census_drivers = compile_drivers(census_driver)

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
    func, nchunks = threads(census_drivers, len(alive), 4*(len(members) + 1)*n_groups)
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
def nonzero_chunk(raw, a, b, out, pos, count):
    """ The number of true values from a to b if `count`, else their indices, written from `pos` """
    if count:
        return true_count(raw, a, b)
    list_true(raw, a, b, out, pos)
    return 0

flatnonzero_drivers = specialize(count_fill, nonzero_chunk, none_or_all=False)

def flatnonzero(raw):
    """ The indices of the true values of an array, as `np.flatnonzero()`, threaded for a large one """
    func, nchunks = threads(flatnonzero_drivers, len(raw))
    if nchunks == 1:
        return np.flatnonzero(raw)
    return func(raw, nchunks)[1]



@jit
def exit_due_chunk(raw, exits, ti, a, b, hit):
    """ For agents a to b: whether each is in the state (`raw`) with its exit due by step ti or not set (NaN), without branches, so it is vectorized; returns how many are """
    rs = raw[a:b].view(np.uint8)
    ex = exits[a:b]
    hs = hit[a:b].view(np.uint8)
    for i in range(len(rs)):
        hs[i] = rs[i] & np.uint8(not ex[i] > ti)
    return true_count(hit, a, b)


def exit_due_driver(threaded):
    loop = nb.prange if threaded else range
    def exit_due(raw, exits, ti, nchunks):
        n = len(raw)
        hit = np.empty(n, dtype=np.bool_)
        found = np.zeros(nchunks + 1, dtype=np.int64)
        for c in loop(nchunks):
            a, b = bounds(c, n, nchunks)
            found[c+1] = exit_due_chunk(raw, exits, ti, a, b, hit)
        starts = np.cumsum(found)
        out = np.empty(starts[-1], dtype=np.int64)
        if starts[-1]:
            for c in loop(nchunks):
                if found[c+1]:
                    a, b = bounds(c, n, nchunks)
                    list_true(hit, a, b, out, starts[c])
        return starts[-1], out
    return exit_due

exit_due_drivers = compile_drivers(exit_due_driver)

def exit_due(raw, exits, ti):
    """ The agents in a state (`raw`) whose exit is due by step ti, or not set (NaN), as `np.flatnonzero(raw & ~(exits > ti))`, threaded for a large population """
    func, nchunks = threads(exit_due_drivers, len(raw))
    return func(raw, exits, ti, nchunks)[1]

@jit
def put_count(raw, uids, value, groups, ng):
    """ Put agents in a state (`value` True) or take them out (False), as `raw[uids] = value`, and how many of them that changed, in each of the `ng` groups (`groups` by UID, or empty for one group) """
    out = np.zeros(ng, dtype=np.int64)
    one = len(groups) == 0
    for u in uids:
        if raw[u] != value:
            raw[u] = value
            out[0 if one else groups[u]] += 1
    return out


@jit
def set_exits(uids, dur, entered, slots, shift, dt, which, ti_exit, exit):
    """
    Schedule the exits of agents who have entered a state: each at `entered` (one value for all, or one each) plus their dwell time `dur` as whole steps, rounded
    up or down by their phase (as `ss.People.phase()` and `whole_steps()`: `floor(dur/dt + phase)`), and by the exit `which`; `slots` by UID, or empty if they are the UIDs
    """
    each = len(entered) > 1
    by_uid = len(slots) == 0
    for j in range(len(uids)):
        u = uids[j]
        slot = u if by_uid else slots[u]
        x = slot*0.6180339887498949 + shift
        phase = x - np.floor(x) # Exactly x % 1 (as ss.People.phase() takes it), since x is not negative, without a call to fmod
        ti_exit[u] = entered[j if each else 0] + np.floor(dur[j]/dt + phase)
        exit[u] = which[j]
    return


@jit
def group_sums_chunk(codes, weights, groups, first, skip, a, b, out):
    """
    For agents a to b (a a multiple of 8): add each one's weight (or 1, if `weights` is empty) to the sum for their code and group, for the codes counted (see
    `group_sums()`); with `skip` (codes of one byte, counted from 1), eight at a time where all are 0
    """
    n_codes = out.shape[0]
    weighted = len(weights) > 0
    one = len(groups) == 0
    end = a + (b - a)//8*8 if skip else a
    if end > a:
        words = codes[a:end].view(np.uint64)
        for w in range(len(words)):
            if words[w]:
                for i in range(a + 8*w, a + 8*w + 8):
                    k = np.int64(codes[i])
                    if k >= first and k < n_codes:
                        out[k, 0 if one else np.int64(groups[i])] += weights[i] if weighted else 1.0
    for i in range(end, b):
        k = np.int64(codes[i])
        if k >= first and k < n_codes:
            out[k, 0 if one else np.int64(groups[i])] += weights[i] if weighted else 1.0
    return


def group_sums_driver(threaded):
    loop = nb.prange if threaded else range
    def group_sums(codes, weights, groups, first, skip, n_codes, ng, nblocks):
        n = len(codes)
        sums = np.zeros((nblocks, n_codes, ng))
        for c in loop(nblocks): # Blocks of a fixed size, a multiple of 8
            group_sums_chunk(codes, weights, groups, first, skip, c*chunk_size, min(n, (c + 1)*chunk_size), sums[c])
        out = np.zeros((n_codes, ng))
        for c in range(nblocks): # In order, so the sums do not depend on the number of threads
            out += sums[c]
        return out
    return group_sums

group_sums_drivers = compile_drivers(group_sums_driver)

def group_sums(codes, weights, groups, n_codes, ng, first=0):
    """
    The number of agents with each code (e.g. a state, 0 to n_codes - 1) in each group, or the sum of their weights: one pass over the agents, threaded for a large population

    The sums are taken in blocks of a fixed size (`chunk_size`), added in order, so they do not depend on the number of threads; with one block (up
    to `chunk_size` agents), they are in the order of the agents, as `np.bincount()` adds them.

    Args:
        codes (array): each agent's code, by UID (e.g. a boolean state, or a state as a small integer); those outside `first` to `n_codes - 1` are not counted
        weights (array): each agent's weight, by UID, or None to count them
        groups (array): each agent's group, by UID, or None for one group
        n_codes, ng (int): the number of codes and of groups
        first (int): the first code counted (e.g. 1, for those in a boolean state)

    Returns an array of floats, one row for each code and one column for each group.
    """
    n = len(codes)
    weights = np.empty(0) if weights is None else weights
    groups = np.empty(0, dtype=np.int64) if groups is None else groups
    func = threads(group_sums_drivers, n)[0]
    nblocks = max(1, -(-n//chunk_size))
    skip = codes.itemsize == 1 and first >= 1 # Eight at a time where none is counted
    return func(codes, weights, groups, first, skip, n_codes, ng, nblocks)


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

infect_drivers = compile_drivers(infect_driver)

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
    if slots is None:
        slots = np.empty(0, dtype=np.int32)
    func, nchunks = threads(infect_drivers, len(susceptible))
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

due_drivers = compile_drivers(due_driver)

def due(auids, ti_dead, ti_removed, ti):
    """ The active agents who have died or left by step ti (as `((ti_dead <= ti) | (ti_removed <= ti)).uids`), in one pass """
    func, nchunks = threads(due_drivers, len(auids))
    return func(auids, ti_dead, ti_removed, ti, nchunks)


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

keep_alive_drivers = compile_drivers(keep_driver)

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
    func, nchunks = threads(keep_alive_drivers, len(auids))
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

due_alive_drivers = compile_drivers(due_alive_driver)

def due_alive(alive, ti_dead, ti_removed, ti):
    """ The agents alive who have died or left by step ti (as `due()`, but by UID), and the number alive """
    func, nchunks = threads(due_alive_drivers, len(alive))
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

count_alive_drivers = compile_drivers(count_alive_driver)

def count_alive(states, alive):
    """ The number of agents alive in each of some states, then the number alive, in one pass by UID """
    func, nchunks = threads(count_alive_drivers, len(alive))
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

keep_raw_drivers = compile_drivers(keep_raw_driver)

def keep_raw(alive, age, dt, aging, kept, expected):
    """
    As `keep_alive()`, but by UID: list the agents alive (in order), and age them, in one pass, if there are as many as expected

    Returns the number alive, or -1 (having changed nothing) if that is not the number expected.
    """
    func, nchunks = threads(keep_raw_drivers, len(alive))
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

random_edges_drivers = compile_drivers(edges_driver)

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
    func, nchunks = threads(random_edges_drivers, len(source))
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
        return out0[:close_gaps(out0, found0, n, nchunks)], out1[:close_gaps(out1, found1, n, nchunks)]
    return candidates

candidates_drivers = compile_drivers(candidates_driver)

index_frac = 0.25 # The largest fraction of the edges to visit by the edges of each agent rather than by a pass over all of them

def candidates(p1, p2, rel_trans, rel_sus, two_way=True, index=None):
    """
    The indices of the edges from someone with `rel_trans` not 0 to someone with `rel_sus` not 0 (by UID), p1 to p2 and (if `two_way`) p2 to p1, in order:
    the edges that might transmit, found for both directions in one pass over the edges (a superset of those that can, which `can_transmit()` then finds among them),
    or, with the edges of each agent (`index`, from `edge_index()`), from the edges of the people infectious or of those susceptible, if they have few
    """
    if index is not None:
        done, out0, out1 = candidates_indexed(rel_trans, rel_sus, two_way, *index, int(index_frac*len(p1)))
        if done:
            return (out0, out1) if two_way else (out0,)
    flags = np.empty(len(rel_trans) if 8*len(p1) > len(rel_trans) else 0, dtype=np.uint8) # A byte for each agent, which the edges read 5-10 times more quickly than two numbers each, unless there are few edges
    func, nchunks = threads(candidates_drivers, max(len(p1), len(flags)))
    out1 = np.empty(len(p1) if two_way else 0, dtype=np.int64)
    found = func(p1, p2, rel_trans, rel_sus, two_way, flags, np.empty(len(p1), dtype=np.int64), out1, nchunks)
    return found if two_way else found[:1]


#%% The edges of each agent (ss.Network.edge_index()), for passes that visit only some agents' edges

@jit
def edge_index(p1, p2, n):
    """
    The edges of each of `n` agents, by the end they are at: agent u is p1 of the edges `e1[s1[u]:s1[u+1]]`, in order, whose p2 are
    `o1[s1[u]:s1[u+1]]`, and p2 of the edges `e2[s2[u]:s2[u+1]]`, whose p1 are `o2[...]` (a counting sort of the edges by each end)
    """
    m = len(p1)
    s1 = np.zeros(n + 1, dtype=np.int64)
    s2 = np.zeros(n + 1, dtype=np.int64)
    for i in range(m):
        s1[p1[i] + 1] += 1
        s2[p2[i] + 1] += 1
    for u in range(n):
        s1[u + 1] += s1[u]
        s2[u + 1] += s2[u]
    e1 = np.empty(m, dtype=np.int64)
    o1 = np.empty(m, dtype=np.int64)
    e2 = np.empty(m, dtype=np.int64)
    o2 = np.empty(m, dtype=np.int64)
    k1 = s1[:-1].copy() # The next place of each agent
    k2 = s2[:-1].copy()
    for i in range(m):
        u = p1[i]
        v = p2[i]
        e1[k1[u]] = i
        o1[k1[u]] = v
        k1[u] += 1
        e2[k2[v]] = i
        o2[k2[v]] = u
        k2[v] += 1
    return s1, e1, o1, s2, e2, o2


debruijn = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,
                     63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], dtype=np.int64) # The position of the bit in w & -w, by its product with 0x03f79d71b4cb0a89

@jit
def in_order(found, n):
    """ The distinct numbers in `found`, each from 0 to n - 1, in order: by a bit for each (so in about n/64 + len(found) steps, rather than a sort's len(found)·log(len(found))) """
    bits = np.zeros((n + 63)//64, dtype=np.uint64)
    for x in found:
        bits[x >> 6] |= np.uint64(1) << np.uint64(x & 63)
    out = np.empty(len(found), dtype=np.int64)
    m = 0
    for i in range(len(bits)):
        w = bits[i]
        while w:
            low = w & (~w + np.uint64(1)) # The lowest bit set
            out[m] = (i << 6) + debruijn[(low*np.uint64(0x03f79d71b4cb0a89)) >> np.uint64(58)]
            m += 1
            w ^= low
    return out


@jit
def candidates_indexed(rel_trans, rel_sus, two_way, s1, e1, o1, s2, e2, o2, most):
    """
    As `candidates()`, from the edges of each agent (`edge_index()`): those of the people infectious, or of those susceptible, whichever have
    fewer, then put in order. Whether it was done (not if it would visit more than `most` edges, when a pass over all of them is quicker), and the two lists
    """
    flags = np.empty(len(rel_trans), dtype=np.uint8)
    flags_chunk(rel_trans, rel_sus, 0, len(rel_trans), flags)
    n = min(len(s1) - 1, len(rel_trans)) # Agents born since the index have no edges in it, and none in the network, which has not changed
    by_src = 0 # The edges visited from the side of the infectious, and of the susceptible
    by_trg = 0
    for u in range(n):
        d1 = s1[u + 1] - s1[u]
        d2 = s2[u + 1] - s2[u]
        if flags[u] & 1:
            by_src += d1 + (d2 if two_way else 0)
        if flags[u] & 2:
            by_trg += d2 + (d1 if two_way else 0)
    none = np.empty(0, dtype=np.int64)
    k = min(by_src, by_trg)
    if k > most:
        return False, none, none
    from_src = by_src <= by_trg
    have = 1 if from_src else 2 # The flag of the agents whose edges are visited, and the one wanted at the other end
    want = 2 if from_src else 1
    sa, ea, oa = (s1, e1, o1) if from_src else (s2, e2, o2) # p1 to p2: the edges of their p1, or of their p2
    sb, eb, ob = (s2, e2, o2) if from_src else (s1, e1, o1) # p2 to p1
    out0 = np.empty(k, dtype=np.int64)
    out1 = np.empty(k if two_way else 0, dtype=np.int64)
    m0 = 0
    m1 = 0
    for u in range(n):
        if flags[u] & have:
            for j in range(sa[u], sa[u + 1]):
                out0[m0] = ea[j] # Branchless, as candidates_chunk()
                m0 += (flags[oa[j]] & want) != 0
            if two_way:
                for j in range(sb[u], sb[u + 1]):
                    out1[m1] = eb[j]
                    m1 += (flags[ob[j]] & want) != 0
    n_edges = len(e1)
    return True, in_order(out0[:m0], n_edges), in_order(out1[:m1], n_edges) # In the order of the edges, as a pass over them finds them


@jit
def partners(inds, s1, o1, s2, o2, n):
    """ The partners of the agents `inds` (by UID; `n` covers every UID), in order, from the edges of each agent (`edge_index()`); as `ss.networks.find_contacts()` """
    is_partner = np.zeros(n, dtype=np.bool_)
    na = len(s1) - 1
    for u in inds:
        if u < na: # Someone born since the index has no edges
            for j in range(s1[u], s1[u + 1]):
                is_partner[o1[j]] = True
            for j in range(s2[u], s2[u + 1]):
                is_partner[o2[j]] = True
    return np.flatnonzero(is_partner)


@jit
def can_chunk(src, trg, rel_trans, rel_sus, beta, out, a, b):
    """ For edges a to b: the indices of those with a chance of transmission above 0, written from a, and how many there are """
    s = src[a:b] # Slices, so the loop's indices are known not to be negative
    t = trg[a:b]
    bs = beta[a:b] if len(beta) == len(src) else beta
    step = 1 if len(bs) == len(s) else 0
    m = 0
    for i in range(len(s)):
        hit = rel_trans[s[i]]*rel_sus[t[i]]*bs[i*step] > 0 # As nb_transmit(), with no fastmath, so a NaN is no transmission
        out[a + m] = a + i # Written every time (branchless) and kept only if m advances
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


can_transmit_drivers = specialize(in_place, can_chunk, outputs=1) # Without counting first (branchless, since the test is often true)
ending_drivers = specialize(count_fill, ending_chunk, none_or_all=True)
touching_drivers = specialize(count_fill, touching_chunk, none_or_all=True)


def can_transmit(src, trg, rel_trans, rel_sus, beta):
    """ The indices of the edges with a chance of transmission above 0 (`rel_trans[src]·rel_sus[trg]·beta`, with `beta` one value for each edge or one for all), in order """
    out = np.empty(len(src), dtype=np.int64)
    func, nchunks = threads(can_transmit_drivers, len(src))
    m = func(src, trg, rel_trans, rel_sus, np.atleast_1d(beta), out, nchunks)
    return out[:m]


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
def women_chunk(auids, female, age, lo, hi, out, a, b):
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


women_drivers = specialize(in_place, women_chunk, outputs=1)

def women(auids, female, age, lo, hi):
    """ The UIDs among `auids` of the women (`female`, by UID) aged at least `lo` and under `hi` (`age`, by UID), in order; compared as NumPy would """
    dtype = np.result_type(age, lo, hi) # E.g. float32 for float32 ages and Python numbers
    out = np.empty(len(auids), dtype=auids.dtype)
    func, nchunks = threads(women_drivers, len(auids))
    m = func(auids, female, age, dtype.type(lo), dtype.type(hi), out, nchunks)
    return out[:m]


#%% Debut (ss.SexualNet)

@jit
def debut_chunk(auids, alive, decided, age, debut, reached, out, a, b):
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


debuts_drivers = specialize(in_place, debut_chunk, outputs=1)

def debuts(auids, alive, decided, age, debut, reached):
    """ The UIDs among `auids` (in order) of those alive with no debut age yet, or (if `reached`) alive, undecided, and at least their debut age (the arrays by UID) """
    out = np.empty(len(auids), dtype=auids.dtype)
    func, nchunks = threads(debuts_drivers, len(auids))
    m = func(auids, alive, decided, age, debut, reached, out, nchunks)
    return out[:m]


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
def exits_chunk(uids, slots, seed, ind, shift, scale, p_exit, edges, pos, which, a, b):
    """
    For agents a to b of `uids`: those who leave a state (their uniform for the call, from `seed` and `ind`, is below `p_exit`), written from a in
    `pos` (their position in `uids`) and `which` (the exit they take: the first i with uniform < edges[i+1], or -1 if none); how many leave
    """
    key = draws.key(seed, ind)
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


competing_exits_drivers = specialize(in_place, exits_chunk, outputs=2)

def competing_exits(uids, slots, seed, ind, dtype, p_exit, edges):
    """
    Who leaves a state by competing rates that are the same for everyone, in one pass that draws each agent's uniform for the call (as
    `hash_uniforms()` gives it: see `ss.Dist.crn_key()`): the positions in `uids` of those who leave, and the exit each takes (by `edges`,
    the cumulative chances of the exits, from 0; -1 for none, where rounding leaves a gap below `p_exit`), in order
    """
    uids = np.asarray(uids, dtype=np.int64)
    pos = np.empty(len(uids), dtype=np.int64) # Only those who leave are written
    which = np.empty(len(uids), dtype=np.int64)
    func, nchunks = threads(competing_exits_drivers, len(uids))
    m = func(uids, np.empty(0, dtype=np.int64) if slots is None else slots, seed, ind, *draws.crn_bits(dtype), float(p_exit), np.asarray(edges, dtype=float), pos, which, nchunks)
    return pos[:m], which[:m]


@jit
def agent_exits_chunk(uids, slots, seed, ind, shift, scale, p_exit, frac, hazards, pos, which, a, b):
    """
    As `exits_chunk()`, with chances that differ between agents: agent i leaves if their uniform is below `p_exit[i]`, by the first
    exit t whose edge, the hazards of exits 0 to t added in order, times `frac[i]`, is above it (the same numbers as NumPy's cumsum)
    """
    key = draws.key(seed, ind)
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
                if u < edge*frac[j]:
                    break
                t += 1
            ps[m] = j
            ws[m] = t if t < n_exits else -1
            m += 1
    return m


competing_agent_exits_drivers = specialize(in_place, agent_exits_chunk, outputs=2)

def competing_agent_exits(uids, slots, seed, ind, dtype, p_exit, frac, hazards):
    """
    As `competing_exits()`, with chances that differ between agents: each agent's chance of leaving (`p_exit`), and the exits' hazards
    (one row for each exit, and a column for each agent), whose running total times `frac` (`p_exit` over the total hazard) are the edges
    """
    uids = np.asarray(uids, dtype=np.int64)
    pos = np.empty(len(uids), dtype=np.int64)
    which = np.empty(len(uids), dtype=np.int64)
    func, nchunks = threads(competing_agent_exits_drivers, len(uids))
    m = func(uids, np.empty(0, dtype=np.int64) if slots is None else slots, seed, ind, *draws.crn_bits(dtype), np.ascontiguousarray(p_exit, dtype=float), np.ascontiguousarray(frac, dtype=float), np.ascontiguousarray(hazards, dtype=float), pos, which, nchunks)
    return pos[:m], which[:m]
