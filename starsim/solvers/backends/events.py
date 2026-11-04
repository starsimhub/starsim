"""
Not for the user: the compiled pass of agents taken event by event in continuous time (`solver='abm-events'`), one call a step: see `ss.solvers.Events`

Each agent has a state, the time and transition of its scheduled exit (a dwell time drawn on entry, kept across steps), and a
version, raised at each move, so that what was queued for it before is skipped. Within the step, events come off one heap in
time order: an agent's own exit (the earliest of its scheduled exit and an exponential of its competing rates), or a contact
of an infectious agent on a route (to a neighbor on a network, a member of a pool, or one of a cell of a contact matrix, the
cell drawn by its weight), accepted with the share its real hazard is of the bound its clock ran at (thinning).

Its Rust copy (`starsim/solvers/rust/src/events.rs`) gives the same events and numbers, bit for bit, with no compiling; this is the reference.

Used with: agents; declared modules
"""
import numpy as np
import numba as nb

__all__ = ['step_events']

jit = nb.jit(nopython=True, cache=True, error_model='numpy')


@jit
def grow(ht, ha, hk, hv, hr, size, room):
    """ The heap's arrays, made larger, with room for `room` more entries """
    m = 2*(size + room)
    nt = np.empty(m); na = np.empty(m, np.int64); nk = np.empty(m, np.int64); nv = np.empty(m, np.int64); nr = np.empty(m, np.int64)
    nt[:size] = ht[:size]; na[:size] = ha[:size]; nk[:size] = hk[:size]; nv[:size] = hv[:size]; nr[:size] = hr[:size]
    return nt, na, nk, nv, nr


@jit
def push(ht, ha, hk, hv, hr, size, t, a, k, v, r):
    """ Push an entry onto the heap (time, agent, kind, version, transition), which has room for it (see grow()); returns the new size """
    i = size
    while i > 0:
        p = (i - 1) >> 1
        if ht[p] <= t:
            break
        ht[i] = ht[p]; ha[i] = ha[p]; hk[i] = hk[p]; hv[i] = hv[p]; hr[i] = hr[p]
        i = p
    ht[i] = t; ha[i] = a; hk[i] = k; hv[i] = v; hr[i] = r
    return size + 1


@jit
def pop(ht, ha, hk, hv, hr, size):
    """ Take the earliest entry off the heap (its values are read at index 0 before the call); returns the new size """
    size -= 1
    t, a, k, v, r = ht[size], ha[size], hk[size], hv[size], hr[size]
    i = 0
    while True:
        c = 2*i + 1
        if c >= size:
            break
        if c + 1 < size and ht[c + 1] < ht[c]:
            c += 1
        if ht[c] >= t:
            break
        ht[i] = ht[c]; ha[i] = ha[c]; hk[i] = hk[c]; hv[i] = hv[c]; hr[i] = hr[c]
        i = c
    ht[i] = t; ha[i] = a; hk[i] = k; hv[i] = v; hr[i] = r
    return size


@jit
def rand(key, a, ctr):
    """
    Agent `a`'s next uniform in (0, 1) of the step: a hash (SplitMix64's) of the step's key, the agent, and its count of draws, so each
    agent's numbers depend only on its own draws, not on the order of everyone's events (and are the same in any language)
    """
    z = key + np.uint64(a)*np.uint64(0x9E3779B97F4A7C15) + np.uint64(ctr[a])*np.uint64(0xD1B54A32D192ED03)
    ctr[a] += 1
    z = (z ^ (z >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (float(z >> np.uint64(11)) + 0.5)/9007199254740992.0 # The top 53 bits, never 0 or 1


@jit
def dwell(tr, a, key, ctr, dw_kind, dw_k, dw_mean, dw_row, pool):
    """ A dwell time of agent `a` for a transition: none (0), Erlang with k stages and a fixed mean (1), or the one drawn for the agent for this step (2) """
    kind = dw_kind[tr]
    if kind == 0:
        return 0.0
    if kind == 1:
        k = dw_k[tr]
        tot = 0.0
        for i in range(k): # Erlang: the sum of k exponentials
            tot -= np.log(rand(key, a, ctr))
        return tot*dw_mean[tr]/k
    return pool[dw_row[tr], a] # Drawn again for each step, so the same only for an agent who enters the state twice in one step


@jit
def draw_exit(a, s, t, key, ctr, br_ptr, br_tr, br_cum, tm_ptr, tm_tr, dw_kind, dw_k, dw_mean, dw_row, pool):
    """ The scheduled exit of agent `a` entering state `s` at time `t`: its branch, chosen on entry, and the earliest of its timed exits; (inf, -1) for none """
    te = np.inf
    tx = -1
    if br_ptr[s] < br_ptr[s+1]:
        u = rand(key, a, ctr)
        j = br_ptr[s]
        while j < br_ptr[s+1] - 1 and u >= br_cum[j]:
            j += 1
        tx = br_tr[j]
        te = t + dwell(tx, a, key, ctr, dw_kind, dw_k, dw_mean, dw_row, pool)
    for j in range(tm_ptr[s], tm_ptr[s+1]):
        d = t + dwell(tm_tr[j], a, key, ctr, dw_kind, dw_k, dw_mean, dw_row, pool)
        if d < te or tx < 0:
            te = d
            tx = tm_tr[j]
    return te, tx


@jit
def queue(a, s, t, t1, key, ctr, ht, ha, hk, hv, hr, size, ver, t_exit, exit_tr, haz_ptr, haz_tr, haz_rate, haz_row, rate_rows,
          r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc):
    """
    Queue what an agent in state `s` does next from time `t`: its own exit (the earliest of the scheduled one and an exponential of
    the competing rates, the exit chosen in proportion to them), if within the step, and its first contact on each route it is a source on;
    the heap has room for one entry and one for each route (see grow()). Returns the heap's new size
    """
    tot = 0.0
    for j in range(haz_ptr[s], haz_ptr[s+1]):
        tot += haz_rate[j] if haz_row[j] < 0 else rate_rows[haz_row[j], a]
    tn = t_exit[a]
    trn = exit_tr[a]
    if tot > 0:
        th = t - np.log(rand(key, a, ctr))/tot
        if th < tn:
            tn = th
            u = rand(key, a, ctr)*tot
            acc = 0.0
            for j in range(haz_ptr[s], haz_ptr[s+1]):
                acc += haz_rate[j] if haz_row[j] < 0 else rate_rows[haz_row[j], a]
                trn = haz_tr[j]
                if u < acc:
                    break
    if tn < t1 and trn >= 0:
        size = push(ht, ha, hk, hv, hr, size, tn, a, -1, ver[a], trn)
    for r in range(len(r_kind)):
        rate = contact_rate(a, s, r, r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc)
        if rate > 0:
            size = push(ht, ha, hk, hv, hr, size, t - np.log(rand(key, a, ctr))/rate, a, r, ver[a], -1)
    return size


@jit
def degree(a, r, x_off, r_n, r_deg):
    """ The number of neighbors of agent `a` on network `r` (as p1 of its edges, and as p2 if the network is two-way), worked out for the step """
    return r_deg[x_off[r] + a] if a < r_n[r] else 0 # The index covers the agents up to the largest on an edge


@jit
def neighbor(a, j, r, x_off, beta_off, s1, e1, o1, s2, e2, o2, w1, w2):
    """ The `j`th neighbor of agent `a` on network `r`, and the weight of the edge from `a` to them: its edges as p1 first (weights w1), then as p2 (w2) """
    i = x_off[r] + a
    d1 = s1[i+1] - s1[i]
    if j < d1:
        m = beta_off[r] + s1[i] + j
        return o1[m], w1[beta_off[r] + e1[m]]
    m = beta_off[r] + s2[i] + j - d1
    return o2[m], w2[beta_off[r] + e2[m]]


@jit
def contact_rate(a, s, r, r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc):
    """ The rate of the contacts of an agent in state `s` on route `r`, at the bound that each is then thinned from: 0 if it is not a source there """
    sh = share[r, s]*w_agent[a]
    if sh <= 0:
        return 0.0
    deg = degree(a, r, x_off, r_n, r_deg) if r_kind[r] == 0 else a_ctc[r, a]
    return r_scale[r]*sh*deg*r_bound[r]


@jit
def step_events(t0, t1, seed, state, ver, t_exit, exit_tr, fresh,
                haz_ptr, haz_tr, haz_rate, haz_row, rate_rows,
                br_ptr, br_tr, br_cum, tm_ptr, tm_tr,
                dw_kind, dw_k, dw_mean, dw_row, pool,
                dst_ptr, dst_state, dst_cum, inf_tr,
                sus_state, sus_agent, w_agent,
                r_kind, x_off, r_n, r_scale, r_bound, share,
                s1, e1, o1, s2, e2, o2, r_deg, w1, w2, beta_off,
                members, mem_eff, mem_off, mem_n, a_ctc,
                r_blk, a_cell, c_ptr, c_first, c_row, c_cum):
    """
    Advance the agents of one module through a step, [t0, t1), event by event

    Args:
        t0, t1 (float): the step, in the sim's unit; seed (int): the step's seed
        state (int array): each agent's state (-1 for none: not alive, or not in this module's people); changed in place, as are ver, t_exit and exit_tr
        ver (int array): each agent's version, raised at each move
        t_exit, exit_tr: each agent's scheduled exit (inf and -1 for none)
        fresh (bool array): the agents who entered their state at the step's start (by code, an intervention, or at the start), whose exits are drawn now
        haz_ptr, haz_tr, haz_rate, haz_row, rate_rows: the competing rates out of each state: their transitions, and a rate for everyone, or (row >= 0) a row of rate_rows for each agent
        br_ptr, br_tr, br_cum: the branches chosen on entry into each state, with their cumulative chances; tm_ptr, tm_tr: the timed exits of each state (the earliest is taken)
        dw_*, pool: each transition's dwell time, and those drawn for each agent for the step (see dwell())
        dst_ptr, dst_state, dst_cum: each transition's destinations (-1 for a death), with their cumulative chances
        inf_tr (int): the transmission (-1 for none)
        sus_state, sus_agent: susceptibility: by state (0 for none) and by agent; w_agent: infectiousness by agent
        r_kind (0 network, 1 pool, 2 contact matrix), x_off, r_n (each network's place in s1, s2 and r_deg, and its number of agents; each matrix's place in a_cell), r_scale (the coefficient per unit of share), r_bound (the bound that each contact is thinned from), share (route x state): the routes
        s1, e1, o1, s2, e2, o2, r_deg, w1, w2, beta_off: the edges of each agent on each network (`ss.Network.edge_index()`), concatenated, with each agent's number of neighbors (r_deg), and the weight of each edge from p1 to p2 (w1) and from p2 to p1 (w2: the same array unless the weights differ by direction, as on a network with acts), each network's edges from beta_off
        members, mem_eff, mem_off, mem_n: the pools: who can be reached, and the contacts of each in that pool; where they start, how many
        a_ctc (route x agent): on each pool or contact matrix, each agent's contacts per unit of the coefficient (0 if not a source there)
        r_blk, a_cell, c_ptr, c_first, c_row, c_cum: the contact matrices, by cells (a group, or a group in a place): each matrix's number of
            cells a contact can reach (r_blk), the cell of each agent on it (-1 for none, from its place in a_cell, x_off), and, for each cell,
            where its members are in `members` (c_ptr, one more than the cells), the first of the cells its contacts can reach (c_first), and
            where its cumulative weights of reaching each of them start in c_cum (c_row)

    Returns:
        The events, as (agent, transition, from, to, time, source, route), the source and route being the infector of an infection and the
        route it was over (-1 otherwise); and whether any contact's chance was above its bound
    """
    n = len(state)
    key = np.uint64(seed)
    ctr = np.zeros(n, np.int64) # Each agent's draws so far in the step
    cap = 1024
    for a in range(n):
        if state[a] >= 0:
            cap += 1 + len(r_kind)
    ht = np.empty(cap); ha = np.empty(cap, np.int64); hk = np.empty(cap, np.int64); hv = np.empty(cap, np.int64); hr = np.empty(cap, np.int64)
    size = 0
    ns = len(haz_ptr) - 1
    busy = np.zeros(ns, dtype=np.bool_) # The states with competing rates or contacts, whose agents have something to draw at each step
    for s in range(ns):
        busy[s] = haz_ptr[s+1] > haz_ptr[s]
        for r in range(len(r_kind)):
            busy[s] |= share[r, s] > 0
    for a in range(n): # Everyone's next exit and first contacts; the competing rates and contacts are drawn again each step, since they are memoryless
        s = state[a]
        if s < 0:
            continue
        if fresh[a]:
            ver[a] += 1
            t_exit[a], exit_tr[a] = draw_exit(a, s, t0, key, ctr, br_ptr, br_tr, br_cum, tm_ptr, tm_tr, dw_kind, dw_k, dw_mean, dw_row, pool)
        if not busy[s] and t_exit[a] >= t1:
            continue
        if size + 1 + len(r_kind) > len(ht):
            ht, ha, hk, hv, hr = grow(ht, ha, hk, hv, hr, size, 1 + len(r_kind))
        size = queue(a, s, t0, t1, key, ctr, ht, ha, hk, hv, hr, size, ver, t_exit, exit_tr, haz_ptr, haz_tr, haz_rate, haz_row, rate_rows,
                                         r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc)
    ecap = 1024
    e_a = np.empty(ecap, np.int64); e_tr = np.empty(ecap, np.int64); e_src = np.empty(ecap, np.int64); e_dst = np.empty(ecap, np.int64); e_t = np.empty(ecap); e_by = np.empty(ecap, np.int64); e_rt = np.empty(ecap, np.int64)
    ne = 0
    over = False # Whether a contact's chance was above the bound its clock ran at, which would make the thinning wrong
    while size > 0:
        t = ht[0]
        if t >= t1:
            break
        a, k, v, trn = ha[0], hk[0], hv[0], hr[0]
        size = pop(ht, ha, hk, hv, hr, size)
        if v != ver[a]: # It has moved since this was queued
            continue
        s = state[a]
        b = a # Who moves now, by transition trn
        by = -1
        if k >= 0: # A contact on route k: of whom, and whether it infects them
            if r_kind[k] == 0:
                deg = degree(a, k, x_off, r_n, r_deg)
                b, x = neighbor(a, min(int(rand(key, a, ctr)*deg), deg - 1), k, x_off, beta_off, s1, e1, o1, s2, e2, o2, w1, w2)
            elif r_kind[k] == 1:
                deg = mem_n[k]
                m = mem_off[k] + min(int(rand(key, a, ctr)*deg), deg - 1)
                b = members[m]
                x = mem_eff[m] # Their contacts in this pool, which bound its thinning
            else: # A contact matrix: a cell drawn by its weight, then one of its members
                c = a_cell[x_off[k] + a]
                row = c_row[c]
                nb = r_blk[k]
                u = rand(key, a, ctr)*c_cum[row + nb - 1]
                i = 0
                while i < nb - 1 and u >= c_cum[row + i]:
                    i += 1
                g = c_first[c] + i
                deg = c_ptr[g+1] - c_ptr[g]
                m = c_ptr[g] + min(int(rand(key, a, ctr)*deg), deg - 1)
                b = members[m] if deg > 0 else a # A cell drawn has members, since its weight is their number
                x = 1.0
            sb = state[b]
            p = 0.0 if sb < 0 else sus_state[sb]*sus_agent[b]*x/r_bound[k]
            rate = contact_rate(a, s, k, r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc) # The next contact of the same agent
            if size + 1 > len(ht):
                ht, ha, hk, hv, hr = grow(ht, ha, hk, hv, hr, size, 1)
            size = push(ht, ha, hk, hv, hr, size, t - np.log(rand(key, a, ctr))/rate, a, k, ver[a], -1)
            if p > 1:
                over = True
            u = rand(key, a, ctr) # Drawn whether or not the contact can infect, so the agent's later numbers do not depend on its contacts' states
            if not (p > 0 and u < p):
                continue
            trn = inf_tr
            by = a

        # The move, then the exits and contacts of the state it enters
        u = rand(key, b, ctr)
        j = dst_ptr[trn]
        while j < dst_ptr[trn+1] - 1 and u >= dst_cum[j]:
            j += 1
        d = dst_state[j]
        if ne == len(e_a):
            m = 2*ne
            e_a2 = np.empty(m, np.int64); e_tr2 = np.empty(m, np.int64); e_src2 = np.empty(m, np.int64); e_dst2 = np.empty(m, np.int64); e_t2 = np.empty(m); e_by2 = np.empty(m, np.int64); e_rt2 = np.empty(m, np.int64)
            e_a2[:ne] = e_a; e_tr2[:ne] = e_tr; e_src2[:ne] = e_src; e_dst2[:ne] = e_dst; e_t2[:ne] = e_t; e_by2[:ne] = e_by; e_rt2[:ne] = e_rt
            e_a, e_tr, e_src, e_dst, e_t, e_by, e_rt = e_a2, e_tr2, e_src2, e_dst2, e_t2, e_by2, e_rt2
        e_a[ne] = b; e_tr[ne] = trn; e_src[ne] = state[b]; e_dst[ne] = d; e_t[ne] = t; e_by[ne] = by; e_rt[ne] = k if by >= 0 else -1
        ne += 1
        state[b] = d
        ver[b] += 1
        if d < 0: # Died
            t_exit[b] = np.inf
            exit_tr[b] = -1
            continue
        t_exit[b], exit_tr[b] = draw_exit(b, d, t, key, ctr, br_ptr, br_tr, br_cum, tm_ptr, tm_tr, dw_kind, dw_k, dw_mean, dw_row, pool)
        if size + 1 + len(r_kind) > len(ht):
            ht, ha, hk, hv, hr = grow(ht, ha, hk, hv, hr, size, 1 + len(r_kind))
        size = queue(b, d, t, t1, key, ctr, ht, ha, hk, hv, hr, size, ver, t_exit, exit_tr, haz_ptr, haz_tr, haz_rate, haz_row, rate_rows,
                                         r_kind, x_off, r_n, r_scale, r_bound, share, w_agent, r_deg, a_ctc)
    return e_a[:ne], e_tr[:ne], e_src[:ne], e_dst[:ne], e_t[:ne], e_by[:ne], e_rt[:ne], over
