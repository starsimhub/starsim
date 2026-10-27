"""
The rounds of moves of a declared disease as agents, compiled: the same-step cascade of `ss.solvers.Agents.finish_moves()`
(and the scheduled exits due at the start of a step, of `Agents.leave()`), for the diseases whose transitions it can
lower to tables (see `Agents.lower()`); not for the user

A round takes the agents who left a state, one transition at a time in the order they first left by it: a transition with
several destinations takes one draw each (`ss.stay` puts them back, with the exit they had), and each arrival is dated, and
has its branch, dwell time, and exit drawn if the state schedules them. Then those who move on at once leave for the next
round: those a trigger moves now (an event of this disease, after a delay drawn for each person), ahead of those whose exit
is due already. The draws are Starsim's counter-based common random numbers, keyed by the stream, the step and the round
(see `Agents.draw()`), so they are the same as the Python engine's.

Values that differ between people (a probability by sex, a dwell time by sex) come in tables by class of agent (`cls`),
worked out by the Python engine for one agent of each class (see `Agents.lower()`).
"""
import numpy as np
import numba as nb
from . import draws, cache as _cache

_cache.depends(__file__, draws.__file__) # The uniforms and inverses come from the draws

jit = nb.jit(nopython=True, cache=True, error_model='numpy') # As the passes over agents

# The kinds of dwell time (and of delay before a triggered move)
CONSTANT = 0 # pars[0]
LOGNORMAL = 1 # exp(pars[0]*ndtri(u))*pars[1] + pars[2], with ndtri at single precision (as SciPy's for the uniforms, which are)
EXPON = 2 # -log1p(-u)*pars[0] + pars[1] (as ss.expon)

STAY = -2 # The destination of someone who stays (ss.stay)
MAX_STATES = 32 # The arrays of each state are passed in tuples of 8, 16, or this many, so the kernel is compiled for a few lengths only, whatever the number of states
MAX_TRIGGERS = 16 # Likewise for the delays of the triggers: 4, 8, or this many


@jit
def uniform(seed, ind, slot, shift, scale):
    """ The uniform of an agent's slot from a stream (as `ss.distributions.hash_uniforms()`) """
    return draws.crn_uniform(draws.key(seed, ind), slot, shift, scale)


@jit
def dwell(kind, pars, u):
    """ A dwell time (or delay) of one of the kinds above from its uniform """
    if kind == CONSTANT:
        return pars[0]
    elif kind == LOGNORMAL:
        x = np.float64(np.float32(draws.ndtri(u)))
        return np.exp(pars[0]*x)*pars[1] + pars[2]
    else:
        return draws.expon(pars[0], u) + pars[1]


@jit
def to_step(val, lowest, never, nan):
    """ A step number as stored (see `ss.arrays.StepArr.from_float()`): the next whole step, within the integers, NaN as not set """
    if val != val:
        return nan
    v = np.ceil(val)
    if v < lowest:
        return lowest
    if v > never:
        return never
    return v


@jit
def grow(arr, n):
    """ The array, larger if it is full at n """
    if n < arr.shape[0]:
        return arr
    out = np.empty(max(2*arr.shape[0], 16), dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    return out


@jit
def first_order(keys, n, rank, out):
    """ Put in `out` the order of n items grouped by key, in the order each key first appears, and in their own order within a group (`rank`, one entry per key, is all -1, and is left so) """
    nr = 0
    for i in range(n):
        if rank[keys[i]] < 0:
            rank[keys[i]] = nr
            nr += 1
    counts = np.zeros(nr + 1, dtype=np.int64)
    for i in range(n):
        counts[rank[keys[i]] + 1] += 1
    for r in range(nr):
        counts[r + 1] += counts[r]
    for i in range(n):
        r = rank[keys[i]]
        out[counts[r]] = i
        counts[r] += 1
    for i in range(n):
        rank[keys[i]] = -1
    return


@jit
def whole(dur, dt, seed, ind, slot, shift, scale):
    """ A dwell time as a whole number of steps, rounded up or down at random keeping its mean (as `Agents.whole_steps()`) """
    steps = dur/dt
    w = np.floor(steps)
    if np.isfinite(steps) and steps != w:
        if uniform(seed, ind, slot, shift, scale) < steps - w:
            w += 1.0
    return w


@jit
def schedule(uid, s, cls, slot, ti, ind, dt, ex_off, ex_tr, ex_after, ex_loop, ex_resc, n_br, br_edges, br_off, dw_kind, dw_pars, adj,
             seeds, br_stream, round_stream, dw_stream, shift, scale):
    """
    For an agent who has just entered state s: choose their branch, draw its dwell time and the timed exits', and schedule the earliest (as
    `Agents.schedule()`); returns the step it is due, its index, and how fast it runs (the multiplier on its rate, if it is rescaled when that changes; else 1)
    """
    c = cls[uid]
    best = np.inf
    which = -1
    nb_ = n_br[s]
    e0 = ex_off[s]
    if nb_ > 0: # Split on entry
        u = uniform(seeds[br_stream[s]], ind, slot, shift, scale)
        base = br_off[s] + c*(nb_ + 1)
        for i in range(nb_):
            if u >= br_edges[base + i] and u < br_edges[base + i + 1]:
                t = ex_tr[e0 + i]
                if dw_kind[t] >= 0:
                    best = dwell(dw_kind[t], dw_pars[t, c], uniform(seeds[dw_stream[t]], ind, slot, shift, scale))/adj[t, c] + ex_after[e0 + i]
                else: # Nobody waits in this branch, beyond a minimum stay
                    best = ex_after[e0 + i] if ex_after[e0 + i] != 0 else -np.inf
                which = i
    for i in range(nb_, ex_off[s+1] - e0): # Competing dwell times: the earliest wins
        t = ex_tr[e0 + i]
        dur = dwell(dw_kind[t], dw_pars[t, c], uniform(seeds[dw_stream[t]], ind, slot, shift, scale))/adj[t, c] + ex_after[e0 + i]
        if dur < best or which < 0:
            best = dur
            which = i
    steps = whole(best, dt, seeds[round_stream[s]], ind, slot, shift, scale)
    if which >= 0 and ex_loop[e0 + which] and best > 0: # Back to the state: taken again no sooner than the next step
        steps = max(steps, 1.0)
    speed = adj[ex_tr[e0 + which], c] if which >= 0 and ex_resc[e0 + which] else 1.0
    return ti + steps, which, speed


@jit
def run(phase, ti_entry, ti_start, shift_val, base_ind, round0, max_depth, dt, n_states, n_trs,
        q_uid, q_tr, q_src, n_q, # The moves of the first round (phase 2: those infected); phase 1 finds them, from the members of each state
        pro_off, pro_uids, # Phase 1: the agents in each state at the start of the step (by state, from the census)
        members, ti_in, shifts, kept, ti_exit, exit_idx, sched, after_due, mults, has_mult, cls, slots, use_slots,
        nan, lowest, never, xnan, xlowest, xnever, anan, alowest, anever, # The stored step numbers of not set, the lowest step, and after the run: of the times of entering, of the scheduled exits, and of the delays
        ex_off, ex_tr, ex_after, ex_loop, ex_resc, n_br, br_edges, br_off, dw_kind, dw_pars, adj,
        tr_dst0, sp_n, sp_dst, sp_edges, sp_stays, tr_trigger,
        tg_tr, tg_event, tg_src, tg_delay, dl_kind, dl_pars, tg_where, ev0_off, ev0,
        seeds, br_stream, round_stream, dw_stream, dst_stream, after_stream, round_after_stream, shift, scale):
    """
    The rounds of moves of a phase of a step (phase 1: from the scheduled exits due at its start; phase 2: from the infections at its end)

    Returns the moves (uid, transition, state left, round), the arrivals (uid, transition, state left, destination, round; STAY
    for those put back), the number of each, the last round, and an error code (1 if the moves did not settle within max_depth rounds)
    """
    N = members[0].shape[0]
    size = max(64, 2*n_q)
    mv_uid = np.empty(size, dtype=np.int64) # The log of moves
    mv_tr = np.empty_like(mv_uid)
    mv_src = np.empty_like(mv_uid)
    mv_rnd = np.empty_like(mv_uid)
    nm = 0
    ar_uid = np.empty(size, dtype=np.int64) # The log of arrivals
    ar_tr = np.empty_like(ar_uid)
    ar_src = np.empty_like(ar_uid)
    ar_dst = np.empty_like(ar_uid)
    ar_rnd = np.empty_like(ar_uid)
    na = 0
    arr_uid = np.empty(size, dtype=np.int64) # Who arrived in a state that schedules exits in a round, and where
    arr_s = np.empty(size, dtype=np.int64)
    nu = np.empty(size, dtype=np.int64) # Whose exit is due already, by which exit, from which state
    nt = np.empty(size, dtype=np.int64)
    ns = np.empty(size, dtype=np.int64)
    tu = np.empty(size, dtype=np.int64) # Whom the triggers move, by which, from which state
    tt = np.empty(size, dtype=np.int64)
    ts = np.empty(size, dtype=np.int64)
    order = np.empty(size, dtype=np.int64) # The order of a round's moves, or arrivals, grouped
    choice = np.empty(size, dtype=np.int64) # The destination of each of a transition's moves
    cand = np.empty(size, dtype=np.int64) # A trigger's candidates
    rank_t = np.full(n_trs, -1, dtype=np.int64)
    rank_s = np.full(n_states, -1, dtype=np.int64)
    kept_exit = np.empty(N, dtype=np.int64) # Each agent's scheduled exit as they left, for those who stay (ss.stay) after a triggered move
    stamp = np.zeros(N, dtype=np.int64) # Taken by a trigger in the round of this number
    n_trig = tg_tr.shape[0]
    rnd = round0
    pend = np.empty((n_trig, N if tg_delay.any() else 0), dtype=np.int64) # Each delayed trigger's agents with a time after its event, found once
    n_pend = np.zeros(n_trig, dtype=np.int64)
    for g in range(n_trig):
        if tg_delay[g]:
            due_arr = after_due[g]
            for uid in range(N):
                if due_arr[uid] != anan:
                    pend[g, n_pend[g]] = uid
                    n_pend[g] += 1

    # The queue of moves for the next round: in phase 1, the scheduled exits due at the start of the step (as Agents.leave())
    qu = np.empty(size, dtype=np.int64)
    qt = np.empty(size, dtype=np.int64)
    qs = np.empty(size, dtype=np.int64)
    qu[:n_q] = q_uid[:n_q]
    qt[:n_q] = q_tr[:n_q]
    qs[:n_q] = q_src[:n_q]
    nq = n_q
    logged = 0 # The moves of the first round in phase 2 are made already (by transmit()), so are not logged again
    if phase == 1:
        nq = 0
        for s in range(n_states):
            if not sched[s] or pro_off[s+1] == pro_off[s]:
                continue
            for i in range(ex_off[s+1] - ex_off[s]):
                for j in range(pro_off[s], pro_off[s+1]):
                    uid = pro_uids[j]
                    if ti_exit[s][uid] <= ti_start and exit_idx[s][uid] == i:
                        qu = grow(qu, nq); qt = grow(qt, nq); qs = grow(qs, nq)
                        qu[nq] = uid; qt[nq] = ex_tr[ex_off[s] + i]; qs[nq] = s
                        nq += 1
        logged = nq
        for j in range(nq): # Leave the state
            uid = qu[j]
            s = qs[j]
            members[s][uid] = False
            if sched[s]:
                ti_exit[s][uid] = xnan
    first = True
    depth = 0
    err = 0
    while True:
        rnd += 1
        ind = base_ind + rnd
        for j in range(logged if phase == 1 and first else (0 if first else nq)):
            mv_uid = grow(mv_uid, nm); mv_tr = grow(mv_tr, nm); mv_src = grow(mv_src, nm); mv_rnd = grow(mv_rnd, nm)
            mv_uid[nm] = qu[j]; mv_tr[nm] = qt[j]; mv_src[nm] = qs[j]; mv_rnd[nm] = rnd
            nm += 1
        if nq > order.shape[0]:
            order = np.empty(2*nq, dtype=np.int64)
            choice = np.empty(2*nq, dtype=np.int64)
            arr_uid = np.empty(2*nq, dtype=np.int64)
            arr_s = np.empty(2*nq, dtype=np.int64)
            nu = np.empty(2*nq, dtype=np.int64)
            nt = np.empty(2*nq, dtype=np.int64)
            ns = np.empty(2*nq, dtype=np.int64)

        # Send each to their destination, one transition at a time, in the order they first left by it (as send())
        narr = 0
        first_order(qt, nq, rank_t, order)
        a0 = na # The arrivals of this round start here
        j = 0
        while j < nq:
            t = qt[order[j]]
            k = j
            while k < nq and qt[order[k]] == t:
                k += 1
            n_dst = sp_n[t]
            if n_dst == 0: # One destination
                d = tr_dst0[t]
                for m in range(j, k):
                    q = order[m]
                    ar_uid = grow(ar_uid, na); ar_tr = grow(ar_tr, na); ar_src = grow(ar_src, na); ar_dst = grow(ar_dst, na); ar_rnd = grow(ar_rnd, na)
                    ar_uid[na] = qu[q]; ar_tr[na] = t; ar_src[na] = qs[q]; ar_dst[na] = d; ar_rnd[na] = rnd
                    na += 1
            else: # Several: one draw each; those who stay are put back first, then each destination in turn (as destinations())
                for m in range(j, k):
                    uid = qu[order[m]]
                    slot = slots[uid] if use_slots else uid
                    u = uniform(seeds[dst_stream[t]], ind, slot, shift, scale)
                    c = cls[uid]
                    ch = -1
                    for i in range(n_dst):
                        if u >= sp_edges[t, c, i] and u < sp_edges[t, c, i+1]:
                            ch = i
                    if ch < 0 and sp_stays[t] and u >= sp_edges[t, c, n_dst]:
                        ch = STAY
                    choice[m] = ch
                for m in range(j, k): # Those who stay: back in the state they left, with the exit they had (after a trigger) or none
                    if choice[m] == STAY:
                        q = order[m]
                        uid = qu[q]
                        s = qs[q]
                        members[s][uid] = True
                        if sched[s]:
                            if tr_trigger[t]:
                                ti_exit[s][uid] = kept_exit[uid]
                            else:
                                ti_exit[s][uid] = xnever
                                exit_idx[s][uid] = -2
                        ar_uid = grow(ar_uid, na); ar_tr = grow(ar_tr, na); ar_src = grow(ar_src, na); ar_dst = grow(ar_dst, na); ar_rnd = grow(ar_rnd, na)
                        ar_uid[na] = uid; ar_tr[na] = t; ar_src[na] = s; ar_dst[na] = STAY; ar_rnd[na] = rnd
                        na += 1
                for i in range(n_dst):
                    d = sp_dst[t, i]
                    for m in range(j, k):
                        if choice[m] == i:
                            q = order[m]
                            ar_uid = grow(ar_uid, na); ar_tr = grow(ar_tr, na); ar_src = grow(ar_src, na); ar_dst = grow(ar_dst, na); ar_rnd = grow(ar_rnd, na)
                            ar_uid[na] = qu[q]; ar_tr[na] = t; ar_src[na] = qs[q]; ar_dst[na] = d; ar_rnd[na] = rnd
                            na += 1
            j = k

        # Enter: in the order of the batches above (as enter() and schedule())
        for a in range(a0, na):
            d = ar_dst[a]
            if d == STAY:
                continue
            uid = ar_uid[a]
            members[d][uid] = True
            if kept[d]:
                ti_in[d][uid] = to_step(ti_entry, lowest, never, nan)
                shifts[d][uid] = shift_val
            if sched[d] and ti_exit[d][uid] == xnan: # Unless code put them back with an exit still scheduled, which they keep (as Agents.schedule())
                slot = slots[uid] if use_slots else uid
                due, which, speed = schedule(uid, d, cls, slot, ti_entry, ind, dt, ex_off, ex_tr, ex_after, ex_loop, ex_resc, n_br, br_edges, br_off,
                                             dw_kind, dw_pars, adj, seeds, br_stream, round_stream, dw_stream, shift, scale)
                ti_exit[d][uid] = to_step(due, xlowest, xnever, xnan)
                exit_idx[d][uid] = which
                if has_mult[d]:
                    mults[d][uid] = speed
            if sched[d]:
                arr_uid[narr] = uid
                arr_s[narr] = d
                narr += 1

        # Who moves on at once: those whose exit is due already, by state in the order they first arrived (as onward())
        nn = 0
        first_order(arr_s, narr, rank_s, order)
        j = 0
        while j < narr:
            s = arr_s[order[j]]
            k = j
            while k < narr and arr_s[order[k]] == s:
                k += 1
            for i in range(ex_off[s+1] - ex_off[s]):
                for m in range(j, k):
                    uid = arr_uid[order[m]]
                    if members[s][uid] and ti_exit[s][uid] <= ti_entry and exit_idx[s][uid] == i:
                        nu[nn] = uid; nt[nn] = ex_tr[ex_off[s] + i]; ns[nn] = s
                        nn += 1
            j = k

        # The triggers, ahead of exits due now (as triggered()): those to whom the trigger's event happened since it last looked, after a delay
        n_tr = 0
        for g in range(n_trig):
            e = tg_event[g]
            nc = 0
            if first:
                for a in range(ev0_off[g], ev0_off[g+1]):
                    cand = grow(cand, nc)
                    cand[nc] = ev0[a]
                    nc += 1
            for a in range(a0, na):
                if ar_tr[a] == e and ar_dst[a] != STAY:
                    cand = grow(cand, nc)
                    cand[nc] = ar_uid[a]
                    nc += 1
            cand[:nc].sort()
            m = 0 # Each once, in order, and in the source
            for i in range(nc):
                uid = cand[i]
                if m > 0 and cand[m-1] == uid:
                    continue
                for s in range(n_states):
                    if tg_src[g, s] and members[s][uid]:
                        cand[m] = uid
                        m += 1
                        break
            nc = m
            if tg_delay[g]: # A time after the event, drawn for each person, as long as they stay in the source (as delayed())
                due_arr = after_due[g]
                m = 0
                for i in range(n_pend[g]): # Those who left the source forget it (and those whose time was taken are dropped)
                    uid = pend[g, i]
                    if due_arr[uid] == anan:
                        continue
                    inside = False
                    for s in range(n_states):
                        if tg_src[g, s] and members[s][uid]:
                            inside = True
                    if not inside:
                        due_arr[uid] = anan
                        continue
                    pend[g, m] = uid
                    m += 1
                for i in range(nc):
                    uid = cand[i]
                    if due_arr[uid] == anan:
                        slot = slots[uid] if use_slots else uid
                        c = cls[uid]
                        dur = dwell(dl_kind[g], dl_pars[g, c], uniform(seeds[after_stream[g]], ind, slot, shift, scale))
                        steps = whole(dur, dt, seeds[round_after_stream[g]], ind, slot, shift, scale)
                        due_arr[uid] = to_step(ti_start + steps, alowest, anever, anan)
                        pend[g, m] = uid
                        m += 1
                n_pend[g] = m
                nc = 0
                for i in range(m):
                    uid = pend[g, i]
                    if due_arr[uid] <= ti_start and stamp[uid] != rnd:
                        cand = grow(cand, nc)
                        cand[nc] = uid
                        nc += 1
                        due_arr[uid] = anan
                cand[:nc].sort() # In order, as the Python engine lists them
            n = 0
            for i in range(nc):
                uid = cand[i]
                if tg_where[g, cls[uid]] and stamp[uid] != rnd:
                    cand[n] = uid
                    n += 1
            if n:
                for s in range(n_states):
                    if tg_src[g, s]:
                        for i in range(n):
                            uid = cand[i]
                            if members[s][uid]:
                                tu = grow(tu, n_tr); tt = grow(tt, n_tr); ts = grow(ts, n_tr)
                                tu[n_tr] = uid; tt[n_tr] = tg_tr[g]; ts[n_tr] = s
                                n_tr += 1
                for i in range(n):
                    stamp[cand[i]] = rnd

        # The next round's moves: the triggered, then the exits due now of those no trigger moves
        nq = 0
        for i in range(n_tr):
            qu = grow(qu, nq); qt = grow(qt, nq); qs = grow(qs, nq)
            qu[nq] = tu[i]; qt[nq] = tt[i]; qs[nq] = ts[i]
            nq += 1
        for i in range(nn):
            if stamp[nu[i]] != rnd:
                qu = grow(qu, nq); qt = grow(qt, nq); qs = grow(qs, nq)
                qu[nq] = nu[i]; qt[nq] = nt[i]; qs[nq] = ns[i]
                nq += 1
        first = False
        if nq == 0:
            break
        if depth > max_depth:
            err = 1
            break
        for j in range(nq): # Leave the state (as move())
            uid = qu[j]
            s = qs[j]
            t = qt[j]
            if sched[s]:
                if sp_stays[t] and tr_trigger[t]: # Those who stay keep the exit they had
                    kept_exit[uid] = ti_exit[s][uid]
                ti_exit[s][uid] = xnan
            members[s][uid] = False
        depth += 1
    return summarize(N, n_states, n_trs, mv_uid[:nm], mv_tr[:nm], mv_src[:nm], mv_rnd[:nm], ar_uid[:na], ar_tr[:na], ar_src[:na], ar_dst[:na], ar_rnd[:na]) + (rnd, err)


@jit
def summarize(N, n_states, n_trs, mv_uid, mv_tr, mv_src, mv_rnd, ar_uid, ar_tr, ar_src, ar_dst, ar_rnd):
    """
    What the Python engine records of the moves (see `Agents.replay()`): for each transition, its flow (those who left by it, less those put back) and its
    events (those who left by it, in order, less those put back in the same round or a later one), and those put back by it; the change in the number in
    each state; and the batches of arrivals, one for each transition and destination in each round, as (start, end) in the arrivals, with the arrivals
    """
    nm = mv_uid.shape[0]
    na = ar_uid.shape[0]
    flow = np.zeros(n_trs, dtype=np.int64)
    dlive = np.zeros(n_states, dtype=np.int64)
    n_ev = np.zeros(n_trs + 1, dtype=np.int64)
    n_st = np.zeros(n_trs + 1, dtype=np.int64)
    for i in range(nm):
        flow[mv_tr[i]] += 1
        dlive[mv_src[i]] -= 1
        n_ev[mv_tr[i] + 1] += 1
    for i in range(na):
        if ar_dst[i] == STAY:
            flow[ar_tr[i]] -= 1
            dlive[ar_src[i]] += 1
            n_st[ar_tr[i] + 1] += 1
        else:
            dlive[ar_dst[i]] += 1
    for t in range(n_trs):
        n_ev[t + 1] += n_ev[t]
        n_st[t + 1] += n_st[t]
    st_uid = np.empty(n_st[n_trs], dtype=np.int64) # Those put back, by transition
    pos = n_st[:n_trs].copy()
    for i in range(na):
        if ar_dst[i] == STAY:
            st_uid[pos[ar_tr[i]]] = ar_uid[i]
            pos[ar_tr[i]] += 1
    last = np.full(N, -1, dtype=np.int64) # The last round each agent was put back by the transition being worked on
    ev_uid = np.empty(nm, dtype=np.int64) # The events of each transition, in order
    ev_off = np.zeros(n_trs + 1, dtype=np.int64)
    k = 0
    for t in range(n_trs):
        ev_off[t] = k
        if n_ev[t + 1] == n_ev[t]:
            continue
        for i in range(na):
            if ar_tr[i] == t and ar_dst[i] == STAY and ar_rnd[i] > last[ar_uid[i]]:
                last[ar_uid[i]] = ar_rnd[i]
        for i in range(nm):
            if mv_tr[i] == t and mv_rnd[i] > last[mv_uid[i]]:
                ev_uid[k] = mv_uid[i]
                k += 1
        for i in range(na):
            if ar_tr[i] == t and ar_dst[i] == STAY:
                last[ar_uid[i]] = -1
    ev_off[n_trs] = k
    nb = 0 # The batches
    starts = np.empty(na + 1, dtype=np.int64)
    prev = -1
    for i in range(na):
        if ar_dst[i] == STAY:
            continue
        if prev < 0 or ar_tr[i] != ar_tr[prev] or ar_dst[i] != ar_dst[prev] or ar_rnd[i] != ar_rnd[prev]:
            starts[nb] = i
            nb += 1
        prev = i
    b_start = starts[:nb].copy()
    b_end = np.empty(nb, dtype=np.int64)
    for b in range(nb): # Each batch runs to the next arrival that is not in it (those put back are never in the middle of one)
        e = b_start[b] + 1
        while e < na and ar_dst[e] != STAY and ar_tr[e] == ar_tr[b_start[b]] and ar_dst[e] == ar_dst[b_start[b]] and ar_rnd[e] == ar_rnd[b_start[b]]:
            e += 1
        b_end[b] = e
    return flow, ev_uid[:k], ev_off, st_uid, n_st, dlive, b_start, b_end, ar_uid, ar_tr, ar_src, ar_dst
