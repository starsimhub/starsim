"""
Not for the user: the compiled moves of the agent engine (`solver='abm'`), one call for the moves of a step: see `ss.solvers.Compiled`

The graph is laid out once as tables of integers (each state's exits, each transition's destinations and dwell time, the moves a
time after each transition's event), and the call takes every agent whose time has come through the step: its scheduled exit
from its state, or a move a time after an event, in the engine's order of the transitions; it then draws where they go and how
long they stay (by the inverse of each distribution's CDF), schedules the moves a time after the event, and takes them on at
once if that is due already, as the agent engine's same-step cascade does. It writes the states, the times of entering, the
scheduled exits and the counts of the flows in place, and logs each move, which the engine records as events.

Used with: agents; declared modules
"""
import numpy as np
import numba as nb
from .draws import ndtri, exp, log, log1p # Starsim's own, as the Rust copy has them, so the two give the same numbers

__all__ = ['due_agents', 'step_moves', 'FAMILIES']

jit = nb.jit(nopython=True, cache=True, error_model='numpy')

FAMILIES = dict(value=-2, none=-1, constant=0, expon=1, lognorm=2, norm=3, uniform=4, weibull_min=5) # The distributions drawn in the call: SciPy's names, with SciPy's parameters (see par())


@jit
def rand(key, a, ctr):
    """ Agent `a`'s next uniform in (0, 1) of the call: a hash (SplitMix64's) of the call's key, the agent, and its count of draws in the step, so each agent's numbers depend only on its own moves """
    z = np.uint64(key) + np.uint64(a)*np.uint64(0x9E3779B97F4A7C15) + np.uint64(ctr[a])*np.uint64(0xD1B54A32D192ED03)
    ctr[a] += 1
    z = (z ^ (z >> np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (float(z >> np.uint64(11)) + 0.5)/9007199254740992.0


@jit
def par(j, a, pcol, PA, PS):
    """ Parameter `j` of a distribution for agent `a`: a column of the table of values by agent, or a number """
    c = pcol[j]
    return PA[a, c] if c >= 0 else PS[-c - 1]


@jit
def draw(d, a, key, ctr, fam, pcol, PA, PS):
    """ A draw for agent `a` from distribution `d`, in the sim's unit of time """
    f = fam[d]
    if f == 0:
        return par(3*d, a, pcol, PA, PS)
    u = rand(key, a, ctr)
    p0 = par(3*d, a, pcol, PA, PS)
    p1 = par(3*d + 1, a, pcol, PA, PS)
    p2 = par(3*d + 2, a, pcol, PA, PS)
    if f == 1: # expon(loc=p0, scale=p1)
        return p0 - p1*log1p(-u)
    if f == 2: # lognorm(s=p0, scale=p1, loc=p2)
        if p0 == 0:
            return p1 + p2
        return p1*exp(p0*ndtri(u)) + p2
    if f == 3: # norm(loc=p0, scale=p1)
        return p0 + p1*ndtri(u)
    if f == 4: # uniform(loc=p0, scale=p1)
        return p0 + p1*u
    if f == 5: # weibull_min(c=p0, scale=p1, loc=p2)
        return p2 + p1*exp(log(-log1p(-u))/p0)
    return np.nan


@jit
def schedule(a, s, entered, key, ctr, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt):
    """ Agent `a` has entered state `s` at `entered`: choose its branch, draw its dwell times (the earliest timed exit wins), and schedule its exit, in whole steps (see `ss.solvers.Agents.schedule()`) """
    lo = ex_ptr[s]
    hi = ex_ptr[s+1]
    if lo == hi:
        return
    best = np.inf
    which = -1
    nb_ = nbr[s]
    if nb_ > 0: # A branch chosen on entry
        u = rand(key, a, ctr)
        c = br_col[s]
        for i in range(nb_):
            if u >= E[a, c + i] and u < E[a, c + i + 1]:
                t = ex_tr[lo + i]
                if has_dw[t]:
                    best = draw(dw[t], a, key, ctr, fam, pcol, PA, PS)/par(3*adj[t], a, pcol, PA, PS) + aft[t]
                else:
                    best = aft[t] if aft[t] > 0 else -np.inf
                which = i
                break
    for i in range(lo + nb_, hi): # Competing dwell times
        t = ex_tr[i]
        d = draw(dw[t], a, key, ctr, fam, pcol, PA, PS)/par(3*adj[t], a, pcol, PA, PS) + aft[t] # A multiplier on the rate divides the duration
        if d < best or which < 0:
            best = d
            which = i - lo
    steps = np.floor(best/dt + phase[a])
    if which >= 0 and loops[lo + which] and best > 0:
        steps = max(steps, 1.0)
    XT[s][a] = entered + steps
    XW[s][a] = which
    return


@jit
def due_agents(S, XT, sched, TT, ntrig, alive, ti_due, ti_delay, mark, idx):
    """ The agents with a move due at the start of a step: a scheduled exit due (or none scheduled, in a state that schedules them), or a move after an event; a pass over each array, listing only those due (few, so the branch is predictable) """
    n = len(alive)
    m = 0
    for s in range(len(S)):
        if sched[s]:
            arr = S[s]
            xt = XT[s]
            for a in range(n):
                x = xt[a]
                if arr[a] and not (x > ti_due) and not mark[a] and alive[a]: # Due, or NaN (none scheduled)
                    mark[a] = True
                    idx[m] = a
                    m += 1
    for g in range(ntrig):
        tt = TT[g]
        for a in range(n):
            if tt[a] <= ti_delay and not mark[a] and alive[a]:
                mark[a] = True
                idx[m] = a
                m += 1
    out = np.sort(idx[:m])
    for i in range(m): # The marks cleared for the next call
        mark[out[i]] = False
    return out


@jit
def step_moves(agents, arr_tr, arr_src, arr_kind, scan, start, ti_due, ti_delay, ti_enter, shift, key, ctr, alive,
               S, STI, SSH, kept, LEFT, has_left, XT, XW, sched,
               ex_ptr, ex_tr, nbr, br_col, loops,
               pos, kind, split, dst_ptr, dst_state, dst_col, dw, adj, aft, has_dw, src_mask, ae_ptr, ae_trig,
               tr_trig, TT, trig_where, WH, trig_dist, trig_fixed, trig_mean,
               fam, pcol, PA, PS, E, phase, dt, w, weighted,
               ev_a, ev_t, ev_s, ev_d, n_ev, flow, flowdst, dlive, max_hops,
               mode, comp_state, LATE, has_late, halved, trig_cond, lazy, pend_a, pend_t, pend_s, pend_k, n_pend):
    """
    The moves of a step, or of some agents, as the agent engine's same-step cascade takes them

    With `scan`, every agent from index `start`; otherwise the agents in `agents`, from `start`, each having just left `arr_src` by
    `arr_tr` (to enter its destination now), or -1 for one to look at only. Exits are due by `ti_due` and moves after an event by
    `ti_delay`; agents enter at `ti_enter`. Stops early, at an agent, when the log of moves may not have room for its moves, and
    returns where it stopped (the number of agents, or of `agents`, if it finished) and the length of the log. A trigger on a condition
    (`trig_cond`, the column of `WH` that holds it, or -1) is due at once for anyone who enters its source while it holds. The shares of a
    split worked out on demand (`lazy`) are NaN in `E` until they are: an agent who reaches one stops there, before anything is drawn for
    that move, and is listed in `pend_*` (the agent, the transition, its state, and 0 to take it again as an arrival or 1 to look again),
    `n_pend[0]` of them, for the caller to work them out and take the agent on from where it stopped, with the same draws
    """
    ns = len(S)
    n = len(alive) if scan else len(agents)
    cap = len(ev_a)
    ntrig = len(tr_trig)
    for idx in range(start, n):
        if n_ev + max_hops > cap:
            return idx, n_ev
        a = idx if scan else agents[idx]
        if not alive[a]:
            continue
        s = -1
        for k in range(ns):
            if S[k][a]:
                s = k
                break
        t_in = -1
        if not scan:
            ak = arr_kind[idx]
            if ak == 0: # To take a transition now (e.g. infected), if still in its source
                if S[arr_src[idx]][a]:
                    t_in = arr_tr[idx]
                    s = arr_src[idx]
            elif ak == 2 and s >= 0: # Gone through a transition already (moved by code): the moves a time after it
                te_ = arr_tr[idx]
                for j in range(ae_ptr[te_], ae_ptr[te_+1]):
                    g = ae_trig[j]
                    if src_mask[tr_trig[g], s] and np.isnan(TT[g][a]):
                        if trig_fixed[g]:
                            dur = draw(trig_dist[g], a, key, ctr, fam, pcol, PA, PS)
                        else:
                            dur = -trig_mean[a if len(trig_mean) > 1 else 0]*log1p(-rand(key, a, ctr))
                        TT[g][a] = ti_delay + np.floor(dur/dt + phase[a])
        hops = 0
        while True:
            hops += 1
            if hops > max_hops:
                raise ValueError('moves with no wait did not settle within a step')
            if s < 0:
                break
            trig = -1
            first = t_in >= 0
            if first:
                t = t_in
            else:
                # What is due: the competing rates, then the scheduled exit of the state, or a move after an event out of it, the first in the engine's order
                t = -1
                if sched[s]:
                    te = XT[s][a]
                    if np.isnan(te): # Put in the state with no exit scheduled (by code, or at the start): scheduled now, from when they entered it
                        entered = STI[s][a] if kept[s] else np.nan
                        if np.isnan(entered):
                            entered = ti_due
                        schedule(a, s, entered, key, ctr, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt)
                        te = XT[s][a]
                    wx = XW[s][a]
                    if te <= ti_due and wx >= 0:
                        t = ex_tr[ex_ptr[s] + wx]
                for g in range(ntrig):
                    tg = tr_trig[g]
                    if src_mask[tg, s] and TT[g][a] <= ti_delay and (t < 0 or (kind[t] != 3 and pos[tg] < pos[t])):
                        t = tg
                        trig = g
                if t < 0:
                    break
            if split[t] and lazy[t] and np.isnan(E[a, dst_col[t] + 1]): # Its shares are not worked out yet: stopped here, as it was
                m = n_pend[0]
                pend_a[m] = a
                pend_t[m] = t
                pend_s[m] = s
                pend_k[m] = 0 if first else 1
                n_pend[0] = m + 1
                break
            t_in = -1
            if trig >= 0:
                TT[trig][a] = np.nan # Taken now, whether or not it applies
                wi = trig_where[trig]
                if wi >= 0 and not WH[wi][a]:
                    continue
            frm = s

            # Where to: one destination, or one of several (beyond the last share, they stay)
            lo = dst_ptr[t]
            hi = dst_ptr[t+1]
            if split[t]:
                u = rand(key, a, ctr)
                c = dst_col[t]
                dst = -2
                for i in range(hi - lo):
                    if u >= E[a, c + i] and u < E[a, c + i + 1]:
                        dst = i
                        break
                if dst < 0: # Stays
                    if kind[t] == 0: # The exit was this transition's: never scheduled again (see ss.solvers.Agents.stay())
                        XT[frm][a] = np.inf
                        XW[frm][a] = -2
                    continue
            else:
                dst = 0
            d = dst_state[lo + dst]

            # Leave
            S[frm][a] = False
            dlive[frm] -= 1
            if sched[frm]:
                XT[frm][a] = np.nan
            if has_left[frm]:
                LEFT[frm][a] = ti_enter
            wt = w[a] if weighted else 1.0
            flow[t] += wt
            flowdst[t, dst] += wt
            ev_a[n_ev] = a
            ev_t[n_ev] = t
            ev_s[n_ev] = frm
            ev_d[n_ev] = d
            n_ev += 1

            if d < 0: # A death (or a removal): out of every state of the module; the engine requests it
                for g in range(ntrig):
                    TT[g][a] = np.nan
                s = -1
                break

            # Enter
            S[d][a] = True
            dlive[d] += 1
            if kept[d]:
                STI[d][a] = ti_enter
                SSH[d][a] = shift
            for g in range(ntrig): # A move after an event is forgotten for those no longer in its source; one on a condition is due now for those it holds for
                if src_mask[tr_trig[g], frm] and not src_mask[tr_trig[g], d]:
                    TT[g][a] = np.nan
                c = trig_cond[g]
                if c >= 0 and src_mask[tr_trig[g], d] and WH[c][a]:
                    TT[g][a] = -np.inf
            if comp_state[d] and has_late[d]: # The share of a step of its competing rates missed (see ss.solvers.Agents.enter()); in the state-change row, the engine draws half a step of them for those who arrived, after the call
                if mode == 0:
                    LATE[d][a] = 0.5
                else:
                    LATE[d][a] = (-0.5 if halved[d] else 0.0) if mode == 1 else (1.0 if ti_enter == ti_delay else 0.0)
            if sched[d]:
                XT[d][a] = np.nan
                schedule(a, d, ti_enter, key, ctr, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt)
            for j in range(ae_ptr[t], ae_ptr[t+1]): # The moves a time after this event
                g = ae_trig[j]
                if src_mask[tr_trig[g], d] and np.isnan(TT[g][a]):
                    if trig_fixed[g]:
                        dur = draw(trig_dist[g], a, key, ctr, fam, pcol, PA, PS)
                    else: # An exponential with the given mean
                        dur = -trig_mean[a if len(trig_mean) > 1 else 0]*log1p(-rand(key, a, ctr))
                    TT[g][a] = ti_delay + np.floor(dur/dt + phase[a])
            s = d
    return n, n_ev
