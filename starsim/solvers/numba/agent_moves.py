"""
Not for the user: the compiled moves of the agent engine (`solver='abm'`), one call for the moves of a step: see `ss.solvers.Compiled`

The graph is laid out once as tables of integers (each state's exits, each transition's destinations and dwell time, the moves a
time after each transition's event), and the call takes every agent whose time has come through the step: its scheduled exit
from its state, or a move a time after an event, in the engine's order of the transitions; it then draws where they go and how
long they stay (by the inverse of each distribution's CDF), schedules the moves a time after the event, and takes them on at
once if that is due already, as the agent engine's same-step cascade does. Each draw is the one the agent engine's own moves make
(see `rand()`), so a run gives the same numbers whichever of the two takes its moves (e.g. with an intervention that the call does
not take, and without it), and common random numbers pair the runs. It writes the states, the times of entering, the scheduled
exits and the counts of the flows in place, and logs each move, which the engine records as events.

Its Rust copy (`starsim/solvers/rust/src/agent_moves.rs`) gives the same numbers, bit for bit; this is the reference.

Used with: agents; declared modules
"""
import math
import numpy as np
import numba as nb
from . import draws
from .draws import ndtri, exp, log, log1p, mix, crn_uniform, U64, GOLDEN # Starsim's own, as the Rust copy has them, so the two give the same numbers
from ..backends import compiling

compiling.depends(__file__, draws.__file__) # The draws are inlined, so the cached kernels are stale when draws.py changes

__all__ = ['due_agents', 'step_moves', 'FAMILIES']

jit = nb.jit(nopython=True, cache=True, error_model='numpy')

FAMILIES = dict(value=-2, none=-1, constant=0, expon=1, lognorm=2, norm=3, uniform=4, weibull_min=5, uniform32=6) # The distributions drawn in the call: SciPy's names, with SciPy's parameters (see par()); uniform32 is ss.uniform with parameters that are plain numbers, which NumPy takes in single precision, as the uniforms are
CONSTANT, EXPON, LOGNORM, NORM, UNIFORM, WEIBULL, UNIFORM32 = [FAMILIES[k] for k in ['constant', 'expon', 'lognorm', 'norm', 'uniform', 'weibull_min', 'uniform32']]

# The codes the caller passes (ss.solvers.Compiled), named once; the Rust copy has the same constants. Numba takes module-level numbers as constants, so they cost nothing
SCHEDULED, TRIGGERED, CONTACT, RATED = range(4) # A transition's kind (`kind`): a scheduled exit, a move a time after an event, an infection, or a competing rate
TAKE, LOOK, MOVED = range(3) # What to do with an agent given (`arr_kind`): take the transition given now, look at what is due only, or take the moves a time after the transition it was moved by (by code)
AT_START, INFECTIONS, LATER = range(3) # When in the step the call is (`mode`): at its start, for the infections, or later (moves by code, and cascades)


@jit
def rand(r, tag, ps, a, cr):
    """
    Agent `a`'s uniform in (0, 1) from stream `r`, in the round of moves the agent engine would draw it in: the pass `ps` of its same-step
    cascade (from the call's first, `ind0`; -1 for the round before it), and the transition `tag` it is for (-1 for none), as
    `ss.solvers.Agents.draw()` gives it (`hash_uniforms()` of the round's seed of the stream and the round's index, at the agent's slot);
    `cr` is the streams' tuple (slots, ind0, RS, ush, usc, z32, dstream)
    """
    slots, ind0, RS, ush, usc = cr[0], cr[1], cr[2], cr[3], cr[4]
    slot = slots[a] if len(slots) else a
    k = mix((U64(RS[r, tag + 1]) ^ (U64(ind0 + ps)*GOLDEN)) + GOLDEN) # As draws.key(), inline
    return crn_uniform(k, slot, ush[r], usc[r])


@jit
def pass_of(t, p, q, pos):
    """ The pass of the cascade in which transition `t` is taken, for an agent in pass `p` that came by the transition at place `q` in the sweep: this one if `t` is later in the sweep, else the next """
    return p if pos[t] > q else p + 1


@jit
def par(j, a, pcol, PA, PS):
    """ Parameter `j` of a distribution for agent `a`: a column of the table of values by agent, or a number """
    c = pcol[j]
    return PA[a, c] if c >= 0 else PS[-c - 1]


@jit
def draw(d, a, tag, ps, fam, pcol, PA, PS, cr):
    """ A draw for agent `a` from distribution `d`, in the sim's unit of time, from the uniform the agent engine would invert for it (see `Moving.draw()` and each distribution's ppf()) """
    z32, dstream = cr[5], cr[6]
    f = fam[d]
    if f == CONSTANT:
        return par(3*d, a, pcol, PA, PS)
    r = dstream[d]
    u = rand(r, tag, ps, a, cr)
    p0 = par(3*d, a, pcol, PA, PS)
    p1 = par(3*d + 1, a, pcol, PA, PS)
    p2 = par(3*d + 2, a, pcol, PA, PS)
    if f == EXPON: # expon(loc=p0, scale=p1)
        return p0 - p1*log1p(-u)
    if f == LOGNORM: # lognorm(s=p0, scale=p1, loc=p2), with NumPy's exp (libm's)
        if p0 == 0:
            return p1 + p2
        z = ndtri(u)
        if z32[r]: # SciPy's ndtri of a single-precision uniform is single precision
            z = float(np.float32(z))
        return p1*math.exp(p0*z) + p2
    if f == NORM: # norm(loc=p0, scale=p1)
        z = ndtri(u)
        if z32[r]:
            z = float(np.float32(z))
        return p0 + p1*z
    if f == UNIFORM: # uniform(loc=p0, scale=p1)
        return p0 + p1*u
    if f == WEIBULL: # weibull_min(c=p0, scale=p1, loc=p2)
        return p2 + p1*exp(log(-log1p(-u))/p0)
    if f == UNIFORM32: # uniform(loc=p0, scale=p1) in single precision
        return float(np.float32(u)*np.float32(p1) + np.float32(p0))
    return np.nan


@jit
def schedule(a, s, entered, tag, ps, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt, cr, bstream):
    """ Agent `a` has entered state `s` at `entered`: choose its branch, draw its dwell times (the earliest timed exit wins), and schedule its exit, in whole steps (see `ss.solvers.Agents.schedule()`), with the draws of the round of the move (`tag`, `ps`) """
    lo = ex_ptr[s]
    hi = ex_ptr[s+1]
    if lo == hi:
        return
    best = np.inf
    which = -1
    nb_ = nbr[s]
    if nb_ > 0: # A branch chosen on entry
        u = rand(bstream[s], tag, ps, a, cr)
        c = br_col[s]
        for i in range(nb_):
            if u >= E[a, c + i] and u < E[a, c + i + 1]:
                t = ex_tr[lo + i]
                if has_dw[t]:
                    best = draw(dw[t], a, tag, ps, fam, pcol, PA, PS, cr)/par(3*adj[t], a, pcol, PA, PS) + aft[t]
                else:
                    best = aft[t] if aft[t] > 0 else -np.inf
                which = i
                break
    for i in range(lo + nb_, hi): # Competing dwell times
        t = ex_tr[i]
        d = draw(dw[t], a, tag, ps, fam, pcol, PA, PS, cr)/par(3*adj[t], a, pcol, PA, PS) + aft[t] # A multiplier on the rate divides the duration
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
def delay(g, a, tag, ps, trig_dist, fam, pcol, PA, PS, phase, dt, ti_delay, cr):
    """ A delay after an event, drawn for agent `a` in the round of the trigger's turn (`tag`, `ps`), as the step it is due in (a delay given as a number is an exponential distribution: see `ss.solvers.Compiled.kadd_mean()`) """
    dur = draw(trig_dist[g], a, tag, ps, fam, pcol, PA, PS, cr)
    return ti_delay + np.floor(dur/dt + phase[a])


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
def step_moves(agents, arr_tr, arr_src, arr_kind, arr_p, arr_q, scan, start, ti_due, ti_delay, ti_enter, shift, alive,
               S, STI, SSH, kept, LEFT, has_left, XT, XW, sched,
               ex_ptr, ex_tr, nbr, br_col, loops,
               pos, kind, split, dst_ptr, dst_state, dst_col, dw, adj, aft, has_dw, src_mask, ae_ptr, ae_trig,
               tr_trig, TT, trig_where, WH, trig_dist,
               fam, pcol, PA, PS, E, phase, dt, w, weighted,
               slots, ind0, RS, ush, usc, z32, dstream, bstream, tstream, back, againx, again,
               ev_a, ev_t, ev_s, ev_d, ev_p, n_ev, flow, flowdst, dlive, max_hops,
               mode, comp_state, LATE, has_late, halved, trig_cond, lazy, pend_a, pend_t, pend_s, pend_k, pend_p, pend_q, n_pend):
    """
    The moves of a step, or of some agents, as the agent engine's same-step cascade takes them, with the draws it would make

    With `scan`, every agent from index `start`; otherwise the agents in `agents`, from `start`, each having just left `arr_src` by
    `arr_tr` (to enter its destination now), as `arr_kind` says (`TAKE`, `LOOK` or `MOVED`). `mode` says when in the step the call is (`AT_START`,
    `INFECTIONS` or `LATER`), which sets the fraction of a step of competing rates those who arrive have missed (`LATE`). Exits are due by `ti_due` and moves after an event by
    `ti_delay`; agents enter at `ti_enter`. Stops early, at an agent, when the log of moves may not have room for its moves, and
    returns where it stopped (the number of agents, or of `agents`, if it finished) and the length of the log. A trigger on a condition
    (`trig_cond`, the column of `WH` that holds it, or -1) is due at once for anyone who enters its source while it holds. The fractions of a
    split worked out on demand (`lazy`) are NaN in `E` until they are: an agent who reaches one stops there, before anything is drawn for
    that move, and is listed in `pend_*` (the agent, the transition, its state, 0 to take it again as an arrival or 1 to look again, and its
    pass and place in the cascade), `n_pend[0]` of them, for the caller to work them out and take the agent on from where it stopped

    Each draw is the agent engine's (see `rand()`): a move by a transition in a pass of the cascade draws, in that transition's round, where
    the agent goes and, on entry, its branch and dwell times; a delay after an event is drawn in the round of the trigger's turn. An agent
    who arrived by a transition in a pass (`arr_p`, at `arr_q`, the transition's place in the sweep, or -1 for none yet) leaves by one later
    in the sweep in the same pass, and by one before it (or itself) in the next; an arrival in a state with a scheduled exit before it, or a
    move by a transition a trigger before it waits for (`back`), means another pass (`again`, by pass), as `Moving.finish_moves()` takes it
    """
    ns = len(S)
    n = len(alive) if scan else len(agents)
    cap = len(ev_a)
    ntrig = len(tr_trig)
    npass = len(again)
    cr = (slots, ind0, RS, ush, usc, z32, dstream)
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
        p = 0 if scan else arr_p[idx] # The pass of the cascade the agent is in, and the place in the sweep of the transition that brought it
        q = -1 if scan else arr_q[idx]
        if not scan:
            ak = arr_kind[idx]
            if ak == TAKE: # To take a transition now (e.g. infected), if still in its source, in the pass given
                if S[arr_src[idx]][a]:
                    t_in = arr_tr[idx]
                    s = arr_src[idx]
            elif ak == MOVED and s >= 0: # Gone through a transition already (moved by code): the moves a time after it, drawn at each trigger's turn
                te_ = arr_tr[idx]
                for j in range(ae_ptr[te_], ae_ptr[te_+1]):
                    g = ae_trig[j]
                    tg = tr_trig[g]
                    if src_mask[tg, s] and np.isnan(TT[g][a]):
                        TT[g][a] = delay(g, a, tg, pass_of(tg, p, q, pos), trig_dist, fam, pcol, PA, PS, phase, dt, ti_delay, cr)
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
                pt = p
            else:
                # What is due: the scheduled exit of the state, or a move after an event out of it, the first the cascade comes to
                t = -1
                pt = 0
                if sched[s]:
                    te = XT[s][a]
                    if np.isnan(te): # Put in the state with no exit scheduled (by code, or at the start): scheduled now, from when they entered it, as at the start of the step
                        entered = STI[s][a] if kept[s] else np.nan
                        if np.isnan(entered): # With no time of entering: they entered it now, and are dated so (see ss.solvers.Agents.schedule())
                            entered = ti_due
                            if kept[s]:
                                STI[s][a] = entered
                                SSH[s][a] = shift
                        schedule(a, s, entered, -1, -1, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt, cr, bstream)
                        te = XT[s][a]
                    wx = XW[s][a]
                    if te <= ti_due and wx >= 0:
                        t = ex_tr[ex_ptr[s] + wx]
                        pt = pass_of(t, p, q, pos)
                for g in range(ntrig if t < 0 or q >= 0 else 0): # (A scheduled exit due before any move in the cascade is taken first, as the engine moves it before the cascade, at the start of the step)
                    tg = tr_trig[g]
                    if src_mask[tg, s] and TT[g][a] <= ti_delay:
                        pg = pass_of(tg, p, q, pos)
                        if t < 0 or (kind[t] != RATED and (pg < pt or (pg == pt and pos[tg] < pos[t]))):
                            t = tg
                            pt = pg
                            trig = g
                if t < 0:
                    break
            if split[t] and lazy[t] and np.isnan(E[a, dst_col[t] + 1]): # Its fractions are not worked out yet: stopped here, as it was
                m = n_pend[0]
                pend_a[m] = a
                pend_t[m] = t
                pend_s[m] = s
                pend_k[m] = 0 if first else 1
                pend_p[m] = pt if first else p
                pend_q[m] = q
                n_pend[0] = m + 1
                break
            t_in = -1
            p = pt
            q = pos[t]
            if trig >= 0:
                TT[trig][a] = np.nan # Taken now, whether or not it applies
                wi = trig_where[trig]
                if wi >= 0 and not WH[wi][a]:
                    continue
            frm = s
            if back[t] and p < npass: # Another pass, for a trigger before it that waits for its event
                again[p] = True

            # Where to: one destination, or one of several (beyond the last fraction, they stay)
            lo = dst_ptr[t]
            hi = dst_ptr[t+1]
            if split[t]:
                u = rand(tstream[t], t, p, a, cr)
                c = dst_col[t]
                dst = -2
                for i in range(hi - lo):
                    if u >= E[a, c + i] and u < E[a, c + i + 1]:
                        dst = i
                        break
                if dst < 0: # Stays
                    if kind[t] == SCHEDULED: # The exit was this transition's: never scheduled again (see ss.solvers.Agents.stay())
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
            wt = w[a] if weighted else 1.0
            flow[t] += wt
            flowdst[t, dst] += wt
            ev_a[n_ev] = a
            ev_t[n_ev] = t
            ev_s[n_ev] = frm
            ev_d[n_ev] = d
            ev_p[n_ev] = p
            n_ev += 1

            if d < 0: # A death (or a removal): out of every state of the module; the engine requests it
                for g in range(ntrig):
                    TT[g][a] = np.nan
                s = -1
                break

            # Enter
            S[d][a] = True
            dlive[d] += 1
            if has_left[frm]: # Dated as leaving the state they came from, as they enter (see ss.solvers.Agents.enter())
                LEFT[frm][a] = ti_enter
            if againx[t, d] and p < npass: # A scheduled exit of the state comes before this transition in the sweep: another pass
                again[p] = True
            if kept[d]:
                STI[d][a] = ti_enter
                SSH[d][a] = shift
            for g in range(ntrig): # A move after an event is forgotten for those no longer in its source; one on a condition is due now for those it holds for
                if src_mask[tr_trig[g], frm] and not src_mask[tr_trig[g], d]:
                    TT[g][a] = np.nan
                c = trig_cond[g]
                if c >= 0 and src_mask[tr_trig[g], d] and WH[c][a]:
                    TT[g][a] = -np.inf
            if comp_state[d] and has_late[d]: # The fraction of a step of its competing rates missed (see ss.solvers.Agents.enter()); in the state-change row, the engine draws half a step of them for those who arrived, after the call
                if mode == AT_START:
                    LATE[d][a] = 0.5
                elif mode == INFECTIONS: # Infected (see ss.solvers.Agents.enter())
                    LATE[d][a] = -0.5 if halved[d] else 0.0
                else:
                    LATE[d][a] = 1.0 if ti_enter == ti_delay else 0.0
            if sched[d]:
                XT[d][a] = np.nan
                schedule(a, d, ti_enter, t, p, XT, XW, ex_ptr, ex_tr, nbr, br_col, loops, dw, adj, aft, has_dw, E, fam, pcol, PA, PS, phase, dt, cr, bstream)
            for j in range(ae_ptr[t], ae_ptr[t+1]): # The moves a time after this event
                g = ae_trig[j]
                tg = tr_trig[g]
                if src_mask[tg, d] and np.isnan(TT[g][a]):
                    TT[g][a] = delay(g, a, tg, pass_of(tg, p, q, pos), trig_dist, fam, pcol, PA, PS, phase, dt, ti_delay, cr)
            s = d
    return n, n_ev
