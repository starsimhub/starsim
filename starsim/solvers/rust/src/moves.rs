//! The compiled moves of the agent engine (`starsim/solvers/backends/moves.py`, which is the reference): the agents with a move due at the start of a step
//! (`due_agents()`), and each agent's moves through the step (`step_moves()`), the same numbers bit for bit

use crate::draws::{crn_uniform, exp, key, log, log1p, max, ndtri};

/// Columns of one type, one for each state (or trigger): the module's arrays, read and written in place (they outlive the call)
pub struct Cols<T> {
    p: Vec<*mut T>,
}

impl<T: Copy> Cols<T> {
    pub fn new(p: Vec<*mut T>) -> Self {
        Cols { p }
    }
    #[inline(always)]
    fn get(&self, k: usize, a: usize) -> T {
        unsafe { *self.p[k].add(a) }
    }
    #[inline(always)]
    fn set(&self, k: usize, a: usize, v: T) {
        unsafe { *self.p[k].add(a) = v }
    }
    fn len(&self) -> usize {
        self.p.len()
    }
}

/// The streams the moves draw from, as the agent engine draws them (`moves.rand()`)
pub struct Crn<'a> {
    pub slots: &'a [i64],
    pub ind0: i64,
    pub rs: &'a [i64], // [stream, 1 + transition]
    pub rs_cols: usize,
    pub ush: &'a [u64],
    pub usc: &'a [f64],
    pub z32: &'a [bool],
    pub dstream: &'a [i64],
    pub bstream: &'a [i64],
    pub tstream: &'a [i64],
}

impl Crn<'_> {
    /// Agent `a`'s uniform in (0, 1) from stream `r`, in pass `ps` of the cascade and the round of transition `tag` (-1 for none) (`moves.rand()`)
    #[inline(always)]
    fn rand(&self, r: usize, tag: i64, ps: i64, a: usize) -> f64 {
        let slot = if self.slots.is_empty() { a as i64 } else { self.slots[a] };
        crn_uniform(key(self.rs[r*self.rs_cols + (tag + 1) as usize], self.ind0 + ps), slot as u64, self.ush[r], self.usc[r])
    }
}

/// The tables of values the moves read: distributions and their parameters, and the shares of branches and splits
pub struct Values<'a> {
    pub fam: &'a [i64],
    pub pcol: &'a [i64],
    pub pa: &'a [f64],
    pub pa_cols: usize,
    pub ps: &'a [f64],
    pub e: &'a [f64],
    pub e_cols: usize,
}

impl Values<'_> {
    /// Parameter `j` of a distribution for agent `a` (`moves.par()`)
    #[inline(always)]
    fn par(&self, j: usize, a: usize) -> f64 {
        let c = self.pcol[j];
        if c >= 0 { self.pa[a*self.pa_cols + c as usize] } else { self.ps[(-c - 1) as usize] }
    }

    /// A draw for agent `a` from distribution `d`, in the round of transition `tag` in pass `ps` (`moves.draw()`)
    fn draw(&self, d: usize, a: usize, tag: i64, ps: i64, cr: &Crn) -> f64 {
        let f = self.fam[d];
        if f == 0 {
            return self.par(3*d, a);
        }
        let r = cr.dstream[d] as usize;
        let u = cr.rand(r, tag, ps, a);
        let p0 = self.par(3*d, a);
        let p1 = self.par(3*d + 1, a);
        let p2 = self.par(3*d + 2, a);
        let z = |u: f64| if cr.z32[r] { ndtri(u) as f32 as f64 } else { ndtri(u) }; // SciPy's ndtri of a single-precision uniform is single precision
        match f {
            1 => p0 - p1*log1p(-u),
            2 => if p0 == 0.0 { p1 + p2 } else { p1*(p0*z(u)).exp() + p2 }, // NumPy's exp (libm's)
            3 => p0 + p1*z(u),
            4 => p0 + p1*u,
            5 => p2 + p1*exp(log(-log1p(-u))/p0),
            6 => ((u as f32)*(p1 as f32) + (p0 as f32)) as f64,
            _ => f64::NAN,
        }
    }

    #[inline(always)]
    fn edge(&self, a: usize, c: usize) -> f64 {
        self.e[a*self.e_cols + c]
    }
}

/// The graph's tables (`ss.solvers.Compiled.lay_out()`)
pub struct Graph<'a> {
    pub sched: &'a [bool],
    pub ex_ptr: &'a [i64],
    pub ex_tr: &'a [i64],
    pub nbr: &'a [i64],
    pub br_col: &'a [i64],
    pub loops: &'a [bool],
    pub pos: &'a [i64],
    pub kind: &'a [i64],
    pub split: &'a [bool],
    pub dst_ptr: &'a [i64],
    pub dst_state: &'a [i64],
    pub dst_col: &'a [i64],
    pub dw: &'a [i64],
    pub adj: &'a [i64],
    pub aft: &'a [f64],
    pub has_dw: &'a [bool],
    pub src_mask: &'a [bool], // [transition, state]
    pub ns: usize,
    pub ae_ptr: &'a [i64],
    pub ae_trig: &'a [i64],
    pub tr_trig: &'a [i64],
    pub trig_where: &'a [i64],
    pub trig_cond: &'a [i64],
    pub lazy: &'a [bool],
    pub trig_dist: &'a [i64],
    pub trig_fixed: &'a [bool],
    pub trig_mean: &'a [f64],
    pub comp_state: &'a [bool],
    pub has_late: &'a [bool],
    pub halved: &'a [bool],
    pub back: &'a [bool],
    pub againx: &'a [bool], // [transition, state]
}

impl Graph<'_> {
    #[inline(always)]
    fn src(&self, t: usize, s: usize) -> bool {
        self.src_mask[t*self.ns + s]
    }
}

/// The module's arrays that the moves read and write
pub struct State {
    pub s: Cols<bool>,
    pub sti: Cols<f32>,
    pub ssh: Cols<i8>,
    pub kept: Vec<bool>,
    pub left: Cols<f32>,
    pub has_left: Vec<bool>,
    pub xt: Cols<f32>,
    pub xw: Cols<i8>,
    pub tt: Cols<f32>,
    pub wh: Cols<bool>,
    pub late: Cols<f32>,
}

/// Agent `a` has entered state `s` at `entered`: its branch, dwell times, and scheduled exit (`moves.schedule()`)
#[allow(clippy::too_many_arguments)]
fn schedule(a: usize, s: usize, entered: f64, tag: i64, ps: i64, st: &State, g: &Graph, v: &Values, cr: &Crn, phase: &[f32], dt: f64) {
    let lo = g.ex_ptr[s] as usize;
    let hi = g.ex_ptr[s + 1] as usize;
    if lo == hi {
        return;
    }
    let mut best = f64::INFINITY;
    let mut which: i64 = -1;
    let nb = g.nbr[s] as usize;
    if nb > 0 {
        let u = cr.rand(cr.bstream[s] as usize, tag, ps, a);
        let c = g.br_col[s] as usize;
        for i in 0..nb {
            if u >= v.edge(a, c + i) && u < v.edge(a, c + i + 1) {
                let t = g.ex_tr[lo + i] as usize;
                if g.has_dw[t] {
                    best = v.draw(g.dw[t] as usize, a, tag, ps, cr)/v.par(3*g.adj[t] as usize, a) + g.aft[t];
                } else {
                    best = if g.aft[t] > 0.0 { g.aft[t] } else { f64::NEG_INFINITY };
                }
                which = i as i64;
                break;
            }
        }
    }
    for i in (lo + nb)..hi {
        let t = g.ex_tr[i] as usize;
        let d = v.draw(g.dw[t] as usize, a, tag, ps, cr)/v.par(3*g.adj[t] as usize, a) + g.aft[t];
        if d < best || which < 0 {
            best = d;
            which = (i - lo) as i64;
        }
    }
    let mut steps = (best/dt + phase[a] as f64).floor();
    if which >= 0 && g.loops[lo + which as usize] && best > 0.0 {
        steps = max(steps, 1.0);
    }
    st.xt.set(s, a, (entered + steps) as f32);
    st.xw.set(s, a, which as i8);
}

/// A delay after an event, drawn for agent `a` in the round of the trigger's turn (`moves.delay()`)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn delay(gi: usize, a: usize, tag: i64, ps: i64, g: &Graph, v: &Values, cr: &Crn, phase: &[f32], dt: f64, ti_delay: f64) -> f32 {
    let dur = if g.trig_fixed[gi] {
        v.draw(g.trig_dist[gi] as usize, a, tag, ps, cr)
    } else {
        -g.trig_mean[if g.trig_mean.len() > 1 { a } else { 0 }]*log1p(-cr.rand(cr.dstream[g.trig_dist[gi] as usize] as usize, tag, ps, a))
    };
    (ti_delay + (dur/dt + phase[a] as f64).floor()) as f32
}

/// The agents with a move due at the start of a step (`moves.due_agents()`)
#[allow(clippy::too_many_arguments)]
pub fn due_agents(s: &Cols<bool>, xt: &Cols<f32>, sched: &[bool], tt: &Cols<f32>, ntrig: usize, alive: &[bool], ti_due: f64, ti_delay: f64, mark: &mut [bool], idx: &mut [i64]) -> Vec<i64> {
    let n = alive.len();
    let mut m = 0usize;
    for k in 0..s.len() {
        if sched[k] {
            for a in 0..n {
                let x = xt.get(k, a) as f64;
                if s.get(k, a) && !(x > ti_due) && !mark[a] && alive[a] {
                    mark[a] = true;
                    idx[m] = a as i64;
                    m += 1;
                }
            }
        }
    }
    for gi in 0..ntrig {
        for a in 0..n {
            if (tt.get(gi, a) as f64) <= ti_delay && !mark[a] && alive[a] {
                mark[a] = true;
                idx[m] = a as i64;
                m += 1;
            }
        }
    }
    let mut out = idx[..m].to_vec();
    out.sort_unstable();
    for &a in &out {
        mark[a as usize] = false;
    }
    out
}

/// What the moves log and count
pub struct Log<'a> {
    pub ev_a: &'a mut [i64],
    pub ev_t: &'a mut [i64],
    pub ev_s: &'a mut [i64],
    pub ev_d: &'a mut [i64],
    pub ev_p: &'a mut [i64],
    pub flow: &'a mut [f64],
    pub flowdst: &'a mut [f64],
    pub flowdst_w: usize,
    pub dlive: &'a mut [i64],
}

/// Agents stopped at a split whose shares are not worked out yet (see `moves.step_moves()`)
pub struct Pend<'a> {
    pub a: &'a mut [i64],
    pub t: &'a mut [i64],
    pub s: &'a mut [i64],
    pub k: &'a mut [i64],
    pub p: &'a mut [i64],
    pub q: &'a mut [i64],
    pub n: &'a mut [i64],
}

/// The moves of some agents (`moves.step_moves()`, with `scan` false): returns where it stopped and the length of the log
#[allow(clippy::too_many_arguments)]
pub fn step_moves(agents: &[i64], arr_tr: &[i64], arr_src: &[i64], arr_kind: &[i64], arr_p: &[i64], arr_q: &[i64], start: usize, ti_due: f64, ti_delay: f64, ti_enter: f64, shift: f64,
                  alive: &[bool], st: &State, g: &Graph, v: &Values, cr: &Crn, again: &mut [bool], phase: &[f32], dt: f64, w: &[f32], weighted: bool,
                  log: &mut Log, mut n_ev: usize, max_hops: usize, mode: i64, pend: &mut Pend) -> (usize, usize) {
    let ns = st.s.len();
    let n = agents.len();
    let cap = log.ev_a.len();
    let ntrig = g.tr_trig.len();
    let npass = again.len() as i64;
    for idx in start..n {
        if n_ev + max_hops > cap {
            return (idx, n_ev);
        }
        let a = agents[idx] as usize;
        if !alive[a] {
            continue;
        }
        let mut s: i64 = -1;
        for k in 0..ns {
            if st.s.get(k, a) {
                s = k as i64;
                break;
            }
        }
        let mut t_in: i64 = -1;
        let mut p = arr_p[idx];
        let mut q = arr_q[idx];
        let ak = arr_kind[idx];
        if ak == 0 {
            let src = arr_src[idx] as usize;
            if st.s.get(src, a) {
                t_in = arr_tr[idx];
                s = src as i64;
            }
        } else if ak == 2 && s >= 0 {
            let te_ = arr_tr[idx] as usize;
            for j in (g.ae_ptr[te_] as usize)..(g.ae_ptr[te_ + 1] as usize) {
                let gi = g.ae_trig[j] as usize;
                let tg = g.tr_trig[gi];
                if g.src(tg as usize, s as usize) && st.tt.get(gi, a).is_nan() {
                    let pg = if g.pos[tg as usize] > q { p } else { p + 1 };
                    let d = delay(gi, a, tg, pg, g, v, cr, phase, dt, ti_delay);
                    st.tt.set(gi, a, d);
                }
            }
        }
        let mut hops = 0usize;
        loop {
            hops += 1;
            if hops > max_hops {
                panic!("moves with no wait did not settle within a step");
            }
            if s < 0 {
                break;
            }
            let su = s as usize;
            let t: usize;
            let pt: i64;
            let mut trig: i64 = -1;
            let first = t_in >= 0;
            if first {
                t = t_in as usize;
                pt = p;
            } else {
                let mut tt: i64 = -1;
                let mut pb: i64 = 0;
                if g.sched[su] {
                    let mut te = st.xt.get(su, a) as f64;
                    if te.is_nan() { // Put in the state with no exit scheduled: scheduled now, from when they entered it, as at the start of the step
                        let mut entered = if st.kept[su] { st.sti.get(su, a) as f64 } else { f64::NAN };
                        if entered.is_nan() { // With no time of entering: they entered it now, and are dated so
                            entered = ti_due;
                            if st.kept[su] {
                                st.sti.set(su, a, entered as f32);
                                st.ssh.set(su, a, shift as i8);
                            }
                        }
                        schedule(a, su, entered, -1, -1, st, g, v, cr, phase, dt);
                        te = st.xt.get(su, a) as f64;
                    }
                    let wx = st.xw.get(su, a);
                    if te <= ti_due && wx >= 0 {
                        tt = g.ex_tr[g.ex_ptr[su] as usize + wx as usize];
                        pb = if g.pos[tt as usize] > q { p } else { p + 1 };
                    }
                }
                let ng = if tt < 0 || q >= 0 { ntrig } else { 0 }; // A scheduled exit due before any move in the cascade is taken first
                for gi in 0..ng {
                    let tg = g.tr_trig[gi];
                    if g.src(tg as usize, su) && (st.tt.get(gi, a) as f64) <= ti_delay {
                        let pg = if g.pos[tg as usize] > q { p } else { p + 1 };
                        if tt < 0 || (g.kind[tt as usize] != 3 && (pg < pb || (pg == pb && g.pos[tg as usize] < g.pos[tt as usize]))) {
                            tt = tg;
                            pb = pg;
                            trig = gi as i64;
                        }
                    }
                }
                if tt < 0 {
                    break;
                }
                t = tt as usize;
                pt = pb;
            }
            if g.split[t] && g.lazy[t] && v.edge(a, g.dst_col[t] as usize + 1).is_nan() { // Its shares are not worked out yet: stopped here, as it was
                let m = pend.n[0] as usize;
                pend.a[m] = a as i64;
                pend.t[m] = t as i64;
                pend.s[m] = su as i64;
                pend.k[m] = if first { 0 } else { 1 };
                pend.p[m] = if first { pt } else { p };
                pend.q[m] = q;
                pend.n[0] = m as i64 + 1;
                break;
            }
            t_in = -1;
            p = pt;
            q = g.pos[t];
            if trig >= 0 {
                let gi = trig as usize;
                st.tt.set(gi, a, f32::NAN);
                let wi = g.trig_where[gi];
                if wi >= 0 && !st.wh.get(wi as usize, a) {
                    continue;
                }
            }
            let frm = su;
            if g.back[t] && p < npass {
                again[p as usize] = true;
            }

            // Where to
            let lo = g.dst_ptr[t] as usize;
            let hi = g.dst_ptr[t + 1] as usize;
            let dst: usize;
            if g.split[t] {
                let u = cr.rand(cr.tstream[t] as usize, t as i64, p, a);
                let c = g.dst_col[t] as usize;
                let mut chosen: i64 = -2;
                for i in 0..(hi - lo) {
                    if u >= v.edge(a, c + i) && u < v.edge(a, c + i + 1) {
                        chosen = i as i64;
                        break;
                    }
                }
                if chosen < 0 { // Stays
                    if g.kind[t] == 0 {
                        st.xt.set(frm, a, f32::INFINITY);
                        st.xw.set(frm, a, -2);
                    }
                    continue;
                }
                dst = chosen as usize;
            } else {
                dst = 0;
            }
            let d = g.dst_state[lo + dst];

            // Leave
            st.s.set(frm, a, false);
            log.dlive[frm] -= 1;
            if g.sched[frm] {
                st.xt.set(frm, a, f32::NAN);
            }
            let wt = if weighted { w[a] as f64 } else { 1.0 };
            log.flow[t] += wt;
            log.flowdst[t*log.flowdst_w + dst] += wt;
            log.ev_a[n_ev] = a as i64;
            log.ev_t[n_ev] = t as i64;
            log.ev_s[n_ev] = frm as i64;
            log.ev_d[n_ev] = d;
            log.ev_p[n_ev] = p;
            n_ev += 1;

            if d < 0 { // A death
                for gi in 0..ntrig {
                    st.tt.set(gi, a, f32::NAN);
                }
                break;
            }
            let du = d as usize;

            // Enter
            st.s.set(du, a, true);
            log.dlive[du] += 1;
            if st.has_left[frm] {
                st.left.set(frm, a, ti_enter as f32);
            }
            if g.againx[t*ns + du] && p < npass {
                again[p as usize] = true;
            }
            if st.kept[du] {
                st.sti.set(du, a, ti_enter as f32);
                st.ssh.set(du, a, shift as i8);
            }
            for gi in 0..ntrig {
                let tg = g.tr_trig[gi] as usize;
                if g.src(tg, frm) && !g.src(tg, du) {
                    st.tt.set(gi, a, f32::NAN);
                }
                let c = g.trig_cond[gi];
                if c >= 0 && g.src(tg, du) && st.wh.get(c as usize, a) { // A trigger on a condition: due now for those it holds for
                    st.tt.set(gi, a, f32::NEG_INFINITY);
                }
            }
            if g.comp_state[du] && g.has_late[du] {
                let val = if mode == 0 { 0.5 } else if mode == 1 { if g.halved[du] { -0.5 } else { 0.0 } } else if ti_enter == ti_delay { 1.0 } else { 0.0 };
                st.late.set(du, a, val);
            }
            if g.sched[du] {
                st.xt.set(du, a, f32::NAN);
                schedule(a, du, ti_enter, t as i64, p, st, g, v, cr, phase, dt);
            }
            for j in (g.ae_ptr[t] as usize)..(g.ae_ptr[t + 1] as usize) {
                let gi = g.ae_trig[j] as usize;
                let tg = g.tr_trig[gi];
                if g.src(tg as usize, du) && st.tt.get(gi, a).is_nan() {
                    let pg = if g.pos[tg as usize] > q { p } else { p + 1 };
                    let dl = delay(gi, a, tg, pg, g, v, cr, phase, dt, ti_delay);
                    st.tt.set(gi, a, dl);
                }
            }
            s = d;
        }
    }
    (n, n_ev)
}

pub mod bind {
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use crate::moves;
    use crate::py::{arr, flat, sl, sl_mut};

    type Arr<'py> = Bound<'py, PyAny>;

    /// A tuple of the module's arrays (one per state, or trigger), all of one type and contiguous, as columns written in place
    fn cols<T: numpy::Element + Copy>(t: &Bound<PyTuple>) -> PyResult<moves::Cols<T>> {
        let mut p = Vec::with_capacity(t.len());
        for o in t.iter() {
            let a = o.cast::<PyArrayDyn<T>>()?;
            if !a.is_c_contiguous() {
                return Err(PyTypeError::new_err("the arrays of the Rust kernels must be contiguous"));
            }
            p.push(a.data());
        }
        Ok(moves::Cols::new(p))
    }

    /// A 2-D table as a flat slice by rows, and its number of columns
    fn table<'a>(o: &'a Arr) -> PyResult<(&'a [f64], usize)> {
        let a = o.cast::<PyArrayDyn<f64>>()?;
        if a.ndim() != 2 {
            return Err(PyTypeError::new_err("a table of the compiled moves must be 2-D"));
        }
        let cols = a.shape()[1];
        Ok((flat(o)?, cols))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn due_agents<'py>(py: Python<'py>, s: &Bound<'py, PyTuple>, xt: &Bound<'py, PyTuple>, sched: &Arr<'py>, tt: &Bound<'py, PyTuple>, ntrig: usize, alive: &Arr<'py>,
                       ti_due: f64, ti_delay: f64, mark: &Arr<'py>, idx: &Arr<'py>) -> PyResult<Arr<'py>> {
        let out = moves::due_agents(&cols::<bool>(s)?, &cols::<f32>(xt)?, sl(sched)?, &cols::<f32>(tt)?, ntrig, sl(alive)?, ti_due, ti_delay, sl_mut(mark)?, sl_mut(idx)?);
        Ok(arr(py, out))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (agents, arr_tr, arr_src, arr_kind, arr_p, arr_q, scan, start, ti_due, ti_delay, ti_enter, shift, alive,
                        s, sti, ssh, kept, left, has_left, xt, xw, sched,
                        ex_ptr, ex_tr, nbr, br_col, loops,
                        pos, kind, split, dst_ptr, dst_state, dst_col, dw, adj, aft, has_dw, src_mask, ae_ptr, ae_trig,
                        tr_trig, tt, trig_where, wh, trig_dist, trig_fixed, trig_mean,
                        fam, pcol, pa, ps, e, phase, dt, w, weighted,
                        slots, ind0, rs, ush, usc, z32, dstream, bstream, tstream, back, againx, again,
                        ev_a, ev_t, ev_s, ev_d, ev_p, n_ev, flow, flowdst, dlive, max_hops,
                        mode, comp_state, late, has_late, halved, trig_cond, lazy, pend_a, pend_t, pend_s, pend_k, pend_p, pend_q, n_pend))]
    fn step_moves<'py>(agents: &Arr<'py>, arr_tr: &Arr<'py>, arr_src: &Arr<'py>, arr_kind: &Arr<'py>, arr_p: &Arr<'py>, arr_q: &Arr<'py>, scan: bool, start: usize, ti_due: f64, ti_delay: f64, ti_enter: f64, shift: f64,
                       alive: &Arr<'py>,
                       s: &Bound<'py, PyTuple>, sti: &Bound<'py, PyTuple>, ssh: &Bound<'py, PyTuple>, kept: &Arr<'py>, left: &Bound<'py, PyTuple>, has_left: &Arr<'py>,
                       xt: &Bound<'py, PyTuple>, xw: &Bound<'py, PyTuple>, sched: &Arr<'py>,
                       ex_ptr: &Arr<'py>, ex_tr: &Arr<'py>, nbr: &Arr<'py>, br_col: &Arr<'py>, loops: &Arr<'py>,
                       pos: &Arr<'py>, kind: &Arr<'py>, split: &Arr<'py>, dst_ptr: &Arr<'py>, dst_state: &Arr<'py>, dst_col: &Arr<'py>, dw: &Arr<'py>, adj: &Arr<'py>, aft: &Arr<'py>,
                       has_dw: &Arr<'py>, src_mask: &Arr<'py>, ae_ptr: &Arr<'py>, ae_trig: &Arr<'py>,
                       tr_trig: &Arr<'py>, tt: &Bound<'py, PyTuple>, trig_where: &Arr<'py>, wh: &Bound<'py, PyTuple>, trig_dist: &Arr<'py>, trig_fixed: &Arr<'py>, trig_mean: &Arr<'py>,
                       fam: &Arr<'py>, pcol: &Arr<'py>, pa: &Arr<'py>, ps: &Arr<'py>, e: &Arr<'py>, phase: &Arr<'py>, dt: f64, w: &Arr<'py>, weighted: bool,
                       slots: &Arr<'py>, ind0: i64, rs: &Arr<'py>, ush: &Arr<'py>, usc: &Arr<'py>, z32: &Arr<'py>, dstream: &Arr<'py>, bstream: &Arr<'py>, tstream: &Arr<'py>,
                       back: &Arr<'py>, againx: &Arr<'py>, again: &Arr<'py>,
                       ev_a: &Arr<'py>, ev_t: &Arr<'py>, ev_s: &Arr<'py>, ev_d: &Arr<'py>, ev_p: &Arr<'py>, n_ev: usize, flow: &Arr<'py>, flowdst: &Arr<'py>, dlive: &Arr<'py>, max_hops: usize,
                       mode: i64, comp_state: &Arr<'py>, late: &Bound<'py, PyTuple>, has_late: &Arr<'py>, halved: &Arr<'py>, trig_cond: &Arr<'py>,
                       lazy: &Arr<'py>, pend_a: &Arr<'py>, pend_t: &Arr<'py>, pend_s: &Arr<'py>, pend_k: &Arr<'py>, pend_p: &Arr<'py>, pend_q: &Arr<'py>, n_pend: &Arr<'py>) -> PyResult<(usize, usize)> {
        if scan {
            return Err(PyTypeError::new_err("the compiled moves take a list of agents (scan=False)"));
        }
        let ns = s.len();
        let st = moves::State {
            s: cols(s)?, sti: cols(sti)?, ssh: cols(ssh)?, kept: sl::<bool>(kept)?.to_vec(), left: cols(left)?, has_left: sl::<bool>(has_left)?.to_vec(),
            xt: cols(xt)?, xw: cols(xw)?, tt: cols(tt)?, wh: cols(wh)?, late: cols(late)?,
        };
        let g = moves::Graph {
            sched: sl(sched)?, ex_ptr: sl(ex_ptr)?, ex_tr: sl(ex_tr)?, nbr: sl(nbr)?, br_col: sl(br_col)?, loops: sl(loops)?,
            pos: sl(pos)?, kind: sl(kind)?, split: sl(split)?, dst_ptr: sl(dst_ptr)?, dst_state: sl(dst_state)?, dst_col: sl(dst_col)?,
            dw: sl(dw)?, adj: sl(adj)?, aft: sl(aft)?, has_dw: sl(has_dw)?, src_mask: flat(src_mask)?, ns,
            ae_ptr: sl(ae_ptr)?, ae_trig: sl(ae_trig)?, tr_trig: sl(tr_trig)?, trig_where: sl(trig_where)?, trig_cond: sl(trig_cond)?, lazy: sl(lazy)?, trig_dist: sl(trig_dist)?, trig_fixed: sl(trig_fixed)?, trig_mean: sl(trig_mean)?,
            comp_state: sl(comp_state)?, has_late: sl(has_late)?, halved: sl(halved)?, back: sl(back)?, againx: flat(againx)?,
        };
        let (pa, pa_cols) = table(pa)?;
        let (e, e_cols) = table(e)?;
        let v = moves::Values { fam: sl(fam)?, pcol: sl(pcol)?, pa, pa_cols, ps: sl(ps)?, e, e_cols };
        let rsa = rs.cast::<PyArrayDyn<i64>>()?;
        if rsa.ndim() != 2 {
            return Err(PyTypeError::new_err("the seeds of the compiled moves' streams must be a 2-D table"));
        }
        let cr = moves::Crn { slots: sl(slots)?, ind0, rs: flat(rs)?, rs_cols: rsa.shape()[1], ush: sl(ush)?, usc: sl(usc)?, z32: sl(z32)?,
                              dstream: sl(dstream)?, bstream: sl(bstream)?, tstream: sl(tstream)? };
        let fd = flowdst.cast::<PyArrayDyn<f64>>()?;
        let flowdst_w = if fd.ndim() == 2 { fd.shape()[1] } else { 1 };
        let mut log = moves::Log { ev_a: sl_mut(ev_a)?, ev_t: sl_mut(ev_t)?, ev_s: sl_mut(ev_s)?, ev_d: sl_mut(ev_d)?, ev_p: sl_mut(ev_p)?, flow: sl_mut(flow)?,
                                   flowdst: crate::py::flat_mut(flowdst)?, flowdst_w, dlive: sl_mut(dlive)? };
        Ok(moves::step_moves(sl(agents)?, sl(arr_tr)?, sl(arr_src)?, sl(arr_kind)?, sl(arr_p)?, sl(arr_q)?, start, ti_due, ti_delay, ti_enter, shift,
                             sl(alive)?, &st, &g, &v, &cr, sl_mut(again)?, sl(phase)?, dt, sl(w)?, weighted, &mut log, n_ev, max_hops, mode,
                             &mut moves::Pend { a: sl_mut(pend_a)?, t: sl_mut(pend_t)?, s: sl_mut(pend_s)?, k: sl_mut(pend_k)?, p: sl_mut(pend_p)?, q: sl_mut(pend_q)?, n: sl_mut(n_pend)? }))
    }

    pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(due_agents, m)?)?;
        m.add_function(wrap_pyfunction!(step_moves, m)?)?;
        Ok(())
    }
}
