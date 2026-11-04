//! The pass of agents taken event by event in continuous time (`solver='abm-events'`) of `starsim/solvers/backends/events.py`: the same
//! arguments and results, the same draws (`rand()`, a hash of the step's key, the agent and its count of draws), the same heap, and the
//! same arithmetic in the same order, so the same events and numbers, bit for bit. `log` is the C library's, as Numba's `np.log` is.

use crate::draws::{GOLDEN, MIX1, MIX2, TWO53};

/// An entry of the heap: its time, the agent, its kind (-1 for the agent's own exit, else the route of a contact), the agent's version when it was queued, and the transition of an exit
#[derive(Clone, Copy)]
struct Entry {
    t: f64,
    a: usize,
    k: i64,
    v: i64,
    r: i64,
}

/// The heap of events, as Numba's arrays: the same moves, so entries with the same time come off in the same order
struct Heap(Vec<Entry>);

impl Heap {
    /// Push an entry (`events.push()`)
    #[inline(always)]
    fn push(&mut self, e: Entry) {
        let h = &mut self.0;
        h.push(e);
        let mut i = h.len() - 1;
        while i > 0 {
            let p = (i - 1) >> 1;
            if h[p].t <= e.t {
                break;
            }
            h[i] = h[p];
            i = p;
        }
        h[i] = e;
    }

    /// Take the earliest entry off the heap, once it has been read at index 0 (`events.pop()`)
    #[inline(always)]
    fn pop(&mut self) {
        let h = &mut self.0;
        let e = h.pop().unwrap();
        let size = h.len();
        if size == 0 {
            return;
        }
        let mut i = 0;
        loop {
            let mut c = 2*i + 1;
            if c >= size {
                break;
            }
            if c + 1 < size && h[c + 1].t < h[c].t {
                c += 1;
            }
            if h[c].t >= e.t {
                break;
            }
            h[i] = h[c];
            i = c;
        }
        h[i] = e;
    }
}

/// Agent `a`'s next uniform in (0, 1) of the step: SplitMix64's hash of the step's key, the agent, and its count of draws (`events.rand()`)
#[inline(always)]
fn rand(key: u64, a: usize, ctr: &mut [u64]) -> f64 {
    let mut z = key.wrapping_add((a as u64).wrapping_mul(GOLDEN)).wrapping_add(ctr[a].wrapping_mul(0xD1B54A32D192ED03));
    ctr[a] += 1;
    z = (z ^ (z >> 30)).wrapping_mul(MIX1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX2);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5)/TWO53 // The top 53 bits, never 0 or 1
}

/// What the pass reads and does not change: the arguments of `events.step_events()` (see its docstring), the tables by rows
pub struct Pass<'a> {
    pub haz_ptr: &'a [i64], pub haz_tr: &'a [i64], pub haz_rate: &'a [f64], pub haz_row: &'a [i64], pub rate_rows: &'a [f64],
    pub br_ptr: &'a [i64], pub br_tr: &'a [i64], pub br_cum: &'a [f64], pub tm_ptr: &'a [i64], pub tm_tr: &'a [i64],
    pub dw_kind: &'a [i64], pub dw_k: &'a [i64], pub dw_mean: &'a [f64], pub dw_row: &'a [i64], pub pool: &'a [f64],
    pub dst_ptr: &'a [i64], pub dst_state: &'a [i64], pub dst_cum: &'a [f64], pub inf_tr: i64,
    pub sus_state: &'a [f64], pub sus_agent: &'a [f64], pub w_agent: &'a [f64],
    pub r_kind: &'a [i64], pub x_off: &'a [i64], pub r_n: &'a [i64], pub r_scale: &'a [f64], pub r_bound: &'a [f64], pub share: &'a [f64],
    pub s1: &'a [i64], pub e1: &'a [i64], pub o1: &'a [i64], pub s2: &'a [i64], pub e2: &'a [i64], pub o2: &'a [i64], pub r_deg: &'a [i64], pub w1: &'a [f64], pub w2: &'a [f64], pub beta_off: &'a [i64],
    pub members: &'a [i64], pub mem_eff: &'a [f64], pub mem_off: &'a [i64], pub mem_n: &'a [i64], pub a_ctc: &'a [f64],
    pub r_blk: &'a [i64], pub a_cell: &'a [i64], pub c_ptr: &'a [i64], pub c_first: &'a [i64], pub c_row: &'a [i64], pub c_cum: &'a [f64],
    pub n: usize, // The number of agents, the length of each row of rate_rows, pool and a_ctc
    pub ns: usize, // The number of states, the length of each row of share
}

/// The events of a step, as `events.step_events()` returns them
pub struct Events {
    pub a: Vec<i64>, pub tr: Vec<i64>, pub src: Vec<i64>, pub dst: Vec<i64>, pub t: Vec<f64>, pub by: Vec<i64>, pub rt: Vec<i64>,
    pub over: bool,
}

impl Pass<'_> {
    /// A dwell time of agent `a` for transition `tr`: none, Erlang with a fixed mean, or the one drawn for the agent (`events.dwell()`)
    #[inline(always)]
    fn dwell(&self, tr: usize, a: usize, key: u64, ctr: &mut [u64]) -> f64 {
        match self.dw_kind[tr] {
            0 => 0.0,
            1 => {
                let k = self.dw_k[tr];
                let mut tot = 0.0;
                for _ in 0..k {
                    tot -= rand(key, a, ctr).ln();
                }
                tot*self.dw_mean[tr]/k as f64
            }
            _ => self.pool[self.dw_row[tr] as usize*self.n + a],
        }
    }

    /// The scheduled exit of agent `a` entering state `s` at time `t` (`events.draw_exit()`)
    fn draw_exit(&self, a: usize, s: usize, t: f64, key: u64, ctr: &mut [u64]) -> (f64, i64) {
        let mut te = f64::INFINITY;
        let mut tx = -1;
        let (b0, b1) = (self.br_ptr[s] as usize, self.br_ptr[s + 1] as usize);
        if b0 < b1 {
            let u = rand(key, a, ctr);
            let mut j = b0;
            while j < b1 - 1 && u >= self.br_cum[j] {
                j += 1;
            }
            tx = self.br_tr[j];
            te = t + self.dwell(tx as usize, a, key, ctr);
        }
        for j in self.tm_ptr[s] as usize..self.tm_ptr[s + 1] as usize {
            let d = t + self.dwell(self.tm_tr[j] as usize, a, key, ctr);
            if d < te || tx < 0 {
                te = d;
                tx = self.tm_tr[j];
            }
        }
        (te, tx)
    }

    /// The competing rate of exit `j` for agent `a`
    #[inline(always)]
    fn haz(&self, j: usize, a: usize) -> f64 {
        let row = self.haz_row[j];
        if row < 0 { self.haz_rate[j] } else { self.rate_rows[row as usize*self.n + a] }
    }

    /// Queue an agent's next exit, if within the step, and its first contact on each route it is a source on (`events.queue()`)
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn queue(&self, heap: &mut Heap, a: usize, s: usize, t: f64, t1: f64, key: u64, ctr: &mut [u64], ver: i64, t_exit: f64, exit_tr: i64) {
        let (h0, h1) = (self.haz_ptr[s] as usize, self.haz_ptr[s + 1] as usize);
        let mut tot = 0.0;
        for j in h0..h1 {
            tot += self.haz(j, a);
        }
        let mut tn = t_exit;
        let mut trn = exit_tr;
        if tot > 0.0 {
            let th = t - rand(key, a, ctr).ln()/tot;
            if th < tn {
                tn = th;
                let u = rand(key, a, ctr)*tot;
                let mut acc = 0.0;
                for j in h0..h1 {
                    acc += self.haz(j, a);
                    trn = self.haz_tr[j];
                    if u < acc {
                        break;
                    }
                }
            }
        }
        if tn < t1 && trn >= 0 {
            heap.push(Entry { t: tn, a, k: -1, v: ver, r: trn });
        }
        for r in 0..self.r_kind.len() {
            let rate = self.contact_rate(a, s, r);
            if rate > 0.0 {
                heap.push(Entry { t: t - rand(key, a, ctr).ln()/rate, a, k: r as i64, v: ver, r: -1 });
            }
        }
    }

    /// The number of neighbors of agent `a` on network `r` (`events.degree()`)
    #[inline(always)]
    fn degree(&self, a: usize, r: usize) -> i64 {
        if (a as i64) < self.r_n[r] { self.r_deg[self.x_off[r] as usize + a] } else { 0 }
    }

    /// The `j`th neighbor of agent `a` on network `r`, and the weight of the edge toward them (`events.neighbor()`)
    #[inline(always)]
    fn neighbor(&self, a: usize, j: i64, r: usize) -> (i64, f64) {
        let i = self.x_off[r] as usize + a;
        let off = self.beta_off[r];
        let d1 = self.s1[i + 1] - self.s1[i];
        if j < d1 {
            let m = (off + self.s1[i] + j) as usize;
            return (self.o1[m], self.w1[(off + self.e1[m]) as usize]);
        }
        let m = (off + self.s2[i] + j - d1) as usize;
        (self.o2[m], self.w2[(off + self.e2[m]) as usize])
    }

    /// The rate of the contacts of an agent in state `s` on route `r`, at their bound (`events.contact_rate()`)
    #[inline(always)]
    fn contact_rate(&self, a: usize, s: usize, r: usize) -> f64 {
        let sh = self.share[r*self.ns + s]*self.w_agent[a];
        if sh <= 0.0 {
            return 0.0;
        }
        let deg = if self.r_kind[r] == 0 { self.degree(a, r) as f64 } else { self.a_ctc[r*self.n + a] };
        self.r_scale[r]*sh*deg*self.r_bound[r]
    }
}

/// One of `deg` items, by a uniform: `min(int(u*deg), deg - 1)`
#[inline(always)]
fn pick(u: f64, deg: i64) -> i64 {
    ((u*deg as f64) as i64).min(deg - 1)
}

/// Advance the agents of one module through a step, [t0, t1), event by event (`events.step_events()`): changes state, ver, t_exit and exit_tr in place
#[allow(clippy::too_many_arguments)]
pub fn step_events(p: &Pass, t0: f64, t1: f64, seed: u64, state: &mut [i64], ver: &mut [i64], t_exit: &mut [f64], exit_tr: &mut [i64], fresh: &[bool]) -> Events {
    let n = state.len();
    let nr = p.r_kind.len();
    let key = seed;
    let mut ctr = vec![0u64; n]; // Each agent's draws so far in the step
    let alive = state.iter().filter(|&&s| s >= 0).count();
    let mut heap = Heap(Vec::with_capacity(1024 + alive*(1 + nr)));
    let busy: Vec<bool> = (0..p.ns).map(|s| p.haz_ptr[s + 1] > p.haz_ptr[s] || (0..nr).any(|r| p.share[r*p.ns + s] > 0.0)).collect(); // The states with competing rates or contacts
    for a in 0..n { // Everyone's next exit and first contacts
        let s = state[a];
        if s < 0 {
            continue;
        }
        let s = s as usize;
        if fresh[a] {
            ver[a] += 1;
            (t_exit[a], exit_tr[a]) = p.draw_exit(a, s, t0, key, &mut ctr);
        }
        if !busy[s] && t_exit[a] >= t1 {
            continue;
        }
        p.queue(&mut heap, a, s, t0, t1, key, &mut ctr, ver[a], t_exit[a], exit_tr[a]);
    }
    let cap = 1024;
    let mut ev = Events { a: Vec::with_capacity(cap), tr: Vec::with_capacity(cap), src: Vec::with_capacity(cap), dst: Vec::with_capacity(cap),
                          t: Vec::with_capacity(cap), by: Vec::with_capacity(cap), rt: Vec::with_capacity(cap), over: false };
    while let Some(&Entry { t, a, k, v, r }) = heap.0.first() {
        if t >= t1 {
            break;
        }
        heap.pop();
        if v != ver[a] { // It has moved since this was queued
            continue;
        }
        let s = state[a] as usize;
        let mut b = a; // Who moves now, by transition trn
        let mut trn = r;
        let mut by = -1;
        if k >= 0 { // A contact on route k: of whom, and whether it infects them
            let k = k as usize;
            let x;
            match p.r_kind[k] {
                0 => {
                    let deg = p.degree(a, k);
                    let (o, w) = p.neighbor(a, pick(rand(key, a, &mut ctr), deg), k);
                    b = o as usize;
                    x = w;
                }
                1 => {
                    let m = (p.mem_off[k] + pick(rand(key, a, &mut ctr), p.mem_n[k])) as usize;
                    b = p.members[m] as usize;
                    x = p.mem_eff[m]; // Their contacts in this pool, which bound its thinning
                }
                _ => { // A contact matrix: a cell drawn by its weight, then one of its members
                    let c = p.a_cell[p.x_off[k] as usize + a] as usize;
                    let row = p.c_row[c] as usize;
                    let nb = p.r_blk[k] as usize;
                    let u = rand(key, a, &mut ctr)*p.c_cum[row + nb - 1];
                    let mut i = 0;
                    while i < nb - 1 && u >= p.c_cum[row + i] {
                        i += 1;
                    }
                    let g = (p.c_first[c] + i as i64) as usize;
                    let deg = p.c_ptr[g + 1] - p.c_ptr[g];
                    let m = p.c_ptr[g] + pick(rand(key, a, &mut ctr), deg);
                    if deg > 0 {
                        b = p.members[m as usize] as usize;
                    }
                    x = 1.0;
                }
            }
            let sb = state[b];
            let prob = if sb < 0 { 0.0 } else { p.sus_state[sb as usize]*p.sus_agent[b]*x/p.r_bound[k] };
            let rate = p.contact_rate(a, s, k); // The next contact of the same agent
            heap.push(Entry { t: t - rand(key, a, &mut ctr).ln()/rate, a, k: k as i64, v: ver[a], r: -1 });
            if prob > 1.0 {
                ev.over = true;
            }
            let u = rand(key, a, &mut ctr); // Drawn whether or not the contact can infect
            if !(prob > 0.0 && u < prob) {
                continue;
            }
            trn = p.inf_tr;
            by = a as i64;
        }

        // The move, then the exits and contacts of the state it enters
        let u = rand(key, b, &mut ctr);
        let tr = trn as usize;
        let mut j = p.dst_ptr[tr] as usize;
        let end = p.dst_ptr[tr + 1] as usize;
        while j < end - 1 && u >= p.dst_cum[j] {
            j += 1;
        }
        let d = p.dst_state[j];
        ev.a.push(b as i64); ev.tr.push(trn); ev.src.push(state[b]); ev.dst.push(d); ev.t.push(t); ev.by.push(by); ev.rt.push(if by >= 0 { k } else { -1 });
        state[b] = d;
        ver[b] += 1;
        if d < 0 { // Died
            t_exit[b] = f64::INFINITY;
            exit_tr[b] = -1;
            continue;
        }
        let d = d as usize;
        (t_exit[b], exit_tr[b]) = p.draw_exit(b, d, t, key, &mut ctr);
        p.queue(&mut heap, b, d, t, t1, key, &mut ctr, ver[b], t_exit[b], exit_tr[b]);
    }
    ev
}


/// The binding, with the same arguments and results as `events.step_events()`
pub mod bind {
    use numpy::{PyArrayDyn, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use crate::events;
    use crate::py::{arr, flat, sl, sl_mut};

    type Arr<'py> = Bound<'py, PyAny>;

    /// A table (a 2-D array of floats) as a flat slice by rows, checked to have rows of length `cols`
    fn table<'a>(o: &'a Arr, cols: usize) -> PyResult<&'a [f64]> {
        let a = o.cast::<PyArrayDyn<f64>>()?;
        if a.ndim() != 2 || a.shape()[1] != cols {
            return Err(PyTypeError::new_err("a table of the events' pass with the wrong shape"));
        }
        flat(o)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (t0, t1, seed, state, ver, t_exit, exit_tr, fresh, haz_ptr, haz_tr, haz_rate, haz_row, rate_rows, br_ptr, br_tr, br_cum, tm_ptr, tm_tr,
                        dw_kind, dw_k, dw_mean, dw_row, pool, dst_ptr, dst_state, dst_cum, inf_tr, sus_state, sus_agent, w_agent,
                        r_kind, x_off, r_n, r_scale, r_bound, share, s1, e1, o1, s2, e2, o2, r_deg, w1, w2, beta_off,
                        members, mem_eff, mem_off, mem_n, a_ctc, r_blk, a_cell, c_ptr, c_first, c_row, c_cum))]
    fn step_events<'py>(py: Python<'py>, t0: f64, t1: f64, seed: &Arr<'py>, state: &Arr<'py>, ver: &Arr<'py>, t_exit: &Arr<'py>, exit_tr: &Arr<'py>, fresh: &Arr<'py>,
                        haz_ptr: &Arr<'py>, haz_tr: &Arr<'py>, haz_rate: &Arr<'py>, haz_row: &Arr<'py>, rate_rows: &Arr<'py>,
                        br_ptr: &Arr<'py>, br_tr: &Arr<'py>, br_cum: &Arr<'py>, tm_ptr: &Arr<'py>, tm_tr: &Arr<'py>,
                        dw_kind: &Arr<'py>, dw_k: &Arr<'py>, dw_mean: &Arr<'py>, dw_row: &Arr<'py>, pool: &Arr<'py>,
                        dst_ptr: &Arr<'py>, dst_state: &Arr<'py>, dst_cum: &Arr<'py>, inf_tr: i64,
                        sus_state: &Arr<'py>, sus_agent: &Arr<'py>, w_agent: &Arr<'py>,
                        r_kind: &Arr<'py>, x_off: &Arr<'py>, r_n: &Arr<'py>, r_scale: &Arr<'py>, r_bound: &Arr<'py>, share: &Arr<'py>,
                        s1: &Arr<'py>, e1: &Arr<'py>, o1: &Arr<'py>, s2: &Arr<'py>, e2: &Arr<'py>, o2: &Arr<'py>, r_deg: &Arr<'py>, w1: &Arr<'py>, w2: &Arr<'py>, beta_off: &Arr<'py>,
                        members: &Arr<'py>, mem_eff: &Arr<'py>, mem_off: &Arr<'py>, mem_n: &Arr<'py>, a_ctc: &Arr<'py>,
                        r_blk: &Arr<'py>, a_cell: &Arr<'py>, c_ptr: &Arr<'py>, c_first: &Arr<'py>, c_row: &Arr<'py>, c_cum: &Arr<'py>)
                        -> PyResult<(Arr<'py>, Arr<'py>, Arr<'py>, Arr<'py>, Arr<'py>, Arr<'py>, Arr<'py>, bool)> {
        let seed = match seed.extract::<u64>() { // A np.uint64, as Numba's np.uint64(seed) takes it
            Ok(x) => x,
            Err(_) => seed.extract::<i64>()? as u64,
        };
        let (st, vr, te, tx) = (sl_mut::<i64>(state)?, sl_mut::<i64>(ver)?, sl_mut::<f64>(t_exit)?, sl_mut::<i64>(exit_tr)?);
        let n = st.len();
        let hp = sl::<i64>(haz_ptr)?;
        let ns = hp.len() - 1;
        let p = events::Pass {
            haz_ptr: hp, haz_tr: sl(haz_tr)?, haz_rate: sl(haz_rate)?, haz_row: sl(haz_row)?, rate_rows: table(rate_rows, n)?,
            br_ptr: sl(br_ptr)?, br_tr: sl(br_tr)?, br_cum: sl(br_cum)?, tm_ptr: sl(tm_ptr)?, tm_tr: sl(tm_tr)?,
            dw_kind: sl(dw_kind)?, dw_k: sl(dw_k)?, dw_mean: sl(dw_mean)?, dw_row: sl(dw_row)?, pool: table(pool, n)?,
            dst_ptr: sl(dst_ptr)?, dst_state: sl(dst_state)?, dst_cum: sl(dst_cum)?, inf_tr,
            sus_state: sl(sus_state)?, sus_agent: sl(sus_agent)?, w_agent: sl(w_agent)?,
            r_kind: sl(r_kind)?, x_off: sl(x_off)?, r_n: sl(r_n)?, r_scale: sl(r_scale)?, r_bound: sl(r_bound)?, share: table(share, ns)?,
            s1: sl(s1)?, e1: sl(e1)?, o1: sl(o1)?, s2: sl(s2)?, e2: sl(e2)?, o2: sl(o2)?, r_deg: sl(r_deg)?, w1: sl(w1)?, w2: sl(w2)?, beta_off: sl(beta_off)?,
            members: sl(members)?, mem_eff: sl(mem_eff)?, mem_off: sl(mem_off)?, mem_n: sl(mem_n)?, a_ctc: table(a_ctc, n)?,
            r_blk: sl(r_blk)?, a_cell: sl(a_cell)?, c_ptr: sl(c_ptr)?, c_first: sl(c_first)?, c_row: sl(c_row)?, c_cum: sl(c_cum)?,
            n, ns,
        };
        if vr.len() != n || te.len() != n || tx.len() != n || sl::<bool>(fresh)?.len() != n {
            return Err(PyTypeError::new_err("the agents' arrays of the events' pass differ in length"));
        }
        let ev = events::step_events(&p, t0, t1, seed, st, vr, te, tx, sl(fresh)?);
        Ok((arr(py, ev.a), arr(py, ev.tr), arr(py, ev.src), arr(py, ev.dst), arr(py, ev.t), arr(py, ev.by), arr(py, ev.rt), ev.over))
    }

    /// Add the kernel to the module
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(step_events, m)?)?;
        Ok(())
    }
}
