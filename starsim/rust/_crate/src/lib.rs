//! Compiled numeric kernels for the Starsim Rust backend.
//!
//! Phase 1 contract: kernels do NOT draw random numbers. Python draws all
//! randoms (with numpy) and passes the arrays in, so equivalence with the pure
//! Python path is byte-identical by construction. Kernels receive zero-copy
//! views of the existing numpy state buffers and return result arrays.
//!
//! Floating-point note: to stay byte-identical with numpy, operations must be
//! performed in the SAME order numpy uses. `compute_transmission` mirrors
//! `(rel_trans[src] * rel_sus[trg]) * beta_per_dt` exactly.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Network transmission kernel; mirrors `Infection.compute_transmission`.
///
/// For each edge i, p = rel_trans_raw[src[i]] * rel_sus_raw[trg[i]] * beta_per_dt[i];
/// the edge transmits iff p > randvals[i]. Returns (target_uids, source_uids) for
/// transmitting edges, in edge order (matching numpy's boolean-index gather).
///
/// Args:
///   src, trg          : edge endpoint UIDs (int64), used both to index the
///                       rel_* buffers and as the returned UIDs
///   rel_trans_raw,
///   rel_sus_raw       : full-length raw state buffers (float64), indexed by UID
///   beta_per_dt       : per-edge transmission scaling (float64), length == src
///   randvals          : per-edge random draws (float64), length == src
#[pyfunction]
fn compute_transmission<'py>(
    py: Python<'py>,
    src: PyReadonlyArray1<'py, i64>,
    trg: PyReadonlyArray1<'py, i64>,
    rel_trans_raw: PyReadonlyArray1<'py, f64>,
    rel_sus_raw: PyReadonlyArray1<'py, f64>,
    beta_per_dt: PyReadonlyArray1<'py, f64>,
    randvals: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    // as_array() handles non-contiguous strides safely
    let src = src.as_array();
    let trg = trg.as_array();
    let rel_trans = rel_trans_raw.as_array();
    let rel_sus = rel_sus_raw.as_array();
    let beta = beta_per_dt.as_array();
    let rand = randvals.as_array();

    let n = src.len();
    let mut target: Vec<i64> = Vec::with_capacity(n);
    let mut source: Vec<i64> = Vec::with_capacity(n);

    for i in 0..n {
        let s = src[i];
        let t = trg[i];
        // Same multiply order as numpy: (rel_trans * rel_sus) * beta
        let p = (rel_trans[s as usize] * rel_sus[t as usize]) * beta[i];
        if p > rand[i] {
            target.push(t);
            source.push(s);
        }
    }

    Ok((
        target.into_pyarray_bound(py),
        source.into_pyarray_bound(py),
    ))
}

// ---------------------------------------------------------------------------
// PCG64 reproduction spike (Phase 3): reproduce numpy's BitGenerator stream
// bit-for-bit by lifting its 128-bit state from Python, so RNG can move into
// Rust without reproducing numpy's SeedSequence.
// ---------------------------------------------------------------------------

const PCG_MULT: u128 = 0x2360ed051fc65da44385df649fccf645;

#[inline]
fn pcg64_step(state: u128, inc: u128) -> u128 {
    state.wrapping_mul(PCG_MULT).wrapping_add(inc)
}

#[inline]
fn pcg64_output(state: u128) -> u64 {
    // PCG XSL-RR 128/64: rotate_right(hi64 ^ lo64, state >> 122)
    let hi = (state >> 64) as u64;
    let lo = state as u64;
    let rot = (state >> 122) as u32;
    (hi ^ lo).rotate_right(rot)
}

fn split(hi: u64, lo: u64) -> u128 {
    ((hi as u128) << 64) | (lo as u128)
}

/// Reproduce numpy Generator.random(n) for float64 (default dtype).
/// next_double = (next_uint64 >> 11) * 2^-53.
#[pyfunction]
fn pcg64_random_f64<'py>(
    py: Python<'py>,
    state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64, n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let mut state = split(state_hi, state_lo);
    let inc = split(inc_hi, inc_lo);
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        state = pcg64_step(state, inc);
        let u = pcg64_output(state);
        out.push(((u >> 11) as f64) * (1.0 / 9007199254740992.0));
    }
    out.into_pyarray_bound(py)
}

/// Reproduce numpy Generator.random(n, dtype=float32).
/// next_uint32 buffers a uint64: returns low 32 bits, caches high 32 bits.
/// next_float = (next_uint32 >> 8) * 2^-24. Returns f32 values (as f64 array).
#[pyfunction]
fn pcg64_random_f32<'py>(
    py: Python<'py>,
    state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64,
    has_uint32: bool, uinteger: u32, n: usize,
) -> Bound<'py, PyArray1<f32>> {
    let mut state = split(state_hi, state_lo);
    let inc = split(inc_hi, inc_lo);
    let mut has = has_uint32;
    let mut cached = uinteger;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n {
        let u32val: u32 = if has {
            has = false;
            cached
        } else {
            state = pcg64_step(state, inc);
            let u = pcg64_output(state);
            has = true;
            cached = (u >> 32) as u32;
            u as u32
        };
        out.push(((u32val >> 8) as f32) * (1.0f32 / 16777216.0f32));
    }
    out.into_pyarray_bound(py)
}

/// Generate `n` float32 uniforms (numpy dtype=float32 path) and return their
/// IEEE bit patterns as u32 (equivalent to `rng.random(n, np.float32).view(uint32)`).
/// Core variant taking a 128-bit state directly (used by the native RNG class).
fn pcg64_f32_bits_core(mut state: u128, inc: u128, mut has: bool, mut cached: u32, n: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..n {
        let u32val: u32 = if has {
            has = false;
            cached
        } else {
            state = pcg64_step(state, inc);
            let u = pcg64_output(state);
            has = true;
            cached = (u >> 32) as u32;
            u as u32
        };
        let f = ((u32val >> 8) as f32) * (1.0f32 / 16777216.0f32);
        out.push(f.to_bits());
    }
    out
}

fn pcg64_f32_bits(
    state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64,
    has_uint32: bool, uinteger: u32, n: usize,
) -> Vec<u32> {
    pcg64_f32_bits_core(split(state_hi, state_lo), split(inc_hi, inc_lo), has_uint32, uinteger, n)
}

/// Combine two CRN float32-bit blocks indexed by slots, exactly as multi_random.
fn combine_blocks(s_block: &[u32], t_block: &[u32],
                  src_slots: numpy::ndarray::ArrayView1<i64>, trg_slots: numpy::ndarray::ArrayView1<i64>) -> Vec<f64> {
    let n = src_slots.len();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let a = s_block[src_slots[i] as usize];
        let b = t_block[trg_slots[i] as usize];
        let combined = a.wrapping_mul(b) ^ a.wrapping_sub(b);
        out.push((combined as f64) * (1.0 / 4294967295.0));
    }
    out
}

/// Full Rust reproduction of `ss.multi_random('source','target').rvs(src, trg)`.
///
/// Generates two CRN float32 blocks (one per underlying ss.random dist) from the
/// lifted PCG64 states, indexes each by its slots, then combines bitwise exactly
/// as `multi_random.combine_rvs`: with a = src bits (u32), b = trg bits (u32),
/// combined = (a*b) ^ (a-b) (wrapping u32), normalized by u32::MAX -> float64.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn multi_random_rvs<'py>(
    py: Python<'py>,
    s_state_hi: u64, s_state_lo: u64, s_inc_hi: u64, s_inc_lo: u64, s_has: bool, s_uint: u32,
    t_state_hi: u64, t_state_lo: u64, t_inc_hi: u64, t_inc_lo: u64, t_has: bool, t_uint: u32,
    src_slots: PyReadonlyArray1<'py, i64>,
    trg_slots: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<f64>> {
    let src_slots = src_slots.as_array();
    let trg_slots = trg_slots.as_array();
    let n = src_slots.len();
    if n == 0 {
        return Vec::<f64>::new().into_pyarray_bound(py);
    }

    // Block size = max(slot)+1, matching process_size()
    let s_size = (src_slots.iter().copied().max().unwrap() + 1) as usize;
    let t_size = (trg_slots.iter().copied().max().unwrap() + 1) as usize;
    let s_block = pcg64_f32_bits(s_state_hi, s_state_lo, s_inc_hi, s_inc_lo, s_has, s_uint, s_size);
    let t_block = pcg64_f32_bits(t_state_hi, t_state_lo, t_inc_hi, t_inc_lo, t_has, t_uint, t_size);
    combine_blocks(&s_block, &t_block, src_slots, trg_slots).into_pyarray_bound(py)
}

/// Native Rust ownership of the trans_rng (`ss.multi_random`) RNG state.
///
/// Built once at sim init from each underlying dist's INITIAL post-seed state
/// (`np.random.default_rng(seed).bit_generator.state`). Thereafter it reproduces
/// Starsim's jumping entirely in Rust: at timestep `ti`, `ind = dt_jump_size*(ti+1)`,
/// incremented by 1 after each draw; the state at any draw is `jumped(state0, ind)`.
/// No numpy RNG calls happen during the run.
#[pyclass]
struct MultiRandomRng {
    s_state0: u128, s_inc: u128,
    t_state0: u128, t_inc: u128,
    dt_jump_size: u64,
    ind: u64,
}

#[pymethods]
impl MultiRandomRng {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        s_state_hi: u64, s_state_lo: u64, s_inc_hi: u64, s_inc_lo: u64,
        t_state_hi: u64, t_state_lo: u64, t_inc_hi: u64, t_inc_lo: u64,
        dt_jump_size: u64,
    ) -> Self {
        MultiRandomRng {
            s_state0: split(s_state_hi, s_state_lo), s_inc: split(s_inc_hi, s_inc_lo),
            t_state0: split(t_state_hi, t_state_lo), t_inc: split(t_inc_hi, t_inc_lo),
            dt_jump_size, ind: 0,
        }
    }

    /// Reset ind for timestep ti, mirroring Starsim's jump_dt (to = dt_jump_size*(ti+1)).
    fn set_timestep(&mut self, ti: u64) {
        self.ind = self.dt_jump_size * (ti + 1);
    }

    /// Reproduce one `trans_rng.rvs(src, trg)` from the current ind, then advance ind.
    /// Takes the edge endpoint UIDs and the raw slot buffer (indexed by UID), so all
    /// slot lookups happen in Rust rather than via Arr.__getitem__ in Python.
    fn rvs<'py>(
        &mut self, py: Python<'py>,
        src: PyReadonlyArray1<'py, i64>, trg: PyReadonlyArray1<'py, i64>,
        slots_raw: PyReadonlyArray1<'py, i64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let src = src.as_array();
        let trg = trg.as_array();
        let slots = slots_raw.as_array();
        let n = src.len();
        let ind = self.ind;
        self.ind += 1;
        if n == 0 {
            return Vec::<f64>::new().into_pyarray_bound(py);
        }
        // Resolve UID -> slot, and the block size needed per side
        let mut src_slots: Vec<i64> = Vec::with_capacity(n);
        let mut trg_slots: Vec<i64> = Vec::with_capacity(n);
        let (mut s_max, mut t_max) = (0i64, 0i64);
        for i in 0..n {
            let ss = slots[src[i] as usize];
            let ts = slots[trg[i] as usize];
            if ss > s_max { s_max = ss; }
            if ts > t_max { t_max = ts; }
            src_slots.push(ss);
            trg_slots.push(ts);
        }
        // jumped() produces a fresh state with no buffered uint32 (has=false, cached=0)
        let s_state = pcg64_jumped(self.s_state0, self.s_inc, ind as u128);
        let t_state = pcg64_jumped(self.t_state0, self.t_inc, ind as u128);
        let s_block = pcg64_f32_bits_core(s_state, self.s_inc, false, 0, (s_max + 1) as usize);
        let t_block = pcg64_f32_bits_core(t_state, self.t_inc, false, 0, (t_max + 1) as usize);
        let mut out: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let a = s_block[src_slots[i] as usize];
            let b = t_block[trg_slots[i] as usize];
            let combined = a.wrapping_mul(b) ^ a.wrapping_sub(b);
            out.push((combined as f64) * (1.0 / 4294967295.0));
        }
        out.into_pyarray_bound(py)
    }
}

// numpy PCG64.jumped() advances the LCG by `jumps * PCG_JUMP_STEP` steps.
const PCG_JUMP_STEP: u128 = 0x9e3779b97f4a7c15f39cc0605cedc835;

/// PCG64 LCG jump-ahead: advance `state` by `delta` steps (mod 2^128).
fn pcg64_advance(state: u128, delta: u128, mult: u128, inc: u128) -> u128 {
    let mut acc_mult: u128 = 1;
    let mut acc_plus: u128 = 0;
    let mut cur_mult = mult;
    let mut cur_plus = inc;
    let mut d = delta;
    while d > 0 {
        if d & 1 == 1 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        d >>= 1;
    }
    acc_mult.wrapping_mul(state).wrapping_add(acc_plus)
}

/// Reproduce `bitgen.jumped(jumps).state` from the initial (post-seed) state.
fn pcg64_jumped(state0: u128, inc: u128, jumps: u128) -> u128 {
    let delta = PCG_JUMP_STEP.wrapping_mul(jumps);
    pcg64_advance(state0, delta, PCG_MULT, inc)
}

/// Verification hook: return the jumped state as (hi, lo) to compare against numpy.
#[pyfunction]
fn pcg64_jumped_state(
    state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64, jumps: u64,
) -> (u64, u64) {
    let s = pcg64_jumped(split(state_hi, state_lo), split(inc_hi, inc_lo), jumps as u128);
    ((s >> 64) as u64, s as u64)
}

/// Reproduce numpy Generator.permutation(source) for an int64 array.
///
/// numpy shuffles a copy via Fisher-Yates from the end, drawing each index with
/// `random_interval(i)` = masked-rejection uniform in [0, i] (using next_uint32
/// when i fits in 32 bits, else next_uint64). next_uint32 buffers a uint64
/// (low 32 bits first, high 32 cached), tracked by has_uint32/uinteger.
/// numpy Generator.shuffle/permutation algorithm: Fisher-Yates from the end,
/// each index via masked-rejection uniform in [0,i]. Mutates `arr` in place.
/// State starts at (state, has_uint32, cached); inc is the PCG64 increment.
fn fisher_yates(arr: &mut [i64], mut state: u128, inc: u128, mut has: bool, mut cached: u32) {
    let make_mask = |mut m: u64| -> u64 {
        m |= m >> 1; m |= m >> 2; m |= m >> 4; m |= m >> 8; m |= m >> 16; m |= m >> 32; m
    };
    let n = arr.len();
    for i in (1..n).rev() {
        let max = i as u64;
        let mask = make_mask(max);
        let j = if max <= 0xffff_ffff {
            loop {
                let u32val: u32 = if has { has = false; cached } else {
                    state = pcg64_step(state, inc);
                    let u = pcg64_output(state);
                    has = true; cached = (u >> 32) as u32;
                    u as u32
                };
                let v = (u32val as u64) & mask;
                if v <= max { break v; }
            }
        } else {
            loop {
                state = pcg64_step(state, inc);
                let v = pcg64_output(state) & mask;
                if v <= max { break v; }
            }
        };
        arr.swap(i, j as usize);
    }
}

#[pyfunction]
fn permutation_int64<'py>(
    py: Python<'py>,
    state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64,
    has_uint32: bool, uinteger: u32,
    source: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let mut arr: Vec<i64> = source.as_array().to_vec();
    fisher_yates(&mut arr, split(state_hi, state_lo), split(inc_hi, inc_lo), has_uint32, uinteger);
    arr.into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Phase 4: native whole-sim loop for SIS + RandomNet (constant dur_inf,
// waning=0, imm_boost=0). Rust owns all state, edges, and RNG; there are no
// per-timestep round-trips to Python. Initial state is lifted from a post-init
// ss.Sim; results are returned for byte-identical validation against ss.SIS.
// ---------------------------------------------------------------------------
#[pyclass]
struct SisRandomNetSim {
    n_agents: usize,
    n_steps: usize,
    // mutable per-agent state (indexed by UID)
    infected: Vec<bool>,
    susceptible: Vec<bool>,
    ti_infected: Vec<f64>,
    ti_recovered: Vec<f64>,
    // fixed inputs
    agent_uids: Vec<i64>, // "born" UIDs used for edge generation
    slots: Vec<i64>,      // slot per UID
    n_edges_per_agent: usize,
    beta_per_dt: f64,
    dur_inf_ts: f64,
    n_initial_cases: i64,
    dt_jump_size: u64,
    // RNG initial (post-seed) states
    net_state0: u128, net_inc: u128,
    s_state0: u128, s_inc: u128,
    t_state0: u128, t_inc: u128,
}

impl SisRandomNetSim {
    /// One direction of transmission RNG: reproduce trans_rng.rvs(src, trg) at `ind`.
    fn trans_randvals(&self, src: &[i64], trg: &[i64], ind: u64) -> Vec<f64> {
        let n = src.len();
        let (mut s_slots, mut t_slots) = (vec![0i64; n], vec![0i64; n]);
        let (mut s_max, mut t_max) = (0i64, 0i64);
        for e in 0..n {
            let a = self.slots[src[e] as usize];
            let b = self.slots[trg[e] as usize];
            s_slots[e] = a; t_slots[e] = b;
            if a > s_max { s_max = a; }
            if b > t_max { t_max = b; }
        }
        let s_state = pcg64_jumped(self.s_state0, self.s_inc, ind as u128);
        let t_state = pcg64_jumped(self.t_state0, self.t_inc, ind as u128);
        let s_block = pcg64_f32_bits_core(s_state, self.s_inc, false, 0, (s_max + 1) as usize);
        let t_block = pcg64_f32_bits_core(t_state, self.t_inc, false, 0, (t_max + 1) as usize);
        let mut out = vec![0f64; n];
        for e in 0..n {
            let a = s_block[s_slots[e] as usize];
            let b = t_block[t_slots[e] as usize];
            let c = a.wrapping_mul(b) ^ a.wrapping_sub(b);
            out[e] = (c as f64) * (1.0 / 4294967295.0);
        }
        out
    }
}

#[pymethods]
impl SisRandomNetSim {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_steps: usize,
        susceptible: Vec<bool>, infected: Vec<bool>,
        ti_infected: Vec<f64>, ti_recovered: Vec<f64>,
        agent_uids: Vec<i64>, slots: Vec<i64>,
        n_edges_per_agent: usize, beta_per_dt: f64, dur_inf_ts: f64,
        n_initial_cases: i64, dt_jump_size: u64,
        net_state_hi: u64, net_state_lo: u64, net_inc_hi: u64, net_inc_lo: u64,
        s_state_hi: u64, s_state_lo: u64, s_inc_hi: u64, s_inc_lo: u64,
        t_state_hi: u64, t_state_lo: u64, t_inc_hi: u64, t_inc_lo: u64,
    ) -> Self {
        SisRandomNetSim {
            n_agents: susceptible.len(), n_steps,
            infected, susceptible, ti_infected, ti_recovered,
            agent_uids, slots, n_edges_per_agent, beta_per_dt, dur_inf_ts,
            n_initial_cases, dt_jump_size,
            net_state0: split(net_state_hi, net_state_lo), net_inc: split(net_inc_hi, net_inc_lo),
            s_state0: split(s_state_hi, s_state_lo), s_inc: split(s_inc_hi, s_inc_lo),
            t_state0: split(t_state_hi, t_state_lo), t_inc: split(t_inc_hi, t_inc_lo),
        }
    }

    /// Run the whole sim natively; return a dict of result arrays.
    fn run<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let n = self.n_agents;
        let (mut r_ninf, mut r_nsus) = (vec![0f64; self.n_steps], vec![0f64; self.n_steps]);
        let (mut r_prev, mut r_new) = (vec![0f64; self.n_steps], vec![0f64; self.n_steps]);
        let nconn = self.n_edges_per_agent;

        for ti in 0..self.n_steps {
            let tif = ti as f64;

            // (1) step_state: SIS recovery (infectious whose ti_recovered has passed)
            for u in 0..n {
                if self.infected[u] && self.ti_recovered[u] <= tif {
                    self.infected[u] = false;
                    self.susceptible[u] = true;
                }
            }

            // (2) RandomNet.step: regenerate edges (dur=0 -> fresh each step)
            let ind = self.dt_jump_size * (ti as u64 + 1);
            let net_state = pcg64_jumped(self.net_state0, self.net_inc, ind as u128);
            let mut p1: Vec<i64> = Vec::with_capacity(self.agent_uids.len() * nconn);
            for &u in &self.agent_uids {
                for _ in 0..nconn { p1.push(u); }
            }
            let mut p2 = p1.clone();
            fisher_yates(&mut p2, net_state, self.net_inc, false, 0);

            // (3) infect: rel_trans/rel_sus snapshot at start, both directions, then apply
            let n_edges = p1.len();
            let mut new_cases: Vec<usize> = Vec::new();
            for (dir, (src, trg)) in [(&p1, &p2), (&p2, &p1)].iter().enumerate() {
                let randvals = self.trans_randvals(src, trg, ind + dir as u64);
                for e in 0..n_edges {
                    let s = src[e] as usize;
                    let t = trg[e] as usize;
                    let rt = if self.infected[s] { 1.0 } else { 0.0 };
                    let rs = if self.susceptible[t] { 1.0 } else { 0.0 };
                    let p = (rt * rs) * self.beta_per_dt;
                    if p > randvals[e] { new_cases.push(t); }
                }
            }
            for &t in &new_cases {
                if self.susceptible[t] { // set_prognoses (idempotent across duplicate edges)
                    self.susceptible[t] = false;
                    self.infected[t] = true;
                    self.ti_infected[t] = tif;
                    self.ti_recovered[t] = tif + self.dur_inf_ts;
                }
            }

            // (4) update_results
            let (mut ninf, mut nsus, mut nnew) = (0i64, 0i64, 0i64);
            for u in 0..n {
                if self.infected[u] { ninf += 1; }
                if self.susceptible[u] { nsus += 1; }
                // NaN guard: Rust casts NaN to 0, which would falsely match ti==0
                if !self.ti_infected[u].is_nan() && (self.ti_infected[u].round() as i64) == ti as i64 {
                    nnew += 1;
                }
            }
            if ti == 0 { nnew -= self.n_initial_cases; }
            r_ninf[ti] = ninf as f64;
            r_nsus[ti] = nsus as f64;
            r_prev[ti] = ninf as f64 / n as f64;
            r_new[ti] = nnew as f64;
        }

        // cum_infections = cumsum(new_infections)
        let mut r_cum = vec![0f64; self.n_steps];
        let mut acc = 0.0;
        for ti in 0..self.n_steps { acc += r_new[ti]; r_cum[ti] = acc; }

        let out = pyo3::types::PyDict::new_bound(py);
        out.set_item("n_infected", r_ninf.into_pyarray_bound(py))?;
        out.set_item("n_susceptible", r_nsus.into_pyarray_bound(py))?;
        out.set_item("prevalence", r_prev.into_pyarray_bound(py))?;
        out.set_item("new_infections", r_new.into_pyarray_bound(py))?;
        out.set_item("cum_infections", r_cum.into_pyarray_bound(py))?;
        Ok(out)
    }
}

#[pymodule]
fn starsim_rust_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_transmission, m)?)?;
    m.add_function(wrap_pyfunction!(pcg64_jumped_state, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_int64, m)?)?;
    m.add_class::<SisRandomNetSim>()?;
    m.add_function(wrap_pyfunction!(pcg64_random_f64, m)?)?;
    m.add_function(wrap_pyfunction!(pcg64_random_f32, m)?)?;
    m.add_function(wrap_pyfunction!(multi_random_rvs, m)?)?;
    m.add_class::<MultiRandomRng>()?;
    Ok(())
}
