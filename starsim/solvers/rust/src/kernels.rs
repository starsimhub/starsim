//! The kernels of the compartmental solvers (`starsim/solvers/backends/kernels.py`, `ss.kernels`), with the same arguments and results: the
//! ODE (RK4 and Euler), the chain binomial, tau-leaping, and the exact solver (by place, and by the compartments that hold anyone),
//! the events of a movement register, and the deliveries. A program, register, deliveries, and tape are the namedtuples of
//! kernels.py, as views of their arrays; the random numbers are the caller's NumPy Generators, drawn as Numba draws them.
//!
//! The ODE and what every solver shares are here; the draws of the stochastic solvers are in `kernels/events.rs`, the exact
//! solver by place in `kernels/places.rs`, and the Generators in `kernels/rng.rs`.

mod dr;
mod events;
mod places;
mod rng;
mod views;
mod ziggurat;

use events::*;
use places::*;
pub use rng::Rng;
pub use views::*;

// The operations a rate can be made of (see kernels.py)
pub const CONST: i64 = 0;
pub const COUNT: i64 = 1;
pub const FRAC: i64 = 2;
pub const EXPOSURE: i64 = 3;
pub const ADD: i64 = 4;
pub const SUB: i64 = 5;
pub const MUL: i64 = 6;
pub const DIV: i64 = 7;
pub const POW: i64 = 8;
pub const MIN: i64 = 9;
pub const MAX: i64 = 10;
pub const LT: i64 = 11;
pub const LE: i64 = 12;
pub const GT: i64 = 13;
pub const GE: i64 = 14;
pub const EXP: i64 = 15;
pub const LOG: i64 = 16;
pub const SQRT: i64 = 17;
pub const ABS: i64 = 18;
pub const SIN: i64 = 19;
pub const COS: i64 = 20;
pub const TOTAL: i64 = 21;
pub const CONVEY: i64 = 22;
pub const LOAD: i64 = 23;
pub const STORE: i64 = 24;
pub const GROUP: i64 = 25; // A count in the group each rate is for (kernels.py's GROUP)

pub const GOLDEN: u64 = 0x9e3779b97f4a7c15;
pub const SPARSE_FLOWS: usize = 256; // The flows above which the exact solver by place follows only the compartments that hold anyone
pub const ROW_GROUPS: usize = 50; // The groups from which a draw for every group is made as a vectorized row

/// The flows of a compartmental model (`kernels.Program`; see `fields` there)
#[derive(Clone, Copy)]
pub struct Program {
    pub n_trans: usize,
    pub src: A1<i64>,
    pub named: A1<u8>,
    pub index: A1<i64>,
    pub record: A1<i64>,
    pub inflow: A2<f64>,
    pub fixed: A2<f64>,
    pub mult: A2<f64>,
    pub depth: usize,
    pub starts: A1<i64>,
    pub ops: A1<i64>,
    pub args: A1<i64>,
    pub consts: A2<f64>,
    pub set_starts: A1<i64>,
    pub set_comps: A1<i64>,
    pub people: A1<i64>,
    pub has_det: bool,
    pub det: A1<u8>,
    pub places: A1<i64>,
    pub n_places: usize,
    pub exp_set: A1<i64>,
    pub rel_trans: A2<f64>,
    pub levels: A2<i64>,
    pub sizes: A1<i64>,
    pub block: A1<i64>,
    pub contacts: A3<f64>,
    pub use_: A2<u8>,
    pub mat: A3<f64>,
    pub sh_pair: A2<i64>,
    pub sh_val: A1<f64>,
    pub tgt_starts: A1<i64>,
    pub tgt_comp: A1<i64>,
    pub tgt_w: A2<f64>,
    pub tgt_map: A1<i64>,
    pub tgt_rec: A1<i64>,
    pub crd_starts: A1<i64>,
    pub crd_index: A1<i64>,
    pub crd_w: A2<f64>,
    pub crd_parent: A1<i64>,
    pub maps: A3<f64>,
    pub n_movers: usize,
    pub mov_starts: A2<i64>,
    pub mov_comps: A1<i64>,
    pub mov_pair: A2<i64>,
    pub mov_rate: A1<f64>,
    pub mov_base: A1<f64>,
    pub mov_par: A1<i64>,
    pub src_starts: A1<i64>,
    pub src_flows: A1<i64>,
    pub src_chain: A1<i64>,
    pub src_stage: A1<i64>,
    pub src_last: A1<i64>,
    pub local: bool,
    pub pl_starts: A1<i64>,
    pub pl_groups: A1<i64>,
}

/// A movement register (`kernels.Register`)
#[derive(Clone, Copy)]
pub struct Register {
    pub starts: A1<i64>,
    pub src: A1<i64>,
    pub dst: A1<i64>,
    pub n: A1<f64>,
    pub sel: A1<i64>,
    pub sel_starts: A1<i64>,
    pub sel_comps: A1<i64>,
    pub sel_to: A1<i64>,
    pub ent_starts: A1<i64>,
    pub ent_comps: A1<i64>,
    pub ent_w: A2<f64>,
    pub pg_starts: A1<i64>,
    pub pg_groups: A1<i64>,
    pub step: i64,
}

/// The deliveries of a whole run (`kernels.Shifts`)
#[derive(Clone, Copy)]
pub struct Shifts {
    pub due: A2<u8>,
    pub sel: A3<f64>,
    pub cov: A3<f64>,
    pub to: A2<i64>,
    pub rest: A2<i64>,
    pub land: A3<f64>,
}

/// The values worked out once a step (`kernels.Tape`)
#[derive(Clone, Copy)]
pub struct Tape {
    pub ops: A1<i64>,
    pub a: A1<i64>,
    pub b: A1<i64>,
    pub const_grouped: A1<u8>,
    pub set_starts: A1<i64>,
    pub state_starts: A1<i64>,
    pub state_comps: A1<i64>,
    pub places: A1<i64>,
    pub n_places: usize,
    pub local: bool,
    pub people: A1<i64>,
    pub has_frac: bool,
}

/// The fixed, mult, consts, and rel_trans tables of a run (see `run()`)
pub type Tables = [A3<f64>; 4];

/// The arrays that the kernels work in, made once so that nothing is allocated inside a step (`workspace()`)
pub struct Work {
    _own: (Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>, Own<f64>),
    pub rates: A2<f64>,     // The rate of each flow, by group
    pub exposures: A2<f64>, // Exposure to each infectious state
    pub stack: A2<f64>,     // For working out a rate
    pub groups: A2<f64>,    // Scratch, by group (7 rows)
    pub cells: A2<f64>,     // Scratch, by cell or place (6 rows; the last is, for the exact solver by place, whether each place is idle)
    pub dy: A3<f64>,        // The stages of the integrator
    pub df: A3<f64>,
    pub yt: A2<f64>,
    pub chain: A1<f64>,     // Where people at a stage of a dwell time end up: see chain_probs()
    pub credits: A3<f64>,   // For credit()
    pub rows: A2<f64>,      // For binomials() and poissons() (11 rows)
}

fn max_i(a: A1<i64>) -> i64 { a.sl().iter().copied().max().unwrap_or(0) }

/// The arrays to work in (`workspace()`)
pub fn workspace(nc: usize, ng: usize, p: &Program) -> Work {
    let nflows = p.n_trans + p.n_movers;
    let size = (max_i(p.sizes) as usize).max(p.n_places).max(p.pl_starts.len().saturating_sub(1));
    let (o0, rates) = own2(p.fixed.n0(), p.fixed.n1(), 0.0);
    let (o1, exposures) = own2(p.exp_set.len(), ng, 0.0);
    let (o2, stack) = own2(p.depth, ng, 0.0);
    let (o3, groups) = own2(7, ng, 0.0);
    let (o4, cells) = own2(6, size, 0.0);
    let (o5, dy) = own3(4, nc, ng, 0.0);
    let (o6, df) = own3(4, nflows, ng, 0.0);
    let (o7, yt) = own2(nc, ng, 0.0);
    let (o8, chain) = own1(if p.src_chain.len() > 0 { max_i(p.src_chain) as usize } else { 0 } + 2, 0.0);
    let widest = (1..p.crd_starts.len()).map(|i| p.crd_starts.at(i) - p.crd_starts.at(i - 1)).max().unwrap_or(0) as usize;
    let (o9, credits) = own3(2, widest + 1, ng, 0.0);
    let (o10, rows) = own2(11, ng, 0.0);
    Work { _own: (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10), rates, exposures, stack, groups, cells, dy, df, yt, chain, credits, rows }
}

/// A function of one value, as an operation on the stack of a rate (`unary()`)
#[inline(always)]
pub fn unary(op: i64, x: f64) -> f64 {
    match op {
        EXP => x.exp(),
        LOG => x.ln(),
        SQRT => x.sqrt(),
        ABS => x.abs(),
        SIN => x.sin(),
        COS => x.cos(),
        _ => f64::NAN,
    }
}

/// A count on the stack of a rate: the number v (COUNT), or its share of the n people there (FRAC), 0 where there is no one (`leaf_share()`)
#[inline(always)]
pub fn leaf_share(op: i64, v: f64, n: f64) -> f64 {
    if op == FRAC { if n > 0.0 { v/n } else { 0.0 } } else { v }
}

/// A function of two values, as an operation on the stack of a rate (`binary()`)
#[inline(always)]
pub fn binary(op: i64, a: f64, b: f64) -> f64 {
    match op {
        ADD => a + b,
        SUB => a - b,
        MUL => a*b,
        DIV => if a != 0.0 { a/b } else { 0.0 }, // Nothing divided by anything is nothing
        POW => a.powf(b),
        MIN => mn(a, b),
        MAX => mx(a, b),
        LT => if a < b { 1.0 } else { 0.0 },
        LE => if a <= b { 1.0 } else { 0.0 },
        GT => if a > b { 1.0 } else { 0.0 },
        GE => if a >= b { 1.0 } else { 0.0 },
        _ => f64::NAN,
    }
}

/// Every value of a tape at the compartments y, by group, into out; grouped is set to whether each differs by group (`evaluate()`)
pub fn evaluate(y: A2<f64>, t: &Tape, consts: A2<f64>, out: A2<f64>, grouped: A1<u8>) {
    let ng = y.n1();
    let (_o1, num) = own1(t.n_places, 0.0);
    let (_o2, people) = own1(t.n_places, 0.0);
    if t.has_frac {
        for g in 0..ng {
            for i in 0..t.people.len() {
                people.add(t.places.u(g), y.at(t.people.u(i), g));
            }
        }
    }
    for i in 0..t.ops.len() {
        let op = t.ops.at(i);
        let arg = t.a.u(i);
        if op == CONST {
            grouped.set(i, t.const_grouped.at(arg));
            for g in 0..ng {
                out.set(i, g, consts.at(arg, g));
            }
        } else if op == COUNT || op == FRAC || op == TOTAL {
            num.fill(0.0);
            for g in 0..ng {
                let mut v = 0.0;
                for s in t.set_starts.u(arg)..t.set_starts.u(arg + 1) { // State by state, added as NumPy adds them
                    let mut vs = 0.0;
                    for k in t.state_starts.u(s)..t.state_starts.u(s + 1) {
                        vs += y.at(t.state_comps.u(k), g);
                    }
                    v += vs;
                }
                num.add(if op == TOTAL { 0 } else { t.places.u(g) }, v);
            }
            grouped.set(i, (t.local && op != TOTAL) as u8);
            for g in 0..ng {
                let pl = if op == TOTAL { 0 } else { t.places.u(g) };
                out.set(i, g, leaf_share(op, num.at(pl), people.at(pl)));
            }
        } else if op >= EXP {
            grouped.set(i, grouped.at(arg));
            for g in 0..ng {
                out.set(i, g, unary(op, out.at(arg, g)));
            }
        } else {
            let b = t.b.u(i);
            grouped.set(i, (grouped.b(arg) || grouped.b(b)) as u8);
            for g in 0..ng {
                out.set(i, g, binary(op, out.at(arg, g), out.at(b, g)));
            }
        }
    }
}

/// BLAS's dgemv, as scipy gives it (the one Numba's np.dot calls), once found: see `bind::blas()`
pub type Dgemv = unsafe extern "C" fn(*const u8, *const i32, *const i32, *const f64, *const f64, *const i32, *const f64, *const i32, *const f64, *mut f64, *const i32);
pub static DGEMV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// The product of the first `size` rows and columns of a mixing matrix with a vector, into out; by BLAS when the whole matrix is used and is larger than 16, as Numba's np.dot (`mix()`)
pub fn mix(matrix: A2<f64>, v: A1<f64>, size: usize, out: A1<f64>) {
    if size == matrix.n0() && size > 16 {
        let f: Dgemv = unsafe { std::mem::transmute::<usize, Dgemv>(*DGEMV.get().expect("BLAS not found")) };
        let cols = matrix.n1();
        let contiguous: Vec<f64>; // np.dot copies a matrix whose rows are not contiguous
        let a = if matrix.n0() < 2 || matrix.row(1).ptr() as usize - matrix.row(0).ptr() as usize == cols*8 {
            matrix.row(0).ptr() as *const f64
        } else {
            contiguous = (0..matrix.n0()).flat_map(|i| matrix.row(i).sl().to_vec()).collect();
            contiguous.as_ptr()
        };
        let mut res = vec![0.0; size];
        let (m, n, lda, inc) = (cols as i32, matrix.n0() as i32, cols as i32, 1i32);
        let (alpha, beta) = (1.0f64, 0.0f64);
        unsafe { f(b"t".as_ptr(), &m, &n, &alpha, a, &lda, v.ptr(), &inc, &beta, res.as_mut_ptr(), &inc) };
        for i in 0..size {
            out.set(i, res[i]);
        }
        return;
    }
    for i in 0..size {
        let mut acc = 0.0;
        for j in 0..size {
            acc += matrix.at(i, j)*v.at(j);
        }
        out.set(i, acc);
    }
}

/// The product of the first `block` rows and columns of a mixing matrix with each block of a vector, into out (`by_block()`)
pub fn by_block(matrix: A2<f64>, v: A1<f64>, size: usize, block: usize, out: A1<f64>) {
    let mut a = 0;
    while a < size {
        for i in 0..block {
            let mut acc = 0.0;
            for j in 0..block {
                acc += matrix.at(i, j)*v.at(a + j);
            }
            out.set(a + i, acc);
        }
        a += block.max(1);
    }
}

/// The product of the time share between places with a vector of places by levels, into out: `v @ share` if `by_columns`, else `share @ v` (`share()`)
pub fn share(pair: A2<i64>, vals: A1<f64>, v: A1<f64>, size: usize, block: usize, out: A1<f64>, by_columns: bool) {
    let places = size/block;
    if vals.len() == places*places && places > 16 {
        let f: Dgemv = unsafe { std::mem::transmute::<usize, Dgemv>(*DGEMV.get().expect("BLAS not found")) };
        let (mut part, mut res) = (vec![0.0; places], vec![0.0; places]);
        let (n, inc) = (places as i32, 1i32);
        let (alpha, beta) = (1.0f64, 0.0f64);
        let trans = if by_columns { b"n" } else { b"t" }; // The rows are BLAS's columns
        for i in 0..block {
            for j in 0..places {
                part[j] = v.at(j*block + i);
            }
            unsafe { f(trans.as_ptr(), &n, &n, &alpha, vals.ptr() as *const f64, &n, part.as_ptr(), &inc, &beta, res.as_mut_ptr(), &inc) };
            for j in 0..places {
                out.set(j*block + i, res[j]);
            }
        }
        return;
    }
    for i in 0..size {
        out.set(i, 0.0);
    }
    for k in 0..vals.len() {
        let (a, b) = (pair.u(0, k), pair.u(1, k));
        if by_columns {
            for i in 0..block {
                out.add(b*block + i, vals.at(k)*v.at(a*block + i));
            }
        } else {
            for i in 0..block {
                out.add(a*block + i, vals.at(k)*v.at(b*block + i));
            }
        }
    }
}

/// The parts of the program that vary in time at a row of their tables (`set_step()`); a negative row counts from the end, as Numba indexes
pub fn set_step(parts: [A2<f64>; 4], ti: i64, tables: &Tables) {
    for k in 0..4 {
        let n = tables[k].n0() as i64;
        if ti < n {
            parts[k].copy_from(tables[k].plane(if ti < 0 { ti + n } else { ti } as usize));
        }
    }
}

/// The rates of the movers whose rate is a parameter, from its value at a step or stage (`set_movers()`)
#[inline(always)]
pub fn set_movers(p: &Program) {
    for e in 0..p.mov_par.len() {
        if p.mov_par.at(e) >= 0 {
            p.mov_rate.set(e, p.mov_base.at(e)*p.consts.at(p.mov_par.u(e), p.mov_pair.u(0, e)));
        }
    }
}

pub fn parts(p: &Program) -> [A2<f64>; 4] { [p.fixed, p.mult, p.consts, p.rel_trans] }

/// Take one timestep (`step()`): the ODE by Euler or RK4, or the rates for the stochastic solvers, which hand them to `draw()` or `event()`
#[allow(clippy::too_many_arguments)]
pub fn step(rng: Rng, y: A2<f64>, p: &Program, w: &Work, dt: f64, mut n_sub: i64, solver: i64, flows: A2<f64>, varying: bool) {
    let (rates, exposures, stack, groups, cells, dy, df, yt) = (w.rates, w.exposures, w.stack, w.groups, w.cells, w.dy, w.df, w.yt);
    let (n, local_n, vals, infectious) = (groups.row(0), groups.row(1), groups.row(2), groups.row(3));
    let (num, den, num2, den2, prev) = (cells.row(0), cells.row(1), cells.row(2), cells.row(3), cells.row(4));
    let (nc, ng) = (y.n0(), y.n1());
    let nf = p.src.len();
    let nflows = p.n_trans + p.n_movers;
    let stages = if solver == 0 { 4 } else { 1 };
    let mut h = dt;
    let timed = solver == 3 || solver == 4; // Tau-leaping and the exact solver take steps of their own length
    let by_place = solver == 4 && p.local;
    let auto = solver < 2 && n_sub <= 0;
    let mut elapsed = 0.0;
    let mut fastest = 0.0;
    let mut sub: i64 = 0;
    let extra = p.starts.len() - nf - 1; // Conveyors, whose operations follow the flows'
    let mut pending = if extra > 0 { step_start(y, p, dt, solver) } else { false };
    let mut sure = false;
    flows.fill(0.0);
    let (_om, moved_in) = if auto && p.n_movers > 0 { own2(nc, ng, 0.0) } else { own2(0, 0, 0.0) };
    let has_moved = moved_in.n0() > 0;
    let stack_top = p.depth.wrapping_sub(1); // The registers are the last rows of the stack
    loop {
        for stage in 0..stages {
            let ys = if stage == 0 { y } else { yt };

            // The rate of every flow, by group
            for f in 0..nf {
                for g in 0..ng {
                    rates.set(f, g, p.fixed.at(f, g));
                }
            }
            if p.ops.len() > 0 && !by_place {
                n.fill(0.0);
                for i in 0..p.people.len() {
                    let c = ys.row(p.people.u(i));
                    for g in 0..ng {
                        n.add(g, c.at(g));
                    }
                }
                for i in 0..p.n_places {
                    num.set(i, 0.0);
                }
                for g in 0..ng {
                    num.add(p.places.u(g), n.at(g));
                }
                for g in 0..ng {
                    local_n.set(g, num.at(p.places.u(g)));
                }

                for e in 0..p.exp_set.len() { // Exposure to each infectious state through each mixing term
                    let s = p.exp_set.at(e);
                    if s < 0 { // The sum of the -s exposures before it, in order
                        let first = (e as i64 + s) as usize;
                        for g in 0..ng {
                            exposures.set(e, g, exposures.at(first, g));
                            for x in first + 1..e {
                                exposures.add(e, g, exposures.at(x, g));
                            }
                        }
                        continue;
                    }
                    let s = s as usize;
                    exposures.row(e).fill(0.0);
                    infectious.fill(0.0);
                    let rel = p.rel_trans.row(e);
                    for k in p.set_starts.u(s)..p.set_starts.u(s + 1) {
                        let c = ys.row(p.set_comps.u(k));
                        for g in 0..ng {
                            infectious.add(g, c.at(g)*rel.at(g));
                        }
                    }
                    for t in 0..p.sizes.len() {
                        let size = p.sizes.u(t);
                        for i in 0..size {
                            num.set(i, 0.0);
                            den.set(i, 0.0);
                        }
                        for g in 0..ng {
                            num.add(p.levels.u(t, g), infectious.at(g));
                            den.add(p.levels.u(t, g), n.at(g));
                        }
                        let block = p.block.u(t);
                        if p.use_.b(0, t) { // Move people to where they spend their time
                            share(p.sh_pair, p.sh_val, num, size, block, num2, true);
                            share(p.sh_pair, p.sh_val, den, size, block, den2, true);
                            for i in 0..size {
                                num.set(i, num2.at(i));
                                den.set(i, den2.at(i));
                            }
                        }
                        for i in 0..size {
                            prev.set(i, if den.at(i) > 0.0 { num.at(i)/den.at(i) } else { 0.0 });
                        }
                        if p.use_.b(1, t) { // Mix between the levels in each place
                            if block == size {
                                mix(p.mat.plane(t), prev, size, num);
                            } else {
                                by_block(p.mat.plane(t), prev, size, block, num);
                            }
                            for i in 0..size {
                                prev.set(i, num.at(i));
                            }
                        }
                        if p.use_.b(2, t) { // Mix between places
                            share(p.sh_pair, p.sh_val, prev, size, block, num, false);
                            for i in 0..size {
                                prev.set(i, num.at(i));
                            }
                        }
                        if p.contacts.n0() > 1 || p.contacts.n2() > 1 { // Contacts by exposure and group
                            for g in 0..ng {
                                exposures.add(e, g, p.contacts.at(e, t, g)*prev.at(p.levels.u(t, g)));
                            }
                        } else {
                            let c = p.contacts.at(0, t, 0);
                            for g in 0..ng {
                                exposures.add(e, g, c*prev.at(p.levels.u(t, g)));
                            }
                        }
                    }
                }

                for f in -1..nf as i64 { // The rates worked out from these, as operations on a stack, after the parts several share (f = -1)
                    let mut top: i64 = -1;
                    let a = if f < 0 { 0 } else { p.starts.u(f as usize) };
                    for k in a..p.starts.u((f + 1) as usize) {
                        let op = p.ops.at(k);
                        let arg = p.args.u(k);
                        if op == CONST {
                            top += 1;
                            let r = stack.row(top as usize);
                            for g in 0..ng {
                                r.set(g, p.consts.at(arg, g));
                            }
                        } else if op == EXPOSURE {
                            top += 1;
                            let r = stack.row(top as usize);
                            for g in 0..ng {
                                r.set(g, exposures.at(arg, g));
                            }
                        } else if op == TOTAL {
                            top += 1;
                            let mut v = 0.0;
                            for g in 0..ng {
                                for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                                    v += ys.at(p.set_comps.u(i), g);
                                }
                            }
                            stack.row(top as usize).fill(v);
                        } else if op == COUNT || op == FRAC {
                            top += 1;
                            for i in 0..p.n_places {
                                num.set(i, 0.0);
                            }
                            for g in 0..ng {
                                let mut v = 0.0;
                                for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                                    v += ys.at(p.set_comps.u(i), g);
                                }
                                vals.set(g, v);
                                num.add(p.places.u(g), v);
                            }
                            for g in 0..ng {
                                stack.set(top as usize, g, leaf_share(op, num.at(p.places.u(g)), local_n.at(g)));
                            }
                        } else if op == GROUP { // The number in some compartments, in the same group
                            top += 1;
                            for g in 0..ng {
                                let mut v = 0.0;
                                for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                                    v += ys.at(p.set_comps.u(i), g);
                                }
                                stack.set(top as usize, g, v);
                            }
                        } else if op == LOAD {
                            top += 1;
                            for g in 0..ng {
                                stack.set(top as usize, g, stack.at(stack_top - arg, g));
                            }
                        } else if op == STORE {
                            for g in 0..ng {
                                stack.set(stack_top - arg, g, stack.at(top as usize, g));
                            }
                            top -= 1;
                        } else if op >= EXP {
                            let r = stack.row(top as usize);
                            for g in 0..ng {
                                r.set(g, unary(op, r.at(g)));
                            }
                        } else {
                            top -= 1;
                            let (r, q) = (stack.row(top as usize), stack.row(top as usize + 1));
                            for g in 0..ng {
                                r.set(g, binary(op, r.at(g), q.at(g)));
                            }
                        }
                    }
                    if top >= 0 && f >= 0 {
                        let f = f as usize;
                        for g in 0..ng {
                            rates.set(f, g, stack.at(0, g)*p.mult.at(f, g));
                        }
                    }
                }
            }

            if sub == 0 && stage == 0 { // A probability of 1 over the step is an infinite rate (see certain())
                for f in 0..nf {
                    for g in 0..ng {
                        sure = sure || rates.at(f, g) == f64::INFINITY;
                    }
                }
            }
            if sure {
                certain(rng, y, p, w, rates, flows, stage == 0, solver);
            }

            if solver == 3 { // Tau-leaping: the fastest rate at which any compartment is being left
                fastest = 0.0;
                for s in 0..p.src_starts.len() - 1 {
                    if p.src.at(p.src_flows.u(p.src_starts.u(s))) >= 0 {
                        for g in 0..ng {
                            let mut leaving = 0.0;
                            for k in p.src_starts.u(s)..p.src_starts.u(s + 1) {
                                leaving += mx(rates.at(p.src_flows.u(k), g), 0.0);
                            }
                            fastest = mx(fastest, leaving);
                        }
                    }
                }
            } else if sub == 0 && stage == 0 && n_sub <= 0 { // The fastest rate of any flow out of a compartment
                fastest = 0.0;
                for f in 0..nf {
                    if p.src.at(f) >= 0 {
                        for g in 0..ng {
                            fastest = mx(fastest, rates.at(f, g));
                        }
                    }
                }
            }
            if sub == 0 && stage == 0 {
                if solver < 2 && n_sub <= 0 {
                    for e in 0..p.mov_rate.len() {
                        fastest = mx(fastest, p.mov_rate.at(e));
                    }
                    n_sub = to_int((fastest*dt/0.25).ceil());
                }
                n_sub = n_sub.max(1);
                h = dt/n_sub as f64;
                if extra > 0 && solver < 2 {
                    release_rates(y, p, rates, dt);
                }
            }
            if solver >= 2 { // The stochastic solvers take it from here
                let mut end = dt;
                if pending && solver == 4 {
                    end = mn(end, dt/2.0);
                }
                if by_place {
                    gillespie_places(rng, y, p, w, end - elapsed, flows, varying || p.has_det);
                    h = end - elapsed;
                } else if solver == 4 {
                    h = event(rng, y, p, w, end - elapsed, flows);
                } else {
                    if solver == 3 {
                        h = mn(end - elapsed, 0.03/mx(fastest, 1e-12));
                    }
                    draw(rng, y, p, w, h, flows, solver == 3);
                }
                if p.has_det && !by_place {
                    settle(y, p, rates, h, flows);
                }
                elapsed += h;
                if pending && elapsed >= dt/2.0 + (if solver == 4 { -1e-12 } else { 1e-12 }) {
                    convey(rng, y, p, w, flows, true);
                    pending = false;
                }
                continue;
            }

            // The rate of change of each compartment, and the rate of each transition and mover
            let (dys, dfs) = (dy.plane(stage), df.plane(stage));
            dys.fill(0.0);
            dfs.fill(0.0);
            for m in 0..p.n_movers {
                for k in p.mov_starts.u(0, m)..p.mov_starts.u(0, m + 1) {
                    let c = p.mov_comps.u(k);
                    for e in p.mov_starts.u(1, m)..p.mov_starts.u(1, m + 1) {
                        let r = p.mov_rate.at(e);
                        if r != 0.0 {
                            let (g, j) = (p.mov_pair.u(0, e), p.mov_pair.u(1, e));
                            dys.set(c, g, dys.at(c, g) - ys.at(c, g)*r);
                            dys.add(c, j, ys.at(c, g)*r);
                            dfs.add(p.n_trans + m, g, ys.at(c, g)*r);
                            if stage == 0 && sub == 0 && has_moved {
                                moved_in.add(c, j, ys.at(c, g)*r);
                            }
                        }
                    }
                }
            }
            if solver == 1 { // Euler: outflows that would take more than everyone in a compartment are scaled down together (yt holds the scale)
                yt.fill(0.0);
                for f in 0..nf {
                    if p.src.at(f) >= 0 {
                        let s = p.src.u(f);
                        for g in 0..ng {
                            yt.add(s, g, rates.at(f, g)*h);
                        }
                    }
                }
                for c in 0..nc {
                    for g in 0..ng {
                        yt.set(c, g, if yt.at(c, g) > 1.0 { 1.0/yt.at(c, g) } else { 1.0 });
                    }
                }
            }
            for f in 0..nf { // The groups inside what each flow does, not the other way round, so a loop over them is simple enough to be vectorized
                let s = p.src.at(f);
                let r = rates.row(f);
                if s >= 0 {
                    let (from, out) = (ys.row(s as usize), dys.row(s as usize));
                    for g in 0..ng {
                        vals.set(g, r.at(g)*from.at(g));
                    }
                    if solver == 1 {
                        let scale = yt.row(s as usize);
                        for g in 0..ng {
                            vals.set(g, vals.at(g)*scale.at(g));
                        }
                    }
                    for g in 0..ng {
                        out.set(g, out.at(g) - vals.at(g));
                    }
                } else {
                    let inflow = p.inflow.row(f);
                    for g in 0..ng {
                        vals.set(g, r.at(g)*inflow.at(g));
                    }
                }
                if p.named.b(f) {
                    let (to, m) = (dfs.row(p.index.u(f)), p.record.at(f));
                    if m < 0 {
                        for g in 0..ng {
                            to.add(g, vals.at(g));
                        }
                    } else {
                        for g in 0..ng {
                            for j in 0..ng {
                                to.add(j, vals.at(g)*p.maps.at(m as usize, g, j));
                            }
                        }
                    }
                }
                for k in p.tgt_starts.u(f)..p.tgt_starts.u(f + 1) {
                    let (w, m) = (p.tgt_w.row(k), p.tgt_map.at(k));
                    if p.tgt_rec.at(k) >= 0 {
                        let to = dfs.row(p.tgt_rec.u(k));
                        for g in 0..ng {
                            to.add(g, vals.at(g)*w.at(g));
                        }
                    }
                    let to = dys.row(p.tgt_comp.u(k));
                    if m < 0 {
                        for g in 0..ng {
                            to.add(g, vals.at(g)*w.at(g));
                        }
                    } else {
                        for g in 0..ng {
                            for j in 0..ng {
                                to.add(j, vals.at(g)*w.at(g)*p.maps.at(m as usize, g, j));
                            }
                        }
                    }
                }
                for k in p.crd_starts.u(f)..p.crd_starts.u(f + 1) {
                    let (w, to) = (p.crd_w.row(k), dfs.row(p.crd_index.u(k)));
                    for g in 0..ng {
                        to.add(g, vals.at(g)*w.at(g));
                    }
                }
            }

            if auto && sub == 0 && stage == 0 { // ... nor grows by more than a quarter in one, through its inflows
                let mut growing = 0.0;
                let mut everyone = 0.0;
                for c in 0..nc {
                    for g in 0..ng {
                        everyone += y.at(c, g);
                    }
                }
                let least = 1e-9*everyone;
                for g in 0..ng {
                    for c in 0..nc {
                        let gain = dy.at(0, c, g) - if has_moved { moved_in.at(c, g) } else { 0.0 };
                        if gain > 0.0 && y.at(c, g) > least {
                            growing = mx(growing, gain/y.at(c, g));
                        }
                    }
                    for e in 0..p.set_starts.len() - 1 {
                        let mut tot = 0.0;
                        let mut grow = 0.0;
                        for k in p.set_starts.u(e)..p.set_starts.u(e + 1) {
                            let c = p.set_comps.u(k);
                            tot += y.at(c, g);
                            grow += dy.at(0, c, g) - if has_moved { moved_in.at(c, g) } else { 0.0 };
                        }
                        if grow > 0.0 && tot > least {
                            growing = mx(growing, grow/tot);
                        }
                    }
                }
                let more = to_int((growing*dt/0.25).ceil());
                if more > n_sub {
                    n_sub = more;
                    h = dt/n_sub as f64;
                }
            }

            if stage < stages - 1 { // Where the next stage is worked out
                let ahead = if stage == 2 { h } else { h/2.0 };
                for c in 0..nc {
                    for g in 0..ng {
                        yt.set(c, g, y.at(c, g) + ahead*dys.at(c, g));
                    }
                }
            }
        }

        if solver >= 2 {
        } else if stages == 1 {
            for c in 0..nc {
                for g in 0..ng {
                    y.add(c, g, h*dy.at(0, c, g));
                }
            }
            for f in 0..nflows {
                for g in 0..ng {
                    flows.add(f, g, h*df.at(0, f, g));
                }
            }
        } else {
            for c in 0..nc {
                for g in 0..ng {
                    y.add(c, g, h/6.0*(dy.at(0, c, g) + 2.0*dy.at(1, c, g) + 2.0*dy.at(2, c, g) + dy.at(3, c, g)));
                }
            }
            for f in 0..nflows {
                for g in 0..ng {
                    flows.add(f, g, h/6.0*(df.at(0, f, g) + 2.0*df.at(1, f, g) + 2.0*df.at(2, f, g) + df.at(3, f, g)));
                }
            }
        }
        sub += 1;
        if (timed && elapsed >= dt - 1e-12) || (!timed && sub >= n_sub) {
            break;
        }
    }
    if solver == 4 { // The exact solver moves people between groups once, over the whole step
        move_people(rng, y, p, dt, flows);
        for e in 0..p.mov_rate.len() {
            if p.mov_rate.at(e) != 0.0 {
                cells.set(5, p.places.u(p.mov_pair.u(0, e)), 0.0);
                cells.set(5, p.places.u(p.mov_pair.u(1, e)), 0.0);
            }
        }
    }
    if extra > 0 {
        convey(rng, y, p, w, flows, false);
    }
    for c in 0..nc {
        for g in 0..ng {
            if y.at(c, g) < 0.0 {
                y.set(c, g, 0.0);
            }
        }
    }
}

/// Whether any of the tables has a row for each step (`varies()`)
pub fn varies(tables: &Tables) -> bool {
    (0..4).any(|k| tables[k].n0() > 1)
}

/// Step ti of a run, from time point ti-1 to ti (`time_point()`): the deliveries at its start, the values of the step, the step, the events of the register at its end, and the places' fixed delays; returns the people the register could not find
#[allow(clippy::too_many_arguments)]
pub fn time_point(rng: Rng, y: A2<f64>, p: &Program, w: &Work, ti: usize, row: i64, dt: f64, n_sub: i64, solver: i64, flows: A2<f64>, tables: &Tables,
                  reg: &Register, sh: &Shifts, tally: A3<f64>) -> f64 {
    let whole = solver >= 2;
    if sh.to.n0() > 0 {
        apply_shifts(rng, y, sh, ti - 1, whole, tally);
        if (0..sh.due.n0()).any(|k| sh.due.b(k, ti - 1)) {
            wake(w);
        }
    }
    set_step(parts(p), row, tables);
    set_movers(p);
    step(rng, y, p, w, dt, n_sub, solver, flows, varies(tables));
    let mut short = 0.0;
    if reg.src.len() > 0 {
        short = apply_events(rng, y, reg, ti, whole);
        if solver == 4 && p.local {
            touch(reg, ti, p.places, w);
        }
    }
    if solver == 4 && p.local && p.has_det {
        settle_places(y, p, dt, flows);
    }
    short
}

/// Every step of a run from time point `start` (`run()`), into stocks and flows (time points from `start` on); returns the people the register could not find
#[allow(clippy::too_many_arguments)]
pub fn run(rng: Rng, y: A2<f64>, p: &Program, dt: f64, npts: usize, solver: i64, n_sub: i64, tables: &Tables, reg: &Register, sh: &Shifts, tally: Nd<f64, 4>,
           start: usize, stocks: A3<f64>, flows: A3<f64>) -> f64 {
    let w = workspace(y.n0(), y.n1(), p);
    let mut short = 0.0;
    stocks.plane(0).copy_from(y);
    for ti in start + 1..npts {
        let k = ti - start;
        short += time_point(rng, y, p, &w, ti, k as i64 - 1, dt, n_sub, solver, flows.plane(k), tables, reg, sh, tally.sub(k));
        stocks.plane(k).copy_from(y);
    }
    short
}

/// The outputs of `run_many()`
pub struct ManyOut {
    pub stocks: Nd<f64, 4>,
    pub flows: Nd<f64, 4>,
    pub stocks_g: Nd<f64, 4>,
    pub flows_g: Nd<f64, 4>,
    pub tally: Nd<f64, 5>,
    pub tally_g: Nd<f64, 5>,
}

/// The steps from time point `start` to npts-1 of many runs that differ only in their random numbers, filling in their results (`run_many()`)
#[allow(clippy::too_many_arguments)]
pub fn run_many(rngs: &[Rng], ys: A3<f64>, p: &Program, dt: f64, start: usize, npts: usize, solver: i64, n_sub: i64, tables: &Tables, reg: &Register, sh: &Shifts,
                o: &ManyOut, state_of: A1<i64>, rows: A1<i64>, grp_state: A1<i64>, grp_flow: A1<i64>) {
    let (nc, ng) = (ys.n1(), ys.n2());
    let nflows = p.n_trans + p.n_movers;
    let n_rows = if rows.len() > 0 { o.stocks_g.shape(1) } else { 0 };
    let w = workspace(nc, ng, p);
    let (_o1, now) = own2(nflows, ng, 0.0);
    let n_sh = sh.to.n0();
    let (_o2, now_sh) = own3(n_sh, 2, ng, 0.0);
    for (r, &rng) in rngs.iter().enumerate() {
        let y = ys.plane(r);
        wake(&w);
        for ti in start..npts {
            if ti > start {
                let k = ti - start;
                for d in 0..n_sh { now_sh.plane(d).fill(0.0); }
                time_point(rng, y, p, &w, ti, k as i64 - 1, dt, n_sub, solver, now, tables, reg, sh, now_sh);
                for d in 0..n_sh {
                    for g in 0..ng {
                        o.tally.add([r, ti, d, 0, 0], now_sh.at(d, 0, g));
                        o.tally.add([r, ti, d, 1, 0], now_sh.at(d, 1, g));
                        if n_rows > 0 {
                            o.tally_g.add([r, rows.u(ti), d, 0, g], now_sh.at(d, 0, g));
                            o.tally_g.add([r, rows.u(ti), d, 1, g], now_sh.at(d, 1, g));
                        }
                    }
                }
                for f in 0..nflows {
                    for g in 0..ng {
                        o.flows.add([r, ti, f, 0], now.at(f, g));
                        if n_rows > 0 && grp_flow.at(f) >= 0 {
                            o.flows_g.add([r, rows.u(ti), grp_flow.u(f), g], now.at(f, g));
                        }
                    }
                }
            } else if ti > 0 { // The start of a later part, which the part before has recorded
                continue;
            }
            let end = n_rows > 0 && (ti == rows.len() - 1 || rows.at(ti + 1) != rows.at(ti));
            for c in 0..nc {
                let s = state_of.u(c);
                for g in 0..ng {
                    o.stocks.add([r, ti, s, 0], y.at(c, g));
                    if end && grp_state.at(s) >= 0 {
                        o.stocks_g.add([r, rows.u(ti), grp_state.u(s), g], y.at(c, g));
                    }
                }
            }
        }
    }
}

/// What `advance()` takes besides the runs, the program, and the compartments
pub struct Advance<'a> {
    pub dt: f64,
    pub start: usize,
    pub end: usize,
    pub solver: i64,
    pub n_sub: i64,
    pub tables: [Nd<f64, 4>; 4], // fixed, mult, consts, rel_trans, each with a first axis for the runs (one entry if every run has the same)
    pub sh_vals: A2<f64>,
    pub reg: &'a Register,
    pub sh: &'a Shifts,
    pub state_of: A1<i64>,
    pub stocks: Nd<f64, 4>,
    pub flows: Nd<f64, 4>,
    pub row0: i64,
}

/// Steps start+1 to end of runs first to last-1 (particles of a particle filter), each from its own compartments, with its own random numbers (`advance()`)
pub fn advance(rngs: &[Rng], p: &Program, ys: A3<f64>, first: usize, last: usize, a: &Advance) {
    let (nc, ng) = (ys.n1(), ys.n2());
    let nflows = p.n_trans + p.n_movers;
    let by_group = a.stocks.shape(3) == ng;
    let w = workspace(nc, ng, p);
    let (_o1, now) = own2(nflows, ng, 0.0);
    let n_sh = a.sh.to.n0();
    let (_o2, tally) = own3(n_sh, 2, ng, 0.0); // Not kept
    let pt = parts(p);
    let (_o3, own) = own3(last - first, nc, ng, 0.0); // A copy of these runs, so threads do not write to the same lines of memory
    for r in first..last {
        own.plane(r - first).copy_from(ys.plane(r));
    }
    for r in first..last {
        let rng = rngs[r];
        let y = own.plane(r - first);
        let tables: Tables = std::array::from_fn(|k| a.tables[k].sub(if a.tables[k].shape(0) > 1 { r } else { 0 }));
        set_step(pt, 0, &tables); // The values of this run, for those that are held for the whole run
        if a.sh_vals.n0() > 1 {
            p.sh_val.copy_from(a.sh_vals.row(r));
        }
        wake(&w);
        for ti in a.start + 1..a.end + 1 {
            if n_sh > 0 {
                apply_shifts(rng, y, a.sh, ti - 1, a.solver >= 2, tally);
                if (0..a.sh.due.n0()).any(|k| a.sh.due.b(k, ti - 1)) {
                    wake(&w);
                }
            }
            let row = ti as i64 - 1 - a.row0;
            set_step(pt, row, &tables);
            set_movers(p);
            step(rng, y, p, &w, a.dt, a.n_sub, a.solver, now, varies(&tables));
            if a.reg.src.len() > 0 {
                apply_events(rng, y, a.reg, ti, a.solver >= 2);
                if a.solver == 4 && p.local {
                    touch(a.reg, ti, p.places, &w);
                }
            }
            if a.solver == 4 && p.local && p.has_det {
                settle_places(y, p, a.dt, now);
            }
            for s in 0..a.stocks.shape(2) {
                for g in 0..a.stocks.shape(3) {
                    a.stocks.set([r, ti, s, g], 0.0);
                }
            }
            for f in 0..nflows {
                for g in 0..a.flows.shape(3) {
                    a.flows.set([r, ti, f, g], 0.0);
                }
            }
            for c in 0..nc {
                for g in 0..ng {
                    a.stocks.add([r, ti, a.state_of.u(c), if by_group { g } else { 0 }], y.at(c, g));
                }
            }
            for f in 0..nflows {
                for g in 0..ng {
                    a.flows.add([r, ti, f, if by_group { g } else { 0 }], now.at(f, g));
                }
            }
        }
    }
    for r in first..last {
        ys.plane(r).copy_from(own.plane(r - first));
    }
}

/// `advance()` for every run, with the runs shared between threads, one program for each share (`advance_threads()`); each run draws from its own generator, so the results do not depend on the threads
pub fn advance_threads(rngs: &[Rng], progs: &[Program], ys: A3<f64>, a: &Advance) {
    let k = progs.len();
    let n = ys.n0();
    crate::passes::set_threads(k);
    crate::passes::par(k, |c| advance(rngs, &progs[c], ys, c*n/k, (c + 1)*n/k, a)); // On the threads of the passes, which wait between calls rather than start on each
}

/// For the exact solver by place: the counts settled over the step, with the people as they are at its end (`settle_places()`)
pub fn settle_places(y: A2<f64>, p: &Program, dt: f64, flows: A2<f64>) {
    let w = workspace(y.n0(), y.n1(), p);
    let (_o, expo) = own1(p.exp_set.len().max(1), 0.0);
    for i in 0..p.pl_starts.len() - 1 {
        place_rates(y, p, p.pl_starts.u(i), p.pl_starts.u(i + 1), w.rates, w.stack, expo);
    }
    settle(y, p, w.rates, dt, flows);
}

pub mod bind;
