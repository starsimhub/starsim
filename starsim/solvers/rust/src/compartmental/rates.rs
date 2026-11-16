//! Working out the rates of the flows (the operations of a rate, on a stack, and the values worked out once a step), and exposure through
//! the mixing terms (the section "Rates" of `starsim/solvers/numba/compartmental.py`, the functions of the same names)

use super::*;

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

/// Member i of the sets of compartments, counted in group g: the number in its compartment, times its weight if it has one (`member()`)
#[inline(always)]
pub fn member(ys: A2<f64>, p: &Program, i: usize, g: usize) -> f64 {
    let v = ys.at(p.set_comps.u(i), g);
    if p.set_w.at(i) < 0 { v } else { v*p.consts.at(p.set_w.u(i), g) }
}

/// A count on the stack of a rate: the number v (COUNT), or its fraction of the n people there (FRAC), 0 where there is no one (`leaf_frac()`)
#[inline(always)]
pub fn leaf_frac(op: i64, v: f64, n: f64) -> f64 {
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
                out.set(i, g, leaf_frac(op, num.at(pl), people.at(pl)));
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

/// The product of the time fraction between places with a vector of places by levels, into out: `v @ frac` if `by_columns`, else `frac @ v` (`frac()`)
pub fn frac(pair: A2<i64>, vals: A1<f64>, v: A1<f64>, size: usize, block: usize, out: A1<f64>, by_columns: bool) {
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
