//! The small kernels of `arrays.py`, `networks.py`, `distributions.py`, `people.py` and `utils.py`: the same arguments and results as the Numba functions they replace

use crate::draws;

/// The UIDs whose value is truthy (or not) (`arrays.nb_true_raw()`)
pub fn true_raw<T: Copy + PartialEq + Default>(auids: &[i64], raw: &[T], want: bool) -> Vec<i64> {
    let zero = T::default();
    let mut out = Vec::with_capacity(auids.len() + 1);
    let spare = out.spare_capacity_mut();
    let mut m = 0;
    for &u in auids { // Written every time (branchless), kept only if m advances
        spare[m].write(u);
        m += ((raw[u as usize] != zero) == want) as usize;
    }
    unsafe { out.set_len(m) };
    out
}

/// The (source, target) of the edges that transmit (`networks.nb_transmit()`)
pub fn transmit<V: Copy + Into<f64>>(src: &[i64], trg: &[i64], p: impl Fn(usize, usize, usize) -> f64, rvs: &[V]) -> (Vec<i64>, Vec<i64>) {
    let (mut s_out, mut t_out) = (Vec::with_capacity(src.len()), Vec::with_capacity(src.len()));
    let (so, to) = (s_out.spare_capacity_mut(), t_out.spare_capacity_mut());
    let mut m = 0;
    for i in 0..src.len() { // Written every time (branchless), kept only if m advances
        so[m].write(src[i]);
        to[m].write(trg[i]);
        m += (p(i, src[i] as usize, trg[i] as usize) > rvs[i].into()) as usize;
    }
    unsafe {
        s_out.set_len(m);
        t_out.set_len(m);
    }
    (s_out, t_out)
}

/// A float to store a draw in
pub trait Real: Copy { fn of(x: f64) -> Self; }
impl Real for f32 { fn of(x: f64) -> Self { x as f32 } }
impl Real for f64 { fn of(x: f64) -> Self { x } }

/// One uniform for each slot, from a counter-based hash (`distributions._hash_uniforms_fill()`)
pub fn hash_uniforms<R: Real>(seed: i64, ind: i64, slots: &[u64], out: &mut [R], shift: u64, scale: f64) {
    let key = draws::key(seed, ind);
    for (o, &slot) in out.iter_mut().zip(slots) {
        *o = R::of(draws::crn_uniform(key, slot, shift, scale));
    }
}

/// The values at some indices (`arrays.nb_indexer()`), negative ones counted from the end, as in Numba
pub fn indexer<T: Copy>(arr: &[T], inds: &[i64]) -> Vec<T> {
    let n = arr.len() as i64;
    inds.iter().map(|&i| arr[(if i < 0 { i + n } else { i }) as usize]).collect()
}

/// The values at some indices of a strided view, of `len` values `stride` items apart from `base` (`arrays.nb_indexer()`, as Numba takes any layout)
pub fn indexer_strided<T: Copy>(base: *const T, stride: isize, len: usize, inds: &[i64]) -> Vec<T> {
    inds.iter().map(|&i| {
        let i = if i < 0 { i + len as i64 } else { i }; // Negative ones counted from the end, as in Numba
        assert!((i as usize) < len, "index out of bounds");
        unsafe { *base.offset(i as isize*stride) }
    }).collect()
}

/// The UIDs whose value is truthy (or not), from the values of the active agents (`arrays.nb_true()`)
pub fn true_vals<T: Copy + PartialEq + Default>(auids: &[i64], vals: &[T], want: bool) -> Vec<i64> {
    let zero = T::default();
    let vals = &vals[..auids.len()];
    let mut out = Vec::with_capacity(auids.len() + 1);
    let spare = out.spare_capacity_mut();
    let mut m = 0;
    for (k, &u) in auids.iter().enumerate() { // Written every time (branchless), kept only if m advances
        spare[m].write(u);
        m += ((vals[k] != zero) == want) as usize;
    }
    unsafe { out.set_len(m) };
    out
}

/// The number of active agents whose value is truthy (`arrays.nb_count()`)
pub fn count<T: Copy + PartialEq + Default>(auids: &[i64], raw: &[T]) -> i64 {
    let zero = T::default();
    auids.iter().map(|&u| (raw[u as usize] != zero) as i64).sum()
}

/// An index, of any integer type
pub trait Ix: Copy {
    fn ix(self) -> usize;
}
macro_rules! ix {
    ($($t:ty),+) => { $(impl Ix for $t { #[inline(always)] fn ix(self) -> usize { self as usize } })+ };
}
ix!(i16, i32, i64, u32, u64);

/// In-place Fisher-Yates shuffle, from uniforms (`networks.fisher_yates_shuffle()`)
pub fn shuffle<T: Copy, R: Copy + Into<f64>>(arr: &mut [T], randvals: &[R]) {
    let n = arr.len();
    for k in 0..n.saturating_sub(1) {
        let i = n - 1 - k; // Position to swap, working from the top down
        let j = draws::to_i64(randvals[k].into()*(i + 1) as f64) as usize;
        arr.swap(i, j);
    }
}

/// A step of a time (an int, or a float, as ti_dead may be)
pub trait Step: Copy {
    const IS_INT: bool;
    fn to_i64(self) -> i64;
    fn to_f64(self) -> f64;
}
macro_rules! step {
    ($is_int:expr, $($t:ty),+) => {
        $(impl Step for $t {
            const IS_INT: bool = $is_int;
            #[inline(always)] fn to_i64(self) -> i64 { self as i64 }
            #[inline(always)] fn to_f64(self) -> f64 { self as f64 }
        })+
    };
}
step!(true, i16, i32, i64);
step!(false, f32, f64);

/// The number of active agents who are no longer alive because they died, and because they left, by step ti (`people.count_gone()`)
pub fn count_gone<T: Copy>(auids: &[i64], alive: &[bool], ti_dead: &[T], ti_removed: &[T], by: impl Fn(T) -> bool) -> (i64, i64) {
    let (mut deaths, mut removed) = (0, 0);
    for &u in auids {
        let u = u as usize;
        if !alive[u] {
            deaths += by(ti_dead[u]) as i64;
            removed += by(ti_removed[u]) as i64;
        }
    }
    (deaths, removed)
}

/// Mark the partners of the agents flagged in is_ind, in both directions of each edge (`utils._mark_partners()`)
pub fn mark_partners<I: Ix>(p1: &[I], p2: &[I], is_ind: &[bool], is_partner: &mut [bool]) {
    for (&a, &b) in p1.iter().zip(p2) {
        let (a, b) = (a.ix(), b.ix());
        if is_ind[a] {
            is_partner[b] = true;
        }
        if is_ind[b] {
            is_partner[a] = true;
        }
    }
}

/// Uniforms combined from those of several distributions, from their bits (as unsigned ints of their width): x*y ^ (x - y), over
/// the largest such int (`multi_random.combine_rvs()` and `combine2_rvs()`); times its reciprocal, as Numba's fastmath divides
macro_rules! combine {
    ($name:ident, $U:ty) => {
        avx2! {
            pub fn $name(rvs: &[&[$U]], int_max: f64) -> Vec<f64> {
                let recip = 1.0/int_max;
                let n = rvs[0].len();
                let mix = |x: $U, y: $U| x.wrapping_mul(y) ^ x.wrapping_sub(y);
                if rvs.len() == 2 { // The usual case (an edge's source and target), in one pass
                    return rvs[0].iter().zip(&rvs[1][..n]).map(|(&x, &y)| mix(x, y) as f64*recip).collect();
                }
                let mut ints = rvs[0].to_vec();
                for r in &rvs[1..] {
                    for (x, &y) in ints.iter_mut().zip(&r[..n]) {
                        *x = mix(*x, y);
                    }
                }
                ints.iter().map(|&x| x as f64*recip).collect()
            }
        }
    };
}
combine!(combine_rvs_32, u32);
combine!(combine_rvs_64, u64);

/// The bindings, with the same arguments and results as the Numba functions they replace
pub mod bind {
    use numpy::{PyArrayDyn, PyUntypedArray, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use crate::small::Step;
    use crate::small;
    use crate::py::{arr, sl, sl_mut, strided};

    #[pyfunction]
    fn true_raw<'py>(py: Python<'py>, auids: &Bound<'py, PyAny>, raw: &Bound<'py, PyAny>, want: bool) -> PyResult<Bound<'py, PyAny>> {
        typed!(raw, T in [bool, u8, i8, i16, i32, i64, f32, f64] => Ok(arr(py, small::true_raw::<T>(sl(auids)?, sl(raw)?, want))))
    }

    #[pyfunction]
    fn transmit<'py>(py: Python<'py>, src: &Bound<'py, PyAny>, trg: &Bound<'py, PyAny>, rel_trans: &Bound<'py, PyAny>, rel_sus: &Bound<'py, PyAny>, beta: &Bound<'py, PyAny>,
                     rvs: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        // rel_trans[u]*rel_sus[v]*beta[i] in the type Numba gives it: float32 if all three are, else float64
        typed!(rvs, V in [f64, f32] => {
            if let (Ok(t), Ok(s), Ok(b)) = (sl::<f32>(rel_trans), sl::<f32>(rel_sus), sl::<f32>(beta)) {
                let (s, t) = small::transmit(sl(src)?, sl(trg)?, |i, u, v| (t[u]*s[v]*b[i]) as f64, sl::<V>(rvs)?);
                return Ok((arr(py, s), arr(py, t)));
            }
            typed!(beta, B in [f64, f32] => with_rel!(rel_trans, rel_sus, rel => {
                let b = sl::<B>(beta)?;
                let (s, t) = small::transmit(sl(src)?, sl(trg)?, |i, u, v| rel(u, v)*(b[i] as f64), sl::<V>(rvs)?);
                Ok((arr(py, s), arr(py, t)))
            }))
        })
    }

    #[pyfunction]
    fn hash_uniforms(seed: i64, ind: i64, slots: &Bound<PyAny>, out: &Bound<PyAny>, shift: u64, scale: f64) -> PyResult<()> {
        typed!(out, R in [f32, f64] => Ok(small::hash_uniforms::<R>(seed, ind, sl(slots)?, sl_mut(out)?, shift, scale)))
    }

    #[pyfunction]
    fn indexer<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, inds: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        typed!(a, T in [bool, u8, i8, i16, i32, i64, u64, f32, f64] => Ok(arr(py, match sl::<T>(a) {
            Ok(x) => small::indexer::<T>(x, sl(inds)?),
            Err(_) => { // A strided view
                let (base, stride, len) = strided::<T>(a)?;
                small::indexer_strided::<T>(base, stride, len, sl(inds)?)
            }
        })))
    }

    #[pyfunction]
    fn true_vals<'py>(py: Python<'py>, auids: &Bound<'py, PyAny>, vals: &Bound<'py, PyAny>, want: bool) -> PyResult<Bound<'py, PyAny>> {
        typed!(vals, T in [bool, u8, i8, i16, i32, i64, f32, f64] => Ok(arr(py, small::true_vals::<T>(sl(auids)?, sl(vals)?, want))))
    }

    #[pyfunction]
    fn count(auids: &Bound<PyAny>, raw: &Bound<PyAny>) -> PyResult<i64> {
        typed!(raw, T in [bool, u8, i8, i16, i32, i64, f32, f64] => Ok(small::count::<T>(sl(auids)?, sl(raw)?)))
    }

    #[pyfunction]
    fn shuffle<'py>(arr: &Bound<'py, PyAny>, randvals: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        typed!(randvals, R in [f64, f32] => typed!(arr, T in [i64, i32, i16, u64, u32, f64, f32] => {
            small::shuffle::<T, R>(sl_mut(arr)?, sl(randvals)?);
            Ok(arr.clone())
        }))
    }

    /// Int steps and an int ti are compared as int64s, and otherwise as float64s (as Numba compares them)
    #[pyfunction]
    fn count_gone(auids: &Bound<PyAny>, alive: &Bound<PyAny>, ti_dead: &Bound<PyAny>, ti_removed: &Bound<PyAny>, ti: &Bound<PyAny>) -> PyResult<(i64, i64)> {
        let (auids, alive) = (sl::<i64>(auids)?, sl::<bool>(alive)?);
        let int_ti = ti.extract::<i64>().ok(); // An int or NumPy int (a float doesn't convert)
        let float_ti = ti.extract::<f64>()?;
        typed!(ti_dead, T in [i16, i32, i64, f32, f64] => {
            let (d, r) = (sl::<T>(ti_dead)?, sl::<T>(ti_removed)?);
            match int_ti {
                Some(t) if T::IS_INT => Ok(small::count_gone::<T>(auids, alive, d, r, |x| x.to_i64() <= t)),
                _ => Ok(small::count_gone::<T>(auids, alive, d, r, |x| x.to_f64() <= float_ti)),
            }
        })
    }

    #[pyfunction]
    fn mark_partners(p1: &Bound<PyAny>, p2: &Bound<PyAny>, is_ind: &Bound<PyAny>, is_partner: &Bound<PyAny>) -> PyResult<()> {
        typed!(p1, I in [i64, i32, i16, u64, u32] => Ok(small::mark_partners::<I>(sl(p1)?, sl(p2)?, sl(is_ind)?, sl_mut(is_partner)?)))
    }

    /// The bits of an array of 8-byte or 4-byte numbers (floats, or the ints of a hash), as `.view(np.uint64)` or `.view(np.uint32)` gives them
    macro_rules! bits {
        ($name:ident, $U:ty, [$($t:ty),+]) => {
            fn $name<'a>(o: &'a Bound<PyAny>) -> PyResult<&'a [$U]> {
                if o.cast::<PyUntypedArray>().is_ok_and(|a| a.len() == 0) { // An empty array of any type (as the agent solver gives for no edges), as .view() takes it
                    return Ok(&[]);
                }
                $( if let Ok(x) = sl::<$t>(o) {
                    return Ok(if x.is_empty() { &[] } else { unsafe { std::slice::from_raw_parts(x.as_ptr() as *const $U, x.len()) } });
                } )+
                Err(PyTypeError::new_err("the Rust combine_rvs() takes arrays of numbers of the width of the int type"))
            }
        };
    }
    bits!(bits64, u64, [f64, u64, i64]);
    bits!(bits32, u32, [f32, u32, i32]);

    /// The combined uniforms, for the int type's width (64 or 32 bits)
    fn combined<'py>(py: Python<'py>, items: &[Bound<'py, PyAny>], int_type: &Bound<'py, PyAny>, int_max: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let int_max = int_max.extract::<u64>()? as f64;
        match int_type.getattr("__name__")?.extract::<String>()?.as_str() {
            "uint64" => Ok(arr(py, small::combine_rvs_64(&items.iter().map(|a| bits64(a)).collect::<PyResult<Vec<_>>>()?, int_max))),
            "uint32" => Ok(arr(py, small::combine_rvs_32(&items.iter().map(|a| bits32(a)).collect::<PyResult<Vec<_>>>()?, int_max))),
            _ => Err(PyTypeError::new_err("the Rust combine_rvs() takes uint32 or uint64")),
        }
    }

    #[pyfunction]
    fn combine_rvs<'py>(py: Python<'py>, rvs_list: &Bound<'py, PyAny>, int_type: &Bound<'py, PyAny>, int_max: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let items = rvs_list.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        combined(py, &items, int_type, int_max)
    }

    #[pyfunction]
    fn combine2_rvs<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>, int_type: &Bound<'py, PyAny>, int_max: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        combined(py, &[a.clone(), b.clone()], int_type, int_max)
    }

    /// Add these kernels to the module
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(true_raw, m)?)?;
        m.add_function(wrap_pyfunction!(transmit, m)?)?;
        m.add_function(wrap_pyfunction!(hash_uniforms, m)?)?;
        m.add_function(wrap_pyfunction!(indexer, m)?)?;
        m.add_function(wrap_pyfunction!(true_vals, m)?)?;
        m.add_function(wrap_pyfunction!(count, m)?)?;
        m.add_function(wrap_pyfunction!(shuffle, m)?)?;
        m.add_function(wrap_pyfunction!(count_gone, m)?)?;
        m.add_function(wrap_pyfunction!(mark_partners, m)?)?;
        m.add_function(wrap_pyfunction!(combine_rvs, m)?)?;
        m.add_function(wrap_pyfunction!(combine2_rvs, m)?)?;
        Ok(())
    }
}
