//! What the bindings of every kernel share: Python's arrays as Rust slices, and a body run for each element type an array may have

use numpy::npyffi::NPY_ARRAY_WRITEABLE;
use numpy::{Element, PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PySlice;

/// A contiguous array of any dimensions as a pointer and its number of items, checked to be writeable if it is to be written to
fn contiguous<T: Element>(o: &Bound<PyAny>, write: bool) -> PyResult<(*mut T, usize)> {
    let a = o.cast::<PyArrayDyn<T>>()?;
    if !a.is_c_contiguous() {
        return Err(PyTypeError::new_err("the arrays of the Rust kernels must be contiguous"));
    }
    if write && unsafe { (*a.as_array_ptr()).fields.flags } & NPY_ARRAY_WRITEABLE == 0 {
        return Err(PyTypeError::new_err("an array the Rust kernels write to must be writeable"));
    }
    Ok((a.data(), a.len()))
}

/// A 1-D array as a pointer and a length (it has to be contiguous)
pub fn col<T: Element>(o: &Bound<PyAny>) -> PyResult<(*mut T, usize)> {
    if o.cast::<PyUntypedArray>()?.ndim() != 1 {
        return Err(PyTypeError::new_err("the arrays of the Rust kernels must be 1-D here"));
    }
    contiguous(o, false)
}

/// A 1-D array as a slice (it has to be contiguous); the caller's arrays outlive the call
pub fn sl<'a, T: Element>(o: &'a Bound<PyAny>) -> PyResult<&'a [T]> {
    let (p, n) = col::<T>(o)?;
    Ok(if n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } })
}

/// A contiguous array of any dimensions as a flat slice, by rows (e.g. a table of hazards)
pub fn flat<'a, T: Element>(o: &'a Bound<PyAny>) -> PyResult<&'a [T]> {
    let (p, n) = contiguous::<T>(o, false)?;
    Ok(if n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p, n) } })
}

/// A 1-D array of any layout (a strided view), as a pointer to its first item, the distance between items (in items), and its length
pub fn strided<T: Element>(o: &Bound<PyAny>) -> PyResult<(*const T, isize, usize)> {
    let a = o.cast::<PyArrayDyn<T>>()?;
    let step = std::mem::size_of::<T>() as isize;
    if a.ndim() != 1 || a.strides()[0] % step != 0 {
        return Err(PyTypeError::new_err("an array the Rust kernels do not take (not 1-D, or misaligned)"));
    }
    Ok((a.data() as *const T, a.strides()[0]/step, a.len()))
}

/// A 1-D array as a slice to write
#[allow(clippy::mut_from_ref)]
pub fn sl_mut<'a, T: Element>(o: &'a Bound<PyAny>) -> PyResult<&'a mut [T]> {
    if o.cast::<PyUntypedArray>()?.ndim() != 1 {
        return Err(PyTypeError::new_err("the arrays of the Rust kernels must be 1-D here"));
    }
    let (p, n) = contiguous::<T>(o, true)?;
    Ok(if n == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(p, n) } })
}

/// A contiguous array of any dimensions as a flat slice to write, by rows (e.g. the scratch rows of the draws)
#[allow(clippy::mut_from_ref)]
pub fn flat_mut<'a, T: Element>(o: &'a Bound<PyAny>) -> PyResult<&'a mut [T]> {
    let (p, n) = contiguous::<T>(o, true)?;
    Ok(if n == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(p, n) } })
}

/// Run `$body` with `$T` the element type of array `$o`, one of the types listed
macro_rules! typed {
    ($o:expr, $T:ident in [$($t:ty),+] => $body:expr) => {{
        $( if $o.cast::<PyArrayDyn<$t>>().is_ok() { type $T = $t; return $body; } )+
        Err(PyTypeError::new_err("an array of a type the Rust kernels do not take"))
    }};
}

/// `rel_trans[u]*rel_sus[v]` for agents u and v, in the type Numba gives it: float32 if both are, else float64
macro_rules! with_rel {
    ($rt:expr, $rs:expr, $rel:ident => $body:expr) => {{
        if let (Ok(t), Ok(s)) = (sl::<f32>($rt), sl::<f32>($rs)) {
            let $rel = |u: usize, v: usize| (t[u]*s[v]) as f64;
            return $body;
        }
        typed!($rt, T in [f32, f64] => typed!($rs, S in [f32, f64] => {
            let (t, s) = (sl::<T>($rt)?, sl::<S>($rs)?);
            let $rel = |u: usize, v: usize| t[u] as f64*s[v] as f64;
            $body
        }))
    }};
}

pub fn arr<'py, T: Element>(py: Python<'py>, v: Vec<T>) -> Bound<'py, PyAny> {
    PyArray1::from_vec(py, v).into_any()
}

/// The first m items of an array (a view, as Numba gives)
pub fn head<'py>(o: &Bound<'py, PyAny>, m: usize) -> PyResult<Bound<'py, PyAny>> {
    o.get_item(PySlice::new(o.py(), 0, m as isize, 1))
}

/// A function compiled twice, for any x86-64 CPU and for one with AVX2 (chosen when called), for loops that vectorize: AVX2
/// makes them twice as wide, and Numba compiles for the CPU it runs on. What the function calls should be `#[inline(always)]`,
/// so it is compiled for AVX2 too. The results are the same: Rust never fuses a multiply and an add
macro_rules! avx2 {
    ($(#[$m:meta])* pub fn $name:ident $(<$($g:ident: $b:ident),*>)? ($($a:ident: $t:ty),*) $(-> $r:ty)? $body:block) => {
        $(#[$m])* pub fn $name $(<$($g: $b),*>)? ($($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn wide $(<$($g: $b),*>)? ($($a: $t),*) $(-> $r)? $body
                if std::arch::is_x86_feature_detected!("avx2") {
                    return unsafe { wide($($a),*) };
                }
            }
            $body
        }
    };
}
