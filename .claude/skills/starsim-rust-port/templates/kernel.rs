//! TEMPLATE: a Phase-1 Starsim Rust kernel.
//!
//! Copy the function below into `starsim/rust/_crate/src/lib.rs` and register it
//! in the `#[pymodule]` block. Rename `my_kernel` and adjust the arguments to
//! the numeric core of the method you are porting.
//!
//! Phase-1 rules (see starsim/rust/SUPPORTED_SUBSET.md):
//!   * Do NOT draw random numbers here. Accept already-drawn arrays as input.
//!   * Take zero-copy views of the existing numpy buffers (PyReadonlyArray1).
//!   * State arrays that Python indexes by UID are passed as their `.raw` buffer
//!     (full length) and indexed by the UID arrays.
//!   * Perform float ops in the SAME ORDER numpy does, or you lose byte-identity.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn my_kernel<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray1<'py, i64>,          // e.g. UIDs to operate on
    state_raw: PyReadonlyArray1<'py, f64>,    // a full-length `.raw` state buffer, indexed by UID
    randvals: PyReadonlyArray1<'py, f64>,     // randoms drawn in Python (Phase 1)
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let idx = idx.as_array();
    let state = state_raw.as_array();
    let rand = randvals.as_array();

    let n = idx.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let u = idx[i];
        // ... numeric core here, mirroring the Python exactly ...
        if state[u as usize] > rand[i] {
            out.push(u);
        }
    }
    Ok(out.into_pyarray_bound(py))
}

// Register in the #[pymodule] fn:
//     m.add_function(wrap_pyfunction!(my_kernel, m)?)?;
