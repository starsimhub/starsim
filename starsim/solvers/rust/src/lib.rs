//! Starsim's compiled kernels, in Rust (see `starsim/solvers/backends/rust.py`), each with the same arguments and results as the Numba function it replaces.
//! Each file holds its kernels and their bindings, and adds them to the module with its `register()`: `draws.rs` (`starsim/solvers/backends/draws.py`,
//! which the others use), `passes.rs` (`starsim/solvers/backends/passes.py`), `small.rs` (the small kernels of `arrays.py`, `networks.py`,
//! `distributions.py`, `people.py` and `utils.py`), `kernels.rs` and `kernels/` (the compartmental kernels, `starsim/solvers/backends/kernels.py`), and
//! `chain.rs` (the chain binomial's handler for JAX, `starsim/solvers/backends/chain.py`)

#[macro_use]
mod py;
mod draws;
mod passes;
mod small;
mod kernels;
mod chain;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    draws::bind::register(m)?;
    passes::bind::register(m)?;
    small::bind::register(m)?;
    kernels::bind::register(m)?;
    chain::bind::register(m)?;
    Ok(())
}
