//! Starsim's compiled kernels, in Rust (see `starsim/solvers/backends/rust.py`), each with the same arguments and results as the Numba function it replaces.
//! Each file holds its kernels and their bindings, and adds them to the module with its `register()`; each is named after its Numba file in `starsim/solvers/numba/`:
//! `draws.rs` (which the others use), `passes.rs`, `agent_events.rs`, and `compartmental.rs` and `compartmental/`; and `small.rs` (the small
//! kernels of `internals/arrays.py`, `networks.py`, `internals/dists.py`, `distributions.py` and `people.py`)

#[macro_use]
mod py;
mod draws;
mod passes;
mod agent_events;
mod small;
mod compartmental;

use pyo3::prelude::*;

#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("source_hash", env!("SOURCE_HASH").parse::<u32>().unwrap())?; // See build.rs
    draws::bind::register(m)?;
    passes::bind::register(m)?;
    agent_events::bind::register(m)?;
    small::bind::register(m)?;
    compartmental::bind::register(m)?;
    Ok(())
}
