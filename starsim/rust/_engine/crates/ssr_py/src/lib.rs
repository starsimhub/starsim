//! PyO3 bindings and the module registry for the fast modular engine.
//!
//! `run(...)` composes a sim from module specs (name + float params), runs the
//! whole loop in Rust, and returns a dict of result arrays. The registry maps
//! module names to constructors; adding a new module (e.g. ssr_sir) means a new
//! dependency + one match arm here -- ssr_core and the other module crates are
//! not recompiled.

use std::collections::HashMap;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ssr_core::{Demographics, Disease, Network, People, Sim};
use ssr_demographics::{Births, Deaths};
use ssr_randomnet::RandomNet;
use ssr_sir::Sir;
use ssr_sis::Sis;

type Pars = HashMap<String, f64>;

fn get(p: &Pars, key: &str, default: f64) -> f64 {
    *p.get(key).unwrap_or(&default)
}

// ---- Registry: name -> constructor ------------------------------------------
fn make_network(name: &str, p: &Pars, seed: u64) -> Box<dyn Network> {
    match name {
        "randomnet" => Box::new(RandomNet::new(
            get(p, "n_contacts", 10.0) as usize,
            get(p, "dur", 0.0) as f32,
            get(p, "beta", 1.0) as f32,
            seed,
        )),
        other => panic!("unknown network module '{other}'"),
    }
}

fn make_disease(name: &str, p: &Pars, n: usize, n_steps: usize, seed: u64) -> Box<dyn Disease> {
    match name {
        "sis" => Box::new(Sis::new(
            n, n_steps,
            get(p, "beta", 0.05),
            get(p, "init_prev", 0.01),
            get(p, "dur_inf", 10.0),
            get(p, "dur_inf_std", 1.0),
            get(p, "waning", 0.05) as f32,
            get(p, "imm_boost", 1.0) as f32,
            seed,
        )),
        "sir" => Box::new(Sir::new(
            n, n_steps,
            get(p, "beta", 0.05),
            get(p, "init_prev", 0.01),
            get(p, "dur_inf", 10.0),
            get(p, "dur_inf_std", 1.0),
            seed,
        )),
        other => panic!("unknown disease module '{other}'"),
    }
}

fn make_demographics(name: &str, p: &Pars, seed: u64) -> Box<dyn Demographics> {
    match name {
        "births" => Box::new(Births::new(get(p, "birth_prob", 0.0), seed)),
        "deaths" => Box::new(Deaths::new(get(p, "death_prob", 0.0), seed)),
        other => panic!("unknown demographics module '{other}'"),
    }
}
// -----------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (n_agents, n_steps, seed, networks, diseases, demographics=Vec::new()))]
fn run<'py>(
    py: Python<'py>,
    n_agents: usize,
    n_steps: usize,
    seed: u64,
    networks: Vec<(String, Pars)>,
    diseases: Vec<(String, Pars)>,
    demographics: Vec<(String, Pars)>,
) -> PyResult<Bound<'py, PyDict>> {
    let nets: Vec<Box<dyn Network>> = networks
        .iter()
        .enumerate()
        .map(|(i, (name, p))| make_network(name, p, seed.wrapping_add(1000 + i as u64)))
        .collect();
    let diss: Vec<Box<dyn Disease>> = diseases
        .iter()
        .enumerate()
        .map(|(i, (name, p))| make_disease(name, p, n_agents, n_steps, seed.wrapping_add(2000 + i as u64)))
        .collect();
    let dems: Vec<Box<dyn Demographics>> = demographics
        .iter()
        .enumerate()
        .map(|(i, (name, p))| make_demographics(name, p, seed.wrapping_add(3000 + i as u64)))
        .collect();

    let mut sim = Sim { people: People::new(n_agents), networks: nets, diseases: diss, demographics: dems, n_steps };
    sim.run();

    let out = PyDict::new_bound(py);
    for d in &sim.diseases {
        let prefix = d.name();
        for (k, v) in d.results() {
            out.set_item(format!("{prefix}_{k}"), v.clone().into_pyarray_bound(py))?;
        }
    }
    Ok(out)
}

#[pymodule]
fn ssr_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
