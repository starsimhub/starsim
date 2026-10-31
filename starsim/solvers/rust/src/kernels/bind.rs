//! The bindings of the compartmental kernels, with the same arguments and results as the Numba functions in kernels.py.
//!
//! The namedtuples are read as views of their arrays, in the order of their fields. An array a kernel writes to must be float64
//! with its last axis contiguous, as the engines make them; one it only reads may be of another type or layout, and is then
//! converted for the call (kept in `Keep` until it returns), as Numba compiles a version for each.

use numpy::{Element, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use super::*;

/// The arrays converted for a call
type Keep<'py> = Vec<Bound<'py, PyAny>>;

/// An array's pointer, shape, and strides (in elements), converted if it is not of type T with a contiguous last axis (or refused, if the kernel writes to it)
fn raw<'py, T: Element + Copy>(o: &Bound<'py, PyAny>, keep: &mut Keep<'py>, write: bool, ndim: usize) -> PyResult<(*mut T, [usize; 8], [usize; 8])> {
    let py = o.py();
    let ok = |a: &Bound<'py, PyAny>| -> Option<(*mut T, [usize; 8], [usize; 8])> {
        let arr = a.cast::<PyArrayDyn<T>>().ok()?;
        if arr.ndim() != ndim {
            return None;
        }
        let size = std::mem::size_of::<T>() as isize;
        let mut shape = [0; 8]; // On the stack, since a step reads some 50 arrays
        shape[..ndim].copy_from_slice(arr.shape());
        let mut strides = [0; 8];
        let empty = arr.len() == 0; // Its strides are never used
        for (k, &s) in arr.strides().iter().enumerate() {
            if shape[k] > 1 && !empty {
                if s < 0 || s % size != 0 || (k == ndim - 1 && s != size) {
                    return None;
                }
                strides[k] = (s/size) as usize;
            }
        }
        Some((arr.data(), shape, strides))
    };
    if let Some(r) = ok(o) {
        return Ok(r);
    }
    let what = o.cast::<PyUntypedArray>().map(|a| format!("{} array of {} dimensions", a.dtype().str().map(|s| s.to_string()).unwrap_or_default(), a.ndim())).unwrap_or_else(|_| o.get_type().to_string());
    if write {
        return Err(PyTypeError::new_err(format!("the Rust compartmental kernels write to a float64 array of {ndim} dimensions with a contiguous last axis, not a {what}")));
    }
    let conv = py.import("numpy")?.call_method1("ascontiguousarray", (o, numpy::dtype::<T>(py)))?;
    let r = ok(&conv).ok_or_else(|| PyTypeError::new_err(format!("the Rust compartmental kernels take an array of {ndim} dimensions here, not a {what}")))?;
    keep.push(conv);
    Ok(r)
}

fn a1<'py, T: Element + Copy>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<A1<T>> {
    let (p, n, _) = raw::<T>(o, k, w, 1)?;
    Ok(A1::new(p, n[0]))
}

fn a2<'py, T: Element + Copy>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<A2<T>> {
    let (p, n, s) = raw::<T>(o, k, w, 2)?;
    Ok(A2::new(p, n[0], n[1], s[0]))
}

fn a3<'py, T: Element + Copy>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<A3<T>> {
    let (p, n, s) = raw::<T>(o, k, w, 3)?;
    Ok(A3::new(p, [n[0], n[1], n[2]], s[0], s[1]))
}

fn nd<'py, const N: usize>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<Nd<f64, N>> {
    let (p, n, s) = raw::<f64>(o, k, w, N)?;
    Ok(Nd::new(p, n[..N].try_into().unwrap(), s[..N].try_into().unwrap()))
}

/// A boolean array, as bytes
fn b1<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<A1<u8>> {
    let a = a1::<bool>(o, k, w)?;
    Ok(A1::new(a.ptr() as *mut u8, a.len()))
}

fn b2<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>, w: bool) -> PyResult<A2<u8>> {
    let (p, n, s) = raw::<bool>(o, k, w, 2)?;
    Ok(A2::new(p as *mut u8, n[0], n[1], s[0]))
}

fn uint(o: &Bound<PyAny>) -> PyResult<usize> { Ok(o.extract::<i64>()?.max(0) as usize) }

/// The items of a namedtuple, checked for their number
fn items<'py>(o: &Bound<'py, PyAny>, n: usize, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let t = o.cast::<PyTuple>()?;
    if t.len() != n {
        return Err(PyTypeError::new_err(format!("the Rust kernels take a {what} of {n} fields, not {}", t.len())));
    }
    Ok(t.iter().collect())
}

/// A program (`kernels.Program`)
fn program<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>) -> PyResult<Program> {
    let f = items(o, 55, "Program")?;
    Ok(Program {
        n_trans: uint(&f[0])?, src: a1(&f[1], k, false)?, named: b1(&f[2], k, false)?, index: a1(&f[3], k, false)?, record: a1(&f[4], k, false)?,
        inflow: a2(&f[5], k, true)?, fixed: a2(&f[6], k, true)?, mult: a2(&f[7], k, true)?, depth: uint(&f[8])?, starts: a1(&f[9], k, false)?,
        ops: a1(&f[10], k, false)?, args: a1(&f[11], k, false)?, consts: a2(&f[12], k, true)?, set_starts: a1(&f[13], k, false)?, set_comps: a1(&f[14], k, false)?,
        people: a1(&f[15], k, false)?, has_det: f[16].is_truthy()?, det: b1(&f[17], k, false)?, places: a1(&f[18], k, false)?, n_places: uint(&f[19])?,
        exp_set: a1(&f[20], k, false)?, rel_trans: a2(&f[21], k, true)?, levels: a2(&f[22], k, false)?, sizes: a1(&f[23], k, false)?, block: a1(&f[24], k, false)?,
        contacts: a3(&f[25], k, false)?, use_: b2(&f[26], k, false)?, mat: a3(&f[27], k, false)?, sh_pair: a2(&f[28], k, false)?, sh_val: a1(&f[29], k, true)?,
        tgt_starts: a1(&f[30], k, false)?, tgt_comp: a1(&f[31], k, false)?, tgt_w: a2(&f[32], k, false)?, tgt_map: a1(&f[33], k, false)?, tgt_rec: a1(&f[34], k, false)?,
        crd_starts: a1(&f[35], k, false)?, crd_index: a1(&f[36], k, false)?, crd_w: a2(&f[37], k, false)?, crd_parent: a1(&f[38], k, false)?, maps: a3(&f[39], k, false)?,
        n_movers: uint(&f[40])?, mov_starts: a2(&f[41], k, false)?, mov_comps: a1(&f[42], k, false)?, mov_pair: a2(&f[43], k, false)?, mov_rate: a1(&f[44], k, true)?,
        mov_base: a1(&f[45], k, false)?, mov_par: a1(&f[46], k, false)?, src_starts: a1(&f[47], k, false)?, src_flows: a1(&f[48], k, false)?, src_chain: a1(&f[49], k, false)?,
        src_stage: a1(&f[50], k, false)?, src_last: a1(&f[51], k, false)?, local: f[52].is_truthy()?, pl_starts: a1(&f[53], k, false)?, pl_groups: a1(&f[54], k, false)?,
    })
}

/// A movement register (`kernels.Register`)
fn register_of<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>) -> PyResult<Register> {
    let f = items(o, 14, "Register")?;
    Ok(Register {
        starts: a1(&f[0], k, false)?, src: a1(&f[1], k, false)?, dst: a1(&f[2], k, false)?, n: a1(&f[3], k, false)?, sel: a1(&f[4], k, false)?,
        sel_starts: a1(&f[5], k, false)?, sel_comps: a1(&f[6], k, false)?, sel_to: a1(&f[7], k, false)?, ent_starts: a1(&f[8], k, false)?,
        ent_comps: a1(&f[9], k, false)?, ent_w: a2(&f[10], k, false)?, pg_starts: a1(&f[11], k, false)?, pg_groups: a1(&f[12], k, false)?, step: f[13].extract()?,
    })
}

/// The deliveries (`kernels.Shifts`)
fn shifts<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>) -> PyResult<Shifts> {
    let f = items(o, 6, "Shifts")?;
    Ok(Shifts { due: b2(&f[0], k, false)?, sel: a3(&f[1], k, false)?, cov: a3(&f[2], k, false)?, to: a2(&f[3], k, false)?, rest: a2(&f[4], k, false)?, land: a3(&f[5], k, false)? })
}

/// A tape (`kernels.Tape`)
fn tape<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>) -> PyResult<Tape> {
    let f = items(o, 12, "Tape")?;
    Ok(Tape {
        ops: a1(&f[0], k, false)?, a: a1(&f[1], k, false)?, b: a1(&f[2], k, false)?, const_grouped: b1(&f[3], k, false)?, set_starts: a1(&f[4], k, false)?,
        state_starts: a1(&f[5], k, false)?, state_comps: a1(&f[6], k, false)?, places: a1(&f[7], k, false)?, n_places: uint(&f[8])?, local: f[9].is_truthy()?,
        people: a1(&f[10], k, false)?, has_frac: f[11].is_truthy()?,
    })
}

/// A Generator's bit generator, from its capsule (as Numba reaches it)
fn gen_of(o: &Bound<PyAny>) -> PyResult<Rng> {
    let cap = o.getattr("bit_generator")?.getattr("capsule")?;
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(cap.as_ptr(), c"BitGenerator".as_ptr()) };
    if ptr.is_null() {
        return Err(PyErr::fetch(o.py()));
    }
    Ok(Rng::new(ptr as *const rng::BitGen))
}

/// The generators of a list (a typed list, as the engines pass them)
fn gens_of<'py>(o: &Bound<'py, PyAny>, k: &mut Keep<'py>) -> PyResult<Vec<Rng>> {
    let mut out = Vec::new();
    for i in 0..o.len()? {
        let g = o.get_item(i)?;
        out.push(gen_of(&g)?);
        k.push(g);
    }
    Ok(out)
}

/// The tables of a step (fixed, mult, consts, rel_trans), each by rows
fn tables_of<'py>(t: [&Bound<'py, PyAny>; 4], k: &mut Keep<'py>) -> PyResult<Tables> {
    Ok([a3(t[0], k, false)?, a3(t[1], k, false)?, a3(t[2], k, false)?, a3(t[3], k, false)?])
}

/// Find BLAS's dgemv where Numba finds it, the first time it is needed (scipy's, for np.dot: see `mix()`)
fn blas(py: Python) -> PyResult<()> {
    if DGEMV.get().is_some() {
        return Ok(());
    }
    let cap = py.import("scipy.linalg.cython_blas")?.getattr("__pyx_capi__")?.get_item("dgemv")?;
    let ptr = unsafe {
        let name = pyo3::ffi::PyCapsule_GetName(cap.as_ptr());
        pyo3::ffi::PyCapsule_GetPointer(cap.as_ptr(), name)
    };
    if ptr.is_null() {
        return Err(PyErr::fetch(py));
    }
    let _ = DGEMV.set(ptr as usize);
    Ok(())
}

/// A new array of zeros, made by NumPy, so one too large raises MemoryError
fn zeros<'py, const N: usize>(py: Python<'py>, dims: [usize; N]) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    Ok(py.import("numpy")?.call_method1("zeros", (dims.to_vec(),))?.cast_into::<PyArrayDyn<f64>>()?)
}

/// A new array with the values of a 2-D view
fn copy2<'py>(py: Python<'py>, y: A2<f64>) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, A2<f64>)> {
    let out = zeros(py, [y.n0(), y.n1()])?;
    let v = A2::new(out.data(), y.n0(), y.n1(), y.n1());
    v.copy_from(y);
    Ok((out, v))
}

fn view2(a: &Bound<PyArrayDyn<f64>>) -> A2<f64> { let s = a.shape(); A2::new(a.data(), s[0], s[1], s[1]) }
fn view3(a: &Bound<PyArrayDyn<f64>>) -> A3<f64> { let s = a.shape(); A3::new(a.data(), [s[0], s[1], s[2]], s[1]*s[2], s[2]) }

#[pyfunction]
#[pyo3(name = "run")]
#[allow(clippy::too_many_arguments)]
fn py_run<'py>(py: Python<'py>, rng: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, dt: f64, npts: usize, solver: i64, n_sub: i64, fixed: &Bound<'py, PyAny>,
            mult: &Bound<'py, PyAny>, consts: &Bound<'py, PyAny>, rel_trans: &Bound<'py, PyAny>, reg: &Bound<'py, PyAny>, sh: &Bound<'py, PyAny>, tally: &Bound<'py, PyAny>,
            start: usize) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, f64)> {
    blas(py)?;
    let mut k = Keep::new();
    let (g, prog, t, r, s) = (gen_of(rng)?, program(p, &mut k)?, tables_of([fixed, mult, consts, rel_trans], &mut k)?, register_of(reg, &mut k)?, shifts(sh, &mut k)?);
    let tl = nd::<4>(tally, &mut k, true)?;
    let (yo, yv) = copy2(py, a2(y, &mut k, false)?)?;
    let (nc, ng) = (yv.n0(), yv.n1());
    let n = npts.checked_sub(start).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("negative dimensions not allowed (start is after npts)"))?;
    let stocks = zeros(py, [n, nc, ng])?;
    let flows = zeros(py, [n, prog.n_trans + prog.n_movers, ng])?;
    let short = super::run(g, yv, &prog, dt, npts, solver, n_sub, &t, &r, &s, tl, start, view3(&stocks), view3(&flows));
    Ok((yo.into_any(), stocks.into_any(), flows.into_any(), short))
}

#[pyfunction]
#[pyo3(name = "run_many")]
#[pyo3(signature = (rngs, ys, p, dt, start, npts, solver, n_sub, fixed, mult, consts, rel_trans, reg, sh, stocks, flows, stocks_g, flows_g, tally, tally_g, state_of, rows, grp_state, grp_flow))]
#[allow(clippy::too_many_arguments)]
fn py_run_many<'py>(py: Python<'py>, rngs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, dt: f64, start: usize, npts: usize, solver: i64, n_sub: i64,
                 fixed: &Bound<'py, PyAny>, mult: &Bound<'py, PyAny>, consts: &Bound<'py, PyAny>, rel_trans: &Bound<'py, PyAny>, reg: &Bound<'py, PyAny>, sh: &Bound<'py, PyAny>,
                 stocks: &Bound<'py, PyAny>, flows: &Bound<'py, PyAny>, stocks_g: &Bound<'py, PyAny>, flows_g: &Bound<'py, PyAny>, tally: &Bound<'py, PyAny>, tally_g: &Bound<'py, PyAny>,
                 state_of: &Bound<'py, PyAny>, rows: &Bound<'py, PyAny>, grp_state: &Bound<'py, PyAny>, grp_flow: &Bound<'py, PyAny>) -> PyResult<()> {
    blas(py)?;
    let mut k = Keep::new();
    let gs = gens_of(rngs, &mut k)?;
    let (prog, t, r, s) = (program(p, &mut k)?, tables_of([fixed, mult, consts, rel_trans], &mut k)?, register_of(reg, &mut k)?, shifts(sh, &mut k)?);
    let o = ManyOut { stocks: nd(stocks, &mut k, true)?, flows: nd(flows, &mut k, true)?, stocks_g: nd(stocks_g, &mut k, true)?, flows_g: nd(flows_g, &mut k, true)?,
                      tally: nd(tally, &mut k, true)?, tally_g: nd(tally_g, &mut k, true)? };
    super::run_many(&gs, a3(ys, &mut k, true)?, &prog, dt, start, npts, solver, n_sub, &t, &r, &s, &o, a1(state_of, &mut k, false)?, a1(rows, &mut k, false)?,
                    a1(grp_state, &mut k, false)?, a1(grp_flow, &mut k, false)?);
    Ok(())
}

/// What advance() and advance_threads() share
#[allow(clippy::too_many_arguments)]
fn advance_args<'a, 'py>(dt: f64, start: usize, end: usize, solver: i64, n_sub: i64, t: [&Bound<'py, PyAny>; 4], sh_vals: &Bound<'py, PyAny>, reg: &'a Register, sh: &'a Shifts,
                         state_of: &Bound<'py, PyAny>, stocks: &Bound<'py, PyAny>, flows: &Bound<'py, PyAny>, row0: i64, k: &mut Keep<'py>) -> PyResult<Advance<'a>> {
    Ok(Advance { dt, start, end, solver, n_sub, tables: [nd(t[0], k, false)?, nd(t[1], k, false)?, nd(t[2], k, false)?, nd(t[3], k, false)?], sh_vals: a2(sh_vals, k, false)?,
                 reg, sh, state_of: a1(state_of, k, false)?, stocks: nd(stocks, k, true)?, flows: nd(flows, k, true)?, row0 })
}

#[pyfunction]
#[pyo3(name = "advance")]
#[pyo3(signature = (rngs, p, ys, first, last, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, row0=0))]
#[allow(clippy::too_many_arguments)]
fn py_advance<'py>(py: Python<'py>, rngs: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, first: usize, last: usize, dt: f64, start: usize, end: usize, solver: i64,
                n_sub: i64, fixed: &Bound<'py, PyAny>, mult: &Bound<'py, PyAny>, consts: &Bound<'py, PyAny>, rel_trans: &Bound<'py, PyAny>, sh_vals: &Bound<'py, PyAny>,
                reg: &Bound<'py, PyAny>, sh: &Bound<'py, PyAny>, state_of: &Bound<'py, PyAny>, stocks: &Bound<'py, PyAny>, flows: &Bound<'py, PyAny>,
                row0: i64) -> PyResult<()> {
    blas(py)?;
    let mut k = Keep::new();
    let gs = gens_of(rngs, &mut k)?;
    let (prog, r, s) = (program(p, &mut k)?, register_of(reg, &mut k)?, shifts(sh, &mut k)?);
    let a = advance_args(dt, start, end, solver, n_sub, [fixed, mult, consts, rel_trans], sh_vals, &r, &s, state_of, stocks, flows, row0, &mut k)?;
    super::advance(&gs, &prog, a3(ys, &mut k, true)?, first, last, &a);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "advance_threads")]
#[pyo3(signature = (rngs, progs, ys, dt, start, end, solver, n_sub, fixed, mult, consts, rel_trans, sh_vals, reg, sh, state_of, stocks, flows, row0=0))]
#[allow(clippy::too_many_arguments)]
fn py_advance_threads<'py>(py: Python<'py>, rngs: &Bound<'py, PyAny>, progs: &Bound<'py, PyAny>, ys: &Bound<'py, PyAny>, dt: f64, start: usize, end: usize, solver: i64, n_sub: i64,
                        fixed: &Bound<'py, PyAny>, mult: &Bound<'py, PyAny>, consts: &Bound<'py, PyAny>, rel_trans: &Bound<'py, PyAny>, sh_vals: &Bound<'py, PyAny>,
                        reg: &Bound<'py, PyAny>, sh: &Bound<'py, PyAny>, state_of: &Bound<'py, PyAny>, stocks: &Bound<'py, PyAny>, flows: &Bound<'py, PyAny>,
                        row0: i64) -> PyResult<()> {
    blas(py)?;
    let mut k = Keep::new();
    let gs = gens_of(rngs, &mut k)?;
    let mut ps = Vec::new();
    for i in 0..progs.len()? {
        let p = progs.get_item(i)?;
        ps.push(program(&p, &mut k)?);
        k.push(p);
    }
    let (r, s) = (register_of(reg, &mut k)?, shifts(sh, &mut k)?);
    let a = advance_args(dt, start, end, solver, n_sub, [fixed, mult, consts, rel_trans], sh_vals, &r, &s, state_of, stocks, flows, row0, &mut k)?;
    let y = a3(ys, &mut k, true)?;
    py.detach(|| super::advance_threads(&gs, &ps, y, &a));
    Ok(())
}

#[pyfunction]
#[pyo3(name = "one_step")]
#[allow(clippy::too_many_arguments)]
fn py_one_step<'py>(py: Python<'py>, rng: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, dt: f64, solver: i64, n_sub: i64, tables: &Bound<'py, PyAny>,
                 row: i64) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    blas(py)?;
    let mut k = Keep::new();
    let (g, prog) = (gen_of(rng)?, program(p, &mut k)?);
    let ti = items(tables, 4, "tuple of tables")?;
    let t = tables_of([&ti[0], &ti[1], &ti[2], &ti[3]], &mut k)?;
    let (yo, yv) = copy2(py, a2(y, &mut k, false)?)?;
    let flows = zeros(py, [prog.n_trans + prog.n_movers, yv.n1()])?;
    let w = workspace(yv.n0(), yv.n1(), &prog);
    if row >= 0 {
        super::set_step(super::parts(&prog), row, &t);
        super::set_movers(&prog);
    }
    step(g, yv, &prog, &w, dt, n_sub, solver, view2(&flows), row >= 0);
    Ok((yo.into_any(), flows.into_any()))
}

#[pyfunction]
#[pyo3(name = "apply_events")]
fn py_apply_events(rng: &Bound<PyAny>, y: &Bound<PyAny>, r: &Bound<PyAny>, ti: usize, whole: bool) -> PyResult<f64> {
    let mut k = Keep::new();
    let (g, reg) = (gen_of(rng)?, register_of(r, &mut k)?);
    Ok(events::apply_events(g, a2(y, &mut k, true)?, &reg, ti, whole))
}

#[pyfunction]
#[pyo3(name = "apply_shifts")]
fn py_apply_shifts(rng: &Bound<PyAny>, y: &Bound<PyAny>, sh: &Bound<PyAny>, ti: usize, whole: bool, tally: &Bound<PyAny>) -> PyResult<()> {
    let mut k = Keep::new();
    let (g, s) = (gen_of(rng)?, shifts(sh, &mut k)?);
    events::apply_shifts(g, a2(y, &mut k, true)?, &s, ti, whole, a3(tally, &mut k, true)?);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "settle_places")]
fn py_settle_places(y: &Bound<PyAny>, p: &Bound<PyAny>, dt: f64, flows: &Bound<PyAny>) -> PyResult<()> {
    let mut k = Keep::new();
    let prog = program(p, &mut k)?;
    super::settle_places(a2(y, &mut k, true)?, &prog, dt, a2(flows, &mut k, true)?);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "evaluate")]
fn py_evaluate(y: &Bound<PyAny>, t: &Bound<PyAny>, consts: &Bound<PyAny>, out: &Bound<PyAny>, grouped: &Bound<PyAny>) -> PyResult<()> {
    let mut k = Keep::new();
    let tp = tape(t, &mut k)?;
    super::evaluate(a2(y, &mut k, false)?, &tp, a2(consts, &mut k, false)?, a2(out, &mut k, true)?, b1(grouped, &mut k, true)?);
    Ok(())
}

/// Add these kernels to the module, as a submodule (`_rs.kernels`), since their names are those of kernels.py
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new(m.py(), "kernels")?;
    sub.add_function(wrap_pyfunction!(py_run, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_run_many, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_advance, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_advance_threads, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_one_step, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_apply_events, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_apply_shifts, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_settle_places, &sub)?)?;
    sub.add_function(wrap_pyfunction!(py_evaluate, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}
