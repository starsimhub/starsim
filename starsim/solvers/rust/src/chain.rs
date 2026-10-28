//! The draws of a step of the chain binomial and tau-leaping in JAX, as `starsim/solvers/backends/chain.py`: `step()` and the kernels it calls,
//! and the XLA FFI handler that JAX calls (`handle()`), which reads XLA's call frame (the structs of xla/ffi/api/c_api.h) by byte offset

use crate::draws;
use std::sync::atomic::{AtomicI32, Ordering};

const N_HEAD: usize = 6; // Before the offsets of the program's parts (see chain.py)
const N_FIELDS: usize = 21;
const CONVEY: i64 = 22; // ss.kernels.CONVEY
const TAU: i64 = 1;
const RELEASE: i64 = 2;
const RELEASE_END: i64 = 4;
const SHIFT: i64 = 8;
const REGISTER: i64 = 16;
const DELIVER: i64 = 32;


//%% Numba's arithmetic, where Rust's differs

/// The smallest value of an array, or its first NaN (as Numba's arr.min())
fn amin(x: &[f64]) -> f64 {
    let mut m = x[0];
    for &v in x {
        if v.is_nan() {
            return v;
        }
        if v < m {
            m = v;
        }
    }
    m
}

/// The largest value of an array, or its first NaN (as Numba's arr.max())
fn amax(x: &[f64]) -> f64 {
    let mut m = x[0];
    for &v in x {
        if v.is_nan() {
            return v;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// a**b for an integer b, by repeated squaring, as Numba computes it
fn int_power(mut a: f64, b: i64) -> f64 {
    let (invert, mut e) = if b < 0 { (true, -b) } else { (false, b) };
    if e > 0x10000 {
        return a.powf(b as f64);
    }
    let mut r = 1.0;
    while e != 0 {
        if e & 1 != 0 {
            r *= a;
        }
        e >>= 1;
        a *= a;
    }
    if invert { 1.0/r } else { r }
}

/// a // b for floats, as Python (and Numba) computes it
fn floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m)/b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        (0.0f64).copysign(a/b)
    }
}


//%% The draws of a step

/// Part i of the program, from the packed array
#[inline(always)]
fn field(s: &[i64], i: usize) -> &[i64] {
    &s[s[N_HEAD+i] as usize..s[N_HEAD+i+1] as usize]
}

/// Row i of a C-contiguous 2D array with rows of n values
#[inline(always)]
fn line(a: &[f64], i: usize, n: usize) -> &[f64] {
    &a[i*n..(i+1)*n]
}

#[inline(always)]
fn line_mut(a: &mut [f64], i: usize, n: usize) -> &mut [f64] {
    &mut a[i*n..(i+1)*n]
}

#[inline(always)]
fn add(out: &mut [f64], x: &[f64]) {
    for (o, v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

/// What every draw of a call reads: its key, the groups, the program, the shares of targets and credits, and the maps between groups
struct Ctx<'a> {
    k: u64,
    ng: usize,
    s: &'a [i64],
    tgt_w: &'a [f64],
    crd_w: &'a [f64],
    maps: &'a [f64],
    vec: bool,
}

/// The scratch of a row of draws: the vectorized draws' (9 rows) and its uniforms
struct Rows<'a> {
    w: &'a mut [f64],
    u: &'a mut [f64],
}

/// A row of binomial draws (one per group), from uniforms row*groups, row*groups+1, ... of the step's stream (as `binomials()`)
fn binomials(c: &Ctx, row: i64, n: &[f64], p: &[f64], out: &mut [f64], r: &mut Rows) -> i64 {
    let ng = n.len() as i64;
    if c.vec {
        draws::uniforms(c.k, (row*ng) as u64, r.u);
        draws::binomials(n, p, r.u, out, r.w, n.len());
    } else {
        for g in 0..n.len() {
            out[g] = draws::binomial(n[g], p[g], draws::uniform(c.k, (row*ng + g as i64) as u64));
        }
    }
    row + 1
}

/// Add arrivals (n, by group) to the groups they land in: shared between groups by map m, if m >= 0, by a chain of binomials for each group (as `land()`)
fn land(c: &Ctx, row: i64, n: &[f64], m: i64, out: &mut [f64]) -> i64 {
    let ng = c.ng;
    if m < 0 {
        add(out, n);
        return row;
    }
    let map = &c.maps[m as usize*ng*ng..(m as usize + 1)*ng*ng];
    for g in 0..ng {
        let mut left = n[g];
        let mut rest = 1.0;
        for h in 0..ng {
            let w = map[g*ng + h];
            if left <= 0.0 {
                break;
            } else if w >= rest - 1e-12 { // The last place they can go
                out[h] += left;
                left = 0.0;
            } else if w > 0.0 {
                let x = draws::binomial(left, w/rest, draws::uniform(c.k, ((row + g as i64)*ng as i64 + h as i64) as u64));
                out[h] += x;
                left -= x;
                rest -= w;
            }
        }
    }
    row + ng as i64
}

/// Share n (by group) between the targets of flow f (but its last, with skip_last): one multinomial draw, as a chain of conditional binomials (as `split()`)
#[allow(clippy::too_many_arguments)]
fn split(c: &Ctx, mut row: i64, n: &[f64], f: usize, new: &mut [f64], flows: &mut [f64], scratch: &mut [f64], r: &mut Rows, skip_last: bool) -> i64 {
    let s = c.s;
    let ng = c.ng;
    let (tgt_starts, tgt_comp, tgt_rec, tgt_map) = (field(s, 6), field(s, 7), field(s, 8), field(s, 12));
    let (a, b) = (tgt_starts[f], tgt_starts[f+1] - skip_last as i64);
    if b - a == 1 && amin(line(c.tgt_w, a as usize, ng)) >= 1.0 - 1e-12 {
        let a = a as usize;
        row = land(c, row, n, tgt_map[a], line_mut(new, tgt_comp[a] as usize, ng));
        if tgt_rec[a] >= 0 {
            add(line_mut(flows, tgt_rec[a] as usize, ng), n);
        }
        return row;
    }
    let (total, sc) = scratch.split_at_mut(ng);
    let (left, sc) = sc.split_at_mut(ng);
    let (drawn, p) = sc.split_at_mut(ng);
    let p = &mut p[..ng];
    let mut whole = true;
    for g in 0..ng {
        total[g] = 0.0;
        left[g] = n[g];
        for t in a..b {
            total[g] += c.tgt_w[t as usize*ng + g];
        }
        whole = whole && (total[g] - 1.0).abs() < 1e-9;
        if 0.0 < total[g] && total[g] < 1.0 - 1e-9 { // Shares that sum below 1 leave the rest nowhere
            total[g] = 1.0;
        }
    }
    for t in a..b {
        let t = t as usize;
        for g in 0..ng {
            p[g] = if total[g] > 0.0 { c.tgt_w[t*ng + g]/total[g] } else { 0.0 };
        }
        row = binomials(c, row, left, p, drawn, r);
        for g in 0..ng {
            left[g] -= drawn[g];
            total[g] -= c.tgt_w[t*ng + g];
        }
        row = land(c, row, drawn, tgt_map[t], line_mut(new, tgt_comp[t] as usize, ng));
        if tgt_rec[t] >= 0 {
            add(line_mut(flows, tgt_rec[t] as usize, ng), drawn);
        }
    }
    if whole && b > a && amax(left) > 0.0 { // Rounding only
        let l = (b - 1) as usize;
        row = land(c, row, left, tgt_map[l], line_mut(new, tgt_comp[l] as usize, ng));
        if tgt_rec[l] >= 0 {
            add(line_mut(flows, tgt_rec[l] as usize, ng), left);
        }
    }
    row
}

/// Count the events of flow f (n, by group) also in the transitions they pass through with no wait: the credits nested in the same one are one multinomial draw, as a chain of binomials (as `credit()`)
fn credit(c: &Ctx, mut row: i64, n: &[f64], f: usize, flows: &mut [f64], r: &mut Rows) -> i64 {
    let s = c.s;
    let (crd_starts, crd_parent, crd_index) = (field(s, 14), field(s, 15), field(s, 16));
    let (a, b) = (crd_starts[f], crd_starts[f+1]);
    if b == a {
        return row;
    }
    let ng = c.ng;
    let rows = (b - a + 1) as usize;
    let mut left = vec![0.0; rows*ng];
    let mut mass = vec![0.0; rows*ng];
    let mut cc = vec![0.0; ng];
    let mut p = vec![0.0; ng];
    left[..ng].copy_from_slice(n);
    mass[..ng].fill(1.0);
    for j in a..b {
        let q = crd_parent[j as usize];
        let i = (j - a + 1) as usize;
        let ju = j as usize;
        let w = line(c.crd_w, ju, ng);
        if q == -2 { // Overlaps the others: drawn on its own
            row = binomials(c, row, n, w, &mut cc, r);
        } else {
            let rr = if q < 0 { 0 } else { (q - a + 1) as usize };
            for g in 0..ng {
                let m = mass[rr*ng + g];
                p[g] = if m > 0.0 { w[g]/m } else { 0.0 };
            }
            row = binomials(c, row, line(&left, rr, ng), &p, &mut cc, r);
            for g in 0..ng {
                left[rr*ng + g] -= cc[g];
                mass[rr*ng + g] -= w[g];
            }
        }
        line_mut(&mut left, i, ng).copy_from_slice(&cc);
        line_mut(&mut mass, i, ng).copy_from_slice(w);
        add(line_mut(flows, crd_index[ju] as usize, ng), &cc);
    }
    row
}

/// The flows of states that are not people (a reservoir), as expected values over the step, integrated by RK4 at the step's rates with the people as they are (as `settle()`)
fn settle(c: &Ctx, y: &mut [f64], hz: &[f64], dt: f64, flows: &mut [f64]) {
    let s = c.s;
    let (src_of, tgt_starts, tgt_comp, tgt_rec, named, index) = (field(s, 2), field(s, 6), field(s, 7), field(s, 8), field(s, 9), field(s, 10));
    let (tgt_map, record, crd_starts, crd_index, det) = (field(s, 12), field(s, 13), field(s, 14), field(s, 16), field(s, 17));
    let (tgt_w, crd_w, maps) = (c.tgt_w, c.crd_w, c.maps);
    let ng = c.ng;
    let nc = y.len()/ng;
    let nflows = flows.len()/ng;
    let mut fastest = 0.0;
    for f in 0..src_of.len() {
        if det[f] != 0 && src_of[f] >= 0 {
            for g in 0..ng {
                fastest = draws::max(fastest, hz[f*ng + g]/dt);
            }
        }
    }
    let n_sub = 1.max(draws::to_i64((fastest*dt/0.25).ceil()));
    let h = dt/n_sub as f64;
    let mut k = vec![0.0; 4*nc*ng];
    let mut kf = vec![0.0; 4*nflows*ng];
    let mut yt = vec![0.0; nc*ng];
    for _ in 0..n_sub {
        k.fill(0.0);
        kf.fill(0.0);
        let y0 = y.to_vec();
        yt.copy_from_slice(y);
        for stage in 0..4 {
            let (ks, kfs) = (&mut k[stage*nc*ng..(stage+1)*nc*ng], &mut kf[stage*nflows*ng..(stage+1)*nflows*ng]);
            for f in 0..src_of.len() {
                if det[f] == 0 {
                    continue;
                }
                let src = src_of[f];
                for g in 0..ng {
                    let amount = hz[f*ng + g]/dt*(if src >= 0 { yt[src as usize*ng + g] } else { 1.0 }); // H is the rate times dt, or for an inflow its arrivals
                    if amount == 0.0 {
                        continue;
                    }
                    if src >= 0 {
                        ks[src as usize*ng + g] -= amount;
                    }
                    if named[f] != 0 {
                        let i = index[f] as usize;
                        if record[f] < 0 {
                            kfs[i*ng + g] += amount;
                        } else {
                            for j in 0..ng {
                                kfs[i*ng + j] += amount*maps[(record[f] as usize*ng + g)*ng + j];
                            }
                        }
                    }
                    for t in tgt_starts[f] as usize..tgt_starts[f+1] as usize {
                        if tgt_rec[t] >= 0 {
                            kfs[tgt_rec[t] as usize*ng + g] += amount*tgt_w[t*ng + g];
                        }
                        let tc = tgt_comp[t] as usize;
                        if tgt_map[t] < 0 {
                            ks[tc*ng + g] += amount*tgt_w[t*ng + g];
                        } else {
                            for j in 0..ng {
                                ks[tc*ng + j] += amount*tgt_w[t*ng + g]*maps[(tgt_map[t] as usize*ng + g)*ng + j];
                            }
                        }
                    }
                    for t in crd_starts[f] as usize..crd_starts[f+1] as usize {
                        kfs[crd_index[t] as usize*ng + g] += amount*crd_w[t*ng + g];
                    }
                }
            }
            if stage < 3 {
                let ahead = if stage == 2 { h } else { h/2.0 };
                for i in 0..nc*ng {
                    yt[i] = draws::max(y0[i] + ahead*ks[i], 0.0);
                }
            }
        }
        let (m, q) = (nc*ng, nflows*ng);
        for i in 0..m {
            y[i] = draws::max(y0[i] + h/6.0*(k[i] + 2.0*k[m+i] + 2.0*k[2*m+i] + k[3*m+i]), 0.0);
        }
        for i in 0..q {
            flows[i] += h/6.0*(kf[i] + 2.0*kf[q+i] + 2.0*kf[2*q+i] + kf[3*q+i]);
        }
    }
}

/// Release the people in the last slot of each conveyor, as `ss.kernels.convey()` does with release=True (as `release()`)
fn release(c: &Ctx, mut row: i64, new: &mut [f64], flows: &mut [f64], scratch: &mut [f64], r: &mut Rows) -> i64 {
    let s = c.s;
    let (src_of, named, index, record) = (field(s, 2), field(s, 9), field(s, 10), field(s, 13));
    let (starts, ops, args) = (field(s, 18), field(s, 19), field(s, 20));
    let ng = c.ng;
    let mut n = vec![0.0; ng];
    for i in 0..starts.len() - src_of.len() - 1 { // Conveyors, whose operations follow the flows'
        let (a, b) = (starts[src_of.len()+i] as usize, starts[src_of.len()+i+1] as usize);
        if ops[a] == CONVEY {
            let f = args[a] as usize;
            let last = line_mut(new, args[b-1] as usize, ng);
            n.copy_from_slice(last);
            last.fill(0.0);
            if named[f] != 0 {
                row = land(c, row, &n, record[f], line_mut(flows, index[f] as usize, ng));
            }
            row = split(c, row, &n, f, new, flows, scratch, r, true); // Not the last target, which is the slot they leave
            row = credit(c, row, &n, f, flows, r);
        }
    }
    row
}

/// The deliveries due at the start of a step, in order, as whole people: each moves some of the people in each compartment and group to another group; tally gets the number each reached (expected) and found eligible, by group (as `deliver()`)
fn deliver(k: u64, y: &mut [f64], ng: usize, due: &[f64], sel: &[f64], cov: &[f64], dst: &[f64], tally: &mut [f64]) {
    let nc = y.len()/ng;
    let mut before = vec![0.0; ng];
    let mut i: u64 = 0;
    for d in 0..due.len() {
        if due[d] <= 0.0 {
            continue;
        }
        for c in 0..nc {
            before.copy_from_slice(line(y, c, ng)); // Each group's people as they were, so nobody moved into a group is moved on again
            for g in 0..ng {
                let e = sel[(d*nc + c)*ng + g];
                let n = before[g];
                if e <= 0.0 || n <= 0.0 {
                    continue;
                }
                let p = cov[(d*nc + c)*ng + g];
                let to = draws::to_i64(dst[(d*2)*ng + g]) as usize;
                let rest = draws::to_i64(dst[(d*2 + 1)*ng + g]) as usize;
                tally[(d*2)*ng + rest] += n*e*p; // In the group of those offered it (see ss.Delivery.mark_offered())
                tally[(d*2 + 1)*ng + g] += n*e;
                let a = draws::binomial(n, e*p, draws::uniform(k, i));
                let b = if e*p < 1.0 { draws::binomial(n - a, e*(1.0 - p)/(1.0 - e*p), draws::uniform(k, i + 1)) } else { 0.0 };
                i += 2;
                if a > 0.0 && to != g {
                    y[c*ng + g] -= a;
                    y[c*ng + to] += a;
                }
                if b > 0.0 && rest != g {
                    y[c*ng + g] -= b;
                    y[c*ng + rest] += b;
                }
            }
        }
    }
}

/// The number of good items in n drawn without replacement, one at a time from uniforms i, i+1, ...; returns it and the next uniform (as `hypergeometric()`)
fn hypergeometric(k: u64, i: i64, good: f64, bad: f64, n: f64) -> (f64, i64) {
    let (mut good, mut bad, mut n) = (good.floor(), bad.floor(), n.floor());
    let flip = n > floordiv(good + bad, 2.0); // Draw the ones left behind instead, if that is fewer
    if flip {
        n = good + bad - n;
    }
    let mut got = 0.0;
    let m = draws::to_i64(n);
    for j in 0..m {
        if draws::uniform(k, (i + j) as u64)*(good + bad) < good {
            good -= 1.0;
            got += 1.0;
        } else {
            bad -= 1.0;
        }
    }
    (if flip { good } else { got }, i + m)
}

/// Part i of the register, from the packed array
#[inline(always)]
fn rfield(r: &[i64], i: usize) -> &[i64] {
    &r[r[1+i] as usize..r[2+i] as usize]
}

/// The events of a movement register dated at time point ti, in order, as whole people; returns the number of people they asked for and could not find (as `apply_events()`)
fn apply_events(k: u64, mut i: i64, y: &mut [f64], ng: usize, rg: &[i64], reg_n: &[f64], ent_w: &[f64], ti: i64) -> f64 {
    let (starts, src_of, dst_of, sel_of, sel_starts, sel_comps, sel_to) = (rfield(rg, 0), rfield(rg, 1), rfield(rg, 2), rfield(rg, 3), rfield(rg, 4), rfield(rg, 5), rfield(rg, 6));
    let (ent_starts, ent_comps, pg_starts, pg_groups) = (rfield(rg, 7), rfield(rg, 8), rfield(rg, 9), rfield(rg, 10));
    let step = rg[0];
    let mut short = 0.0;
    if ti >= starts.len() as i64 - 1 {
        return short;
    }
    let ti = ti as usize;
    for e in starts[ti] as usize..starts[ti+1] as usize {
        let (src, dst, s) = (src_of[e], dst_of[e], sel_of[e] as usize);
        let mut n = reg_n[e];
        if src >= 0 { // Move people, or remove them
            let su = src as usize;
            let mut total = 0.0;
            for j in sel_starts[s] as usize..sel_starts[s+1] as usize {
                for q in pg_starts[su] as usize..pg_starts[su+1] as usize {
                    total += y[sel_comps[j] as usize*ng + pg_groups[q] as usize];
                }
            }
            if n > total {
                short += n - total;
                n = total;
            }
            let mut left = n;
            let mut rest = total;
            for j in sel_starts[s] as usize..sel_starts[s+1] as usize {
                let c = sel_comps[j] as usize;
                for q in pg_starts[su] as usize..pg_starts[su+1] as usize {
                    let g = pg_groups[q];
                    let have = y[c*ng + g as usize];
                    if left <= 0.0 || rest <= 0.0 {
                        break;
                    }
                    if have <= 0.0 {
                        continue;
                    }
                    let take;
                    (take, i) = hypergeometric(k, i, have, rest - have, left);
                    rest -= have;
                    left -= take;
                    y[c*ng + g as usize] -= take;
                    if dst >= 0 {
                        y[sel_to[j] as usize*ng + (g + (dst - src)*step) as usize] += take;
                    }
                }
            }
        } else { // Add people
            let du = dst as usize;
            let mut left = n;
            for j in ent_starts[s]..ent_starts[s+1] {
                let mut part = draws::binomial(left, 1.0/(ent_starts[s+1] - j) as f64, draws::uniform(k, i as u64)); // Shared between the states not yet given theirs
                i += 1;
                left -= part;
                let ju = j as usize;
                let mut wrest = 0.0;
                for q in pg_starts[du] as usize..pg_starts[du+1] as usize {
                    wrest += ent_w[ju*ng + pg_groups[q] as usize];
                }
                for q in pg_starts[du] as usize..pg_starts[du+1] as usize {
                    let g = pg_groups[q] as usize;
                    let wt = ent_w[ju*ng + g];
                    if part <= 0.0 || wrest <= 0.0 {
                        break;
                    }
                    let take = draws::binomial(part, wt/wrest, draws::uniform(k, i as u64));
                    i += 1;
                    y[ent_comps[ju] as usize*ng + g] += take;
                    part -= take;
                    wrest -= wt;
                }
            }
        }
    }
    short
}

/// The arrays of one call of `step()`, each flat (C order; see `step()` in chain.py for what each is)
pub struct Call<'a> {
    pub k: u64,
    pub y: &'a [f64],
    pub hz: &'a [f64],
    pub s: &'a [i64],
    pub tgt_w: &'a [f64],
    pub crd_w: &'a [f64],
    pub maps: &'a [f64],
    pub mov_prob: &'a [f64],
    pub dt: f64,
    pub flags: i64,
    pub sh_due: &'a [f64],
    pub sh_sel: &'a [f64],
    pub sh_cov: &'a [f64],
    pub sh_dst: &'a [f64],
    pub reg: &'a [i64],
    pub reg_n: &'a [f64],
    pub ent_w: &'a [f64],
    pub ti: i64,
}

/// All of a step's draws (or a substep's, or a leap's): the compartments at its end (new) and the count of each transition and mover over it (flows), as `ss.kernels.draw()` gives them (as `step()`)
pub fn step(a: &Call, new: &mut [f64], flows: &mut [f64], short: &mut [f64], tally: &mut [f64]) {
    let s = a.s;
    new.copy_from_slice(a.y);
    flows.fill(0.0);
    short[0] = 0.0;
    tally.fill(0.0);
    let (k, hz, flags) = (a.k, a.hz, a.flags);
    let ng = s[1] as usize;
    if flags & DELIVER != 0 {
        deliver(k, new, ng, a.sh_due, a.sh_sel, a.sh_cov, a.sh_dst, tally);
        return;
    }
    let tau = flags & TAU > 0;
    let (nflows, n_movers) = (s[2] as usize, s[3] as usize);
    let c = Ctx { k, ng, s, tgt_w: a.tgt_w, crd_w: a.crd_w, maps: a.maps, vec: s[4] > 0 };
    let (src_starts, src_flows, src_of, src_chain, src_stage, src_last) = (field(s, 0), field(s, 1), field(s, 2), field(s, 3), field(s, 4), field(s, 5));
    let (named, index, mov_comps, record, det) = (field(s, 9), field(s, 10), field(s, 11), field(s, 13), field(s, 17));
    let (tgt_starts, tgt_comp, tgt_rec, tgt_map, crd_starts) = (field(s, 6), field(s, 7), field(s, 8), field(s, 12), field(s, 14));
    let base = s[N_HEAD+N_FIELDS] as usize; // Then the movers' starts and pairs, as two rows
    let width = (s.len() - base)/2;
    let mut buf = vec![0.0; (23 + s[5] as usize)*ng]; // Scratch, in one piece: every row by group
    let (w, rest_buf) = buf.split_at_mut(9*ng);
    let (scratch, rest_buf) = rest_buf.split_at_mut(4*ng);
    let (one, pp) = rest_buf.split_at_mut(8*ng);
    let mut it = one.chunks_exact_mut(ng);
    let [total, left, n, p, u, ends, rest, mass] = std::array::from_fn(|_| it.next().unwrap());
    let mut r = Rows { w, u };
    let mut y = a.y.to_vec(); // Who leaves is drawn from y, which is the people as they are at the start, or after a release, or (in a crossing) after the last disease's flows
    let mut row: i64 = 0;
    let (starts, ops, args) = (field(s, 18), field(s, 19), field(s, 20));
    let n_conv = starts.len() - src_of.len() - 1; // Conveyors, whose operations follow the flows' (see ss.kernels.convey())
    if n_conv > 0 && flags & RELEASE != 0 { // Release the people in the last slot of each, at the start of the step, once the rates are worked out from the people as they were
        row = release(&c, row, new, flows, scratch, &mut r);
        y.copy_from_slice(new);
    }
    for si in 0..src_starts.len() - 1 {
        let (a0, b0) = (src_starts[si] as usize, src_starts[si+1] as usize);
        if a0 == b0 { // In the program of a crossing, the end of one disease's flows: the next disease's are drawn from where these left people (each from the rates at the start of the step), as ss.kernels.draw() does
            y.copy_from_slice(new);
            continue;
        }
        let src = src_of[src_flows[a0] as usize];
        let d = if tau { 1 } else { src_chain[si] }; // Tau-leaping takes the stages of a dwell time one at a time, as any other compartment
        if src >= 0 && det[src_flows[a0] as usize] != 0 { // A state that is not people: see settle()
            continue;
        }
        if d > 1 { // The stages of a dwell time: as many as people pass in the step (see ss.kernels.draw_chain())
            let src = src as usize;
            let du = d as usize;
            let stage = src_stage[si] as usize;
            let stride = tgt_comp[tgt_starts[stage] as usize] - src as i64;
            for g in 0..ng {
                let x = hz[stage*ng + g];
                let mut hx = 0.0;
                for j in a0..b0 {
                    if src_flows[j] as usize != stage {
                        hx += hz[src_flows[j] as usize*ng + g];
                    }
                }
                total[g] = hx;
                let (stay, mut term, mut term_t, mut cdf_t, mut tot) = ((-hx).exp(), (-x).exp(), (-x - hx).exp(), 0.0, 0.0);
                for m in 0..du {
                    pp[m*ng + g] = stay*term;
                    tot += pp[m*ng + g];
                    cdf_t += term_t;
                    term *= x/(m + 1) as f64;
                    term_t *= (x + hx)/(m + 1) as f64;
                }
                pp[du*ng + g] = if x + hx > 0.0 { int_power(x/(x + hx), d)*draws::max(1.0 - cdf_t, 0.0) } else { 0.0 };
                pp[(du+1)*ng + g] = draws::max(1.0 - tot - pp[du*ng + g], 0.0);
            }
            row = binomials(&c, row, line(&y, src, ng), line(pp, du + 1, ng), left, &mut r); // Leave by another exit
            for g in 0..ng {
                rest[g] = y[src*ng + g] - left[g];
                mass[g] = 1.0 - pp[(du+1)*ng + g];
                p[g] = if mass[g] > 0.0 { pp[du*ng + g]/mass[g] } else { 1.0 };
            }
            row = binomials(&c, row, rest, p, ends, &mut r); // Leave by the last stage
            for g in 0..ng {
                rest[g] -= ends[g];
                mass[g] -= pp[du*ng + g];
                new[src*ng + g] -= left[g] + ends[g];
            }
            for m in 1..du { // Move on some stages; the rest stay
                for g in 0..ng {
                    p[g] = if mass[g] > 0.0 { pp[m*ng + g]/mass[g] } else { 0.0 };
                }
                row = binomials(&c, row, rest, p, n, &mut r);
                for g in 0..ng {
                    rest[g] -= n[g];
                    mass[g] -= pp[m*ng + g];
                    new[src*ng + g] -= n[g];
                    new[(src as i64 + m as i64*stride) as usize*ng + g] += n[g];
                }
            }
            let last = src_last[si] as usize;
            if named[last] != 0 {
                row = land(&c, row, ends, record[last], line_mut(flows, index[last] as usize, ng));
            }
            row = split(&c, row, ends, last, new, flows, scratch, &mut r, false);
            row = credit(&c, row, ends, last, flows, &mut r);
        } else if src >= 0 { // Who leaves
            let src = src as usize;
            for g in 0..ng {
                total[g] = 0.0;
                for j in a0..b0 {
                    total[g] += hz[src_flows[j] as usize*ng + g];
                }
                p[g] = if tau { total[g]*y[src*ng + g] } else { -(-total[g]).exp_m1() };
            }
            if tau { // Poisson, and no more than are there (clipped once, so the order of the flows does not matter)
                if c.vec {
                    draws::uniforms(k, (row*ng as i64) as u64, r.u);
                    draws::poissons(p, r.u, left, r.w, ng);
                } else {
                    for g in 0..ng {
                        left[g] = draws::poisson(p[g], draws::uniform(k, (row*ng as i64 + g as i64) as u64));
                    }
                }
                row += 1;
                for g in 0..ng {
                    left[g] = draws::min(left[g], y[src*ng + g]);
                }
            } else {
                row = binomials(&c, row, line(&y, src, ng), p, left, &mut r);
            }
            for g in 0..ng {
                new[src*ng + g] -= left[g];
            }
        }
        for j in a0..b0 { // How they split between the flows (or the arrivals)
            let f = src_flows[j] as usize;
            if (d > 1 && f as i64 == src_stage[si]) || det[f] != 0 { // Taken above, or settled below
                continue;
            }
            let hf = line(hz, f, ng);
            if src < 0 {
                if c.vec {
                    draws::uniforms(k, (row*ng as i64) as u64, r.u);
                    draws::poissons(hf, r.u, n, r.w, ng);
                } else {
                    for g in 0..ng {
                        n[g] = draws::poisson(hf[g], draws::uniform(k, (row*ng as i64 + g as i64) as u64));
                    }
                }
                row += 1;
            } else {
                for g in 0..ng {
                    p[g] = if total[g] > 0.0 { hf[g]/total[g] } else { 0.0 };
                }
                row = binomials(&c, row, left, p, n, &mut r);
                for g in 0..ng {
                    total[g] -= hf[g];
                    left[g] -= n[g];
                }
            }
            if named[f] != 0 && record[f] < 0 { // Counted in the group it happens in, and (below) everyone to one compartment in the same group: as land() and split() do, but written out, since the calls cost more than the draws of a small model
                add(line_mut(flows, index[f] as usize, ng), n);
            } else if named[f] != 0 {
                row = land(&c, row, n, record[f], line_mut(flows, index[f] as usize, ng));
            }
            let t = tgt_starts[f] as usize;
            if tgt_starts[f+1] as usize == t + 1 && tgt_map[t] < 0 && tgt_rec[t] < 0 && amin(line(c.tgt_w, t, ng)) >= 1.0 - 1e-12 {
                add(line_mut(new, tgt_comp[t] as usize, ng), n);
            } else {
                row = split(&c, row, n, f, new, flows, scratch, &mut r, false);
            }
            if crd_starts[f+1] > crd_starts[f] {
                row = credit(&c, row, n, f, flows, &mut r);
            }
        }
    }
    let mut i = row*ng as i64; // Movement between groups, from where people are after the transitions
    for m in 0..n_movers {
        for e in s[base+width+m] as usize..s[base+width+m+1] as usize {
            if a.mov_prob[e] > 0.0 {
                let (g, h) = (s[base+n_movers+1+e] as usize, s[base+width+n_movers+1+e] as usize);
                for j in s[base+m] as usize..s[base+m+1] as usize {
                    let cj = mov_comps[j] as usize;
                    let moved = draws::binomial(new[cj*ng + g], a.mov_prob[e], draws::uniform(k, i as u64));
                    i += 1;
                    new[cj*ng + g] -= moved;
                    new[cj*ng + h] += moved;
                    flows[(nflows - n_movers + m)*ng + g] += moved;
                }
            }
        }
    }
    if det.iter().any(|&x| x != 0) {
        settle(&c, new, hz, a.dt, flows);
    }
    row = i/ng as i64 + 1;
    if n_conv > 0 && flags & RELEASE_END != 0 { // Tau-leaping releases them at the middle of the step, after the leap that reaches it
        row = release(&c, row, new, flows, scratch, &mut r);
    }
    if n_conv > 0 && flags & SHIFT != 0 { // Everyone in a conveyor moves along one slot
        for ci in 0..n_conv {
            let (a0, b0) = (starts[src_of.len()+ci] as usize, starts[src_of.len()+ci+1] as usize);
            if ops[a0] == CONVEY {
                for j in (a0+2..b0).rev() {
                    let (to, from) = (args[j] as usize, args[j-1] as usize);
                    new.copy_within(from*ng..(from+1)*ng, to*ng);
                }
                line_mut(new, args[a0+1] as usize, ng).fill(0.0);
            }
        }
    }
    if flags & REGISTER != 0 { // The events dated at the step's end
        short[0] = apply_events(k, row*ng as i64, new, ng, a.reg, a.reg_n, a.ent_w, a.ti);
    }
}


//%% The XLA FFI handler

static MAJOR: AtomicI32 = AtomicI32::new(0); // The XLA FFI API version the handler reports (see `api_version()` in chain.py)
static MINOR: AtomicI32 = AtomicI32::new(0);

const IN_RANKS: [i64; 16] = [1, 2, 2, 1, 2, 2, 3, 1, 0, 1, 3, 3, 3, 1, 1, 2]; // The rank of each argument of one run: ctl, y, H, S, tgt_w, crd_w, maps, mov_prob, dt, sh_due, sh_sel, sh_cov, sh_dst, R, reg_n, ent_w
const OUT_RANKS: [i64; 4] = [2, 2, 1, 3]; // ... and of each output: new, flows, short, tally

#[inline(always)]
unsafe fn load(addr: i64) -> i64 {
    unsafe { *(addr as *const i64) }
}

/// n values at an address, as a slice (empty if n is 0, when the address may be null)
#[inline(always)]
unsafe fn at<'a, T>(addr: i64, n: i64) -> &'a [T] {
    if n <= 0 { &[] } else { unsafe { std::slice::from_raw_parts(addr as *const T, n as usize) } }
}

#[inline(always)]
unsafe fn at_mut<'a, T>(addr: i64, n: i64) -> &'a mut [T] {
    if n <= 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(addr as *mut T, n as usize) } }
}

/// XLA_FFI_Buffer: struct_size, extension_start, dtype, data (+24), rank (+32), dims (+40). Returns its data and, for an argument
/// of rank rank0 in one run, how far apart the runs are in it (0 if it is the same for all of them) and the size of one; or with
/// runs=0, the number of runs (as `buffer()`)
unsafe fn buffer(addr: i64, rank0: i64, runs: i64) -> (i64, i64, i64) {
    unsafe {
        let rank = load(addr + 32);
        let dims: &[i64] = at(load(addr + 40), rank);
        let size: i64 = dims.iter().product();
        let batch = if rank > rank0 { dims[0] } else { 1 };
        if runs == 0 {
            return (load(addr + 24), batch, size);
        }
        let one = if batch > 0 { size/batch } else { 0 };
        (load(addr + 24), if batch > 1 { one } else { 0 }, one)
    }
}

/// XLA_FFI_Error* handler(XLA_FFI_CallFrame*): the arguments of step() in, new, flows, short, and tally out. Runs batched by vmap
/// have the runs first, in the outputs and in each argument that is batched (as `handle()`)
pub unsafe extern "C" fn handle(frame: *mut u8) -> *mut u8 {
    unsafe {
        let frame = frame as i64;
        // XLA_FFI_CallFrame: struct_size, extension_start (+8), api, ctx, stage (+32), args (+40), rets (+80)
        let ext = load(frame + 8);
        if ext != 0 && *((ext + 8) as *const i32) == 1 { // XLA_FFI_Extension_Metadata: report the API version
            let meta = load(ext + 24);
            let version = at_mut::<i64>(meta + 8, 2); // XLA_FFI_Api_Version: struct_size, extension_start, major and minor
            version[0] = 24;
            version[1] = 0;
            let ints = at_mut::<i32>(meta + 24, 2);
            ints[0] = MAJOR.load(Ordering::Relaxed);
            ints[1] = MINOR.load(Ordering::Relaxed);
            *((meta + 32) as *mut u32) = 0; // Traits
            return std::ptr::null_mut();
        }
        if *((frame + 32) as *const i32) != 3 { // Only XLA_FFI_ExecutionStage_EXECUTE
            return std::ptr::null_mut();
        }
        let args = load(frame + 40 + 32); // XLA_FFI_Args: struct_size, extension_start, size, types, args (+32)
        let rets = load(frame + 80 + 32);
        let n_in = IN_RANKS.len();
        let mut data = [0i64; 16];
        let mut gap = [0i64; 16]; // How far apart the runs are in each argument, in elements
        let mut one = [0i64; 16]; // The size of one run's
        let (n_data, runs, _) = buffer(load(rets), OUT_RANKS[0], 0);
        let (f_data, _, _) = buffer(load(rets + 8), OUT_RANKS[1], 0);
        let (s_data, _, _) = buffer(load(rets + 16), OUT_RANKS[2], 0);
        let (t_data, _, _) = buffer(load(rets + 24), OUT_RANKS[3], 0);
        for j in 0..n_in {
            (data[j], gap[j], one[j]) = buffer(load(args + 8*j as i64), IN_RANKS[j], runs);
        }
        for r in 0..runs {
            let p = |j: usize| data[j] + 8*r*gap[j]; // Where each argument of this run is
            let ctl: &[i64] = at(p(0), 4); // The seed, the call's count, its flags, and the time point
            let s: &[i64] = at(p(3), one[3]);
            let (nc, ng, nflows) = (s[0], s[1], s[2]);
            let kd = one[9];
            let call = Call {
                k: draws::key(ctl[0], ctl[1]),
                y: at(p(1), nc*ng),
                hz: at(p(2), one[2]),
                s,
                tgt_w: at(p(4), one[4]),
                crd_w: at(p(5), one[5]),
                maps: at(p(6), one[6]),
                mov_prob: at(p(7), one[7]),
                dt: *(p(8) as *const f64),
                flags: ctl[2],
                sh_due: at(p(9), kd),
                sh_sel: at(p(10), kd*nc*ng),
                sh_cov: at(p(11), kd*nc*ng),
                sh_dst: at(p(12), kd*2*ng),
                reg: at(p(13), one[13]),
                reg_n: at(p(14), one[14]),
                ent_w: at(p(15), one[15]),
                ti: ctl[3],
            };
            let new = at_mut(n_data + 8*r*nc*ng, nc*ng);
            let flows = at_mut(f_data + 8*r*nflows*ng, nflows*ng);
            let short = at_mut(s_data + 8*r, 1);
            let tally = at_mut(t_data + 8*r*kd*2*ng, kd*2*ng);
            step(&call, new, flows, short, tally);
        }
        std::ptr::null_mut()
    }
}


/// The binding: the handler's address, for `jax.ffi.pycapsule()`
pub mod bind {
    use pyo3::prelude::*;
    use std::sync::atomic::Ordering;

    /// The address of the handler, which reports XLA FFI API version major.minor (see `register()` in chain.py)
    #[pyfunction]
    fn chain_handler(major: i32, minor: i32) -> usize {
        super::MAJOR.store(major, Ordering::Relaxed);
        super::MINOR.store(minor, Ordering::Relaxed);
        super::handle as *const () as usize
    }

    /// Add the binding to the module
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(chain_handler, m)?)?;
        Ok(())
    }
}
