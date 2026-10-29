//! The exact solver one place at a time, when places do not affect each other within a step (`gillespie_places()` and what it calls)

use super::*;

/// The rate of every flow for the groups of one place (p.pl_groups[a:b]), when people mix only within their place (`place_rates()`)
pub fn place_rates(ys: A2<f64>, p: &Program, a: usize, b: usize, rates: A2<f64>, stack: A2<f64>, expo: A1<f64>) {
    let nf = p.src.len();
    let top_reg = stack.n0().wrapping_sub(1);
    let mut n = 0.0;
    for j in a..b {
        let g = p.pl_groups.u(j);
        for i in 0..p.people.len() {
            n += ys.at(p.people.u(i), g);
        }
    }
    for e in 0..p.exp_set.len() {
        let s = p.exp_set.at(e);
        if s < 0 {
            expo.set(e, sum_exposures(expo, e, s));
            continue;
        }
        let s = s as usize;
        let mut num = 0.0;
        for j in a..b {
            let g = p.pl_groups.u(j);
            for k in p.set_starts.u(s)..p.set_starts.u(s + 1) {
                num += ys.at(p.set_comps.u(k), g)*p.rel_trans.at(e, g);
            }
        }
        let prev = if n > 0.0 { num/n } else { 0.0 };
        expo.set(e, 0.0);
        for t in 0..p.sizes.len() {
            expo.add(e, p.contacts.at(0, t, 0)*prev);
        }
    }
    for f in 0..nf {
        for j in a..b {
            let g = p.pl_groups.u(j);
            rates.set(f, g, p.fixed.at(f, g));
            if p.starts.at(f) == p.starts.at(f + 1) {
                continue;
            }
            let mut top: i64 = -1;
            for k in p.starts.u(f)..p.starts.u(f + 1) {
                let op = p.ops.at(k);
                let arg = p.args.u(k);
                if op == CONST {
                    top += 1;
                    stack.set(top as usize, 0, p.consts.at(arg, g));
                } else if op == EXPOSURE {
                    top += 1;
                    stack.set(top as usize, 0, expo.at(arg));
                } else if op == TOTAL {
                    top += 1;
                    let mut v = 0.0;
                    for gg in 0..ys.n1() {
                        for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                            v += ys.at(p.set_comps.u(i), gg);
                        }
                    }
                    stack.set(top as usize, 0, v);
                } else if op == COUNT || op == FRAC {
                    top += 1;
                    let mut v = 0.0;
                    for jj in a..b {
                        for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                            v += ys.at(p.set_comps.u(i), p.pl_groups.u(jj));
                        }
                    }
                    stack.set(top as usize, 0, leaf_share(op, v, n));
                } else if op == GROUP {
                    top += 1;
                    let mut v = 0.0;
                    for i in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                        v += ys.at(p.set_comps.u(i), g);
                    }
                    stack.set(top as usize, 0, v);
                } else {
                    top = stack_op(stack, top, op, arg, top_reg);
                }
            }
            if top >= 0 {
                rates.set(f, g, stack.at(0, 0)*p.mult.at(f, g));
            }
        }
    }
}

/// The operations on the stack of one group's rate other than the leaves: LOAD, STORE, and functions; returns the new top
#[inline(always)]
fn stack_op(stack: A2<f64>, mut top: i64, op: i64, arg: usize, top_reg: usize) -> i64 {
    if op == LOAD {
        top += 1;
        stack.set(top as usize, 0, stack.at(top_reg - arg, 0));
    } else if op == STORE {
        stack.set(top_reg - arg, 0, stack.at(top as usize, 0));
        top -= 1;
    } else if op >= EXP {
        stack.set(top as usize, 0, unary(op, stack.at(top as usize, 0)));
    } else {
        top -= 1;
        stack.set(top as usize, 0, binary(op, stack.at(top as usize, 0), stack.at(top as usize + 1, 0)));
    }
    top
}

/// The sum of the -s exposures before exposure e, added in order (`sum_exposures()`)
#[inline(always)]
fn sum_exposures(expo: A1<f64>, e: usize, s: i64) -> f64 {
    let first = (e as i64 + s) as usize;
    let mut v = expo.at(first);
    for x in first + 1..e {
        v += expo.at(x);
    }
    v
}

/// A number uniform on (0, 1) from the counter of a random stream, with the bits taken to the middle of their step (`uniform()`)
#[inline(always)]
fn uniform(z: u64) -> f64 {
    ((dr::mix(z) >> 11) as f64 + 0.5)/9007199254740992.0
}

/// The seed of place i's random numbers in this step (as place_events() works it out)
#[inline(always)]
fn place_seed(seed: u64, i: usize) -> u64 {
    dr::mix(seed.wrapping_add(dr::mix((i as u64 + 1).wrapping_mul(GOLDEN))))
}

/// Where an event goes and what else it counts as, once its flow and group are known; for both kinds of place events. Returns the target compartment and the counter after the event's numbers
fn finish_event(y: A2<f64>, p: &Program, which: usize, place: usize, flows: A2<f64>, mut z: u64, mut arrive: impl FnMut(usize)) -> u64 {
    let (ka, kb) = (p.tgt_starts.u(which), p.tgt_starts.u(which + 1)); // One of its targets, in proportion to their shares
    z = z.wrapping_add(GOLDEN);
    let u = uniform(z);
    let mut tot = 0.0;
    for k in ka..kb {
        tot += p.tgt_w.at(k, place);
    }
    let whole = !(0.0 < tot && tot < 1.0 - 1e-9); // Shares that sum below 1 leave the rest nowhere, as in the ODE
    let tot = if whole { tot } else { 1.0 };
    let mut cum = 0.0;
    for k in ka..kb {
        cum += p.tgt_w.at(k, place);
        if u*tot < cum || (whole && k == kb - 1) {
            let c = p.tgt_comp.u(k);
            arrive(c);
            y.add(c, place, 1.0);
            if p.tgt_rec.at(k) >= 0 {
                flows.add(p.tgt_rec.u(k), place, 1.0);
            }
            break;
        }
    }
    z = z.wrapping_add(GOLDEN);
    let u = uniform(z); // The transitions it also counts as (immediate branches)
    let (ca, cb) = (p.crd_starts.u(which), p.crd_starts.u(which + 1));
    if cb > ca {
        let mut us = vec![0.0; cb - ca];
        us[0] = u;
        for k in 1..cb - ca {
            z = z.wrapping_add(GOLDEN);
            us[k] = uniform(z);
        }
        credit_one(&us, ca, cb, p, place, flows);
    }
    z
}

/// The events over a time dt of the places active[first..last], each drawn from its own random numbers (`place_events()`)
#[allow(clippy::too_many_arguments)]
pub fn place_events(y: A2<f64>, p: &Program, w: &Work, active: &[usize], dt: f64, flows: A2<f64>, seed: u64) {
    let (rates, idle) = (w.rates, w.cells.row(5));
    let nf = p.src.len();
    let (_o1, stack) = own2(p.depth, 1, 0.0);
    let (_o2, expo) = own1(p.exp_set.len().max(1), 0.0);
    for &i in active {
        let (a, b) = (p.pl_starts.u(i), p.pl_starts.u(i + 1));
        idle.set(i, 0.0);
        let mut t = 0.0;
        let mut z: u64 = 0;
        loop {
            place_rates(y, p, a, b, rates, stack, expo);
            let mut total = 0.0;
            for f in 0..nf {
                if p.det.b(f) {
                    continue;
                }
                let s = p.src.at(f);
                for j in a..b {
                    let g = p.pl_groups.u(j);
                    total += mx(rates.at(f, g)*(if s >= 0 { y.at(s as usize, g) } else { p.inflow.at(f, g) }), 0.0);
                }
            }
            if total <= 0.0 {
                idle.set(i, 1.0);
                break;
            }
            if t == 0.0 { // The first event in this step: the counter of the place's random numbers in this step
                z = place_seed(seed, i);
            }
            z = z.wrapping_add(GOLDEN);
            t -= uniform(z).ln()/total;
            if t > dt {
                break;
            }
            z = z.wrapping_add(GOLDEN);
            let target = uniform(z)*total;
            let mut cum = 0.0;
            let (mut which, mut place) = (0, 0);
            'flows: for f in 0..nf {
                if p.det.b(f) {
                    continue;
                }
                let s = p.src.at(f);
                for j in a..b {
                    let g = p.pl_groups.u(j);
                    let r = mx(rates.at(f, g)*(if s >= 0 { y.at(s as usize, g) } else { p.inflow.at(f, g) }), 0.0);
                    if r > 0.0 {
                        cum += r;
                        which = f;
                        place = g;
                        if cum > target {
                            break 'flows;
                        }
                    }
                }
            }
            if p.src.at(which) >= 0 {
                y.set(p.src.u(which), place, y.at(p.src.u(which), place) - 1.0);
            }
            if p.named.b(which) {
                flows.add(p.index.u(which), place, 1.0);
            }
            z = finish_event(y, p, which, place, flows, z, |_| ());
        }
    }
}

/// For `sparse_events()`: the sets each compartment is in, the exposures to each set, the flows out of each compartment, the inflows, which compartments hold people, and whether any rate is a count in every place (`sparse_maps()`)
pub struct SparseMaps {
    cs_starts: Vec<usize>,
    cs_sets: Vec<usize>,
    se_starts: Vec<usize>,
    se_exps: Vec<usize>,
    cf_starts: Vec<usize>,
    cf_flows: Vec<usize>,
    inflows: Vec<usize>,
    people: Vec<bool>,
    has_total: bool,
}

fn cumsum(v: &mut [usize]) {
    for i in 1..v.len() {
        v[i] += v[i - 1];
    }
}

pub fn sparse_maps(p: &Program, nc: usize) -> SparseMaps {
    let n_sets = p.set_starts.len() - 1;
    let mut cs_starts = vec![0; nc + 1];
    for k in 0..p.set_comps.len() {
        cs_starts[p.set_comps.u(k) + 1] += 1;
    }
    cumsum(&mut cs_starts);
    let mut cs_sets = vec![0; p.set_comps.len()];
    let mut fill = cs_starts[..nc].to_vec();
    for s in 0..n_sets { // In order, so each compartment's sets are too
        for k in p.set_starts.u(s)..p.set_starts.u(s + 1) {
            let c = p.set_comps.u(k);
            cs_sets[fill[c]] = s;
            fill[c] += 1;
        }
    }
    let mut se_starts = vec![0; n_sets + 1];
    for e in 0..p.exp_set.len() {
        if p.exp_set.at(e) >= 0 {
            se_starts[p.exp_set.u(e) + 1] += 1;
        }
    }
    cumsum(&mut se_starts);
    let mut se_exps = vec![0; se_starts[n_sets]];
    let mut fill = se_starts[..n_sets].to_vec();
    for e in 0..p.exp_set.len() {
        if p.exp_set.at(e) >= 0 {
            let s = p.exp_set.u(e);
            se_exps[fill[s]] = e;
            fill[s] += 1;
        }
    }
    let mut cf_starts = vec![0; nc + 1]; // The flows that are events, by the compartment they leave
    for f in 0..p.src.len() {
        if !p.det.b(f) && p.src.at(f) >= 0 {
            cf_starts[p.src.u(f) + 1] += 1;
        }
    }
    cumsum(&mut cf_starts);
    let mut cf_flows = vec![0; cf_starts[nc]];
    let mut inflows = Vec::new();
    let mut fill = cf_starts[..nc].to_vec();
    for f in 0..p.src.len() {
        if p.det.b(f) {
            continue;
        }
        if p.src.at(f) >= 0 {
            let s = p.src.u(f);
            cf_flows[fill[s]] = f;
            fill[s] += 1;
        } else {
            inflows.push(f);
        }
    }
    let mut people = vec![false; nc];
    for i in 0..p.people.len() {
        people[p.people.u(i)] = true;
    }
    let has_total = (0..p.ops.len()).any(|k| p.ops.at(k) == TOTAL);
    SparseMaps { cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total }
}

/// Put a key into the first n of a sorted array, in order; returns the new number of keys (`add_key()`)
#[inline(always)]
fn add_key(keys: &mut [usize], n: usize, key: usize) -> usize {
    let mut m = n;
    while m > 0 && keys[m - 1] > key {
        keys[m] = keys[m - 1];
        m -= 1;
    }
    keys[m] = key;
    n + 1
}

/// Take a key out of the first n of a sorted array; returns the new number of keys (`drop_key()`)
#[inline(always)]
fn drop_key(keys: &mut [usize], n: usize, key: usize) -> usize {
    let m = keys[..n].partition_point(|&k| k < key);
    for k in m..n.saturating_sub(1) {
        keys[k] = keys[k + 1];
    }
    n - 1
}

/// The events of the places to step, as `place_events()`, for a large program most of whose compartments are empty: each place keeps the compartments that hold anyone, and the flows out of them, in order, and a rate until the counts and exposures it is worked out from change (`sparse_events()`)
#[allow(clippy::too_many_arguments)]
pub fn sparse_events(y: A2<f64>, p: &Program, w: &Work, active: &[usize], dt: f64, flows: A2<f64>, seed: u64, maps: &SparseMaps) {
    let idle = w.cells.row(5);
    let nc = y.n0();
    let nf = p.src.len();
    let top_reg = p.depth.wrapping_sub(1);
    let SparseMaps { cs_starts, cs_sets, se_starts, se_exps, cf_starts, cf_flows, inflows, people, has_total } = maps;
    let mut width = 1;
    for &i in active {
        width = width.max(p.pl_starts.u(i + 1) - p.pl_starts.u(i));
    }
    let (mut gbits, mut cbits) = (0, 0); // The bits of a key that hold the group in the place, and the compartment
    while (1usize << gbits) < width {
        gbits += 1;
    }
    while (1usize << cbits) < nc {
        cbits += 1;
    }
    let (gmask, cmask) = ((1usize << gbits) - 1, (1usize << cbits) - 1);
    let mut held = vec![0usize; nc*width];   // The compartments that hold anyone in the place, as keys of (group, compartment), in order
    let mut keys = vec![0usize; nf*width];   // The flows out of them and the inflows, as keys of (flow, group), in order
    let mut props = vec![0.0; nf*width];     // The rate of each
    let mut kept = vec![0.0; nf << gbits];   // By key, the rate of a flow worked out from the counts and exposures,
    let mut fresh = vec![false; nf << gbits]; // and whether it is at their current values
    let n_counts = p.set_starts.len() - 1;
    let n_exp = p.exp_set.len();
    let mut counts = vec![0.0; n_counts];
    let mut num = vec![0.0; n_exp];
    let mut before = vec![0.0; n_counts + n_exp + 1]; // The counts, exposures, and people at the last event
    let (_o1, stack) = own2(p.depth, 1, 0.0);
    let (_o2, expo) = own1(n_exp.max(1), 0.0);
    for &i in active {
        let (a, b) = (p.pl_starts.u(i), p.pl_starts.u(i + 1));
        let ng = b - a;
        idle.set(i, 0.0);
        let mut n_held = 0;
        for j in 0..ng {
            let g = p.pl_groups.u(a + j);
            for c in 0..nc {
                if y.at(c, g) != 0.0 {
                    held[n_held] = (j << cbits) | c;
                    n_held += 1;
                }
            }
        }
        let mut n_keys = 0;
        for &f in inflows {
            for j in 0..ng {
                if p.inflow.at(f, p.pl_groups.u(a + j)) != 0.0 {
                    keys[n_keys] = (f << gbits) | j;
                    n_keys += 1;
                }
            }
        }
        for h in 0..n_held {
            let (j, c) = (held[h] >> cbits, held[h] & cmask);
            for k in cf_starts[c]..cf_starts[c + 1] {
                keys[n_keys] = (cf_flows[k] << gbits) | j;
                n_keys += 1;
            }
        }
        keys[..n_keys].sort_unstable();
        let mut t = 0.0;
        let mut z: u64 = 0;
        let mut begun = false;
        loop {
            let mut n = 0.0; // The counts and exposures, as place_rates() works them out, from the compartments that hold anyone
            counts.fill(0.0);
            num.fill(0.0);
            for h in 0..n_held {
                let (j, c) = (held[h] >> cbits, held[h] & cmask);
                let g = p.pl_groups.u(a + j);
                let v = y.at(c, g);
                if people[c] {
                    n += v;
                }
                for k in cs_starts[c]..cs_starts[c + 1] {
                    let s = cs_sets[k];
                    counts[s] += v;
                    for q in se_starts[s]..se_starts[s + 1] {
                        let e = se_exps[q];
                        num[e] += v*p.rel_trans.at(e, g);
                    }
                }
            }
            for e in 0..n_exp {
                if p.exp_set.at(e) < 0 {
                    expo.set(e, sum_exposures(expo, e, p.exp_set.at(e)));
                    continue;
                }
                let prev = if n > 0.0 { num[e]/n } else { 0.0 };
                expo.set(e, 0.0);
                for tt in 0..p.sizes.len() {
                    expo.add(e, p.contacts.at(0, tt, 0)*prev);
                }
            }
            let mut same = begun && !has_total && n == before[n_counts + n_exp]; // Whether the rates kept still hold
            for k in 0..n_counts {
                same = same && counts[k] == before[k];
            }
            for e in 0..n_exp {
                same = same && expo.at(e) == before[n_counts + e];
            }
            before[..n_counts].copy_from_slice(&counts);
            for e in 0..n_exp {
                before[n_counts + e] = expo.at(e);
            }
            before[n_counts + n_exp] = n;
            begun = true;
            let mut total = 0.0;
            for q in 0..n_keys { // The rate of each flow, as place_rates() works it out
                let key = keys[q];
                let f = key >> gbits;
                let g = p.pl_groups.u(a + (key & gmask));
                let rate;
                if p.starts.at(f) == p.starts.at(f + 1) {
                    rate = p.fixed.at(f, g);
                } else if same && fresh[key] {
                    rate = kept[key];
                } else {
                    let mut r = p.fixed.at(f, g);
                    let mut top: i64 = -1;
                    for k in p.starts.u(f)..p.starts.u(f + 1) {
                        let op = p.ops.at(k);
                        let arg = p.args.u(k);
                        if op == CONST {
                            top += 1;
                            stack.set(top as usize, 0, p.consts.at(arg, g));
                        } else if op == EXPOSURE {
                            top += 1;
                            stack.set(top as usize, 0, expo.at(arg));
                        } else if op == TOTAL {
                            top += 1;
                            let mut v = 0.0;
                            for gg in 0..y.n1() {
                                for ii in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                                    v += y.at(p.set_comps.u(ii), gg);
                                }
                            }
                            stack.set(top as usize, 0, v);
                        } else if op == COUNT || op == FRAC {
                            top += 1;
                            stack.set(top as usize, 0, leaf_share(op, counts[arg], n));
                        } else if op == GROUP {
                            top += 1;
                            let mut v = 0.0;
                            for ii in p.set_starts.u(arg)..p.set_starts.u(arg + 1) {
                                v += y.at(p.set_comps.u(ii), g);
                            }
                            stack.set(top as usize, 0, v);
                        } else {
                            top = stack_op(stack, top, op, arg, top_reg);
                        }
                    }
                    if top >= 0 {
                        r = stack.at(0, 0)*p.mult.at(f, g);
                    }
                    kept[key] = r;
                    fresh[key] = true;
                    rate = r;
                }
                let s = p.src.at(f);
                props[q] = mx(rate*(if s >= 0 { y.at(s as usize, g) } else { p.inflow.at(f, g) }), 0.0);
                total += props[q];
            }
            if total <= 0.0 {
                idle.set(i, 1.0);
                break;
            }
            if t == 0.0 {
                z = place_seed(seed, i);
            }
            z = z.wrapping_add(GOLDEN);
            t -= uniform(z).ln()/total;
            if t > dt {
                break;
            }
            z = z.wrapping_add(GOLDEN);
            let target = uniform(z)*total;
            let mut cum = 0.0;
            let mut pick = 0;
            for q in 0..n_keys {
                if props[q] > 0.0 {
                    cum += props[q];
                    pick = q;
                    if cum > target {
                        break;
                    }
                }
            }
            let which = keys[pick] >> gbits;
            let j = keys[pick] & gmask;
            let place = p.pl_groups.u(a + j);
            let s = p.src.at(which);
            if s >= 0 {
                let s = s as usize;
                y.set(s, place, y.at(s, place) - 1.0);
                if y.at(s, place) == 0.0 { // Its compartment is now empty
                    n_held = drop_key(&mut held, n_held, (j << cbits) | s);
                    for k in cf_starts[s]..cf_starts[s + 1] {
                        n_keys = drop_key(&mut keys, n_keys, (cf_flows[k] << gbits) | j);
                    }
                }
            }
            if p.named.b(which) {
                flows.add(p.index.u(which), place, 1.0);
            }
            z = finish_event(y, p, which, place, flows, z, |c| {
                if y.at(c, place) == 0.0 { // Its compartment now holds someone
                    n_held = add_key(&mut held, n_held, (j << cbits) | c);
                    for kk in cf_starts[c]..cf_starts[c + 1] {
                        n_keys = add_key(&mut keys, n_keys, (cf_flows[kk] << gbits) | j);
                        fresh[(cf_flows[kk] << gbits) | j] = false;
                    }
                }
            });
        }
    }
}

/// The exact solver over a time dt, one place at a time, skipping places where no event can happen until something changes them (`gillespie_places()`); each place draws from its own random numbers, seeded by one number from the generator
pub fn gillespie_places(rng: Rng, y: A2<f64>, p: &Program, w: &Work, dt: f64, flows: A2<f64>, mut fresh: bool) {
    let idle = w.cells.row(5);
    if (0..p.ops.len()).any(|k| p.ops.at(k) == TOTAL) { // A count in every place, which any place can change: every place is stepped
        fresh = true;
    }
    let active: Vec<usize> = (0..p.pl_starts.len() - 1).filter(|&i| fresh || idle.at(i) == 0.0).collect();
    let seed = rng.integers62() as u64;
    if p.src.len() > SPARSE_FLOWS {
        sparse_events(y, p, w, &active, dt, flows, seed, &sparse_maps(p, y.n0()));
    } else {
        place_events(y, p, w, &active, dt, flows, seed);
    }
}

/// For the exact solver by place: the places that the register's events dated in step ti change, which are stepped again (`touch()`)
pub fn touch(r: &Register, ti: usize, places: A1<i64>, w: &Work) {
    if ti + 1 >= r.starts.len() {
        return;
    }
    let idle = w.cells.row(5);
    for e in r.starts.u(ti)..r.starts.u(ti + 1) {
        for place in [r.src.at(e), r.dst.at(e)] {
            if place >= 0 {
                let place = place as usize;
                for j in r.pg_starts.u(place)..r.pg_starts.u(place + 1) {
                    idle.set(places.u(r.pg_groups.u(j)), 0.0);
                }
            }
        }
    }
}

/// For the exact solver by place: every place is stepped again (`wake()`)
pub fn wake(w: &Work) {
    w.cells.row(5).fill(0.0);
}
