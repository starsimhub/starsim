//! The events of a movement register, and the deliveries (the section "Register" of `starsim/solvers/numba/compartmental.py`,
//! the functions of the same names)

use super::*;

/// The events of a movement register dated in step ti, in order (`apply_events()`); returns the number of people they asked for and could not find
pub fn apply_events(rng: Rng, y: A2<f64>, r: &Register, ti: usize, whole: bool) -> f64 {
    let mut short = 0.0;
    if ti + 1 >= r.starts.len() {
        return short;
    }
    for e in r.starts.u(ti)..r.starts.u(ti + 1) {
        let (src, dst, s) = (r.src.at(e), r.dst.at(e), r.sel.u(e));
        let mut n = r.n.at(e);
        if src >= 0 { // Move people, or remove them
            let srcu = src as usize;
            let mut total = 0.0;
            for k in r.sel_starts.u(s)..r.sel_starts.u(s + 1) {
                for j in r.pg_starts.u(srcu)..r.pg_starts.u(srcu + 1) {
                    total += y.at(r.sel_comps.u(k), r.pg_groups.u(j));
                }
            }
            if n > total {
                short += n - total;
                n = total;
            }
            let mut left = n;
            let mut rest = total;
            for k in r.sel_starts.u(s)..r.sel_starts.u(s + 1) {
                let c = r.sel_comps.u(k);
                for j in r.pg_starts.u(srcu)..r.pg_starts.u(srcu + 1) {
                    let g = r.pg_groups.u(j);
                    let have = y.at(c, g);
                    if left <= 0.0 || rest <= 0.0 {
                        break;
                    }
                    if have <= 0.0 {
                        continue;
                    }
                    let take = if whole { hypergeometric(rng, have, rest - have, left) } else { left*have/rest };
                    rest -= have;
                    left -= take;
                    y.set(c, g, y.at(c, g) - take);
                    if dst >= 0 {
                        y.add(r.sel_to.u(k), (g as i64 + (dst - src)*r.step) as usize, take);
                    }
                }
            }
        } else { // Add people
            let dstu = dst as usize;
            let mut left = n;
            for k in r.ent_starts.u(s)..r.ent_starts.u(s + 1) {
                let remaining = (r.ent_starts.u(s + 1) - k) as f64; // The states not yet given their share
                let mut part = if whole { binomial(rng, left, 1.0/remaining) } else { left/remaining };
                left -= part;
                let mut wrest = 0.0;
                for j in r.pg_starts.u(dstu)..r.pg_starts.u(dstu + 1) {
                    wrest += r.ent_w.at(k, r.pg_groups.u(j));
                }
                for j in r.pg_starts.u(dstu)..r.pg_starts.u(dstu + 1) {
                    let g = r.pg_groups.u(j);
                    let w = r.ent_w.at(k, g);
                    if part <= 0.0 || wrest <= 0.0 {
                        break;
                    }
                    let take = if whole { binomial(rng, part, w/wrest) } else { part*w/wrest };
                    y.add(r.ent_comps.u(k), g, take);
                    part -= take;
                    wrest -= w;
                }
            }
        }
    }
    short
}

/// The deliveries that happen at time point ti, at the start of the step from it, in order (`apply_shifts()`); tally is added to
pub fn apply_shifts(rng: Rng, y: A2<f64>, sh: &Shifts, ti: usize, whole: bool, tally: A3<f64>) {
    let (nc, ng) = (y.n0(), y.n1());
    let (_o1, arrived) = own1(ng, 0.0);
    let (_o2, before) = own1(ng, 0.0);
    for k in 0..sh.to.n0() {
        if !sh.due.b(k, ti) {
            continue;
        }
        let land = sh.land.plane(k);
        let mut sum = 0.0;
        for c in 0..land.n0() {
            for g in 0..land.n1() {
                sum += land.at(c, g);
            }
        }
        let moves = sum > 0.0;
        arrived.fill(0.0);
        for c in 0..nc {
            before.copy_from(y.row(c)); // Each group's people as they were, so nobody moved into a group is moved on again
            for g in 0..ng {
                let e = sh.sel.at(k, c, g);
                let n = before.at(g);
                if e <= 0.0 || n <= 0.0 {
                    continue;
                }
                let p = sh.cov.at(k, c, g);
                tally.add(k, 0, sh.rest.u(k, g), n*e*p);
                tally.add(k, 1, g, n*e);
                let (a, b);
                if whole {
                    a = binomial(rng, n, e*p);
                    b = if e*p < 1.0 { binomial(rng, n - a, e*(1.0 - p)/(1.0 - e*p)) } else { 0.0 };
                } else {
                    a = n*e*p;
                    b = n*e*(1.0 - p);
                }
                let (to, rest) = (sh.to.u(k, g), sh.rest.u(k, g));
                if moves && a > 0.0 {
                    y.set(c, g, y.at(c, g) - a);
                    arrived.add(to, a);
                } else if a > 0.0 && to != g {
                    y.set(c, g, y.at(c, g) - a);
                    y.add(c, to, a);
                }
                if b > 0.0 && rest != g {
                    y.set(c, g, y.at(c, g) - b);
                    y.add(c, rest, b);
                }
            }
        }
        if moves {
            enter_state(rng, y, land, arrived, whole);
        }
    }
}

/// Put the people who have moved into a state (by group) into its compartments, by the share in each (`enter_state()`)
pub fn enter_state(rng: Rng, y: A2<f64>, w: A2<f64>, arrived: A1<f64>, whole: bool) {
    let (nc, ng) = (y.n0(), y.n1());
    for g in 0..ng {
        let (mut left, mut rest) = (arrived.at(g), 1.0);
        if left <= 0.0 {
            continue;
        }
        let mut last = usize::MAX;
        for c in 0..nc {
            if w.at(c, g) > 0.0 {
                last = c;
            }
        }
        for c in 0..nc {
            if w.at(c, g) <= 0.0 {
                continue;
            }
            if !whole {
                y.add(c, g, arrived.at(g)*w.at(c, g));
            } else if c == last {
                y.add(c, g, left);
            } else {
                let n = binomial(rng, left, mn(mx(w.at(c, g)/rest, 0.0), 1.0));
                y.add(c, g, n);
                left -= n;
                rest -= w.at(c, g);
            }
        }
    }
}
