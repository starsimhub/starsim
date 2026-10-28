//! The draws of the stochastic solvers (the chain binomial, tau-leaping, and the exact solver), what every solver does with
//! people who must move at once (infinite rates, conveyors), the counts of states that are not people, and the events of a
//! movement register and the deliveries (`kernels.py`, the functions of the same names)

use super::*;

/// The number of n people that something with a probability happens to, from one uniform (`binomial()`)
#[inline(always)]
pub fn binomial(rng: Rng, n: f64, prob: f64) -> f64 {
    if n <= 0.0 || prob <= 0.0 {
        0.0
    } else if prob >= 1.0 {
        n
    } else {
        dr::binomial(n, prob, rng.random())
    }
}

/// A Poisson count with mean lam, from one uniform (`poisson()`)
#[inline(always)]
pub fn poisson(rng: Rng, lam: f64) -> f64 {
    if lam > 0.0 { dr::poisson(lam, rng.random()) } else { 0.0 }
}

/// A binomial for every group, into out: one at a time for fewer than ROW_GROUPS groups, else as a vectorized row from counter-based uniforms (`binomials()`)
#[inline(always)]
pub fn binomials(rng: Rng, n: A1<f64>, prob: A1<f64>, out: A1<f64>, rows: A2<f64>) {
    let ng = n.len();
    if ng < ROW_GROUPS {
        for g in 0..ng {
            out.set(g, binomial(rng, n.at(g), prob.at(g)));
        }
    } else {
        let u = dr::uniforms(rng.integers62(), rows.row(0));
        dr::binomials(n, prob, u, out, rows.rows(2, rows.n0()));
    }
}

/// A Poisson draw for every group, into out, as `binomials()` (`poissons()`)
#[inline(always)]
pub fn poissons(rng: Rng, lam: A1<f64>, out: A1<f64>, rows: A2<f64>) {
    let ng = lam.len();
    if ng < ROW_GROUPS {
        for g in 0..ng {
            out.set(g, poisson(rng, lam.at(g)));
        }
    } else {
        let u = dr::uniforms(rng.integers62(), rows.row(0));
        dr::poissons(lam, u, out, rows.rows(2, rows.n0()));
    }
}

/// As `binomials()`, but as a row only if ROW_GROUPS groups or more have anyone to draw for (`some_binomials()`)
#[inline(always)]
pub fn some_binomials(rng: Rng, n: A1<f64>, prob: A1<f64>, out: A1<f64>, rows: A2<f64>) {
    let mut k = 0;
    for g in 0..n.len() {
        k += (n.at(g) > 0.0) as usize;
    }
    if k < ROW_GROUPS {
        for g in 0..n.len() {
            out.set(g, binomial(rng, n.at(g), prob.at(g)));
        }
    } else {
        let u = dr::uniforms(rng.integers62(), rows.row(0));
        dr::binomials(n, prob, u, out, rows.rows(2, rows.n0()));
    }
}

/// The number of good items in n drawn without replacement from good and bad ones, drawn one at a time (`hypergeometric()`)
pub fn hypergeometric(rng: Rng, good: f64, bad: f64, n: f64) -> f64 {
    let (mut good, mut bad, mut n) = (to_int(good), to_int(bad), to_int(n));
    let flip = n > (good + bad).div_euclid(2);
    if flip {
        n = good + bad - n;
    }
    let mut k = 0;
    for _ in 0..n.max(0) {
        if rng.random()*((good + bad) as f64) < good as f64 {
            good -= 1;
            k += 1;
        } else {
            bad -= 1;
        }
    }
    (if flip { good } else { k }) as f64
}

/// Add some arrivals to the groups they land in: a draw, where they are shared between groups (`land()`)
pub fn land(rng: Rng, n: A1<f64>, m: i64, maps: A3<f64>, out: A1<f64>) {
    let ng = n.len();
    if m < 0 {
        for g in 0..ng {
            out.add(g, n.at(g));
        }
        return;
    }
    let m = m as usize;
    for g in 0..ng {
        let mut left = n.at(g);
        let mut rest = 1.0;
        for h in 0..ng {
            let w = maps.at(m, g, h);
            if left <= 0.0 {
                break;
            } else if w >= rest - 1e-12 { // The last place they can go
                out.add(h, left);
                left = 0.0;
            } else if w > 0.0 {
                let k = binomial(rng, left, w/rest);
                out.add(h, k);
                left -= k;
                rest -= w;
            }
        }
    }
}

/// Share n whole events between the targets of a flow (or the transitions it is also counted as), at random (`split()`)
#[allow(clippy::too_many_arguments)]
pub fn split(rng: Rng, n: A1<f64>, a: usize, b: usize, weights: A2<f64>, credit: bool, out: A2<f64>, comps: A1<i64>, which: A1<i64>, maps: A3<f64>, scratch: A2<f64>,
             rec: A1<i64>, counted: A2<f64>, rows: A2<f64>) {
    let ng = n.len();
    let (total, left, drawn, prob) = (scratch.row(0), scratch.row(1), scratch.row(2), rows.row(1));
    if b == a + 1 && amin(weights.row(a)) >= 1.0 - 1e-12 {
        land(rng, n, if credit { -1 } else { which.at(a) }, maps, out.row(comps.u(a)));
        if !credit && rec.at(a) >= 0 {
            for g in 0..ng {
                counted.add(rec.u(a), g, n.at(g));
            }
        }
    } else if b > a {
        let mut whole = true;
        for g in 0..ng {
            total.set(g, 0.0);
            left.set(g, n.at(g));
            for k in a..b {
                total.add(g, weights.at(k, g));
            }
            whole = whole && (total.at(g) - 1.0).abs() < 1e-9;
            if 0.0 < total.at(g) && total.at(g) < 1.0 - 1e-9 { // Shares that sum below 1 leave the rest nowhere, as in the ODE
                total.set(g, 1.0);
            }
        }
        for k in a..b { // One multinomial draw, as a chain of conditional binomials
            for g in 0..ng {
                prob.set(g, if total.at(g) > 0.0 { weights.at(k, g)/total.at(g) } else { 0.0 });
            }
            some_binomials(rng, left, prob, drawn, rows);
            for g in 0..ng {
                left.set(g, left.at(g) - drawn.at(g));
                total.set(g, total.at(g) - weights.at(k, g));
            }
            land(rng, drawn, if credit { -1 } else { which.at(k) }, maps, out.row(comps.u(k)));
            if !credit && rec.at(k) >= 0 {
                for g in 0..ng {
                    counted.add(rec.u(k), g, drawn.at(g));
                }
            }
        }
        if !credit && whole && amax(left) > 0.0 { // Rounding only
            land(rng, left, which.at(b - 1), maps, out.row(comps.u(b - 1)));
            if rec.at(b - 1) >= 0 {
                for g in 0..ng {
                    counted.add(rec.u(b - 1), g, left.at(g));
                }
            }
        }
    }
}

/// Count the events of a flow also in the transitions they pass through with no wait (`credit()`)
#[allow(clippy::too_many_arguments)]
pub fn credit(rng: Rng, n: A1<f64>, a: usize, b: usize, parent: A1<i64>, weights: A2<f64>, index: A1<i64>, scratch: A3<f64>, flows: A2<f64>, rows: A2<f64>) {
    if b == a {
        return;
    }
    let ng = n.len();
    let (left, mass, prob) = (scratch.plane(0), scratch.plane(1), rows.row(1));
    for g in 0..ng {
        left.set(0, g, n.at(g));
        mass.set(0, g, 1.0);
    }
    for k in a..b {
        let q = parent.at(k);
        let i = k - a + 1;
        let j = if q < 0 { 0 } else { q as usize - a + 1 };
        if q == -2 {
            some_binomials(rng, n, weights.row(k), left.row(i), rows);
        } else {
            for g in 0..ng {
                prob.set(g, if mass.at(j, g) > 0.0 { weights.at(k, g)/mass.at(j, g) } else { 0.0 });
            }
            some_binomials(rng, left.row(j), prob, left.row(i), rows);
        }
        for g in 0..ng {
            if q != -2 {
                left.set(j, g, left.at(j, g) - left.at(i, g));
                mass.set(j, g, mass.at(j, g) - weights.at(k, g));
            }
            mass.set(i, g, weights.at(k, g));
            flows.add(index.u(k), g, left.at(i, g));
        }
    }
}

/// As `credit()`, for one event in group g, from a uniform for each credit (`credit_one()`)
pub fn credit_one(us: &[f64], a: usize, b: usize, p: &Program, g: usize, flows: A2<f64>) {
    let mut left = vec![0.0; b - a + 1];
    let mut mass = vec![0.0; b - a + 1];
    left[0] = 1.0;
    mass[0] = 1.0;
    for k in a..b {
        let q = p.crd_parent.at(k);
        let i = k - a + 1;
        let w = p.crd_w.at(k, g);
        let hit;
        if q == -2 {
            hit = us[i - 1] < w;
        } else {
            let j = if q < 0 { 0 } else { q as usize - a + 1 };
            hit = left[j] > 0.0 && us[i - 1]*mass[j] < w;
            if hit {
                left[j] = 0.0;
            }
            mass[j] -= w;
        }
        left[i] = if hit { 1.0 } else { 0.0 };
        mass[i] = w;
        if hit {
            flows.add(p.crd_index.u(k), g, 1.0);
        }
    }
}

/// Where people at a stage of a dwell time are at the end of a step, with d stages left (`chain_probs()`)
pub fn chain_probs(r: f64, h: f64, dt: f64, d: usize, out: A1<f64>) {
    let x = r*dt;
    let xt = (r + h)*dt;
    let stay = (-h*dt).exp();
    let mut term = (-x).exp();
    let mut term_t = (-xt).exp();
    let mut cdf_t = 0.0;
    let mut total = 0.0;
    for m in 0..d {
        out.set(m, stay*term);
        total += out.at(m);
        cdf_t += term_t;
        term *= x/(m + 1) as f64;
        term_t *= xt/(m + 1) as f64;
    }
    out.set(d, if r + h > 0.0 { (r/(r + h)).powf(d as f64)*mx(1.0 - cdf_t, 0.0) } else { 0.0 });
    out.set(d + 1, mx(1.0 - total - out.at(d), 0.0));
}

/// For the chain-binomial solver: the people at a stage of a dwell time move through as many stages as they pass in the step (`draw_chain()`)
#[allow(clippy::too_many_arguments)]
pub fn draw_chain(rng: Rng, y: A2<f64>, p: &Program, w: &Work, s: usize, dt: f64, new: A2<f64>, flows: A2<f64>) {
    let ng = y.n1();
    let (rates, groups, probs) = (w.rates, w.groups, w.chain);
    let (total, left, n, ends) = (groups.row(3), groups.row(4), groups.row(5), groups.row(6));
    let (a, b) = (p.src_starts.u(s), p.src_starts.u(s + 1));
    let d = p.src_chain.u(s);
    let stage = p.src_stage.u(s);
    let src = p.src.u(stage);
    let stride = p.tgt_comp.u(p.tgt_starts.u(stage)) - src; // How far apart its stages are: 1, except in the program of a crossing
    for g in 0..ng {
        let r = mx(rates.at(stage, g), 0.0);
        let mut h = 0.0;
        for k in a..b {
            let f = p.src_flows.u(k);
            if f != stage {
                h += mx(rates.at(f, g), 0.0);
            }
        }
        chain_probs(r, h, dt, d, probs);
        let mut rest = y.at(src, g);
        let mut mass = 1.0;
        left.set(g, binomial(rng, rest, probs.at(d + 1))); // Leave by another exit
        rest -= left.at(g);
        mass -= probs.at(d + 1);
        ends.set(g, if mass > 0.0 { binomial(rng, rest, probs.at(d)/mass) } else { rest }); // Leave by the last stage
        rest -= ends.at(g);
        mass -= probs.at(d);
        let mut moved = left.at(g) + ends.at(g);
        for m in 1..d { // Move on some stages; the rest stay
            let k = if mass > 0.0 { binomial(rng, rest, probs.at(m)/mass) } else { 0.0 };
            new.add(src + m*stride, g, k);
            rest -= k;
            mass -= probs.at(m);
            moved += k;
        }
        new.set(src, g, new.at(src, g) - moved);
        total.set(g, h);
    }
    for k in a..b { // Another exit: as in draw()
        let f = p.src_flows.u(k);
        if f == stage {
            continue;
        }
        for g in 0..ng {
            let hazard = mx(rates.at(f, g), 0.0);
            n.set(g, if total.at(g) > 0.0 { binomial(rng, left.at(g), hazard/total.at(g)) } else { 0.0 });
            total.set(g, total.at(g) - hazard);
            left.set(g, left.at(g) - n.at(g));
        }
        if p.named.b(f) {
            land(rng, n, p.record.at(f), p.maps, flows.row(p.index.u(f)));
        }
        split(rng, n, p.tgt_starts.u(f), p.tgt_starts.u(f + 1), p.tgt_w, false, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w.rows);
        credit(rng, n, p.crd_starts.u(f), p.crd_starts.u(f + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, w.rows);
    }
    let f = p.src_last.u(s); // The last stage
    if p.named.b(f) {
        land(rng, ends, p.record.at(f), p.maps, flows.row(p.index.u(f)));
    }
    split(rng, ends, p.tgt_starts.u(f), p.tgt_starts.u(f + 1), p.tgt_w, false, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w.rows);
    credit(rng, ends, p.crd_starts.u(f), p.crd_starts.u(f + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, w.rows);
}

/// Draw whole numbers of events over a time dt, and add them to the flows (`draw()`): for each compartment, the number who leave it, then which flow each leaves by
pub fn draw(rng: Rng, y: A2<f64>, p: &Program, w: &Work, dt: f64, flows: A2<f64>, tau: bool) {
    let ng = y.n1();
    let (rates, groups, new, rows) = (w.rates, w.groups, w.yt, w.rows);
    let (total, left, n, prob) = (groups.row(3), groups.row(4), groups.row(5), rows.row(1));
    new.copy_from(y);
    for s in 0..p.src_starts.len() - 1 {
        let (a, b) = (p.src_starts.u(s), p.src_starts.u(s + 1));
        if a == b { // In the program of a crossing, the end of one disease's flows
            y.copy_from(new);
            continue;
        }
        let src = p.src.at(p.src_flows.u(a));
        if src >= 0 && p.det.b(p.src_flows.u(a)) { // A state that is not people: see settle()
            continue;
        }
        if !tau && p.src_chain.at(s) > 1 {
            draw_chain(rng, y, p, w, s, dt, new, flows);
            continue;
        }
        if src >= 0 {
            let src = src as usize;
            for g in 0..ng {
                let mut t = 0.0;
                for k in a..b {
                    t += mx(rates.at(p.src_flows.u(k), g), 0.0);
                }
                total.set(g, t);
                prob.set(g, if tau { t*y.at(src, g)*dt } else { -(-t*dt).exp_m1() });
            }
            if tau {
                poissons(rng, prob, left, rows);
            } else {
                binomials(rng, y.row(src), prob, left, rows);
            }
            for g in 0..ng {
                left.set(g, mn(left.at(g), y.at(src, g))); // For tau-leaping, everyone who leaves, clipped once
                new.set(src, g, new.at(src, g) - left.at(g));
            }
        }
        for k in a..b {
            let f = p.src_flows.u(k);
            if p.det.b(f) {
                continue;
            }
            if src < 0 { // An inflow
                for g in 0..ng {
                    prob.set(g, mx(rates.at(f, g)*p.inflow.at(f, g)*dt, 0.0));
                }
                poissons(rng, prob, n, rows);
            } else { // Which flow they leave by: one multinomial draw, as a chain of conditional binomials
                for g in 0..ng {
                    let hazard = mx(rates.at(f, g), 0.0);
                    prob.set(g, if total.at(g) > 0.0 { hazard/total.at(g) } else { 0.0 });
                    total.set(g, total.at(g) - hazard);
                }
                binomials(rng, left, prob, n, rows);
                for g in 0..ng {
                    left.set(g, left.at(g) - n.at(g));
                }
            }
            if p.named.b(f) && p.record.at(f) < 0 {
                let r = flows.row(p.index.u(f));
                for g in 0..ng {
                    r.add(g, n.at(g));
                }
            } else if p.named.b(f) {
                land(rng, n, p.record.at(f), p.maps, flows.row(p.index.u(f)));
            }
            let t = p.tgt_starts.u(f);
            if p.tgt_starts.u(f + 1) == t + 1 && p.tgt_map.at(t) < 0 && p.tgt_rec.at(t) < 0 && amin(p.tgt_w.row(t)) >= 1.0 - 1e-12 { // Everyone to one compartment, in the same group
                let r = new.row(p.tgt_comp.u(t));
                for g in 0..ng {
                    r.add(g, n.at(g));
                }
            } else {
                split(rng, n, t, p.tgt_starts.u(f + 1), p.tgt_w, false, new, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, rows);
            }
            credit(rng, n, p.crd_starts.u(f), p.crd_starts.u(f + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, rows);
        }
    }
    move_people(rng, new, p, dt, flows); // Movement between groups, from where the flows left people
    y.copy_from(new);
}

/// Move people between groups over a time dt, as whole numbers drawn at random (`move()`)
pub fn move_people(rng: Rng, y: A2<f64>, p: &Program, dt: f64, flows: A2<f64>) {
    for m in 0..p.mov_starts.n1().saturating_sub(1) {
        for e in p.mov_starts.u(1, m)..p.mov_starts.u(1, m + 1) {
            let rate = p.mov_rate.at(e);
            if rate != 0.0 {
                let (g, h) = (p.mov_pair.u(0, e), p.mov_pair.u(1, e));
                let prob = -(-rate*dt).exp_m1();
                for k in p.mov_starts.u(0, m)..p.mov_starts.u(0, m + 1) {
                    let c = p.mov_comps.u(k);
                    let moved = binomial(rng, y.at(c, g), prob);
                    y.set(c, g, y.at(c, g) - moved);
                    y.add(c, h, moved);
                    flows.add(p.n_trans + m, g, moved);
                }
            }
        }
    }
}

/// The rate at which events of a flow happen in a group (`propensity()`)
#[inline(always)]
pub fn propensity(y: A2<f64>, p: &Program, rates: A2<f64>, f: usize, g: usize) -> f64 {
    let s = p.src.at(f);
    if p.det.b(f) {
        return 0.0;
    }
    mx(rates.at(f, g)*(if s >= 0 { y.at(s as usize, g) } else { p.inflow.at(f, g) }), 0.0)
}

/// For the exact solver: the time to the next event and, if it comes within dt, the event itself (`event()`); returns the time taken
pub fn event(rng: Rng, y: A2<f64>, p: &Program, w: &Work, dt: f64, flows: A2<f64>) -> f64 {
    let (nf, ng) = (p.src.len(), y.n1());
    let (rates, groups) = (w.rates, w.groups);
    let n = groups.row(5);
    let mut total = 0.0;
    for f in 0..nf {
        for g in 0..ng {
            total += propensity(y, p, rates, f, g);
        }
    }
    if total <= 0.0 {
        return dt;
    }
    let wait = rng.exponential(1.0/total);
    if wait > dt {
        return dt;
    }

    // Which event: the last one with any chance, if rounding leaves the draw past the end
    let target = rng.random()*total;
    let mut cum = 0.0;
    let (mut which, mut place) = (0, 0);
    'flows: for f in 0..nf {
        for g in 0..ng {
            let rate = propensity(y, p, rates, f, g);
            if rate > 0.0 {
                cum += rate;
                which = f;
                place = g;
                if cum > target {
                    break 'flows;
                }
            }
        }
    }
    n.fill(0.0);
    n.set(place, 1.0);
    if p.src.at(which) >= 0 {
        y.set(p.src.u(which), place, y.at(p.src.u(which), place) - 1.0);
    }
    if p.named.b(which) {
        land(rng, n, p.record.at(which), p.maps, flows.row(p.index.u(which)));
    }
    split(rng, n, p.tgt_starts.u(which), p.tgt_starts.u(which + 1), p.tgt_w, false, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w.rows);
    credit(rng, n, p.crd_starts.u(which), p.crd_starts.u(which + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, w.rows);
    wait
}

/// The conveyors of fixed delays and minimum stays: release the people in the last slot of each, or move them all along one slot (`convey()`)
pub fn convey(rng: Rng, y: A2<f64>, p: &Program, w: &Work, flows: A2<f64>, release: bool) {
    let ng = y.n1();
    let nf = p.src.len();
    let (groups, idle) = (w.groups, w.cells.row(5));
    let n = groups.row(5);
    for i in 0..p.starts.len() - nf - 1 {
        let (a, b) = (p.starts.u(nf + i), p.starts.u(nf + i + 1));
        if p.ops.at(a) != CONVEY {
            continue;
        }
        if release {
            let last = p.args.u(b - 1);
            let f = p.args.u(a);
            for g in 0..ng {
                n.set(g, y.at(last, g));
                y.set(last, g, 0.0);
                if n.at(g) > 0.0 {
                    idle.set(p.places.u(g), 0.0);
                }
            }
            if p.named.b(f) {
                land(rng, n, p.record.at(f), p.maps, flows.row(p.index.u(f)));
            }
            split(rng, n, p.tgt_starts.u(f), p.tgt_starts.u(f + 1) - 1, p.tgt_w, false, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, w.rows); // Not the last target, which is the slot they leave
            credit(rng, n, p.crd_starts.u(f), p.crd_starts.u(f + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, w.rows);
        } else {
            for g in 0..ng {
                let mut k = b - 1;
                while k > a + 1 { // The last slot is empty by now (or nearly, by rounding, in the ODE)
                    y.set(p.args.u(k), g, y.at(p.args.u(k - 1), g));
                    if y.at(p.args.u(k), g) > 0.0 {
                        idle.set(p.places.u(g), 0.0);
                    }
                    k -= 1;
                }
                y.set(p.args.u(a + 1), g, 0.0);
            }
        }
    }
}

/// When a flow has an infinite rate: everyone in its compartment leaves by it, at the start of each substep (`certain()`)
#[allow(clippy::too_many_arguments)]
pub fn certain(rng: Rng, y: A2<f64>, p: &Program, w: &Work, rates: A2<f64>, flows: A2<f64>, mv: bool, solver: i64) {
    let ng = y.n1();
    let (groups, rows) = (w.groups, w.rows);
    let (_o1, left) = own1(ng, 0.0);
    let (_o2, n) = own1(ng, 0.0);
    let (_o3, prob) = own1(ng, 0.0);
    let (_o4, k) = own1(ng, 0.0);
    for s in 0..p.src_starts.len() - 1 {
        let (a, b) = (p.src_starts.u(s), p.src_starts.u(s + 1));
        if a == b || p.src.at(p.src_flows.u(a)) < 0 {
            continue;
        }
        let src = p.src.u(p.src_flows.u(a));
        if mv {
            for g in 0..ng {
                left.set(g, y.at(src, g));
                k.set(g, 0.0);
                for j in a..b {
                    k.add(g, (rates.at(p.src_flows.u(j), g) == f64::INFINITY) as u8 as f64);
                }
            }
            for j in a..b {
                let f = p.src_flows.u(j);
                let mut some = false;
                for g in 0..ng {
                    prob.set(g, 0.0);
                    if rates.at(f, g) == f64::INFINITY { // One of k[g] equal shares of those left
                        prob.set(g, 1.0/k.at(g));
                        k.set(g, k.at(g) - 1.0);
                        some = true;
                    }
                }
                if !some {
                    continue;
                }
                let drawn = solver >= 2 && !p.det.b(f);
                if drawn {
                    binomials(rng, left, prob, n, rows);
                } else {
                    for g in 0..ng {
                        n.set(g, left.at(g)*prob.at(g));
                    }
                }
                for g in 0..ng {
                    left.set(g, left.at(g) - n.at(g));
                    y.set(src, g, y.at(src, g) - n.at(g));
                }
                if drawn { // Whole people, as draw() puts them
                    if p.named.b(f) {
                        land(rng, n, p.record.at(f), p.maps, flows.row(p.index.u(f)));
                    }
                    split(rng, n, p.tgt_starts.u(f), p.tgt_starts.u(f + 1), p.tgt_w, false, y, p.tgt_comp, p.tgt_map, p.maps, groups, p.tgt_rec, flows, rows);
                    credit(rng, n, p.crd_starts.u(f), p.crd_starts.u(f + 1), p.crd_parent, p.crd_w, p.crd_index, w.credits, flows, rows);
                    continue;
                }
                for g in 0..ng { // Expected numbers, as the ODE puts them
                    let ng_ = n.at(g);
                    if ng_ == 0.0 {
                        continue;
                    }
                    if p.named.b(f) {
                        if p.record.at(f) < 0 {
                            flows.add(p.index.u(f), g, ng_);
                        } else {
                            for h in 0..ng {
                                flows.add(p.index.u(f), h, ng_*p.maps.at(p.record.u(f), g, h));
                            }
                        }
                    }
                    for t in p.tgt_starts.u(f)..p.tgt_starts.u(f + 1) {
                        if p.tgt_rec.at(t) >= 0 {
                            flows.add(p.tgt_rec.u(t), g, ng_*p.tgt_w.at(t, g));
                        }
                        if p.tgt_map.at(t) < 0 {
                            y.add(p.tgt_comp.u(t), g, ng_*p.tgt_w.at(t, g));
                        } else {
                            for h in 0..ng {
                                y.add(p.tgt_comp.u(t), h, ng_*p.tgt_w.at(t, g)*p.maps.at(p.tgt_map.u(t), g, h));
                            }
                        }
                    }
                    for t in p.crd_starts.u(f)..p.crd_starts.u(f + 1) {
                        flows.add(p.crd_index.u(t), g, ng_*p.crd_w.at(t, g));
                    }
                }
            }
        }
        for j in a..b {
            for g in 0..ng {
                if rates.at(p.src_flows.u(j), g) == f64::INFINITY {
                    rates.set(p.src_flows.u(j), g, 0.0);
                }
            }
        }
    }
}

/// In a program with conveyors: for the ODE, the people due to leave each conveyor over the step, as its flow's inflow; returns whether a stochastic solver has people to release (`step_start()`)
pub fn step_start(y: A2<f64>, p: &Program, dt: f64, solver: i64) -> bool {
    let nf = p.src.len();
    let ng = y.n1();
    let mut pending = false;
    for i in 0..p.starts.len() - nf - 1 {
        let (a, b) = (p.starts.u(nf + i), p.starts.u(nf + i + 1));
        if p.ops.at(a) == CONVEY {
            pending = solver >= 2;
            if solver < 2 {
                for g in 0..ng {
                    p.inflow.set(p.args.u(a), g, y.at(p.args.u(b - 1), g)/dt);
                }
            }
        }
    }
    pending
}

/// For the ODE, in a program with conveyors: the rate at which the people due to leave each conveyor leave it over the step, lowered for its other exits (`release_rates()`)
pub fn release_rates(y: A2<f64>, p: &Program, rates: A2<f64>, dt: f64) {
    let nf = p.src.len();
    let ng = y.n1();
    let mut is_last = vec![false; y.n0()];
    for i in 0..p.starts.len() - nf - 1 {
        let (a, b) = (p.starts.u(nf + i), p.starts.u(nf + i + 1));
        if p.ops.at(a) == CONVEY {
            is_last[p.args.u(b - 1)] = true;
        }
    }
    let mut exits = vec![false; nf];
    for i in 0..p.starts.len() - nf - 1 {
        let (a, b) = (p.starts.u(nf + i), p.starts.u(nf + i + 1));
        if p.ops.at(a) != CONVEY {
            continue;
        }
        let last = p.args.at(b - 1);
        let f_out = p.args.u(a);
        exits.fill(false);
        for f in 0..nf {
            if p.src.at(f) == last {
                exits[f] = p.tgt_starts.at(f + 1) == p.tgt_starts.at(f); // A flow out of the model (a death) is an exit
                for t in p.tgt_starts.u(f)..p.tgt_starts.u(f + 1) {
                    if !is_last[p.tgt_comp.u(t)] {
                        exits[f] = true;
                    }
                }
            }
        }
        for g in 0..ng {
            let mut x = 0.0;
            for f in 0..nf {
                if exits[f] {
                    x += mx(rates.at(f, g), 0.0)*dt;
                }
            }
            if x > 1e-12 {
                p.inflow.set(f_out, g, p.inflow.at(f_out, g)*(x/x.exp_m1()));
            }
        }
    }
}

/// For the solvers that draw whole events: the flows of states that are not people, as expected values over a time dt, by RK4 in enough substeps (`settle()`)
pub fn settle(y: A2<f64>, p: &Program, rates: A2<f64>, dt: f64, flows: A2<f64>) {
    let mut fastest = 0.0;
    for f in 0..p.src.len() {
        if p.det.b(f) && p.src.at(f) >= 0 {
            for g in 0..y.n1() {
                fastest = mx(fastest, rates.at(f, g));
            }
        }
    }
    let n_sub = to_int((fastest*dt/0.25).ceil()).max(1);
    for _ in 0..n_sub {
        settle_step(y, p, rates, dt/n_sub as f64, flows);
    }
}

/// For `settle()`: one RK4 step of the counts over a time dt (`settle_step()`)
pub fn settle_step(y: A2<f64>, p: &Program, rates: A2<f64>, dt: f64, flows: A2<f64>) {
    let (nc, ng) = (y.n0(), y.n1());
    let nf = p.src.len();
    let nflows = flows.n0();
    let (_o1, k) = own3(4, nc, ng, 0.0);
    let (_o2, kf) = own3(4, nflows, ng, 0.0);
    let (_o3, y0) = own2(nc, ng, 0.0);
    let (_o4, yt) = own2(nc, ng, 0.0);
    y0.copy_from(y);
    yt.copy_from(y);
    for stage in 0..4 {
        let (ks, kfs) = (k.plane(stage), kf.plane(stage));
        for f in 0..nf {
            if !p.det.b(f) {
                continue;
            }
            let s = p.src.at(f);
            for g in 0..ng {
                let amount = mx(rates.at(f, g)*(if s >= 0 { yt.at(s as usize, g) } else { p.inflow.at(f, g) }), 0.0);
                if amount == 0.0 {
                    continue;
                }
                if s >= 0 {
                    ks.set(s as usize, g, ks.at(s as usize, g) - amount);
                }
                if p.named.b(f) {
                    if p.record.at(f) < 0 {
                        kfs.add(p.index.u(f), g, amount);
                    } else {
                        for j in 0..ng {
                            kfs.add(p.index.u(f), j, amount*p.maps.at(p.record.u(f), g, j));
                        }
                    }
                }
                for t in p.tgt_starts.u(f)..p.tgt_starts.u(f + 1) {
                    if p.tgt_rec.at(t) >= 0 {
                        kfs.add(p.tgt_rec.u(t), g, amount*p.tgt_w.at(t, g));
                    }
                    if p.tgt_map.at(t) < 0 {
                        ks.add(p.tgt_comp.u(t), g, amount*p.tgt_w.at(t, g));
                    } else {
                        for j in 0..ng {
                            ks.add(p.tgt_comp.u(t), j, amount*p.tgt_w.at(t, g)*p.maps.at(p.tgt_map.u(t), g, j));
                        }
                    }
                }
                for t in p.crd_starts.u(f)..p.crd_starts.u(f + 1) {
                    kfs.add(p.crd_index.u(t), g, amount*p.crd_w.at(t, g));
                }
            }
        }
        if stage < 3 {
            let ahead = if stage == 2 { dt } else { dt/2.0 };
            for c in 0..nc {
                for g in 0..ng {
                    yt.set(c, g, mx(y0.at(c, g) + ahead*ks.at(c, g), 0.0));
                }
            }
        }
    }
    for c in 0..nc {
        for g in 0..ng {
            y.set(c, g, mx(y0.at(c, g) + dt/6.0*(k.at(0, c, g) + 2.0*k.at(1, c, g) + 2.0*k.at(2, c, g) + k.at(3, c, g)), 0.0));
        }
    }
    for f in 0..nflows {
        for g in 0..ng {
            flows.add(f, g, dt/6.0*(kf.at(0, f, g) + 2.0*kf.at(1, f, g) + 2.0*kf.at(2, f, g) + kf.at(3, f, g)));
        }
    }
}

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
