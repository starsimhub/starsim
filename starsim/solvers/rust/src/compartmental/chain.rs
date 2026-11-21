//! The draws of the chain binomial and tau-leaping, by flow and group, and moving people between groups (the section "Chain" of
//! `starsim/solvers/numba/compartmental.py`, the functions of the same names)

use super::*;

/// Gamma white noise over a time h: a multiplier with mean 1 and variance sigma²/h, from one uniform; 1 with no draw for sigma = 0 (`noise()`)
#[inline(always)]
pub fn noise(rng: Rng, sigma: f64, h: f64) -> f64 {
    if sigma <= 0.0 {
        return 1.0;
    }
    let var = sigma*sigma;
    let (g, _) = crate::draws::gamma(h/var, crate::draws::stream(rng.random()));
    g*var/h
}

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
            if 0.0 < total.at(g) && total.at(g) < 1.0 - 1e-9 { // Fractions that sum below 1 leave the rest nowhere, as in the ODE
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
    let stride = p.tgt_comp.u(p.tgt_starts.u(stage)) - src; // How far apart its stages are: 1, except in the program of joint states
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
pub fn draw(rng: Rng, y: A2<f64>, p: &Program, w: &Work, dt: f64, flows: A2<f64>, tau: bool, whole: bool) {
    let ng = y.n1();
    let (rates, groups, new, rows) = (w.rates, w.groups, w.yt, w.rows);
    let (total, left, n, prob) = (groups.row(3), groups.row(4), groups.row(5), rows.row(1));
    new.copy_from(y);
    for s in 0..p.src_starts.len() - 1 {
        let (a, b) = (p.src_starts.u(s), p.src_starts.u(s + 1));
        if a == b { // In the program of joint states, the end of one disease's flows
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
            let transmits = (0..p.set_comps.len()).any(|k| p.set_comps.at(k) == src); // As draw(): whether people here are a source of infection
            let src = src as usize;
            for g in 0..ng {
                let mut t = 0.0;
                for k in a..b {
                    t += mx(rates.at(p.src_flows.u(k), g), 0.0);
                }
                total.set(g, t);
                prob.set(g, if tau { t*y.at(src, g)*dt } else if transmits && !whole { (t*dt).min(1.0) } else { -(-t*dt).exp_m1() }); // As draw()
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
