//! What every solver does with people who must move at once: infinite rates and the conveyors of fixed delays and minimum stays
//! (the section "At once" of `starsim/solvers/numba/compartmental.py`, the functions of the same names)

use super::*;

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
                while k > a + 1 { // The last slot is empty by now (or nearly, in the ODE, whose remainder there leaves with the next)
                    y.set(p.args.u(k), g, y.at(p.args.u(k - 1), g) + if k == b - 1 { y.at(p.args.u(k), g) } else { 0.0 });
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
                let drawn = solver >= BINOMIAL && !p.det.b(f);
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
                    if n.at(g) != 0.0 {
                        put_expected(p, f, g, n.at(g), y, flows);
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
            pending = solver >= BINOMIAL;
            if solver < BINOMIAL {
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
