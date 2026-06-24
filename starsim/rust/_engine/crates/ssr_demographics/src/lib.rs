//! Vital dynamics: Births (append agents) and Deaths (flip `alive`).
//! Separately compiled; depends only on `ssr_core`. Demonstrates the engine's
//! dynamic-population support (births grow every module's per-agent arrays).

use ssr_core::{Demographics, People, Rng};

/// Births: each alive agent produces a new (susceptible) agent with probability
/// `birth_prob` per step. New agents are appended to the population.
pub struct Births {
    birth_prob: f64,
    rng: Rng,
}

impl Births {
    pub fn new(birth_prob: f64, seed: u64) -> Self {
        Self { birth_prob, rng: Rng::new(seed) }
    }
}

impl Demographics for Births {
    fn step(&mut self, people: &mut People) -> usize {
        let mut k = 0usize;
        for i in 0..people.n {
            if people.alive[i] && self.rng.rand_f64() < self.birth_prob {
                k += 1;
            }
        }
        if k > 0 {
            people.grow(k);
        }
        k
    }
    fn name(&self) -> &str { "births" }
}

/// Deaths: each alive agent dies with probability `death_prob` per step.
pub struct Deaths {
    death_prob: f64,
    rng: Rng,
}

impl Deaths {
    pub fn new(death_prob: f64, seed: u64) -> Self {
        Self { death_prob, rng: Rng::new(seed) }
    }
}

impl Demographics for Deaths {
    fn step(&mut self, people: &mut People) -> usize {
        for i in 0..people.n {
            if people.alive[i] && self.rng.rand_f64() < self.death_prob {
                people.alive[i] = false;
            }
        }
        0 // deaths add no agents
    }
    fn name(&self) -> &str { "deaths" }
}
