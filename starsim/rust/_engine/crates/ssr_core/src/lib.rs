//! Core of the Starsim fast modular Rust engine.
//!
//! Design goals (see starsim/rust/STATUS.md):
//!   * FAST, not byte-identical: a quick RNG (xoshiro256**), lazy per-edge
//!     transmission draws, no CRN. Results match Starsim *statistically*, not
//!     bit-for-bit.
//!   * MODULAR: diseases and networks are trait objects. Each concrete module
//!     (SIS, RandomNet, SIR, ...) lives in its own crate and is composed at
//!     runtime via `Box<dyn Disease>` / `Box<dyn Network>`. Adding a module
//!     does not require recompiling the others or this core.
//!
//! The loop dispatch is Rust->Rust vtable calls (nanoseconds), once per module
//! per timestep -- not per agent -- so modularity is essentially free.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// RNG: xoshiro256** seeded via SplitMix64 (fast; not numpy-compatible)
// ---------------------------------------------------------------------------
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1)
    #[inline]
    pub fn rand_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller
    pub fn randn(&mut self) -> f64 {
        loop {
            let u1 = self.rand_f64();
            if u1 > 0.0 {
                let u2 = self.rand_f64();
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    /// Lognormal with given mean and (here) exponential-ish SD = mean (matches ss.lognorm_ex default)
    pub fn lognormal_ex(&mut self, mean: f64, std: f64, n: usize) -> Vec<f64> {
        if mean <= 0.0 {
            return vec![mean; n];
        }
        let sigma = (1.0 + (std / mean).powi(2)).ln().sqrt();
        let mu = mean.ln() - sigma * sigma / 2.0;
        (0..n).map(|_| (mu + sigma * self.randn()).exp()).collect()
    }

    #[inline]
    pub fn rand_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle in place
    pub fn shuffle(&mut self, v: &mut [u32]) {
        for i in (1..v.len()).rev() {
            let j = self.rand_below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Population with a liveness mask. `n` is the total number of slots ever
/// created (= len of all per-agent arrays); births append, deaths flip `alive`.
pub struct People {
    pub n: usize,
    pub alive: Vec<bool>,
}

impl People {
    pub fn new(n: usize) -> Self {
        People { n, alive: vec![true; n] }
    }
    pub fn n_alive(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }
    /// Append `k` new (alive) agents; returns the new total `n`.
    pub fn grow(&mut self, k: usize) -> usize {
        self.alive.extend(std::iter::repeat(true).take(k));
        self.n += k;
        self.n
    }
}

/// Network edges as parallel arrays.
#[derive(Default)]
pub struct Edges {
    pub p1: Vec<u32>,
    pub p2: Vec<u32>,
    pub beta: Vec<f32>,
    pub dur: Vec<f32>,
}

pub type Results = BTreeMap<String, Vec<f64>>;

// ---------------------------------------------------------------------------
// Module traits (object-safe)
// ---------------------------------------------------------------------------
pub trait Network {
    fn step(&mut self, people: &People);
    fn edges(&self) -> &Edges;
    fn name(&self) -> &str;
    /// Extend internal per-agent state for `k` newly added agents (default: nothing).
    fn grow(&mut self, _k: usize) {}
}

pub trait Disease {
    fn step_state(&mut self, people: &People);
    fn infect(&mut self, networks: &[Box<dyn Network>], people: &People);
    fn update_results(&mut self, ti: usize, people: &People);
    fn finalize(&mut self);
    fn results(&self) -> &Results;
    fn name(&self) -> &str;
    /// Extend per-agent state arrays for `k` newly added (susceptible) agents.
    fn grow(&mut self, _k: usize) {}
}

/// Vital dynamics: births (append agents) and deaths (flip `alive`).
pub trait Demographics {
    /// Apply this timestep; return the number of agents *added* (births), so the
    /// driver can grow the other modules' per-agent arrays to match.
    fn step(&mut self, people: &mut People) -> usize;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Loop driver: fixed Starsim order, all in Rust, no per-timestep Python round-trip
// ---------------------------------------------------------------------------
pub struct Sim {
    pub people: People,
    pub networks: Vec<Box<dyn Network>>,
    pub diseases: Vec<Box<dyn Disease>>,
    pub demographics: Vec<Box<dyn Demographics>>,
    pub n_steps: usize,
}

impl Sim {
    pub fn run(&mut self) {
        // Destructure so the borrow checker sees disjoint fields
        let Sim { people, networks, diseases, demographics, n_steps } = self;
        for ti in 0..*n_steps {
            // Vital dynamics first: births append agents, deaths flip `alive`.
            for dem in demographics.iter_mut() {
                let n_new = dem.step(people);
                if n_new > 0 { // grow every module's per-agent arrays to match
                    for d in diseases.iter_mut() { d.grow(n_new); }
                    for n in networks.iter_mut() { n.grow(n_new); }
                }
            }
            for d in diseases.iter_mut() { d.step_state(people); }     // recovery / autonomous
            for n in networks.iter_mut() { n.step(people); }           // regenerate contacts
            for d in diseases.iter_mut() { d.infect(networks, people); } // transmission
            for d in diseases.iter_mut() { d.update_results(ti, people); }
        }
        for d in diseases.iter_mut() { d.finalize(); }
    }
}
