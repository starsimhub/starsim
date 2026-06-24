//! SIS disease: susceptible <-> infected with waning immunity.
//! Separately compiled; depends only on `ssr_core`.
//!
//! FAST mode: transmission uses a lazy per-edge Bernoulli draw (only where an
//! infected meets a susceptible), not Starsim's CRN block. Results match
//! Starsim statistically, not bit-for-bit.

use ssr_core::{Disease, Network, People, Results, Rng};

pub struct Sis {
    n: usize,
    beta: f64,        // per-contact-per-step transmission probability
    dur_inf: f64,     // mean infectious duration in timesteps
    dur_inf_std: f64, // SD of infectious duration (ss.lognorm_ex uses ~1.0)
    waning: f32,
    imm_boost: f32,
    ti: f64,
    susceptible: Vec<bool>,
    infected: Vec<bool>,
    ti_recovered: Vec<f32>,
    immunity: Vec<f32>,
    rel_sus: Vec<f32>,
    rng: Rng,
    results: Results,
    new_this_step: usize,
}

impl Sis {
    #[allow(clippy::too_many_arguments)]
    pub fn new(n: usize, n_steps: usize, beta: f64, init_prev: f64, dur_inf: f64,
               dur_inf_std: f64, waning: f32, imm_boost: f32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut susceptible = vec![true; n];
        let mut infected = vec![false; n];
        let mut ti_recovered = vec![0.0f32; n];

        // Seed initial infections (bernoulli init_prev), set recovery times
        let mut initial: Vec<usize> = Vec::new();
        for i in 0..n {
            if rng.rand_f64() < init_prev {
                initial.push(i);
            }
        }
        let durs = rng.lognormal_ex(dur_inf, dur_inf_std, initial.len());
        for (j, &i) in initial.iter().enumerate() {
            susceptible[i] = false;
            infected[i] = true;
            ti_recovered[i] = durs[j] as f32;
        }

        let mut results = Results::new();
        for key in ["n_susceptible", "n_infected", "prevalence", "new_infections", "cum_infections"] {
            results.insert(key.to_string(), vec![0.0; n_steps]);
        }

        Self {
            n, beta, dur_inf, dur_inf_std, waning, imm_boost, ti: 0.0,
            susceptible, infected, ti_recovered,
            immunity: vec![0.0; n], rel_sus: vec![1.0; n],
            rng, results, new_this_step: 0,
        }
    }

    fn set_prognoses(&mut self, uids: &[usize]) {
        let durs = self.rng.lognormal_ex(self.dur_inf, self.dur_inf_std, uids.len());
        for (j, &u) in uids.iter().enumerate() {
            self.susceptible[u] = false;
            self.infected[u] = true;
            self.immunity[u] += self.imm_boost;
            self.ti_recovered[u] = (self.ti + durs[j]) as f32;
        }
        self.new_this_step = uids.len();
    }
}

impl Disease for Sis {
    fn step_state(&mut self, people: &People) {
        // Recovery: alive infectious whose recovery time has passed
        for i in 0..self.susceptible.len() {
            if people.alive[i] && self.infected[i] && self.ti_recovered[i] as f64 <= self.ti {
                self.infected[i] = false;
                self.susceptible[i] = true;
            }
        }
        // Waning immunity
        if self.waning > 0.0 {
            let factor = 1.0 - self.waning;
            for i in 0..self.susceptible.len() {
                if self.immunity[i] > 0.0 {
                    self.immunity[i] *= factor;
                    self.rel_sus[i] = (1.0 - self.immunity[i]).max(0.0);
                }
            }
        }
    }

    fn infect(&mut self, networks: &[Box<dyn Network>], people: &People) {
        let alive = &people.alive;
        let mut targets: Vec<usize> = Vec::new();
        for net in networks {
            let e = net.edges();
            for i in 0..e.p1.len() {
                let a = e.p1[i] as usize;
                let b = e.p2[i] as usize;
                if !(alive[a] && alive[b]) { continue; }
                if self.infected[a] && self.susceptible[b] {
                    let p = self.beta * e.beta[i] as f64 * self.rel_sus[b] as f64;
                    if self.rng.rand_f64() < p { targets.push(b); }
                }
                if self.infected[b] && self.susceptible[a] {
                    let p = self.beta * e.beta[i] as f64 * self.rel_sus[a] as f64;
                    if self.rng.rand_f64() < p { targets.push(a); }
                }
            }
        }
        targets.sort_unstable();
        targets.dedup();
        self.new_this_step = 0;
        if !targets.is_empty() {
            self.set_prognoses(&targets);
        }
    }

    fn update_results(&mut self, ti: usize, people: &People) {
        let mut n_sus = 0usize;
        let mut n_inf = 0usize;
        for i in 0..self.susceptible.len() {
            if !people.alive[i] { continue; }
            if self.susceptible[i] { n_sus += 1; }
            if self.infected[i] { n_inf += 1; }
        }
        let n_alive = people.n_alive().max(1);
        self.results.get_mut("n_susceptible").unwrap()[ti] = n_sus as f64;
        self.results.get_mut("n_infected").unwrap()[ti] = n_inf as f64;
        self.results.get_mut("prevalence").unwrap()[ti] = n_inf as f64 / n_alive as f64;
        self.results.get_mut("new_infections").unwrap()[ti] = self.new_this_step as f64;
        self.ti += 1.0;
    }

    fn grow(&mut self, k: usize) {
        // New agents are alive & susceptible
        self.susceptible.extend(std::iter::repeat(true).take(k));
        self.infected.extend(std::iter::repeat(false).take(k));
        self.ti_recovered.extend(std::iter::repeat(0.0f32).take(k));
        self.immunity.extend(std::iter::repeat(0.0f32).take(k));
        self.rel_sus.extend(std::iter::repeat(1.0f32).take(k));
        self.n += k;
    }

    fn finalize(&mut self) {
        let new = self.results.get("new_infections").unwrap().clone();
        let cum = self.results.get_mut("cum_infections").unwrap();
        let mut acc = 0.0;
        for ti in 0..new.len() {
            acc += new[ti];
            cum[ti] = acc;
        }
    }

    fn results(&self) -> &Results { &self.results }
    fn name(&self) -> &str { "sis" }
}
