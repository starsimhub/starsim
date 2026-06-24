//! SIR disease: susceptible -> infected -> recovered (permanent immunity).
//! Separately compiled; depends only on `ssr_core`. Added AFTER ssr_sis/ssr_core
//! were already built, to demonstrate that a new module compiles without
//! recompiling the others.

use ssr_core::{Disease, Network, People, Results, Rng};

pub struct Sir {
    n: usize,
    beta: f64,
    dur_inf: f64,
    dur_inf_std: f64,
    ti: f64,
    susceptible: Vec<bool>,
    infected: Vec<bool>,
    recovered: Vec<bool>,
    ti_recovered: Vec<f32>,
    rng: Rng,
    results: Results,
    new_this_step: usize,
}

impl Sir {
    pub fn new(n: usize, n_steps: usize, beta: f64, init_prev: f64, dur_inf: f64,
               dur_inf_std: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut susceptible = vec![true; n];
        let mut infected = vec![false; n];
        let mut ti_recovered = vec![0.0f32; n];

        let mut initial: Vec<usize> = Vec::new();
        for i in 0..n {
            if rng.rand_f64() < init_prev { initial.push(i); }
        }
        let durs = rng.lognormal_ex(dur_inf, dur_inf_std, initial.len());
        for (j, &i) in initial.iter().enumerate() {
            susceptible[i] = false;
            infected[i] = true;
            ti_recovered[i] = durs[j] as f32;
        }

        let mut results = Results::new();
        for key in ["n_susceptible", "n_infected", "n_recovered", "prevalence", "new_infections", "cum_infections"] {
            results.insert(key.to_string(), vec![0.0; n_steps]);
        }

        Self {
            n, beta, dur_inf, dur_inf_std, ti: 0.0,
            susceptible, infected, recovered: vec![false; n], ti_recovered,
            rng, results, new_this_step: 0,
        }
    }

    fn set_prognoses(&mut self, uids: &[usize]) {
        let durs = self.rng.lognormal_ex(self.dur_inf, self.dur_inf_std, uids.len());
        for (j, &u) in uids.iter().enumerate() {
            self.susceptible[u] = false;
            self.infected[u] = true;
            self.ti_recovered[u] = (self.ti + durs[j]) as f32;
        }
        self.new_this_step = uids.len();
    }
}

impl Disease for Sir {
    fn step_state(&mut self, people: &People) {
        // Infected -> recovered (permanent), never back to susceptible
        for i in 0..self.susceptible.len() {
            if people.alive[i] && self.infected[i] && self.ti_recovered[i] as f64 <= self.ti {
                self.infected[i] = false;
                self.recovered[i] = true;
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
                    if self.rng.rand_f64() < self.beta * e.beta[i] as f64 { targets.push(b); }
                }
                if self.infected[b] && self.susceptible[a] {
                    if self.rng.rand_f64() < self.beta * e.beta[i] as f64 { targets.push(a); }
                }
            }
        }
        targets.sort_unstable();
        targets.dedup();
        self.new_this_step = 0;
        if !targets.is_empty() { self.set_prognoses(&targets); }
    }

    fn update_results(&mut self, ti: usize, people: &People) {
        let (mut s, mut inf, mut r) = (0usize, 0usize, 0usize);
        for i in 0..self.susceptible.len() {
            if !people.alive[i] { continue; }
            if self.susceptible[i] { s += 1; }
            if self.infected[i] { inf += 1; }
            if self.recovered[i] { r += 1; }
        }
        let n_alive = people.n_alive().max(1);
        self.results.get_mut("n_susceptible").unwrap()[ti] = s as f64;
        self.results.get_mut("n_infected").unwrap()[ti] = inf as f64;
        self.results.get_mut("n_recovered").unwrap()[ti] = r as f64;
        self.results.get_mut("prevalence").unwrap()[ti] = inf as f64 / n_alive as f64;
        self.results.get_mut("new_infections").unwrap()[ti] = self.new_this_step as f64;
        self.ti += 1.0;
    }

    fn grow(&mut self, k: usize) {
        self.susceptible.extend(std::iter::repeat(true).take(k));
        self.infected.extend(std::iter::repeat(false).take(k));
        self.recovered.extend(std::iter::repeat(false).take(k));
        self.ti_recovered.extend(std::iter::repeat(0.0f32).take(k));
        self.n += k;
    }

    fn finalize(&mut self) {
        let new = self.results.get("new_infections").unwrap().clone();
        let cum = self.results.get_mut("cum_infections").unwrap();
        let mut acc = 0.0;
        for ti in 0..new.len() { acc += new[ti]; cum[ti] = acc; }
    }

    fn results(&self) -> &Results { &self.results }
    fn name(&self) -> &str { "sir" }
}
