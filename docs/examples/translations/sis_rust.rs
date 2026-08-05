/// Test simulation performance -- Rust version
///
/// Compile and run:
///   rustc -O sis_rust.rs -o sis_rust && ./sis_rust

use std::time::Instant;

// Use f32 for state arrays (matches Julia's F = Float32)
type F = f32;


// ---- RNG (xoshiro256**) ----

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix64 seeding
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

    fn next_u64(&mut self) -> u64 {
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
    fn rand_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller
    fn randn(&mut self) -> f64 {
        loop {
            let u1 = self.rand_f64();
            if u1 > 0.0 {
                let u2 = self.rand_f64();
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    /// Random usize in [0, n)
    fn rand_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle
    fn shuffle(&mut self, v: &mut [usize]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.rand_below(i + 1);
            v.swap(i, j);
        }
    }

    /// Return a random permutation of 0..n
    fn randperm(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        self.shuffle(&mut v);
        v
    }
}


// ---- Simulation types ----

struct Pars {
    n_agents: usize,
    dur: usize,
}

struct RandomNet {
    n_agents: usize,
    n_contacts: usize,
    dur: usize,
    beta: f64,
    p1: Vec<usize>,
    p2: Vec<usize>,
    edge_beta: Vec<F>,
    edge_dur: Vec<F>,
}

impl RandomNet {
    fn new(pars: &Pars, n_contacts: usize, dur: usize, beta: f64) -> Self {
        Self {
            n_agents: pars.n_agents,
            n_contacts,
            dur,
            beta,
            p1: Vec::new(),
            p2: Vec::new(),
            edge_beta: Vec::new(),
            edge_dur: Vec::new(),
        }
    }

    /// Remove expired edges
    fn end_pairs(&mut self) {
        for d in self.edge_dur.iter_mut() {
            *d -= 1.0;
        }
        let mut i = 0;
        while i < self.edge_dur.len() {
            if self.edge_dur[i] > 0.0 {
                i += 1;
            } else {
                self.p1.swap_remove(i);
                self.p2.swap_remove(i);
                self.edge_beta.swap_remove(i);
                self.edge_dur.swap_remove(i);
            }
        }
    }

    /// Generate new random edges
    fn add_pairs(&mut self, rng: &mut Rng) {
        let target_edges = (self.n_agents * self.n_contacts) / 2;
        let current_edges = self.p1.len();
        if current_edges >= target_edges {
            return;
        }
        let needed = target_edges - current_edges;
        let total_contacts = self.n_agents * self.n_contacts;
        let scale = needed as f64 / total_contacts as f64;

        // Build scaled contact counts per agent
        let n_conn: Vec<usize> = (0..self.n_agents)
            .map(|_| (self.n_contacts as f64 * scale).round() as usize)
            .collect();

        // Build source array
        let n_half_edges: usize = n_conn.iter().sum();
        let mut source = vec![0usize; n_half_edges];
        let mut idx = 0;
        for (person_id, &nc) in n_conn.iter().enumerate() {
            for _ in 0..nc {
                source[idx] = person_id;
                idx += 1;
            }
        }

        // Shuffle a copy to get targets
        let mut target = source.clone();
        rng.shuffle(&mut target);

        let n_new = source.len();
        let beta_val = self.beta as F;
        let dur_val = self.dur as F;

        self.p1.reserve(n_new);
        self.p2.reserve(n_new);
        self.edge_beta.reserve(n_new);
        self.edge_dur.reserve(n_new);

        for i in 0..n_new {
            self.p1.push(source[i]);
            self.p2.push(target[i]);
            self.edge_beta.push(beta_val);
            self.edge_dur.push(dur_val);
        }
    }

    fn step(&mut self, rng: &mut Rng) {
        self.end_pairs();
        self.add_pairs(rng);
    }
}


struct SIS {
    n_agents: usize,
    #[allow(dead_code)]
    dur: usize,
    beta: f64,
    #[allow(dead_code)]
    init_prev: f64,
    dur_inf: f64,
    waning: f64,
    imm_boost: f64,
    ti: f64,
    susceptible: Vec<bool>,
    infected: Vec<bool>,
    ti_recovered: Vec<F>,
    immunity: Vec<F>,
    rel_sus: Vec<F>,
    n_susceptible: Vec<F>,
    n_infected: Vec<F>,
    mean_rel_sus: Vec<F>,
}

impl SIS {
    fn new(
        pars: &Pars,
        rng: &mut Rng,
        beta: f64,
        init_prev: f64,
        dur_inf: f64,
        waning: f64,
        imm_boost: f64,
    ) -> Self {
        let n = pars.n_agents;

        let mut susceptible = vec![true; n];
        let mut infected = vec![false; n];
        let mut ti_recovered = vec![0.0 as F; n];
        let immunity = vec![0.0 as F; n];
        let rel_sus = vec![1.0 as F; n];

        // Seed initial infections
        let n_inf = (init_prev * n as f64).round() as usize;
        let perm = rng.randperm(n);
        let dur_samples = Self::lognormal_ex(rng, dur_inf, 1.0, n_inf);
        for (j, &i) in perm[..n_inf].iter().enumerate() {
            susceptible[i] = false;
            infected[i] = true;
            ti_recovered[i] = dur_samples[j] as F;
        }

        let n_susceptible = vec![0.0 as F; pars.dur];
        let n_infected = vec![0.0 as F; pars.dur];
        let mean_rel_sus = vec![0.0 as F; pars.dur];

        Self {
            n_agents: n,
            dur: pars.dur,
            beta,
            init_prev,
            dur_inf,
            waning,
            imm_boost,
            ti: 0.0,
            susceptible,
            infected,
            ti_recovered,
            immunity,
            rel_sus,
            n_susceptible,
            n_infected,
            mean_rel_sus,
        }
    }

    /// Lognormal with specified mean and exponential SD
    fn lognormal_ex(rng: &mut Rng, mean: f64, std: f64, n: usize) -> Vec<f64> {
        let sigma = (1.0 + (std / mean).powi(2)).ln().sqrt();
        let mu = mean.ln() - sigma * sigma / 2.0;
        (0..n).map(|_| (mu + sigma * rng.randn()).exp()).collect()
    }

    /// Progress infectious -> recovered
    fn step_state(&mut self) {
        for i in 0..self.n_agents {
            if self.infected[i] && self.ti_recovered[i] as f64 <= self.ti {
                self.infected[i] = false;
                self.susceptible[i] = true;
            }
        }
    }

    /// Wane immunity and update relative susceptibility
    fn update_immunity(&mut self) {
        let ti = self.ti as usize;
        let waning_factor = 1.0 - self.waning as F;
        let mut sum_rel_sus: f64 = 0.0;
        for i in 0..self.n_agents {
            if self.immunity[i] > 0.0 {
                self.immunity[i] *= waning_factor;
                self.rel_sus[i] = (1.0 - self.immunity[i]).max(0.0);
            }
            sum_rel_sus += self.rel_sus[i] as f64;
        }
        self.mean_rel_sus[ti] = (sum_rel_sus / self.n_agents as f64) as F;
    }

    /// Count susceptible and infected
    fn update_results(&mut self) {
        let ti = self.ti as usize;
        let mut n_sus: usize = 0;
        let mut n_inf: usize = 0;
        for i in 0..self.n_agents {
            if self.susceptible[i] {
                n_sus += 1;
            }
            if self.infected[i] {
                n_inf += 1;
            }
        }
        self.n_susceptible[ti] = n_sus as F;
        self.n_infected[ti] = n_inf as F;
    }

    /// Set prognoses for newly infected agents
    fn set_prognoses(&mut self, rng: &mut Rng, uids: &[usize]) {
        let dur_inf = Self::lognormal_ex(rng, self.dur_inf, 1.0, uids.len());
        for (j, &uid) in uids.iter().enumerate() {
            self.susceptible[uid] = false;
            self.infected[uid] = true;
            self.immunity[uid] += self.imm_boost as F;
            self.ti_recovered[uid] = (self.ti + dur_inf[j]) as F;
        }
    }

    /// Calculate transmission across network edges
    fn infect(&mut self, net: &RandomNet, rng: &mut Rng) {
        let n_edges = net.p1.len();
        let mut targets: Vec<usize> = Vec::new();

        for i in 0..n_edges {
            let a = net.p1[i];
            let b = net.p2[i];

            // p1 infected, p2 susceptible
            if self.infected[a] && self.susceptible[b] {
                let beta = self.beta * net.edge_beta[i] as f64 * self.rel_sus[b] as f64;
                if rng.rand_f64() < beta {
                    targets.push(b);
                }
            }
            // p2 infected, p1 susceptible
            if self.infected[b] && self.susceptible[a] {
                let beta = self.beta * net.edge_beta[i] as f64 * self.rel_sus[a] as f64;
                if rng.rand_f64() < beta {
                    targets.push(a);
                }
            }
        }

        // Deduplicate
        targets.sort_unstable();
        targets.dedup();

        if !targets.is_empty() {
            self.set_prognoses(rng, &targets);
        }
    }

    fn step(&mut self, net: &RandomNet, rng: &mut Rng) {
        self.step_state();
        self.update_immunity();
        self.infect(net, rng);
        self.update_results();
        self.ti += 1.0;
    }
}


struct Sim {
    #[allow(dead_code)]
    n_agents: usize,
    dur: usize,
    disease: SIS,
    network: RandomNet,
}

impl Sim {
    fn new(pars: &Pars, network: RandomNet, disease: SIS) -> Self {
        Self {
            n_agents: pars.n_agents,
            dur: pars.dur,
            disease,
            network,
        }
    }

    fn step(&mut self, rng: &mut Rng) {
        self.network.step(rng);
        self.disease.step(&self.network, rng);
    }

    fn run(&mut self, rng: &mut Rng) {
        for _ in 0..self.dur {
            self.step(rng);
        }
    }
}


// ---- Tests ----

fn test_inf(sim: &Sim, n_agents: usize) {
    let n_inf = &sim.disease.n_infected;
    let inf0 = n_inf[0];
    let infm = n_inf.iter().cloned().fold(F::NEG_INFINITY, F::max);
    let inf1 = n_inf[n_inf.len() - 1];

    let tests = [
        (format!("Initial infections are nonzero: {}", inf0), inf0 > 0.005 * n_agents as F),
        (format!("Initial infections start low: {}", inf0), inf0 < 0.05 * n_agents as F),
        (format!("Infections peak high: {}", infm), infm > 0.5 * n_agents as F),
        (format!("Infections stabilize: {}", inf1), inf1 < infm),
    ];

    let mut all_pass = true;
    for (msg, pass) in &tests {
        if *pass {
            println!("✓ {}", msg);
        } else {
            println!("× {}", msg);
            all_pass = false;
        }
    }
    assert!(all_pass);
}


// ---- Main ----

fn main() {
    let pars = Pars {
        n_agents: 100_000,
        dur: 100,
    };

    let mut rng = Rng::new(12345);

    // Warmup run
    let net = RandomNet::new(&pars, 10, 0, 1.0);
    let dis = SIS::new(&pars, &mut rng, 0.05, 0.01, 10.0, 0.05, 1.0);
    let mut sim = Sim::new(&pars, net, dis);
    let t0 = Instant::now();
    sim.run(&mut rng);
    println!("Warmup: {:.3}s", t0.elapsed().as_secs_f64());

    // Timed run
    let net = RandomNet::new(&pars, 10, 0, 1.0);
    let dis = SIS::new(&pars, &mut rng, 0.05, 0.01, 10.0, 0.05, 1.0);
    let mut sim = Sim::new(&pars, net, dis);
    let t0 = Instant::now();
    sim.run(&mut rng);
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "Time for SIS-Rust, n_agents={}, dur={}: {:.3}s",
        pars.n_agents, pars.dur, elapsed
    );

    test_inf(&sim, pars.n_agents);
}
