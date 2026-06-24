//! RandomNet: random bidirectional contacts, regenerated as edges expire.
//! Separately compiled; depends only on `ssr_core`.

use ssr_core::{Edges, Network, People, Rng};

pub struct RandomNet {
    n_contacts: usize, // total contacts per agent (edges per agent = n_contacts/2)
    dur: f32,          // edge duration in timesteps (0 -> regenerate every step)
    beta: f32,
    edges: Edges,
    rng: Rng,
}

impl RandomNet {
    pub fn new(n_contacts: usize, dur: f32, beta: f32, seed: u64) -> Self {
        Self { n_contacts, dur, beta, edges: Edges::default(), rng: Rng::new(seed) }
    }

    fn end_pairs(&mut self) {
        let e = &mut self.edges;
        for d in e.dur.iter_mut() { *d -= 1.0; }
        let mut i = 0;
        while i < e.dur.len() {
            if e.dur[i] > 0.0 {
                i += 1;
            } else {
                e.p1.swap_remove(i);
                e.p2.swap_remove(i);
                e.beta.swap_remove(i);
                e.dur.swap_remove(i);
            }
        }
    }

    fn add_pairs(&mut self, people: &People) {
        // Only alive agents form contacts (so births join, deaths drop out)
        let alive: Vec<u32> = (0..people.n as u32).filter(|&i| people.alive[i as usize]).collect();
        let n_alive = alive.len();
        let target_edges = (n_alive * self.n_contacts) / 2;
        let current = self.edges.p1.len();
        if current >= target_edges || n_alive == 0 {
            return;
        }
        let needed = target_edges - current;
        // Half-edges per agent (n_contacts/2), scaled to the number still needed
        let total_contacts = n_alive * self.n_contacts;
        let scale = needed as f64 / total_contacts as f64;
        let per_agent = (self.n_contacts as f64 * scale).round() as usize;

        let n_half = per_agent * n_alive;
        let mut source: Vec<u32> = Vec::with_capacity(n_half);
        for &person in &alive {
            for _ in 0..per_agent { source.push(person); }
        }
        let mut target = source.clone();
        self.rng.shuffle(&mut target);

        let e = &mut self.edges;
        e.p1.reserve(n_half);
        e.p2.reserve(n_half);
        e.beta.reserve(n_half);
        e.dur.reserve(n_half);
        for i in 0..source.len() {
            e.p1.push(source[i]);
            e.p2.push(target[i]);
            e.beta.push(self.beta);
            e.dur.push(self.dur);
        }
    }
}

impl Network for RandomNet {
    fn step(&mut self, people: &People) {
        self.end_pairs();
        self.add_pairs(people);
    }
    fn edges(&self) -> &Edges {
        &self.edges
    }
    fn name(&self) -> &str {
        "randomnet"
    }
}
