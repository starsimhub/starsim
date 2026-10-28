//! The passes over the agents and over a network's edges of `starsim/solvers/backends/passes.py`: the same arguments and results, and the same
//! chunks, which several threads take when there is more than one (see `par()`). Lists are kept in order and counts are whole numbers,
//! so the results do not depend on the number of threads. Each takes the arrays as slices (see `py.rs`).

use crate::draws;
use std::marker::PhantomData;
use std::ops::AddAssign;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};


//%% Threads and chunks

static THREADS: AtomicUsize = AtomicUsize::new(1); // As many as Numba has (set by the bindings: see bind::chunks())
static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// Take this many threads for the next pass of more than one chunk
pub fn set_threads(n: usize) {
    THREADS.store(n.max(1), Ordering::Relaxed);
}

/// What the threads are given to do: the work of the latest pass (its number), and how many are still at it
struct Job {
    pass: u64,
    work: Option<&'static (dyn Fn() + Sync)>,
    busy: usize,
    failed: bool,
}

/// The threads that take the chunks of a pass beside the calling thread: started the first time a pass has more than one chunk (and again if
/// the number changes, or in a forked process, which has none of them); between passes they wait without spinning, so they take no time from
/// Python's (as a pool that spins would)
struct Pool {
    n: usize,
    pid: u32,
    shared: Arc<(Mutex<Job>, Condvar, Condvar)>, // The job, a signal to start, and a signal that all are done
}

impl Pool {
    fn new(n: usize) -> Pool {
        let shared = Arc::new((Mutex::new(Job { pass: 0, work: None, busy: 0, failed: false }), Condvar::new(), Condvar::new()));
        for _ in 1..n {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let (job, start, done) = &*sh;
                let mut seen = 0;
                loop {
                    let work = {
                        let mut j = job.lock().unwrap();
                        while j.pass == seen {
                            j = start.wait(j).unwrap();
                        }
                        seen = j.pass;
                        match j.work {
                            Some(w) => w,
                            None => return, // The pool is dropped
                        }
                    };
                    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_ok();
                    let mut j = job.lock().unwrap();
                    j.failed |= !ok;
                    j.busy -= 1;
                    if j.busy == 0 {
                        done.notify_one();
                    }
                }
            });
        }
        Pool { n, pid: std::process::id(), shared }
    }

    /// Run work on every thread, this one too, and wait for all
    fn run(&self, work: &(dyn Fn() + Sync)) {
        let (job, start, done) = &*self.shared;
        let w: &'static (dyn Fn() + Sync) = unsafe { std::mem::transmute(work) }; // The threads drop it before this returns
        {
            let mut j = job.lock().unwrap();
            *j = Job { pass: j.pass + 1, work: Some(w), busy: self.n - 1, failed: false };
        }
        start.notify_all();
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)); // Waited on before failing, since the others use the work
        let mut j = job.lock().unwrap();
        while j.busy > 0 {
            j = done.wait(j).unwrap();
        }
        j.work = None;
        let failed = j.failed;
        drop(j);
        if let Err(e) = mine {
            std::panic::resume_unwind(e);
        }
        assert!(!failed, "a thread of the Rust kernels failed");
    }
}

impl Drop for Pool {
    /// The threads end
    fn drop(&mut self) {
        let (job, start, _) = &*self.shared;
        let mut j = job.lock().unwrap();
        j.pass += 1;
        j.work = None;
        start.notify_all();
    }
}

/// The result of f for each chunk, in order: on the threads if there is more than one chunk (as `nb.prange`), else in this thread
pub fn par<R: Send>(nchunks: usize, f: impl Fn(usize) -> R + Send + Sync + Copy) -> Vec<R> {
    let threads = THREADS.load(Ordering::Relaxed);
    if nchunks <= 1 || threads <= 1 {
        return (0..nchunks).map(f).collect();
    }
    let mut out: Vec<Option<R>> = (0..nchunks).map(|_| None).collect();
    let next = AtomicUsize::new(0);
    {
        let o = Shared::new(&mut out);
        let work = || {
            let g = f; // A copy of its own on each thread, so what it captures is known not to change as it writes, and is kept in registers
            loop {
                let c = next.fetch_add(1, Ordering::Relaxed); // The next chunk not yet taken
                if c >= nchunks {
                    break;
                }
                unsafe { *o.at(c) = Some(g(c)) };
            }
        };
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner()); // As it was, if a pass failed
        let pid = std::process::id();
        if pool.as_ref().is_none_or(|p| p.n != threads || p.pid != pid) {
            if let Some(old) = pool.take() {
                if old.pid != pid {
                    std::mem::forget(old); // Its threads are in the parent
                }
            }
            *pool = Some(Pool::new(threads));
        }
        pool.as_ref().unwrap().run(&work);
    }
    out.into_iter().map(|r| r.unwrap()).collect()
}

/// The agents that chunk c of n chunks covers (`passes.bounds()`)
#[inline(always)]
pub fn bounds(c: usize, n: usize, nchunks: usize) -> (usize, usize) {
    let size = n.div_ceil(nchunks);
    let a = (c*size).min(n);
    (a, (a + size).min(n))
}

/// An array that the chunks write on several threads, each in parts of its own
pub struct Shared<'a, T>(*mut T, usize, PhantomData<&'a mut [T]>);
unsafe impl<T: Send> Send for Shared<'_, T> {}
unsafe impl<T: Send> Sync for Shared<'_, T> {}
impl<T> Clone for Shared<'_, T> { fn clone(&self) -> Self { *self } }
impl<T> Copy for Shared<'_, T> {}

impl<'a, T> Shared<'a, T> {
    pub fn new(s: &'a mut [T]) -> Self {
        Shared(s.as_mut_ptr(), s.len(), PhantomData)
    }

    /// Items a to b, which no other chunk may take
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn part(&self, a: usize, b: usize) -> &mut [T] {
        assert!(a <= b && b <= self.1);
        unsafe { std::slice::from_raw_parts_mut(self.0.add(a), b - a) }
    }

    /// Item i, which no other chunk may take
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    pub unsafe fn at(&self, i: usize) -> &mut T {
        assert!(i < self.1);
        unsafe { &mut *self.0.add(i) }
    }
}

/// A vector of n items, each written before it is read (as np.empty(): zeroing it would take as long as some passes)
fn empty<T: Copy>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    unsafe { v.set_len(n) };
    v
}

/// Where each chunk's part of a list starts, from how many each found, and (last) the length of the list
fn starts(found: &[usize]) -> Vec<usize> {
    let mut out = vec![0; found.len() + 1];
    for (c, &f) in found.iter().enumerate() {
        out[c + 1] = out[c] + f;
    }
    out
}

/// A list made in chunks, each found[c] long, one after another: fill(c, part) writes chunk c's part
fn gather<T: Copy + Send>(found: &[usize], fill: impl Fn(usize, &mut [T]) + Send + Sync + Copy) -> Vec<T> {
    let st = &starts(found);
    let mut out = empty(st[found.len()]);
    if !out.is_empty() {
        let o = Shared::new(&mut out);
        par(found.len(), move |c| if found[c] > 0 { fill(c, unsafe { o.part(st[c], st[c + 1]) }) });
    }
    out
}

/// Move the part of out that each chunk wrote (found[c] from its start) to follow the one before; the length of the list (`passes.close_gaps()`)
fn close_gaps<T: Copy>(out: &mut [T], found: &[usize], n: usize) -> usize {
    let mut p = found[0];
    for (c, &f) in found.iter().enumerate().skip(1) {
        let a = bounds(c, n, found.len()).0;
        out.copy_within(a..a + f, p);
        p += f;
    }
    p
}

/// Each chunk of n items writes its list at the start of its own part of out (work(a, b, part) gives how many), and the lists are joined: their length
fn in_place<T: Copy + Send>(out: &mut [T], n: usize, nchunks: usize, work: impl Fn(usize, usize, &mut [T]) -> usize + Send + Sync + Copy) -> usize {
    let found = {
        let o = Shared::new(out);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            work(a, b, unsafe { o.part(a, b) })
        })
    };
    close_gaps(out, &found, n)
}

/// The counts of each chunk, added up
fn sum(per: Vec<Vec<i64>>, width: usize) -> Vec<i64> {
    let mut out = vec![0i64; width];
    for p in per {
        for (o, x) in out.iter_mut().zip(p) {
            *o += x;
        }
    }
    out
}


//%% Types

/// A whole number that indexes an array: a group, a level, a slot
pub trait Ix: Copy + Send + Sync {
    fn ix(self) -> usize;
    fn int(self) -> i64;
}
macro_rules! ix { ($($t:ty),+) => { $( impl Ix for $t { #[inline(always)] fn ix(self) -> usize { self as usize } #[inline(always)] fn int(self) -> i64 { self as i64 } } )+ } }
ix!(i8, i16, i32, i64);

/// A number as Numba compares it: as a float64 (floats, and whole numbers with floats), or as an int64 with a step
pub trait Num: Copy + Send + Sync {
    type Step: Copy + Send + Sync;
    fn f(self) -> f64;
    /// Step ti in this type, to compare with (see le())
    fn step(ti: i64) -> Self::Step;
    /// Whether it is at most a step, as Numba compares it with ti
    fn le(self, s: Self::Step) -> bool;
}
macro_rules! num_int { ($($t:ty),+) => { $( impl Num for $t {
    type Step = i64;
    #[inline(always)] fn f(self) -> f64 { self as f64 }
    #[inline(always)] fn step(ti: i64) -> i64 { ti }
    #[inline(always)] fn le(self, s: i64) -> bool { (self as i64) <= s }
} )+ } }
num_int!(i16, i32, i64);
impl Num for f64 {
    type Step = f64;
    #[inline(always)] fn f(self) -> f64 { self }
    #[inline(always)] fn step(ti: i64) -> f64 { ti as f64 }
    #[inline(always)] fn le(self, s: f64) -> bool { self <= s }
}
impl Num for f32 {
    type Step = f32;
    #[inline(always)] fn f(self) -> f64 { self as f64 }
    /// The largest float32 at most ti (as a float64): a float32 is at most it if and only if it is at most ti, so they are compared as float32s, which is faster
    fn step(ti: i64) -> f32 {
        let (t, c) = (ti as f64, ti as f64 as f32);
        if c as f64 > t { c.next_down() } else { c }
    }
    #[inline(always)] fn le(self, s: f32) -> bool { self <= s }
}

/// A product in the type Numba gives it: float32 only if both are
pub trait Times<B>: Copy + Send + Sync {
    type Out: Copy + Num;
    fn times(self, b: B) -> Self::Out;
}
impl Times<f32> for f32 { type Out = f32; #[inline(always)] fn times(self, b: f32) -> f32 { self*b } }
impl Times<f64> for f32 { type Out = f64; #[inline(always)] fn times(self, b: f64) -> f64 { self as f64*b } }
impl Times<f32> for f64 { type Out = f64; #[inline(always)] fn times(self, b: f32) -> f64 { self*b as f64 } }
impl Times<f64> for f64 { type Out = f64; #[inline(always)] fn times(self, b: f64) -> f64 { self*b } }

/// A value whose whole part is a level, or none if it is NaN (`int()` in Numba)
pub trait Level: Copy {
    fn level(self) -> Option<i32>;
}
macro_rules! level_int { ($($t:ty),+) => { $( impl Level for $t { #[inline(always)] fn level(self) -> Option<i32> { Some(self as i32) } } )+ } }
level_int!(i8, i16, i32, i64, u8);
impl Level for bool { fn level(self) -> Option<i32> { Some(self as i32) } }
macro_rules! level_float { ($($t:ty),+) => { $( impl Level for $t {
    #[inline(always)]
    fn level(self) -> Option<i32> {
        let k = if self.abs() < 9223372036854775808.0 { unsafe { self.to_int_unchecked::<i64>() } } else { i64::MIN }; // As Numba's int() on x86 (i64::MIN when out of range), without Rust's checks for saturating
        if self == self { Some(k as i32) } else { None } // Worked out before the test, so it is a select, not a branch
    }
} )+ } }
level_float!(f32, f64);


//%% Booleans, eight at a time

/// Booleans as bytes
#[inline(always)]
pub fn bytes(raw: &[bool]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u8, raw.len()) } // A bool is a byte that is 0 or 1
}

#[inline(always)]
fn word(c: &[u8]) -> u64 {
    u64::from_ne_bytes(c.try_into().unwrap())
}

/// The sum of the bytes of a word, each a count up to 255
#[inline(always)]
fn byte_sum(w: u64) -> usize {
    let pairs = (w & 0x00ff00ff00ff00ff) + ((w >> 8) & 0x00ff00ff00ff00ff); // Four counts of up to 510
    (pairs.wrapping_mul(0x0001000100010001) >> 48) as usize
}

/// The number of true values, eight at a time: each byte is 0 or 1, so words are added as they are, each byte a count, and the bytes summed every 255 words (`passes.true_count()`)
pub fn true_count(raw: &[u8]) -> usize {
    let mut m = 0;
    for block in raw.chunks(8*255) {
        let words = block.chunks_exact(8);
        let rest = words.remainder();
        m += byte_sum(words.map(word).fold(0, |s, w| s + w)) + rest.iter().map(|&b| b as usize).sum::<usize>();
    }
    m
}

/// The number for whom two boolean arrays are both true, eight at a time (`passes.both_count()`), as true_count()
pub fn both_count(x: &[u8], y: &[u8]) -> usize {
    let mut m = 0;
    for (bx, by) in x.chunks(8*255).zip(y.chunks(8*255)) {
        let (wx, wy) = (bx.chunks_exact(8), by.chunks_exact(8));
        let (rx, ry) = (wx.remainder(), wy.remainder());
        m += byte_sum(wx.zip(wy).map(|(a, b)| word(a) & word(b)).fold(0, |s, w| s + w)) + rx.iter().zip(ry).map(|(&a, &b)| (a & b) as usize).sum::<usize>();
    }
    m
}

/// Write the indices of the true values of raw from a to b into out, from position p, eight at a time where they are all false; the position after (`passes.list_true()`)
pub fn list_true(raw: &[u8], a: usize, b: usize, out: &mut [i64], mut p: usize) -> usize {
    let chunks = raw[a..b].chunks_exact(8);
    let rest = chunks.remainder();
    for (w, c) in chunks.enumerate() {
        if word(c) != 0 {
            for (j, &x) in c.iter().enumerate() {
                if x != 0 {
                    out[p] = (a + 8*w + j) as i64;
                    p += 1;
                }
            }
        }
    }
    let r0 = b - rest.len();
    for (j, &x) in rest.iter().enumerate() {
        if x != 0 {
            out[p] = (r0 + j) as i64;
            p += 1;
        }
    }
    p
}

/// `max(x, 0.0)` as Python and Numba take it (so a NaN stays NaN)
#[inline(always)]
fn max0(x: f64) -> f64 {
    if 0.0 > x { 0.0 } else { x }
}

/// The chance of infection in a step at a rate (`-np.expm1(-max(rate, 0.0)*dt)`)
#[inline(always)]
fn chance(rate: f64, dt: f64) -> f64 {
    -(-max0(rate)*dt).exp_m1()
}


//%% Levels (ss.dimensions), and who could be infected (ss.solvers.Agents)

/// The level of each agent in a dimension, as whole numbers (0 for an agent with none) (`passes.levels()`)
pub fn levels<T: Level>(raw: &[T]) -> Vec<i32> {
    raw.iter().map(|x| x.level().unwrap_or(0)).collect()
}

/// The agents whose value of a binned property is no longer in the bin of their level (NaN is in the last), in order (`passes.crossed()`)
pub fn crossed<T: Num, L: Ix>(raw: &[T], level: &[L], lo: &[f64], hi: &[f64]) -> Vec<i64> {
    let last = lo.len() - 1;
    let mut out = empty::<i64>(raw.len());
    let mut m = 0;
    for (i, (&v, &k)) in raw.iter().zip(level).enumerate() {
        let (k, x) = (k.ix(), v.f());
        if !((x >= lo[k] && x < hi[k]) || (x != x && k == last)) {
            out[m] = i as i64;
            m += 1;
        }
    }
    out.truncate(m);
    out
}

/// The agents who could be infected this step, and the chance for each (`passes.infection_probs()`)
#[allow(clippy::too_many_arguments)]
pub fn infection_probs<G: Ix, R: Num>(susceptible: &[bool], auids: &[i64], groups: &[G], rate: &[f64], rel_sus: &[R], state_sus: f64, adjust: &[f64], mult: f64, dt: f64) -> (Vec<i64>, Vec<f64>) {
    let usual: Vec<f64> = (0..rate.len()).map(|g| chance(rate[g]*1.0*state_sus*adjust[g]*mult, dt)).collect(); // For someone of average susceptibility, which is most people
    let each = (rel_sus.len() > 1) as usize; // One for each agent, or one for all (see bind::bcast())
    let mut uids = empty::<i64>(auids.len());
    let mut probs = empty::<f64>(auids.len());
    let mut n = 0;
    for &uid in auids { // Written every time and kept only if n advances
        let u = uid as usize;
        let g = groups[u].ix();
        let r = rel_sus[u*each].f();
        let prob = if r == 1.0 { usual[g] } else { chance(rate[g]*r*state_sus*adjust[g]*mult, dt) };
        uids[n] = uid;
        probs[n] = prob;
        n += (susceptible[u] & (prob > 0.0)) as usize;
    }
    uids.truncate(n);
    probs.truncate(n);
    (uids, probs)
}


//%% Counts by group, those in a state, and who is infected (ss.solvers.Agents)

/// For agents a to b: the number in each state by group, and (in the last row) the number alive, in four running counts (`passes.census_chunk()`)
fn census_chunk<G: Ix>(members: &[&[bool]], alive: &[bool], groups: &[G], ng: usize, a: usize, b: usize) -> Vec<i64> {
    let ns = members.len();
    let w = (ns + 1)*ng;
    let whole = groups.is_empty();
    let mut counts = vec![0i64; 4*w];
    for uid in a..b {
        if !alive[uid] { // Nobody who has died is in a state
            continue;
        }
        let g = if whole { 0 } else { groups[uid].ix() };
        let r = (uid & 3)*w; // Four running counts, taken in turn, so an addition need not wait on the last when neighbors are in the same group
        counts[r + ns*ng + g] += 1;
        let mut k = 0;
        let mut last = 0;
        for (s, m) in members.iter().enumerate() {
            if m[uid] {
                k += 1;
                last = s;
            }
        }
        if k == 1 { // The usual case: in one state
            counts[r + last*ng + g] += 1;
        } else if k > 1 {
            for (s, m) in members.iter().enumerate() {
                if m[uid] {
                    counts[r + s*ng + g] += 1;
                }
            }
        }
    }
    counts
}

/// The number alive in each state by group (and everyone alive, in the last row), the agents in the states listed, and where each state's start (`passes.census()`)
pub fn census<G: Ix>(members: &[&[bool]], alive: &[bool], groups: &[G], ng: usize, want: &[i64], n_want: usize, nchunks: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (ns, n) = (members.len(), alive.len());
    let w = (ns + 1)*ng;
    let counts = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        census_chunk(members, alive, groups, ng, a, b)
    });
    let nw = n_want.max(1);
    let mut found = vec![0usize; nchunks*nw]; // The number in each state listed, in each chunk
    for s in 0..ns {
        if want[s] >= 0 {
            for (c, cs) in counts.iter().enumerate() {
                found[c*nw + want[s] as usize] = (0..4).map(|r| cs[r*w + s*ng..r*w + (s + 1)*ng].iter().sum::<i64>()).sum::<i64>() as usize;
            }
        }
    }
    let mut start = vec![0usize; nchunks*nw]; // Where each chunk's agents in each state go
    let mut offsets = vec![0i64; n_want + 1];
    let mut pos = 0;
    for k in 0..n_want {
        offsets[k] = pos as i64;
        for c in 0..nchunks {
            start[c*nw + k] = pos;
            pos += found[c*nw + k];
        }
    }
    offsets[n_want] = pos as i64;
    let (found, start) = (&found, &start);
    let mut out = empty::<i64>(pos);
    if pos > 0 {
        let o = Shared::new(&mut out);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            for s in 0..ns {
                if want[s] >= 0 {
                    let i = c*nw + want[s] as usize;
                    list_true(bytes(members[s]), a, b, unsafe { o.part(start[i], start[i] + found[i]) }, 0);
                }
            }
        });
    }
    let mut total = vec![0i64; w];
    for cs in &counts {
        for r in 0..4 {
            for (t, x) in total.iter_mut().zip(&cs[r*w..(r + 1)*w]) {
                *t += x;
            }
        }
    }
    (total, out, offsets)
}

/// The indices of the true values of an array (`passes.flatnonzero()`)
pub fn flatnonzero(raw: &[bool], nchunks: usize) -> Vec<i64> {
    let (raw, n) = (bytes(raw), raw.len());
    let found = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        true_count(&raw[a..b])
    });
    gather(&found, move |c, part| {
        let (a, b) = bounds(c, n, nchunks);
        list_true(raw, a, b, part, 0);
    })
}

/// The agents infected this step under homogeneous mixing, each drawn from their uniform for the call, in order (`passes.infect()`)
#[allow(clippy::too_many_arguments)]
pub fn infect<G: Ix, R: Num, L: Ix>(susceptible: &[bool], groups: &[G], rate: &[f64], rel_sus: &[R], state_sus: f64, adjust: &[f64], mult: f64, dt: f64, slots: &[L],
                                     key: u64, shift: u64, scale: f64, nchunks: usize) -> Vec<i64> {
    let n = susceptible.len();
    let usual = &(0..rate.len()).map(|g| chance(rate[g]*1.0*state_sus*adjust[g]*mult, dt)).collect::<Vec<f64>>();
    let each = (rel_sus.len() > 1) as usize; // As infection_probs()
    let by_uid = slots.is_empty(); // The slots are the UIDs
    let mut hit = empty::<u8>(n);
    let found = {
        let h = Shared::new(&mut hit);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            let (sus, gs, rs) = (&susceptible[a..b], &groups[a..b], if each == 1 { &rel_sus[a..b] } else { rel_sus }); // Slices, as passes.infect_chunk()
            let sl = if by_uid { &slots[..0] } else { &slots[a..b] };
            let mut m = 0;
            for (i, ((x, &su), &g)) in unsafe { h.part(a, b) }.iter_mut().zip(sus).zip(gs).enumerate() {
                let u = a + i;
                let mut got = false;
                if su {
                    let g = g.ix();
                    let r = rs[i*each].f();
                    let prob = if r == 1.0 { usual[g] } else { chance(rate[g]*r*state_sus*adjust[g]*mult, dt) };
                    if prob > 0.0 {
                        got = draws::crn_uniform(key, if by_uid { u as u64 } else { sl[i].int() as u64 }, shift, scale) < prob;
                    }
                }
                *x = got as u8;
                m += got as usize;
            }
            m
        })
    };
    let hit = &hit;
    gather(&found, move |c, part| {
        let (a, b) = bounds(c, n, nchunks);
        list_true(hit, a, b, part, 0);
    })
}


//%% For people: the passes over the active agents at the end of each step

/// The active agents who have died or left by step ti, in order (`passes.due()`)
pub fn due<T: Num>(auids: &[i64], ti_dead: &[T], ti_removed: &[T], ti: i64, nchunks: usize) -> Vec<i64> {
    let st = T::step(ti);
    let n = auids.len();
    let mut hit = empty::<bool>(n);
    let found = {
        let h = Shared::new(&mut hit);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            let mut m = 0;
            for (x, &uid) in unsafe { h.part(a, b) }.iter_mut().zip(&auids[a..b]) {
                let u = uid as usize;
                let got = ti_dead[u].le(st) | ti_removed[u].le(st);
                *x = got;
                m += got as usize;
            }
            m
        })
    };
    let hit = &hit;
    gather(&found, move |c, part| {
        let (a, b) = bounds(c, n, nchunks);
        let mut p = 0;
        for i in a..b {
            if hit[i] {
                part[p] = auids[i];
                p += 1;
            }
        }
    })
}

/// The number of active agents in each state, then the number who died and who left by step ti (`passes.count_people()`)
pub fn count_people<T: Num>(auids: &[i64], states: &[&[bool]], alive: &[bool], ti_dead: &[T], ti_removed: &[T], ti: i64, nchunks: usize) -> Vec<i64> {
    let st = T::step(ti);
    let (n, ns) = (auids.len(), states.len());
    let per = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        let us = &auids[a..b];
        let mut out: Vec<i64> = states.iter().map(|x| us.iter().map(|&u| x[u as usize] as i64).sum()).collect(); // A pass for each state, which is quicker than a loop over them for each agent
        let (mut dead, mut removed) = (0, 0);
        for &uid in us {
            let u = uid as usize;
            if !alive[u] {
                dead += ti_dead[u].le(st) as i64;
                removed += ti_removed[u].le(st) as i64;
            }
        }
        out.extend([dead, removed]);
        out
    });
    sum(per, ns + 2)
}

/// Take the agents no longer alive out of the active agents (those alive go to kept, in order), and age those who are: the number alive, and the others (`passes.keep_alive()`)
pub fn keep_alive<A: Copy + AddAssign + Send + Sync>(auids: &[i64], alive: &[bool], age: &mut [A], dt: A, aging: bool, kept: &mut [i64], nchunks: usize) -> (i64, Vec<i64>) {
    let n = auids.len();
    let found = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        auids[a..b].iter().map(|&u| alive[u as usize] as usize).sum::<usize>()
    });
    let st = &starts(&found);
    let n_kept = st[nchunks];
    let mut gone = empty::<i64>(n - n_kept);
    let (ages, kp, gn) = (Shared::new(age), Shared::new(kept), Shared::new(&mut gone));
    par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        let k = unsafe { kp.part(st[c], st[c + 1]) };
        let g = unsafe { gn.part(a - st[c], b - st[c + 1]) };
        let (mut i, mut j) = (0, 0);
        for &uid in &auids[a..b] {
            let u = uid as usize;
            if alive[u] {
                k[i] = uid;
                i += 1;
                if aging {
                    unsafe { *ages.at(u) += dt };
                }
            } else {
                g[j] = uid;
                j += 1;
            }
        }
    });
    (n_kept as i64, gone)
}

/// The agents alive who have died or left by step ti (by UID), and the number alive (`passes.due_alive()`)
pub fn due_alive<T: Num>(alive: &[bool], ti_dead: &[T], ti_removed: &[T], ti: i64, nchunks: usize) -> (Vec<i64>, i64) {
    let st = T::step(ti);
    let n = alive.len();
    let al = bytes(alive);
    let mut hit = empty::<u8>(n);
    let per = {
        let h = Shared::new(&mut hit);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            let hs = unsafe { h.part(a, b) };
            for (((h, &y), &d), &r) in hs.iter_mut().zip(&al[a..b]).zip(&ti_dead[a..b]).zip(&ti_removed[a..b]) { // Without branches, so it is vectorized
                *h = (d.le(st) as u8 | r.le(st) as u8) & y;
            }
            (true_count(hs), true_count(&al[a..b])) // Counted after, eight at a time, which is quicker than adding each byte to a count as it goes
        })
    };
    let found: Vec<usize> = per.iter().map(|p| p.0).collect();
    let hit = &hit;
    let out = gather(&found, move |c, part| {
        let (a, b) = bounds(c, n, nchunks);
        list_true(hit, a, b, part, 0);
    });
    (out, per.iter().map(|p| p.1 as i64).sum())
}

/// The number of agents alive in each state, then the number alive (`passes.count_alive()`)
pub fn count_alive(states: &[&[bool]], alive: &[bool], nchunks: usize) -> Vec<i64> {
    let (n, ns) = (alive.len(), states.len());
    let al = bytes(alive);
    let per = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        let mut out: Vec<i64> = states.iter().map(|s| both_count(&bytes(s)[a..b], &al[a..b]) as i64).collect();
        out.push(true_count(&al[a..b]) as i64);
        out
    });
    sum(per, ns + 1)
}

/// List the agents alive, and age them by dt, if there are as many as expected; the number alive, or -1 having changed nothing (`passes.keep_raw()`)
pub fn keep_raw<A: Copy + AddAssign + Send + Sync>(alive: &[bool], age: &mut [A], dt: A, aging: bool, kept: &mut [i64], expected: i64, nchunks: usize) -> i64 {
    let n = alive.len();
    let found = par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        true_count(&bytes(alive)[a..b])
    });
    let st = &starts(&found);
    if st[nchunks] as i64 != expected { // Not the agents expected: nothing is changed
        return -1;
    }
    let (ages, kp) = (Shared::new(age), Shared::new(kept));
    par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        let k = unsafe { kp.part(st[c], st[c + 1]) };
        let mut p = 0;
        if aging {
            let ag = unsafe { ages.part(a, b) };
            for (i, &al) in alive[a..b].iter().enumerate() {
                if al {
                    k[p] = (a + i) as i64;
                    p += 1;
                    ag[i] += dt;
                }
            }
        } else {
            for (i, &al) in alive[a..b].iter().enumerate() {
                if al {
                    k[p] = (a + i) as i64;
                    p += 1;
                }
            }
        }
    });
    st[nchunks] as i64
}


//%% New random edges (ss.RandomNet)

/// The two ends of new random edges: each agent the first end once for each of its stubs, and the other end drawn at random (`passes.random_edges()`)
pub fn random_edges(inds: &[i64], starts: &[i64], seed: i64, by_agent: bool, source: &mut [i64], target: &mut [i64], nchunks: usize) {
    let ni = inds.len();
    {
        let src = Shared::new(source);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, ni, nchunks);
            let s0 = starts[a] as usize;
            let part = unsafe { src.part(s0, starts[b] as usize) };
            let mut j = 0;
            for (i, &u) in inds[a..b].iter().enumerate() {
                for _ in starts[a + i]..starts[a + i + 1] { // A loop, not fill(), which is slower for the few stubs of each agent
                    part[j] = u;
                    j += 1;
                }
            }
        });
    }
    let key = draws::key(seed, 0);
    let pool: &[i64] = if by_agent { inds } else { source };
    let np = pool.len() as i64;
    let n = target.len();
    let trg = Shared::new(target);
    par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        for (j, t) in unsafe { trg.part(a, b) }.iter_mut().enumerate() {
            let r = unsafe { (draws::crn_uniform(key, (a + j) as u64, 11, draws::INV53)*np as f64).to_int_unchecked::<i64>() }.min(np - 1); // Uniform on 0 to n-1; in (0, n), so the float needs no checks to become an integer (as Numba's)
            *t = pool[r as usize];
        }
    });
}


//%% The edges that can transmit, that end, or that touch someone removed

/// The edges from someone infectious to someone susceptible, p1 to p2 in out0 and (if two_way) p2 to p1 in out1, in order: how many of each (`passes.candidates()`)
#[allow(clippy::too_many_arguments)]
pub fn candidates<T: Num, S: Num>(p1: &[i64], p2: &[i64], rel_trans: &[T], rel_sus: &[S], two_way: bool, flags: &mut [u8], out0: &mut [i64], out1: &mut [i64], nchunks: usize) -> (usize, usize) {
    let flag = |u: usize| (rel_trans[u].f() != 0.0) as u8 + 2*(rel_sus[u].f() != 0.0) as u8;
    let na = flags.len();
    {
        let fl = Shared::new(flags);
        par(nchunks, move |c| {
            let (a, b) = bounds(c, na, nchunks);
            for ((f, t), s) in unsafe { fl.part(a, b) }.iter_mut().zip(&rel_trans[a..b]).zip(&rel_sus[a..b]) {
                *f = (t.f() != 0.0) as u8 + 2*(s.f() != 0.0) as u8;
            }
        });
    }
    let flags: &[u8] = flags;
    let by_flags = !flags.is_empty(); // A byte for each agent, which the edges read more quickly than two numbers each, unless there are few edges
    let n = p1.len();
    let found = {
        let (o0, o1) = (Shared::new(out0), Shared::new(out1));
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            let q0 = unsafe { o0.part(a, b) };
            let q1 = if two_way { unsafe { o1.part(a, b) } } else { &mut [] };
            let (mut m0, mut m1) = (0, 0);
            for (i, (&u, &v)) in p1[a..b].iter().zip(&p2[a..b]).enumerate() {
                let (u, v) = (u as usize, v as usize);
                let (f1, f2) = if by_flags { (flags[u], flags[v]) } else { (flag(u), flag(v)) };
                unsafe { *q0.get_unchecked_mut(m0) = (a + i) as i64 }; // Branchless: written every time, kept only if m0 advances (so m0 <= i, within the part)
                m0 += ((f1 & 1) & (f2 >> 1)) as usize;
                if two_way {
                    unsafe { *q1.get_unchecked_mut(m1) = (a + i) as i64 };
                    m1 += ((f2 & 1) & (f1 >> 1)) as usize;
                }
            }
            (m0, m1)
        })
    };
    let f0: Vec<usize> = found.iter().map(|f| f.0).collect();
    let f1: Vec<usize> = found.iter().map(|f| f.1).collect();
    (close_gaps(out0, &f0, n), if two_way { close_gaps(out1, &f1, n) } else { 0 })
}

/// The edges with a chance of transmission above 0 (`rel_trans[src]·rel_sus[trg]·beta`, in Numba's types), in order: how many, written from the start of out (`passes.can_transmit()`)
pub fn can_transmit<T: Times<S>, S: Copy + Send + Sync, B: Copy + Send + Sync>(src: &[i64], trg: &[i64], rel_trans: &[T], rel_sus: &[S], beta: &[B], out: &mut [i64], nchunks: usize) -> usize
where T::Out: Times<B> {
    let n = src.len();
    let each = beta.len() == n; // One for each edge, or one for all
    in_place(out, n, nchunks, move |a, b, o| {
        let bs = if each { &beta[a..b] } else { beta };
        let step = (bs.len() == b - a) as usize;
        let mut m = 0;
        for i in 0..b - a {
            let x = rel_trans[src[a + i] as usize].times(rel_sus[trg[a + i] as usize]).times(bs[i*step]);
            o[m] = (a + i) as i64; // Written every time (branchless) and kept only if m advances
            m += (x.f() > 0.0) as usize; // With no fastmath, so a NaN is no transmission
        }
        m
    })
}

/// For edges a to b (their values xs, e.g. durations, and ends p1 and p2): the number for which ends() is true, or (into out) their indices; with branches when writing, as `passes.ending_chunk()`, since few are
fn edges_where<X: Copy>(xs: &[X], p1: &[i64], p2: &[i64], a: usize, ends: impl Fn(X, &i64, &i64) -> bool, out: Option<&mut [i64]>) -> usize {
    let each = xs.iter().zip(p1).zip(p2).map(|((&x, u), v)| ends(x, u, v)); // The ends by reference, so they are read only if needed
    match out {
        None => each.map(|e| e as usize).sum(),
        Some(o) => {
            let mut m = 0;
            for (i, e) in each.enumerate() {
                if e {
                    o[m] = (a + i) as i64;
                    m += 1;
                }
            }
            m
        }
    }
}

/// The number of edges for which ends() is true, counted first so only what is found is written, and their indices if not none or all
fn some_edges<X: Copy + Sync>(xs: &[X], p1: &[i64], p2: &[i64], nchunks: usize, ends: impl Fn(X, &i64, &i64) -> bool + Send + Sync + Copy) -> (usize, Vec<i64>) {
    let (n, ends) = (xs.len(), &ends);
    let part = |c| {
        let (a, b) = bounds(c, n, nchunks);
        (a, &xs[a..b], &p1[a..b], &p2[a..b])
    };
    let found = par(nchunks, move |c| {
        let (a, x, u, v) = part(c);
        edges_where(x, u, v, a, ends, None)
    });
    let total = found.iter().sum::<usize>();
    if total == 0 || total == n { // None or all: no list is needed
        return (total, Vec::new());
    }
    (total, gather(&found, move |c, out| {
        let (a, x, u, v) = part(c);
        edges_where(x, u, v, a, ends, Some(out));
    }))
}

/// The partnerships that end in a step (fewer than two steps left, or a partner not alive): how many, and their indices if not none or all (`passes.ending()`)
pub fn ending<F: Num>(dur: &[F], p1: &[i64], p2: &[i64], alive: &[bool], nchunks: usize) -> (usize, Vec<i64>) {
    let n = dur.len();
    let all = par(nchunks, move |c| { // Whether every edge ends for its steps alone, as on a random network: checked first, in blocks without branches (so it is vectorized), stopping at the first edge that lasts
        let (a, b) = bounds(c, n, nchunks);
        dur[a..b].chunks(4096).all(|block| block.iter().map(|d| !(d.f() > 1.0) as usize).sum::<usize>() == block.len())
    });
    if all.iter().all(|&x| x) {
        return (n, Vec::new());
    }
    some_edges(dur, p1, p2, nchunks, |d, u, v| !(d.f() > 1.0) || !(alive[*u as usize] & alive[*v as usize])) // A branch on the steps left, which is predicted when all or none end (as on a random network), and none on who is alive, which few edges that end at random would mispredict
}

/// The edges with either end removed: how many, and their indices if not none or all (`passes.touching()`)
pub fn touching(p1: &[i64], p2: &[i64], removed: &[bool], nchunks: usize) -> (usize, Vec<i64>) {
    some_edges(p1, p1, p2, nchunks, |_, u, v| removed[*u as usize] | removed[*v as usize])
}


//%% Women of an age (ss.Births), and debut (ss.SexualNet)

/// The UIDs among auids of the women aged at least lo and under hi, in order: how many, written from the start of out (`passes.women()`)
pub fn women<A: Num>(auids: &[i64], female: &[bool], age: &[A], lo: f64, hi: f64, out: &mut [i64], nchunks: usize) -> usize {
    in_place(out, auids.len(), nchunks, move |a, b, o| {
        let mut m = 0;
        for &u in &auids[a..b] {
            let x = age[u as usize].f();
            o[m] = u; // Branchless, as can_transmit()
            m += (female[u as usize] & (x >= lo) & (x < hi)) as usize;
        }
        m
    })
}

/// The UIDs among auids of those alive with no debut age yet, or (if reached) alive, undecided, and at least their debut age, in order: how many, written from the start of out (`passes.debuts()`)
#[allow(clippy::too_many_arguments)]
pub fn debuts<A: Num, D: Num>(auids: &[i64], alive: &[bool], decided: &[bool], age: &[A], debut: &[D], reached: bool, out: &mut [i64], nchunks: usize) -> usize {
    in_place(out, auids.len(), nchunks, move |a, b, o| {
        let mut m = 0;
        for &u in &auids[a..b] {
            let k = u as usize;
            o[m] = u; // Branchless, as can_transmit()
            m += if reached { alive[k] & !decided[k] & (age[k].f() >= debut[k].f()) } else { alive[k] & debut[k].f().is_nan() } as usize;
        }
        m
    })
}


//%% Removing edges

/// A copy of arr without the elements at the indices gone (in order), into out, in runs between them (`passes.compress()`)
pub fn compress<T: Copy + Send + Sync>(arr: &[T], gone: &[i64], out: &mut [T], nchunks: usize) {
    let n = arr.len();
    let first = |x: usize| gone.partition_point(|&g| g < x as i64); // The first index removed at or after x, as np.searchsorted()
    let o = Shared::new(out);
    par(nchunks, move |c| {
        let (a, b) = bounds(c, n, nchunks);
        let (ga, gb) = (first(a), first(b));
        let dst = unsafe { o.part(a - ga, b - gb) };
        let (mut g, mut x, mut p) = (ga, a, 0);
        while x < b {
            let y = if g < gone.len() { (gone[g] as usize).min(b) } else { b }; // The end of this run
            dst[p..p + y - x].copy_from_slice(&arr[x..y]);
            p += y - x;
            x = y + 1; // Past the element removed
            g += 1;
        }
    });
}


//%% Competing exits from a state (ss.solvers.Agents)

/// For agents a to b of uids: those who leave (their uniform is below p_exit), at their own positions in pos and which, and how many (`passes.exits_chunk()`)
#[allow(clippy::too_many_arguments)]
fn exits_chunk<L: Ix>(uids: &[i64], slots: &[L], key: u64, shift: u64, scale: f64, p_exit: f64, edges: &[f64], a: usize, ps: &mut [i64], ws: &mut [i64]) -> usize {
    let by_uid = slots.is_empty();
    let n_exits = edges.len() - 1;
    let mut m = 0;
    for (i, &uid) in uids.iter().enumerate() {
        let u = draws::crn_uniform(key, if by_uid { uid as u64 } else { slots[uid as usize].int() as u64 }, shift, scale);
        if u < p_exit {
            let mut t = 0;
            while t < n_exits && !(u < edges[t + 1]) {
                t += 1;
            }
            ps[m] = (a + i) as i64;
            ws[m] = if t < n_exits { t as i64 } else { -1 };
            m += 1;
        }
    }
    m
}

/// Who leaves a state by competing rates that are the same for everyone: their positions in uids, and the exit each takes (-1 for none), written from the start of pos and which; how many (`passes.competing_exits()`)
#[allow(clippy::too_many_arguments)]
pub fn competing_exits<L: Ix>(uids: &[i64], slots: &[L], key: u64, shift: u64, scale: f64, p_exit: f64, edges: &[f64], pos: &mut [i64], which: &mut [i64], nchunks: usize) -> usize {
    let n = uids.len();
    let found = {
        let (ps, ws) = (Shared::new(pos), Shared::new(which));
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            exits_chunk(&uids[a..b], slots, key, shift, scale, p_exit, edges, a, unsafe { ps.part(a, b) }, unsafe { ws.part(a, b) })
        })
    };
    close_gaps(which, &found, n);
    close_gaps(pos, &found, n)
}

/// As competing_exits(), with chances that differ between agents: each one's chance of leaving, and the exits' hazards (a row for each exit), whose running total times share are the edges (`passes.competing_agent_exits()`)
#[allow(clippy::too_many_arguments)]
pub fn competing_agent_exits<L: Ix>(uids: &[i64], slots: &[L], key: u64, shift: u64, scale: f64, p_exit: &[f64], share: &[f64], hazards: &[f64], n_exits: usize, pos: &mut [i64], which: &mut [i64], nchunks: usize) -> usize {
    let n = uids.len();
    let by_uid = slots.is_empty();
    let cols = if n_exits > 0 { hazards.len()/n_exits } else { 0 };
    let found = {
        let (ps, ws) = (Shared::new(pos), Shared::new(which));
        par(nchunks, move |c| {
            let (a, b) = bounds(c, n, nchunks);
            let (ps, ws) = unsafe { (ps.part(a, b), ws.part(a, b)) };
            let mut m = 0;
            for (i, &uid) in uids[a..b].iter().enumerate() {
                let j = a + i;
                let u = draws::crn_uniform(key, if by_uid { uid as u64 } else { slots[uid as usize].int() as u64 }, shift, scale);
                if u < p_exit[j] {
                    let mut t = 0;
                    let mut edge = 0.0;
                    while t < n_exits {
                        edge += hazards[t*cols + j];
                        if u < edge*share[j] {
                            break;
                        }
                        t += 1;
                    }
                    ps[m] = j as i64;
                    ws[m] = if t < n_exits { t as i64 } else { -1 };
                    m += 1;
                }
            }
            m
        })
    };
    close_gaps(which, &found, n);
    close_gaps(pos, &found, n)
}


/// The bindings, with the same arguments and results as the Numba drivers they replace (serial and threaded alike: the number of chunks says which)
pub mod bind {
    use numpy::{Element, PyArray1, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use crate::{draws, passes};
    use crate::py::{arr, flat, head, sl, sl_mut};

    type Arr<'py> = Bound<'py, PyAny>;

    /// The number of chunks, taking as many threads as Numba has if there is more than one
    fn chunks(py: Python, nchunks: i64) -> PyResult<usize> {
        if nchunks > 1 {
            passes::set_threads(py.import("numba")?.call_method0("get_num_threads")?.extract()?);
        }
        Ok(nchunks.max(1) as usize)
    }

    /// A seed or the index of a call, as the bits Numba's `U64()` gives it
    fn bits(o: &Bound<PyAny>) -> PyResult<u64> {
        match o.extract::<i64>() {
            Ok(x) => Ok(x as u64),
            Err(_) => o.extract::<u64>(),
        }
    }

    /// The key of a call's uniforms, from its seed and index (`draws.key()`)
    fn key(seed: &Bound<PyAny>, ind: &Bound<PyAny>) -> PyResult<u64> {
        Ok(draws::key(bits(seed)? as i64, bits(ind)? as i64))
    }

    /// Each of a tuple of boolean arrays, as a slice
    fn bools<'a>(items: &'a [Bound<'a, PyAny>]) -> PyResult<Vec<&'a [bool]>> {
        items.iter().map(sl::<bool>).collect()
    }

    /// An array as a slice, or (if it is one value broadcast to every agent, as NumPy's broadcast_to() gives) that value alone
    fn bcast<'a, T: Element>(o: &'a Bound<PyAny>) -> PyResult<&'a [T]> {
        let a = o.cast::<PyArrayDyn<T>>()?;
        if a.ndim() == 1 && a.len() > 0 && a.strides()[0] == 0 {
            return Ok(unsafe { std::slice::from_raw_parts(a.data(), 1) });
        }
        sl(o)
    }

    #[pyfunction]
    fn levels<'py>(py: Python<'py>, raw: &Arr<'py>) -> PyResult<Arr<'py>> {
        typed!(raw, T in [f32, f64, bool, i8, i16, i32, i64, u8] => Ok(arr(py, passes::levels::<T>(sl(raw)?))))
    }

    #[pyfunction]
    fn crossed<'py>(py: Python<'py>, raw: &Arr<'py>, level: &Arr<'py>, lo: &Arr<'py>, hi: &Arr<'py>) -> PyResult<Arr<'py>> {
        typed!(raw, T in [f32, f64, i16, i32, i64] => typed!(level, L in [i16, i32, i8, i64] => Ok(arr(py, passes::crossed::<T, L>(sl(raw)?, sl(level)?, sl(lo)?, sl(hi)?)))))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn infection_probs<'py>(py: Python<'py>, susceptible: &Arr<'py>, auids: &Arr<'py>, groups: &Arr<'py>, rate: &Arr<'py>, rel_sus: &Arr<'py>, state_sus: f64, adjust: &Arr<'py>, mult: f64, dt: f64)
                            -> PyResult<(Arr<'py>, Arr<'py>)> {
        typed!(groups, G in [i16, i32, i64, i8] => typed!(rel_sus, R in [f32, f64] => {
            let (uids, probs) = passes::infection_probs::<G, R>(sl(susceptible)?, sl(auids)?, sl(groups)?, sl(rate)?, bcast(rel_sus)?, state_sus, sl(adjust)?, mult, dt);
            Ok((arr(py, uids), arr(py, probs)))
        }))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn census<'py>(py: Python<'py>, members: &Bound<'py, PyTuple>, alive: &Arr<'py>, groups: &Arr<'py>, n_groups: usize, want: &Arr<'py>, n_want: usize, nchunks: i64) -> PyResult<(Arr<'py>, Arr<'py>, Arr<'py>)> {
        let nchunks = chunks(py, nchunks)?;
        let items: Vec<Arr> = members.iter().collect();
        let ms = bools(&items)?;
        typed!(groups, G in [i16, i32, i64, i8] => {
            let (total, out, offsets) = passes::census::<G>(&ms, sl(alive)?, sl(groups)?, n_groups, sl(want)?, n_want, nchunks);
            Ok((PyArray1::from_vec(py, total).reshape([ms.len() + 1, n_groups])?.into_any(), arr(py, out), arr(py, offsets)))
        })
    }

    #[pyfunction]
    fn flatnonzero<'py>(py: Python<'py>, raw: &Arr<'py>, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        Ok(arr(py, passes::flatnonzero(sl(raw)?, nchunks)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn infect<'py>(py: Python<'py>, susceptible: &Arr<'py>, groups: &Arr<'py>, rate: &Arr<'py>, rel_sus: &Arr<'py>, state_sus: f64, adjust: &Arr<'py>, mult: f64, dt: f64, slots: &Arr<'py>,
                   seed: &Arr<'py>, ind: &Arr<'py>, shift: u64, scale: f64, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        let k = key(seed, ind)?;
        typed!(groups, G in [i16, i32, i64, i8] => typed!(rel_sus, R in [f32, f64] => typed!(slots, L in [i32, i64] => {
            Ok(arr(py, passes::infect::<G, R, L>(sl(susceptible)?, sl(groups)?, sl(rate)?, bcast(rel_sus)?, state_sus, sl(adjust)?, mult, dt, sl(slots)?, k, shift, scale, nchunks)))
        })))
    }

    #[pyfunction]
    fn due<'py>(py: Python<'py>, auids: &Arr<'py>, ti_dead: &Arr<'py>, ti_removed: &Arr<'py>, ti: i64, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        typed!(ti_dead, T in [f32, f64, i16, i32, i64] => Ok(arr(py, passes::due::<T>(sl(auids)?, sl(ti_dead)?, sl(ti_removed)?, ti, nchunks))))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn count_people<'py>(py: Python<'py>, auids: &Arr<'py>, states: &Bound<'py, PyTuple>, alive: &Arr<'py>, ti_dead: &Arr<'py>, ti_removed: &Arr<'py>, ti: i64, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        let items: Vec<Arr> = states.iter().collect();
        let st = bools(&items)?;
        typed!(ti_dead, T in [f32, f64, i16, i32, i64] => Ok(arr(py, passes::count_people::<T>(sl(auids)?, &st, sl(alive)?, sl(ti_dead)?, sl(ti_removed)?, ti, nchunks))))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn keep_alive<'py>(py: Python<'py>, auids: &Arr<'py>, alive: &Arr<'py>, age: &Arr<'py>, dt: f64, aging: bool, kept: &Arr<'py>, nchunks: i64) -> PyResult<(i64, Arr<'py>)> {
        let nchunks = chunks(py, nchunks)?;
        typed!(age, A in [f32, f64] => {
            let (n, gone) = passes::keep_alive::<A>(sl(auids)?, sl(alive)?, sl_mut(age)?, dt as A, aging, sl_mut(kept)?, nchunks);
            Ok((n, arr(py, gone)))
        })
    }

    #[pyfunction]
    fn due_alive<'py>(py: Python<'py>, alive: &Arr<'py>, ti_dead: &Arr<'py>, ti_removed: &Arr<'py>, ti: i64, nchunks: i64) -> PyResult<(Arr<'py>, i64)> {
        let nchunks = chunks(py, nchunks)?;
        typed!(ti_dead, T in [f32, f64, i16, i32, i64] => {
            let (out, n) = passes::due_alive::<T>(sl(alive)?, sl(ti_dead)?, sl(ti_removed)?, ti, nchunks);
            Ok((arr(py, out), n))
        })
    }

    #[pyfunction]
    fn count_alive<'py>(py: Python<'py>, states: &Bound<'py, PyTuple>, alive: &Arr<'py>, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        let items: Vec<Arr> = states.iter().collect();
        Ok(arr(py, passes::count_alive(&bools(&items)?, sl(alive)?, nchunks)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn keep_raw(py: Python, alive: &Arr, age: &Arr, dt: f64, aging: bool, kept: &Arr, expected: i64, nchunks: i64) -> PyResult<i64> {
        let nchunks = chunks(py, nchunks)?;
        typed!(age, A in [f32, f64] => Ok(passes::keep_raw::<A>(sl(alive)?, sl_mut(age)?, dt as A, aging, sl_mut(kept)?, expected, nchunks)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn random_edges(py: Python, inds: &Arr, starts: &Arr, seed: &Arr, by_agent: bool, source: &Arr, target: &Arr, nchunks: i64) -> PyResult<()> {
        let nchunks = chunks(py, nchunks)?;
        passes::random_edges(sl(inds)?, sl(starts)?, bits(seed)? as i64, by_agent, sl_mut(source)?, sl_mut(target)?, nchunks);
        Ok(())
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn candidates<'py>(py: Python<'py>, p1: &Arr<'py>, p2: &Arr<'py>, rel_trans: &Arr<'py>, rel_sus: &Arr<'py>, two_way: bool, flags: &Arr<'py>, out0: &Arr<'py>, out1: &Arr<'py>, nchunks: i64)
                       -> PyResult<(Arr<'py>, Arr<'py>)> {
        let nchunks = chunks(py, nchunks)?;
        typed!(rel_trans, T in [f32, f64] => typed!(rel_sus, S in [f32, f64] => {
            let (m0, m1) = passes::candidates::<T, S>(sl(p1)?, sl(p2)?, sl(rel_trans)?, sl(rel_sus)?, two_way, sl_mut(flags)?, sl_mut(out0)?, sl_mut(out1)?, nchunks);
            Ok((head(out0, m0)?, head(out1, m1)?))
        }))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn can_transmit<'py>(py: Python<'py>, src: &Arr<'py>, trg: &Arr<'py>, rel_trans: &Arr<'py>, rel_sus: &Arr<'py>, beta: &Arr<'py>, out: &Arr<'py>, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        typed!(rel_trans, T in [f32, f64] => typed!(rel_sus, S in [f32, f64] => typed!(beta, B in [f32, f64] => {
            head(out, passes::can_transmit::<T, S, B>(sl(src)?, sl(trg)?, sl(rel_trans)?, sl(rel_sus)?, sl(beta)?, sl_mut(out)?, nchunks))
        })))
    }

    #[pyfunction]
    fn ending<'py>(py: Python<'py>, dur: &Arr<'py>, p1: &Arr<'py>, p2: &Arr<'py>, alive: &Arr<'py>, nchunks: i64) -> PyResult<(usize, Arr<'py>)> {
        let nchunks = chunks(py, nchunks)?;
        typed!(dur, F in [f32, f64, i16, i32, i64] => {
            let (n, out) = passes::ending::<F>(sl(dur)?, sl(p1)?, sl(p2)?, sl(alive)?, nchunks);
            Ok((n, arr(py, out)))
        })
    }

    #[pyfunction]
    fn touching<'py>(py: Python<'py>, p1: &Arr<'py>, p2: &Arr<'py>, removed: &Arr<'py>, nchunks: i64) -> PyResult<(usize, Arr<'py>)> {
        let nchunks = chunks(py, nchunks)?;
        let (n, out) = passes::touching(sl(p1)?, sl(p2)?, sl(removed)?, nchunks);
        Ok((n, arr(py, out)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn women<'py>(py: Python<'py>, auids: &Arr<'py>, female: &Arr<'py>, age: &Arr<'py>, lo: f64, hi: f64, out: &Arr<'py>, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        typed!(age, A in [f32, f64] => head(out, passes::women::<A>(sl(auids)?, sl(female)?, sl(age)?, lo, hi, sl_mut(out)?, nchunks)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn debuts<'py>(py: Python<'py>, auids: &Arr<'py>, alive: &Arr<'py>, decided: &Arr<'py>, age: &Arr<'py>, debut: &Arr<'py>, reached: bool, out: &Arr<'py>, nchunks: i64) -> PyResult<Arr<'py>> {
        let nchunks = chunks(py, nchunks)?;
        typed!(age, A in [f32, f64] => typed!(debut, D in [f32, f64] => {
            head(out, passes::debuts::<A, D>(sl(auids)?, sl(alive)?, sl(decided)?, sl(age)?, sl(debut)?, reached, sl_mut(out)?, nchunks))
        }))
    }

    /// An array of any type as a slice of whole numbers of its width, since copying it needs no more
    #[allow(clippy::mut_from_ref)]
    fn raw<'a, T>(o: &'a Bound<PyAny>, width: usize) -> PyResult<&'a mut [T]> {
        let a = o.cast::<PyUntypedArray>()?;
        if !a.is_c_contiguous() || a.ndim() != 1 || a.dtype().itemsize() != width || a.dtype().has_object() { // Objects' references would be copied without being counted
            return Err(PyTypeError::new_err("an array of a type the Rust kernels do not take"));
        }
        if unsafe { (*a.as_array_ptr()).fields.flags } & numpy::npyffi::NPY_ARRAY_WRITEABLE == 0 {
            return Err(PyTypeError::new_err("an array the Rust kernels write to must be writeable"));
        }
        let n = a.len();
        Ok(if n == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut((*a.as_array_ptr()).fields.data as *mut T, n) } })
    }

    #[pyfunction]
    fn compress(py: Python, arr: &Arr, gone: &Arr, out: &Arr, nchunks: i64) -> PyResult<()> {
        let nchunks = chunks(py, nchunks)?;
        let width = arr.cast::<PyUntypedArray>()?.dtype().itemsize();
        match width {
            1 => passes::compress::<u8>(raw(arr, 1)?, sl(gone)?, raw(out, 1)?, nchunks),
            2 => passes::compress::<u16>(raw(arr, 2)?, sl(gone)?, raw(out, 2)?, nchunks),
            4 => passes::compress::<u32>(raw(arr, 4)?, sl(gone)?, raw(out, 4)?, nchunks),
            8 => passes::compress::<u64>(raw(arr, 8)?, sl(gone)?, raw(out, 8)?, nchunks),
            16 => passes::compress::<u128>(raw(arr, 16)?, sl(gone)?, raw(out, 16)?, nchunks),
            _ => return Err(PyTypeError::new_err("an array of a type the Rust kernels do not take")),
        }
        Ok(())
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn competing_exits(py: Python, uids: &Arr, slots: &Arr, seed: &Arr, ind: &Arr, shift: u64, scale: f64, p_exit: f64, edges: &Arr, pos: &Arr, which: &Arr, nchunks: i64) -> PyResult<usize> {
        let nchunks = chunks(py, nchunks)?;
        let k = key(seed, ind)?;
        typed!(slots, L in [i64, i32] => Ok(passes::competing_exits::<L>(sl(uids)?, sl(slots)?, k, shift, scale, p_exit, sl(edges)?, sl_mut(pos)?, sl_mut(which)?, nchunks)))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn competing_agent_exits(py: Python, uids: &Arr, slots: &Arr, seed: &Arr, ind: &Arr, shift: u64, scale: f64, p_exit: &Arr, share: &Arr, hazards: &Arr, pos: &Arr, which: &Arr, nchunks: i64) -> PyResult<usize> {
        let nchunks = chunks(py, nchunks)?;
        let k = key(seed, ind)?;
        let n_exits = hazards.cast::<PyArrayDyn<f64>>()?.shape()[0];
        typed!(slots, L in [i64, i32] => Ok(passes::competing_agent_exits::<L>(sl(uids)?, sl(slots)?, k, shift, scale, sl(p_exit)?, sl(share)?, flat(hazards)?, n_exits, sl_mut(pos)?, sl_mut(which)?, nchunks)))
    }

    /// Add these kernels to the module
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(levels, m)?)?;
        m.add_function(wrap_pyfunction!(crossed, m)?)?;
        m.add_function(wrap_pyfunction!(infection_probs, m)?)?;
        m.add_function(wrap_pyfunction!(census, m)?)?;
        m.add_function(wrap_pyfunction!(flatnonzero, m)?)?;
        m.add_function(wrap_pyfunction!(infect, m)?)?;
        m.add_function(wrap_pyfunction!(due, m)?)?;
        m.add_function(wrap_pyfunction!(count_people, m)?)?;
        m.add_function(wrap_pyfunction!(keep_alive, m)?)?;
        m.add_function(wrap_pyfunction!(due_alive, m)?)?;
        m.add_function(wrap_pyfunction!(count_alive, m)?)?;
        m.add_function(wrap_pyfunction!(keep_raw, m)?)?;
        m.add_function(wrap_pyfunction!(random_edges, m)?)?;
        m.add_function(wrap_pyfunction!(candidates, m)?)?;
        m.add_function(wrap_pyfunction!(can_transmit, m)?)?;
        m.add_function(wrap_pyfunction!(ending, m)?)?;
        m.add_function(wrap_pyfunction!(touching, m)?)?;
        m.add_function(wrap_pyfunction!(women, m)?)?;
        m.add_function(wrap_pyfunction!(debuts, m)?)?;
        m.add_function(wrap_pyfunction!(compress, m)?)?;
        m.add_function(wrap_pyfunction!(competing_exits, m)?)?;
        m.add_function(wrap_pyfunction!(competing_agent_exits, m)?)?;
        Ok(())
    }
}
