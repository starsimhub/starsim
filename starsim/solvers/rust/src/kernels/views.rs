//! Arrays as the kernels see them: a pointer and a shape, read and written in place, as Numba's are (several views may share an
//! array, and a kernel writes to the arrays it is passed). The last axis is contiguous (the bindings see to it), so a row is a slice.

/// A 1-D array
#[derive(Clone, Copy)]
pub struct A1<T> { p: *mut T, n: usize }

/// A 2-D array, with rows `s0` apart
#[derive(Clone, Copy)]
pub struct A2<T> { p: *mut T, n0: usize, n1: usize, s0: usize }

/// A 3-D array
#[derive(Clone, Copy)]
pub struct A3<T> { p: *mut T, n0: usize, n1: usize, n2: usize, s0: usize, s1: usize }

// The views are passed to the threads of advance_threads(), which each write to their own runs only
unsafe impl<T> Send for A1<T> {}
unsafe impl<T> Sync for A1<T> {}
unsafe impl<T> Send for A2<T> {}
unsafe impl<T> Sync for A2<T> {}
unsafe impl<T> Send for A3<T> {}
unsafe impl<T> Sync for A3<T> {}

impl<T: Copy> A1<T> {
    pub fn new(p: *mut T, n: usize) -> Self { A1 { p, n } }
    #[inline(always)] pub fn len(&self) -> usize { self.n }
    #[inline(always)] pub fn at(&self, i: usize) -> T { debug_assert!(i < self.n); unsafe { *self.p.add(i) } }
    #[inline(always)] pub fn set(&self, i: usize, v: T) { debug_assert!(i < self.n); unsafe { *self.p.add(i) = v } }
    #[inline(always)] pub fn sl(&self) -> &[T] { if self.n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.p, self.n) } } }
    #[allow(clippy::mut_from_ref)]
    #[inline(always)] pub fn sl_mut(&self) -> &mut [T] { if self.n == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.p, self.n) } } }
    pub fn ptr(&self) -> *mut T { self.p }
    pub fn fill(&self, v: T) { for i in 0..self.n { self.set(i, v) } }
    pub fn copy_from(&self, o: A1<T>) { for i in 0..self.n { self.set(i, o.at(i)) } }
}

impl A1<f64> {
    #[inline(always)] pub fn add(&self, i: usize, v: f64) { self.set(i, self.at(i) + v) }
}

impl A1<i64> {
    /// An entry that is an index
    #[inline(always)] pub fn u(&self, i: usize) -> usize { self.at(i) as usize }
}

impl A1<u8> {
    /// An entry of a boolean array
    #[inline(always)] pub fn b(&self, i: usize) -> bool { self.at(i) != 0 }
}

impl<T: Copy> A2<T> {
    pub fn new(p: *mut T, n0: usize, n1: usize, s0: usize) -> Self { A2 { p, n0, n1, s0 } }
    #[inline(always)] pub fn n0(&self) -> usize { self.n0 }
    #[inline(always)] pub fn n1(&self) -> usize { self.n1 }
    #[inline(always)] pub fn at(&self, i: usize, j: usize) -> T { debug_assert!(i < self.n0 && j < self.n1); unsafe { *self.p.add(i*self.s0 + j) } }
    #[inline(always)] pub fn set(&self, i: usize, j: usize, v: T) { debug_assert!(i < self.n0 && j < self.n1); unsafe { *self.p.add(i*self.s0 + j) = v } }
    #[inline(always)] pub fn row(&self, i: usize) -> A1<T> { debug_assert!(i < self.n0); A1 { p: unsafe { self.p.add(i*self.s0) }, n: self.n1 } }
    /// Rows a to b-1
    pub fn rows(&self, a: usize, b: usize) -> A2<T> { debug_assert!(a <= b && b <= self.n0); A2 { p: unsafe { self.p.add(a*self.s0) }, n0: b - a, n1: self.n1, s0: self.s0 } }
    /// Its rows as one slice, for a kernel that takes their stride (`s0()`)
    #[allow(clippy::mut_from_ref)]
    pub fn flat(&self) -> &mut [T] { if self.n0 == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.p, (self.n0 - 1)*self.s0 + self.n1) } } }
    #[inline(always)] pub fn s0(&self) -> usize { self.s0 }
    pub fn fill(&self, v: T) { for i in 0..self.n0 { self.row(i).fill(v) } }
    pub fn copy_from(&self, o: A2<T>) { for i in 0..self.n0 { self.row(i).copy_from(o.row(i)) } }
}

impl A2<f64> {
    #[inline(always)] pub fn add(&self, i: usize, j: usize, v: f64) { self.set(i, j, self.at(i, j) + v) }
}

impl A2<i64> {
    #[inline(always)] pub fn u(&self, i: usize, j: usize) -> usize { self.at(i, j) as usize }
}

impl A2<u8> {
    #[inline(always)] pub fn b(&self, i: usize, j: usize) -> bool { self.at(i, j) != 0 }
}

impl<T: Copy> A3<T> {
    pub fn new(p: *mut T, n: [usize; 3], s0: usize, s1: usize) -> Self { A3 { p, n0: n[0], n1: n[1], n2: n[2], s0, s1 } }
    #[inline(always)] pub fn n0(&self) -> usize { self.n0 }
    #[inline(always)] pub fn n1(&self) -> usize { self.n1 }
    #[inline(always)] pub fn n2(&self) -> usize { self.n2 }
    #[inline(always)] pub fn at(&self, i: usize, j: usize, k: usize) -> T { debug_assert!(i < self.n0 && j < self.n1 && k < self.n2); unsafe { *self.p.add(i*self.s0 + j*self.s1 + k) } }
    #[inline(always)] pub fn set(&self, i: usize, j: usize, k: usize, v: T) { debug_assert!(i < self.n0 && j < self.n1 && k < self.n2); unsafe { *self.p.add(i*self.s0 + j*self.s1 + k) = v } }
    #[inline(always)] pub fn plane(&self, i: usize) -> A2<T> { debug_assert!(i < self.n0); A2 { p: unsafe { self.p.add(i*self.s0) }, n0: self.n1, n1: self.n2, s0: self.s1 } }
}

impl A3<f64> {
    #[inline(always)] pub fn add(&self, i: usize, j: usize, k: usize, v: f64) { self.set(i, j, k, self.at(i, j, k) + v) }
}

/// An array a kernel makes for itself, which keeps its view valid (the buffer never moves)
pub struct Own<T> { _buf: Vec<T> }

pub fn own1<T: Copy>(n: usize, v: T) -> (Own<T>, A1<T>) {
    let mut buf = vec![v; n];
    let a = A1::new(buf.as_mut_ptr(), n);
    (Own { _buf: buf }, a)
}

pub fn own2<T: Copy>(n0: usize, n1: usize, v: T) -> (Own<T>, A2<T>) {
    let mut buf = vec![v; n0*n1];
    let a = A2::new(buf.as_mut_ptr(), n0, n1, n1);
    (Own { _buf: buf }, a)
}

pub fn own3<T: Copy>(n0: usize, n1: usize, n2: usize, v: T) -> (Own<T>, A3<T>) {
    let mut buf = vec![v; n0*n1*n2];
    let a = A3::new(buf.as_mut_ptr(), [n0, n1, n2], n1*n2, n2);
    (Own { _buf: buf }, a)
}

// Python's max(a, b) and min(a, b), and int() of a float, as Numba compiles them
pub use crate::draws::{max as mx, min as mn, to_i64 as to_int};

/// NumPy's (and Numba's) max of a float array: the first NaN if there is one
pub fn amax(a: A1<f64>) -> f64 {
    let mut m = a.at(0);
    if m.is_nan() { return m; }
    for i in 1..a.len() {
        let v = a.at(i);
        if v.is_nan() { return v; }
        if v > m { m = v; }
    }
    m
}

/// NumPy's min of a float array
pub fn amin(a: A1<f64>) -> f64 {
    let mut m = a.at(0);
    if m.is_nan() { return m; }
    for i in 1..a.len() {
        let v = a.at(i);
        if v.is_nan() { return v; }
        if v < m { m = v; }
    }
    m
}

/// An array of N dimensions, for the outputs of the entries that take many runs (4 or 5 axes)
#[derive(Clone, Copy)]
pub struct Nd<T, const N: usize> { p: *mut T, n: [usize; N], s: [usize; N] }

unsafe impl<T, const N: usize> Send for Nd<T, N> {}
unsafe impl<T, const N: usize> Sync for Nd<T, N> {}

impl<T: Copy, const N: usize> Nd<T, N> {
    pub fn new(p: *mut T, n: [usize; N], s: [usize; N]) -> Self { Nd { p, n, s } }
    #[inline(always)] pub fn shape(&self, k: usize) -> usize { self.n[k] }
    #[inline(always)]
    fn off(&self, i: [usize; N]) -> usize {
        let mut o = 0;
        for k in 0..N { debug_assert!(i[k] < self.n[k]); o += i[k]*self.s[k]; }
        o
    }
    #[inline(always)] pub fn at(&self, i: [usize; N]) -> T { unsafe { *self.p.add(self.off(i)) } }
    #[inline(always)] pub fn set(&self, i: [usize; N], v: T) { unsafe { *self.p.add(self.off(i)) = v } }
}

impl<const N: usize> Nd<f64, N> {
    #[inline(always)] pub fn add(&self, i: [usize; N], v: f64) { self.set(i, self.at(i) + v) }
}

impl<T: Copy> Nd<T, 4> {
    /// The 3-D array at index i of the first axis
    pub fn sub(&self, i: usize) -> A3<T> { debug_assert!(i < self.n[0]); A3::new(unsafe { self.p.add(i*self.s[0]) }, [self.n[1], self.n[2], self.n[3]], self.s[1], self.s[2]) }
}
