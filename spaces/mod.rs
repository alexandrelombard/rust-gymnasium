/// Space implementations (Step 4 of README)

pub mod space;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

pub use space::Space;

/// A discrete space of integers in [0, n).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Discrete {
    n: u32,
}

impl Discrete {
    pub fn new(n: u32) -> Self {
        assert!(n > 0, "Discrete space requires n > 0");
        Self { n }
    }

    pub fn n(&self) -> u32 { self.n }
}

impl Space for Discrete {
    type Element = u32;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Element {
        // Uniform over [0, n)
        if self.n == 1 { return 0; }
        let dist = Uniform::from(0..self.n);
        dist.sample(rng)
    }

    fn contains(&self, elem: &Self::Element) -> bool { *elem < self.n }
}

/// A fixed-length binary vector space of size `n`.
/// Elements are vectors of 0/1 values (u8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiBinary {
    n: usize,
}

impl MultiBinary {
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "MultiBinary requires n > 0");
        Self { n }
    }

    pub fn n(&self) -> usize { self.n }
}

impl Space for MultiBinary {
    type Element = Vec<u8>;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Element {
        // Sample each bit independently with p=0.5 using a uniform over {0,1}
        let dist = Uniform::from(0u8..=1u8);
        (0..self.n).map(|_| dist.sample(rng)).collect()
    }

    fn contains(&self, elem: &Self::Element) -> bool {
        elem.len() == self.n && elem.iter().all(|&v| v == 0 || v == 1)
    }
}

/// A multi-dimensional discrete space with per-dimension sizes nvec[i] (values in [0, nvec[i])).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiDiscrete {
    nvec: Vec<u32>,
}

impl MultiDiscrete {
    pub fn new<I: Into<Vec<u32>>>(nvec: I) -> Self {
        let nvec = nvec.into();
        assert!(!nvec.is_empty(), "MultiDiscrete requires at least one dimension");
        for (i, &n) in nvec.iter().enumerate() {
            assert!(n > 0, "MultiDiscrete nvec[{i}] must be > 0");
        }
        Self { nvec }
    }

    pub fn nvec(&self) -> &[u32] { &self.nvec }
    pub fn ndim(&self) -> usize { self.nvec.len() }
}

impl Space for MultiDiscrete {
    type Element = Vec<u32>;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Element {
        self.nvec
            .iter()
            .map(|&n| if n == 1 { 0 } else { Uniform::from(0..n).sample(rng) })
            .collect()
    }

    fn contains(&self, elem: &Self::Element) -> bool {
        if elem.len() != self.nvec.len() { return false; }
        elem.iter().zip(self.nvec.iter()).all(|(&v, &n)| v < n)
    }
}

/// A simple Box-like space with element type `T` and fixed compile-time length `N`.
/// Uses per-dimension inclusive lower/upper bounds for validation and sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxSpace<T: Copy + PartialOrd, const N: usize> {
    low: [T; N],
    high: [T; N],
}

impl<T: Copy + PartialOrd, const N: usize> BoxSpace<T, N> {
    pub fn new(low: [T; N], high: [T; N]) -> Self {
        // Validate low <= high elementwise
        for i in 0..N {
            assert!(low[i] <= high[i], "low[{i}] > high[{i}]");
        }
        Self { low, high }
    }

    pub fn low(&self) -> &[T; N] { &self.low }
    pub fn high(&self) -> &[T; N] { &self.high }
}

impl<T, const N: usize> Space for BoxSpace<T, N>
where
    T: Copy + PartialOrd + rand::distributions::uniform::SampleUniform,
{
    type Element = [T; N];

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::Element {
        // Sample each dimension independently from Uniform[low, high]
        let mut arr = self.low;
        for i in 0..N {
            let dist = Uniform::new_inclusive(self.low[i], self.high[i]);
            arr[i] = dist.sample(rng);
        }
        arr
    }

    fn contains(&self, elem: &Self::Element) -> bool {
        (0..N).all(|i| self.low[i] <= elem[i] && elem[i] <= self.high[i])
    }
}
