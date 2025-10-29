//! Deterministic seeding and RNG utilities (Step 5 of README)
//!
//! This module provides:
//! - SeedSequence: expands a root u64 seed into deterministic sub-seeds
//! - RngStream: a reproducible PRNG stream (ChaCha8)
//! - Helpers to split seeds for vectorized envs

use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Type alias for the default RNG stream used across the crate.
pub type RngStream = ChaCha8Rng;

/// SplitMix64 mixer used to expand a 64-bit seed into a sequence of pseudo-random u64 values.
/// This is fast and deterministic, ideal for deriving sub-seeds.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    state: u128,  // keep some extra space to avoid trivial cycles when mixing
}

impl SeedSequence {
    /// Create a new seed sequence from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        // Initialize using a 128-bit state derived from the seed using a fixed constant.
        // The constants are taken from SplitMix64 reference to ensure good bit diffusion.
        let init = (seed as u128) ^ 0x9E3779B97F4A7C15u128;
        Self { state: init }
    }

    /// Generate the next sub-seed deterministically.
    pub fn next_subseed(&mut self) -> u64 {
        // SplitMix64 step operating on the low 64 bits while evolving a 128-bit state
        // to reduce correlation across long runs.
        let mut z = (self.state as u64).wrapping_add(0x9E3779B97F4A7C15);
        self.state = (self.state ^ (z as u128)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Create an RNG stream seeded from the next subseed.
    pub fn next_rng(&mut self) -> RngStream {
        let s = self.next_subseed();
        RngStream::seed_from_u64(s)
    }

    /// Produce `n` sub-seeds deterministically from this sequence.
    pub fn split_n(&mut self, n: usize) -> Vec<u64> {
        (0..n).map(|_| self.next_subseed()).collect()
    }
}

/// Split a root seed into N sub-seeds deterministically (convenience helper).
pub fn split_n(seed: u64, n: usize) -> Vec<u64> {
    let mut ss = SeedSequence::new(seed);
    ss.split_n(n)
}

/// Create a new RNG stream from a root seed (convenience).
pub fn rng_from_seed(seed: u64) -> RngStream {
    RngStream::seed_from_u64(seed)
}

/// Sample a u64 from an RNG (utility to have a stable method surface without pulling rand prelude).
pub fn sample_u64(rng: &mut impl RngCore) -> u64 {
    rng.next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{BoxSpace, Discrete, Space};

    #[test]
    fn split_n_is_deterministic() {
        let a = split_n(12345, 5);
        let b = split_n(12345, 5);
        assert_eq!(a, b);
        let c = split_n(12346, 5);
        assert_ne!(a, c);
    }

    #[test]
    fn rng_stream_is_reproducible() {
        let mut r1 = rng_from_seed(7);
        let mut r2 = rng_from_seed(7);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn spaces_sample_deterministically_with_seed_seq() {
        let mut ss = SeedSequence::new(999);
        let mut rng1 = ss.next_rng();
        let mut rng2 = SeedSequence::new(999).next_rng();
        let d = Discrete::new(10);
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng1), d.sample(&mut rng2));
        }

        let b = BoxSpace::new([0.0, -1.0], [1.0, 1.0]);
        let mut rng3 = SeedSequence::new(2024).next_rng();
        let mut rng4 = SeedSequence::new(2024).next_rng();
        for _ in 0..20 {
            assert_eq!(b.sample(&mut rng3), b.sample(&mut rng4));
        }
    }
}
