use rust_gymnasium::{BoxSpace, Discrete, MultiBinary, MultiDiscrete, spaces::Space};
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

proptest! {
    // Discrete sampling always within bounds and deterministic per seed
    #[test]
    fn discrete_sampling_contains_and_deterministic(n in 1u32..1000, seed in any::<u64>()) {
        let d = Discrete::new(n);
        let mut rng1 = StdRng::seed_from_u64(seed);
        let mut rng2 = StdRng::seed_from_u64(seed);
        for _ in 0..100 {
            let v1 = d.sample(&mut rng1);
            let v2 = d.sample(&mut rng2);
            prop_assert!(d.contains(&v1));
            prop_assert!(d.contains(&v2));
            // Same seed, same sequence
            prop_assert_eq!(v1, v2);
        }
    }

    // MultiBinary sampling only 0/1 and correct length
    #[test]
    fn multibinary_sampling_valid(n in 1usize..256, seed in any::<u64>()) {
        let mb = MultiBinary::new(n);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..50 {
            let v = mb.sample(&mut rng);
            prop_assert!(mb.contains(&v));
            prop_assert_eq!(v.len(), n);
        }
    }

    // MultiDiscrete per-dimension ranges honored
    #[test]
    fn multidiscrete_sampling_valid(nvec in proptest::collection::vec(1u32..10_000, 1..8), seed in any::<u64>()) {
        let md = MultiDiscrete::new(nvec.clone());
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..50 {
            let v = md.sample(&mut rng);
            prop_assert!(md.contains(&v));
            prop_assert_eq!(v.len(), nvec.len());
        }
    }
}

// BoxSpace needs concrete element type and const N known at compile time.
// Provide a few representative cases and check contains + determinism.
#[test]
fn boxspace_sampling_contains_and_deterministic() {
    let mut rng1 = StdRng::seed_from_u64(12345);
    let mut rng2 = StdRng::seed_from_u64(12345);

    let b3 = BoxSpace::new([0.0, -1.0, 2.5], [1.0, 1.0, 3.5]);
    for _ in 0..100 {
        let v1 = b3.sample(&mut rng1);
        let v2 = b3.sample(&mut rng2);
        assert!(b3.contains(&v1));
        assert!(b3.contains(&v2));
        assert_eq!(v1, v2);
    }

    let b2 = BoxSpace::new([-10i32, 5i32], [0i32, 10i32]);
    let mut r1 = StdRng::seed_from_u64(999);
    let mut r2 = StdRng::seed_from_u64(999);
    for _ in 0..50 {
        let v1 = b2.sample(&mut r1);
        let v2 = b2.sample(&mut r2);
        assert!(b2.contains(&v1));
        assert!(b2.contains(&v2));
        assert_eq!(v1, v2);
    }
}
