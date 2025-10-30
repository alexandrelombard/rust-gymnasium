pub mod core;
pub mod spaces;
pub mod utils;
pub mod envs;
pub mod wrappers;
pub mod vector;

pub use crate::core::{Env, GymError, Info, InfoValue, RenderFrame, Result, Step};
pub use crate::spaces::{BoxSpace, Discrete, MultiBinary, MultiDiscrete, Space};
pub use crate::envs::{CartPoleEnv, MountainCarEnv, MountainCarContinuousEnv, AcrobotEnv, PendulumEnv, LunarLanderEnv};
pub use crate::wrappers::{TimeLimit, ClipAction, ClipReward, TransformObservation, TransformAction, TransformReward, RecordEpisodeStatistics};
pub use crate::utils::{encode_png, save_png};
pub use crate::vector::SyncVectorEnv;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// A tiny dummy environment to validate the trait compiles and basic methods work.
    struct CounterEnv {
        state: i32,
    }

    impl Env for CounterEnv {
        type Obs = i32;
        type Act = i32;

        fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Info) {
            self.state = 0;
            (self.state, Info::new())
        }

        fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
            self.state += action;
            let terminated = self.state >= 3;
            Step::new(self.state, 1.0, terminated, false, Info::new())
        }

        fn render(&self) -> Option<RenderFrame> {
            Some(RenderFrame::Text(format!("state={}", self.state)))
        }
    }

    #[test]
    fn dummy_env_runs() {
        let mut env = CounterEnv { state: 0 };
        let (_obs, _info) = env.reset(None);
        let s1 = env.step(1);
        assert_eq!(s1.observation, 1);
        assert!(!s1.terminated);
        let s2 = env.step(2);
        assert_eq!(s2.observation, 3);
        assert!(s2.terminated);
        assert!(matches!(env.render(), Some(RenderFrame::Text(_))));
        env.close();
    }

    #[test]
    fn spaces_discrete_and_box() {
        let mut rng = StdRng::seed_from_u64(42);
        let d = Discrete::new(5);
        for _ in 0..100 {
            let v = d.sample(&mut rng);
            assert!(d.contains(&v));
        }

        let b = BoxSpace::new([0.0, -1.0, 2.5], [1.0, 1.0, 3.5]);
        for _ in 0..100 {
            let v = b.sample(&mut rng);
            assert!(b.contains(&v));
            assert!(v[0] >= 0.0 && v[0] <= 1.0);
        }
    }

    #[test]
    fn spaces_multi_binary_and_multi_discrete() {
        let mut rng = StdRng::seed_from_u64(123);

        let mb = MultiBinary::new(8);
        for _ in 0..50 {
            let v = mb.sample(&mut rng);
            assert!(mb.contains(&v));
            assert_eq!(v.len(), 8);
            assert!(v.iter().all(|&x| x == 0 || x == 1));
        }

        let md = MultiDiscrete::new(vec![1, 2, 5, 10]);
        for _ in 0..50 {
            let v = md.sample(&mut rng);
            assert!(md.contains(&v));
            assert_eq!(v.len(), 4);
            assert!(v[0] == 0); // n=1 always samples/contains only 0
            assert!(v[1] < 2 && v[2] < 5 && v[3] < 10);
        }

        // Negative containment checks
        let bad_mb = vec![0, 1, 2, 0, 1, 0, 1, 0];
        assert!(!mb.contains(&bad_mb));
        let bad_md = vec![0, 2, 5, 10];
        assert!(!md.contains(&bad_md));
    }

    #[test]
    fn classic_control_cartpole_runs() {
        let mut env = CartPoleEnv::default();
        let (_o, _info) = env.reset(Some(0));
        for _ in 0..10 {
            let s = env.step(1);
            assert!(s.reward >= 1.0 - 1e-6);
            if s.terminated || s.truncated { break; }
        }
    }

    #[test]
    fn classic_control_mountaincar_runs() {
        let mut env = MountainCarEnv::default();
        let (_o, _info) = env.reset(Some(0));
        for _ in 0..10 {
            let s = env.step(2);
            assert!(s.reward <= 0.0);
            if s.terminated || s.truncated { break; }
        }
    }

    #[test]
    fn render_text_exists() {
        let env = CartPoleEnv::default();
        let frame = env.render();
        assert!(matches!(frame, Some(RenderFrame::Text(_)) | Some(RenderFrame::Pixels { .. })));
    }

    #[test]
    fn lunar_lander_runs_and_renders() {
        let mut env = LunarLanderEnv::default();
        let (_o, _info) = env.reset(Some(0));
        for _ in 0..5 {
            let s = env.step(2); // fire main engine a bit
            assert!(s.observation.len() == 8);
            if s.terminated || s.truncated { break; }
        }
        let frame = env.render();
        assert!(matches!(frame, Some(RenderFrame::Pixels { .. }) | Some(RenderFrame::Text(_))));
    }

    #[cfg(not(feature = "image"))]
    #[test]
    fn encode_png_without_feature_not_supported() {
        let frame = RenderFrame::Pixels { width: 2, height: 2, data: vec![255, 0, 0, 255,  0, 255, 0, 255,  0, 0, 255, 255,  255, 255, 255, 255] };
        let err = encode_png(&frame).unwrap_err();
        match err {
            GymError::NotSupported(_) => {}
            other => panic!("Expected NotSupported, got {:?}", other),
        }
    }

    #[cfg(feature = "image")]
    #[test]
    fn encode_png_with_feature_produces_png_signature() {
        // 2x2 RGBA pixels
        let frame = RenderFrame::Pixels { width: 2, height: 2, data: vec![255, 0, 0, 255,  0, 255, 0, 255,  0, 0, 255, 255,  255, 255, 255, 255] };
        let bytes = encode_png(&frame).expect("PNG encoding should succeed");
        let sig = &bytes[..8];
        assert_eq!(sig, &[137, 80, 78, 71, 13, 10, 26, 10]);
    }
}


#[cfg(test)]
mod mountain_car_continuous_tests {
    use super::*;

    #[test]
    fn classic_control_mountaincar_continuous_runs_and_renders() {
        let mut env = MountainCarContinuousEnv::default();
        let (_o, _info) = env.reset(Some(0));
        for _ in 0..10 {
            let s = env.step(0.5);
            // Reward can be negative due to action penalty; ensure obs length is 2
            assert!(s.observation.len() == 2);
            if s.terminated || s.truncated { break; }
        }
        let frame = env.render();
        assert!(matches!(frame, Some(RenderFrame::Pixels { .. }) | Some(RenderFrame::Text(_))));
    }
}
