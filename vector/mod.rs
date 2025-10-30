// Vectorized environments (Step 7 of README)
// A simple synchronous vector environment running N copies of an Env in a loop.

use crate::core::{Env, RenderFrame, Step};

/// Runs N copies of an environment in the current thread.
///
/// - Construct with `SyncVectorEnv::new(n, || MyEnv::default())`
/// - Step with a batch of actions: `step_all(actions)`
/// - Reset all envs (optionally with a base seed): `reset_all(Some(0))`
pub struct SyncVectorEnv<E: Env> {
    envs: Vec<E>,
}

impl<E: Env> SyncVectorEnv<E> {
    /// Create N copies using the provided factory closure.
    pub fn new<F>(n: usize, mut factory: F) -> Self
    where
        F: FnMut() -> E,
    {
        let mut envs = Vec::with_capacity(n);
        for _ in 0..n {
            envs.push(factory());
        }
        Self { envs }
    }

    /// Number of contained environments.
    pub fn len(&self) -> usize { self.envs.len() }
    /// Whether there are no environments.
    pub fn is_empty(&self) -> bool { self.envs.is_empty() }

    /// Reset all environments. If `base_seed` is provided, each env gets base_seed + i.
    pub fn reset_all(&mut self, base_seed: Option<u64>) -> Vec<(E::Obs, crate::core::Info)> {
        self.envs
            .iter_mut()
            .enumerate()
            .map(|(i, e)| {
                let seed = base_seed.map(|s| s + i as u64);
                e.reset(seed)
            })
            .collect()
    }

    /// Step all environments with a batch of actions.
    /// The length of `actions` must equal `self.len()`.
    pub fn step_all(&mut self, actions: Vec<E::Act>) -> Vec<Step<E::Obs>> {
        assert_eq!(actions.len(), self.envs.len(), "actions len must match envs len");
        self.envs
            .iter_mut()
            .zip(actions.into_iter())
            .map(|(e, a)| e.step(a))
            .collect()
    }

    /// Render all environments; returns a vector of optional frames (one per env).
    pub fn render_all(&self) -> Vec<Option<RenderFrame>> {
        self.envs.iter().map(|e| e.render()).collect()
    }

    /// Close all environments.
    pub fn close_all(&mut self) {
        for e in &mut self.envs { e.close(); }
    }

    /// Get immutable access to underlying envs (advanced usage).
    pub fn envs(&self) -> &[E] { &self.envs }
    /// Get mutable access to underlying envs (advanced usage).
    pub fn envs_mut(&mut self) -> &mut [E] { &mut self.envs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Info, RenderFrame};

    // A tiny dummy environment to validate vector stepping
    #[derive(Clone, Default)]
    struct DummyEnv { s: i32 }
    impl Env for DummyEnv {
        type Obs = i32;
        type Act = i32;
        fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Info) { self.s = 0; (self.s, Info::new()) }
        fn step(&mut self, a: Self::Act) -> Step<Self::Obs> {
            self.s += a;
            Step::new(self.s, 1.0, self.s >= 5, false, Info::new())
        }
        fn render(&self) -> Option<RenderFrame> { Some(RenderFrame::Text(format!("s={}", self.s))) }
    }

    #[test]
    fn vector_env_runs_batch() {
        let mut v = SyncVectorEnv::new(3, || DummyEnv::default());
        let _ = v.reset_all(Some(123));
        let steps = v.step_all(vec![1, 2, 3]);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].observation, 1);
        assert_eq!(steps[1].observation, 2);
        assert_eq!(steps[2].observation, 3);
        let frames = v.render_all();
        assert_eq!(frames.len(), 3);
        assert!(matches!(frames[0], Some(RenderFrame::Text(_))));
        v.close_all();
    }
}
