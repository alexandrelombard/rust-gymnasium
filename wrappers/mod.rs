// Wrappers module (Step 6 of README): base wrappers for Env composition.
//
// Provided wrappers in this first pass:
// - TimeLimit
// - ClipAction
// - ClipReward
// - TransformObservation / TransformAction / TransformReward
// - RecordEpisodeStatistics

use crate::core::{Env, Info, InfoValue, Step};

/// A wrapper that enforces a maximum number of steps per episode, marking truncation when exceeded.
pub struct TimeLimit<E: Env> {
    inner: E,
    max_steps: u32,
    steps: u32,
}

impl<E: Env> TimeLimit<E> {
    pub fn new(inner: E, max_steps: u32) -> Self {
        Self { inner, max_steps, steps: 0 }
    }

    pub fn inner(&self) -> &E { &self.inner }
    pub fn inner_mut(&mut self) -> &mut E { &mut self.inner }
    pub fn into_inner(self) -> E { self.inner }
}

impl<E: Env> Env for TimeLimit<E> {
    type Obs = E::Obs;
    type Act = E::Act;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        self.steps = 0;
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let mut s = self.inner.step(action);
        self.steps += 1;
        if !s.terminated && !s.truncated && self.steps >= self.max_steps {
            s.truncated = true;
        }
        s
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// ClipAction clamps scalar actions to a range [min, max]. Useful for continuous controls.
/// For non-scalar actions, prefer TransformAction to perform custom clipping.
pub struct ClipAction<E: Env, A: PartialOrd + Copy> {
    inner: E,
    min: A,
    max: A,
    _marker: core::marker::PhantomData<A>,
}

impl<E, A> ClipAction<E, A>
where
    E: Env<Act = A>,
    A: PartialOrd + Copy,
{
    pub fn new(inner: E, min: A, max: A) -> Self {
        Self { inner, min, max, _marker: core::marker::PhantomData }
    }
}

impl<E, A> Env for ClipAction<E, A>
where
    E: Env<Act = A>,
    A: PartialOrd + Copy,
{
    type Obs = E::Obs;
    type Act = A;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) { self.inner.reset(seed) }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let a = if action < self.min { self.min } else if action > self.max { self.max } else { action };
        self.inner.step(a)
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// ClipReward clamps rewards into [min, max].
pub struct ClipReward<E: Env> {
    inner: E,
    min: f32,
    max: f32,
}

impl<E: Env> ClipReward<E> {
    pub fn new(inner: E, min: f32, max: f32) -> Self { Self { inner, min, max } }
}

impl<E: Env> Env for ClipReward<E> {
    type Obs = E::Obs;
    type Act = E::Act;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) { self.inner.reset(seed) }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let mut s = self.inner.step(action);
        if s.reward < self.min { s.reward = self.min; }
        if s.reward > self.max { s.reward = self.max; }
        s
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// TransformObservation maps an environment's observations through a user-provided function.
pub struct TransformObservation<E, F, O2>
where
    E: Env,
    F: Fn(&E::Obs) -> O2,
{
    inner: E,
    f: F,
    _marker: core::marker::PhantomData<O2>,
}

impl<E, F, O2> TransformObservation<E, F, O2>
where
    E: Env,
    F: Fn(&E::Obs) -> O2,
{
    pub fn new(inner: E, f: F) -> Self { Self { inner, f, _marker: core::marker::PhantomData } }
}

impl<E, F, O2> Env for TransformObservation<E, F, O2>
where
    E: Env,
    F: Fn(&E::Obs) -> O2,
{
    type Obs = O2;
    type Act = E::Act;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        let (obs, info) = self.inner.reset(seed);
        ((self.f)(&obs), info)
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let s = self.inner.step(action);
        Step::new((self.f)(&s.observation), s.reward, s.terminated, s.truncated, s.info)
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// TransformAction maps caller-provided actions into the inner environment's action type.
pub struct TransformAction<E, F, A2>
where
    E: Env,
    F: Fn(A2) -> E::Act,
{
    inner: E,
    f: F,
    _marker: core::marker::PhantomData<A2>,
}

impl<E, F, A2> TransformAction<E, F, A2>
where
    E: Env,
    F: Fn(A2) -> E::Act,
{
    pub fn new(inner: E, f: F) -> Self { Self { inner, f, _marker: core::marker::PhantomData } }
}

impl<E, F, A2> Env for TransformAction<E, F, A2>
where
    E: Env,
    F: Fn(A2) -> E::Act,
    A2: Clone,
{
    type Obs = E::Obs;
    type Act = A2;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) { self.inner.reset(seed) }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let inner_action = (self.f)(action);
        self.inner.step(inner_action)
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// TransformReward maps rewards through a user-provided function (e.g., scaling).
pub struct TransformReward<E, F>
where
    E: Env,
    F: Fn(f32) -> f32,
{
    inner: E,
    f: F,
}

impl<E, F> TransformReward<E, F>
where
    E: Env,
    F: Fn(f32) -> f32,
{
    pub fn new(inner: E, f: F) -> Self { Self { inner, f } }
}

impl<E, F> Env for TransformReward<E, F>
where
    E: Env,
    F: Fn(f32) -> f32,
{
    type Obs = E::Obs;
    type Act = E::Act;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) { self.inner.reset(seed) }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let mut s = self.inner.step(action);
        s.reward = (self.f)(s.reward);
        s
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

/// RecordEpisodeStatistics tracks cumulative return and episode length.
/// On episode end (terminated or truncated), it injects keys into the returned Step's Info:
/// - "episode_return": f64
/// - "episode_length": i64
pub struct RecordEpisodeStatistics<E: Env> {
    inner: E,
    ep_return: f64,
    ep_length: i64,
}

impl<E: Env> RecordEpisodeStatistics<E> {
    pub fn new(inner: E) -> Self { Self { inner, ep_return: 0.0, ep_length: 0 } }
}

impl<E: Env> Env for RecordEpisodeStatistics<E> {
    type Obs = E::Obs;
    type Act = E::Act;

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        self.ep_return = 0.0;
        self.ep_length = 0;
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let mut s = self.inner.step(action);
        self.ep_return += s.reward as f64;
        self.ep_length += 1;
        if s.terminated || s.truncated {
            let mut info = s.info;
            info.insert("episode_return", InfoValue::from(self.ep_return));
            info.insert("episode_length", InfoValue::from(self.ep_length));
            s.info = info;
            // reset counters for next episode
            self.ep_return = 0.0;
            self.ep_length = 0;
        }
        s
    }

    fn render(&self) -> Option<crate::core::RenderFrame> { self.inner.render() }
    fn close(&mut self) { self.inner.close() }
}

// Re-exports for convenience
pub use {
    ClipAction as _ClipAction,
    ClipReward as _ClipReward,
    RecordEpisodeStatistics as _RecordEpisodeStatistics,
    TimeLimit as _TimeLimit,
    TransformAction as _TransformAction,
    TransformObservation as _TransformObservation,
    TransformReward as _TransformReward,
};
