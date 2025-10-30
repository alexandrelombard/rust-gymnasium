// Registration and Specs (README Step 8)
/// Minimal registry system to construct environments by id with associated EnvSpec.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::core::{Env, Info, RenderFrame, Result, Step};
use crate::core::GymError;

/// Key-value kwargs for make(). Keep simple for now: stringly-typed values.
pub type KwArgs = HashMap<String, String>;

/// Environment specification metadata.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvSpec {
    /// Unique identifier like "CartPole-v1".
    pub id: String,
    /// Suggested max episode steps for TimeLimit wrapper.
    pub max_episode_steps: Option<u32>,
    /// Target reward threshold for a "solved" score, if defined.
    pub reward_threshold: Option<f32>,
    /// Whether environment has inherent nondeterminism beyond RNG seed.
    pub nondeterministic: bool,
    /// Whether to enforce order between reset/step/etc.
    pub order_enforce: bool,
    /// Version string or semver-like number (free-form for now).
    pub version: Option<String>,
}

impl EnvSpec {
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self {
            id: id.into(),
            max_episode_steps: None,
            reward_threshold: None,
            nondeterministic: false,
            order_enforce: true,
            version: None,
        }
    }
}

/// A type-erased environment trait to allow Box<dyn EnvDyn> results from make().
pub trait EnvDyn {
    fn reset(&mut self, seed: Option<u64>) -> (Box<dyn Any>, Info);
    fn step(&mut self, action: Box<dyn Any>) -> Step<Box<dyn Any>>;
    fn render(&self) -> Option<RenderFrame>;
    fn close(&mut self);
}

/// Wrapper to adapt any Env into EnvDyn by boxing Obs/Act via Any.
struct DynEnv<E: Env>(E);

impl<E: Env> EnvDyn for DynEnv<E>
where
    E::Obs: Any + 'static,
    E::Act: Any + 'static,
{
    fn reset(&mut self, seed: Option<u64>) -> (Box<dyn Any>, Info) {
        let (obs, info) = self.0.reset(seed);
        (Box::new(obs), info)
    }

    fn step(&mut self, action: Box<dyn Any>) -> Step<Box<dyn Any>> {
        let action = *action
            .downcast::<E::Act>()
            .map_err(|_| ())
            .expect("invalid action type for DynEnv");
        let s = self.0.step(action);
        Step::new(Box::new(s.observation) as Box<dyn Any>, s.reward, s.terminated, s.truncated, s.info)
    }

    fn render(&self) -> Option<RenderFrame> { self.0.render() }
    fn close(&mut self) { self.0.close() }
}

/// Factory closure type for constructing environments with kwargs.
pub type FactoryFn = Box<dyn Fn(KwArgs) -> Box<dyn EnvDyn + Send + Sync> + Send + Sync>;

#[derive(Default)]
struct RegistryInner {
    specs: HashMap<String, EnvSpec>,
    factories: HashMap<String, FactoryFn>,
}

struct Registry {
    inner: RwLock<RegistryInner>,
}

impl Registry {
    fn new() -> Self { Self { inner: RwLock::new(RegistryInner::default()) } }

    fn register(&self, spec: EnvSpec, factory: FactoryFn) -> Result<()> {
        let mut g = self.inner.write().map_err(|_| GymError::Other("registry poisoned".into()))?;
        if g.specs.contains_key(&spec.id) {
            return Err(GymError::Other(format!("Env id already registered: {}", spec.id)));
        }
        g.factories.insert(spec.id.clone(), factory);
        g.specs.insert(spec.id.clone(), spec);
        Ok(())
    }

    fn get_spec(&self, id: &str) -> Option<EnvSpec> {
        let g = self.inner.read().ok()?;
        g.specs.get(id).cloned()
    }

    fn make(&self, id: &str, kwargs: KwArgs) -> Result<Box<dyn EnvDyn + Send + Sync>> {
        let guard = self.inner.read().map_err(|_| GymError::Other("registry poisoned".into()))?;
        match guard.factories.get(id) {
            Some(f) => Ok((f)(kwargs)),
            None => Err(GymError::Other(format!("Unknown environment id: {}", id)))
        }
    }
}

static REGISTRY: OnceLock<Registry> = OnceLock::new();

fn registry() -> &'static Registry {
    REGISTRY.get_or_init(|| Registry::new())
}

/// Register an environment spec and its factory globally.
pub fn register(spec: EnvSpec, factory: FactoryFn) -> Result<()> { registry().register(spec, factory) }

/// Fetch a registered EnvSpec by id.
pub fn get_spec(id: &str) -> Option<EnvSpec> { registry().get_spec(id) }

/// Construct an environment by id with kwargs, returning a boxed dynamic env.
pub fn make<S: AsRef<str>>(id: S, kwargs: KwArgs) -> Result<Box<dyn EnvDyn + Send + Sync>> { registry().make(id.as_ref(), kwargs) }

/// Helper to adapt a concrete Env into a factory function easily.
pub fn factory_of<E, F>(ctor: F) -> FactoryFn
where
    E: Env + Send + Sync + 'static,
    E::Obs: Any + 'static,
    E::Act: Any + 'static,
    F: Fn(KwArgs) -> E + Send + Sync + 'static,
{
    Box::new(move |kwargs: KwArgs| {
        let env = ctor(kwargs);
        Box::new(DynEnv::<E>(env)) as Box<dyn EnvDyn + Send + Sync>
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Env, Step};

    #[derive(Default)]
    struct Dummy;
    impl Env for Dummy {
        type Obs = i32;
        type Act = i32;
        fn reset(&mut self, _seed: Option<u64>) -> (Self::Obs, Info) { (0, Info::new()) }
        fn step(&mut self, a: Self::Act) -> Step<Self::Obs> { Step::new(a, 0.0, true, false, Info::new()) }
        fn render(&self) -> Option<RenderFrame> { Some(RenderFrame::Text("dummy".into())) }
        fn close(&mut self) {}
    }

    #[test]
    fn register_and_make_dummy() {
        let spec = EnvSpec { id: "Dummy-v0".into(), max_episode_steps: Some(10), reward_threshold: None, nondeterministic: false, order_enforce: true, version: Some("0".into()) };
        register(spec.clone(), factory_of::<Dummy, _>(|_k| Dummy::default())).expect("register ok");
        let mut env = make("Dummy-v0", KwArgs::new()).expect("make ok");
        let (obs, _info) = env.reset(None);
        assert!(obs.downcast_ref::<i32>().is_some());
        let s = env.step(Box::new(5));
        assert!(s.observation.downcast_ref::<i32>() == Some(&5));
        assert!(matches!(env.render(), Some(RenderFrame::Text(_))));
    }
}
