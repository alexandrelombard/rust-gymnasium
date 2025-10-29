// Core traits and types for Rust Gymnasium (Step 3 of README)

/// A minimal, serde-friendly info map (without pulling serde as a dependency).
/// It stores small numbers of key-value pairs and is sufficient for early phases.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Info {
    entries: Vec<(String, InfoValue)>,
}

impl Info {
    /// Create an empty Info map.
    pub fn new() -> Self { Self { entries: Vec::new() } }

    /// Insert or replace a key with the given value.
    pub fn insert<K: Into<String>>(&mut self, key: K, value: InfoValue) {
        let k = key.into();
        if let Some((_, v)) = self.entries.iter_mut().find(|(kk, _)| kk == &k) {
            *v = value;
        } else {
            self.entries.push((k, value));
        }
    }

    /// Get a reference to a value by key.
    pub fn get(&self, key: &str) -> Option<&InfoValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &InfoValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Number of entries.
    pub fn len(&self) -> usize { self.entries.len() }
}

/// A small set of value types commonly used in info maps.
#[derive(Clone, Debug, PartialEq)]
pub enum InfoValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
}

impl From<bool> for InfoValue { fn from(v: bool) -> Self { InfoValue::Bool(v) } }
impl From<i64> for InfoValue { fn from(v: i64) -> Self { InfoValue::I64(v) } }
impl From<i32> for InfoValue { fn from(v: i32) -> Self { InfoValue::I64(v as i64) } }
impl From<f64> for InfoValue { fn from(v: f64) -> Self { InfoValue::F64(v) } }
impl From<f32> for InfoValue { fn from(v: f32) -> Self { InfoValue::F64(v as f64) } }
impl From<&str> for InfoValue { fn from(v: &str) -> Self { InfoValue::Str(v.to_string()) } }
impl From<String> for InfoValue { fn from(v: String) -> Self { InfoValue::Str(v) } }

/// A frame returned by `Env::render`.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderFrame {
    /// Textual representation of a frame (e.g., ASCII art or debug string).
    Text(String),
    /// Raw pixel buffer in row-major RGB or RGBA format.
    Pixels {
        width: u32,
        height: u32,
        /// Pixel data. Convention: RGB uses 3 bytes per pixel, RGBA uses 4.
        data: Vec<u8>,
    },
}

/// A step result from the environment.
#[derive(Clone, Debug, PartialEq)]
pub struct Step<Obs> {
    pub observation: Obs,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Info,
}

impl<Obs> Step<Obs> {
    pub fn new(observation: Obs, reward: f32, terminated: bool, truncated: bool, info: Info) -> Self {
        Self { observation, reward, terminated, truncated, info }
    }
}

/// Recoverable errors across Gymnasium APIs.
#[derive(thiserror::Error, Debug)]
pub enum GymError {
    #[error("Invalid action: {0}")]
    InvalidAction(String),
    #[error("Invalid observation: {0}")]
    InvalidObservation(String),
    #[error("Environment not ready: {0}")]
    NotReady(String),
    #[error("Operation not supported: {0}")]
    NotSupported(String),
    #[error("Other error: {0}")]
    Other(String),
}

/// Convenience alias for results using GymError.
pub type Result<T> = std::result::Result<T, GymError>;

/// Core environment trait following the Gymnasium contract.
pub trait Env {
    type Obs;
    type Act;

    /// Reset the environment to an initial state.
    /// Implementations should re-seed internal RNGs when `seed` is provided.
    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info);

    /// Apply an action and advance the environment by one step.
    fn step(&mut self, action: Self::Act) -> Step<Self::Obs>;

    /// Render a frame of the current state, if supported.
    fn render(&self) -> Option<RenderFrame> { None }

    /// Close and release any external resources.
    fn close(&mut self) {}
}
