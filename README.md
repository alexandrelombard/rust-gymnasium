# Gymnasium for Rust

This repository aims to be a pure Rust port of the Python Gymnasium library. The goal is API parity where it makes sense in Rust, performance and safety guarantees idiomatic to Rust, and first‑class support for RL research and production environments.

## Porting Plan (Pure Rust)

The plan below outlines phases, scope, architecture, milestones, risks, and validation strategy to deliver a production‑ready Rust port. Changes will be kept incremental and continuously releasable.

### 1. Scope and Non‑Goals
- In scope
  - Core Env interface: reset/step/render/close/seeding, info dicts, termination/truncation semantics
  - Spaces: Box, Discrete, MultiBinary, MultiDiscrete, Dict, Tuple
  - Wrappers: ObservationWrapper, ActionWrapper, RewardWrapper, TransformObservation/Action/Reward, TimeLimit, ClipAction, ClipReward, RecordEpisodeStatistics
  - Vectorized environments: Sync/Async vector APIs (initially sync)
  - Seeding and RNG: gymnasium.utils.seeding equivalent with reproducible PRNG streams
  - Registration API: make(id)/registry and EnvSpec with metadata
  - Compatibility utilities: datatypes and conversions for common numeric backends (ndarray, nalgebra)
  - Examples: classic control environments (CartPole, MountainCar, Acrobot) with tests
- Non‑goals (initially)
  - MPI/distributed features
  - Rendering backends beyond minimal (e.g., only text + optional pixels buffer)
  - Full Atari/Mujoco integrations (can be separate crates)

### 2. Architecture Overview
- Crate layout (monorepo single crate initially; later workspace split if needed):
  - src/lib.rs: prelude and top‑level exports
  - envs/: base traits, core environments, registry
  - spaces/: space definitions and sampling
  - wrappers/: type‑safe wrapper layer with composition
  - vector/: synchronous vector envs (async as follow‑up)
  - utils/: seeding, error types, time, statistics
- Design principles
  - Zero‑cost abstractions: dynamic dispatch optional; prefer generics with trait objects where needed
  - No_std ready design where practical, with std features for I/O and rendering
  - Determinism first: explicit RNG streams passed or owned per env
  - Strong typing: typed observations/actions where possible, with an untyped fallback (DynSpace)

### 3. Core Traits and Types
- Env trait
  - fn reset(&mut self, seed: Option<u64>) -> (Obs, Info)
  - fn step(&mut self, action: Act) -> Step<Obs>
  - fn render(&self) -> Option<RenderFrame>
  - fn close(&mut self)
  - Associated types: Obs, Act
- Step struct
  - observation: Obs
  - reward: f32 (configurable via type parameter later)
  - terminated: bool
  - truncated: bool
  - info: Info (map‑like)
- Info type
  - Lightweight map: smallvec‑optimized Vec<(Key, Value)> initially; serde‑friendly
- Error model
  - GymError enum for recoverable errors; Result types on API boundaries

### 4. Spaces
- BoxSpace<T, const N: usize>: contiguous numeric spaces with shape and bounds
- Discrete: u32 range [0, n)
- MultiBinary: fixed‑length bit vector
- MultiDiscrete: per‑dimension discrete ranges
- TupleSpace / DictSpace: composite spaces with nested sampling and validation
- Sampling: uses RNG trait; no global state
- Validation: validate(action/observation) -> bool with descriptive errors

### 5. Seeding and RNG
- utils::rng with:
  - SeedSequence equivalent that expands u64 seed to per‑stream keys
  - Deterministic PRNG (e.g., PCG32 or ChaCha8) via rand_chacha
  - Reproducible sampling across spaces and environments
- API
  - Env::reset(seed) re‑seeds the internal RNG
  - Vector env creates sub‑streams for each worker

### 6. Wrappers
- Base traits: ObservationWrapper, ActionWrapper, RewardWrapper; unified Wrapper<E: Env>
- Provided wrappers (Phase 1)
  - TimeLimit
  - ClipAction
  - ClipReward
  - TransformObservation/Action/Reward (user closures)
  - RecordEpisodeStatistics
- Composition model: Wrapper<W<Env>> with newtype pattern; minimal dynamic dispatch with trait objects for heterogeneous pipelines

### 7. Vector Environments
- vector::SyncVectorEnv<E: Env>
  - Runs N copies in a single thread loop or rayon parallel iterator (feature‑gated)
  - Batched step taking Vec<Act> and returning Vec<Step<Obs>>
- vector::AsyncVectorEnv (Phase 2)
  - Crossbeam channel workers per env; builder for thread pool size

### 8. Registration and Specs
- Registry with id -> EnvSpec and factory closures
- make(id, kwargs) -> Box<dyn EnvDyn>
- EnvSpec fields: id, max_episode_steps, reward_threshold, nondeterministic, order_enforce, version
- Serialization via serde for specs

### 9. Rendering
- Minimal rendering trait returning either text frame or pixel buffer (RGB/RGBA)
- Feature gates for image encoders; no GUI dependency by default

### 10. Numeric Backends Interop
- Traits to convert BoxSpace payloads to/from ndarray and nalgebra when features enabled
- Default payloads as Vec<T> or small arrays; avoid hard dependency on ndarray

### 11. Testing and Validation Strategy
- Unit tests
  - Space sampling determinism with fixed seeds
  - Env step/reset contracts and episode termination behavior
  - Wrapper invariants (e.g., TimeLimit enforces truncation)
- Integration tests
  - CartPole-v1 reference scores over N seeds within tolerance
  - Vector env consistency vs single env rollouts
- Property tests (proptest) for space validation and sampling

### 12. Documentation and Examples
- Doc comments and module‑level guides
- Examples folder with:
  - cartpole_basic.rs
  - wrappers_demo.rs
  - vector_sync.rs

### 13. Milestones and Deliverables
- M0: Repo scaffolding, plan, CI, rustfmt/clippy, basic crate exports (current)
- M1: Core traits, Step type, error types, utils::rng, Box/Discrete spaces; simple dummy env ✓ deliver: 0.1.0-alpha.1
- M2: Tuple/Dict/Multi* spaces, validation, serde; initial wrappers; CartPole env ✓ deliver: 0.1.0-alpha.2
- M3: SyncVectorEnv + RecordEpisodeStatistics; MountainCar, Acrobot ✓ deliver: 0.1.0-alpha.3
- M4: Registry/make API; rendering frames; docs and examples ✓ deliver: 0.1.0
- M5: AsyncVectorEnv, backend interop, feature gates, benchmarks ✓ deliver: 0.2.0

### 14. CI and Tooling
- CI: check, test, clippy -D warnings, fmt, doc
- Optional MSRV: 1.75+
- Features: std (default), rayon, ndarray, nalgebra, serde, image

### 15. Risks and Mitigations
- API drift vs Python: provide reference docs and mapping table; keep changelog
- Performance regressions: microbenchmarks, criterion
- Complexity of wrappers generics: offer dyn Env for ergonomics when needed

### 16. Contribution Guide (short)
- Prefer small PRs per milestone item
- Include tests and docs for each feature
- Keep determinism and safety as priorities

---

## Current Status
- Skeleton crate with placeholder modules in envs/, spaces/, wrappers/, vector/, utils/
- Next up: implement core traits and utils::rng per M1

## License
Licensed under the Apache-2.0 or MIT license, at your option.