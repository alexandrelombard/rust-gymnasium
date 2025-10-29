// Re-export core definitions by including the single-source file at the crate root.
// This keeps implementation in one place while exposing it as `crate::core`.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/core.rs"));
