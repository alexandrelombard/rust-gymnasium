// Re-export registry implementation by including the single-source file at the crate root.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/registry.rs"));
