// Re-export common Space trait by including single-source file at the crate root.
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/spaces/space.rs"));
