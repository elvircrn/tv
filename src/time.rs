// Bare wasm32-unknown-unknown has no OS clock; std::time::Instant::now()
// panics there. web-time is API-compatible, backed by performance.now() on
// wasm and re-exporting std::time everywhere else.
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;
