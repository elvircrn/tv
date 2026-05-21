#![no_main]
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    let _ = trace_viewer::find_key(data, b"traceEvents");
    let _ = trace_viewer::find_key(data, b"ph");
    let _ = trace_viewer::find_key(data, b"name");
    let _ = trace_viewer::find_key(data, b"ts");
    let _ = trace_viewer::find_key(data, b"dur");
    let _ = trace_viewer::find_key(data, b"cat");
    let _ = trace_viewer::find_key(data, b"args");

    let mut strs = Vec::new();
    let mut idx = HashMap::new();
    let mut pairs = Vec::new();
    trace_viewer::parse_args_flat(data, &mut strs, &mut idx, &mut pairs);
});
