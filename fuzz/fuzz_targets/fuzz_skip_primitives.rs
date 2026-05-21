#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 { return; }
    let pos = data[0] as usize % data.len();

    let _ = trace_viewer::find_key(data, b"traceEvents");
    let _ = trace_viewer::find_key(data, b"name");

    let _ = trace_viewer::skip_ws(data, pos);
    let _ = trace_viewer::skip_value(data, pos);

    if pos < data.len() && data[pos] == b'"' {
        let _ = trace_viewer::skip_string(data, pos);
    }

    let _ = trace_viewer::parse_f64(data);

    let mut strs = Vec::new();
    let mut idx = std::collections::HashMap::new();
    let mut pairs = Vec::new();
    trace_viewer::parse_args_flat(data, &mut strs, &mut idx, &mut pairs);
});
