use super::*;
use crate::parse::*;
use crate::loader::{load_trace, detect_rank_groups, merge_traces};
use crate::state::parse_rank;
use imgui::ImColor32;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

fn test_counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn ev(ts: f64, dur: f64, name: u32, depth: u16) -> Event {
    Event { ts, dur, name, cat: 0, args_off: 0, args_len: 0, depth }
}

fn make_trace(names: Vec<&str>, tracks: Vec<(&str, bool, Vec<Event>)>) -> Trace {
    let name_strs: Vec<String> = names.into_iter().map(String::from).collect();
    let mut trs = Vec::new();
    let mut max_ts: f64 = 0.0;
    let mut total_events = 0;
    for (label, gpu, events) in tracks {
        for e in &events {
            let end = e.ts + e.dur;
            if end > max_ts { max_ts = end; }
        }
        let max_depth = events.iter().map(|e| e.depth + 1).max().unwrap_or(1);
        let mut prefix_max_dur = Vec::with_capacity(events.len());
        let mut running_max = 0.0f64;
        for ev in &events {
            running_max = running_max.max(ev.dur);
            prefix_max_dur.push(running_max);
        }
        total_events += events.len();
        trs.push(Track { label: label.to_string(), gpu, events, max_depth, prefix_max_dur, raw_buf_idx: 0 });
    }
    Trace {
        tracks: trs, names: name_strs, cats: vec![String::new()],
        raw_bufs: Vec::new(), stats: Vec::new(),
        max_ts, min_ts: 0.0, total_events, device: String::new(),
        flow_pairs: Vec::new(),
    }
}

fn make_state(trace: Trace) -> AppState {
    let hidden_names = vec![false; trace.names.len()];
    let collapsed = vec![false; trace.tracks.len()];
    let mut buf = DrawBuf::default();
    let mut cum = 0.0f32;
    for (i, t) in trace.tracks.iter().enumerate() {
        buf.visible.push(i);
        let h = track_height(t.max_depth, false, 1.0);
        buf.heights.push(h);
        buf.y_offsets.push(cum);
        cum += h;
    }
    let mut pane = Pane::new();
    // Mirror the layout into the pane-owned geom, exactly as draw_timeline's
    // per-frame snapshot does. Selection/diff/copy read pane.geom, not buf.
    pane.geom.visible = buf.visible.clone();
    pane.geom.heights = buf.heights.clone();
    pane.geom.y_offsets = buf.y_offsets.clone();
    pane.hidden_names = hidden_names;
    pane.collapsed = collapsed;
    pane.trace = Some(trace);
    AppState {
        panes: vec![pane],
        active: 0,
        divider_xs: Vec::new(),
        buf,
        bottom_h: DETAIL_H,
        drag: DragKind::None,
        show_diff: false,
        diff_popup_open: false,
        diff_result: None,
        diff_bar_scroll: 0.0,
        diff_bar_zoom: 1.0,
        diff_pane_indices: None,
    }
}

// --- JSON scanning primitives ---

#[test]
fn test_find_key() {
    let raw = br#"{"traceEvents": [], "other": 1}"#;
    assert_eq!(find_key(raw, b"traceEvents"), Some(1));
    assert_eq!(find_key(raw, b"other"), Some(20));
    assert_eq!(find_key(raw, b"missing"), None);
}

#[test]
fn test_find_key_multiple() {
    let raw = br#"{"a": 1, "traceEvents": []}"#;
    let pos = find_key(raw, b"traceEvents").unwrap();
    assert_eq!(&raw[pos..pos + 13], br#""traceEvents""#);
}

#[test]
fn test_skip_string() {
    let raw = br#""hello" rest"#;
    assert_eq!(skip_string(raw, 0), 7);
    let escaped = br#""he\"llo" rest"#;
    assert_eq!(skip_string(escaped, 0), 9);
}

#[test]
fn test_skip_number() {
    assert_eq!(skip_number(b"123.45,", 0), 6);
    assert_eq!(skip_number(b"-1.5e10}", 0), 7);
    assert_eq!(skip_number(b"0 ", 0), 1);
}

#[test]
fn test_skip_value_nested() {
    let raw = br#"{"a": [1, 2]}, next"#;
    assert_eq!(skip_value(raw, 0), 13);
}

#[test]
fn test_skip_ws_comma() {
    assert_eq!(skip_ws_comma(b"  , , x", 0), 6);
    assert_eq!(skip_ws_comma(b"abc", 0), 0);
}

#[test]
fn test_parse_f64() {
    assert_eq!(parse_f64(b"123.5"), 123.5);
    assert_eq!(parse_f64(b"-1e3"), -1000.0);
    assert_eq!(parse_f64(b""), 0.0);
    assert_eq!(parse_f64(b"garbage"), 0.0);
    assert_eq!(parse_f64(b"1716234567890123"), 1716234567890123.0);
    assert_eq!(parse_f64(b"123456.789"), 123456.789);
    assert_eq!(parse_f64(b"1.5e-3"), 0.0015);
    assert_eq!(parse_f64(b"1e6"), 1000000.0);
    assert_eq!(parse_f64(b"0"), 0.0);
    assert_eq!(parse_f64(b"-42.5"), -42.5);
    assert_eq!(parse_f64(b"1.5E+3"), 1500.0);
    assert_eq!(parse_f64(b".5"), 0.5);
}

// --- JSON unescape ---

#[test]
fn test_json_unescape_roundtrip() {
    let cases = ["hello", "with\\\"quotes", "back\\\\slash", "new\\nline", ""];
    let expected = ["hello", "with\"quotes", "back\\slash", "new\nline", ""];
    for (inner, want) in cases.iter().zip(expected.iter()) {
        assert_eq!(json_unescape(inner), *want);
    }
}

// --- Hashing ---

#[test]
fn test_fnv1a_deterministic() {
    assert_eq!(fnv1a(b"hello"), fnv1a(b"hello"));
    assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
}

#[test]
fn test_fnv1a_empty() {
    assert_eq!(fnv1a(b""), 14695981039346656037);
}

// --- Interning ---

#[test]
fn test_intern_dedup() {
    let mut table = Vec::new();
    let mut index = FnvMap::default();
    let a = intern(b"kernel_a", &mut table, &mut index);
    let b = intern(b"kernel_b", &mut table, &mut index);
    let a2 = intern(b"kernel_a", &mut table, &mut index);
    assert_eq!(a, a2);
    assert_ne!(a, b);
    assert_eq!(table.len(), 2);
}

// --- Time formatting ---

#[test]
fn test_write_time_ns() {
    let mut buf = String::new();
    write_time(&mut buf, 0.5);
    assert_eq!(buf, "500ns");
}

#[test]
fn test_write_time_us() {
    let mut buf = String::new();
    write_time(&mut buf, 42.7);
    assert_eq!(buf, "42.7us");
}

#[test]
fn test_write_time_ms() {
    let mut buf = String::new();
    write_time(&mut buf, 1500.0);
    assert_eq!(buf, "1.50ms");
}

#[test]
fn test_write_time_s() {
    let mut buf = String::new();
    write_time(&mut buf, 2_500_000.0);
    assert_eq!(buf, "2.500s");
}

#[test]
fn test_write_time_zero() {
    let mut buf = String::new();
    write_time(&mut buf, 0.0);
    assert_eq!(buf, "0");
}

// --- nice_interval ---

#[test]
fn test_nice_interval_zero() {
    assert_eq!(nice_interval(0.0), 1.0);
    assert_eq!(nice_interval(-10.0), 1.0);
}

#[test]
fn test_nice_interval_values() {
    let n = nice_interval(100.0);
    assert!(n == 10.0 || n == 20.0 || n == 50.0,
        "nice_interval(100) = {n}, expected 10/20/50");
}

// --- Color functions ---

#[test]
fn test_name_color_deterministic() {
    let c1 = name_color("kernel_abc");
    let c2 = name_color("kernel_abc");
    assert_eq!(Into::<u32>::into(c1), Into::<u32>::into(c2));
}

#[test]
fn test_name_color_darkened() {
    let c: u32 = name_color("x").into();
    let r = c & 0xFF;
    let g = (c >> 8) & 0xFF;
    let b = (c >> 16) & 0xFF;
    assert!(r <= 140 && g <= 140 && b <= 140, "colors should be darkened to <=140");
}

#[test]
fn test_brighten() {
    let c = col32(100, 100, 100, 255);
    let b: u32 = brighten(c, 30).into();
    assert_eq!(b & 0xFF, 130);
    assert_eq!((b >> 8) & 0xFF, 130);
}

#[test]
fn test_brighten_saturates() {
    let c = col32(250, 250, 250, 255);
    let b: u32 = brighten(c, 30).into();
    assert_eq!(b & 0xFF, 255);
}

// --- Selection stats ---

#[test]
fn test_selection_stats_basic() {
    let names = vec!["", "A", "B"];
    let events = vec![ev(0.0, 10.0, 1, 0), ev(10.0, 20.0, 2, 0), ev(30.0, 10.0, 1, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([0.0, 40.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);

    assert_eq!(p.selection_stats.len(), 2);
    let a = p.selection_stats.iter().find(|s| s.name == 1).unwrap();
    assert_eq!(a.count, 2);
    assert_eq!(a.total_dur, 20.0);
}

#[test]
fn test_selection_stats_respects_hidden() {
    let names = vec!["", "A", "B"];
    let events = vec![ev(0.0, 10.0, 1, 0), ev(10.0, 20.0, 2, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.hidden_names[1] = true;
    p.selection = Some([0.0, 30.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);

    assert_eq!(p.selection_stats.len(), 1);
    assert_eq!(p.selection_stats[0].name, 2);
}

#[test]
fn test_selection_stats_cpu_hidden() {
    let names = vec!["", "gpu_kern", "cpu_kern"];
    let trace = make_trace(names, vec![
        ("GPU 0", true, vec![ev(0.0, 10.0, 1, 0)]),
        ("Thread 1", false, vec![ev(0.0, 10.0, 2, 0)]),
    ]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.show_cpu = false;
    p.selection = Some([0.0, 10.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);

    assert_eq!(p.selection_stats.len(), 1);
    assert_eq!(p.selection_stats[0].name, 1);
}

// --- Selection state machine ---
// Invariants:
//   During drag:  selection=Some, finished_sel=None  (rectangle + highlights)
//   After release: selection=None, finished_sel=Some  (highlights only, no rectangle)
//   After clear:   selection=None, finished_sel=None  (nothing)

#[test]
fn test_finish_selection_clears_active_keeps_finished() {
    let names = vec!["", "A", "B"];
    let events = vec![ev(0.0, 10.0, 1, 0), ev(10.0, 20.0, 2, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];

    p.selection = Some([0.0, 30.0, 0.0, 1e9]);
    assert!(p.finished_sel.is_none());

    p.finish_selection(&mut state.buf);

    assert!(p.selection.is_none(), "active selection must be cleared on finish");
    assert!(p.finished_sel.is_some(), "finished_sel must be set on finish");
    assert_eq!(p.finished_sel.unwrap(), [0.0, 30.0, 0.0, 1e9]);
    assert!(!p.selection_stats.is_empty(), "stats must be computed from finished_sel");
}

#[test]
fn test_clear_selection_clears_both() {
    let names = vec!["", "A"];
    let events = vec![ev(0.0, 10.0, 1, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];

    p.selection = Some([0.0, 10.0, 0.0, 1e9]);
    p.finish_selection(&mut state.buf);
    assert!(p.finished_sel.is_some());

    p.clear_selection();

    assert!(p.selection.is_none(), "active selection must be cleared");
    assert!(p.finished_sel.is_none(), "finished_sel must be cleared");
    assert!(p.selection_stats.is_empty(), "stats must be cleared");
}

#[test]
fn test_finish_selection_stats_match_region() {
    let names = vec!["", "A", "B"];
    let events = vec![
        ev(0.0, 10.0, 1, 0), ev(10.0, 10.0, 2, 0),
        ev(20.0, 10.0, 1, 0), ev(30.0, 10.0, 2, 0),
    ];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];

    p.selection = Some([0.0, 15.0, 0.0, 1e9]);
    p.finish_selection(&mut state.buf);

    let names_in_stats: Vec<u32> = p.selection_stats.iter().map(|s| s.name).collect();
    assert!(names_in_stats.contains(&1), "A is in the selected region");
    assert!(names_in_stats.contains(&2), "B overlaps the selected region");

    let a = p.selection_stats.iter().find(|s| s.name == 1).unwrap();
    assert_eq!(a.count, 1, "only one A event is in the region, not all As");
}

// --- Search ---

#[test]
fn test_rebuild_search() {
    let names = vec!["", "attention_kernel", "moe_kernel", "allreduce"];
    let events = vec![
        ev(0.0, 10.0, 1, 0), ev(10.0, 10.0, 2, 0), ev(20.0, 10.0, 3, 0),
    ];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.search = "kernel".to_string();
    p.rebuild_search();

    assert!(p.search_mask[1]);
    assert!(p.search_mask[2]);
    assert!(!p.search_mask[3]);
    assert_eq!(p.search_nav.len(), 2);
}

#[test]
fn test_rebuild_search_case_insensitive() {
    let names = vec!["", "AttentionKernel"];
    let events = vec![ev(0.0, 10.0, 1, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.search = "attention".to_string();
    p.rebuild_search();

    assert!(p.search_mask[1]);
}

#[test]
fn test_rebuild_search_empty() {
    let names = vec!["", "A"];
    let events = vec![ev(0.0, 10.0, 1, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.search = "  ".to_string();
    p.rebuild_search();

    assert!(p.search_mask.iter().all(|&m| !m));
}

// --- Trace loading (file-based) ---

#[test]
fn test_load_trace_json() {
    let json = r#"{"traceEvents": [
        {"ph":"X","ts":100,"dur":50,"pid":1,"tid":1,"name":"kern_a","cat":"kernel","args":{"op":"matmul"}},
        {"ph":"X","ts":200,"dur":30,"pid":1,"tid":1,"name":"kern_b","cat":"kernel"},
        {"ph":"X","ts":300,"dur":40,"pid":1,"tid":2,"name":"cpu_fn","cat":"cpu_op"},
        {"ph":"M","ts":0,"pid":1,"tid":1,"name":"thread_name","args":{"name":"GPU Stream 0"}}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_load");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test.json");
    std::fs::write(&path, json).unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    assert_eq!(trace.tracks.len(), 2);
    assert!(trace.names.contains(&"kern_a".to_string()));
    assert!(trace.names.contains(&"kern_b".to_string()));
    assert!(trace.names.contains(&"cpu_fn".to_string()));
    assert_eq!(trace.total_events, 3);
    assert!(trace.max_ts > 0.0);

    let gpu_track = trace.tracks.iter().find(|t| t.label == "GPU Stream 0").unwrap();
    assert!(gpu_track.gpu);
    assert_eq!(gpu_track.events.len(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_trace_gz() {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let json = br#"{"traceEvents": [
        {"ph":"X","ts":0,"dur":100,"pid":1,"tid":1,"name":"k","cat":"kernel"}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_gz");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test.json.gz");
    let file = std::fs::File::create(&path).unwrap();
    let mut gz = GzEncoder::new(file, Compression::fast());
    std::io::Write::write_all(&mut gz, json).unwrap();
    gz.finish().unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    assert_eq!(trace.total_events, 1);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_trace_no_events() {
    let json = r#"{"traceEvents": []}"#;
    let dir = std::env::temp_dir().join("tv_test_empty");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("empty.json");
    std::fs::write(&path, json).unwrap();

    let result = load_trace(path.to_str().unwrap(), &test_counter(), 0, None);
    assert!(result.is_err());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_cache_roundtrip() {
    let json = r#"{"traceEvents": [
        {"ph":"X","ts":100,"dur":50,"pid":1,"tid":1,"name":"kern_a","cat":"kernel","args":{"op":"matmul"}},
        {"ph":"X","ts":200,"dur":30,"pid":1,"tid":1,"name":"kern_b","cat":"kernel"},
        {"ph":"X","ts":300,"dur":40,"pid":1,"tid":2,"name":"cpu_fn","cat":"cpu_op"},
        {"ph":"M","ts":0,"pid":1,"tid":1,"name":"thread_name","args":{"name":"GPU Stream 0"}}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_cache");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("cache_test.json");
    std::fs::write(&path, json).unwrap();
    let path_str = path.to_str().unwrap();

    let original = load_trace(path_str, &test_counter(), 0, None).unwrap();
    let cache_path = format!("{path_str}.tvcache");
    assert!(std::path::Path::new(&cache_path).exists(), "cache file should be created");

    let cached = crate::loader::load_cache(path_str, None).expect("cache should load");
    assert_eq!(cached.total_events, original.total_events);
    assert_eq!(cached.tracks.len(), original.tracks.len());
    assert_eq!(cached.names, original.names);
    assert_eq!(cached.cats, original.cats);
    assert_eq!(cached.max_ts, original.max_ts);
    assert_eq!(cached.device, original.device);
    assert_eq!(cached.stats.len(), original.stats.len());
    for (a, b) in cached.tracks.iter().zip(original.tracks.iter()) {
        assert_eq!(a.label, b.label);
        assert_eq!(a.gpu, b.gpu);
        assert_eq!(a.max_depth, b.max_depth);
        assert_eq!(a.events.len(), b.events.len());
        for (ea, eb) in a.events.iter().zip(b.events.iter()) {
            assert_eq!(ea.ts, eb.ts);
            assert_eq!(ea.dur, eb.dur);
            assert_eq!(ea.name, eb.name);
            assert_eq!(ea.depth, eb.depth);
            assert_eq!(ea.args_off, eb.args_off);
            assert_eq!(ea.args_len, eb.args_len);
        }
        assert_eq!(a.prefix_max_dur, b.prefix_max_dur);
    }
    let orig_args = &original.raw_bufs[0];
    let cached_args = &cached.raw_bufs[0];
    assert_eq!(orig_args.len(), cached_args.len());
    assert_eq!(&orig_args[..], &cached_args[..]);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&cache_path).ok();
}

#[test]
fn test_load_trace_depth_assignment() {
    let json = r#"{"traceEvents": [
        {"ph":"X","ts":0,"dur":100,"pid":1,"tid":1,"name":"outer","cat":"kernel"},
        {"ph":"X","ts":10,"dur":30,"pid":1,"tid":1,"name":"inner1","cat":"kernel"},
        {"ph":"X","ts":50,"dur":30,"pid":1,"tid":1,"name":"inner2","cat":"kernel"}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_depth");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("depth.json");
    std::fs::write(&path, json).unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    let track = &trace.tracks[0];
    assert_eq!(track.events.len(), 3);
    let depths: Vec<u16> = track.events.iter().map(|e| e.depth).collect();
    assert_eq!(depths[0], 0);
    assert!(depths[1] > 0 || depths[2] > 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_trace_truncated_json() {
    let json = r#"{"traceEvents": [
        {"ph":"X","ts":100,"dur":50,"pid":1,"tid":1,"name":"kern_a","cat":"kernel"},
        {"ph":"X","ts":200,"dur":30,"pid":1,"tid":1,"name":"kern_b","cat":"kernel"},
        {"ph":"X","ts":300,"dur":40,"pid":1,"tid":1,"name":"kern_c","cat":"ke"#;
    let dir = std::env::temp_dir().join("tv_test_truncated");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("truncated.json");
    std::fs::write(&path, json).unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    assert!(trace.total_events >= 2, "should parse complete events from truncated JSON, got {}", trace.total_events);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_trace_truncated_gz() {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let json = br#"{"traceEvents": [
        {"ph":"X","ts":100,"dur":50,"pid":1,"tid":1,"name":"kern_a","cat":"kernel"},
        {"ph":"X","ts":200,"dur":30,"pid":1,"tid":1,"name":"kern_b","cat":"kernel"},
        {"ph":"X","ts":300,"dur":40,"pid":1,"tid":1,"name":"kern_c","cat":"kernel"}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_truncgz");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("truncated.json.gz");

    let mut buf = Vec::new();
    {
        let mut gz = GzEncoder::new(&mut buf, Compression::fast());
        std::io::Write::write_all(&mut gz, json).unwrap();
        gz.finish().unwrap();
    }
    std::fs::write(&path, &buf[..buf.len() - 10]).unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    assert!(trace.total_events >= 2, "should parse events from truncated gz, got {}", trace.total_events);

    std::fs::remove_file(&path).ok();
}

// --- Args parsing ---

#[test]
fn test_parse_args_flat() {
    let blob = br#"{"key1": "val1", "key2": 42}"#;
    let mut strs = Vec::new();
    let mut idx = FnvMap::default();
    let mut pairs = Vec::new();
    parse_args_flat(blob, &mut strs, &mut idx, &mut pairs);

    assert_eq!(pairs.len(), 2);
    assert_eq!(strs[pairs[0][0] as usize], "key1");
    assert_eq!(strs[pairs[0][1] as usize], "val1");
    assert_eq!(strs[pairs[1][0] as usize], "key2");
    assert_eq!(strs[pairs[1][1] as usize], "42");
}

#[test]
fn test_parse_args_flat_empty() {
    let mut strs = Vec::new();
    let mut idx = FnvMap::default();
    let mut pairs = Vec::new();
    parse_args_flat(b"{}", &mut strs, &mut idx, &mut pairs);
    assert!(pairs.is_empty());
    parse_args_flat(b"", &mut strs, &mut idx, &mut pairs);
    assert!(pairs.is_empty());
}

#[test]
fn test_select_from_search() {
    let trace = make_trace(
        vec!["alpha_kernel", "beta_kernel", "gamma_kernel"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),
            ev(10.0, 20.0, 1, 0),
            ev(30.0, 5.0, 0, 0),
            ev(35.0, 15.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selected = Some(EventRef { track_idx: 0, event_idx: 1 });
    p.selection = Some([5.0, 25.0, 0.0, 1e9]);

    p.search = "alpha".to_string();
    p.rebuild_search();
    p.select_from_search(&mut state.buf);

    assert!(p.selected.is_none(), "individual selection should be cleared");
    assert!(p.selection.is_none(), "spatial selection should be cleared");
    assert_eq!(p.selection_stats.len(), 1);
    assert_eq!(p.selection_stats[0].name, 0);
    assert_eq!(p.selection_stats[0].count, 2);
    assert!((p.selection_stats[0].total_dur - 15.0).abs() < 1e-9);
}

#[test]
fn test_select_from_search_no_match_keeps_old() {
    let trace = make_trace(
        vec!["kern"],
        vec![("GPU", true, vec![ev(0.0, 10.0, 0, 0)])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([0.0, 10.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    assert_eq!(p.selection_stats.len(), 1);

    p.search = "nonexistent".to_string();
    p.rebuild_search();
    p.select_from_search(&mut state.buf);
    assert_eq!(p.selection_stats.len(), 1);
}

#[test]
fn test_select_from_search_respects_hidden() {
    let trace = make_trace(
        vec!["alpha", "beta"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),
            ev(10.0, 20.0, 1, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.hidden_names[1] = true;

    p.search = "a".to_string();
    p.rebuild_search();
    p.select_from_search(&mut state.buf);

    assert_eq!(p.selection_stats.len(), 1);
    assert_eq!(p.selection_stats[0].name, 0);
}

// --- Diff algorithm ---

fn seq(items: &[(&str, f64)]) -> Vec<(String, f64)> {
    items.iter().map(|(n, d)| (n.to_string(), *d)).collect()
}

fn kinds(diff: &DiffResult) -> Vec<DiffKind> {
    diff.lines.iter().map(|l| l.kind).collect()
}

#[test]
fn test_diff_identical_sequences() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0), ("layernorm", 3.0)]);
    let diff = diff::compute_diff(&a, &a);
    assert_eq!(diff.lines.len(), 3);
    assert!(diff.lines.iter().all(|l| l.kind == DiffKind::Same));
    assert_eq!(diff.count_a, 3);
    assert_eq!(diff.count_b, 3);
    for line in &diff.lines {
        assert_eq!(line.dur_a, line.dur_b);
    }
}

#[test]
fn test_diff_empty_sequences() {
    let diff = diff::compute_diff(&[], &[]);
    assert_eq!(diff.lines.len(), 0);
    assert_eq!(diff.count_a, 0);
    assert_eq!(diff.count_b, 0);
    assert_eq!(diff.total_dur_a, 0.0);
    assert_eq!(diff.total_dur_b, 0.0);
}

#[test]
fn test_diff_left_empty() {
    let b = seq(&[("matmul", 10.0), ("softmax", 5.0)]);
    let diff = diff::compute_diff(&[], &b);
    assert_eq!(diff.lines.len(), 2);
    assert!(diff.lines.iter().all(|l| l.kind == DiffKind::Added));
    assert_eq!(diff.count_a, 0);
    assert_eq!(diff.count_b, 2);
}

#[test]
fn test_diff_right_empty() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0)]);
    let diff = diff::compute_diff(&a, &[]);
    assert_eq!(diff.lines.len(), 2);
    assert!(diff.lines.iter().all(|l| l.kind == DiffKind::Removed));
    assert_eq!(diff.count_a, 2);
    assert_eq!(diff.count_b, 0);
}

#[test]
fn test_diff_single_insertion() {
    let a = seq(&[("matmul", 10.0), ("layernorm", 3.0)]);
    let b = seq(&[("matmul", 10.0), ("softmax", 5.0), ("layernorm", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k, vec![DiffKind::Same, DiffKind::Added, DiffKind::Same]);
    assert_eq!(diff.lines[1].name, "softmax");
    assert_eq!(diff.lines[1].dur_a, None);
    assert_eq!(diff.lines[1].dur_b, Some(5.0));
}

#[test]
fn test_diff_single_removal() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0), ("layernorm", 3.0)]);
    let b = seq(&[("matmul", 10.0), ("layernorm", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k, vec![DiffKind::Same, DiffKind::Removed, DiffKind::Same]);
    assert_eq!(diff.lines[1].name, "softmax");
    assert_eq!(diff.lines[1].dur_a, Some(5.0));
    assert_eq!(diff.lines[1].dur_b, None);
}

#[test]
fn test_diff_duration_change() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0)]);
    let b = seq(&[("matmul", 12.0), ("softmax", 4.0)]);
    let diff = diff::compute_diff(&a, &b);
    assert_eq!(diff.lines.len(), 2);
    assert!(diff.lines.iter().all(|l| l.kind == DiffKind::Same));
    assert_eq!(diff.lines[0].dur_a, Some(10.0));
    assert_eq!(diff.lines[0].dur_b, Some(12.0));
    assert_eq!(diff.lines[1].dur_a, Some(5.0));
    assert_eq!(diff.lines[1].dur_b, Some(4.0));
}

#[test]
fn test_diff_total_durations() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0)]);
    let b = seq(&[("matmul", 12.0), ("softmax", 4.0), ("relu", 2.0)]);
    let diff = diff::compute_diff(&a, &b);
    assert_eq!(diff.total_dur_a, 15.0);
    assert_eq!(diff.total_dur_b, 18.0);
}

#[test]
fn test_diff_completely_different() {
    let a = seq(&[("matmul", 10.0), ("softmax", 5.0)]);
    let b = seq(&[("conv", 8.0), ("relu", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    assert_eq!(diff.lines.len(), 4);
    let k = kinds(&diff);
    assert!(k.contains(&DiffKind::Removed));
    assert!(k.contains(&DiffKind::Added));
}

#[test]
fn test_diff_prefix_match_suffix_differs() {
    let a = seq(&[("A", 1.0), ("B", 2.0), ("C", 3.0)]);
    let b = seq(&[("A", 1.0), ("B", 2.0), ("D", 4.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k[0], DiffKind::Same); // A
    assert_eq!(k[1], DiffKind::Same); // B
    assert_eq!(diff.lines[0].name, "A");
    assert_eq!(diff.lines[1].name, "B");
}

#[test]
fn test_diff_multiple_insertions() {
    let a = seq(&[("A", 1.0), ("D", 4.0)]);
    let b = seq(&[("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k[0], DiffKind::Same);  // A
    assert_eq!(diff.lines.last().unwrap().kind, DiffKind::Same); // D
    assert_eq!(diff.lines.last().unwrap().name, "D");
    let added: Vec<&str> = diff.lines.iter()
        .filter(|l| l.kind == DiffKind::Added)
        .map(|l| l.name.as_str())
        .collect();
    assert_eq!(added, vec!["B", "C"]);
}

#[test]
fn test_diff_preserves_order() {
    let a = seq(&[("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0), ("E", 5.0)]);
    let b = seq(&[("A", 1.0), ("C", 3.0), ("E", 5.0)]);
    let diff = diff::compute_diff(&a, &b);
    let names: Vec<&str> = diff.lines.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, vec!["A", "B", "C", "D", "E"]);
    let k = kinds(&diff);
    assert_eq!(k, vec![DiffKind::Same, DiffKind::Removed, DiffKind::Same, DiffKind::Removed, DiffKind::Same]);
}

#[test]
fn test_diff_same_lines_have_both_durations() {
    let a = seq(&[("X", 7.0)]);
    let b = seq(&[("X", 9.0)]);
    let diff = diff::compute_diff(&a, &b);
    assert_eq!(diff.lines.len(), 1);
    assert_eq!(diff.lines[0].kind, DiffKind::Same);
    assert_eq!(diff.lines[0].dur_a, Some(7.0));
    assert_eq!(diff.lines[0].dur_b, Some(9.0));
}

#[test]
fn test_diff_removed_has_only_dur_a() {
    let a = seq(&[("X", 7.0)]);
    let diff = diff::compute_diff(&a, &[]);
    assert_eq!(diff.lines[0].kind, DiffKind::Removed);
    assert_eq!(diff.lines[0].dur_a, Some(7.0));
    assert_eq!(diff.lines[0].dur_b, None);
}

#[test]
fn test_diff_added_has_only_dur_b() {
    let b = seq(&[("X", 7.0)]);
    let diff = diff::compute_diff(&[], &b);
    assert_eq!(diff.lines[0].kind, DiffKind::Added);
    assert_eq!(diff.lines[0].dur_a, None);
    assert_eq!(diff.lines[0].dur_b, Some(7.0));
}

#[test]
fn test_diff_extract_selection_events() {
    let trace = make_trace(
        vec!["matmul", "softmax", "relu"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),
            ev(10.0, 5.0, 1, 0),
            ev(15.0, 3.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.finished_sel = Some([0.0, 20.0, 0.0, 100.0]);
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].0, "matmul");
    assert_eq!(events[1].0, "softmax");
    assert_eq!(events[2].0, "relu");
}

#[test]
fn test_diff_extract_selection_respects_hidden() {
    let trace = make_trace(
        vec!["matmul", "softmax", "relu"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),
            ev(10.0, 5.0, 1, 0),
            ev(15.0, 3.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.hidden_names[1] = true;
    p.finished_sel = Some([0.0, 20.0, 0.0, 100.0]);
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].0, "matmul");
    assert_eq!(events[1].0, "relu");
}

#[test]
fn test_diff_extract_selection_partial_time_range() {
    let trace = make_trace(
        vec!["matmul", "softmax", "relu"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),
            ev(10.0, 5.0, 1, 0),
            ev(15.0, 3.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.finished_sel = Some([5.0, 12.0, 0.0, 100.0]);
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].0, "matmul");
    assert_eq!(events[1].0, "softmax");
}

#[test]
fn test_diff_extract_selection_sorted_by_timestamp() {
    let trace = make_trace(
        vec!["late", "early", "mid"],
        vec![("GPU", true, vec![
            ev(20.0, 5.0, 0, 0),
            ev(0.0, 5.0, 1, 0),
            ev(10.0, 5.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.finished_sel = Some([0.0, 30.0, 0.0, 100.0]);
    let events = p.extract_selection_events();
    assert_eq!(events[0].0, "early");
    assert_eq!(events[1].0, "mid");
    assert_eq!(events[2].0, "late");
}

// Merged rows Tetris-pack their events and strip grandparent wrappers (whole-
// stream spans). A selection over a merged row must read the packed set that was
// actually drawn — not re-scan the raw track — or it picks up "ghost" wrappers
// that were never rendered. `geom.merged` here mimics draw_timeline's snapshot:
// the raw track has a `wrapper` span at index 0, but it is absent from the packed
// events, so selection must never return it.
#[test]
fn test_merged_selection_excludes_unrendered_wrapper() {
    let trace = make_trace(
        vec!["wrapper", "kA", "kB"],
        vec![("GPU", true, vec![
            ev(0.0, 100.0, 0, 0),  // idx 0: whole-stream wrapper — stripped from merged row
            ev(0.0, 10.0, 1, 1),   // idx 1: kA
            ev(20.0, 10.0, 2, 1),  // idx 2: kB
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    // Packed row: kA at depth 0, kB at depth 1; wrapper (idx 0) intentionally omitted.
    p.geom.merged = vec![MergedGeom { vi: 0, max_depth: 2, events: vec![(0, 1, 0), (0, 2, 1)] }];
    p.geom.heights[0] = 40.0; // max_depth 2 * SUB_LANE_H(20)
    p.geom.y_offsets[0] = 0.0;

    // Full-height selection over the whole time range: kA + kB, never the wrapper.
    p.finished_sel = Some([0.0, 30.0, 0.0, 40.0]);
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|(n, _)| n != "wrapper"));
    assert_eq!(events[0].0, "kA");
    assert_eq!(events[1].0, "kB");
}

// The renderer only highlights packed events whose depth lane intersects the
// selection rectangle; the stats/extract must apply the same y-test so they stay
// in sync with the highlight. sub_h = 40/2 = 20, so a y-range of [0,10] hits only
// depth-0 (kA), not depth-1 (kB).
#[test]
fn test_merged_selection_respects_depth_yrange() {
    let trace = make_trace(
        vec!["wrapper", "kA", "kB"],
        vec![("GPU", true, vec![
            ev(0.0, 100.0, 0, 0),
            ev(0.0, 10.0, 1, 1),
            ev(20.0, 10.0, 2, 1),
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.geom.merged = vec![MergedGeom { vi: 0, max_depth: 2, events: vec![(0, 1, 0), (0, 2, 1)] }];
    p.geom.heights[0] = 40.0;
    p.geom.y_offsets[0] = 0.0;

    p.finished_sel = Some([0.0, 30.0, 0.0, 10.0]);
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "kA");
}

#[test]
fn test_diff_cost_tiebreak_prefers_fewer_skips() {
    // A: X Y Z     B: X Q Y Z
    // Both sides find a match: A[1]=Y found in B at cost 1, B[1]=Q not in A.
    // With cost_b(1) <= cost_a(not found), we should skip Q as Added.
    // Inverting the comparison would incorrectly skip Y as Removed.
    let a = seq(&[("X", 1.0), ("Y", 2.0), ("Z", 3.0)]);
    let b = seq(&[("X", 1.0), ("Q", 1.5), ("Y", 2.0), ("Z", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k, vec![DiffKind::Same, DiffKind::Added, DiffKind::Same, DiffKind::Same]);
    assert_eq!(diff.lines[1].name, "Q");

    // Reverse: A has extra, B doesn't
    // A: X Q Y Z     B: X Y Z
    let diff2 = diff::compute_diff(&b, &a);
    let k2 = kinds(&diff2);
    assert_eq!(k2, vec![DiffKind::Same, DiffKind::Removed, DiffKind::Same, DiffKind::Same]);
    assert_eq!(diff2.lines[1].name, "Q");
}

#[test]
fn test_diff_asymmetric_cost_picks_cheaper_side() {
    // A: X P Q R Y     B: X S Y
    // cost_a to find S: not found. cost_b to find X's next=P: not found.
    // After X matches, A[1]=P not in B, B[1]=S not in A -> both skip.
    // Then A has P,Q,R before Y; B goes straight to Y.
    // The diff should prefer skipping the shorter side (B's S) when both have matches to Y.
    let a = seq(&[("X", 1.0), ("P", 1.0), ("Q", 1.0), ("R", 1.0), ("Y", 1.0)]);
    let b = seq(&[("X", 1.0), ("S", 1.0), ("Y", 1.0)]);
    let diff = diff::compute_diff(&a, &b);
    let k = kinds(&diff);
    assert_eq!(k[0], DiffKind::Same); // X
    assert_eq!(diff.lines.last().unwrap().name, "Y");
    assert_eq!(diff.lines.last().unwrap().kind, DiffKind::Same);
    // All of P, Q, R should be Removed and S should be Added
    let removed: Vec<&str> = diff.lines.iter()
        .filter(|l| l.kind == DiffKind::Removed)
        .map(|l| l.name.as_str()).collect();
    let added: Vec<&str> = diff.lines.iter()
        .filter(|l| l.kind == DiffKind::Added)
        .map(|l| l.name.as_str()).collect();
    assert!(removed.contains(&"P"));
    assert!(removed.contains(&"Q"));
    assert!(removed.contains(&"R"));
    assert!(added.contains(&"S"));
}

#[test]
fn test_diff_inline_added_has_correct_dur_fields() {
    // When an Added line appears mid-sequence (not in trailing drain),
    // dur_a must be None and dur_b must be Some.
    let a = seq(&[("A", 1.0), ("C", 3.0)]);
    let b = seq(&[("A", 1.0), ("B", 2.0), ("C", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    let added = diff.lines.iter().find(|l| l.kind == DiffKind::Added).unwrap();
    assert_eq!(added.name, "B");
    assert_eq!(added.dur_a, None);
    assert_eq!(added.dur_b, Some(2.0));
}

#[test]
fn test_diff_inline_removed_has_correct_dur_fields() {
    let a = seq(&[("A", 1.0), ("B", 2.0), ("C", 3.0)]);
    let b = seq(&[("A", 1.0), ("C", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    let removed = diff.lines.iter().find(|l| l.kind == DiffKind::Removed).unwrap();
    assert_eq!(removed.name, "B");
    assert_eq!(removed.dur_a, Some(2.0));
    assert_eq!(removed.dur_b, None);
}

#[test]
fn test_diff_both_sides_match_added_dur_fields() {
    // Triggers (Some(bk), Some(ak)) arm where cost_b <= cost_a.
    // A: [X, C], B: [X, B1, B2, C] — both sides find C, but B has cheaper cost.
    // Also A[1]=C is found in B at bk=3 (cost 2), B[1]=B1 is found in A? No.
    // Actually need both found. Let's force it:
    // A: [X, B1, C], B: [X, Q, B1, C]
    // At X match -> i=1,j=1. A[1]=B1 found in B at bk=2 (cost 1).
    //   B[1]=Q found in A? No. -> (Some(2), None) arm.
    // Need: A: [X, Q, C], B: [X, P, Q, C]
    // i=1,j=1. A[1]=Q found in B at bk=2 (cost 1). B[1]=P found in A? No. -> (Some, None).
    // Need both found: A: [X, P, C], B: [X, Q, P, C]
    // i=1,j=1. A[1]=P found in B at bk=2 (cost 1). B[1]=Q found in A? No. -> (Some, None).
    // Try: A: [P, Q, Z], B: [Q, P, Z]
    // i=0,j=0. A[0]=P, B[0]=Q. P found in B at bk=1 (cost 1). Q found in A at ak=1 (cost 1).
    // Both found, equal cost -> cost_b <= cost_a -> emit B[0..1] as Added (Q).
    let a = seq(&[("P", 1.0), ("Q", 2.0), ("Z", 3.0)]);
    let b = seq(&[("Q", 2.5), ("P", 1.5), ("Z", 3.0)]);
    let diff = diff::compute_diff(&a, &b);
    // With cost_b <= cost_a (both 1), we skip B[0] as Added, then match P.
    for line in &diff.lines {
        match line.kind {
            DiffKind::Added => {
                assert_eq!(line.dur_a, None, "Added line '{}' should have dur_a=None", line.name);
                assert!(line.dur_b.is_some(), "Added line '{}' should have dur_b", line.name);
            }
            DiffKind::Removed => {
                assert!(line.dur_a.is_some(), "Removed line '{}' should have dur_a", line.name);
                assert_eq!(line.dur_b, None, "Removed line '{}' should have dur_b=None", line.name);
            }
            DiffKind::Same => {
                assert!(line.dur_a.is_some(), "Same line '{}' should have dur_a", line.name);
                assert!(line.dur_b.is_some(), "Same line '{}' should have dur_b", line.name);
            }
        }
    }
}

// --- parse_rank ---

#[test]
fn test_parse_rank() {
    assert_eq!(parse_rank("Rank 0 GPU 0"), Some(0));
    assert_eq!(parse_rank("Rank 12 CPU"), Some(12));
    assert_eq!(parse_rank("Rank 99 Stream"), Some(99));
    assert_eq!(parse_rank("GPU 0"), None);
    assert_eq!(parse_rank(""), None);
    assert_eq!(parse_rank("Rank abc GPU"), None);
}

// --- detect_rank_groups ---

#[test]
fn test_detect_rank_groups_basic() {
    let paths = vec![
        "trace-rank-0.json.gz".to_string(),
        "trace-rank-1.json.gz".to_string(),
        "trace-rank-2.json.gz".to_string(),
        "standalone.json".to_string(),
    ];
    let (groups, standalone) = detect_rank_groups(&paths);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 3);
    assert_eq!(groups[0][0].0, 0);
    assert_eq!(groups[0][1].0, 1);
    assert_eq!(groups[0][2].0, 2);
    assert_eq!(standalone, vec!["standalone.json"]);
}

#[test]
fn test_detect_rank_groups_ep_format() {
    let paths = vec![
        "dp0_pp0_tp0_dcp0_ep0_rank0.123.pt.trace.json.gz".to_string(),
        "dp1_pp0_tp0_dcp0_ep1_rank0.456.pt.trace.json.gz".to_string(),
        "dp2_pp0_tp0_dcp0_ep2_rank0.789.pt.trace.json.gz".to_string(),
    ];
    let (groups, standalone) = detect_rank_groups(&paths);
    assert_eq!(groups.len(), 1, "should group all ep files together");
    assert_eq!(groups[0].len(), 3);
    assert_eq!(groups[0][0].0, 0);
    assert_eq!(groups[0][1].0, 1);
    assert_eq!(groups[0][2].0, 2);
    assert!(standalone.is_empty());
}

#[test]
fn test_detect_rank_groups_single_ep_file() {
    let paths = vec![
        "dp1_pp0_tp0_dcp0_ep1_rank0.123.pt.trace.json.gz".to_string(),
    ];
    let (groups, standalone) = detect_rank_groups(&paths);
    assert!(groups.is_empty(), "single file should not form a group");
    assert_eq!(standalone.len(), 1, "single ep file must end up in standalone");
}

#[test]
fn test_detect_rank_groups_no_ranks() {
    let paths = vec![
        "trace_a.json".to_string(),
        "trace_b.json".to_string(),
    ];
    let (groups, standalone) = detect_rank_groups(&paths);
    assert!(groups.is_empty());
    assert_eq!(standalone.len(), 2);
}

#[test]
fn test_is_trace_file() {
    assert!(crate::loader::is_trace_file("foo.json"));
    assert!(crate::loader::is_trace_file("foo.json.gz"));
    assert!(crate::loader::is_trace_file("foo.tar.gz"));
    assert!(crate::loader::is_trace_file("foo.tgz"));
    assert!(!crate::loader::is_trace_file("foo.txt"));
    assert!(!crate::loader::is_trace_file("foo.csv"));
    assert!(!crate::loader::is_trace_file("foo.gz"));
}

// --- merge_traces ---

#[test]
fn test_merge_traces() {
    let t0 = make_trace(
        vec!["", "kern_a"],
        vec![("GPU 0", true, vec![ev(100.0, 50.0, 1, 0)])],
    );
    let t1 = make_trace(
        vec!["", "kern_b"],
        vec![("GPU 0", true, vec![ev(200.0, 60.0, 1, 0)])],
    );
    let merged = merge_traces(vec![(0, t0), (1, t1)]);

    assert_eq!(merged.tracks.len(), 2);
    assert!(merged.tracks[0].label.contains("Rank 0"));
    assert!(merged.tracks[1].label.contains("Rank 1"));
    assert!(merged.names.contains(&"kern_a".to_string()));
    assert!(merged.names.contains(&"kern_b".to_string()));
    let r0_ev = &merged.tracks[0].events[0];
    let r1_ev = &merged.tracks[1].events[0];
    assert!(r0_ev.ts < r1_ev.ts, "rank 0 event should be earlier");
}

// Measures the per-frame cost of the merged-GPU buffer build for two depth
// filters: leaf-only (delta=0, current) vs. keep-one-parent-level (delta=1,
// the 047226a design that un-hides parent blocks). Run with:
//   cargo test --release bench_merge_filter -- --ignored --nocapture
#[test]
#[ignore]
fn bench_merge_filter() {
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("set TV_BENCH_TRACE=<path.tvcache> to run this bench"); return; }
    };
    let counter = test_counter();
    let trace = load_trace(&path, &counter, 8, None).expect("load trace");
    eprintln!(
        "loaded: {} tracks, {} events, span {:.0}..{:.0} us",
        trace.tracks.len(), trace.total_events, trace.min_ts, trace.max_ts
    );

    // Group GPU tracks by rank, mirroring ui.rs merge pre-grouping.
    let mut groups: Vec<(Option<usize>, Vec<usize>)> = Vec::new();
    for i in 0..trace.tracks.len() {
        if !trace.tracks[i].gpu { continue; }
        let rank = parse_rank(&trace.tracks[i].label);
        if let Some(g) = groups.iter_mut().find(|(r, _)| *r == rank) {
            g.1.push(i);
        } else {
            groups.push((rank, vec![i]));
        }
    }
    let gpu_tracks: usize = groups.iter().map(|(_, t)| t.len()).sum();
    eprintln!("gpu rank groups: {}, gpu tracks: {}", groups.len(), gpu_tracks);

    // How deep do the source GPU tracks actually nest?
    let mut src_max_depth = 0u16;
    let mut depth_hist = [0usize; 8];
    for (_r, tracks) in &groups {
        for &ti in tracks {
            for e in &trace.tracks[ti].events {
                src_max_depth = src_max_depth.max(e.depth + 1);
                depth_hist[(e.depth as usize).min(7)] += 1;
            }
        }
    }
    eprintln!("source max nesting depth: {}  per-depth counts: {:?}", src_max_depth, depth_hist);

    // trace.min_ts/max_ts are unreliable on the merged cache; derive the real
    // GPU time span from the events themselves.
    let mut gmin = f64::INFINITY;
    let mut gmax = f64::NEG_INFINITY;
    for (_r, tracks) in &groups {
        for &ti in tracks {
            let evs = &trace.tracks[ti].events;
            if let Some(first) = evs.first() { gmin = gmin.min(first.ts); }
            for e in evs { gmax = gmax.max(e.ts + e.dur); }
        }
    }
    eprintln!("gpu span: {:.0}..{:.0} us ({:.0} us)", gmin, gmax, gmax - gmin);

    // One full merge-buffer build over [t0,t1] for the given depth delta.
    // Returns (surviving events, max tetris depth).
    let build = |delta: i32, t0: f64, t1: f64| -> (usize, u16) {
        let mut total_ev = 0usize;
        let mut max_md = 0u16;
        for (_rank, tracks) in &groups {
            let mut ev_list: Vec<(f64, f64, u32, u32)> = Vec::new();
            for &ti in tracks {
                let gt = &trace.tracks[ti];
                let start = crate::types::bisect_overlap(&gt.events, &gt.prefix_max_dur, t0);
                let end = gt.events.partition_point(|e| e.ts <= t1);
                for ei in start..end {
                    let ev = &gt.events[ei];
                    let strip = gt.events[ei + 1..].iter()
                        .take_while(|e2| e2.ts <= ev.ts + ev.dur)
                        .any(|e2| e2.depth as i32 > ev.depth as i32 + delta);
                    if strip { continue; }
                    ev_list.push((ev.ts, ev.dur, ti as u32, ei as u32));
                }
            }
            ev_list.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut depth_ends: Vec<f64> = Vec::new();
            let mut max_depth: u16 = 0;
            for &(ts, dur, _, _) in &ev_list {
                let d = depth_ends.iter().position(|&end| end <= ts)
                    .unwrap_or_else(|| { depth_ends.push(0.0); depth_ends.len() - 1 });
                depth_ends[d] = ts + dur;
                let d16 = d as u16;
                if d16 >= max_depth { max_depth = d16 + 1; }
            }
            total_ev += ev_list.len();
            if max_depth > max_md { max_md = max_depth; }
        }
        (total_ev, max_md)
    };

    let span = gmax - gmin;
    let mid = (gmin + gmax) / 2.0;
    for &frac in &[1.0f64, 0.25, 0.05] {
        let half = span * frac / 2.0;
        let (t0, t1) = (mid - half, mid + half);
        for &(label, delta) in &[("leaf-only  ", 0i32), ("keep-parent", 1i32), ("keep-all   ", 10000i32)] {
            let (n, md) = build(delta, t0, t1); // warmup + result
            let iters = 30;
            let start = std::time::Instant::now();
            for _ in 0..iters { let _ = build(delta, t0, t1); }
            let per = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            eprintln!(
                "zoom={:>5.0}%  {}  {:>9} events  max_depth={:>3}  {:>7.3} ms/build",
                frac * 100.0, label, n, md, per
            );
        }
    }
}


#[test]
fn test_zoom_to_search_frames_first_match_80pct_fill() {
    // Two tracks; "foo" matches appear on both. The earliest match (ts=50 on
    // track 1) is what we frame, not the bounding box of all matches.
    let trace = make_trace(
        vec!["", "foo", "bar"],
        vec![
            ("GPU 0", true, vec![
                ev(0.0, 5.0, 2, 0),      // bar
                ev(100.0, 4.0, 1, 0),    // foo (later)
                ev(900.0, 5.0, 2, 0),    // bar
            ]),
            ("GPU 1", true, vec![
                ev(50.0, 4.0, 1, 0),     // foo (EARLIEST match)
                ev(800.0, 4.0, 1, 0),    // foo (later)
            ]),
        ],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    pane.view.t0 = 0.0;
    pane.view.t1 = 1000.0;
    pane.search.push_str("foo");
    pane.rebuild_search();
    pane.zoom_to_search();

    // Vertical focus is the track holding the earliest match (track 1).
    assert_eq!(pane.pending_focus, Some(1));

    let a = pane.view.anim.as_ref().expect("search zoom should start an animation");
    // first match = [50,54] => dur 4, range = 4/0.8 = 5, center 52
    assert!((a.to_t0 - 49.5).abs() < 1e-6, "to_t0={}", a.to_t0);
    assert!((a.to_t1 - 54.5).abs() < 1e-6, "to_t1={}", a.to_t1);
    // 80% fill: the event's duration occupies 4/5 = 0.8 of the framed range.
    let framed = a.to_t1 - a.to_t0;
    assert!((4.0 / framed - 0.8).abs() < 1e-9);

    // Driving the animation to completion lands exactly on target and clears.
    // dt is clamped per tick, so step until it finishes (bounded loop).
    let mut guard = 0;
    while pane.view.tick_anim(0.05) { guard += 1; assert!(guard < 100); }
    assert!(pane.view.anim.is_none());
    assert!((pane.view.t0 - 49.5).abs() < 1e-6);
    assert!((pane.view.t1 - 54.5).abs() < 1e-6);
}

#[test]
fn test_nav_search_wraps_and_frames_each_match() {
    // Three "foo" matches, sorted by ts into search_nav: [50, 100, 800].
    let trace = make_trace(
        vec!["", "foo", "bar"],
        vec![
            ("GPU 0", true, vec![
                ev(0.0, 5.0, 2, 0),      // bar
                ev(100.0, 4.0, 1, 0),    // foo (cursor 1)
                ev(800.0, 4.0, 1, 0),    // foo (cursor 2)
            ]),
            ("GPU 1", true, vec![
                ev(50.0, 6.0, 1, 0),     // foo (cursor 0, earliest)
            ]),
        ],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    pane.view.t0 = 0.0;
    pane.view.t1 = 1000.0;
    pane.search.push_str("foo");
    pane.rebuild_search();
    assert_eq!(pane.search_nav.len(), 3);
    assert_eq!(pane.search_cursor, 0);

    // Forward: cursor 0 -> 1, frames the ts=100 match (track 0), dur 4.
    pane.nav_search(true);
    assert_eq!(pane.search_cursor, 1);
    assert_eq!(pane.pending_focus, Some(0));
    let sel = pane.selected.expect("nav selects the match");
    assert_eq!((sel.track_idx, sel.event_idx), (0, 1));
    let a = pane.view.anim.as_ref().unwrap();
    assert!((a.to_t1 - a.to_t0 - 4.0 / SEARCH_ZOOM_FILL).abs() < 1e-6);

    // Backward from cursor 1 -> 0: earliest match on track 1.
    pane.nav_search(false);
    assert_eq!(pane.search_cursor, 0);
    assert_eq!(pane.pending_focus, Some(1));

    // Backward again wraps 0 -> 2 (last match).
    pane.nav_search(false);
    assert_eq!(pane.search_cursor, 2);
    let sel = pane.selected.unwrap();
    assert_eq!((sel.track_idx, sel.event_idx), (0, 2));

    // Forward from last wraps 2 -> 0.
    pane.nav_search(true);
    assert_eq!(pane.search_cursor, 0);
}

#[test]
fn test_rebuild_multi_select_stats_aggregates_by_name() {
    // "foo" (name id 1) appears 3x on GPU tracks and 1x on a CPU track.
    let trace = make_trace(
        vec!["", "foo", "bar"],
        vec![
            ("GPU 0", true, vec![
                ev(0.0, 10.0, 1, 0),   // foo
                ev(20.0, 2.0, 2, 0),   // bar
                ev(30.0, 20.0, 1, 0),  // foo
            ]),
            ("GPU 1", true, vec![
                ev(5.0, 30.0, 1, 0),   // foo
            ]),
            ("CPU", false, vec![
                ev(1.0, 99.0, 1, 0),   // foo, but on a CPU track
            ]),
        ],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    assert!(!pane.show_cpu, "default hides CPU tracks");

    // Double-click selects every "foo": with CPU hidden, only the 3 GPU ones.
    pane.multi_select_name = Some(1);
    pane.rebuild_multi_select_stats();
    assert_eq!(pane.selection_stats.len(), 1);
    let se = &pane.selection_stats[0];
    assert_eq!(se.name, 1);
    assert_eq!(se.count, 3);
    assert!((se.total_dur - 60.0).abs() < 1e-9, "total={}", se.total_dur);
    // median of [10,20,30] = 20
    assert!((pane.sel_median - 20.0).abs() < 1e-9, "median={}", pane.sel_median);
    assert_eq!(pane.sel_individual.len(), 3);

    // Enabling CPU includes the 4th instance.
    pane.show_cpu = true;
    pane.rebuild_multi_select_stats();
    assert_eq!(pane.selection_stats[0].count, 4);
    assert!((pane.selection_stats[0].total_dur - 159.0).abs() < 1e-9);
}

#[test]
fn test_copy_selection_text_survives_stale_selection() {
    // Regression: after a reload/merge the trace can shrink, leaving a stale
    // EventRef whose index is out of bounds. Indexing it must not panic.
    let trace = make_trace(
        vec!["", "foo", "bar"],
        vec![("GPU 0", true, vec![
            ev(0.0, 5.0, 2, 0),
            ev(10.0, 4.0, 1, 0),
            ev(20.0, 3.0, 2, 0),
        ])],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];

    // A valid selection produces text.
    pane.selected = Some(crate::types::EventRef { track_idx: 0, event_idx: 1 });
    let out = pane.copy_selection_text().expect("valid selection yields text");
    assert!(out.contains("foo"), "got {out:?}");

    // Stale event index (track has only 3 events) must not panic.
    pane.selected = Some(crate::types::EventRef { track_idx: 0, event_idx: 23226 });
    assert!(pane.copy_selection_text().is_none());

    // Stale track index must not panic either.
    pane.selected = Some(crate::types::EventRef { track_idx: 99, event_idx: 0 });
    assert!(pane.copy_selection_text().is_none());
}

#[test]
fn test_nav_search_no_matches_noop() {
    let trace = make_trace(
        vec!["", "foo"],
        vec![("GPU 0", true, vec![ev(0.0, 5.0, 1, 0)])],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    pane.search.push_str("zzz");
    pane.rebuild_search();
    pane.nav_search(true);
    assert!(pane.view.anim.is_none());
    assert!(pane.selected.is_none());
}

#[test]
fn test_zoom_to_search_no_matches_no_anim() {
    let trace = make_trace(
        vec!["", "foo"],
        vec![("GPU 0", true, vec![ev(0.0, 5.0, 1, 0)])],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    pane.search.push_str("zzz");
    pane.rebuild_search();
    pane.zoom_to_search();
    assert!(pane.view.anim.is_none());
}
