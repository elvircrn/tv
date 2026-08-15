use super::*;
use crate::parse::*;
use crate::loader::{load_trace, detect_rank_groups, merge_traces};
use crate::state::{parse_rank, find_exec_context_names, default_track_order};
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
        vllm_version: String::new(),
        dist_rank: -1, dist_world: 0,
        flow_pairs: Vec::new(),
        rank_paths: Vec::new(),
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
    // name_color scales the (saturation-boosted) palette color by a 155/255
    // brightness multiplier, so no channel can exceed that ceiling — still
    // "darkened" relative to a raw, undimmed palette color (255 max).
    let c: u32 = name_color("x").into();
    let r = c & 0xFF;
    let g = (c >> 8) & 0xFF;
    let b = (c >> 16) & 0xFF;
    assert!(r <= 155 && g <= 155 && b <= 155, "colors should be darkened to <=155");
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
    p.finish_selection(&mut state.buf);

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
    p.finish_selection(&mut state.buf);

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
    p.finish_selection(&mut state.buf);

    assert_eq!(p.selection_stats.len(), 1);
    assert_eq!(p.selection_stats[0].name, 1);
}

// Regression: a finished selection used to be a raw pixel Y range, re-tested
// against whatever the CURRENT layout happened to be every time it was read.
// Toggling Show CPU (inserting/removing tracks), reordering tracks, or
// resizing the bottom panel (which can change per-track heights) would then
// silently reassign an existing selection to different tracks, without the
// user doing anything to their selection. capture_sel_events resolves a
// finished selection to a frozen (track_idx, event_idx) set at drag-finish
// time instead, so it must survive layout changes unchanged.
#[test]
fn test_selection_survives_layout_change() {
    let trace = make_trace(
        vec!["", "gpu_kern", "cpu_kern"],
        vec![
            ("GPU 0", true, vec![ev(0.0, 10.0, 1, 0)]),
            ("Thread 1", false, vec![ev(0.0, 10.0, 2, 0)]),
        ],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    // Select just the GPU row's own Y span.
    let gpu_h = p.geom.heights[0] as f64;
    p.selection = Some([0.0, 10.0, 0.0, gpu_h]);
    p.finish_selection(&mut state.buf);
    let before = p.extract_selection_events();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].0, "gpu_kern");

    // Simulate a layout change after the selection was made: tracks
    // reordered and the previously-selected row's height/offset changed —
    // exactly what enabling Show CPU or resizing the bottom panel can do.
    p.geom.visible = vec![1, 0];
    p.geom.y_offsets = vec![0.0, 500.0];
    p.geom.heights = vec![gpu_h as f32, 999.0];

    let after = p.extract_selection_events();
    assert_eq!(after, before, "a finished selection must not change when the layout changes");
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
fn test_load_cache_rejects_future_version() {
    // Forward-compat: a cache format newer than this build understands must
    // still be rejected (its layout may have diverged in ways this reader
    // can't handle) — unaffected by making *older* versions (see
    // test_compute_kernel_stats_min_dur) load via a fallback instead.
    let mut buf = vec![0u8; 80];
    buf[0..4].copy_from_slice(b"TRV2");
    buf[4..8].copy_from_slice(&999u32.to_le_bytes());
    assert!(crate::loader::load_cache_from_bytes(&buf).is_none(), "a from-the-future cache version must not load");
}

#[test]
fn test_compute_kernel_stats_min_dur() {
    // v4 grew KernelStats (added min_dur); cache files written before that
    // (any version < 4) have the old, smaller per-entry stats layout on
    // disk. load_cache_from_bytes skips those bytes and recomputes stats
    // from the raw events instead (see compute_kernel_stats) — this checks
    // that recompute path actually produces the right min, not a guess.
    let names = vec!["kernel_a"];
    let events = vec![ev(0.0, 30.0, 0, 0), ev(50.0, 10.0, 0, 0), ev(100.0, 20.0, 0, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let stats = crate::loader::compute_kernel_stats(&trace.tracks);
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].min_dur, 10.0);
    assert_eq!(stats[0].max_dur, 30.0);
    assert_eq!(stats[0].count, 3);
}

#[test]
fn test_cache_roundtrip() {
    let json = r#"{"vllm_version": "0.26.1rc1.dev528+gf8d03e774", "distributedInfo": {"backend": "nccl", "rank": 3, "world_size": 8, "pg_config": [{"pg_name": "0", "ranks": [0, 1, 2, 3, 4, 5, 6, 7]}]}, "traceEvents": [
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
    assert_eq!(original.vllm_version, "0.26.1rc1.dev528+gf8d03e774");
    assert_eq!(cached.vllm_version, original.vllm_version);
    assert_eq!(original.dist_rank, 3);
    assert_eq!(original.dist_world, 8);
    assert_eq!(cached.dist_rank, original.dist_rank);
    assert_eq!(cached.dist_world, original.dist_world);
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
fn test_rank_paths_trailer_roundtrip() {
    let mut trace = make_trace(
        vec!["kern_a"],
        vec![("  Rank 0 GPU", true, vec![Event { ts: 0.0, dur: 1.0, name: 0, cat: 0, args_off: 0, depth: 0, args_len: 0 }])],
    );
    trace.rank_paths = vec![(0, "dp0_tp0_rank0.json".to_string()), (1, "dp0_tp1_rank1.json".to_string())];

    let dir = std::env::temp_dir().join("tv_test_rank_paths_trailer");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("rank_paths_test.json");
    std::fs::write(&path, "{}").unwrap();
    let path_str = path.to_str().unwrap();

    crate::loader::save_cache(&trace, path_str, None);
    let cache_path = format!("{path_str}.tvcache");
    let cached = crate::loader::load_cache(path_str, None).expect("cache should load");
    assert_eq!(cached.rank_paths, trace.rank_paths);

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&cache_path).ok();
}

#[test]
fn test_rank_paths_trailer_absent_is_empty() {
    // A trace with no rank_paths (the common single-rank case) must not
    // write/read back a spurious nonempty section.
    let trace = make_trace(
        vec!["kern_a"],
        vec![("gpu0", true, vec![Event { ts: 0.0, dur: 1.0, name: 0, cat: 0, args_off: 0, depth: 0, args_len: 0 }])],
    );
    assert!(trace.rank_paths.is_empty());

    let dir = std::env::temp_dir().join("tv_test_rank_paths_absent");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("rank_paths_absent_test.json");
    std::fs::write(&path, "{}").unwrap();
    let path_str = path.to_str().unwrap();

    crate::loader::save_cache(&trace, path_str, None);
    let cache_path = format!("{path_str}.tvcache");
    let cached = crate::loader::load_cache(path_str, None).expect("cache should load");
    assert!(cached.rank_paths.is_empty());

    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&cache_path).ok();
}

#[test]
fn test_even_spacing_heights_all_content_is_uniform() {
    let has_content = vec![true; 4];
    let heights = crate::ui::even_spacing_heights(&has_content, 100.0);
    assert_eq!(heights, vec![25.0; 4]);
}

#[test]
fn test_even_spacing_heights_all_empty_is_uniform() {
    // Nothing to expand into the freed space, so this must not collapse
    // every row to the empty-strip height and lose the rest of the viewport.
    let has_content = vec![false; 4];
    let heights = crate::ui::even_spacing_heights(&has_content, 100.0);
    assert_eq!(heights, vec![25.0; 4]);
}

#[test]
fn test_even_spacing_heights_one_of_many_fills_the_rest() {
    // The scenario the merged multi-rank view actually hits: 8 rank rows,
    // only one has an event in the current (zoomed-in) view window.
    let mut has_content = vec![false; 8];
    has_content[3] = true;
    let heights = crate::ui::even_spacing_heights(&has_content, 100.0);
    for (i, &h) in heights.iter().enumerate() {
        if i == 3 {
            assert_eq!(h, 100.0 - 7.0 * 6.0, "the one non-empty row should get the freed height");
        } else {
            assert_eq!(h, 6.0, "empty rows should collapse to the thin strip");
        }
    }
}

#[test]
fn test_even_spacing_heights_empty_strip_never_exceeds_uniform_share() {
    // A tiny viewport with many rows: EMPTY_ROW_H (6.0) alone would exceed
    // an equal per-row share, which must clamp instead of overflowing avail.
    let has_content = vec![false, true, false, false, false, false, false, false, false, false];
    let heights = crate::ui::even_spacing_heights(&has_content, 10.0);
    let total: f32 = heights.iter().sum();
    assert!(total <= 10.0 + 1e-4, "collapsed rows must not overflow the available height: {total}");
}

#[test]
fn test_even_spacing_heights_empty_input() {
    assert!(crate::ui::even_spacing_heights(&[], 100.0).is_empty());
}

#[test]
fn test_bucket_durations_spreads_across_range() {
    let durs = vec![0.0, 25.0, 50.0, 75.0, 100.0];
    let (bins, min, max) = crate::ui::bucket_durations(&durs, 4);
    assert_eq!(min, 0.0);
    assert_eq!(max, 100.0);
    assert_eq!(bins.iter().sum::<u32>(), 5);
    // 0.0->bucket0, 25.0->bucket1, 50.0->bucket2, 75.0->bucket3,
    // 100.0 is exactly the max boundary and must clamp into the last
    // bucket rather than overflow past the end of `bins`.
    assert_eq!(bins, vec![1, 1, 1, 2]);
}

#[test]
fn test_bucket_durations_identical_values_go_in_first_bucket() {
    // range == 0 must not divide by zero.
    let durs = vec![5.0, 5.0, 5.0];
    let (bins, min, max) = crate::ui::bucket_durations(&durs, 8);
    assert_eq!(min, 5.0);
    assert_eq!(max, 5.0);
    assert_eq!(bins[0], 3);
    assert_eq!(bins.iter().sum::<u32>(), 3);
}

#[test]
fn test_bucket_durations_empty_input() {
    let (bins, min, max) = crate::ui::bucket_durations(&[], 8);
    assert_eq!(bins.iter().sum::<u32>(), 0);
    assert!(min > max, "empty input should produce the invalid min>max sentinel");
}

#[test]
fn test_fit_font_size_unshrunk_above_reference_height() {
    // A lane at or above the tuned reference height renders pixel-identical
    // to before this existed: no shrinking.
    assert_eq!(crate::ui::fit_font_size(15.0, 16.0), 15.0);
    assert_eq!(crate::ui::fit_font_size(15.0, 100.0), 15.0);
}

#[test]
fn test_fit_font_size_shrinks_proportionally() {
    // Half the reference height (16.0) should shrink to roughly half size.
    let size = crate::ui::fit_font_size(16.0, 8.0);
    assert!((size - 8.0).abs() < 0.01, "expected ~8.0, got {size}");
}

#[test]
fn test_fit_font_size_floors_at_min_text_px() {
    // An extremely squashed lane must still floor at MIN_TEXT_PX rather
    // than shrinking to near-zero or negative.
    let size = crate::ui::fit_font_size(15.0, 0.5);
    assert_eq!(size, crate::types::MIN_TEXT_PX);
    let size = crate::ui::fit_font_size(15.0, 0.0);
    assert_eq!(size, crate::types::MIN_TEXT_PX);
}

#[test]
fn test_rank_summary() {
    use crate::rank_summary;
    let fname = "dp0_pp0_tp3_dcp0_ep3_rank3.1786304095590565996.pt.trace.json.gz";
    // Full info: distributedInfo rank/world + filename coords (dcp0 omitted).
    assert_eq!(rank_summary(fname, 3, 32), "rank 3/32 · tp3 pp0 dp0 ep3");
    // No distributedInfo, coords still parse from the filename.
    assert_eq!(rank_summary(fname, -1, 0), "tp3 pp0 dp0 ep3");
    // Merged trace: world known, no single rank.
    assert_eq!(rank_summary("8 ranks: prefix", -1, 8), "8 ranks");
    // Non-vLLM filename with no coords and no dist info.
    assert_eq!(rank_summary("chrome_trace.json", -1, 0), "");
    // dcp shown when non-zero.
    assert_eq!(rank_summary("tp1_dcp2_ep0", -1, 0), "tp1 ep0 dcp2");
}

#[test]
fn test_find_exec_context_names() {
    let names: Vec<String> = ["", "foo", "execute_context_0(0)_generation_15(15)", "bar",
        "execute_context_0(0)_generation_16(16)"]
        .iter().map(|s| s.to_string()).collect();
    assert_eq!(find_exec_context_names(&names), vec![2, 4]);
    assert!(find_exec_context_names(&["foo".to_string(), "bar".to_string()]).is_empty());
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
    p.finish_selection(&mut state.buf);
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
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
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
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
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
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
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
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
    let events = p.extract_selection_events();
    assert_eq!(events[0].0, "early");
    assert_eq!(events[1].0, "mid");
    assert_eq!(events[2].0, "late");
}

// Merged rows Tetris-pack their events and strip grandparent wrappers (whole-
// stream spans). capture_sel_events reads the SAME packed geom.merged
// snapshot the renderer just drew from, so a selection matches the rendered
// row precisely instead of sweeping in ghost events that were never drawn —
// and because it's captured ONCE (not re-derived later), it stays correct
// regardless of what the layout does afterward.
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
    p.geom.merged = vec![MergedGeom { vi: 0, events: vec![(0, 1, 0), (0, 2, 1)] }];
    p.geom.heights[0] = 40.0; // max_depth 2 * SUB_LANE_H(20)
    p.geom.y_offsets[0] = 0.0;

    // Full-height selection over the whole time range: kA + kB, never the wrapper.
    p.finished_sel = Some([0.0, 30.0, 0.0, 40.0]);
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|(n, _)| n != "wrapper"));
    assert_eq!(events[0].0, "kA");
    assert_eq!(events[1].0, "kB");
}

// The renderer only highlights packed events whose depth lane intersects the
// selection rectangle; stats/extract must apply the same y-test so they stay
// in sync with the highlight — this is the exact precision that regressed
// when an earlier version of this fix dropped merged-row depth matching
// entirely ("selecting one item selected all items"). sub_h = 40/2 = 20, so a
// y-range of [0,10] hits only depth-0 (kA), not depth-1 (kB).
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
    p.geom.merged = vec![MergedGeom { vi: 0, events: vec![(0, 1, 0), (0, 2, 1)] }];
    p.geom.heights[0] = 40.0;
    p.geom.y_offsets[0] = 0.0;

    p.finished_sel = Some([0.0, 30.0, 0.0, 10.0]);
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
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

// A merged multi-rank view used to sort tracks lexicographically by their
// "  Rank {N} ..." label string, so "Rank 10" sorted before "Rank 2" (both
// start with a smaller leading digit character). 12 ranks (0..=11) is
// enough to actually exercise that: fewer than 10 ranks never hits a
// two-digit-vs-one-digit comparison at all.
#[test]
fn test_merge_traces_sorts_ranks_numerically_not_lexicographically() {
    let traces: Vec<(usize, Trace)> = (0..12).map(|r| {
        (r, make_trace(vec!["", "k"], vec![("GPU 0", true, vec![ev(r as f64, 1.0, 1, 0)])]))
    }).collect();
    let merged = merge_traces(traces);
    assert_eq!(merged.tracks.len(), 12);
    for (i, track) in merged.tracks.iter().enumerate() {
        assert!(track.label.contains(&format!("Rank {i} ")), "position {i} should be rank {i}, got {:?}", track.label);
    }
}

// The actual user-visible fix: even a Trace whose *stored* track order is
// already wrong (e.g. a .tvcache exported before merge_traces sorted
// correctly — the label text still has the real rank in it, only the
// on-disk order predates the fix) should still *display* in rank order,
// because the view computes its own default order rather than trusting
// storage order. Deliberately construct tracks in bad (lexicographic)
// order here to simulate that old-file case.
#[test]
fn test_default_track_order_fixes_bad_stored_order() {
    let bad_order = ["Rank 0", "Rank 1", "Rank 10", "Rank 11", "Rank 2"];
    let tracks: Vec<Track> = bad_order.iter().map(|r| Track {
        label: format!("  {r} GPU 0"), gpu: true, events: Vec::new(),
        max_depth: 0, prefix_max_dur: Vec::new(), raw_buf_idx: 0,
    }).collect();

    let order = default_track_order(&tracks);
    let visible_ranks: Vec<usize> = order.iter().map(|&i| parse_rank(&tracks[i].label).unwrap()).collect();
    assert_eq!(visible_ranks, vec![0, 1, 2, 10, 11], "view should show ranks in numeric order regardless of storage order");
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
fn test_sel_generation_bumps_on_every_rebuild() {
    // draw_stats_table's sort cache is keyed on this counter to avoid
    // re-sorting on every redraw when the selection hasn't changed — it must
    // advance on every real rebuild (so a genuinely new selection is never
    // mistaken for a stale one) and never go backwards.
    let trace = make_trace(
        vec!["", "foo"],
        vec![("GPU 0", true, vec![ev(0.0, 5.0, 1, 0), ev(10.0, 5.0, 1, 0)])],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    assert_eq!(pane.sel_generation, 0);

    pane.multi_select_name = Some(1);
    pane.rebuild_multi_select_stats();
    let g1 = pane.sel_generation;
    assert!(g1 > 0);

    pane.rebuild_multi_select_stats();
    let g2 = pane.sel_generation;
    assert!(g2 > g1, "a second rebuild must advance the generation again");

    pane.selection = Some([0.0, 5.0, 0.0, 100.0]);
    pane.finish_selection(&mut state.buf);
    assert!(state.panes[0].sel_generation > g2);
}

#[test]
fn test_individual_stats_occupancy_limit_lookup() {
    use crate::ui::kernel_occ_limit;

    // A normal small-grid kernel (WARPS-limited, trustworthy) and a
    // large-shared-memory kernel whose "SMEM" verdict is a known calculator
    // artifact (opts into more than the 48KB default static limit).
    let json = r#"{"traceEvents": [
        {"ph":"X","ts":100,"dur":10,"pid":1,"tid":1,"name":"warps_kernel","cat":"kernel",
         "args":{"shared memory":1024,"occupancy":{"limitingFactors":"WARPS"}}},
        {"ph":"X","ts":200,"dur":20,"pid":1,"tid":1,"name":"smem_kernel","cat":"kernel",
         "args":{"shared memory":180000,"occupancy":{"limitingFactors":"SMEM"}}}
    ]}"#;
    let dir = std::env::temp_dir().join("tv_test_occ");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("occ.json");
    std::fs::write(&path, json).unwrap();

    let trace = load_trace(path.to_str().unwrap(), &test_counter(), 0, None).unwrap();
    let warps_name = trace.names.iter().position(|n| n == "warps_kernel").unwrap() as u32;
    let smem_name = trace.names.iter().position(|n| n == "smem_kernel").unwrap() as u32;

    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    pane.show_cpu = true; // both events land on a non-GPU track here

    pane.multi_select_name = Some(warps_name);
    pane.rebuild_multi_select_stats();
    assert_eq!(pane.sel_individual_refs.len(), 1);
    let (ti, ei) = pane.sel_individual_refs[0];
    let (limit, suspect) = kernel_occ_limit(pane.trace.as_ref().unwrap(), ti, ei);
    assert_eq!(limit, "WARPS");
    assert!(!suspect, "1024B shared mem is under the 48KB default cap");

    pane.multi_select_name = Some(smem_name);
    pane.rebuild_multi_select_stats();
    let (ti, ei) = pane.sel_individual_refs[0];
    let (limit, suspect) = kernel_occ_limit(pane.trace.as_ref().unwrap(), ti, ei);
    assert_eq!(limit, "SMEM");
    assert!(suspect, "180000B shared mem exceeds the 48KB default cap");

    std::fs::remove_file(&path).ok();
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
fn test_nav_search_skips_hidden_cpu_matches() {
    // A "foo" match on a hidden CPU track precedes two on GPU tracks. With CPU
    // hidden (the default), navigation must frame and step only through the
    // visible GPU matches — never the off-screen CPU one.
    let trace = make_trace(
        vec!["", "foo"],
        vec![
            ("python (CPU)", false, vec![ev(10.0, 5.0, 1, 0)]), // search_nav[0], hidden
            ("GPU 0", true, vec![
                ev(50.0, 4.0, 1, 0),   // search_nav[1], visible
                ev(100.0, 4.0, 1, 0),  // search_nav[2], visible
            ]),
        ],
    );
    let mut state = make_state(trace);
    let pane = &mut state.panes[0];
    assert!(!pane.show_cpu, "CPU hidden by default");
    pane.view.t0 = 0.0;
    pane.view.t1 = 1000.0;
    pane.search.push_str("foo");
    pane.rebuild_search();
    assert_eq!(pane.search_nav.len(), 3);

    // Enter frames the first *visible* match (GPU ts=50) and syncs the cursor.
    pane.zoom_to_search();
    assert_eq!(pane.search_cursor, 1);
    let sel = pane.selected.expect("frames a visible match");
    assert_eq!((sel.track_idx, sel.event_idx), (1, 0));

    // Next advances to the other GPU match, not the CPU one.
    pane.nav_search(true);
    assert_eq!(pane.search_cursor, 2);
    assert_eq!(pane.selected.unwrap().track_idx, 1);

    // Next wraps past the hidden CPU match (index 0) back to the first GPU match.
    pane.nav_search(true);
    assert_eq!(pane.search_cursor, 1);
    assert_eq!(pane.selected.unwrap().track_idx, 1);

    // Showing CPU makes the previously-skipped match reachable.
    pane.show_cpu = true;
    pane.nav_search(false);
    assert_eq!(pane.search_cursor, 0);
    assert_eq!(pane.selected.unwrap().track_idx, 0);
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

#[test]
#[ignore]
fn bench_stats_sort_cost() {
    // Profiling harness (not part of the normal suite): measures the actual
    // cost of one `draw_stats_table` resort at realistic row counts, using
    // real trace data through the real `kernel_occ_limit` code path. Run with:
    //   TV_BENCH_TRACE=/path/to/trace.json.gz cargo test --release -- --ignored --nocapture bench_stats_sort_cost
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("skipped: set TV_BENCH_TRACE"); return; }
    };
    let trace = load_trace(&path, &test_counter(), 0, None).unwrap();

    let mut all_stats: Vec<KernelStats> = Vec::new();
    let mut all_refs: Vec<(u32, u32)> = Vec::new();
    for (ti, track) in trace.tracks.iter().enumerate() {
        for (ei, ev) in track.events.iter().enumerate() {
            all_stats.push(KernelStats { name: ev.name, count: 1, total_dur: ev.dur, median_dur: ev.dur, max_dur: ev.dur, min_dur: ev.dur });
            all_refs.push((ti as u32, ei as u32));
        }
    }

    for &n in &[15_000usize, 100_000, all_stats.len()] {
        let n = n.min(all_stats.len());
        let stats = &all_stats[..n];
        let refs = &all_refs[..n];
        eprintln!("--- rows: {} ---", n);

        // Baseline: sort by Total (column 2), a plain float compare.
        let mut idx: Vec<usize> = (0..n).collect();
        let t0 = std::time::Instant::now();
        idx.sort_by(|&a, &b| stats[a].total_dur.partial_cmp(&stats[b].total_dur).unwrap());
        eprintln!("  sort by Total:     {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);

        // Occ Limit column, naive: parses each event's raw args JSON on every
        // comparison (the bug this benchmark exists to catch a regression of).
        let occ_limit = |si: usize| -> &str {
            let (ti, ei) = refs[si];
            crate::ui::kernel_occ_limit(&trace, ti, ei).0
        };
        let mut idx2: Vec<usize> = (0..n).collect();
        let t1 = std::time::Instant::now();
        idx2.sort_by(|&a, &b| occ_limit(a).cmp(occ_limit(b)));
        eprintln!("  sort by Occ Limit (naive):  {:.2}ms", t1.elapsed().as_secs_f64() * 1000.0);

        // Occ Limit column, fixed: parse once (O(n)), then sort the cache —
        // what draw_stats_table actually does now.
        let mut idx3: Vec<usize> = (0..n).collect();
        let t2 = std::time::Instant::now();
        let cache: Vec<&str> = (0..n).map(occ_limit).collect();
        idx3.sort_by(|&a, &b| cache[a].cmp(cache[b]));
        eprintln!("  sort by Occ Limit (cached): {:.2}ms", t2.elapsed().as_secs_f64() * 1000.0);

        // Exact comparator SHAPE from draw_stats_table's sort_by (the 8-way
        // match, with avg/pct closures alive in scope) sorting by Max (column
        // 6) — a plain f64 field, same as Total, but through the real match
        // arm structure instead of an isolated closure. Checks whether the
        // extra branches/closures cost something under lower optimization
        // (the real app runs dev-release: opt-level=2, no LTO, 16 codegen
        // units) even on a numeric column that never touches Occ Limit.
        let total_sum: f64 = stats.iter().map(|s| s.total_dur).sum();
        let avg = |s: &KernelStats| if s.count > 0 { s.total_dur / s.count as f64 } else { 0.0 };
        let pct = |s: &KernelStats| if total_sum > 0.0 { s.total_dur / total_sum } else { 0.0 };
        let sort_col = 6usize;
        let sort_asc = true;
        let mut idx4: Vec<usize> = (0..n).collect();
        let t3 = std::time::Instant::now();
        idx4.sort_by(|&a, &b| {
            let (sa, sb) = (&stats[a], &stats[b]);
            let ord = match sort_col {
                0 => trace.names[sa.name as usize].cmp(&trace.names[sb.name as usize]),
                1 => sa.count.cmp(&sb.count),
                2 => sa.total_dur.partial_cmp(&sb.total_dur).unwrap(),
                3 => pct(sa).partial_cmp(&pct(sb)).unwrap(),
                4 => avg(sa).partial_cmp(&avg(sb)).unwrap(),
                5 => sa.median_dur.partial_cmp(&sb.median_dur).unwrap(),
                6 => sa.max_dur.partial_cmp(&sb.max_dur).unwrap(),
                _ => occ_limit(a).cmp(occ_limit(b)),
            };
            if sort_asc { ord } else { ord.reverse() }
        });
        eprintln!("  sort by Max via full match arm:  {:.2}ms", t3.elapsed().as_secs_f64() * 1000.0);
    }
}

#[test]
fn test_collect_merged_track_events_excludes_events_before_view_t0() {
    // bisect_overlap's starting index is a conservative lower bound: it
    // accounts for the longest duration seen up to any given index, not
    // that specific event's own duration. A long early event (A) can make
    // the bisect land on index 0, after which collect_merged_track_events
    // used to include every subsequent event up to view_t1 without checking
    // whether each one individually still overlapped the window — so a
    // short, long-since-finished event (B) got swept in as if it were
    // visible at the current (zoomed-in) time, well after it actually ended.
    let names = vec!["a", "b"];
    let events = vec![
        ev(0.0, 5000.0, 0, 0),  // A: ends at 5000, legitimately overlaps below
        ev(100.0, 1.0, 1, 0),   // B: ends at 101, long over by the window below
    ];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let hidden = vec![false; trace.names.len()];
    let mut out = Vec::new();
    crate::ui::collect_merged_track_events(&trace.tracks[0], 0, 4500.0, 4600.0, &hidden, &mut out);
    assert_eq!(out.len(), 1, "only A should overlap [4500, 4600], not B: {out:?}");
    assert_eq!(out[0].3, 0, "the surviving event should be A (event index 0)");
}

#[test]
#[ignore]
fn bench_merge_gpu_collect_cost() {
    // Profiling harness: measures collect_merged_track_events (the merged/
    // Tetris-packed-view row builder that runs every redraw when "Merge
    // Streams" is on) over every real GPU stream track in a trace, with
    // execute_context wrapper spans visible vs. hidden. Its has_grandchild
    // check scans forward from each surviving event to its own end, so a
    // whole-generation-step wrapper (hundreds of ms, spanning thousands of
    // descendants) is the worst case for that scan. Run with:
    //   TV_BENCH_TRACE=/path/to/trace.json.gz cargo test --release -- --ignored --nocapture bench_merge_gpu_collect_cost
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("skipped: set TV_BENCH_TRACE"); return; }
    };
    let trace = load_trace(&path, &test_counter(), 0, None).unwrap();

    let exec_names: std::collections::HashSet<u32> = trace.names.iter().enumerate()
        .filter(|(_, n)| n.contains("execute_context"))
        .map(|(i, _)| i as u32)
        .collect();
    eprintln!("names matching execute_context: {}", exec_names.len());

    let mut hidden_none = vec![false; trace.names.len()];
    let mut hidden_exec = vec![false; trace.names.len()];
    for &n in &exec_names { hidden_exec[n as usize] = true; }
    let _ = &mut hidden_none;

    for (ti, track) in trace.tracks.iter().enumerate() {
        if !track.gpu || track.events.len() < 1000 { continue; }
        let n_exec_here = track.events.iter().filter(|e| exec_names.contains(&e.name)).count();
        let mut out = Vec::new();
        let t0 = std::time::Instant::now();
        crate::ui::collect_merged_track_events(track, ti, trace.min_ts, trace.max_ts, &hidden_none, &mut out);
        let visible_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kept_visible = out.len();

        out.clear();
        let t1 = std::time::Instant::now();
        crate::ui::collect_merged_track_events(track, ti, trace.min_ts, trace.max_ts, &hidden_exec, &mut out);
        let hidden_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let kept_hidden = out.len();

        eprintln!(
            "track {ti} ({} events, {n_exec_here} execute_context): visible={visible_ms:.2}ms (kept {kept_visible})  hidden={hidden_ms:.2}ms (kept {kept_hidden})",
            track.events.len(),
        );
    }
}

#[test]
#[ignore]
fn bench_selection_histogram_rebuild_cost() {
    // Profiling harness: measures draw_selection_histogram's per-frame cost
    // (buf.sel_bars rebuild + sort_unstable_by), which — unlike the stats
    // table — has no generation-based cache and reruns on every redraw
    // regardless of whether the selection changed. Run with:
    //   TV_BENCH_TRACE=/path/to/trace.json.gz cargo test --release -- --ignored --nocapture bench_selection_histogram_rebuild_cost
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("skipped: set TV_BENCH_TRACE"); return; }
    };
    let trace = load_trace(&path, &test_counter(), 0, None).unwrap();

    let mut all_durs: Vec<(f64, u32)> = Vec::new();
    for track in &trace.tracks {
        for ev in &track.events {
            all_durs.push((ev.dur, ev.name));
        }
    }

    for &n in &[15_000usize, 71_000, 100_000, all_durs.len()] {
        let n = n.min(all_durs.len());
        let src = &all_durs[..n];
        let t0 = std::time::Instant::now();
        let mut sel_bars: Vec<(f64, u32)> = Vec::with_capacity(n);
        for &(d, name) in src { sel_bars.push((d, name)); }
        sel_bars.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!("rows: {n:>8}  build+sort: {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);
    }
}

#[test]
#[ignore]
fn bench_parallel_occ_limit_parse() {
    // Verifies parallel_occ_limit (used for the Occ Limit column's O(n)
    // sort-cache build) is both correct (matches a sequential pass exactly)
    // and actually faster. On a 9M-event trace this measured 1678ms
    // sequential vs 227ms parallel (14 cores). Run with:
    //   TV_BENCH_TRACE=/path/to/trace cargo test --profile dev-release -- --ignored --nocapture bench_parallel_occ_limit_parse
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("skipped: set TV_BENCH_TRACE"); return; }
    };
    let trace = crate::loader::load_trace(&path, &test_counter(), 0, None).unwrap();

    let mut refs: Vec<(u32, u32)> = Vec::new();
    for (ti, t) in trace.tracks.iter().enumerate() {
        for ei in 0..t.events.len() {
            refs.push((ti as u32, ei as u32));
        }
    }
    eprintln!("total events: {}", refs.len());
    let f = |i: usize| -> (&str, bool) {
        let (ti, ei) = refs[i];
        crate::ui::kernel_occ_limit(&trace, ti, ei)
    };
    let t0 = std::time::Instant::now();
    let cache: Vec<(&str, bool)> = (0..refs.len()).map(f).collect();
    eprintln!("sequential: {:.2}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let t1 = std::time::Instant::now();
    let cache2 = crate::ui::parallel_occ_limit(refs.len(), &f);
    eprintln!("parallel:   {:.2}ms (available_parallelism={})",
        t1.elapsed().as_secs_f64() * 1000.0,
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(0));
    assert_eq!(cache, cache2, "parallel result must match sequential");
}
