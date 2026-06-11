use super::*;
use crate::parse::*;
use crate::loader::load_trace;
use imgui::ImColor32;
use std::collections::HashMap;

fn ev(ts: f64, dur: f64, name: u32, depth: u16) -> Event {
    Event { ts, dur, name, cat: 0, args_start: 0, args_count: 0, depth }
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
        trs.push(Track { label: label.to_string(), gpu, events, max_depth, prefix_max_dur });
    }
    Trace {
        tracks: trs, names: name_strs, cats: vec![String::new()],
        arg_strs: Vec::new(), arg_pairs: Vec::new(), stats: Vec::new(),
        max_ts, total_events, device: String::new(),
    }
}

fn make_state(trace: Trace) -> AppState {
    let event_labels = trace.tracks.iter()
        .map(|t| vec![None; t.events.len()])
        .collect();
    let hidden_names = vec![false; trace.names.len()];
    let collapsed = vec![false; trace.tracks.len()];
    let mut pane = Pane::new();
    pane.event_labels = event_labels;
    pane.hidden_names = hidden_names;
    pane.collapsed = collapsed;
    pane.trace = Some(trace);
    AppState {
        panes: vec![pane],
        active: 0,
        divider_xs: Vec::new(),
        buf: DrawBuf::default(),
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

// --- JSON escape/unescape ---

#[test]
fn test_json_escape_roundtrip() {
    let cases = ["hello", "with\"quotes", "back\\slash", "new\nline", ""];
    for s in cases {
        let escaped = json_escape(s);
        let inner = &escaped[1..escaped.len() - 1];
        assert_eq!(json_unescape(inner), s);
    }
}

#[test]
fn test_json_escape_format() {
    assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
    assert_eq!(json_escape("a\\b"), "\"a\\\\b\"");
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
    let mut index = HashMap::new();
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

#[test]
fn test_event_color_unlabeled() {
    let el: Vec<Vec<Option<u8>>> = vec![vec![None; 3]];
    let labels: Vec<Label> = Vec::new();
    let c = event_color(0, 0, "kern", &el, &labels);
    assert_eq!(Into::<u32>::into(c), Into::<u32>::into(name_color("kern")));
}

#[test]
fn test_event_color_labeled() {
    let color = ImColor32::from_rgba(0x21, 0x96, 0xF3, 255);
    let el = vec![vec![None, Some(0u8), None]];
    let labels = vec![Label { name: "attn".into(), color, pattern: vec![1] }];
    let c = event_color(0, 1, "kern", &el, &labels);
    assert_eq!(Into::<u32>::into(c), Into::<u32>::into(color));
}

#[test]
fn test_event_color_out_of_bounds() {
    let el: Vec<Vec<Option<u8>>> = vec![vec![None]];
    let c = event_color(5, 0, "kern", &el, &[]);
    assert_eq!(Into::<u32>::into(c), Into::<u32>::into(name_color("kern")));
}

// --- Labeling and pattern matching ---

fn gpu_trace() -> (Trace, Vec<&'static str>) {
    let names = vec!["", "exec_ctx", "attn_qkv", "attn_proj", "moe_gate", "moe_expert", "allreduce"];
    let layer = [1, 2, 3, 4, 5, 6];
    let mut events = Vec::new();
    for rep in 0..3 {
        let base = rep as f64 * 600.0;
        for (j, &n) in layer.iter().enumerate() {
            events.push(ev(base + j as f64 * 100.0, 90.0, n, 0));
        }
    }
    let trace = make_trace(names.clone(), vec![("GPU 0", true, events)]);
    (trace, names)
}

#[test]
fn test_label_pattern_exact_match() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([0.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");

    assert_eq!(p.labels.len(), 1);
    assert_eq!(p.labels[0].name, "attn");
    assert_eq!(p.labels[0].pattern, vec![1, 2, 3]);
}

#[test]
fn test_label_repeating_pattern_labels_all_occurrences() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");

    let labeled: Vec<(usize, usize)> = p.event_labels[0].iter().enumerate()
        .filter_map(|(i, l)| l.map(|_| (0, i)))
        .collect();
    assert_eq!(labeled.len(), 6);
}

#[test]
fn test_label_no_orphan_islands() {
    let names = vec!["", "A", "B", "C"];
    let events = vec![
        ev(0.0, 10.0, 1, 0), ev(10.0, 10.0, 2, 0), ev(20.0, 10.0, 3, 0),
        ev(30.0, 10.0, 1, 0), ev(40.0, 10.0, 2, 0), ev(50.0, 10.0, 3, 0),
        ev(60.0, 10.0, 1, 0), ev(70.0, 10.0, 3, 0), ev(80.0, 10.0, 1, 0),
    ];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([0.0, 29.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("abc");

    assert_eq!(p.labels[0].pattern, vec![1, 2, 3]);
    let labeled: Vec<usize> = p.event_labels[0].iter().enumerate()
        .filter_map(|(i, l)| l.map(|_| i))
        .collect();
    assert_eq!(labeled, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_label_hidden_names_excluded_from_pattern() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.hidden_names[1] = true;
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");

    assert!(!p.labels[0].pattern.contains(&1));
}

#[test]
fn test_label_hidden_names_excluded_from_matching() {
    let names = vec!["", "wrap", "A", "B"];
    let events = vec![
        ev(0.0, 10.0, 1, 0), ev(10.0, 10.0, 2, 0), ev(20.0, 10.0, 3, 0),
        ev(30.0, 10.0, 1, 0), ev(40.0, 10.0, 2, 0), ev(50.0, 10.0, 3, 0),
    ];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.hidden_names[1] = true;
    p.selection = Some([10.0, 30.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("ab");

    assert_eq!(p.labels[0].pattern, vec![2, 3]);
    let labeled: Vec<usize> = p.event_labels[0].iter().enumerate()
        .filter_map(|(i, l)| l.map(|_| i))
        .collect();
    assert_eq!(labeled, vec![1, 2, 4, 5]);
}

#[test]
fn test_delete_label() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");
    assert_eq!(p.labels.len(), 1);

    p.delete_label(0);
    assert!(p.labels.is_empty());
    assert!(p.event_labels[0].iter().all(|l| l.is_none()));
}

#[test]
fn test_multiple_labels_no_overlap() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");
    p.selection = Some([300.0, 490.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("moe");

    assert_eq!(p.labels.len(), 2);
    for track_labels in &p.event_labels {
        for label in track_labels.iter().flatten() {
            assert!(*label < 2);
        }
    }
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

// --- Label stats ---

#[test]
fn test_label_stats() {
    let (trace, _) = gpu_trace();
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");

    assert!(!p.label_stats.is_empty());
    let ls = &p.label_stats[0];
    assert_eq!(ls.label_idx, 0);
    assert!(ls.count > 0);
    assert!(ls.total_dur > 0.0);
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

    let trace = load_trace(path.to_str().unwrap()).unwrap();
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

    let trace = load_trace(path.to_str().unwrap()).unwrap();
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

    let result = load_trace(path.to_str().unwrap());
    assert!(result.is_err());

    std::fs::remove_file(&path).ok();
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

    let trace = load_trace(path.to_str().unwrap()).unwrap();
    let track = &trace.tracks[0];
    assert_eq!(track.events.len(), 3);
    let depths: Vec<u16> = track.events.iter().map(|e| e.depth).collect();
    assert_eq!(depths[0], 0);
    assert!(depths[1] > 0 || depths[2] > 0);

    std::fs::remove_file(&path).ok();
}

// --- Label save/load roundtrip ---

#[test]
fn test_label_save_load_roundtrip() {
    let (trace, _) = gpu_trace();
    let dir = std::env::temp_dir().join("tv_test_labels");
    let _ = std::fs::create_dir_all(&dir);
    let trace_path = dir.join("trace.json").to_string_lossy().to_string();

    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.trace_path = trace_path.clone();
    p.selection = Some([100.0, 290.0, 0.0, 1e9]);
    p.rebuild_selection_stats(&mut state.buf);
    p.apply_label("attn");

    let orig_pattern = p.labels[0].pattern.clone();
    let orig_labeled: Vec<Option<u8>> = p.event_labels[0].clone();

    let (trace2, _) = gpu_trace();
    let mut state2 = make_state(trace2);
    let p2 = &mut state2.panes[0];
    p2.trace_path = trace_path.clone();
    p2.load_labels();

    assert_eq!(p2.labels.len(), 1);
    assert_eq!(p2.labels[0].name, "attn");
    assert_eq!(p2.labels[0].pattern, orig_pattern);
    assert_eq!(p2.event_labels[0], orig_labeled);

    let label_path = format!("{}.labels.json", trace_path);
    std::fs::remove_file(&label_path).ok();
}

// --- Args parsing ---

#[test]
fn test_parse_args_flat() {
    let blob = br#"{"key1": "val1", "key2": 42}"#;
    let mut strs = Vec::new();
    let mut idx = HashMap::new();
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
    let mut idx = HashMap::new();
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
