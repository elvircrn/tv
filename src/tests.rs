use super::*;
use crate::parse::*;
use crate::loader::{load_trace, detect_rank_groups, merge_traces, resolve_tool_in};
use crate::state::{parse_rank, find_exec_context_names, default_track_order, compute_steps, compute_gaps_in, selection_gpu_summary, Pane};
/// Test-local shorthand for the unfiltered sweep (production code always
/// goes through compute_gaps_in with a real GapFilter).
fn compute_gaps(trace: &Trace) -> Vec<GapInfo> {
    compute_gaps_in(trace, GapFilter::default())
}
use crate::ui::{heat_color, heat_dim, gap_why};
use imgui::ImColor32;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// imgui's context is a process-global singleton (C ImGui's current-context
/// rule): `Context::create` panics while another context is still alive, and
/// cargo's test harness runs tests on parallel threads. Every test that
/// builds a Context holds this lock from just before `create` until the
/// context is dropped — declare the guard FIRST so it outlives the context
/// (locals drop in reverse order). Poisoning is ignored: a panicking test
/// unwinds through the context's drop first, so the global is clear either
/// way and one failure shouldn't cascade into every other context test.
static IMGUI_CTX_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn test_counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn ev(ts: f64, dur: f64, name: u32, depth: u16) -> Event {
    Event { ts, dur, name, cat: 0, args_off: 0, args_len: 0, depth }
}

/// `ev` with an explicit cat. `make_trace` hardcodes cats to one empty
/// string, so gap tests overwrite `trace.cats` afterwards and build events
/// with real category ids via this helper. Category constants per test:
/// 0 = kernel, 1 = cpu_op, 2 = runtime, 3 = gpu_user_annotation.
fn evc(ts: f64, dur: f64, name: u32, cat: u32) -> Event {
    Event { ts, dur, name, cat, args_off: 0, args_len: 0, depth: 0 }
}

fn make_trace(names: Vec<&str>, tracks: Vec<(&str, bool, Vec<Event>)>) -> Trace {    let name_strs: Vec<String> = names.into_iter().map(String::from).collect();
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

/// Hand-builds a `MergedPack` from (ts, dur, track_idx, event_idx, depth,
/// lo, hi) tuples for tests that need to pin the pane's merged geometry
/// directly (selection tests etc.) without running the real packing.
/// prefix_max_dur/name/ends are derived to match the pack invariants.
fn make_merged_pack(entries: &[(f64, f64, u32, u32, u16, u16, u16)]) -> MergedPack {
    let mut ts = Vec::with_capacity(entries.len());
    let mut prefix_max_dur = Vec::with_capacity(entries.len());
    let mut running_max = 0.0f64;
    for &(t, d, _, _, _, _, _) in entries {
        ts.push(t);
        if d > running_max { running_max = d; }
        prefix_max_dur.push(running_max);
    }
    MergedPack {
        ts,
        ends: entries.iter().map(|&(t, d, _, _, _, _, _)| t + d).collect(),
        prefix_max_dur,
        refs: entries.iter().map(|&(_, _, ti, ei, _, _, _)| (ti, ei)).collect(),
        name: entries.iter().map(|&(_, _, ti, ei, _, _, _)| {
            // Caller's tuples reference real trace events; name comes from
            // the trace at pack time — fill it in after, via override below,
            // since the helper doesn't see the trace. Use 0; tests that need
            // real names set them through capture/extract paths that read
            // `refs`, not `name` (name is only read on the sub-pixel render
            // path).
            let _ = (ti, ei);
            0
        }).collect(),
        depth: entries.iter().map(|&(_, _, _, _, d, _, _)| d).collect(),
        stretch: entries.iter().map(|&(_, _, _, _, _, lo, hi)| (lo, hi)).collect(),
        max_depth: entries.iter().map(|&(_, _, _, _, d, _, _)| d + 1).max().unwrap_or(1),
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
        pending_active_tab: None,
        buf,
        bottom_h: DETAIL_H,
        tabs_w: 0.0,
        drag: DragKind::None,
        show_diff: false,
        diff_popup_open: false,
        diff_result: None,
        diff_bar_scroll: 0.0,
        diff_bar_zoom: 1.0,
        diff_pane_indices: None,
        kernel_ctx_menu: None,
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

// --- step_band_px (ruler step-band clamping) ---

/// A 1200px ruler like the real one: [x0, y0, x1, y1].
const RULER: [f32; 4] = [100.0, 0.0, 1300.0, 20.0];

#[test]
fn test_step_band_px_inside_view_unclamped() {
    // Step fully inside the view maps 1:1 to its proportional slice.
    let view = View { t0: 0.0, t1: 1000.0, ..Default::default() };
    let (x0, x1) = step_band_px(250.0, 750.0, &view, RULER);
    assert!((x0 - 400.0).abs() < 0.5, "x0 {x0} != 400");
    assert!((x1 - 1000.0).abs() < 0.5, "x1 {x1} != 1000");
}

#[test]
fn test_step_band_px_wider_than_view_clamps_to_ruler_edges() {
    // Step spanning past both edges fills the whole ruler, at ANY zoom —
    // this is the frame_step layout (step framed with slack), and the case
    // the old `.min(range)` clamp broke: it clamped a pixel offset by the
    // view's µs range, so any zoom tighter than ~1 band-per-screen
    // collapsed the fill to a left-edge sliver that tracked the zoom level.
    let view = View { t0: 400.0, t1: 600.0, ..Default::default() };
    let (x0, x1) = step_band_px(100.0, 900.0, &view, RULER);
    assert!((x0 - RULER[0]).abs() < 0.5, "x0 {x0} should pin to the left edge");
    assert!((x1 - RULER[2]).abs() < 0.5, "x1 {x1} should pin to the right edge");
}

#[test]
fn test_step_band_px_stays_pinned_while_zooming_in() {
    // The "moves erratically when I zoom in" regression: keep the view
    // centered inside one long step and shrink the range from the whole step
    // down to kernel scale — the band must stay glued to both ruler edges
    // through every zoom level, never sliding or narrowing.
    let mut view = View { t0: 0.0, t1: 1000.0, ..Default::default() };
    let mut range = 1000.0;
    while range >= 0.001 {
        let c = 500.0;
        view.t0 = c - range / 2.0;
        view.t1 = c + range / 2.0;
        let (x0, x1) = step_band_px(0.0, 1000.0, &view, RULER);
        assert!((x0 - RULER[0]).abs() < 0.5 && (x1 - RULER[2]).abs() < 0.5,
            "range {range}: band [{x0}, {x1}] left the ruler edges");
        range /= 4.0;
    }
}

#[test]
fn test_step_band_px_right_edge_clamp_is_pixels_not_time() {
    // A step whose end is past the right edge at a zoom where the OLD clamp
    // bound: pixel offset (600) < true extent, but the µs range (300) is
    // smaller still, so the old code truncated the band to 300px. The band
    // must instead run to the ruler's right edge.
    let view = View { t0: 0.0, t1: 300.0, ..Default::default() };
    let (x0, x1) = step_band_px(150.0, 1000.0, &view, RULER);
    assert!((x0 - 700.0).abs() < 0.5, "x0 {x0} != 700");
    assert!((x1 - RULER[2]).abs() < 0.5, "x1 {x1} should clamp at the ruler edge, not at range px");
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

// --- Tab strip visibility ---

#[test]
fn test_should_show_tab_strip() {
    // Single default pane, nothing ever opened -> no tab row.
    let panes = vec![Pane::new()];
    assert!(!should_show_tab_strip(&panes));

    // Same single pane, but something has been opened into it (trace_path
    // gets set immediately by Pane::open, before loading even finishes).
    let mut opened = Pane::new();
    opened.trace_path = "foo.json".to_string();
    assert!(should_show_tab_strip(&[opened]));

    // More than one pane, even if both are still empty (e.g. right after
    // add_pane before a drop's open() call lands) -> show the row.
    assert!(should_show_tab_strip(&[Pane::new(), Pane::new()]));
}

// --- Merge-view cache invalidation ---

#[test]
fn test_reload_resets_merge_cache_key() {
    // Regression test: the merged multi-rank view's Tetris-packing cache
    // (Pane::merge_cache_key) is keyed on (view range, hidden names, track
    // order) — none of which necessarily change across a reload, even
    // though the reloaded trace's events (and their indices) can be
    // completely different. A stale cache hit after a reload panics with an
    // out-of-bounds index once the render loop looks up a cached
    // (track_idx, event_idx) pair against the new, smaller trace.
    let trace1 = make_trace(vec!["", "k"], vec![("GPU 0", true, vec![ev(0.0, 1.0, 1, 0)])]);
    let mut pane = Pane::new();
    pane.trace = Some(trace1);
    pane.merge_cache_key = Some((vec![false], vec![0]));

    let trace2 = make_trace(vec!["", "k"], vec![("GPU 0", true, vec![ev(0.0, 1.0, 1, 0)])]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(Ok(trace2)).unwrap();
    pane.loading = Some(rx);
    pane.poll_loading();

    assert!(pane.merge_cache_key.is_none(), "reload must invalidate the merge cache");
}

// Regression test for the left-edge popping bug: while panning right (holding
// D), events straddling the left edge visibly re-laned and re-stretched frame
// after frame. Cause: the Tetris packing was window-relative — an off-screen
// event overlapping a left-edge-straddling event influenced its depth, and
// dropped out of the collect window as the pan progressed, re-packing
// everything near the edge. Fix: the pack is whole-span/view-independent and
// only (hidden_names, track_order) key it. This test pins the structural
// property: a left-edge event's packed depth and stretch must be IDENTICAL
// whether the window is the whole trace or a tight pan that cuts off its
// off-screen influencers.
#[test]
fn test_merged_pack_is_view_independent_at_left_edge() {
    // Track 0: event B [0,50) — the off-screen influencer once panned past.
    // Track 1: event A [10,20) — overlaps B, so packs at depth 1 while B is
    // in the window; A's stretch can't claim depth 0 while B spans it.
    // Event C [30,40) overlaps only B (depth 1).
    let trace = make_trace(vec!["a"], vec![
        ("GPU 0", true, vec![ev(0.0, 50.0, 0, 0)]),
        ("GPU 1", true, vec![ev(10.0, 10.0, 0, 0), ev(30.0, 10.0, 0, 0)]),
    ]);
    let hidden = vec![false; trace.names.len()];

    let pack = crate::ui::build_merged_group_events(&trace, &[0, 1], &hidden);

    // Find A (track 1, idx 0) and C (track 1, idx 1) in the packed refs.
    let a = pack.refs.iter().position(|&(ti, ei)| ti == 1 && ei == 0).unwrap();
    let c = pack.refs.iter().position(|&(ti, ei)| ti == 1 && ei == 1).unwrap();

    // Whole-trace expectations: B takes depth 0, both track-1 events sit at
    // depth 1 with no stretch (depth 0 is occupied across their spans).
    assert_eq!(pack.depth[a], 1, "A packs above B while both are packed");
    assert_eq!(pack.depth[c], 1);
    assert_eq!(pack.stretch[a], (1, 1), "A cannot stretch into B's occupied depth 0");
    assert_eq!(pack.stretch[c], (1, 1));

    // The pack has no window — the SAME arrays serve every view. Simulate
    // the render loop's windowed read for a pan that has scrolled B's start
    // out of view (B still overlaps [10,45) so it's still in the slice via
    // overlap_start): A's depth/stretch read from the pack are unchanged.
    let start = pack.overlap_start(10.0);
    let end = pack.ts_end(45.0);
    let a_slice = pack.refs[start..end].iter().position(|&(ti, ei)| ti == 1 && ei == 0).unwrap() + start;
    assert_eq!(pack.depth[a_slice], pack.depth[a], "A's depth must not depend on the window");
    assert_eq!(pack.stretch[a_slice], pack.stretch[a], "A's stretch must not depend on the window");
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
fn test_reselecting_same_region_twice_reproduces_stats() {
    let names = vec!["", "A", "B"];
    let events = vec![ev(0.0, 10.0, 1, 0), ev(10.0, 20.0, 2, 0), ev(30.0, 10.0, 1, 0)];
    let trace = make_trace(names, vec![("GPU 0", true, events)]);
    let mut state = make_state(trace);
    let p = &mut state.panes[0];

    p.selection = Some([0.0, 40.0, 0.0, 1e9]);
    p.finish_selection(&mut state.buf);
    assert_eq!(p.selection_stats.len(), 2, "first selection");

    // Redo the exact same drag.
    p.selection = Some([0.0, 40.0, 0.0, 1e9]);
    p.finish_selection(&mut state.buf);
    assert_eq!(p.selection_stats.len(), 2, "second identical selection");

    p.selection = Some([0.0, 40.0, 0.0, 1e9]);
    p.finish_selection(&mut state.buf);
    assert_eq!(p.selection_stats.len(), 2, "third identical selection");
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
fn test_capture_sel_events_direction_independent() {
    // A shift-drag from bottom-right to top-left stores the same rectangle
    // as top-left to bottom-right, just with start/end (and therefore corner
    // order) swapped — capture_sel_events must resolve both to the exact
    // same set of events.
    let trace = make_trace(
        vec!["", "a", "b", "c"],
        vec![
            ("GPU 0", true, vec![ev(0.0, 10.0, 1, 0), ev(20.0, 10.0, 2, 0)]),
            ("GPU 1", true, vec![ev(5.0, 10.0, 3, 0)]),
        ],
    );
    let state = make_state(trace);
    let p = &state.panes[0];
    let y_bottom = p.geom.y_offsets[1] as f64 + p.geom.heights[1] as f64;

    // Rect spans both tracks and t in [0, 30] however you'd naturally drag it.
    let top_left_to_bottom_right = [0.0, 30.0, 0.0, y_bottom];
    let bottom_right_to_top_left = [30.0, 0.0, y_bottom, 0.0];

    let a = p.capture_sel_events(top_left_to_bottom_right);
    let b = p.capture_sel_events(bottom_right_to_top_left);
    assert_eq!(a.len(), 3, "sanity: all 3 events should be within this rect");
    let a_set: std::collections::HashSet<_> = a.into_iter().collect();
    let b_set: std::collections::HashSet<_> = b.into_iter().collect();
    assert_eq!(a_set, b_set, "drag direction must not change which events get selected");
}

#[test]
fn test_collapsed_track_selection_matches_rendered_lanes() {
    // A collapsed track renders ONLY its depth-0 events, spanning the whole
    // (single-lane) row. capture_sel_events must resolve the same thing: a
    // drag over a collapsed row selects the visible top-level events, not
    // the hidden deeper lanes (which the old depth*sub_h mapping also
    // squeezed into the same row).
    let trace = make_trace(
        vec!["", "outer", "mid", "inner"],
        vec![(
            "Thread 0",
            false,
            vec![
                ev(0.0, 100.0, 1, 0),
                ev(10.0, 50.0, 2, 1),
                ev(20.0, 20.0, 3, 2),
            ],
        )],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.show_cpu = true;
    p.collapsed = vec![true];
    // Mirror what draw_timeline's layout does for a collapsed track: the row
    // is one SUB_LANE_H lane regardless of max_depth.
    p.geom.heights[0] = track_height(3, true, 1.0);

    let row_bottom = p.geom.heights[0] as f64;
    let sel = p.capture_sel_events([0.0, 100.0, 0.0, row_bottom]);
    assert_eq!(sel.len(), 1, "collapsed row must select only the depth-0 event, got {:?} (depths of selected: hidden children must not leak in)", sel);
    let (_, ei) = sel[0];
    assert_eq!(ei, 0, "the selected event must be the top-level one");

    // And the depth-0 event must be capturable from the BOTTOM of the row
    // too — the render maps it across the full lane, not a top sliver.
    let sel_bottom_half = p.capture_sel_events([0.0, 100.0, row_bottom * 0.5, row_bottom]);
    assert_eq!(sel_bottom_half.len(), 1, "dragging over the bottom half of a collapsed row must still select the depth-0 event");
}

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
    // Opening a trace no longer writes caches as a side effect — export
    // explicitly, the way File > Export Merged TV Cache does.
    let cache_path = crate::loader::export_single_cache(&original, path_str)
        .expect("explicit cache export should write the cache file");

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
fn test_median_inplace() {
    use crate::types::median_inplace;
    assert_eq!(median_inplace(&mut []), 0.0);
    assert_eq!(median_inplace(&mut [5.0]), 5.0);
    // Odd length: exact middle element, regardless of input order.
    assert_eq!(median_inplace(&mut [3.0, 1.0, 2.0]), 2.0);
    // Even length: average of the two middle elements.
    assert_eq!(median_inplace(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    // Duplicates and negatives, still unsorted going in.
    assert_eq!(median_inplace(&mut [-1.0, -1.0, 5.0, 5.0]), 2.0);
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
fn test_stretch_bounds_lone_event_fills_whole_row() {
    // depth 0 has our event, depths 1 and 2 have nothing at all — it should
    // claim the entire row.
    let per_depth = vec![vec![(0.0, 10.0)], vec![], vec![]];
    assert_eq!(crate::ui::stretch_bounds(&per_depth, 0, 0.0, 10.0), (0, 2));
}

#[test]
fn test_stretch_bounds_blocked_by_real_overlap() {
    // depth 1 has a sibling that overlaps our event's window, so it can't
    // stretch past depth 0; depth 2 is free the whole time and still claimable.
    let per_depth = vec![vec![(0.0, 10.0)], vec![(5.0, 6.0)], vec![]];
    assert_eq!(crate::ui::stretch_bounds(&per_depth, 0, 0.0, 10.0), (0, 0));
}

#[test]
fn test_stretch_bounds_not_blocked_by_non_overlapping_neighbor() {
    // depth 1's occupant runs entirely before our event starts, so it
    // doesn't block the stretch even though depth 1 isn't literally empty.
    let per_depth = vec![vec![(10.0, 20.0)], vec![(0.0, 5.0)]];
    assert_eq!(crate::ui::stretch_bounds(&per_depth, 0, 10.0, 20.0), (0, 1));
}

#[test]
fn test_stretch_bounds_middle_depth_stretches_both_ways_when_free() {
    let per_depth = vec![vec![], vec![(0.0, 10.0)], vec![]];
    assert_eq!(crate::ui::stretch_bounds(&per_depth, 1, 0.0, 10.0), (0, 2));
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
fn test_format_arg_value_expands_nested_object() {
    let mut out = String::new();
    format_arg_value(&mut out, r#"{"limitingFactors": "REGS", "blockLimitRegs": 1}"#, 0);
    assert_eq!(out, "\n  limitingFactors: REGS\n  blockLimitRegs: 1");
}

#[test]
fn test_format_arg_value_passes_through_scalars() {
    let mut out = String::new();
    format_arg_value(&mut out, "REGS", 0);
    assert_eq!(out, "REGS");

    let mut out = String::new();
    format_arg_value(&mut out, "{}", 0);
    assert_eq!(out, "{}");
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
    p.geom.merged = vec![MergedGeom { vi: 0, pack: make_merged_pack(&[
        (0.0, 10.0, 0, 1, 0, 0, 0),
        (20.0, 10.0, 0, 2, 1, 1, 1),
    ]).into() }];
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

// Merged-row selection precision, both halves of the rule:
//  - an event FULLY INSIDE the dragged time span is captured wherever its
//    lane sits: a thin horizontal band means "this time range on this rank",
//    not "these two rendering lanes" — events sharing a Tetris lane are
//    disjoint by construction, so a lane-sliced capture can't overlap and
//    the overlap-aware totals silently equal the plain sums;
//  - events STRADDLING the span's edges keep exact lane precision, which is
//    what preserves the older fix this test originally pinned ("selecting
//    one item selected all items"): a shift-click is a zero-width rect that
//    can't fully contain anything, so it still selects exactly the one bar
//    under the cursor.
// sub_h = 40/2 = 20, so a y-range of [0,10] hits only depth 0.
#[test]
fn test_merged_selection_respects_depth_yrange() {
    let trace = make_trace(
        vec!["wrapper", "kA", "kB", "kC", "kD"],
        vec![("GPU", true, vec![
            ev(0.0, 100.0, 0, 0),  // idx 0: whole-stream wrapper — stripped from merged row
            ev(0.0, 10.0, 1, 0),   // idx 1: kA [0,10), depth 0
            ev(2.0, 6.0, 4, 1),    // idx 2: kD [2,8), depth 1 — concurrent with kA
            ev(12.0, 6.0, 3, 1),   // idx 3: kC [12,18), depth 1 — fully inside the band's span
            ev(20.0, 10.0, 2, 1),  // idx 4: kB [20,30), depth 1 — straddles the span's end
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.geom.merged = vec![MergedGeom { vi: 0, pack: make_merged_pack(&[
        (0.0, 10.0, 0, 1, 0, 0, 0),
        (2.0, 6.0, 0, 2, 1, 1, 1),
        (12.0, 6.0, 0, 3, 1, 1, 1),
        (20.0, 10.0, 0, 4, 1, 1, 1),
    ]).into() }];
    p.geom.heights[0] = 40.0;
    p.geom.y_offsets[0] = 0.0;

    // Thin band over depth 0 only, time span [5,25): kA straddles the start
    // (lane test → depth 0, touched → in), kD straddles the start (lane test
    // → depth 1, untouched → out), kC is fully contained (in regardless of
    // lane), kB straddles the end (lane test → depth 1 → out).
    p.finished_sel = Some([5.0, 25.0, 0.0, 10.0]);
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
    let events = p.extract_selection_events();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["kA", "kC"], "straddlers respect the depth y-range; contained events are lane-agnostic");

    // Shift-click (zero-width point rect) inside kA's bar: containment is
    // impossible for a nonzero-duration event, so only the lane test applies
    // — the concurrent kD one lane down must NOT ride along.
    p.finished_sel = Some([4.0, 4.0, 5.0, 5.0]);
    p.finished_sel_events = p.capture_sel_events(p.finished_sel.unwrap()).into_iter().collect();
    let events = p.extract_selection_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "kA");
}

// The reported bug: with "Overlap aware" off (the boot default), select a
// thin horizontal band over a merged rank row and then enable it — nothing
// changed. The band only crossed one Tetris lane, and same-lane events are
// disjoint by construction, so the captured set's union equaled its sum no
// matter how the totals were computed. The band must capture the rank's
// whole kernel set in its time range for the toggle to have anything to
// shrink.
#[test]
fn test_merged_thin_band_captures_all_lanes_for_overlap_totals() {
    // One rank row: kernel K launched twice concurrently (exactly why
    // packing puts the two launches in different lanes), plus a sequential
    // kernel S behind them.
    let trace = make_trace(
        vec!["K", "S"],
        vec![("GPU", true, vec![
            ev(0.0, 10.0, 0, 0),  // idx 0: K, packs to depth 0
            ev(0.0, 10.0, 0, 0),  // idx 1: K, concurrent — packs to depth 1
            ev(20.0, 5.0, 1, 0),  // idx 2: S, depth 0
        ])],
    );
    let mut state = make_state(trace);
    let p = &mut state.panes[0];
    p.geom.merged = vec![MergedGeom { vi: 0, pack: make_merged_pack(&[
        (0.0, 10.0, 0, 0, 0, 0, 0),
        (0.0, 10.0, 0, 1, 1, 1, 1),
        (20.0, 5.0, 0, 2, 0, 0, 0),
    ]).into() }];
    p.geom.heights[0] = 40.0; // max_depth 2 * SUB_LANE_H(20)
    p.geom.y_offsets[0] = 0.0;

    // Thin band over lane 0 only (y in [0,10]), time span covering
    // everything: the concurrent K on lane 1 must be captured too.
    p.selection = Some([0.0, 30.0, 0.0, 10.0]);
    p.finish_selection(&mut state.buf);

    let ki = p.selection_stats.iter().position(|s| s.name == 0).unwrap();
    assert_eq!(p.selection_stats[ki].count, 2, "both concurrent K launches must be captured from a thin band");
    assert_eq!(p.selection_stats[ki].total_dur, 20.0);
    let si = p.selection_stats.iter().position(|s| s.name == 1).unwrap();
    assert_eq!(p.selection_stats[si].count, 1, "the touched lane's own events are still captured");

    // Overlap aware: the two K launches cover the same wall time and K is
    // the only selected kernel in it — shared time 10, not the summed
    // 20. Before the capture fix only one launch was captured, so this
    // equaled the sum and the toggle "did nothing".
    p.sel_overlap = true;
    p.ensure_sel_overlap_totals();
    assert_eq!(p.sel_overlap_totals[ki], 10.0, "shared time of the two concurrent launches, not the 20us sum");
    assert!(p.sel_overlap_totals[ki] < p.selection_stats[ki].total_dur);
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
    assert_eq!(groups[0].0.len(), 3);
    assert_eq!(groups[0].0[0].0, 0);
    assert_eq!(groups[0].0[1].0, 1);
    assert_eq!(groups[0].0[2].0, 2);
    assert_eq!(groups[0].1, None);
    assert_eq!(standalone, vec![("standalone.json".to_string(), None)]);
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
    assert_eq!(groups[0].0.len(), 3);
    assert_eq!(groups[0].0[0].0, 0);
    assert_eq!(groups[0].0[1].0, 1);
    assert_eq!(groups[0].0[2].0, 2);
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
fn test_detect_rank_groups_isolates_directories() {
    // Two different runs that happen to use identically-named rank files
    // (the common case: the same trace tool dumps "trace-rank-N.json" every
    // time) must never merge into one group just because they share a
    // directory-relative filename — each directory argument is its own
    // isolated grouping universe.
    let base = std::env::temp_dir().join("tv_test_dir_isolation");
    let dir_a = base.join("run_a");
    let dir_b = base.join("run_b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();
    for dir in [&dir_a, &dir_b] {
        std::fs::write(dir.join("trace-rank-0.json"), "{}").unwrap();
        std::fs::write(dir.join("trace-rank-1.json"), "{}").unwrap();
    }

    let paths = vec![dir_a.to_str().unwrap().to_string(), dir_b.to_str().unwrap().to_string()];
    let (groups, standalone) = detect_rank_groups(&paths);

    assert_eq!(groups.len(), 2, "each directory's rank files must form their own group");
    assert!(standalone.is_empty());
    for (group, _) in &groups {
        assert_eq!(group.len(), 2);
    }
    let dirs_seen: std::collections::HashSet<_> = groups.iter().map(|(_, d)| d.clone()).collect();
    assert_eq!(dirs_seen.len(), 2, "each group must be tagged with its own source directory");
    assert!(dirs_seen.contains(&Some(dir_a.to_str().unwrap().to_string())));
    assert!(dirs_seen.contains(&Some(dir_b.to_str().unwrap().to_string())));

    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn test_is_trace_file() {
    assert!(crate::loader::is_trace_file("foo.json"));
    assert!(crate::loader::is_trace_file("foo.json.gz"));
    assert!(crate::loader::is_trace_file("foo.tar.gz"));
    assert!(crate::loader::is_trace_file("foo.tgz"));
    assert!(crate::loader::is_trace_file("foo.pt.trace.json.ktz"));
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

// Measures a *full* draw_timeline call (real imgui context, no renderer
// backend needed — draw lists are plain in-memory data) at several zoom
// levels, to see where the frame budget actually goes beyond the
// merge-group build that bench_merge_filter isolates. Run with:
//   cargo test --release bench_draw_timeline -- --ignored --nocapture
//
// TV_BENCH_TRACE accepts either a single pre-merged .tvcache file, or a
// directory of per-rank files (the same thing a `tv <dir>` CLI open takes) —
// the latter is loaded and merged in parallel here, mirroring the `--bench`
// CLI path in main.rs, so this bench works straight off a fresh sweep-result
// folder without needing an already-exported merged cache first.
#[test]
#[ignore]
fn bench_draw_timeline() {
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("set TV_BENCH_TRACE=<path.tvcache or a directory> to run this bench"); return; }
    };
    let trace = if std::path::Path::new(&path).is_dir() {
        let (rank_groups, standalone) = detect_rank_groups(&[path.clone()]);
        let mut all_paths: Vec<(usize, String)> = Vec::new();
        for (group, _) in &rank_groups { all_paths.extend(group.clone()); }
        for (i, (p, _)) in standalone.iter().enumerate() { all_paths.push((all_paths.len() + i, p.clone())); }
        let counter = test_counter();
        use rayon::prelude::*;
        let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / all_paths.len().max(1)).max(2);
        let traces: Vec<(usize, Trace)> = all_paths.par_iter().filter_map(|(rank, p)| {
            load_trace(p, &counter, tpf, None).ok().map(|t| (*rank, t))
        }).collect();
        eprintln!("loaded {} rank files from {path}", traces.len());
        merge_traces(traces)
    } else {
        let counter = test_counter();
        load_trace(&path, &counter, 8, None).expect("load trace")
    };
    eprintln!("loaded: {} tracks, {} events, span 0..{:.0} us", trace.tracks.len(), trace.total_events, trace.max_ts);

    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();

    let mut pane = Pane::new();
    pane.merge_gpu = true;
    pane.track_order = default_track_order(&trace.tracks);
    pane.hidden_names = vec![false; trace.names.len()];
    pane.collapsed = vec![false; trace.tracks.len()];
    pane.track_scales = vec![1.0; trace.tracks.len()];
    let max_ts = trace.max_ts;
    pane.trace = Some(trace);

    let mut buf = DrawBuf::default();
    let mut drag = DragKind::None;

    // One forced whole-span pack build (what a hidden-names/track-order
    // change, or the first merged frame after opening, costs), then the
    // per-frame cost with the view CHANGING every frame — a continuous
    // pan/zoom. The pack is view-independent now (see MergedPack), so pan
    // frames are pure windowed rendering; the "cache miss every frame"
    // reset the old bench did would force a full repack per frame, which
    // real interaction never triggers.
    pane.view.t0 = 0.0;
    pane.view.t1 = max_ts;
    pane.merge_cache_key = None;
    let t_pack = std::time::Instant::now();
    {
        let ui = imgui.new_frame();
        ui.window("bench")
            .position([0.0, 0.0], imgui::Condition::Always)
            .size([1600.0, 900.0], imgui::Condition::Always)
            .build(|| {
                let trace_ref = pane.trace.as_ref().unwrap();
                draw_timeline(
                    ui, trace_ref, &mut pane.view, pane.show_cpu, &mut buf,
                    [0.0, 0.0, 1600.0, 900.0], 0, false, false, false,
                    [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, false, false,
                    &pane.search_mask, pane.selection, &pane.finished_sel_events,
                    &mut pane.collapsed, &pane.hidden_names, pane.selected, pane.multi_select_name,
                    &pane.sel_mask, pane.label_w, &mut pane.track_scales, &mut pane.even_spacing,
                    &mut pane.geom, &mut pane.track_order, &mut drag, pane.merge_gpu, 0.016,
                    &mut pane.pending_focus, &mut pane.merge_cache_key, &mut pane.merged_gpu_groups,
                    &pane.steps, pane.step_cursor, false, None, None,
                );
            });
        imgui.render();
    }
    eprintln!("one-time whole-span pack build: {:.1} ms", t_pack.elapsed().as_secs_f64() * 1000.0);

    // Continuous pan at full zoom-out: view.t0/t1 slide every frame, pack
    // cache stays valid.
    let start = std::time::Instant::now();
    let iters = 30;
    for i in 0..iters {
        pane.view.t0 = (i as f64) * 100.0;
        pane.view.t1 = pane.view.t0 + max_ts;
        let ui = imgui.new_frame();
        ui.window("bench")
            .position([0.0, 0.0], imgui::Condition::Always)
            .size([1600.0, 900.0], imgui::Condition::Always)
            .build(|| {
                let trace_ref = pane.trace.as_ref().unwrap();
                draw_timeline(
                    ui, trace_ref, &mut pane.view, pane.show_cpu, &mut buf,
                    [0.0, 0.0, 1600.0, 900.0], 0, false, false, false,
                    [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, false, false,
                    &pane.search_mask, pane.selection, &pane.finished_sel_events,
                    &mut pane.collapsed, &pane.hidden_names, pane.selected, pane.multi_select_name,
                    &pane.sel_mask, pane.label_w, &mut pane.track_scales, &mut pane.even_spacing,
                    &mut pane.geom, &mut pane.track_order, &mut drag, pane.merge_gpu, 0.016,
                    &mut pane.pending_focus, &mut pane.merge_cache_key, &mut pane.merged_gpu_groups,
                    &pane.steps, pane.step_cursor, false, None, None,
                );
            });
        imgui.render();
    }
    let per = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    eprintln!("{:<22} {:>7.3} ms/frame (pan at full zoom, view moves every frame)", "zoom=100% pan", per);

    // And the idle-hover case: same view every frame, cache hits throughout.
    pane.view.t0 = 0.0;
    pane.view.t1 = max_ts;
    let start = std::time::Instant::now();
    let iters = 30;
    for i in 0..iters {
        let ui = imgui.new_frame();
        ui.window("bench")
            .position([0.0, 0.0], imgui::Condition::Always)
            .size([1600.0, 900.0], imgui::Condition::Always)
            .build(|| {
                let trace_ref = pane.trace.as_ref().unwrap();
                draw_timeline(
                    ui, trace_ref, &mut pane.view, pane.show_cpu, &mut buf,
                    [0.0, 0.0, 1600.0, 900.0], 0, false, false, false,
                    [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, false, false,
                    &pane.search_mask, pane.selection, &pane.finished_sel_events,
                    &mut pane.collapsed, &pane.hidden_names, pane.selected, pane.multi_select_name,
                    &pane.sel_mask, pane.label_w, &mut pane.track_scales, &mut pane.even_spacing,
                    &mut pane.geom, &mut pane.track_order, &mut drag, pane.merge_gpu, 0.016,
                    &mut pane.pending_focus, &mut pane.merge_cache_key, &mut pane.merged_gpu_groups,
                    &pane.steps, pane.step_cursor, false, None, None,
                );
            });
        imgui.render();
        let _ = i;
    }
    let per = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    eprintln!("{:<22} {:>7.3} ms/frame (idle, view unchanged, cache hit)", "zoom=100% idle", per);
}

// Selecting a large region calls Pane::rebuild_selection_stats, which was
// spending most of its time in an eager sort of the individual-event rows
// that draw_stats_table immediately re-sorts anyway (see its own sort_idx
// cache). Accepts the same TV_BENCH_TRACE (file or directory) as
// bench_draw_timeline. Run with:
//   cargo test --release bench_selection_stats -- --ignored --nocapture
#[test]
#[ignore]
fn bench_selection_stats() {
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("set TV_BENCH_TRACE=<path.tvcache or a directory> to run this bench"); return; }
    };
    let trace = if std::path::Path::new(&path).is_dir() {
        let (rank_groups, standalone) = detect_rank_groups(&[path.clone()]);
        let mut all_paths: Vec<(usize, String)> = Vec::new();
        for (group, _) in &rank_groups { all_paths.extend(group.clone()); }
        for (i, (p, _)) in standalone.iter().enumerate() { all_paths.push((all_paths.len() + i, p.clone())); }
        let counter = test_counter();
        use rayon::prelude::*;
        let tpf = (std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4) / all_paths.len().max(1)).max(2);
        let traces: Vec<(usize, Trace)> = all_paths.par_iter().filter_map(|(rank, p)| {
            load_trace(p, &counter, tpf, None).ok().map(|t| (*rank, t))
        }).collect();
        merge_traces(traces)
    } else {
        let counter = test_counter();
        load_trace(&path, &counter, 8, None).expect("load trace")
    };

    let mut sel: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for (ti, t) in trace.tracks.iter().enumerate() {
        if !t.gpu { continue; }
        for ei in 0..t.events.len() {
            sel.insert((ti as u32, ei as u32));
        }
    }
    eprintln!("selecting {} events across {} names", sel.len(), trace.names.len());

    let mut pane = Pane::new();
    pane.finished_sel = Some([0.0, 0.0, 0.0, 0.0]);
    pane.finished_sel_events = sel;
    pane.trace = Some(trace);
    let mut buf = DrawBuf::default();
    let t0 = std::time::Instant::now();
    pane.rebuild_selection_stats(&mut buf);
    eprintln!("total rebuild_selection_stats: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
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
fn test_build_merged_group_events_stretch_matches_naive_recompute() {
    // Track 0/1 each have an event at [0,10) -- they overlap, so one lands
    // at depth 0 and the other at depth 1. Track 0's second event at
    // [100,10) doesn't overlap anything and gets depth 0 (first available),
    // but depth 1 is genuinely empty across its own [100,110) window, so it
    // should stretch to claim both depths.
    let trace = make_trace(vec!["a"], vec![
        ("GPU 0", true, vec![ev(0.0, 10.0, 0, 0), ev(100.0, 10.0, 0, 0)]),
        ("GPU 1", true, vec![ev(0.0, 10.0, 0, 0)]),
    ]);
    let hidden = vec![false; trace.names.len()];
    let pack = crate::ui::build_merged_group_events(&trace, &[0, 1], &hidden);
    assert_eq!(pack.max_depth, 2);
    assert_eq!(pack.depth.len(), 3);
    assert_eq!(pack.stretch.len(), 3);

    // The render loop now trusts this cached `stretch` instead of rebuilding
    // per_depth and calling stretch_bounds fresh every frame -- confirm it's
    // identical to what that naive per-frame recompute would have produced.
    let mut per_depth: Vec<Vec<(f64, f64)>> = vec![Vec::new(); pack.max_depth as usize];
    for (i, &(ti, ei)) in pack.refs.iter().enumerate() {
        let e = &trace.tracks[ti as usize].events[ei as usize];
        per_depth[pack.depth[i] as usize].push((e.ts, e.ts + e.dur));
    }
    for (i, &(ti, ei)) in pack.refs.iter().enumerate() {
        let e = &trace.tracks[ti as usize].events[ei as usize];
        let expected = crate::ui::stretch_bounds(&per_depth, pack.depth[i], e.ts, e.ts + e.dur);
        assert_eq!(pack.stretch[i], expected, "event {i} (track {ti} idx {ei})");
    }

    let lone_idx = pack.refs.iter().position(|&(ti, ei)| ti == 0 && ei == 1).unwrap();
    assert_eq!(pack.stretch[lone_idx], (0, 1), "lone event should stretch across both depths");
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

// Phase breakdown of the merged-view pack build (`build_merged_group_events`)
// on a real trace. The pack is view-independent (built once per
// hidden_names/track_order change, see MergedPack), so this measures the
// one-time build cost per rank group — collect phase in isolation for
// attribution, production builder end-to-end — plus, for reference, the
// per-frame windowed-render cost the old per-frame rebuild used to dominate.
// Accepts a single pre-merged .tvcache (same as bench_merge_filter). Run with:
//   TV_BENCH_TRACE=<merged.tvcache> cargo test --release bench_merge_phases -- --ignored --nocapture
#[test]
#[ignore]
fn bench_merge_phases() {
    let path = match std::env::var("TV_BENCH_TRACE") {
        Ok(p) => p,
        Err(_) => { eprintln!("set TV_BENCH_TRACE=<merged.tvcache> to run this bench"); return; }
    };
    let counter = test_counter();
    let trace = load_trace(&path, &counter, 8, None).expect("load trace");
    eprintln!("{} tracks, {} events", trace.tracks.len(), trace.total_events);

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
    eprintln!("{} rank groups", groups.len());

    let hidden: Vec<bool> = Vec::new();
    let iters = 10;

    // One-time pack build per group (view-independent — no zoom loop).
    let mut t_collect = std::time::Duration::ZERO;
    let mut t_full = std::time::Duration::ZERO;
    let mut total_packed = 0usize;
    for _ in 0..iters {
        for (_r, tracks) in &groups {
            // Collect exactly as the builder does, timed in isolation
            // for attribution.
            let mut ev_list: Vec<(f64, f64, u32, u32)> = Vec::new();
            let s = std::time::Instant::now();
            for &ti in tracks {
                let gt = &trace.tracks[ti];
                crate::ui::collect_merged_track_events(gt, ti, f64::NEG_INFINITY, f64::INFINITY, &hidden, &mut ev_list);
            }
            t_collect += s.elapsed();
            drop(ev_list);

            // The production builder end-to-end (it re-collects
            // internally, so merge+pack+stretch is t_full minus the
            // collect phase above).
            let s = std::time::Instant::now();
            let pack = crate::ui::build_merged_group_events(&trace, tracks, &hidden);
            t_full += s.elapsed();
            total_packed += pack.depth.len();
        }
    }
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / iters as f64;
    let full_ms = ms(t_full);
    let other_ms = (full_ms - ms(t_collect)).max(0.0);
    eprintln!(
        "pack build (whole span)   packed={:>9}  full build {full_ms:>7.2} ms  (collect {:.2} + merge/pack/stretch {other_ms:.2})  [sequential]",
        total_packed / iters, ms(t_collect),
    );

    // Per-frame windowed slice cost (what every pan/zoom frame now pays
    // instead of a rebuild): binary search + iterate the visible slice.
    let max_ts = trace.max_ts;
    let packs: Vec<crate::types::MergedPack> = groups.iter()
        .map(|(_r, tracks)| crate::ui::build_merged_group_events(&trace, tracks, &hidden))
        .collect();
    for &(label, frac) in &[("100%", 1.0), ("25%", 0.25), ("5%", 0.05)] {
        let t0 = 0.0;
        let t1 = max_ts * frac;
        let s = std::time::Instant::now();
        let mut visited = 0usize;
        for _ in 0..iters {
            for p in &packs {
                let start = p.overlap_start(t0);
                let end = p.ts_end(t1);
                for gi in start..end {
                    if p.ends[gi] >= t0 { visited += 1; }
                }
            }
        }
        let per = s.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!("windowed slice zoom={label:>4}  visited={:>9}  {:>7.2} ms/frame [sequential]", visited / iters, per);
    }
}

// ---- vLLM step derivation (compute_steps / nav_step / frame_step) ----

/// Build a trace like `make_trace`, but with a proper cats table so
/// gpu_user_annotation / user_annotation / kernel categories survive
/// (make_trace hardcodes cat 0).
fn make_stepped_trace(
    cats: Vec<&str>,
    names: Vec<&str>,
    tracks: Vec<(&str, bool, Vec<(f64, f64, usize, usize)>)>, // (ts, dur, name_idx, cat_idx)
) -> Trace {
    let cat_strs: Vec<String> = cats.into_iter().map(String::from).collect();
    let name_strs: Vec<String> = names.into_iter().map(String::from).collect();
    let mut trs = Vec::new();
    let mut max_ts: f64 = 0.0;
    let mut total_events = 0;
    for (label, gpu, raw) in tracks {
        let mut events: Vec<Event> = raw.into_iter()
            .map(|(ts, dur, name, cat)| Event { ts, dur, name: name as u32, cat: cat as u32, args_off: 0, args_len: 0, depth: 0 })
            .collect();
        // The loader guarantees ts-sorted track event lists (compute_steps'
        // binary searches rely on it); the test helper must too.
        events.sort_unstable_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap());
        for e in &events { max_ts = max_ts.max(e.ts + e.dur); }
        total_events += events.len();
        let n_events = events.len();
        trs.push(Track {
            label: label.to_string(), gpu, events,
            max_depth: 1,
            prefix_max_dur: vec![0.0; n_events],
            raw_buf_idx: 0,
        });
    }
    Trace {
        tracks: trs, names: name_strs, cats: cat_strs,
        raw_bufs: Vec::new(), stats: Vec::new(),
        max_ts, min_ts: 0.0, total_events, device: String::new(),
        vllm_version: String::new(),
        dist_rank: -1, dist_world: 0,
        flow_pairs: Vec::new(),
        rank_paths: Vec::new(),
    }
}

/// `make_stepped_trace` + the `rank_paths` a `merge_traces` of the given
/// per-rank source filenames would carry — needed by tests that exercise
/// DP-group batch aggregation (`compute_steps` reads DP indices from these).
fn make_stepped_trace_dp(
    cats: Vec<&str>,
    names: Vec<&str>,
    tracks: Vec<(&str, bool, Vec<(f64, f64, usize, usize)>)>,
    rank_paths: Vec<(usize, String)>,
) -> Trace {
    let mut t = make_stepped_trace(cats, names, tracks);
    t.rank_paths = rank_paths;
    t
}

#[test]
fn test_compute_steps_multi_rank() {
    // 3 steps on the reference rank (Rank 0 stream), skewed +5us on Rank 1,
    // plus kernels inside the steps. Names: 0=execute_context..., 1=kernel.
    let ec = 0usize; // name idx
    let k = 1usize; // name idx
    let c_gua = 0usize; // cat idx: gpu_user_annotation
    let c_kernel = 1usize; // cat idx: kernel
    // Reference track spans: [0,10), [20,30), [40,50)
    let r0 = vec![
        (0.0, 10.0, ec, c_gua), (20.0, 10.0, ec, c_gua), (40.0, 10.0, ec, c_gua),
        (1.0, 2.0, k, c_kernel), (22.0, 2.0, k, c_kernel),
    ];
    // Rank 1: same steps, +5 skew, its own kernels
    let r1 = vec![
        (5.0, 10.0, ec, c_gua), (25.0, 10.0, ec, c_gua), (45.0, 10.0, ec, c_gua),
        (7.0, 1.0, k, c_kernel),
    ];
    let trace = make_stepped_trace(
        vec!["gpu_user_annotation", "kernel"],
        vec!["execute_context_0(0)_generation_1(1)", "some_kernel"],
        vec![("Rank 0 stream 1", true, r0), ("Rank 1 stream 1", true, r1)],
    );
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    assert_eq!(steps.len(), 3);
    // Step 0: min t0=0, max t1=15 (Rank 1's span ends at 15)
    assert!((steps[0].t0 - 0.0).abs() < 1e-9);
    assert!((steps[0].t1 - 15.0).abs() < 1e-9);
    // gpu_dur = mean of assigned span durs = 10
    assert!((steps[0].gpu_dur - 10.0).abs() < 1e-9);
    // both ranks assigned
    assert_eq!(steps[0].n_ranks, 2);
    // kernels in [0,15): r0's at ts=1 and r1's at ts=7
    assert_eq!(steps[0].kernel_count, 2);
    // idle before step 1: 20 - 15 = 5
    assert!((steps[1].idle_before - 5.0).abs() < 1e-9);
    // step 2's kernels: only r0's ts=22 falls in [35,55)... no wait, that's
    // step 1's range. Step 1: [20, 35); kernels at ts=22 -> 1.
    assert_eq!(steps[1].kernel_count, 1);
    // step 2: [40, 55); no kernels
    assert_eq!(steps[2].kernel_count, 0);
}

#[test]
fn test_compute_steps_deterministic_under_full_tie() {
    // `gpu_user_annotation` projects each step onto EVERY CUDA stream, so on
    // a real multi-stream trace dozens of tracks tie at exactly one span per
    // step. All 32 tracks here tie on both count and total duration, so the
    // reference pick runs through to the track-index tie-break — and the
    // result must be identical on every call. The old count-only
    // `max_by_key` over a `HashMap` resolved these ties by hash iteration
    // order, a fresh lottery per call: the same trace computed different
    // step bounds on every recompute (load, clock sync, auto-reload).
    // Track 0 brackets the steps at [0,100) and [200,300); the other 31
    // (identical to each other) at [120,220) and [320,420). Which geometry
    // wins changes the assignment — e.g. under a shifted reference, track
    // 0's [200,300) span overlaps only the shifted step-0 bracket, so it
    // counts toward step 0 and stretches it to [0,300).
    let ec = 0usize;
    let c_gua = 0usize;
    let mut tracks = vec![
        ("Rank 0 stream 0", true, vec![
            (0.0, 100.0, ec, c_gua), (200.0, 100.0, ec, c_gua),
        ]),
    ];
    for _ in 1..32 {
        // Labels needn't be distinct: the pick is by track index, and
        // parse_rank maps them all to rank 0 regardless.
        tracks.push((
            "Rank 0 stream 1",
            true,
            vec![(120.0, 100.0, ec, c_gua), (320.0, 100.0, ec, c_gua)],
        ));
    }
    let trace = make_stepped_trace(
        vec!["gpu_user_annotation"],
        vec!["execute_context_0(0)_generation_1(1)"],
        tracks,
    );
    let ec_names = find_exec_context_names(&trace.names);
    let project = |steps: &[crate::types::StepInfo]| -> Vec<(f64, f64, f64)> {
        steps.iter().map(|s| (s.t0, s.t1, s.gpu_dur)).collect()
    };
    let first = project(&compute_steps(&trace, &ec_names));
    for _ in 0..30 {
        assert_eq!(project(&compute_steps(&trace, &ec_names)), first,
            "compute_steps returned different steps across calls");
    }
    // Lowest index wins the full tie: track 0's brackets are the reference.
    assert_eq!(first.len(), 2);
    assert!((first[0].0 - 0.0).abs() < 1e-9 && (first[0].1 - 100.0).abs() < 1e-9);
    assert!((first[1].0 - 120.0).abs() < 1e-9 && (first[1].1 - 420.0).abs() < 1e-9);
}

#[test]
fn test_compute_steps_duration_tiebreak_beats_track_order() {
    // Count tie between 15 clipped/shifted streams (brackets only part of
    // each step) at the LOWER track indices and one full-bracket stream
    // above them — the real-trace shape, where most projections are partial
    // and only a few streams bracket every step fully. Total span duration
    // must win the reference: a projection can't exceed its CPU annotation's
    // window, so the longest per-step brackets are the ones closest to each
    // step's true GPU extent. A count-only pick that fell back to track
    // order would take a partial stream, whose intervals leave the full
    // stream's spans overlapping nothing — the nearest-interval fallback
    // then attaches [200,300) to step 0 and the step swallows the inter-step
    // gap ([0,300) instead of [0,100)).
    let ec = 0usize;
    let c_gua = 0usize;
    let mut tracks = Vec::new();
    for _ in 0..15 {
        // Partial brackets: [120,190) and [320,390), total dur 140.
        tracks.push((
            "Rank 0 stream P",
            true,
            vec![(120.0, 70.0, ec, c_gua), (320.0, 70.0, ec, c_gua)],
        ));
    }
    // Full brackets: [0,100) and [200,300), total dur 200.
    tracks.push((
        "Rank 0 stream F",
        true,
        vec![(0.0, 100.0, ec, c_gua), (200.0, 100.0, ec, c_gua)],
    ));
    let trace = make_stepped_trace(
        vec!["gpu_user_annotation"],
        vec!["execute_context_0(0)_generation_1(1)"],
        tracks,
    );
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    assert_eq!(steps.len(), 2);
    assert!((steps[0].t0 - 0.0).abs() < 1e-9);
    assert!((steps[0].t1 - 100.0).abs() < 1e-9);
    assert!((steps[1].t0 - 120.0).abs() < 1e-9);
    assert!((steps[1].t1 - 390.0).abs() < 1e-9);
}

#[test]
fn test_compute_steps_gpu_dur_is_median_of_rank_unions() {
    // GPU busy must merge a rank's parallel-stream projections (they overlap
    // in time) before aggregating across ranks, and use the median so one
    // straggler rank can't inflate it. The old metric summed every span and
    // divided by rank count: rank 0's two identical 30us projections counted
    // as 60, and rank 2's straggler dragged the mean — 140/3 = 46.7 instead
    // of the true 30.
    let ec = 0usize;
    let c_gua = 0usize;
    let step = |t0: f64| (t0, 30.0, ec, c_gua);
    let trace = make_stepped_trace(
        vec!["gpu_user_annotation"],
        vec!["execute_context_0(0)_generation_1(1)"],
        vec![
            ("Rank 0 stream 0", true, vec![step(100.0), step(200.0)]),
            ("Rank 0 stream 1", true, vec![step(100.0), step(200.0)]),
            ("Rank 1 stream 0", true, vec![step(100.0), step(200.0)]),
            // Straggler: single 50us projection in step 0 only.
            ("Rank 2 stream 0", true, vec![(100.0, 50.0, ec, c_gua)]),
        ],
    );
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    assert_eq!(steps.len(), 2);
    // Step 0 bounds stretched by the straggler's end, but its busy time
    // doesn't leak into the median: rank unions are 30, 30, 50.
    assert!((steps[0].t0 - 100.0).abs() < 1e-9);
    assert!((steps[0].t1 - 150.0).abs() < 1e-9);
    assert_eq!(steps[0].n_ranks, 3);
    assert!((steps[0].gpu_dur - 30.0).abs() < 1e-9);
    assert!((steps[1].gpu_dur - 30.0).abs() < 1e-9);
}

#[test]
fn test_sel_overlap_totals_split_shared_time() {
    // "Overlap aware" totals split every instant of wall time equally among
    // the selected kernels running it. Same-name concurrency counts once
    // (partial's two overlapping launches on one rank span 25us of wall,
    // not 30; solo's concurrent pair spans 10, not 20); time overlapped by
    // another kernel is discounted to its share, not dropped (dup, covered
    // by partial, is worth 5 of its 10; hidden, inside cover, 2.5 of 5).
    // The conservation law: the totals sum to the busy union of the
    // selection (67us here), so concurrency is never double-counted.
    // SelectionStats' `durations` carry no timestamps, so the sweep must
    // come from the trace through `event_refs`.
    let trace = make_trace(
        vec!["kernel_dup", "kernel_seq", "kernel_partial",
             "kernel_cover", "kernel_hidden", "kernel_solo"],
        vec![
            ("Rank 0 stream 0", true, vec![
                ev(100.0, 10.0, 0, 0),  // dup: [100,110)
                ev(200.0, 6.0, 1, 0),   // seq A: [200,206)
                ev(100.0, 15.0, 2, 0),  // partial A: [100,115)
                ev(300.0, 20.0, 3, 0),  // cover: [300,320)
                ev(305.0, 5.0, 4, 0),   // hidden: [305,310), inside cover
                ev(400.0, 10.0, 5, 0),  // solo A: [400,410)
            ]),
            ("Rank 0 stream 1", true, vec![
                ev(100.0, 10.0, 0, 0),  // dup again, concurrent with itself
                ev(210.0, 6.0, 1, 0),   // seq B: [210,216), after seq A
                ev(110.0, 15.0, 2, 0),  // partial B: [110,125), overlaps A by 5
                ev(400.0, 10.0, 5, 0),  // solo B, concurrent with solo A
            ]),
        ],
    );
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    // Populate selection_stats the way rebuild_selection_stats would;
    // compute_aggregates maps it 1:1 into sel_agg_stats, so row order here
    // is the row order of the overlap totals.
    pane.selection_stats = vec![
        crate::types::SelectionEntry {
            name: 0, count: 2, total_dur: 20.0,
            durations: vec![10.0, 10.0],
            event_refs: vec![(0, 0), (1, 0)],
        },
        crate::types::SelectionEntry {
            name: 1, count: 2, total_dur: 12.0,
            durations: vec![6.0, 6.0],
            event_refs: vec![(0, 1), (1, 1)],
        },
        crate::types::SelectionEntry {
            name: 2, count: 2, total_dur: 30.0,
            durations: vec![15.0, 15.0],
            event_refs: vec![(0, 2), (1, 2)],
        },
        crate::types::SelectionEntry {
            name: 3, count: 1, total_dur: 20.0,
            durations: vec![20.0],
            event_refs: vec![(0, 3)],
        },
        crate::types::SelectionEntry {
            name: 4, count: 1, total_dur: 5.0,
            durations: vec![5.0],
            event_refs: vec![(0, 4)],
        },
        crate::types::SelectionEntry {
            name: 5, count: 2, total_dur: 20.0,
            durations: vec![10.0, 10.0],
            event_refs: vec![(0, 5), (1, 3)],
        },
    ];
    pane.sel_generation += 1; // as compute_aggregates does on every rebuild
    pane.ensure_sel_overlap_totals();
    assert_eq!(pane.sel_overlap_totals.len(), 6);
    // dup shares [100,110) with partial: half of 10.
    assert!((pane.sel_overlap_totals[0] - 5.0).abs() < 1e-9);
    // seq is always alone: full sum.
    assert!((pane.sel_overlap_totals[1] - 12.0).abs() < 1e-9);
    // partial: [100,110) shared with dup (5) + [110,115) both launches,
    // one name (5) + [115,125) alone (10) = 20, not the 30us sum.
    assert!((pane.sel_overlap_totals[2] - 20.0).abs() < 1e-9);
    // cover: [300,305) alone (5) + [305,310) shared with hidden (2.5)
    // + [310,320) alone (10).
    assert!((pane.sel_overlap_totals[3] - 17.5).abs() < 1e-9);
    // hidden: half of its 5us inside cover.
    assert!((pane.sel_overlap_totals[4] - 2.5).abs() < 1e-9);
    // solo: two concurrent launches, one name, nobody else: 10, not 20.
    assert!((pane.sel_overlap_totals[5] - 10.0).abs() < 1e-9);
    // Conservation: totals sum to the busy union — [100,125) + [200,206)
    // + [210,216) + [300,320) + [400,410) = 67us — never the 87us sum.
    let total: f64 = pane.sel_overlap_totals.iter().sum();
    assert!((total - 67.0).abs() < 1e-9);
    // Cached: a repeat call on the same generation must not recompute.
    pane.ensure_sel_overlap_totals();
    assert!((pane.sel_overlap_totals[2] - 20.0).abs() < 1e-9);
    // A new selection (generation bump) invalidates the cache. Truncated to
    // just the dup entry, dup is now the only selected kernel, so its name
    // owns the whole [100,110) span — 10, not the 5 it got while partial
    // shared it.
    pane.selection_stats.truncate(1);
    pane.sel_generation += 1;
    pane.ensure_sel_overlap_totals();
    assert_eq!(pane.sel_overlap_totals.len(), 1);
    assert!((pane.sel_overlap_totals[0] - 10.0).abs() < 1e-9);
}

#[test]
fn test_sel_overlap_totals_scope_per_rank() {
    // Shared time is swept per rank scope (the same parse_rank grouping
    // merged rows use) and summed, so a kernel only shares against kernels
    // on its own rank. A single global sweep would halve rank 1's K
    // against rank 0's J; the plain sum would say K 20 / J 20.
    let trace = make_trace(
        vec!["kernel_k", "kernel_j"],
        vec![
            ("Rank 0 stream 0", true, vec![
                ev(100.0, 10.0, 0, 0),  // K: [100,110)
                ev(95.0, 20.0, 1, 0),   // J: [95,115), covers K
            ]),
            ("Rank 1 stream 0", true, vec![
                ev(100.0, 10.0, 0, 0),  // K again, concurrent across ranks
            ]),
        ],
    );
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    pane.selection_stats = vec![
        crate::types::SelectionEntry {
            name: 0, count: 2, total_dur: 20.0,
            durations: vec![10.0, 10.0],
            event_refs: vec![(0, 0), (1, 0)],
        },
        crate::types::SelectionEntry {
            name: 1, count: 1, total_dur: 20.0,
            durations: vec![20.0],
            event_refs: vec![(0, 1)],
        },
    ];
    pane.sel_generation += 1;
    pane.ensure_sel_overlap_totals();
    // K: rank 0 shares [100,110) with J (5) + rank 1 alone (10) = 15.
    assert!((pane.sel_overlap_totals[0] - 15.0).abs() < 1e-9);
    // J: [95,100) alone (5) + [100,110) shared (5) + [110,115) alone (5).
    assert!((pane.sel_overlap_totals[1] - 15.0).abs() < 1e-9);
    // Conservation across scopes: busy union 20 (rank 0) + 10 (rank 1).
    let total: f64 = pane.sel_overlap_totals.iter().sum();
    assert!((total - 30.0).abs() < 1e-9);
}

#[test]
fn stats_table_sorts_by_overlap_totals_when_overlap_aware() {
    // Two kernels whose summed and shared totals rank differently:
    // "heavy_sum" launches 4x10us all concurrently (sum 40, shared 12 —
    // the span its overlapping launches cover, counted once);
    // "light_sum" launches 3x5us sequentially (sum 15, shared 15).
    // Sorting by Total desc must put heavy_sum first normally and
    // light_sum first once overlap-aware — the re-ranking the
    // Selection-tab checkbox exists to surface. Both draw calls share the
    // pane's sort state and table id (exactly what toggling the checkbox
    // does mid-selection), so this also pins that the sort cache key
    // distinguishes the two modes.
    let trace = make_trace(
        vec!["", "heavy_sum", "light_sum"],
        vec![
            ("Rank 0 stream 0", true, vec![
                ev(100.0, 10.0, 1, 0),
                ev(210.0, 5.0, 2, 0),
                ev(220.0, 5.0, 2, 0),
                ev(230.0, 5.0, 2, 0),
            ]),
            ("Rank 0 stream 1", true, vec![
                ev(100.0, 10.0, 1, 0),
                ev(101.0, 10.0, 1, 0),
                ev(102.0, 10.0, 1, 0),
            ]),
        ],
    );
    let stats = vec![
        crate::types::KernelStats {
            name: 1, count: 4, total_dur: 40.0,
            median_dur: 10.0, max_dur: 10.0, min_dur: 10.0,
        },
        crate::types::KernelStats {
            name: 2, count: 3, total_dur: 15.0,
            median_dur: 5.0, max_dur: 5.0, min_dur: 5.0,
        },
    ];
    let overlap = vec![12.0, 15.0];

    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();
    // Pane-owned sort state, shared across both calls like the real toggle.
    let mut search = String::new();
    let mut search_changed = false;
    let mut sort_col = 2usize; // Total
    let mut sort_asc = false;  // descending
    let mut cache: Option<(u64, usize, bool, usize, bool, bool)> = None;
    let mut sort_idx: Vec<usize> = Vec::new();
    let mut buf = DrawBuf::default();
    let mut run = |overlap: Option<&[f64]>| {
        let ui = imgui.new_frame();
        ui.window("##tblwin")
            .flags(imgui::WindowFlags::NO_SAVED_SETTINGS | imgui::WindowFlags::NO_DECORATION)
            .size([1600.0, 400.0], imgui::Condition::Always)
            .build(|| {
                draw_stats_table(
                    &ui, &trace, &stats, None, overlap, 7,
                    &mut search, &mut search_changed,
                    &mut sort_col, &mut sort_asc, &mut cache, &mut sort_idx,
                    &mut buf, "##selstats",
                );
            });
        drop(ui);
        imgui.render();
        sort_idx.clone()
    };
    assert_eq!(run(None), vec![0, 1], "plain mode: heavy_sum (40us sum) sorts first");
    assert_eq!(run(Some(&overlap)), vec![1, 0],
        "overlap mode: light_sum (15us union) sorts first, and the cache key must force a re-sort");
}

#[test]
fn test_compute_steps_prefers_gpu_side_and_falls_back() {
    let ec = 0usize;
    // CPU-only trace: user_annotation cat, no GPU side
    let cpu_track = vec![(0.0, 5.0, ec, 0usize), (10.0, 5.0, ec, 0usize)];
    let trace = make_stepped_trace(
        vec!["user_annotation"],
        vec!["execute_context_0(0)_generation_1(1)"],
        vec![("Thread 7", false, cpu_track)],
    );
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].n_ranks, 1); // no rank in label; per-track distinct
}

#[test]
fn test_compute_steps_empty_for_non_vllm() {
    let trace = make_stepped_trace(
        vec!["kernel"],
        vec!["some_kernel"],
        vec![("GPU 0", true, vec![(0.0, 1.0, 0usize, 0usize)])],
    );
    assert!(compute_steps(&trace, &[]).is_empty());
}

#[test]
fn test_compute_steps_sums_batches_across_dp_groups() {
    // tp2 x dp2: 4 ranks, two DP groups of two TP ranks each. Every rank in
    // a group annotates the same local batch (1 gen token); the two groups
    // have different local batches for step 0 (1 vs 3 tokens) so the sum is
    // checkable. The token counts must NOT be summed per-rank (which would
    // double each group's contribution to 8).
    let ec0 = 0usize; // name idx: dp0 group's annotation (1 token)
    let ec1 = 1usize; // name idx: dp1 group's annotation (3 tokens)
    let c_gua = 0usize; // cat idx: gpu_user_annotation
    // Group-local annotation: every rank of a DP group carries the same
    // interned name; the two groups differ (their schedulers ran different
    // local batches).
    let step = |t0: f64, name: usize| vec![
        (t0, 10.0, name, c_gua),
        (t0 + 1.0, 2.0, 2usize, 1usize), // kernel
    ];
    let trace = make_stepped_trace_dp(
        vec!["gpu_user_annotation", "kernel"],
        vec![
            "execute_1_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq1sk12sqsq1sqsk12)",
            "execute_3_context_0(sq0sk0sqsq0sqsk0)_generation_1(sq3sk30sqsq9sqsk30)",
            "some_kernel",
        ],
        vec![
            ("Rank 0 stream 1", true, step(0.0, ec0)), // dp0 tp0
            ("Rank 1 stream 1", true, step(0.5, ec0)), // dp0 tp1 (same scheduler)
            ("Rank 2 stream 1", true, step(1.0, ec1)), // dp1 tp0
            ("Rank 3 stream 1", true, step(1.5, ec1)), // dp1 tp1 (same scheduler)
        ],
        vec![
            (0, "dp0_pp0_tp0_dcp0_ep0_rank0.x.json.gz".to_string()),
            (1, "dp0_pp0_tp1_dcp0_ep1_rank1.x.json.gz".to_string()),
            (2, "dp1_pp0_tp0_dcp2_ep2_rank2.x.json.gz".to_string()),
            (3, "dp1_pp0_tp1_dcp0_ep3_rank3.x.json.gz".to_string()),
        ],
    );
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    assert_eq!(steps.len(), 1);
    let b = steps[0].batch.expect("batch must parse");
    // 1 token (dp0) + 3 tokens (dp1) = 4 — not 8 (per-rank double count).
    assert_eq!(b.total_tokens, 4);
    assert_eq!(b.n_gen, 2); // 1 + 1 requests
    assert_eq!(b.gen_tokens, 4);
    // KV sums are per-group local sums: 12 (dp0) + 30 (dp1).
    assert_eq!(b.gen_kv, Some(42));
    // Per-group breakdown is kept alongside the sum.
    assert_eq!(steps[0].group_batches.len(), 2);
    assert_eq!(steps[0].group_batches[0], (0, parse_exec_context_name(&trace.names[ec0]).unwrap()));
    assert_eq!(steps[0].group_batches[1], (1, parse_exec_context_name(&trace.names[ec1]).unwrap()));
    assert_eq!(steps[0].n_ranks, 4);
    assert_eq!(steps[0].kernel_count, 4);
}

#[test]
fn test_nav_step_and_frame_step() {
    let ec = 0usize;
    let r = vec![(0.0, 10.0, ec, 0usize), (20.0, 10.0, ec, 0usize), (40.0, 10.0, ec, 0usize)];
    let trace = make_stepped_trace(
        vec!["gpu_user_annotation"],
        vec!["execute_context_0(0)_generation_1(1)"],
        vec![("Rank 0 stream 1", true, r)],
    );
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    // poll_loading normally does this before recompute_steps.
    pane.exec_context_names = find_exec_context_names(&pane.trace.as_ref().unwrap().names);
    pane.recompute_steps();
    assert_eq!(pane.steps.len(), 3);

    // View centered inside step 0; nav forward snaps the cursor to step 0,
    // then advances to 1.
    pane.view.t0 = -1.0;
    pane.view.t1 = 11.0; // center = 5, inside step 0 [0,10)
    pane.nav_step(true);
    assert_eq!(pane.step_cursor, 1);
    let anim = pane.view.anim.as_ref().unwrap();
    // frames step 1 [20,30) with 5% pad either side: range = 10/(1-0.1)
    let range = 10.0 / (1.0 - 2.0 * STEP_PAD);
    let center = 25.0;
    assert!((anim.to_t0 - (center - range / 2.0)).abs() < 1e-9);
    assert!((anim.to_t1 - (center + range / 2.0)).abs() < 1e-9);

    // Back from a view centered inside step 1: snaps to 1, then back to 0
    pane.view.t0 = 19.0;
    pane.view.t1 = 31.0; // center = 25, inside step 1
    pane.view.anim = None;
    pane.nav_step(false);
    assert_eq!(pane.step_cursor, 0);

    // Wrap-around: view centered inside the last step, forward wraps to 0
    pane.step_cursor = 2;
    pane.view.anim = None;
    pane.view.t0 = 39.0;
    pane.view.t1 = 41.0; // center = 40, inside step 2's [40,50)
    pane.nav_step(true);
    assert_eq!(pane.step_cursor, 0); // wrapped
}

#[test]
#[ignore] // loads a 85MB local trace; run explicitly with --ignored
fn test_steps_on_real_merged_trace() {
    let path = "/Users/ecrncevi/vllm-profiles/20260923-114925.tvcache/_merged.tvcache";
    let trace = match load_trace(path, &test_counter(), 8, None) {
        Ok(t) => t,
        Err(e) => { eprintln!("could not load {path}: {e}, skipping"); return; }
    };
    let ec_names = find_exec_context_names(&trace.names);
    let t0 = std::time::Instant::now();
    let steps = compute_steps(&trace, &ec_names);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("computed {} steps in {ms:.1} ms", steps.len());
    assert!(!steps.is_empty(), "real merged trace must have steps");
    for (i, s) in steps.iter().enumerate() {
        let batch = match s.batch {
            Some(b) => format!(" tokens={} ctx={}({}) gen={}({})",
                b.total_tokens, b.n_ctx, b.ctx_tokens, b.n_gen, b.gen_tokens),
            None => String::new(),
        };
        eprintln!("step {i:>2}: [{:>12.1}, {:>12.1}] wall={:>7.1} gpu={:>7.1} idle={:>7.1} ranks={} kernels={}{}",
            s.t0, s.t1, s.t1 - s.t0, s.gpu_dur, s.idle_before, s.n_ranks, s.kernel_count, batch);
    }
}

#[test]
fn test_parse_exec_context_name_all_formats() {
    use crate::state::parse_exec_context_name;
    // Detailed format (20260914-125236, vLLM with detailed_trace_annotation)
    let b = parse_exec_context_name(
        "execute_512_context_1(sq512sk388352sqsq262144sqsk198836224)_generation_0(sq0sk0sqsq0sqsk0)",
    ).unwrap();
    assert_eq!(b.total_tokens, 512);
    assert_eq!(b.n_ctx, 1);
    assert_eq!(b.ctx_tokens, 512);
    assert_eq!(b.ctx_kv, Some(388352));
    assert_eq!(b.n_gen, 0);
    assert_eq!(b.gen_tokens, 0);
    assert_eq!(b.gen_kv, Some(0));

    // Mixed prefill+decode
    let b = parse_exec_context_name(
        "execute_5_context_1(sq4sk4sqsq16sqsk16)_generation_1(sq1sk11sqsq1sqsk11)",
    ).unwrap();
    assert_eq!(b.total_tokens, 5);
    assert_eq!(b.n_ctx, 1);
    assert_eq!(b.ctx_tokens, 4);
    assert_eq!(b.ctx_kv, Some(4));
    assert_eq!(b.n_gen, 1);
    assert_eq!(b.gen_tokens, 1);
    assert_eq!(b.gen_kv, Some(11));

    // Legacy format (older traces): tokens in parens, no kv
    let b = parse_exec_context_name("execute_context_1(113)_generation_15(15)").unwrap();
    assert_eq!(b.total_tokens, 128); // 113 + 15, derived
    assert_eq!(b.n_ctx, 1);
    assert_eq!(b.ctx_tokens, 113);
    assert_eq!(b.ctx_kv, None);
    assert_eq!(b.n_gen, 15);
    assert_eq!(b.gen_tokens, 15);
    assert_eq!(b.gen_kv, None);

    // Zero-value legacy (20260923-114925): counts are literally 0(0)/1(1)
    let b = parse_exec_context_name("execute_context_0(0)_generation_1(1)").unwrap();
    assert_eq!(b.total_tokens, 1);
    assert_eq!(b.n_gen, 1);
    assert_eq!(b.gen_tokens, 1);

    // Non-annotations must not parse
    assert!(parse_exec_context_name("execute_model").is_none());
    assert!(parse_exec_context_name("execute_mm_encoder").is_none());
    assert!(parse_exec_context_name("vllm::silu_and_mul_kernel<...>").is_none());
}

#[test]
fn test_find_exec_context_names_all_formats() {
    use crate::state::find_exec_context_names;
    let names = vec![
        "execute_model".to_string(),
        "execute_context_0(0)_generation_1(1)".to_string(),
        "execute_512_context_1(sq512sk388352sqsq262144sqsk198836224)_generation_0(sq0sk0sqsq0sqsk0)".to_string(),
        "execute_mm_encoder".to_string(),
        "some_kernel".to_string(),
    ];
    let idxs = find_exec_context_names(&names);
    assert_eq!(idxs, vec![1, 2]);
}

#[test]
#[ignore] // exercises the Steps tab against a real trace; run with --ignored
fn test_steps_table_real_trace() {
    let path = "/Users/ecrncevi/vllm-profiles/20260923-114925.tvcache/_merged.tvcache";
    let trace = match load_trace(path, &test_counter(), 8, None) {
        Ok(t) => t,
        Err(e) => { eprintln!("could not load {path}: {e}, skipping"); return; }
    };
    let ec_names = find_exec_context_names(&trace.names);
    let steps = compute_steps(&trace, &ec_names);
    eprintln!("{} steps", steps.len());
    assert!(!steps.is_empty());

    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();
    let mut buf = DrawBuf::default();
    let mut step_clicked: Option<usize> = None;
    // Render the table the way the bottom panel does, several frames.
    for _ in 0..3 {
        let ui = imgui.new_frame();
        ui.window("steps")
            .position([0.0, 700.0], imgui::Condition::Always)
            .size([1600.0, 200.0], imgui::Condition::Always)
            .build(|| {
                let mut clicked: Option<usize> = None;
                draw_steps_table(&ui, &steps, &mut buf, &mut clicked);
                step_clicked = clicked;
            });
        imgui.render();
    }
    eprintln!("table rendered OK");
}

#[test]
#[ignore] // real-trace smoke test for the detailed-annotation format
fn test_steps_detailed_annotation_trace() {
    // 20260914-125236: vLLM with detailed_trace_annotation — batch counts
    // encoded in span names. Use its merged cache.
    let dir = "/Users/ecrncevi/vllm-profiles/20260914-125236.tvcache";
    let path = format!("{dir}/_merged.tvcache");
    let trace = match load_trace(&path, &test_counter(), 8, None) {
        Ok(t) => t,
        Err(e) => { eprintln!("could not load {path}: {e}, skipping"); return; }
    };
    let ec_names = find_exec_context_names(&trace.names);
    assert!(!ec_names.is_empty(), "detailed-format names must be found");
    let steps = compute_steps(&trace, &ec_names);
    eprintln!("{} steps", steps.len());
    for (i, s) in steps.iter().enumerate() {
        let b = s.batch.unwrap();
        eprintln!("step {i}: wall={:>8.1} gpu={:>8.1} tokens={} ctx={}(sq{}) gen={}(sq{})",
            s.t1 - s.t0, s.gpu_dur, b.total_tokens, b.n_ctx, b.ctx_tokens, b.n_gen, b.gen_tokens);
    }
    assert!(steps.iter().all(|s| s.batch.is_some()));
    assert!(steps.iter().any(|s| s.batch.unwrap().total_tokens > 0));
}

// =========================== ktz decoding ===========================
// Synthetic files hand-built to the ktz.cpp spec (varints, sections, zstd
// framing) and decoded through ktz::decode; expected output is what
// `ktz d` would emit, since that binary is the format's reference reader.

fn ktz_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 { b |= 0x80; }
        buf.push(b);
        if v == 0 { break; }
    }
}

fn ktz_section(payload: &mut Vec<u8>, data: &[u8]) {
    ktz_varint(payload, data.len() as u64);
    payload.extend_from_slice(data);
}

fn ktz_zz(v: i64) -> u64 { ((v as u64) << 1) ^ ((v >> 63) as u64) }
fn ktz_zz1(v: i64) -> u64 { ktz_zz(v) + 1 }

fn ktz_symtab(syms: &[&[u8]]) -> Vec<u8> {
    let mut s = Vec::new();
    ktz_varint(&mut s, syms.len() as u64);
    for sym in syms {
        ktz_varint(&mut s, sym.len() as u64);
        s.extend_from_slice(sym);
    }
    s
}

fn ktz_head_foot(head: &[u8], foot: &[u8]) -> Vec<u8> {
    let mut s = Vec::new();
    ktz_varint(&mut s, head.len() as u64);
    s.extend_from_slice(head);
    ktz_varint(&mut s, foot.len() as u64);
    s.extend_from_slice(foot);
    s
}

/// Frame a payload as a KTZ3/KTZ4 container (one zstd frame).
fn ktz_framed(magic: &[u8; 4], n_events: u64, payload: &[u8]) -> Vec<u8> {
    let comp = zstd::bulk::compress(payload, 3).unwrap();
    let mut f = Vec::new();
    f.extend_from_slice(magic);
    ktz_varint(&mut f, n_events);
    ktz_varint(&mut f, 1); // nFrames
    ktz_varint(&mut f, payload.len() as u64);
    ktz_varint(&mut f, comp.len() as u64);
    f.extend_from_slice(&comp);
    f
}

#[test]
fn ktz_v4_synthetic() {
    let syms: Vec<&[u8]> = vec![
        b"",               // 0: reserved (absent)
        b"\"X\"",          // 1: ph
        b"1234",           // 2: pid (raw number span)
        b"\"T-1\"",        // 3: tid
        b"\"Op\"",         // 4: name
        b"\"cat\"",        // 5: cat
        b"\"a\"",          // 6: column/blob key
        b"\"b\"",          // 7: column key
        b"\"x\"",          // 8: column str value
        b"\"k\"",          // 9: blob key
        b"\"s\"",          // 10: blob key
        b"\"hello\"",      // 11: blob string value
        b"5678",           // 12: second pid
        b"\"nested\"",     // 13: blob key
        b"\"arr\"",        // 14: blob key
    ];

    // Event 1's blob: TK_INT (dmap update), TK_STR, a nested TK_OBJ with a
    // TK_DINT, and a TK_ARR with TK_INTABS/TK_RAWNUM/TK_TRUE/TK_FALSE/TK_NULL.
    let mut blob = Vec::new();
    blob.push(3u8); ktz_varint(&mut blob, 9);  // "k":
    blob.push(4u8); ktz_varint(&mut blob, ktz_zz(100)); // TK_INT 100
    blob.push(3u8); ktz_varint(&mut blob, 10); // "s":
    blob.push(3u8); ktz_varint(&mut blob, 11); // TK_STR "hello"
    blob.push(3u8); ktz_varint(&mut blob, 13); // "nested":
    blob.push(1u8); ktz_varint(&mut blob, 1);  // TK_OBJ {1 member}
    blob.push(3u8); ktz_varint(&mut blob, 6);  //   "a":
    blob.push(9u8); ktz_varint(&mut blob, ktz_zz(5)); // TK_DINT +5 (vs dmap=0)
    blob.push(3u8); ktz_varint(&mut blob, 14); // "arr":
    blob.push(2u8); ktz_varint(&mut blob, 5);  // TK_ARR {5 elems}
    blob.push(10u8); ktz_varint(&mut blob, ktz_zz(5)); // TK_INTABS 5
    blob.push(5u8); ktz_varint(&mut blob, 4); blob.extend_from_slice(b"1.25"); // TK_RAWNUM
    blob.push(6u8); // true
    blob.push(7u8); // false
    blob.push(8u8); // null

    // Per-name stream for name sym 4: 2 columns ("a" int, "b" str), wrap,
    // flags 1/2/1, one blen for the flag-2 event.
    let col_a: Vec<u8> = [ktz_zz(7), ktz_zz(9 - 7)].iter().flat_map(|v| {
        let mut b = Vec::new(); ktz_varint(&mut b, *v); b
    }).collect();
    let col_b: Vec<u8> = [ktz_zz(8), ktz_zz(0)].iter().flat_map(|v| {
        let mut b = Vec::new(); ktz_varint(&mut b, *v); b
    }).collect();
    let flags = [1u64, 2, 1];
    let blens = [blob.len() as u64];
    let mut stream = Vec::new();
    ktz_varint(&mut stream, 2);          // nKeys
    stream.push(1);                      // wrap: columns inside one "args"
    ktz_varint(&mut stream, 6); stream.push(0); // key "a", kind int
    ktz_varint(&mut stream, 7); stream.push(1); // key "b", kind str
    ktz_varint(&mut stream, col_a.len() as u64); stream.extend_from_slice(&col_a);
    ktz_varint(&mut stream, col_b.len() as u64); stream.extend_from_slice(&col_b);
    ktz_varint(&mut stream, flags.len() as u64);
    for f in flags { ktz_varint(&mut stream, f); }
    ktz_varint(&mut stream, blens.len() as u64);
    for b in blens { ktz_varint(&mut stream, b); }
    stream.extend_from_slice(&blob);

    let mut payload = Vec::new();
    for col in [
        vec![1u64, 1, 1],        // ph
        vec![2u64, 12, 2],       // pid
        vec![3u64, 3, 3],        // tid
        vec![4u64, 4, 4],        // name
        vec![5u64, 0, 0],        // cat (0 = absent)
        vec![ktz_zz1(0), ktz_zz1(2_500_000), ktz_zz1(3_000_000)], // ts deltas
        vec![ktz_zz1(1500), 0, ktz_zz1(-1500)],                   // dur
    ] {
        let mut sec = Vec::new();
        for v in col { ktz_varint(&mut sec, v); }
        ktz_section(&mut payload, &sec);
    }
    ktz_varint(&mut payload, 1); // nStreams
    ktz_varint(&mut payload, 4); // nid = name sym 4
    ktz_section(&mut payload, &stream);
    ktz_section(&mut payload, &ktz_symtab(&syms));
    ktz_section(&mut payload, &ktz_head_foot(b"{\"traceEvents\": [\n", b"]\n}\n"));

    let file = ktz_framed(b"KTZ4", 3, &payload);
    let got = ktz::decode(&file, &test_counter()).expect("decode");
    let expected = concat!(
        "{\"traceEvents\": [\n",
        "  {\"ph\":\"X\",\"pid\":1234,\"tid\":\"T-1\",\"name\":\"Op\",\"cat\":\"cat\",\"ts\":0.000,\"dur\":1.500,\"args\":{\"a\":7,\"b\":\"x\"}},\n",
        "  {\"ph\":\"X\",\"pid\":5678,\"tid\":\"T-1\",\"name\":\"Op\",\"ts\":2500.000,\"k\":100,\"s\":\"hello\",\"nested\":{\"a\":5},\"arr\":[5,1.25,true,false,null]},\n",
        "  {\"ph\":\"X\",\"pid\":1234,\"tid\":\"T-1\",\"name\":\"Op\",\"ts\":3000.000,\"dur\":-1.500,\"args\":{\"a\":9,\"b\":\"x\"}}\n",
        "]\n}\n",
    );
    assert_eq!(std::str::from_utf8(&got).unwrap(), expected);
}

#[test]
fn ktz_v4_range_resets() {
    // 8194 events crossing the 8192-event range boundary: column delta
    // bases and the blob dmap must both reset at event index 8192. Events
    // 8191/8192 are blob events sharing key "k" — without the dmap reset
    // the TK_DINT at 8192 would decode as 5007 instead of 7.
    const N: u64 = 8194;
    let syms: Vec<&[u8]> = vec![b"", b"\"X\"", b"1234", b"7", b"\"Op\"", b"\"a\"", b"\"k\""];

    let mut blob_8191 = Vec::new();
    blob_8191.push(3u8); ktz_varint(&mut blob_8191, 6);   // "k":
    blob_8191.push(4u8); ktz_varint(&mut blob_8191, ktz_zz(5000)); // TK_INT 5000
    let mut blob_8192 = Vec::new();
    blob_8192.push(3u8); ktz_varint(&mut blob_8192, 6);   // "k":
    blob_8192.push(9u8); ktz_varint(&mut blob_8192, ktz_zz(7)); // TK_DINT +7 vs reset dmap

    // Column "a" (int): flag-1 events are 0..=8190 (values = index) and
    // 8193 (value 8193, first of range 1 so delta vs a reset base of 0).
    let mut col_a = Vec::new();
    for i in 0..=8190u64 {
        ktz_varint(&mut col_a, ktz_zz(if i == 0 { 0 } else { 1 }));
    }
    ktz_varint(&mut col_a, ktz_zz(8193));

    let mut stream = Vec::new();
    ktz_varint(&mut stream, 1);          // nKeys
    stream.push(1);                      // wrap
    ktz_varint(&mut stream, 5); stream.push(0); // key "a", kind int
    ktz_varint(&mut stream, col_a.len() as u64); stream.extend_from_slice(&col_a);
    let mut flags = Vec::new();
    for i in 0..N {
        ktz_varint(&mut flags, if i == 8191 || i == 8192 { 2 } else { 1 });
    }
    ktz_varint(&mut stream, flags.len() as u64); stream.extend_from_slice(&flags);
    ktz_varint(&mut stream, 2); // two blens (one per flag-2 event)
    ktz_varint(&mut stream, blob_8191.len() as u64);
    ktz_varint(&mut stream, blob_8192.len() as u64);
    stream.extend_from_slice(&blob_8191);
    stream.extend_from_slice(&blob_8192);

    let mut payload = Vec::new();
    let mut rep = |vals: &dyn Fn(u64) -> u64| {
        let mut sec = Vec::new();
        for i in 0..N { ktz_varint(&mut sec, vals(i)); }
        ktz_section(&mut payload, &sec);
    };
    rep(&|_| 1);                 // ph
    rep(&|_| 2);                 // pid
    rep(&|_| 3);                 // tid
    rep(&|_| 4);                 // name
    rep(&|_| 0);                 // cat absent
    rep(&|i| if i == 0 { ktz_zz1(0) } else { ktz_zz1(1000) }); // ts: i us
    rep(&|_| 0);                 // dur absent
    ktz_varint(&mut payload, 1); // nStreams
    ktz_varint(&mut payload, 4); // nid
    ktz_section(&mut payload, &stream);
    ktz_section(&mut payload, &ktz_symtab(&syms));
    ktz_section(&mut payload, &ktz_head_foot(b"[", b"]\n"));

    let file = ktz_framed(b"KTZ4", N, &payload);
    let got = ktz::decode(&file, &test_counter()).expect("decode");
    let mut expected = String::from("[");
    for i in 0..N {
        expected.push_str("  {\"ph\":\"X\",\"pid\":1234,\"tid\":7,\"name\":\"Op\",\"ts\":");
        expected.push_str(&format!("{i}.000"));
        match i {
            8191 => expected.push_str(",\"k\":5000"),
            8192 => expected.push_str(",\"k\":7"), // dmap reset: 7, not 5007
            _ => expected.push_str(&format!(",\"args\":{{\"a\":{i}}}")),
        }
        expected.push_str(if i + 1 < N { "},\n" } else { "}\n" });
    }
    expected.push_str("]\n");
    assert_eq!(std::str::from_utf8(&got).unwrap(), expected);
}

#[test]
fn ktz_v3_synthetic() {
    // KTZ3: framed like KTZ4 but per-name streams are blens+blobs sections.
    let syms: Vec<&[u8]> = vec![b"", b"\"X\"", b"1", b"2", b"\"Op\"", b"\"k\""];
    let mut blob = Vec::new();
    blob.push(3u8); ktz_varint(&mut blob, 5);            // "k":
    blob.push(4u8); ktz_varint(&mut blob, ktz_zz(42));   // TK_INT 42

    let mut blens = Vec::new();
    ktz_varint(&mut blens, blob.len() as u64); // event 0: blob
    ktz_varint(&mut blens, 0);                 // event 1: no args

    let mut payload = Vec::new();
    for col in [
        vec![1u64, 1],                        // ph
        vec![2u64, 2],                        // pid
        vec![3u64, 3],                        // tid
        vec![4u64, 4],                        // name
        vec![0u64, 0],                        // cat
        vec![ktz_zz1(500), ktz_zz1(250)],     // ts deltas (same pid,tid)
        vec![0u64, 0],                        // dur
    ] {
        let mut sec = Vec::new();
        for v in col { ktz_varint(&mut sec, v); }
        ktz_section(&mut payload, &sec);
    }
    ktz_varint(&mut payload, 1); // nStreams
    ktz_varint(&mut payload, 4); // nid
    ktz_section(&mut payload, &blens);
    ktz_section(&mut payload, &blob);
    ktz_section(&mut payload, &ktz_symtab(&syms));
    ktz_section(&mut payload, &ktz_head_foot(b"{\n\"traceEvents\": [\n", b"]\n}\n"));

    let file = ktz_framed(b"KTZ3", 2, &payload);
    let got = ktz::decode(&file, &test_counter()).expect("decode");
    let expected = concat!(
        "{\n\"traceEvents\": [\n",
        "  {\"ph\":\"X\",\"pid\":1,\"tid\":2,\"name\":\"Op\",\"ts\":0.500,\"k\":42},\n",
        "  {\"ph\":\"X\",\"pid\":1,\"tid\":2,\"name\":\"Op\",\"ts\":0.750}\n",
        "]\n}\n",
    );
    assert_eq!(std::str::from_utf8(&got).unwrap(), expected);
}

#[test]
fn ktz_v2_synthetic() {
    // KTZ2: the whole file is one zstd frame holding KTZ2 + nEvents + the
    // same v2-style sections as KTZ3.
    let syms: Vec<&[u8]> = vec![b"", b"\"X\"", b"1", b"2", b"\"Op\"", b"\"k\""];
    let mut blob = Vec::new();
    blob.push(3u8); ktz_varint(&mut blob, 5);          // "k":
    blob.push(4u8); ktz_varint(&mut blob, ktz_zz(42)); // TK_INT 42
    let mut blens = Vec::new();
    ktz_varint(&mut blens, blob.len() as u64); // event 0: blob
    ktz_varint(&mut blens, 0);                 // event 1: no args

    let mut payload = Vec::new();
    payload.extend_from_slice(b"KTZ2");
    ktz_varint(&mut payload, 2); // nEvents
    for col in [
        vec![1u64, 1],                        // ph
        vec![2u64, 2],                        // pid
        vec![3u64, 3],                        // tid
        vec![4u64, 4],                        // name
        vec![0u64, 0],                        // cat
        vec![ktz_zz1(500), ktz_zz1(250)],     // ts deltas (same pid,tid)
        vec![0u64, 0],                        // dur
    ] {
        let mut sec = Vec::new();
        for v in col { ktz_varint(&mut sec, v); }
        ktz_section(&mut payload, &sec);
    }
    ktz_varint(&mut payload, 1); // nStreams
    ktz_varint(&mut payload, 4); // nid
    ktz_section(&mut payload, &blens);
    ktz_section(&mut payload, &blob);
    ktz_section(&mut payload, &ktz_symtab(&syms));
    ktz_section(&mut payload, &ktz_head_foot(b"{\n\"traceEvents\": [\n", b"]\n}\n"));

    let file = zstd::bulk::compress(&payload, 3).unwrap();
    let got = ktz::decode(&file, &test_counter()).expect("decode");
    let expected = concat!(
        "{\n\"traceEvents\": [\n",
        "  {\"ph\":\"X\",\"pid\":1,\"tid\":2,\"name\":\"Op\",\"ts\":0.500,\"k\":42},\n",
        "  {\"ph\":\"X\",\"pid\":1,\"tid\":2,\"name\":\"Op\",\"ts\":0.750}\n",
        "]\n}\n",
    );
    assert_eq!(std::str::from_utf8(&got).unwrap(), expected);
}

#[test]
fn ktz_rejects_garbage() {
    let c = test_counter();
    assert!(ktz::decode(b"not a ktz file at all", &c).is_err());
    assert!(ktz::decode(b"KTZ4", &c).is_err()); // truncated header
    let mut f = Vec::new();
    f.extend_from_slice(b"KTZ4");
    ktz_varint(&mut f, 1);
    ktz_varint(&mut f, 1);
    ktz_varint(&mut f, 100);
    ktz_varint(&mut f, 5);
    f.extend_from_slice(&[0u8; 3]); // frame shorter than declared
    assert!(ktz::decode(&f, &c).is_err());
}

#[test]
#[ignore]
fn ktz_real_file_matches_ktz_d() {
    let ktz_bin = "/Users/ecrncevi/CLionProjects/compression/ktz";
    let sample = match std::env::var("TV_KTZ_SAMPLE") {
        Ok(p) => p,
        Err(_) => "/Users/ecrncevi/dsv4traces/dsv41-pd-1p1d-dep8-spec3-prof-001-20261022-2124-pd1p1d-dep8-spec3-vllm-profile-prefill/wide-ep-dss-vllm-dsv41-flash-0-5b1b47f7-prefill-0/dsv41-pd-1p1d-dep8-spec3-prof-001-20261022-2124-pd1p1d-dep8-spec3-c16-prefill_dp7_pp0_tp0_dcp0_ep7_rank0.1792698681095973251.pt.trace.json.ktz".to_string(),
    };
    if !std::path::Path::new(ktz_bin).exists() || !std::path::Path::new(&sample).exists() {
        eprintln!("ktz binary or sample trace missing, skipping");
        return;
    }
    let out_path = std::env::temp_dir().join(format!("tv_ktz_test_reference_{}.json", std::process::id()));
    let status = std::process::Command::new(ktz_bin)
        .args(["d", &sample, out_path.to_str().unwrap()])
        .status()
        .expect("spawn ktz");
    assert!(status.success(), "ktz d failed");
    let expected = std::fs::read(&out_path).expect("read ktz d output");
    let data = std::fs::read(sample).expect("read ktz file");
    let got = ktz::decode(&data, &test_counter()).expect("decode");
    if got != expected {
        let n = got.len().min(expected.len());
        let pos = (0..n).find(|&i| got[i] != expected[i]).unwrap_or(n);
        eprintln!(
            "diverge at byte {pos}/{} (got {} bytes): got {:?} vs ktz d {:?}",
            expected.len(),
            got.len(),
            String::from_utf8_lossy(&got[pos.saturating_sub(60)..(pos + 60).min(got.len())]),
            String::from_utf8_lossy(&expected[pos.saturating_sub(60)..(pos + 60).min(expected.len())]),
        );
    }
    assert_eq!(got, expected, "decoded output must match `ktz d` byte-for-byte");
}

#[test]
#[ignore]
fn ktz_real_file_loads_end_to_end() {
    let sample = "/Users/ecrncevi/dsv4traces/dsv41-pd-1p1d-dep8-spec3-prof-001-20261022-2124-pd1p1d-dep8-spec3-vllm-profile-prefill/wide-ep-dss-vllm-dsv41-flash-0-5b1b47f7-prefill-0/dsv41-pd-1p1d-dep8-spec3-prof-001-20261022-2124-pd1p1d-dep8-spec3-c16-prefill_dp7_pp0_tp0_dcp0_ep7_rank0.1792698681095973251.pt.trace.json.ktz";
    if !std::path::Path::new(sample).exists() {
        eprintln!("sample trace missing, skipping");
        return;
    }
    let trace = load_trace(sample, &test_counter(), 8, None).expect("load ktz trace");
    eprintln!("loaded {} tracks, {} events", trace.tracks.len(), trace.total_events);
    assert!(trace.total_events > 1_000_000);
    assert!(!trace.tracks.is_empty());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resolve_tool_finds_first_match_on_path() {
    // A tool present in a listed PATH dir resolves to that dir's copy,
    // in listing order (first hit wins).
    let tmp = std::env::temp_dir().join("tv-resolve-tool-test");
    std::fs::create_dir_all(&tmp).unwrap();
    let tool = tmp.join("tv-fake-tool");
    std::fs::write(&tool, b"#!/bin/sh\n").unwrap();
    let path = format!("{}:{}", tmp.display(), std::path::Path::new("/nonexistent-dir").display());
    assert_eq!(resolve_tool_in(Some(&path), "tv-fake-tool"), tool);
    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resolve_tool_falls_back_to_bare_name_when_missing() {
    // Not on PATH and not in the Homebrew prefixes (name is unique
    // enough): the bare name comes back so a spawn error names the tool.
    let missing = resolve_tool_in(Some("/nonexistent-dir"), "tv-no-such-tool-anywhere");
    assert_eq!(missing, std::path::Path::new("tv-no-such-tool-anywhere"));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn resolve_tool_handles_no_path() {
    // Dock/Finder launches can have no useful PATH at all; must not panic
    // and must still fall through to the prefix candidates.
    let r = resolve_tool_in(None, "tv-no-such-tool-anywhere");
    assert_eq!(r, std::path::Path::new("tv-no-such-tool-anywhere"));
}

#[test]
fn tab_strip_hugs_content_and_lines_up_with_toolbar() {
    // Reproduces the ##tracetabs strip exactly as main.rs builds it now:
    // an ALWAYS_AUTO_RESIZE window pivoted at its right edge. The strip
    // must (a) end up exactly as wide as its content — no estimate to
    // drift from imgui's real per-tab overhead (the old label+32 estimate
    // was ~9px short per strip plus the window padding, which squeezed
    // tabs and left the strip misaligned from the right edge), (b) keep
    // its right edge glued to the display's right edge, and (c) be
    // exactly toolbar_h tall so it lines up with the toolbar row (the old
    // fixed-height window was 2px short of the tab bar's real height).
    let trace = make_trace(
        vec!["", "kernel_a", "kernel_b"],
        vec![("GPU 0", true, vec![ev(0.0, 10.0, 1, 0)])],
    );
    let state = make_state(trace);
    let labels: Vec<String> = state.panes.iter()
        .map(|p| tab_label(p).to_string())
        .collect();

    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();

    let mut toolbar_h = 0.0f32;
    let mut measured_w = 0.0f32;
    let mut measured_h = 0.0f32;
    let mut measured_right = 0.0f32;
    // Auto-resize converges after the first frame; read the converged
    // geometry on the last pass.
    for _frame in 0..3 {
        let ui = imgui.new_frame();
        if _frame == 0 {
            toolbar_h = ui.text_line_height() + 12.0;
        }
        {
            let _pad = ui.push_style_var(imgui::StyleVar::WindowPadding([8.0, 3.0]));
            let _min = ui.push_style_var(imgui::StyleVar::WindowMinSize([1.0, 1.0]));
            ui.window("##tracetabs_test")
                .position([1600.0, 0.0], imgui::Condition::Always)
                .position_pivot([1.0, 0.0])
                .size_constraints([1.0, 1.0], [1600.0 - 420.0, toolbar_h])
                .flags(
                    imgui::WindowFlags::NO_DECORATION
                        | imgui::WindowFlags::NO_MOVE
                        | imgui::WindowFlags::NO_SAVED_SETTINGS
                        | imgui::WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS
                        | imgui::WindowFlags::ALWAYS_AUTO_RESIZE,
                )
                .build(|| {
                    if let Some(_bar) = ui.tab_bar_with_flags(
                        "##tracetabbar_test",
                        imgui::TabBarFlags::FITTING_POLICY_SCROLL,
                    ) {
                        for (i, l) in labels.iter().enumerate() {
                            let id = format!("{}##t{}", l, i);
                            let mut open = true;
                            if let Some(_t) = imgui::TabItem::new(&id).opened(&mut open).begin(&ui) {}
                        }
                    }
                    let win = ui.window_size();
                    let pos = ui.window_pos();
                    measured_w = win[0];
                    measured_h = win[1];
                    measured_right = pos[0] + win[0];
                });
        }
        drop(ui);
        imgui.render();
    }
    eprintln!("strip: {}x{:.1} right_edge={:.1} toolbar_h={:.1}",
              measured_w, measured_h, measured_right, toolbar_h);

    assert!((measured_h - toolbar_h).abs() <= 0.5,
        "tab strip height {measured_h:.1} must line up with the toolbar row ({toolbar_h:.1})");
    assert!((measured_right - 1600.0).abs() <= 0.5,
        "tab strip right edge {measured_right:.1} must be glued to the display's right edge");
    // Sanity: a real, content-sized width (close button + paddings counted),
    // not the old collapsed-to-padding or estimate-based size.
    assert!(measured_w > 60.0,
        "strip width {measured_w:.1} must reflect real content, not an estimate");
}

// TEMP investigation dump: loads a real multi-rank trace dir exactly like the
// app does (detect_rank_groups -> load_multi_progressive, no clock sync) and
// prints the internals of compute_steps: which track becomes the reference,
// per-rank span timelines vs the reference intervals, and where each step's
// min/max bounds stretch past the reference interval.
//   TV_STEPS_TRACE=<dir> cargo test debug_steps_dump -- --ignored --nocapture
#[test]
#[ignore]
fn debug_steps_dump() {
    let path = std::env::var("TV_STEPS_TRACE").expect("set TV_STEPS_TRACE");
    let (groups, standalone) = crate::loader::detect_rank_groups(&[path.clone()]);
    let mut rank_paths: Vec<(usize, String)> = Vec::new();
    for (g, _) in groups { rank_paths.extend(g); }
    for (p, _) in standalone { rank_paths.push((rank_paths.len(), p)); }
    assert!(!rank_paths.is_empty(), "no trace files under {path}");
    eprintln!("rank_paths: {:#?}", rank_paths.iter().map(|(r, p)| (r, p.rsplit('/').next().unwrap_or(p))).collect::<Vec<_>>());

    let (tx, rx) = std::sync::mpsc::channel();
    let counter = test_counter();
    crate::loader::load_multi_progressive(rank_paths, &counter, 2, &tx, None, true);
    let trace = rx.recv().expect("load result").expect("trace loaded");
    eprintln!("merged: {} tracks, {} events, span 0..{:.0} us", trace.tracks.len(), trace.total_events, trace.max_ts);

    let exec_names = find_exec_context_names(&trace.names);
    eprintln!("exec-context names: {:?}",
        exec_names.iter().map(|&i| trace.names[i].as_str()).collect::<Vec<_>>());
    let name_set: std::collections::HashSet<u32> = exec_names.iter().map(|&i| i as u32).collect();

    // Mirror collect_exec_spans for both categories.
    let collect = |cat: &str| -> Vec<(f64, f64, usize, u32)> {
        let cat_ids: Vec<u32> = trace.cats.iter().enumerate()
            .filter(|(_, c)| c.as_str() == cat).map(|(i, _)| i as u32).collect();
        let mut out = Vec::new();
        for (ti, track) in trace.tracks.iter().enumerate() {
            for ev in &track.events {
                if name_set.contains(&ev.name) && cat_ids.contains(&ev.cat) {
                    out.push((ev.ts, ev.ts + ev.dur, ti, ev.name));
                }
            }
        }
        out
    };
    let gpu_side = collect("gpu_user_annotation");
    let cpu_side = collect("user_annotation");
    eprintln!("gpu_user_annotation spans: {}, user_annotation spans: {}",
        gpu_side.len(), cpu_side.len());
    let spans = if gpu_side.is_empty() { cpu_side } else { gpu_side };

    // Per-track (count, total duration) -> reference track, exactly like
    // compute_steps: most spans, then longest total duration, then lowest
    // track index. The tiebreak chain is a total order, so the winner is the
    // same on every call (the old count-only max_by_key was a HashMap lottery
    // among ~200 tracks tied at one span per step).
    let mut per_track: Vec<(usize, usize, f64)> = Vec::new(); // (ti, count, dur)
    for &(s0, s1, ti, _) in &spans {
        match per_track.iter_mut().find(|(t, _, _)| *t == ti) {
            Some((_, c, d)) => { *c += 1; *d += s1 - s0; }
            None => per_track.push((ti, 1, s1 - s0)),
        }
    }
    let (ref_track, _, _) = *per_track.iter().max_by(|a, b| {
        a.1.cmp(&b.1)
            .then(a.2.total_cmp(&b.2))
            .then(b.0.cmp(&a.0))
    }).unwrap();
    eprintln!("reference track: {} ({} of {} tracks with spans)",
        trace.tracks[ref_track].label, per_track.len(), trace.tracks.len());

    // Reference intervals + per-rank span lists (ts-sorted).
    let mut ref_spans: Vec<(f64, f64)> = spans.iter()
        .filter(|&&(_, _, ti, _)| ti == ref_track)
        .map(|&(t0, t1, _, _)| (t0, t1)).collect();
    ref_spans.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eprintln!("reference track {} has {} intervals", trace.tracks[ref_track].label, ref_spans.len());
    let mut by_rank: Vec<(usize, Vec<(f64, f64, usize)>)> = Vec::new(); // (rank, (s0,s1,ti))
    for &(s0, s1, ti, _) in &spans {
        let rank = parse_rank(&trace.tracks[ti].label).unwrap_or(ti);
        match by_rank.iter_mut().find(|(r, _)| *r == rank) {
            Some((_, v)) => v.push((s0, s1, ti)),
            None => by_rank.push((rank, vec![(s0, s1, ti)])),
        }
    }
    for (_rank, v) in &mut by_rank { v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); }
    by_rank.sort_by_key(|(r, _)| *r);

    // Rank phase/skew vs the reference: for each rank, its k-th span start
    // minus the reference's k-th span start (only ranks with equal counts
    // line up index-for-index, but the drift pattern is what matters).
    eprintln!("\nper-rank k-th span start minus reference k-th span start (us), k=0,10,20,...:");
    for (rank, v) in &by_rank {
        let mut line = format!("  rank {rank:2} ({} spans):", v.len());
        for k in (0..v.len()).step_by(10) {
            if k < ref_spans.len() {
                line += &format!(" [{k}]{:+.0}", v[k].0 - ref_spans[k].0);
            }
        }
        eprintln!("{line}");
    }

    // The steps compute_steps actually produces.
    let steps = compute_steps(&trace, &exec_names);
    eprintln!("\n{} steps from compute_steps:", steps.len());
    for (i, s) in steps.iter().enumerate().take(40) {
        let (r0, r1) = ref_spans.get(i).copied().unwrap_or((f64::NAN, f64::NAN));
        let (bt0, bt1) = (s.t0, s.t1);
        eprintln!("  step {i:3}: ref [{r0:10.1},{r1:10.1}] -> bounds [{bt0:10.1},{bt1:10.1}] \
wall {:8.1} gpu_busy {:8.1} idle {:7.1} ranks {} kernels {}",
            s.t1 - s.t0, s.gpu_dur, s.idle_before, s.n_ranks, s.kernel_count);
    }

    // Determinism check: call compute_steps repeatedly on the SAME trace and
    // confirm the bounds don't move. Guards the total-order reference pick —
    // the old count-only max_by_key let HashMap iteration order decide among
    // ~200 tracks tied at 16 spans each, giving different bounds per call.
    eprintln!("\ncompute_steps stability across 8 calls (step-8 t0 / wall):");
    let first = compute_steps(&trace, &exec_names);
    let mut all_same = true;
    for k in 0..8 {
        let s = compute_steps(&trace, &exec_names);
        let same = s.len() == first.len() && s.iter().zip(&first)
            .all(|(a, b)| a.t0 == b.t0 && a.t1 == b.t1 && a.gpu_dur == b.gpu_dur);
        if !same { all_same = false; }
        let st = s.get(8);
        if let Some(st) = st {
            eprintln!("  call {k}: t0={:10.1} t1={:10.1} wall {:8.1} gpu_busy {:8.1}{}",
                st.t0, st.t1, st.t1 - st.t0, st.gpu_dur, if same { "" } else { "  <-- DIFFERS" });
        }
    }
    assert!(all_same, "compute_steps is nondeterministic across calls");

    // Assignment detail: for every step, which assigned spans reach outside
    // the reference interval (the min/max stretch contributors)?
    eprintln!("\nstretch contributors (spans outside the ref interval), per step:");
    for (i, &(r0, r1)) in ref_spans.iter().enumerate() {
        let mut viol: Vec<(f64, f64, &str)> = Vec::new();
        for &(s0, s1, ti, _) in &spans {
            let ov = (s1.min(r1) - s0.max(r0)).max(0.0);
            if ov <= 0.0 { continue; }
            // best-overlap interval for this span (same rule as compute_steps)
            let mut best = (0usize, 0f64);
            for (j, &(q0, q1)) in ref_spans.iter().enumerate() {
                if q0 >= s1 { break; }
                let o = (s1.min(q1) - s0.max(q0)).max(0.0);
                if o > best.1 { best = (j, o); }
            }
            if best.0 != i { continue; }
            if s0 < r0 - 1.0 || s1 > r1 + 1.0 {
                viol.push((s0, s1, &trace.tracks[ti].label));
            }
        }
        viol.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let n_out = viol.len();
        if n_out > 0 {
            eprintln!("  step {i:3} ref [{r0:9.1},{r1:9.1}]: {n_out} spans outside, earliest 6:");
            for (s0, s1, label) in viol.iter().take(6) {
                eprintln!("      [{s0:9.1},{s1:9.1}] dur {:8.1} left-stretch {:7.1}  {}",
                    s1 - s0, r0 - s0, label);
            }
        }
    }

    // gpu_busy sanity: with per-rank union + median, GPU busy time can never
    // exceed the step's wall time. The old sum-of-parallel-streams metric
    // reported ~3.5x the wall time on this trace (38ms busy for an 11.5ms
    // step), which is physically impossible.
    eprintln!("\nsteps where gpu_busy > wall (should be none):");
    let mut n_bad = 0;
    for (i, s) in steps.iter().enumerate() {
        if s.gpu_dur > s.t1 - s.t0 + 1.0 {
            eprintln!("  step {i:3}: wall {:8.1} gpu_busy {:8.1}", s.t1 - s.t0, s.gpu_dur);
            n_bad += 1;
        }
    }
    if n_bad == 0 { eprintln!("  (none)"); }

    // Exact replication of compute_steps' assignment (including the
    // nearest-interval fallback), tracking WHICH span sets each step's
    // t0 (argmin) and t1 (argmax).
    eprintln!("\nargmin/argmax span per step (exact compute_steps assignment):");
    {
        let mut bounds: Vec<(f64, f64, usize, usize)> = (0..ref_spans.len())
            .map(|_| (f64::MAX, f64::MIN, usize::MAX, usize::MAX)).collect(); // (t0, t1, argmin span idx, argmax span idx)
        for (si, &(s0, s1, _ti, _)) in spans.iter().enumerate() {
            let mut best = (0usize, 0f64);
            for (j, &(q0, q1)) in ref_spans.iter().enumerate() {
                if q0 >= s1 { break; }
                let o = (s1.min(q1) - s0.max(q0)).max(0.0);
                if o > best.1 { best = (j, o); }
            }
            let i = if best.1 > 0.0 { best.0 } else {
                ref_spans.iter().enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = (a.0 - s0).abs() + (a.1 - s1).abs();
                        let db = (b.0 - s0).abs() + (b.1 - s1).abs();
                        da.partial_cmp(&db).unwrap()
                    }).map(|(j, _)| j).unwrap_or(0)
            };
            let b = &mut bounds[i];
            if s0 < b.0 { b.0 = s0; b.2 = si; }
            if s1 > b.1 { b.1 = s1; b.3 = si; }
        }
        for (i, b) in bounds.iter().enumerate() {
            let (t0, t1, ai, ax) = *b;
            if ai == usize::MAX { continue; }
            let (s0, s1, ti0, _) = spans[ai];
            let (e0, e1, tix, _) = spans[ax];
            let (r0, r1) = ref_spans[i];
            eprintln!("  step {i:3} ref [{r0:9.1},{r1:9.1}] -> [{t0:9.1},{t1:9.1}]");
            eprintln!("      t0 from [{s0:9.1},{s1:9.1}] dur {:8.1}  {}",
                s1 - s0, trace.tracks[ti0].label);
            eprintln!("      t1 from [{e0:9.1},{e1:9.1}] dur {:8.1}  {}",
                e1 - e0, trace.tracks[tix].label);
        }
    }

    // Rank phase: per rank, the span assigned to each step whose start is the
    // EARLIEST for that (rank, step) — i.e. that rank's own step start —
    // versus the reference interval start.
    eprintln!("\nper-rank own-step start minus ref interval start (us), steps 0..15:");
    for (rank, v) in &by_rank {
        // group this rank's spans by assigned step
        let mut per_step: Vec<Vec<(f64, f64)>> = vec![Vec::new(); ref_spans.len()];
        for &(s0, s1, _ti) in v {
            let mut best = (0usize, 0f64);
            for (j, &(q0, q1)) in ref_spans.iter().enumerate() {
                if q0 >= s1 { break; }
                let o = (s1.min(q1) - s0.max(q0)).max(0.0);
                if o > best.1 { best = (j, o); }
            }
            if best.1 > 0.0 { per_step[best.0].push((s0, s1)); }
        }
        let mut line = format!("  rank {rank:2}:");
        for (i, ps) in per_step.iter().enumerate().take(16) {
            match ps.iter().map(|&(s0, _)| s0).fold(f64::INFINITY, f64::min) {
                f64::INFINITY => line += "     -",
                s => line += &format!(" {:+7.0}", s - ref_spans[i].0),
            }
        }
        eprintln!("{line}");
    }
}

// TEMP investigation dump: loads a real multi-rank trace dir exactly like the
// app does and, for a set of kernel names (matched by substring), compares
// the Selection tab's "Overlap aware" semantics on real data:
//   - self-union: union of the kernel's OWN launch intervals — shrinks only
//     when launches of the SAME kernel overlap (parallel streams /
//     concurrent ranks);
//   - shared: every instant of wall time split equally among the kernels
//     running it — what the toggle computes since the shared-time rewrite.
//     Overlapped kernels are discounted (not zeroed: exclusive time was
//     tried and pays only straggler tails in dense windows).
// Both are computed globally (all ranks) and per-rank (what a merged-row band
// selection captures), plus the pairwise overlap between the watched kernels.
// The tail blocks cross-check the production `ensure_sel_overlap_totals`
// against ground truths: shares over all names must sum to the busy union,
// and a name never covered by any other must keep its full sum.
//   TV_SEL_TRACE=<dir> cargo test debug_sel_overlap_trace -- --ignored --nocapture
#[test]
#[ignore]
fn debug_sel_overlap_trace() {
    let path = std::env::var("TV_SEL_TRACE").expect("set TV_SEL_TRACE");
    let (groups, standalone) = crate::loader::detect_rank_groups(&[path.clone()]);
    let mut rank_paths: Vec<(usize, String)> = Vec::new();
    for (g, _) in groups { rank_paths.extend(g); }
    for (p, _) in standalone { rank_paths.push((rank_paths.len(), p)); }
    assert!(!rank_paths.is_empty(), "no trace files under {path}");

    let (tx, rx) = std::sync::mpsc::channel();
    let counter = test_counter();
    crate::loader::load_multi_progressive(rank_paths, &counter, 2, &tx, None, true);
    let trace = rx.recv().expect("load result").expect("trace loaded");
    eprintln!("merged: {} tracks, {} events, span 0..{:.0} us", trace.tracks.len(), trace.total_events, trace.max_ts);

    let needles = ["elastic::combine_impl", "humming<MmaOpClass", "combine_reduce_epilogue_impl"];
    let watch: Vec<u32> = trace.names.iter().enumerate()
        .filter(|(_, n)| needles.iter().any(|nd| n.contains(nd)))
        .map(|(i, _)| i as u32).collect();
    assert!(!watch.is_empty(), "none of {needles:?} found in trace names");
    for &w in &watch {
        eprintln!("  watch [{}] {}...", w, &trace.names[w as usize][..trace.names[w as usize].len().min(70)]);
    }

    // GPU events grouped by rank: rank -> Vec<(ts, end, name)>.
    let mut per_rank: std::collections::BTreeMap<usize, Vec<(f64, f64, u32)>> = Default::default();
    let mut global: Vec<(f64, f64, u32)> = Vec::new();
    for track in &trace.tracks {
        if !track.gpu { continue; }
        let rank = crate::state::parse_rank(&track.label).unwrap_or(9999);
        for ev in &track.events {
            let e = (ev.ts, ev.ts + ev.dur, ev.name);
            per_rank.entry(rank).or_default().push(e);
            global.push(e);
        }
    }
    eprintln!("ranks with GPU events: {}", per_rank.len());

    // One sweep over a scope's events: per watched name -> shared time (each
    // instant split equally among the distinct active names); also the
    // scope's busy union. `active` maps name -> live-launch count (matching
    // the production sweep): a set of names evicts a self-overlapping name
    // at its first end while it is still live, wrongly crediting whatever
    // else runs.
    let shared_sweep = |events: &[(f64, f64, u32)]| -> (Vec<f64>, f64) {
        let mut points: Vec<(f64, i32, u32)> = Vec::with_capacity(events.len() * 2);
        for &(ts, end, name) in events {
            if end > ts {
                points.push((ts, 1, name));
                points.push((end, -1, name));
            }
        }
        // Ends before starts at equal timestamps: touching intervals don't
        // count as overlapping each other.
        points.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut active: std::collections::HashMap<u32, u32> = Default::default();
        let mut share = vec![0.0f64; watch.len()];
        let mut union = 0.0f64;
        let mut live = 0usize;
        let mut prev = f64::NAN;
        for &(t, d, name) in &points {
            if !prev.is_nan() && live > 0 {
                let dt = t - prev;
                union += dt;
                let k = active.len() as f64;
                for &n in active.keys() {
                    if let Some(wi) = watch.iter().position(|&w| w == n) {
                        share[wi] += dt / k;
                    }
                }
            }
            prev = t;
            live = (live as i32 + d) as usize;
            if d > 0 {
                *active.entry(name).or_insert(0) += 1;
            } else if let Some(c) = active.get_mut(&name) {
                *c -= 1;
                if *c == 0 { active.remove(&name); }
            }
        }
        (share, union)
    };
    let counts_sums = |events: &[(f64, f64, u32)]| -> Vec<(u32, f64, f64)> {
        // per watched name: (count, sum, self-union)
        watch.iter().map(|&w| {
            let mut spans: Vec<(f64, f64)> = events.iter()
                .filter(|&&(_, _, n)| n == w).map(|&(ts, end, _)| (ts, end)).collect();
            let count = spans.len() as u32;
            let sum: f64 = spans.iter().map(|&(a, b)| b - a).sum();
            let su = {
                spans.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (mut total, mut cur) = (0.0, f64::NEG_INFINITY);
                for &(a, b) in spans.iter() {
                    if b > cur { total += b - a.max(cur); cur = b; }
                }
                total
            };
            (count, sum, su)
        }).collect()
    };

    // Pairwise intersection of two watched names' unions within a scope.
    let pairwise = |events: &[(f64, f64, u32)], a: u32, b: u32| -> f64 {
        let mut ia: Vec<(f64, f64)> = events.iter().filter(|&&(_, _, n)| n == a).map(|&(ts, e, _)| (ts, e)).collect();
        let mut ib: Vec<(f64, f64)> = events.iter().filter(|&&(_, _, n)| n == b).map(|&(ts, e, _)| (ts, e)).collect();
        ia.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        ib.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let (mut i, mut j, mut total) = (0usize, 0usize, 0.0f64);
        while i < ia.len() && j < ib.len() {
            let lo = ia[i].0.max(ib[j].0);
            let hi = ia[i].1.min(ib[j].1);
            if hi > lo { total += hi - lo; }
            if ia[i].1 < ib[j].1 { i += 1; } else { j += 1; }
        }
        total
    };

    let mut scopes: Vec<(String, &Vec<(f64, f64, u32)>)> = vec![("GLOBAL".to_string(), &global)];
    for (r, evs) in &per_rank {
        scopes.push((format!("rank {r}"), evs));
    }
    for (scope, events) in scopes {
        let (share, union) = shared_sweep(events);
        let cs: Vec<(u32, f64, f64)> = counts_sums(events);
        let sum_all: f64 = events.iter().map(|&(a, b, _)| b - a).sum();
        eprintln!("\n=== scope {scope} (events {}, busy union {union:.1} us, raw sum {sum_all:.1} us)", events.len());
        for (wi, &w) in watch.iter().enumerate() {
            let (count, sum, su) = cs[wi];
            if count == 0 { continue; }
            let short: String = trace.names[w as usize].chars().take(46).collect();
            eprintln!("  {short:46} n={count:5} sum {sum:9.1}  self-union {su:9.1} ({:4.0}% of sum)  shared {:9.1} ({:4.0}% of sum)",
                su / sum * 100.0, share[wi], share[wi] / sum * 100.0);
        }
        // Pairwise overlap between watched kernels (how much of A is covered by B).
        for (x, &a) in watch.iter().enumerate() {
            for (y, &b) in watch.iter().enumerate().skip(x + 1) {
                if cs[x].0 == 0 || cs[y].0 == 0 { continue; }
                let inter = pairwise(events, a, b);
                if inter > 0.0 {
                    let na: String = trace.names[a as usize].chars().take(28).collect();
                    let nb: String = trace.names[b as usize].chars().take(28).collect();
                    eprintln!("  overlap {na:28} x {nb:28}: {inter:9.1} us");
                }
            }
        }
    }

    let r0 = per_rank.keys().next().copied().unwrap();
    // Independent ground truths for shared time, both pure interval
    // algebra with no active-name bookkeeping:
    //   (a) conservation — shares over ALL names must sum to the busy
    //       union of the scope (each instant is split, not duplicated);
    //   (b) a name whose launches are never covered by any other name's
    //       union (checked by union-intersection) must keep its full sum.
    let rank0_events = &per_rank[&r0];
    let union_of = |spans: &mut Vec<(f64, f64)>| -> Vec<(f64, f64)> {
        spans.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut out: Vec<(f64, f64)> = Vec::new();
        for &(a, b) in spans.iter() {
            if let Some(last) = out.last_mut() {
                if a <= last.1 { last.1 = last.1.max(b); continue; }
            }
            out.push((a, b));
        }
        out
    };
    let inter_len = |a: &[(f64, f64)], b: &[(f64, f64)]| -> f64 {
        let (mut i, mut j, mut t) = (0usize, 0usize, 0.0);
        while i < a.len() && j < b.len() {
            let lo = a[i].0.max(b[j].0);
            let hi = a[i].1.min(b[j].1);
            if hi > lo { t += hi - lo; }
            if a[i].1 < b[j].1 { i += 1; } else { j += 1; }
        }
        t
    };
    eprintln!("\n=== shared-time ground truth, rank {r0}");
    {
        let mut all: Vec<(f64, f64)> = rank0_events.iter()
            .map(|&(a, b, _)| (a, b)).collect();
        let busy: f64 = union_of(&mut all).iter().map(|&(a, b)| b - a).sum();
        eprintln!("  busy union (independent interval merge): {busy:.1} us");
        for &w in &watch {
            let mut hs: Vec<(f64, f64)> = rank0_events.iter()
                .filter(|&&(_, _, n)| n == w).map(|&(a, b, _)| (a, b)).collect();
            if hs.is_empty() { continue; }
            let mut os: Vec<(f64, f64)> = rank0_events.iter()
                .filter(|&&(_, _, n)| n != w).map(|&(a, b, _)| (a, b)).collect();
            let hu = union_of(&mut hs);
            let ou = union_of(&mut os);
            let covered = inter_len(&hu, &ou);
            let sum: f64 = rank0_events.iter()
                .filter(|&&(_, _, n)| n == w).map(|&(a, b, _)| b - a).sum();
            let note = if covered == 0.0 { "never covered: share must equal sum" } else { "" };
            let short: String = trace.names[w as usize].chars().take(46).collect();
            eprintln!("  {short:46} sum {sum:9.1}  covered-by-others {covered:9.1}  {note}");
        }
    }

    // Cross-check the production path: build a Pane holding the whole trace
    // with a selection of one rank's complete GPU events (what a band over
    // that merged row captures) and run the real ensure_sel_overlap_totals.
    // Its totals must match the per-rank shared numbers printed above, and
    // the sum over all names must equal the busy union.
    let mut by_name: std::collections::HashMap<u32, crate::types::SelectionEntry> =
        Default::default();
    for (ti, track) in trace.tracks.iter().enumerate() {
        if !track.gpu || crate::state::parse_rank(&track.label) != Some(r0) { continue; }
        for (ei, ev) in track.events.iter().enumerate() {
            let se = by_name.entry(ev.name).or_insert_with(|| crate::types::SelectionEntry {
                name: ev.name, count: 0, total_dur: 0.0,
                durations: Vec::new(), event_refs: Vec::new(),
            });
            se.count += 1;
            se.total_dur += ev.dur;
            se.durations.push(ev.dur);
            se.event_refs.push((ti as u32, ei as u32));
        }
    }
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    pane.selection_stats = by_name.into_values().collect();
    pane.sel_generation += 1;
    pane.ensure_sel_overlap_totals();
    let trace = pane.trace.as_ref().unwrap();
    eprintln!("\n=== production ensure_sel_overlap_totals, selection = rank {r0} all GPU events");
    let total_share: f64 = pane.sel_overlap_totals.iter().sum();
    {
        let mut all: Vec<(f64, f64)> = rank0_events.iter()
            .map(|&(a, b, _)| (a, b)).collect();
        let busy: f64 = union_of(&mut all).iter().map(|&(a, b)| b - a).sum();
        eprintln!("  sum of all shares {total_share:.1} vs busy union {busy:.1} (gap {:+.4})",
            total_share - busy);
    }
    for &w in &watch {
        let Some(si) = pane.selection_stats.iter().position(|s| s.name == w) else { continue };
        let short: String = trace.names[w as usize].chars().take(46).collect();
        eprintln!("  {short:46} sum {:9.1}  shared {:9.1}",
            pane.selection_stats[si].total_dur, pane.sel_overlap_totals[si]);
    }
}

// ---- View > Color > By Duration ----

fn col_rgb(c: ImColor32) -> (u8, u8, u8) {
    let v: u32 = c.into();
    ((v & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, ((v >> 16) & 0xFF) as u8)
}

#[test]
fn heat_color_ramp_runs_cold_to_hot() {
    let (cr, _cg, cb) = col_rgb(heat_color(0.0));
    assert!(cb > cr, "cold end must be blue-dominant, got r={cr} b={cb}");
    let (hr, _hg, hb) = col_rgb(heat_color(1.0));
    assert!(hr > hb, "hot end must be red-dominant, got r={hr} b={hb}");
    // Out-of-range t clamps instead of extrapolating past the stops.
    assert_eq!(col_rgb(heat_color(-0.5)), col_rgb(heat_color(0.0)));
    assert_eq!(col_rgb(heat_color(2.0)), col_rgb(heat_color(1.0)));
    // Middle of the ramp is greenish: both red and blue below green.
    let (mr, mg, mb) = col_rgb(heat_color(0.5));
    assert!(mg > mr && mg > mb, "mid ramp must peak green, got r={mr} g={mg} b={mb}");
}

#[test]
fn heat_dim_scales_channels() {
    let (r, g, b) = col_rgb(heat_dim(ImColor32::from_rgba(100, 200, 50, 255)));
    assert_eq!(r, 55);
    assert_eq!(g, 110);
    assert_eq!(b, 27);
}

#[test]
fn heat_bounds_percentiles_resist_outliers() {
    // 100 ordinary kernels plus one giant memcpy: the p95 bound must stay
    // near the bulk, not stretch to the outlier.
    let mut events = Vec::new();
    for i in 0..100 {
        events.push(ev(i as f64 * 20.0, 10.0, 0, 0));
    }
    events.push(ev(1e6, 1_000_000.0, 0, 0));
    let trace = make_trace(vec!["k"], vec![("stream 7", true, events)]);
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    pane.ensure_heat_bounds();
    let (lo, hi) = pane.heat_bounds.expect("bounds computed");
    assert!(lo <= 1.0, "p5 bound below the bulk's log10(10), got {lo}");
    assert!(hi >= 1.0 && hi < 5.0, "p95 bound near the bulk, got {hi}");
    // Idempotent: a second call keeps the same bounds.
    pane.ensure_heat_bounds();
    assert_eq!(pane.heat_bounds, Some((lo, hi)));
}

#[test]
fn heat_bounds_fall_back_without_gpu_events() {
    let trace = make_trace(
        vec!["op"],
        vec![("python (CPU)", false, vec![ev(0.0, 5.0, 0, 0)])],
    );
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    pane.ensure_heat_bounds();
    assert_eq!(pane.heat_bounds, Some((0.0, 1.0)));
    // And no trace at all.
    let mut pane = Pane::new();
    pane.ensure_heat_bounds();
    assert_eq!(pane.heat_bounds, Some((0.0, 1.0)));
}

/// One full draw_timeline frame with View > Color > By Duration active:
/// exercises the heat LUT lookup in all render paths and the bottom-right
/// legend overlay (chip, gradient segments, write_time labels). Assertions
/// are "runs without panicking" plus the bounds sanity — imgui gives no
/// pixel access to assert colors against.
#[test]
fn draw_timeline_heat_mode_renders() {
    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();

    let trace = make_trace(
        vec!["fast", "slow"],
        vec![
            ("python (CPU)", false, vec![ev(0.0, 30.0, 0, 0), ev(5.0, 2.0, 0, 1)]),
            ("stream 7", true, vec![ev(1.0, 1.0, 0, 0), ev(10.0, 40.0, 1, 0)]),
        ],
    );
    let mut pane = Pane::new();
    pane.trace = Some(trace);
    pane.collapsed = vec![false; 2];
    pane.track_scales = vec![1.0; 2];
    pane.track_order = vec![0, 1];
    pane.hidden_names = vec![false; 2];
    pane.show_cpu = true;
    pane.view.t0 = 0.0;
    pane.view.t1 = 60.0;
    pane.color_by_dur = true;
    pane.ensure_heat_bounds();
    let (lo, hi) = pane.heat_bounds.expect("bounds computed");
    assert!(hi > lo, "degenerate heat bounds {lo}..{hi}");

    let mut buf = DrawBuf::default();
    let mut drag = DragKind::None;
    let ui = imgui.new_frame();
    ui.window("t")
        .position([0.0, 0.0], imgui::Condition::Always)
        .size([1600.0, 900.0], imgui::Condition::Always)
        .build(|| {
            draw_timeline(
                ui, pane.trace.as_ref().unwrap(), &mut pane.view, pane.show_cpu, &mut buf,
                [0.0, 0.0, 1600.0, 900.0], 0, false, false, false,
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, false, false,
                &pane.search_mask, pane.selection, &pane.finished_sel_events,
                &mut pane.collapsed, &pane.hidden_names, pane.selected, pane.multi_select_name,
                &pane.sel_mask, pane.label_w, &mut pane.track_scales, &mut pane.even_spacing,
                &mut pane.geom, &mut pane.track_order, &mut drag, pane.merge_gpu, 0.016,
                &mut pane.pending_focus, &mut pane.merge_cache_key, &mut pane.merged_gpu_groups,
                &pane.steps, pane.step_cursor, pane.color_by_dur, pane.heat_bounds, None,
            );
        });
    imgui.render();
}

// ------------------------------------------------------------------ Gaps ----
// compute_gaps / gap_why coverage. Category ids referenced by evc calls in
// these tests: 0 = kernel, 1 = cpu_op, 2 = runtime, 3 = gpu_user_annotation
// (set per-test via trace.cats, since make_trace ships a single empty cat).

fn gaps_trace(tracks: Vec<(&str, bool, Vec<Event>)>, cats: &[&str]) -> Trace {
    let names = vec!["forward", "backward", "wrapper", "nested", "launch", "sync"];
    let mut trace = make_trace(names, tracks);
    trace.cats = cats.iter().map(|c| c.to_string()).collect();
    trace
}

#[test]
fn gaps_basic() {
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),   // "forward" [0,10]
            evc(30.0, 10.0, 1, 0), // "backward" [30,40]
        ])],
        &["kernel"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    let g = &gaps[0];
    // Unlabeled tracks share one scope → rank None.
    assert_eq!(g.rank, None);
    assert_eq!(g.t0, 10.0);
    assert_eq!(g.t1, 30.0);
    // "forward" closes the busy interval before the gap, "backward" opens
    // the one after.
    assert_eq!(g.prev_name, Some(0));
    assert_eq!(g.next_name, Some(1));
    assert_eq!(g.focus_track, 0);
    // No CPU track in this scope: has_cpu distinguishes "no data" from idle.
    assert!(!g.has_cpu);
    assert_eq!(g.cpu_busy_frac, 0.0);
    assert_eq!(g.runtime_frac, 0.0);
    assert!(g.cpu_top.is_empty());
}

#[test]
fn gaps_min_length_floor() {
    // A 0.5µs gap is below MIN_GAP_US and must be dropped — without the
    // floor, sub-µs scheduling jitter would flood the table.
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(10.5, 10.0, 1, 0),
        ])],
        &["kernel"],
    );
    assert!(compute_gaps(&trace).is_empty());
}

#[test]
fn gaps_zero_duration_events_are_not_busy() {
    // Instant events add no busy time; they must not erase or split a gap
    // either.
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(15.0, 0.0, 1, 0),
            evc(30.0, 10.0, 1, 0),
        ])],
        &["kernel"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    assert_eq!((gaps[0].t0, gaps[0].t1), (10.0, 30.0));
}

#[test]
fn gaps_gpu_user_annotation_is_not_busy() {
    // gpu_user_annotation is a CPU bracket projected onto the stream, so it
    // spans the step's internal idle. Counting it as GPU work would erase
    // exactly the gaps this view exists to find.
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(12.0, 16.0, 1, 3), // annotation [12,28] covering the gap
            evc(30.0, 10.0, 1, 0),
        ])],
        &["kernel", "cpu_op", "runtime", "gpu_user_annotation"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    assert_eq!((gaps[0].t0, gaps[0].t1), (10.0, 30.0));
}

#[test]
fn gaps_gpu_memcpy_counts_as_work() {
    // gpu_memcpy is real GPU work and must erase a gap it covers.
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(10.0, 20.0, 1, 4), // memcpy [10,30]
            evc(30.0, 10.0, 1, 0),
        ])],
        &["kernel", "cpu_op", "runtime", "gpu_user_annotation", "gpu_memcpy"],
    );
    assert!(compute_gaps(&trace).is_empty());
}

#[test]
fn gaps_same_rank_streams_unioned() {
    // Both streams carry the same rank label, so their busy intervals are
    // unioned before gaps are taken: a full cover erases the gap, a partial
    // cover shortens it instead of leaving two gaps on one stream each.
    let trace = gaps_trace(
        vec![
            ("Rank 0 stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("Rank 0 stream 3", true, vec![evc(10.0, 15.0, 0, 0)]),
        ],
        &["kernel"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].rank, Some(0));
    // Busy union is [0,25] ∪ [30,40]: only [25,30] remains.
    assert_eq!((gaps[0].t0, gaps[0].t1), (25.0, 30.0));
}

#[test]
fn gaps_full_cover_by_second_stream_erases_gap() {
    let trace = gaps_trace(
        vec![
            ("Rank 0 stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("Rank 0 stream 3", true, vec![evc(10.0, 20.0, 0, 0)]),
        ],
        &["kernel"],
    );
    assert!(compute_gaps(&trace).is_empty());
}

#[test]
fn gaps_per_rank_scoping() {
    // A gap on rank 0 while rank 1 is busy still counts: gaps are scoped
    // per rank (device level), not across the whole merged trace.
    let trace = gaps_trace(
        vec![
            ("Rank 0 stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("Rank 1 stream 7", true, vec![evc(0.0, 40.0, 0, 0)]),
        ],
        &["kernel"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].rank, Some(0));
    assert_eq!((gaps[0].t0, gaps[0].t1), (10.0, 30.0));
}

#[test]
fn gaps_cpu_fractions_are_union_not_sum() {
    // The CPU annotation must use interval unions: a nested call stack
    // inside a wrapper must not double-count wall time. Wrapper [5,32]
    // covers the whole gap; nested [12,16] sits inside it.
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("cpu", false, vec![
                evc(5.0, 27.0, 2, 1),   // wrapper [5,32]
                evc(12.0, 4.0, 3, 1),   // nested [12,16]
                evc(20.0, 5.0, 4, 2),   // runtime [20,25]
            ]),
        ],
        &["kernel", "cpu_op", "runtime"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    let g = &gaps[0];
    assert!(g.has_cpu);
    // Gap [10,30]: union of {[10,30], [12,16], [20,25]} is 20µs → 100%.
    // A naive sum would give 24/20 = 1.2.
    assert!((g.cpu_busy_frac - 1.0).abs() < 1e-12);
    // Only the runtime event's 5µs counts toward runtime_frac.
    assert!((g.runtime_frac - 0.25).abs() < 1e-12);
    // cpu_top is by covered time, union per name: wrapper 20, nested 4,
    // runtime 5 → wrapper, runtime, nested.
    assert_eq!(g.cpu_top.len(), 3);
    assert_eq!(g.cpu_top[0].0, 2);
    assert!((g.cpu_top[0].2 - 20.0).abs() < 1e-12);
    assert_eq!(g.cpu_top[1].0, 4);
    assert!((g.cpu_top[1].2 - 5.0).abs() < 1e-12);
    assert_eq!(g.cpu_top[2].0, 3);
    assert!((g.cpu_top[2].2 - 4.0).abs() < 1e-12);
    // Counts are per-name launch counts inside the gap.
    assert_eq!(g.cpu_top.iter().map(|&(_, c, _)| c).sum::<u32>(), 3);
}

#[test]
fn gaps_cpu_present_but_idle() {
    // A CPU track exists for the scope but had nothing running inside the
    // gap: has_cpu is true and the fractions are 0 — "CPU idle", which
    // gap_why reports as starved rather than "no CPU data".
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("cpu", false, vec![evc(40.0, 5.0, 2, 1)]),
        ],
        &["kernel", "cpu_op"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    assert!(gaps[0].has_cpu);
    assert_eq!(gaps[0].cpu_busy_frac, 0.0);
    assert_eq!(gaps[0].runtime_frac, 0.0);
}

#[test]
fn gaps_python_frames_are_not_cpu_busy() {
    // Real with_stack traces hold python_function / user_annotation frames
    // open while a thread is PARKED (queue.get, lock.acquire) — they record
    // stack extent, not executed work. Counting them as busy made every
    // starved gap read "host-bound" on such traces (cpu_busy 1.000 from a
    // thread blocked for seconds inside one open frame). Only executed
    // host work (cpu_op / CUDA API cats) feeds the fractions and cpu_top.
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("cpu", false, vec![
                evc(5.0, 27.0, 2, 1),   // python wrapper [5,32] spans the gap
                evc(12.0, 4.0, 3, 2),   // user annotation [12,16], same story
            ]),
        ],
        &["kernel", "python_function", "user_annotation"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    let g = &gaps[0];
    assert!(g.has_cpu);
    assert_eq!(g.cpu_busy_frac, 0.0);
    assert_eq!(g.runtime_frac, 0.0);
    assert!(g.cpu_top.is_empty());
    assert_eq!(gap_why(g).0, "CPU idle (starved)");
}

#[test]
fn gaps_cuda_runtime_cat_aliases() {
    // Real kineto traces carry the CUDA API cats as `cuda_runtime` and
    // `cuda_driver`, not the older `runtime` spelling — launch and sync
    // calls must still count toward runtime_frac (and busy), or every gap
    // reads as zero runtime and the launch-bound case disappears.
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(30.0, 10.0, 1, 0),
            ]),
            ("cpu", false, vec![
                evc(10.0, 8.0, 4, 1),   // cuda_runtime launch [10,18]
                evc(20.0, 6.0, 5, 2),   // cuda_driver sync [20,26]
            ]),
        ],
        &["kernel", "cuda_runtime", "cuda_driver"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), 1);
    let g = &gaps[0];
    // Gap [10,30]: 8µs launch + 6µs sync = 14/20 busy, all of it runtime.
    assert!((g.cpu_busy_frac - 0.7).abs() < 1e-12);
    assert!((g.runtime_frac - 0.7).abs() < 1e-12);
    assert_eq!(g.cpu_top.len(), 2);
    assert_eq!(gap_why(g).0, "runtime-bound");
}

#[test]
fn gaps_filter_window_clips_and_drops() {
    // Kernels [0,10] and [50,60]: one 40µs gap [10,50]. The window clips
    // each gap to its intersection with [w0, w1] and drops ones that fall
    // outside it entirely (or shave to under MIN_GAP_US at an edge).
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(50.0, 10.0, 1, 0),
        ])],
        &["kernel"],
    );
    let f = |w0: f64, w1: f64| GapFilter { window: Some((w0, w1)), max_len: None, ..Default::default() };

    // Window strictly inside the gap: clipped to the intersection.
    let gaps = compute_gaps_in(&trace, f(20.0, 35.0));
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].t0, 20.0);
    assert_eq!(gaps[0].t1, 35.0);

    // Straddling the gap's left edge: clipped from the window start.
    let gaps = compute_gaps_in(&trace, f(5.0, 15.0));
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].t0, 10.0);
    assert_eq!(gaps[0].t1, 15.0);

    // Edge shaving below MIN_GAP_US disappears.
    assert!(compute_gaps_in(&trace, f(49.5, 60.0)).is_empty());

    // Window entirely outside the gap: nothing left.
    assert!(compute_gaps_in(&trace, f(70.0, 90.0)).is_empty());
}

#[test]
fn gaps_filter_max_len_drops_artifacts() {
    // Kernels at [0,10], [20,30], [50,60]: gaps [10,20] (10µs) and
    // [30,50] (20µs, the "artifact"). max_len 15µs keeps only the real
    // bubble. 0/None keeps both.
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(0.0, 10.0, 0, 0),
            evc(20.0, 10.0, 1, 0),
            evc(50.0, 10.0, 0, 0),
        ])],
        &["kernel"],
    );
    let gaps = compute_gaps_in(&trace, GapFilter { window: None, max_len: Some(15.0), ..Default::default() });
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].t0, 10.0);
    assert_eq!(gaps[0].t1, 20.0);
    assert_eq!(compute_gaps_in(&trace, GapFilter::default()).len(), 2);
}

#[test]
fn gaps_filter_applies_before_top_n() {
    // Same 600-gap ladder as gaps_top_n_truncation (lengths 1..600µs).
    // The max-length cut must happen BEFORE the top-N truncation: with
    // max 400µs the 200 artifact-sized gaps are dropped first and all 400
    // real ones listed — not the top 500 of the unfiltered list minus the
    // artifacts (which would leave only 100).
    let mut events = Vec::new();
    let mut t = 0.0;
    for i in 1..=600 {
        events.push(evc(t, 1.0, 0, 0));
        t += 1.0;
        t += i as f64;
    }
    events.push(evc(t, 1.0, 0, 0));
    let trace = gaps_trace(vec![("stream 7", true, events)], &["kernel"]);
    let gaps = compute_gaps_in(&trace, GapFilter { window: None, max_len: Some(400.0), ..Default::default() });
    assert_eq!(gaps.len(), 400);
    assert_eq!(gaps[0].t1 - gaps[0].t0, 400.0);
    assert_eq!(gaps[399].t1 - gaps[399].t0, 1.0);
}

#[test]
fn gaps_filter_scopes_to_selected_ranks() {
    // Two ranks, each with a 40µs gap at [10,50]. Scoping to one rank's
    // scope keeps only its gap; an empty scope list (a selection over empty
    // space) yields nothing; None keeps both.
    let trace = gaps_trace(
        vec![
            ("Rank 0 stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(50.0, 10.0, 1, 0),
            ]),
            ("Rank 1 stream 7", true, vec![
                evc(0.0, 10.0, 0, 0),
                evc(50.0, 10.0, 1, 0),
            ]),
        ],
        &["kernel"],
    );
    let f = |scopes: Option<Vec<usize>>| GapFilter { scopes, ..Default::default() };

    // Whole trace, no scoping: one gap per rank.
    let gaps = compute_gaps_in(&trace, f(None));
    assert_eq!(gaps.len(), 2);

    // Scoped to rank 0 (the key compute_gaps_in itself uses): only its gap.
    let gaps = compute_gaps_in(&trace, f(Some(vec![0])));
    assert_eq!(gaps.len(), 1);
    assert_eq!(gaps[0].rank, Some(0));
    assert_eq!(gaps[0].t0, 10.0);
    assert_eq!(gaps[0].t1, 50.0);

    // Empty scope set: no rank is in it, so nothing is listed.
    assert!(compute_gaps_in(&trace, f(Some(Vec::new()))).is_empty());
}

#[test]
fn selection_gpu_summary_measures_idle_between_kernels() {
    // Two GPU streams plus a CPU event the summary must ignore: the span
    // runs from the leftmost kernel start to the rightmost kernel end,
    // busy is the UNION across streams (overlap [12,18] inside [10,20]
    // must not double-count), idle = span - busy.
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![evc(10.0, 10.0, 0, 0), evc(50.0, 10.0, 1, 0)]),
            ("stream 8", true, vec![evc(12.0, 6.0, 2, 0)]),
            ("cpu", false, vec![evc(25.0, 20.0, 3, 1)]),
        ],
        &["kernel", "cpu_op"],
    );
    let events: std::collections::HashSet<(u32, u32)> =
        [(0, 0), (0, 1), (1, 0), (2, 0)].into_iter().collect();
    let summary = selection_gpu_summary(&trace, &events, [0.0, 100.0, 0.0, 0.0]);
    // Unlabeled streams share the single-rank scope: one entry.
    assert_eq!(summary.len(), 1);
    let g = &summary[0];
    assert_eq!(g.t0, 10.0);
    assert_eq!(g.t1, 60.0);
    assert!((g.busy - 20.0).abs() < 1e-12);
    // Idle 30µs of the 50µs span.
    assert!((g.t1 - g.t0 - g.busy - 30.0).abs() < 1e-12);
}

#[test]
fn selection_gpu_summary_clips_to_selection_edges() {
    // A kernel straddling the selection's left edge counts only its inside
    // portion (the span and busy both start at the clipped edge).
    let trace = gaps_trace(
        vec![("stream 7", true, vec![
            evc(5.0, 10.0, 0, 0),   // [5,15]; selection starts at 10
            evc(50.0, 10.0, 1, 0),  // [50,60]
        ])],
        &["kernel"],
    );
    let events: std::collections::HashSet<(u32, u32)> =
        [(0, 0), (0, 1)].into_iter().collect();
    let summary = selection_gpu_summary(&trace, &events, [10.0, 100.0, 0.0, 0.0]);
    assert_eq!(summary.len(), 1);
    let g = &summary[0];
    assert_eq!(g.t0, 10.0);
    assert_eq!(g.t1, 60.0);
    assert!((g.busy - 15.0).abs() < 1e-12);
}

#[test]
fn selection_gpu_summary_requires_gpu_work() {
    // CPU-only selections — and GPU tracks carrying only annotations —
    // have nothing to measure: None, so the summary line stays hidden.
    let trace = gaps_trace(
        vec![
            ("cpu", false, vec![evc(10.0, 10.0, 0, 1)]),
            ("stream 7", true, vec![evc(20.0, 5.0, 1, 2)]), // gpu_user_annotation
        ],
        &["kernel", "cpu_op", "gpu_user_annotation"],
    );
    let all: std::collections::HashSet<(u32, u32)> = [(0, 0), (1, 0)].into_iter().collect();
    assert!(selection_gpu_summary(&trace, &all, [0.0, 100.0, 0.0, 0.0]).is_empty());
    let cpu_only: std::collections::HashSet<(u32, u32)> = [(0, 0)].into_iter().collect();
    assert!(selection_gpu_summary(&trace, &cpu_only, [0.0, 100.0, 0.0, 0.0]).is_empty());
}

#[test]
fn selection_gpu_summary_is_per_rank() {
    // Rank 0: kernels [10,20] and [50,60] → span 50, busy 20, idle 30.
    // Rank 1: one kernel [15,25] → span 10, busy 10, idle 0. Rank 1
    // working must not mask rank 0's idle (and vice versa), so each rank
    // gets its own entry, sorted by rank.
    let trace = gaps_trace(
        vec![
            ("Rank 0 stream 7", true, vec![evc(10.0, 10.0, 0, 0), evc(50.0, 10.0, 1, 0)]),
            ("Rank 1 stream 7", true, vec![evc(15.0, 10.0, 0, 0)]),
        ],
        &["kernel"],
    );
    let events: std::collections::HashSet<(u32, u32)> =
        [(0, 0), (0, 1), (1, 0)].into_iter().collect();
    let summary = selection_gpu_summary(&trace, &events, [0.0, 100.0, 0.0, 0.0]);
    assert_eq!(summary.len(), 2);
    assert_eq!(summary[0].rank, Some(0));
    assert!((summary[0].t1 - summary[0].t0 - summary[0].busy - 30.0).abs() < 1e-12);
    assert_eq!(summary[1].rank, Some(1));
    assert!((summary[1].t1 - summary[1].t0 - summary[1].busy).abs() < 1e-12);
}

#[test]
fn gaps_top_n_truncation() {
    // 600 gaps of strictly increasing length: only GAP_TOP_N survive, and
    // they must be the biggest ones, biggest first.
    let mut events = Vec::new();
    let mut t = 0.0;
    for i in 1..=600 {
        events.push(evc(t, 1.0, 0, 0));
        t += 1.0;
        t += i as f64;
    }
    events.push(evc(t, 1.0, 0, 0));
    let trace = gaps_trace(
        vec![("stream 7", true, events)],
        &["kernel"],
    );
    let gaps = compute_gaps(&trace);
    assert_eq!(gaps.len(), GAP_TOP_N);
    assert_eq!(gaps[0].t1 - gaps[0].t0, 600.0);
    assert_eq!(gaps[GAP_TOP_N - 1].t1 - gaps[GAP_TOP_N - 1].t0, 101.0);
    // Sorted descending by length.
    for w in gaps.windows(2) {
        assert!(w[0].t1 - w[0].t0 >= w[1].t1 - w[1].t0);
    }
}

#[test]
fn gaps_no_gpu_work_cats() {
    // A trace with GPU tracks but no kernel/gpu_* cats (e.g. an
    // annotation-only projection) has no GPU busy intervals to gap.
    let trace = gaps_trace(
        vec![
            ("stream 7", true, vec![evc(0.0, 10.0, 0, 1)]),
            ("stream 8", true, vec![evc(30.0, 10.0, 1, 1)]),
        ],
        &["kernel", "cpu_op"],
    );
    assert!(compute_gaps(&trace).is_empty());
}

#[test]
fn gap_why_classification() {
    let mk = |has_cpu: bool, cpu: f64, rt: f64| GapInfo {
        rank: None, t0: 0.0, t1: 1.0,
        cpu_busy_frac: cpu, runtime_frac: rt, has_cpu,
        cpu_top: Vec::new(), prev_name: None, next_name: None, focus_track: 0,
    };
    // No CPU tracks at all → can't tell, say so instead of "idle".
    assert_eq!(gap_why(&mk(false, 0.0, 0.0)).0, "no CPU data");
    // CPU tracks existed but were ~empty → the GPU starved.
    assert_eq!(gap_why(&mk(true, 0.1, 0.0)).0, "CPU idle (starved)");
    // Runtime covered at least half of the CPU's busy time → launch-bound.
    assert_eq!(gap_why(&mk(true, 0.8, 0.5)).0, "runtime-bound");
    // CPU busy with framework work rather than issuing launches.
    assert_eq!(gap_why(&mk(true, 0.8, 0.1)).0, "host-bound");
}

/// Regression for "most gaps are not clickable": mimics draw_gaps_table
/// (span-all-columns selectables labeled by visible rank with a `##gap{row}`
/// ID suffix, tooltip-bearing text cells) and clicks a middle row, on a
/// column to the right of the selectable's own cell. The label suffix is
/// load-bearing — with the bare rank as the label, rows of the same rank
/// share an ImGui ID and clicks stop registering (reproduced here before the
/// fix); the Steps table avoids this by labeling rows with their index.
#[test]
fn gaps_table_rows_are_clickable() {
    let _imgui_ctx = IMGUI_CTX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut imgui = imgui::Context::create();
    imgui.io_mut().display_size = [1600.0, 900.0];
    imgui.fonts().build_rgba32_texture();

    const N_ROWS: i32 = 20;
    let mut row_rects: Vec<([f32; 2], [f32; 2])> = Vec::new();
    let mut clicked: Vec<i32> = Vec::new();

    let run_frame = |imgui: &mut imgui::Context, row_rects: &mut Vec<([f32; 2], [f32; 2])>, clicked: &mut Vec<i32>| {
        let ui = imgui.new_frame();
        ui.window("bottom")
            .position([0.0, 300.0], imgui::Condition::Always)
            .size([1600.0, 600.0], imgui::Condition::Always)
            .flags(imgui::WindowFlags::NO_DECORATION | imgui::WindowFlags::NO_SAVED_SETTINGS)
            .build(|| {
                use imgui::{TableColumnFlags, TableColumnSetup, TableFlags};
                let headers: [&str; 4] = ["Rank", "Start", "Length", "Why"];
                let flags = TableFlags::RESIZABLE
                    | TableFlags::ROW_BG
                    | TableFlags::BORDERS_INNER_V
                    | TableFlags::SCROLL_Y
                    | TableFlags::NO_SAVED_SETTINGS
                    | TableFlags::SIZING_FIXED_FIT;
                let mk_col = |i: usize| {
                    let mut c = TableColumnSetup::new(headers[i]);
                    if i == 3 { c.flags |= TableColumnFlags::WIDTH_STRETCH; }
                    c
                };
                let cols: [TableColumnSetup<&str>; 4] = std::array::from_fn(mk_col);
                if let Some(_t) = ui.begin_table_header_with_sizing("##gaps", cols, flags, [1600.0, 550.0], 0.0) {
                    let row_h = ui.current_font_size() + ROW_PAD;
                    let clipper = imgui::ListClipper::new(N_ROWS)
                        .items_height(row_h)
                        .begin(&ui);
                    for row in clipper.iter() {
                        ui.table_next_row();
                        ui.table_set_column_index(0);
                        // Rank-duplicated visible text + unique ID suffix,
                        // exactly like the fixed draw_gaps_table.
                        let label = format!("{}##gap{row}", row % 4);
                        let pressed = ui.selectable_config(label)
                            .span_all_columns(true)
                            .build();
                        let min = ui.item_rect_min();
                        let max = [min[0] + ui.item_rect_size()[0], min[1] + ui.item_rect_size()[1]];
                        if row_rects.len() == row as usize { row_rects.push((min, max)); }
                        if pressed { clicked.push(row); }
                        ui.table_set_column_index(1);
                        ui.text("100.0 ms");
                        ui.table_set_column_index(2);
                        ui.text("20.0 ms");
                        ui.table_set_column_index(3);
                        ui.text_colored([0.9, 0.5, 0.5, 1.0], "runtime-bound");
                        if ui.is_item_hovered() {
                            ui.tooltip_text("tooltip body");
                        }
                    }
                }
            });
        imgui.render();
    };

    // Frame 1: layout only, capture row rects.
    imgui.io_mut().add_mouse_pos_event([-1000.0, -1000.0]);
    run_frame(&mut imgui, &mut row_rects, &mut clicked);
    assert!(row_rects.len() == N_ROWS as usize);

    // Click row 12 in the middle of the "Why" column (right of col 0).
    let (min, max) = row_rects[12];
    let target = [min[0] + (max[0] - min[0]) * 0.5, (min[1] + max[1]) / 2.0];
    // A few hover frames so tooltips/steady state settle.
    for _ in 0..3 {
        imgui.io_mut().add_mouse_pos_event(target);
        run_frame(&mut imgui, &mut row_rects, &mut clicked);
    }
    imgui.io_mut().add_mouse_pos_event(target);
    imgui.io_mut().add_mouse_button_event(imgui::MouseButton::Left, true);
    run_frame(&mut imgui, &mut row_rects, &mut clicked);
    imgui.io_mut().add_mouse_pos_event(target);
    imgui.io_mut().add_mouse_button_event(imgui::MouseButton::Left, false);
    run_frame(&mut imgui, &mut row_rects, &mut clicked);
    assert_eq!(clicked, vec![12], "click on row 12 must register exactly row 12");
}

