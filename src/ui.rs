use crate::parse::fnv1a;
use crate::types::*;
use imgui::{ImColor32, StyleColor, StyleVar, WindowFlags};
use std::fmt::Write;
use std::os::raw::c_char;
use winit::keyboard::KeyCode;

fn draw_text_clipped(col: ImColor32, text: &str, pos: [f32; 2], clip: [f32; 4]) {
    unsafe {
        let raw_dl = imgui_sys::igGetWindowDrawList();
        let font = imgui_sys::igGetFont();
        let font_size = imgui_sys::igGetFontSize();
        let start = text.as_ptr() as *const c_char;
        let end = (start as usize + text.len()) as *const c_char;
        imgui_sys::ImDrawList_AddText_FontPtr(
            raw_dl,
            font,
            font_size,
            imgui_sys::ImVec2 { x: pos[0], y: pos[1] },
            col.to_bits(),
            start,
            end,
            0.0,
            &imgui_sys::ImVec4 { x: clip[0], y: clip[1], z: clip[2], w: clip[3] },
        );
    }
}

fn draw_text_wrapped(col: ImColor32, text: &str, pos: [f32; 2], wrap_width: f32, clip: [f32; 4]) {
    // imgui word-wrap only breaks on blanks (space/tab/ideographic) plus .,;!?".
    // A kernel signature like `void vllm::silu_and_mul_kernel<c10::BFloat16>(...)`
    // has no such break point except the space after "void", so imgui wraps there
    // and shoves the giant identifier wholesale onto the next line — a one-line-tall
    // lane then shows only "void". Swap ASCII spaces for U+00A0 (SF Mono renders it
    // with an identical glyph and advance, but it is not a wrap point) so the whole
    // name is one atomic word that imgui slices character-by-character, filling every
    // line of the lane top-to-bottom.
    let nbsp_buf;
    let text = if text.as_bytes().contains(&b' ') {
        nbsp_buf = text.replace(' ', "\u{a0}");
        nbsp_buf.as_str()
    } else {
        text
    };
    unsafe {
        let raw_dl = imgui_sys::igGetWindowDrawList();
        let font = imgui_sys::igGetFont();
        let font_size = imgui_sys::igGetFontSize();
        let start = text.as_ptr() as *const c_char;
        let end = (start as usize + text.len()) as *const c_char;
        imgui_sys::ImDrawList_AddText_FontPtr(
            raw_dl,
            font,
            font_size,
            imgui_sys::ImVec2 { x: pos[0], y: pos[1] },
            col.to_bits(),
            start,
            end,
            wrap_width,
            &imgui_sys::ImVec4 { x: clip[0], y: clip[1], z: clip[2], w: clip[3] },
        );
    }
}

pub fn draw_selection_histogram(
    ui: &imgui::Ui,
    trace: &Trace,
    stats: &[SelectionEntry],
    aggregate: bool,
    buf: &mut DrawBuf,
) {
    let avail_w = ui.window_size()[0] - 2.0 * ui.clone_style().window_padding[0];
    let bar_h = HISTOGRAM_BAR_H;
    let win_x = ui.window_pos()[0] + ui.clone_style().window_padding[0];
    let cursor_y = ui.cursor_screen_pos()[1];
    let cursor = [win_x, cursor_y];
    let dl = ui.get_window_draw_list();

    buf.sel_bars.clear();
    if aggregate {
        for (i, s) in stats.iter().enumerate() {
            buf.sel_bars.push((s.total_dur, i as u32));
        }
    } else {
        let mut idx = 0u32;
        for s in stats {
            for &d in &s.durations {
                buf.sel_bars.push((d, idx));
                idx += 1;
            }
        }
    }
    buf.sel_bars.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let total_dur: f64 = buf.sel_bars.iter().map(|b| b.0).sum();
    if total_dur <= 0.0 { return; }

    let mut x = cursor[0];
    let y = cursor[1];

    let mut flat_names: Vec<u32> = Vec::new();
    if !aggregate {
        for s in stats {
            for _ in &s.durations { flat_names.push(s.name); }
        }
    }

    let font_h = ui.current_font_size();
    let label_h = font_h + 2.0;

    for &(dur, idx) in &buf.sel_bars {
        let w = (dur / total_dur * avail_w as f64) as f32;
        if w < 0.5 { x += w; continue; }
        let name_idx = if aggregate { stats[idx as usize].name } else { flat_names[idx as usize] };
        let name = &trace.names[name_idx as usize];
        let color = name_color(name);
        dl.add_rect([x, y], [x + w, y + bar_h], color).filled(true).build();
        if w > 1.0 {
            dl.add_line([x, y], [x, y + bar_h], col32(20, 20, 20, 200)).build();
        }
        if w > 60.0 {
            buf.fmt.clear();
            if aggregate { buf.fmt.push_str(name); }
            else { write_time(&mut buf.fmt, dur); }
            let text_w = ui.calc_text_size(&buf.fmt)[0];
            if text_w < w - 6.0 {
                dl.add_text([x + 3.0, y + 2.0], col32(240, 240, 240, 255), &buf.fmt);
            }
        }
        buf.fmt.clear();
        write_time(&mut buf.fmt, dur);
        let tw = ui.calc_text_size(&buf.fmt)[0];
        if tw < w - 4.0 {
            let tx = x + (w - tw) / 2.0;
            dl.add_text([tx, y + bar_h + 1.0], col32(160, 160, 160, 255), &buf.fmt);
        }
        x += w;
    }

    ui.dummy([avail_w, bar_h + label_h + 2.0]);
}

/// Column order for the stats table. Also the sort key indices persisted in
/// `Pane::sort_col`, so keep the two in sync. "Occ Limit" is always present
/// (so a persisted sort_col/column layout survives switching aggregate modes)
/// but is only ever populated in individual-row mode, since an aggregate row
/// can span kernel launches with different limiting factors.
const STATS_HEADERS: [&str; 8] = ["Name", "Count", "Total", "%", "Mean", "Median", "Max", "Occ Limit"];

/// CUDA's default static shared-memory-per-block cap (48KB) on every
/// architecture since Kepler. A kernel that opts into more via
/// `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` — routine for
/// GEMM/attention/NCCL kernels — makes CUPTI's launch-config occupancy
/// calculator check against this (wrong, too-low) ceiling instead, which
/// falsely reports "SMEM" as the limiting factor with zero active blocks.
const CUDA_DEFAULT_SHARED_MEM_PER_BLOCK: u64 = 49152;

/// Look up CUPTI's `occupancy.limitingFactors` (e.g. "WARPS", "SMEM",
/// "REGS|BLOCKS") for one event from its raw args JSON, and whether it's
/// likely a calculator artifact rather than a real occupancy limit (see
/// `CUDA_DEFAULT_SHARED_MEM_PER_BLOCK`). Returns ("", false) for anything
/// that isn't a CUDA kernel launch (CPU ops, no args, etc) — most events
/// simply don't have this field, which isn't an error.
pub(crate) fn kernel_occ_limit(trace: &Trace, track_idx: u32, event_idx: u32) -> (&str, bool) {
    if track_idx == u32::MAX { return ("", false); }
    let track = match trace.tracks.get(track_idx as usize) { Some(t) => t, None => return ("", false) };
    let ev = match track.events.get(event_idx as usize) { Some(e) => e, None => return ("", false) };
    if ev.args_off == 0 { return ("", false); }
    let raw = match trace.raw_bufs.get(track.raw_buf_idx as usize) { Some(r) => r, None => return ("", false) };
    let off = ev.args_off as usize;
    if off >= raw.len() { return ("", false); }
    let end = crate::parse::skip_value(raw, off);
    let args = &raw[off..end];
    let limit = crate::parse::find_str_field(args, b"limitingFactors").unwrap_or("");
    let suspect = limit.contains("SMEM")
        && crate::parse::find_int_field(args, 0, b"shared memory")
            .map(|smem| smem as u64 > CUDA_DEFAULT_SHARED_MEM_PER_BLOCK)
            .unwrap_or(false);
    (limit, suspect)
}

/// Renders the "N hidden" indicator and its "Clear" (unhide-all) button, placed
/// on the current row after `spacing` px. Draws nothing when nothing is hidden.
/// Lives beside the Selection table's "Hide Selected" so all hide/unhide
/// controls sit together instead of being scattered into the toolbar.
pub fn draw_hidden_clear(ui: &imgui::Ui, spacing: f32, hidden_names: &mut [bool], fmt: &mut String) {
    let n_hidden = hidden_names.iter().filter(|&&h| h).count();
    if n_hidden == 0 {
        return;
    }
    ui.same_line_with_spacing(0.0, spacing);
    fmt.clear();
    write!(fmt, "{} hidden", n_hidden).unwrap();
    ui.text_colored([1.0, 0.7, 0.3, 1.0], fmt);
    ui.same_line_with_spacing(0.0, 4.0);
    if ui.small_button("Clear##unhide") {
        for h in hidden_names.iter_mut() {
            *h = false;
        }
    }
}

/// Run `f` over `0..n` in parallel, splitting the range evenly across
/// available cores — used for the Occ Limit column's per-row JSON parse,
/// which is independent per row and CPU-bound (unlike the other columns'
/// plain field reads). Falls back to sequential below `MIN_PARALLEL`, where
/// thread spawn overhead would dominate the actual work.
pub(crate) fn parallel_occ_limit<'a>(n: usize, f: &(impl Fn(usize) -> (&'a str, bool) + Sync)) -> Vec<(&'a str, bool)> {
    const MIN_PARALLEL: usize = 20_000;
    if n < MIN_PARALLEL {
        return (0..n).map(f).collect();
    }
    // rayon sizes its pool from the environment (== available_parallelism by
    // default) and preserves index order on a range's `.collect()`, so the
    // manual available_parallelism()-based chunking this replaced is
    // redundant — rayon already splits/rebalances the range itself.
    use rayon::prelude::*;
    (0..n).into_par_iter().map(|i| f(i)).collect()
}

pub fn draw_stats_table(
    ui: &imgui::Ui,
    trace: &Trace,
    stats: &[KernelStats],
    // (track_idx, event_idx) parallel to `stats`, for the Occ Limit column.
    // `None` in aggregate mode, where a row can span many launches and no
    // single limiting factor applies.
    event_refs: Option<&[(u32, u32)]>,
    // Bumped by `compute_aggregates` whenever the selection is rebuilt; part
    // of the sort cache key (see `DrawBuf::sort_cache_key`) so an unchanged
    // selection doesn't get re-sorted every redraw.
    generation: u64,
    search: &mut String,
    search_changed: &mut bool,
    sort_col: &mut usize,
    sort_asc: &mut bool,
    buf: &mut DrawBuf,
    table_id: &str,
) {
    use imgui::{TableColumnFlags, TableColumnSetup, TableFlags};

    // A real imgui table: native resizing, per-column hide/reorder via the
    // header context menu, row backgrounds and borders. Column layout, width,
    // order and visibility live in imgui's internal per-table state (keyed by
    // this id inside the per-pane `##bottom{pi}` window), so the two panes no
    // longer share a single width array the way the old hand-rolled table did.
    let flags = TableFlags::RESIZABLE
        | TableFlags::REORDERABLE
        | TableFlags::HIDEABLE
        | TableFlags::SORTABLE
        | TableFlags::ROW_BG
        | TableFlags::BORDERS_INNER_V
        | TableFlags::BORDERS_OUTER
        | TableFlags::SCROLL_Y
        | TableFlags::NO_SAVED_SETTINGS
        | TableFlags::SIZING_FIXED_FIT;

    let avail = ui.content_region_avail();
    // Pin each numeric column to a stable width up front. If we left them to
    // auto-fit (init width 0), imgui re-measures against only the rows the
    // clipper renders, so the columns jitter horizontally while scrolling. A
    // width sized to the header and a representative value keeps them steady.
    let num_w = ui
        .calc_text_size("0000.00 ms")[0]
        .max(ui.calc_text_size("Median")[0] + 22.0);
    // "WARPS|BLOCKS" is the widest realistic limiting-factor combo.
    let occ_w = ui.calc_text_size("WARPS|BLOCKS")[0] + 12.0;
    let cols: [TableColumnSetup<&str>; 8] = std::array::from_fn(|i| {
        let mut c = TableColumnSetup::new(STATS_HEADERS[i]);
        if i == 0 {
            // The name is textual and usually long, so let it stretch to absorb
            // all the slack. The numeric columns fit their (short) content and
            // get pushed to the right. Name can't be hidden (it's the key).
            c.flags |= TableColumnFlags::WIDTH_STRETCH | TableColumnFlags::NO_HIDE;
        } else {
            c.flags |= TableColumnFlags::WIDTH_FIXED;
            c.init_width_or_weight = if i == 7 { occ_w } else { num_w };
        }
        if i == *sort_col {
            c.flags |= TableColumnFlags::DEFAULT_SORT;
            c.flags |= if *sort_asc {
                TableColumnFlags::PREFER_SORT_ASCENDING
            } else {
                TableColumnFlags::PREFER_SORT_DESCENDING
            };
        }
        c
    });

    let Some(_t) = ui.begin_table_header_with_sizing(table_id, cols, flags, [avail[0], avail[1]], 0.0)
    else {
        return;
    };

    // Pull sort direction from imgui when the user clicks a header, and mirror
    // it into the pane-persisted sort state so it survives data rebuilds.
    if let Some(specs) = ui.table_sort_specs_mut() {
        specs.conditional_sort(|specs| {
            if let Some(spec) = specs.iter().next() {
                *sort_col = spec.column_idx();
                *sort_asc = matches!(
                    spec.sort_direction(),
                    Some(imgui::TableSortDirection::Ascending)
                );
            }
        });
    }

    let total_sum: f64 = stats.iter().map(|s| s.total_dur).sum();
    let avg = |s: &KernelStats| if s.count > 0 { s.total_dur / s.count as f64 } else { 0.0 };
    let pct = |s: &KernelStats| if total_sum > 0.0 { s.total_dur / total_sum } else { 0.0 };

    let occ_limit_uncached = |si: usize| -> (&str, bool) {
        match event_refs.and_then(|r| r.get(si)) {
            Some(&(ti, ei)) => kernel_occ_limit(trace, ti, ei),
            None => ("", false),
        }
    };
    // This table resorts on every call — i.e. every redraw, which fires on
    // every mouse-move, not just when the user actually changes the sort or
    // the selection. Skip the O(n log n) sort_by entirely when nothing that
    // could change the order has changed (measured ~120ms wasted per redraw
    // on a 1M-row selection even for a plain numeric column).
    let stats_is_individual = event_refs.is_some();
    let cache_key = (generation, *sort_col, *sort_asc, stats.len(), stats_is_individual);
    if buf.sort_cache_key != Some(cache_key) {
        // Sorting by Occ Limit naively (parsing each row's raw JSON args
        // inside the comparator) measured at 140ms for just 15k rows on its
        // own. Parse each row exactly once up front instead, only when it's
        // the active sort key; row *rendering* below still looks values up
        // lazily (bounded by the clipper to the visible rows). Even parsed
        // once, this is a genuinely CPU-bound O(n) pass — GPU kernel events
        // carry a much larger args blob (occupancy dict, grid/block arrays)
        // than CPU ops, so a kernel-heavy selection can still cost 100ms+ at
        // ~100k rows single-threaded. Each row is independent, so split the
        // parse across threads the same way loader.rs parallelizes per-track
        // work at load time.
        let occ_limit_sort_cache: Option<Vec<(&str, bool)>> = (*sort_col == 7)
            .then(|| parallel_occ_limit(stats.len(), &occ_limit_uncached));
        let occ_limit_for_sort = |si: usize| -> (&str, bool) {
            match &occ_limit_sort_cache {
                Some(cache) => cache[si],
                None => occ_limit_uncached(si),
            }
        };

        buf.sort_idx.clear();
        buf.sort_idx.extend(0..stats.len());
        buf.sort_idx.sort_by(|&a, &b| {
            let (sa, sb) = (&stats[a], &stats[b]);
            let ord = match *sort_col {
                0 => trace.names[sa.name as usize].cmp(&trace.names[sb.name as usize]),
                1 => sa.count.cmp(&sb.count),
                2 => sa.total_dur.partial_cmp(&sb.total_dur).unwrap(),
                3 => pct(sa).partial_cmp(&pct(sb)).unwrap(),
                4 => avg(sa).partial_cmp(&avg(sb)).unwrap(),
                5 => sa.median_dur.partial_cmp(&sb.median_dur).unwrap(),
                6 => sa.max_dur.partial_cmp(&sb.max_dur).unwrap(),
                _ => occ_limit_for_sort(a).0.cmp(occ_limit_for_sort(b).0),
            };
            if *sort_asc { ord } else { ord.reverse() }
        });
        buf.sort_cache_key = Some(cache_key);
    }
    let occ_limit = occ_limit_uncached;

    let row_h = ui.current_font_size() + ROW_PAD;
    let dl = ui.get_window_draw_list();
    // Without an explicit item height, ImGuiListClipper auto-detects it by
    // measuring one throwaway row — which doesn't reliably skip ahead when
    // interleaved with table_next_row(), degrading to visiting every row
    // (not just the visible ones) regardless of total count. This was the
    // real cost behind "sorting is slow": it ran unconditionally on every
    // call, independent of the sort cache above, so it stayed slow even on a
    // cache hit. Measured at 112k rows: this alone was ~160ms.
    let clipper = imgui::ListClipper::new(buf.sort_idx.len() as i32)
        .items_height(row_h)
        .begin(ui);
    for row in clipper.iter() {
        let si = buf.sort_idx[row as usize];
        let s = &stats[si];
        let name = &trace.names[s.name as usize];
        ui.table_next_row();

        // Name cell: color swatch + (clipped) name, clickable to search.
        ui.table_set_column_index(0);
        let cur = ui.cursor_screen_pos();
        dl.add_rect(
            [cur[0], cur[1] + EV_INSET],
            [cur[0] + SWATCH_W, cur[1] + row_h - EV_INSET],
            name_color(name),
        )
        .filled(true)
        .rounding(EV_ROUNDING)
        .build();
        ui.set_cursor_screen_pos([cur[0] + SWATCH_PAD, cur[1]]);
        ui.text(name);
        if ui.is_item_hovered() {
            ui.tooltip_text(name);
        }
        if ui.is_item_clicked() {
            search.clear();
            search.push_str(name);
            *search_changed = true;
        }
        if ui.is_item_clicked_with_button(imgui::MouseButton::Right) {
            ui.set_clipboard_text(name);
        }

        if ui.table_set_column_index(1) {
            buf.fmt.clear();
            write!(buf.fmt, "{}", s.count).unwrap();
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(2) {
            buf.fmt.clear();
            write_time(&mut buf.fmt, s.total_dur);
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(3) {
            buf.fmt.clear();
            write!(buf.fmt, "{:.1}%", pct(s) * 100.0).unwrap();
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(4) {
            buf.fmt.clear();
            write_time(&mut buf.fmt, avg(s));
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(5) {
            buf.fmt.clear();
            write_time(&mut buf.fmt, s.median_dur);
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(6) {
            buf.fmt.clear();
            write_time(&mut buf.fmt, s.max_dur);
            ui.text(&buf.fmt);
        }
        if ui.table_set_column_index(7) {
            let (limit, suspect) = occ_limit(si);
            if !limit.is_empty() {
                if suspect {
                    ui.text_colored([1.0, 0.7, 0.3, 1.0], limit);
                } else {
                    ui.text(limit);
                }
                if ui.is_item_hovered() {
                    if suspect {
                        ui.tooltip_text(
                            "Likely a calculator artifact.\n\n\
                            This kernel requests shared memory above CUDA's default\n\
                            48KB static limit (common for GEMM/attention/NCCL\n\
                            kernels). CUPTI's occupancy calculator checks against\n\
                            that default instead of the larger opt-in limit the\n\
                            kernel actually used.\n\n\
                            Real occupancy is probably higher than this implies.",
                        );
                    } else {
                        ui.tooltip_text(
                            "CUDA occupancy-limiting factor for this launch,\n\
                            from CUPTI's launch-config calculator.",
                        );
                    }
                }
            }
        }
    }
}

pub fn winit_to_imgui(code: KeyCode) -> Option<imgui::Key> {
    Some(match code {
        KeyCode::Tab => imgui::Key::Tab,
        KeyCode::ArrowLeft => imgui::Key::LeftArrow,
        KeyCode::ArrowRight => imgui::Key::RightArrow,
        KeyCode::ArrowUp => imgui::Key::UpArrow,
        KeyCode::ArrowDown => imgui::Key::DownArrow,
        KeyCode::PageUp => imgui::Key::PageUp,
        KeyCode::PageDown => imgui::Key::PageDown,
        KeyCode::Home => imgui::Key::Home,
        KeyCode::End => imgui::Key::End,
        KeyCode::Delete => imgui::Key::Delete,
        KeyCode::Backspace => imgui::Key::Backspace,
        KeyCode::Enter => imgui::Key::Enter,
        KeyCode::Escape => imgui::Key::Escape,
        KeyCode::Space => imgui::Key::Space,
        KeyCode::ControlLeft => imgui::Key::LeftCtrl,
        KeyCode::ControlRight => imgui::Key::RightCtrl,
        KeyCode::ShiftLeft => imgui::Key::LeftShift,
        KeyCode::ShiftRight => imgui::Key::RightShift,
        KeyCode::AltLeft => imgui::Key::LeftAlt,
        KeyCode::AltRight => imgui::Key::RightAlt,
        KeyCode::SuperLeft => imgui::Key::LeftSuper,
        KeyCode::SuperRight => imgui::Key::RightSuper,
        KeyCode::KeyA => imgui::Key::A,
        KeyCode::KeyC => imgui::Key::C,
        KeyCode::KeyF => imgui::Key::F,
        KeyCode::KeyN => imgui::Key::N,
        KeyCode::KeyV => imgui::Key::V,
        KeyCode::KeyX => imgui::Key::X,
        KeyCode::KeyZ => imgui::Key::Z,
        KeyCode::KeyY => imgui::Key::Y,
        KeyCode::Slash => imgui::Key::Slash,
        _ => return None,
    })
}

/// Collect one GPU stream track's events for the merged/Tetris-packed view:
/// visible (not hidden by name) within the current time window, and not a
/// "grandparent" wrapper (an event that itself has a child with a further
/// child) — those would each claim their own tetris row while contributing
/// no information beyond their descendants', so they're stripped. Appends
/// (ts, dur, track_idx, event_idx) tuples to `out`.
///
/// The `has_grandchild` check scans forward from each surviving event to its
/// own end (`take_while ts <= ev.ts + ev.dur`), so its cost scales with how
/// many descendants fall inside that window — cheap for ordinary short
/// kernels, but a whole-generation-step wrapper spanning hundreds of
/// milliseconds forces a scan across everything nested under it. Hiding such
/// a wrapper by name (see `hidden_names`) skips it before this scan runs.
pub(crate) fn collect_merged_track_events(
    gt: &Track,
    ti: usize,
    view_t0: f64,
    view_t1: f64,
    hidden_names: &[bool],
    out: &mut Vec<(f64, f64, u32, u32)>,
) {
    let start = bisect_overlap(&gt.events, &gt.prefix_max_dur, view_t0);
    let end = gt.events.partition_point(|e| e.ts <= view_t1);
    for ei in start..end {
        let ev = &gt.events[ei];
        // bisect_overlap's `start` is a conservative lower bound (it accounts
        // for the longest duration seen up to any given index, not this
        // event's own), so it can land well before events that don't
        // actually reach view_t0 — a single long-running event earlier in
        // the stream is enough to pull in everything after it, regardless of
        // whether each one individually overlaps the current window. Skip
        // those explicitly instead of treating "start..end" as "overlapping".
        if ev.ts + ev.dur < view_t0 { continue; }
        if hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
        let has_grandchild = gt.events[ei + 1..].iter()
            .take_while(|e2| e2.ts <= ev.ts + ev.dur)
            .any(|e2| e2.depth > ev.depth + 1);
        if has_grandchild { continue; }
        out.push((ev.ts, ev.dur, ti as u32, ei as u32));
    }
}

/// Row heights for "even spacing" mode: `has_content[vi]` says whether row
/// `vi` has any event in the current view window. Rows without one collapse
/// to a thin fixed strip (`EMPTY_ROW_H`, clamped so it can't itself exceed
/// an equal share if `avail` is very small); the height freed by collapsing
/// them is split evenly among the rows that do have content — so, e.g., a
/// zoomed-in merged multi-rank view with only one rank active at this
/// instant lets that one row fill nearly the whole available height instead
/// of splitting it uniformly with dozens of blank rank rows. Falls back to
/// a plain uniform split if every row (or no row) has content, since
/// there's nothing to collapse into.
pub(crate) fn even_spacing_heights(has_content: &[bool], avail: f32) -> Vec<f32> {
    let n = has_content.len();
    if n == 0 { return Vec::new(); }
    const EMPTY_ROW_H: f32 = 6.0;
    let n_empty = has_content.iter().filter(|&&c| !c).count();
    let n_content = n - n_empty;
    let empty_h = EMPTY_ROW_H.min(avail / n as f32);
    let content_h = if n_content > 0 {
        ((avail - n_empty as f32 * empty_h) / n_content as f32).max(0.0)
    } else {
        avail / n as f32
    };
    has_content.iter().map(|&c| if n_content > 0 && !c { empty_h } else { content_h }).collect()
}

#[allow(clippy::too_many_arguments)]
pub fn draw_timeline(
    ui: &imgui::Ui,
    trace: &Trace,
    view: &mut View,
    show_cpu: bool,
    buf: &mut DrawBuf,
    rect: [f32; 4],
    pane_idx: usize,
    hovered: bool,
    clicked: bool,
    active: bool,
    mouse_pos: [f32; 2],
    mouse_delta: [f32; 2],
    scroll: [f32; 2],
    pinch: f32,
    ctrl: bool,
    shift: bool,
    search_mask: &[bool],
    selection: Option<[f64; 4]>,
    // The frozen (track_idx, event_idx) set a finished region-selection
    // refers to (see `Pane::capture_sel_events`, `Pane::finished_sel`). Used
    // instead of re-deriving track membership from the selection's raw Y
    // range against the CURRENT layout, which would silently reassign the
    // highlight to different tracks after Show CPU toggles, track
    // reordering, or height changes.
    finished_sel_events: &std::collections::HashSet<(u32, u32)>,
    collapsed: &mut Vec<bool>,
    hidden_names: &[bool],
    selected: Option<EventRef>,
    multi_select_name: Option<u32>,
    sel_mask: &[bool],
    label_w: f32,
    track_scales: &mut Vec<f32>,
    even_spacing: &mut bool,
    geom: &mut PaneGeom,
    track_order: &mut Vec<usize>,
    drag: &mut DragKind,
    merge_gpu: bool,
    dt: f32,
    focus: &mut Option<u32>,
) -> (Option<EventRef>, Option<EventRef>, Option<Option<[f64; 4]>>) {
    let dl = ui.get_window_draw_list();
    let tl_left = rect[0] + label_w;
    let tl_w = (rect[2] - tl_left).max(1.0);

    if let DragKind::TrackResize(pi, ti) = *drag {
        if pi == pane_idx {
            if ui.io().mouse_down[0] {
                let base_h = track_height(trace.tracks[ti].max_depth, false, 1.0);
                if base_h > 0.0 {
                    let delta_scale = mouse_delta[1] / base_h;
                    if let Some(s) = track_scales.get_mut(ti) {
                        *s = (*s + delta_scale).clamp(TRACK_SCALE_MIN, TRACK_SCALE_MAX);
                    }
                }
            } else {
                *drag = DragKind::None;
            }
        }
    }

    buf.visible.clear();
    buf.heights.clear();
    buf.y_offsets.clear();
    let n_old_groups = buf.merged_gpu_groups.len();
    let mut cumulative = 0.0f32;

    // Pre-group GPU tracks by rank when merging (simple Vec, no BTreeMap)
    let mut rank_group_idxs: Vec<(Option<usize>, usize)> = Vec::new(); // (rank, group_slot)
    let mut group_slot = 0usize;
    if merge_gpu {
        for &i in track_order.iter() {
            if !trace.tracks[i].gpu { continue; }
            let rank = crate::state::parse_rank(&trace.tracks[i].label);
            if let Some(&(_, gi)) = rank_group_idxs.iter().find(|(r, _)| *r == rank) {
                if gi < n_old_groups {
                    buf.merged_gpu_groups[gi].tracks.push(i);
                } else {
                    buf.merged_gpu_groups[gi - n_old_groups].tracks.push(i);
                }
            } else {
                let gi = group_slot;
                group_slot += 1;
                if gi < n_old_groups {
                    let g = &mut buf.merged_gpu_groups[gi];
                    g.tracks.clear();
                    g.tracks.push(i);
                    g.events.clear();
                    g.max_depth = 0;
                    g.vi = 0;
                    g.label.clear();
                    match rank {
                        Some(r) => write!(g.label, "  Rank {}", r).ok(),
                        None => write!(g.label, "GPU").ok(),
                    };
                } else {
                    let label = match rank {
                        Some(r) => format!("  Rank {}", r),
                        None => "GPU".to_string(),
                    };
                    buf.merged_gpu_groups.push(MergedGpuGroup {
                        tracks: vec![i], events: Vec::new(), max_depth: 0, vi: 0, label,
                    });
                }
                rank_group_idxs.push((rank, gi));
            }
        }
    }
    buf.merged_gpu_groups.truncate(group_slot);

    // Parallel to buf.visible/heights: whether each row has any event
    // actually overlapping the current view window. Feeds the even-spacing
    // pass below so rows with nothing to show at this zoom collapse instead
    // of claiming an equal share of the height a row with real content
    // could otherwise expand into — most useful in the merged multi-rank
    // view, where ranks are rarely in perfect lockstep, so zooming in tight
    // enough often leaves only one rank's row with anything visible while
    // the rest sit empty.
    let mut has_content: Vec<bool> = Vec::new();
    let mut emitted_ranks: Vec<bool> = vec![false; rank_group_idxs.len()];
    for &i in track_order.iter() {
        let t = &trace.tracks[i];
        if merge_gpu && t.gpu {
            let rank = crate::state::parse_rank(&t.label);
            if let Some(ri) = rank_group_idxs.iter().position(|(r, _)| *r == rank) {
                if emitted_ranks[ri] { continue; }
                emitted_ranks[ri] = true;
                let gi = rank_group_idxs[ri].1;
                let g = &buf.merged_gpu_groups[gi];
                let group_tracks: Vec<usize> = g.tracks.clone();
                let mut ev_list: Vec<(f64, f64, u32, u32)> = Vec::new();
                for &ti in &group_tracks {
                    let gt = &trace.tracks[ti];
                    collect_merged_track_events(gt, ti, view.t0, view.t1, hidden_names, &mut ev_list);
                }
                ev_list.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let mut depth_ends: Vec<f64> = Vec::new();
                let mut max_depth: u16 = 0;
                let g = &mut buf.merged_gpu_groups[gi];
                g.events.clear();
                for &(ts, dur, ti, ei) in &ev_list {
                    let d = depth_ends.iter().position(|&end| end <= ts)
                        .unwrap_or_else(|| { depth_ends.push(0.0); depth_ends.len() - 1 });
                    depth_ends[d] = ts + dur;
                    let d16 = d as u16;
                    if d16 >= max_depth { max_depth = d16 + 1; }
                    g.events.push((ti, ei, d16));
                }
                let md = max_depth.max(1);
                let first = group_tracks[0];
                let scale = track_scales.get(first).copied().unwrap_or(1.0);
                let h = md as f32 * SUB_LANE_H * scale;
                let vi = buf.visible.len();
                g.max_depth = md;
                g.vi = vi;
                buf.visible.push(first);
                buf.heights.push(h);
                buf.y_offsets.push(cumulative);
                has_content.push(!ev_list.is_empty());
                cumulative += h;
            }
            continue;
        }
        if !show_cpu && !t.gpu { continue; }
        buf.visible.push(i);
        let h = track_height(
            t.max_depth,
            collapsed.get(i).copied().unwrap_or(false),
            track_scales.get(i).copied().unwrap_or(1.0),
        );
        buf.heights.push(h);
        buf.y_offsets.push(cumulative);
        // Same conservative-lower-bound caveat as collect_merged_track_events:
        // `start` can land on an event well before view.t0, so check each
        // candidate's actual end time instead of assuming the first one
        // (or its ts alone) means real overlap.
        let start = bisect_overlap(&t.events, &t.prefix_max_dur, view.t0);
        let end = t.events.partition_point(|e| e.ts <= view.t1);
        has_content.push(t.events[start..end].iter().any(|e| e.ts + e.dur >= view.t0));
        cumulative += h;
    }
    let mut total_h = cumulative;
    let tracks_top = rect[1] + RULER_H;

    // Even-spacing mode: override the drawn row heights so visible tracks
    // fill the viewport height, down to the bottom pane. It recomputes each
    // frame from the current viewport, so dragging the bottom divider
    // (which changes rect[3]) or panning/zooming (which changes which rows
    // have anything in view) re-flows the lanes live. It intentionally does
    // not touch track_scales, so toggling the mode off restores the manual
    // layout — an easy undo. Toggled by double-clicking the last lane, or
    // the bottom-divider icon. See `even_spacing_heights` for how rows with
    // nothing in the current view collapse instead of claiming an equal
    // share of the height.
    if *even_spacing && !buf.visible.is_empty() {
        let avail = (rect[3] - tracks_top).max(1.0);
        let heights = even_spacing_heights(&has_content, avail);
        let mut cum = 0.0;
        for (vi, &h) in heights.iter().enumerate() {
            buf.heights[vi] = h;
            buf.y_offsets[vi] = cum;
            cum += h;
        }
        total_h = cum;
    }

    // Snapshot the final row layout into the pane-owned geometry. `buf` is shared
    // across panes, so after the render loop it only holds the last-drawn pane's
    // layout. Selection stats, diff extraction and clipboard copy run later and
    // must read this per-pane copy, or they'd compute against the wrong pane.
    geom.visible.clear();
    geom.visible.extend_from_slice(&buf.visible);
    geom.heights.clear();
    geom.heights.extend_from_slice(&buf.heights);
    geom.y_offsets.clear();
    geom.y_offsets.extend_from_slice(&buf.y_offsets);
    geom.merged.clear();
    for g in &buf.merged_gpu_groups {
        geom.merged.push(MergedGeom {
            vi: g.vi,
            events: g.events.clone(),
        });
    }

    if let DragKind::TrackDrag(pi, ref mut dragged_vi, _grab_off) = *drag {
        if pi == pane_idx {
            if ui.io().mouse_down[0] {
                let rel_y = mouse_pos[1] - tracks_top + view.scroll_y;
                let mut target_vi = buf.visible.len();
                for vi in 0..buf.visible.len() {
                    let mid = buf.y_offsets[vi] + buf.heights[vi] * 0.5;
                    if rel_y < mid {
                        target_vi = vi;
                        break;
                    }
                }
                if target_vi != *dragged_vi && !(target_vi == *dragged_vi + 1) {
                    let ti = buf.visible[*dragged_vi];
                    let order_pos = track_order.iter().position(|&x| x == ti).unwrap();
                    track_order.remove(order_pos);
                    let insert_order_pos = if target_vi < buf.visible.len() {
                        let target_ti = buf.visible[target_vi];
                        track_order.iter().position(|&x| x == target_ti).unwrap()
                    } else {
                        track_order.len()
                    };
                    track_order.insert(insert_order_pos, ti);
                    *dragged_vi = if target_vi > *dragged_vi { target_vi - 1 } else { target_vi };
                }
            } else {
                *drag = DragKind::None;
            }
        }
    }

    let time_range = (view.t1 - view.t0).max(1e-9);
    let px_per_us = tl_w as f64 / time_range;

    #[inline]
    fn t2x(t: f64, t0: f64, ppus: f64, left: f32) -> f32 { left + ((t - t0) * ppus) as f32 }
    #[inline]
    fn x2t(x: f32, t0: f64, ppus: f64, left: f32) -> f64 { t0 + (x - left) as f64 / ppus }

    // A live search zoom drives the view unless the user grabs it back with a
    // mouse zoom/pan; cancel then so the two don't fight for t0/t1.
    let user_zoomed = hovered && (pinch != 0.0 || (ctrl && scroll[1] != 0.0) || scroll[0] != 0.0);
    let user_panned = active && !drag.is_active() && !shift && mouse_delta[0] != 0.0;
    if user_zoomed || user_panned {
        view.anim = None;
        *focus = None;
    }
    // Resolve a pending vertical focus (from a search zoom) into the animation's
    // scroll target, now that the final row layout + viewport height are known.
    // Vertical layout doesn't depend on the horizontal zoom, so this is stable
    // even while t0/t1 are still animating.
    if let Some(ti) = focus.take() {
        if let Some(a) = view.anim.as_mut() {
            let row_vi = buf.merged_gpu_groups.iter()
                .find(|g| g.tracks.contains(&(ti as usize)))
                .map(|g| g.vi)
                .or_else(|| buf.visible.iter().position(|&v| v == ti as usize));
            if let Some(vi) = row_vi {
                let visible_h = (rect[3] - tracks_top).max(1.0);
                let max_scroll = (total_h - visible_h).max(0.0);
                let center = buf.y_offsets[vi] + buf.heights[vi] * 0.5;
                a.from_scroll = view.scroll_y;
                a.to_scroll = (center - visible_h * 0.5).clamp(0.0, max_scroll);
            }
        }
    }
    view.tick_anim(dt);

    if hovered {
        if pinch != 0.0 {
            let factor = (1.0 + pinch as f64).max(0.1);
            let ct = x2t(mouse_pos[0], view.t0, px_per_us, tl_left);
            view.t0 = ct + (view.t0 - ct) / factor;
            view.t1 = ct + (view.t1 - ct) / factor;
        }
        if ctrl {
            if scroll[1] != 0.0 {
                let factor = (scroll[1] as f64 / SCROLL_ZOOM_SENSITIVITY).exp();
                let ct = x2t(mouse_pos[0], view.t0, px_per_us, tl_left);
                view.t0 = ct + (view.t0 - ct) / factor;
                view.t1 = ct + (view.t1 - ct) / factor;
            }
        } else {
            view.scroll_y -= scroll[1];
            let max_scroll = (total_h - (rect[3] - rect[1] - RULER_H)).max(0.0);
            view.scroll_y = view.scroll_y.clamp(0.0, max_scroll);
            if scroll[0] != 0.0 {
                let dt = -scroll[0] as f64 / px_per_us;
                view.t0 += dt;
                view.t1 += dt;
            }
        }
    }

    let mut sel_change: Option<Option<[f64; 4]>> = None;

    if active && !drag.is_active() {
        if shift {
            let t = x2t(mouse_pos[0], view.t0, px_per_us, tl_left);
            let y = (mouse_pos[1] - rect[1] - RULER_H + view.scroll_y) as f64;
            if clicked {
                sel_change = Some(Some([t, t, y, y]));
            } else {
                sel_change = Some(match selection {
                    Some([s0, _, y0, _]) => Some([s0, t, y0, y]),
                    None => Some([t, t, y, y]),
                });
            }
        } else {
            let dx = mouse_delta[0] as f64;
            let dt = -dx / px_per_us;
            view.t0 += dt;
            view.t1 += dt;
        }
    }

    if clicked && !shift && !drag.is_active() {
        sel_change = Some(None);
    }

    let pad = trace.max_ts * TIMELINE_PAD_FRAC;
    let min_range = MIN_TIME_RANGE;
    let max_range = trace.max_ts + 2.0 * pad;
    let cur_range = view.t1 - view.t0;
    if cur_range < min_range {
        let c = (view.t0 + view.t1) / 2.0;
        view.t0 = c - min_range / 2.0;
        view.t1 = c + min_range / 2.0;
    } else if cur_range > max_range {
        let c = (view.t0 + view.t1) / 2.0;
        view.t0 = c - max_range / 2.0;
        view.t1 = c + max_range / 2.0;
    }
    let rn = view.t1 - view.t0;
    if view.t0 < -pad {
        view.t0 = -pad;
        view.t1 = view.t0 + rn;
    }
    if view.t1 > trace.max_ts + pad {
        view.t1 = trace.max_ts + pad;
        view.t0 = view.t1 - rn;
    }

    let time_range = (view.t1 - view.t0).max(1e-9);
    let px_per_us = tl_w as f64 / time_range;

    dl.add_rect([rect[0], rect[1]], [rect[2], rect[3]], BG_TIMELINE).filled(true).build();

    let ruler_rect = [tl_left, rect[1], rect[2], rect[1] + RULER_H];
    draw_ruler(&dl, ruler_rect, view, &mut buf.fmt);

    dl.add_rect([rect[0], rect[1]], [tl_left, rect[3]], BG_LABELS).filled(true).build();
    dl.add_line([tl_left, rect[1]], [tl_left, rect[3]], DIVIDER).build();
    dl.add_line([rect[0], rect[1] + RULER_H], [rect[2], rect[1] + RULER_H], DIVIDER).build();

    let mut hover_result: Option<EventRef> = None;
    let mut click_result: Option<EventRef> = None;
    let hover_in_timeline = hovered && mouse_pos[1] > tracks_top;
    let searching = search_mask.iter().any(|&m| m);
    let has_sel_mask = !sel_mask.is_empty() && sel_mask.iter().any(|&m| m);
    let filtering = searching || has_sel_mask;

    let active_sel = sel_change.unwrap_or(selection);
    // A LIVE drag tests events against raw pixel Y — the layout can't
    // meaningfully change mid-drag, so this is simple and exact. A FINISHED
    // selection instead looks up the frozen (track_idx, event_idx) set
    // captured when the drag ended (see Pane::capture_sel_events) — testing
    // finished_sel's raw Y range against the CURRENT layout here would
    // silently reassign the highlight to whatever tracks now occupy those
    // pixels after a Show CPU toggle, track reorder, or height change.
    let active_bounds: Option<(f64, f64, f32, f32)> = active_sel.map(|[s0, s1, y0, y1]| {
        let (sa, sb) = if s0 <= s1 { (s0, s1) } else { (s1, s0) };
        let (ya, yb) = if y0 <= y1 { (y0 as f32, y1 as f32) } else { (y1 as f32, y0 as f32) };
        (sa, sb, ya, yb)
    });
    let is_selected = |ti: u32, ei: u32, ev_top: f32, sub_h: f32, ts: f64, dur: f64| -> bool {
        if let Some((sa, sb, ya, yb)) = active_bounds {
            return ts + dur >= sa && ts <= sb && ev_top + sub_h >= ya && ev_top <= yb;
        }
        finished_sel_events.contains(&(ti, ei))
    };

    dl.with_clip_rect([tl_left, tracks_top], [rect[2], rect[3]], || {
        let interval = nice_interval(view.t1 - view.t0);
        if interval > 0.0 {
            let first = (view.t0 / interval).floor() * interval;
            let mut tick = first;
            let mut count = 0;
            while tick <= view.t1 && count < 500 {
                let x = t2x(tick, view.t0, px_per_us, tl_left);
                if x > tl_left && x < rect[2] {
                    dl.add_line([x, tracks_top], [x, rect[3]], GRID).build();
                }
                tick += interval;
                count += 1;
            }
        }

        for vi in 0..buf.visible.len() {
            let track_h = buf.heights[vi];
            let y = tracks_top + buf.y_offsets[vi] - view.scroll_y;
            if y + track_h < tracks_top || y > rect[3] { continue; }

            let bg = if vi % 2 == 0 { ROW_BG_A } else { ROW_BG_B };
            dl.add_rect([rect[0], y], [rect[2], y + track_h], bg).filled(true).build();

            let merged_group = buf.merged_gpu_groups.iter().find(|g| g.vi == vi);

            if let Some(group) = merged_group {
                let total_depth = group.max_depth;
                let sub_h = track_h / total_depth as f32;
                let lane_h = sub_h - LANE_GAP;
                buf.last_px.clear();
                buf.last_px.resize(total_depth as usize, -1i32);

                for &(ti32, ei32, eff_depth) in &group.events {
                    let orig_ti = ti32 as usize;
                    let ei = ei32 as usize;
                    let ev = &trace.tracks[orig_ti].events[ei];
                    let x0 = t2x(ev.ts, view.t0, px_per_us, tl_left).max(tl_left);
                    let x1 = t2x(ev.ts + ev.dur, view.t0, px_per_us, tl_left).min(rect[2]);
                    let w = x1 - x0;

                    let matches = !filtering
                        || (searching && (ev.name as usize) < search_mask.len() && search_mask[ev.name as usize])
                        || (has_sel_mask && sel_mask.get(ev.name as usize).copied().unwrap_or(false));

                    if w < MIN_EV_PX {
                        let px = x0 as i32;
                        if px == buf.last_px[eff_depth as usize] { continue; }
                        buf.last_px[eff_depth as usize] = px;
                        let ev_y = y + eff_depth as f32 * sub_h + EV_INSET;
                        let color = if matches {
                            name_color(&trace.names[ev.name as usize])
                        } else {
                            dim_color(&trace.names[ev.name as usize])
                        };
                        dl.add_rect([x0, ev_y], [x0 + 1.0, ev_y + lane_h], color).filled(true).build();
                        continue;
                    }

                    let ev_y = y + eff_depth as f32 * sub_h + EV_INSET;
                    let name = &trace.names[ev.name as usize];
                    let color = if matches {
                        name_color(name)
                    } else {
                        dim_color(name)
                    };
                    let ev_rect = [x0, ev_y, x1, ev_y + lane_h];

                    let is_hovered = hover_in_timeline
                        && mouse_pos[0] >= ev_rect[0] && mouse_pos[0] <= ev_rect[2]
                        && mouse_pos[1] >= ev_rect[1] && mouse_pos[1] <= ev_rect[3];

                    let is_primary = selected.map_or(false, |s| s.track_idx == ti32 && s.event_idx == ei32);
                    let is_multi = multi_select_name.map_or(false, |n| ev.name == n);
                    let ev_track_y = buf.y_offsets[vi] + eff_depth as f32 * sub_h;
                    let is_selected = is_selected(ti32, ei32, ev_track_y, sub_h, ev.ts, ev.dur);
                    let is_sel_mask = !sel_mask.is_empty() && sel_mask.get(ev.name as usize).copied().unwrap_or(false);

                    let fill = if is_hovered { brighten(color, 30) } else if is_selected || is_sel_mask { brighten(color, 20) } else { color };
                    dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], fill)
                        .filled(true).rounding(EV_ROUNDING).build();

                    if is_primary {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 220, 50, 255))
                            .rounding(EV_ROUNDING).build();
                    } else if is_hovered {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 255, 255, 255))
                            .rounding(EV_ROUNDING).build();
                    } else if is_selected || is_sel_mask {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], ACCENT_SOFT)
                            .rounding(EV_ROUNDING).build();
                    } else if is_multi {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 220, 50, 140))
                            .rounding(EV_ROUNDING).build();
                    } else if searching && matches {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], ACCENT_SOFT)
                            .rounding(EV_ROUNDING).build();
                    }

                    if is_hovered {
                        let r = EventRef { track_idx: ti32, event_idx: ei32 };
                        let prefer = hover_result.map_or(true, |prev| {
                            let prev_ev = &trace.tracks[prev.track_idx as usize].events[prev.event_idx as usize];
                            ev.depth > prev_ev.depth || (ev.depth == prev_ev.depth && ev.dur < prev_ev.dur)
                        });
                        if prefer {
                            if clicked && !shift { click_result = Some(r); }
                            hover_result = Some(r);
                        }
                    }

                    if w > TEXT_MIN_PX {
                        let tx = ev_rect[0] + 3.0;
                        let ty = ev_rect[1] + 2.0;
                        let text_col = if matches { col32(240, 240, 240, 255) } else { col32(120, 120, 120, 255) };
                        draw_text_wrapped(text_col, name, [tx, ty], w - 6.0, ev_rect);
                    }
                }
            } else {
                let orig_ti = buf.visible[vi];
                let track = &trace.tracks[orig_ti];
                let is_collapsed = collapsed.get(orig_ti).copied().unwrap_or(false);
                let sub_h = track_h / track.max_depth.max(1) as f32;
                let lane_h = sub_h - LANE_GAP;
                let start = bisect_overlap(&track.events, &track.prefix_max_dur, view.t0);
                let end = track.events.partition_point(|e| e.ts <= view.t1);

                buf.last_px.clear();
                buf.last_px.resize(track.max_depth as usize, -1i32);

                for ei in start..end {
                    let ev = &track.events[ei];
                    if is_collapsed && ev.depth > 0 { continue; }
                    if hidden_names.get(ev.name as usize).copied().unwrap_or(false) { continue; }
                    let x0 = t2x(ev.ts, view.t0, px_per_us, tl_left).max(tl_left);
                    let x1 = t2x(ev.ts + ev.dur, view.t0, px_per_us, tl_left).min(rect[2]);
                    let w = x1 - x0;

                    let matches = !filtering
                        || (searching && (ev.name as usize) < search_mask.len() && search_mask[ev.name as usize])
                        || (has_sel_mask && sel_mask.get(ev.name as usize).copied().unwrap_or(false));

                    if w < MIN_EV_PX {
                        let px = x0 as i32;
                        if px == buf.last_px[ev.depth as usize] { continue; }
                        buf.last_px[ev.depth as usize] = px;
                        let ev_y = y + ev.depth as f32 * sub_h + EV_INSET;
                        let color = if matches {
                            name_color(&trace.names[ev.name as usize])
                        } else {
                            dim_color(&trace.names[ev.name as usize])
                        };
                        dl.add_rect([x0, ev_y], [x0 + 1.0, ev_y + lane_h], color).filled(true).build();
                        continue;
                    }

                    let ev_y = y + ev.depth as f32 * sub_h + EV_INSET;
                    let name = &trace.names[ev.name as usize];
                    let color = if matches {
                        name_color(name)
                    } else {
                        dim_color(name)
                    };
                    let ev_rect = [x0, ev_y, x1, ev_y + lane_h];

                    let is_hovered = hover_in_timeline
                        && mouse_pos[0] >= ev_rect[0] && mouse_pos[0] <= ev_rect[2]
                        && mouse_pos[1] >= ev_rect[1] && mouse_pos[1] <= ev_rect[3];

                    let is_primary = selected.map_or(false, |s| s.track_idx == orig_ti as u32 && s.event_idx == ei as u32);
                    let is_multi = multi_select_name.map_or(false, |n| ev.name == n);
                    let ev_track_y = buf.y_offsets[vi] + ev.depth as f32 * sub_h;
                    let is_selected = is_selected(orig_ti as u32, ei as u32, ev_track_y, sub_h, ev.ts, ev.dur);
                    let is_sel_mask = !sel_mask.is_empty() && sel_mask.get(ev.name as usize).copied().unwrap_or(false);

                    let fill = if is_hovered { brighten(color, 30) } else if is_selected || is_sel_mask { brighten(color, 20) } else { color };
                    dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], fill)
                        .filled(true).rounding(EV_ROUNDING).build();

                    if is_primary {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 220, 50, 255))
                            .rounding(EV_ROUNDING).build();
                    } else if is_hovered {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 255, 255, 255))
                            .rounding(EV_ROUNDING).build();
                    } else if is_selected || is_sel_mask {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], ACCENT_SOFT)
                            .rounding(EV_ROUNDING).build();
                    } else if is_multi {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 220, 50, 140))
                            .rounding(EV_ROUNDING).build();
                    } else if searching && matches {
                        dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], ACCENT_SOFT)
                            .rounding(EV_ROUNDING).build();
                    }

                    if is_hovered {
                        let r = EventRef { track_idx: orig_ti as u32, event_idx: ei as u32 };
                        let prefer = hover_result.map_or(true, |prev| {
                            let prev_ev = &trace.tracks[prev.track_idx as usize].events[prev.event_idx as usize];
                            ev.depth > prev_ev.depth || (ev.depth == prev_ev.depth && ev.dur < prev_ev.dur)
                        });
                        if prefer {
                            if clicked && !shift { click_result = Some(r); }
                            hover_result = Some(r);
                        }
                    }

                    if w > TEXT_MIN_PX {
                        let tx = ev_rect[0] + 3.0;
                        let ty = ev_rect[1] + 2.0;
                        let text_col = if matches { col32(240, 240, 240, 255) } else { col32(120, 120, 120, 255) };
                        draw_text_wrapped(text_col, name, [tx, ty], w - 6.0, ev_rect);
                    }
                }
            }
        }

        if !trace.flow_pairs.is_empty() && show_cpu {
            if let Some(sel) = selected {
                let sel_ti = sel.track_idx as usize;
                let sel_track = &trace.tracks[sel_ti];
                let sel_ev = sel_track.events[sel.event_idx as usize];
                let ti32 = sel_ti as u32;

                let mut flow_start = usize::MAX;
                let mut flow_end = 0;
                let mut cur_ei = sel.event_idx as usize;
                loop {
                    let ev = &sel_track.events[cur_ei];
                    let idx = trace.flow_pairs.partition_point(|f|
                        (f.src_track, f.src_ts) < (ti32, ev.ts - 0.001));
                    let mut found = false;
                    for k in idx..trace.flow_pairs.len() {
                        let f = &trace.flow_pairs[k];
                        if f.src_track != ti32 || f.src_ts > ev.ts + ev.dur + 0.001 { break; }
                        if f.src_ts >= ev.ts - 0.001 {
                            if !found { flow_start = k; found = true; }
                            flow_end = k + 1;
                        }
                    }
                    if found { break; }
                    if ev.depth == 0 { break; }
                    let target_depth = ev.depth - 1;
                    let mut parent = None;
                    for i in (0..cur_ei).rev() {
                        let pe = &sel_track.events[i];
                        if pe.depth == target_depth && pe.ts <= ev.ts && pe.ts + pe.dur >= ev.ts + ev.dur {
                            parent = Some(i);
                            break;
                        }
                        if pe.depth < target_depth { break; }
                    }
                    match parent { Some(p) => cur_ei = p, None => break }
                }

                if flow_start < flow_end {
                    let find_vi_and_depth = |ti: usize, ei: usize, depth: u16| -> Option<(usize, u16)> {
                        if let Some(vi) = buf.visible.iter().position(|&v| v == ti) {
                            return Some((vi, depth));
                        }
                        for g in &buf.merged_gpu_groups {
                            if g.tracks.contains(&ti) {
                                let md = g.events.iter()
                                    .find(|&&(t, e, _)| t == ti as u32 && e == ei as u32)
                                    .map(|&(_, _, d)| d)
                                    .unwrap_or(0);
                                return Some((g.vi, md));
                            }
                        }
                        None
                    };
                    let sel_loc = find_vi_and_depth(sel_ti, sel.event_idx as usize, sel_ev.depth);
                    if let Some((sel_vi, sel_eff_depth)) = sel_loc {
                        let sel_gpu = sel_track.gpu;
                        let total_depth: u16 = buf.merged_gpu_groups.iter()
                            .find(|g| g.vi == sel_vi)
                            .map_or(sel_track.max_depth.max(1), |g| g.max_depth);
                        let src_sub_h = buf.heights[sel_vi] / total_depth as f32;
                        let src_lane_h = src_sub_h - LANE_GAP;
                        let src_y = tracks_top + buf.y_offsets[sel_vi] - view.scroll_y
                            + sel_eff_depth as f32 * src_sub_h + EV_INSET + src_lane_h / 2.0;
                        let src_x = if sel_gpu {
                            t2x(sel_ev.ts, view.t0, px_per_us, tl_left)
                        } else {
                            t2x(sel_ev.ts + sel_ev.dur, view.t0, px_per_us, tl_left)
                        };

                        let arrow_col = col32(255, 180, 50, 200);
                        for fi in flow_start..flow_end {
                            let fp = &trace.flow_pairs[fi];
                            if fp.src_track != ti32 { continue; }
                            let dst_ti = fp.dst_track as usize;
                            let dst_evs = &trace.tracks[dst_ti].events;
                            let p = dst_evs.partition_point(|e| e.ts < fp.dst_ts - 0.001);
                            let mut dst_ei_found = None;
                            for k in p..dst_evs.len().min(p + 10) {
                                if (dst_evs[k].ts - fp.dst_ts).abs() < 0.001 { dst_ei_found = Some(k); break; }
                            }
                            let dst_ei = match dst_ei_found { Some(v) => v, None => continue };
                            let dst_ev = dst_evs[dst_ei];

                            let (dst_x, dst_y) = if let Some((dst_vi, dst_eff_depth)) = find_vi_and_depth(dst_ti, dst_ei, dst_ev.depth) {
                                let dst_total: u16 = buf.merged_gpu_groups.iter()
                                    .find(|g| g.vi == dst_vi)
                                    .map_or(trace.tracks[dst_ti].max_depth.max(1), |g| g.max_depth);
                                let dst_sub_h = buf.heights[dst_vi] / dst_total as f32;
                                let dst_lane_h = dst_sub_h - LANE_GAP;
                                let dy = tracks_top + buf.y_offsets[dst_vi] - view.scroll_y
                                    + dst_eff_depth as f32 * dst_sub_h + EV_INSET + dst_lane_h / 2.0;
                                let dx = if sel_gpu {
                                    t2x(dst_ev.ts + dst_ev.dur, view.t0, px_per_us, tl_left)
                                } else {
                                    t2x(dst_ev.ts, view.t0, px_per_us, tl_left)
                                };
                                (dx, dy)
                            } else {
                                let dx = if sel_gpu {
                                    t2x(dst_ev.ts + dst_ev.dur, view.t0, px_per_us, tl_left)
                                } else {
                                    t2x(dst_ev.ts, view.t0, px_per_us, tl_left)
                                };
                                (dx, rect[3])
                            };

                            dl.add_line([src_x, src_y], [dst_x, dst_y], arrow_col).thickness(2.0).build();

                            let dx = dst_x - src_x;
                            let dy = dst_y - src_y;
                            let len = (dx * dx + dy * dy).sqrt();
                            if len > 1.0 {
                                let ux = dx / len;
                                let uy = dy / len;
                                let arrow_size = 8.0f32;
                                let left = [dst_x - ux * arrow_size - uy * arrow_size * 0.5,
                                            dst_y - uy * arrow_size + ux * arrow_size * 0.5];
                                let right = [dst_x - ux * arrow_size + uy * arrow_size * 0.5,
                                             dst_y - uy * arrow_size - ux * arrow_size * 0.5];
                                dl.add_triangle([dst_x, dst_y], left, right, arrow_col).filled(true).build();
                            }
                        }
                    }
                }
            }
        }

        if let Some([s0, s1, y0, y1]) = active_sel {
            let (sa, sb) = if s0 <= s1 { (s0, s1) } else { (s1, s0) };
            let (ya, yb) = if y0 <= y1 { (y0 as f32, y1 as f32) } else { (y1 as f32, y0 as f32) };
            let sx0 = t2x(sa, view.t0, px_per_us, tl_left).max(tl_left);
            let sx1 = t2x(sb, view.t0, px_per_us, tl_left).min(rect[2]);
            let sy0 = (tracks_top + ya - view.scroll_y).max(tracks_top);
            let sy1 = (tracks_top + yb - view.scroll_y).min(rect[3]);
            if sx1 > sx0 && sy1 > sy0 {
                dl.add_rect([sx0, sy0], [sx1, sy1], ACCENT_FILL)
                    .filled(true).build();
                dl.add_line([sx0, sy0], [sx0, sy1], ACCENT_LINE).build();
                dl.add_line([sx1, sy0], [sx1, sy1], ACCENT_LINE).build();
                dl.add_line([sx0, sy0], [sx1, sy0], ACCENT_LINE).build();
                dl.add_line([sx0, sy1], [sx1, sy1], ACCENT_LINE).build();
                buf.fmt.clear();
                write_time(&mut buf.fmt, sb - sa);
                let text_sz = ui.calc_text_size(&buf.fmt);
                let tx = ((sx0 + sx1) / 2.0 - text_sz[0] / 2.0).max(sx0 + 2.0);
                let ty = sy0.max(tracks_top) + 2.0;
                let pad = 3.0;
                dl.add_rect([tx - pad, ty - 1.0], [tx + text_sz[0] + pad, ty + text_sz[1] + 1.0], col32(20, 20, 20, 220))
                    .filled(true).rounding(3.0).build();
                dl.add_rect([tx - pad, ty - 1.0], [tx + text_sz[0] + pad, ty + text_sz[1] + 1.0], ACCENT_SOFT)
                    .rounding(3.0).build();
                dl.add_text([tx, ty], col32(220, 230, 255, 255), &buf.fmt);
            }
        }
    });

    drop(dl);

    let win_pos = ui.window_pos();
    for vi in 0..buf.visible.len() {
        let orig_ti = buf.visible[vi];
        let track = &trace.tracks[orig_ti];
        let track_h = buf.heights[vi];
        let y = tracks_top + buf.y_offsets[vi] - view.scroll_y;
        if y + track_h < tracks_top || y > rect[3] { continue; }
        let vis_top = y.max(tracks_top);
        let vis_h = (y + track_h).min(rect[3]) - vis_top;
        if vis_h <= 0.0 { continue; }

        let label_area_w = tl_left - 4.0 - rect[0];
        ui.set_cursor_pos([rect[0] - win_pos[0], vis_top - win_pos[1]]);

        buf.fmt.clear();
        write!(buf.fmt, "##tl{vi}").ok();
        let _pad = ui.push_style_var(StyleVar::WindowPadding([2.0, 2.0]));
        if let Some(_child) = ui.child_window(&buf.fmt)
            .size([label_area_w, vis_h])
            .border(false)
            .flags(WindowFlags::NO_SCROLLBAR | WindowFlags::NO_SCROLL_WITH_MOUSE | WindowFlags::NO_BACKGROUND | WindowFlags::NO_INPUTS)
            .begin()
        {
            buf.fmt.clear();
            if let Some(g) = buf.merged_gpu_groups.iter().find(|g| g.vi == vi) {
                write!(buf.fmt, "{}", g.label).ok();
            } else {
                write!(buf.fmt, "{}", track.label).ok();
            }
            let text_h = ui.calc_text_size_with_opts(&buf.fmt, false, label_area_w - 4.0)[1];
            let pad_y = ((vis_h - text_h) * 0.5).max(0.0);
            ui.set_cursor_pos([ui.cursor_pos()[0], pad_y]);
            let _col = ui.push_style_color(StyleColor::Text, [0.82, 0.82, 0.82, 1.0]);
            ui.text_wrapped(&buf.fmt);
        }
        drop(_pad);
    }

    let dl = ui.get_window_draw_list();

    // Track separator lines + resize drag
    let mut near_border = false;
    let mut hovered_border_y: Option<f32> = None;
    for vi in 0..buf.visible.len() {
        let border_y = tracks_top + buf.y_offsets[vi] + buf.heights[vi] - view.scroll_y;
        if border_y < tracks_top || border_y > rect[3] { continue; }
        dl.add_line([rect[0], border_y], [rect[2], border_y], DIVIDER).build();

        // !shift: suppress resize affordance during shift-drag selection.
        // !*even_spacing: heights are auto-managed in even mode, so per-track
        // border resizing is disabled (double-click the last lane to exit).
        if !shift && !*even_spacing && !drag.is_active() && hovered && mouse_pos[1] > tracks_top {
            if (mouse_pos[1] - border_y).abs() < RESIZE_GRAB_H {
                hovered_border_y = Some(border_y);
                near_border = true;
                if clicked {
                    *drag = DragKind::TrackResize(pane_idx, buf.visible[vi]);
                }
            }
        }
    }
    if let Some(by) = hovered_border_y {
        dl.add_line([rect[0], by], [rect[2], by], col32(100, 180, 255, 200)).thickness(2.0).build();
        ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
    }
    if let DragKind::TrackResize(pi, ti) = *drag {
        if pi == pane_idx {
            ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
            if let Some(vi) = buf.visible.iter().position(|&v| v == ti) {
                let by = tracks_top + buf.y_offsets[vi] + buf.heights[vi] - view.scroll_y;
                dl.add_line([rect[0], by], [rect[2], by], col32(100, 180, 255, 200)).thickness(2.0).build();
            }
        }
    }

    if clicked && !near_border && !drag.is_active() && hovered && mouse_pos[0] < tl_left && mouse_pos[1] > tracks_top {
        for vi in 0..buf.visible.len() {
            let track_h = buf.heights[vi];
            let y = tracks_top + buf.y_offsets[vi] - view.scroll_y;
            if mouse_pos[1] >= y && mouse_pos[1] < y + track_h {
                let grab_off = mouse_pos[1] - y;
                *drag = DragKind::TrackDrag(pane_idx, vi, grab_off);
                break;
            }
        }
    }

    if let DragKind::TrackDrag(pi, dragged_vi, _) = *drag {
        if pi == pane_idx {
            ui.set_mouse_cursor(Some(imgui::MouseCursor::Hand));
            let y = tracks_top + buf.y_offsets[dragged_vi] - view.scroll_y;
            let h = buf.heights[dragged_vi];
            dl.add_rect(
                [rect[0], y.max(tracks_top)],
                [tl_left - 4.0, (y + h).min(rect[3])],
                col32(100, 180, 255, 80),
            ).filled(true).build();
        }
    }

    if ui.is_mouse_double_clicked(imgui::MouseButton::Left) && !drag.is_active() && hovered && mouse_pos[0] < tl_left && mouse_pos[1] > tracks_top {
        for vi in 0..buf.visible.len() {
            let orig_ti = buf.visible[vi];
            let track_h = buf.heights[vi];
            let y = tracks_top + buf.y_offsets[vi] - view.scroll_y;
            if mouse_pos[1] >= y && mouse_pos[1] < y + track_h {
                if trace.tracks[orig_ti].max_depth > 1 {
                    if let Some(c) = collapsed.get_mut(orig_ti) {
                        *c = !*c;
                    }
                }
                break;
            }
        }
    }

    // Double-click the last lane (or the empty space beneath it) to toggle
    // "even spacing": every visible track shares the viewport height equally and
    // re-flows live as the viewport changes (e.g. dragging the bottom-pane
    // divider). The mode overrides the drawn heights without touching
    // track_scales, so double-clicking again restores the manual layout — an
    // easy undo. Handy for the merged multi-rank view, where the few rank rows
    // otherwise leave a lot of dead space. Guarded by hover_result.is_none() so
    // double-clicking an event still multi-selects, and by x > tl_left so it
    // never collides with the label-column collapse toggle.
    if ui.is_mouse_double_clicked(imgui::MouseButton::Left)
        && !drag.is_active()
        && hovered
        && hover_result.is_none()
        && mouse_pos[0] > tl_left
        && mouse_pos[0] <= rect[2]
    {
        let n = buf.visible.len();
        if n > 0 {
            let last_top = tracks_top + buf.y_offsets[n - 1] - view.scroll_y;
            if mouse_pos[1] >= last_top && mouse_pos[1] <= rect[3] {
                *even_spacing = !*even_spacing;
            }
        }
    }

    (hover_result, click_result, sel_change)
}

fn draw_ruler(dl: &imgui::DrawListMut, rect: [f32; 4], view: &View, fmt: &mut String) {
    dl.add_rect([rect[0], rect[1]], [rect[2], rect[3]], RULER_BG)
        .filled(true).build();
    let range = view.t1 - view.t0;
    let interval = nice_interval(range);
    if interval <= 0.0 { return; }
    let first = (view.t0 / interval).floor() * interval;
    let px_per_us = (rect[2] - rect[0]) as f64 / range;
    let mut tick = first;
    let mut count = 0;
    while tick <= view.t1 && count < 500 {
        let x = rect[0] + ((tick - view.t0) * px_per_us) as f32;
        if x >= rect[0] && x <= rect[2] {
            dl.add_line([x, rect[1]], [x, rect[3]], RULER_TICK).build();
            fmt.clear();
            write_time(fmt, tick);
            dl.add_text([x + 3.0, rect[1] + 4.0], RULER_TEXT, &*fmt);
        }
        tick += interval;
        count += 1;
    }
}

pub fn nice_interval(range: f64) -> f64 {
    let rough = range / 8.0;
    if rough <= 0.0 { return 1.0; }
    let mag = 10f64.powf(rough.log10().floor());
    let r = rough / mag;
    let nice = if r <= 1.5 { 1.0 } else if r <= 3.5 { 2.0 } else if r <= 7.5 { 5.0 } else { 10.0 };
    nice * mag
}

pub fn write_time(buf: &mut String, us: f64) {
    let abs = us.abs();
    if abs == 0.0 {
        buf.push('0');
    } else if abs < 1.0 {
        let _ = write!(buf, "{:.0}ns", us * 1000.0);
    } else if abs < 1000.0 {
        let _ = write!(buf, "{:.1}us", us);
    } else if abs < 1_000_000.0 {
        let _ = write!(buf, "{:.2}ms", us / 1000.0);
    } else {
        let _ = write!(buf, "{:.3}s", us / 1_000_000.0);
    }
}

pub fn draw_diff_popup(
    ui: &imgui::Ui,
    diff: &DiffResult,
    buf: &mut DrawBuf,
    _display: [f32; 2],
    name_a: &str,
    name_b: &str,
    bar_scroll: f64,
    bar_zoom: f64,
) -> (f64, f64) {
    buf.fmt.clear();
    write!(buf.fmt, "{}: {} events, ", name_a, diff.count_a).unwrap();
    write_time(&mut buf.fmt, diff.total_dur_a);
    write!(buf.fmt, "  |  {}: {} events, ", name_b, diff.count_b).unwrap();
    write_time(&mut buf.fmt, diff.total_dur_b);
    if diff.total_dur_a > 0.0 {
        let pct = (diff.total_dur_b - diff.total_dur_a) / diff.total_dur_a * 100.0;
        write!(buf.fmt, "  |  Delta: {:+.1}%", pct).unwrap();
    }
    ui.text(&buf.fmt);
    ui.separator();

    let avail = ui.content_region_avail();
    let font_h = ui.current_font_size();
    let label_h = font_h + 2.0;
    let style = ui.clone_style();
    let spacing = style.item_spacing[1];
    let bar_h = DIFF_BAR_H;
    let bar_gap = DIFF_BAR_GAP;
    let bars_reserve = label_h + bar_h + bar_gap + label_h + bar_h + DIFF_BAR_GAP + spacing * 2.0 + EV_INSET;
    let char_w = ui.calc_text_size("M")[0];
    let dur_w = 10.0 * char_w;
    let delta_w = 10.0 * char_w;
    let mid_w = 3.0 * char_w;
    let half_w = (avail[0] - mid_w - delta_w) / 2.0;
    let name_w = half_w - dur_w;
    let left_name = 0.0;
    let left_dur = name_w;
    let mid = half_w;
    let right_name = mid + mid_w;
    let right_dur = right_name + name_w;
    let delta_x = right_dur + dur_w;

    let header_col = [0.55, 0.55, 0.55, 1.0];
    let cx0 = ui.window_content_region_min()[0];
    ui.set_cursor_pos([ui.cursor_pos()[0] + left_name, ui.cursor_pos()[1]]);
    ui.text_colored(header_col, name_a);
    ui.same_line_with_pos(left_dur + cx0);
    ui.text_colored(header_col, "Dur");
    ui.same_line_with_pos(right_name + cx0);
    ui.text_colored(header_col, name_b);
    ui.same_line_with_pos(right_dur + cx0);
    ui.text_colored(header_col, "Dur");
    ui.same_line_with_pos(delta_x + cx0);
    ui.text_colored(header_col, "Delta");
    ui.separator();

    ui.child_window("##diffscroll")
        .size([avail[0], -bars_reserve])
        .build(|| {
            let row_h = ui.current_font_size() + ROW_PAD;
            let total_rows = diff.lines.len();
            let scroll_y = ui.scroll_y();
            let content_h = ui.content_region_avail()[1];
            let first = (scroll_y / row_h) as usize;
            let visible = (content_h / row_h) as usize + 2;
            let last = total_rows.min(first + visible);
            let cx = ui.window_content_region_min()[0];

            if first > 0 {
                ui.dummy([0.0, first as f32 * row_h]);
            }

            let dl = ui.get_window_draw_list();
            let win_pos = ui.window_pos();

            let name_chars_adj = ((name_w - SWATCH_PAD - 8.0) / char_w).max(4.0) as usize;
            let text_col = [0.85, 0.85, 0.85, 1.0];
            let dur_col = [0.7, 0.7, 0.7, 1.0];
            let dim_col = [0.35, 0.35, 0.35, 1.0];

            for i in first..last {
                let line = &diff.lines[i];
                let row_y = ui.cursor_screen_pos()[1];
                let swatch_color = name_color(&line.name);
                let has_left = line.kind == DiffKind::Same || line.kind == DiffKind::Removed;
                let has_right = line.kind == DiffKind::Same || line.kind == DiffKind::Added;

                // Left side
                ui.set_cursor_pos([cx + left_name, ui.cursor_pos()[1]]);
                if has_left {
                    let sx = win_pos[0] + cx + left_name;
                    dl.add_rect(
                        [sx, row_y + EV_INSET], [sx + SWATCH_W, row_y + row_h - EV_INSET],
                        swatch_color,
                    ).filled(true).rounding(EV_ROUNDING).build();
                    ui.dummy([SWATCH_PAD, 1.0]);
                    ui.same_line();
                    truncated_text(ui, buf, &line.name, name_chars_adj, text_col);
                    ui.same_line_with_pos(cx + left_dur);
                    buf.fmt.clear();
                    write_time(&mut buf.fmt, line.dur_a.unwrap_or(0.0));
                    ui.text_colored(dur_col, &buf.fmt);
                } else {
                    ui.text_colored(dim_col, "");
                }

                // Separator
                ui.same_line_with_pos(cx + mid);
                ui.text_colored([0.4, 0.4, 0.4, 1.0], "|");

                // Right side
                if has_right {
                    ui.same_line_with_pos(cx + right_name);
                    let sx = win_pos[0] + cx + right_name;
                    dl.add_rect(
                        [sx, row_y + EV_INSET], [sx + SWATCH_W, row_y + row_h - EV_INSET],
                        swatch_color,
                    ).filled(true).rounding(EV_ROUNDING).build();
                    ui.dummy([SWATCH_PAD, 1.0]);
                    ui.same_line();
                    truncated_text(ui, buf, &line.name, name_chars_adj, text_col);
                    ui.same_line_with_pos(cx + right_dur);
                    buf.fmt.clear();
                    write_time(&mut buf.fmt, line.dur_b.unwrap_or(0.0));
                    ui.text_colored(dur_col, &buf.fmt);

                    // Delta
                    if has_left {
                        let da = line.dur_a.unwrap_or(0.0);
                        let db = line.dur_b.unwrap_or(0.0);
                        if da > 0.0 {
                            ui.same_line_with_pos(cx + delta_x);
                            let pct = (db - da) / da * 100.0;
                            buf.fmt.clear();
                            write!(buf.fmt, "{:+.1}%", pct).unwrap();
                            let col = if pct < -1.0 { [0.3, 0.9, 0.3, 1.0] }
                                else if pct > 1.0 { [0.9, 0.3, 0.3, 1.0] }
                                else { [0.5, 0.5, 0.5, 1.0] };
                            ui.text_colored(col, &buf.fmt);
                        }
                    }
                }
            }

            if last < total_rows {
                ui.dummy([0.0, (total_rows - last) as f32 * row_h]);
            }

            let _ = dl;
        });

    ui.separator();
    let (new_scroll, new_zoom) = draw_diff_bars(ui, diff, buf, avail[0], name_a, name_b, bar_scroll, bar_zoom);
    (new_scroll, new_zoom)
}

fn draw_diff_bars(
    ui: &imgui::Ui, diff: &DiffResult, buf: &mut DrawBuf, width: f32,
    name_a: &str, name_b: &str, mut scroll: f64, mut zoom: f64,
) -> (f64, f64) {
    let bar_h = DIFF_BAR_H;
    let gap = DIFF_BAR_GAP;
    let label_h = ui.current_font_size() + EV_INSET;
    let cursor = ui.cursor_screen_pos();

    let total_h = label_h + bar_h + gap + label_h + bar_h + DIFF_BAR_GAP;
    let bar_x = cursor[0];
    let bar_w = width.max(1.0);

    struct Segment { start: usize, end: usize, dur_a: f64, dur_b: f64 }
    let mut segments: Vec<Segment> = Vec::new();
    let mut i = 0;
    let lines = &diff.lines;
    while i < lines.len() {
        if lines[i].kind == DiffKind::Same {
            segments.push(Segment {
                start: i, end: i + 1,
                dur_a: lines[i].dur_a.unwrap_or(0.0),
                dur_b: lines[i].dur_b.unwrap_or(0.0),
            });
            i += 1;
        } else {
            let start = i;
            let mut da = 0.0;
            let mut db = 0.0;
            while i < lines.len() && lines[i].kind != DiffKind::Same {
                match lines[i].kind {
                    DiffKind::Removed => da += lines[i].dur_a.unwrap_or(0.0),
                    DiffKind::Added => db += lines[i].dur_b.unwrap_or(0.0),
                    _ => {}
                }
                i += 1;
            }
            segments.push(Segment { start, end: i, dur_a: da, dur_b: db });
        }
    }

    let total_budget: f64 = segments.iter().map(|s| s.dur_a.max(s.dur_b)).sum();
    if total_budget <= 0.0 {
        ui.dummy([width, total_h]);
        return (scroll, zoom);
    }

    let seg_gap = DIFF_SEG_GAP;

    ui.invisible_button("##diffbar_area", [width, total_h]);
    let hovered = ui.is_item_hovered();

    if hovered {
        let wheel = ui.io().mouse_wheel;
        if wheel != 0.0 {
            let mouse_x = ui.io().mouse_pos[0];
            let frac = ((mouse_x - bar_x) as f64 / bar_w as f64).clamp(0.0, 1.0);
            let world_at_mouse = scroll + frac / zoom;
            let factor = if wheel > 0.0 { ZOOM_STEP } else { 1.0 / ZOOM_STEP };
            zoom = (zoom * factor).clamp(1.0, MAX_ZOOM);
            scroll = world_at_mouse - frac / zoom;
        }
        unsafe {
            let io = imgui_sys::igGetIO();
            (*io).MouseWheel = 0.0;
            (*io).MouseWheelH = 0.0;
        }
    }
    if hovered && ui.is_mouse_dragging(imgui::MouseButton::Left) {
        let dx = ui.io().mouse_delta[0];
        scroll -= dx as f64 / (bar_w as f64 * zoom);
    }
    let max_scroll = 1.0 - 1.0 / zoom;
    scroll = scroll.clamp(0.0, max_scroll.max(0.0));

    let label_a_y = cursor[1];
    let top_y = label_a_y + label_h;
    let label_b_y = top_y + bar_h + gap;
    let bot_y = label_b_y + label_h;

    let mouse = ui.io().mouse_pos;
    let mut hover_name: Option<(String, f64)> = None;

    let dl = ui.get_window_draw_list();
    dl.add_text([cursor[0], label_a_y], col32(160, 160, 160, 255), name_a);
    dl.add_text([cursor[0], label_b_y], col32(160, 160, 160, 255), name_b);
    dl.with_clip_rect([bar_x, cursor[1]], [bar_x + bar_w, cursor[1] + total_h], || {

    let zoomed_w = bar_w as f64 * zoom;
    let n_gaps = if segments.len() > 1 { segments.len() - 1 } else { 0 };
    let usable_zoomed = zoomed_w - n_gaps as f64 * seg_gap as f64;

    let mut vx = -(scroll * zoomed_w) as f32;
    for (si, seg) in segments.iter().enumerate() {
        let seg_w = (seg.dur_a.max(seg.dur_b) / total_budget * usable_zoomed) as f32;
        let x0 = bar_x + vx;
        let x1 = x0 + seg_w;

        if x1 >= bar_x && x0 <= bar_x + bar_w && seg_w >= 0.5 {
            let mut xa = x0;
            let mut xb = x0;
            let max_dur = seg.dur_a.max(seg.dur_b);
            for li in seg.start..seg.end {
                let line = &lines[li];
                let color = name_color(&line.name);
                if let Some(da) = line.dur_a {
                    if line.kind != DiffKind::Added {
                        let w = (da / max_dur * seg_w as f64) as f32;
                        if w >= 0.5 {
                            dl.add_rect([xa, top_y], [xa + w, top_y + bar_h], color).filled(true).build();
                            if w > 1.0 { dl.add_line([xa, top_y], [xa, top_y + bar_h], col32(20, 20, 20, 180)).build(); }
                            let char_w = ui.calc_text_size("M")[0];
                            if w > char_w * 4.0 {
                                buf.fmt.clear();
                                buf.fmt.push_str(&line.name);
                                buf.fmt.push_str("  ");
                                write_time(&mut buf.fmt, da);
                                let clip = [xa, top_y, xa + w, top_y + bar_h];
                                draw_text_clipped(col32(240, 240, 240, 255), &buf.fmt, [xa + 3.0, top_y + 3.0], clip);
                            }
                            if hovered && mouse[0] >= xa && mouse[0] < xa + w && mouse[1] >= top_y && mouse[1] < top_y + bar_h {
                                hover_name = Some((line.name.clone(), da));
                            }
                        }
                        xa += w;
                    }
                }
                if let Some(db) = line.dur_b {
                    if line.kind != DiffKind::Removed {
                        let w = (db / max_dur * seg_w as f64) as f32;
                        if w >= 0.5 {
                            dl.add_rect([xb, bot_y], [xb + w, bot_y + bar_h], color).filled(true).build();
                            if w > 1.0 { dl.add_line([xb, bot_y], [xb, bot_y + bar_h], col32(20, 20, 20, 180)).build(); }
                            let char_w = ui.calc_text_size("M")[0];
                            if w > char_w * 4.0 {
                                buf.fmt.clear();
                                buf.fmt.push_str(&line.name);
                                buf.fmt.push_str("  ");
                                write_time(&mut buf.fmt, db);
                                let clip = [xb, bot_y, xb + w, bot_y + bar_h];
                                draw_text_clipped(col32(240, 240, 240, 255), &buf.fmt, [xb + 3.0, bot_y + 3.0], clip);
                            }
                            if hovered && mouse[0] >= xb && mouse[0] < xb + w && mouse[1] >= bot_y && mouse[1] < bot_y + bar_h {
                                hover_name = Some((line.name.clone(), db));
                            }
                        }
                        xb += w;
                    }
                }
            }
        }

        vx += seg_w;
        if si + 1 < segments.len() {
            vx += seg_gap;
        }
    }

    }); // clip rect
    drop(dl);

    if let Some((name, dur)) = hover_name {
        buf.fmt.clear();
        buf.fmt.push_str(&name);
        buf.fmt.push_str("  ");
        write_time(&mut buf.fmt, dur);
        ui.tooltip_text(&buf.fmt);
    }

    (scroll, zoom)
}

fn truncated_text(ui: &imgui::Ui, buf: &mut DrawBuf, name: &str, max_chars: usize, col: [f32; 4]) {
    if name.len() > max_chars && max_chars > 3 {
        buf.fmt.clear();
        buf.fmt.push_str(&name[..max_chars - 3]);
        buf.fmt.push_str("...");
        ui.text_colored(col, &buf.fmt);
        if ui.is_item_hovered() {
            ui.tooltip_text(name);
        }
    } else {
        ui.text_colored(col, name);
    }
}

/// Side length of the logical box `draw_vllm_logo`'s vertex coordinates are
/// normalized to — exposed so callers can compute the logo's on-screen size
/// (`VLLM_LOGO_GRID * scale`) without duplicating that number.
pub const VLLM_LOGO_GRID: f32 = 32.0;

/// Exact vector geometry (not a rasterized approximation) of the official
/// mark, from `vLLM-Logo.svg` in `vllm-project/media-kit` (96x96 viewBox: a
/// yellow triangle and a blue convex quadrilateral). Coordinates are that
/// SVG's own path data, tightly cropped to the mark's bounding box (plus a
/// small margin) and rescaled so the box is `VLLM_LOGO_GRID` units square —
/// `scale` then maps 1 unit to 1 pixel, same convention the old pixel-grid
/// version used. `YELLOW`/`BLUE` are the exact fill colors from that SVG
/// (`#fdb515`/`#30a2ff`).
pub fn draw_vllm_logo(dl: &imgui::DrawListMut, x: f32, y: f32, scale: f32) {
    const YELLOW: ImColor32 = ImColor32::from_rgba(253, 181, 21, 255);
    const BLUE: ImColor32 = ImColor32::from_rgba(48, 162, 255, 255);
    const YELLOW_PTS: [[f32; 2]; 3] = [[13.417, 7.835], [13.417, 30.286], [2.191, 7.835]];
    const BLUE_PTS: [[f32; 2]; 4] =
        [[13.416, 30.286], [22.237, 30.286], [29.809, 1.714], [19.427, 7.179]];

    let tp = |p: [f32; 2]| [x + p[0] * scale, y + p[1] * scale];
    dl.add_triangle(tp(YELLOW_PTS[0]), tp(YELLOW_PTS[1]), tp(YELLOW_PTS[2]), YELLOW)
        .filled(true)
        .build();
    dl.add_polyline(BLUE_PTS.iter().map(|&p| tp(p)).collect::<Vec<_>>(), BLUE)
        .filled(true)
        .build();
}

/// Radius of the lock icon's shackle circle, in pixels.
pub const LOCK_ICON_RADIUS: f32 = 4.0;

/// Draws the bottom divider's "fit tracks to the visible height" indicator
/// (`Pane::even_spacing`, toggled by double-clicking the divider or this
/// icon — see main.rs): a shackle (circle outline) sitting on a body
/// (filled rounded rect), the classic padlock silhouette — the body's fill
/// occludes the circle's lower half, so only the top arc reads as a loop,
/// no arc-drawing primitive needed. Not itself an interactive element, just
/// a visual indicator. `fit` controls both color (accent when on, gray when
/// off) and the shackle's vertical offset: flush against the body when on,
/// lifted with a visible gap when off — mirroring a closed vs. open padlock.
pub fn draw_lock_icon(dl: &imgui::DrawListMut, cx: f32, cy: f32, fit: bool) {
    let color = if fit { ACCENT_LINE } else { col32(120, 120, 120, 255) };
    let body_w = 11.0;
    let body_h = 8.0;
    let body_top = cy + 1.0;
    let shackle_gap = if fit { 0.0 } else { 2.5 };
    let shackle_cy = body_top - LOCK_ICON_RADIUS * 0.9 - shackle_gap;

    dl.add_circle([cx, shackle_cy], LOCK_ICON_RADIUS, color)
        .thickness(1.6)
        .num_segments(16)
        .build();
    dl.add_rect(
        [cx - body_w * 0.5, body_top],
        [cx + body_w * 0.5, body_top + body_h],
        color,
    )
    .filled(true)
    .rounding(1.5)
    .build();
}

pub fn device_name_color() -> ImColor32 {
    ImColor32::from_rgba(118, 185, 0, 255)
}

pub fn col32(r: u8, g: u8, b: u8, a: u8) -> ImColor32 {
    ImColor32::from_rgba(r, g, b, a)
}

/// How much further each channel is pushed from the color's own max channel
/// (i.e. away from gray, at constant hue/value) — a cheap saturation boost
/// that needs no HSV round-trip. 1.0 leaves `PALETTE` (a deliberately muted
/// Tableau-style set) unchanged; higher values pull the three channels
/// apart, making different tracks/kernels easier to tell apart at a glance.
const SATURATION_BOOST: f32 = 1.15;

fn boost_saturation(r: u8, g: u8, b: u8, factor: f32) -> (u8, u8, u8) {
    let max = r.max(g).max(b) as f32;
    if max == 0.0 { return (r, g, b); }
    let push = |c: u8| (max - (max - c as f32) * factor).clamp(0.0, 255.0) as u8;
    (push(r), push(g), push(b))
}

pub fn palette_color(name: &str, brightness: u32) -> ImColor32 {
    let h = fnv1a(name.as_bytes()) as usize;
    let c = PALETTE[h % PALETTE.len()];
    let (pr, pg, pb) = boost_saturation(
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8) & 0xFF) as u8,
        (c & 0xFF) as u8,
        SATURATION_BOOST,
    );
    let r = pr as u32 * brightness / 255;
    let g = pg as u32 * brightness / 255;
    let b = pb as u32 * brightness / 255;
    ImColor32::from_rgba(r as u8, g as u8, b as u8, 255)
}

pub fn name_color(name: &str) -> ImColor32 { palette_color(name, 155) }
pub fn dim_color(name: &str) -> ImColor32 { palette_color(name, 85) }

pub fn brighten(c: ImColor32, amt: u8) -> ImColor32 {
    let v: u32 = c.into();
    let r = (v & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let b = ((v >> 16) & 0xFF) as u8;
    let a = ((v >> 24) & 0xFF) as u8;
    ImColor32::from_rgba(r.saturating_add(amt), g.saturating_add(amt), b.saturating_add(amt), a)
}

// Cohesive dark theme applied once at init. Replaces imgui's stock defaults so
// the chrome (toolbar, buttons, dropdowns, scrollbars, popups, tables) reads as
// one intentional surface: consistent rounding, quieter borders, and a single
// accent (the logo blue) instead of imgui's default blue. Shared between
// renderer.rs (native/Metal) and renderer_web.rs (wasm/WebGL2) — those two
// modules are mutually exclusive per build, so this lives here instead of
// duplicated in both, and both call it right after building the font atlas.
pub fn apply_style(style: &mut imgui::Style) {
    use imgui::StyleColor as C;

    // Geometry: gentle, consistent rounding + calmer borders + a bit of air.
    style.window_rounding = 6.0;
    style.child_rounding = 4.0;
    style.frame_rounding = 4.0;
    style.popup_rounding = 4.0;
    style.grab_rounding = 3.0;
    style.scrollbar_rounding = 4.0;
    style.tab_rounding = 4.0;
    style.window_border_size = 1.0;
    style.child_border_size = 1.0;
    style.frame_border_size = 0.0;
    style.popup_border_size = 1.0;
    style.frame_padding = [8.0, 4.0];
    // Leave item_spacing at imgui's default [8, 4]: the hand-rolled virtualized
    // tables (draw_stats_table, labels table) assume a row pitch of
    // font_size + ROW_PAD (4) == default item_spacing.y. Bumping it opens a gap
    // after the header and desyncs the scroll virtualization.
    style.item_spacing = [8.0, 4.0];
    style.item_inner_spacing = [6.0, 4.0];
    style.scrollbar_size = 12.0;
    style.grab_min_size = 10.0;

    // One accent, expressed at a few alphas.
    let accent = [0.188, 0.635, 1.0, 1.0];
    let accent_a = |a: f32| [accent[0], accent[1], accent[2], a];

    let c = style.colors.as_mut_slice();
    c[C::Text as usize] = [1.0, 1.0, 1.0, 1.0];
    c[C::TextDisabled as usize] = [0.5, 0.5, 0.5, 1.0];
    c[C::WindowBg as usize] = [0.086, 0.086, 0.086, 1.0];
    c[C::ChildBg as usize] = [0.0, 0.0, 0.0, 0.0];
    c[C::PopupBg as usize] = [0.08, 0.08, 0.08, 0.98];
    c[C::Border as usize] = [0.24, 0.24, 0.24, 0.5];
    c[C::BorderShadow as usize] = [0.0, 0.0, 0.0, 0.0];
    c[C::FrameBg as usize] = [0.16, 0.16, 0.16, 1.0];
    c[C::FrameBgHovered as usize] = [0.22, 0.22, 0.22, 1.0];
    c[C::FrameBgActive as usize] = [0.26, 0.26, 0.26, 1.0];
    c[C::TitleBg as usize] = [0.10, 0.10, 0.10, 1.0];
    c[C::TitleBgActive as usize] = [0.12, 0.12, 0.12, 1.0];
    c[C::TitleBgCollapsed as usize] = [0.08, 0.08, 0.08, 1.0];
    c[C::MenuBarBg as usize] = [0.12, 0.12, 0.12, 1.0];
    c[C::ScrollbarBg as usize] = [0.0, 0.0, 0.0, 0.0];
    c[C::ScrollbarGrab as usize] = [0.30, 0.30, 0.30, 1.0];
    c[C::ScrollbarGrabHovered as usize] = [0.38, 0.38, 0.38, 1.0];
    c[C::ScrollbarGrabActive as usize] = [0.46, 0.46, 0.46, 1.0];
    c[C::CheckMark as usize] = accent;
    c[C::SliderGrab as usize] = accent_a(0.85);
    c[C::SliderGrabActive as usize] = accent;
    c[C::Button as usize] = [0.18, 0.18, 0.18, 1.0];
    c[C::ButtonHovered as usize] = [0.26, 0.26, 0.26, 1.0];
    c[C::ButtonActive as usize] = accent_a(0.55);
    c[C::Header as usize] = [0.20, 0.20, 0.20, 1.0];
    c[C::HeaderHovered as usize] = [0.26, 0.26, 0.26, 1.0];
    c[C::HeaderActive as usize] = accent_a(0.45);
    c[C::Separator as usize] = [0.24, 0.24, 0.24, 1.0];
    c[C::SeparatorHovered as usize] = accent_a(0.55);
    c[C::SeparatorActive as usize] = accent;
    c[C::ResizeGrip as usize] = [0.26, 0.26, 0.26, 0.0];
    c[C::ResizeGripHovered as usize] = accent_a(0.5);
    c[C::ResizeGripActive as usize] = accent_a(0.85);
    c[C::Tab as usize] = [0.14, 0.14, 0.14, 1.0];
    c[C::TabHovered as usize] = accent_a(0.55);
    c[C::TabActive as usize] = [0.22, 0.22, 0.22, 1.0];
    c[C::TabUnfocused as usize] = [0.11, 0.11, 0.11, 1.0];
    c[C::TabUnfocusedActive as usize] = [0.16, 0.16, 0.16, 1.0];
    c[C::TextSelectedBg as usize] = accent_a(0.35);
    c[C::NavHighlight as usize] = accent;
}
