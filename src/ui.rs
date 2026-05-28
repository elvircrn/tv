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

pub fn draw_stats_table(
    ui: &imgui::Ui,
    trace: &Trace,
    stats: &[KernelStats],
    search: &mut String,
    search_changed: &mut bool,
    sort_col: &mut usize,
    sort_asc: &mut bool,
    buf: &mut DrawBuf,
) {
    let avail = ui.content_region_avail();
    ui.child_window("##statstable")
        .size([avail[0], avail[1]])
        .build(|| {
            let col_w = STATS_COL_W;
            let swatch_w = 14.0;
            let name_w = avail[0] - col_w * 6.0 - swatch_w - 16.0;
            let headers = ["Name", "Count", "Total", "%", "Mean", "Median", "Max"];
            let positions = [swatch_w, swatch_w + name_w, swatch_w + name_w + col_w, swatch_w + name_w + col_w * 2.0, swatch_w + name_w + col_w * 3.0, swatch_w + name_w + col_w * 4.0, swatch_w + name_w + col_w * 5.0];
            for (ci, &label) in headers.iter().enumerate() {
                if ci > 0 { ui.same_line_with_pos(positions[ci]); }
                else { ui.set_cursor_pos([positions[0], ui.cursor_pos()[1]]); }
                let arrow = if *sort_col == ci { if *sort_asc { " ^" } else { " v" } } else { "" };
                let col = if *sort_col == ci { [0.85, 0.85, 0.55, 1.0] } else { [0.55, 0.55, 0.55, 1.0] };
                buf.fmt.clear();
                write!(buf.fmt, "{label}{arrow}").unwrap();
                ui.text_colored(col, &buf.fmt);
                if ui.is_item_clicked() {
                    if *sort_col == ci { *sort_asc = !*sort_asc; }
                    else { *sort_col = ci; *sort_asc = ci == 0; }
                }
            }
            ui.separator();

            buf.sort_idx.clear();
            buf.sort_idx.extend(0..stats.len());
            let total_sum: f64 = stats.iter().map(|s| s.total_dur).sum();
            buf.sort_idx.sort_by(|&a, &b| {
                let sa = &stats[a];
                let sb = &stats[b];
                let avg = |s: &KernelStats| if s.count > 0 { s.total_dur / s.count as f64 } else { 0.0 };
                let pct = |s: &KernelStats| if total_sum > 0.0 { s.total_dur / total_sum } else { 0.0 };
                let ord = match *sort_col {
                    0 => trace.names[sa.name as usize].cmp(&trace.names[sb.name as usize]),
                    1 => sa.count.cmp(&sb.count),
                    2 => sa.total_dur.partial_cmp(&sb.total_dur).unwrap(),
                    3 => pct(sa).partial_cmp(&pct(sb)).unwrap(),
                    4 => avg(sa).partial_cmp(&avg(sb)).unwrap(),
                    5 => sa.median_dur.partial_cmp(&sb.median_dur).unwrap(),
                    _ => sa.max_dur.partial_cmp(&sb.max_dur).unwrap(),
                };
                if *sort_asc { ord } else { ord.reverse() }
            });

            let row_h = ui.current_font_size() + ROW_PAD;
            let total_rows = buf.sort_idx.len();
            let scroll_y = ui.scroll_y();
            let content_h = ui.content_region_avail()[1];
            let first = (scroll_y / row_h) as usize;
            let visible = (content_h / row_h) as usize + 2;
            let last = total_rows.min(first + visible);

            if first > 0 {
                ui.dummy([0.0, first as f32 * row_h]);
            }

            let dl = ui.get_window_draw_list();
            let char_w = ui.calc_text_size("M")[0];
            let max_name_chars = ((name_w - 8.0) / char_w) as usize;

            for i in first..last {
                let si = buf.sort_idx[i];
                let s = &stats[si];
                let name = &trace.names[s.name as usize];
                let avg = if s.count > 0 { s.total_dur / s.count as f64 } else { 0.0 };
                let pct = if total_sum > 0.0 { s.total_dur / total_sum * 100.0 } else { 0.0 };
                let text_color = if i % 2 == 0 { [0.85, 0.85, 0.85, 1.0] } else { [0.75, 0.75, 0.75, 1.0] };

                let cursor = ui.cursor_screen_pos();
                let swatch_color = name_color(name);
                dl.add_rect([cursor[0], cursor[1] + EV_INSET], [cursor[0] + SWATCH_W, cursor[1] + row_h - EV_INSET], swatch_color)
                    .filled(true).rounding(EV_ROUNDING).build();

                ui.set_cursor_pos([positions[0], ui.cursor_pos()[1]]);
                if name.len() > max_name_chars && max_name_chars > 3 {
                    buf.fmt.clear();
                    buf.fmt.push_str(&name[..max_name_chars - 3]);
                    buf.fmt.push_str("...");
                    ui.text_colored(text_color, &buf.fmt);
                } else {
                    ui.text_colored(text_color, name);
                }
                if ui.is_item_hovered() && name.len() > max_name_chars {
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
                ui.same_line_with_pos(positions[1]);
                buf.fmt.clear();
                write!(buf.fmt, "{}", s.count).unwrap();
                ui.text_colored(text_color, &buf.fmt);
                ui.same_line_with_pos(positions[2]);
                buf.fmt.clear();
                write_time(&mut buf.fmt, s.total_dur);
                ui.text_colored(text_color, &buf.fmt);
                ui.same_line_with_pos(positions[3]);
                buf.fmt.clear();
                write!(buf.fmt, "{pct:.1}%").unwrap();
                ui.text_colored(text_color, &buf.fmt);
                ui.same_line_with_pos(positions[4]);
                buf.fmt.clear();
                write_time(&mut buf.fmt, avg);
                ui.text_colored(text_color, &buf.fmt);
                ui.same_line_with_pos(positions[5]);
                buf.fmt.clear();
                write_time(&mut buf.fmt, s.median_dur);
                ui.text_colored(text_color, &buf.fmt);
                ui.same_line_with_pos(positions[6]);
                buf.fmt.clear();
                write_time(&mut buf.fmt, s.max_dur);
                ui.text_colored(text_color, &buf.fmt);
            }

            if last < total_rows {
                ui.dummy([0.0, (total_rows - last) as f32 * row_h]);
            }
        });
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

#[allow(clippy::too_many_arguments)]
pub fn draw_timeline(
    ui: &imgui::Ui,
    trace: &Trace,
    view: &mut View,
    show_cpu: bool,
    buf: &mut DrawBuf,
    rect: [f32; 4],
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
    finished_sel: Option<[f64; 4]>,
    collapsed: &mut Vec<bool>,
    event_labels: &[Vec<Option<u8>>],
    labels: &[Label],
    hidden_names: &[bool],
    selected: Option<EventRef>,
    multi_select_name: Option<u32>,
    sel_mask: &[bool],
    label_w: f32,
    track_scales: &mut Vec<f32>,
    drag: &mut DragKind,
) -> (Option<EventRef>, Option<EventRef>, Option<Option<[f64; 4]>>) {
    let dl = ui.get_window_draw_list();
    let tl_left = rect[0] + label_w;
    let tl_w = (rect[2] - tl_left).max(1.0);

    if let DragKind::TrackResize(ti) = *drag {
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

    buf.visible.clear();
    buf.heights.clear();
    buf.y_offsets.clear();
    let mut cumulative = 0.0f32;
    for (i, t) in trace.tracks.iter().enumerate() {
        if !show_cpu && !t.gpu { continue; }
        buf.visible.push(i);
        let h = track_height(
            t.max_depth,
            collapsed.get(i).copied().unwrap_or(false),
            track_scales.get(i).copied().unwrap_or(1.0),
        );
        buf.heights.push(h);
        buf.y_offsets.push(cumulative);
        cumulative += h;
    }
    let total_h = cumulative;

    let time_range = (view.t1 - view.t0).max(1e-9);
    let px_per_us = tl_w as f64 / time_range;

    #[inline]
    fn t2x(t: f64, t0: f64, ppus: f64, left: f32) -> f32 { left + ((t - t0) * ppus) as f32 }
    #[inline]
    fn x2t(x: f32, t0: f64, ppus: f64, left: f32) -> f64 { t0 + (x - left) as f64 / ppus }

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

    let bg = col32(24, 24, 24, 255);
    dl.add_rect([rect[0], rect[1]], [rect[2], rect[3]], bg).filled(true).build();

    let ruler_rect = [tl_left, rect[1], rect[2], rect[1] + RULER_H];
    draw_ruler(&dl, ruler_rect, view, &mut buf.fmt);

    let lbg = col32(20, 20, 20, 255);
    dl.add_rect([rect[0], rect[1]], [tl_left, rect[3]], lbg).filled(true).build();
    dl.add_line([tl_left, rect[1]], [tl_left, rect[3]], col32(50, 50, 50, 255)).build();
    dl.add_line([rect[0], rect[1] + RULER_H], [rect[2], rect[1] + RULER_H], col32(50, 50, 50, 255)).build();

    let tracks_top = rect[1] + RULER_H;
    let mut hover_result: Option<EventRef> = None;
    let mut click_result: Option<EventRef> = None;
    let hover_in_timeline = hovered && mouse_pos[1] > tracks_top;
    let searching = search_mask.iter().any(|&m| m);
    let has_sel_mask = !sel_mask.is_empty() && sel_mask.iter().any(|&m| m);
    let filtering = searching || has_sel_mask;

    let active_sel = sel_change.unwrap_or(selection);
    let highlight_sel = active_sel.or(finished_sel);
    let sel_bounds: Option<(f64, f64, f32, f32)> = highlight_sel.map(|[s0, s1, y0, y1]| {
        let (sa, sb) = if s0 <= s1 { (s0, s1) } else { (s1, s0) };
        let (ya, yb) = if y0 <= y1 { (y0 as f32, y1 as f32) } else { (y1 as f32, y0 as f32) };
        (sa, sb, ya, yb)
    });

    dl.with_clip_rect([tl_left, tracks_top], [rect[2], rect[3]], || {
        let interval = nice_interval(view.t1 - view.t0);
        if interval > 0.0 {
            let first = (view.t0 / interval).floor() * interval;
            let mut tick = first;
            let mut count = 0;
            while tick <= view.t1 && count < 500 {
                let x = t2x(tick, view.t0, px_per_us, tl_left);
                if x > tl_left && x < rect[2] {
                    dl.add_line([x, tracks_top], [x, rect[3]], col32(40, 40, 40, 255)).build();
                }
                tick += interval;
                count += 1;
            }
        }

        for vi in 0..buf.visible.len() {
            let orig_ti = buf.visible[vi];
            let track = &trace.tracks[orig_ti];
            let track_h = buf.heights[vi];
            let is_collapsed = collapsed.get(orig_ti).copied().unwrap_or(false);
            let y = tracks_top + buf.y_offsets[vi] - view.scroll_y;
            if y + track_h < tracks_top || y > rect[3] { continue; }

            let bg = if vi % 2 == 0 { col32(28, 28, 28, 255) } else { col32(32, 32, 32, 255) };
            dl.add_rect([rect[0], y], [rect[2], y + track_h], bg).filled(true).build();

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
                        event_color(orig_ti, ei, &trace.names[ev.name as usize], event_labels, labels)
                    } else {
                        event_dim_color(orig_ti, ei, &trace.names[ev.name as usize], event_labels, labels)
                    };
                    dl.add_rect([x0, ev_y], [x0 + 1.0, ev_y + lane_h], color).filled(true).build();
                    continue;
                }

                let ev_y = y + ev.depth as f32 * sub_h + EV_INSET;
                let name = &trace.names[ev.name as usize];
                let color = if matches {
                    event_color(orig_ti, ei, name, event_labels, labels)
                } else {
                    event_dim_color(orig_ti, ei, name, event_labels, labels)
                };
                let ev_rect = [x0, ev_y, x1, ev_y + lane_h];

                let is_hovered = hover_in_timeline
                    && mouse_pos[0] >= ev_rect[0] && mouse_pos[0] <= ev_rect[2]
                    && mouse_pos[1] >= ev_rect[1] && mouse_pos[1] <= ev_rect[3];

                let is_primary = selected.map_or(false, |s| s.track_idx == orig_ti as u32 && s.event_idx == ei as u32);
                let is_multi = multi_select_name.map_or(false, |n| ev.name == n);
                let is_selected = sel_bounds.map_or(false, |(sa, sb, ya, yb)| {
                    let ev_track_y = buf.y_offsets[vi] + ev.depth as f32 * sub_h;
                    let ev_bot = ev_track_y + sub_h;
                    ev.ts + ev.dur >= sa && ev.ts <= sb && ev_bot >= ya && ev_track_y <= yb
                });
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
                    dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(100, 180, 255, 180))
                        .rounding(EV_ROUNDING).build();
                } else if is_multi {
                    dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(255, 220, 50, 140))
                        .rounding(EV_ROUNDING).build();
                } else if searching && matches {
                    dl.add_rect([ev_rect[0], ev_rect[1]], [ev_rect[2], ev_rect[3]], col32(100, 180, 255, 180))
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

        if let Some([s0, s1, y0, y1]) = active_sel {
            let (sa, sb) = if s0 <= s1 { (s0, s1) } else { (s1, s0) };
            let (ya, yb) = if y0 <= y1 { (y0 as f32, y1 as f32) } else { (y1 as f32, y0 as f32) };
            let sx0 = t2x(sa, view.t0, px_per_us, tl_left).max(tl_left);
            let sx1 = t2x(sb, view.t0, px_per_us, tl_left).min(rect[2]);
            let sy0 = (tracks_top + ya - view.scroll_y).max(tracks_top);
            let sy1 = (tracks_top + yb - view.scroll_y).min(rect[3]);
            if sx1 > sx0 && sy1 > sy0 {
                dl.add_rect([sx0, sy0], [sx1, sy1], ImColor32::from_rgba(60, 130, 220, 40))
                    .filled(true).build();
                dl.add_line([sx0, sy0], [sx0, sy1], ImColor32::from_rgba(80, 160, 255, 200)).build();
                dl.add_line([sx1, sy0], [sx1, sy1], ImColor32::from_rgba(80, 160, 255, 200)).build();
                dl.add_line([sx0, sy0], [sx1, sy0], ImColor32::from_rgba(80, 160, 255, 200)).build();
                dl.add_line([sx0, sy1], [sx1, sy1], ImColor32::from_rgba(80, 160, 255, 200)).build();
                buf.fmt.clear();
                write_time(&mut buf.fmt, sb - sa);
                let text_sz = ui.calc_text_size(&buf.fmt);
                let tx = ((sx0 + sx1) / 2.0 - text_sz[0] / 2.0).max(sx0 + 2.0);
                let ty = sy0.max(tracks_top) + 2.0;
                let pad = 3.0;
                dl.add_rect([tx - pad, ty - 1.0], [tx + text_sz[0] + pad, ty + text_sz[1] + 1.0], col32(20, 20, 20, 220))
                    .filled(true).rounding(3.0).build();
                dl.add_rect([tx - pad, ty - 1.0], [tx + text_sz[0] + pad, ty + text_sz[1] + 1.0], col32(80, 160, 255, 180))
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
            let is_collapsed = collapsed.get(orig_ti).copied().unwrap_or(false);
            let indicator = if track.max_depth <= 1 { " " } else if is_collapsed { ">" } else { "v" };

            buf.fmt.clear();
            write!(buf.fmt, "{indicator}  {}", track.label).ok();
            let _col = ui.push_style_color(StyleColor::Text, [0.67, 0.67, 0.67, 1.0]);
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
        dl.add_line([rect[0], border_y], [rect[2], border_y], col32(50, 50, 50, 255)).build();

        // !shift: suppress resize affordance during shift-drag selection
        if !shift && !drag.is_active() && hovered && mouse_pos[1] > tracks_top {
            if (mouse_pos[1] - border_y).abs() < RESIZE_GRAB_H {
                hovered_border_y = Some(border_y);
                near_border = true;
                if clicked {
                    *drag = DragKind::TrackResize(buf.visible[vi]);
                }
            }
        }
    }
    if let Some(by) = hovered_border_y {
        dl.add_line([rect[0], by], [rect[2], by], col32(100, 180, 255, 200)).thickness(2.0).build();
        ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
    }
    if let DragKind::TrackResize(ti) = *drag {
        ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
        if let Some(vi) = buf.visible.iter().position(|&v| v == ti) {
            let by = tracks_top + buf.y_offsets[vi] + buf.heights[vi] - view.scroll_y;
            dl.add_line([rect[0], by], [rect[2], by], col32(100, 180, 255, 200)).thickness(2.0).build();
        }
    }

    if clicked && !near_border && !drag.is_active() && hovered && mouse_pos[0] < tl_left && mouse_pos[1] > tracks_top {
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

    (hover_result, click_result, sel_change)
}

fn draw_ruler(dl: &imgui::DrawListMut, rect: [f32; 4], view: &View, fmt: &mut String) {
    dl.add_rect([rect[0], rect[1]], [rect[2], rect[3]], col32(18, 18, 18, 255))
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
            dl.add_line([x, rect[1]], [x, rect[3]], col32(60, 60, 60, 255)).build();
            fmt.clear();
            write_time(fmt, tick);
            dl.add_text([x + 3.0, rect[1] + 4.0], col32(160, 160, 160, 255), &*fmt);
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

pub fn draw_vllm_logo(dl: &imgui::DrawListMut, x: f32, y: f32, scale: f32) {
    const ICON: [[u8; 16]; 16] = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2],
        [0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0],
        [1,1,1,1,1,1,0,0,0,0,2,2,2,2,2,0],
        [0,1,1,1,1,1,0,0,0,0,2,2,2,2,2,0],
        [0,1,1,1,1,1,0,0,0,2,2,2,2,2,0,0],
        [0,0,1,1,1,1,0,0,0,2,2,2,2,2,0,0],
        [0,0,1,1,1,1,0,0,0,2,2,2,2,2,0,0],
        [0,0,0,1,1,1,0,0,0,2,2,2,2,2,0,0],
        [0,0,0,1,1,1,0,0,2,2,2,2,2,0,0,0],
        [0,0,0,0,1,1,0,0,2,2,2,2,2,0,0,0],
        [0,0,0,0,1,1,0,0,2,2,2,2,2,0,0,0],
        [0,0,0,0,0,1,0,2,2,2,2,2,2,0,0,0],
        [0,0,0,0,0,1,0,2,2,2,2,2,0,0,0,0],
        [0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,0],
        [0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,0],
    ];
    const YELLOW: ImColor32 = ImColor32::from_rgba(253, 181, 21, 255);
    const BLUE: ImColor32 = ImColor32::from_rgba(48, 162, 255, 255);
    for (r, row) in ICON.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v == 0 { continue; }
            let color = if v == 1 { YELLOW } else { BLUE };
            let px = x + c as f32 * scale;
            let py = y + r as f32 * scale;
            dl.add_rect([px, py], [px + scale, py + scale], color).filled(true).build();
        }
    }
}

pub fn device_name_color() -> ImColor32 {
    ImColor32::from_rgba(118, 185, 0, 255)
}

pub fn col32(r: u8, g: u8, b: u8, a: u8) -> ImColor32 {
    ImColor32::from_rgba(r, g, b, a)
}

pub fn event_color(track_idx: usize, event_idx: usize, name: &str, event_labels: &[Vec<Option<u8>>], labels: &[Label]) -> ImColor32 {
    if let Some(li) = event_labels.get(track_idx).and_then(|t| t.get(event_idx)).copied().flatten() {
        labels[li as usize].color
    } else {
        name_color(name)
    }
}

pub fn event_dim_color(track_idx: usize, event_idx: usize, name: &str, event_labels: &[Vec<Option<u8>>], labels: &[Label]) -> ImColor32 {
    if let Some(li) = event_labels.get(track_idx).and_then(|t| t.get(event_idx)).copied().flatten() {
        let c: u32 = labels[li as usize].color.into();
        let r = (c & 0xFF) as u32 * 77 / 255;
        let g = ((c >> 8) & 0xFF) as u32 * 77 / 255;
        let b = ((c >> 16) & 0xFF) as u32 * 77 / 255;
        ImColor32::from_rgba(r as u8, g as u8, b as u8, 255)
    } else {
        dim_color(name)
    }
}

pub fn palette_color(name: &str, brightness: u32) -> ImColor32 {
    let h = fnv1a(name.as_bytes()) as usize;
    let c = PALETTE[h % PALETTE.len()];
    let r = ((c >> 16) & 0xFF) * brightness / 255;
    let g = ((c >> 8) & 0xFF) * brightness / 255;
    let b = (c & 0xFF) * brightness / 255;
    ImColor32::from_rgba(r as u8, g as u8, b as u8, 255)
}

pub fn name_color(name: &str) -> ImColor32 { palette_color(name, 140) }
pub fn dim_color(name: &str) -> ImColor32 { palette_color(name, 77) }

pub fn brighten(c: ImColor32, amt: u8) -> ImColor32 {
    let v: u32 = c.into();
    let r = (v & 0xFF) as u8;
    let g = ((v >> 8) & 0xFF) as u8;
    let b = ((v >> 16) & 0xFF) as u8;
    let a = ((v >> 24) & 0xFF) as u8;
    ImColor32::from_rgba(r.saturating_add(amt), g.saturating_add(amt), b.saturating_add(amt), a)
}
