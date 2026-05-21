mod parse;
mod types;
mod renderer;
mod loader;
mod state;
mod ui;
mod diff;

use types::*;
use renderer::MetalRenderer;
use state::*;
use ui::*;

use imgui::{Condition, InputTextCallback, InputTextCallbackHandler, StyleVar, TextCallbackData, WindowFlags};
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

extern "C" {
    fn objc_autoreleasePoolPush() -> *mut std::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut std::ffi::c_void);
}

struct App {
    window: Option<Window>,
    imgui: Option<imgui::Context>,
    renderer: Option<MetalRenderer>,
    state: AppState,
    last_frame: Instant,
    scale_factor: f64,
    scroll_accum: [f32; 2],
    pinch_accum: f32,
    mod_super: bool,
    mod_ctrl: bool,
    mod_shift: bool,
    pending_files: Vec<String>,
    last_mouse_x: f32,
}

impl App {
    fn new(files: Vec<String>) -> Self {
        Self {
            window: None,
            imgui: None,
            renderer: None,
            state: AppState {
                panes: [Pane::new(), Pane::new()],
                active: 0,
                split: false,
                split_x: 0.0,
                buf: DrawBuf::default(),
                bottom_h: DETAIL_H,
                label_w: LABEL_W,
                drag: DragKind::None,
                show_diff: false,
                diff_popup_open: false,
                diff_result: None,
                diff_bar_scroll: 0.0,
                diff_bar_zoom: 1.0,
            },
            last_frame: Instant::now(),
            scale_factor: 1.0,
            scroll_accum: [0.0; 2],
            pinch_accum: 0.0,
            mod_super: false,
            mod_ctrl: false,
            mod_shift: false,
            pending_files: files,
            last_mouse_x: 0.0,
        }
    }
}

struct MacClipboard;
impl imgui::ClipboardBackend for MacClipboard {
    fn get(&mut self) -> Option<String> {
        std::process::Command::new("pbpaste")
            .output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
    }
    fn set(&mut self, value: &str) {
        if let Ok(mut child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped()).spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(value.as_bytes()).ok();
            }
            child.wait().ok();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Trace Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 800.0));
        let window = event_loop.create_window(attrs).unwrap();
        self.scale_factor = window.scale_factor();

        let mut imgui = imgui::Context::create();
        imgui.io_mut().config_mac_os_behaviors = false;
        imgui.set_clipboard_backend(MacClipboard);
        let renderer = MetalRenderer::new(&window, &mut imgui, self.scale_factor);

        self.window = Some(window);
        self.imgui = Some(imgui);
        self.renderer = Some(renderer);

        if !self.pending_files.is_empty() {
            self.state.panes[0].open(self.pending_files.remove(0));
            if !self.pending_files.is_empty() {
                self.state.panes[1].open(self.pending_files.remove(0));
                self.state.split = true;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _wid: WindowId, event: WindowEvent) {
        let imgui = match self.imgui.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let io = imgui.io_mut();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &self.renderer {
                    r.resize(size.width, size.height);
                }
                let s = self.scale_factor as f32;
                io.display_size = [size.width as f32 / s, size.height as f32 / s];
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor;
                io.display_framebuffer_scale = [scale_factor as f32; 2];
            }

            WindowEvent::CursorMoved { position, .. } => {
                let s = self.scale_factor as f32;
                let x = position.x as f32 / s;
                io.add_mouse_pos_event([x, position.y as f32 / s]);
                self.last_mouse_x = x;
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => imgui::MouseButton::Left,
                    winit::event::MouseButton::Right => imgui::MouseButton::Right,
                    winit::event::MouseButton::Middle => imgui::MouseButton::Middle,
                    _ => return,
                };
                io.add_mouse_button_event(btn, state == ElementState::Pressed);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let (h, v) = match delta {
                    MouseScrollDelta::LineDelta(h, v) => (h * 20.0, v * 20.0),
                    MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                self.scroll_accum[0] += h;
                self.scroll_accum[1] += v;
                io.add_mouse_wheel_event([h / 20.0, v / 20.0]);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = winit_to_imgui(code) {
                        io.add_key_event(key, event.state == ElementState::Pressed);
                    }
                    if event.state == ElementState::Pressed && self.mod_ctrl {
                        if code == KeyCode::KeyA {
                            self.state.panes[self.state.active].select_all_pending = true;
                        }
                    }
                }
                if event.state == ElementState::Pressed && !self.mod_super && !self.mod_ctrl {
                    if let Some(text) = &event.text {
                        for ch in text.chars() {
                            if ch >= ' ' && ch != '\x7f' {
                                io.add_input_character(ch);
                            }
                        }
                    }
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                let s = mods.state();
                self.mod_ctrl = s.control_key();
                self.mod_super = s.super_key();
                self.mod_shift = s.shift_key();
                io.add_key_event(imgui::Key::LeftCtrl, s.control_key());
                io.add_key_event(imgui::Key::LeftShift, s.shift_key());
                io.add_key_event(imgui::Key::LeftAlt, s.alt_key());
                io.add_key_event(imgui::Key::LeftSuper, s.super_key());
            }

            WindowEvent::Focused(_) => {}

            WindowEvent::PinchGesture { delta, .. } => {
                self.pinch_accum += delta as f32;
            }

            WindowEvent::DroppedFile(path) => {
                let path_str: String = path.to_string_lossy().into();
                let pane0_busy = self.state.panes[0].has_trace() || self.state.panes[0].loading.is_some();
                if !self.state.split && pane0_busy {
                    self.state.split = true;
                    self.state.active = 1;
                    self.state.panes[1].open(path_str);
                } else if self.state.split {
                    let target = if self.last_mouse_x < self.state.split_x { 0 } else { 1 };
                    self.state.active = target;
                    self.state.panes[target].open(path_str);
                } else {
                    self.state.panes[0].open(path_str);
                }
            }

            WindowEvent::RedrawRequested => {
                let pool = unsafe { objc_autoreleasePoolPush() };
                self.render_frame();
                unsafe { objc_autoreleasePoolPop(pool); }
                return;
            }

            _ => return,
        }

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.panes[0].loading.is_some() || self.state.panes[1].loading.is_some() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        }
    }
}

impl App {
    fn render_frame(&mut self) {
        let imgui = match self.imgui.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let window = self.window.as_ref().unwrap();
        let phys = window.inner_size();
        let s = self.scale_factor as f32;

        let io = imgui.io_mut();
        io.display_size = [phys.width as f32 / s, phys.height as f32 / s];
        io.display_framebuffer_scale = [s, s];
        let now = Instant::now();
        let dt = self.last_frame.elapsed().as_secs_f32().max(0.0001);
        io.delta_time = dt;
        self.last_frame = now;

        self.state.panes[0].poll_loading();
        self.state.panes[1].poll_loading();

        let scroll = self.scroll_accum;
        self.scroll_accum = [0.0; 2];
        let pinch = self.pinch_accum;
        self.pinch_accum = 0.0;
        let ctrl = self.mod_ctrl || self.mod_super;
        let shift = self.mod_shift;

        let ui = imgui.new_frame();
        let display = ui.io().display_size;
        let mouse_pos = ui.io().mouse_pos;
        let mouse_delta = ui.io().mouse_delta;

        let state = &mut self.state;

        if state.split && state.split_x < 1.0 {
            state.split_x = display[0] * 0.5;
        }

        let n_panes = if state.split { 2 } else { 1 };
        let pane_xs: [f32; 2] = if state.split { [0.0, state.split_x] } else { [0.0, 0.0] };
        let pane_ws: [f32; 2] = if state.split { [state.split_x, display[0] - state.split_x] } else { [display[0], 0.0] };

        let any_has_trace = (0..n_panes).any(|pi| state.panes[pi].has_trace());
        let bottom_h = if any_has_trace { state.bottom_h } else { 0.0 };
        let status_h = if any_has_trace { STATUS_H } else { 0.0 };

        // ---- Drag handling (bottom divider, label divider, split divider) ----
        if any_has_trace && !state.diff_popup_open {
            let divider_y = display[1] - bottom_h - status_h;
            let near_h = (mouse_pos[1] - divider_y).abs() < 4.0 && mouse_pos[1] > TOOLBAR_H;

            if (near_h && !state.drag.is_active()) || state.drag == DragKind::BottomDivider {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeNS));
            }

            if ui.io().mouse_down[0] && !state.drag.is_active() && near_h {
                state.drag = DragKind::BottomDivider;
            }

            if ui.io().mouse_down[0] && state.drag == DragKind::BottomDivider {
                state.bottom_h -= mouse_delta[1];
                state.bottom_h = state.bottom_h.clamp(60.0, display[1] - TOOLBAR_H - status_h - 60.0);
            } else if state.drag == DragKind::BottomDivider && !ui.io().mouse_down[0] {
                state.drag = DragKind::None;
            }
        }

        // Label divider (check each pane)
        if !state.diff_popup_open {
            let divider_y = display[1] - bottom_h - status_h;
            let mut near_v = false;
            for pi in 0..n_panes {
                if !state.panes[pi].has_trace() { continue; }
                let label_x = pane_xs[pi] + state.label_w;
                let in_pane = mouse_pos[0] >= pane_xs[pi] && mouse_pos[0] < pane_xs[pi] + pane_ws[pi];
                if in_pane && (mouse_pos[0] - label_x).abs() < 4.0
                    && mouse_pos[1] > TOOLBAR_H && mouse_pos[1] < divider_y
                {
                    near_v = true;
                }
            }

            if (near_v && !state.drag.is_active()) || state.drag == DragKind::LabelDivider {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeEW));
            }
            if ui.io().mouse_down[0] && !state.drag.is_active() && near_v {
                state.drag = DragKind::LabelDivider;
            }
            if ui.io().mouse_down[0] && state.drag == DragKind::LabelDivider {
                state.label_w += mouse_delta[0];
                let max_w = if state.split { state.split_x * 0.5 } else { display[0] * 0.5 };
                state.label_w = state.label_w.clamp(60.0, max_w);
            } else if state.drag == DragKind::LabelDivider && !ui.io().mouse_down[0] {
                state.drag = DragKind::None;
            }
        }

        // Split divider
        if state.split && !state.diff_popup_open {
            let near_split = (mouse_pos[0] - state.split_x).abs() < 4.0
                && mouse_pos[1] > TOOLBAR_H;

            if (near_split && !state.drag.is_active()) || state.drag == DragKind::SplitDivider {
                ui.set_mouse_cursor(Some(imgui::MouseCursor::ResizeEW));
            }
            if ui.io().mouse_down[0] && !state.drag.is_active() && near_split {
                state.drag = DragKind::SplitDivider;
            }
            if ui.io().mouse_down[0] && state.drag == DragKind::SplitDivider {
                state.split_x += mouse_delta[0];
                state.split_x = state.split_x.clamp(200.0, display[0] - 200.0);
            } else if state.drag == DragKind::SplitDivider && !ui.io().mouse_down[0] {
                state.drag = DragKind::None;
            }
        }

        // ---- Per-pane toolbars ----
        let mut search_changed = [false; 2];
        let mut close_pane: Option<usize> = None;
        let mut diff_clicked = false;
        let can_diff = state.split
            && !state.panes[0].selection_stats.is_empty()
            && !state.panes[1].selection_stats.is_empty();
        let toolbar_names = ["##toolbar0", "##toolbar1"];
        for pi in 0..n_panes {
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 6.0]));
            ui.window(toolbar_names[pi])
                .position([pane_xs[pi], 0.0], Condition::Always)
                .size([pane_ws[pi], TOOLBAR_H], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let pane = &mut state.panes[pi];
                    if let Some(t) = &pane.trace {
                        if state.split {
                            let win_size = ui.window_size();
                            ui.set_cursor_pos([win_size[0] - 22.0, ui.cursor_pos()[1]]);
                            state.buf.fmt.clear();
                            write!(state.buf.fmt, "x##close{}", pi).unwrap();
                            if ui.small_button(&state.buf.fmt) {
                                close_pane = Some(pi);
                            }
                            ui.same_line_with_pos(ui.cursor_start_pos()[0]);
                        }
                        ui.checkbox("CPU", &mut pane.show_cpu);
                        ui.same_line();
                        if ui.button("Fit") {
                            let pad = t.max_ts * 0.02;
                            pane.view.t0 = -pad;
                            pane.view.t1 = t.max_ts + pad;
                            pane.view.scroll_y = 0.0;
                        }
                        if can_diff {
                            ui.same_line();
                            if ui.button("Diff") {
                                diff_clicked = true;
                            }
                        }
                        ui.same_line_with_spacing(0.0, 16.0);
                        if pane.search_focus {
                            ui.set_keyboard_focus_here();
                            pane.search_focus = false;
                        }
                        ui.set_next_item_width(200.0);
                        let prev_len = pane.prev_search.len();
                        struct SelectAllCb<'a>(&'a mut bool);
                        impl InputTextCallbackHandler for SelectAllCb<'_> {
                            fn on_always(&mut self, mut data: TextCallbackData) {
                                if *self.0 { data.select_all(); *self.0 = false; }
                            }
                        }
                        let enter = ui.input_text("##search", &mut pane.search)
                            .hint("Search (/) ")
                            .flags(imgui::InputTextFlags::ENTER_RETURNS_TRUE
                                | imgui::InputTextFlags::AUTO_SELECT_ALL)
                            .callback(InputTextCallback::ALWAYS, SelectAllCb(&mut pane.select_all_pending))
                            .build();
                        if pane.search.len() != prev_len || pane.search != pane.prev_search {
                            search_changed[pi] = true;
                            pane.prev_search.clear();
                            pane.prev_search.push_str(&pane.search);
                        }
                        if enter && !pane.search.trim().is_empty() {
                            if search_changed[pi] {
                                pane.rebuild_search();
                                search_changed[pi] = false;
                            }
                            pane.select_from_search(&mut state.buf);
                            pane.pending_tab = Some(BottomTab::Selection);
                        }
                        let search_active = pane.search_mask.iter().any(|&m| m);
                        if search_active {
                            ui.same_line();
                            state.buf.fmt.clear();
                            write!(state.buf.fmt, "{} matches", pane.search_nav.len()).unwrap();
                            ui.text_colored([0.6, 0.8, 1.0, 1.0], &state.buf.fmt);
                        }
                        let n_hidden = pane.hidden_names.iter().filter(|&&h| h).count();
                        if n_hidden > 0 {
                            ui.same_line_with_spacing(0.0, 16.0);
                            state.buf.fmt.clear();
                            write!(state.buf.fmt, "{} hidden", n_hidden).unwrap();
                            ui.text_colored([1.0, 0.7, 0.3, 1.0], &state.buf.fmt);
                            ui.same_line();
                            if ui.small_button("Clear##unhide") {
                                for h in &mut pane.hidden_names { *h = false; }
                            }
                        }

                    } else if pane.loading.is_some() {
                        ui.text("Loading...");
                    } else {
                        ui.text("Drop a trace file here, or: tv <file.json[.gz]>");
                    }
                    if let Some(e) = &pane.error {
                        ui.same_line();
                        ui.text_colored([1.0, 0.4, 0.4, 1.0], e);
                    }
                });
        }

        if let Some(pi) = close_pane {
            if pi == 0 {
                state.panes.swap(0, 1);
            }
            state.panes[1] = Pane::new();
            state.split = false;
            state.active = 0;
        }

        // ---- Diff trigger ----
        if diff_clicked {
            let seq_a = state.panes[0].extract_selection_events();
            let seq_b = state.panes[1].extract_selection_events();
            state.diff_result = Some(diff::compute_diff(&seq_a, &seq_b));
            state.diff_bar_scroll = 0.0;
            state.diff_bar_zoom = 1.0;
            state.show_diff = true;
        }

        // ---- Divider lines (skip when diff popup covers everything) ----
        if !state.diff_popup_open {
            if any_has_trace {
                let divider_y = display[1] - bottom_h - status_h;
                let dl = ui.get_foreground_draw_list();
                let near = (mouse_pos[1] - divider_y).abs() < 4.0 || state.drag == DragKind::BottomDivider;
                let col = if near { col32(120, 120, 120, 255) } else { col32(60, 60, 60, 255) };
                dl.add_line([0.0, divider_y], [display[0], divider_y], col).build();
            }
            if state.split {
                let dl = ui.get_foreground_draw_list();
                let near = (mouse_pos[0] - state.split_x).abs() < 4.0 || state.drag == DragKind::SplitDivider;
                let col = if near { col32(120, 120, 120, 255) } else { col32(60, 60, 60, 255) };
                dl.add_line([state.split_x, 0.0], [state.split_x, display[1]], col).build();
            }
        }

        // ---- Per-pane bottom panels ----
        let mut labels_changed = [false; 2];
        let mut pending_delete_label: [Option<usize>; 2] = [None; 2];
        let bottom_names = ["##bottom0", "##bottom1"];
        let bottom_tab_names = ["##bottomtabs0", "##bottomtabs1"];
        for pi in 0..n_panes {
            if !state.panes[pi].has_trace() { continue; }
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 6.0]));
            ui.window(bottom_names[pi])
                .position([pane_xs[pi], display[1] - bottom_h - status_h], Condition::Always)
                .size([pane_ws[pi], bottom_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let pane = &mut state.panes[pi];
                    let trace = pane.trace.as_ref().unwrap();
                    if let Some(_tab_bar) = ui.tab_bar(bottom_tab_names[pi]) {
                        let pending = pane.pending_tab.take();
                        let detail_flags = if pending == Some(BottomTab::Detail) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Detail").flags(detail_flags).begin(&ui) {
                            if let Some(sel) = &pane.selected {
                                let track = &trace.tracks[sel.track_idx as usize];
                                let ev = &track.events[sel.event_idx as usize];
                                let name = &trace.names[ev.name as usize];
                                let is_hidden = pane.hidden_names.get(ev.name as usize).copied().unwrap_or(false);
                                state.buf.fmt.clear();
                                if is_hidden {
                                    write!(state.buf.fmt, "Show##hideev").unwrap();
                                } else {
                                    write!(state.buf.fmt, "Hide##hideev").unwrap();
                                }
                                if ui.small_button(&state.buf.fmt) {
                                    if (ev.name as usize) < pane.hidden_names.len() {
                                        pane.hidden_names[ev.name as usize] = !is_hidden;
                                    }
                                }
                                ui.same_line_with_spacing(0.0, 8.0);
                                if ui.small_button("Copy All") {
                                    ui.set_clipboard_text(&state.buf.detail_buf);
                                }
                                state.buf.detail_buf.clear();
                                write!(state.buf.detail_buf, "{}\n", name).unwrap();
                                state.buf.detail_buf.push_str("Dur: ");
                                write_time(&mut state.buf.detail_buf, ev.dur);
                                state.buf.detail_buf.push_str("  |  Start: +");
                                write_time(&mut state.buf.detail_buf, ev.ts);
                                write!(state.buf.detail_buf, "\nCat: {}  |  Track: {}", trace.cats[ev.cat as usize], track.label).unwrap();
                                if ev.args_count > 0 {
                                    state.buf.detail_buf.push('\n');
                                    let pairs = &trace.arg_pairs[ev.args_start as usize
                                        ..(ev.args_start as usize + ev.args_count as usize)];
                                    for &[k, v] in pairs {
                                        write!(state.buf.detail_buf, "\n{}: {}", trace.arg_strs[k as usize], trace.arg_strs[v as usize]).unwrap();
                                    }
                                }
                                let avail = ui.content_region_avail();
                                ui.input_text_multiline("##detail_text", &mut state.buf.detail_buf, [avail[0], avail[1]])
                                    .flags(imgui::InputTextFlags::READ_ONLY)
                                    .build();
                            } else {
                                ui.text_colored([0.5, 0.5, 0.5, 1.0], "Click an event to see details");
                            }
                        }

                        let stats_flags = if pending == Some(BottomTab::Stats) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Stats").flags(stats_flags).begin(&ui) {
                            draw_stats_table(&ui, trace, &trace.stats, &mut pane.search, &mut search_changed[pi], &mut pane.sort_col, &mut pane.sort_asc, &mut state.buf);
                        }

                        let sel_flags = if pending == Some(BottomTab::Selection) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Selection").flags(sel_flags).begin(&ui) {
                            if !pane.selection_stats.is_empty() {
                                ui.checkbox("Aggregate", &mut pane.sel_aggregate);
                                ui.same_line();
                                if ui.button("Hide Selected") {
                                    for se in &pane.selection_stats {
                                        if let Some(h) = pane.hidden_names.get_mut(se.name as usize) { *h = true; }
                                    }
                                }
                                ui.same_line();
                                let sel_total_count: u32 = pane.selection_stats.iter().map(|s| s.count).sum();
                                let sel_total_dur: f64 = pane.selection_stats.iter().map(|s| s.total_dur).sum();
                                state.buf.fmt.clear();
                                write!(state.buf.fmt, "{} events, ", sel_total_count).unwrap();
                                write_time(&mut state.buf.fmt, sel_total_dur);
                                write!(state.buf.fmt, " total").unwrap();
                                ui.text_colored([0.6, 0.6, 0.6, 1.0], &state.buf.fmt);
                                draw_selection_histogram(&ui, trace, &pane.selection_stats, pane.sel_aggregate, &mut state.buf);
                                ui.separator();
                                if pane.sel_aggregate {
                                    let sel_entries: Vec<KernelStats> = pane.selection_stats.iter()
                                        .map(|s| {
                                            let mut sorted = s.durations.clone();
                                            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                                            let n = sorted.len();
                                            let median = if n == 0 { 0.0 } else if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
                                            KernelStats {
                                                name: s.name, count: s.count, total_dur: s.total_dur,
                                                median_dur: median,
                                                max_dur: s.durations.iter().copied().fold(0.0f64, f64::max),
                                            }
                                        }).collect();
                                    draw_stats_table(&ui, trace, &sel_entries, &mut pane.search, &mut search_changed[pi], &mut pane.sort_col, &mut pane.sort_asc, &mut state.buf);
                                } else {
                                    let mut individual: Vec<KernelStats> = Vec::new();
                                    for se in &pane.selection_stats {
                                        for &d in &se.durations {
                                            individual.push(KernelStats { name: se.name, count: 1, total_dur: d, median_dur: d, max_dur: d });
                                        }
                                    }
                                    individual.sort_unstable_by(|a, b| b.total_dur.partial_cmp(&a.total_dur).unwrap());
                                    draw_stats_table(&ui, trace, &individual, &mut pane.search, &mut search_changed[pi], &mut pane.sort_col, &mut pane.sort_asc, &mut state.buf);
                                }
                            } else {
                                ui.text_colored([0.5, 0.5, 0.5, 1.0], "Shift+drag to select a time range");
                            }
                        }

                        let labels_flags = if pending == Some(BottomTab::Labels) {
                            imgui::TabItemFlags::SET_SELECTED
                        } else { imgui::TabItemFlags::empty() };
                        if let Some(_t) = imgui::TabItem::new("Labels").flags(labels_flags).begin(&ui) {
                            if pane.selection.is_some() || !pane.selection_stats.is_empty() {
                                ui.set_next_item_width(160.0);
                                let label_enter = ui.input_text("##labelinput", &mut pane.label_input)
                                    .hint("Label name")
                                    .flags(imgui::InputTextFlags::ENTER_RETURNS_TRUE)
                                    .build();
                                ui.same_line();
                                let label_btn = ui.button("Label");
                                if (label_enter || label_btn) && !pane.label_input.is_empty() {
                                    labels_changed[pi] = true;
                                }
                            } else {
                                ui.text_colored([0.5, 0.5, 0.5, 1.0], "(select a region first)");
                            }
                            if pane.labels.is_empty() {
                                if pane.selection.is_none() && pane.selection_stats.is_empty() {
                                    ui.text_colored([0.5, 0.5, 0.5, 1.0], "Select a range and label it to categorize kernels");
                                }
                            } else {
                                let avail = ui.content_region_avail();
                                let total_trace_dur: f64 = trace.tracks.iter()
                                    .flat_map(|t| t.events.iter())
                                    .map(|e| e.dur)
                                    .sum();
                                ui.child_window("##labelstable")
                                    .size([avail[0], avail[1]])
                                    .build(|| {
                                        let col_w = 80.0;
                                        ui.text_colored([0.55, 0.55, 0.55, 1.0], "Label");
                                        ui.same_line_with_pos(160.0);
                                        ui.text_colored([0.55, 0.55, 0.55, 1.0], "Kernels");
                                        ui.same_line_with_pos(160.0 + col_w);
                                        ui.text_colored([0.55, 0.55, 0.55, 1.0], "Events");
                                        ui.same_line_with_pos(160.0 + col_w * 2.0);
                                        ui.text_colored([0.55, 0.55, 0.55, 1.0], "Total");
                                        ui.same_line_with_pos(160.0 + col_w * 3.0);
                                        ui.text_colored([0.55, 0.55, 0.55, 1.0], "%");
                                        ui.separator();

                                        for ls in &pane.label_stats {
                                            let li = ls.label_idx as usize;
                                            let label = &pane.labels[li];
                                            let lc: u32 = label.color.into();
                                            let r = (lc & 0xFF) as f32 / 255.0;
                                            let g = ((lc >> 8) & 0xFF) as f32 / 255.0;
                                            let b = ((lc >> 16) & 0xFF) as f32 / 255.0;
                                            let color = [r, g, b, 1.0];
                                            ui.text_colored(color, &label.name);
                                            if ui.is_item_clicked() {
                                                let n = pane.trace.as_ref().map(|t| t.names.len()).unwrap_or(0);
                                                pane.search_mask.clear();
                                                pane.search_mask.resize(n, false);
                                                for &kn in &label.pattern {
                                                    if (kn as usize) < pane.search_mask.len() {
                                                        pane.search_mask[kn as usize] = true;
                                                    }
                                                }
                                                pane.search.clear();
                                                pane.search.push_str(&label.name);
                                                pane.prev_search.clear();
                                                pane.prev_search.push_str(&label.name);
                                            }
                                            ui.same_line_with_pos(160.0);
                                            state.buf.fmt.clear();
                                            write!(state.buf.fmt, "{}", label.pattern.len()).unwrap();
                                            ui.text(&state.buf.fmt);
                                            ui.same_line_with_pos(160.0 + col_w);
                                            state.buf.fmt.clear();
                                            write!(state.buf.fmt, "{}", ls.count).unwrap();
                                            ui.text(&state.buf.fmt);
                                            ui.same_line_with_pos(160.0 + col_w * 2.0);
                                            state.buf.fmt.clear();
                                            write_time(&mut state.buf.fmt, ls.total_dur);
                                            ui.text(&state.buf.fmt);
                                            ui.same_line_with_pos(160.0 + col_w * 3.0);
                                            let pct = if total_trace_dur > 0.0 { ls.total_dur / total_trace_dur * 100.0 } else { 0.0 };
                                            state.buf.fmt.clear();
                                            write!(state.buf.fmt, "{:.1}%", pct).unwrap();
                                            ui.text(&state.buf.fmt);
                                            ui.same_line_with_pos(160.0 + col_w * 4.0);
                                            state.buf.fmt.clear();
                                            write!(state.buf.fmt, "x##del{}", li).unwrap();
                                            if ui.small_button(&state.buf.fmt) {
                                                pending_delete_label[pi] = Some(li);
                                            }
                                        }
                                        if pending_delete_label[pi].is_some() {
                                            labels_changed[pi] = true;
                                        }
                                    });
                            }
                        }
                    }
                });
        }

        // ---- Per-pane status bars ----
        let status_names = ["##status0", "##status1"];
        for pi in 0..n_panes {
            if !state.panes[pi].has_trace() { continue; }
            let pane = &state.panes[pi];
            let t = pane.trace.as_ref().unwrap();
            let _pad = ui.push_style_var(StyleVar::WindowPadding([8.0, 2.0]));
            ui.window(status_names[pi])
                .position([pane_xs[pi], display[1] - status_h], Condition::Always)
                .size([pane_ws[pi], status_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let dl = ui.get_window_draw_list();
                    let win_pos = ui.window_pos();
                    let win_size = ui.window_size();
                    let text_h = ui.calc_text_size("X")[1];
                    let cy = win_pos[1] + (status_h - text_h) / 2.0;

                    state.buf.fmt.clear();
                    let fname = pane.trace_path.rsplit('/').next().unwrap_or(&pane.trace_path);
                    write!(state.buf.fmt, "{} | {} events | {} tracks", fname, t.total_events, t.tracks.len()).unwrap();
                    dl.add_text([win_pos[0] + 8.0, cy], col32(153, 153, 153, 255), &state.buf.fmt);

                    let logo_scale = text_h / 16.0;
                    let logo_w = 16.0 * logo_scale;
                    let mut right_x = win_pos[0] + win_size[0] - 8.0;
                    if !t.device.is_empty() {
                        let dev_size = ui.calc_text_size(&t.device);
                        right_x -= dev_size[0];
                        dl.add_text([right_x, cy], device_name_color(), &t.device);
                        right_x -= 6.0;
                        let on_size = ui.calc_text_size("on");
                        right_x -= on_size[0];
                        dl.add_text([right_x, cy], col32(120, 120, 120, 255), "on");
                        right_x -= 6.0;
                    }
                    right_x -= logo_w;
                    draw_vllm_logo(&dl, right_x, cy, logo_scale);
                });
        }

        // ---- Per-pane timelines ----
        let mut hover_results: [Option<EventRef>; 2] = [None; 2];
        let mut click_results: [Option<EventRef>; 2] = [None; 2];
        let mut new_selections: [Option<Option<[f64; 4]>>; 2] = [None; 2];
        let mut double_clicks: [bool; 2] = [false; 2];

        let timeline_names = ["##timeline0", "##timeline1"];
        let canvas_names = ["##canvas0", "##canvas1"];
        for pi in 0..n_panes {
            if !state.panes[pi].has_trace() { continue; }
            let tl_top = TOOLBAR_H;
            let tl_h = display[1] - TOOLBAR_H - bottom_h - status_h;

            let _pad = ui.push_style_var(StyleVar::WindowPadding([0.0, 0.0]));
            ui.window(timeline_names[pi])
                .position([pane_xs[pi], tl_top], Condition::Always)
                .size([pane_ws[pi], tl_h], Condition::Always)
                .flags(
                    WindowFlags::NO_DECORATION
                        | WindowFlags::NO_MOVE
                        | WindowFlags::NO_SAVED_SETTINGS
                        | WindowFlags::NO_SCROLL_WITH_MOUSE
                        | WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
                )
                .build(|| {
                    let pos = ui.cursor_screen_pos();
                    let avail = ui.content_region_avail();
                    ui.invisible_button(canvas_names[pi], avail);
                    let hovered = ui.is_item_hovered();
                    let clicked = ui.is_item_clicked();
                    let double_clicked = ui.is_mouse_double_clicked(imgui::MouseButton::Left) && hovered;
                    let active = ui.is_item_active();

                    if clicked {
                        state.active = pi;
                    }

                    let pane = &mut state.panes[pi];
                    let trace = pane.trace.as_ref().unwrap();
                    let rect = [pos[0], pos[1], pos[0] + avail[0], pos[1] + avail[1]];
                    let (h, c, sel) = draw_timeline(
                        &ui,
                        trace,
                        &mut pane.view,
                        pane.show_cpu,
                        &mut state.buf,
                        rect,
                        hovered,
                        clicked,
                        active,
                        mouse_pos,
                        mouse_delta,
                        scroll,
                        pinch,
                        ctrl,
                        shift,
                        &pane.search_mask,
                        pane.selection,
                        pane.finished_sel,
                        &mut pane.collapsed,
                        &pane.event_labels,
                        &pane.labels,
                        &pane.hidden_names,
                        pane.selected,
                        pane.multi_select_name,
                        &pane.sel_mask,
                        state.label_w,
                        &mut pane.track_scales,
                        &mut state.drag,
                    );
                    hover_results[pi] = h;
                    click_results[pi] = c;
                    new_selections[pi] = sel;
                    double_clicks[pi] = double_clicked;
                });
        }

        // ---- Process click/selection results per pane ----
        for pi in 0..n_panes {
            if let Some(c) = click_results[pi] {
                let pane = &mut state.panes[pi];
                let trace = pane.trace.as_ref().unwrap();
                let ev = &trace.tracks[c.track_idx as usize].events[c.event_idx as usize];
                if double_clicks[pi] {
                    pane.multi_select_name = Some(ev.name);
                } else {
                    pane.multi_select_name = None;
                }
                pane.selected = Some(c);
                pane.clear_selection();
                state.active = pi;
                pane.pending_tab = Some(BottomTab::Detail);
            }

            if let Some(sel) = new_selections[pi] {
                let pane = &mut state.panes[pi];
                pane.selection = sel;
                if sel.is_some() {
                    pane.selected = None;
                    pane.multi_select_name = None;
                    pane.selection_dirty = true;
                } else {
                    pane.clear_selection();
                }
            }

            if state.panes[pi].selection_dirty && !ui.io().mouse_down[0] {
                state.panes[pi].finish_selection(&mut state.buf);
                state.panes[pi].pending_tab = Some(BottomTab::Selection);
                state.panes[pi].selection_dirty = false;
                state.active = pi;
            }
        }

        for pi in 0..n_panes {
            if labels_changed[pi] {
                let pane = &mut state.panes[pi];
                if !pane.label_input.is_empty() {
                    let name = pane.label_input.clone();
                    pane.apply_label(&name);
                    pane.label_input.clear();
                    pane.clear_selection();
                }
                if let Some(di) = pending_delete_label[pi] {
                    pane.delete_label(di);
                }
                pane.pending_tab = Some(BottomTab::Labels);
            }
        }

        // ---- Hover tooltip ----
        for pi in 0..n_panes {
            if state.diff_popup_open { break; }
            if let Some(r) = &hover_results[pi] {
                if let Some(trace) = &state.panes[pi].trace {
                    let track = &trace.tracks[r.track_idx as usize];
                    let ev = &track.events[r.event_idx as usize];
                    ui.tooltip(|| {
                        ui.text(&trace.names[ev.name as usize]);
                        state.buf.fmt.clear();
                        state.buf.fmt.push_str("Dur: ");
                        write_time(&mut state.buf.fmt, ev.dur);
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        state.buf.fmt.push_str("Start: +");
                        write_time(&mut state.buf.fmt, ev.ts);
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        write!(state.buf.fmt, "Cat: {}", trace.cats[ev.cat as usize]).unwrap();
                        ui.text(&state.buf.fmt);
                        state.buf.fmt.clear();
                        write!(state.buf.fmt, "Track: {}", track.label).unwrap();
                        ui.text(&state.buf.fmt);
                    });
                }
            }
        }

        // ---- Diff window ----
        state.diff_popup_open = false;
        if state.show_diff {
            let mut opened = true;
            let size_cond = if state.diff_result.is_some() { Condition::Once } else { Condition::Never };
            if let Some(_token) = ui.window("Diff")
                .size([display[0] * 0.7, display[1] * 0.75], size_cond)
                .position([display[0] * 0.15, display[1] * 0.12], size_cond)
                .resizable(true)
                .collapsible(false)
                .opened(&mut opened)
                .flags(WindowFlags::NO_SCROLLBAR | WindowFlags::NO_SCROLL_WITH_MOUSE)
                .begin()
            {
                state.diff_popup_open = true;
                if let Some(diff) = &state.diff_result {
                    let na = std::path::Path::new(&state.panes[0].trace_path)
                        .file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                    let nb = std::path::Path::new(&state.panes[1].trace_path)
                        .file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                    let (scroll, zoom) = draw_diff_popup(&ui, diff, &mut state.buf, display, &na, &nb,
                        state.diff_bar_scroll, state.diff_bar_zoom);
                    state.diff_bar_scroll = scroll;
                    state.diff_bar_zoom = zoom;
                }
                if ui.is_key_pressed(imgui::Key::Escape) {
                    opened = false;
                }
            }
            if !opened {
                state.show_diff = false;
            }
        }

        // ---- Keybinds (active pane) ----
        let any_text_focused = ui.is_any_item_active();
        let ai = state.active;

        if ui.is_key_pressed(imgui::Key::Home) {
            if let Some(t) = &state.panes[ai].trace {
                let pad = t.max_ts * 0.02;
                state.panes[ai].view.t0 = -pad;
                state.panes[ai].view.t1 = t.max_ts + pad;
                state.panes[ai].view.scroll_y = 0.0;
            }
        }
        if ui.is_key_pressed(imgui::Key::Escape) {
            state.panes[ai].search.clear();
            state.panes[ai].clear_selection();
            search_changed[ai] = true;
        }
        if !any_text_focused {
            if ui.is_key_pressed(imgui::Key::Slash) || (ctrl && ui.is_key_pressed(imgui::Key::F)) {
                state.panes[ai].search_focus = true;
            }
            if ui.is_key_pressed(imgui::Key::N) {
                let pane = &mut state.panes[ai];
                if !pane.search_nav.is_empty() {
                    if shift {
                        if pane.search_cursor == 0 {
                            pane.search_cursor = pane.search_nav.len();
                        }
                        pane.search_cursor -= 1;
                    } else {
                        if pane.search_cursor >= pane.search_nav.len() {
                            pane.search_cursor = 0;
                        }
                    }
                    let (ts, ti, ei) = pane.search_nav[pane.search_cursor];
                    pane.selected = Some(EventRef { track_idx: ti, event_idx: ei });
                    pane.pending_tab = Some(BottomTab::Detail);
                    let ev = &pane.trace.as_ref().unwrap().tracks[ti as usize].events[ei as usize];
                    let pad = (pane.view.t1 - pane.view.t0) * 0.1;
                    pane.view.t0 = ts - pad;
                    pane.view.t1 = ts + ev.dur + pad;
                    if !shift { pane.search_cursor += 1; }
                }
            }
        }

        for pi in 0..n_panes {
            if search_changed[pi] {
                state.panes[pi].rebuild_search();
            }
        }

        let draw_data = imgui.render();
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.render(draw_data);
        }
    }
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(files);
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
