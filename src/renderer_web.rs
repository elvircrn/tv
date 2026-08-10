use imgui::FontSource;
use winit::window::Window;

/// Phase-1 stub: satisfies the same `new`/`resize`/`render` shape as
/// `MetalRenderer` (see main.rs's `PlatformRenderer` alias) so the rest of
/// the app — input handling, UI logic, event loop — compiles and runs
/// end-to-end on wasm32 before the real WebGL2 draw path exists. `render`
/// is a no-op for now; nothing is drawn to the canvas yet.
pub struct WebGl2Renderer;

impl WebGl2Renderer {
    pub fn new(_window: &Window, imgui: &mut imgui::Context, scale_factor: f64) -> Self {
        let hidpi_size = (15.0 * scale_factor as f32).round();
        imgui.fonts().add_font(&[FontSource::DefaultFontData {
            config: Some(imgui::FontConfig { size_pixels: hidpi_size, ..Default::default() }),
        }]);
        imgui.fonts().build_rgba32_texture();
        imgui.fonts().tex_id = imgui::TextureId::new(0);
        imgui.io_mut().font_global_scale = 1.0 / scale_factor as f32;
        Self
    }

    pub fn resize(&self, _w: u32, _h: u32) {}

    pub fn render(&mut self, _draw_data: &imgui::DrawData) {}
}
