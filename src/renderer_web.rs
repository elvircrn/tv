use imgui::{DrawCmd, DrawVert, FontSource};
use wasm_bindgen::JsCast;
use web_sys::{WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram, WebGlTexture, WebGlUniformLocation};
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;

const VERT_SRC: &str = r#"#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
uniform mat4 uProj;
out vec2 vUV;
out vec4 vColor;
void main() {
    vUV = aUV;
    vColor = aColor;
    gl_Position = uProj * vec4(aPos, 0.0, 1.0);
}
"#;

const FRAG_SRC: &str = r#"#version 300 es
precision mediump float;
in vec2 vUV;
in vec4 vColor;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    fragColor = vColor * texture(uTex, vUV);
}
"#;

pub struct WebGl2Renderer {
    gl: Gl,
    canvas: web_sys::HtmlCanvasElement,
    program: WebGlProgram,
    u_proj: WebGlUniformLocation,
    font_tex: WebGlTexture,
    vtx_buf: WebGlBuffer,
    idx_buf: WebGlBuffer,
    // Scratch buffers rebuilt every frame from imgui's per-draw-list data,
    // mirroring MetalRenderer's single contiguous vtx/idx upload.
    vtx_scratch: Vec<u8>,
    idx_scratch: Vec<u8>,
}

fn compile_shader(gl: &Gl, kind: u32, src: &str) -> web_sys::WebGlShader {
    let shader = gl.create_shader(kind).expect("create_shader failed");
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if !gl.get_shader_parameter(&shader, Gl::COMPILE_STATUS).as_bool().unwrap_or(false) {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        panic!("shader compile failed: {log}");
    }
    shader
}

impl WebGl2Renderer {
    pub fn new(window: &Window, imgui: &mut imgui::Context, scale_factor: f64) -> Self {
        let canvas = window.canvas().expect("winit created no canvas");
        let gl: Gl = canvas
            .get_context("webgl2")
            .expect("get_context failed")
            .expect("no webgl2 context")
            .dyn_into()
            .expect("context is not WebGl2RenderingContext");

        let vert = compile_shader(&gl, Gl::VERTEX_SHADER, VERT_SRC);
        let frag = compile_shader(&gl, Gl::FRAGMENT_SHADER, FRAG_SRC);
        let program = gl.create_program().expect("create_program failed");
        gl.attach_shader(&program, &vert);
        gl.attach_shader(&program, &frag);
        gl.link_program(&program);
        if !gl.get_program_parameter(&program, Gl::LINK_STATUS).as_bool().unwrap_or(false) {
            let log = gl.get_program_info_log(&program).unwrap_or_default();
            panic!("program link failed: {log}");
        }
        gl.delete_shader(Some(&vert));
        gl.delete_shader(Some(&frag));
        let u_proj = gl.get_uniform_location(&program, "uProj").expect("uProj not found");
        let u_tex = gl.get_uniform_location(&program, "uTex");

        // Font atlas: imgui's bundled `DefaultFontData` is a tiny pixel font
        // (ProggyClean) meant for a native-resolution, unscaled UI — it reads
        // as blurry/blocky once stb_truetype rasterizes it at a real HiDPI
        // pixel size. There's no filesystem to read a system font from in
        // the browser (see renderer.rs's SF Mono read, native-only), so
        // DejaVu Sans Mono is vendored instead (third-party/, Bitstream
        // Vera/DejaVu license — explicitly redistributable, unlike Apple's
        // SF Mono). Same oversampling/gamma-lift config as native's SF Mono
        // setup to keep thin glyph edges from reading muddy on the dark
        // theme.
        const FONT_TTF: &[u8] = include_bytes!("../third-party/DejaVuSansMono.ttf");
        let hidpi_size = (15.0 * scale_factor as f32).round();
        imgui.fonts().add_font(&[FontSource::TtfData {
            data: FONT_TTF,
            size_pixels: hidpi_size,
            config: Some(imgui::FontConfig {
                oversample_h: 3,
                oversample_v: 1,
                pixel_snap_h: true,
                rasterizer_multiply: 1.5,
                ..Default::default()
            }),
        }]);
        let atlas = imgui.fonts().build_rgba32_texture();
        let (aw, ah) = (atlas.width, atlas.height);
        let mut atlas_px = atlas.data.to_vec();
        for px in atlas_px.chunks_exact_mut(4) {
            let a = px[3] as f32 / 255.0;
            px[3] = (a.powf(0.72) * 255.0 + 0.5) as u8;
        }
        imgui.fonts().tex_id = imgui::TextureId::new(0);
        imgui.io_mut().font_global_scale = 1.0 / scale_factor as f32;

        let font_tex = gl.create_texture().expect("create_texture failed");
        gl.bind_texture(Gl::TEXTURE_2D, Some(&font_tex));
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            Gl::TEXTURE_2D, 0, Gl::RGBA as i32, aw as i32, ah as i32, 0,
            Gl::RGBA, Gl::UNSIGNED_BYTE, Some(&atlas_px),
        ).expect("tex_image_2d failed");

        let vtx_buf = gl.create_buffer().expect("create_buffer failed");
        let idx_buf = gl.create_buffer().expect("create_buffer failed");

        gl.use_program(Some(&program));
        if let Some(loc) = &u_tex { gl.uniform1i(Some(loc), 0); }
        gl.enable(Gl::BLEND);
        gl.blend_equation(Gl::FUNC_ADD);
        gl.blend_func_separate(Gl::SRC_ALPHA, Gl::ONE_MINUS_SRC_ALPHA, Gl::ONE, Gl::ONE_MINUS_SRC_ALPHA);
        gl.disable(Gl::DEPTH_TEST);
        gl.disable(Gl::CULL_FACE);
        gl.enable(Gl::SCISSOR_TEST);

        Self {
            gl, canvas, program, u_proj, font_tex, vtx_buf, idx_buf,
            vtx_scratch: Vec::new(), idx_scratch: Vec::new(),
        }
    }

    pub fn resize(&self, w: u32, h: u32) {
        self.canvas.set_width(w);
        self.canvas.set_height(h);
    }

    pub fn render(&mut self, draw_data: &imgui::DrawData) {
        let gl = &self.gl;
        let fb_w = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_h = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if fb_w <= 0.0 || fb_h <= 0.0 { return; }

        let vtx_size_of = std::mem::size_of::<DrawVert>();
        let idx_size_of = std::mem::size_of::<imgui::DrawIdx>();
        self.vtx_scratch.clear();
        self.idx_scratch.clear();
        for dl in draw_data.draw_lists() {
            let vb = dl.vtx_buffer();
            let ib = dl.idx_buffer();
            self.vtx_scratch.extend_from_slice(unsafe {
                std::slice::from_raw_parts(vb.as_ptr() as *const u8, vb.len() * vtx_size_of)
            });
            self.idx_scratch.extend_from_slice(unsafe {
                std::slice::from_raw_parts(ib.as_ptr() as *const u8, ib.len() * idx_size_of)
            });
        }

        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&self.vtx_buf));
        gl.buffer_data_with_u8_array(Gl::ARRAY_BUFFER, &self.vtx_scratch, Gl::DYNAMIC_DRAW);
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&self.idx_buf));
        gl.buffer_data_with_u8_array(Gl::ELEMENT_ARRAY_BUFFER, &self.idx_scratch, Gl::DYNAMIC_DRAW);

        gl.enable_vertex_attrib_array(0);
        gl.enable_vertex_attrib_array(1);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(0, 2, Gl::FLOAT, false, vtx_size_of as i32, 0);
        gl.vertex_attrib_pointer_with_i32(1, 2, Gl::FLOAT, false, vtx_size_of as i32, 8);
        gl.vertex_attrib_pointer_with_i32(2, 4, Gl::UNSIGNED_BYTE, true, vtx_size_of as i32, 16);

        let l = draw_data.display_pos[0];
        let r = l + draw_data.display_size[0];
        let t = draw_data.display_pos[1];
        let b = t + draw_data.display_size[1];
        #[rustfmt::skip]
        let proj: [f32; 16] = [
            2.0/(r-l),      0.0,            0.0, 0.0,
            0.0,            2.0/(t-b),      0.0, 0.0,
            0.0,            0.0,            1.0, 0.0,
            (r+l)/(l-r),   (t+b)/(b-t),    0.0, 1.0,
        ];

        gl.viewport(0, 0, fb_w as i32, fb_h as i32);
        gl.clear_color(0.094, 0.094, 0.094, 1.0);
        gl.clear(Gl::COLOR_BUFFER_BIT);
        gl.use_program(Some(&self.program));
        gl.uniform_matrix4fv_with_f32_array(Some(&self.u_proj), false, &proj);
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_texture(Gl::TEXTURE_2D, Some(&self.font_tex));

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        let idx_type = if idx_size_of == 2 { Gl::UNSIGNED_SHORT } else { Gl::UNSIGNED_INT };

        let mut global_vtx_bytes = 0i32;
        let mut global_idx_bytes = 0i32;
        for dl in draw_data.draw_lists() {
            for cmd in dl.commands() {
                match cmd {
                    DrawCmd::Elements { count, cmd_params } => {
                        let cx0 = (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0];
                        let cy0 = (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1];
                        let cx1 = (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0];
                        let cy1 = (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1];
                        if cx1 <= cx0 || cy1 <= cy0 { continue; }
                        let sx = cx0.max(0.0);
                        let sy = cy0.max(0.0);
                        let sw = (cx1 - sx).max(1.0);
                        let sh = (cy1 - sy).max(1.0);
                        // WebGL's scissor/viewport origin is bottom-left;
                        // imgui's clip rect is top-left. Flip Y.
                        let gl_y = fb_h - (sy + sh);
                        gl.scissor(sx as i32, gl_y as i32, sw as i32, sh as i32);

                        let vtx_byte_off = global_vtx_bytes + (cmd_params.vtx_offset * vtx_size_of) as i32;
                        gl.vertex_attrib_pointer_with_i32(0, 2, Gl::FLOAT, false, vtx_size_of as i32, vtx_byte_off);
                        gl.vertex_attrib_pointer_with_i32(1, 2, Gl::FLOAT, false, vtx_size_of as i32, vtx_byte_off + 8);
                        gl.vertex_attrib_pointer_with_i32(2, 4, Gl::UNSIGNED_BYTE, true, vtx_size_of as i32, vtx_byte_off + 16);

                        let idx_byte_off = global_idx_bytes + (cmd_params.idx_offset * idx_size_of) as i32;
                        gl.draw_elements_with_i32(Gl::TRIANGLES, count as i32, idx_type, idx_byte_off);
                    }
                    _ => {}
                }
            }
            global_vtx_bytes += (dl.vtx_buffer().len() * vtx_size_of) as i32;
            global_idx_bytes += (dl.idx_buffer().len() * idx_size_of) as i32;
        }
    }
}
