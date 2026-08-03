use imgui::{DrawCmd, DrawVert, FontSource};
use metal::*;
use raw_window_handle::HasWindowHandle;
use winit::window::Window;

use crate::types::INITIAL_BUF;

const MSL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 pos [[attribute(0)]];
    float2 uv  [[attribute(1)]];
    float4 col [[attribute(2)]];
};

struct VertexOut {
    float4 pos [[position]];
    float2 uv;
    float4 col;
};

vertex VertexOut vertex_main(
    VertexIn in [[stage_in]],
    constant float4x4 &proj [[buffer(1)]]
) {
    VertexOut out;
    out.pos = proj * float4(in.pos, 0.0, 1.0);
    out.uv  = in.uv;
    out.col = in.col;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]]
) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    return in.col * tex.sample(s, in.uv);
}
"#;

pub struct MetalRenderer {
    device: Device,
    queue: CommandQueue,
    pipeline: RenderPipelineState,
    font_tex: Texture,
    vtx_buf: Buffer,
    idx_buf: Buffer,
    proj_buf: Buffer,
    vtx_cap: usize,
    idx_cap: usize,
    raw_layer: raw_window_metal::Layer,
}

impl MetalRenderer {
    pub fn new(window: &Window, imgui: &mut imgui::Context, scale_factor: f64) -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();

        let handle = window.window_handle().unwrap();
        let ns_view = match handle.as_raw() {
            raw_window_handle::RawWindowHandle::AppKit(h) => h.ns_view,
            _ => panic!("not macOS"),
        };
        let raw_layer = unsafe { raw_window_metal::Layer::from_ns_view(ns_view) };
        let layer = layer_ref(&raw_layer);
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let size = window.inner_size();
        layer.set_drawable_size(core_graphics_types::geometry::CGSize::new(
            size.width as f64,
            size.height as f64,
        ));

        let opts = CompileOptions::new();
        let lib = device.new_library_with_source(MSL_SHADER, &opts).expect("MSL compile failed");
        let vert = lib.get_function("vertex_main", None).unwrap();
        let frag = lib.get_function("fragment_main", None).unwrap();

        let vd = VertexDescriptor::new();
        let attrs = vd.attributes();
        attrs.object_at(0).unwrap().set_format(MTLVertexFormat::Float2);
        attrs.object_at(0).unwrap().set_offset(0);
        attrs.object_at(0).unwrap().set_buffer_index(0);
        attrs.object_at(1).unwrap().set_format(MTLVertexFormat::Float2);
        attrs.object_at(1).unwrap().set_offset(8);
        attrs.object_at(1).unwrap().set_buffer_index(0);
        attrs.object_at(2).unwrap().set_format(MTLVertexFormat::UChar4Normalized);
        attrs.object_at(2).unwrap().set_offset(16);
        attrs.object_at(2).unwrap().set_buffer_index(0);
        let layout = vd.layouts().object_at(0).unwrap();
        layout.set_stride(std::mem::size_of::<DrawVert>() as u64);
        layout.set_step_function(MTLVertexStepFunction::PerVertex);

        let pdesc = RenderPipelineDescriptor::new();
        pdesc.set_vertex_function(Some(&vert));
        pdesc.set_fragment_function(Some(&frag));
        pdesc.set_vertex_descriptor(Some(vd));
        let ca = pdesc.color_attachments().object_at(0).unwrap();
        ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        ca.set_blending_enabled(true);
        ca.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        ca.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.set_source_alpha_blend_factor(MTLBlendFactor::One);
        ca.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        let pipeline = device
            .new_render_pipeline_state(&pdesc)
            .expect("pipeline failed");

        let hidpi_size = (15.0 * scale_factor as f32).round();
        let system_font = std::fs::read("/System/Library/Fonts/SFNSMono.ttf")
            .or_else(|_| std::fs::read("/Library/Fonts/SF-Mono.ttf"))
            .ok();
        let fonts = imgui.fonts();
        if let Some(ref data) = system_font {
            fonts.add_font(&[FontSource::TtfData {
                data,
                size_pixels: hidpi_size,
                config: Some(imgui::FontConfig {
                    oversample_h: 3,
                    oversample_v: 1,
                    pixel_snap_h: true,
                    rasterizer_multiply: 1.5,
                    ..Default::default()
                }),
            }]);
        } else {
            fonts.add_font(&[FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    size_pixels: hidpi_size,
                    ..Default::default()
                }),
            }]);
        }
        let atlas = fonts.build_rgba32_texture();
        let tdesc = TextureDescriptor::new();
        tdesc.set_texture_type(MTLTextureType::D2);
        tdesc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        tdesc.set_width(atlas.width as u64);
        tdesc.set_height(atlas.height as u64);
        tdesc.set_usage(MTLTextureUsage::ShaderRead);
        let font_tex = device.new_texture(&tdesc);
        font_tex.replace_region(
            MTLRegion::new_2d(0, 0, atlas.width as u64, atlas.height as u64),
            0,
            atlas.data.as_ptr() as *const _,
            (atlas.width * 4) as u64,
        );
        imgui.fonts().tex_id = imgui::TextureId::new(0);
        imgui.io_mut().font_global_scale = 1.0 / scale_factor as f32;
        let style = imgui.style_mut();
        style.colors[imgui::StyleColor::Text as usize] = [1.0, 1.0, 1.0, 1.0];

        let vtx_buf = device.new_buffer(INITIAL_BUF as u64, MTLResourceOptions::StorageModeShared);
        let idx_buf = device.new_buffer(INITIAL_BUF as u64, MTLResourceOptions::StorageModeShared);
        let proj_buf = device.new_buffer(64, MTLResourceOptions::StorageModeShared);

        Self {
            device, queue, pipeline, font_tex, vtx_buf, idx_buf, proj_buf,
            vtx_cap: INITIAL_BUF, idx_cap: INITIAL_BUF, raw_layer,
        }
    }

    pub fn layer(&self) -> &MetalLayerRef {
        layer_ref(&self.raw_layer)
    }

    pub fn resize(&self, w: u32, h: u32) {
        self.layer().set_drawable_size(core_graphics_types::geometry::CGSize::new(w as f64, h as f64));
    }

    pub fn render(&mut self, draw_data: &imgui::DrawData) {
        let drawable = match layer_ref(&self.raw_layer).next_drawable() {
            Some(d) => d,
            None => return,
        };
        let fb_w = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_h = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if fb_w <= 0.0 || fb_h <= 0.0 { return; }

        let vtx_size_of = std::mem::size_of::<DrawVert>();
        let idx_size_of = std::mem::size_of::<imgui::DrawIdx>();
        let vtx_total = draw_data.total_vtx_count as usize * vtx_size_of;
        let idx_total = draw_data.total_idx_count as usize * idx_size_of;

        if vtx_total > self.vtx_cap {
            self.vtx_cap = (vtx_total * 2).max(INITIAL_BUF);
            self.vtx_buf = self.device.new_buffer(self.vtx_cap as u64, MTLResourceOptions::StorageModeShared);
        }
        if idx_total > self.idx_cap {
            self.idx_cap = (idx_total * 2).max(INITIAL_BUF);
            self.idx_buf = self.device.new_buffer(self.idx_cap as u64, MTLResourceOptions::StorageModeShared);
        }

        let vtx_dst = self.vtx_buf.contents() as *mut u8;
        let idx_dst = self.idx_buf.contents() as *mut u8;
        let mut vo = 0usize;
        let mut io_off = 0usize;
        for dl in draw_data.draw_lists() {
            let vb = dl.vtx_buffer();
            let ib = dl.idx_buffer();
            unsafe {
                std::ptr::copy_nonoverlapping(vb.as_ptr() as *const u8, vtx_dst.add(vo), vb.len() * vtx_size_of);
                std::ptr::copy_nonoverlapping(ib.as_ptr() as *const u8, idx_dst.add(io_off), ib.len() * idx_size_of);
            }
            vo += vb.len() * vtx_size_of;
            io_off += ib.len() * idx_size_of;
        }

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
        unsafe {
            std::ptr::copy_nonoverlapping(proj.as_ptr(), self.proj_buf.contents() as *mut f32, 16);
        }

        let rp = RenderPassDescriptor::new();
        let att = rp.color_attachments().object_at(0).unwrap();
        att.set_texture(Some(drawable.texture()));
        att.set_load_action(MTLLoadAction::Clear);
        att.set_clear_color(MTLClearColor::new(0.094, 0.094, 0.094, 1.0));
        att.set_store_action(MTLStoreAction::Store);

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_render_command_encoder(rp);
        enc.set_render_pipeline_state(&self.pipeline);
        enc.set_vertex_buffer(1, Some(&self.proj_buf), 0);
        enc.set_fragment_texture(0, Some(&self.font_tex));
        enc.set_viewport(MTLViewport {
            originX: 0.0, originY: 0.0,
            width: fb_w as f64, height: fb_h as f64,
            znear: 0.0, zfar: 1.0,
        });

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;
        let idx_type = if idx_size_of == 2 { MTLIndexType::UInt16 } else { MTLIndexType::UInt32 };

        let mut global_vtx = 0usize;
        let mut global_idx = 0usize;
        for dl in draw_data.draw_lists() {
            for cmd in dl.commands() {
                match cmd {
                    DrawCmd::Elements { count, cmd_params } => {
                        let cx0 = (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0];
                        let cy0 = (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1];
                        let cx1 = (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0];
                        let cy1 = (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1];
                        if cx1 <= cx0 || cy1 <= cy0 { continue; }
                        let sx = (cx0.max(0.0)) as u64;
                        let sy = (cy0.max(0.0)) as u64;
                        let sw = ((cx1 - cx0.max(0.0)).max(1.0)) as u64;
                        let sh = ((cy1 - cy0.max(0.0)).max(1.0)) as u64;
                        enc.set_scissor_rect(MTLScissorRect { x: sx, y: sy, width: sw, height: sh });
                        enc.set_vertex_buffer(
                            0, Some(&self.vtx_buf),
                            (global_vtx + cmd_params.vtx_offset * vtx_size_of) as u64,
                        );
                        enc.draw_indexed_primitives(
                            MTLPrimitiveType::Triangle, count as u64, idx_type,
                            &self.idx_buf,
                            (global_idx + cmd_params.idx_offset * idx_size_of) as u64,
                        );
                    }
                    _ => {}
                }
            }
            global_vtx += dl.vtx_buffer().len() * vtx_size_of;
            global_idx += dl.idx_buffer().len() * idx_size_of;
        }

        enc.end_encoding();
        cb.present_drawable(drawable);
        cb.commit();
        cb.wait_until_scheduled();
    }
}

fn layer_ref(raw: &raw_window_metal::Layer) -> &MetalLayerRef {
    unsafe { &*(raw.as_ptr().as_ptr() as *const MetalLayerRef) }
}
