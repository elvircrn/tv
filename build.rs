use std::path::PathBuf;

fn main() {
    let home = std::env::var("HOME").unwrap();
    let registry = PathBuf::from(&home).join(".cargo/registry/src");
    let Ok(entries) = std::fs::read_dir(&registry) else { return };
    for entry in entries.flatten() {
        let imgui_sys = entry.path().join("imgui-sys-0.12.0");
        if !imgui_sys.exists() { continue; }

        for variant in &["imgui-master", "imgui-master-freetype"] {
            let imconfig = imgui_sys.join(format!("third-party/{variant}/imgui/imconfig.h"));
            if let Ok(s) = std::fs::read_to_string(&imconfig) {
                if s.contains("//#define ImDrawIdx unsigned int") {
                    let patched = s.replace(
                        "//#define ImDrawIdx unsigned int",
                        "#define ImDrawIdx unsigned int",
                    );
                    std::fs::write(&imconfig, patched).ok();
                }
            }
            let cimgui = imgui_sys.join(format!("third-party/{variant}/cimgui.h"));
            if let Ok(s) = std::fs::read_to_string(&cimgui) {
                if s.contains("typedef unsigned short ImDrawIdx;") {
                    let patched = s.replace(
                        "typedef unsigned short ImDrawIdx;",
                        "typedef unsigned int ImDrawIdx;",
                    );
                    std::fs::write(&cimgui, patched).ok();
                }
            }
        }

        for bindings in &["src/bindings.rs", "src/freetype_bindings.rs"] {
            let path = imgui_sys.join(bindings);
            if let Ok(s) = std::fs::read_to_string(&path) {
                if s.contains("pub type ImDrawIdx = cty::c_ushort;") {
                    let patched = s.replace(
                        "pub type ImDrawIdx = cty::c_ushort;",
                        "pub type ImDrawIdx = cty::c_uint;",
                    );
                    std::fs::write(&path, patched).ok();
                }
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
