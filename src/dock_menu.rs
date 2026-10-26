//! macOS Dock right-click menu: a "New Window" item that spawns a whole new
//! process (what `open -n` does), since winit apps are single-window and
//! the plain Dock click only activates the running instance. The menu is
//! installed on NSApplication after winit creates it in `EventLoop::new()`.
//!
//! Written against objc2 0.5 / objc2-app-kit 0.2 — the same versions those
//! crates use among themselves in this project's lockfile, so there is
//! exactly one copy of each in the build.
#![cfg(target_os = "macos")]

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2::{declare_class, msg_send, msg_send_id, sel, ClassType, DeclaredClass};
use objc2_app_kit::{NSApplication, NSMenu, NSMenuItem};
use objc2_foundation::{MainThreadMarker, NSString};

declare_class!(
    struct TVDockMenuTarget;

    // SAFETY:
    // - The superclass NSObject does not have any subclassing requirements.
    // - Interior mutability is a safe default.
    // - `TVDockMenuTarget` does not implement `Drop`.
    unsafe impl ClassType for TVDockMenuTarget {
        type Super = NSObject;
        type Mutability = objc2::mutability::InteriorMutable;
        const NAME: &'static str = "TVDockMenuTarget";
    }

    impl DeclaredClass for TVDockMenuTarget {
        type Ivars = ();
    }

    unsafe impl TVDockMenuTarget {
        // The menu item's action target; `sender` is the item, unused.
        #[method(newInstance:)]
        fn new_instance(&self, _sender: Option<&AnyObject>) {
            if let Ok(exe) = std::env::current_exe() {
                let _ = new_instance_command(&exe).spawn();
            }
        }
    }
);

/// The command that launches one more instance of this app. Bundled, it goes
/// through LaunchServices (`open -n`) so the new process gets its own Dock
/// tile and proper activation; running as a bare binary it just re-execs
/// the same path. Split out so the mapping is unit-testable.
pub(crate) fn new_instance_command(exe: &std::path::Path) -> std::process::Command {
    let path = exe.to_string_lossy();
    if let Some(idx) = path.find(".app/") {
        let mut cmd = std::process::Command::new("open");
        cmd.arg("-n").arg(path[..idx + ".app".len()].to_string());
        cmd
    } else {
        std::process::Command::new(exe)
    }
}

pub fn install() {
    let Some(mtm) = MainThreadMarker::new() else { return };
    let app = NSApplication::sharedApplication(mtm);
    let target: Retained<TVDockMenuTarget> = unsafe { msg_send_id![TVDockMenuTarget::class(), new] };
    let item = NSMenuItem::new(mtm);
    let menu = NSMenu::new(mtm);
    unsafe {
        item.setTitle(&NSString::from_str("New Window"));
        item.setAction(Some(sel!(newInstance:)));
        // setTarget: takes a bare `id`, so go through msg_send to hand it the
        // subclass instance without an upcast; same for NSApplication's
        // setDockMenu:, which objc2-app-kit 0.2 doesn't expose.
        let _: () = msg_send![&item, setTarget: &*target];
        menu.setAutoenablesItems(false);
        menu.addItem(&item);
        let _: () = msg_send![&*app, setDockMenu: &*menu];
    }
    // NSMenuItem does not guarantee retaining its target; this object must
    // outlive the menu, so leak the single instance for the app's lifetime.
    std::mem::forget(target);
}

#[cfg(test)]
mod tests {
    use super::new_instance_command;
    use std::path::Path;

    fn argv(cmd: &std::process::Command) -> Vec<String> {
        std::iter::once(cmd.get_program().to_string_lossy().into_owned())
            .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
            .collect()
    }

    #[test]
    fn bundled_exe_launches_via_open_dash_n() {
        let cmd = new_instance_command(Path::new("/Apps/TraceViewer.app/Contents/MacOS/tv"));
        assert_eq!(argv(&cmd), vec!["open", "-n", "/Apps/TraceViewer.app"]);
    }

    #[test]
    fn bare_exe_reexecs_itself() {
        let cmd = new_instance_command(Path::new("/Users/x/trace-viewer/target/release/tv"));
        assert_eq!(argv(&cmd), vec!["/Users/x/trace-viewer/target/release/tv"]);
    }
}
