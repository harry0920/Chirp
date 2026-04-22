//! Windows `WH_KEYBOARD_LL` hook used *only* for suppression.
//!
//! Detection happens in `hotkey_raw_input`. This module's single job is to
//! prevent hotkey keys from reaching the focused application while the combo
//! is pressed. It reads `ActiveCombo` (set by the Raw Input thread) and
//! returns 1 from the hook when the current VK is in the configured combo
//! AND the combo is currently active.
//!
//! Why split detection from suppression? `WH_KEYBOARD_LL` is fragile — the
//! OS silently unhooks it under load, during UAC prompts, or on
//! resume-from-sleep. When that happens, detection via Raw Input keeps
//! working; the user loses suppression but the app still responds to the
//! hotkey. If this hook dies, it stays dead — no revival, no heartbeat.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows_sys::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, SetWindowsHookExW, TranslateMessage,
    UnhookWindowsHookEx, HHOOK, KBDLLHOOKSTRUCT, MSG, WH_KEYBOARD_LL,
};

struct HookConfig {
    vk_set: HashSet<u16>,
    active_combo: Arc<crate::hotkey::ActiveCombo>,
}

static HOOK_CONFIG: OnceLock<Mutex<Option<HookConfig>>> = OnceLock::new();
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
static THREAD_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

fn config_slot() -> &'static Mutex<Option<HookConfig>> {
    HOOK_CONFIG.get_or_init(|| Mutex::new(None))
}

pub fn start(
    vk_set: HashSet<u16>,
    active_combo: Arc<crate::hotkey::ActiveCombo>,
) -> Result<(), String> {
    stop();
    SHUTDOWN.store(false, Ordering::SeqCst);

    if let Ok(mut slot) = config_slot().lock() {
        *slot = Some(HookConfig { vk_set, active_combo });
    }

    let handle = std::thread::Builder::new()
        .name("hotkey-ll-suppress".into())
        .spawn(move || unsafe {
            let hook: HHOOK =
                SetWindowsHookExW(WH_KEYBOARD_LL, Some(hook_proc), std::ptr::null_mut(), 0);
            if hook.is_null() {
                log::error!("SetWindowsHookExW(WH_KEYBOARD_LL) returned null");
                return;
            }

            let mut msg: MSG = std::mem::zeroed();
            while !SHUTDOWN.load(Ordering::SeqCst) {
                let r = GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0);
                if r <= 0 {
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }

            UnhookWindowsHookEx(hook);
        })
        .map_err(|e| format!("Failed to spawn LL suppression thread: {e}"))?;

    if let Ok(mut guard) = THREAD_HANDLE.lock() {
        *guard = Some(handle);
    }
    Ok(())
}

pub fn stop() {
    SHUTDOWN.store(true, Ordering::SeqCst);
    if let Ok(mut slot) = config_slot().lock() {
        *slot = None;
    }
    if let Ok(mut guard) = THREAD_HANDLE.lock() {
        *guard = None;
    }
}

unsafe extern "system" fn hook_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code < 0 {
        return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
    }

    let kb = &*(lparam as *const KBDLLHOOKSTRUCT);
    let vk = kb.vkCode as u16;

    // Normalize generic VK_SHIFT/VK_CONTROL/VK_MENU to left/right using the
    // extended flag + scancode so the set membership check matches the
    // detection side.
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::*;
    const LLKHF_EXTENDED: u32 = 0x00000001;
    let extended = (kb.flags & LLKHF_EXTENDED) != 0;
    let resolved = match vk {
        VK_SHIFT => MapVirtualKeyW(kb.scanCode, 3) as u16,
        VK_CONTROL => if extended { VK_RCONTROL } else { VK_LCONTROL },
        VK_MENU => if extended { VK_RMENU } else { VK_LMENU },
        other => other,
    };
    let resolved = if resolved == 0 { vk } else { resolved };

    let should_suppress = if let Ok(slot) = config_slot().lock() {
        if let Some(cfg) = slot.as_ref() {
            cfg.active_combo.load() && cfg.vk_set.contains(&resolved)
        } else {
            false
        }
    } else {
        false
    };

    if should_suppress {
        return 1;
    }
    CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
}
