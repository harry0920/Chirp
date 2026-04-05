#[cfg(not(windows))]
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::thread;
use std::time::Duration;

#[cfg(windows)]
use crate::clipboard_win;
#[cfg(windows)]
use crate::richtext;

/// Inject text at the current cursor position.
/// On Windows: Win32 clipboard with exclusion flags + optional rich text.
/// On macOS: arboard clipboard + Cmd+V (unchanged).
pub fn inject_text(text: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        inject_text_windows(text)
    }
    #[cfg(not(windows))]
    {
        inject_text_arboard(text)
    }
}

/// Windows: Win32 clipboard with exclusion flags, rich text support, Ctrl+V paste
#[cfg(windows)]
fn inject_text_windows(text: &str) -> Result<(), String> {
    // 1. Save current clipboard
    let saved = clipboard_win::save_clipboard()?;

    // 2. Generate rich text formats (only if text has structure)
    let html = richtext::text_to_cf_html(text);
    let rtf = richtext::text_to_rtf(text);

    if html.is_some() || rtf.is_some() {
        log::debug!("Rich text generated: HTML={}, RTF={}", html.is_some(), rtf.is_some());
    }

    // 3. Set clipboard with retry + verification
    let mut verified = false;
    for attempt in 0..5 {
        if let Err(e) = clipboard_win::set_clipboard_with_exclusion(
            text,
            html.as_deref(),
            rtf.as_deref(),
        ) {
            log::warn!("Clipboard set attempt {}/5: {e}", attempt + 1);
            thread::sleep(Duration::from_millis(20));
            continue;
        }

        thread::sleep(Duration::from_millis(30));

        match clipboard_win::verify_clipboard_text(text) {
            Ok(true) => {
                log::info!("Clipboard verified on attempt {}/5", attempt + 1);
                verified = true;
                break;
            }
            Ok(false) => {
                log::warn!("Clipboard mismatch on attempt {}/5", attempt + 1);
            }
            Err(e) => {
                log::warn!("Clipboard verify failed on attempt {}/5: {e}", attempt + 1);
            }
        }

        thread::sleep(Duration::from_millis(20));
    }

    if !verified {
        return Err("Failed to set clipboard — text did not persist after 5 attempts".into());
    }

    // 4. Simulate Ctrl+V
    // First, release all modifiers to clear any stale state from the hotkey combo.
    // Without this, Shift/Meta can appear "stuck" after injection.
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| format!("Failed to init enigo: {e}"))?;
    let _ = enigo.key(Key::Shift, Direction::Release);
    let _ = enigo.key(Key::Control, Direction::Release);
    let _ = enigo.key(Key::Meta, Direction::Release);
    let _ = enigo.key(Key::Alt, Direction::Release);

    enigo
        .key(Key::Control, Direction::Press)
        .map_err(|e| format!("Key press failed: {e}"))?;
    enigo
        .key(Key::Unicode('v'), Direction::Click)
        .map_err(|e| format!("Key click failed: {e}"))?;
    enigo
        .key(Key::Control, Direction::Release)
        .map_err(|e| format!("Key release failed: {e}"))?;

    // 5. Restore clipboard after delay (background thread)
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(3));
        if let Err(e) = clipboard_win::restore_clipboard(saved) {
            log::warn!("Failed to restore clipboard: {e}");
        }
    });

    Ok(())
}

/// macOS/Linux: arboard clipboard + Cmd+V / Ctrl+V paste (original implementation)
#[cfg(not(windows))]
fn inject_text_arboard(text: &str) -> Result<(), String> {
    // Save current clipboard content using a short-lived Clipboard instance
    let saved = {
        let mut cb = Clipboard::new().map_err(|e| format!("Failed to access clipboard: {e}"))?;
        let s = cb.get_text().ok();
        log::debug!(
            "Clipboard before inject: {:?}",
            s.as_deref().map(|t| if t.len() > 80 { &t[..80] } else { t })
        );
        s
    };

    // Set new text with retry loop and read-back verification.
    let mut verified = false;
    for attempt in 0..5 {
        {
            let mut cb =
                Clipboard::new().map_err(|e| format!("Failed to access clipboard: {e}"))?;
            if let Err(e) = cb.set_text(text.to_string()) {
                log::warn!("Clipboard set_text attempt {}/5: {e}", attempt + 1);
                thread::sleep(Duration::from_millis(20));
                continue;
            }
        }

        // Small delay then verify with a fresh handle
        thread::sleep(Duration::from_millis(30));

        {
            let mut cb =
                Clipboard::new().map_err(|e| format!("Failed to access clipboard: {e}"))?;
            match cb.get_text() {
                Ok(current) if current == text => {
                    log::info!("Clipboard verified on attempt {}/5", attempt + 1);
                    verified = true;
                    break;
                }
                Ok(current) => {
                    log::warn!(
                        "Clipboard mismatch on attempt {}/5: expected '{}', got '{}'",
                        attempt + 1,
                        if text.len() > 60 { &text[..60] } else { text },
                        if current.len() > 60 {
                            &current[..60]
                        } else {
                            &current
                        }
                    );
                }
                Err(e) => {
                    log::warn!("Clipboard read-back failed on attempt {}/5: {e}", attempt + 1);
                }
            }
        }

        thread::sleep(Duration::from_millis(20));
    }

    if !verified {
        return Err("Failed to set clipboard — text did not persist after 5 attempts".into());
    }

    // Simulate Cmd+V (macOS) or Ctrl+V (other platforms)
    let paste_modifier = if cfg!(target_os = "macos") {
        Key::Meta
    } else {
        Key::Control
    };

    let mut enigo =
        Enigo::new(&Settings::default()).map_err(|e| {
            let msg = format!("{e}");
            // On macOS, enigo fails with a permission error when Accessibility is not granted
            if cfg!(target_os = "macos") && (msg.contains("accessibility") || msg.contains("permission") || msg.contains("trusted")) {
                "accessibility_denied".to_string()
            } else {
                format!("Failed to init enigo: {e}")
            }
        })?;
    enigo
        .key(paste_modifier, Direction::Press)
        .map_err(|e| format!("Key press failed: {e}"))?;
    enigo
        .key(Key::Unicode('v'), Direction::Click)
        .map_err(|e| format!("Key click failed: {e}"))?;
    enigo
        .key(paste_modifier, Direction::Release)
        .map_err(|e| format!("Key release failed: {e}"))?;

    // Restore clipboard in a background thread after a generous delay.
    if let Some(original) = saved {
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(3));
            match Clipboard::new() {
                Ok(mut cb) => {
                    if let Err(e) = cb.set_text(original) {
                        log::warn!("Failed to restore clipboard: {e}");
                    } else {
                        log::debug!("Clipboard restored after 3s delay");
                    }
                }
                Err(e) => log::warn!("Failed to open clipboard for restore: {e}"),
            }
        });
    }

    Ok(())
}
