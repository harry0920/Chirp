use crate::state::{Settings, SnippetEntry};
use std::path::PathBuf;

/// Migrate old Tauri shortcut format (e.g., "CmdOrCtrl+Shift+Space") to new
/// event.code-based format (e.g., "MetaLeft+ShiftLeft+Space").
fn migrate_hotkey(hotkey: &str) -> String {
    // Quick check: if it already uses new-style identifiers, return as-is
    let has_new_style = hotkey.contains("Left") || hotkey.contains("Right")
        || hotkey.contains("Key") || hotkey.contains("Digit") || hotkey == "Fn";
    let has_old_style = hotkey.contains("CmdOrCtrl") || hotkey.contains("Cmd")
        || (hotkey.contains("Ctrl") && !hotkey.contains("Control"))
        || (hotkey.contains("Shift") && !hotkey.contains("ShiftLeft") && !hotkey.contains("ShiftRight"))
        || (hotkey.contains("Alt") && !hotkey.contains("AltGr"));

    if has_new_style && !has_old_style {
        return hotkey.to_string();
    }
    if !has_old_style {
        return hotkey.to_string();
    }

    let parts: Vec<&str> = hotkey.split('+').collect();
    let mut new_parts: Vec<String> = Vec::new();

    for part in parts {
        let migrated = match part.trim() {
            "CmdOrCtrl" => {
                if cfg!(target_os = "macos") { "MetaLeft" } else { "ControlLeft" }
            }
            "Ctrl" | "Control" => "ControlLeft",
            "Cmd" | "Command" | "Meta" | "Super" => "MetaLeft",
            "Shift" => "ShiftLeft",
            "Alt" | "Option" => "Alt",
            "Space" => "Space",
            "Tab" => "Tab",
            "Backspace" => "Backspace",
            "Delete" => "Delete",
            "Enter" | "Return" => "Enter",
            "Escape" | "Esc" => "Escape",
            "Up" => "ArrowUp",
            "Down" => "ArrowDown",
            "Left" => "ArrowLeft",
            "Right" => "ArrowRight",
            s if s.len() == 1 && s.chars().next().unwrap().is_ascii_alphabetic() => {
                new_parts.push(format!("Key{}", s.to_uppercase()));
                continue;
            }
            s if s.starts_with('F') && s[1..].parse::<u32>().is_ok() => s,
            other => other,
        };
        new_parts.push(migrated.to_string());
    }

    new_parts.join("+")
}

/// Get the app config directory (%APPDATA%/com.chirp.app/)
pub fn config_dir() -> PathBuf {
    let base = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join("com.chirp.app")
}

/// Get the models directory
pub fn models_dir() -> PathBuf {
    config_dir().join("models")
}

fn settings_path() -> PathBuf {
    config_dir().join("settings.json")
}

fn vocabulary_path() -> PathBuf {
    config_dir().join("vocabulary.json")
}

/// Remove old model files from previous versions (Qwen GGUF, chirp-cleanup T5)
fn cleanup_old_models() {
    let llm_dir = config_dir().join("llm");
    let qwen_path = llm_dir.join("qwen2.5-3b-instruct-q4_k_m.gguf");
    if qwen_path.exists() {
        match std::fs::remove_file(&qwen_path) {
            Ok(_) => log::info!("Cleaned up old Qwen model ({})", qwen_path.display()),
            Err(e) => log::warn!("Failed to remove old Qwen model: {e}"),
        }
    }

    let chirp_cleanup_dir = config_dir().join("chirp-cleanup");
    if chirp_cleanup_dir.exists() {
        match std::fs::remove_dir_all(&chirp_cleanup_dir) {
            Ok(_) => log::info!("Cleaned up old chirp-cleanup model dir"),
            Err(e) => log::warn!("Failed to remove chirp-cleanup dir: {e}"),
        }
    }
}

/// Load settings from disk, returning defaults if file doesn't exist
pub fn load_settings() -> Settings {
    let path = settings_path();
    let mut settings = match std::fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
            log::warn!("Corrupted settings JSON, using defaults: {e}");
            Settings::default()
        }),
        Err(_) => Settings::default(),
    };

    // Migrate old whisper model IDs to new default
    match settings.model.as_str() {
        "tiny" | "base" | "small" | "medium" => {
            settings.model = "parakeet-tdt-0.6b".into();
        }
        _ => {}
    }

    // Migrate old cleanup model to Gemma
    if settings.cleanup_model != "gemma" {
        log::info!("Migrated cleanup_model '{}' → 'gemma'", settings.cleanup_model);
        settings.cleanup_model = "gemma".into();
    }

    // Clean up old model files (Qwen, chirp-cleanup) in background
    cleanup_old_models();

    // Migrate old Tauri shortcut format to new event.code-based format
    let migrated = migrate_hotkey(&settings.hotkey);
    if migrated != settings.hotkey {
        log::info!("Migrated hotkey '{}' → '{}'", settings.hotkey, migrated);
        settings.hotkey = migrated;
        let _ = save_settings(&settings);
    }

    settings
}

/// Save settings to disk
pub fn save_settings(settings: &Settings) -> Result<(), String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create config dir: {e}"))?;
    let data =
        serde_json::to_string_pretty(settings).map_err(|e| format!("Failed to serialize: {e}"))?;
    std::fs::write(settings_path(), data).map_err(|e| format!("Failed to write settings: {e}"))
}

/// Load vocabulary from disk
pub fn load_vocabulary() -> Vec<String> {
    let path = vocabulary_path();
    match std::fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
            log::warn!("Corrupted vocabulary JSON, resetting: {e}");
            Vec::new()
        }),
        Err(_) => {
            // Migrate from old dictionary.json if it exists
            let old_path = config_dir().join("dictionary.json");
            if let Ok(old_data) = std::fs::read_to_string(&old_path) {
                #[derive(serde::Deserialize)]
                struct OldEntry { #[allow(dead_code)] from: String, to: String }
                if let Ok(entries) = serde_json::from_str::<Vec<OldEntry>>(&old_data) {
                    let vocab: Vec<String> = entries.into_iter().map(|e| e.to).collect();
                    if !vocab.is_empty() {
                        log::info!("Migrated {} dictionary entries to vocabulary", vocab.len());
                        let _ = save_vocabulary(&vocab);
                        let _ = std::fs::remove_file(&old_path);
                        return vocab;
                    }
                }
            }
            Vec::new()
        }
    }
}

fn snippets_path() -> PathBuf {
    config_dir().join("snippets.json")
}

/// Load snippets from disk, providing defaults on first run
pub fn load_snippets() -> Vec<SnippetEntry> {
    let path = snippets_path();
    match std::fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
            log::warn!("Corrupted snippets JSON, resetting: {e}");
            default_snippets()
        }),
        Err(_) => default_snippets(),
    }
}

/// Save snippets to disk
pub fn save_snippets(entries: &[SnippetEntry]) -> Result<(), String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create config dir: {e}"))?;
    let data = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("Failed to serialize: {e}"))?;
    std::fs::write(snippets_path(), data)
        .map_err(|e| format!("Failed to write snippets: {e}"))
}

fn default_snippets() -> Vec<SnippetEntry> {
    vec![
        SnippetEntry {
            trigger: "my email address".into(),
            expansion: "user@example.com".into(),
        },
        SnippetEntry {
            trigger: "my signature".into(),
            expansion: "Best regards,\n[Your Name]".into(),
        },
    ]
}

/// Path to the Silero VAD model
pub fn vad_model_path() -> PathBuf {
    models_dir().join("silero_vad.onnx")
}

/// Check if the VAD model exists on disk
pub fn vad_model_exists() -> bool {
    vad_model_path().exists()
}

const VAD_MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

/// Download the Silero VAD model (~2MB)
pub async fn download_vad_model() -> Result<(), String> {
    let dest = vad_model_path();
    if dest.exists() {
        return Ok(());
    }

    let dir = models_dir();
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("Failed to create models dir: {e}"))?;

    let tmp_path = dest.with_extension("onnx.tmp");

    let client = reqwest::Client::new();
    let response = client
        .get(VAD_MODEL_URL)
        .send()
        .await
        .map_err(|e| format!("VAD model download failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("VAD download failed with status: {}", response.status()));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read VAD model: {e}"))?;

    tokio::fs::write(&tmp_path, &bytes)
        .await
        .map_err(|e| format!("Failed to write VAD model: {e}"))?;

    tokio::fs::rename(&tmp_path, &dest)
        .await
        .map_err(|e| format!("Failed to finalize VAD model: {e}"))?;

    log::info!("Silero VAD model downloaded ({} bytes)", bytes.len());
    Ok(())
}

/// Save vocabulary to disk
pub fn save_vocabulary(words: &[String]) -> Result<(), String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed to create config dir: {e}"))?;
    let data = serde_json::to_string_pretty(words)
        .map_err(|e| format!("Failed to serialize: {e}"))?;
    std::fs::write(vocabulary_path(), data)
        .map_err(|e| format!("Failed to write vocabulary: {e}"))
}
