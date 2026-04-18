use std::path::{Path, PathBuf};
use tauri::{AppHandle, Emitter};

use crate::settings;

/// Check if a filename is a binary we want to extract
fn is_wanted_binary(basename: &str) -> bool {
    if cfg!(windows) {
        (basename.ends_with(".exe") && basename.contains("llama-server"))
            || basename.ends_with(".dll")
    } else {
        basename == "llama-server"
            || basename.ends_with(".dylib")
            || basename.ends_with(".so")
    }
}

/// Extract llama-server binary from either a .zip or .tar.gz archive
fn extract_binary_archive(archive_path: &Path, dest_dir: &Path, is_targz: bool) -> Result<(), String> {
    let mut found_server = false;

    if is_targz {
        let file = std::fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive: {e}"))?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(gz);

        for entry in archive.entries().map_err(|e| format!("Failed to read tar: {e}"))? {
            let mut entry = entry.map_err(|e| format!("Failed to read tar entry: {e}"))?;
            let basename = entry.path()
                .map_err(|e| format!("Invalid tar path: {e}"))?
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            if basename.is_empty() || basename.contains("..") { continue; }

            if is_wanted_binary(&basename) {
                let dest_file = dest_dir.join(&basename);
                // Tar symlinks (e.g. libfoo.0.dylib → libfoo.0.9.8.dylib) have
                // no file content. Create a proper symlink on Unix, or skip on
                // other platforms (the versioned file is already extracted).
                if entry.header().entry_type() == tar::EntryType::Symlink {
                    #[cfg(unix)]
                    if let Ok(link_target) = entry.link_name() {
                        if let Some(target) = link_target {
                            let target_name = target.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("");
                            if !target_name.is_empty() && !target_name.contains("..") {
                                let _ = std::fs::remove_file(&dest_file);
                                let _ = std::os::unix::fs::symlink(target_name, &dest_file);
                            }
                        }
                    }
                    continue;
                }
                let mut out = std::fs::File::create(&dest_file)
                    .map_err(|e| format!("Failed to create {basename}: {e}"))?;
                std::io::copy(&mut entry, &mut out)
                    .map_err(|e| format!("Failed to extract {basename}: {e}"))?;
                if basename.contains("llama-server") {
                    found_server = true;
                }
            }
        }
    } else {
        let file = std::fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open zip: {e}"))?;
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| format!("Failed to read zip: {e}"))?;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)
                .map_err(|e| format!("Failed to read zip entry: {e}"))?;
            let name = entry.name().to_string();
            let basename = name.rsplit('/').next().unwrap_or(&name);

            if basename.contains("..") || basename.contains('/') || basename.contains('\\')
                || std::path::Path::new(basename).is_absolute()
            {
                log::warn!("Skipping suspicious filename in archive: {basename}");
                continue;
            }

            if is_wanted_binary(basename) {
                let dest_file = dest_dir.join(basename);
                let mut out = std::fs::File::create(&dest_file)
                    .map_err(|e| format!("Failed to create {basename}: {e}"))?;
                std::io::copy(&mut entry, &mut out)
                    .map_err(|e| format!("Failed to extract {basename}: {e}"))?;
                if basename.contains("llama-server") {
                    found_server = true;
                }
            }
        }
    }

    if !found_server {
        return Err("llama-server binary not found in archive".to_string());
    }

    // Set executable permission on Unix platforms
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let server_path = dest_dir.join("llama-server");
        if server_path.exists() {
            let _ = std::fs::set_permissions(
                &server_path,
                std::fs::Permissions::from_mode(0o755),
            );
        }
    }

    Ok(())
}

const BASE_SYSTEM_PROMPT: &str = "\
You clean up speech-to-text output. Your job is to make it read like the person typed it, while preserving every piece of information they communicated.

Do:
- Fix ASR errors (misheard words) using context clues
- Remove filler words (um, uh, like, you know, so, basically)
- Remove stutters and word repetitions
- Keep only the corrected version when someone corrects themselves
- Combine fragmented sentences into clear prose
- When the same idea is stated multiple times, state it once clearly

Do not:
- Add information the speaker did not say
- Summarize or omit meaningful content
- Add formatting, headers, or bullet points

Example 1:
Input: We were um looking at the the new model and it's basically it's really fast actually no it's not that fast but it's faster than what we had before
Output: We were looking at the new model. It's faster than what we had before, though not extremely fast.

Example 2:
Input: So I think we should um we should probably move the the database to actually no not the database the cache to Redis because it's faster
Output: I think we should move the cache to Redis because it's faster.

Output only the cleaned text.";

const EMAIL_SYSTEM_PROMPT: &str = "\
You clean up speech-to-text output and format it as an email. Your job is to make it read like the person typed the email directly.

Do:
- Fix ASR errors (misheard words) using context clues
- Remove filler words (um, uh, like, you know, so, basically)
- Remove stutters and word repetitions
- Keep only the corrected version when someone corrects themselves
- Combine fragmented sentences into clear prose
- When the same idea is stated multiple times, state it once clearly

Do not:
- Add information the speaker did not say
- Summarize or omit meaningful content

Email formatting:
- If the speech starts with a greeting (Hey/Hi/Hello/Dear + name), format as a full email: greeting on its own line, blank line, body paragraphs, blank line, sign-off
- If the speech ends with a sign-off but no greeting, add a blank line before the sign-off
- If there is no greeting or sign-off, just clean up the text normally

Output only the cleaned text.";

fn system_prompt_for_mode(mode: &str) -> &'static str {
    match mode {
        "email" => EMAIL_SYSTEM_PROMPT,
        _ => BASE_SYSTEM_PROMPT,
    }
}

// Gemma 4 E2B Instruct (Google, Apache 2.0). Mobile/edge-friendly 3.11 GB
// quant via the unsloth GGUF mirror. Chosen for mobile/MLX portability and
// because it handled the plain-text + in-prompt few-shot format cleanly in
// prior dogfooding (v1.2.6 shipped this exact config).
const MODEL_FILENAME: &str = "gemma-4-E2B-it-Q4_K_M.gguf";
const MODEL_URL: &str = "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf";
const MODEL_SIZE: u64 = 3_110_000_000;

/// llama-server release info
const LLAMA_CPP_VERSION: &str = "b8653";

fn llama_server_url() -> String {
    let (platform_suffix, ext) = if cfg!(target_os = "windows") {
        ("bin-win-vulkan-x64", "zip")
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            ("bin-macos-arm64", "tar.gz")
        } else {
            ("bin-macos-x64", "tar.gz")
        }
    } else {
        ("bin-ubuntu-x64", "tar.gz")
    };
    format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{}/llama-{}-{}.{ext}",
        LLAMA_CPP_VERSION, LLAMA_CPP_VERSION, platform_suffix
    )
}

/// Directory for LLM files: %APPDATA%/com.chirp.app/llm/
pub fn llm_dir() -> PathBuf {
    settings::config_dir().join("llm")
}

fn binary_path() -> PathBuf {
    if cfg!(windows) {
        llm_dir().join("llama-server.exe")
    } else {
        llm_dir().join("llama-server")
    }
}

fn model_path() -> PathBuf {
    llm_dir().join(MODEL_FILENAME)
}

fn version_marker_path() -> PathBuf {
    llm_dir().join("llama-server.version")
}

/// Check if llama-server binary exists and matches the expected version.
pub fn binary_exists() -> bool {
    if !binary_path().exists() {
        return false;
    }
    // If the version marker is missing or stale, the binary needs to be replaced.
    match std::fs::read_to_string(version_marker_path()) {
        Ok(v) if v.trim() == LLAMA_CPP_VERSION => true,
        _ => false,
    }
}

/// Check if the model GGUF exists
pub fn model_exists() -> bool {
    model_path().exists()
}

/// Download llama-server binary from llama.cpp releases
pub async fn download_binary(app_handle: &AppHandle) -> Result<(), String> {
    let dir = llm_dir();
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("Failed to create LLM dir: {e}"))?;

    // Skip download if binary exists and version matches
    if binary_exists() {
        return Ok(());
    }

    // Clean up stale binaries from a previous llama.cpp version
    let dest = binary_path();
    if dest.exists() {
        log::info!("Removing stale llama-server (upgrading to {})", LLAMA_CPP_VERSION);
        // Remove all DLLs/dylibs and the old binary so we get a clean slate
        if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".dll") || name_str.ends_with(".dylib")
                    || name_str.ends_with(".so") || name_str.contains("llama-server")
                {
                    let _ = tokio::fs::remove_file(entry.path()).await;
                }
            }
        }
        let _ = tokio::fs::remove_file(version_marker_path()).await;
    }

    let url = llama_server_url();
    let is_targz = url.ends_with(".tar.gz");
    let archive_ext = if is_targz { "tar.gz" } else { "zip" };
    let archive_path = dir.join(format!("llama-server.{archive_ext}"));
    let tmp_path = dir.join(format!("llama-server.{archive_ext}.tmp"));

    // Download the archive
    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(15_000_000);
    let mut downloaded: u64 = 0;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;

        downloaded += chunk.len() as u64;
        let progress = ((downloaded as f64 / total_size as f64) * 90.0).min(90.0) as u32;
        let _ = app_handle.emit("llm-download-progress", progress);
    }
    drop(file);

    tokio::fs::rename(&tmp_path, &archive_path)
        .await
        .map_err(|e| format!("Failed to finalize download: {e}"))?;

    let _ = app_handle.emit("llm-download-progress", 95u32);

    // Extract llama-server binary (and DLLs on Windows) from the archive
    let archive_clone = archive_path.clone();
    let dir_clone = dir.clone();
    tokio::task::spawn_blocking(move || {
        extract_binary_archive(&archive_clone, &dir_clone, is_targz)
    })
    .await
    .map_err(|e| format!("Extract task failed: {e}"))??;

    // Clean up archive
    if let Err(e) = tokio::fs::remove_file(&archive_path).await {
        log::warn!("Failed to clean up LLM archive: {e}");
    }

    // Write version marker so we know this binary matches our expected version
    let _ = tokio::fs::write(version_marker_path(), LLAMA_CPP_VERSION).await;
    log::info!("llama-server {} downloaded and extracted", LLAMA_CPP_VERSION);

    let _ = app_handle.emit("llm-download-progress", 100u32);
    Ok(())
}

/// Download the model GGUF
pub async fn download_model(app_handle: &AppHandle) -> Result<(), String> {
    let dir = llm_dir();
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("Failed to create LLM dir: {e}"))?;

    let dest = model_path();
    if dest.exists() {
        return Ok(());
    }

    // Clean up old model files from previous versions
    if let Ok(mut entries) = tokio::fs::read_dir(&dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".gguf") && name_str.as_ref() != MODEL_FILENAME {
                log::info!("Removing old model file: {}", name_str);
                let _ = tokio::fs::remove_file(entry.path()).await;
            }
        }
    }

    let file_name = dest.file_name()
        .ok_or_else(|| "Model path has no filename".to_string())?;
    let tmp_path = dir.join(format!("{}.tmp", file_name.to_string_lossy()));

    let client = reqwest::Client::new();
    let response = client
        .get(MODEL_URL)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(MODEL_SIZE);
    let mut downloaded: u64 = 0;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;

        downloaded += chunk.len() as u64;
        let progress = ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u32;
        let _ = app_handle.emit("llm-download-progress", progress);
    }
    drop(file);

    tokio::fs::rename(&tmp_path, &dest)
        .await
        .map_err(|e| format!("Failed to finalize model download: {e}"))?;

    Ok(())
}

/// Start llama-server on a given port. Returns the child process.
pub async fn start_server(port: u16) -> Result<tokio::process::Child, String> {
    let binary = binary_path();
    let model = model_path();

    if !binary.exists() {
        return Err("llama-server binary not found".to_string());
    }
    if !model.exists() {
        return Err("Model not found".to_string());
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut cmd = tokio::process::Command::new(binary.to_string_lossy().to_string());
    cmd.arg("--model")
        .arg(model.to_string_lossy().to_string())
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg("2048")
        .arg("--n-predict")
        .arg("512")
        .arg("--threads")
        .arg(n_threads.to_string())
        .arg("--gpu-layers")
        .arg("99")
        .arg("--flash-attn").arg("on")
        .arg("--batch-size")
        .arg("512")
        .arg("--parallel")
        .arg("1")
        .arg("--reasoning-budget").arg("0")
        // Use the Jinja chat template embedded in the GGUF. Without this,
        // llama-server falls back to a generic template that doesn't match
        // Qwen3's training format, which combined with greedy decoding and
        // few-shot-in-system caused total mode collapse on stock Qwen3-0.6B.
        .arg("--jinja")
        .arg("--log-disable")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        #[allow(unused_imports)]
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to start llama-server: {e}"))?;

    // Wait for server to be ready
    let health_url = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::new();

    for _ in 0..60 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if let Ok(resp) = client.get(&health_url).send().await {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                if body.get("status").and_then(|s| s.as_str()) == Some("ok") {
                    log::info!("llama-server ready on port {port}");
                    return Ok(child);
                }
            }
        }
    }

    // Kill the orphan process before returning error
    let _ = child.kill().await;
    let _ = child.wait().await;
    log::warn!("Killed llama-server after startup timeout");

    Err("llama-server failed to start within 30s".to_string())
}

/// Stop a running llama-server process
pub async fn stop_server(child: &mut tokio::process::Child) {
    let _ = child.kill().await;
    let _ = child.wait().await;
    log::info!("llama-server stopped");
}

/// Send text through the LLM for cleanup.
///
/// Vocabulary biasing happens at the ASR layer (sherpa-onnx hotwords) and via
/// the post-ASR find/replace pass, not via prompt injection. We do not append
/// vocab to the system prompt — small models tend to hallucinate when given
/// out-of-distribution instructions, and vocab terms can land in unrelated
/// places in the output.
pub async fn cleanup_text(port: u16, text: &str, tone_mode: &str, client: &reqwest::Client) -> Result<String, String> {
    let system_prompt = system_prompt_for_mode(tone_mode);

    // Word-count-based max_tokens. With the regex pre-pass, output should be
    // close to input length or shorter; clamp keeps runaway generation bounded.
    let word_count = text.split_whitespace().count();
    let max_tokens = ((word_count as f64 * 1.5) as usize).clamp(64, 512);

    let payload = serde_json::json!({
        "model": "gemma",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": max_tokens,
        "stream": false,
        "cache_prompt": true,
    });

    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("LLM request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("LLM returned status: {}", resp.status()));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse LLM response: {e}"))?;

    let result = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or(text)
        .trim()
        .to_string();

    // Guard: if LLM returned empty, fall back to original text
    if result.is_empty() {
        log::warn!("LLM returned empty response, using original text");
        return Ok(text.to_string());
    }

    // Prompt-injection defense: if output is much longer than input, the LLM
    // likely followed the text as an instruction instead of cleaning it.
    let input_words = text.split_whitespace().count();
    let output_words = result.split_whitespace().count();
    if output_words > input_words * 3 / 2 + 10 {
        log::warn!(
            "Cleanup output ({output_words} words) much longer than input ({input_words} words), using original"
        );
        return Ok(text.to_string());
    }

    Ok(result)
}

// ── PID file management ──────────────────────────────────────────────
// Persists the llama-server PID so we can clean up orphans after crashes.

fn pid_file_path() -> PathBuf {
    llm_dir().join("llama-server.pid")
}

/// Save the llama-server PID to a file for crash recovery.
pub fn save_server_pid(pid: u32) {
    let dir = llm_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(pid_file_path(), pid.to_string());
}

/// Kill a stale llama-server from a previous session (crash recovery) and remove the PID file.
pub fn kill_stale_server() {
    let path = pid_file_path();
    if let Ok(pid_str) = std::fs::read_to_string(&path) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            log::info!("Killing stale llama-server (PID {pid})");
            #[cfg(windows)]
            {
                #[allow(unused_imports)]
                use std::os::windows::process::CommandExt;
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .output();
            }
            #[cfg(unix)]
            {
                let _ = std::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .output();
            }
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// Remove the PID file (called on clean shutdown).
pub fn clear_server_pid() {
    let _ = std::fs::remove_file(pid_file_path());
}

/// LLM status for frontend
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmStatus {
    pub binary_downloaded: bool,
    pub model_downloaded: bool,
    pub server_running: bool,
}

