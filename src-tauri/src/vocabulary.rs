use crate::state::VocabularyEntry;
use std::path::PathBuf;

/// Generate a hotwords file for sherpa-onnx from vocabulary entries.
/// Format: one entry per line, `word :boost_score`
/// Returns the path to the generated file, or None if vocabulary is empty.
pub fn generate_hotwords_file(
    entries: &[VocabularyEntry],
    app_data_dir: &std::path::Path,
) -> Result<Option<PathBuf>, String> {
    if entries.is_empty() {
        return Ok(None);
    }

    let hotwords_path = app_data_dir.join("hotwords.txt");
    let mut content = String::new();

    for entry in entries {
        if entry.word.trim().is_empty() {
            continue;
        }
        // sherpa-onnx format: "word :score" (one per line)
        content.push_str(&format!("{} :{:.1}\n", entry.word.trim(), entry.boost));
    }

    if content.is_empty() {
        return Ok(None);
    }

    std::fs::write(&hotwords_path, &content)
        .map_err(|e| format!("Failed to write hotwords file: {e}"))?;

    log::info!("Generated hotwords file with {} entries at {}", entries.len(), hotwords_path.display());
    Ok(Some(hotwords_path))
}
