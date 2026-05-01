//! OS keychain wrapper for BYOK cleanup API keys.
//!
//! Keys are stored under service `chirp-cleanup` with the provider id as the
//! account name (e.g. `openai_compatible`, `anthropic`, `gemini`). Provider
//! ids are sanitized against an allowlist before being passed to the keyring
//! to prevent arbitrary entry creation.

const SERVICE: &str = "chirp-cleanup";

pub const PROVIDER_LOCAL: &str = "local";
pub const PROVIDER_OPENAI_COMPATIBLE: &str = "openai_compatible";
pub const PROVIDER_ANTHROPIC: &str = "anthropic";
pub const PROVIDER_GEMINI: &str = "gemini";

const CLOUD_PROVIDERS: &[&str] = &[
    PROVIDER_OPENAI_COMPATIBLE,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
];

pub fn is_known_provider(provider: &str) -> bool {
    provider == PROVIDER_LOCAL || CLOUD_PROVIDERS.contains(&provider)
}

fn entry(provider: &str) -> Result<keyring::Entry, String> {
    if !CLOUD_PROVIDERS.contains(&provider) {
        return Err(format!("Unknown cleanup provider: {provider}"));
    }
    keyring::Entry::new(SERVICE, provider).map_err(|e| format!("Keychain entry error: {e}"))
}

/// Read a stored API key. Returns None if the entry is missing or empty.
pub fn get_api_key(provider: &str) -> Option<String> {
    let entry = entry(provider).ok()?;
    match entry.get_password() {
        Ok(s) if !s.trim().is_empty() => Some(s),
        _ => None,
    }
}

/// Store an API key. An empty key deletes the entry.
pub fn set_api_key(provider: &str, key: &str) -> Result<(), String> {
    let entry = entry(provider)?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        // Best-effort delete; ignore "no entry" errors so this is idempotent.
        match entry.delete_credential() {
            Ok(_) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(format!("Failed to delete keychain entry: {e}")),
        }
    } else {
        entry
            .set_password(trimmed)
            .map_err(|e| format!("Failed to store key in keychain: {e}"))
    }
}
