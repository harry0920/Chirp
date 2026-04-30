//! Resolves raw foreground process names (e.g. "Slack.exe") into display
//! names (e.g. "Slack"). Mapping lives in `resources/app-names.json`,
//! embedded at compile time so there's no runtime filesystem dependency.
//!
//! Lookup is case-insensitive on the raw key. Unknown processes fall back
//! to a title-cased stripped form.

use std::collections::HashMap;
use std::sync::OnceLock;

const APP_NAMES_JSON: &str = include_str!("../resources/app-names.json");

fn map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        match serde_json::from_str::<HashMap<String, String>>(APP_NAMES_JSON) {
            Ok(parsed) => parsed
                .into_iter()
                .map(|(k, v)| (k.to_lowercase(), v))
                .collect(),
            Err(e) => {
                log::warn!("Failed to parse embedded app-names.json: {e}");
                HashMap::new()
            }
        }
    })
}

/// Resolve a raw process file name to a friendly display name.
///
/// Examples:
///   `display_name("Slack.exe")` → `"Slack"`
///   `display_name("unknown_tool.exe")` → `"Unknown Tool"`
///   `display_name("foo")` → `"Foo"`
pub fn display_name(raw: &str) -> String {
    let key = raw.trim().to_lowercase();
    if let Some(known) = map().get(&key) {
        return known.clone();
    }
    let stripped = key
        .strip_suffix(".exe")
        .or_else(|| key.strip_suffix(".app"))
        .unwrap_or(&key);

    title_case(stripped)
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut new_word = true;
    for ch in s.chars() {
        if ch == ' ' || ch == '_' || ch == '-' || ch == '.' {
            out.push(' ');
            new_word = true;
            continue;
        }
        if new_word {
            for up in ch.to_uppercase() {
                out.push(up);
            }
            new_word = false;
        } else {
            out.push(ch);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_lookup() {
        assert_eq!(display_name("Slack.exe"), "Slack");
        assert_eq!(display_name("slack.exe"), "Slack");
        assert_eq!(display_name("SLACK.EXE"), "Slack");
    }

    #[test]
    fn unknown_falls_back() {
        assert_eq!(display_name("unknown_tool.exe"), "Unknown Tool");
        assert_eq!(display_name("my-app.exe"), "My App");
        assert_eq!(display_name("foo"), "Foo");
    }
}
