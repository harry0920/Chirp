use regex::Regex;
use std::sync::OnceLock;

use crate::state::VocabEntry;

/// Pre-compiled regex patterns for text cleanup
struct CleanupRegexes {
    fillers: Vec<Regex>,
    dangling_comma: Regex,
    leading_comma: Regex,
    whitespace: Regex,
    sentence_end: Regex,
    standalone_i: Regex,
    i_contraction: Regex,
    punctuation: Vec<(Regex, &'static str)>,
    space_before_punct: Regex,
    no_space_after: Regex,
    email: Regex,
    numeric_contexts: Vec<Regex>,
    number_words: Vec<&'static str>,
    percentage: Regex,
    hundred_pct: Regex,
}

fn regexes() -> &'static CleanupRegexes {
    static REGEXES: OnceLock<CleanupRegexes> = OnceLock::new();
    REGEXES.get_or_init(|| {
        let filler_patterns = [
            r"(?i)\bum+\b",
            r"(?i)\buh+\b",
            r"(?i)\buh huh\b",
            r"(?i)\bmm+ ?hmm+\b",
            r"(?i)\bhmm+\b",
            r"(?i)\byou know\b(?=\s*,?\s)",
            r"(?i)\blike\b(?=\s+(the|a|an|i|we|they|he|she|it|my|our|this|that)\b)",
            r"(?i)\bbasically\b(?=\s*,)",
            r"(?i)\bactually\b(?=\s*,)",
            r"(?i)\bso\b(?=\s*,\s)",
            r"(?i)\bi mean\b(?=\s*,)",
            r"(?i)\bkind of\b(?=\s+(like|a|the)\b)",
            r"(?i)\bsort of\b(?=\s+(like|a|the)\b)",
            r"(?i)\bright\s*\?\s*(?=\b)",
        ];

        let number_word_patterns = [
            (r"\b(?i)zero\b", "0"),
            (r"\b(?i)one\b", "1"),
            (r"\b(?i)two\b", "2"),
            (r"\b(?i)three\b", "3"),
            (r"\b(?i)four\b", "4"),
            (r"\b(?i)five\b", "5"),
            (r"\b(?i)six\b", "6"),
            (r"\b(?i)seven\b", "7"),
            (r"\b(?i)eight\b", "8"),
            (r"\b(?i)nine\b", "9"),
            (r"\b(?i)ten\b", "10"),
        ];

        let numeric_context_patterns = [
            r"(?i)\b(number|step|item|option|version|v|chapter|page|line|row|column|level|grade|score|count|total)\s+",
            r"(?i)\b(is|are|was|were|equals?|=)\s+",
            r"(?i)\b(about|around|approximately|roughly|nearly|over|under)\s+",
        ];

        // Pre-compile combined numeric context + number word patterns
        let mut compiled_contexts = Vec::new();
        let mut compiled_numbers = Vec::new();
        for ctx_pattern in &numeric_context_patterns {
            for (word_pattern, digit) in &number_word_patterns {
                let combined = format!("({ctx_pattern})({word_pattern})");
                if let Ok(re) = Regex::new(&combined) {
                    compiled_contexts.push(re);
                    compiled_numbers.push(*digit);
                }
            }
        }

        let punctuation_map: Vec<(Regex, &'static str)> = [
            (r"(?i)\bperiod\b", "."),
            (r"(?i)\bcomma\b", ","),
            (r"(?i)\bquestion mark\b", "?"),
            (r"(?i)\bexclamation (?:mark|point)\b", "!"),
            (r"(?i)\bcolon\b", ":"),
            (r"(?i)\bsemicolon\b", ";"),
            (r"(?i)\bdash\b", " —"),
            (r"(?i)\bhyphen\b", "-"),
            (r"(?i)\bopen (?:paren|parenthesis)\b", "("),
            (r"(?i)\bclose (?:paren|parenthesis)\b", ")"),
            // "new line" and "new paragraph" handled by LLM, not regex
        ]
        .iter()
        .filter_map(|(p, r)| Regex::new(p).ok().map(|re| (re, *r)))
        .collect();

        // Store pre-compiled context+number pairs as parallel vecs in the struct
        // We'll use numeric_contexts for the compiled combined regexes
        // and number_words for the corresponding digit strings
        CleanupRegexes {
            fillers: filler_patterns
                .iter()
                .filter_map(|p| Regex::new(p).ok())
                .collect(),
            dangling_comma: Regex::new(r",\s*,").unwrap(),
            leading_comma: Regex::new(r"^\s*,\s*").unwrap(),
            whitespace: Regex::new(r"\s{2,}").unwrap(),
            sentence_end: Regex::new(r"([.!?])\s+([a-z])").unwrap(),
            standalone_i: Regex::new(r"\bi\b").unwrap(),
            i_contraction: Regex::new(r"\bI'([msdtv])").unwrap(),
            punctuation: punctuation_map,
            space_before_punct: Regex::new(r"\s+([.,!?;:)])").unwrap(),
            no_space_after: Regex::new(r"([.,!?;:])([A-Za-z])").unwrap(),
            email: Regex::new(r"(?i)\b(\w+)\s+at\s+(\w+)\s+dot\s+(com|org|net|io|dev|co)\b").unwrap(),
            numeric_contexts: compiled_contexts,
            number_words: compiled_numbers,
            percentage: Regex::new(r"(?i)\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+percent\b").unwrap(),
            hundred_pct: Regex::new(r"(?i)\b(one )?hundred percent\b").unwrap(),
        }
    })
}

/// Full cleanup pipeline: filler removal → capitalization → regex formatting.
///
/// Self-correction handling is now performed by the LLM cleanup pass
/// (chirp-cleanup-v2), which is fine-tuned for it and applies semantic
/// context. The previous regex-based `strip_corrections` step was removed
/// because its `.*` greedy match across sentence boundaries silently deleted
/// large portions of dictations whenever a discourse filler like "I mean" /
/// "actually" / "sorry" appeared mid-monologue.
pub fn cleanup_text(text: &str, smart_formatting: bool) -> String {
    if text.is_empty() {
        return String::new();
    }

    if !smart_formatting {
        return text.to_string();
    }

    // Remove fillers (um, uh, filler "like", etc.)
    let result = remove_fillers(text);
    if result.is_empty() {
        return String::new();
    }

    // Capitalize first letter (filler removal may have stripped a leading "Um,")
    let result = capitalize_first(&result);

    // Regex-based formatting (spoken punctuation, numbers, etc.)
    smart_format(&result)
}

/// Remove common filler words from transcript
fn remove_fillers(text: &str) -> String {
    let re = regexes();
    let mut result = text.to_string();

    for filler in &re.fillers {
        result = filler.replace_all(&result, "").to_string();
    }

    // Clean up extra whitespace and dangling commas from removal
    result = re.dangling_comma.replace_all(&result, ",").to_string();
    result = re.leading_comma.replace(&result, "").to_string();
    re.whitespace.replace_all(result.trim(), " ").to_string()
}

/// Capitalize the first character of a string
fn capitalize_first(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut chars = trimmed.chars();
    let first = chars.next().unwrap();
    first.to_uppercase().to_string() + chars.as_str()
}

/// Smart formatting: punctuation, capitalization, numbers, common patterns
fn smart_format(text: &str) -> String {
    let mut result = text.to_string();

    // Expand spoken numbers to digits for common cases
    result = format_spoken_numbers(&result);

    // Format common spoken patterns
    result = format_spoken_patterns(&result);

    result
}

/// Convert spoken number words to digits for common short numbers
fn format_spoken_numbers(text: &str) -> String {
    let re = regexes();
    let mut result = text.to_string();

    // Apply pre-compiled combined context+number patterns
    for (i, ctx_re) in re.numeric_contexts.iter().enumerate() {
        let digit = re.number_words[i];
        result = ctx_re
            .replace_all(&result, |caps: &regex::Captures| {
                format!("{}{}", &caps[1], digit)
            })
            .to_string();
    }

    // Percentages
    result = re
        .percentage
        .replace_all(&result, |caps: &regex::Captures| {
            let num = match caps[1].to_lowercase().as_str() {
                "twenty" => "20",
                "thirty" => "30",
                "forty" => "40",
                "fifty" => "50",
                "sixty" => "60",
                "seventy" => "70",
                "eighty" => "80",
                "ninety" => "90",
                _ => &caps[1],
            };
            format!("{num}%")
        })
        .to_string();

    // "hundred percent" → "100%"
    result = re.hundred_pct.replace_all(&result, "100%").to_string();

    result
}

/// Format common spoken patterns (email, URLs, punctuation commands)
fn format_spoken_patterns(text: &str) -> String {
    let re = regexes();
    let mut result = text.to_string();

    // Spoken punctuation → actual punctuation
    for (pattern, replacement) in &re.punctuation {
        result = pattern.replace_all(&result, *replacement).to_string();
    }

    // Clean up spaces before punctuation
    result = re.space_before_punct.replace_all(&result, "$1").to_string();

    // Ensure space after punctuation
    result = re.no_space_after.replace_all(&result, "$1 $2").to_string();

    // Email pattern
    result = re.email.replace_all(&result, "$1@$2.$3").to_string();

    result
}

/// Apply user-configured find/replace from the vocabulary's `replaces` lists.
///
/// For each VocabEntry, every string in `replaces` is matched
/// case-insensitively at word boundaries and substituted with the canonical
/// `term`. This is the deterministic post-ASR fixup layer for things sherpa
/// hotword biasing can't fix — homophones (Pieter/Peter), brand spellings,
/// stable mishearings.
///
/// Runs BEFORE the LLM cleanup pass so the LLM sees the corrected names in
/// context. chirp-cleanup-v2 at temp 0.0 won't reintroduce alternative
/// spellings the input doesn't contain.
///
/// Edge cases:
///   - Self-replacements (case-insensitive `from == to`) are skipped
///   - Empty / whitespace-only `from` strings are skipped
///   - `from` strings are `regex::escape`'d, so literal special chars are safe
///   - Match is case-insensitive; replacement always uses the canonical case
pub fn apply_replacements(text: &str, vocabulary: &[VocabEntry]) -> String {
    let mut result = text.to_string();
    for entry in vocabulary {
        let canonical = entry.term.trim();
        if canonical.is_empty() {
            continue;
        }
        for from in &entry.replaces {
            let from_trim = from.trim();
            if from_trim.is_empty() {
                continue;
            }
            if from_trim.eq_ignore_ascii_case(canonical) {
                continue;
            }
            // Build a case-insensitive regex for the literal `from`. `\b` only
            // matches at the transition between a word and non-word char, so
            // we only anchor with `\b` on sides where the pattern's edge IS a
            // word character. Otherwise the boundary assertion would fail
            // (e.g. `\bco (corp)\b` — the trailing `)` is non-word, the
            // following space is also non-word, so no boundary exists there).
            let starts_with_word = from_trim
                .chars()
                .next()
                .map(|c| c.is_alphanumeric() || c == '_')
                .unwrap_or(false);
            let ends_with_word = from_trim
                .chars()
                .last()
                .map(|c| c.is_alphanumeric() || c == '_')
                .unwrap_or(false);
            let pattern = format!(
                "(?i){}{}{}",
                if starts_with_word { r"\b" } else { "" },
                regex::escape(from_trim),
                if ends_with_word { r"\b" } else { "" },
            );
            match Regex::new(&pattern) {
                Ok(re) => {
                    result = re.replace_all(&result, canonical).to_string();
                }
                Err(e) => {
                    log::warn!("Failed to compile replacement pattern '{from_trim}': {e}");
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spoken_punctuation() {
        let result = smart_format("hello comma how are you question mark");
        assert!(result.contains("hello, how are you?"));
    }

    #[test]
    fn test_percentage() {
        let result = smart_format("it was about fifty percent done");
        assert!(result.contains("50%"));
    }

    #[test]
    fn test_email() {
        let result = smart_format("send it to john at example dot com");
        assert!(result.contains("john@example.com"));
    }

    #[test]
    fn test_new_paragraph_passthrough() {
        // "new paragraph" should pass through to LLM as words, not be converted by regex
        let result = smart_format("hello new paragraph world");
        assert!(result.contains("new paragraph"));
    }

    #[test]
    fn test_full_cleanup() {
        let result = cleanup_text("send an email to bob at test dot com", true);
        assert!(result.contains("bob@test.com"));
    }

    fn vocab(entries: &[(&str, &[&str])]) -> Vec<VocabEntry> {
        entries
            .iter()
            .map(|(term, replaces)| VocabEntry {
                term: (*term).to_string(),
                replaces: replaces.iter().map(|s| (*s).to_string()).collect(),
            })
            .collect()
    }

    #[test]
    fn test_apply_replacements_basic() {
        let v = vocab(&[("Pieter", &["Peter"])]);
        assert_eq!(apply_replacements("My name is Peter.", &v), "My name is Pieter.");
    }

    #[test]
    fn test_apply_replacements_case_insensitive() {
        let v = vocab(&[("Pieter", &["Peter"])]);
        assert_eq!(apply_replacements("PETER and peter", &v), "Pieter and Pieter");
    }

    #[test]
    fn test_apply_replacements_word_boundary() {
        // "petersburg" should NOT become "pietersburg"
        let v = vocab(&[("Pieter", &["Peter"])]);
        assert_eq!(apply_replacements("I went to Petersburg.", &v), "I went to Petersburg.");
    }

    #[test]
    fn test_apply_replacements_apostrophe() {
        // "Peter's" should become "Pieter's" — \b matches between r and '
        let v = vocab(&[("Pieter", &["Peter"])]);
        assert_eq!(apply_replacements("Peter's car.", &v), "Pieter's car.");
    }

    #[test]
    fn test_apply_replacements_multiple_froms() {
        let v = vocab(&[("Lakshamanan", &["Lakshmanan", "lock shamanan"])]);
        assert_eq!(
            apply_replacements("Hi Lakshmanan and lock shamanan", &v),
            "Hi Lakshamanan and Lakshamanan"
        );
    }

    #[test]
    fn test_apply_replacements_self_replacement_skipped() {
        // Replacing "Peter" with "Peter" should be a no-op (no infinite loop)
        let v = vocab(&[("Peter", &["peter", "Peter", "PETER"])]);
        // All three case-insensitively equal to canonical → all skipped
        assert_eq!(apply_replacements("hello peter", &v), "hello peter");
    }

    #[test]
    fn test_apply_replacements_empty_replaces() {
        // Term with no replaces does nothing — purely for ASR biasing
        let v = vocab(&[("Akilan", &[])]);
        assert_eq!(apply_replacements("hello world", &v), "hello world");
    }

    #[test]
    fn test_apply_replacements_special_chars_in_from() {
        // regex::escape handles parens, dots, etc. in the `from` string
        let v = vocab(&[("Co.", &["co (corp)"])]);
        assert_eq!(
            apply_replacements("the co (corp) released a product", &v),
            "the Co. released a product"
        );
    }
}
