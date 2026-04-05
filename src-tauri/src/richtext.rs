use regex::Regex;
use std::sync::OnceLock;

/// Pre-compiled regex for detecting numbered list items
fn list_item_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\d+\.\s+(.*)").unwrap())
}

/// Check if text has structure worth converting to rich text (paragraphs or lists)
fn has_rich_structure(text: &str) -> bool {
    let re = list_item_re();
    text.contains("\n\n") || text.lines().any(|line| re.is_match(line))
}

/// Escape HTML special characters
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Escape RTF special characters and encode non-ASCII as \uN?
fn rtf_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            c if c as u32 > 0x7F => {
                // RTF Unicode: \uN? where N is signed 16-bit
                let code = c as u32;
                if code <= 0xFFFF {
                    let signed = code as i16;
                    out.push_str(&format!("\\u{}?", signed));
                } else {
                    // Supplementary plane: encode as UTF-16 surrogate pair
                    let code = code - 0x10000;
                    let high = (code >> 10) + 0xD800;
                    let low = (code & 0x3FF) + 0xDC00;
                    out.push_str(&format!("\\u{}?\\u{}?", high as i16, low as i16));
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// Convert structured plain text to CF_HTML clipboard format.
/// Returns None if text has no rich structure (single paragraph, no lists).
pub fn text_to_cf_html(text: &str) -> Option<String> {
    if !has_rich_structure(text) {
        return None;
    }

    let re = list_item_re();
    let blocks: Vec<&str> = text.split("\n\n").collect();
    let mut html_body = String::new();

    for block in &blocks {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let lines: Vec<&str> = block.lines().collect();
        let all_list_items = lines.iter().all(|line| re.is_match(line));

        if all_list_items && lines.len() > 1 {
            html_body.push_str("<ol>");
            for line in &lines {
                if let Some(caps) = re.captures(line) {
                    html_body.push_str("<li>");
                    html_body.push_str(&html_escape(&caps[1]));
                    html_body.push_str("</li>");
                }
            }
            html_body.push_str("</ol>");
        } else {
            html_body.push_str("<p>");
            let escaped_lines: Vec<String> = lines.iter().map(|l| html_escape(l)).collect();
            html_body.push_str(&escaped_lines.join("<br>"));
            html_body.push_str("</p>");
        }
    }

    // Build CF_HTML with byte-offset header
    // Use 10-digit zero-padded placeholders so header length is constant
    let header_template = "Version:0.9\r\nStartHTML:%%START_H%%\r\nEndHTML:%%END_HTML%%\r\nStartFragment:%%START_F%%\r\nEndFragment:%%ENDFRAGM%%\r\n";
    let before_fragment = "<html><body><!--StartFragment-->";
    let after_fragment = "<!--EndFragment--></body></html>";

    // Calculate with placeholder length (10 chars each)
    // Compute header length with 10-digit placeholders
    let header_with_placeholders = header_template
        .replace("%%START_H%%", "0000000000")
        .replace("%%END_HTML%%", "0000000000")
        .replace("%%START_F%%", "0000000000")
        .replace("%%ENDFRAGM%%", "0000000000");

    let header_len = header_with_placeholders.len();
    let start_html = header_len;
    let start_fragment = header_len + before_fragment.len();
    let end_fragment = start_fragment + html_body.len();
    let end_html = end_fragment + after_fragment.len();

    let header = header_template
        .replace("%%START_H%%", &format!("{:010}", start_html))
        .replace("%%END_HTML%%", &format!("{:010}", end_html))
        .replace("%%START_F%%", &format!("{:010}", start_fragment))
        .replace("%%ENDFRAGM%%", &format!("{:010}", end_fragment));

    let mut result = String::with_capacity(end_html);
    result.push_str(&header);
    result.push_str(before_fragment);
    result.push_str(&html_body);
    result.push_str(after_fragment);

    Some(result)
}

/// Convert structured plain text to RTF.
/// Returns None if text has no rich structure (single paragraph, no lists).
pub fn text_to_rtf(text: &str) -> Option<String> {
    if !has_rich_structure(text) {
        return None;
    }

    let re = list_item_re();
    let blocks: Vec<&str> = text.split("\n\n").collect();

    let mut rtf = String::from("{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Calibri;}}\\f0\\fs22 ");

    for (i, block) in blocks.iter().enumerate() {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }

        let lines: Vec<&str> = block.lines().collect();
        let all_list_items = lines.iter().all(|line| re.is_match(line));

        if all_list_items && lines.len() > 1 {
            for (j, line) in lines.iter().enumerate() {
                if let Some(caps) = re.captures(line) {
                    let num = j + 1;
                    rtf.push_str(&format!(
                        "{{\\pntext {}\\tab}}{}\\par\n",
                        num,
                        rtf_escape(&caps[1])
                    ));
                }
            }
        } else {
            for line in &lines {
                rtf.push_str(&rtf_escape(line));
                rtf.push_str("\\line\n");
            }
        }

        // Double paragraph break between blocks
        if i < blocks.len() - 1 {
            rtf.push_str("\\par\n");
        }
    }

    rtf.push('}');
    Some(rtf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text_returns_none() {
        assert!(text_to_cf_html("Hello world. This is a sentence.").is_none());
        assert!(text_to_rtf("Hello world. This is a sentence.").is_none());
    }

    #[test]
    fn test_paragraphs_html() {
        let text = "First paragraph here.\n\nSecond paragraph here.";
        let html = text_to_cf_html(text).unwrap();
        assert!(html.contains("<p>First paragraph here.</p>"));
        assert!(html.contains("<p>Second paragraph here.</p>"));
        assert!(html.contains("Version:0.9"));
        assert!(html.contains("StartFragment"));
    }

    #[test]
    fn test_paragraphs_rtf() {
        let text = "First paragraph here.\n\nSecond paragraph here.";
        let rtf = text_to_rtf(text).unwrap();
        assert!(rtf.contains("First paragraph here."));
        assert!(rtf.contains("Second paragraph here."));
        assert!(rtf.starts_with("{\\rtf1"));
        assert!(rtf.ends_with('}'));
    }

    #[test]
    fn test_numbered_list_html() {
        let text = "1. Update the API\n2. Test it\n3. Deploy it";
        let html = text_to_cf_html(text).unwrap();
        assert!(html.contains("<ol>"));
        assert!(html.contains("<li>Update the API</li>"));
        assert!(html.contains("<li>Test it</li>"));
        assert!(html.contains("<li>Deploy it</li>"));
        assert!(html.contains("</ol>"));
    }

    #[test]
    fn test_numbered_list_rtf() {
        let text = "1. Update the API\n2. Test it\n3. Deploy it";
        let rtf = text_to_rtf(text).unwrap();
        assert!(rtf.contains("{\\pntext 1\\tab}Update the API"));
        assert!(rtf.contains("{\\pntext 2\\tab}Test it"));
        assert!(rtf.contains("{\\pntext 3\\tab}Deploy it"));
    }

    #[test]
    fn test_mixed_paragraph_and_list() {
        let text = "Here is what we need to do:\n\n1. First thing\n2. Second thing\n3. Third thing";
        let html = text_to_cf_html(text).unwrap();
        assert!(html.contains("<p>Here is what we need to do:</p>"));
        assert!(html.contains("<ol>"));
        assert!(html.contains("<li>First thing</li>"));
    }

    #[test]
    fn test_html_escaping() {
        let text = "Use <div> & \"quotes\".\n\nAnother paragraph.";
        let html = text_to_cf_html(text).unwrap();
        assert!(html.contains("&lt;div&gt;"));
        assert!(html.contains("&amp;"));
    }

    #[test]
    fn test_rtf_escaping() {
        let text = "Braces {} and backslash \\.\n\nAnother paragraph.";
        let rtf = text_to_rtf(text).unwrap();
        assert!(rtf.contains("\\{\\}"));
        assert!(rtf.contains("\\\\"));
    }

    #[test]
    fn test_cf_html_offsets() {
        let text = "Para one.\n\nPara two.";
        let html = text_to_cf_html(text).unwrap();
        // Parse the offsets from the header
        let start_html: usize = html[html.find("StartHTML:").unwrap() + 10..][..10]
            .trim()
            .parse()
            .unwrap();
        let end_html: usize = html[html.find("EndHTML:").unwrap() + 8..][..10]
            .trim()
            .parse()
            .unwrap();
        let start_frag: usize = html[html.find("StartFragment:").unwrap() + 14..][..10]
            .trim()
            .parse()
            .unwrap();
        let end_frag: usize = html[html.find("EndFragment:").unwrap() + 12..][..10]
            .trim()
            .parse()
            .unwrap();

        assert_eq!(&html[start_html..start_html + 6], "<html>");
        assert_eq!(&html[end_html - 7..end_html], "</html>");
        assert!(start_frag > start_html);
        assert!(end_frag > start_frag);
        assert!(end_html > end_frag);
        // Content between fragment markers should contain our paragraphs
        let fragment = &html[start_frag..end_frag];
        assert!(fragment.contains("Para one."));
        assert!(fragment.contains("Para two."));
    }

    #[test]
    fn test_non_ascii_rtf() {
        let text = "Caf\u{00e9} na\u{00ef}ve.\n\nSecond.";
        let rtf = text_to_rtf(text).unwrap();
        assert!(rtf.contains("\\u233?")); // é = U+00E9 = 233
        assert!(rtf.contains("\\u239?")); // ï = U+00EF = 239
    }

    #[test]
    fn test_single_list_item_not_treated_as_list() {
        // A single line starting with "1." is a paragraph, not a list
        let text = "1. Just one item here.\n\nAnother paragraph.";
        let html = text_to_cf_html(text).unwrap();
        // Single "1." line should be a paragraph, not an <ol>
        assert!(html.contains("<p>1. Just one item here.</p>"));
    }
}
