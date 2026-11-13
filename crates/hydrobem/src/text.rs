//! Text of the input files: files read as UTF-8 lines, the free-format numbers of the
//! geometry files, and the shortest round-trip form of a double.
//! The bytes of a file are a `Vec<u8>`; the line functions take the text as `&str`.
//!
//! The line breaks, whitespace and float text follow Python's str.splitlines, str.split, str.strip
//! and repr(float), which the readers had before they moved into the engine, so that a file reads
//! the same through either. The path functions decompose a path as Windows does (drive letters,
//! UNC names, both slashes) and as POSIX elsewhere.

use std::fs;
use std::io::{Read, Write};

use crate::error::{Error, Result};

// The byte length of the whitespace character at s[i] (Python's str.isspace, in UTF-8), else 0.
fn space_at(s: &[u8], i: usize) -> usize {
    let c = s[i];
    if c == b' ' || (0x09..=0x0d).contains(&c) || (0x1c..=0x1f).contains(&c) {
        return 1;
    }
    let at = |k: usize| if i + k < s.len() { s[i + k] } else { 0u8 };
    if c == 0xc2 && (at(1) == 0x85 || at(1) == 0xa0) {
        return 2; // U+0085, U+00A0
    }
    if c == 0xe1 && at(1) == 0x9a && at(2) == 0x80 {
        return 3; // U+1680
    }
    if c == 0xe2
        && at(1) == 0x80
        && ((at(2) >= 0x80 && at(2) <= 0x8a) || at(2) == 0xa8 || at(2) == 0xa9 || at(2) == 0xaf)
    {
        return 3; // U+2000-200A, 2028, 2029, 202F
    }
    if c == 0xe2 && at(1) == 0x81 && at(2) == 0x9f {
        return 3; // U+205F
    }
    if c == 0xe3 && at(1) == 0x80 && at(2) == 0x80 {
        return 3; // U+3000
    }
    0
}

// The byte length of the line break at s[i] (Python's str.splitlines), else 0.
fn break_at(s: &[u8], i: usize) -> usize {
    let c = s[i];
    let at = |k: usize| if i + k < s.len() { s[i + k] } else { 0u8 };
    if c == b'\r' {
        return if at(1) == b'\n' { 2 } else { 1 };
    }
    if c == b'\n' || c == 0x0b || c == 0x0c || (0x1c..=0x1e).contains(&c) {
        return 1;
    }
    if c == 0xc2 && at(1) == 0x85 {
        return 2;
    }
    if c == 0xe2 && at(1) == 0x80 && (at(2) == 0xa8 || at(2) == 0xa9) {
        return 3;
    }
    0
}

fn digit(c: u8) -> bool {
    c.is_ascii_digit()
}

// ---------------------------------------------------------------- the decomposition of a path
// (the rules of Windows on Windows, POSIX elsewhere)

fn is_slash(c: u8) -> bool {
    if cfg!(windows) { c == b'/' || c == b'\\' } else { c == b'/' }
}

// The end of the root name: X: (a drive), \\server, \\?\ \??\ \\.\ (Windows); none elsewhere.
fn root_name_end(s: &[u8]) -> usize {
    if !cfg!(windows) || s.len() < 2 {
        return 0;
    }
    if s[0].is_ascii_alphabetic() && s[1] == b':' {
        return 2;
    }
    if !is_slash(s[0]) {
        return 0;
    }
    if s.len() >= 4
        && is_slash(s[3])
        && (s.len() == 4 || !is_slash(s[4]))
        && ((is_slash(s[1]) && (s[2] == b'?' || s[2] == b'.')) || (s[1] == b'?' && s[2] == b'?'))
    {
        return 3;
    }
    if s.len() >= 3 && is_slash(s[1]) && !is_slash(s[2]) {
        let mut k = 3;
        while k < s.len() && !is_slash(s[k]) {
            k += 1;
        }
        return k;
    }
    0
}

// The start of the relative path: after the root name and the root directory.
fn relative_start(s: &[u8]) -> usize {
    let mut k = root_name_end(s);
    while k < s.len() && is_slash(s[k]) {
        k += 1;
    }
    k
}

// The lexical normal form (path::lexically_normal) of an absolute POSIX path: no "." and ".."
// elements, no repeated separators; a trailing separator kept.
#[cfg(not(windows))]
fn lexically_normal(p: &str) -> String {
    let rooted = p.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    let elems: Vec<&str> = p.split('/').collect();
    let trailing = elems.len() > 1 && (p.ends_with('/') || p.ends_with("/.") || p.ends_with("/.."));
    for e in elems {
        match e {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|l| *l != "..") {
                    parts.pop();
                } else if !rooted {
                    parts.push("..");
                }
            }
            _ => parts.push(e),
        }
    }
    let mut out = if rooted { "/".to_string() } else { String::new() };
    out += &parts.join("/");
    if trailing && !parts.is_empty() {
        out.push('/');
    }
    if out.is_empty() {
        out.push('.');
    }
    out
}

/// The absolute form of a path (UTF-8), against the current folder; the path as given when it has
/// none.
pub fn absolute_path(path: &str) -> String {
    match std::path::absolute(path) {
        Err(_) => path.to_string(),
        Ok(a) => {
            // on Windows std::path::absolute (GetFullPathNameW) gives the normal form
            let s = a.to_string_lossy().into_owned();
            #[cfg(not(windows))]
            let s = lexically_normal(&s);
            s
        }
    }
}

/// The bytes of a file (path UTF-8); an ErrorKind::Io error when it cannot be read.
pub fn read_file(path: &str) -> Result<Vec<u8>> {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Err(Error::io(format!("{path}: cannot open the file"))),
    };
    let mut out = Vec::new();
    if f.read_to_end(&mut out).is_err() {
        return Err(Error::io(format!("{path}: cannot read the file")));
    }
    Ok(out)
}

/// Writes text (bytes) to a file (path UTF-8), replacing it; an ErrorKind::Io error when it cannot be written.
pub fn write_file(path: &str, text: &[u8]) -> Result<()> {
    let mut f = match fs::File::create(path) {
        Ok(f) => f,
        Err(_) => return Err(Error::io(format!("{path}: cannot write the file"))),
    };
    if f.write_all(text).and_then(|_| f.flush()).is_err() {
        return Err(Error::io(format!("{path}: cannot write the file")));
    }
    Ok(())
}

/// Makes the folder of a file path when it is missing (the writers).
pub fn make_parent_folder(path: &str) {
    let parent = parent_folder(path);
    if !parent.is_empty() {
        let _ = fs::create_dir_all(&parent); // a failure shows when the file is written
    }
}

/// True when a file exists (path UTF-8).
pub fn file_exists(path: &str) -> bool {
    fs::metadata(path).map(|m| m.is_file()).unwrap_or(false)
}

/// The folder of a path and a path relative to a folder (UTF-8).
pub fn parent_folder(path: &str) -> String {
    // path::parent_path: the filename removed, then the separators before it (not the root directory)
    let s = path.as_bytes();
    let rel = relative_start(s);
    let mut last = s.len();
    while last > rel && !is_slash(s[last - 1]) {
        last -= 1;
    }
    while last > rel && is_slash(s[last - 1]) {
        last -= 1;
    }
    path[..last].to_string()
}

/// Lines as Python's str.splitlines: split at \n, \r\n, \r, \v, \f, \x1c-\x1e and the UTF-8
/// encodings of U+0085, U+2028 and U+2029, without the breaks.
pub fn split_lines(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let (mut start, mut i) = (0usize, 0usize);
    while i < b.len() {
        let n = break_at(b, i);
        if n == 0 {
            i += 1;
            continue;
        }
        out.push(text[start..i].to_string());
        i += n;
        start = i;
    }
    if start < b.len() {
        out.push(text[start..].to_string());
    }
    out
}

/// The whitespace-separated tokens of a line after replacing commas by spaces (Python's
/// line.replace(",", " ").split(), with the Unicode whitespace of str.split in UTF-8).
pub fn tokens(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        let n = if b[i] == b',' { 1 } else { space_at(b, i) };
        if n > 0 {
            if i > start {
                out.push(line[start..i].to_string());
            }
            i += n;
            start = i;
        } else {
            i += 1;
        }
    }
    if b.len() > start {
        out.push(line[start..].to_string());
    }
    out
}

/// The line without leading and trailing whitespace (Python's str.strip).
pub fn strip(s: &str) -> String {
    let b = s.as_bytes();
    let (mut a, e) = (0usize, b.len());
    while a < e {
        let n = space_at(b, a);
        if n == 0 {
            break;
        }
        a += n;
    }
    // trailing whitespace: scan forward so that multi-byte characters are found at their start
    let (mut end, mut i) = (a, a);
    while i < e {
        let n = space_at(b, i);
        if n > 0 {
            i += n;
        } else {
            i += 1;
            end = i;
        }
    }
    // end is after a byte that is not whitespace: the end of a character or a continuation byte
    // of one whose last byte follows; the bytes up to end are valid UTF-8 only at a boundary
    while end < e && !s.is_char_boundary(end) {
        end += 1;
    }
    s[a..end].to_string()
}

/// A free-format number: [+-]?(d+.?d*|.d+)([eEdD][+-]?d+)?
pub fn is_number(token: &str) -> bool {
    let t = token.as_bytes();
    let mut i = 0usize;
    if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
        i += 1;
    }
    let (mut int_digits, mut frac_digits) = (0usize, 0usize);
    while i < t.len() && digit(t[i]) {
        i += 1;
        int_digits += 1;
    }
    if i < t.len() && t[i] == b'.' {
        i += 1;
        while i < t.len() && digit(t[i]) {
            i += 1;
            frac_digits += 1;
        }
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < t.len() && (t[i] == b'e' || t[i] == b'E' || t[i] == b'd' || t[i] == b'D') {
        i += 1;
        if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
            i += 1;
        }
        let mut exp_digits = 0usize;
        while i < t.len() && digit(t[i]) {
            i += 1;
            exp_digits += 1;
        }
        if exp_digits == 0 {
            return false;
        }
    }
    i == t.len()
}

/// Its value (D or d as the exponent letter; correctly rounded, independent of the locale).
pub fn parse_number(token: &str) -> Result<f64> {
    let t: String = token.chars().map(|c| if c == 'd' || c == 'D' { 'e' } else { c }).collect();
    parse_double(&t)
}

// The text std::from_chars (chars_format::general) reads after the sign: [-](digits[.digits]|.digits)
// [eE[+-]digits], inf, infinity, nan, nan(chars) without regard to case.
fn from_chars_form(t: &str) -> bool {
    let u = t.strip_prefix('-').unwrap_or(t);
    let l = u.to_ascii_lowercase();
    if l == "inf" || l == "infinity" || l == "nan" {
        return true;
    }
    if let Some(inner) = l.strip_prefix("nan(").and_then(|r| r.strip_suffix(')')) {
        return inner.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_');
    }
    let b = u.as_bytes();
    let mut i = 0usize;
    let (mut int_digits, mut frac_digits) = (0usize, 0usize);
    while i < b.len() && digit(b[i]) {
        i += 1;
        int_digits += 1;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        while i < b.len() && digit(b[i]) {
            i += 1;
            frac_digits += 1;
        }
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let mut exp_digits = 0usize;
        while i < b.len() && digit(b[i]) {
            i += 1;
            exp_digits += 1;
        }
        if exp_digits == 0 {
            return false;
        }
    }
    i == b.len()
}

/// A decimal string, correctly rounded, independent of the locale ("1e400" is inf).
pub fn parse_double(s: &str) -> Result<f64> {
    let mut t = s;
    let mut negative = false;
    if let Some(c) = t.as_bytes().first() {
        if *c == b'+' || *c == b'-' {
            negative = *c == b'-';
            t = &t[1..];
        }
    }
    if !from_chars_form(t) {
        return Err(Error::value(format!("invalid number '{s}'")));
    }
    // parse is correctly rounded and gives inf on an overflow and 0 on an underflow, the values the
    // numbers out of range take
    let l = t.to_ascii_lowercase();
    let v = if l.trim_start_matches('-').starts_with("nan") {
        let n = f64::NAN;
        if t.starts_with('-') { -n } else { n }
    } else {
        match t.parse::<f64>() {
            Ok(v) => v,
            Err(_) => return Err(Error::value(format!("invalid number '{s}'"))),
        }
    };
    if negative { Ok(-v) } else { Ok(v) }
}

/// The shortest round-trip text of a double as Python's repr: fixed notation for decimal
/// exponents -4 <= e < 16 with ".0" for integers, else d.ddde+XX; "inf", "-inf", "nan".
pub fn py_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    let mut sci = format!("{v:e}"); // shortest round-trip digits: [-]d[.ddd]e[-]X
    let mut sign = "";
    if sci.starts_with('-') {
        sign = "-";
        sci.remove(0);
    }
    let e = sci.find('e').unwrap();
    let digits: String = sci[..e].chars().filter(|&c| c != '.').collect();
    let exp10: i32 = sci[e + 1..].parse().unwrap();
    let decpt = exp10 + 1; // value = 0.digits * 10^decpt
    let nd = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let mut m = digits[..1].to_string();
        if nd > 1 {
            m += ".";
            m += &digits[1..];
        }
        let x = decpt - 1;
        return format!("{sign}{m}{}{:02}", if x < 0 { "e-" } else { "e+" }, x.abs());
    }
    if decpt <= 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize));
    }
    if decpt < nd {
        return format!("{sign}{}.{}", &digits[..decpt as usize], &digits[decpt as usize..]);
    }
    format!("{sign}{digits}{}.0", "0".repeat((decpt - nd) as usize))
}

/// Python's repr of a str: in single quotes (double quotes when it holds a single quote and no double
/// quote), the backslash, newline, carriage return, tab and the other control characters escaped.
pub fn py_str_repr(s: &str) -> String {
    let dq = s.contains('\'') && !s.contains('"');
    let q = if dq { '"' } else { '\'' };
    let mut out = String::new();
    out.push(q);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for ch in s.chars() {
        let c = ch as u32;
        if ch == q || ch == '\\' {
            out.push('\\');
            out.push(ch);
        } else if ch == '\n' {
            out += "\\n";
        } else if ch == '\r' {
            out += "\\r";
        } else if ch == '\t' {
            out += "\\t";
        } else if c < 0x20 || c == 0x7f {
            out += "\\x";
            out.push(HEX[(c >> 4) as usize] as char);
            out.push(HEX[(c & 15) as usize] as char);
        } else {
            out.push(ch);
        }
    }
    out.push(q);
    out
}

/// An object of a list in the messages: "tanks 'aft'" by its name, "tanks 2" (one-based position k + 1)
/// without one.
pub fn element_text(what: &str, k: usize, name: &str) -> String {
    if name.is_empty() { format!("{what} {}", k + 1) } else { format!("{what} {}", py_str_repr(name)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_of_doubles() {
        // the values of Python's repr(float)
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e15, "1000000000000000.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (5e-324, "5e-324"),
            (2.2250738585072014e-308, "2.2250738585072014e-308"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (1e100, "1e+100"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.0 / 3.0, "0.6666666666666666"),
            (100.0, "100.0"),
            (1234.5, "1234.5"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (9.81, "9.81"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for &(v, t) in cases {
            assert_eq!(py_repr(v), t, "repr of {v:?}");
        }
    }

    #[test]
    fn numbers() -> Result<()> {
        assert!(is_number("1") && is_number("-1.") && is_number(".5") && is_number("+1.5D-3") && is_number("2e10"));
        assert!(!is_number(".") && !is_number("1e") && !is_number("e5") && !is_number("1.5x") && !is_number(""));
        assert!(!is_number("inf") && !is_number("nan") && !is_number("1e+"));
        assert_eq!(parse_number("1.5D-3")?, 1.5e-3);
        assert_eq!(parse_number("-2d2")?, -200.0);
        assert_eq!(parse_double("1e400")?, f64::INFINITY);
        assert_eq!(parse_double("-1e400")?, f64::NEG_INFINITY);
        let z = parse_double("-1e-400")?;
        assert!(z == 0.0 && z.is_sign_negative());
        assert_eq!(parse_double("0.1")?, 0.1);
        assert_eq!(parse_double("2.2250738585072011e-308")?, 2.225073858507201e-308);
        assert_eq!(parse_double("--5")?, 5.0); // the sign, then from_chars reads -5
        assert!(parse_double("nan")?.is_nan());
        assert!(parse_double("-+5").is_err());
        assert!(parse_double("1,5").is_err());
        assert!(parse_double("").is_err());
        let e = parse_double("x1").unwrap_err();
        assert_eq!(e.message(), "invalid number 'x1'");
        Ok(())
    }

    #[test]
    fn lines_and_tokens() {
        assert_eq!(
            split_lines("a\nb\r\nc\rd\x0be\x0cf\x1cg\u{85}h\u{2028}i\u{2029}j"),
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        );
        assert_eq!(split_lines("a\n\nb\n"), vec!["a", "", "b"]);
        assert_eq!(split_lines(""), Vec::<String>::new());
        assert_eq!(tokens(" 1,2  3\t4\u{a0}5\u{3000}6,,7 "), vec!["1", "2", "3", "4", "5", "6", "7"]);
        assert_eq!(tokens("é,ü"), vec!["é", "ü"]);
        assert_eq!(strip("\u{2003} a b \u{a0}\t"), "a b");
        assert_eq!(strip("   "), "");
        assert_eq!(strip("xé "), "xé");
    }

    #[test]
    fn str_repr() {
        assert_eq!(py_str_repr("abc"), "'abc'");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_str_repr("a\\b\n\r\t\x01\x7f"), "'a\\\\b\\n\\r\\t\\x01\\x7f'");
        assert_eq!(py_str_repr("é"), "'é'");
    }

    #[test]
    fn paths() {
        if cfg!(windows) {
            assert_eq!(parent_folder("C:\\a\\b.gdf"), "C:\\a");
            assert_eq!(parent_folder("C:\\b.gdf"), "C:\\");
            assert_eq!(parent_folder("C:\\"), "C:\\");
            assert_eq!(parent_folder("a/b/"), "a/b");
            assert_eq!(parent_folder("a//b"), "a");
            assert_eq!(parent_folder("b.gdf"), "");
            assert_eq!(parent_folder("\\\\server\\share\\x"), "\\\\server\\share");
        } else {
            assert_eq!(parent_folder("/a/b.gdf"), "/a");
        }
    }
}
