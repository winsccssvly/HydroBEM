//! A small JSON reader and writer (RFC 8259) for the geometry files (hbg.rs): values with the
//! members of an object in their order, duplicate member names refused, numbers as f64 (with
//! whether the text was an integer), the nesting limited to 64 levels.

use crate::error::{Error, Result};
use crate::text::py_repr;

/// A JSON value; the members of an object in the order of the text.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// the value and whether the text had neither a fraction nor an exponent
    Number(f64, bool),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The name of the kind of the value, for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Number(..) => "a number",
            Json::String(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> Error {
        // the line and column (one-based, in bytes) of the position
        let before = &self.s[..self.i.min(self.s.len())];
        let line = before.iter().filter(|&&c| c == b'\n').count() + 1;
        let col = before.iter().rev().take_while(|&&c| c != b'\n').count() + 1;
        Error::value(format!("JSON at line {line}, column {col}: {what}"))
    }
    fn space(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn literal(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.error("an unknown word (true, false or null expected)"))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Json> {
        self.space();
        if depth >= MAX_DEPTH && matches!(self.peek(), Some(b'{') | Some(b'[')) {
            return Err(self.error("nested more than 64 levels"));
        }
        match self.peek() {
            None => Err(self.error("a value expected, the text ended")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.error("a value expected")),
        }
    }
    fn object(&mut self, depth: usize) -> Result<Json> {
        self.i += 1;
        let mut members: Vec<(String, Json)> = Vec::new();
        self.space();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.space();
            if self.peek() != Some(b'"') {
                return Err(self.error("a member name (a string) expected"));
            }
            let at = self.i;
            let key = self.string()?;
            if members.iter().any(|(k, _)| *k == key) {
                self.i = at;
                return Err(self.error(&format!("the member \"{key}\" is given twice")));
            }
            self.space();
            if self.peek() != Some(b':') {
                return Err(self.error("':' expected after a member name"));
            }
            self.i += 1;
            let v = self.value(depth + 1)?;
            members.push((key, v));
            self.space();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.error("',' or '}' expected")),
            }
        }
    }
    fn array(&mut self, depth: usize) -> Result<Json> {
        self.i += 1;
        let mut items = Vec::new();
        self.space();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.space();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("',' or ']' expected")),
            }
        }
    }
    fn hex4(&mut self) -> Result<u32> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.error("four hexadecimal digits expected"))?;
        let t = std::str::from_utf8(h).map_err(|_| self.error("four hexadecimal digits expected"))?;
        if !t.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(self.error("four hexadecimal digits expected"));
        }
        let v = u32::from_str_radix(t, 16).map_err(|_| self.error("four hexadecimal digits expected"))?;
        self.i += 4;
        Ok(v)
    }
    fn string(&mut self) -> Result<String> {
        self.i += 1; // the opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.error("the string is not closed"));
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(e) = self.peek() else {
                        return Err(self.error("the string is not closed"));
                    };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut v = self.hex4()?;
                            if (0xD800..0xDC00).contains(&v) {
                                // a surrogate pair
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err(self.error("a high surrogate without its low surrogate"));
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.error("a high surrogate without its low surrogate"));
                                }
                                v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            char::from_u32(v).ok_or_else(|| self.error("an invalid \\u escape"))?
                        }
                        _ => return Err(self.error("an invalid escape")),
                    };
                    let mut b = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                }
                c if c < 0x20 => return Err(self.error("a control character in a string")),
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.error("a string that is not UTF-8"))
    }
    fn number(&mut self) -> Result<Json> {
        // -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
        let start = self.i;
        let digits = |p: &mut Self| {
            let a = p.i;
            while p.peek().is_some_and(|c| c.is_ascii_digit()) {
                p.i += 1;
            }
            p.i - a
        };
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        let first = self.peek();
        let n = digits(self);
        if n == 0 || (first == Some(b'0') && n > 1) {
            return Err(self.error("an invalid number"));
        }
        let mut integer = true;
        if self.peek() == Some(b'.') {
            self.i += 1;
            integer = false;
            if digits(self) == 0 {
                return Err(self.error("an invalid number"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            integer = false;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err(self.error("an invalid number"));
            }
        }
        let t = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII digits");
        let v: f64 = t.parse().map_err(|_| self.error("an invalid number"))?;
        if !v.is_finite() {
            return Err(self.error("a number beyond the range of a double"));
        }
        Ok(Json::Number(v, integer))
    }
}

/// The value of a JSON text (one value, then only white space).
pub fn parse(text: &str) -> Result<Json> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    // a byte-order mark is not JSON (RFC 8259 §8.1) but is tolerated, as many editors write one
    if p.s.starts_with(&[0xEF, 0xBB, 0xBF]) {
        p.i = 3;
    }
    let v = p.value(0)?;
    p.space();
    if p.i != p.s.len() {
        return Err(p.error("text after the value"));
    }
    Ok(v)
}

fn quoted(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn number(v: f64, integer: bool, out: &mut String) {
    if integer && v.fract() == 0.0 && v.abs() < 9.007_199_254_740_992e15 {
        out.push_str(&format!("{}", v as i64));
    } else {
        out.push_str(&py_repr(v)); // the shortest text that reads back as the same double
    }
}

fn flat(v: &Json) -> bool {
    !matches!(v, Json::Array(_) | Json::Object(_))
}

// a value on one line
fn inline(v: &Json, out: &mut String) {
    match v {
        Json::Array(a) => {
            out.push('[');
            for (k, x) in a.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                inline(x, out);
            }
            out.push(']');
        }
        _ => write(v, 0, out),
    }
}

fn write(v: &Json, indent: usize, out: &mut String) {
    let pad = |n: usize| " ".repeat(2 * n);
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(x, i) => number(*x, *i, out),
        Json::String(s) => quoted(s, out),
        Json::Array(a) => {
            // an array of scalars on one line, also an array of such arrays (a row of points)
            // while it fits in 100 characters; otherwise one item per line
            let short = a.iter().all(|x| matches!(x, Json::Array(b) if b.iter().all(flat))) && {
                let mut t = String::new();
                inline(v, &mut t);
                t.len() <= 100
            };
            if short {
                inline(v, out);
            } else if a.iter().all(flat) {
                out.push('[');
                for (k, x) in a.iter().enumerate() {
                    if k > 0 {
                        out.push_str(", ");
                    }
                    write(x, indent, out);
                }
                out.push(']');
            } else {
                out.push_str("[\n");
                for (k, x) in a.iter().enumerate() {
                    out.push_str(&pad(indent + 1));
                    write(x, indent + 1, out);
                    out.push_str(if k + 1 < a.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(indent));
                out.push(']');
            }
        }
        Json::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (k, (key, x)) in m.iter().enumerate() {
                out.push_str(&pad(indent + 1));
                quoted(key, out);
                out.push_str(": ");
                write(x, indent + 1, out);
                out.push_str(if k + 1 < m.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(indent));
            out.push('}');
        }
    }
}

/// The text of a value: objects one member per line, arrays of numbers on one line, two spaces
/// per level; ends with a newline.
pub fn to_text(v: &Json) -> String {
    let mut out = String::new();
    write(v, 0, &mut out);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let t = r#"{"a": [1, -2.5e-3, 0.1, 1e300], "b": {"c": "x\"\né😀", "d": [true, false, null]}}"#;
        let v = parse(t).unwrap();
        let back = parse(&to_text(&v)).unwrap();
        assert_eq!(v, back);
        let Json::Object(m) = &v else { panic!() };
        assert_eq!(
            m[0].1,
            Json::Array(vec![
                Json::Number(1.0, true),
                Json::Number(-2.5e-3, false),
                Json::Number(0.1, false),
                Json::Number(1e300, false)
            ])
        );
        let Json::Object(b) = &m[1].1 else { panic!() };
        assert_eq!(b[0].1, Json::String("x\"\n\u{e9}\u{1F600}".into()));
    }

    #[test]
    fn invalid_texts_are_refused() {
        for (t, msg) in [
            ("{\"a\": 1, \"a\": 2}", "given twice"),
            ("[1, 2", "',' or ']' expected"),
            ("01", "invalid number"),
            ("1.", "invalid number"),
            ("[1] x", "text after the value"),
            ("\"a\u{1}\"", "control character"),
            ("1e400", "beyond the range"),
            ("{'a': 1}", "member name"),
            ("nul", "unknown word"),
        ] {
            let e = parse(t).unwrap_err();
            assert!(e.message().contains(msg), "{t}: {}", e.message());
        }
        let nest = |n: usize| "[".repeat(n) + &"]".repeat(n);
        assert!(parse(&nest(64)).is_ok());
        assert!(parse(&nest(65)).unwrap_err().message().contains("64 levels"));
        assert!(parse(r#""\u+041""#).unwrap_err().message().contains("hexadecimal"));
        assert!(parse("\n  [1,\n   x]").unwrap_err().message().starts_with("JSON at line 3, column 4"));
    }
}
