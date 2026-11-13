//! Little-endian records of the model and results files: unsigned integers of 1 to
//! 8 bytes, IEEE 754 doubles, texts with a 4-byte length, and a reader that refuses a record cut
//! short. A record is a `Vec<u8>`, read from a `&[u8]`.

use crate::error::{Error, Result};

pub fn put_u(out: &mut Vec<u8>, v: u64, bytes: i32) {
    for k in 0..bytes {
        out.push(((v >> (8 * k)) & 0xff) as u8);
    }
}

pub fn put_i64(out: &mut Vec<u8>, v: i64) {
    put_u(out, v as u64, 8);
}

pub fn put_f64(out: &mut Vec<u8>, v: f64) {
    put_u(out, v.to_bits(), 8);
}

pub fn put_text(out: &mut Vec<u8>, s: &str) {
    put_u(out, s.len() as u64, 4);
    out.extend_from_slice(s.as_bytes());
}

/// Reads the values of a record in the order they were put; a read past its end is ErrorKind::Value.
pub struct Reader<'a> {
    b: &'a [u8],
    where_: String,
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8], where_: String) -> Reader<'a> {
        Reader { b: bytes, where_, at: 0 }
    }
    pub fn u(&mut self, bytes: i32) -> Result<u64> {
        self.need(bytes as usize)?;
        let mut v: u64 = 0;
        for k in 0..bytes as usize {
            v |= (self.b[self.at + k] as u64) << (8 * k);
        }
        self.at += bytes as usize;
        Ok(v)
    }
    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u(8)? as i64)
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u(8)?))
    }
    /// A text (an invalid UTF-8 sequence is replaced; the texts of a record were checked when they
    /// were set).
    pub fn text(&mut self) -> Result<String> {
        let n = self.u(4)? as usize;
        self.need(n)?;
        let s = String::from_utf8_lossy(&self.b[self.at..self.at + n]).into_owned();
        self.at += n;
        Ok(s)
    }
    pub fn done(&self) -> bool {
        self.at == self.b.len()
    }
    /// The bytes not read yet (a bound for the counts a record gives).
    pub fn left(&self) -> usize {
        self.b.len() - self.at
    }
    pub fn where_(&self) -> &str {
        &self.where_
    }
    fn need(&self, n: usize) -> Result<()> {
        if n > self.b.len() - self.at {
            return Err(Error::value(format!("{}: the record is damaged (cut short)", self.where_)));
        }
        Ok(())
    }
}
