"""Writes the expected strings of REPR_CASES in crates/hydrobem/tests/io.rs from its bit patterns.

Each row is (the bits of a double, Python's repr of it, its Fortran field 1PE14.6 and 0PE14.6),
"-" where the field does not fit (an exponent of three digits). The strings are Python's: repr
(the shortest that round trips) and the correctly rounded digits of "%.*e" (the C standard's
conversion), placed in the fields as the Fortran E edit descriptor does: 1P one digit before the
point, 0P none and the exponent one higher, two exponent digits, right-justified in 14 columns.
The bit patterns stay as they are in the file (values of every magnitude and sign).

Usage: python scripts/gen_repr_cases.py [--check]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "crates" / "hydrobem" / "tests" / "io.rs"
ROW = re.compile(r'^    \((0x[0-9a-f]{16}), "[^"]*", "[^"]*", "[^"]*"\),$')


def field(x: float, one_p: bool, width: int = 14, digits: int = 6) -> str:
    if x == 0.0:
        body = "0." + "0" * digits + "E+00"
    else:
        s = f"{abs(x):.{digits if one_p else digits - 1}e}"
        mant, e = s.split("e")
        e = int(e) + (0 if one_p else 1)
        if abs(e) > 99:
            return "-"
        exp = f"E{'-' if e < 0 else '+'}{abs(e):02d}"
        body = mant + exp if one_p else "0." + mant.replace(".", "") + exp
    body = ("-" if x < 0 else "") + body
    return "-" if len(body) > width else body.rjust(width)


def row(bits: str) -> str:
    x = struct.unpack("<d", int(bits, 16).to_bytes(8, "little"))[0]
    return f'    ({bits}, "{x!r}", "{field(x, True)}", "{field(x, False)}"),'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that the rows are up to date")
    a = ap.parse_args()
    lines = FILE.read_bytes().decode("utf-8").split("\n")
    out, n = [], 0
    for line in lines:
        m = ROW.match(line)
        if m:
            line = row(m[1])
            n += 1
        out.append(line)
    text = "\n".join(out)
    if a.check:
        if text != "\n".join(lines):
            sys.exit(f"{FILE}: REPR_CASES is not up to date: run scripts/gen_repr_cases.py")
        print(f"up to date ({n} rows)")
        return
    FILE.write_bytes(text.encode("utf-8"))
    print(f"wrote {n} rows of {FILE}")


if __name__ == "__main__":
    main()
