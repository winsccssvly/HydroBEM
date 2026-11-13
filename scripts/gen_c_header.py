"""Generates crates/hydrobem-c/include/hydrobem.h from the sources of the C ABI.

The header is the C declaration of what crates/hydrobem-c/src/*.rs exports, read from them:
- `pub const HB_...: T = v;` a macro, `pub type` a typedef (the progress callback a function
  pointer type);
- a `#[repr(C)]` struct its C struct, a public struct `Hb...` without it an opaque type;
- every `#[unsafe(no_mangle)] pub [unsafe] extern "C" fn` its prototype;
with the doc comments of each. The Rust types map to C ones (`i32` int32_t, `usize` size_t,
`*const T` const T *, ...); a type that has no mapping fails the run, so that the header never
declares something the library does not export.

Usage: python scripts/gen_c_header.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "crates" / "hydrobem-c" / "src"
OUT = ROOT / "crates" / "hydrobem-c" / "include" / "hydrobem.h"
# the order of the sections of the header
FILES = ["status.rs", "table.rs", "info.rs", "model.rs", "object.rs", "mesh.rs", "query.rs", "results.rs"]
BASE = {"i32": "int32_t", "i64": "int64_t", "u32": "uint32_t", "u64": "uint64_t", "usize": "size_t",
        "f64": "double", "f32": "float", "c_char": "char", "c_void": "void"}  # fmt: skip
VALUES = {"u64::MAX": "UINT64_MAX"}


def c_type(t: str, names: set[str]) -> str:
    t = t.strip()
    for head, const in (("*const ", True), ("*mut ", False)):
        if t.startswith(head):
            inner = t[len(head) :].strip()
            c = c_type(inner, names)
            if inner.startswith("*"):
                return f"{c}const *" if const else f"{c}*"
            return f"const {c} *" if const else f"{c} *"
    if t in BASE:
        return BASE[t]
    if t in names:
        return t
    sys.exit(f"gen_c_header: no C type for the Rust type {t!r}")


def decl(t: str, name: str, names: set[str]) -> str:
    c = c_type(t, names)
    return f"{c}{name}" if c.endswith("*") else f"{c} {name}"


def comment(doc: list[str], indent: str = "") -> list[str]:
    if not doc:
        return []
    doc = ["Arguments:" if d == "# Safety" else d for d in doc]  # the contract of the pointers
    return [f"{indent}/*"] + [f"{indent} *{(' ' + d) if d else ''}" for d in doc] + [f"{indent} */"]


def params(text: str) -> list[tuple[str, str]]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    res = []
    for p in out:
        name, t = p.split(":", 1)
        res.append((name.strip(), t.strip()))
    return res


def parse(path: Path):
    """The items of a source file in order: (kind, name, doc, details)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    items, doc, repr_c, no_mangle, k = [], [], False, False, 0
    while k < len(lines):
        s = lines[k].strip()
        if s.startswith("#[cfg(test)]"):
            break
        if s.startswith("///"):
            doc.append(s[3:].removeprefix(" "))
        elif s == "#[repr(C)]":
            repr_c = True
        elif s == "#[unsafe(no_mangle)]":
            no_mangle = True
        elif s.startswith("#["):
            pass
        elif m := re.match(r"pub const (HB_\w+): ([\w:]+) = (.+);$", s):
            items.append(("const", m[1], doc, VALUES.get(m[3], m[3])))
        elif m := re.match(r"pub type (Hb\w+) = (.+);$", s):
            items.append(("type", m[1], doc, m[2]))
        elif m := re.match(r"pub struct (Hb\w+) \{$", s):
            fields = []
            k += 1
            while lines[k].strip() != "}":
                f = lines[k].strip()
                if f.startswith("pub "):
                    name, t = f[4:].rstrip(",").split(":", 1)
                    fields.append((name.strip(), t.strip()))
                k += 1
            items.append(("struct" if repr_c else "opaque", m[1], doc, fields))
        elif m := re.match(r"pub (?:unsafe )?extern \"C\" fn (hb_\w+)\((.*)$", s):
            if not no_mangle:
                sys.exit(f"gen_c_header: {m[1]} is not #[unsafe(no_mangle)]")
            sig = m[2]
            while not re.search(r"\)\s*(->\s*[\w*\s]+)?\{$", sig):
                k += 1
                sig += " " + lines[k].strip()
            body = re.match(r"(.*)\)\s*(?:->\s*([\w*\s]+?))?\s*\{$", sig)
            items.append(("fn", m[1], doc, (params(body[1]), (body[2] or "").strip())))
        else:
            if s and not s.startswith("//"):
                doc = []
                repr_c = no_mangle = False
            k += 1
            continue
        if not (s.startswith("///") or s.startswith("#[")):
            doc, repr_c, no_mangle = [], False, False
        k += 1
    return items


def header() -> str:
    files = {f: parse(SRC / f) for f in FILES}
    found = sorted(p.name for p in SRC.glob("*.rs") if p.name != "lib.rs")
    if found != sorted(FILES):
        sys.exit(f"gen_c_header: the sources {found} are not those of FILES {sorted(FILES)}")
    names = {i[1] for f in files.values() for i in f if i[0] in ("type", "struct", "opaque")}
    out = [
        "/*",
        " * hydrobem.h: the C ABI of the HydroBEM engine (hydrobem.dll). Generated by",
        " * scripts/gen_c_header.py from crates/hydrobem-c/src; never edited by hand.",
        " * The conventions are those of docs/reference/c-api.md.",
        " */",
        "#ifndef HYDROBEM_H",
        "#define HYDROBEM_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#if defined(_WIN32)",
        "#define HB_API __declspec(dllimport)",
        "#else",
        "#define HB_API",
        "#endif",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    # the opaque types first: the functions of every section name them
    for f in FILES:
        for kind, name, doc, _ in files[f]:
            if kind == "opaque":
                out += comment(doc) + [f"typedef struct {name} {name};", ""]
    for f in FILES:
        out.append(f"/* ---- {f.removesuffix('.rs')} */")
        out.append("")
        for kind, name, doc, d in files[f]:
            if kind == "const":
                out += comment(doc) + [f"#define {name} {d}", ""]
            elif kind == "type":
                fn = re.match(r"Option<unsafe extern \"C\" fn\((.*)\)\s*->\s*(\w+)>$", d)
                if fn:
                    args = ", ".join(decl(t, n, names) for n, t in params(fn[1]))
                    out += comment(doc) + [f"typedef {c_type(fn[2], names)} (*{name})({args});", ""]
                else:
                    out += comment(doc) + [f"typedef {c_type(d, names)} {name};", ""]
            elif kind == "struct":
                out += comment(doc) + [f"typedef struct {name} {{"]
                out += [f"    {decl(t, n, names)};" for n, t in d]
                out += [f"}} {name};", ""]
            elif kind == "fn":
                ps, ret = d
                r = c_type(ret, names) if ret else "void"
                args = ", ".join(decl(t, n, names) for n, t in ps) or "void"
                sep = "" if r.endswith("*") else " "
                out += comment(doc) + [f"HB_API {r}{sep}{name}({args});", ""]
    out += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* HYDROBEM_H */", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only check that the header is up to date")
    a = ap.parse_args()
    text = header()
    if a.check:
        if not OUT.exists() or OUT.read_bytes().decode("utf-8") != text:
            sys.exit(f"{OUT} is not up to date: run scripts/gen_c_header.py")
        print("up to date")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("utf-8"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
