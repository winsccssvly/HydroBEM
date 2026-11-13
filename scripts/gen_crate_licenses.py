"""Writes licenses/rust-crates.txt: the licences of the Rust crates compiled into the extension
module hydrobem._core (THIRD_PARTY_NOTICES.md).

    python scripts/gen_crate_licenses.py [--check]

The crates are those cargo resolves (Cargo.lock, `cargo metadata`) for crates/hydrobem-python on
x86_64-pc-windows-msvc, following the normal dependencies only: build scripts, proc macros and
development dependencies are not part of the binary. For each crate the file gives its version,
its licence (the SPDX expression of its Cargo.toml), its authors and repository, and the licence
and notice files of its package (LICENSE*, COPYING*, NOTICE*), each distinct text once. A crate
whose package carries no licence file gets the text of its licence with its authors as the
copyright holders (MIT), or a reference to the Apache License 2.0 in LICENSE. Run it after a
change of the dependencies (Cargo.lock); --check fails when the file is not current.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "licenses" / "rust-crates.txt"
TARGET = "x86_64-pc-windows-msvc"
TOP = "hydrobem-python"

MIT = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def linked_crates() -> list[dict]:
    """The packages compiled into the extension module, by name and version."""
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", TARGET],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
    )
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    top = next(p["id"] for p in meta["packages"] if p["name"] == TOP)
    seen, todo = set(), [top]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for d in nodes[pid]["deps"]:
            if not any(k["kind"] is None for k in d["dep_kinds"]):
                continue  # build or development dependency only
            if any("proc-macro" in t["kind"] for t in pkgs[d["pkg"]]["targets"]):
                continue  # runs in the compiler, not part of the binary
            todo.append(d["pkg"])
    ws = set(meta["workspace_members"])
    return sorted((pkgs[i] for i in seen if i not in ws), key=lambda p: (p["name"], p["version"]))


def licence_files(pkg: dict) -> list[Path]:
    base = Path(pkg["manifest_path"]).parent
    names = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE")
    return sorted(f for f in base.iterdir() if f.is_file() and f.name.upper().startswith(names))


def is_apache_text(body: str) -> bool:
    """Whether a licence file is the Apache License 2.0 alone (with at most its appendix template):
    such a file is the text of LICENSE; a file holding more (another licence, notices of code it
    derives from, a filled-in copyright line) is kept verbatim."""
    start, end = body.find("Apache License"), body.find("END OF TERMS AND CONDITIONS")
    if start < 0 or end < start or "Version 2.0" not in body[start:end]:
        return False
    head, tail = body[:start].strip(), body[end + len("END OF TERMS AND CONDITIONS"):].strip()
    template = (tail.startswith("APPENDIX") and tail.endswith("limitations under the License.")
                and ("[yyyy]" in tail or "{yyyy}" in tail))
    return not head and (not tail or template)


def text() -> str:
    crates = linked_crates()
    out = [
        "Licences of the Rust crates compiled into the extension module hydrobem._core",
        "",
        "Written by scripts/gen_crate_licenses.py from Cargo.lock; do not edit. Each crate is used",
        "under its licence as given in its Cargo.toml (for a choice such as \"MIT OR Apache-2.0\",",
        "any of the licences offered). The Apache License 2.0 is the text of LICENSE.",
        "",
        "Crates: " + ", ".join(f"{p['name']} {p['version']}" for p in crates) + ".",
        "",
    ]
    shown: dict[str, str] = {}
    for p in crates:
        out += ["=" * 100, f"{p['name']} {p['version']}", f"Licence: {p.get('license') or 'see the files below'}"]
        if p.get("authors"):
            out.append("Authors: " + ", ".join(p["authors"]))
        if p.get("repository"):
            out.append(f"Repository: {p['repository']}")
        files = licence_files(p)
        for f in files:
            body = f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").strip() + "\n"
            key = hashlib.sha256(body.encode()).hexdigest()
            if is_apache_text(body):
                out += ["", f"--- {f.name}: the Apache License 2.0 (the text of LICENSE)"]
            elif key in shown:
                out += ["", f"--- {f.name}: the same text as {shown[key]}"]
            else:
                shown[key] = f"{f.name} of {p['name']} {p['version']}"
                out += ["", f"--- {f.name}", "", body.rstrip()]
        if not any(f.name.upper().startswith(("LICENSE", "LICENCE", "UNLICENSE")) for f in files):
            lic = p.get("license") or ""
            names = [a.split("<")[0].strip() for a in p.get("authors") or []]
            holders = ", ".join(names) or f"the authors of {p['name']}"
            if "MIT" in lic:
                out += ["", "--- (no licence file in the package) the MIT License", "",
                        f"Copyright (c) {holders}", "", MIT.rstrip()]
            elif "Apache-2.0" in lic:
                out += ["", "--- (no licence file in the package) the Apache License 2.0 (the text of LICENSE)"]
            else:
                raise SystemExit(f"{p['name']} {p['version']}: no licence file and licence {lic!r}")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    new = text()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != new:
            print(f"{OUT.relative_to(ROOT)} is not current: run scripts/gen_crate_licenses.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(new, encoding="utf-8", newline="\n")
    print(f"{OUT.relative_to(ROOT)}: {new.count(chr(10) + '=' * 100)} crates")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
