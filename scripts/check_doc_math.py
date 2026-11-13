"""Checks the notation of the published documents, whose formulas are written in GitHub math
(docs/development/rules.md): every Markdown file of docs/, README.md, CHANGELOG.md and
examples/README.md.

    python scripts/check_doc_math.py [--dump FILE] [file.md ...]

- Formulas: inline `$...$` on one line (no space after the opening or before the closing `$`, no
  word character just outside it), displayed `$$` on lines of their own; braces, \\left/\\right and
  \\begin/\\end balanced; no `\\{` `\\}` (written `\\lbrace` `\\rbrace`, which Markdown leaves
  alone) and, in a table row, no `|` (written `\\vert`, `\\lvert`, `\\rvert`).
- No Unicode math symbols (Greek letters, operators, super- and subscripts, primes, ...) outside
  formulas and code (fenced blocks and backtick spans).
- No characters of a wrong code page: the CJK ranges and the replacement character U+FFFD, which
  a text saved in another encoding leaves (a section sign read in a Korean code page, for example),
  anywhere in the file.
- Captions: an image is followed by the paragraph `*Figure N. ...*`, a table is preceded by the
  paragraph `*Table N. ...*` (the marker comment of a generated table may stand between them),
  numbered from 1 per page.

--dump FILE writes every formula as JSON ([{"tex", "display", "where"}]) for a TeX parser.
Exits with 1 and lists the findings if any.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the characters of a wrong code page: CJK symbols, kana, the unified ideographs, Hangul, and the
# replacement character
MOJIBAKE = re.compile("[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef\ufffd]")
DIRS = [ROOT / "docs"]
FILES = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "examples" / "README.md"]


def is_math_char(ch):
    """A character that belongs in a formula, not in prose."""
    o = ord(ch)
    return (0x0370 <= o <= 0x03FF  # Greek
            or 0x1D400 <= o <= 0x1D7FF  # mathematical alphanumerics
            or 0x2200 <= o <= 0x22FF  # mathematical operators
            or 0x27C0 <= o <= 0x27EF  # angle brackets and other math symbols
            or 0x2070 <= o <= 0x209F  # super- and subscripts
            or 0x2190 <= o <= 0x21FF  # arrows
            or 0x2032 <= o <= 0x2037  # primes
            or 0x1D2C <= o <= 0x1D6A  # modifier letters (ᵀ, ...)
            or 0x2016 == o  # double bar
            or 0x0302 <= o <= 0x0305  # combining hat, tilde, bar
            or ch in "²³¹×÷±·¼½¾°ˆ˜")  # fmt: skip


def code_free(line):
    """The line with its backtick spans blanked."""
    return re.sub(r"(`+)(.+?)\1", lambda m: " " * len(m.group(0)), line)


def balanced(tex):
    """The structural errors of one formula."""
    errs = []
    depth = 0
    i = 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            if tex[i + 1:i + 2] in "{}":
                errs.append("\\{ or \\}: write \\lbrace, \\rbrace")
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                errs.append("unbalanced }")
                depth = 0
        i += 1
    if depth:
        errs.append("unbalanced {")
    if len(re.findall(r"\\left(?![a-zA-Z])", tex)) != len(re.findall(r"\\right(?![a-zA-Z])", tex)):
        errs.append("\\left and \\right unbalanced")
    if re.findall(r"\\begin\{(\w+)\}", tex) != re.findall(r"\\end\{(\w+)\}", tex):
        errs.append("\\begin and \\end unbalanced")
    if re.search(r"\\begin\{(align|equation|eqnarray)\*?\}", tex):
        errs.append("align/equation environment: use aligned inside $$")
    return errs


def check(path, rel):
    """(findings, formulas, figures, tables) of one file."""
    findings, formulas = [], []
    lines = path.read_text(encoding="utf-8").split("\n")
    fenced = False
    display = None  # (start line, text) inside $$
    figures = tables = 0
    prev_table = False
    for no, raw in enumerate(lines, 1):
        where = f"{rel}:{no}"
        stripped = raw.strip()
        wrong = sorted(set(MOJIBAKE.findall(raw)))
        if wrong:
            findings.append(f"{where}: characters of a wrong code page: {' '.join(wrong)}: {stripped[:100]}")
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if display is not None:
            if stripped == "$$":
                tex = "\n".join(display[1])
                formulas.append({"tex": tex, "display": True, "where": f"{rel}:{display[0]}"})
                findings += [f"{rel}:{display[0]}: {m}" for m in balanced(tex)]
                display = None
            elif "$" in code_free(raw):
                findings.append(f"{where}: $ inside a displayed formula")
            else:
                display[1].append(raw)
            continue
        if stripped == "$$":
            display = (no, [])
            continue
        line = code_free(raw)
        if "$$" in line:
            findings.append(f"{where}: $$ not on a line of its own")
            continue
        # a table row: a line of | cells in a table (a header followed by its |---| line)
        nxt_line = lines[no].strip() if no < len(lines) else ""
        in_table = stripped.startswith("|") and (prev_table or re.match(r"\|\s*:?-", nxt_line) is not None)
        # the inline formulas
        parts = line.split("$")
        if len(parts) % 2 == 0:
            findings.append(f"{where}: odd number of $")
            continue
        pos = 0
        prose = []
        for k, part in enumerate(parts):
            if k % 2 == 0:
                prose.append(part)
            else:
                tex = part
                start = pos - 1  # index of the opening $
                end = pos + len(part)  # index of the closing $
                if not tex or tex[0].isspace() or tex[-1].isspace():
                    findings.append(f"{where}: space inside $...$: ${tex}$")
                if start > 0 and (line[start - 1].isalnum() or line[start - 1] == "\\"):
                    findings.append(f"{where}: word character before $: ...{line[max(0, start - 8):start + 8]}")
                if end + 1 < len(line) and line[end + 1].isalnum():
                    findings.append(f"{where}: word character after $: {line[end - 8:end + 4]}...")
                if in_table and "|" in tex:
                    findings.append(f"{where}: | in a formula in a table: ${tex}$")
                findings += [f"{where}: {m}: ${tex}$" for m in balanced(tex)]
                formulas.append({"tex": tex, "display": False, "where": where})
            pos += len(part) + 1
        text = "".join(prose)
        bad = sorted({c for c in text if is_math_char(c)})
        if bad:
            findings.append(f"{where}: Unicode math outside a formula: {' '.join(bad)}: {stripped[:100]}")
        # captions
        if re.match(r"!\[[^\]]*\]\([^)]*\)\s*$", stripped):
            figures += 1
            nxt = next((x.strip() for x in lines[no:] if x.strip()), "")
            if not nxt.startswith(f"*Figure {figures}. "):
                findings.append(f"{where}: figure without the caption '*Figure {figures}. ...*' below it")
        if in_table and not prev_table:
            tables += 1
            # the paragraph before the table (its lines up to the blank line before it; the
            # marker comment of a generated table between them is skipped)
            k = no - 2
            while k >= 0 and (not lines[k].strip() or lines[k].strip().startswith("<!--")):
                k -= 1
            last = lines[k].strip() if k >= 0 else ""
            while k > 0 and lines[k - 1].strip():
                k -= 1
            first = lines[k].strip() if k >= 0 else ""
            if not (last.endswith("*") and first.startswith(f"*Table {tables}. ")):
                findings.append(f"{where}: table without the caption '*Table {tables}. ...*' above it")
        prev_table = in_table
    if display is not None:
        findings.append(f"{rel}:{display[0]}: $$ not closed")
    if fenced:
        findings.append(f"{rel}: fenced block not closed")
    return findings, formulas, figures, tables


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    dump = None
    if "--dump" in args:
        i = args.index("--dump")
        dump = Path(args[i + 1])
        del args[i:i + 2]
    files = [Path(a).resolve() for a in args] or FILES + sorted(p for d in DIRS for p in d.rglob("*.md"))
    findings, formulas = [], []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        f, m, nfig, ntab = check(p, rel)
        findings += f
        formulas += m
        print(f"{rel}: {sum(not x['display'] for x in m)} inline, {sum(x['display'] for x in m)} displayed, "
              f"{nfig} figures, {ntab} tables")  # fmt: skip
    if dump:
        dump.write_text(json.dumps(formulas, ensure_ascii=False), encoding="utf-8", newline="\n")
    for f in findings:
        print(f)
    if findings:
        print(f"check_doc_math: {len(findings)} findings", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
