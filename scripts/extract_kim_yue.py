"""Extracts the semi-analytic second-order potential force QTFs of a bottom-mounted vertical
cylinder (h/a = 1) from table 3 of M.-H. Kim and D. K. P. Yue (1990), The complete second-order
diffraction solution for an axisymmetric body. Part 2. Bichromatic incident waves and body
motions, J. Fluid Mech. 211, 557-593:
|f_p| / (rho g a A_j A_l) at (nu_j a, nu_l a) = (1.0, 2.0) and (1.4, 1.6), difference and sum
frequency (tests/data/kim_yue1990.txt).

The input is a local text conversion of the paper (`pdftotext -layout`; the row checked against
the PDF page 574; the paper itself is not part of the repository); the numbers are parsed, not
typed.

Usage: python scripts/extract_kim_yue.py <text conversion of the paper>
Writes tests/data/kim_yue1990.txt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "kim_yue1990.txt"


def main():
    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    pages = text.split("\f")
    # the page of table 3: its caption and the row of the semi-analytic values above it
    page = next(p for p in pages if re.search(r"TABLE\s*3\.\s*Magnitudes of the second-order sum", p))
    head, rest = page.split("Difference frequency", 1)[1].split("(b-a)/h", 1)
    # the column pairs (the conversion reads "1.0" as "1 .o" in places)
    pairs = re.findall(r"\(\s*(\d\.\d)\s*,\s*(\d\.\d)\s*\)", head.replace(" .", ".").replace(".o", ".0"))
    # the semi-analytic values: the conversion splits their row and may reorder it, so each is put in
    # the column whose converged numerical value (the last partition radius) is within 1 % of it
    semi = [float(x) for x in re.findall(r"\d+\.\d{3}", head.split(")", len(pairs))[-1])]
    last = [float(x) for x in re.findall(r"\d+\.\d{3}", rest.split("TABLE")[0].strip().splitlines()[-1])]
    if len(pairs) != 4 or len(semi) != 4 or len(last) != 4:
        sys.exit(f"table 3 not recognized: pairs {pairs}, values {semi}, {last}")
    vals = []
    for x in last:
        near = [v for v in semi if abs(v - x) <= 0.01 * x]
        if len(near) != 1:
            sys.exit(f"table 3: no single semi-analytic value next to {x}: {semi}")
        vals.append(near[0])
    out = [
        "# M.-H. Kim and D. K. P. Yue (1990), J. Fluid Mech. 211, 557-593, table 3, semi-analytic",
        "# (extracted by scripts/extract_kim_yue.py): bottom-mounted vertical cylinder, h/a = 1;",
        "# |f_p| / (rho g a A_j A_l) of the second-order potential force",
        "# kind nu_j_a nu_l_a value",
    ]
    for (a, b), v, kind in zip(pairs, vals, ("difference", "difference", "sum", "sum"), strict=True):
        out.append(f"{kind} {a} {b} {v}")
    OUT.write_bytes(("\n".join(out) + "\n").encode())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
