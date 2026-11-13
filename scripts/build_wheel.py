"""Build the wheel to distribute (maturin), without the paths of the building machine.

    python scripts/build_wheel.py [--out DIR]   (default dist/)

The panic messages of the compiled module name source locations; rustc writes them with the
paths of the build (the checkout, the cargo registry, the toolchain), e.g. the user's home folder.
This script remaps them to neutral prefixes (``--remap-path-prefix``). The flags are passed as
CARGO_ENCODED_RUSTFLAGS, which replaces the rustflags of .cargo/config.toml, so those are read
from it and passed along (paths with spaces are kept whole by the encoded form). The wheel built is
then searched for the home folder and the checkout; a file that names one fails the build (the
SBOM of the crates, which names them, is left out in pyproject.toml). ``pip install .``
and ``scripts/build.bat`` build without the remapping: their module stays on the machine.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "x86_64-pc-windows-msvc"


def rustflags() -> list[str]:
    """The rustflags of .cargo/config.toml for the target, and the path remapping."""
    config = tomllib.loads((ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8"))
    flags = list(config.get("target", {}).get(TARGET, {}).get("rustflags", []))
    home = Path.home()
    cargo_home = Path(os.environ.get("CARGO_HOME", home / ".cargo"))
    rustup_home = Path(os.environ.get("RUSTUP_HOME", home / ".rustup"))
    # the longest prefixes first: rustc applies the last matching flag, so the general home folder
    # goes before the folders inside it
    for path, to in ((home, "~"), (rustup_home, "rustup"), (cargo_home, "cargo"), (ROOT, "hydrobem")):
        flags.append(f"--remap-path-prefix={path}={to}")
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    a = ap.parse_args()
    env = dict(os.environ, CARGO_ENCODED_RUSTFLAGS="\x1f".join(rustflags()))
    env.pop("RUSTFLAGS", None)
    cmd = [sys.executable, "-m", "maturin", "build", "--release", "--out", str(a.out)]
    start = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if rc == 0:
        for wheel in a.out.glob("*.whl"):
            if wheel.stat().st_mtime >= start - 1:
                rc = rc or check_paths(wheel)
    sys.exit(rc)


def check_paths(wheel: Path) -> int:
    """1 (and the members named) if a file of the wheel contains a path of the building machine."""
    paths = {str(Path.home()), str(ROOT)}
    needles = {v.lower().encode() for p in paths for v in (p, p.replace("\\", "/"), p.replace("\\", "\\\\"))}
    found = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            data = z.read(name).lower()
            if any(n in data for n in needles):
                found.append(name)
    for name in found:
        print(f"{wheel.name}: {name} contains a path of this machine", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    main()
