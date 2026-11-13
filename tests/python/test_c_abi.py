"""The C ABI from C: the header compiles with a C compiler (MSVC, as C with /std:c11 and as C++), the program
crates/hydrobem-c/tests/c_api.c links the import library and runs on the DLL that
scripts/build.bat built (target/release). The functions themselves are tested from Rust
(crates/hydrobem-c/tests/c_api.rs); the header is checked against the sources by
scripts/gen_c_header.py --check (scripts/test.bat). Skipped where no MSVC is installed."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "crates" / "hydrobem-c"
RELEASE = ROOT / "target" / "release"
PROGRAMS = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
VSWHERE = PROGRAMS / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def _vcvars():
    if os.name != "nt" or not VSWHERE.exists():
        return None
    out = subprocess.run([str(VSWHERE), "-latest", "-products", "*", "-requires",
                          "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                         capture_output=True, text=True)  # fmt: skip
    p = Path(out.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if out.stdout.strip() else None
    return p if p and p.exists() else None


def test_a_c_program_on_the_header_and_the_dll(tmp_path):
    vcvars = _vcvars()
    if vcvars is None:
        pytest.skip("no MSVC (vswhere and vcvars64.bat) on this machine")
    dll, lib = RELEASE / "hydrobem.dll", RELEASE / "hydrobem.dll.lib"
    assert dll.exists() and lib.exists(), "build the workspace first (scripts\\build.bat Release)"
    include = CRATE / "include"
    shutil.copy(dll, tmp_path / "hydrobem.dll")
    exe, header_exe = tmp_path / "c_api.exe", tmp_path / "header.exe"
    cpp = tmp_path / "header.cpp"
    cpp.write_text('#include "hydrobem.h"\nint main() { return hb_abi_version() == HB_ABI_VERSION ? 0 : 1; }\n')
    # a batch file: cmd reads its quoting as written
    lines = [
        "@echo off",
        f'call "{vcvars}" >nul || exit /b 1',
        f'cl /nologo /std:c11 /W4 /WX /I"{include}" "{CRATE / "tests" / "c_api.c"}" /Fe"{exe}" "{lib}" || exit /b 1',
        f'cl /nologo /W4 /WX /EHsc /I"{include}" "{cpp}" /Fe"{header_exe}" "{lib}" || exit /b 1',
    ]
    bat = tmp_path / "build.bat"
    bat.write_text(os.linesep.join(lines) + os.linesep)
    b = subprocess.run(["cmd", "/d", "/c", str(bat)], capture_output=True, text=True, cwd=tmp_path)
    assert b.returncode == 0, b.stdout + b.stderr
    runs = {p: subprocess.run([str(p)], capture_output=True, text=True, cwd=tmp_path) for p in (exe, header_exe)}
    for r in runs.values():
        assert r.returncode == 0, r.stdout + r.stderr
    assert "c_api: ok" in runs[exe].stdout
