@echo off
rem Build the cargo workspace (the Rust engine hydrobem_core, pure Rust, the Python extension
rem module hydrobem._core, PyO3, and the C ABI target\<profile>\hydrobem.dll with its import
rem library), run the Rust tests and put the extension module
rem into the Python package in place (python\hydrobem\_core.pyd, as "maturin develop
rem --skip-install" does); scripts\test.bat then tests this checkout's package.
rem Usage: scripts\build.bat [Release|Debug]
rem
rem An installed package instead: "pip install ." (maturin builds the module and packs the package,
rem pyproject.toml) or "maturin develop --release" (an editable install).
rem
rem Requires a Rust toolchain for x86_64-pc-windows-msvc (rustup) with the Visual Studio Build
rem Tools (rustc finds the MSVC linker itself); nothing else is installed. The Python of the
rem extension's build (PyO3): PYO3_PYTHON, else that of HB_PYTHON_ENV or CONDA_PREFIX when set, else
rem the python on PATH.
setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=Release
set ROOT=%~dp0..
if /i "%CONFIG%"=="Release" (
  set "PROFILE_FLAG=--release"
  set "PROFILE_DIR=release"
) else if /i "%CONFIG%"=="Debug" (
  set "PROFILE_FLAG="
  set "PROFILE_DIR=debug"
) else (
  echo build.bat: unknown configuration "%CONFIG%" ^(Release or Debug^) 1>&2
  exit /b 1
)
where cargo >nul 2>nul || (
  echo build.bat: cargo not found; install Rust from https://rustup.rs 1>&2
  exit /b 1
)
set "ENV=%HB_PYTHON_ENV%"
if "%ENV%"=="" set "ENV=%CONDA_PREFIX%"
if "%PYO3_PYTHON%"=="" if not "%ENV%"=="" (
  if exist "%ENV%\python.exe" (set "PYO3_PYTHON=%ENV%\python.exe") else set "PYO3_PYTHON=%ENV%\Scripts\python.exe"
)

echo build.bat: %CONFIG%
cargo build %PROFILE_FLAG% --manifest-path "%ROOT%\Cargo.toml" || exit /b 1
cargo test %PROFILE_FLAG% --manifest-path "%ROOT%\Cargo.toml" || exit /b 1

copy /y "%ROOT%\target\%PROFILE_DIR%\_core.dll" "%ROOT%\python\hydrobem\_core.pyd" >nul || exit /b 1
