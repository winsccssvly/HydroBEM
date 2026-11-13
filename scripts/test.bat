@echo off
rem Runs the Rust component tests, the Python test suite and ruff.
rem Usage: scripts\test.bat [full|fast] [Release|Debug]
rem   full (default): every test, and the kernel timings (HB_TIMING);
rem   fast: without the tests marked slow (about a minute).
rem Requires a prior scripts\build.bat of the same configuration (the extension module in the
rem package, python\hydrobem\_core.pyd: the tests import this checkout's package, before an
rem installed one) and a Python environment with NumPy, SciPy, mpmath, pytest and ruff: the
rem environment directory is taken from HB_PYTHON_ENV, else from CONDA_PREFIX (an activated conda
rem environment); a venv directory works as well.
setlocal
set ROOT=%~dp0..
set "MODE=%~1"
if "%MODE%"=="" set MODE=full
set "CONFIG=%~2"
if "%CONFIG%"=="" set CONFIG=Release
if /i not "%MODE%"=="full" if /i not "%MODE%"=="fast" (
  echo test.bat: unknown mode "%MODE%" ^(full or fast^) 1>&2
  exit /b 1
)
if not exist "%ROOT%\python\hydrobem\_core.pyd" (
  echo test.bat: the package has no extension module: run scripts\build.bat %CONFIG% 1>&2
  exit /b 1
)
set "PROFILE_FLAG="
if /i "%CONFIG%"=="Release" set "PROFILE_FLAG=--release"
set "ENV=%HB_PYTHON_ENV%"
if "%ENV%"=="" set "ENV=%CONDA_PREFIX%"
if "%ENV%"=="" (
  echo test.bat: set HB_PYTHON_ENV or activate the conda environment 1>&2
  exit /b 1
)
set "PY=%ENV%\python.exe"
if not exist "%PY%" set "PY=%ENV%\Scripts\python.exe"
rem NumPy's linear algebra DLLs live in Library\bin of a conda environment
set "PATH=%ENV%;%ENV%\Library\bin;%ENV%\Scripts;%PATH%"
rem this checkout's package (pytest reads hydrobem.errors from pyproject.toml before the tests' conftest)
set "PYTHONPATH=%ROOT%\python;%PYTHONPATH%"
rem full: every Rust test of the workspace (the unit tests, the component tests core, model_api and
rem pools, the C ABI tests), with the kernel timings; fast: the component tests core only
if /i "%MODE%"=="full" (
  set HB_TIMING=1
  cargo test %PROFILE_FLAG% --manifest-path "%ROOT%\Cargo.toml" --workspace -- --nocapture || exit /b 1
) else (
  cargo test %PROFILE_FLAG% --manifest-path "%ROOT%\Cargo.toml" --test core || exit /b 1
)
set HB_TIMING=
if /i "%MODE%"=="fast" (
  "%PY%" -m pytest -q -m "not slow" "%ROOT%\tests\python" || exit /b 1
) else (
  "%PY%" -m pytest -q "%ROOT%\tests\python" || exit /b 1
)
"%PY%" "%ROOT%\scripts\gen_model_classes.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_result_views.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_crate_licenses.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_log_gauss.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_c_header.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_repr_cases.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\gen_sphere_array_reference.py" --check || exit /b 1
"%PY%" "%ROOT%\scripts\check_doc_math.py" || exit /b 1
"%PY%" "%ROOT%\scripts\rewrite_doc_refs.py" --check || exit /b 1
ruff check "%ROOT%\python" "%ROOT%\tests" "%ROOT%\scripts" || exit /b 1
if exist "%ROOT%\examples" (ruff check "%ROOT%\examples" || exit /b 1)
