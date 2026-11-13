#!/usr/bin/env bash
# Runs the Rust component tests, the Python test suite and ruff.
# Usage: scripts/test.sh [full|fast] [Release|Debug]
#   full (default): every test, and the kernel timings (HB_TIMING);
#   fast: without the tests marked slow (about a minute).
# Requires a prior scripts/build.sh of the same configuration (it puts the extension module into
# the package, python/hydrobem; the tests import this checkout's package, before an installed
# one) and a Python with NumPy, SciPy, mpmath, pytest and ruff (PYTHON, default python3).
set -euo pipefail
MODE="${1:-full}"
CONFIG="${2:-Release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
case "$MODE" in
    full|fast) ;;
    *) echo "test.sh: unknown mode $MODE (full or fast)" >&2; exit 1 ;;
esac
if ! ls "$ROOT"/python/hydrobem/_core*.so >/dev/null 2>&1; then
    echo "test.sh: the package has no extension module: run scripts/build.sh $CONFIG" >&2
    exit 1
fi
FLAG=()
if [ "$CONFIG" = Release ]; then FLAG=(--release); fi
# full: every Rust test of the workspace (unit, component, C ABI), with the kernel timings;
# fast: the component tests core only
if [ "$MODE" = fast ]; then
    cargo test "${FLAG[@]}" --manifest-path "$ROOT/Cargo.toml" --test core
    "$PY" -m pytest -q -m "not slow" "$ROOT/tests/python"
else
    HB_TIMING=1 cargo test "${FLAG[@]}" --manifest-path "$ROOT/Cargo.toml" --workspace -- --nocapture
    "$PY" -m pytest -q "$ROOT/tests/python"
fi
"$PY" "$ROOT/scripts/gen_model_classes.py" --check
"$PY" "$ROOT/scripts/gen_result_views.py" --check
"$PY" "$ROOT/scripts/gen_crate_licenses.py" --check
"$PY" "$ROOT/scripts/gen_log_gauss.py" --check
"$PY" "$ROOT/scripts/gen_c_header.py" --check
"$PY" "$ROOT/scripts/gen_repr_cases.py" --check
"$PY" "$ROOT/scripts/gen_sphere_array_reference.py" --check
"$PY" "$ROOT/scripts/check_doc_math.py"
"$PY" "$ROOT/scripts/rewrite_doc_refs.py" --check
LINT=("$ROOT/python" "$ROOT/tests" "$ROOT/scripts")
if [ -d "$ROOT/examples" ]; then LINT+=("$ROOT/examples"); fi
ruff check "${LINT[@]}"
