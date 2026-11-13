#!/usr/bin/env bash
# Build the cargo workspace (the Rust engine hydrobem_core, pure Rust, the Python extension
# module hydrobem._core, PyO3, and the C ABI library target/<profile>/libhydrobem.so), run the
# Rust tests and put the extension module into the Python package in place
# (python/hydrobem/_core.abi3.so, as "maturin develop --skip-install" does); scripts/test.sh then
# tests this checkout's package.
# Usage: scripts/build.sh [Release|Debug]
# An installed package instead: "pip install ." (maturin, pyproject.toml) or "maturin develop --release".
# The Python of the extension's build (PyO3): PYO3_PYTHON, else PYTHON, else python3.
set -euo pipefail
CONFIG="${1:-Release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$CONFIG" in
    Release) FLAG=(--release); DIR=release ;;
    Debug) FLAG=(); DIR=debug ;;
    *) echo "build.sh: unknown configuration $CONFIG (Release or Debug)" >&2; exit 1 ;;
esac
export PYO3_PYTHON="${PYO3_PYTHON:-$(command -v "${PYTHON:-python3}")}"
cargo build "${FLAG[@]}" --manifest-path "$ROOT/Cargo.toml"
cargo test "${FLAG[@]}" --manifest-path "$ROOT/Cargo.toml"
cp "$ROOT/target/$DIR/lib_core.so" "$ROOT/python/hydrobem/_core.abi3.so"
