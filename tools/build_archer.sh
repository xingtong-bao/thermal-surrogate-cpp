#!/usr/bin/env bash
# tools/build_archer.sh -- build LLVM's Archer OMPT tool so ThreadSanitizer understands OpenMP
#
# Why
#   ThreadSanitizer only sees synchronisation it instrumented. The OpenMP
#   runtime (libomp) is a prebuilt, uninstrumented library, so the barriers
#   between the red and black half-sweeps in src/fd_solver.cpp are invisible to
#   TSan and every cross-thread access looks like a race (32 false reports on
#   FdSolver.ThreadsParameterDoesNotChangeTheContractAnswer without this tool).
#   Archer (openmp/tools/archer in llvm-project) is an OMPT tool: libomp calls
#   it at every barrier / task / lock event and it tells TSan "happens-before"
#   through the annotation API, so TSan then reports only real races.
#
# What it does
#   1. sparse-clones llvm-project at the tag matching the installed libomp
#      (brew on macOS; override with LLVM_TAG), only the openmp/ directory;
#   2. compiles ompt-tsan.cpp into libarcher.dylib / .so against the
#      installed omp-tools.h; TSan symbols are resolved by dlsym at run time,
#      so the tool needs no TSan at link time (-undefined dynamic_lookup).
#
# Usage
#   tools/build_archer.sh [out_dir]          # default: build-tsan/archer
#   OMP_TOOL_LIBRARIES=build-tsan/archer/libarcher.dylib \
#   TSAN_OPTIONS=ignore_noninstrumented_modules=1 \
#     ./build-tsan/ts_tests --gtest_filter='FdSolver.*:Concurrency.*'
set -euo pipefail
out="${1:-build-tsan/archer}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

if [[ "$(uname)" == "Darwin" ]]; then
  omp_prefix="$(brew --prefix libomp)"
  ver="$(brew list --versions libomp | awk '{print $2}')"
  ext=dylib
else
  omp_prefix="${OMP_PREFIX:-/usr}"
  ver="${LLVM_VERSION:?set LLVM_VERSION to the installed libomp version, e.g. 18.1.8}"
  ext=so
fi
tag="${LLVM_TAG:-llvmorg-$ver}"
inc="$omp_prefix/include"
[[ -f "$inc/omp-tools.h" ]] || { echo "omp-tools.h not found under $inc" >&2; exit 1; }

src="$out/llvm-project"
if [[ ! -f "$src/openmp/tools/archer/ompt-tsan.cpp" ]]; then
  rm -rf "$src"
  git clone -q --filter=blob:none --sparse --depth 1 --branch "$tag" https://github.com/llvm/llvm-project.git "$src"
  git -C "$src" sparse-checkout set openmp >/dev/null
fi
echo "archer source: $src/openmp/tools/archer (tag $tag)"

extra=()
[[ "$ext" == dylib ]] && extra=(-undefined dynamic_lookup)
clang++ -std=c++17 -O2 -fPIC -shared -I"$inc" "${extra[@]}" \
  "$src/openmp/tools/archer/ompt-tsan.cpp" -o "$out/libarcher.$ext"
echo "built $out/libarcher.$ext"
echo "run tests with: OMP_TOOL_LIBRARIES=$out/libarcher.$ext TSAN_OPTIONS=ignore_noninstrumented_modules=1 ./build-tsan/ts_tests --gtest_filter='FdSolver.*'"
