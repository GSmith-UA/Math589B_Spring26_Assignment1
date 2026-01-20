#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Build a shared library:
#  - Linux: librod.so
#  - macOS: librod.dylib

CXX="${CXX:-g++}"
CXXFLAGS="-O3 -std=c++17 -fPIC"

UNAME="$(uname -s)"
if [[ "$UNAME" == "Darwin" ]]; then
  OUT="librod.dylib"
  $CXX $CXXFLAGS -dynamiclib -o "$OUT" rod_energy.cpp
else
  OUT="librod.so"
  $CXX $CXXFLAGS -shared -o "$OUT" rod_energy.cpp
fi

echo "Built: $(pwd)/$OUT"
