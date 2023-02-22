#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build IREE's runtime using CMake. Designed for CI, but can be run manually.
# This uses previously cached build results and does not clear build
# directories.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_RUNTIME_SMALL_BUILD_DIR:-build-runtime-small}}"

source build_tools/cmake/setup_build.sh
source build_tools/cmake/setup_ccache.sh

IREE_OPTS="-DIREE_PLATFORM_GENERIC \
  -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 \
  -DIREE_FILE_IO_ENABLE=0 \
  -DIREE_TIME_NOW_FN=\"{ return 0; }\""

"${CMAKE_BIN?}" -B "${BUILD_DIR}" \
  -G Ninja . \
  -DPython3_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DPYTHON_EXECUTABLE="${IREE_PYTHON3_EXECUTABLE}" \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DIREE_SIZE_OPTIMIZED=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_HOST_BIN_DIR=$(realpath build-host/install/bin) \
  -DIREE_BUILD_SAMPLES=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DCMAKE_SYSTEM_NAME=Generic \
  -DIREE_ENABLE_THREADING=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF \
  -DCMAKE_C_FLAGS="${IREE_OPTS}"

"${CMAKE_BIN?}" --build "${BUILD_DIR}" --target samples/static_library/all
