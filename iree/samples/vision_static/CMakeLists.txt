# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER})
  return()
endif()

# Set iree-translate binary.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

## Example with VM bytecode module.
# Setup args for iree-translate.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-input-type=mhlo")
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-embedded=false")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=mnist.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/../models/mnist.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "mnist.vmfb")

# Custom command for iree-translate to generate static library and bytecode.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/mnist.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist.o
    ${CMAKE_CURRENT_BINARY_DIR}/mnist.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../models/mnist.mlir"
)

# Tell cmake about the mnist library so it will link it.
add_library(mnist
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/mnist.o)

SET_TARGET_PROPERTIES(
  mnist
  PROPERTIES
  LINKER_LANGUAGE C)

# Note: If you're cross compiling the simple_mul for a different backend, you'll
# need to run iree-translate manually to produce the desired '.vmfb' and '.h/.o'
# files. Substitute the 'simple_mul' dependency in iree_cc_binary() below with
# your own static library and the `simple_mul.vmfb` in the iree_c_embed_data()
# rule. You can use paths to files, i.e. 'path/to/generated/output.vmfb' in
# place of CMake targets.

# Generate the embed data with the bytecode module
iree_c_embed_data(
  NAME
    mnist_c
  IDENTIFIER
    iree_samples_vision_static_mnist
  GENERATED_SRCS
    mnist.vmfb
  C_FILE_OUTPUT
    mnist_c.c
  H_FILE_OUTPUT
    mnist_c.h
  FLATTEN
  PUBLIC
)

iree_cc_binary(
NAME
  mnist_static_bytecode
SRCS
  "create_bytecode_module.c"
  "mnist_static.c"
DATA
  "iree::samples::vision_static::mnist_test.png"
DEPS
  ::mnist_c
  iree::runtime
  iree::hal::local::loaders::static_library_loader
  iree::hal::local::sync_driver
  iree::tools::utils::image_util
  mnist
)

iree_lit_test(
  NAME
    mnist_static_bytecode_test
  TEST_FILE
    "mnist_static_bytecode_test.txt"
  DATA
    ::mnist_static_bytecode
    iree::tools::IreeFileCheck
  LABELS
    "hostonly"
)

if(NOT (${IREE_ENABLE_EMITC} OR DEFINED IREE_HOST_BINARY_ROOT))
  return()
endif()

## Example with VM C module.
# Setup args for iree-translate.
# Build the static library and the c code from the mnist.mlir in iree/samples/models.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-input-type=mhlo")
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-c-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-embedded=false")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=mnist_c_module.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/../models/mnist.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "mnist_emitc.h")

# Custom command for iree-translate to generate static library and C module.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_c_module.h
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_c_module.o
    ${CMAKE_CURRENT_BINARY_DIR}/mnist_emitc.h
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/../models/mnist.mlir"
)

add_custom_target(
  mnist_gen
  DEPENDS
    "mnist_emitc.h"
)

add_library(mnist_c_module
 STATIC
 ${CMAKE_CURRENT_BINARY_DIR}/mnist_c_module.o
)
add_dependencies(mnist_c_module mnist_gen)

SET_TARGET_PROPERTIES(
  mnist_c_module
  PROPERTIES
  LINKER_LANGUAGE C
)

iree_cc_binary(
  NAME
    mnist_static_c
  OUT
    mnist_static_c
  SRCS
    "create_c_module.c"
    "mnist_static.c"
    "mnist_emitc.h"
  DATA
    "iree::samples::vision_static::mnist_test.png"
  DEPS
    iree::runtime
    iree::hal::local::loaders::static_library_loader
    iree::hal::local::sync_driver
    iree::vm::shims_emitc
    iree::tools::utils::image_util
    mnist_c_module
)

iree_lit_test(
  NAME
    mnist_static_c_test
  TEST_FILE
    "mnist_static_c_test.txt"
  DATA
    ::mnist_static_c
    iree::tools::IreeFileCheck
  LABELS
    "hostonly"
)
