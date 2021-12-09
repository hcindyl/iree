// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/samples/vision_static/mnist_emitc.h"

// A function to create the C module.
iree_status_t create_module(iree_vm_module_t** module) {
  return module_create(iree_allocator_system(), module);
}

void print_success() { fprintf(stdout, "mnist_static_c passed\n"); }
