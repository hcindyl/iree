// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This sample uses iree/tools/utils/image_util to load a hand-written image
// as an iree_hal_buffer_view_t then passes it to the bytecode module built
// from mnist.mlir on the dylib-llvm-aot backend.

#include <float.h>

#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/hal/local/sync_device.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/samples/vision_static/mnist_c_module.h"
#include "iree/tools/utils/image_util.h"

extern iree_status_t create_module(iree_vm_module_t** module);
extern void print_success();

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
iree_status_t create_device_with_static_loader(iree_allocator_t host_allocator,
                                               iree_hal_device_t** out_device) {
  iree_status_t status = iree_ok_status();

  // Set paramters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  // Load the statically embedded library
  const iree_hal_executable_library_header_t** static_library =
      mnist_linked_llvm_library_query(
          IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION, /*reserved=*/NULL);
  const iree_hal_executable_library_header_t** libraries[1] = {static_library};

  iree_hal_executable_loader_t* library_loader = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_static_library_loader_create(
        IREE_ARRAYSIZE(libraries), libraries,
        iree_hal_executable_import_provider_null(), host_allocator,
        &library_loader);
  }

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("sync");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  // Create the device and release the executor and loader afterwards.
  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create(
        identifier, &params, /*loader_count=*/1, &library_loader,
        device_allocator, host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(library_loader);
  return status;
}
iree_status_t Run(const iree_string_view_t image_path) {
  iree_status_t status = iree_ok_status();
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(&instance_options,
                                          iree_allocator_system(), &instance);
  }

  // Create dylib device with static loader.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = create_device_with_static_loader(iree_allocator_system(), &device);
  }

  // Create one session per loaded module to hold the module state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  // Load bytecode module from the embedded data. Append to the session.
  iree_vm_module_t* module = NULL;

  if (iree_status_is_ok(status)) {
    status = create_module(&module);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(session, module);
  }

  iree_runtime_call_t call;
  memset(&call, 0, sizeof(call));
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view("module.predict"), &call);
  }

  // Prepare the input hal buffer view with image_util library.
  // The input of the mmist model is single 28x28 pixel image as a
  // tensor<1x28x28x1xf32>, with pixels in [0.0, 1.0].
  iree_hal_buffer_view_t* buffer_view = NULL;
  iree_hal_dim_t buffer_shape[] = {1, 28, 28, 1};
  iree_hal_element_type_t hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  float input_range[2] = {0.0f, 1.0f};
  if (iree_status_is_ok(status)) {
    status = iree_tools_utils_buffer_view_from_image_rescaled(
        image_path, buffer_shape, IREE_ARRAYSIZE(buffer_shape),
        hal_element_type, iree_hal_device_allocator(device), input_range,
        IREE_ARRAYSIZE(input_range), &buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, buffer_view);
  }
  iree_hal_buffer_view_release(buffer_view);

  // Invoke call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  // Retreive output buffer view with results from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call,
                                                             &ret_buffer_view);
  }

  // Read back the results. The output of the mnist model is a 1x10 prediction
  // confidence values for each digit in [0, 9].
  iree_hal_buffer_mapping_t mapped_memory;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(ret_buffer_view),
        IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &mapped_memory);
  }
  float result_val = FLT_MIN;
  int result_idx = 0;
  if (iree_status_is_ok(status)) {
    const float* data_ptr = (const float*)mapped_memory.contents.data;
    for (int i = 0; i < mapped_memory.contents.data_length / sizeof(float);
         ++i) {
      if (data_ptr[i] > result_val) {
        result_val = data_ptr[i];
        result_idx = i;
      }
    }
    iree_hal_buffer_unmap_range(&mapped_memory);
  }
  // Get the highest index from the output.
  fprintf(stdout, "Detected number: %d\n", result_idx);
  iree_hal_buffer_view_release(ret_buffer_view);

  iree_runtime_call_deinitialize(&call);
  iree_hal_device_release(device);
  iree_vm_module_release(module);
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return status;
}

int main(int argc, char** argv) {
  if (argc > 2) {
    fprintf(stderr, "Usage: iree-run-mnist-module <image file>\n");
    return -1;
  }
  iree_string_view_t image_path;
  if (argc == 1) {
    image_path = iree_make_cstring_view("mnist_test.png");
  } else {
    image_path = iree_make_cstring_view(argv[1]);
  }
  iree_status_t result = Run(image_path);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }
  print_success();
  return 0;
}
