#include <stdio.h>

#include "iree/samples/vision_static/mnist_c.h"
#include "iree/vm/bytecode_module.h"

// A function to create the bytecode module.
iree_status_t create_module(iree_vm_module_t** module) {
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_vision_static_mnist_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);

  return iree_vm_bytecode_module_create(module_data, iree_allocator_null(),
                                        iree_allocator_system(), module);
}

void print_success() {
  fprintf(stdout, "mnist_static_bytecode passed\n");
}
