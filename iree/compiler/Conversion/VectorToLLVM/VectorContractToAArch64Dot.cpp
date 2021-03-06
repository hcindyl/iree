// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct ConvertVectorContract4x4x4_i8i8i32_ToAArch64InlineAsmPattern
    : public ConvertOpToLLVMPattern<vector::ContractionOp> {
 public:
  explicit ConvertVectorContract4x4x4_i8i8i32_ToAArch64InlineAsmPattern(
      LLVMTypeConverter &typeConv)
      : ConvertOpToLLVMPattern<vector::ContractionOp>(typeConv, 1) {}
  LogicalResult matchAndRewrite(
      vector::ContractionOp contractionOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = contractionOp.getLoc();

    auto asm_block = rewriter.getStringAttr(
        "sdot $0.4s, $4.16b, $5.4b[0] \n"
        "sdot $1.4s, $4.16b, $5.4b[1] \n"
        "sdot $2.4s, $4.16b, $5.4b[2] \n"
        "sdot $3.4s, $4.16b, $5.4b[3]");
    auto constrains = rewriter.getStringAttr("=w,=w,=w,=w,w,w,0,1,2,3");

    auto int32x4VType = VectorType::get({4}, rewriter.getIntegerType(32));

    auto dstVec0 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, contractionOp.acc(), rewriter.getI64ArrayAttr({0}));
    auto dstVec1 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, contractionOp.acc(), rewriter.getI64ArrayAttr({1}));
    auto dstVec2 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, contractionOp.acc(), rewriter.getI64ArrayAttr({2}));
    auto dstVec3 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, contractionOp.acc(), rewriter.getI64ArrayAttr({3}));

    auto returnType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {int32x4VType, int32x4VType, int32x4VType, int32x4VType});

    auto packedResult = rewriter.create<LLVM::InlineAsmOp>(
        loc, returnType,
        ArrayRef<Value>({dstVec0, dstVec1, dstVec2, dstVec3,
                         contractionOp.lhs(), contractionOp.rhs()}),
        asm_block, constrains, rewriter.getUnitAttr(), rewriter.getUnitAttr(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    auto resVec0 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, packedResult.res(), rewriter.getI64ArrayAttr({0}));
    auto resVec1 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, packedResult.res(), rewriter.getI64ArrayAttr({1}));
    auto resVec2 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, packedResult.res(), rewriter.getI64ArrayAttr({2}));
    auto resVec3 = rewriter.create<LLVM::ExtractValueOp>(
        loc, int32x4VType, packedResult.res(), rewriter.getI64ArrayAttr({3}));

    auto int32x4x4Type = LLVM::LLVMArrayType::get(int32x4VType, 4);
    auto result = rewriter.create<LLVM::UndefOp>(loc, int32x4x4Type);

    rewriter.create<LLVM::InsertValueOp>(loc, result, resVec0,
                                         rewriter.getI64ArrayAttr({0}));
    rewriter.create<LLVM::InsertValueOp>(loc, result, resVec1,
                                         rewriter.getI64ArrayAttr({1}));
    rewriter.create<LLVM::InsertValueOp>(loc, result, resVec2,
                                         rewriter.getI64ArrayAttr({2}));
    rewriter.create<LLVM::InsertValueOp>(loc, result, resVec3,
                                         rewriter.getI64ArrayAttr({3}));

    rewriter.replaceOp(contractionOp, {result});
    return success();
  }
};

}  // namespace

namespace {
struct VectorToAArch64InlineAsmPass
    : public PassWrapper<VectorToAArch64InlineAsmPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void populateMatmul_4x4x4_i8_i8_i32ToAArch64InlineAsm(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  patterns.insert<ConvertVectorContract4x4x4_i8i8i32_ToAArch64InlineAsmPattern>(
      converter);
}

void VectorToAArch64InlineAsmPass::runOnOperation() {
  LLVMTypeConverter converter(&getContext());

  LLVMConversionTarget target(getContext());

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalOp<LLVM::DialectCastOp>();

  target.addIllegalOp<ReturnOp>();

  OwningRewritePatternList patterns;

  populateMatmul_4x4x4_i8_i8_i32ToAArch64InlineAsm(converter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
  return;
}

std::unique_ptr<OperationPass<ModuleOp>>
createVectorToAArch64InlineAssemblyPass() {
  return std::make_unique<VectorToAArch64InlineAsmPass>();
}

static PassRegistration<VectorToAArch64InlineAsmPass> pass(
    "iree-codegen-vector-to-aarch64-asm",
    "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to "
    "LLVMIR dialect",
    [] { return std::make_unique<VectorToAArch64InlineAsmPass>(); });

}  // namespace iree_compiler
}  // namespace mlir