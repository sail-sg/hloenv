/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "hloenv/hlo_graph.h"

#include "gtest/gtest.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace {

Shape r0f32_ = ShapeUtil::MakeShape(F32, {});

std::unique_ptr<HloModule> CreateNewVerifiedModule() {
  HloModuleConfig config;
  return absl::make_unique<HloModule>("test_module", config);
}

// Create a computation which returns a constant.
std::unique_ptr<HloComputation> CreateConstantComputation() {
  auto builder = HloComputation::Builder("Constant");
  builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  return builder.Build();
}

// Creates a computation which calls the given zero-parameter computations.
std::unique_ptr<HloComputation> CreateCallComputation(
    absl::Span<HloComputation* const> computations) {
  auto builder = HloComputation::Builder("Call");
  for (auto computation : computations) {
    builder.AddInstruction(HloInstruction::CreateCall(r0f32_, {}, computation));
  }
  return builder.Build();
}

TEST(HloGraphTest, OneComputationPostOrder) {
  // Create a module with a single computation.
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(CreateConstantComputation());
  hloenv::HloGraph graph(module.get());

  // TODO(ohcy, wangyzh) Restore tests once hash is updated
  // EXPECT_EQ(graph.Hash(), module->CalledComputationHash());
}

TEST(HloGraphTest, TwoComputationsPostOrder) {
  // Create a module with two unconnected computations.
  auto module = CreateNewVerifiedModule();
  auto computation1 = module->AddEntryComputation(CreateConstantComputation());
  auto computation2 =
      module->AddEmbeddedComputation(CreateConstantComputation());
  hloenv::HloGraph graph(module.get());

  // TODO(ohcy, wangyzh) Restore tests once hash is updated
  // EXPECT_EQ(graph.Hash(), module->CalledComputationHash());
}

}  // namespace

}  // namespace xla
