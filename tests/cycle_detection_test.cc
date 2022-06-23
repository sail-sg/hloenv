
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "gtest/gtest.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace xla {
namespace {

std::unique_ptr<HloModule> CreateNewVerifiedModule() {
  HloModuleConfig config;
  return absl::make_unique<HloModule>("test_module", config);
}

TEST(CycleDetectionTestBase, Basic) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder("CycleDetection");
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd, constant1, constant1));
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add1, add1));
  auto add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add2, add2));
  auto add4 = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kAdd, add3, add3));
  // Create cycle
  add1->ReplaceOperandWith(0, add3);

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/add4));

  std::cout << computation->ToString() << std::endl;

  EXPECT_TRUE(computation->HasCycle());
  EXPECT_TRUE(computation->HasCycle(add1));
  EXPECT_TRUE(computation->HasCycle(add2));
  EXPECT_TRUE(computation->HasCycle(add3));
}

}
} // namespace xla
