// Copyright 2021 Garena Online Private Limited

#include "altgraph/evaluation/evaluator.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

namespace xla {
namespace {

std::unique_ptr<HloModule> GetHloModule(const std::string& fn) {
  std::string hlo_filename = "tests/" + fn;
  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };
  std::unique_ptr<xla::HloModule> ret =
      LoadModuleFromFile(hlo_filename, xla::hlo_module_loader_details::Config(),
                         "txt", config_modifier_hook)
          .ValueOrDie();
  return ret;
}

TEST(EvaluatorTestBase, Basic) {
  std::unique_ptr<HloModule> hlo = GetHloModule("fn_hlo.txt");
  xla::Evaluator evaluator;
  auto client = xla::GetCpuClient(true).ValueOrDie();
  evaluator.Compile(hlo->ToProto(), /* rerun_hlo = */ false, client.get());

  std::vector<int> seeds = {1, 10086, 1, 2, 10086};
  std::unordered_map<int, std::shared_ptr<Literal>> seed_to_results;
  for (int seed : seeds) {
    evaluator.GenerateParameters(seed);
    auto result = evaluator.Evaluate();
    for (absl::Duration duration : result.durations) {
      EXPECT_GT(duration, absl::Nanoseconds(1));
    }
    auto literal = result.output[0][0]->ToLiteral().ValueOrDie();
    if (seed_to_results.count(seed)) {
      EXPECT_EQ(*literal, *seed_to_results[seed]);
    } else {
      seed_to_results[seed] = literal;
    }
  }
  EXPECT_NE(*seed_to_results[1], *seed_to_results[10086]);
  EXPECT_NE(*seed_to_results[2], *seed_to_results[10086]);
}

}  // namespace
}  // namespace xla
