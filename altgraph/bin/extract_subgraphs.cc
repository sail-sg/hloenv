// Copyright 2021 Garena Online Private Limited

#include <gflags/gflags.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "altgraph/utils/hlo_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

std::vector<std::unique_ptr<HloModule>> ExtractSubgraphs(
    const std::unique_ptr<HloModule>& module, int target_inst_count) {
  std::vector<std::unique_ptr<HloModule>> ret;
  if (module->instruction_count() < target_inst_count) {
    return ret;
  }
  // Select a random instruction in a random computation.
  auto comps = module->MakeComputationPostOrder();
  // Pick computation only when its instruction count is large enough.
  auto filtered_comps = FilterComputations(comps, [&](HloComputation* c) {
    return c->instruction_count() > target_inst_count;
  });
  for (auto comp : filtered_comps) {
    LOG(INFO) << "[Computation]: " << comp->name();
    auto instructions = comp->MakeInstructionPostOrder();
    for (auto inst : instructions) {
      int hmin = 1;
      int hmax = target_inst_count;
      int h = 1;
      while (true) {
        auto submodule = ExtractModule(inst, h);
        int new_inst_count = submodule->instruction_count();
        // update max & min
        if (new_inst_count >= target_inst_count) {
          hmax = h;
        } else if (new_inst_count < target_inst_count) {
          hmin = h;
        }
        // iterate to next
        h = (hmin + hmax) / 2;
        if (hmax - hmin <= 1) {
          auto submodule = ExtractModule(inst, hmax);
          if (submodule->instruction_count() > target_inst_count &&
              FindInstruction(submodule.get(), HloOpcode::kCall) == nullptr) {
            LOG(INFO) << " [Root]: " << inst->name()
                      << " [Inst Count]: " << submodule->instruction_count();
            ret.push_back(std::move(submodule));
          }
          break;
        }
      }
    }
  }
  return ret;
}

}  // namespace xla

template <typename T>
std::string IntToHex(T i) {
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << i;
  return stream.str();
}

DEFINE_string(hlo, "-", "hlo text file");  // by default read from stdin
DEFINE_int32(num_inst, 20, "instruction number");

int main(int argc, char** argv) {
  tensorflow::port::InitMain("", &argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Load HloModule from file.
  std::unique_ptr<xla::HloModule> hlo_module;
  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };

  if (FLAGS_hlo == "-") {
    std::stringstream ss;
    std::string s;
    while (std::getline(std::cin, s)) {
      ss << s << "\n";
    }
    hlo_module = LoadModuleFromData(ss.str(), "txt",
                                    xla::hlo_module_loader_details::Config(),
                                    config_modifier_hook)
                     .ValueOrDie();
  } else {
    hlo_module =
        LoadModuleFromFile(FLAGS_hlo, xla::hlo_module_loader_details::Config(),
                           "txt", config_modifier_hook)
            .ValueOrDie();
  }
  auto vec = xla::ExtractSubgraphs(hlo_module, FLAGS_num_inst);
  for (auto& m : vec) {
    auto name = IntToHex(absl::HashOf(*m)) + ".txt";
    std::ofstream file;
    file.open(name);
    file << m->ToString();
    file.close();
  }
}
