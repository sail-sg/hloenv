// Copyright 2022 Garena Online Private Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HLOENV_HLO_MODULE_H_
#define HLOENV_HLO_MODULE_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hloenv/utils/hlo_utils.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

namespace py = pybind11;

namespace hloenv {

class AltHloModule {
 public:
  explicit AltHloModule(std::unique_ptr<xla::HloModule> hlo_module) {
    hlo_module_ = std::move(hlo_module);
  }

  explicit AltHloModule(const std::string& input, const std::string& format="path") {
    std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
        [](xla::HloModuleConfig* config) { config->set_seed(42); };

    if (format == "path") {
      hlo_module_ = std::move(
          LoadModuleFromFile(input, xla::hlo_module_loader_details::Config(),
                             "txt", config_modifier_hook)
              .ValueOrDie());
    } else  if (format == "text") {
      hlo_module_ =
          std::move(LoadModuleFromData(input, format,
                                       xla::hlo_module_loader_details::Config(),
                                       config_modifier_hook)
                        .ValueOrDie());
    } else {
      LOG(FATAL) << "Unrecognized HloModule format, use 'path' or 'text'.";
    }
  }

  explicit AltHloModule(AltHloModule&& other) {
    hlo_module_ = std::move(other.hlo_module_);
  }
  AltHloModule& operator=(AltHloModule&& other) {
    if (this != &other) {
      hlo_module_ = std::move(other.hlo_module_);
    }
    return *this;
  }

  virtual ~AltHloModule() {}

  xla::HloModule* hlo_module_ptr() { return hlo_module_.get(); }

  std::shared_ptr<AltHloModule> Clone() {
    std::unique_ptr<xla::HloModule> saved_hlo_module = std::move(hlo_module_);
    hlo_module_ = std::move(saved_hlo_module->Clone());
    return std::make_shared<AltHloModule>(std::move(saved_hlo_module));
  }

  std::string ToString() { return hlo_module_->ToString(); }

  uint64_t Hash() { return xla::HloModuleHash(hlo_module_.get()); }

  std::shared_ptr<AltHloModule> ExtractRandomSubmodule(
      int instruction_count_threshold, int height) {
    auto returned_submodule = xla::ExtractRandomSubmodule(
        hlo_module_, instruction_count_threshold, height);
    return returned_submodule == nullptr
               ? nullptr
               : std::make_shared<AltHloModule>(std::move(returned_submodule));
  }

  // TODO(wanxy): Might need a better reprensentation for HloInstruction besides
  // serialized string
  std::vector<std::pair<std::string, std::shared_ptr<AltHloModule>>>
  ExtractInstructionsAsModule(int repeat = 1000) {
    std::vector<std::pair<std::string, std::shared_ptr<AltHloModule>>> ret;
    for (auto& ins : xla::ExtractInstructionsAsModule(*hlo_module_, repeat)) {
      ret.emplace_back(std::make_pair(
          ins.first->ToString(),
          std::make_shared<AltHloModule>(std::move(ins.second))));
    }
    return ret;
  }

  std::vector<std::shared_ptr<AltHloModule>> ExtractFusionsAsModule(
      int repeat = 1000) {
    std::vector<std::shared_ptr<AltHloModule>> ret;
    for (auto& module : xla::ExtractFusionsAsModule(*hlo_module_, repeat)) {
      ret.emplace_back(std::make_shared<AltHloModule>(std::move(module)));
    }
    return ret;
  }

  xla::HloModuleProto ToProto() { return hlo_module_->ToProto(); }

  bool IsBefEnabled() { return xla::gpu::IsBefEnabled(hlo_module_->config()); }

  const xla::HloModuleConfig& config() const { return hlo_module_->config(); }

  void SetHloProfiling(bool enabled) {
    xla::HloModuleConfig& hlo_module_config = hlo_module_->config();
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    options.set_xla_hlo_profile(enabled);
    hlo_module_config.set_debug_options(options);
  }

  xla::HloCostAnalysis::Properties CostAnalysis() {
    auto analysis =
        HloEnvGpuBackend::PjRtClient()->GetHloCostAnalysis().ValueOrDie();
    hlo_module_->entry_computation()->Accept(analysis.get());
    return analysis->properties();
  }

  int64_t InstructionCount() { return hlo_module_->instruction_count(); }

  int64_t ComputationCount() { return hlo_module_->computation_count(); }

 private:
  std::unique_ptr<xla::HloModule> hlo_module_;
};

}  // namespace hloenv

#endif  // HLOENV_HLO_MODULE_H_
