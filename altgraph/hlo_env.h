// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_HLO_ENV_H_
#define ALTGRAPH_HLO_ENV_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "altgraph/evaluation/evaluator.h"
#include "altgraph/gpu_backend.h"
#include "altgraph/hlo_graph.h"
#include "altgraph/hlo_module.h"
#include "altgraph/schedule.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace altgraph {

class HloEnv {
 public:
  struct EvaluationResult {
    std::vector<uint64_t> durations;
    std::vector<uint64_t> async_durations;
    std::vector<uint64_t> compute_durations;
    std::vector<std::vector<py::object>> output;
  };

 private:
  std::shared_ptr<AltHloModule> alt_hlo_module_;
  const std::string platform_;
  Evaluator evaluator_;

 public:
  // Currently, JAX automatically preallocates 90% of the currently-available
  // GPU memory when the first JAX op is run. Setting preallocate to false will
  // disable this, though it might result in more allocation overhead and
  // memory fragmentation. Conversely, setting it to true may cause out of
  // memory errors.
  //
  // memory_fraction can be used to control the percentage of
  // currently available GPU memory that is preallocated. However if preallocat
  // is set to false, this parameter will be ignored.
  explicit HloEnv(std::shared_ptr<AltHloModule>, const std::string& platform);

  explicit HloEnv(const std::string& hlo_filepath, const std::string& format,
                  const std::string& platform);

  explicit HloEnv(const std::string& hlo_filepath, const std::string& platform)
      : HloEnv(hlo_filepath, "path", platform) {}

  void Init(bool preallocate, double memory_fraction);

  std::shared_ptr<AltHloModule> CloneHloModule();

  void LoadHloModule(std::shared_ptr<AltHloModule> saved_hlo_module);

  void LoadHloModule(const std::string& hlo_input, const std::string& format);

  std::string ExportHloModuleToStr();

  void PrepareForEvaluation();

  EvaluationResult Evaluate(int times, bool do_not_prep_for_eval=false);

  bool HasEqualOutputAs(std::shared_ptr<AltHloModule> other_module,
                        int times = 1);

  bool HasEqualOutput(std::shared_ptr<AltHloModule> first_module,
                      std::shared_ptr<AltHloModule> second_module,
                      int times = 1);

  bool Run(std::shared_ptr<PassInterface> pass);

  // TODO(ohcy): Move to utility/AltHloModule
  uint64_t GetHloModuleHash();

  std::shared_ptr<AltHloModule> GetHloModule();

  void ApplyAlternatives(py::array_t<size_t> decisions);

  void OriginalOptimizeHloModule();
};

}  // namespace altgraph

#endif  // ALTGRAPH_HLO_ENV_H_
