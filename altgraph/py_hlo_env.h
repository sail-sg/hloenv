// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PY_HLO_ENV_H_
#define ALTGRAPH_PY_HLO_ENV_H_

#include <pybind11/pybind11.h>

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
#include "altgraph/py_hlo_graph.h"
#include "altgraph/py_hlo_module.h"
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

namespace py = pybind11;

class PyHloEnv {
 public:
  struct EvaluationResult {
    std::vector<uint64_t> durations;
    std::vector<std::vector<py::object>> output;
  };

 private:
  std::shared_ptr<PyHloModule> py_hlo_module_;
  const std::string platform_;
  xla::Evaluator evaluator_;

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
  explicit PyHloEnv(std::shared_ptr<PyHloModule>, const std::string& platform);

  explicit PyHloEnv(const std::string& hlo_filepath, const std::string& format,
                    const std::string& platform);

  explicit PyHloEnv(const std::string& hlo_filepath,
                    const std::string& platform)
      : PyHloEnv(hlo_filepath, "path", platform) {}

  void Init(bool preallocate, double memory_fraction);

  std::shared_ptr<PyHloModule> SaveHloModule();

  void LoadHloModule(std::shared_ptr<PyHloModule> saved_hlo_module);

  void LoadHloModule(const std::string& hlo_input, const std::string& format);

  std::string ExportHloModuleToStr();

  EvaluationResult Evaluate(int times);

  bool HasEqualOutputAs(std::shared_ptr<PyHloModule> other_module,
                        int times = 1);

  bool HasEqualOutput(std::shared_ptr<PyHloModule> first_module,
                      std::shared_ptr<PyHloModule> second_module,
                      int times = 1);

  void PreFusionOptimizations();

  void PreFusionDryPasses();

  void FusionDryRun(bool may_duplicate = true);

  void PostFusionDryPasses();

  void GeneralFusionDryRun();

  void PostFusionOptimizations();

  // TODO(ohcy): Move to utility/PyHloModule
  uint64_t GetHloModuleHash();

  PyHloGraph GetHloGraph(bool do_hash_verification);

  std::shared_ptr<PyHloModule> GetHloModule();

  void ApplyAlternatives(py::array_t<size_t> decisions);

  void OriginalRunHloPasses();

 private:
  void PrepareHloModuleForIrEmitting();
};

#endif  // ALTGRAPH_PY_HLO_ENV_H_
