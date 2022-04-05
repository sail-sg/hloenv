// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PY_HLO_IR_H_
#define ALTGRAPH_PY_HLO_IR_H_

#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "altgraph/evaluation/evaluator.h"
#include "altgraph/py_hlo_graph.h"
#include "tensorflow/compiler/xla/literal.h"
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

class PyHloIr {
 public:
  struct EvaluationResult {
    std::vector<uint64_t> durations;
    std::vector<std::vector<py::object>> output;
  };

 private:
  std::shared_ptr<xla::HloModule> hlo_module_;
  xla::Intercept<xla::cpu::CpuCompiler> cpu_intercept_;
  xla::Intercept<xla::gpu::GpuCompiler> gpu_intercept_;
  const std::string platform_;
  std::unique_ptr<xla::PjRtClient> client_;
  xla::Evaluator evaluator_;

 public:
  explicit PyHloIr(const std::string& hlo_filepath,
                   const std::string& platform);

  std::shared_ptr<xla::HloModule> SaveHloModule();

  void RestoreHloModule(std::shared_ptr<xla::HloModule> saved_hlo_module);

  std::string ExportHloModuleToStr();

  EvaluationResult Evaluate(int times);

  void PreFusionOptimizations();

  void FusionDryRun();

  void PostFusionOptimizations();

  PyHloGraph GetHloGraph();

  void ApplyAlternatives(py::array_t<size_t> decisions);
};

#endif  // ALTGRAPH_PY_HLO_IR_H_
