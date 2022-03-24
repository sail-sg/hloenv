// Copyright 2021 Garena Online Private Limited

#ifndef TENSORFLOW_XLA_STANDALONE_PY_HLO_IR_H_
#define TENSORFLOW_XLA_STANDALONE_PY_HLO_IR_H_

#include <pybind11/pybind11.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "altgraph/py_hlo_graph.h"

namespace py = pybind11;

class PyHloIr {
 public:
  std::unique_ptr<xla::HloModule> hlo_module_;

 private:
  xla::Intercept<xla::cpu::CpuCompiler> cpu_intercept_;
  xla::Intercept<xla::gpu::GpuCompiler> gpu_intercept_;
  PyHloGraph py_hlo_graph_;
  const std::string platform_;
  std::unique_ptr<xla::PjRtClient> client_;

 public:
  explicit PyHloIr(const std::string& hlo_filepath,
                   const std::string& platform);

  void PreFusionOptimizations();

  void FusionDryRun();

  void PostFusionOptimizations();

  PyHloGraph& GetHloGraph();

  void ApplyAlternatives(py::array_t<size_t> decisions);
};

#endif  // TENSORFLOW_XLA_STANDALONE_PY_HLO_IR_H_
