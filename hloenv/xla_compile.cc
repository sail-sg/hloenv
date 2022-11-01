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

// An example for reading a HloModule from a HloProto file and execute the
// module on PJRT CPU client.

#include <gflags/gflags.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hloenv/hlo_graph.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

DEFINE_string(hlo, "-", "hlo text file");  // by default read from stdin
DEFINE_string(dry, "", "which pass to dry run");
DEFINE_string(platform, "gpu", "gpu or cpu, defaults to gpu");

int main(int argc, char** argv) {
  tensorflow::port::InitMain("", &argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  setenv("DRY", FLAGS_dry.c_str(), 1);
  xla::HloPassPipeline::dry_sandwich_set =
      xla::HloPassPipeline::ExtractDrySandwichSetFromEnv();

  xla::Intercept<xla::cpu::CpuCompiler> cpu_intercept;
  xla::Intercept<xla::gpu::GpuCompiler> gpu_intercept;

  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };

  // Load HloModule from file.
  std::unique_ptr<xla::HloModule> hlo_module;
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
  const xla::HloModuleProto hlo_module_proto = hlo_module->ToProto();

  // Run it using JAX C++ Runtime (PJRT).
  // Get a CPU client.

  std::unique_ptr<xla::PjRtClient> client;
  if (FLAGS_platform == "gpu") {
    client = xla::GetGpuClient(/*asynchronous=*/true, xla::GpuAllocatorConfig(),
                               nullptr, 0)
                 .ValueOrDie();
  } else if (FLAGS_platform == "cpu") {
    client = xla::GetCpuClient(/*asynchronous=*/true).ValueOrDie();
  } else {
    LOG(FATAL) << "Unknown platform " << FLAGS_platform;
  }

  // Compile XlaComputation to PjRtExecutable.
  xla::XlaComputation xla_computation(hlo_module_proto);
  xla::CompileOptions compile_options;

  try {
    std::unique_ptr<xla::PjRtExecutable> executable =
        client->Compile(xla_computation, compile_options).ValueOrDie();
  } catch (xla::Intercept<xla::cpu::CpuCompiler>& e) {
    cpu_intercept = std::move(e);
  } catch (xla::Intercept<xla::gpu::GpuCompiler>& e) {
    gpu_intercept = std::move(e);
    hloenv::HloGraph graph(gpu_intercept.module.get());
    graph.ShowStats();
    gpu_intercept.compiler->RunHloPasses(gpu_intercept.module.get(),
                                         gpu_intercept.stream_exec,
                                         gpu_intercept.options);
  }

  // intercept.compiler->RunHloPasses(intercept.module.get(),
  //                                intercept.stream_exec, intercept.options);

  /// There's a very long chain here
  /// pjrtclient -> local_client -> local_service -> service -> BuildExecutable
  /// -> backend->compiler->RunHloPasses
}
