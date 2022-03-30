/* Copyright 2021 Google LLC

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

// An example for reading a HloModule from a HloProto file and execute the
// module on PJRT CPU client.
//
// To build a HloModule,
//
// $ python3 jax/tools/jax_to_hlo.py \
// --fn examples.jax_cpp.prog.fn \
// --input_shapes '[("x", "f32[2,2]"), ("y", "f32[2,2]")]' \
// --constants '{"z": 2.0}' \
// --hlo_text_dest /tmp/fn_hlo.txt \
// --hlo_proto_dest /tmp/fn_hlo.pb
//
// To load and run the HloModule,
//
// $ bazel build examples/jax_cpp:main --experimental_repo_remote_exec
// --check_visibility=false $ bazel-bin/examples/jax_cpp/main 2021-01-12
// 15:35:28.316880: I examples/jax_cpp/main.cc:65] result = ( f32[2,2] {
//   { 1.5, 1.5 },
//   { 3.5, 3.5 }
// }
// )

#include <memory>
#include <string>
#include <vector>

#include "altgraph/evaluation/evaluator.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

int main(int argc, char** argv) {
  std::string hlo_fn;
  int times = 1;
  bool gpu = false;
  bool rerun_hlo = false;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("hlo_fn", &hlo_fn, "HLO filename"),
      tensorflow::Flag("times", &times, "Times of running"),
      tensorflow::Flag("rerun_hlo", &rerun_hlo,
                       "Whether to rerun the hlo passes"),
      tensorflow::Flag("gpu", &gpu, "Use gpu instead of cpu")};
  const std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(parse_ok && argc == 1) << "\n" << usage;

  QCHECK(!hlo_fn.empty()) << "--hlo_fn is required";

  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };
  std::unique_ptr<xla::HloModule> test_module =
      LoadModuleFromFile(hlo_fn, xla::hlo_module_loader_details::Config(),
                         "txt", config_modifier_hook)
          .ValueOrDie();
  xla::Evaluator evaluator;
  xla::GpuAllocatorConfig gpu_config;
  auto client =
      gpu ? xla::GetGpuClient(true, gpu_config, nullptr, 0).ValueOrDie()
          : xla::GetCpuClient(true).ValueOrDie();
  evaluator.Compile(test_module->ToProto(), rerun_hlo, client.get());
  auto result = evaluator.Evaluate(times);
  LOG(INFO) << "run time:" << result.duration << std::endl;
  return 0;
}
