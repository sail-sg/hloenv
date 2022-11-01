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

#ifndef ALTGRAPH_EVALUATION_EVALUATOR_H_
#define ALTGRAPH_EVALUATION_EVALUATOR_H_

#include <memory>
#include <vector>

#include "absl/time/clock.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace altgraph {

class Evaluator {
 public:
  Evaluator() = default;
  virtual ~Evaluator() = default;

  // Sets and compiles the target HloModule used for evaluation
  void Compile(const xla::HloModuleProto& hlo_module_proto, bool rerun_hlo,
               xla::PjRtClient* client);

  void GenerateParameters(int rand_seed = 53);

  typedef std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> BufferPack;
  struct EvaluationResult {
    std::vector<uint64_t> durations;
    std::vector<absl::Duration> full_durations;
    std::vector<uint64_t> compute_durations;

    BufferPack output;
    uint64_t memory_consumed;
  };
  EvaluationResult Evaluate(int times = 1);

 private:
  static void GenerateParametersImpl(const xla::HloModule& hlo_module,
                                     int rand_seed, xla::PjRtClient* client,
                                     BufferPack* parameters);
  BufferPack parameters_;
  std::unique_ptr<xla::PjRtExecutable> executable_;
};

}  // namespace altgraph

#endif  // ALTGRAPH_EVALUATION_EVALUATOR_H_
