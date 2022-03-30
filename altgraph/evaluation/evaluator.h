// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_EVALUATION_EVALUATOR_H_
#define ALTGRAPH_EVALUATION_EVALUATOR_H_

#include <memory>
#include <vector>

#include "absl/time/clock.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

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
    absl::Duration duration;

    BufferPack output;
    uint64_t memory_consumed;
  };
  EvaluationResult Evaluate(int times = 1);

 private:
  static void GenerateParametersImpl(const xla::HloModule& hlo_module,
                                     int rand_seed, PjRtClient* client,
                                     BufferPack* parameters);
  BufferPack parameters_;
  std::unique_ptr<xla::PjRtExecutable> executable_;
};

}  // namespace xla

#endif  // ALTGRAPH_EVALUATION_EVALUATOR_H_
