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

#include "hloenv/evaluation/evaluator.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"

#define NUM_RUNS_IGNORED 3

namespace hloenv {
namespace {
xla::Literal CreateRandomLiteral(
    const xla::Shape& shape,
    const std::function<double(absl::Span<const int64_t>)>& generator) {
  switch (shape.element_type()) {
    case xla::F32:
      return xla::LiteralUtil::CreateLiteralWithGenerator<xla::F32>(shape,
                                                                    generator)
          .ValueOrDie();
    case xla::F64:
      return xla::LiteralUtil::CreateLiteralWithGenerator<xla::F64>(shape,
                                                                    generator)
          .ValueOrDie();
    case xla::TUPLE: {
      std::vector<xla::Literal> tuple;
      for (int i = 0; i < shape.tuple_shapes_size(); i++) {
        tuple.push_back(CreateRandomLiteral(shape.tuple_shapes(i), generator));
      }
      return xla::LiteralUtil::MakeTupleOwned(std::move(tuple));
    }
    case xla::F16: {
      std::function<xla::half(absl::Span<const int64_t>)> wrap =
          [&generator](absl::Span<const int64_t> args) {
            return static_cast<xla::half>(generator(args));
          };
      return xla::LiteralUtil::CreateLiteralWithGenerator<xla::F16>(shape, wrap)
          .ValueOrDie();
    }
    case xla::BF16: {
      std::function<xla::bfloat16(absl::Span<const int64_t>)> wrap =
          [&generator](absl::Span<const int64_t> args) {
            return static_cast<xla::bfloat16>(generator(args));
          };
      return xla::LiteralUtil::CreateLiteralWithGenerator<xla::BF16>(shape,
                                                                     wrap)
          .ValueOrDie();
    }
    default:
      // Zero init
      return xla::Literal::CreateFromShape(shape);
  }
}
}  // namespace

void Evaluator::Compile(const xla::HloModuleProto& hlo_module_proto,
                        bool rerun_hlo, xla::PjRtClient* client) {
  xla::XlaComputation xla_computation(hlo_module_proto);
  xla::CompileOptions compile_options;
  if (!rerun_hlo) {
    compile_options.executable_build_options.set_run_backend_only(true);
  }
  auto executable_status = client->Compile(xla_computation, compile_options);
  executable_ = std::move(executable_status.ValueOrDie());
  this->GenerateParameters();
}

void Evaluator::GenerateParameters(int rand_seed) {
  GenerateParametersImpl(*executable_->GetHloModules().ValueOrDie().front(),
                         rand_seed, executable_->client(), &parameters_);
}

void Evaluator::GenerateParametersImpl(const xla::HloModule& hlo_module,
                                       int rand_seed, xla::PjRtClient* client,
                                       BufferPack* parameters) {
  parameters->clear();
  std::vector<xla::HloInstruction*> parameter_instructions =
      hlo_module.entry_computation()->parameter_instructions();
  // We keep the literals in this pool to keep them alive until all parameters's
  // BlockHostUntilReady
  std::vector<xla::Literal> literal_pool;
  parameters->emplace_back();
  std::minstd_rand0 engine;
  engine.seed(rand_seed);
  std::normal_distribution<double> distribution(0.0, 1.0);
  std::function<double(absl::Span<const int64_t>)> generator =
      [&distribution, &engine](absl::Span<const int64_t>) {
        return distribution(engine);
      };
  for (xla::HloInstruction* parameter : parameter_instructions) {
    xla::Shape shape = parameter->shape();
    literal_pool.emplace_back(CreateRandomLiteral(shape, generator));
    LOG(INFO) << "Assumed parameter(" << parameter->name()
              << "): " << literal_pool.back().shape().ToString();
    parameters->back().push_back(
        client
            ->BufferFromHostLiteral(literal_pool.back(),
                                    client->addressable_devices()[0])
            .ValueOrDie());
  }
  for (auto& parameter : parameters->back()) {
    parameter->BlockHostUntilReady();
  }
}

Evaluator::EvaluationResult Evaluator::Evaluate(int times) {
  if (!executable_) {
    LOG(FATAL) << "Please asssign a hlomodule to evaluator first";
  }
  EvaluationResult ret;
  xla::ExecuteOptions execute_options;
  std::vector<std::vector<xla::PjRtBuffer*>> parameters;
  for (auto& p : parameters_) {
    parameters.emplace_back();
    for (auto& pp : p) {
      parameters.back().push_back(pp.get());
    }
  }

  absl::Time start;
  BufferPack result;
  for (int i = 0; i < times + NUM_RUNS_IGNORED; i++) {
    start = absl::Now();
    // TODO(wanxy): Not sure whether this is async yet
    // Might need to make sure the execution is complete after function returns
    result = std::move(
        executable_->Execute(parameters, execute_options).ValueOrDie());

    for (auto& p : result) {
      for (auto& pp : p) {
        pp->BlockHostUntilReady();
      }
    }

    if (i >= NUM_RUNS_IGNORED) {
      ret.durations.push_back(executable_->async_exec_time_ns);
      ret.full_durations.push_back(absl::Now() - start);
      ret.compute_durations.push_back(executable_->compute_time_ns);
    }
  }

  for (auto& p : result) {
    ret.output.emplace_back();
    for (auto& pp : p) {
      ret.output.back().emplace_back(std::move(pp));
    }
  }
  ret.memory_consumed = 0;
  return ret;
}

}  // namespace hloenv
