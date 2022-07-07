// Copyright 2021 Garena Online Private Limited

#include "altgraph/hlo_env.h"

namespace altgraph {

HloEnv::HloEnv(std::shared_ptr<AltHloModule> py_hlo_module,
               const std::string& platform)
    : platform_(platform) {
  alt_hlo_module_ = py_hlo_module;
}

HloEnv::HloEnv(const std::string& hlo_input, const std::string& format,
               const std::string& platform)
    : platform_(platform) {
  alt_hlo_module_ = std::make_shared<AltHloModule>(hlo_input, format);
}

bool HloEnv::HasEqualOutputAs(std::shared_ptr<AltHloModule> other_module,
                              int times) {
  return HasEqualOutput(alt_hlo_module_, other_module, times);
}

// Callback helper that prints the different literals when they are
// not equal
void OnMiscompare(const xla::LiteralSlice& expected,
                  const xla::LiteralSlice& actual,
                  const xla::LiteralSlice& mismatches,
                  const xla::ShapeIndex& /*shape_index*/) {
  LOG(INFO) << "expected: " << xla::ShapeUtil::HumanString(expected.shape())
            << " " << xla::literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << xla::ShapeUtil::HumanString(actual.shape())
            << " " << xla::literal_comparison::ToStringTruncated(actual);
}

bool HloEnv::HasEqualOutput(std::shared_ptr<AltHloModule> first_module,
                            std::shared_ptr<AltHloModule> second_module,
                            int times) {
  if (platform_ == "gpu") {
    for (int run = 0; run < times; run++) {
      evaluator_.Compile(first_module->hlo_module_ptr()->ToProto(),
                         /* rerun_hlo = */ false,
                         HloEnvGpuBackend::PjRtClient());
      auto first_ret = evaluator_.Evaluate();
      Evaluator::BufferPack& first_output = first_ret.output;

      evaluator_.Compile(second_module->hlo_module_ptr()->ToProto(),
                         /* rerun_hlo = */ false,
                         HloEnvGpuBackend::PjRtClient());
      auto second_ret = evaluator_.Evaluate();
      Evaluator::BufferPack& second_output = second_ret.output;

      if (first_output.size() != second_output.size()) {
        LOG(ERROR)
            << "Evaluation output length of compared HloModule is different!";
        return false;
      }

      for (int i = 0; i < first_output.size(); i++) {
        auto& first_buf_vector = first_output[i];
        auto& second_buf_vector = second_output[i];
        if (first_buf_vector.size() != second_buf_vector.size()) {
          LOG(ERROR) << "Evaluation output (internal vector) of compared "
                        "HloModule length is different!";
          return false;
        }

        for (int j = 0; j < first_buf_vector.size(); j++) {
          auto first_literal = std::make_shared<xla::Literal>(
              first_buf_vector[j]->on_device_shape());
          auto second_literal = std::make_shared<xla::Literal>(
              second_buf_vector[j]->on_device_shape());

          first_buf_vector[j]->ToLiteralSync(first_literal.get());
          second_buf_vector[j]->ToLiteralSync(second_literal.get());

          xla::ErrorSpec error_spec(static_cast<float>(1e-6),
                                    static_cast<float>(1e-6));

          xla::Status comparison_res = xla::literal_comparison::Near(
              /*expected=*/*first_literal,
              /*actual=*/*second_literal,
              /*error=*/error_spec,
              /*detailed_message=*/true, &OnMiscompare);

          return comparison_res.ok();
        }
      }
    }
    return true;
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

HloEnv::EvaluationResult HloEnv::Evaluate(int times) {
  HloEnv::EvaluationResult result;
  result.durations.reserve(times);

  if (platform_ == "gpu") {
    evaluator_.Compile(alt_hlo_module_->hlo_module_ptr()->ToProto(),
                       /* rerun_hlo = */ false, HloEnvGpuBackend::PjRtClient());
    auto ret = evaluator_.Evaluate(times);

    Evaluator::BufferPack& output = ret.output;

    for (auto& pjrt_buf_vector : output) {
      result.output.push_back(std::vector<py::object>());
      for (auto& pjrt_buf_ptr : pjrt_buf_vector) {
        std::shared_ptr<xla::Literal> literal =
            pjrt_buf_ptr->ToLiteralSync().ValueOrDie();
        result.output.back().push_back(
            std::move(xla::LiteralToPython(literal).ValueOrDie()));
      }
    }
    std::transform(ret.durations.begin(), ret.durations.end(),
                   std::back_inserter(result.durations),
                   [](absl::Duration duration) -> uint64_t {
                     return duration / absl::Nanoseconds(1);
                   });
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
  return result;
}

std::shared_ptr<AltHloModule> HloEnv::SaveHloModule() {
  return alt_hlo_module_->Clone();
}

void HloEnv::LoadHloModule(std::shared_ptr<AltHloModule> saved_hlo_module) {
  alt_hlo_module_ = saved_hlo_module;
}

void HloEnv::LoadHloModule(const std::string& hlo_input,
                           const std::string& format) {
  alt_hlo_module_ = std::make_shared<AltHloModule>(hlo_input, format);
}

std::string HloEnv::ExportHloModuleToStr() {
  return alt_hlo_module_->ToString();
}

void HloEnv::PreFusionOptimizations() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModulePreFusion(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::PreFusionDryPasses() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRunPre(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::FusionDryRun(bool may_duplicate) {
  if (platform_ == "gpu") {
    alt_hlo_module_->hlo_module_ptr()->SetDry(true);
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRun(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator(), may_duplicate);
    alt_hlo_module_->hlo_module_ptr()->SetDry(false);
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::PostFusionDryPasses() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRunPost(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::PostFusionOptimizations() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModulePostFusion(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
    // TODO(ohcy) To be refactored out of PostFusionOptimizations when we
    // can do multiple passes and have a pre/post passes phase
    this->PrepareHloModuleForIrEmitting();
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::PrepareHloModuleForIrEmitting() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->PrepareHloModuleForIrEmitting(
        alt_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void HloEnv::OriginalOptimizeHloModule() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModule(
        alt_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());

    HloEnvGpuBackend::GpuCompiler()->PrepareHloModuleForIrEmitting(
        alt_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

// Returns whether or not the pass is run to completion or has stopped in the
// middle (due to completing a dry_run pass, and needing apply alternatives)
bool HloEnv::Run(std::shared_ptr<PassInterface> pass) {
  if (platform_ == "gpu") {
    return pass->Run(alt_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

std::shared_ptr<AltHloModule> HloEnv::GetHloModule() { return alt_hlo_module_; }

// TODO(ohcy): Make it take a (uid_ptr, decision) arg instead, save time on
// rebuilding the HloGraph
void HloEnv::ApplyAlternatives(py::array_t<size_t> decisions) {
  if (platform_ == "gpu") {
    HloGraph hlo_graph = HloGraph(alt_hlo_module_->hlo_module_ptr(), false);
    const NodeFeats& node_feats = hlo_graph.get_node_feats();

    py::buffer_info decisions_buf = decisions.request();
    size_t* decisions_ptr = static_cast<size_t*>(decisions_buf.ptr);
    int num_decisions = decisions_buf.shape[0];

    // TODO(ohcy): Remove this
    // OCYTEMP -> sanity checks while debugging
    if (decisions_buf.shape[0] !=
        hlo_graph.get_alternative_indices_ptr()->size()) {
      LOG(FATAL) << "Decisions length != num alternatives length!";
    }
    if (decisions_buf.shape[1] != 2) {
      LOG(FATAL) << "Incorrect decisions shape!";
    }

    absl::flat_hash_map<int, xla::HloInstruction*>& uid_to_inst =
        hlo_graph.get_uid_to_inst();
    for (size_t decisions_idx = 0; decisions_idx < num_decisions;
         decisions_idx++) {
      size_t node_uid = node_feats.uids->at(decisions_ptr[decisions_idx * 2]);
      size_t decision = decisions_ptr[decisions_idx * 2 + 1];

      xla::HloInstruction* instruction = uid_to_inst.at(node_uid);

      // OCYTEMP -> sanity checks while debugging
      if (instruction->opcode() != xla::HloOpcode::kAlternatives) {
        LOG(FATAL) << "Applying alternatives to non-kAlternatives node -> "
                   << decisions_ptr[decisions_idx * 2] << " -> "
                   << instruction->name();
      }
      static_cast<xla::HloAlternatives*>(instruction)->Select(decision);
    }

    alt_hlo_module_->hlo_module_ptr()->Prune();
    // Remove unused computations created during fusion
    alt_hlo_module_->hlo_module_ptr()->RemoveUnusedComputations();
    alt_hlo_module_->hlo_module_ptr()->Cleanup();

  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

uint64_t HloEnv::GetHloModuleHash() { return alt_hlo_module_->Hash(); }

}  // namespace altgraph
