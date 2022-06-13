// Copyright 2021 Garena Online Private Limited

#include "altgraph/py_hlo_env.h"

PyHloEnv::PyHloEnv(std::shared_ptr<PyHloModule> py_hlo_module,
                   const std::string& platform)
    : platform_(platform) {
  py_hlo_module_ = py_hlo_module;
}

PyHloEnv::PyHloEnv(const std::string& hlo_input, const std::string& format,
                   const std::string& platform)
    : platform_(platform) {
  py_hlo_module_ = std::make_shared<PyHloModule>(hlo_input, format);
}

bool PyHloEnv::HasEqualOutputAs(std::shared_ptr<PyHloModule> other_module,
                                int times) {
  return HasEqualOutput(py_hlo_module_, other_module, times);
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

bool PyHloEnv::HasEqualOutput(std::shared_ptr<PyHloModule> first_module,
                              std::shared_ptr<PyHloModule> second_module,
                              int times) {
  if (platform_ == "gpu") {
    for (int run = 0; run < times; run++) {
      evaluator_.Compile(first_module->hlo_module_ptr()->ToProto(),
                         /* rerun_hlo = */ false,
                         HloEnvGpuBackend::PjRtClient());
      auto first_ret = evaluator_.Evaluate();
      xla::Evaluator::BufferPack& first_output = first_ret.output;

      evaluator_.Compile(second_module->hlo_module_ptr()->ToProto(),
                         /* rerun_hlo = */ false,
                         HloEnvGpuBackend::PjRtClient());
      auto second_ret = evaluator_.Evaluate();
      xla::Evaluator::BufferPack& second_output = second_ret.output;

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

PyHloEnv::EvaluationResult PyHloEnv::Evaluate(int times) {
  PyHloEnv::EvaluationResult result;
  result.durations.reserve(times);

  if (platform_ == "gpu") {
    evaluator_.Compile(py_hlo_module_->hlo_module_ptr()->ToProto(),
                       /* rerun_hlo = */ false, HloEnvGpuBackend::PjRtClient());
    auto ret = evaluator_.Evaluate(times);

    xla::Evaluator::BufferPack& output = ret.output;

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

std::shared_ptr<PyHloModule> PyHloEnv::SaveHloModule() {
  return py_hlo_module_->Clone();
}

void PyHloEnv::LoadHloModule(std::shared_ptr<PyHloModule> saved_hlo_module) {
  py_hlo_module_ = saved_hlo_module;
}

void PyHloEnv::LoadHloModule(const std::string& hlo_input,
                             const std::string& format) {
  py_hlo_module_ = std::make_shared<PyHloModule>(hlo_input, format);
}

std::string PyHloEnv::ExportHloModuleToStr() {
  return py_hlo_module_->ToString();
}

void PyHloEnv::PreFusionOptimizations() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModulePreFusion(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::PreFusionDryPasses() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRunPre(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::FusionDryRun(bool may_duplicate) {
  if (platform_ == "gpu") {
    py_hlo_module_->hlo_module_ptr()->SetDry(true);
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRun(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator(), may_duplicate);
    py_hlo_module_->hlo_module_ptr()->SetDry(false);
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::PostFusionDryPasses() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleFusionRunPost(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::GeneralFusionDryRun() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModuleGeneralFusionRun(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::PostFusionOptimizations() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModulePostFusion(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());
    // TODO(ohcy) To be refactored out of PostFusionOptimizations when we
    // can do multiple passes and have a pre/post passes phase
    this->PrepareHloModuleForIrEmitting();
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::PrepareHloModuleForIrEmitting() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->PrepareHloModuleForIrEmitting(
        py_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::OriginalRunHloPasses() {
  if (platform_ == "gpu") {
    HloEnvGpuBackend::GpuCompiler()->OptimizeHloModule(
        py_hlo_module_->hlo_module_ptr(), HloEnvGpuBackend::StreamExecutor(),
        HloEnvGpuBackend::DeviceMemoryAllocator());

    HloEnvGpuBackend::GpuCompiler()->PrepareHloModuleForIrEmitting(
        py_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

PyHloGraph PyHloEnv::GetHloGraph(bool do_hash_verification) {
  return PyHloGraph(py_hlo_module_->hlo_module_ptr(), do_hash_verification);
}

std::shared_ptr<PyHloModule> PyHloEnv::GetHloModule() { return py_hlo_module_; }

// TODO(ohcy): Make it take a (uid_ptr, decision) arg instead, save time on
// rebuilding the HloGraph
void PyHloEnv::ApplyAlternatives(py::array_t<size_t> decisions) {
  if (platform_ == "gpu") {
    PyHloGraph py_hlo_graph =
        PyHloGraph(py_hlo_module_->hlo_module_ptr(), false);
    xla::NodeFeats& node_feats = py_hlo_graph.py_get_node_features();

    py::buffer_info decisions_buf = decisions.request();
    size_t* decisions_ptr = static_cast<size_t*>(decisions_buf.ptr);
    int num_decisions = decisions_buf.shape[0];

    // TODO(ohcy): Remove this
    // OCYTEMP -> sanity checks while debugging
    if (decisions_buf.shape[0] !=
        py_hlo_graph.get_alternative_indices_ptr()->size()) {
      LOG(FATAL) << "Decisions length != num alternatives length!";
    }
    if (decisions_buf.shape[1] != 2) {
      LOG(FATAL) << "Incorrect decisions shape!";
    }

    absl::flat_hash_map<int, xla::HloInstruction*>& uid_to_inst =
        py_hlo_graph.get_uid_to_inst();
    for (size_t decisions_idx = 0; decisions_idx < num_decisions;
         decisions_idx++) {
      size_t node_uid = node_feats.uids->at(decisions_ptr[decisions_idx * 2]);
      size_t decision = decisions_ptr[decisions_idx * 2 + 1];

      xla::HloInstruction* instruction = uid_to_inst.at(node_uid);

      // OCYTEMP -> sanity checks while debugging
      if (instruction->opcode() != xla::HloOpcode::kAlternatives) {
        LOG(FATAL) << "Trying to apply alternatives to non-kAlternatives node!";
      }
      static_cast<xla::HloAlternatives*>(instruction)->Select(decision);
    }

    for (xla::HloComputation* computation :
         py_hlo_module_->hlo_module_ptr()->MakeNonfusionComputations()) {
      // Remove the residue
      computation->Prune();
    }
    // Remove unused computations created during fusion
    py_hlo_module_->hlo_module_ptr()->RemoveUnusedComputations();

  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

uint64_t PyHloEnv::GetHloModuleHash() { return py_hlo_module_->Hash(); }

PYBIND11_MODULE(hlo_env, m) {
  // TODO(ohcy) Change PyHloGraph and PyHloEnv names to remove the Py prefix
  py::class_<PyHloGraph> py_hlo_graph(m, "PyHloGraph");

  py_hlo_graph.def(py::init<const xla::HloModule*>())
      .def("hash", &PyHloGraph::py_hash)
      .def("get_graph_load_errors", &PyHloGraph::py_get_graph_load_errors)
      .DEF_PYBIND_READONLY(PyHloGraph, out_edge_offsets)
      .DEF_PYBIND_READONLY(PyHloGraph, out_edge_indices)
      .DEF_PYBIND_READONLY(PyHloGraph, in_edge_offsets)
      .DEF_PYBIND_READONLY(PyHloGraph, in_edge_indices)
      .DEF_PYBIND_READONLY(PyHloGraph, alternative_indices)
      .DEF_PYBIND_READONLY(PyHloGraph, opcode_attr_counts)
      .DEF_PYBIND_READONLY(PyHloGraph, node_features)
      .DEF_PYBIND_READONLY(PyHloGraph, in_edge_features)
      .DEF_PYBIND_READONLY(PyHloGraph, out_edge_features);

  // TODO(ohcy): write this without copy as nparray
  py::class_<PyNodeFeats>(m, "NodeFeats")
      .DEF_PYBIND_READONLY(PyNodeFeats, uids)
      .DEF_PYBIND_READONLY(PyNodeFeats, names)
      .DEF_PYBIND_READONLY(PyNodeFeats, gids)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_users)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_operands)
      .DEF_PYBIND_READONLY(PyNodeFeats, opcodes)
      .DEF_PYBIND_READONLY(PyNodeFeats, opcode_attrs)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_opcode_attrs)
      .DEF_PYBIND_READONLY(PyNodeFeats, is_alternative)
      .DEF_PYBIND_READONLY(PyNodeFeats, in_tensor_sizes)
      .DEF_PYBIND_READONLY(PyNodeFeats, out_tensor_sizes)
      .DEF_PYBIND_READONLY(PyNodeFeats, has_max_in_tensor)
      .DEF_PYBIND_READONLY(PyNodeFeats, has_max_out_tensor);

  // TODO(ohcy): write this without copy as nparray
  py::class_<PyEdgeFeats>(m, "EdgeFeats")
      .def("get_tensor_size", &PyEdgeFeats::GetTensorSize)
      .DEF_PYBIND_READONLY(PyEdgeFeats, uids)
      .DEF_PYBIND_READONLY(PyEdgeFeats, srcs)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dsts)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dims)
      .DEF_PYBIND_READONLY(PyEdgeFeats, layouts)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dtypes);

  py::class_<PyHloEnv::EvaluationResult>(m, "EvaluationResult")
      .def_readonly("durations", &PyHloEnv::EvaluationResult::durations)
      .def_readonly("output", &PyHloEnv::EvaluationResult::output);

  py::class_<PyHloModule, std::shared_ptr<PyHloModule>>(m, "PyHloModule")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, const std::string&>())
      .def("to_string", &PyHloModule::ToString)
      .def("hash", &PyHloModule::Hash)
      .def("extract_random_submodule", &PyHloModule::ExtractRandomSubmodule)
      .def("extract_instructions_as_module",
           &PyHloModule::ExtractInstructionsAsModule)
      .def("clone", &PyHloModule::Clone);

  py::class_<PyHloEnv>(m, "PyHloEnv")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("hlo_filepath"), py::arg("platform"))
      .def(py::init<const std::string&, const std::string&,
                    const std::string&>(),
           py::arg("hlo_data"), py::arg("format"), py::arg("platform"))
      .def(py::init<std::shared_ptr<PyHloModule>, const std::string&>(),
           py::arg("py_hlo_module"), py::arg("platform"))
      .def("evaluate", &PyHloEnv::Evaluate)
      .def("has_equal_output", &PyHloEnv::HasEqualOutput,
           py::arg("first_module"), py::arg("second_module"),
           py::arg("times") = 1)
      .def("has_equal_output_as", &PyHloEnv::HasEqualOutputAs,
           py::arg("other_module"), py::arg("times") = 1)
      .def("save_hlo", &PyHloEnv::SaveHloModule)
      .def("load_hlo",
           static_cast<void (PyHloEnv::*)(std::shared_ptr<PyHloModule>)>(
               &PyHloEnv::LoadHloModule))
      .def("load_hlo",
           static_cast<void (PyHloEnv::*)(const std::string&,
                                          const std::string&)>(
               &PyHloEnv::LoadHloModule),
           py::arg("hlo_data"), py::arg("format") = "path")
      .def("export_hlo_to_str", &PyHloEnv::ExportHloModuleToStr)
      .def("get_hlo_module", &PyHloEnv::GetHloModule)
      .def("get_hlo_graph", &PyHloEnv::GetHloGraph,
           py::arg("do_hash_verification") = true)
      .def("pre_fusion_optimizations", &PyHloEnv::PreFusionOptimizations)
      .def("fusion_dry_run", &PyHloEnv::FusionDryRun,
           py::arg("may_duplicate") = true)
      .def("post_fusion_dry_passes", &PyHloEnv::PostFusionDryPasses)
      .def("pre_fusion_dry_passes", &PyHloEnv::PreFusionDryPasses)
      .def("general_fusion_dry_run", &PyHloEnv::GeneralFusionDryRun)
      .def("post_fusion_optimizations", &PyHloEnv::PostFusionOptimizations)
      .def("original_run_hlo_passes", &PyHloEnv::OriginalRunHloPasses)
      .def("get_hlo_module_hash", &PyHloEnv::GetHloModuleHash)
      .def("apply_alternatives", &PyHloEnv::ApplyAlternatives);

  py::class_<xla::Literal, std::shared_ptr<xla::Literal>>(m, "Literal")
      .def("__repr__", &xla::Literal::ToString);
}
