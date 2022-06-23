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
    this->DedupTupleInstructions();
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

void PyHloEnv::OriginalOptimizeHloModule() {
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

// Returns whether or not the pass is run to completion or has stopped in the
// middle (due to completing a dry_run pass, and needing apply alternatives)
bool PyHloEnv::Run(std::shared_ptr<PassInterface> pass) {
  if (platform_ == "gpu") {
    return pass->Run(py_hlo_module_->hlo_module_ptr());
  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

PyHloGraph PyHloEnv::GetHloGraph(bool inline_fused_comp,
                                 bool do_hash_verification) {
  return PyHloGraph(py_hlo_module_->hlo_module_ptr(), inline_fused_comp,
                    do_hash_verification);
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

    py_hlo_module_->hlo_module_ptr()->Prune();
    // Remove unused computations created during fusion
    py_hlo_module_->hlo_module_ptr()->RemoveUnusedComputations();
    py_hlo_module_->hlo_module_ptr()->Cleanup();

  } else if (platform_ == "cpu") {
    LOG(FATAL) << "HloEnv currently not enabled for platform == cpu";
  }
}

void PyHloEnv::DedupTupleInstructions() {
  if (platform_ == "gpu") {
    py_hlo_module_->hlo_module_ptr()->DedupTupleInstructions();
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
      .DEF_PYBIND_READONLY(PyEdgeFeats, lehmercodes)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dtypes);

  py::class_<PyHloEnv::EvaluationResult>(m, "EvaluationResult")
      .def_readonly("durations", &PyHloEnv::EvaluationResult::durations)
      .def_readonly("output", &PyHloEnv::EvaluationResult::output);

  py::class_<xla::DebugOptions>(m, "DebugOptions")
      .def_property_readonly(
          "xla_llvm_enable_alias_scope_metadata",
          &xla::DebugOptions::xla_llvm_enable_alias_scope_metadata)
      .def_property_readonly(
          "xla_llvm_enable_noalias_metadata",
          &xla::DebugOptions::xla_llvm_enable_noalias_metadata)
      .def_property_readonly(
          "xla_llvm_enable_invariant_load_metadata",
          &xla::DebugOptions::xla_llvm_enable_invariant_load_metadata)
      .def_property_readonly(
          "xla_llvm_disable_expensive_passes",
          &xla::DebugOptions::xla_llvm_disable_expensive_passes)
      .def_property_readonly("xla_backend_optimization_level",
                             &xla::DebugOptions::xla_backend_optimization_level)
      .def_property_readonly("xla_gpu_deterministic_ops",
                             &xla::DebugOptions::xla_gpu_deterministic_ops)
      .def_property_readonly("xla_gpu_autotune_level",
                             &xla::DebugOptions::xla_gpu_autotune_level)
      .def_property_readonly("xla_cpu_multi_thread_eigen",
                             &xla::DebugOptions::xla_cpu_multi_thread_eigen)
      .def_property_readonly("xla_gpu_cuda_data_dir",
                             &xla::DebugOptions::xla_gpu_cuda_data_dir)
      .def_property_readonly("xla_gpu_asm_extra_flags",
                             &xla::DebugOptions::xla_gpu_asm_extra_flags)
      .def_property_readonly(
          "xla_eliminate_hlo_implicit_broadcast",
          &xla::DebugOptions::xla_eliminate_hlo_implicit_broadcast)
      .def_property_readonly("xla_dump_hlo_as_html",
                             &xla::DebugOptions::xla_dump_hlo_as_html)
      .def_property_readonly("xla_dump_fusion_visualization",
                             &xla::DebugOptions::xla_dump_fusion_visualization)
      .def_property_readonly("xla_dump_include_timestamp",
                             &xla::DebugOptions::xla_dump_include_timestamp)
      .def_property_readonly("xla_dump_max_hlo_modules",
                             &xla::DebugOptions::xla_dump_max_hlo_modules)
      .def_property_readonly("xla_dump_module_metadata",
                             &xla::DebugOptions::xla_dump_module_metadata)
      .def_property_readonly("xla_dump_hlo_as_long_text",
                             &xla::DebugOptions::xla_dump_hlo_as_long_text)
      .def_property_readonly("xla_cpu_use_mkl_dnn",
                             &xla::DebugOptions::xla_cpu_use_mkl_dnn)
      .def_property_readonly(
          "xla_gpu_max_kernel_unroll_factor",
          &xla::DebugOptions::xla_gpu_max_kernel_unroll_factor)
      .def_property_readonly(
          "xla_gpu_disable_multi_streaming",
          &xla::DebugOptions::xla_gpu_disable_multi_streaming)
      .def_property_readonly("xla_cpu_enable_fast_math",
                             &xla::DebugOptions::xla_cpu_enable_fast_math)
      .def_property_readonly("xla_cpu_fast_math_honor_nans",
                             &xla::DebugOptions::xla_cpu_fast_math_honor_nans)
      .def_property_readonly("xla_cpu_fast_math_honor_infs",
                             &xla::DebugOptions::xla_cpu_fast_math_honor_infs)
      .def_property_readonly(
          "xla_cpu_fast_math_honor_functions",
          &xla::DebugOptions::xla_cpu_fast_math_honor_functions)
      .def_property_readonly(
          "xla_cpu_fast_math_honor_division",
          &xla::DebugOptions::xla_cpu_fast_math_honor_division)
      .def_property_readonly("xla_cpu_enable_fast_min_max",
                             &xla::DebugOptions::xla_cpu_enable_fast_min_max)
      .def_property_readonly("xla_gpu_enable_cudnn_frontend",
                             &xla::DebugOptions::xla_gpu_enable_cudnn_frontend)
      .def_property_readonly("xla_gpu_enable_fast_min_max",
                             &xla::DebugOptions::xla_gpu_enable_fast_min_max)
      .def_property_readonly(
          "xla_gpu_strict_conv_algorithm_picker",
          &xla::DebugOptions::xla_gpu_strict_conv_algorithm_picker)
      .def_property_readonly("xla_allow_excess_precision",
                             &xla::DebugOptions::xla_allow_excess_precision)
      .def_property_readonly(
          "xla_force_host_platform_device_count",
          &xla::DebugOptions::xla_force_host_platform_device_count)
      .def_property_readonly(
          "xla_gpu_all_reduce_combine_threshold_bytes",
          &xla::DebugOptions::xla_gpu_all_reduce_combine_threshold_bytes)
      .def_property_readonly("xla_gpu_all_reduce_contiguous",
                             &xla::DebugOptions::xla_gpu_all_reduce_contiguous)
      .def_property_readonly(
          "xla_gpu_all_reduce_blueconnect_num_devices_per_host",
          &xla::DebugOptions::
              xla_gpu_all_reduce_blueconnect_num_devices_per_host)
      .def_property_readonly(
          "xla_gpu_enable_async_all_reduce",
          &xla::DebugOptions::xla_gpu_enable_async_all_reduce)
      .def_property_readonly("xla_cpu_enable_xprof_traceme",
                             &xla::DebugOptions::xla_cpu_enable_xprof_traceme)
      .def_property_readonly(
          "xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
          &xla::DebugOptions::
              xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found)
      .def_property_readonly(
          "xla_multiheap_size_constraint_per_heap",
          &xla::DebugOptions::xla_multiheap_size_constraint_per_heap)
      .def_property_readonly(
          "xla_detailed_logging_and_dumping",
          &xla::DebugOptions::xla_detailed_logging_and_dumping)
      .def_property_readonly("xla_gpu_bef_executable",
                             &xla::DebugOptions::xla_gpu_bef_executable)
      .def_property_readonly("xla_gpu_bef_thunk",
                             &xla::DebugOptions::xla_gpu_bef_thunk)
      .def_property_readonly(
          "xla_gpu_nccl_termination_timeout_seconds",
          &xla::DebugOptions::xla_gpu_nccl_termination_timeout_seconds)
      .def_property_readonly(
          "xla_gpu_enable_shared_constants",
          &xla::DebugOptions::xla_gpu_enable_shared_constants)
      .def_property_readonly(
          "xla_gpu_redzone_scratch_max_megabytes",
          &xla::DebugOptions::xla_gpu_redzone_scratch_max_megabytes);

  py::class_<xla::HloModuleConfig>(m, "HloModuleConfig")
      .def(py::init<>())
      .def_property_readonly("debug_options",
                             &xla::HloModuleConfig::debug_options)
      .def_property_readonly("seed", &xla::HloModuleConfig::seed)
      .def_property_readonly("launch_id", &xla::HloModuleConfig::launch_id)
      .def_property_readonly("replica_count",
                             &xla::HloModuleConfig::replica_count)
      .def_property_readonly("num_partitions",
                             &xla::HloModuleConfig::num_partitions)
      .def_property_readonly("use_spmd_partitioning",
                             &xla::HloModuleConfig::use_spmd_partitioning)
      .def_property_readonly("use_auto_spmd_partitioning",
                             &xla::HloModuleConfig::use_auto_spmd_partitioning)
      .def_property_readonly("deduplicate_hlo",
                             &xla::HloModuleConfig::deduplicate_hlo)
      .def_property_readonly(
          "intra_op_parallelism_threads",
          &xla::HloModuleConfig::intra_op_parallelism_threads)
      .def_property_readonly(
          "has_static_device_assignment",
          &xla::HloModuleConfig::has_static_device_assignment)
      .def_property_readonly("alias_passthrough_params",
                             &xla::HloModuleConfig::alias_passthrough_params)
      .def_property_readonly(
          "content_aware_computation_sorting",
          &xla::HloModuleConfig::content_aware_computation_sorting)
      .def_property_readonly("phase_index", &xla::HloModuleConfig::phase_index)
      .def_property_readonly(
          "allow_spmd_sharding_propagation_to_output",
          &xla::HloModuleConfig::allow_spmd_sharding_propagation_to_output);

  py::class_<PyHloModule, std::shared_ptr<PyHloModule>>(m, "PyHloModule")
      .def(py::init<const std::string&>())
      .def(py::init<const std::string&, const std::string&>())
      .def("to_string", &PyHloModule::ToString)
      .def_property_readonly("config", &PyHloModule::config)
      .def("hash", &PyHloModule::Hash)
      .def("extract_random_submodule", &PyHloModule::ExtractRandomSubmodule)
      .def("extract_instructions_as_module",
           &PyHloModule::ExtractInstructionsAsModule)
      .def("is_bef_enabled", &PyHloModule::IsBefEnabled)
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
           py::arg("inline_fused_comp") = false,
           py::arg("do_hash_verification") = true)
      .def("pre_fusion_optimizations", &PyHloEnv::PreFusionOptimizations)
      .def("fusion_dry_run", &PyHloEnv::FusionDryRun,
           py::arg("may_duplicate") = true)
      .def("post_fusion_dry_passes", &PyHloEnv::PostFusionDryPasses)
      .def("pre_fusion_dry_passes", &PyHloEnv::PreFusionDryPasses)
      .def("general_fusion_dry_run", &PyHloEnv::GeneralFusionDryRun)
      .def("post_fusion_optimizations", &PyHloEnv::PostFusionOptimizations)
      .def("optimize_hlo_module", &PyHloEnv::OriginalOptimizeHloModule)
      .def("prepare_hlo_module_for_ir_emitting",
           &PyHloEnv::PrepareHloModuleForIrEmitting)
      .def("run", &PyHloEnv::Run)
      .def("get_hlo_module_hash", &PyHloEnv::GetHloModuleHash)
      .def("get_hlo_module_hash", &PyHloEnv::GetHloModuleHash)
      .def("dedup_tuples", &PyHloEnv::DedupTupleInstructions)
      .def("apply_alternatives", &PyHloEnv::ApplyAlternatives);

  py::class_<HloEnvGpuBackend, std::unique_ptr<HloEnvGpuBackend, py::nodelete>>
      gpu_backend(m, "GpuBackend");
  gpu_backend
      .def(py::init([]() {
        return std::unique_ptr<HloEnvGpuBackend, py::nodelete>(
            &HloEnvGpuBackend::Instance());
      }))
      .def_property_readonly_static(
          "stream_exec_platform",
          [](py::object /* self */) {
            return HloEnvGpuBackend::GetStreamExecPlatform();
          })
      .def_static("cuda_is_at_least", &HloEnvGpuBackend::CudaComputeIsAtLeast,
                  py::arg("other_major"), py::arg("other_minor") = 1);
  py::enum_<xla::se::CudaComputeCapability::CudaComputeCapabilities>(
      gpu_backend, "CudaComputeCapability")
      .value("PASCAL",
             xla::se::CudaComputeCapability::CudaComputeCapabilities::PASCAL_)
      .value("VOLTA",
             xla::se::CudaComputeCapability::CudaComputeCapabilities::VOLTA)
      .value("AMPERE",
             xla::se::CudaComputeCapability::CudaComputeCapabilities::AMPERE)
      .export_values();

  py::class_<xla::Literal, std::shared_ptr<xla::Literal>>(m, "Literal")
      .def("__repr__", &xla::Literal::ToString);

  // Bindings for Hlo Passes
  py::module hlo_passes_m = m.def_submodule("hlo_pass", "Hlo Pass definitions");
  py_init_hlo_passes(hlo_passes_m);

  // General pipeline interface

  py::class_<PassInterface, std::shared_ptr<PassInterface>>(m, "PassInterface")
      .def("Run",
           static_cast<bool (PassInterface::*)(std::shared_ptr<PyHloModule>)>(
               &PassInterface::Run),
           py::arg("hlo_pass"))
      .def_property_readonly("name", &PassInterface::name);

  py::class_<Pass, PassInterface, std::shared_ptr<Pass>>(m, "Pass")
      .def(py::init<std::shared_ptr<xla::HloPassInterface>, int>(),
           py::arg("hlo_pass"), py::arg("loop_count") = 1)
      .def_property_readonly("changed", &Pass::changed)
      .def_property_readonly("name", &Pass::name);

  py::class_<Pipeline, PassInterface, std::shared_ptr<Pipeline>>(m, "Pipeline")
      .def(py::init<const std::string&, int>(), py::arg("name"),
           py::arg("loop_count") = 1)
      .def(
          "add_pass",
          static_cast<void (Pipeline::*)(std::shared_ptr<xla::HloPassInterface>,
                                         int)>(&Pipeline::AddPass),
          py::arg("hlo_pass"), py::arg("loop_count") = 1)
      .def("add_pass",
           static_cast<void (Pipeline::*)(std::shared_ptr<PassInterface>)>(
               &Pipeline::AddPass))
      .def("add_invariant_checker",
           static_cast<void (Pipeline::*)(  // NOLINT(whitespace/parens)
               std::shared_ptr<xla::HloPassInterface>)>(
               &Pipeline::AddInvariantChecker),
           py::arg("hlo_pass"))
      .def("add_invariant_checker",
           static_cast<void (Pipeline::*)(std::shared_ptr<PassInterface>)>(
               &Pipeline::AddInvariantChecker))
      .def_property_readonly("name", &Pipeline::name)
      .def_property_readonly("changed", &Pipeline::changed);

  py::class_<AltPipeline, PassInterface, std::shared_ptr<AltPipeline>>(
      m, "AltPipeline")
      .def(py::init<std::shared_ptr<PassInterface>, int>(), py::arg("pass"),
           py::arg("loop_count") = 1)
      .def_property_readonly("name", &AltPipeline::name)
      .def_property_readonly("changed", &AltPipeline::changed);
}
