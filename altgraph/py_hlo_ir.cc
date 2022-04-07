// Copyright 2021 Garena Online Private Limited

#include "altgraph/py_hlo_ir.h"

PyHloIr::PyHloIr(const std::string& hlo_filepath, const std::string& platform)
    : platform_(platform) {
  std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
      [](xla::HloModuleConfig* config) { config->set_seed(42); };

  std::unique_ptr<xla::HloModule> temp_hlo_module =
      LoadModuleFromFile(hlo_filepath, xla::hlo_module_loader_details::Config(),
                         "txt", config_modifier_hook)
          .ValueOrDie();
  const xla::HloModuleProto hlo_module_proto = temp_hlo_module->ToProto();

  if (platform == "gpu") {
    client_ = xla::GetGpuClient(/*asynchronous=*/true,
                                xla::GpuAllocatorConfig(), nullptr, 0)
                  .ValueOrDie();
  } else if (platform == "cpu") {
    LOG(FATAL) << "HloIr currently not enabled for platform == cpu";
    // client = GetCpuClient(/*asynchronous=*/true).ValueOrDie();
  } else {
    LOG(FATAL) << "Unknown platform " << platform;
  }

  // Compile XlaComputation to PjRtExecutable.
  xla::XlaComputation xla_computation(hlo_module_proto);
  xla::CompileOptions compile_options;

  try {
    std::unique_ptr<xla::PjRtExecutable> executable =
        client_->Compile(xla_computation, compile_options).ValueOrDie();
  } catch (xla::Intercept<xla::cpu::CpuCompiler>& e) {
    cpu_intercept_ = std::move(e);
  } catch (xla::Intercept<xla::gpu::GpuCompiler>& e) {
    gpu_intercept_ = std::move(e);
    hlo_module_ = std::move(gpu_intercept_.module);
  }
}

PyHloIr::EvaluationResult PyHloIr::Evaluate(int times) {
  PyHloIr::EvaluationResult result;
  result.durations.reserve(times);

  if (platform_ == "gpu") {
    evaluator_.Compile(hlo_module_->ToProto(),
                       /* rerun_hlo = */ false, client_.get());
    auto ret = evaluator_.Evaluate(times);

    xla::Evaluator::BufferPack& output = ret.output;

    for (auto& pjrt_buf_vector : output) {
      result.output.push_back(std::vector<py::object>());
      for (auto& pjrt_buf_ptr : pjrt_buf_vector) {
        std::shared_ptr<xla::Literal> literal =
            pjrt_buf_ptr->ToLiteral().ValueOrDie();
        result.output.back().push_back(
            std::move(xla::LiteralToPython(literal).ValueOrDie()));
      }
    }
    std::transform(ret.durations.begin(), ret.durations.end(),
                   std::back_inserter(result.durations),
                   [](absl::Duration duration) -> uint64_t {
                     return duration / absl::Nanoseconds(1);
                   });
  }
  return result;
}

std::shared_ptr<xla::HloModule> PyHloIr::SaveHloModule() {
  std::shared_ptr<xla::HloModule> saved_hlo_module = std::move(hlo_module_);
  hlo_module_ = saved_hlo_module->Clone();
  return saved_hlo_module;
}

void PyHloIr::RestoreHloModule(
    std::shared_ptr<xla::HloModule> saved_hlo_module) {
  hlo_module_ = saved_hlo_module;
}

std::string PyHloIr::ExportHloModuleToStr() { return hlo_module_->ToString(); }

void PyHloIr::PreFusionOptimizations() {
  if (platform_ == "gpu") {
    gpu_intercept_.compiler->OptimizeHloModulePreFusion(
        hlo_module_.get(), gpu_intercept_.stream_exec,
        gpu_intercept_.options.device_allocator);
  }
}

void PyHloIr::FusionDryRun() {
  if (platform_ == "gpu") {
    for (xla::HloComputation* computation :
         hlo_module_.get()->MakeNonfusionComputations()) {
      computation->set_dry(true);
    }
    gpu_intercept_.compiler->OptimizeHloModuleFusionRun(
        hlo_module_.get(), gpu_intercept_.stream_exec,
        gpu_intercept_.options.device_allocator);
    for (xla::HloComputation* computation :
         hlo_module_.get()->MakeNonfusionComputations()) {
      computation->set_dry(false);
    }
  }
}

void PyHloIr::PostFusionOptimizations() {
  if (platform_ == "gpu") {
    gpu_intercept_.compiler->OptimizeHloModulePostFusion(
        hlo_module_.get(), gpu_intercept_.stream_exec,
        gpu_intercept_.options.device_allocator);
  }
}

PyHloGraph PyHloIr::GetHloGraph(bool do_hash_verification) {
  return PyHloGraph(hlo_module_.get(), do_hash_verification);
}

// TODO(ohcy): Make it take a (uid_ptr, decision) arg instead, save time on
// rebuilding the HloGraph
void PyHloIr::ApplyAlternatives(py::array_t<size_t> decisions) {
  if (platform_ == "gpu") {
    PyHloGraph py_hlo_graph = PyHloGraph(hlo_module_.get(), false);
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
         decisions_idx += 2) {
      size_t node_idx = decisions_ptr[decisions_idx];
      size_t decision = decisions_ptr[decisions_idx + 1];
      int uid = node_feats.uids->at(node_idx);

      xla::HloInstruction* instruction = uid_to_inst.at(uid);

      // OCYTEMP -> sanity checks while debugging
      if (instruction->opcode() != xla::HloOpcode::kAlternatives) {
        LOG(FATAL) << "Trying to apply alternatives to non-kAlternatives node!";
      }
      static_cast<xla::HloAlternatives*>(instruction)->Select(decision);
    }

    for (xla::HloComputation* computation :
         hlo_module_.get()->MakeNonfusionComputations()) {
      // Remove the residue
      computation->Prune();
    }
  }
}

PYBIND11_MODULE(hlo_ir, m) {
  // TODO(ohcy) Change PyHloGraph and PyHloIr names to remove the Py prefix
  py::class_<PyHloGraph> py_hlo_graph(m, "PyHloGraph");
  py_hlo_graph.def(py::init<const xla::HloModule*>())
      .def("hash", &PyHloGraph::py_hash)
      .DEF_PYBIND_READONLY(PyHloGraph, out_edge_offsets)
      .DEF_PYBIND_READONLY(PyHloGraph, out_edge_indices)
      .DEF_PYBIND_READONLY(PyHloGraph, in_edge_offsets)
      .DEF_PYBIND_READONLY(PyHloGraph, in_edge_indices)
      .DEF_PYBIND_READONLY(PyHloGraph, alternative_indices)
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

  py::class_<PyHloIr::EvaluationResult>(m, "EvaluationResult")
      .def_readonly("durations", &PyHloIr::EvaluationResult::durations)
      .def_readonly("output", &PyHloIr::EvaluationResult::output);

  py::class_<PyHloIr>(m, "PyHloIr")
      .def(py::init<const std::string&, const std::string&>())
      .def("evaluate", &PyHloIr::Evaluate)
      .def("save_hlo", &PyHloIr::SaveHloModule)
      .def("restore_hlo", &PyHloIr::RestoreHloModule)
      .def("export_hlo_to_str", &PyHloIr::ExportHloModuleToStr)
      .def("get_hlo_graph", &PyHloIr::GetHloGraph,
           py::arg("do_hash_verification") = true)
      .def("pre_fusion_optimizations", &PyHloIr::PreFusionOptimizations)
      .def("fusion_dry_run", &PyHloIr::FusionDryRun)
      .def("post_fusion_optimizations", &PyHloIr::PostFusionOptimizations)
      .def("apply_alternatives", &PyHloIr::ApplyAlternatives);

  py::class_<xla::HloModule, std::shared_ptr<xla::HloModule>>(m, "HloModule");

  py::class_<xla::Literal, std::shared_ptr<xla::Literal>>(m, "Literal")
      .def("__repr__", &xla::Literal::ToString);
}
