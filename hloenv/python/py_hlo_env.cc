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

#include "hloenv/python/py_hlo_env.h"

namespace hloenv {

PYBIND11_MODULE(py_hlo_env, m) {
  // TODO(ohcy) Change PyHloGraph and PyHloEnv names to remove the Py prefix
  py::class_<PyHloGraph> py_hlo_graph(m, "HloGraph", 
    "The graph representation of a HloModule the describes its structure and individual instructions features. See :ref:`Playing with HLO graph features` for more details."
    );

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
      .DEF_PYBIND_READONLY(PyNodeFeats, fused_comp_ids)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_users)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_operands)
      .DEF_PYBIND_READONLY(PyNodeFeats, opcodes)
      .DEF_PYBIND_READONLY(PyNodeFeats, opcode_attrs)
      .DEF_PYBIND_READONLY(PyNodeFeats, num_opcode_attrs)
      .DEF_PYBIND_READONLY(PyNodeFeats, is_alternative)
      .DEF_PYBIND_READONLY(PyNodeFeats, is_in_fusion)
      .DEF_PYBIND_READONLY(PyNodeFeats, in_tensor_sizes)
      .DEF_PYBIND_READONLY(PyNodeFeats, out_tensor_sizes)
      .DEF_PYBIND_READONLY(PyNodeFeats, has_max_in_tensor)
      .DEF_PYBIND_READONLY(PyNodeFeats, has_max_out_tensor)
      .DEF_PYBIND_READONLY(PyNodeFeats, normalized_num_group_inst);

  // TODO(ohcy): write this without copy as nparray
  py::class_<PyEdgeFeats>(m, "EdgeFeats")
      .def("get_tensor_size", &PyEdgeFeats::GetTensorSize)
      .DEF_PYBIND_READONLY(PyEdgeFeats, uids)
      .DEF_PYBIND_READONLY(PyEdgeFeats, srcs)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dsts)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dims)
      .DEF_PYBIND_READONLY(PyEdgeFeats, layouts)
      .DEF_PYBIND_READONLY(PyEdgeFeats, lehmercodes)
      .DEF_PYBIND_READONLY(PyEdgeFeats, types)
      .DEF_PYBIND_READONLY(PyEdgeFeats, dtypes);

  py::class_<PyHloEnv::EvaluationResult>(m, "EvaluationResult", 
                    "A structure containing the duration and output of the HloModule evaluation.")
      .def_readonly("durations", &PyHloEnv::EvaluationResult::durations,
                    "The default duration in nanoseconds. This returns the execution duration as measured within the Tensorflow evaluation code, starting from the point when the executable has been enqueued on the compute stream till the completion of the executable."
                    )
      .def_readonly("compute_durations",
                    &PyHloEnv::EvaluationResult::compute_durations,
                    "The duration in nanoseconds of the computation, without data transfer, as measured on the device."
                    )
      .def_readonly("full_durations",
                    &PyHloEnv::EvaluationResult::full_durations,
                    "The full duration in nanoseconds as measured within HloEnv.evaluate(). This captures the entire execution process including processes such as enqueueing the computation on the compute stream, and is hence more subject to timing noise.")
      .def_readonly("output", &PyHloEnv::EvaluationResult::output,
                    "The output of the HloModule."
                    );

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

  py::class_<xla::HloCostAnalysis::Properties>(m, "CostAnalysisProperties");

  py::class_<AltHloModule, std::shared_ptr<AltHloModule>>(m, "HloModule",
    "The class representing an XLA HloModule. Each HloModule can be loaded into the :class:`HloEnv`, where we can evaluate it, obtain its hash, or run specific :class:`Pass` and :class:`Pipeline` on it ")
      .def(py::init<const std::string&, const std::string&>(),
           R"hloenvdoc(
Creates a :class:`HloEnv` and loads in a HloModule from a specified filepath.

Args:
    input (str): The Hlo text input in the form of a string or filepath.
    format (str, optional): The format of the input. Can be either "path" for a filepath to a Hlo text file, or "txt" for the raw Hlo text string. Defaults to "path".
           )hloenvdoc",
           py::arg("input"), py::arg("format")="path"
       )
      .def("to_string", &AltHloModule::ToString, 
        "Converts the HloModule to a string representation. This string representation can also used to initialize a new HloEnv or loaded into an existing one.")
      .def_property_readonly("config", &AltHloModule::config,
        "The config options of the HloModule")
      .def("hash", &AltHloModule::Hash,
        "The DAGHash of the HloModule. This DAGHash is a custom hash implementation that differs from Tensorflow's existing HloModule hash implementation to better account for the structure and parameters of the Hlo Instructions.")
      .def("extract_random_submodule", &AltHloModule::ExtractRandomSubmodule)
      .def("extract_instructions_as_module",
           &AltHloModule::ExtractInstructionsAsModule)
      .def("extract_fusions_as_module", &AltHloModule::ExtractFusionsAsModule)
      .def("is_bef_enabled", &AltHloModule::IsBefEnabled, 
        "Returns whethe Binary Executable Format (BEF) is enabled for the executable.")
      .def_property_readonly("instruction_count",
                             &AltHloModule::InstructionCount,
                             "The number of instructions in the HloModule.")
      .def_property_readonly("computation_count",
                             &AltHloModule::ComputationCount,
                             "The number of computations in the HloModule.")
      .def("cost_analysis", &AltHloModule::CostAnalysis, 
        "Returns a dictionary containing the flops, transcendentals, bytes accessed and optimal seconds of the Hlo Module.")
      .def("clone", &AltHloModule::Clone, "Clones the HloModule.");

  py::class_<PyHloEnv>(m, "HloEnv", 
                       "The class representing the HloEnv. Each HloEnv instance can be loaded with a single HloModule, and used to run Passes/Pipelines on that module, as well as extract features, evaluate and obtain the graph features for that HloModule.")
      .def(py::init<const std::string&, const std::string&>(), 
           R"hloenvdoc(
Creates a :class:`HloEnv` and loads in a HloModule from a specified filepath.

Args:
    hlo_filepath (str): The path of the HloModule text file
    platform (str): The platform we wish to run the HloModule on. Currently only 'gpu' is supported
           )hloenvdoc",           
           py::arg("hlo_filepath"), py::arg("platform"))
      .def(py::init<const std::string&, const std::string&,
                    const std::string&>(), 
           R"hloenvdoc(
Creates a :class:`HloEnv` and loads in a HloModule from its string representation.

Args:
    hlo_data (str): The HloModule string
    platform (str): The platform we wish to run the HloModule on. Currently only 'gpu' is supported
           )hloenvdoc",                       
           py::arg("hlo_data"), py::arg("format"), py::arg("platform"))
      .def(py::init<std::shared_ptr<AltHloModule>, const std::string&>(),
           R"hloenvdoc(
Creates a :class:`HloEnv` and loads in a HloModule from an existing HloModule object.

Args:
    alt_hlo_module (:class:`HloModule`): The HloModule object
    platform (str): The platform we wish to run the HloModule on. Currently only 'gpu' is supported
           )hloenvdoc",   
           py::arg("alt_hlo_module"), py::arg("platform"))
      .def("evaluate", &PyHloEnv::Evaluate, 
           R"hloenvdoc(
              Evaluates the :class:`HloModule` loaded into the environment N times and returns both the output and the duration of each evaluation.

              Args:
                  times (int): The number of evaluations to perform
                  do_not_prep_for_eval (bool, optional): Whether to prepare the HloModule for evaluation. This can result in changes to the HloModule (e.g. insertion of Copy instructions), so set this to True if the HloModule has already gone through this process. Defaults to false.

              Returns:
                  :class:`EvaluationResult`: The structure containing the durations and output of the evaluation
           )hloenvdoc",   
           py::arg("times") = 20,
           py::arg("do_not_prep_for_eval") = false)
      .def("has_equal_output", &PyHloEnv::HasEqualOutput,
           // R"hloenvdoc(
           //    Checks whether two HloModules return the same output given identical random input.

           //    Args:
           //        first_module (:class:`HloModule`): The reference module.
           //        second_module (:class:`HloModule`): The module to compare the reference module to.
           //        times (int, optional): The number of times to repeat the evaluation when comparing the two modules. Defaults to 1.

           //    Returns:
           //        bool: True if the output is identical, False otherwise.
           // )hloenvdoc", 
           py::arg("first_module"), py::arg("second_module"),
           py::arg("times") = 1)
      .def("has_equal_output_as", &PyHloEnv::HasEqualOutputAs,
           // R"hloenvdoc(
           //    Checks whether a HloModule returns the same output as the HloModule loaded in the HloEnv given identical random input.

           //    Args:
           //        second_module (:class:`HloModule`): The module to compare the loaded HloModule to.
           //        times (int, optional): The number of times to repeat the evaluation when comparing the two modules. Defaults to 1.

           //    Returns:
           //        bool: True if the output is identical, False otherwise.
           // )hloenvdoc",         
           py::arg("other_module"), py::arg("times") = 1)
      .def("clone_hlo", &PyHloEnv::CloneHloModule, 
           "Clones the currently loaded :class:`HloModule` and returns it.")
      .def("load_hlo",
           static_cast<void (PyHloEnv::*)(std::shared_ptr<AltHloModule>)>(
               &PyHloEnv::LoadHloModule),
           R"hloenvdoc(
              Loads in a new :class:`HloModule` from an existing :class:`HloModule` object.

              Args:
                  hlo_module (:class:`HloModule`): The HloModule to be loaded in.
           )hloenvdoc",   
           py::arg("hlo_module")
         )
      .def("load_hlo",
           static_cast<void (PyHloEnv::*)(const std::string&,
                                          const std::string&)>(
               &PyHloEnv::LoadHloModule),
           R"hloenvdoc(
              Loads in a new :class:`HloModule` from text data.

              Args:
                  hlo_data (str): The HloModule data to be loaded in in the form of a filepath or raw Hlo text string.
                  format (str, optional): The format of the Hlo data. Defaults to "path".
           )hloenvdoc",   
           py::arg("hlo_data"), py::arg("format") = "path")
      .def("export_hlo_to_str", &PyHloEnv::ExportHloModuleToStr,
        "Exports the currently loaded :class:`HloModule` into its text representation. This text can be loaded by the HloEnv."
        )
      .def("get_hlo_module", &PyHloEnv::GetHloModule,
        "Get the :class:`HloModule` object loaded in the :class:`HloEnv`")
      .def("get_hlo_graph", &PyHloEnv::GetHloGraph, py::arg("debug") = false,
           "Converts the :class:`HloModule` into a :class:'HloGraph' object that describes the features and structure of the HloModule.",
           py::arg("inline_fused_comp") = false,
           py::arg("do_hash_verification") = false 
        )
      .def("optimize_hlo_module", &PyHloEnv::OriginalOptimizeHloModule,
           "Runs the original Xla Optimization Pipeline on the HloModule to obtain a baseline reference against XLA.")
      .def("prepare_for_eval", &PyHloEnv::PrepareForEvaluation,
           "Prepare the HloModule for IR emitting and evaluation. This step is automatically run during HloEnv.evaluate, unless the do_not_prep_for_eval parameter is set to True, hence in most cases you will not have to run this function.")
      .def("run", &PyHloEnv::Run,
           R"hloenvdoc(
              Runs the specified :class:`Pass` or :class:`Pipeline` on the :class:`HloModule` loaded in the environment.

              Args:
                  pass_pipeline (Union[:class:`Pass`, :class:`Pipeline`]): The Pass or Pipeline we wish to run on the :class:`HloModule` loaded in the environment.

              Returns:
                  bool: True if alternatives were generated (i.e. the pass or one of the passes in the pipeline is an AltPipeline), False otherwise. Note that if this returns True, it indicates that an apply_alternatives call must be run to pick the decisions at each alternative before the HloModule can be evaluated.
           )hloenvdoc",     
           py::arg("pass_pipeline")
        )
      .def("get_hlo_module_hash", &PyHloEnv::GetHloModuleHash,
          "Returns the HloDagHash of the class:`HloModule` loaded in the environment."
        )
      .def("apply_alternatives", &PyHloEnv::ApplyAlternatives,
           R"hloenvdoc(
              Applies the specified decisions to the alternative nodes in the HloModule graph, and prunes the resulting graph.

              Args:
                  decisions (ndarray): 2D array of (node_uid, decision) pairs where decision = i indicates that we wish to select the ith alternative at the alternative node corresponding to node_uid.
           )hloenvdoc",  
           py::arg("decisions")
        );

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
           static_cast<bool (PassInterface::*)(std::shared_ptr<AltHloModule>)>(
               &PassInterface::Run),
           py::arg("hlo_pass"))
      .def_property_readonly("name", &PassInterface::name);

  py::class_<Pass, PassInterface, std::shared_ptr<Pass>>(m, "Pass")
      .def(py::init<std::shared_ptr<xla::HloPassInterface>, int>(),
           R"hloenvdoc(
              Creates a new Pipeline.

              Args:
                  hlo_pass (:class:`HloPassInterface`): The XLA Pass/Pipeline we wish to run. See :ref:`List of currently enabled XLA Hlo Passes` for more details.
                  name (loop_count): The number of times to run this Pipeline. Set this to -1 to run it until no further changes to the HloModule occur, up to a maximum of 25 times. Defaults to 1.

              Examples:
                  fusion_pass = hloenv.HloPass.GpuInstructionFusion(may_duplicate=True)
           )hloenvdoc",            
           py::arg("hlo_pass"), py::arg("loop_count") = 1)
      .def_property_readonly("changed", &Pass::changed)
      .def_property_readonly("name", &Pass::name);

  py::class_<Pipeline, PassInterface, std::shared_ptr<Pipeline>>(m, "Pipeline")
      .def(py::init<const std::string&, int>(), 
           R"hloenvdoc(
              Creates a new Pipeline.

              Args:
                  name (str): The name of this Pipeline.
                  loop_count (int, optional): The number of times to run this Pipeline. Set this to -1 to run it until no further changes to the HloModule occur, up to a maximum of 25 times. Defaults to 1.
           )hloenvdoc",           
           py::arg("name"), py::arg("loop_count") = 1)
      .def("add_pass",
          static_cast<void (Pipeline::*)(std::shared_ptr<xla::HloPassInterface>,
                                         int)>(&Pipeline::AddPass),
          py::arg("hlo_pass"), py::arg("loop_count") = 1)
      .def("add_pass",
           static_cast<void (Pipeline::*)(std::shared_ptr<PassInterface>)>(
               &Pipeline::AddPass),
           "Add a :class:`Pass` or :class:`Pipeline` to this Pipeline.")
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

}  // namespace hloenv
