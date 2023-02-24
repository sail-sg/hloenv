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

#include "hloenv/python/py_hlo_passes.h"

namespace hloenv {

void py_init_hlo_passes(const py::module& m) {
  py::class_<xla::HloPassInterface, std::shared_ptr<xla::HloPassInterface>>(
      m, "HloPassInterface");

  // GPU PASSES:
  py::class_<xla::AlgebraicSimplifier,
             std::shared_ptr<xla::AlgebraicSimplifier>, xla::HloPassInterface>(
      m, "AlgebraicSimplifier")
      // TODO(ohcy): Change this to use protobuf
      .def(
          py::init([](py::dict options_dict) {
            xla::AlgebraicSimplifierOptions options;
            for (auto item : options_dict) {
              const std::string option = item.first.cast<const std::string>();

              if (option == "is_layout_sensitive") {
                options.set_is_layout_sensitive(item.second.cast<bool>());
              } else if (option == "enable_dot_strength_reduction") {
                options.set_enable_dot_strength_reduction(
                    item.second.cast<bool>());
              } else if (option == "enable_dot_to_multiply_rewrite") {
                options.set_enable_dot_to_multiply_rewrite(
                    item.second.cast<bool>());
              } else if (option == "enable_conv_simplification") {
                options.set_enable_conv_simplification(
                    item.second.cast<bool>());
              } else if (option == "enable_conv_operand_swap") {
                options.set_enable_conv_operand_swap(item.second.cast<bool>());
              } else if (option == "enable_scalar_multiply_reduction") {
                options.set_enable_scalar_multiply_reduction(
                    item.second.cast<bool>());
              } else if (option == "enable_floats_are_real") {
                options.set_enable_floats_are_real(item.second.cast<bool>());
              } else if (option ==
                         "enable_window_reduce_to_reduce_replacement") {
                options.set_enable_window_reduce_to_reduce_replacement(
                    item.second.cast<bool>());
              } else if (option == "very_small_gather_size") {
                options.set_very_small_gather_size(item.second.cast<int64_t>());
              } else if (option ==
                         "cudnn_batchnorm_forward_training_metadata") {
                options.set_cudnn_batchnorm_forward_training_metadata(
                    item.second.cast<const std::string>());
              } else if (option == "enable_reduce_of_reshape") {
                options.set_enable_reduce_of_reshape(item.second.cast<bool>());
              } else if (option == "enable_negative_padding_replacement") {
                options.set_enable_negative_padding_replacement(
                    item.second.cast<bool>());
              } else if (option == "enable_sink_broadcast") {
                options.set_enable_sink_broadcast(item.second.cast<bool>());
              } else if (option == "replace_transpose_with_bitcast") {
                options.set_replace_transpose_with_bitcast(
                    item.second.cast<bool>());
              } else if (option == "minmax_propagate_nan") {
                options.set_minmax_propagate_nan(item.second.cast<bool>());
              } else {
                LOG(FATAL) << "Option '" << option
                           << "' not an option for AlgebraicSimplifierOptions";
              }
            }
            return std::make_shared<xla::AlgebraicSimplifier>(options);
          }),
          py::arg("options"));

  py::class_<xla::AllGatherBroadcastReorder,
             std::shared_ptr<xla::AllGatherBroadcastReorder>,
             xla::HloPassInterface>(m, "AllGatherBroadcastReorder")
      .def(py::init<>());

  py::class_<xla::AllGatherCombiner, std::shared_ptr<xla::AllGatherCombiner>,
             xla::HloPassInterface>(m, "AllGatherCombiner")
      .def(py::init<int64_t, int64_t>(), py::arg("combine_threshold_in_bytes"),
           py::arg("combine_threshold_count"));

  py::class_<xla::AllReduceCombiner, std::shared_ptr<xla::AllReduceCombiner>,
             xla::HloPassInterface>(m, "AllReduceCombiner")
      .def(py::init<int64_t, int64_t>(), py::arg("combine_threshold_in_bytes"),
           py::arg("combine_threshold_count"));

  py::class_<xla::AllReduceContiguous,
             std::shared_ptr<xla::AllReduceContiguous>, xla::HloPassInterface>(
      m, "AllReduceContiguous")
      .def(py::init<>());

  py::class_<xla::AllReduceFolder, std::shared_ptr<xla::AllReduceFolder>,
             xla::HloPassInterface>(m, "AllReduceFolder")
      .def(py::init<>());

  py::class_<xla::AllReduceReassociate,
             std::shared_ptr<xla::AllReduceReassociate>, xla::HloPassInterface>(
      m, "AllReduceReassociate")
      .def(py::init<>());

  py::class_<xla::AsyncCollectiveCreator,
             std::shared_ptr<xla::AsyncCollectiveCreator>,
             xla::HloPassInterface>(m, "AsyncCollectiveCreator")
      .def(py::init([]() {
        // TODO(ohcy): Currently this specialization uses the function defined
        // in gpu_compiler.h:OptimizeHloModule
        xla::AsyncCollectiveCreator::CollectiveCreatorConfig config;
        config.convert_all_reduce = [](const xla::HloInstruction*) {
          return true;
        };
        return std::make_shared<xla::AsyncCollectiveCreator>(config);
      }));

  py::class_<xla::BatchNormExpander, std::shared_ptr<xla::BatchNormExpander>,
             xla::HloPassInterface>(m, "BatchNormExpander")
      .def(py::init<bool, bool, bool>(), py::arg("rewrite_training_op") = false,
           py::arg("rewrite_inference_op") = false,
           py::arg("rewrite_grad_op") = false);

  py::class_<xla::BFloat16Normalization,
             std::shared_ptr<xla::BFloat16Normalization>,
             xla::HloPassInterface>(m, "BFloat16Normalization")
      .def(py::init([](bool supports_matrix_multiplication) {
        // TODO(ohcy): Would be better to change it to take a unique_ptr and
        // let the TF side manage the lifetime of the support object, but that
        // requires more changes to TF. KIV
        xla::gpu::GpuBfloat16Support* bf16 = new xla::gpu::GpuBfloat16Support(
            supports_matrix_multiplication, HloEnvGpuBackend::StreamExecutor());
        auto deleter = [](xla::BFloat16Normalization* pass) {
          delete pass->bfloat16_support();
        };

        return std::shared_ptr<xla::BFloat16Normalization>(
            new xla::BFloat16Normalization{bf16}, deleter);
      }));

  py::class_<xla::CallInliner, std::shared_ptr<xla::CallInliner>,
             xla::HloPassInterface>(m, "CallInliner")
      .def(py::init<bool, bool>(), py::arg("single_call_site") = false,
           py::arg("update_domain") = false);

  py::class_<xla::CollectivesScheduleLinearizer,
             std::shared_ptr<xla::CollectivesScheduleLinearizer>,
             xla::HloPassInterface>(m, "CollectivesScheduleLinearizer")
      .def(py::init<>());

  py::class_<xla::ConditionalCanonicalizer,
             std::shared_ptr<xla::ConditionalCanonicalizer>,
             xla::HloPassInterface>(m, "ConditionalCanonicalizer")
      .def(py::init<>());

  py::class_<xla::ConditionalSimplifier,
             std::shared_ptr<xla::ConditionalSimplifier>,
             xla::HloPassInterface>(m, "ConditionalSimplifier")
      .def(py::init<>());

  py::class_<xla::DotDecomposer, std::shared_ptr<xla::DotDecomposer>,
             xla::HloPassInterface>(m, "DotDecomposer")
      .def(py::init<>());

  py::class_<xla::DotMerger, std::shared_ptr<xla::DotMerger>,
             xla::HloPassInterface>(m, "DotMerger")
      .def(py::init<int64_t>(), py::arg("max_size_to_merge"));

  py::class_<xla::DynamicDimensionSimplifier,
             std::shared_ptr<xla::DynamicDimensionSimplifier>,
             xla::HloPassInterface>(m, "DynamicDimensionSimplifier")
      .def(py::init<>());

  py::class_<xla::DynamicIndexSplitter,
             std::shared_ptr<xla::DynamicIndexSplitter>, xla::HloPassInterface>(
      m, "DynamicIndexSplitter")
      .def(py::init<>());

  py::class_<xla::DynamicPadder, std::shared_ptr<xla::DynamicPadder>,
             xla::HloPassInterface>
      dynamic_padder(m, "DynamicPadder");
  dynamic_padder.def(py::init([](py::dict options_dict) {
    auto options = xla::DynamicPadderOptions();
    for (auto item : options_dict) {
      const std::string option = item.first.cast<const std::string>();

      if (option == "slice_dynamic_output") {
        options.slice_dynamic_output = item.second.cast<bool>();
      } else if (option == "shape_check_mode") {
        options.slice_dynamic_output =
            item.second.cast<xla::DynamicDimensionInference::ShapeCheckMode>();
      } else if (option == "op_supports_dynamism_handler") {
        // TODO(ohcy)
        LOG(FATAL) << "op_supports_dynamism_handler not yet handled for "
                      "DynamicPadderOptions";
      } else if (option == "custom_call_handler") {
        // TODO(ohcy)
        LOG(FATAL) << "custom_call_handler not yet handled for "
                      "DynamicPadderOptions";
      } else {
        LOG(FATAL) << "Option '" << option
                   << "' not an option for DynamicPadderOptions";
      }
    }
    return std::make_shared<xla::DynamicPadder>(options);
  }));
  py::enum_<xla::DynamicDimensionInference::ShapeCheckMode>(dynamic_padder, 
                                                            "ShapeCheckMode",
                                                            py::module_local())
      .value("kInvalid",
             xla::DynamicDimensionInference::ShapeCheckMode::kInvalid)
      .value("kCompileTime",
             xla::DynamicDimensionInference::ShapeCheckMode::kCompileTime)
      .value("kRuntimeTime",
             xla::DynamicDimensionInference::ShapeCheckMode::kRuntimeTime)
      .value("kIgnore", xla::DynamicDimensionInference::ShapeCheckMode::kIgnore)
      .export_values();

  py::class_<xla::FlattenCallGraph, std::shared_ptr<xla::FlattenCallGraph>,
             xla::HloPassInterface>(m, "FlattenCallGraph")
      .def(py::init<>());

  py::class_<xla::HloConstantFolding, std::shared_ptr<xla::HloConstantFolding>,
             xla::HloPassInterface>(m, "HloConstantFolding")
      .def(py::init<>());

  py::class_<xla::HloCSE, std::shared_ptr<xla::HloCSE>, xla::HloPassInterface>(
      m, "HloCSE")
      .def(py::init<bool, bool>(), py::arg("is_layout_sensitive"),
           py::arg("only_fusion_computations") = false);

  py::class_<xla::HloDCE, std::shared_ptr<xla::HloDCE>, xla::HloPassInterface>(
      m, "HloDCE")
      .def(py::init<>())
      .def(py::init<bool>(), py::arg("remove_cross_partition_collective_ops"));

  py::class_<xla::gpu::GpuInstructionFusion,
             std::shared_ptr<xla::gpu::GpuInstructionFusion>,
             xla::HloPassInterface>(m, "GpuInstructionFusion")
      .def(py::init<bool>(), py::arg("may_duplicate"));

  py::class_<xla::gpu::GpuLayoutAssignment,
             std::shared_ptr<xla::gpu::GpuLayoutAssignment>,
             xla::HloPassInterface>(m, "GpuLayoutAssignment")
      // .def(py::init<ComputationLayout*, se::StreamExecutor*,
      //               ChannelLayoutConstraints*>(),
      //      py::arg("entry_computation_layout"), py::arg("stream_executor"),
      //      py::arg("channel_constraints") = nullptr);
      .def(py::init([](std::shared_ptr<AltHloModule> alt_hlo_module) {
        // TODO(ohcy): Currently we just default to normal LayoutConstraints,
        // we do not have the ability to add more LayoutConstraints
        xla::ChannelLayoutConstraints layout_constraints;
        return std::make_shared<xla::gpu::GpuLayoutAssignment>(
            alt_hlo_module->hlo_module_ptr()
                ->mutable_entry_computation_layout(),
            HloEnvGpuBackend::StreamExecutor(), &layout_constraints);
      }));

  py::class_<xla::AllToAllDecomposer, std::shared_ptr<xla::AllToAllDecomposer>,
             xla::HloPassInterface>(m, "AllToAllDecomposer")
      .def(py::init<bool, int64_t>(), py::arg("decompose_to_tuple") = true,
           py::arg("min_array_rank") = 0);

  py::class_<xla::BitcastDtypesExpander,
             std::shared_ptr<xla::BitcastDtypesExpander>,
             xla::HloPassInterface>(m, "BitcastDtypesExpander")
      .def(py::init<>());

  py::class_<xla::ComparisonExpander, std::shared_ptr<xla::ComparisonExpander>,
             xla::HloPassInterface>(m, "ComparisonExpander")
      .def(py::init<>());

  py::class_<xla::Convolution4DExpander,
             std::shared_ptr<xla::Convolution4DExpander>,
             xla::HloPassInterface>(m, "Convolution4DExpander")
      .def(py::init<>());

  py::class_<xla::EighExpander, std::shared_ptr<xla::EighExpander>,
             xla::HloPassInterface>(m, "EighExpander")
      .def(py::init<>());

  py::class_<xla::GatherExpander, std::shared_ptr<xla::GatherExpander>,
             xla::HloPassInterface>
      gather_expander(m, "GatherExpander");
  gather_expander.def(py::init<xla::GatherExpander::Mode>(), py::arg("m"));
  py::enum_<xla::GatherExpander::Mode>(gather_expander, "Mode", py::module_local())
      .value("kEliminateAllGathers",
             xla::GatherExpander::Mode::kEliminateAllGathers)
      .value("kEliminateSimpleGathers",
             xla::GatherExpander::Mode::kEliminateSimpleGathers)
      .export_values();

  py::class_<xla::LogisticExpander, std::shared_ptr<xla::LogisticExpander>,
             xla::HloPassInterface>
      logistic_expander(m, "LogisticExpander");
  logistic_expander.def(py::init<xla::LogisticExpansionType>(),
                        py::arg("expansion_type"));
  py::enum_<xla::LogisticExpansionType>(logistic_expander,
                                        "LogisticExpansionType",
                                        py::module_local())
      .value("kTanh", xla::LogisticExpansionType::kTanh)
      .value("kExp", xla::LogisticExpansionType::kExp)
      .export_values();

  py::class_<xla::OperandUpcaster, std::shared_ptr<xla::OperandUpcaster>,
             xla::HloPassInterface>(m, "OperandUpcaster")
      // .def(py::init<PatternExtraFilter>(), py::arg("extra_filter") =
      // nullptr);
      .def(py::init([]() {
        // TODO(ohcy): Currently this specialization uses the function defined
        // in gpu_compiler.h:OptimizeHloModule
        xla::OpExpanderPass::PatternExtraFilter upcaster_filter =
            [&](const xla::HloInstruction* instr) {
              return !(HloEnvGpuBackend::StreamExecutor()
                           ->GetDeviceDescription()
                           .cuda_compute_capability()
                           .IsAtLeast(xla::se::CudaComputeCapability::VOLTA)) ||
                     !xla::gpu::IsMatrixMultiplication(*instr);
            };
        return std::make_shared<xla::OperandUpcaster>(upcaster_filter);
      }));

  py::class_<xla::OptimizationBarrierExpander,
             std::shared_ptr<xla::OptimizationBarrierExpander>,
             xla::HloPassInterface>(m, "OptimizationBarrierExpander")
      .def(py::init<>());

  py::class_<xla::QrExpander, std::shared_ptr<xla::QrExpander>,
             xla::HloPassInterface>(m, "QrExpander")
      .def(py::init<>());

  py::class_<xla::RealImagExpander, std::shared_ptr<xla::RealImagExpander>,
             xla::HloPassInterface>(m, "RealImagExpander")
      .def(py::init<>());

  py::class_<xla::ResultCaster, std::shared_ptr<xla::ResultCaster>,
             xla::HloPassInterface>(m, "ResultCaster")
      // .def(py::init<PatternExtraFilter>(), py::arg("extra_filter") =
      // nullptr);
      .def(py::init([]() {
        // TODO(ohcy): Currently this specialization uses the function defined
        // in gpu_compiler.h:OptimizeHloModule
        xla::OpExpanderPass::PatternExtraFilter upcaster_filter =
            [&](const xla::HloInstruction* instr) {
              return !(HloEnvGpuBackend::StreamExecutor()
                           ->GetDeviceDescription()
                           .cuda_compute_capability()
                           .IsAtLeast(xla::se::CudaComputeCapability::VOLTA)) ||
                     !xla::gpu::IsMatrixMultiplication(*instr);
            };
        return std::make_shared<xla::ResultCaster>(upcaster_filter);
      }));

  py::class_<xla::RngBitGeneratorExpander,
             std::shared_ptr<xla::RngBitGeneratorExpander>,
             xla::HloPassInterface>
      rng_bit_generator_expander(m, "RngBitGeneratorExpander");
  rng_bit_generator_expander.def(py::init<xla::RandomAlgorithm>(),
                                 py::arg("default_algorithm"));
  py::enum_<xla::RandomAlgorithm>(rng_bit_generator_expander, "RandomAlgorithm", 
                                  py::module_local())
      .value("RNG_DEFAULT", xla::RandomAlgorithm::RNG_DEFAULT)
      .value("RNG_THREE_FRY", xla::RandomAlgorithm::RNG_THREE_FRY)
      .value("RNG_PHILOX", xla::RandomAlgorithm::RNG_PHILOX)
      .export_values();

  py::class_<xla::RngExpander, std::shared_ptr<xla::RngExpander>,
             xla::HloPassInterface>(m, "RngExpander")
      .def(py::init<>());

  py::class_<xla::ScatterExpander, std::shared_ptr<xla::ScatterExpander>,
             xla::HloPassInterface>
      scatter_expander(m, "ScatterExpander");
  scatter_expander.def(py::init<xla::ScatterExpander::Mode>(), py::arg("m"));
  py::enum_<xla::ScatterExpander::Mode>(scatter_expander, "Mode", 
                                        py::module_local())
      .value("kEliminateAllScatters",
             xla::ScatterExpander::Mode::kEliminateAllScatters)
      .value("kEliminateSimpleScatters",
             xla::ScatterExpander::Mode::kEliminateSimpleScatters)
      .export_values();

  py::class_<xla::GpuScatterExpander, std::shared_ptr<xla::GpuScatterExpander>,
             xla::HloPassInterface>(m, "GpuScatterExpander")
      .def(py::init<>());

  py::class_<xla::StableSortExpander, std::shared_ptr<xla::StableSortExpander>,
             xla::HloPassInterface>(m, "StableSortExpander")
      .def(py::init<>());

  py::class_<xla::ReduceScatterCombiner,
             std::shared_ptr<xla::ReduceScatterCombiner>,
             xla::HloPassInterface>(m, "ReduceScatterCombiner")
      .def(py::init<int64_t, int64_t>(), py::arg("combine_threshold_in_bytes"),
           py::arg("combine_threshold_count"));

  py::class_<xla::ReshapeMover, std::shared_ptr<xla::ReshapeMover>,
             xla::HloPassInterface>(m, "ReshapeMover")
      .def(py::init<>());

  py::class_<xla::ShardingPropagation,
             std::shared_ptr<xla::ShardingPropagation>, xla::HloPassInterface>(
      m, "ShardingPropagation")
      .def(py::init<bool, bool, bool, bool>(), py::arg("is_spmd") = false,
           py::arg("propagate_metadata") = false,
           py::arg("allow_spmd_sharding_propagation_to_output") = false,
           py::arg("cse_prevention_only") = false);

  py::class_<xla::ShardingRemover, std::shared_ptr<xla::ShardingRemover>,
             xla::HloPassInterface>(m, "ShardingRemover")
      .def(py::init<>());

  py::class_<xla::SliceSinker, std::shared_ptr<xla::SliceSinker>,
             xla::HloPassInterface>(m, "SliceSinker")
      .def(py::init<>());

  py::class_<xla::SortSimplifier, std::shared_ptr<xla::SortSimplifier>,
             xla::HloPassInterface>(m, "SortSimplifier")
      .def(py::init<>());

  py::class_<xla::TransposeFolding, std::shared_ptr<xla::TransposeFolding>,
             xla::HloPassInterface>(m, "TransposeFolding")
      .def(py::init([]() {
        // TODO(ohcy): Currently this specialization uses the function defined
        // in gpu_compiler.h:OptimizeHloModule
        // TODO(ohcy): Does not modify default transposable_conv_operands
        xla::TransposeFolding::TransposableGemmOperandsFn
            transposable_gemm_operands =
                [](const xla::HloInstruction& dot,
                   const xla::TransposeFolding::OperandIndices&
                       candidate_operands) {
                  return xla::gpu::IsMatrixMultiplication(dot)
                             ? candidate_operands
                             : xla::TransposeFolding::OperandIndices{};
                };
        return std::make_shared<xla::TransposeFolding>(
            transposable_gemm_operands);
      }));

  py::class_<xla::TupleSimplifier, std::shared_ptr<xla::TupleSimplifier>,
             xla::HloPassInterface>(m, "TupleSimplifier")
      .def(py::init<>())
      .def(py::init<bool>(), py::arg("exclude_entry_computation"));

  py::class_<xla::WhileLoopConstantSinking,
             std::shared_ptr<xla::WhileLoopConstantSinking>,
             xla::HloPassInterface>(m, "WhileLoopConstantSinking")
      .def(py::init<>());

  py::class_<xla::WhileLoopSimplifier,
             std::shared_ptr<xla::WhileLoopSimplifier>, xla::HloPassInterface>(
      m, "WhileLoopSimplifier")
      .def(py::init<>());

  py::class_<xla::WhileLoopTripCountAnnotator,
             std::shared_ptr<xla::WhileLoopTripCountAnnotator>,
             xla::HloPassInterface>(m, "WhileLoopTripCountAnnotator")
      .def(py::init<>());

  py::class_<xla::ZeroSizedHloElimination,
             std::shared_ptr<xla::ZeroSizedHloElimination>,
             xla::HloPassInterface>(m, "ZeroSizedHloElimination")
      .def(py::init<>());

  py::class_<xla::AllReduceBlueConnect,
             std::shared_ptr<xla::AllReduceBlueConnect>, xla::HloPassInterface>(
      m, "AllReduceBlueConnect")
      .def(py::init<size_t>(), py::arg("num_devices_per_host"));

  py::class_<xla::gpu::FusionBitcastLift,
             std::shared_ptr<xla::gpu::FusionBitcastLift>,
             xla::HloPassInterface>(m, "FusionBitcastLift")
      .def(py::init<>());

  py::class_<xla::gpu::FusionMerger, std::shared_ptr<xla::gpu::FusionMerger>,
             xla::HloPassInterface>(m, "FusionMerger")
      .def(py::init<>());

  py::class_<xla::gpu::GeneralFusion, std::shared_ptr<xla::gpu::GeneralFusion>,
             xla::HloPassInterface>(m, "GeneralFusion")
      .def(py::init<>());

  py::class_<xla::gpu::ReduceScatterCreator,
             std::shared_ptr<xla::gpu::ReduceScatterCreator>,
             xla::HloPassInterface>(m, "ReduceScatterCreator")
      .def(py::init<>());

  py::class_<xla::gpu::GpuHorizontalInputFusion,
             std::shared_ptr<xla::gpu::GpuHorizontalInputFusion>,
             xla::HloPassInterface>(m, "GpuHorizontalInputFusion")
      .def(py::init<>());

  py::class_<xla::gpu::GpuHorizontalLoopFusion,
             std::shared_ptr<xla::gpu::GpuHorizontalLoopFusion>,
             xla::HloPassInterface>(m, "GpuHorizontalLoopFusion")
      .def(py::init<>());

  py::class_<xla::gpu::GpuMultiOutputFusion,
             std::shared_ptr<xla::gpu::GpuMultiOutputFusion>,
             xla::HloPassInterface>(m, "GpuMultiOutputFusion")
      .def(py::init<>());

  py::class_<xla::gpu::VariadicOpSplitter,
             std::shared_ptr<xla::gpu::VariadicOpSplitter>,
             xla::HloPassInterface>(m, "VariadicOpSplitter")
      .def(py::init<>());

  py::class_<xla::spmd::StatefulRngSpmdPartitioner,
             std::shared_ptr<xla::spmd::StatefulRngSpmdPartitioner>,
             xla::HloPassInterface>(m, "StatefulRngSpmdPartitioner")
      .def(py::init<int64_t, int64_t>(), py::arg("num_partitions"),
           py::arg("num_replicas"));

  py::class_<xla::gpu::CudnnFusedConvRewriter,
             std::shared_ptr<xla::gpu::CudnnFusedConvRewriter>,
             xla::HloPassInterface>(m, "CudnnFusedConvRewriter")
      .def(py::init<>());

  py::class_<xla::gpu::CudnnPadForConvolutions,
             std::shared_ptr<xla::gpu::CudnnPadForConvolutions>,
             xla::HloPassInterface>(m, "CudnnPadForConvolutions")
      .def(py::init([]() {
        return std::make_shared<xla::gpu::CudnnPadForConvolutions>(
            HloEnvGpuBackend::StreamExecutor()
                ->GetDeviceDescription()
                .cuda_compute_capability());
      }));

  py::class_<xla::gpu::CudnnVectorizeConvolutions,
             std::shared_ptr<xla::gpu::CudnnVectorizeConvolutions>,
             xla::HloPassInterface>(m, "CudnnVectorizeConvolutions")
      .def(py::init([]() {
        return std::make_shared<xla::gpu::CudnnVectorizeConvolutions>(
            HloEnvGpuBackend::StreamExecutor()
                ->GetDeviceDescription()
                .cuda_compute_capability());
      }));

  py::class_<xla::gpu::CublasPadForGemms,
             std::shared_ptr<xla::gpu::CublasPadForGemms>,
             xla::HloPassInterface>
      cublas_pad_for_gemms(m, "CublasPadForGemms");
  cublas_pad_for_gemms.def(py::init<xla::PrimitiveType, int32_t>(),
                           py::arg("datatype"), py::arg("pad_to_multiple_of"));
  py::enum_<xla::PrimitiveType>(cublas_pad_for_gemms, "PrimitiveType", py::module_local())
      .value("PRIMITIVE_TYPE_INVALID",
             xla::PrimitiveType::PRIMITIVE_TYPE_INVALID)
      .value("PRED", xla::PrimitiveType::PRED)
      .value("S8", xla::PrimitiveType::S8)
      .value("S16", xla::PrimitiveType::S16)
      .value("S32", xla::PrimitiveType::S32)
      .value("S64", xla::PrimitiveType::S64)
      .value("U8", xla::PrimitiveType::U8)
      .value("U16", xla::PrimitiveType::U16)
      .value("U32", xla::PrimitiveType::U32)
      .value("U64", xla::PrimitiveType::U64)
      .value("F16", xla::PrimitiveType::F16)
      .value("F32", xla::PrimitiveType::F32)
      .value("BF16", xla::PrimitiveType::BF16)
      .value("F64", xla::PrimitiveType::F64)
      .value("C64", xla::PrimitiveType::C64)
      .value("C128", xla::PrimitiveType::C128)
      .value("TUPLE", xla::PrimitiveType::TUPLE)
      .value("OPAQUE_TYPE", xla::PrimitiveType::OPAQUE_TYPE)
      .value("TOKEN", xla::PrimitiveType::TOKEN)
      .export_values();

  py::class_<xla::gpu::GemmAlgorithmPicker,
             std::shared_ptr<xla::gpu::GemmAlgorithmPicker>,
             xla::HloPassInterface>(m, "GemmAlgorithmPicker")
      .def(py::init([]() {
        return std::make_shared<xla::gpu::GemmAlgorithmPicker>(
            HloEnvGpuBackend::StreamExecutor(),
            HloEnvGpuBackend::DeviceMemoryAllocator());
      }));

  py::class_<xla::gpu::GpuTreeReductionRewriter,
             std::shared_ptr<xla::gpu::GpuTreeReductionRewriter>,
             xla::HloPassInterface>(m, "GpuTreeReductionRewriter")
      .def(py::init([]() {
        return std::make_shared<xla::gpu::GpuTreeReductionRewriter>(
            HloEnvGpuBackend::StreamExecutor()
                ->GetDeviceDescription()
                .cuda_compute_capability());
      }));

  py::class_<xla::gpu::GemmBroadcastFoldingRewriter,
             std::shared_ptr<xla::gpu::GemmBroadcastFoldingRewriter>,
             xla::HloPassInterface>(m, "GemmBroadcastFoldingRewriter")
      .def(py::init<>());

  py::class_<xla::gpu::GemmRewriter, std::shared_ptr<xla::gpu::GemmRewriter>,
             xla::HloPassInterface>(m, "GemmRewriter")
      .def(py::init<>());

  py::class_<xla::gpu::GpuConvAlgorithmPicker,
             std::shared_ptr<xla::gpu::GpuConvAlgorithmPicker>,
             xla::HloPassInterface>(m, "GpuConvAlgorithmPicker")
      .def(py::init([]() {
        return std::make_shared<xla::gpu::GpuConvAlgorithmPicker>(
            HloEnvGpuBackend::StreamExecutor(),
            HloEnvGpuBackend::DeviceMemoryAllocator());
      }));

  py::class_<xla::gpu::GpusolverRewriter,
             std::shared_ptr<xla::gpu::GpusolverRewriter>,
             xla::HloPassInterface>(m, "GpusolverRewriter")
      .def(py::init<>());

  py::class_<xla::gpu::GpuConvPaddingLegalization,
             std::shared_ptr<xla::gpu::GpuConvPaddingLegalization>,
             xla::HloPassInterface>(m, "GpuConvPaddingLegalization")
      .def(py::init<>());

  py::class_<xla::gpu::GpuConvRewriter,
             std::shared_ptr<xla::gpu::GpuConvRewriter>, xla::HloPassInterface>(
      m, "GpuConvRewriter")
      .def(py::init<>());

  py::class_<xla::gpu::ReductionDegenerateDimRemover,
             std::shared_ptr<xla::gpu::ReductionDegenerateDimRemover>,
             xla::HloPassInterface>(m, "ReductionDegenerateDimRemover")
      .def(py::init<>());

  py::class_<xla::gpu::ReductionDimensionGrouper,
             std::shared_ptr<xla::gpu::ReductionDimensionGrouper>,
             xla::HloPassInterface>(m, "ReductionDimensionGrouper")
      .def(py::init<>());

  py::class_<xla::gpu::ReductionLayoutNormalizer,
             std::shared_ptr<xla::gpu::ReductionLayoutNormalizer>,
             xla::HloPassInterface>(m, "ReductionLayoutNormalizer")
      .def(py::init<>());

  py::class_<xla::gpu::ReductionSplitter,
             std::shared_ptr<xla::gpu::ReductionSplitter>,
             xla::HloPassInterface>(m, "ReductionSplitter")
      .def(py::init<>());

  py::class_<xla::gpu::TriangularSolveRewriter,
             std::shared_ptr<xla::gpu::TriangularSolveRewriter>,
             xla::HloPassInterface>(m, "TriangularSolveRewriter")
      .def(py::init<>());

  py::class_<xla::HloVerifier, std::shared_ptr<xla::HloVerifier>,
             xla::HloPassInterface>(m, "HloVerifier")
      .def(py::init<bool, bool>(), py::arg("layout_sensitive"),
           py::arg("allow_mixed_precision"));

  // // OTHER PASSES (EMPTY CONSTRUCTOR)

  // py::class_<xla::BatchDotSimplification,
  // std::shared_ptr<xla::BatchDotSimplification>, xla::HloPassInterface>(m,
  // "BatchDotSimplification")
  //     .def(py::init<>());

  // py::class_<xla::BFloat16ConversionFolding,
  // std::shared_ptr<xla::BFloat16ConversionFolding>, xla::HloPassInterface>(m,
  // "BFloat16ConversionFolding")
  //     .def(py::init<>());

  // py::class_<xla::BFloat16MixedPrecisionRemoval,
  // std::shared_ptr<xla::BFloat16MixedPrecisionRemoval>,
  // xla::HloPassInterface>(m, "BFloat16MixedPrecisionRemoval")
  //     .def(py::init<>());

  // py::class_<xla::ConditionalToSelect,
  // std::shared_ptr<xla::ConditionalToSelect>, xla::HloPassInterface>(m,
  // "ConditionalToSelect")
  //     .def(py::init<>());

  // py::class_<xla::Defuser, std::shared_ptr<xla::Defuser>,
  // xla::HloPassInterface>(m, "Defuser")
  //     .def(py::init<>());

  // py::class_<xla::Despecializer, std::shared_ptr<xla::Despecializer>,
  // xla::HloPassInterface>(m, "Despecializer")
  //     .def(py::init<>());

  // py::class_<xla::ControlDepRemover, std::shared_ptr<xla::ControlDepRemover>,
  // xla::HloPassInterface>(m, "ControlDepRemover")
  //     .def(py::init<>());

  // py::class_<xla::DryModeOn, std::shared_ptr<xla::DryModeOn>,
  // xla::HloPassInterface>(m, "DryModeOn")
  //     .def(py::init<>());

  // py::class_<xla::DryModeOff, std::shared_ptr<xla::DryModeOff>,
  // xla::HloPassInterface>(m, "DryModeOff")
  //     .def(py::init<>());

  // py::class_<xla::HloGetDimensionSizeRewriter,
  // std::shared_ptr<xla::HloGetDimensionSizeRewriter>,
  // xla::HloPassInterface>(m, "HloGetDimensionSizeRewriter")
  //     .def(py::init<>());

  // py::class_<xla::HloTrivialScheduler,
  // std::shared_ptr<xla::HloTrivialScheduler>, xla::HloPassInterface>(m,
  // "HloTrivialScheduler")
  //     .def(py::init<>());

  // py::class_<xla::HloDescheduler, std::shared_ptr<xla::HloDescheduler>,
  // xla::HloPassInterface>(m, "HloDescheduler")
  //     .def(py::init<>());

  // py::class_<xla::HloModuleDCE, std::shared_ptr<xla::HloModuleDCE>,
  // xla::HloPassInterface>(m, "HloModuleDCE")
  //     .def(py::init<>());

  // py::class_<xla::FooToBarModulePass,
  // std::shared_ptr<xla::FooToBarModulePass>, xla::HloPassInterface>(m,
  // "FooToBarModulePass")
  //     .def(py::init<>());

  // py::class_<xla::BarBlowerUpper, std::shared_ptr<xla::BarBlowerUpper>,
  // xla::HloPassInterface>(m, "BarBlowerUpper")
  //     .def(py::init<>());

  // py::class_<xla::HloSubcomputationUnification,
  // std::shared_ptr<xla::HloSubcomputationUnification>,
  // xla::HloPassInterface>(m, "HloSubcomputationUnification")
  //     .def(py::init<>());

  // py::class_<xla::IndexedArrayAnalysisPrinterPass,
  // std::shared_ptr<xla::IndexedArrayAnalysisPrinterPass>,
  // xla::HloPassInterface>(m, "IndexedArrayAnalysisPrinterPass")
  //     .def(py::init<>());

  // py::class_<xla::cpu::std::shared_ptr<xla::cpu>, CpuInstructionFusion,
  // xla::HloPassInterface>(m, "CpuInstructionFusion")
  //     .def(py::init<>());

  // py::class_<xla::InstructionFusionForTesting,
  // std::shared_ptr<xla::InstructionFusionForTesting>,
  // xla::HloPassInterface>(m, "InstructionFusionForTesting")
  //     .def(py::init<>());

  // py::class_<xla::LoopScheduleLinearizer,
  // std::shared_ptr<xla::LoopScheduleLinearizer>, xla::HloPassInterface>(m,
  // "LoopScheduleLinearizer")
  //     .def(py::init<>());

  // py::class_<xla::MapInliner, std::shared_ptr<xla::MapInliner>,
  // xla::HloPassInterface>(m, "MapInliner")
  //     .def(py::init<>());

  // py::class_<xla::MemorySpacePropagation,
  // std::shared_ptr<xla::MemorySpacePropagation>, xla::HloPassInterface>(m,
  // "MemorySpacePropagation")
  //     .def(py::init<>());

  // py::class_<xla::MultiOutputFusion, std::shared_ptr<xla::MultiOutputFusion>,
  // xla::HloPassInterface>(m, "MultiOutputFusion")
  //     .def(py::init<>());

  // py::class_<xla::OpExpanderPass, std::shared_ptr<xla::OpExpanderPass>,
  // xla::HloPassInterface>(m, "OpExpanderPass")
  //     .def(py::init<>());

  // py::class_<xla::CholeskyExpander, std::shared_ptr<xla::CholeskyExpander>,
  // xla::HloPassInterface>(m, "CholeskyExpander")
  //     .def(py::init<>());

  // py::class_<xla::ConvertOperandFolding,
  // std::shared_ptr<xla::ConvertOperandFolding>, xla::HloPassInterface>(m,
  // "ConvertOperandFolding")
  //     .def(py::init<>());

  // py::class_<xla::OptimizeInputOutputBufferAlias,
  // std::shared_ptr<xla::OptimizeInputOutputBufferAlias>,
  // xla::HloPassInterface>(m, "OptimizeInputOutputBufferAlias")
  //     .def(py::init<>());

  // py::class_<xla::ReduceScatterDecomposer,
  // std::shared_ptr<xla::ReduceScatterDecomposer>, xla::HloPassInterface>(m,
  // "ReduceScatterDecomposer")
  //     .def(py::init<>());

  // py::class_<xla::RootInstructionSinker,
  // std::shared_ptr<xla::RootInstructionSinker>, xla::HloPassInterface>(m,
  // "RootInstructionSinker")
  //     .def(py::init<>());

  // py::class_<xla::WhileLoopAllReduceCodeMotion,
  // std::shared_ptr<xla::WhileLoopAllReduceCodeMotion>,
  // xla::HloPassInterface>(m, "WhileLoopAllReduceCodeMotion")
  //     .def(py::init<>());

  // py::class_<xla::gpu::AliasPassthroughParams,
  // std::shared_ptr<xla::gpu::AliasPassthroughParams>,
  // xla::HloPassInterface>(m, "AliasPassthroughParams")
  //     .def(py::init<>());

  // py::class_<xla::gpu::GpuSanitizeConstantNames,
  // std::shared_ptr<xla::gpu::GpuSanitizeConstantNames>,
  // xla::HloPassInterface>(m, "GpuSanitizeConstantNames")
  //     .def(py::init<>());

  // py::class_<xla::spmd::std::shared_ptr<xla::spmd>,
  // CanonicalizeAllGatherForCSE, xla::HloPassInterface>(m,
  // "CanonicalizeAllGatherForCSE")
  //     .def(py::init<>());

  // // OTHER PASSES:

  // py::class_<xla::AllGatherDecomposer,
  // std::shared_ptr<xla::AllGatherDecomposer>, xla::HloPassInterface>(m,
  // "AllGatherDecomposer")
  //     .def(py::init<std::function<bool(const HloAllGatherInstruction&)>>(),
  //     py::arg("should_decompose")) .def(py::init<>());

  // py::class_<xla::AllReduceSimplifier,
  // std::shared_ptr<xla::AllReduceSimplifier>, xla::HloPassInterface>(m,
  // "AllReduceSimplifier")
  //     .def(py::init<int64_t>(), py::arg("replica_count"));

  // py::class_<xla::ArCrsCombiner, std::shared_ptr<xla::ArCrsCombiner>,
  // xla::HloPassInterface>(m, "ArCrsCombiner")
  //     .def(py::init<int, bool>(), py::arg("num_spatial_partitions"),
  //     py::arg("spmd_partition"));

  // py::class_<xla::BFloat16Propagation,
  // std::shared_ptr<xla::BFloat16Propagation>, xla::HloPassInterface>(m,
  // "BFloat16Propagation")
  //     .def(py::init<const BFloat16Support*>(), py::arg("bfloat16_support"));

  // py::class_<xla::ConditionalCodeMotion,
  // std::shared_ptr<xla::ConditionalCodeMotion>, xla::HloPassInterface>(m,
  // "ConditionalCodeMotion")
  //     .def(py::init<bool, bool, int64_t>(), py::arg("is_layout_sensitive"),
  //     py::arg("pursue_full_conditional_code_motion"),
  //     py::arg("search_config") = 0);

  // py::class_<xla::ConvolutionGroupConverter,
  // std::shared_ptr<xla::ConvolutionGroupConverter>, xla::HloPassInterface>(m,
  // "ConvolutionGroupConverter")
  //     .def(py::init<std::function<bool(HloInstruction*)>,
  //     std::function<bool(HloInstruction*)>, bool, bool>(),
  //     py::arg("should_expand"), py::arg("is_cost_viable"),
  //     py::arg("convert_batch_groups_only"), py::arg("filter_expansion") =
  //     true);

  // py::class_<xla::CopyInsertion, std::shared_ptr<xla::CopyInsertion>,
  // xla::HloPassInterface>(m, "CopyInsertion")
  //     .def(py::init<const HloDataflowAnalysis::CanShareBuffer&, int64_t>(),
  //     py::arg("can_share_buffer") = nullptr,
  //     py::arg("use_region_based_live_range_analysis") =
  //     kUseRegionAnalysisLimit);

  // py::class_<xla::HloDomainIsolator, std::shared_ptr<xla::HloDomainIsolator>,
  // xla::HloPassInterface>(m, "HloDomainIsolator")
  //     .def(py::init<DomainCreatorFactory>(), py::arg("creator_factory_"));

  // py::class_<xla::HloDomainRemover, std::shared_ptr<xla::HloDomainRemover>,
  // xla::HloPassInterface>(m, "HloDomainRemover")
  //     .def(py::init<absl::string_view, std::function<Status(const, const
  //     DomainMetadata*>(), py::arg("kind"),
  //     py::arg("DomainMetadata::Domain&"), py::arg("metadata"));

  // py::class_<xla::HloDomainVerifier, std::shared_ptr<xla::HloDomainVerifier>,
  // xla::HloPassInterface>(m, "HloDomainVerifier")
  //     .def(py::init<std::vector<std::string>>(), py::arg("kinds"));

  // py::class_<xla::HloElementTypeConverter,
  // std::shared_ptr<xla::HloElementTypeConverter>, xla::HloPassInterface>(m,
  // "HloElementTypeConverter")
  //     .def(py::init<PrimitiveType, PrimitiveType>(),
  //     py::arg("eliminate_type"), py::arg("replace_with_type"));

  // py::class_<xla::HloMemoryScheduler,
  // std::shared_ptr<xla::HloMemoryScheduler>, xla::HloPassInterface>(m,
  // "HloMemoryScheduler")
  //     .def(py::init<const LogicalBuffer::SizeFunction&, const
  //     ModuleSchedulerAlgorithm&>(), py::arg("size_function"),
  //     py::arg("algorithm") = {});

  // py::class_<xla::HloRematerialization,
  // std::shared_ptr<xla::HloRematerialization>, xla::HloPassInterface>(m,
  // "HloRematerialization")
  //     .def(py::init<const ShapeSizeFunction&, int64_t,
  //     RematerializationSizes*, RematerializationPass, int, int,
  //     CompactShapeFunction, RematerializationMode, int64_t>(),
  //     py::arg("size_function"), py::arg("memory_limit_bytes"),
  //     py::arg("sizes"), py::arg("pass_location"),
  //     py::arg("block_size_limit"), py::arg("block_rematerialization_factor"),
  //     py::arg("compact_shape_function") = nullptr, py::arg("mode") =
  //     RematerializationMode::kRecomputeAndCompress, py::arg("min_remat_size")
  //     = 0);

  // py::class_<xla::InstructionFusion, std::shared_ptr<xla::InstructionFusion>,
  // xla::HloPassInterface>(m, "InstructionFusion")
  //     .def(py::init<std::function<bool(const HloInstruction& instruction)>,
  //     bool, FusionConfigCollection>(), py::arg("is_expensive"),
  //     py::arg("may_duplicate") = true, py::arg("config_collection_mode") =
  //     FusionConfigCollection::kOff);

  // py::class_<xla::LayoutAssignment, std::shared_ptr<xla::LayoutAssignment>,
  // xla::HloPassInterface>(m, "LayoutAssignment")
  //     .def(py::init<ComputationLayout*, ChannelLayoutConstraints*, bool>(),
  //     py::arg("entry_computation_layout"), py::arg("channel_constraints") =
  //     nullptr, py::arg("reverse_computation_order") = false);

  // py::class_<xla::OperandsMustBeTheSameLayoutAssignment,
  // std::shared_ptr<xla::OperandsMustBeTheSameLayoutAssignment>,
  // xla::HloPassInterface>(m, "OperandsMustBeTheSameLayoutAssignment")
  //     .def(py::init<ComputationLayout*>(),
  //     py::arg("entry_computation_layout"));

  // py::class_<xla::cpu::std::shared_ptr<xla::cpu>, CpuLayoutAssignment,
  // xla::HloPassInterface>(m, "CpuLayoutAssignment")
  //     .def(py::init<ComputationLayout*, const TargetMachineFeatures*,
  //     ChannelLayoutConstraints*>(), py::arg("entry_computation_layout"),
  //     py::arg("target_machine_features"), py::arg("channel_constraints") =
  //     nullptr);

  // py::class_<xla::TriangularSolveExpander,
  // std::shared_ptr<xla::TriangularSolveExpander>, xla::HloPassInterface>(m,
  // "TriangularSolveExpander")
  //     .def(py::init<int64_t>(), py::arg("block_size") = 128);

  // py::class_<xla::SpaceToBatchConverter,
  // std::shared_ptr<xla::SpaceToBatchConverter>, xla::HloPassInterface>(m,
  // "SpaceToBatchConverter")
  //     .def(py::init<SpaceToBatchController>(), py::arg("ctrl"));

  // py::class_<xla::TopkRewriter, std::shared_ptr<xla::TopkRewriter>,
  // xla::HloPassInterface>(m, "TopkRewriter")
  //     .def(py::init<std::function<bool(const, int64_t)>>(),
  //     py::arg("HloSortInstruction*"), py::arg("is_profitable_to_convert"));

  // py::class_<xla::TreeReductionRewriter,
  // std::shared_ptr<xla::TreeReductionRewriter>, xla::HloPassInterface>(m,
  // "TreeReductionRewriter")
  //     .def(py::init<int64_t>(), py::arg("reduce_window_size") = 32);

  // py::class_<xla::WhileLoopConcatCodeMotion,
  // std::shared_ptr<xla::WhileLoopConcatCodeMotion>, xla::HloPassInterface>(m,
  // "WhileLoopConcatCodeMotion")
  //     .def(py::init<int64_t>(), py::arg("min_operand_count_to_optimize"));

  // py::class_<xla::WhileLoopExpensiveInvariantCodeMotion,
  //      std::shared_ptr<xla::WhileLoopExpensiveInvariantCodeMotion>,
  //      xla::HloPassInterface>(m, "WhileLoopExpensiveInvariantCodeMotion")
  //     .def(py::init<std::function<bool(const HloInstruction&)>,
  //     ShapeSizeFunction>(), py::arg("worth_hoisting_individually"),
  //     py::arg("shape_size_function") = ShapeUtil::ByteSizeOfElements);

  // py::class_<xla::WhileLoopInvariantCodeMotion,
  // std::shared_ptr<xla::WhileLoopInvariantCodeMotion>,
  // xla::HloPassInterface>(m, "WhileLoopInvariantCodeMotion")
  //     .def(py::init<bool, bool, bool, absl::optional<float>,
  //     ShapeSizeFunction>(), py::arg("hoist_constants") = false,
  //     py::arg("hoist_reshapes") = false, py::arg("hoist_other") = true,
  //     py::arg("hoist_size_inflation_ratio") = absl::nullopt,
  //     py::arg("shape_size_function") = ShapeUtil::ByteSizeOfElements);

  // py::class_<xla::cpu::std::shared_ptr<xla::cpu>, ConvCanonicalization,
  // xla::HloPassInterface>(m, "ConvCanonicalization")
  //     .def(py::init<const TargetMachineFeatures*>(),
  //     py::arg("target_machine_features"));

  // py::class_<xla::cpu::std::shared_ptr<xla::cpu>, ParallelTaskAssigner,
  // xla::HloPassInterface>(m, "ParallelTaskAssigner")
  //     .def(py::init<const int64_t, const HloCostAnalysis::ShapeSizeFunction&,
  //     const TargetMachineFeatures*>(), py::arg("max_parallelism"),
  //     py::arg("shape_size"), py::arg("target_machine_features"));

  // py::class_<xla::spmd::std::shared_ptr<xla::spmd>,
  // ScheduleAwareCollectiveOpsCSE, xla::HloPassInterface>(m,
  // "ScheduleAwareCollectiveOpsCSE")
  //     .def(py::init<int64_t, bool>(), py::arg("distance_threshold"),
  //     py::arg("for_replicas"));

  // py::class_<xla::spmd::std::shared_ptr<xla::spmd>, SpmdPartitioner,
  // xla::HloPassInterface>(m, "SpmdPartitioner")
  //     .def(py::init<int64_t, int64_t, SpmdPartitionerOptions>(),
  //     py::arg("num_partitions"), py::arg("num_replicas"), py::arg("options"))
  //     .def(py::init<int64_t, int64_t, SpmdPartitionerOptions,
  //     SPMDCollectiveOpsCreator>(), py::arg("num_partitions"),
  //     py::arg("num_replicas"), py::arg("options"),
  //     py::arg("collective_ops_creator"));

  // py::class_<xla::HloControlFlowFlattening,
  // std::shared_ptr<xla::HloControlFlowFlattening>, xla::HloPassInterface>(m,
  // "HloControlFlowFlattening")
  //     .def(py::init<const Options&>(), py::arg("options"));
}  // NOLINT(readability/fn_size)

}  // namespace hloenv
