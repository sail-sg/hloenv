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

#ifndef HLOENV_HLO_PASS_DEFS_H_
#define HLOENV_HLO_PASS_DEFS_H_

// -------------------------------------
//   GPU PASSES
// -------------------------------------
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"
#include "tensorflow/compiler/xla/service/all_gather_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_combiner.h"
#include "tensorflow/compiler/xla/service/all_reduce_contiguous.h"
#include "tensorflow/compiler/xla/service/all_reduce_folder.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"
#include "tensorflow/compiler/xla/service/async_collective_creator.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/collectives_schedule_linearizer.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_4d_expander.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dot_merger.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gpu/all_reduce_blueconnect.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_pad_for_gemms.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_pad_for_convolutions.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_vectorize_convolutions.h"
#include "tensorflow/compiler/xla/service/gpu/cusolver_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_bitcast_lift.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_broadcast_folding_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/general_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_reduce_scatter_creator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_scatter_expander.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_input_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_splitter.h"
#include "tensorflow/compiler/xla/service/gpu/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/variadic_op_splitter.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/logistic_expander.h"
#include "tensorflow/compiler/xla/service/operand_upcaster.h"
#include "tensorflow/compiler/xla/service/optimization_barrier_expander.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/real_imag_expander.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_combiner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/stable_sort_expander.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"

// // -------------------------------------
// //   OTHERS - ARG INCLUDES
// // -------------------------------------
// #include "tensorflow/compiler/xla/service/all_gather_decomposer.h"
// #include "tensorflow/compiler/xla/service/all_reduce_simplifier.h"
// #include "tensorflow/compiler/xla/service/ar_crs_combiner.h"
// #include "tensorflow/compiler/xla/service/bfloat16_propagation.h"
// #include "tensorflow/compiler/xla/service/conditional_code_motion.h"
// #include "tensorflow/compiler/xla/service/convolution_group_converter.h"
// #include "tensorflow/compiler/xla/service/copy_insertion.h"
// #include "tensorflow/compiler/xla/service/hlo_domain_isolator.h"
// #include "tensorflow/compiler/xla/service/hlo_domain_remover.h"
// #include "tensorflow/compiler/xla/service/hlo_domain_verifier.h"
// #include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
// #include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
// #include "tensorflow/compiler/xla/service/hlo_rematerialization.h"
// #include "tensorflow/compiler/xla/service/instruction_fusion.h"
// #include "tensorflow/compiler/xla/service/layout_assignment.h"
// #include "tensorflow/compiler/xla/service/layout_assignment_test.cc"
// #include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"
// #include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
// #include "tensorflow/compiler/xla/service/space_to_batch_converter.h"
// #include "tensorflow/compiler/xla/service/topk_rewriter.h"
// #include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"
// #include "tensorflow/compiler/xla/service/while_loop_concat_code_motion.h"
// #include
// "tensorflow/compiler/xla/service/while_loop_expensive_invariant_code_motion.h"
// #include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
// #include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
// #include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
// #include
// "tensorflow/compiler/xla/service/spmd/schedule_aware_collective_ops_cse.h"
// #include "tensorflow/compiler/xla/service/spmd/spmd_partitioner.h"
// // -------------------------------------
// //   OTHERS - EMPTY INCLUDES
// // -------------------------------------
// #include "tensorflow/compiler/xla/service/all_gather_decomposer.h"
// #include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
// #include "tensorflow/compiler/xla/service/bfloat16_conversion_folding.h"
// #include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
// #include "tensorflow/compiler/xla/service/conditional_to_select.h"
// #include "tensorflow/compiler/xla/service/copy_insertion.h"
// #include "tensorflow/compiler/xla/service/defuser.h"
// #include "tensorflow/compiler/xla/service/despecializer.h"
// #include "tensorflow/compiler/xla/service/despecializer.h"
// #include "tensorflow/compiler/xla/service/dry_mode.h"
// #include "tensorflow/compiler/xla/service/dry_mode.h"
// #include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
// #include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
// #include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
// #include "tensorflow/compiler/xla/service/hlo_module_dce.h"
// #include "tensorflow/compiler/xla/service/hlo_pass_pipeline_test.cc"
// #include "tensorflow/compiler/xla/service/hlo_pass_pipeline_test.cc"
// #include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
// #include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
// #include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
// #include "tensorflow/compiler/xla/service/instruction_fusion_test.cc"
// #include "tensorflow/compiler/xla/service/loop_schedule_linearizer.h"
// #include "tensorflow/compiler/xla/service/map_inliner.h"
// #include "tensorflow/compiler/xla/service/memory_space_propagation.h"
// #include "tensorflow/compiler/xla/service/multi_output_fusion.h"
// #include "tensorflow/compiler/xla/service/op_expander_pass.h"
// #include "tensorflow/compiler/xla/service/cholesky_expander.h"
// #include "tensorflow/compiler/xla/service/convert_operand_folding.h"
// #include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
// #include
// "tensorflow/compiler/xla/service/optimize_input_output_buffer_alias.h"
// #include "tensorflow/compiler/xla/service/reduce_scatter_decomposer.h"
// #include "tensorflow/compiler/xla/service/root_instruction_sinker.h"
// #include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"
// #include
// "tensorflow/compiler/xla/service/while_loop_all_reduce_code_motion.h"
// #include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
// #include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"
// #include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
// #include
// "tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.h"

#endif  // HLOENV_HLO_PASS_DEFS_H_
