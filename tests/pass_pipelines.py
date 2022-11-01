# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod, abstractproperty
from hloenv import HloEnv, HloModule, Pass, HloPass, AltPipeline, Pipeline, GpuBackend


class PassPipelines(ABC):

  @abstractproperty
  def pre_pass_optimizations(self):
    pass

  @abstractproperty
  def post_pass_optimizations(self):
    pass

  @abstractproperty
  def pre_dry_pass_passes(self):
    pass

  @abstractproperty
  def post_dry_pass_passes(self):
    pass

  @abstractproperty
  def xla_passes(self):
    pass

  @abstractproperty
  def pass_dry_run(self):
    pass


class SingleFusionPipeline(PassPipelines):

  def __init__(self, hlo_ir: HloEnv) -> None:
    self._hlo_ir = hlo_ir

    hlo_module = self._hlo_ir.get_hlo_module()
    config = hlo_module.config
    debug_options = config.debug_options

    self.pre_fusion_optimizations = Pipeline("pre-fusion-optimizations")

    optimization_pipeline = Pipeline("optimization")
    optimization_pipeline.add_pass(HloPass.AllToAllDecomposer())
    optimization_pipeline.add_pass(HloPass.OperandUpcaster())
    optimization_pipeline.add_pass(HloPass.ResultCaster())
    optimization_pipeline.add_pass(HloPass.RngExpander())
    optimization_pipeline.add_pass(
      HloPass.RngBitGeneratorExpander(
        HloPass.RngBitGeneratorExpander.RandomAlgorithm.RNG_PHILOX
      )
    )
    optimization_pipeline.add_pass(HloPass.ComparisonExpander())
    optimization_pipeline.add_pass(HloPass.ZeroSizedHloElimination())

    if debug_options.xla_gpu_deterministic_ops:
      optimization_pipeline.add_pass(
        HloPass.ScatterExpander(
          HloPass.ScatterExpander.Mode.kEliminateAllScatters
        )
      )
    else:
      optimization_pipeline.add_pass(HloPass.GpuScatterExpander())

    optimization_pipeline.add_pass(HloPass.QrExpander())
    optimization_pipeline.add_pass(HloPass.EighExpander())
    optimization_pipeline.add_pass(HloPass.DynamicIndexSplitter())
    optimization_pipeline.add_pass(HloPass.CallInliner())
    optimization_pipeline.add_pass(HloPass.DotDecomposer())
    optimization_pipeline.add_pass(HloPass.Convolution4DExpander())
    optimization_pipeline.add_pass(HloPass.StableSortExpander())

    optimization_pipeline.add_pass(HloPass.BFloat16Normalization(True))
    optimization_pipeline.add_pass(HloPass.BatchNormExpander(True, True, True))
    optimization_pipeline.add_pass(
      HloPass.LogisticExpander(
        HloPass.LogisticExpander.LogisticExpansionType.kExp
      )
    )
    optimization_pipeline.add_pass(HloPass.ConditionalCanonicalizer())
    optimization_pipeline.add_pass(HloPass.DynamicDimensionSimplifier())
    dp_options = {
      "shape_check_mode": HloPass.DynamicPadder.ShapeCheckMode.kCompileTime
    }
    optimization_pipeline.add_pass(HloPass.DynamicPadder(dp_options))

    # ***********************************************************************

    simplification_pipeline = Pipeline("simplification", loop_count=-1)
    simplification_pipeline.add_pass(HloPass.ZeroSizedHloElimination())
    simplification_pipeline.add_pass(
      HloPass.GatherExpander(
        HloPass.GatherExpander.Mode.kEliminateSimpleGathers
      )
    )
    simplification_pipeline.add_pass(
      HloPass.ScatterExpander(
        HloPass.ScatterExpander.Mode.kEliminateSimpleScatters
      )
    )
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    if (GpuBackend.stream_exec_platform == "ROCM"):
      algebraic_config_options["enable_conv_operand_swap"] = False

    simplification_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options)
    )

    simplification_pipeline.add_pass(HloPass.BitcastDtypesExpander())
    simplification_pipeline.add_pass(HloPass.DotDecomposer())
    simplification_pipeline.add_pass(
      HloPass.DotMerger(max_size_to_merge=16 << 20)
    )
    simplification_pipeline.add_pass(HloPass.SortSimplifier())
    simplification_pipeline.add_pass(HloPass.TupleSimplifier())
    simplification_pipeline.add_pass(HloPass.WhileLoopConstantSinking())
    simplification_pipeline.add_pass(HloPass.WhileLoopSimplifier())
    simplification_pipeline.add_pass(HloPass.ReshapeMover())
    simplification_pipeline.add_pass(HloPass.HloConstantFolding())
    simplification_pipeline.add_pass(HloPass.ConditionalSimplifier())
    simplification_pipeline.add_pass(HloPass.RealImagExpander())
    simplification_pipeline.add_pass(HloPass.TransposeFolding())
    simplification_pipeline.add_pass(HloPass.HloCSE(is_layout_sensitive=False))
    simplification_pipeline.add_pass(HloPass.HloDCE())
    optimization_pipeline.add_pass(simplification_pipeline)

    # Run WhileLoopTripCountAnnotator at the end of the simplification
    # pipeline, before layout assignment and fusion.  This pass does some
    # pattern-matching on while bodies/conditions, and this is where the HLO is
    # "nicest".
    #
    # It's important that we don't make semantic changes (e.g. unrolling) to
    # any `while` loops after this point, because otherwise the trip-count
    # annotations added by this pass may not be correct after the
    # modifications.
    optimization_pipeline.add_pass(HloPass.WhileLoopTripCountAnnotator())
    # pre_fusion_pipeline.add_pass(optmiization_pipeline)
    self.pre_fusion_optimizations.add_pass(optimization_pipeline)

    # --------------------------------------------
    # Collectives Pipeline
    # --------------------------------------------

    collectives_pipeline = Pipeline("collective-optimizations")
    collectives_pipeline.add_pass(HloPass.AllReduceFolder())
    collectives_pipeline.add_pass(HloPass.ReduceScatterCreator())
    collectives_pipeline.add_pass(HloPass.AllReduceReassociate())
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    collectives_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options)
    )
    collectives_pipeline.add_pass(HloPass.AllGatherBroadcastReorder())
    self.pre_fusion_optimizations.add_pass(collectives_pipeline)

    # --------------------------------------------
    # Convolution Canonicalization Pipeline
    # --------------------------------------------

    # TODO(ohcy): Account for AMD GPU case
    # Note, this is specific to Nvidia GPUs. For AMD GPUs, some of the passes,
    # e.g. Cudnn passes should be excluded
    conv_canon_pipeline = Pipeline("conv-canonicalization")
    conv_canon_pipeline.add_pass(HloPass.GpusolverRewriter())
    conv_canon_pipeline.add_pass(HloPass.GpuConvRewriter())
    conv_canon_pipeline.add_pass(HloPass.CudnnFusedConvRewriter())
    conv_canon_pipeline.add_pass(HloPass.GpuConvPaddingLegalization())
    conv_canon_pipeline.add_pass(HloPass.CudnnPadForConvolutions())
    conv_canon_pipeline.add_pass(HloPass.CudnnVectorizeConvolutions())
    conv_canon_pipeline.add_pass(HloPass.CallInliner())
    conv_canon_pipeline.add_pass(HloPass.TupleSimplifier())
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
    }
    conv_canon_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options),
      loop_count=-1
    )
    conv_canon_pipeline.add_pass(HloPass.HloConstantFolding())
    self.pre_fusion_optimizations.add_pass(conv_canon_pipeline)

    # --------------------------------------------
    # Layout Assignment Pipeline
    # --------------------------------------------

    layout_assignment_pipeline = Pipeline("layout-assignment")
    layout_assignment_pipeline.add_pass(HloPass.FlattenCallGraph())
    layout_assignment_pipeline.add_pass(
      HloPass.GpuLayoutAssignment(hlo_module)
    )
    self.pre_fusion_optimizations.add_pass(layout_assignment_pipeline)

    # --------------------------------------------
    # Post Layout Assignment Pipeline
    # --------------------------------------------

    # *******************
    # NVIDIA GPU Specific Passes Stage 1 - START
    post_layout_ass_pipeline_nv_pre = Pipeline("post-layout-assignment-nv-pre")
    if (GpuBackend.cuda_is_at_least(GpuBackend.CudaComputeCapability.AMPERE)):
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.BF16,
          pad_to_multiple_of=8
        )
      )

    if (GpuBackend.cuda_is_at_least(GpuBackend.CudaComputeCapability.VOLTA)):
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.S8,
          pad_to_multiple_of=4
        )
      )
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.F16,
          pad_to_multiple_of=8
        )
      )

    post_layout_ass_pipeline_nv_pre.add_pass(HloPass.HloConstantFolding())
    # NVIDIA GPU Specific Passes Stage 1 - END
    # *******************

    post_layout_ass_pipeline = Pipeline("post-layout-assignment")
    post_layout_ass_pipeline.add_pass(HloPass.ReductionDegenerateDimRemover())
    post_layout_ass_pipeline.add_pass(HloPass.ReductionLayoutNormalizer())
    post_layout_ass_pipeline.add_pass(HloPass.ReductionDimensionGrouper())
    post_layout_ass_pipeline.add_pass(
      HloPass.ReductionSplitter(), loop_count=-1
    )
    post_layout_ass_pipeline.add_pass(
      HloPass.GpuTreeReductionRewriter(), loop_count=-1
    )

    algebraic_config_options = {
      "is_layout_sensitive": True,
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    post_layout_ass_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(algebraic_config_options), loop_count=-1
    )
    post_layout_ass_pipeline.add_pass(HloPass.TransposeFolding())
    post_layout_ass_pipeline.add_pass(HloPass.GemmRewriter())
    post_layout_ass_pipeline.add_pass(HloPass.GemmBroadcastFoldingRewriter())
    post_layout_ass_pipeline.add_pass(HloPass.BFloat16Normalization(False))
    post_layout_ass_pipeline.add_pass(HloPass.GpuConvAlgorithmPicker())
    post_layout_ass_pipeline.add_pass(HloPass.TupleSimplifier())
    post_layout_ass_pipeline.add_pass(HloPass.HloCSE(True))

    # *******************
    # NVIDIA GPU Specific Passes Stage 2 - START
    post_layout_ass_pipeline_nv_post = Pipeline(
      "post-layout-assignment-nv-post"
    )

    post_layout_ass_pipeline_nv_post.add_pass(HloPass.GemmAlgorithmPicker())
    if (hlo_module.is_bef_enabled):
      post_layout_ass_pipeline_nv_post.add_pass(
        HloPass.TriangularSolveRewriter()
      )
    # NVIDIA GPU Specific Passes Stage 2 - END
    # *******************

    self.pre_fusion_optimizations.add_pass(post_layout_ass_pipeline_nv_pre)
    self.pre_fusion_optimizations.add_pass(post_layout_ass_pipeline)
    self.pre_fusion_optimizations.add_pass(post_layout_ass_pipeline_nv_post)

    #  -----------------------------------------------------------------------
    #                               FUSION PIPELINE
    #  -----------------------------------------------------------------------

    # --------------------------------------------
    # Original Vertical Fusion Pipeline
    # --------------------------------------------

    self.vert_fusion_pipeline = Pipeline("vertical-fusion", loop_count=-1)
    self.vert_fusion_pipeline.add_pass(HloPass.VariadicOpSplitter())
    self.vert_fusion_pipeline.add_pass(HloPass.GpuInstructionFusion(False))
    self.vert_fusion_pipeline.add_pass(HloPass.GpuInstructionFusion(True))
    self.vert_fusion_pipeline.add_pass(HloPass.FusionMerger())
    self.vert_fusion_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    self.vert_fusion_pipeline.add_pass(HloPass.HloCSE(True, True))
    self.vert_fusion_pipeline.add_pass(HloPass.HloDCE())

    # --------------------------------------------
    # Vertical Fusion Pipeline
    # --------------------------------------------

    self.fusion_pipeline_pre = Pipeline("fusion-pre")
    self.fusion_pipeline_pre.add_pass(HloPass.VariadicOpSplitter())

    self.fusion_dry_pass = AltPipeline(
      Pass(
        HloPass.GpuInstructionFusion(True),  # may_duplicate
      )
    )

    self.fusion_pipeline_post = Pipeline("fusion-post")
    self.fusion_pipeline_post.add_pass(HloPass.FusionMerger())
    self.fusion_pipeline_post.add_pass(HloPass.GpuMultiOutputFusion())
    self.fusion_pipeline_post.add_pass(HloPass.HloCSE(True, True))
    self.fusion_pipeline_post.add_pass(HloPass.HloDCE())

    # --------------------------------------------
    # Horizontal Fusion Pipeline
    # --------------------------------------------
    self.post_fusion_optimizations = Pipeline("post-fusion-optimizations")

    hori_fusion_pipeline = Pipeline("horizontal-fusion", loop_count=-1)
    hori_fusion_pipeline.add_pass(HloPass.GpuHorizontalLoopFusion())
    hori_fusion_pipeline.add_pass(HloPass.GpuHorizontalInputFusion())
    hori_fusion_pipeline.add_pass(HloPass.HloCSE(True, True))
    hori_fusion_pipeline.add_pass(HloPass.HloDCE())

    self.post_fusion_optimizations.add_pass(hori_fusion_pipeline)
    #  -----------------------------------------------------------------------
    #                               POST PIPELINE
    #  -----------------------------------------------------------------------

    post_fusion_pipeline = Pipeline("post-fusion")

    post_fusion_pipeline.add_pass(
      HloPass.AllGatherCombiner(
        combine_threshold_in_bytes=1024 * 1024 * 1024,
        combine_threshold_count=256
      )
    )
    post_fusion_pipeline.add_pass(
      HloPass.AllReduceCombiner(
        combine_threshold_in_bytes=debug_options
        .xla_gpu_all_reduce_combine_threshold_bytes,
        combine_threshold_count=256
      )
    )
    post_fusion_pipeline.add_pass(
      HloPass.ReduceScatterCombiner(
        combine_threshold_in_bytes=30 * 1024 * 1024,
        combine_threshold_count=256
      )
    )

    if debug_options.xla_gpu_all_reduce_contiguous:
      post_fusion_pipeline.add_pass(HloPass.AllReduceContiguous())

    blueconnect_num_devices_per_host = debug_options.xla_gpu_all_reduce_blueconnect_num_devices_per_host
    if (blueconnect_num_devices_per_host > 0):
      post_fusion_pipeline.add_pass(
        HloPass.AllReduceBlueConnect(blueconnect_num_devices_per_host)
      )

    if debug_options.xla_gpu_enable_async_all_reduce:
      post_fusion_pipeline.add_pass(HloPass.AsyncCollectiveCreator())

    post_fusion_pipeline.add_pass(HloPass.CollectivesScheduleLinearizer())
    algebraic_config_options = {
      "is_layout_sensitive": True,
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    post_fusion_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(algebraic_config_options)
    )
    post_fusion_pipeline.add_pass(HloPass.OptimizationBarrierExpander())
    post_fusion_pipeline.add_pass(HloPass.TupleSimplifier())

    self.post_fusion_optimizations.add_pass(post_fusion_pipeline)

    self.xla_pipeline = Pipeline("xla-pipeline")
    self.xla_pipeline.add_pass(self.pre_fusion_optimizations)
    self.xla_pipeline.add_pass(self.vert_fusion_pipeline)
    self.xla_pipeline.add_pass(self.post_fusion_optimizations)
    # -------------------------------------------

  @property
  def pre_pass_optimizations(self):
    return self.pre_fusion_optimizations

  @property
  def post_pass_optimizations(self):
    return self.post_fusion_optimizations

  @property
  def pre_dry_pass_passes(self):
    return self.fusion_pipeline_pre

  @property
  def post_dry_pass_passes(self):
    return self.fusion_pipeline_post

  @property
  def xla_passes(self):
    return self.xla_pipeline

  @property
  def pass_dry_run(self):
    return self.fusion_dry_pass


class GeneralFusionPipeline(SingleFusionPipeline):

  def __init__(self, hlo_ir: HloEnv) -> None:
    super(GeneralFusionPipeline, self).__init__(hlo_ir)

    self.fusion_dry_pass = AltPipeline(Pass(HloPass.GeneralFusion(),))
    self.fusion_pipeline_post = Pipeline("post-general-fusion")
    # self.fusion_pipeline_post.add_pass(HloPass.GpuMultiOutputFusion())
    self.fusion_pipeline_post.add_pass(HloPass.HloCSE(True, True))
    self.fusion_pipeline_post.add_pass(HloPass.HloDCE())
