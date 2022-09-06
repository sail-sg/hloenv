// Copyright 2021 Garena Online Private Limited

#include "altgraph/utils/hlo_utils.h"

#include <unordered_set>

namespace xla {
void GetInstructionAttributesAndCounts(HloInstruction* inst,
                                       std::vector<int>* attrs,
                                       std::vector<int>* attr_counts) {
  auto add_enum = [&attrs](int item, int count) {
    std::vector<int> one_hot(count, 0);
    one_hot[item] = 1;
    attrs->insert(attrs->end(), one_hot.begin(), one_hot.end());
  };
  auto add_attr_counts = [&attr_counts](int int_count, int enum_count) {
    attr_counts->push_back(int_count);
    attr_counts->push_back(enum_count);
  };
  attrs->clear();
  attr_counts->clear();
  // We cap the max possible number of dim at 6.
  // Covers 99.99% time.
  const int kInvalidDim = 6;
  // set dim enum size to be 6 + 1 invalid dim
  const int kDimEnumSize = 7;
  switch (inst->opcode()) {
    // dimension indices
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose: {
      // counts: 0,6
      // pad to six to keep a fixed size
      for (int i = 0; i < std::min<int>(6, inst->dimensions().size()); ++i) {
        add_enum(/*item=*/inst->dimensions()[i], /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < std::max<int>(0, (6 - inst->dimensions().size()));
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_attr_counts(/*int_count=*/0, /*enum_count=*/6 * 7);
      break;
    }
    case HloOpcode::kCompare: {
      // counts: 0,2
      auto comp_inst = dynamic_cast<HloCompareInstruction*>(inst);
      add_enum(/*item=*/static_cast<int>(comp_inst->direction()), /*count=*/6);
      add_enum(/*item=*/static_cast<int>(comp_inst->type()), /*count=*/4);
      add_attr_counts(/*int_count=*/0, /*enum_count=*/6 + 4);
      break;
    }
    case HloOpcode::kConvolution: {
      // counts: 16,24
      auto conv_inst = dynamic_cast<HloConvolutionInstruction*>(inst);
      // window
      auto w_dims = conv_inst->window().dimensions();
      if (w_dims.size() == 2) {
        // greped all conv in current hlo dataset
        // boldly assume only one 2D window for convolution
        // to make sure feature size for kConv is fixed.
        for (int d = 0; d < w_dims.size(); ++d) {
          attrs->push_back(w_dims[d].size());
          attrs->push_back(w_dims[d].stride());
          attrs->push_back(w_dims[d].padding_low());
          attrs->push_back(w_dims[d].padding_high());
          attrs->push_back(w_dims[d].base_dilation());
          attrs->push_back(w_dims[d].window_dilation());
          attrs->push_back(w_dims[d].window_reversal());
        }
      } else {
        attrs->resize(14, 0);
      }
      // feature group count and batch group count
      attrs->push_back(inst->feature_group_count());
      attrs->push_back(inst->batch_group_count());
      // dim_labels
      auto conv_dims = conv_inst->convolution_dimension_numbers();
      add_enum(/*item=*/conv_dims.input_batch_dimension(),
               /*count=*/kDimEnumSize);
      add_enum(/*item=*/conv_dims.input_feature_dimension(),
               /*count=*/kDimEnumSize);
      for (auto input_spatial_dim : conv_dims.input_spatial_dimensions()) {
        add_enum(/*item=*/input_spatial_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - conv_dims.input_spatial_dimensions().size();
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_enum(/*item=*/conv_dims.kernel_input_feature_dimension(),
               /*count=*/kDimEnumSize);
      add_enum(/*item=*/conv_dims.kernel_output_feature_dimension(),
               /*count=*/kDimEnumSize);
      for (auto kernel_spatial_dim : conv_dims.kernel_spatial_dimensions()) {
        add_enum(/*item=*/kernel_spatial_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - conv_dims.kernel_spatial_dimensions().size();
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_enum(/*item=*/conv_dims.output_batch_dimension(),
               /*count=*/kDimEnumSize);
      add_enum(/*item=*/conv_dims.output_feature_dimension(),
               /*count=*/kDimEnumSize);
      for (auto output_spatial_dim : conv_dims.output_spatial_dimensions()) {
        add_enum(/*item=*/output_spatial_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - conv_dims.output_spatial_dimensions().size();
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_attr_counts(/*int_count=*/16, /*enum_count=*/24 * 7);
      break;
    }
    case HloOpcode::kDot: {
      // counts: 0,25
      auto dot_inst = dynamic_cast<HloDotInstruction*>(inst);
      auto dot_dims = dot_inst->dot_dimension_numbers();
      for (auto lhs_batch_dim : dot_dims.lhs_batch_dimensions()) {
        add_enum(/*item=*/lhs_batch_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - dot_dims.lhs_batch_dimensions().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      for (auto lhs_c_dim : dot_dims.lhs_contracting_dimensions()) {
        add_enum(/*item=*/lhs_c_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - dot_dims.lhs_contracting_dimensions().size();
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      for (auto rhs_batch_dim : dot_dims.rhs_batch_dimensions()) {
        add_enum(/*item=*/rhs_batch_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - dot_dims.rhs_batch_dimensions().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      for (auto rhs_c_dim : dot_dims.rhs_contracting_dimensions()) {
        add_enum(/*item=*/rhs_c_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - dot_dims.rhs_contracting_dimensions().size();
           ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      // we force to take only one precision config. 0 is default, 1 and 2 are
      // high and highest separately.
      auto operand_precision = dot_inst->precision_config().operand_precision();
      if (operand_precision.size() > 0) {
        add_enum(/*item=*/static_cast<int>(operand_precision[0]), /*count=*/3);
      } else {
        add_enum(/*item=*/0, /*count=*/3);
      }
      add_attr_counts(/*int_count=*/0, /*enum_count=*/24 * 7 + 3);
      break;
    }
    case HloOpcode::kDynamicSlice: {
      // counts: 6,0
      auto ds_inst = dynamic_cast<HloDynamicSliceInstruction*>(inst);
      attrs->resize(6, -1);
      int idx = 0;
      for (auto d : ds_inst->dynamic_slice_sizes()) {
        attrs->at(idx++) = d;
      }
      add_attr_counts(/*int_count=*/6, /*enum_count=*/0);
      break;
    }
    case HloOpcode::kGather: {
      // counts: 7,19
      auto gather_inst = dynamic_cast<HloGatherInstruction*>(inst);
      auto gather_dims = gather_inst->gather_dimension_numbers();
      attrs->resize(7, -1);
      int idx = 0;
      attrs->at(idx++) = gather_dims.index_vector_dim();
      for (auto d : gather_inst->gather_slice_sizes()) {
        attrs->at(idx++) = d;
      }
      for (auto offset_dim : gather_dims.offset_dims()) {
        add_enum(/*item=*/offset_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - gather_dims.offset_dims().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }

      for (auto cs_dim : gather_dims.collapsed_slice_dims()) {
        add_enum(/*item=*/cs_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - gather_dims.collapsed_slice_dims().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }

      for (auto s_dim : gather_dims.start_index_map()) {
        add_enum(/*item=*/s_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - gather_dims.start_index_map().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_enum(/*item=*/gather_inst->indices_are_sorted(), /*count=*/2);
      add_attr_counts(/*int_count=*/7, /*enum_count=*/18 * 7 + 2);
      break;
    }
    case HloOpcode::kGetTupleElement: {
      // counts: 1,0
      auto gte_inst = dynamic_cast<HloGetTupleElementInstruction*>(inst);
      attrs->push_back(gte_inst->tuple_index());
      add_attr_counts(/*int_count=*/1, /*enum_count=*/0);
      break;
    }
    case HloOpcode::kIota: {
      // counts: 0,1
      auto iota_inst = dynamic_cast<HloIotaInstruction*>(inst);
      add_enum(/*item=*/std::min<int>(6, iota_inst->iota_dimension()),
               /*count=*/kDimEnumSize);
      add_attr_counts(/*int_count=*/0, /*enum_count=*/7);
      break;
    }
    case HloOpcode::kPad: {
      // counts: 18,0
      auto pad_inst = dynamic_cast<HloPadInstruction*>(inst);
      auto pad_dims = pad_inst->padding_config().dimensions();
      for (auto dim : pad_dims) {
        attrs->push_back(dim.edge_padding_low());
        attrs->push_back(dim.edge_padding_high());
        attrs->push_back(dim.interior_padding());
      }
      for (int i = 0; i < 6 - pad_dims.size(); ++i) {
        attrs->push_back(-1);
        attrs->push_back(-1);
        attrs->push_back(-1);
      }
      add_attr_counts(/*int_count=*/18, /*enum_count=*/0);
      break;
    }
    case HloOpcode::kScatter: {
      // counts: 1,20
      auto sc_inst = dynamic_cast<HloScatterInstruction*>(inst);
      auto scatter_dims = sc_inst->scatter_dimension_numbers();
      attrs->push_back(scatter_dims.index_vector_dim());
      for (auto uw_dim : scatter_dims.update_window_dims()) {
        add_enum(/*item=*/uw_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - scatter_dims.update_window_dims().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      for (auto iw_dim : scatter_dims.inserted_window_dims()) {
        add_enum(/*item=*/iw_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - scatter_dims.inserted_window_dims().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      for (auto sd2od_dim : scatter_dims.scatter_dims_to_operand_dims()) {
        add_enum(/*item=*/sd2od_dim, /*count=*/kDimEnumSize);
      }
      for (int i = 0;
           i < 6 - scatter_dims.scatter_dims_to_operand_dims().size(); ++i) {
        add_enum(/*item=*/kInvalidDim, /*count=*/kDimEnumSize);
      }
      add_enum(/*item=*/sc_inst->indices_are_sorted(), /*count=*/2);
      add_enum(/*item=*/sc_inst->unique_indices(), /*count=*/2);
      add_attr_counts(/*int_count=*/1, /*enum_count=*/18 * 7 + 2 * 2);
      break;
    }
    case HloOpcode::kSlice: {
      // counts: 18,0
      auto slice_inst = dynamic_cast<HloSliceInstruction*>(inst);
      auto slice_starts = slice_inst->slice_starts();
      auto slice_limits = slice_inst->slice_limits();
      auto slice_strides = slice_inst->slice_strides();
      for (int i = 0; i < slice_starts.size(); ++i) {
        attrs->push_back(slice_starts[i]);
        attrs->push_back(slice_limits[i]);
        attrs->push_back(slice_strides[i]);
      }
      for (int i = 0; i < 6 - slice_starts.size(); ++i) {
        attrs->push_back(-1);
        attrs->push_back(-1);
        attrs->push_back(-1);
      }
      add_attr_counts(/*int_count=*/18, /*enum_count=*/0);
      break;
    }
    case HloOpcode::kSort: {
      // counts: 0,2
      auto sort_inst = dynamic_cast<HloSortInstruction*>(inst);
      add_enum(/*item=*/sort_inst->sort_dimension(), /*count=*/kDimEnumSize);
      add_enum(/*item=*/sort_inst->is_stable(), /*count=*/2);
      add_attr_counts(/*int_count=*/0, /*enum_count=*/7 + 2);
      break;
    }
    case HloOpcode::kTriangularSolve: {
      // counts: 0,4
      auto tri_inst = dynamic_cast<HloTriangularSolveInstruction*>(inst);
      auto trisol_options = tri_inst->triangular_solve_options();
      add_enum(/*item=*/trisol_options.left_side(), /*count=*/2);
      add_enum(/*item=*/trisol_options.lower(), /*count=*/2);
      add_enum(/*item=*/trisol_options.unit_diagonal(), /*count=*/2);
      add_enum(/*item=*/static_cast<int>(trisol_options.transpose_a()),
               /*count=*/4);
      add_attr_counts(/*int_count=*/0, /*enum_count=*/2 * 3 + 4);
      break;
    }
    case HloOpcode::kCustomCall: {
      // counts: 0,1
      auto cc_inst = dynamic_cast<HloCustomCallInstruction*>(inst);
      auto cc_target = cc_inst->custom_call_target();
      CustomCallTargetType cc_target_enum =
          CustomCallTargetStringToEnum(cc_target);
      add_enum(/*item=*/cc_target_enum, /*count=*/13);
      add_attr_counts(/*int_count=*/0, /*enum_count=*/1 * 13);
      break;
    }
    default: {
      // counts: 0,0
      add_attr_counts(/*int_count=*/0, /*enum_count=*/0);
      break;
    }
  }
}

bool isCommutative(xla::HloInstruction* inst) {
  using xla::HloOpcode;
  switch (inst->opcode()) {
    // nullary ops
    case HloOpcode::kConstant:
    case HloOpcode::kIota:
    case HloOpcode::kParameter:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
    case HloOpcode::kRngGetAndUpdateState:
    // unary ops
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCholesky:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCos:
    case HloOpcode::kDomain:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFloor:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSendDone:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh:
    case HloOpcode::kTrace:
    case HloOpcode::kTranspose:
    case HloOpcode::kWhile:
      return true;

    // COMMUTATIVE:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kDot:
    case HloOpcode::kSort:
    case HloOpcode::kAlternatives:
      return true;

    // NOT COMMUTATIVE:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSubtract:
      return false;

    // UNSURE:
    // TODO(ohcy): Confirm whether these are not commutative
    // binary ops
    case HloOpcode::kAddDependency:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvolution:
    case HloOpcode::kGather:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSend:
    case HloOpcode::kTriangularSolve:
    // ternary ops
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kClamp:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kTupleSelect:
    // quinary ops
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    // variadic ops
    case HloOpcode::kAfterAll:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kCall:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFusion:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kRng:
    case HloOpcode::kTuple:
      return false;
    default:
      // We should not get here
      LOG(FATAL) << "Unrecognized Hlo Instruction: " << inst->opcode();
  }
  return false;
}

uint64_t HloModuleHash(xla::HloModule* module) {
  HloModuleHashWrapper hash_wrapper = HloModuleHashWrapper(module);
  return absl::HashOf(hash_wrapper);
}

// Note that calling FindInstruction() on a large/full hlo module is not
// the intended way to use this function.
HloInstruction* FindInstruction(HloModule* module, HloOpcode opcode) {
  for (const HloComputation* c : module->computations()) {
    auto instructions = c->instructions();
    auto it = absl::c_find_if(
        instructions, [&](HloInstruction* i) { return i->opcode() == opcode; });
    if (it != instructions.end()) {
      return *it;
    }
  }
  return nullptr;
}

std::unique_ptr<HloModule> ExtractRandomSubmodule(
    const std::unique_ptr<HloModule>& module, int instruction_count_threshold,
    int height) {
  // Return nullptr if total instruction count is too small
  // or instruction count threshold is larger than the total
  // instruction count.
  const int kInstructionCountThreshold = 10;
  if (module->instruction_count() < kInstructionCountThreshold ||
      instruction_count_threshold > module->instruction_count()) {
    return nullptr;
  }
  // Select a random instruction in a random computation.
  auto comps = module->MakeComputationPostOrder();
  // Pick computation only when its instruction count is large enough.
  auto filtered_comps = FilterComputations(comps, [](HloComputation* c) {
    return c->instruction_count() > kInstructionCountThreshold;
  });
  if (filtered_comps.empty()) {
    LOG(ERROR) << "no submodule generated!";
    return nullptr;
  }
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<int> rand_comp(0, filtered_comps.size() - 1);
  auto instructions =
      filtered_comps[rand_comp(generator)]->MakeInstructionPostOrder();
  std::uniform_int_distribution<int> rand_inst(0, instructions.size() - 1);
  auto inst = instructions[rand_inst(generator)];
  auto submodule = ExtractModule(inst, height);
  const int kRetry = 10;
  int tried = 1;
  while (FindInstruction(submodule.get(), HloOpcode::kCall) != nullptr ||
         submodule->instruction_count() < instruction_count_threshold) {
    // Resample
    inst = instructions[rand_inst(generator)];
    submodule = ExtractModule(inst, height);
    tried++;
    if (tried == kRetry) {
      LOG(ERROR) << "no submodule generated!";
      return nullptr;
    }
  }
  return submodule;
}

std::vector<std::pair<HloInstruction*, std::unique_ptr<HloModule>>>
ExtractInstructionsAsModule(const HloModule& module, int repeat) {
  std::vector<std::pair<HloInstruction*, std::unique_ptr<HloModule>>> ret;
  for (xla::HloComputation* computation : module.computations()) {
    for (auto instruction : computation->instructions()) {
      auto instruction_proto = instruction->ToProto();
      // We skip instructions that calls other computations, like call, reduce,
      // etc
      if (!instruction->called_computations().empty()) {
        continue;
      }

      // Skip trivial ops
      std::unordered_set<xla::HloOpcode> uninterested_ops = {
          xla::HloOpcode::kParameter};
      if (uninterested_ops.count(
              xla::StringToHloOpcode(instruction_proto.opcode())
                  .ValueOrDie())) {
        continue;
      }

      absl::flat_hash_map<int64_t, xla::HloInstruction*> id_to_params;
      xla::HloComputation::Builder computation_builder("new_computation");

      int param_num = 0;
      // Build params
      for (int i = 0; i < instruction->operand_count(); i++) {
        auto parameter = xla::HloInstruction::CreateParameter(
            param_num++, instruction->operand(i)->shape(),
            "param" + std::to_string(i));
        id_to_params.try_emplace(instruction->operand(i)->unique_id(),
                                 parameter.get());
        computation_builder.AddInstruction(std::move(parameter));
      }
      // Build instructions
      xla::HloInstruction* last_instruction = nullptr;
      for (int rep = 0; rep < repeat; rep++) {
        auto new_instruction = xla::HloInstruction::CreateFromProto(
                                   instruction_proto, id_to_params, {})
                                   .ValueOrDie();
        new_instruction->ClearUniqueIdInternal();
        if (last_instruction != nullptr) {
          // old must run before new instruction
          CHECK(last_instruction->AddControlDependencyTo(new_instruction.get())
                    .ok());
        }
        last_instruction = new_instruction.get();
        computation_builder.AddInstruction(std::move(new_instruction));
      }
      auto new_computation = computation_builder.Build(last_instruction);

      xla::HloModuleConfig config;
      std::unique_ptr<xla::HloModule> module =
          std::make_unique<xla::HloModule>("module", config);
      module->AddEntryComputation(std::move(new_computation));
      ret.emplace_back(std::make_pair(last_instruction, std::move(module)));
    }
  }
  return ret;
}

std::vector<std::unique_ptr<HloModule>> ExtractFusionsAsModule(
    const HloModule& module, int repeat) {
  std::vector<std::unique_ptr<HloModule>> ret;
  xla::HloComputation* computation = module.entry_computation();
  for (auto instruction : computation->instructions()) {
    // find fusion instructions
    if (instruction->opcode() == xla::HloOpcode::kFusion) {
      std::vector<xla::HloInstruction*> params;
      xla::HloComputation::Builder computation_builder("fused_comp");

      int param_num = 0;
      // Build params
      for (int i = 0; i < instruction->operand_count(); i++) {
        auto parameter = xla::HloInstruction::CreateParameter(
            param_num++, instruction->operand(i)->shape(),
            "param" + std::to_string(i));
        params.emplace_back(parameter.get());
        computation_builder.AddInstruction(std::move(parameter));
      }
      xla::HloModuleConfig config;
      std::unique_ptr<xla::HloModule> module =
          std::make_unique<xla::HloModule>("module", config);
      // Repeat repeat times, add control dependency
      xla::HloInstruction* last_fusion = nullptr;
      for (int rep = 0; rep < repeat; rep++) {
        // Add fusion itself and fused_computation (clone)
        auto new_fused_computation = module->AddEmbeddedComputation(
            instruction->fused_instructions_computation()->Clone("clone"));
        auto cloned_fusion = absl::make_unique<HloFusionInstruction>(
            instruction->shape(), instruction->fusion_kind(), params,
            new_fused_computation);
        if (last_fusion != nullptr) {
          // old must run before new instruction
          CHECK(last_fusion->AddControlDependencyTo(cloned_fusion.get()).ok());
        }
        last_fusion = cloned_fusion.get();
        computation_builder.AddInstruction(std::move(cloned_fusion));
      }
      auto new_computation = computation_builder.Build(last_fusion);
      module->AddEntryComputation(std::move(new_computation));
      // emplace_back to vector
      ret.emplace_back(std::move(module));
    }
  }
  return ret;
}

}  // namespace xla
