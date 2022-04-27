// Copyright 2021 Garena Online Private Limited

#include "altgraph/utils/hlo_utils.h"

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
  const int kInvalidDim = 6;
  // set dim enum size to be 6 + 1 invalid dim
  const int kDimEnumSize = 7;
  switch (inst->opcode()) {
    // dimension sizes
    case HloOpcode::kBroadcast:
    case HloOpcode::kSetDimensionSize: {
      // counts: 6,0
      attrs->resize(6, -1);
      int idx = 0;
      for (auto d : inst->dimensions()) {
        (*attrs)[idx++] = d;
      }
      add_attr_counts(/*int_count=*/6, /*enum_count=*/0);
      break;
    }
    // dimension indices
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose: {
      // counts: 0,6
      // pad to six to keep a fixed size
      for (auto d : inst->dimensions()) {
        add_enum(/*item=*/d, /*count=*/kDimEnumSize);
      }
      for (int i = 0; i < 6 - inst->dimensions().size(); ++i) {
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
        (*attrs)[idx++] = d;
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
      (*attrs)[idx++] = gather_dims.index_vector_dim();
      for (auto d : gather_inst->gather_slice_sizes()) {
        (*attrs)[idx++] = d;
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
      add_enum(/*item=*/iota_inst->iota_dimension(), /*count=*/kDimEnumSize);
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
    case HloOpcode::kCustomCall:
    default: {
      // counts: 0,0
      add_attr_counts(/*int_count=*/0, /*enum_count=*/0);
      break;
    }
  }
}

}  // namespace xla
