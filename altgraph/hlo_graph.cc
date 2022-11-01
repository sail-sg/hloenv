// Copyright 2021 Garena Online Private Limited

#include "altgraph/hlo_graph.h"

#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "altgraph/utils/hlo_utils.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"

namespace altgraph {

HloGraph::HloGraph(const xla::HloModule* m, bool debug, bool inline_fused_comp,
                   bool do_hash_verification)
    : graph_load_errors_(0),
      kNumOpcodes(xla::HloOpcodeCount()),
      parent_hlo_module_(const_cast<xla::HloModule*>(m)),
      uid_(m->unique_id()),
      name_(m->name()) {
  user_list_offsets_ = std::make_shared<std::vector<size_t>>();
  user_list_indices_ = std::make_shared<std::vector<size_t>>();
  operand_list_offsets_ = std::make_shared<std::vector<size_t>>();
  operand_list_indices_ = std::make_shared<std::vector<size_t>>();
  opcode_attr_counts_ = std::make_shared<std::vector<int>>();
  alternative_indices_ = std::make_shared<std::vector<int>>();

  Build(m, debug, inline_fused_comp, do_hash_verification);
}

void HloGraph::Clear() {
  node_feats_.Clear();
  in_edge_feats_.Clear();
  out_edge_feats_.Clear();

  user_list_offsets_->clear();
  operand_list_offsets_->clear();
  user_list_indices_->clear();
  operand_list_indices_->clear();
  opcode_attr_counts_->clear();
  alternative_indices_->clear();

  inst_list_.clear();
  uid_to_node_idx_.clear();
  uid_to_inst_.clear();
  in_edge_lists_.clear();
  out_edge_lists_.clear();
  called_comp_lists_.clear();
}

void HloGraph::BuildGraphTopology(const xla::HloComputation* c, int gid,
                                  bool is_fusion_comp) {
  // build in/out edge lists with toposort order.
  int num_group_inst = c->instruction_count();
  float normalized_num_group_inst = 0;
  if (gid > 0 && num_group_inst > 0) {
    normalized_num_group_inst = 1.0 / num_group_inst;
  }
  for (auto inst : c->MakeInstructionPostOrder()) {
    int uid = inst->unique_id();

    // add into in edge lists
    for (auto operand : inst->unique_operands()) {
      size_t op_uid = operand->unique_id();
      in_edge_lists_[uid].push_back(op_uid);
    }

    // add into out edge lists
    for (auto user : inst->users()) {
      size_t user_uid = user->unique_id();
      out_edge_lists_[uid].push_back(user_uid);
    }

    inst_list_.push_back(inst);
    std::string name = inst->name();
    int opcode = static_cast<int>(inst->opcode());
    // Add to basic node_feats_
    node_feats_.uids->push_back(uid);
    node_feats_.names->push_back(name);
    node_feats_.gids->push_back(gid);
    node_feats_.is_in_fusion->push_back(is_fusion_comp);
    node_feats_.opcodes->push_back(opcode);
    node_feats_.normalized_num_group_inst->push_back(normalized_num_group_inst);
    std::vector<int> opcode_attrs;
    std::vector<int> opcode_attr_counts;
    GetInstructionAttributesAndCounts(inst, &opcode_attrs, &opcode_attr_counts);
    node_feats_.opcode_attrs->insert(node_feats_.opcode_attrs->end(),
                                     opcode_attrs.begin(), opcode_attrs.end());
    node_feats_.num_opcode_attrs->insert(node_feats_.num_opcode_attrs->end(),
                                         opcode_attr_counts.begin(),
                                         opcode_attr_counts.end());
    uid_to_inst_.insert({uid, inst});
  }

  return;
}

void HloGraph::SetFusedCompId(
    const absl::flat_hash_map<xla::HloComputation*, int>& comp_id_map) {
  int num_nodes = node_feats_.uids->size();
  for (int ii = 0; ii < num_nodes; ++ii) {
    int uid = node_feats_.uids->at(ii);
    uid_to_node_idx_[uid] = ii;
    auto instruction = uid_to_inst_[uid];
    // For fusion instruction, record the computation id of its fused
    // computation; For all non-fusion instructions, set their fused_comp_ids to
    // zero.
    auto called_comps = instruction->called_computations();
    if (!called_comps.empty()) {
      for (auto comp : called_comps) {
        int comp_id = comp_id_map.at(comp);
        called_comp_lists_[ii].push_back(comp_id);
      }
    }
  }
}

std::vector<int> HloGraph::FindInstIndicesOfFusedComp(int fused_comp_id) {
  std::vector<int> indices;
  auto gid_vector = node_feats_.gids.get();
  auto it = gid_vector->begin();
  while ((it = std::find_if(it, gid_vector->end(), [&](size_t const& e) {
            return e == fused_comp_id;
          })) != gid_vector->end()) {
    indices.push_back(std::distance(gid_vector->begin(), it));
    it++;
  }
  return indices;
}

bool HloGraph::ShouldInline(int inst_idx, xla::HloInstruction* inst) {
  return (!called_comp_lists_[inst_idx].empty() &&
          !(inst->opcode() == xla::HloOpcode::kReduce ||
            inst->opcode() == xla::HloOpcode::kReduceWindow ||
            inst->opcode() == xla::HloOpcode::kScatter ||
            inst->opcode() == xla::HloOpcode::kSelectAndScatter ||
            inst->opcode() == xla::HloOpcode::kSort));
}

void HloGraph::FusedComputationInlining() {
  int num_nodes = node_feats_.uids->size();
  for (int ii = 0; ii < num_nodes; ++ii) {
    // get fusion instruction's uid and fused computation id it points to.
    int uid = node_feats_.uids->at(ii);
    auto current_instruction = uid_to_inst_[uid];
    LOG(ERROR) << "considering: " << current_instruction->name();
    int operand_fusion;
    if (ShouldInline(ii, current_instruction)) {
      for (int fused_comp_id : called_comp_lists_[ii]) {
        // If instruction is fusion, we inline the fused computation.
        auto instr_indices = FindInstIndicesOfFusedComp(fused_comp_id);
        if (node_feats_.normalized_num_group_inst->at(ii) == 0) {
          // update fusion node's normalized_num_group_inst to be the same as
          // instruction's normalized_num_group_inst in the sub computation.
          node_feats_.normalized_num_group_inst->at(ii) =
              node_feats_.normalized_num_group_inst->at(instr_indices[1]);
        }
        for (int idx_instr_indices = 0;
             idx_instr_indices < instr_indices.size(); ++idx_instr_indices) {
          // Iterate over all instructions in the fused computation.
          int fused_comp_uid =
              node_feats_.uids->at(instr_indices[idx_instr_indices]);
          auto instruction = uid_to_inst_[fused_comp_uid];
          if (instruction->opcode() == xla::HloOpcode::kParameter) {
            // param of fused comp instructions, we rewire fusion
            // instruction's operand to params inside fused computation
            operand_fusion = in_edge_lists_[uid].front();
            in_edge_lists_[uid].erase(in_edge_lists_[uid].begin());
            if (fused_comp_id != called_comp_lists_[ii].back()) {
              // if not the last comp, add operand_fusion back as it will be
              // needed for the next comp too.
              in_edge_lists_[uid].push_back(operand_fusion);
            }
            in_edge_lists_[fused_comp_uid].push_back(operand_fusion);
            // remove fusion instruction from user list of its operand.
            auto& operand_oel = out_edge_lists_[operand_fusion];
            if (fused_comp_id == called_comp_lists_[ii].back()) {
              operand_oel.erase(
                  std::remove(operand_oel.begin(), operand_oel.end(), uid),
                  operand_oel.end());
            }
            operand_oel.push_back(fused_comp_uid);
          }
          if (instruction->IsRoot()) {
            // root instruction of fused comp, we set user of fused
            // computation root to fusion instruction.
            out_edge_lists_[fused_comp_uid].push_back(uid);
            in_edge_lists_[uid].push_back(fused_comp_uid);
            // For fusion instruction, set their gids to the computation id it
            // points to.
            node_feats_.gids->at(ii) =
                node_feats_.gids->at(instr_indices[idx_instr_indices]);
          }
        }
      }
    }
  }
}

bool HloGraph::HasCycleForward_(int idx, std::vector<bool>* visited,
                                std::vector<bool>* stack) {
  if (!visited->at(idx)) {
    visited->at(idx) = true;
    stack->at(idx) = true;

    int uid = node_feats_.uids->at(idx);
    for (int neighbor_uid : out_edge_lists_[uid]) {
      int new_idx = uid_to_node_idx_[neighbor_uid];
      if (!visited->at(new_idx) && HasCycleForward_(new_idx, visited, stack)) {
        auto current_instruction = uid_to_inst_[uid];
        auto neighbor_instruction = uid_to_inst_[neighbor_uid];
        LOG(INFO) << "passing cycle between: " << current_instruction->name();
        LOG(INFO) << "and: " << neighbor_instruction->name();
        return true;
      } else if (stack->at(new_idx)) {
        auto current_instruction = uid_to_inst_[uid];
        auto neighbor_instruction = uid_to_inst_[neighbor_uid];
        LOG(INFO) << "source cycle between: " << current_instruction->name();
        LOG(INFO) << "and: " << neighbor_instruction->name();
        return true;
      }
    }
  }
  stack->at(idx) = false;
  return false;
}

bool HloGraph::HasCycleBackward_(int idx, std::vector<bool>* visited,
                                 std::vector<bool>* stack) {
  if (!visited->at(idx)) {
    visited->at(idx) = true;
    stack->at(idx) = true;

    int uid = node_feats_.uids->at(idx);
    for (int neighbor_uid : in_edge_lists_[uid]) {
      int new_idx = uid_to_node_idx_[neighbor_uid];
      if (!visited->at(new_idx) && HasCycleBackward_(new_idx, visited, stack)) {
        return true;
      } else if (stack->at(new_idx)) {
        return true;
      }
    }
  }
  stack->at(idx) = false;
  return false;
}

bool HloGraph::HasCycle() {
  int num_nodes = node_feats_.uids->size();
  std::vector<bool> visited(num_nodes, false);
  std::vector<bool> stack(num_nodes, false);
  for (int idx = 0; idx < num_nodes; ++idx) {
    if (!visited[idx] && HasCycleForward_(idx, &visited, &stack)) {
      return true;
    }
  }
  std::fill(visited.begin(), visited.end(), false);
  std::fill(stack.begin(), stack.end(), false);
  for (int idx = 0; idx < num_nodes; ++idx) {
    if (!visited[idx] && HasCycleBackward_(idx, &visited, &stack)) {
      return true;
    }
  }
  return false;
}

void HloGraph::BuildRaggedTensors(
    const absl::flat_hash_map<xla::HloComputation*, int>& comp_id_map) {
  int num_nodes = node_feats_.uids->size();
  // Resize offsets arrays (need one extra space at the end)
  user_list_offsets_->resize(num_nodes + 1);
  operand_list_offsets_->resize(num_nodes + 1);

  user_list_offsets_->at(0) = 0;
  operand_list_offsets_->at(0) = 0;
  for (int i = 1; i <= node_feats_.uids->size(); ++i) {
    // build offset arrays
    int uid = node_feats_.uids->at(i - 1);
    int ucount = (out_edge_lists_.find(uid) == out_edge_lists_.end())
                     ? 0
                     : out_edge_lists_[uid].size();
    int opcount = (in_edge_lists_.find(uid) == in_edge_lists_.end())
                      ? 0
                      : in_edge_lists_[uid].size();

    operand_list_offsets_->at(i) = operand_list_offsets_->at(i - 1) + opcount;
    user_list_offsets_->at(i) = user_list_offsets_->at(i - 1) + ucount;
    // build indices arrays
    if (ucount > 0) {
      for (int neighbor_uid : out_edge_lists_[uid]) {
        user_list_indices_->push_back(uid_to_node_idx_[neighbor_uid]);
      }
    }
    if (opcount > 0) {
      for (int neighbor_uid : in_edge_lists_[uid]) {
        operand_list_indices_->push_back(uid_to_node_idx_[neighbor_uid]);
      }
    }
  }
}

void HloGraph::PrepareFeatures() {
  // combines uid of source and destination node into a int64.
  auto genuid = [](int src_uid, int dst_uid) -> int64_t {
    int64_t suid = absl::bit_cast<int>(src_uid);
    int64_t duid = absl::bit_cast<int>(dst_uid);
    return (suid << 32) | duid;
  };

  int num_nodes = inst_list_.size();
  node_feats_.has_max_in_tensor->assign(num_nodes, false);
  node_feats_.has_max_out_tensor->assign(num_nodes, false);
  node_feats_.is_alternative->assign(num_nodes, false);
  for (int i = 0; i < num_nodes; ++i) {
    size_t user_offset = user_list_offsets_->at(i);
    size_t operand_offset = operand_list_offsets_->at(i);
    int ucount = user_list_offsets_->at(i + 1) - user_offset;
    int opcount = operand_list_offsets_->at(i + 1) - operand_offset;
    int cur_uid = node_feats_.uids->at(i);
    auto cur_inst = uid_to_inst_[cur_uid];

    // add to node features
    if (cur_inst->opcode() == xla::HloOpcode::kAlternatives) {
      node_feats_.is_alternative->at(i) = true;
      alternative_indices_->push_back(i);
    }
    node_feats_.num_users->push_back(ucount);
    node_feats_.num_operands->push_back(opcount);

    // add to out edge features
    int64_t out_tensor_size = 0;
    for (int s = user_offset; s < user_list_offsets_->at(i + 1); ++s) {
      int user_node_idx = user_list_indices_->at(s);
      int user_uid = node_feats_.uids->at(user_node_idx);

      int64_t euid = genuid(cur_uid, user_uid);
      out_edge_feats_.uids->push_back(euid);
      uid_to_out_edge_idx_.insert({euid, s});
      out_edge_feats_.srcs->push_back(i);
      out_edge_feats_.dsts->push_back(user_node_idx);
      int src_gid = node_feats_.gids->at(i);
      int dst_gid = node_feats_.gids->at(user_node_idx);
      // edge_type:
      // outside any fusion, 0
      // inside fusion, 1
      // cross fusion, 2
      uint8_t edge_type = (src_gid == dst_gid) ? ((src_gid == 0) ? 0 : 1) : 2;
      out_edge_feats_.types->push_back(edge_type);
      // put in shapes, layouts, lehmer codes, and dtypes for cur_inst
      xla::Shape shape = cur_inst->shape();
      auto minor_to_major = shape.layout().minor_to_major();
      int dim_size = shape.dimensions_size();
      for (int k = 0; k < 8; ++k) {
        if (k < dim_size) {
          out_edge_feats_.dims->push_back(shape.dimensions(k));
          out_edge_feats_.layouts->push_back(minor_to_major[k]);
          int lehmer = 0;
          for (int l = 0; l < k; ++l) {
            lehmer += (minor_to_major[l] > minor_to_major[k]);
          }
          out_edge_feats_.lehmercodes->push_back(lehmer);
        } else {
          out_edge_feats_.dims->push_back(-1);
          out_edge_feats_.layouts->push_back(-1);
          out_edge_feats_.lehmercodes->push_back(0);
        }
      }
      out_tensor_size += out_edge_feats_.GetTensorSize(s);
      out_edge_feats_.dtypes->push_back(static_cast<int>(shape.element_type()));
    }
    node_feats_.out_tensor_sizes->push_back(out_tensor_size);

    // add to in edge features
    int64_t in_tensor_size = 0;
    for (int s = operand_offset; s < operand_list_offsets_->at(i + 1); ++s) {
      int operand_node_idx = operand_list_indices_->at(s);
      int operand_uid = node_feats_.uids->at(operand_node_idx);
      auto operand_inst = uid_to_inst_[operand_uid];

      int64_t euid = genuid(operand_uid, cur_uid);
      in_edge_feats_.uids->push_back(euid);
      uid_to_in_edge_idx_.insert({euid, s});
      in_edge_feats_.srcs->push_back(operand_node_idx);
      in_edge_feats_.dsts->push_back(i);
      int src_gid = node_feats_.gids->at(operand_node_idx);
      int dst_gid = node_feats_.gids->at(i);
      // edge_type:
      // outside any fusion, 0
      // inside fusion, 1
      // cross fusion, 2
      uint8_t edge_type = (src_gid == dst_gid) ? ((src_gid == 0) ? 0 : 1) : 2;
      in_edge_feats_.types->push_back(edge_type);
      // put in shapes, layouts, and dtypes for operand_inst
      xla::Shape shape = operand_inst->shape();
      auto minor_to_major = shape.layout().minor_to_major();
      int dim_size = shape.dimensions_size();
      for (int k = 0; k < 8; ++k) {
        if (k < dim_size) {
          in_edge_feats_.dims->push_back(shape.dimensions(k));
          in_edge_feats_.layouts->push_back(minor_to_major[k]);
          int lehmer = 0;
          for (int l = 0; l < k; ++l) {
            lehmer += (minor_to_major[l] > minor_to_major[k]);
          }
          in_edge_feats_.lehmercodes->push_back(lehmer);
        } else {
          in_edge_feats_.dims->push_back(-1);
          in_edge_feats_.layouts->push_back(-1);
          in_edge_feats_.lehmercodes->push_back(0);
        }
      }
      in_tensor_size += in_edge_feats_.GetTensorSize(s);
      in_edge_feats_.dtypes->push_back(static_cast<int>(shape.element_type()));
    }
    node_feats_.in_tensor_sizes->push_back(in_tensor_size);
  }
  auto max_input_tensor_size = std::max_element(
      node_feats_.in_tensor_sizes->begin(), node_feats_.in_tensor_sizes->end());
  auto max_output_tensor_size =
      std::max_element(node_feats_.out_tensor_sizes->begin(),
                       node_feats_.out_tensor_sizes->end());
  for (int i = 0; i < num_nodes; ++i) {
    if (node_feats_.in_tensor_sizes->at(i) == *max_input_tensor_size) {
      node_feats_.has_max_in_tensor->at(i) = true;
    }
    if (node_feats_.out_tensor_sizes->at(i) == *max_output_tensor_size) {
      node_feats_.has_max_out_tensor->at(i) = true;
    }
  }
}

bool HloGraph::Build(const xla::HloModule* m, bool debug,
                     bool inline_fused_comp, bool do_hash_verification) {
  parent_hlo_module_ = const_cast<xla::HloModule*>(m);
  uid_ = m->unique_id();
  name_ = m->name();

  // Clear to rebuild.
  Clear();
  GenOpcodeAttrCounts();

  // For each sub computation, build its graph topology.
  auto post_order_comps = m->MakeComputationPostOrder();
  int gid = 0;
  absl::flat_hash_map<xla::HloComputation*, int> fused_comp_id_map;
  while (!post_order_comps.empty()) {
    // We iterate in reverse post order, so remove from the back of the
    // vector.
    xla::HloComputation* comp = post_order_comps.back();
    post_order_comps.pop_back();
    fused_comp_id_map[comp] = gid;
    bool is_fusion = comp->IsFusionComputation();
    BuildGraphTopology(comp, gid++, is_fusion);
  }

  // Inline fused computation.
  SetFusedCompId(fused_comp_id_map);
  if (inline_fused_comp) {
    FusedComputationInlining();
  }

  if (debug && HasCycle()) {
    LOG(FATAL) << "ERROR: Detected cycles in graph!";
    return false;
  }

  // Fill in content of node/edge features.
  BuildRaggedTensors(fused_comp_id_map);
  PrepareFeatures();

  // Final touch, add the root_index_
  int entry_root_uid = m->entry_computation()->root_instruction()->unique_id();
  root_index_ = uid_to_node_idx_[entry_root_uid];

  LOG(INFO) << "HloGraph build finished";

  // Optionally do hash verification.
  if (do_hash_verification) {
    uint64_t hlograph_hash = Hash();
    uint64_t hlomodule_hash = parent_hlo_module_->CalledComputationHash();
    // TODO(wangyzh/ohcy) Rewrite the HloGraph::Hash according to the new
    // mechanism.
    if (hlograph_hash == hlomodule_hash) {
      LOG(INFO) << "HloGraph build verified.";
      return true;
    } else {
      LOG(INFO) << "HloGraph hash NOT verified.";
      return false;
    }
  }

  return true;
}

uint64_t HloGraph::Hash() {
  // only consider if cross-computation edges are correctly built.
  // Since all operands and users are added properly by calling
  // xla::HloInstruction's operands() and users() function.
  uint64_t hash_value = absl::HashOf(parent_hlo_module_);
  for (int i = 0; i < in_edge_feats_.uids->size(); ++i) {
    // add euid of cross computation edge to hash_value
    int src = in_edge_feats_.srcs->at(i);
    int dst = in_edge_feats_.dsts->at(i);
    if (node_feats_.gids->at(src) != node_feats_.gids->at(dst)) {
      hash_value = tensorflow::Hash64CombineUnordered(
          hash_value, in_edge_feats_.uids->at(i));
    }
  }
  for (int i = 0; i < out_edge_feats_.uids->size(); ++i) {
    // add euid of cross computation edge to hash_value
    int src = out_edge_feats_.srcs->at(i);
    int dst = out_edge_feats_.dsts->at(i);
    if (node_feats_.gids->at(src) != node_feats_.gids->at(dst)) {
      hash_value = tensorflow::Hash64CombineUnordered(
          hash_value, out_edge_feats_.uids->at(i));
    }
  }
  return hash_value;
}

// TODO(wangyzh): Ideally this could be a static member for the HloGraph.
void HloGraph::GenOpcodeAttrCounts() {
  opcode_attr_counts_->resize(kNumOpcodes * 2, 0);
  auto update_opcode_attr_counts = [&](int idx, int int_count, int enum_count) {
    opcode_attr_counts_->at(idx * 2) = int_count;
    opcode_attr_counts_->at(idx * 2 + 1) = enum_count;
  };

  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kBroadcast), 0,
                            6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kSetDimensionSize),
                            0, 6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kConcatenate), 0,
                            6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kReduce), 0,
                            6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kReverse), 0,
                            6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kTranspose), 0,
                            6 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kCompare), 0,
                            6 + 4);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kConvolution), 16,
                            24 * 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kDot), 0,
                            24 * 7 + 3);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kDynamicSlice), 6,
                            0);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kGather), 7,
                            18 * 7 + 2);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kGetTupleElement),
                            1, 0);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kIota), 0, 7);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kPad), 18, 0);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kScatter), 1,
                            18 * 7 + 2 * 2);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kSlice), 18, 0);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kSort), 0, 7 + 2);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kTriangularSolve),
                            0, 2 * 3 + 4);
  update_opcode_attr_counts(static_cast<int>(xla::HloOpcode::kCustomCall), 0,
                            1 * 13);
  return;
}

}  // namespace altgraph
