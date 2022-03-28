// Copyright 2021 Garena Online Private Limited

#include "altgraph/hlo_graph.h"

#include <string>

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

HloGraph::HloGraph(const HloModule* m)
    : parent_hlo_module_(const_cast<HloModule*>(m)),
      uid_(m->unique_id()),
      name_(m->name()) {
  Build(m);
}

void HloGraph::Clear() {
  node_feats.Clear();
  in_edge_feats.Clear();
  out_edge_feats.Clear();
  inst_list.clear();
  user_list_offsets.clear();
  operand_list_offsets.clear();
  user_list_indices.clear();
  operand_list_indices.clear();
  uid_to_node_idx_.clear();
  uid_set_.clear();
  uid_to_inst_.clear();
  alternative_indices_.clear();
  in_edge_lists.clear();
  out_edge_lists.clear();
}

void HloGraph::BuildGraphTopology(const HloComputation* c, int gid) {
  // build in/out edge lists with toposort order.
  for (auto inst : c->MakeInstructionPostOrder()) {
    int uid = inst->unique_id();
    if (uid_set_.contains(uid)) {
      continue;
    }
    uid_set_.insert(uid);

    // add into in edge lists
    if (inst->called_computations().empty()) {
      // normal instruction, put operands and users to its in_edge_lists
      // and out_edge_lists.
      for (auto operand : inst->operands()) {
        size_t op_uid = operand->unique_id();
        in_edge_lists[uid].push_back(op_uid);
      }
    } else {
      // instruction's in edges will be called computation's root insts.
      auto operands = inst->operands();
      for (auto c : inst->called_computations()) {
        BuildGraphTopology(c, gid + 1);
        auto params = c->parameter_instructions();
        CHECK_EQ(params.size(), operands.size());
        for (int i = 0; i < params.size(); ++i) {
          int op_uid = operands[i]->unique_id();
          int param_uid = params[i]->unique_id();
          out_edge_lists[op_uid].push_back(param_uid);
          in_edge_lists[param_uid].push_back(op_uid);
        }
        size_t root_uid = c->root_instruction()->unique_id();
        in_edge_lists[uid].push_back(root_uid);
        out_edge_lists[root_uid].push_back(uid);
      }
    }

    // add into out edge lists
    for (auto user : inst->users()) {
      if (user->called_computations().empty()) {
        size_t user_uid = user->unique_id();
        out_edge_lists[uid].push_back(user_uid);
      }
    }

    inst_list.push_back(inst);
    std::string name = inst->name();
    // Add to basic node_feats
    node_feats.uids.push_back(uid);
    node_feats.names.push_back(name);
    node_feats.gids.push_back(gid);
    uid_to_inst_.insert({uid, inst});
  }
  return;
}

void HloGraph::BuildRaggedTensors() {
  int num_nodes = node_feats.uids.size();
  user_list_offsets.resize(num_nodes + 1);
  operand_list_offsets.resize(num_nodes + 1);

  for (int ii = 0; ii < num_nodes; ++ii) {
    uid_to_node_idx_[node_feats.uids[ii]] = ii;
  }

  user_list_offsets[0] = 0;
  operand_list_offsets[0] = 0;
  for (int i = 1; i <= node_feats.uids.size(); ++i) {
    // build offset arrays
    int uid = node_feats.uids[i - 1];
    int ucount = (out_edge_lists.find(uid) == out_edge_lists.end())
                     ? 0
                     : out_edge_lists[uid].size();
    int opcount = (in_edge_lists.find(uid) == in_edge_lists.end())
                      ? 0
                      : in_edge_lists[uid].size();

    operand_list_offsets[i] = operand_list_offsets[i - 1] + opcount;
    user_list_offsets[i] = user_list_offsets[i - 1] + ucount;
    // build indices arrays
    if (ucount > 0) {
      for (int neighbor_uid : out_edge_lists[uid]) {
        user_list_indices.push_back(uid_to_node_idx_[neighbor_uid]);
      }
    }
    if (opcount > 0) {
      for (int neighbor_uid : in_edge_lists[uid]) {
        operand_list_indices.push_back(uid_to_node_idx_[neighbor_uid]);
      }
    }
  }
}

void HloGraph::PrepareFeatures() {
  auto genuid = [](int src_uid, int dst_uid) -> int64_t {
    int64_t suid = absl::bit_cast<int>(src_uid);
    int64_t duid = absl::bit_cast<int>(dst_uid);
    return (suid << 32) | duid;
  };

  int num_nodes = inst_list.size();
  node_feats.has_max_in_tensor.assign(num_nodes, false);
  node_feats.has_max_out_tensor.assign(num_nodes, false);
  node_feats.is_alternative.assign(num_nodes, false);
  for (int i = 0; i < num_nodes; ++i) {
    size_t user_offset = user_list_offsets[i];
    size_t operand_offset = operand_list_offsets[i];
    int ucount = user_list_offsets[i + 1] - user_offset;
    int opcount = operand_list_offsets[i + 1] - operand_offset;
    int cur_uid = node_feats.uids[i];
    auto cur_inst = uid_to_inst_[cur_uid];

    // add to node features
    if (cur_inst->opcode() == HloOpcode::kAlternatives) {
      node_feats.is_alternative[i] = true;
      alternative_indices_.push_back(i);
    }
    node_feats.num_users.push_back(ucount);
    node_feats.num_operands.push_back(opcount);

    // add to out edge features
    int64_t out_tensor_size = 0;
    for (int s = user_offset; s < user_list_offsets[i + 1]; ++s) {
      int user_node_idx = user_list_indices[s];
      int user_uid = node_feats.uids[user_node_idx];

      int64_t euid = genuid(cur_uid, user_uid);
      out_edge_feats.uids.push_back(euid);
      uid_to_out_edge_idx_.insert({euid, s});
      out_edge_feats.srcs.push_back(i);
      out_edge_feats.dsts.push_back(user_node_idx);
      // put in shapes, layouts, and dtypes for cur_inst
      Shape shape = cur_inst->shape();
      auto minor_to_major = shape.layout().minor_to_major();
      int dim_size = shape.dimensions_size();
      for (int k = 0; k < 8; ++k) {
        if (k < dim_size) {
          out_edge_feats.dims.push_back(shape.dimensions(k));
          out_edge_feats.layouts.push_back(minor_to_major[k]);
        } else {
          out_edge_feats.dims.push_back(-1);
          out_edge_feats.layouts.push_back(-1);
        }
      }
      out_tensor_size += out_edge_feats.GetTensorSize(s);
      out_edge_feats.dtypes.push_back(shape.element_type());
    }
    node_feats.out_tensor_sizes.push_back(out_tensor_size);

    // add to in edge features
    int64_t in_tensor_size = 0;
    for (int s = operand_offset; s < operand_list_offsets[i + 1]; ++s) {
      int operand_node_idx = operand_list_indices[s];
      int operand_uid = node_feats.uids[operand_node_idx];
      auto operand_inst = uid_to_inst_[operand_uid];

      int64_t euid = genuid(operand_uid, cur_uid);
      in_edge_feats.uids.push_back(euid);
      uid_to_in_edge_idx_.insert({euid, s});
      in_edge_feats.srcs.push_back(operand_node_idx);
      in_edge_feats.dsts.push_back(i);
      // put in shapes, layouts, and dtypes for operand_inst
      Shape shape = operand_inst->shape();
      auto minor_to_major = shape.layout().minor_to_major();
      int dim_size = shape.dimensions_size();
      for (int k = 0; k < 8; ++k) {
        if (k < dim_size) {
          in_edge_feats.dims.push_back(shape.dimensions(k));
          in_edge_feats.layouts.push_back(minor_to_major[k]);
        } else {
          in_edge_feats.dims.push_back(-1);
          in_edge_feats.layouts.push_back(-1);
        }
      }
      in_tensor_size += in_edge_feats.GetTensorSize(s);
      in_edge_feats.dtypes.push_back(shape.element_type());
    }
    node_feats.in_tensor_sizes.push_back(in_tensor_size);
  }
  auto max_input_tensor_size = std::max_element(
      node_feats.in_tensor_sizes.begin(), node_feats.in_tensor_sizes.end());
  auto max_output_tensor_size = std::max_element(
      node_feats.out_tensor_sizes.begin(), node_feats.out_tensor_sizes.end());
  for (int i = 0; i < num_nodes; ++i) {
    if (node_feats.in_tensor_sizes[i] == *max_input_tensor_size) {
      node_feats.has_max_in_tensor[i] = true;
    }
    if (node_feats.out_tensor_sizes[i] == *max_output_tensor_size) {
      node_feats.has_max_out_tensor[i] = true;
    }
  }
}

bool HloGraph::Build(const HloModule* m) {
  Clear();
  BuildGraphTopology(m->entry_computation(), 0);
  BuildRaggedTensors();
  PrepareFeatures();
  // final touch, add the root_index
  int entry_root_uid = m->entry_computation()->root_instruction()->unique_id();
  root_index = uid_to_node_idx_[entry_root_uid];

  LOG(ERROR) << "HloGraph build finished";

  uint64_t hlograph_hash = Hash();
  uint64_t hlomodule_hash = parent_hlo_module_->CalledComputationHash();
  if (hlograph_hash == hlomodule_hash) {
    LOG(ERROR) << "HloGraph build verified.";
  }

  return true;
}

uint64_t HloGraph::Hash() {
  // only consider if cross-computation edges are correctly built.
  // Since all operands and users are added properly by calling
  // HloInstruction's operands() and users() function.
  uint64_t hash_value = parent_hlo_module_->entry_computation_layout().Hash();
  for (int i = 0; i < in_edge_feats.uids.size(); ++i) {
    // add euid of cross computation edge to hash_value
    int src = in_edge_feats.srcs[i];
    int dst = in_edge_feats.dsts[i];
    if (node_feats.gids[src] != node_feats.gids[dst]) {
      hash_value =
          tensorflow::Hash64CombineUnordered(hash_value, in_edge_feats.uids[i]);
    }
  }
  for (int i = 0; i < out_edge_feats.uids.size(); ++i) {
    // add euid of cross computation edge to hash_value
    int src = out_edge_feats.srcs[i];
    int dst = out_edge_feats.dsts[i];
    if (node_feats.gids[src] != node_feats.gids[dst]) {
      hash_value = tensorflow::Hash64CombineUnordered(hash_value,
                                                      out_edge_feats.uids[i]);
    }
  }
  return hash_value;
}

void HloGraph::ShowStats() {
  auto oedge_offsets = get_out_edge_offsets();
  auto iedge_offsets = get_in_edge_offsets();
  auto oedge_indices = get_out_edge_indices();
  auto iedge_indices = get_in_edge_indices();
  LOG(ERROR) << "module name: " << name_;
  LOG(ERROR) << "number of nodes: " << oedge_offsets.size() - 1;
  LOG(ERROR) << "number of in edges: " << iedge_offsets.back();
  LOG(ERROR) << "number of out edges: " << oedge_offsets.back();
  LOG(ERROR) << "================================";

  auto print_vector = [](int64_t* vec) -> std::string {
    std::string os;
    os = "[";
    for (int i = 0; i < 8; ++i) {
      os += std::to_string(vec[i]);
      os += ",";
    }
    os += "]";
    return os;
  };

  auto names = get_node_names();
  auto gids = get_gids();
  auto ucounts = get_user_counts();
  auto opcounts = get_operand_counts();

  auto oedge_uids = get_out_edge_uids();
  auto oedge_srcs = get_out_edge_srcs();
  auto oedge_dsts = get_out_edge_dsts();
  auto oedge_dims = get_out_edge_dims();
  auto oedge_layouts = get_out_edge_layouts();
  auto oedge_dtypes = get_out_edge_dtypes();

  auto iedge_uids = get_in_edge_uids();
  auto iedge_srcs = get_in_edge_srcs();
  auto iedge_dsts = get_in_edge_dsts();
  auto iedge_dims = get_in_edge_dims();
  auto iedge_layouts = get_in_edge_layouts();
  auto iedge_dtypes = get_in_edge_dtypes();

  for (int i = 0; i < oedge_offsets.size() - 1; ++i) {
    LOG(ERROR) << "node index: " << i;
    LOG(ERROR) << "node name: " << names[i];
    LOG(ERROR) << "gid: " << gids[i];
    LOG(ERROR) << "user_count: " << ucounts[i];
    LOG(ERROR) << "operand_count: " << opcounts[i];
    int start_idx = oedge_offsets[i];
    int end_idx = oedge_offsets[i + 1];
    for (int ii = start_idx; ii < end_idx; ++ii) {
      int idx = oedge_indices[ii];
      LOG(ERROR) << "  out edge: " << idx << " " << names[idx];
      LOG(ERROR) << "  " << oedge_uids[ii] << " | " << oedge_srcs[ii] << "->"
                 << oedge_dsts[ii];
      LOG(ERROR) << "  dims: " << print_vector(&oedge_dims[ii * 8]);
      LOG(ERROR) << "  layouts: " << print_vector(&oedge_layouts[ii * 8]);
      LOG(ERROR) << "  dtype: " << oedge_dtypes[ii];
    }
    start_idx = iedge_offsets[i];
    end_idx = iedge_offsets[i + 1];
    for (int ii = start_idx; ii < end_idx; ++ii) {
      int idx = iedge_indices[ii];
      LOG(ERROR) << "  in edge: " << idx << " " << names[idx];
      LOG(ERROR) << "  " << iedge_uids[ii] << " | " << iedge_srcs[ii] << "->"
                 << iedge_dsts[ii];
      LOG(ERROR) << "  dims: " << print_vector(&iedge_dims[ii * 8]);
      LOG(ERROR) << "  layouts: " << print_vector(&iedge_layouts[ii * 8]);
      LOG(ERROR) << "  dtype: " << iedge_dtypes[ii];
    }
    LOG(ERROR) << "--------------------------------";
  }
}

}  // namespace xla
