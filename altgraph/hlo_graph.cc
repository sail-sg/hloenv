// Copyright 2021 Garena Online Private Limited

#include "altgraph/hlo_graph.h"

#include <string>

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
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
  user_list_offsets.clear();
  operand_list_offsets.clear();
  user_list_indices.clear();
  operand_list_indices.clear();
  alternative_indices_.clear();
  uid_to_node_ind_.clear();
  uid_to_inst_.clear();
}

bool HloGraph::Build(const HloModule* m) {
  int gid = -1;
  int node_ind = 0;
  Clear();

  std::vector<HloInstruction*> inst_list;

  // Since CSR/CSC is a flatten linear structure where each item's
  // global index needs to be decided first before values being
  // filled up. We need to first compute the sizes/offsets, then
  // put indices into the neighbor list array (xxx_indices).
  for (auto c : m->MakeComputationPostOrder()) {
    gid++;
    for (auto inst : c->MakeInstructionPostOrder()) {
      inst_list.push_back(inst);
      int uid = inst->unique_id();
      std::string name = inst->name();

      // Add to node_feats
      node_feats.uids.push_back(uid);
      node_feats.names.push_back(name);
      node_feats.gids.push_back(gid);

      // Add to offsets
      user_list_offsets.push_back(inst->user_count());
      if (inst->called_computations().empty()) {
        operand_list_offsets.push_back(inst->operand_count());
      } else {
        // if instruction calls computation(s), it should have
        // no operand other than that computation's root instruction.
        // current operands will all point to that computation's
        // params.
        operand_list_offsets.push_back(0);
      }

      // add uid to node id hash map
      // when we build neighbor list we will need these indices.
      uid_to_node_ind_.insert({uid, node_ind++});
      uid_to_inst_.insert({uid, inst});
    }
  }

  // For instruction that calls computations, we:
  // 1) add instruction as its computation's root inst's extra user
  // 2) connect instruction's operands to its called computation's params in
  // order
  //    2.1) for instruction's each operand, add one user
  //    2.2) for each called computation's param, add one operand

  // Update user/operand count
  for (auto inst : inst_list) {
    int uid = inst->unique_id();
    auto called_comps = inst->called_computations();
    if (called_comps.size() > 0) {
      // for kCall and kFusion, called_comps.size() should be 1
      // for kWhile and kCondition, called_comps.size() should be 2
      // and body_func/cond_func in kWhile and cond_true/cond_false func
      // in kCondition share the same set of param.
      // So it's safe to apply operand of current instruction to
      // both computations's params.

      // add one operand count to current instruction (from called comp).
      operand_list_offsets[uid_to_node_ind_[uid]]++;
      for (auto cc : called_comps) {
        int cc_uid = cc->root_instruction()->unique_id();
        user_list_offsets[uid_to_node_ind_[cc_uid]]++;
        for (auto param : cc->parameter_instructions()) {
          int param_uid = param->unique_id();
          // add operand count to called comp's each parameter instruction.
          operand_list_offsets[uid_to_node_ind_[param_uid]]++;
        }
      }
    }
  }

  // running offsets are used to keep track of inserting indices
  // when creating column indices arrays.
  std::vector<int> running_user_offset;
  std::vector<int> running_operand_offset;
  size_t user_offset = 0;
  size_t operand_offset = 0;

  // compute exclusive prefix sum
  for (int i = 0; i < user_list_offsets.size(); ++i) {
    auto inst = uid_to_inst_[node_feats.uids[i]];
    size_t ucount = user_list_offsets[i];
    size_t opcount = operand_list_offsets[i];
    node_feats.num_users.push_back(ucount);
    node_feats.num_operands.push_back(opcount);
    user_list_offsets[i] = user_offset;
    operand_list_offsets[i] = operand_offset;
    running_user_offset.push_back(user_offset);
    running_operand_offset.push_back(operand_offset);
    user_offset += ucount;
    operand_offset += opcount;
  }
  user_list_offsets.push_back(user_offset);
  operand_list_offsets.push_back(operand_offset);
  user_list_indices.resize(user_offset);
  operand_list_indices.resize(operand_offset);

  // prepare column indices
  for (auto inst : inst_list) {
    // find current node's index in offsets lists
    // and insert node indices of its users and operands
    // to according indices lists.
    int uid = inst->unique_id();
    int node_idx = uid_to_node_ind_[uid];
    int empty_comp_ucount = 0;

    // Handle cases when instruction's called computation is empty.
    auto called_comps = inst->called_computations();
    for (auto u : inst->users()) {
      if (u->called_computations().empty()) {
        user_list_indices[running_user_offset[node_idx]++] =
            uid_to_node_ind_[u->unique_id()];
      }
    }
    if (called_comps.empty()) {
      for (auto u : inst->operands()) {
        operand_list_indices[running_operand_offset[node_idx]++] =
            uid_to_node_ind_[u->unique_id()];
      }
    }

    // Handle cases when instruction with called_computations:
    auto operands = inst->operands();
    if (called_comps.size() > 0) {
      for (auto cc : called_comps) {
        auto params = cc->parameter_instructions();
        // 1. make sure number of operands of current instruction equals to
        // each called_computation's parameter count. (as mentioned above they
        // should have a one-to-one mapping)
        CHECK_EQ(params.size(), operands.size());

        // 2. at each index, add comp's param as user to inst's operand
        // 3. at each index, add inst's operand as operand to comp's param
        for (int idx = 0; idx < params.size(); ++idx) {
          int operand_inst_idx = uid_to_node_ind_[operands[idx]->unique_id()];
          int param_inst_idx = uid_to_node_ind_[params[idx]->unique_id()];
          user_list_indices[running_user_offset[operand_inst_idx]++] =
              param_inst_idx;
          operand_list_indices[running_operand_offset[param_inst_idx]++] =
              operand_inst_idx;
        }
      }
      // 4. for root instruction of each computation, add curent node index
      // to their user list indices.
      for (auto cc : called_comps) {
        int cc_uid = cc->root_instruction()->unique_id();
        int cc_root_inst_idx = uid_to_node_ind_[cc_uid];

        // 5. for current node, add each computation's root instruction as its
        // operand. add current instruction as computation's root instruction's
        // user.
        user_list_indices[running_user_offset[cc_root_inst_idx]++] = node_idx;
        operand_list_indices[running_operand_offset[node_idx]++] =
            cc_root_inst_idx;
      }
    }
  }

  // add to edge features
  // uids, srcs, dsts, shapes, layouts, dtypes
  auto genuid = [](int src_uid, int dst_uid) -> int64_t {
    int64_t suid = absl::bit_cast<int>(src_uid);
    int64_t duid = absl::bit_cast<int>(dst_uid);
    return (suid << 32) | duid;
  };
  for (int i = 0; i < user_list_offsets.size() - 1; ++i) {
    size_t user_offset = user_list_offsets[i];
    size_t operand_offset = operand_list_offsets[i];
    int cur_uid = node_feats.uids[i];
    auto cur_inst = uid_to_inst_[cur_uid];

    if (cur_inst->opcode() == HloOpcode::kAlternatives) {
      alternative_indices_.push_back(i);
    }
    for (int s = user_offset; s < user_list_offsets[i + 1]; ++s) {
      int user_node_idx = user_list_indices[s];
      int user_uid = node_feats.uids[user_node_idx];
      auto user_inst = uid_to_inst_[user_uid];

      // add to out edge features
      int64_t euid = genuid(cur_uid, user_uid);
      out_edge_feats.uids.push_back(euid);
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
      out_edge_feats.dtypes.push_back(shape.element_type());
    }
    for (int s = operand_offset; s < operand_list_offsets[i + 1]; ++s) {
      int operand_node_idx = operand_list_indices[s];
      int operand_uid = node_feats.uids[operand_node_idx];
      auto operand_inst = uid_to_inst_[operand_uid];

      // add to out edge features
      int64_t euid = genuid(operand_uid, cur_uid);
      in_edge_feats.uids.push_back(euid);
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
      in_edge_feats.dtypes.push_back(shape.element_type());
    }
  }

  // final touch, add the root_index
  int entry_root_uid = m->entry_computation()->root_instruction()->unique_id();
  root_index_ = uid_to_node_ind_[entry_root_uid];

  LOG(ERROR) << "HloGraph build finished";

  return true;
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
