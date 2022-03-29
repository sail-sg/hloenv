// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_HLO_GRAPH_H_
#define ALTGRAPH_HLO_GRAPH_H_

#include <cstdio>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

// NodeFeatures holds all tensors with #node length
// uids: unique_id of HloInstruction
// names: opcode in string
// gids: sub-computation id HloInstruction belongs to
// num_users: number of outgoing edges (users and called computations)
// num_operands: number of incoming edges (operands)
// is_alternative: whether the node is kAlternative
// in_tensor_sizes: sum of input tensor sizes
// out_tensor_sizes: sum of output tensor sizes
// has_max_in_tensor: if has the largest input tensor size
// has_max_out_tensor: if has the largest output tensor size
struct NodeFeats {
  // basic node feats
  std::vector<int> uids;
  std::vector<std::string> names;
  std::vector<size_t> gids;

  // advanced node feats
  std::vector<int> num_users;
  std::vector<int> num_operands;
  std::vector<uint8_t> is_alternative;
  std::vector<int64_t> in_tensor_sizes;
  std::vector<int64_t> out_tensor_sizes;
  std::vector<uint8_t> has_max_in_tensor;
  std::vector<uint8_t> has_max_out_tensor;

  NodeFeats() {}
  // NodeFeats(const NodeFeats&) {}

  void Clear() {
    uids.clear();
    names.clear();
    gids.clear();
    num_users.clear();
    num_operands.clear();
    is_alternative.clear();
    in_tensor_sizes.clear();
    out_tensor_sizes.clear();
    has_max_in_tensor.clear();
    has_max_out_tensor.clear();
  }
};

// EdgeFeatures holds all tensors with #edge length
// uids: unique_id of both src and dst node.
// srcs: indices of source nodes
// dsts: indices of destination nodes
// dims: a fixed-length (8) array to present tensor shape
// layouts: a fixed-length (8) array to present tensor layout
// dtypes: tensor dtype
// enum PrimitiveType {
//   S16 = 0,
//   S32,
//   S64,
//   U8,
//   U16,
//   U32,
//   U64,
//   F16,
//   BF16,
//   F32,
//   F64,
//   C64,
//   C128
// };
struct EdgeFeats {
  std::vector<int64_t> uids;
  std::vector<int> srcs;
  std::vector<int> dsts;
  std::vector<int64_t> dims;
  std::vector<int64_t> layouts;
  // PrimitiveType as is defined in xla_data.proto.
  std::vector<PrimitiveType> dtypes;

  EdgeFeats() {}
  // EdgeFeats(const EdgeFeats&) {}

  void Clear() {
    uids.clear();
    srcs.clear();
    dsts.clear();
    dims.clear();
    layouts.clear();
    dtypes.clear();
  }

  int64_t GetTensorSize(size_t idx) {
    int64_t res = 1;
    for (int i = idx * 8; i < idx * 8 + 8; ++i) {
      res *= dims[i];
    }
    return abs(res);
  }
};

// A class for representing a HLO graph in the module
// To make things simpler, only string, f32, i32, and i64 are allowed as dtype.
class HloGraph {
 public:
  HloGraph() {}
  explicit HloGraph(const HloModule* m);

  bool Build(const HloModule* m);

  void Clear();

  uint64_t Hash();

  void ShowStats();

  // return CSR/CSC
  std::vector<size_t>& get_out_edge_offsets() { return user_list_offsets; }
  std::vector<size_t>& get_out_edge_indices() { return user_list_indices; }
  std::vector<size_t>& get_in_edge_offsets() { return operand_list_offsets; }
  std::vector<size_t>& get_in_edge_indices() { return operand_list_indices; }

  // return node features.
  const std::vector<int>& get_node_uids() { return node_feats.uids; }
  const std::vector<std::string>& get_node_names() { return node_feats.names; }
  const std::vector<size_t>& get_gids() { return node_feats.gids; }
  const std::vector<int>& get_user_counts() { return node_feats.num_users; }
  const std::vector<int>& get_operand_counts() {
    return node_feats.num_operands;
  }

  // return edge features.
  const std::vector<int64_t>& get_in_edge_uids() { return in_edge_feats.uids; }
  const std::vector<int>& get_in_edge_srcs() { return in_edge_feats.srcs; }
  const std::vector<int>& get_in_edge_dsts() { return in_edge_feats.dsts; }
  const std::vector<int64_t>& get_in_edge_dims() { return in_edge_feats.dims; }
  const std::vector<int64_t>& get_in_edge_layouts() {
    return in_edge_feats.layouts;
  }
  const std::vector<PrimitiveType>& get_in_edge_dtypes() {
    return in_edge_feats.dtypes;
  }

  const std::vector<int64_t>& get_out_edge_uids() {
    return out_edge_feats.uids;
  }
  const std::vector<int>& get_out_edge_srcs() { return out_edge_feats.srcs; }
  const std::vector<int>& get_out_edge_dsts() { return out_edge_feats.dsts; }
  const std::vector<int64_t>& get_out_edge_dims() {
    return out_edge_feats.dims;
  }
  const std::vector<int64_t>& get_out_edge_layouts() {
    return out_edge_feats.layouts;
  }
  const std::vector<PrimitiveType>& get_out_edge_dtypes() {
    return out_edge_feats.dtypes;
  }

  NodeFeats& get_node_feats() { return node_feats; }
  EdgeFeats& get_in_edge_feats() { return in_edge_feats; }
  EdgeFeats& get_out_edge_feats() { return out_edge_feats; }

  std::vector<int>& get_alternative_indices() { return alternative_indices_; }
  absl::flat_hash_map<int, HloInstruction*>& get_uid_to_inst() {
    return uid_to_inst_;
  }

 protected:
  void BuildGraphTopology(const HloComputation* c, int gid);
  void BuildRaggedTensors();
  void PrepareFeatures();

 private:
  HloModule* parent_hlo_module_;
  int uid_;
  std::string name_;

  std::vector<HloInstruction*> inst_list;
  absl::flat_hash_map<int, std::vector<int> > in_edge_lists;
  absl::flat_hash_map<int, std::vector<int> > out_edge_lists;
  // Use CSR to represent graph (and CSC inverse graph) topology
  std::vector<size_t> user_list_offsets;
  std::vector<size_t> user_list_indices;
  std::vector<size_t> operand_list_offsets;
  std::vector<size_t> operand_list_indices;

  // Ignore control deps for now
  // utility to lookup node and its neighbor
  absl::flat_hash_set<int> uid_set_;
  absl::flat_hash_map<int, int> uid_to_node_idx_;
  absl::flat_hash_map<int, HloInstruction*> uid_to_inst_;
  absl::flat_hash_map<int64_t, int> uid_to_in_edge_idx_;
  absl::flat_hash_map<int64_t, int> uid_to_out_edge_idx_;

  // Indices of alternative nodes
  std::vector<int> alternative_indices_;

  // index of root instruction of entry computation
  int root_index;

  // Node features
  NodeFeats node_feats;

  // Edge features
  EdgeFeats in_edge_feats;
  EdgeFeats out_edge_feats;
};

}  // namespace xla

#endif  // ALTGRAPH_HLO_GRAPH_H_
