// Copyright 2021 Garena Online Private Limited

#ifndef HLOENV_HLO_GRAPH_H_
#define HLOENV_HLO_GRAPH_H_

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace hloenv {

// NodeFeatures holds all tensors with #node length
// uids: unique_id of HloInstruction
// names: opcode in string
// gids: sub-computation id HloInstruction belongs to
// fused_comp_ids: called-computation id instruction points to
//                 0: not pointing to a called-computation
//                 other positive integers: fused computation id
// num_users: number of outgoing edges (users and called computations)
// num_operands: number of incoming edges (operands)
// opcodes: enumerator of opcodes in HLO
// opcode_attrs: integers that represent opcode's attributes
// num_opcode_attrs: two consecutive integers represent count
// of integer attrs and count of enumerator attrs.
// is_alternative: whether the node is kAlternative
// is_in_fusion: whether the node is in a fused computation
// in_tensor_sizes: sum of input tensor sizes
// out_tensor_sizes: sum of output tensor sizes
// has_max_in_tensor: if has the largest input tensor size
// has_max_out_tensor: if has the largest output tensor size
// normalized_num_group_inst: computed as 1/#instruction in the group.
struct NodeFeats {
  // basic node feats
  std::shared_ptr<std::vector<int>> uids;
  std::shared_ptr<std::vector<std::string>> names;
  std::shared_ptr<std::vector<size_t>> gids;
  std::shared_ptr<std::vector<size_t>> fused_comp_ids;

  // advanced node feats
  std::shared_ptr<std::vector<int>> num_users;
  std::shared_ptr<std::vector<int>> num_operands;
  // opcode enum is defined here:
  // https://git.insea.io/sail/aisys/tensorflow/-/blob/main/tensorflow/compiler/xla/service/hlo_opcode.h#L46
  std::shared_ptr<std::vector<int>> opcodes;
  std::shared_ptr<std::vector<int>> opcode_attrs;
  std::shared_ptr<std::vector<int>> num_opcode_attrs;
  std::shared_ptr<std::vector<uint8_t>> is_alternative;
  std::shared_ptr<std::vector<uint8_t>> is_in_fusion;
  std::shared_ptr<std::vector<int64_t>> in_tensor_sizes;
  std::shared_ptr<std::vector<int64_t>> out_tensor_sizes;
  std::shared_ptr<std::vector<uint8_t>> has_max_in_tensor;
  std::shared_ptr<std::vector<uint8_t>> has_max_out_tensor;
  std::shared_ptr<std::vector<float>> normalized_num_group_inst;

  NodeFeats() {
    uids = std::make_shared<std::vector<int>>();
    names = std::make_shared<std::vector<std::string>>();
    gids = std::make_shared<std::vector<size_t>>();
    num_users = std::make_shared<std::vector<int>>();
    num_operands = std::make_shared<std::vector<int>>();
    opcodes = std::make_shared<std::vector<int>>();
    opcode_attrs = std::make_shared<std::vector<int>>();
    num_opcode_attrs = std::make_shared<std::vector<int>>();
    is_alternative = std::make_shared<std::vector<uint8_t>>();
    is_in_fusion = std::make_shared<std::vector<uint8_t>>();
    in_tensor_sizes = std::make_shared<std::vector<int64_t>>();
    out_tensor_sizes = std::make_shared<std::vector<int64_t>>();
    has_max_in_tensor = std::make_shared<std::vector<uint8_t>>();
    has_max_out_tensor = std::make_shared<std::vector<uint8_t>>();
    normalized_num_group_inst = std::make_shared<std::vector<float>>();
  }

  void Clear() {
    uids->clear();
    names->clear();
    gids->clear();
    num_users->clear();
    num_operands->clear();
    opcodes->clear();
    opcode_attrs->clear();
    num_opcode_attrs->clear();
    is_alternative->clear();
    is_in_fusion->clear();
    in_tensor_sizes->clear();
    out_tensor_sizes->clear();
    has_max_in_tensor->clear();
    has_max_out_tensor->clear();
    normalized_num_group_inst->clear();
  }
};

// EdgeFeatures holds all tensors with #edge length
// uids: unique_id of both src and dst node.
// srcs: indices of source nodes
// dsts: indices of destination nodes
// dims: a fixed-length (8) array to present tensor shape
// layouts: a fixed-length (8) array to present tensor layout
// lehmercodes: a fixed-length (8) array to present lehmer code of layout
// dtypes: tensor dtype, integer mapping to xla_data primitivetype:
// enum PrimitiveType {
//   PRIMITIVE_TYPE_INVALID = 0,
//   PRED,
//   S8,
//   S16,
//   S32,
//   S64,
//   U8,
//   U16,
//   U32,
//   U64,
//   F16,
//   F32,
//   F64,
//   TUPLE,
//   OPAQUE_TYPE,
//   C64,
//   BF16,
//   TOKEN,
//   C128
// };
struct EdgeFeats {
  std::shared_ptr<std::vector<int64_t>> uids;
  std::shared_ptr<std::vector<int>> srcs;
  std::shared_ptr<std::vector<int>> dsts;
  std::shared_ptr<std::vector<int64_t>> dims;
  std::shared_ptr<std::vector<int64_t>> layouts;
  std::shared_ptr<std::vector<int64_t>> lehmercodes;
  std::shared_ptr<std::vector<uint8_t>> types;
  // PrimitiveType as is defined in xla_data.proto.
  std::shared_ptr<std::vector<int>> dtypes;

  EdgeFeats() {
    uids = std::make_shared<std::vector<int64_t>>();
    srcs = std::make_shared<std::vector<int>>();
    dsts = std::make_shared<std::vector<int>>();
    dims = std::make_shared<std::vector<int64_t>>();
    layouts = std::make_shared<std::vector<int64_t>>();
    lehmercodes = std::make_shared<std::vector<int64_t>>();
    types = std::make_shared<std::vector<uint8_t>>();
    dtypes = std::make_shared<std::vector<int>>();
  }

  void Clear() {
    uids->clear();
    srcs->clear();
    dsts->clear();
    dims->clear();
    layouts->clear();
    lehmercodes->clear();
    types->clear();
    dtypes->clear();
  }

  int64_t GetTensorSize(size_t idx) {
    int64_t res = 1;
    for (int i = idx * 8; i < idx * 8 + 8; ++i) {
      res *= dims->at(i);
    }
    return abs(res);
  }
};

// A class for representing a HLO graph in the module
// To make things simpler, only string, f32, i32, and i64 are allowed as dtype.
class HloGraph {
 public:
  HloGraph() : kNumOpcodes(xla::HloOpcodeCount()) {}
  explicit HloGraph(const xla::HloModule* m, bool debug = false,
                    bool inline_fused_comp = false,
                    bool do_hash_verification = false);

  bool Build(const xla::HloModule* m, bool debug = false,
             bool inline_fused_comp = false, bool do_hash_verification = false);

  void Clear();

  uint64_t Hash();

  // return CSR/CSC
  std::shared_ptr<std::vector<size_t>> get_out_edge_offsets_ptr() {
    return user_list_offsets_;
  }
  std::shared_ptr<std::vector<size_t>> get_out_edge_indices_ptr() {
    return user_list_indices_;
  }
  std::shared_ptr<std::vector<size_t>> get_in_edge_offsets_ptr() {
    return operand_list_offsets_;
  }
  std::shared_ptr<std::vector<size_t>> get_in_edge_indices_ptr() {
    return operand_list_indices_;
  }
  std::shared_ptr<std::vector<int>> get_opcode_attr_counts_ptr() {
    return opcode_attr_counts_;
  }

  // return node features.
  const std::vector<int>& get_node_uids() { return *node_feats_.uids; }
  const std::vector<std::string>& get_node_names() {
    return *node_feats_.names;
  }
  const std::vector<size_t>& get_gids() { return *node_feats_.gids; }
  const std::vector<int>& get_user_counts() { return *node_feats_.num_users; }
  const std::vector<int>& get_operand_counts() {
    return *node_feats_.num_operands;
  }
  const std::vector<int>& get_opcodes() { return *node_feats_.opcodes; }
  const std::vector<int>& get_opcode_attrs() {
    return *node_feats_.opcode_attrs;
  }
  const std::vector<int>& get_num_opcode_attrs() {
    return *node_feats_.num_opcode_attrs;
  }

  // return edge features.
  const std::vector<int64_t>& get_in_edge_uids() {
    return *in_edge_feats_.uids;
  }
  const std::vector<int>& get_in_edge_srcs() { return *in_edge_feats_.srcs; }
  const std::vector<int>& get_in_edge_dsts() { return *in_edge_feats_.dsts; }
  const std::vector<int64_t>& get_in_edge_dims() {
    return *in_edge_feats_.dims;
  }
  const std::vector<int64_t>& get_in_edge_layouts() {
    return *in_edge_feats_.layouts;
  }
  const std::vector<int64_t>& get_in_edge_lehmercodes() {
    return *in_edge_feats_.lehmercodes;
  }
  const std::vector<int>& get_in_edge_dtypes() {
    return *in_edge_feats_.dtypes;
  }

  const std::vector<int64_t>& get_out_edge_uids() {
    return *out_edge_feats_.uids;
  }
  const std::vector<int>& get_out_edge_srcs() { return *out_edge_feats_.srcs; }
  const std::vector<int>& get_out_edge_dsts() { return *out_edge_feats_.dsts; }
  const std::vector<int64_t>& get_out_edge_dims() {
    return *out_edge_feats_.dims;
  }
  const std::vector<int64_t>& get_out_edge_layouts() {
    return *out_edge_feats_.layouts;
  }
  const std::vector<int64_t>& get_out_edge_lehmercodes() {
    return *out_edge_feats_.lehmercodes;
  }
  const std::vector<int>& get_out_edge_dtypes() {
    return *out_edge_feats_.dtypes;
  }

  const NodeFeats& get_node_feats() { return node_feats_; }
  const EdgeFeats& get_in_edge_feats() { return in_edge_feats_; }
  const EdgeFeats& get_out_edge_feats() { return out_edge_feats_; }

  std::shared_ptr<std::vector<int>> get_alternative_indices_ptr() {
    return alternative_indices_;
  }
  absl::flat_hash_map<int, xla::HloInstruction*>& get_uid_to_inst() {
    return uid_to_inst_;
  }

 protected:
  // For each computation, build in/out edge lists for all instructions.
  void BuildGraphTopology(const xla::HloComputation* c, int gid,
                          bool is_fusion_comp);

  // Inlining all fused computations into entry computation.
  void FusedComputationInlining();

  // Fill content of each in/out edge lists according to topology.
  void BuildRaggedTensors(
      const absl::flat_hash_map<xla::HloComputation*, int>& comp_id_map);

  // Fill the rest of contents in node/edge features.
  void PrepareFeatures();

  int graph_load_errors_;

 private:
  // Internal function call to set fused_comp_ids for each instruction.
  void SetFusedCompId(
      const absl::flat_hash_map<xla::HloComputation*, int>& comp_id_map);

  // Internal function that returns indices of instructions from one specific
  // fused computation.
  std::vector<int> FindInstIndicesOfFusedComp(int fused_comp_id);

  // Internal function that generate attribute count for each opcode.
  void GenOpcodeAttrCounts();

  bool HasCycleForward_(int idx, std::vector<bool>* visited,
                        std::vector<bool>* stack);
  bool HasCycleBackward_(int idx, std::vector<bool>* visited,
                         std::vector<bool>* stack);

  bool HasCycle();

  bool ShouldInline(int inst_idx, xla::HloInstruction* inst);

  const int kNumOpcodes;
  xla::HloModule* parent_hlo_module_;
  int uid_;
  std::string name_;

  std::vector<xla::HloInstruction*> inst_list_;
  absl::flat_hash_map<int, std::vector<int>> in_edge_lists_;
  absl::flat_hash_map<int, std::vector<int>> out_edge_lists_;
  absl::flat_hash_map<int, std::vector<int>> called_comp_lists_;

  // Ignore control deps for now
  // utility to lookup node and its neighbor
  absl::flat_hash_map<int, int> uid_to_node_idx_;
  absl::flat_hash_map<int, xla::HloInstruction*> uid_to_inst_;
  absl::flat_hash_map<int64_t, int> uid_to_in_edge_idx_;
  absl::flat_hash_map<int64_t, int> uid_to_out_edge_idx_;

  // Use CSR to represent graph (and CSC inverse graph) topology
  std::shared_ptr<std::vector<size_t>> user_list_offsets_;
  std::shared_ptr<std::vector<size_t>> user_list_indices_;
  std::shared_ptr<std::vector<size_t>> operand_list_offsets_;
  std::shared_ptr<std::vector<size_t>> operand_list_indices_;
  std::shared_ptr<std::vector<int>> opcode_attr_counts_;

  // Indices of alternative nodes
  std::shared_ptr<std::vector<int>> alternative_indices_;

  // index of root instruction of entry computation
  int root_index_;

  // Node features
  NodeFeats node_feats_;

  // Edge features
  EdgeFeats in_edge_feats_;
  EdgeFeats out_edge_feats_;
};

}  // namespace hloenv

#endif  // HLOENV_HLO_GRAPH_H_
