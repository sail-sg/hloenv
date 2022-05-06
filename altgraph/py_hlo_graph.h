// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PY_HLO_GRAPH_H_
#define ALTGRAPH_PY_HLO_GRAPH_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "altgraph/hlo_graph.h"

namespace py = pybind11;

// Capsule stores a copy of the shared_ptr to the data
// will delete the shared_ptr when it goes out of scope in python
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(
    std::shared_ptr<Sequence> s_ptr) {
  return py::array_t<typename Sequence::value_type>{
      s_ptr->size(), s_ptr->data(),
      py::capsule(
          new auto(s_ptr),  // <- can leak
          [](void* p) { delete reinterpret_cast<decltype(s_ptr)*>(p); })};
}

#define SHARED_VEC_TO_PYARRAY(NAME, TYPE, SHARED_PTR) \
  py::array_t<TYPE> py_get_##NAME() { return as_pyarray(SHARED_PTR); }

#define DEF_PYBIND_READONLY(CLASS, NAME) \
  def_property_readonly(#NAME, &CLASS::py_get_##NAME)

// TODO(ohcy) -> shift HloGraph out from xla::namespace?
struct PyNodeFeats : public xla::NodeFeats {
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_READONLY macro
  SHARED_VEC_TO_PYARRAY(uids, int, uids)
  SHARED_VEC_TO_PYARRAY(gids, size_t, gids)
  SHARED_VEC_TO_PYARRAY(num_users, int, num_users)
  SHARED_VEC_TO_PYARRAY(num_operands, int, num_operands)
  SHARED_VEC_TO_PYARRAY(opcodes, int, opcodes)
  SHARED_VEC_TO_PYARRAY(opcode_attrs, int, opcode_attrs)
  SHARED_VEC_TO_PYARRAY(num_opcode_attrs, int, num_opcode_attrs)
  SHARED_VEC_TO_PYARRAY(is_alternative, uint8_t, is_alternative)
  SHARED_VEC_TO_PYARRAY(in_tensor_sizes, int64_t, in_tensor_sizes)
  SHARED_VEC_TO_PYARRAY(out_tensor_sizes, int64_t, out_tensor_sizes)
  SHARED_VEC_TO_PYARRAY(has_max_in_tensor, uint8_t, has_max_in_tensor)
  SHARED_VEC_TO_PYARRAY(has_max_out_tensor, uint8_t, has_max_out_tensor)
  std::vector<std::string>& py_get_names() { return *names; }

  PyNodeFeats() {}
  explicit PyNodeFeats(const xla::NodeFeats& node_feats)
      : xla::NodeFeats(node_feats) {}
};

struct PyEdgeFeats : public xla::EdgeFeats {
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_READONLY macro
  SHARED_VEC_TO_PYARRAY(uids, int64_t, uids);
  SHARED_VEC_TO_PYARRAY(srcs, int, srcs);
  SHARED_VEC_TO_PYARRAY(dsts, int, dsts);
  SHARED_VEC_TO_PYARRAY(dims, int64_t, dims);
  SHARED_VEC_TO_PYARRAY(layouts, int64_t, layouts);
  SHARED_VEC_TO_PYARRAY(dtypes, int, dtypes);

  PyEdgeFeats() {}
  explicit PyEdgeFeats(const xla::EdgeFeats& edge_feats)
      : xla::EdgeFeats(edge_feats) {}
};

class PyHloGraph : public xla::HloGraph {
 public:
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_READONLY macro
  SHARED_VEC_TO_PYARRAY(out_edge_offsets, size_t, get_out_edge_offsets_ptr());
  SHARED_VEC_TO_PYARRAY(out_edge_indices, size_t, get_out_edge_indices_ptr());
  SHARED_VEC_TO_PYARRAY(in_edge_offsets, size_t, get_in_edge_offsets_ptr());
  SHARED_VEC_TO_PYARRAY(in_edge_indices, size_t, get_in_edge_indices_ptr());
  SHARED_VEC_TO_PYARRAY(opcode_attr_counts, int, get_opcode_attr_counts_ptr());
  SHARED_VEC_TO_PYARRAY(alternative_indices, int,
                        get_alternative_indices_ptr());
  int py_get_graph_load_errors() { return graph_load_errors_; }

  PyNodeFeats node_features_;
  PyEdgeFeats in_edge_features_;
  PyEdgeFeats out_edge_features_;

  PyHloGraph() {}
  explicit PyHloGraph(const xla::HloModule* m, bool do_hash_verification = true)
      : xla::HloGraph(m, do_hash_verification) {
    node_features_ = PyNodeFeats(get_node_feats());
    in_edge_features_ = PyEdgeFeats(get_in_edge_feats());
    out_edge_features_ = PyEdgeFeats(get_out_edge_feats());
  }

  PyNodeFeats& py_get_node_features() { return node_features_; }
  PyEdgeFeats& py_get_in_edge_features() { return in_edge_features_; }
  PyEdgeFeats& py_get_out_edge_features() { return out_edge_features_; }

  uint64_t py_hash() { return Hash(); }
};

#endif  // ALTGRAPH_PY_HLO_GRAPH_H_
