// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PY_HLO_GRAPH_H_
#define ALTGRAPH_PY_HLO_GRAPH_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
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

#define VECTOR_TO_PYARRAY(NAME, TYPE)         \
  std::shared_ptr<std::vector<TYPE>> NAME##_;    \
  py::array_t<TYPE> py_get_##NAME() { return as_pyarray(NAME##_); }

#define MAKE_VECTOR_SHARED(NAME, TYPE, VECTOR) \
  NAME##_ = std::make_shared<std::vector<TYPE>>(VECTOR)

#define DEF_PYBIND_READONLY(CLASS, NAME) \
  def_property_readonly(#NAME, &CLASS::py_get_##NAME)

// TODO(ohcy) -> shift HloGraph out from xla::namespace?
// TODO(ohcy) -> Should move this to NodeFeats, though for now leave HloGraph
// be first till changes to it are finalized.
struct PyNodeFeats : public xla::NodeFeats {
  // Note, the shared_pointers must be initialized in constructor with MAKE_VECTOR_SHARED
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_GETTER macro
  VECTOR_TO_PYARRAY(uids, int);
  VECTOR_TO_PYARRAY(gids, size_t);
  VECTOR_TO_PYARRAY(num_users, int);
  VECTOR_TO_PYARRAY(num_operands, int);
  VECTOR_TO_PYARRAY(is_alternative, uint8_t);
  VECTOR_TO_PYARRAY(in_tensor_sizes, int64_t);
  VECTOR_TO_PYARRAY(out_tensor_sizes, int64_t);
  VECTOR_TO_PYARRAY(has_max_in_tensor, uint8_t);
  VECTOR_TO_PYARRAY(has_max_out_tensor, uint8_t);
  std::vector<std::string>& py_get_names() { return names; };

  PyNodeFeats() { }
  PyNodeFeats(xla::NodeFeats && obj) : xla::NodeFeats(std::move(obj)) {
    MAKE_VECTOR_SHARED(uids, int, uids);
    MAKE_VECTOR_SHARED(gids, size_t, gids);
    MAKE_VECTOR_SHARED(num_users, int, num_users);
    MAKE_VECTOR_SHARED(num_operands, int, num_operands);
    MAKE_VECTOR_SHARED(is_alternative, uint8_t, is_alternative);
    MAKE_VECTOR_SHARED(in_tensor_sizes, int64_t, in_tensor_sizes);
    MAKE_VECTOR_SHARED(out_tensor_sizes, int64_t, out_tensor_sizes);
    MAKE_VECTOR_SHARED(has_max_in_tensor, uint8_t, has_max_in_tensor);
    MAKE_VECTOR_SHARED(has_max_out_tensor, uint8_t, has_max_out_tensor);
  }
};

// TODO(ohcy) -> Should move this to EdgeFeats, though for now leave HloGraph
// be first till changes to it are finalized.
struct PyEdgeFeats : public xla::EdgeFeats {
  // Note, the shared_pointers must be initialized in constructor with MAKE_VECTOR_SHARED
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_GETTER macro
  VECTOR_TO_PYARRAY(uids, int64_t);
  VECTOR_TO_PYARRAY(srcs, int);
  VECTOR_TO_PYARRAY(dsts, int);
  VECTOR_TO_PYARRAY(dims, int64_t);
  VECTOR_TO_PYARRAY(layouts, int64_t);
  std::vector<xla::PrimitiveType>& py_get_dtypes() { return dtypes; };

  PyEdgeFeats() { }
  PyEdgeFeats(xla::EdgeFeats && obj) : xla::EdgeFeats(std::move(obj)) {
    MAKE_VECTOR_SHARED(uids, int64_t, uids);
    MAKE_VECTOR_SHARED(srcs, int, srcs);
    MAKE_VECTOR_SHARED(dsts, int, dsts);
    MAKE_VECTOR_SHARED(dims, int64_t, dims);
    MAKE_VECTOR_SHARED(layouts, int64_t, layouts);
  }
};

class PyHloGraph : public xla::HloGraph {

 public:

  // Note, the shared_pointers must be initialized in constructor with MAKE_VECTOR_SHARED
  // Expose to pybind interface in py_hlo_ir.cc using DEF_PYBIND_GETTER macro
  VECTOR_TO_PYARRAY(out_edge_offsets, size_t);
  VECTOR_TO_PYARRAY(out_edge_indices, size_t);
  VECTOR_TO_PYARRAY(in_edge_offsets, size_t);
  VECTOR_TO_PYARRAY(in_edge_indices, size_t);
  VECTOR_TO_PYARRAY(alternative_indices, int);

  PyNodeFeats node_features_;
  PyEdgeFeats in_edge_features_;
  PyEdgeFeats out_edge_features_;

  PyHloGraph() { }
  explicit PyHloGraph(const xla::HloModule* m) : xla::HloGraph(m) {
    // Note, this has to be aligned with VECTOR_TO_PYARRAY
    MAKE_VECTOR_SHARED(out_edge_offsets, size_t, get_out_edge_offsets());
    MAKE_VECTOR_SHARED(out_edge_indices, size_t, get_out_edge_indices());
    MAKE_VECTOR_SHARED(in_edge_offsets, size_t, get_in_edge_offsets());
    MAKE_VECTOR_SHARED(in_edge_indices, size_t, get_in_edge_indices());
    MAKE_VECTOR_SHARED(alternative_indices, int, get_alternative_indices());

    node_features_ = PyNodeFeats(std::move(get_node_feats()));
    in_edge_features_ = PyEdgeFeats(std::move(get_in_edge_feats()));
    out_edge_features_ = PyEdgeFeats(std::move(get_out_edge_feats()));
  }

  // py::array_t<size_t> py_get_out_edge_offsets();
  // py::array_t<size_t> py_get_out_edge_indices();
  // py::array_t<size_t> py_get_in_edge_offsets();
  // py::array_t<size_t> py_get_in_edge_indices();

  PyNodeFeats& py_get_node_features() { return node_features_; }
  PyEdgeFeats& py_get_in_edge_features() { return in_edge_features_; }
  PyEdgeFeats& py_get_out_edge_features() { return out_edge_features_; }

  uint64_t py_hash() { return Hash(); }
};

#endif  // ALTGRAPH_PY_HLO_GRAPH_H_
