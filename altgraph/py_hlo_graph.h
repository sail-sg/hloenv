// Copyright 2021 Garena Online Private Limited

#ifndef TENSORFLOW_XLA_STANDALONE_PY_HLO_GRAPH_H_
#define TENSORFLOW_XLA_STANDALONE_PY_HLO_GRAPH_H_

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
  return py::array_t<size_t>{
      s_ptr->size(), s_ptr->data(),
      py::capsule(
          new auto(s_ptr),  // <- can leak
          [](void* p) { delete reinterpret_cast<decltype(s_ptr)*>(p); })};
}

class PyHloGraph : public xla::HloGraph {
  std::shared_ptr<std::vector<size_t>> out_edge_offsets_;
  std::shared_ptr<std::vector<size_t>> out_edge_indices_;
  std::shared_ptr<std::vector<size_t>> in_edge_offsets_;
  std::shared_ptr<std::vector<size_t>> in_edge_indices_;

 public:
  PyHloGraph() {}
  explicit PyHloGraph(const xla::HloModule* m) : xla::HloGraph(m) {
    out_edge_offsets_ = std::make_shared<std::vector<size_t>>(
        std::move(get_out_edge_offsets()));
    out_edge_indices_ = std::make_shared<std::vector<size_t>>(
        std::move(get_out_edge_indices()));
    in_edge_offsets_ =
        std::make_shared<std::vector<size_t>>(std::move(get_in_edge_offsets()));
    in_edge_indices_ =
        std::make_shared<std::vector<size_t>>(std::move(get_in_edge_indices()));
  }

  py::array_t<size_t> py_get_out_edge_offsets();
  py::array_t<size_t> py_get_out_edge_indices();
  py::array_t<size_t> py_get_in_edge_offsets();
  py::array_t<size_t> py_get_in_edge_indices();

  xla::NodeFeats& py_get_node_feats() { return get_node_feats(); }
  xla::EdgeFeats& py_get_in_edge_feats() { return get_in_edge_feats(); }
  xla::EdgeFeats& py_get_out_edge_feats() { return get_out_edge_feats(); }

  std::vector<int>& py_get_alternative_indices() {
    return get_alternative_indices();
  }
};

#endif  // TENSORFLOW_XLA_STANDALONE_PY_HLO_GRAPH_H_
