// Copyright 2021 Garena Online Private Limited

#ifndef HLOENV_HLO_REWRITE_GRAPH_H_
#define HLOENV_HLO_REWRITE_GRAPH_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace py = pybind11;

namespace hloenv {

// A class for representing all rewrite plans in a HloModule
class HloRewriteGraph {
 public:
  explicit HloRewriteGraph(xla::HloModule* hlo_module);

  bool ApplyAllRewritesDebug();

  std::vector<std::pair<int, xla::RewriteStatus>> ApplyRewrites(
      py::array_t<size_t> decisions);

  xla::RewriteStatus ApplyRewrite(int id);

  int NumRewrites() { return rewrites_.size(); }

  void Log();

  std::vector<xla::Rewrite*>& rewrites() { return rewrites_; }

  int num_rewrites() { return rewrites_.size(); }

  // TODO(ohcy): optimize here, don't return copy
  std::vector<std::vector<bool>> adjacency_matrix() {
    return adjacency_matrix_;
  }

 private:
  xla::HloModule* hlo_module_;
  std::vector<xla::Rewrite*> rewrites_;
  std::vector<bool> adjacent_rewrite_ids_;
  std::vector<bool> applied_;
  std::vector<std::vector<bool>> adjacency_matrix_;

  std::unordered_map<xla::Rewrite*, int> rewrite_id_map_;
};

}  // namespace hloenv

#endif  // HLOENV_HLO_REWRITE_GRAPH_H_
