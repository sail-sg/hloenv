// Copyright 2022 Garena Online Private Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HLOENV_PYTHON_PY_HLO_REWRITE_GRAPH_H_
#define HLOENV_PYTHON_PY_HLO_REWRITE_GRAPH_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <vector>

#include "hloenv/hlo_rewrite_graph.h"
#include "hloenv/python/py_hlo_graph.h"
#include "tensorflow/compiler/xla/tools/hlo_extractor.h"

namespace py = pybind11;

namespace hloenv {

// Extract the subgraph starting with the original/replacement instruction
// and ending with the operands of the rewrite.
std::unique_ptr<xla::HloModule> extract_subgraph(xla::HloInstruction* root_inst,
                                                 xla::Rewrite* rewrite,
                                                 bool do_verify = false) {
  const xla::HloInstructionSet& operands = rewrite->operands();
  absl::flat_hash_set<const xla::HloInstruction*> boundary(operands.begin(),
                                                           operands.end());
  xla::ExtractionVisitor visitor(*root_inst->GetModule(), &boundary);
  // xla::ExtractionVisitor visitor(*root_inst->GetModule(), nullptr);
  CHECK(root_inst->Accept(&visitor).ok());

  // The first pass may leave unused parameter instructions. Do another
  // extraction pass to remove unused parameters. This is done because
  // HloComputation does not allow removing parameters after the computation
  // has been built.
  xla::ExtractionVisitor cleanup_visitor(*visitor.module(),
                                         /*boundary=*/nullptr);
  TF_CHECK_OK(visitor.module()->entry_computation()->root_instruction()->Accept(
      &cleanup_visitor));

  if (do_verify) {
    xla::HloVerifier verifier(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/true);
    if (!verifier.Run(cleanup_visitor.module()).status().ok()) {
      std::cout << "-------------------------" << std::endl;
      std::cout << "create_order: " << rewrite->create_order() << std::endl;
      std::cout << "pass_name: " << rewrite->pass_name() << std::endl;
      std::cout << "original: " << rewrite->original()->name() << std::endl;
      std::cout << "replacement: " << rewrite->replacement()->name()
                << std::endl;
      std::cout << "users: " << std::endl;
      for (auto* user : rewrite->users()) {
        std::cout << "    " << user->name() << std::endl;
      }
      std::cout << "affected edges: " << std::endl;
      for (auto const& pair : rewrite->affected_edges()) {
        std::cout << "    [" << pair.first->name() << "]->["
                  << pair.second->name() << "]" << std::endl;
      }
      std::cout << "-------------------------" << std::endl;
      std::cout << visitor.module()->ToString() << std::endl;
      std::cout << "-------------------------" << std::endl;
    }
    TF_CHECK_OK(verifier.Run(cleanup_visitor.module()).status());
  }

  return cleanup_visitor.ConsumeModule();
}

class PyRewrite {
 public:
  explicit PyRewrite(xla::Rewrite* rewrite) {
    rewrite_ = rewrite;
    pass_name_ = rewrite_->pass_name();

    for (auto* instruction : rewrite_->affected_insts()) {
      orig_subgraph_uids_.push_back(instruction->orig_unique_id());
    }
  }

  const std::string& pass_name() const { return pass_name_; }

  std::shared_ptr<PyHloGraph> orig_subgraph() {
    if (orig_hlo_subgraph_ == nullptr) {
      orig_hlo_module_ = extract_subgraph(rewrite_->original(), rewrite_);
      orig_hlo_module_->Prune();
      orig_hlo_module_->Cleanup();
      orig_hlo_subgraph_ = std::make_shared<PyHloGraph>(orig_hlo_module_.get(),
                                                        false, false, false);
    }
    return orig_hlo_subgraph_;
  }
  std::shared_ptr<PyHloGraph> replacement_subgraph() {
    if (replacement_hlo_subgraph_ == nullptr) {
      replacement_hlo_module_ =
          extract_subgraph(rewrite_->replacement(), rewrite_);
      replacement_hlo_module_->Prune();
      replacement_hlo_module_->Cleanup();
      replacement_hlo_subgraph_ = std::make_shared<PyHloGraph>(
          replacement_hlo_module_.get(), false, false, false);
    }
    return replacement_hlo_subgraph_;
  }

  std::string orig_subgraph_str() {
    orig_subgraph();
    return orig_hlo_module_->ToString();
  }

  const std::vector<int>& orig_subgraph_uids() { return orig_subgraph_uids_; }

  std::string replacement_subgraph_str() {
    replacement_subgraph();
    return replacement_hlo_module_->ToString();
  }

  int order_idx() { return rewrite_->create_order(); }

 private:
  std::shared_ptr<PyHloGraph> orig_hlo_subgraph_;
  std::shared_ptr<PyHloGraph> replacement_hlo_subgraph_;
  std::unique_ptr<xla::HloModule> orig_hlo_module_;
  std::unique_ptr<xla::HloModule> replacement_hlo_module_;
  std::vector<int> orig_subgraph_uids_;
  xla::Rewrite* rewrite_;
  std::string pass_name_;
};

class PyHloRewriteGraph : public HloRewriteGraph {
 public:
  explicit PyHloRewriteGraph(std::shared_ptr<AltHloModule> alt_hlo_module)
      : PyHloRewriteGraph(alt_hlo_module->hlo_module_ptr()) {}

  explicit PyHloRewriteGraph(xla::HloModule* hlo_module)
      : HloRewriteGraph(hlo_module) {
    for (auto* rewrite : this->rewrites()) {
      py_rewrites_.push_back(std::make_shared<PyRewrite>(rewrite));
    }
  }

  // TODO(ohcy): optimize here, don't return copy
  std::vector<std::shared_ptr<PyRewrite>> rewrite_data() {
    return py_rewrites_;
  }

 private:
  std::vector<std::shared_ptr<PyRewrite>> py_rewrites_;
};

}  // namespace hloenv

#endif  // HLOENV_PYTHON_PY_HLO_REWRITE_GRAPH_H_
