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

#include "hloenv/hlo_rewrite_graph.h"

#include "absl/base/casts.h"
#include "hloenv/utils/hlo_utils.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/types.h"

namespace hloenv {

bool AreDifferentGteOfSameMultiOutputFusion(xla::HloInstruction* left,
                                            xla::HloInstruction* right) {
  if (left == right) {
    return false;
  }
  if (left->opcode() == xla::HloOpcode::kGetTupleElement &&
      right->opcode() == xla::HloOpcode::kGetTupleElement) {
    if (left->operand(0) == right->operand(0) &&
        left->operand(0)->IsMultiOutputFusion()) {
      return true;
    }
  }
  return false;
}

HloRewriteGraph::HloRewriteGraph(xla::HloModule* hlo_module) {
  int curr_rewrite_id = 0;
  hlo_module_ = hlo_module;

  xla::HloInstructionPairMap<std::set<xla::Rewrite*>> edge_pair_to_rewrite_map;
  for (auto* hlo_computation : hlo_module->MakeComputationPostOrder()) {
    for (auto* hlo_instruction : hlo_computation->MakeInstructionPostOrder()) {
      for (auto* rewrite : hlo_instruction->rewrite_plans()) {
        if (rewrite_id_map_.find(rewrite) == rewrite_id_map_.end()) {
          rewrite_id_map_[rewrite] = curr_rewrite_id;
          rewrites_.push_back(rewrite);
          curr_rewrite_id++;
          CHECK_EQ(curr_rewrite_id, rewrites_.size());
        }

        for (auto edge_pair : rewrite->affected_edges()) {
          edge_pair_to_rewrite_map[edge_pair].insert(rewrite);
        }
      }
    }
  }

  // TODO(ohcy) testing if this is neccessary, clean up overlap with above if
  // it is not.
  // Add a sorting step to sort according to create order
  sort(rewrites_.begin(), rewrites_.end(),
       [](const xla::Rewrite* lhs, const xla::Rewrite* rhs) {
         return lhs->create_order() < rhs->create_order();
       });
  curr_rewrite_id = 0;
  for (auto* rewrite : rewrites_) {
    rewrite_id_map_[rewrite] = curr_rewrite_id;
    curr_rewrite_id++;
  }

  adjacent_rewrite_ids_.resize(curr_rewrite_id, false);
  applied_.resize(curr_rewrite_id, false);
  adjacency_matrix_.resize(curr_rewrite_id,
                           std::vector<bool>(curr_rewrite_id, false));
  for (auto const& rewrites_it : edge_pair_to_rewrite_map) {
    const std::set<xla::Rewrite*>& shared_rewrites = rewrites_it.second;
    for (std::set<xla::Rewrite*>::const_iterator iter1 =
             shared_rewrites.begin();
         iter1 != shared_rewrites.end(); ++iter1) {
      for (std::set<xla::Rewrite*>::const_iterator iter2 = iter1;
           iter2 != shared_rewrites.end(); ++iter2) {        
        xla::Rewrite* rewrite_a = *iter1;
        xla::Rewrite* rewrite_b = *iter2;
        // Handle special case of tuple outputs since those will always
        // share affected edges, but both should be able to be applied.
        if (!AreDifferentGteOfSameMultiOutputFusion(rewrite_a->original(),
                                                    rewrite_b->original())) {
          int rewrite_a_id = rewrite_id_map_[rewrite_a];
          int rewrite_b_id = rewrite_id_map_[rewrite_b];
          adjacency_matrix_[rewrite_a_id][rewrite_b_id] = true;
          adjacency_matrix_[rewrite_b_id][rewrite_a_id] = true;
        }
      }
    }
  }
}

void HloRewriteGraph::Log() {
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Logging HloRewriteGraph" << std::endl;
  std::cout << "Num rewrites: " << NumRewrites() << std::endl;
  int idx = 0;
  for (auto* rewrite : rewrites_) {
    std::cout << "-------------------------" << std::endl;
    std::cout << "idx: " << idx++ << std::endl;
    std::cout << "idx (debug): " << rewrite_id_map_[rewrite] << std::endl;
    std::cout << "create_order: " << rewrite->create_order() << std::endl;
    std::cout << "pass_name: " << rewrite->pass_name() << std::endl;
    std::cout << "original: " << rewrite->original()->name() << std::endl;
    std::cout << "replacement: " << rewrite->replacement()->name() << std::endl;
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
  }
  std::cout << "--------------------------------------------------------"
            << std::endl;
}

bool HloRewriteGraph::ApplyAllRewritesDebug() {
  int num_rewrites_applied = 0;
  bool rewrite_applied = true;
  while (rewrite_applied) {
    rewrite_applied = false;
    for (int i = 0; i < this->NumRewrites(); i++) {
      if (this->ApplyRewrite(i) == xla::RewriteStatus::OK) {
        rewrite_applied = true;
        num_rewrites_applied++;
        break;
      }
    }
  }
  hlo_module_->Prune();
  hlo_module_->RemoveUnusedComputations();
  hlo_module_->RewriteCleanup();
  hlo_module_->Cleanup();

  return num_rewrites_applied > 0;
}

std::vector<std::pair<int, xla::RewriteStatus>> HloRewriteGraph::ApplyRewrites(
    py::array_t<size_t> decisions) {
  py::buffer_info decisions_buf = decisions.request();

  std::vector<std::pair<int, xla::RewriteStatus>> results;

  size_t* decisions_ptr = static_cast<size_t*>(decisions_buf.ptr);
  int num_decisions = decisions_buf.shape[0];

  if (num_decisions > this->NumRewrites()) {
    LOG(FATAL) << "Decisions length [" << num_decisions << "] > num rewrites ["
               << this->NumRewrites() << "] length!";
  }

  for (size_t decisions_idx = 0; decisions_idx < num_decisions;
       decisions_idx++) {
    size_t rewrite_idx = decisions_ptr[decisions_idx];
    xla::RewriteStatus status = ApplyRewrite(rewrite_idx);
    results.push_back(std::make_pair(rewrite_idx, status));
  }
  // At this point, we check if the replacement instructions are still in the
  // graph and set the status to PRUNED if not.
  // We do this only for rewrites that were succesfully applied
  hlo_module_->DetachRewriteInstructions();
  hlo_module_->RemoveUnusedComputations();
  hlo_module_->Prune();

  // At this point, we check if the replacement instructions are still in the
  // graph and set the status to PRUNED if not.
  // We do this only for rewrites that were succesfully applied
  for (auto& result : results) {
    size_t idx = result.first;
    xla::HloInstruction* replacement_inst = rewrites_[idx]->replacement();
    if (result.second == xla::RewriteStatus::OK) {
      if (replacement_inst->IsDead()) {
        result.second = xla::RewriteStatus::PRUNED;
      }
    }
  }

  hlo_module_->Cleanup();
  hlo_module_->RewriteCleanup();

  return results;
}

xla::RewriteStatus HloRewriteGraph::ApplyRewrite(int id) {
  if (adjacent_rewrite_ids_[id]) {
    return xla::RewriteStatus::ADJACENCY;
  }

  LOG(ERROR) << "-------------------------";
  LOG(ERROR) << rewrites_.size();
  LOG(ERROR) << "Applying rewrite [" << id << "]";
  LOG(ERROR) << "total rewrites: " << rewrites_.size();
  LOG(ERROR) << "original: " << rewrites_[id]->original()->name();
  LOG(ERROR) << "replacement: " << rewrites_[id]->replacement()->name();
  LOG(ERROR) << "users: ";
  for (auto* user : rewrites_[id]->users()) {
    LOG(ERROR) << "    " << user->name();
  }
  LOG(ERROR) << "-------------------------";

  xla::RewriteStatus status = rewrites_[id]->Apply();
  if (status == xla::RewriteStatus::OK) {
    applied_[id] = true;
    for (int i = 0; i < adjacency_matrix_[id].size(); i++) {
      bool is_connected = adjacency_matrix_[id][i];
      // Set all connected rewrites as no longer applicable.
      if (is_connected) {
        adjacent_rewrite_ids_[i] = true;
        // rewrites_[i]->SetApplicable(false);
      }
    }
  }

  return status;
}

}  // namespace hloenv
