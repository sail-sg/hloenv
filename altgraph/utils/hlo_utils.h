// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_UTILS_HLO_UTILS_H_
#define ALTGRAPH_UTILS_HLO_UTILS_H_

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
// given an instruction, return its attributes (int vector)
// and attribute counts:
// {2,3,4}, {LE} (out of enum {LE, EQ, GE}), {SAME} (out of {PADDING,SAME}) will
// results: attrs: 2,3,0,3,1,2 attr_counts: 3,2 we assume all dimensions has
// size 6, if less than 6 will pad -1
// TODO(wangyzh): handle custom-call, custom_call_target, api_version,
// backend_config
void GetInstructionAttributesAndCounts(HloInstruction* inst,
                                       std::vector<int>* attrs,
                                       std::vector<int>* attr_counts);

bool isCommutative(xla::HloInstruction* inst);

class HloComputationHashWrapper;

class HloInstructionHashWrapper {
 public:
  explicit HloInstructionHashWrapper(
      xla::HloInstruction* inst,
      std::shared_ptr<std::vector<uint64_t>> operand_hashes,
      std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map,
      std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map)
      : inst_(inst),
        operand_hashes_(operand_hashes),
        inst_hash_map_(inst_hash_map),
        comp_hash_map_(comp_hash_map) {}

  template <typename H>
  friend H AbslHashValue(H h, const HloInstructionHashWrapper& inst_wrapper) {
    xla::HloInstruction* inst = inst_wrapper.inst_;

    std::shared_ptr<std::vector<uint64_t>> operand_hashes =
        inst_wrapper.operand_hashes_;
    std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map =
        inst_wrapper.inst_hash_map_;
    std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map =
        inst_wrapper.comp_hash_map_;

    std::vector<int> opcode_attrs;
    std::vector<int> opcode_attr_counts;
    GetInstructionAttributesAndCounts(inst, &opcode_attrs, &opcode_attr_counts);

    // Base instruction hash
    h = H::combine(std::move(h), inst->opcode(), inst->shape());
    if (inst->opcode() == HloOpcode::kFusion) {
      h = H::combine(std::move(h), *inst->fused_expression_root(),
                     inst->fusion_kind(), inst->fused_instruction_count(),
                     inst->fused_parameters().size());
    }

    // Hash in instruction attributes
    h = H::combine(std::move(h), *inst);
    h = H::combine(std::move(h), opcode_attrs, opcode_attr_counts);

    // TODO(ohcy) -> Do we need to handle the special case of
    // inst->IsCrossModuleAllReduce()?

    // Add the hash of the computations if it has called_computations
    if (inst->called_computations().size() > 0) {
      for (xla::HloComputation* comp : inst->called_computations()) {
        auto iter = comp_hash_map->find(comp);
        uint64_t comp_hash;
        // cache the hash of the same computation
        if (iter == comp_hash_map->end()) {
          HloComputationHashWrapper comp_hash_wrapper =
              HloComputationHashWrapper(comp, inst_hash_map, comp_hash_map);
          comp_hash = absl::HashOf(comp_hash_wrapper);
          comp_hash_map->insert({comp, comp_hash});
        } else {
          comp_hash = iter->second;
        }

        h = H::combine(std::move(h), comp_hash);
      }
    }

    // Hash in operand hashes
    if (inst->opcode() == HloOpcode::kAlternatives) {
      h = H::combine_unordered(std::move(h), operand_hashes->begin(),
                               operand_hashes->end());
    } else {
      for (uint64_t operand_hash : *operand_hashes) {
        h = H::combine(std::move(h), operand_hash);
      }
      // h = H::combine_contiguous(std::move(h), operand_hashes->data(),
      //                           operand_hashes->size());
    }
    h = H::combine(std::move(h), operand_hashes->size());
    return h;
  }

 private:
  xla::HloInstruction* inst_;
  std::shared_ptr<std::vector<uint64_t>> operand_hashes_;
  std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map_;
  std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map_;
};

class HloComputationHashWrapper {
 public:
  explicit HloComputationHashWrapper(
      xla::HloComputation* computation,
      std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map,
      std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map)
      : computation_(computation),
        inst_hash_map_(inst_hash_map),
        comp_hash_map_(comp_hash_map) {}

  template <typename H>
  friend H AbslHashValue(H h, const HloComputationHashWrapper& comp_wrapper) {
    std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map =
        comp_wrapper.inst_hash_map_;
    std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map =
        comp_wrapper.comp_hash_map_;

    HloComputation* computation = comp_wrapper.computation_;

    std::vector<std::tuple<xla::HloInstruction*,
                           std::shared_ptr<std::vector<uint64_t>>>>
        stack;
    std::shared_ptr<std::vector<uint64_t>> root_operand_hashes =
        std::make_shared<std::vector<uint64_t>>(0);
    stack.push_back(
        {computation->root_instruction(), std::move(root_operand_hashes)});

    while (!stack.empty()) {
      std::tuple<xla::HloInstruction*, std::shared_ptr<std::vector<uint64_t>>>&
          inst_op_pair = stack.back();
      HloInstruction* inst = std::get<0>(inst_op_pair);
      std::shared_ptr<std::vector<uint64_t>> operand_hashes =
          std::get<1>(inst_op_pair);
      size_t operand_hash_count = operand_hashes->size();
      // if this instruction is ready to compute hash
      if (operand_hash_count ==
          inst->operand_count() + inst->control_predecessors().size()) {
        // pop it out
        stack.pop_back();
        // compute the instruction hash
        HloInstructionHashWrapper inst_hash_wrapper = HloInstructionHashWrapper(
            inst, operand_hashes, inst_hash_map, comp_hash_map);
        uint64_t inst_hash = absl::HashOf(inst_hash_wrapper);
        // cache the instruction hash once computed
        inst_hash_map->insert({inst, inst_hash});
        // add to its users (which should be the next element in the stack)
        if (stack.empty()) {
          return H::combine(std::move(h), inst_hash);
        } else {
          std::get<1>(stack.back())->push_back(inst_hash);
        }
      } else {
        // If the inst is not ready to compute hash
        // get its next operand in the row
        xla::HloInstruction* operand;
        if (operand_hash_count < inst->operand_count()) {
          operand = inst->mutable_operand(operand_hash_count);
        } else {
          operand = inst->control_predecessors().at(operand_hash_count -
                                                    inst->operand_count());
        }
        auto iter = inst_hash_map->find(operand);
        if (iter == inst_hash_map->end()) {
          // Push the operand to the end of the stack if its hash doesn't exist
          std::shared_ptr<std::vector<uint64_t>> new_operand_hashes =
              std::make_shared<std::vector<uint64_t>>(0);
          stack.push_back({operand, std::move(new_operand_hashes)});
        } else {
          // If operand already has hash cached
          operand_hashes->push_back(iter->second);
        }
      }
    }
  }

 private:
  xla::HloComputation* computation_;
  std::unordered_map<xla::HloInstruction*, uint64_t>* inst_hash_map_;
  std::unordered_map<xla::HloComputation*, uint64_t>* comp_hash_map_;
};

class HloModuleHashWrapper {
 public:
  explicit HloModuleHashWrapper(xla::HloModule* module) : module_(module) {}

  template <typename H>
  friend H AbslHashValue(H h, const HloModuleHashWrapper& module_wrapper) {
    xla::HloModule* module = module_wrapper.module_;

    h = H::combine(std::move(h), module->entry_computation_layout());
    // Use MakeComputationSorted() instead of MakeComputationPostOrder()
    // because naming may affect the order of MakeComputationPostOrder() but not
    // MakeComputationSorted().
    auto computations = module->MakeComputationSorted();
    std::unordered_map<xla::HloInstruction*, uint64_t> inst_hash_map;
    std::unordered_map<xla::HloComputation*, uint64_t> comp_hash_map;
    for (auto* computation : computations) {
      HloComputationHashWrapper comp_hash_wrapper = HloComputationHashWrapper(
          computation, &inst_hash_map, &comp_hash_map);
      h = H::combine(std::move(h), absl::HashOf(comp_hash_wrapper));
    }
    return H::combine(std::move(h), computations.size());
  }

 private:
  xla::HloModule* module_;
};

uint64_t HloModuleHash(xla::HloModule* module);

}  // namespace xla

#endif  // ALTGRAPH_UTILS_HLO_UTILS_H_
