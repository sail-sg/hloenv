// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_UTILS_HLO_UTILS_H_
#define ALTGRAPH_UTILS_HLO_UTILS_H_

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
}  // namespace xla

#endif  // ALTGRAPH_UTILS_HLO_UTILS_H_
