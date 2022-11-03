# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A demo to expose and play all the features (graph, node, edge)
# of a hlo module in python script.
# Usage: python hlo_play.py

import os
import pathlib
import hloenv
import numpy as np
from hloenv import HloEnv

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  hlo_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "hlo_texts/jax-md/module_0013.jit__lambda_.7.before_optimizations.txt"
  )

  hlo_env = HloEnv(hlo_path, "gpu")

  # hlo_graph is the entry point of the features of hlo_env
  hlo_graph = hlo_env.get_hlo_graph()

  print("=========graph_features==========")
  print(hlo_graph.out_edge_offsets)
  print(len(hlo_graph.out_edge_offsets))
  print(hlo_graph.out_edge_indices)
  print(len(hlo_graph.out_edge_indices))
  print(hlo_graph.in_edge_offsets)
  print(hlo_graph.in_edge_indices)
  print(hlo_graph.alternative_indices)
  print(hlo_graph.opcode_attr_counts)
  print(hlo_graph.hash())

  node_features = hlo_graph.node_features
  in_edge_features = hlo_graph.in_edge_features
  out_edge_features = hlo_graph.out_edge_features

  print("=========node_features==========")
  print(node_features.uids)
  print(node_features.gids)
  print(node_features.num_users)
  print(node_features.num_operands)
  print(node_features.opcodes)
  print(node_features.opcode_attrs)
  print(node_features.num_opcode_attrs.reshape(-1, 2))
  print(node_features.is_alternative)
  print(node_features.in_tensor_sizes)
  print(node_features.out_tensor_sizes)
  print(node_features.has_max_in_tensor)
  print(node_features.has_max_out_tensor)
  print(node_features.names)

  print("=========in_edge_features===========")
  print(in_edge_features.uids)
  print(in_edge_features.srcs)
  print(in_edge_features.dsts)
  print(in_edge_features.dims.reshape(-1, 8))
  print(in_edge_features.layouts.reshape(-1, 8))
  print(in_edge_features.dtypes)

  print("=========out_edge_features===========")
  print(out_edge_features.uids)
  print(out_edge_features.srcs)
  print(out_edge_features.dsts)
  print(np.array(out_edge_features.dims).reshape(-1, 8))
  print(np.array(out_edge_features.layouts).reshape(-1, 8))
  print(out_edge_features.dtypes)
