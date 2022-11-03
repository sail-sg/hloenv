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

# A demo to where we always pick the first alternative and compare the
# evaluation timing of the resulting hlo module to the one optimized by the
# original xla pipeline.

import numpy as np
import os
from hloenv import AltPipeline, HloEnv, HloPass, Pass, Pipeline
from general_fusion_pipeline import GeneralFusionPipeline

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  base_dir = os.path.dirname(os.path.realpath(__file__))
  hlo_base_dir = os.path.join(base_dir, "hlo_texts")
  for root, dirs, files in os.walk(hlo_base_dir):
    for file in files:

      filepath = os.path.join(root, file)

      hlo_env = HloEnv(filepath, "gpu")
      general_fusion_pipeline = GeneralFusionPipeline(hlo_env)
      instruction_count = hlo_env.get_hlo_module().instruction_count

      print("-------------------------------------------------")
      print("Testing general fusion for file: " + filepath)
      print("    num instructions = %d" % instruction_count)

      orig_hlo_module = hlo_env.clone_hlo()
      # Original TF pipelines
      hlo_env.optimize_hlo_module()

      ref_results = hlo_env.evaluate(100)
      ref_timing = min(ref_results.durations)

      # Save reference copy of the module after a non dry-run RunHloPasses call
      reference_hlo_module = hlo_env.clone_hlo()
      hlo_env.load_hlo(orig_hlo_module)

      hlo_env.run(general_fusion_pipeline.pre_pass_optimizations)
      num_alts = 1
      while num_alts > 0:
        hlo_env.run(general_fusion_pipeline.pre_dry_pass_passes)
        hlo_env.run(general_fusion_pipeline.pass_dry_run)

        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)

        if num_alts > 0:
          decisions = []
          for alt_idx in hlo_graph.alternative_indices:
            node_uid = node_features.uids[alt_idx]

            # Always pick the first alternative available
            decisions.append([node_uid, 1])

          decisions = np.asarray(decisions)
          hlo_env.apply_alternatives(decisions)

          hlo_env.run(general_fusion_pipeline.post_dry_pass_passes)

      hlo_env.run(general_fusion_pipeline.post_pass_optimizations)

      pick_one_results = hlo_env.evaluate(100)
      pick_one_timing = min(pick_one_results.durations)

      print(
        "Ref timing: %.2f, Pick one timing: %.2f" %
        (ref_timing, pick_one_timing)
      )
