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

      print("-------------------------------------------------")
      print("Testing general fusion for file: " + filepath)

      hlo_env = HloEnv(filepath, "gpu")
      general_fusion_pipeline = GeneralFusionPipeline(hlo_env)
      rewrite_fusion_pass = RewritePipeline(Pass(HloPass.GeneralFusion(),))

      orig_hlo_module = hlo_env.clone_hlo()
      # Original TF pipelines
      hlo_env.optimize_hlo_module()

      ref_results = hlo_env.evaluate(100)
      ref_timing = min(ref_results.durations)

      hlo_env.load_hlo(orig_hlo_module)

      hlo_env.run(general_fusion_pipeline.pre_pass_optimizations)

      rewrite_applied = True
      count = 0
      while rewrite_applied:
        hlo_env.run(general_fusion_pipeline.pre_dry_pass_passes)

        hlo_env.run(rewrite_fusion_pass)

        rewrite_graph = hlo_env.get_hlo_rewrite_graph()
        adjacency_matrix = rewrite_graph.adjacency_matrix
        num_rewrites = len(adjacency_matrix)

        rewrites = rewrite_graph.rewrite_data
        rewrites = rewrite_graph.rewrite_data
        # Simplistic algorithm where we start with first rewrite, and walk through
        # and add any rewrites that aren't adjacent to already added rewrites
        # to our decisions
        start_idx = 0
        rewrite_applied = False
        while (start_idx < num_rewrites):
          applicable = [True for i in range(num_rewrites)]
          decisions = []
          for i in range(start_idx, num_rewrites):
            # do_anyway = random.randint(0,10) == 0
            # if applicable[i] or do_anyway:
            if applicable[i]:
              decisions.append(i)
              for other_idx in range(i, num_rewrites):
                is_adj = adjacency_matrix[i][other_idx]
                if is_adj:
                  applicable[other_idx] = False

          results = hlo_env.apply_rewrites(decisions)
          any_applied = sum([1 for (idx, applied) in results if applied == RewriteStatus.OK]) > 0

          not_applied = [(idx, applied) for (idx, applied) in results if applied != RewriteStatus.OK]
          num_not_applied = len(not_applied)
          if num_not_applied > 0:
            print("%d out of %d rewrites not applied" % (num_not_applied, len(results)))
            print(not_applied)

          # If we successfully applied at least 1 rewrite, good! we're happy
          # Otherwise let's try again and ignore rewrites we've already tried to
          # apply
          if any_applied:
            rewrite_applied = True
            break
          else:
            start_idx += 1

        hlo_env.run(general_fusion_pipeline.post_dry_pass_passes)

        count += 1

      pick_one_results = hlo_env.evaluate(100)
      pick_one_timing = min(pick_one_results.durations)

      print(
        "Ref timing: %.2f, Pick one timing: %.2f" %
        (ref_timing, pick_one_timing)
      )
