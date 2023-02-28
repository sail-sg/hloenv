import numpy as np
from hloenv import AltPipeline, RewritePipeline, HloEnv, HloPass, Pass, \
                   Pipeline, RewriteStatus
from pass_pipelines import GeneralFusionPipeline
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_file(filepath):
  hlo_env = HloEnv(filepath, "gpu")
  general_fusion_pipeline = GeneralFusionPipeline(hlo_env)
  rewrite_fusion_pass = RewritePipeline(Pass(HloPass.GeneralFusion(),))

  hlo_env.run(general_fusion_pipeline.pre_pass_optimizations)

  rewrite_applied = True
  count = 0
  while rewrite_applied:
    hlo_env.run(general_fusion_pipeline.pre_dry_pass_passes)

    hlo_env.run(rewrite_fusion_pass)

    rewrite_graph = hlo_env.get_hlo_rewrite_graph()
    adjacency_matrix = rewrite_graph.adjacency_matrix
    num_rewrites = len(adjacency_matrix)

    # Print rewrite data
    print("\nLogging all rewrites (for debug only")
    rewrite_graph.log()

    rewrites = rewrite_graph.rewrite_data
    # Get individual rewrite data
    print("\nGetting rewrite data (orig subgraph + pass name)")
    for (idx, rewrite) in enumerate(rewrites):
      print("**********************************************")
      print("Rewrite %d:" % idx)
      print("    Pass Name: " + rewrite.pass_name)
      print("    Idx: %d" % rewrite.order_idx)
      print("    Node uids: " + str(rewrite.orig_subgraph.node_features.uids))
      print(
        "    Node uids (repl): " +
        str(rewrite.replacement_subgraph.node_features.uids)
      )
      print("Printing original subgraph:")
      print(rewrite.orig_subgraph_to_str())
      print("Printing rewrite subgraph:")
      print(rewrite.replacement_subgraph_to_str())
      print("**********************************************")

    print("\nExample applying rewrites...")
    rewrites = rewrite_graph.rewrite_data
    print("num_rewrites: %d" % num_rewrites)
    assert(num_rewrites == len(rewrites))
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
      # [1, 2, 5]
      if any_applied:
        # print(results)
        rewrite_applied = True
        break
      else:
        start_idx += 1

    hlo_env.run(general_fusion_pipeline.post_dry_pass_passes)

    count += 1

  hlo_env.run(general_fusion_pipeline.post_pass_optimizations)
  rewrite_mod = hlo_env.clone_hlo()

  hlo_env = HloEnv(filepath, "gpu")
  hlo_env.run(general_fusion_pipeline.xla_pipeline)

  assert (hlo_env.has_equal_output_as(rewrite_mod))
  return


# filepath = "./hlo_texts/test_hlos/NuX/1648647409115717.module_0337.jit_loss.9581.before_optimizations.txt"
# test_file(filepath)
# exit()

count = 0
hlo_base_dir = "../tests/hlo_texts/test_hlos"
for root, dirs, files in os.walk(hlo_base_dir):
  for file in files:

    filepath = os.path.join(root, file)
    print("Testing..." + filepath)
    test_file(filepath)

    # count += 1
    # if count > 10:
    #   break
