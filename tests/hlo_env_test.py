import os
import random

from absl import logging
from absl.testing import absltest

SELECT_ORIGINAL_CHANCE = 0.1


def get_rand_action(num_operands):
  select_original = random.random() <= SELECT_ORIGINAL_CHANCE
  if select_original:
    return 0
  else:
    return random.randrange(1, num_operands)


class HloEnvTest(absltest.TestCase):
  """Placeholder for some real tests
  """

  def setUp(self) -> None:
    logging.set_verbosity(logging.INFO)
    logging.info("setting up")
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/hlo_texts"

    self.hlo_main_test_file = dir_path + "/hlo_test.txt"

  def test_import(self) -> None:
    import altgraph

    import tensorflow
    logging.info("altgraph module imported at %s", altgraph)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_graph_interfaces(self) -> None:
    from altgraph import HloEnv
    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_graph = hlo_env.get_hlo_graph()

    assert (len(hlo_graph.out_edge_offsets) > 0)
    assert (len(hlo_graph.out_edge_indices) > 0)
    assert (len(hlo_graph.in_edge_offsets) > 0)
    assert (len(hlo_graph.in_edge_indices) > 0)

    _ = hlo_graph.alternative_indices
    _ = hlo_graph.hash()
    node_features = hlo_graph.node_features
    in_edge_features = hlo_graph.in_edge_features
    out_edge_features = hlo_graph.out_edge_features

    num_nodes = len(node_features.uids)
    assert (num_nodes > 0)
    assert (len(hlo_graph.opcode_attr_counts) == 236)
    assert (
      len(node_features.opcode_attrs) == sum(node_features.num_opcode_attrs)
    )
    assert (len(node_features.names) == num_nodes)
    assert (len(node_features.gids) == num_nodes)
    assert (len(node_features.num_users) == num_nodes)
    assert (len(node_features.num_operands) == num_nodes)
    assert (len(node_features.opcodes) == num_nodes)
    assert (len(node_features.is_alternative) == num_nodes)
    assert (len(node_features.in_tensor_sizes) == num_nodes)
    assert (len(node_features.out_tensor_sizes) == num_nodes)
    _ = node_features.has_max_in_tensor
    _ = node_features.has_max_out_tensor

    num_in_edges = len(in_edge_features.uids)
    assert (num_in_edges > 0)
    assert (len(in_edge_features.srcs) == num_in_edges)
    assert (len(in_edge_features.dsts) == num_in_edges)
    assert (len(in_edge_features.dims) == num_in_edges * 8)
    assert (len(in_edge_features.layouts) == num_in_edges * 8)
    assert (len(in_edge_features.lehmercodes) == num_in_edges * 8)
    assert (len(in_edge_features.dtypes) == num_in_edges)
    _ = in_edge_features.get_tensor_size(0)

    num_out_edges = len(out_edge_features.uids)
    assert (len(out_edge_features.srcs) == num_out_edges)
    assert (len(out_edge_features.dsts) == num_out_edges)
    assert (len(out_edge_features.dims) == num_out_edges * 8)
    assert (len(out_edge_features.layouts) == num_out_edges * 8)
    assert (len(out_edge_features.lehmercodes) == num_out_edges * 8)
    assert (len(out_edge_features.dtypes) == num_out_edges)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_alt_hlo_module(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv, HloModule

    import tensorflow

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    hlo_module_ref = HloModule(self.hlo_main_test_file)
    hlo_module_str_ref = hlo_module_ref.to_string()
    hlo_module_hash_ref = hlo_module_ref.hash()

    hlo_module_from_ir = hlo_env.get_hlo_module()
    hlo_module_from_ir_str = hlo_module_from_ir.to_string()
    hlo_module_from_ir_hash = hlo_module_from_ir.hash()

    hlo_env_hlo_str = hlo_env.export_hlo_to_str()
    hlo_env_hlo_hash = hlo_env.get_hlo_module_hash()

    assert (hlo_module_str_ref == hlo_module_from_ir_str == hlo_env_hlo_str)
    assert (hlo_module_hash_ref == hlo_module_from_ir_hash == hlo_env_hlo_hash)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_basic(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    import tensorflow

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()
      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, get_rand_action(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
      else:
        logging.info("No more alternatives, ending run...")
      eval_result = hlo_env.evaluate(1)
      total_time = 0
      for eval_time_ns in eval_result.durations:
        assert eval_time_ns > 0
        total_time += eval_time_ns
      logging.info("Running time estimation: %d ns", total_time / 10)

      count += 1

    assert (count > 1)

    logging.info("Running post_fusion_optimizations...")
    hlo_env.post_fusion_optimizations()

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_create_from_module_handle(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv, HloModule

    import tensorflow

    hlo_module = HloModule(self.hlo_main_test_file)

    hlo_env = HloEnv(hlo_module, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()

      saved_hlo_module = hlo_env.save_hlo()
      hlo_env = HloEnv(saved_hlo_module, "gpu")

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, get_rand_action(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
      else:
        logging.info("No more alternatives, ending run...")
        eval_result = hlo_env.evaluate(1)
        for eval_time_ns in eval_result.durations:
          assert eval_time_ns > 0
          logging.info("Running time estimation: %d ns", eval_time_ns)

      count += 1

    assert (count > 1)
    logging.info("Running post_fusion_optimizations...")
    hlo_env.post_fusion_optimizations()
    eval_result = hlo_env.evaluate(1)
    for eval_time_ns in eval_result.durations:
      assert eval_time_ns > 0
      logging.info("Running time estimation: %d ns", eval_time_ns)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_may_duplicate(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    import tensorflow

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info(
        "Running fusion dry run, may_duplicate = %s" % (count % 2 == 0)
      )
      hlo_env.fusion_dry_run(may_duplicate=(count % 2 == 0))
      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, get_rand_action(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
      else:
        logging.info("No more alternatives, ending run...")
      count += 1

    logging.info("Running post_fusion_optimizations...")
    hlo_env.post_fusion_optimizations()
    hlo_env.evaluate(1)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_save_load(self) -> None:
    from altgraph import HloEnv

    # Test normal save/loading
    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    init_hlo_str = hlo_env.export_hlo_to_str()
    init_hlo_hash = hlo_env.get_hlo_module_hash()
    saved_hlo_module = hlo_env.save_hlo()
    hlo_env.pre_fusion_optimizations()
    post_fusion_hlo_str = hlo_env.export_hlo_to_str()
    post_fusion_hlo_hash = hlo_env.get_hlo_module_hash()
    hlo_env.load_hlo(saved_hlo_module)
    restored_hlo_hash = hlo_env.get_hlo_module_hash()
    restored_hlo_str = hlo_env.export_hlo_to_str()
    assert (init_hlo_str != post_fusion_hlo_str)
    assert (init_hlo_hash != post_fusion_hlo_hash)
    assert (init_hlo_str == restored_hlo_str)
    assert (init_hlo_hash == restored_hlo_hash)

    # Test loading from string
    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    init_hlo_str = hlo_env.export_hlo_to_str()
    init_hlo_hash = hlo_env.get_hlo_module_hash()
    saved_hlo_module = hlo_env.save_hlo()
    hlo_env.pre_fusion_optimizations()
    post_fusion_hlo_str = hlo_env.export_hlo_to_str()
    post_fusion_hlo_hash = hlo_env.get_hlo_module_hash()
    hlo_env.load_hlo(init_hlo_str, "txt")
    restored_hlo_hash = hlo_env.get_hlo_module_hash()
    restored_hlo_str = hlo_env.export_hlo_to_str()
    assert (init_hlo_str != post_fusion_hlo_str)
    assert (init_hlo_hash != post_fusion_hlo_hash)
    assert (init_hlo_str == restored_hlo_str)
    assert (init_hlo_hash == restored_hlo_hash)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_evaluation(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

    hlo_env.pre_fusion_optimizations()
    saved_hlo_module = hlo_env.save_hlo()
    # Restore back to original, where we only did pre_fusion_optimizations
    hlo_env.post_fusion_optimizations()

    orig_res = hlo_env.evaluate(1)
    orig_post_opt_module = hlo_env.save_hlo()

    hlo_env.load_hlo(saved_hlo_module)

    num_alts = 1
    while num_alts > 0:
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()
      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, get_rand_action(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()

    hlo_env.post_fusion_optimizations()
    mod_res = hlo_env.evaluate(1)
    assert (hlo_env.has_equal_output_as(orig_post_opt_module))

    est_time_orig = sum(orig_res.durations) / len(orig_res.durations)
    est_time_mod = sum(mod_res.durations) / len(mod_res.durations)

    logging.info(
      "Running time estimation: orig: %d ns, altgraph: %d ns", est_time_orig,
      est_time_mod
    )

    assert (len(orig_res.output) == len(mod_res.output))
    for i in range(len(orig_res.output)):
      assert (len(orig_res.output[i]) == len(mod_res.output[i]))
      for j in range(len(orig_res.output[i])):
        assert (len(orig_res.output[i][j]) == len(mod_res.output[i][j]))
        for k in range(len(orig_res.output[i][j])):
          assert (
            np.allclose(mod_res.output[i][j][k], orig_res.output[i][j][k])
          )

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_validation(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    base_dir = os.path.dirname(os.path.realpath(__file__))
    hlo_base_dir = base_dir + "/hlo_texts/test_hlos"
    for root, dirs, files in os.walk(hlo_base_dir):
      for file in files:

        filepath = os.path.join(root, file)
        logging.info("Testing validation for file: " + filepath)

        hlo_env = HloEnv(filepath, "gpu")

        saved_hlo_module = hlo_env.save_hlo()
        # Original TF pipelines
        hlo_env.optimize_hlo_module()
        hlo_env.prepare_hlo_module_for_ir_emitting()

        # Save reference copy of the module after a non dry-run RunHloPasses call
        reference_hlo_module = hlo_env.save_hlo()
        hlo_env.load_hlo(saved_hlo_module)

        hlo_env.pre_fusion_optimizations()
        num_alts = 1
        while num_alts > 0:
          hlo_env.pre_fusion_dry_passes()
          hlo_env.fusion_dry_run()
          hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
          node_features = hlo_graph.node_features
          num_operands = node_features.num_operands
          num_alts = len(hlo_graph.alternative_indices)

          if num_alts > 0:
            decisions = []
            for alt_idx in hlo_graph.alternative_indices:
              node_uid = node_features.uids[alt_idx]
              decisions.append(
                [alt_idx, get_rand_action(num_operands[alt_idx])]
              )

            decisions = np.asarray(decisions)
            hlo_env.apply_alternatives(decisions)
            hlo_env.post_fusion_dry_passes()

        hlo_env.post_fusion_optimizations()
        post_fusion_module = hlo_env.save_hlo()

        assert (
          hlo_env.has_equal_output(post_fusion_module, reference_hlo_module)
        )

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_load_from_string(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    import tensorflow

    def check_load_from_string(hlo_env):
      hlo_string = hlo_env.export_hlo_to_str()
      hlo_env_loaded_from_str = HloEnv(hlo_string, "txt", "gpu")
      assert (
        hlo_env.get_hlo_module_hash() ==
        hlo_env_loaded_from_str.get_hlo_module_hash()
      )

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    logging.info(
      "Checking load_from_string after: hlo_env = HloEnv(%s, %s)" %
      (self.hlo_main_test_file, "gpu")
    )
    check_load_from_string(hlo_env)

    hlo_string = hlo_env.export_hlo_to_str()
    hlo_env = HloEnv(hlo_string, "txt", "gpu")

    hlo_env.pre_fusion_optimizations()
    logging.info(
      "Checking load_from_string after: hlo_env.pre_fusion_optimizations"
    )
    check_load_from_string(hlo_env)

    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()
      logging.info("Checking load_from_string after: hlo_env.fusion_dry_run")
      check_load_from_string(hlo_env)

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, get_rand_action(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
        logging.info(
          "Checking load_from_string after: hlo_env.apply_alternatives"
        )
        check_load_from_string(hlo_env)

      else:
        logging.info("No more alternatives, ending run...")
      count += 1

    assert (count > 1)

    logging.info("Running post_fusion_optimizations...")
    hlo_env.post_fusion_optimizations()
    logging.info(
      "Checking load_from_string after: hlo_env.post_fusion_optimizations"
    )
    check_load_from_string(hlo_env)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_hash(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    base_dir = os.path.dirname(os.path.realpath(__file__))
    hlo_base_dir = base_dir + "/hlo_texts/test_hlos"
    for root, dirs, files in os.walk(hlo_base_dir):
      for file in files:

        filepath = os.path.join(root, file)
        logging.info("Testing hash for file: " + filepath)

        hlo_env = HloEnv(filepath, "gpu")

        saved_hlo_module = hlo_env.save_hlo()
        cloned_hash = saved_hlo_module.hash()
        original_hash = hlo_env.get_hlo_module_hash()
        assert (cloned_hash == original_hash)
        hlo_env.load_hlo(saved_hlo_module)

        hlo_env.pre_fusion_optimizations()
        saved_hlo_module = hlo_env.save_hlo()
        cloned_hash = saved_hlo_module.hash()
        original_hash = hlo_env.get_hlo_module_hash()
        assert (cloned_hash == original_hash)
        hlo_env.load_hlo(saved_hlo_module)

        num_alts = 1
        while num_alts > 0:
          prev_hash = hlo_env.get_hlo_module_hash()
          hlo_env.pre_fusion_dry_passes()
          hlo_env.fusion_dry_run()

          hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
          node_features = hlo_graph.node_features
          num_operands = node_features.num_operands
          num_alts = len(hlo_graph.alternative_indices)

          if num_alts > 0:
            # Test that hash changes after fusion_dry_run
            new_hash = hlo_env.get_hlo_module_hash()
            saved_hlo_module = hlo_env.save_hlo()
            cloned_hash = saved_hlo_module.hash()
            assert (cloned_hash == new_hash)
            assert (prev_hash != new_hash)
            hlo_env.load_hlo(saved_hlo_module)

            # Test that hash changes after apply_alternatives
            prev_hash = hlo_env.get_hlo_module_hash()
            decisions = []
            for alt_idx in hlo_graph.alternative_indices:
              node_uid = node_features.uids[alt_idx]
              decisions.append(
                [alt_idx, get_rand_action(num_operands[alt_idx])]
              )

            decisions = np.asarray(decisions)
            hlo_env.apply_alternatives(decisions)
            new_hash = hlo_env.get_hlo_module_hash()
            hlo_env.post_fusion_dry_passes()
            assert (prev_hash != new_hash)

        hlo_env.post_fusion_optimizations()

  # Test that if we choose the original nodes, graph and graph hash
  # stays constant
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_apply_original(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv

    base_dir = os.path.dirname(os.path.realpath(__file__))
    hlo_base_dir = base_dir + "/hlo_texts/test_hlos"
    for root, dirs, files in os.walk(hlo_base_dir):
      for file in files:

        filepath = os.path.join(root, file)
        logging.info("Testing hash for file: " + filepath)

        hlo_env = HloEnv(filepath, "gpu")
        hlo_env.pre_fusion_optimizations()

        num_alts = 1
        while num_alts > 0:
          prev_hash = hlo_env.get_hlo_module_hash()
          hlo_env.pre_fusion_dry_passes()
          original_hash = hlo_env.get_hlo_module_hash()
          hlo_env.fusion_dry_run()

          hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
          node_features = hlo_graph.node_features
          num_operands = node_features.num_operands
          num_alts = len(hlo_graph.alternative_indices)

          if num_alts > 0:
            # Test that hash does not change after apply_alternatives zero
            decisions = []
            for alt_idx in hlo_graph.alternative_indices:
              node_uid = node_features.uids[alt_idx]
              decisions.append([alt_idx, 0])

            decisions = np.asarray(decisions)
            hlo_env.apply_alternatives(decisions)
            new_hash = hlo_env.get_hlo_module_hash()
            assert (original_hash == new_hash)

            break

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_extract_instruction(self) -> None:
    from altgraph import HloEnv, HloModule

    hlo_ir = HloEnv(self.hlo_main_test_file, "gpu")
    for (instruction, hlo_graph
        ) in hlo_ir.get_hlo_module().extract_instructions_as_module(10):
      assert (len(instruction) > 0)
      assert (len(hlo_graph.to_string()) > 0)
      print(instruction)
      print(hlo_graph.to_string())

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_extract_fusions(self) -> None:
    from altgraph import HloEnv, HloModule

    hlo_ir = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_ir.pre_fusion_optimizations()
    hlo_ir.pre_fusion_dry_passes()
    hlo_ir.fusion_dry_run()
    m = hlo_ir.get_hlo_module()
    fusions = m.extract_fusions_as_module(10)
    assert (len(fusions) > 0)
    assert (len(fusions[0].to_string()) > 0)
    print(fusions[0].to_string())

  # Test general pipeline
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_main(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0

    fusion_pre_pipeline = Pipeline("fusion_pre")

    fusion_pre_pipeline.add_pass(HloPass.VariadicOpSplitter())
    # Note you have to make an Pass, cannot just run the HloPass directly.
    fusion_dry_pass = AltPipeline(
      Pass(
        HloPass.GpuInstructionFusion(True),  # may_duplicate
      )
    )

    fusion_post_pipeline = Pipeline(name="fusion_pre")
    fusion_post_pipeline.add_pass(HloPass.FusionMerger(), loop_count=1)
    # Note: default values for dry_mode is false, loop_count is 1
    fusion_post_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    fusion_post_pipeline.add_pass(HloPass.HloCSE(True, True))

    fusion_example_pipeline = Pipeline(name="fusion_pre_hlo_dce")
    fusion_example_pipeline.add_pass(HloPass.HloDCE())
    # Note, a pipeline can be added as a pass to a pipeline (can nest this)
    fusion_post_pipeline.add_pass(fusion_example_pipeline)

    init_hlo = hlo_env.save_hlo()
    while num_alts > 0:
      hlo_env.run(fusion_pre_pipeline)

      # You can run a pass on it's own
      hlo_env.run(fusion_dry_pass)

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        hlo_env.apply_alternatives(decisions)
        hlo_env.run(fusion_post_pipeline)

        count += 1

    general_count = count
    general_pipeline_hlo = hlo_env.save_hlo()

    hlo_env.load_hlo(init_hlo)

    num_alts = 1
    count = 0
    while num_alts > 0:
      print(count)
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
        count += 1

    original_count = count
    original_pipeline_hlo = hlo_env.save_hlo()

    assert (original_count == general_count)
    assert (
      hlo_env.has_equal_output(original_pipeline_hlo, general_pipeline_hlo)
    )

  # Test general pipeline run till next dry pass functionality
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_run_to_dry(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0

    fusion_pipeline = Pipeline("fusion")

    fusion_pipeline.add_pass(HloPass.VariadicOpSplitter())
    fusion_dry_pass = AltPipeline(
      Pass(
        HloPass.GpuInstructionFusion(may_duplicate=True
                                    ),  # Test named arguments
      )
    )
    fusion_pipeline.add_pass(fusion_dry_pass)
    fusion_pipeline.add_pass(HloPass.FusionMerger())
    fusion_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    fusion_pipeline.add_pass(
      HloPass.HloCSE(True, only_fusion_computations=True)
    )
    fusion_pipeline.add_pass(HloPass.HloDCE())

    init_hlo = hlo_env.save_hlo()
    has_alt = True
    while has_alt:
      has_alt = hlo_env.run(fusion_pipeline)
      print("COUNT: ", count, has_alt)
      # We hit a dry run pass
      if has_alt:
        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)
        assert (num_alts > 0)
        if num_alts > 0:
          decisions = []
          for alt_idx in hlo_graph.alternative_indices:
            node_uid = node_features.uids[alt_idx]
            decisions.append([alt_idx, min(1, num_operands[alt_idx])])

          decisions = np.asarray(decisions)
          # pass the decision back to compilerp
          hlo_env.apply_alternatives(decisions)
          count += 1

      # Continue running the rest of the fusion_pipeline
      rest_has_alt = hlo_env.run(fusion_pipeline)
      # We should have no alts now, since the rest of the passes are not dry
      assert (not rest_has_alt)

    general_count = count
    general_pipeline_hlo = hlo_env.save_hlo()
    hlo_env.load_hlo(init_hlo)

    num_alts = 1
    count = 0
    while num_alts > 0:
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
        count += 1

    original_count = count
    original_pipeline_hlo = hlo_env.save_hlo()

    assert (original_count == general_count)
    assert (
      hlo_env.has_equal_output(original_pipeline_hlo, general_pipeline_hlo)
    )

  # Test general pipeline fixed pipeline functionality
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_fixed(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0

    fusion_pipeline = Pipeline("fusion-pipeline", loop_count=-1)

    fusion_pipeline.add_pass(HloPass.VariadicOpSplitter())
    fusion_dry_pass = AltPipeline(Pass(HloPass.GpuInstructionFusion(True)))
    fusion_pipeline.add_pass(fusion_dry_pass)
    fusion_pipeline.add_pass(HloPass.FusionMerger())
    fusion_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    fusion_pipeline.add_pass(HloPass.HloCSE(True, True))
    fusion_pipeline.add_pass(HloPass.HloDCE())

    init_hlo = hlo_env.save_hlo()
    has_alt = True
    # Since the pipeline is fixed, it will run till there are no changes
    while has_alt:
      has_alt = hlo_env.run(fusion_pipeline)
      # We hit a dry run pass
      if has_alt:
        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)
        if num_alts > 0:
          decisions = []
          for alt_idx in hlo_graph.alternative_indices:
            node_uid = node_features.uids[alt_idx]
            decisions.append([alt_idx, min(1, num_operands[alt_idx])])

          decisions = np.asarray(decisions)
          # pass the decision back to compilerp
          hlo_env.apply_alternatives(decisions)
          count += 1

    general_count = count
    general_pipeline_hlo = hlo_env.save_hlo()

    hlo_env.load_hlo(init_hlo)

    num_alts = 1
    count = 0
    while num_alts > 0:
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
        count += 1

    original_count = count
    original_pipeline_hlo = hlo_env.save_hlo()

    assert (original_count == general_count)
    assert (
      hlo_env.has_equal_output(original_pipeline_hlo, general_pipeline_hlo)
    )

  # Test general pipeline fixed single pass functionality
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_loop_count(self) -> None:
    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0
    loop_count = 7

    fusion_dry_pass = AltPipeline(
      Pass(HloPass.GpuInstructionFusion(may_duplicate=True),),
      loop_count=loop_count
    )

    init_hlo = hlo_env.save_hlo()
    has_alt = True
    while (has_alt):
      hlo_env.pre_fusion_dry_passes()
      has_alt = hlo_env.run(fusion_dry_pass)
      # We hit a dry run pass
      if has_alt:
        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)

        # Since pass is not complete, there must be a change, i.e.
        # num_alts > 0
        assert (num_alts > 0)
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()

        count += 1

    assert (count == loop_count)
    assert (hlo_env.has_equal_output_as(init_hlo))

  # Test general pipeline fixed single pass functionality
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_fixed_pass(self) -> None:
    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0

    fusion_dry_pass = AltPipeline(
      Pass(HloPass.GpuInstructionFusion(may_duplicate=True), loop_count=-1)
    )

    init_hlo = hlo_env.save_hlo()
    has_alt = True
    while (has_alt):
      print(count)
      hlo_env.pre_fusion_dry_passes()
      has_alt = hlo_env.run(fusion_dry_pass)
      # We hit a dry run pass
      if has_alt:
        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)

        # Since pass is not complete, there must be a change, i.e.
        # num_alts > 0
        assert (num_alts > 0)
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()

        count += 1

    assert (count > 1)
    assert (hlo_env.has_equal_output_as(init_hlo))

  # Test general pipeline
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_identical_hash(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.pre_fusion_optimizations()

    num_alts = 1
    count = 0

    fusion_pre_pipeline = Pipeline("fusion_pre")

    fusion_pre_pipeline.add_pass(HloPass.VariadicOpSplitter())
    # Note you have to make an Pass, cannot just run the HloPass directly.
    fusion_dry_pass = AltPipeline(
      Pass(
        HloPass.GpuInstructionFusion(True),  # may_duplicate
      )
    )

    fusion_post_pipeline = Pipeline(name="fusion_pre")
    fusion_post_pipeline.add_pass(HloPass.FusionMerger(), loop_count=1)
    # Note: default values for dry_mode is false, loop_count is 1
    # We ignore GpuMultiOutputFusion since its changes are undeterministic
    # fusion_post_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    fusion_post_pipeline.add_pass(HloPass.HloCSE(True, True))
    fusion_post_pipeline.add_pass(HloPass.HloDCE())

    init_hlo = hlo_env.save_hlo()
    while num_alts > 0:
      hlo_env.run(fusion_pre_pipeline)

      # You can run a pass on it's own
      hlo_env.run(fusion_dry_pass)

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        hlo_env.apply_alternatives(decisions)
        hlo_env.run(fusion_post_pipeline)

        count += 1

    general_count = count
    general_pipeline_hash = hlo_env.get_hlo_module_hash()

    hlo_env.load_hlo(init_hlo)

    num_alts = 1
    count = 0
    while num_alts > 0:
      print(count)
      hlo_env.pre_fusion_dry_passes()
      hlo_env.fusion_dry_run()

      hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          node_uid = node_features.uids[alt_idx]
          decisions.append([alt_idx, min(1, num_operands[alt_idx])])

        hlo_env.apply_alternatives(decisions)
        hlo_env.run(fusion_post_pipeline)  # So as to exclude MultiOutputFusion

        count += 1

    original_count = count
    original_pipeline_hash = hlo_env.get_hlo_module_hash()

    assert (original_count == general_count)
    assert (original_pipeline_hash == general_pipeline_hash)

  # Test general pipeline reproducing the full OptimizeHloModule pipeline
  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_pipeline_full_optimize_hlo(self) -> None:
    from altgraph import AltPipeline, GpuBackend, HloEnv, HloPass, Pass, Pipeline

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_module = hlo_env.get_hlo_module()
    config = hlo_module.config
    debug_options = config.debug_options

    #  -----------------------------------------------------------------------
    #                             PRE FUSION PIPELINE
    #  -----------------------------------------------------------------------

    pre_fusion_pipeline = Pipeline("pre-fusion")

    # --------------------------------------------
    # SPMD Paritioning Pipeline
    # --------------------------------------------

    if (config.use_spmd_partitioning):
      spmd_pipeline = Pipeline("spmd-partitioner")
      num_partitions = config.num_partitions
      if (num_partitions > 1):
        spmd_pipeline.add_invariant_checker(HloPass.HloVerifier(False, False))

        spmd_pipeline.add_pass(HloPass.CallInliner())
        spmd_pipeline.add_pass(HloPass.ZeroSizedHloElimination())
        spmd_pipeline.add_pass(HloPass.ConditionalCanonicalizer())

        spmd_simplify_pipeline = Pipeline("spmd-simplify", loop_count=-1)
        algebraic_config_options = {
          "replace_transpose_with_bitcast": False,
          "enable_conv_operand_swap": False,
          "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
        }
        spmd_simplify_pipeline.add_pass(
          HloPass.AlgebraicSimplifier(options=algebraic_config_options)
        )
        spmd_simplify_pipeline.add_pass(HloPass.SortSimplifier())
        spmd_simplify_pipeline.add_pass(HloPass.TupleSimplifier())
        spmd_simplify_pipeline.add_pass(
          HloPass.ScatterExpander(
            HloPass.ScatterExpander.Mode.kEliminateSimpleScatters
          )
        )
        spmd_simplify_pipeline.add_pass(
          HloPass.GatherExpander(
            HloPass.GatherExpander.Mode.kEliminateSimpleGathers
          )
        )
        spmd_simplify_pipeline.add_pass(HloPass.WhileLoopConstantSinking())
        spmd_simplify_pipeline.add_pass(HloPass.WhileLoopSimplifier())
        spmd_simplify_pipeline.add_pass(HloPass.ReshapeMover())
        spmd_simplify_pipeline.add_pass(HloPass.HloConstantFolding())
        spmd_simplify_pipeline.add_pass(HloPass.ConditionalSimplifier())
        spmd_simplify_pipeline.add_pass(HloPass.HloDCE())
        spmd_pipeline.add_pass(spmd_simplify_pipeline)

        spmd_pipeline.add_pass(HloPass.ShardingPropagation(True))
        spmd_pipeline.add_pass(
          HloPass.StatefulRngSpmdPartitioner(
            num_partitions, config.replica_count
          )
        )
      else:
        spmd_simplify_pipeline.add_pass(HloPass.ShardingRemover())
        spmd_simplify_pipeline.add_pass(HloPass.HloDCE())

      pre_fusion_pipeline.add_pass(spmd_pipeline)

    # --------------------------------------------
    # Optimization Pipeline
    # --------------------------------------------

    optimization_pipeline = Pipeline("optimization")
    optimization_pipeline.add_invariant_checker(
      HloPass.HloVerifier(False, False)
    )

    optimization_pipeline.add_pass(HloPass.AllToAllDecomposer())
    optimization_pipeline.add_pass(HloPass.OperandUpcaster())
    optimization_pipeline.add_pass(HloPass.ResultCaster())
    optimization_pipeline.add_pass(HloPass.RngExpander())
    optimization_pipeline.add_pass(
      HloPass.RngBitGeneratorExpander(
        HloPass.RngBitGeneratorExpander.RandomAlgorithm.RNG_PHILOX
      )
    )
    optimization_pipeline.add_pass(HloPass.ComparisonExpander())
    optimization_pipeline.add_pass(HloPass.ZeroSizedHloElimination())

    if debug_options.xla_gpu_deterministic_ops:
      optimization_pipeline.add_pass(
        HloPass.ScatterExpander(
          HloPass.ScatterExpander.Mode.kEliminateAllScatters
        )
      )
    else:
      optimization_pipeline.add_pass(HloPass.GpuScatterExpander())

    optimization_pipeline.add_pass(HloPass.QrExpander())
    optimization_pipeline.add_pass(HloPass.EighExpander())
    optimization_pipeline.add_pass(HloPass.DynamicIndexSplitter())
    optimization_pipeline.add_pass(HloPass.CallInliner())
    optimization_pipeline.add_pass(HloPass.DotDecomposer())
    optimization_pipeline.add_pass(HloPass.Convolution4DExpander())
    optimization_pipeline.add_pass(HloPass.StableSortExpander())

    optimization_pipeline.add_pass(HloPass.BFloat16Normalization(True))
    optimization_pipeline.add_pass(HloPass.BatchNormExpander(True, True, True))
    optimization_pipeline.add_pass(
      HloPass.LogisticExpander(
        HloPass.LogisticExpander.LogisticExpansionType.kExp
      )
    )
    optimization_pipeline.add_pass(HloPass.ConditionalCanonicalizer())
    optimization_pipeline.add_pass(HloPass.DynamicDimensionSimplifier())
    dp_options = {
      "shape_check_mode": HloPass.DynamicPadder.ShapeCheckMode.kCompileTime
    }
    optimization_pipeline.add_pass(HloPass.DynamicPadder(dp_options))

    simplification_pipeline = Pipeline("simplification", loop_count=-1)
    simplification_pipeline.add_pass(HloPass.ZeroSizedHloElimination())
    simplification_pipeline.add_pass(
      HloPass.GatherExpander(
        HloPass.GatherExpander.Mode.kEliminateSimpleGathers
      )
    )
    simplification_pipeline.add_pass(
      HloPass.ScatterExpander(
        HloPass.ScatterExpander.Mode.kEliminateSimpleScatters
      )
    )
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    if (GpuBackend.stream_exec_platform == "ROCM"):
      algebraic_config_options["enable_conv_operand_swap"] = False
    simplification_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options)
    )
    simplification_pipeline.add_pass(HloPass.BitcastDtypesExpander())
    simplification_pipeline.add_pass(HloPass.DotDecomposer())
    simplification_pipeline.add_pass(
      HloPass.DotMerger(max_size_to_merge=16 << 20)
    )
    simplification_pipeline.add_pass(HloPass.SortSimplifier())
    simplification_pipeline.add_pass(HloPass.TupleSimplifier())
    simplification_pipeline.add_pass(HloPass.WhileLoopConstantSinking())
    simplification_pipeline.add_pass(HloPass.WhileLoopSimplifier())
    simplification_pipeline.add_pass(HloPass.ReshapeMover())
    simplification_pipeline.add_pass(HloPass.HloConstantFolding())
    simplification_pipeline.add_pass(HloPass.ConditionalSimplifier())
    simplification_pipeline.add_pass(HloPass.RealImagExpander())
    simplification_pipeline.add_pass(HloPass.TransposeFolding())
    simplification_pipeline.add_pass(HloPass.HloCSE(is_layout_sensitive=False))
    simplification_pipeline.add_pass(HloPass.HloDCE())
    optimization_pipeline.add_pass(simplification_pipeline)

    # Run WhileLoopTripCountAnnotator at the end of the simplification
    # pipeline, before layout assignment and fusion.  This pass does some
    # pattern-matching on while bodies/conditions, and this is where the HLO is
    # "nicest".
    #
    # It's important that we don't make semantic changes (e.g. unrolling) to
    # any `while` loops after this point, because otherwise the trip-count
    # annotations added by this pass may not be correct after the
    # modifications.
    optimization_pipeline.add_pass(HloPass.WhileLoopTripCountAnnotator())
    pre_fusion_pipeline.add_pass(optimization_pipeline)

    # --------------------------------------------
    # Collectives Pipeline
    # --------------------------------------------

    collectives_pipeline = Pipeline("collective-optimizations")
    collectives_pipeline.add_pass(HloPass.AllReduceFolder())
    collectives_pipeline.add_pass(HloPass.ReduceScatterCreator())
    collectives_pipeline.add_pass(HloPass.AllReduceReassociate())
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    collectives_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options)
    )
    collectives_pipeline.add_pass(HloPass.AllGatherBroadcastReorder())
    # pre_fusion_pipeline.add_pass(collectives_pipeline)

    # --------------------------------------------
    # Convolution Canonicalization Pipeline
    # --------------------------------------------

    # TODO(ohcy): Account for AMD GPU case
    # Note, this is specific to Nvidia GPUs. For AMD GPUs, some of the passes,
    # e.g. Cudnn passes should be excluded
    conv_canon_pipeline = Pipeline("conv-canonicalization")
    conv_canon_pipeline.add_pass(HloPass.GpusolverRewriter())
    conv_canon_pipeline.add_pass(HloPass.GpuConvRewriter())
    conv_canon_pipeline.add_pass(HloPass.CudnnFusedConvRewriter())
    conv_canon_pipeline.add_pass(HloPass.GpuConvPaddingLegalization())
    conv_canon_pipeline.add_pass(HloPass.CudnnPadForConvolutions())
    conv_canon_pipeline.add_pass(HloPass.CudnnVectorizeConvolutions())
    conv_canon_pipeline.add_pass(HloPass.CallInliner())
    conv_canon_pipeline.add_pass(HloPass.TupleSimplifier())
    algebraic_config_options = {
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
    }
    conv_canon_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(options=algebraic_config_options),
      loop_count=-1
    )
    conv_canon_pipeline.add_pass(HloPass.HloConstantFolding())
    pre_fusion_pipeline.add_pass(conv_canon_pipeline)

    # --------------------------------------------
    # Layout Assignment Pipeline
    # --------------------------------------------

    layout_assignment_pipeline = Pipeline("layout-assignment")
    layout_assignment_pipeline.add_pass(HloPass.FlattenCallGraph())
    layout_assignment_pipeline.add_pass(
      HloPass.GpuLayoutAssignment(hlo_module)
    )
    pre_fusion_pipeline.add_pass(layout_assignment_pipeline)

    # --------------------------------------------
    # Post Layout Assignment Pipeline
    # --------------------------------------------

    # *******************
    # NVIDIA GPU Specific Passes Stage 1 - START
    post_layout_ass_pipeline_nv_pre = Pipeline("post-layout-assignment-nv-pre")
    if (GpuBackend.cuda_is_at_least(GpuBackend.CudaComputeCapability.AMPERE)):
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.BF16,
          pad_to_multiple_of=8
        )
      )

    if (GpuBackend.cuda_is_at_least(GpuBackend.CudaComputeCapability.VOLTA)):
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.S8,
          pad_to_multiple_of=4
        )
      )
      post_layout_ass_pipeline_nv_pre.add_pass(
        HloPass.CublasPadForGemms(
          datatype=HloPass.CublasPadForGemms.PrimitiveType.F16,
          pad_to_multiple_of=8
        )
      )

    post_layout_ass_pipeline_nv_pre.add_pass(HloPass.HloConstantFolding())
    # NVIDIA GPU Specific Passes Stage 1 - END
    # *******************

    post_layout_ass_pipeline = Pipeline("post-layout-assignment")
    post_layout_ass_pipeline.add_pass(HloPass.ReductionDegenerateDimRemover())
    post_layout_ass_pipeline.add_pass(HloPass.ReductionLayoutNormalizer())
    post_layout_ass_pipeline.add_pass(HloPass.ReductionDimensionGrouper())
    post_layout_ass_pipeline.add_pass(
      HloPass.ReductionSplitter(), loop_count=-1
    )
    post_layout_ass_pipeline.add_pass(
      HloPass.GpuTreeReductionRewriter(), loop_count=-1
    )

    algebraic_config_options = {
      "is_layout_sensitive": True,
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    post_layout_ass_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(algebraic_config_options), loop_count=-1
    )
    post_layout_ass_pipeline.add_pass(HloPass.TransposeFolding())
    post_layout_ass_pipeline.add_pass(HloPass.GemmRewriter())
    post_layout_ass_pipeline.add_pass(HloPass.GemmBroadcastFoldingRewriter())
    post_layout_ass_pipeline.add_pass(HloPass.BFloat16Normalization(False))
    post_layout_ass_pipeline.add_pass(HloPass.GpuConvAlgorithmPicker())
    post_layout_ass_pipeline.add_pass(HloPass.TupleSimplifier())
    post_layout_ass_pipeline.add_pass(HloPass.HloCSE(True))

    # *******************
    # NVIDIA GPU Specific Passes Stage 2 - START
    post_layout_ass_pipeline_nv_post = Pipeline(
      "post-layout-assignment-nv-post"
    )

    post_layout_ass_pipeline_nv_post.add_pass(HloPass.GemmAlgorithmPicker())
    if (hlo_module.is_bef_enabled):
      post_layout_ass_pipeline_nv_post.add_pass(
        HloPass.TriangularSolveRewriter()
      )
    # NVIDIA GPU Specific Passes Stage 2 - END
    # *******************

    pre_fusion_pipeline.add_pass(post_layout_ass_pipeline_nv_pre)
    pre_fusion_pipeline.add_pass(post_layout_ass_pipeline)
    pre_fusion_pipeline.add_pass(post_layout_ass_pipeline_nv_post)

    #  -----------------------------------------------------------------------
    #                               FUSION PIPELINE
    #  -----------------------------------------------------------------------

    fusion_pipeline = Pipeline("fusion")

    # --------------------------------------------
    # Vertical Fusion Pipeline
    # --------------------------------------------

    vert_fusion_pipeline = Pipeline("vertical-fusion", loop_count=-1)
    vert_fusion_pipeline.add_pass(HloPass.VariadicOpSplitter())
    vert_fusion_pipeline.add_pass(HloPass.GpuInstructionFusion(False))
    vert_fusion_pipeline.add_pass(HloPass.GpuInstructionFusion(True))
    vert_fusion_pipeline.add_pass(HloPass.FusionMerger())
    vert_fusion_pipeline.add_pass(HloPass.GpuMultiOutputFusion())
    vert_fusion_pipeline.add_pass(HloPass.HloCSE(True, True))
    vert_fusion_pipeline.add_pass(HloPass.HloDCE())

    # --------------------------------------------
    # Horizontal Fusion Pipeline
    # --------------------------------------------

    hori_fusion_pipeline = Pipeline("horizontal-fusion", loop_count=-1)
    hori_fusion_pipeline.add_pass(HloPass.GpuHorizontalLoopFusion())
    hori_fusion_pipeline.add_pass(HloPass.GpuHorizontalInputFusion())
    hori_fusion_pipeline.add_pass(HloPass.HloCSE(True, True))
    hori_fusion_pipeline.add_pass(HloPass.HloDCE())

    fusion_pipeline.add_pass(vert_fusion_pipeline)
    fusion_pipeline.add_pass(hori_fusion_pipeline)

    #  -----------------------------------------------------------------------
    #                               POST PIPELINE
    #  -----------------------------------------------------------------------

    post_fusion_pipeline = Pipeline("fusion")

    post_fusion_pipeline.add_pass(
      HloPass.AllGatherCombiner(
        combine_threshold_in_bytes=1024 * 1024 * 1024,
        combine_threshold_count=256
      )
    )
    post_fusion_pipeline.add_pass(
      HloPass.AllReduceCombiner(
        combine_threshold_in_bytes=debug_options
        .xla_gpu_all_reduce_combine_threshold_bytes,
        combine_threshold_count=256
      )
    )
    post_fusion_pipeline.add_pass(
      HloPass.ReduceScatterCombiner(
        combine_threshold_in_bytes=30 * 1024 * 1024,
        combine_threshold_count=256
      )
    )

    if debug_options.xla_gpu_all_reduce_contiguous:
      post_fusion_pipeline.add_pass(HloPass.AllReduceContiguous())

    blueconnect_num_devices_per_host = debug_options.xla_gpu_all_reduce_blueconnect_num_devices_per_host
    if (blueconnect_num_devices_per_host > 0):
      post_fusion_pipeline.add_pass(
        HloPass.AllReduceBlueConnect(blueconnect_num_devices_per_host)
      )

    if debug_options.xla_gpu_enable_async_all_reduce:
      post_fusion_pipeline.add_pass(HloPass.AsyncCollectiveCreator())

    post_fusion_pipeline.add_pass(HloPass.CollectivesScheduleLinearizer())

    algebraic_config_options = {
      "is_layout_sensitive": True,
      "replace_transpose_with_bitcast": False,
      "enable_conv_operand_swap": False,
      "minmax_propagate_nan": debug_options.xla_gpu_enable_fast_min_max,
    }
    post_fusion_pipeline.add_pass(
      HloPass.AlgebraicSimplifier(algebraic_config_options)
    )
    post_fusion_pipeline.add_pass(HloPass.OptimizationBarrierExpander())
    post_fusion_pipeline.add_pass(HloPass.TupleSimplifier())

    #  -----------------------------------------------------------------------
    #                        FULL OPTIMIZE HLO MODULE PIPELINE
    #  -----------------------------------------------------------------------

    optimize_hlo_pipeline = Pipeline("fusion")

    optimize_hlo_pipeline.add_pass(pre_fusion_pipeline)
    optimize_hlo_pipeline.add_pass(fusion_pipeline)
    optimize_hlo_pipeline.add_pass(post_fusion_pipeline)

    hlo_env.run(optimize_hlo_pipeline)
    general_pipeline_hash = hlo_env.get_hlo_module_hash()

    hlo_env = HloEnv(self.hlo_main_test_file, "gpu")
    hlo_env.optimize_hlo_module()
    original_pipeline_hash = hlo_env.get_hlo_module_hash()

    assert (general_pipeline_hash == original_pipeline_hash)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_general_fusion(self) -> None:
    from random import randrange

    import numpy as np
    from altgraph import HloEnv, AltPipeline, HloPass, Pass, Pipeline

    # Note you have to make an Pass, cannot just run the HloPass directly.
    general_fusion_dry_pass = AltPipeline(Pass(HloPass.GeneralFusion(),))

    post_general_fusion_dry_passes = Pipeline("post-general-fusion")
    post_general_fusion_dry_passes.add_pass(HloPass.HloCSE(True, True))
    post_general_fusion_dry_passes.add_pass(HloPass.HloDCE())

    base_dir = os.path.dirname(os.path.realpath(__file__))
    hlo_base_dir = base_dir + "/hlo_texts/test_hlos"
    for root, dirs, files in os.walk(hlo_base_dir):
      for file in files:

        filepath = os.path.join(root, file)
        logging.info("Testing general fusion for file: " + filepath)

        hlo_env = HloEnv(filepath, "gpu")

        saved_hlo_module = hlo_env.save_hlo()
        # Original TF pipelines
        hlo_env.optimize_hlo_module()
        hlo_env.prepare_hlo_module_for_ir_emitting()

        # Save reference copy of the module after a non dry-run RunHloPasses call
        reference_hlo_module = hlo_env.save_hlo()
        hlo_env.load_hlo(saved_hlo_module)

        hlo_env.pre_fusion_optimizations()
        num_alts = 1
        while num_alts > 0:
          hlo_env.pre_fusion_dry_passes()
          hlo_env.run(general_fusion_dry_pass)

          hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
          node_features = hlo_graph.node_features
          num_operands = node_features.num_operands
          num_alts = len(hlo_graph.alternative_indices)

          if num_alts > 0:
            decisions = []
            for alt_idx in hlo_graph.alternative_indices:
              node_uid = node_features.uids[alt_idx]
              decisions.append(
                [alt_idx, get_rand_action(num_operands[alt_idx])]
              )

            decisions = np.asarray(decisions)
            hlo_env.apply_alternatives(decisions)

            hlo_env.run(post_general_fusion_dry_passes)

        hlo_env.post_fusion_optimizations()
        post_fusion_module = hlo_env.save_hlo()

        assert (
          hlo_env.has_equal_output(post_fusion_module, reference_hlo_module)
        )

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_deterministic_node_ids(self) -> None:
    from random import randrange
    import numpy as np
    from altgraph import HloEnv, AltPipeline, HloPass, Pass, Pipeline

    general_fusion_dry_pass = AltPipeline(Pass(HloPass.GeneralFusion(),))
    post_general_fusion_dry_passes = Pipeline("post-general-fusion")
    post_general_fusion_dry_passes.add_pass(HloPass.HloCSE(True, True))
    post_general_fusion_dry_passes.add_pass(HloPass.HloDCE())

    # filepath = "./hlo_texts/large_hlo.txt"
    filepath = "hlo_texts/test_hlos/maml_flax/module_0082.jit_divmod.28.before_optimizations.txt"
    print("Testing general fusion for file: " + filepath)

    all_alt_indices = []
    all_alt_ids = []
    all_dag_hashes = []

    for i in range(10):
      hlo_env = HloEnv(self.hlo_main_test_file, "gpu")

      run_alt_indices = []
      run_alt_ids = []

      hashes = []
      hashes.append(hlo_env.get_hlo_module_hash())

      hlo_env.pre_fusion_optimizations()
      hashes.append(hlo_env.get_hlo_module_hash())

      num_alts = 1
      while num_alts > 0:
        hlo_env.pre_fusion_dry_passes()
        hashes.append(hlo_env.get_hlo_module_hash())

        hlo_env.run(general_fusion_dry_pass)
        hashes.append(hlo_env.get_hlo_module_hash())

        hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
        node_features = hlo_graph.node_features
        num_operands = node_features.num_operands
        num_alts = len(hlo_graph.alternative_indices)

        alt_ids = [
          node_features.uids[idx] for idx in hlo_graph.alternative_indices
        ]
        run_alt_indices += list(hlo_graph.alternative_indices)
        run_alt_ids += alt_ids

        if num_alts > 0:
          decisions = []
          for alt_idx in hlo_graph.alternative_indices:
            node_uid = node_features.uids[alt_idx]
            decisions.append([alt_idx, 1])

          decisions = np.asarray(decisions)
          hlo_env.apply_alternatives(decisions)
          hashes.append(hlo_env.get_hlo_module_hash())

          hlo_env.run(post_general_fusion_dry_passes)
          hashes.append(hlo_env.get_hlo_module_hash())

      hlo_env.post_fusion_optimizations()
      hashes.append(hlo_env.get_hlo_module_hash())
      all_dag_hashes.append(hashes)
      all_alt_indices.append(run_alt_indices)
      all_alt_ids.append(run_alt_ids)

    assert (all_alt_indices.count(all_alt_indices[0]) == len(all_alt_indices))
    assert (all_alt_ids.count(all_alt_ids[0]) == len(all_alt_ids))
    assert (all_dag_hashes.count(all_dag_hashes[0]) == len(all_dag_hashes))

    logging.info("---------IDX--------")
    for alt_idx in all_alt_indices:
      logging.info(alt_idx[:25])

    logging.info("--------IDS---------")
    for alt_id in all_alt_ids:
      logging.info(alt_id[:25])

    logging.info("-------HASH--------")
    for dag_hash in all_dag_hashes:
      logging.info(dag_hash)


if __name__ == "__main__":
  absltest.main()
