import os

from absl import logging
from absl.testing import absltest


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
    assert (len(in_edge_features.dtypes) == num_in_edges)
    _ = in_edge_features.get_tensor_size(0)

    num_out_edges = len(out_edge_features.uids)
    assert (len(out_edge_features.srcs) == num_out_edges)
    assert (len(out_edge_features.dsts) == num_out_edges)
    assert (len(out_edge_features.dims) == num_out_edges * 8)
    assert (len(out_edge_features.layouts) == num_out_edges * 8)
    assert (len(out_edge_features.dtypes) == num_out_edges)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_py_hlo_module(self) -> None:
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
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_env.apply_alternatives(decisions)
        hlo_env.post_fusion_dry_passes()
      else:
        logging.info("No more alternatives, ending run...")
      eval_result = hlo_env.evaluate(10)
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
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
    import tensorflow
    from altgraph import HloEnv
    from random import randrange
    import numpy as np

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
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
        hlo_env.original_run_hlo_passes()
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
              decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
    from altgraph import HloEnv
    from random import randrange
    import numpy as np

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
              decisions.append([alt_idx, randrange(num_operands[alt_idx])])

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
    from altgraph import HloEnv
    from random import randrange
    import numpy as np

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


if __name__ == "__main__":
  absltest.main()
