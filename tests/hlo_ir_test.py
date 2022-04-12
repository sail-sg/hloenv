import os
from absl import logging
from absl.testing import absltest


class HloIRTest(absltest.TestCase):
  """Placeholder for some real tests
  """

  def setUp(self) -> None:
    logging.set_verbosity(logging.INFO)
    logging.info("setting up")
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/hlo_texts"

    self.hlo_main_test_file = dir_path + "/hlo_test.txt"

  def test_import(self) -> None:
    import tensorflow
    import altgraph
    logging.info("altgraph module imported at %s", altgraph)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_graph_interfaces(self) -> None:
    from altgraph import HloIr
    hlo_ir = HloIr(self.hlo_main_test_file, "gpu")
    hlo_graph = hlo_ir.get_hlo_graph()

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
    assert (num_out_edges == num_out_edges)
    assert (len(out_edge_features.srcs) == num_out_edges)
    assert (len(out_edge_features.dsts) == num_out_edges)
    assert (len(out_edge_features.dims) == num_out_edges * 8)
    assert (len(out_edge_features.layouts) == num_out_edges * 8)
    assert (len(out_edge_features.dtypes) == num_out_edges)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_basic(self) -> None:
    import tensorflow
    from altgraph import HloIr
    from random import randrange
    import numpy as np

    hlo_ir = HloIr(self.hlo_main_test_file, "gpu")

    hlo_ir.pre_fusion_optimizations()

    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_ir.fusion_dry_run()
      hlo_graph = hlo_ir.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        # pass the decision back to compilerp
        logging.info("Applying alternatives...")
        hlo_ir.apply_alternatives(decisions)
      else:
        logging.info("No more alternatives, ending run...")
      eval_result = hlo_ir.evaluate(10)
      for eval_time_ns in eval_result.durations:
        assert eval_time_ns > 0
      logging.info("Running time estimation: %d ns", eval_time_ns / 10)

      count += 1

    logging.info("Running post_fusion_optimizations...")
    hlo_ir.post_fusion_optimizations()

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_save_restore(self) -> None:
    from altgraph import HloIr
    hlo_ir = HloIr(self.hlo_main_test_file, "gpu")

    init_hlo_str = hlo_ir.export_hlo_to_str()
    saved_hlo_module = hlo_ir.save_hlo()
    hlo_ir.pre_fusion_optimizations()
    post_fusion_hlo_str = hlo_ir.export_hlo_to_str()
    hlo_ir.restore_hlo(saved_hlo_module)
    restored_hlo_str = hlo_ir.export_hlo_to_str()
    assert (init_hlo_str != post_fusion_hlo_str)
    assert (init_hlo_str == restored_hlo_str)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_evaluation(self) -> None:
    from altgraph import HloIr
    from random import randrange
    import numpy as np

    hlo_ir = HloIr(self.hlo_main_test_file, "gpu")

    hlo_ir.pre_fusion_optimizations()
    saved_hlo_module = hlo_ir.save_hlo()
    # Restore back to original, where we only did pre_fusion_optimizations
    hlo_ir.post_fusion_optimizations()

    orig_res = hlo_ir.evaluate(1)
    orig_post_opt_module = hlo_ir.save_hlo()

    hlo_ir.restore_hlo(saved_hlo_module)

    num_alts = 1
    while num_alts > 0:
      hlo_ir.fusion_dry_run()
      hlo_graph = hlo_ir.get_hlo_graph(do_hash_verification=False)
      node_features = hlo_graph.node_features
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.alternative_indices)

      if num_alts > 0:
        decisions = []
        for alt_idx in hlo_graph.alternative_indices:
          decisions.append([alt_idx, randrange(num_operands[alt_idx])])

        decisions = np.asarray(decisions)
        hlo_ir.apply_alternatives(decisions)

    hlo_ir.post_fusion_optimizations()
    mod_res = hlo_ir.evaluate(1)
    assert (hlo_ir.has_equal_output_as(orig_post_opt_module))

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
    from altgraph import HloIr
    from random import randrange
    import numpy as np

    base_dir = os.path.dirname(os.path.realpath(__file__))
    hlo_base_dir = base_dir + "/hlo_texts/test_hlos"
    for root, dirs, files in os.walk(hlo_base_dir):
      for file in files:

        filepath = os.path.join(root, file)
        logging.info("Testing validation for file: " + filepath)

        hlo_ir = HloIr(filepath, "gpu")

        saved_hlo_module = hlo_ir.save_hlo()
        hlo_ir.original_run_hlo_passes()
        # Save reference copy of the module after a non dry-run RunHloPasses call
        reference_hlo_module = hlo_ir.save_hlo()
        hlo_ir.restore_hlo(saved_hlo_module)

        hlo_ir.pre_fusion_optimizations()
        num_alts = 1
        while num_alts > 0:
          hlo_ir.fusion_dry_run()
          hlo_graph = hlo_ir.get_hlo_graph(do_hash_verification=False)
          node_features = hlo_graph.node_features
          num_operands = node_features.num_operands
          num_alts = len(hlo_graph.alternative_indices)

          if num_alts > 0:
            decisions = []
            for alt_idx in hlo_graph.alternative_indices:
              decisions.append([alt_idx, randrange(num_operands[alt_idx])])

            decisions = np.asarray(decisions)
            hlo_ir.apply_alternatives(decisions)

        hlo_ir.post_fusion_optimizations()
        post_fusion_module = hlo_ir.save_hlo()

        assert (
          hlo_ir.has_equal_output(post_fusion_module, reference_hlo_module)
        )


if __name__ == "__main__":
  absltest.main()
