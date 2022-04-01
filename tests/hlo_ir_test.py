import os
from absl import logging
from absl.testing import absltest


class HloIRTest(absltest.TestCase):
  """Placeholder for some real tests
  """

  def setUp(self) -> None:
    logging.set_verbosity(logging.INFO)
    logging.info("setting up")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    self.hlo_file = dir_path + "/hlo_test.txt"

  def test_import(self) -> None:
    import altgraph
    logging.info("altgraph module imported at %s", altgraph)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_graph_interfaces(self) -> None:
    from altgraph import HloIr
    hlo_ir = HloIr(self.hlo_file, "gpu")
    hlo_graph = hlo_ir.get_hlo_graph()

    assert(len(hlo_graph.out_edge_offsets) > 0)
    assert(len(hlo_graph.out_edge_indices) > 0)
    assert(len(hlo_graph.in_edge_offsets) > 0)
    assert(len(hlo_graph.in_edge_indices) > 0)

    _ = hlo_graph.alternative_indices
    _ = hlo_graph.hash()
    node_features = hlo_graph.node_features
    in_edge_features = hlo_graph.in_edge_features
    out_edge_features = hlo_graph.out_edge_features

    num_nodes = len(node_features.uids)
    assert(num_nodes > 0)
    assert(len(node_features.names) == num_nodes)
    assert(len(node_features.gids) == num_nodes)
    assert(len(node_features.num_users) == num_nodes)
    assert(len(node_features.num_operands) == num_nodes)
    assert(len(node_features.opcodes) == num_nodes)
    assert(len(node_features.is_alternative) == num_nodes)
    assert(len(node_features.in_tensor_sizes) == num_nodes)
    assert(len(node_features.out_tensor_sizes) == num_nodes)
    _ = node_features.has_max_in_tensor
    _ = node_features.has_max_out_tensor

    num_in_edges = len(in_edge_features.uids)
    assert(num_in_edges > 0)
    assert(len(in_edge_features.srcs) == num_in_edges)
    assert(len(in_edge_features.dsts) == num_in_edges)
    assert(len(in_edge_features.dims) == num_in_edges * 8)
    assert(len(in_edge_features.layouts) == num_in_edges * 8)
    assert(len(in_edge_features.dtypes) == num_in_edges)
    _ = in_edge_features.get_tensor_size(0)

    num_out_edges = len(out_edge_features.uids)
    assert(num_out_edges == num_out_edges)
    assert(len(out_edge_features.srcs) == num_out_edges)
    assert(len(out_edge_features.dsts) == num_out_edges)
    assert(len(out_edge_features.dims) == num_out_edges * 8)
    assert(len(out_edge_features.layouts) == num_out_edges * 8)
    assert(len(out_edge_features.dtypes) == num_out_edges)

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_basic(self) -> None:
    from altgraph import HloIr
    from random import randrange
    import numpy as np

    hlo_ir = HloIr(self.hlo_file, "gpu")

    hlo_ir.pre_fusion_optimizations()
    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_ir.fusion_dry_run()
      hlo_graph = hlo_ir.get_hlo_graph()
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
      eval_time_ns = hlo_ir.evaluate(10)
      assert eval_time_ns > 0;
      logging.info("Running time estimation: %d ns", eval_time_ns / 10)

      count += 1

    # TODO(OCY) Obtain actual executable
    logging.info("Running post_fusion_optimizations...")
    hlo_ir.post_fusion_optimizations()


if __name__ == "__main__":
  absltest.main()

