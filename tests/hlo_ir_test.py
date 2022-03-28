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
    from altgraph import PyHloIr
    hlo_ir = PyHloIr(self.hlo_file, "gpu")
    hlo_graph = hlo_ir.get_hlo_graph()

    _ = hlo_graph.get_out_edge_offsets()
    _ = hlo_graph.get_out_edge_indices()
    _ = hlo_graph.get_in_edge_offsets()
    _ = hlo_graph.get_in_edge_indices()

    _ = hlo_graph.get_alternative_indices()
    _ = hlo_graph.hash()
    node_features = hlo_graph.get_node_features()
    in_edge_features = hlo_graph.get_in_edge_features()
    out_edge_features = hlo_graph.get_out_edge_features()

    _ = node_features.uids
    _ = node_features.names
    _ = node_features.gids
    _ = node_features.num_users
    _ = node_features.num_operands
    _ = node_features.is_alternative
    _ = node_features.in_tensor_sizes
    _ = node_features.out_tensor_sizes
    _ = node_features.has_max_in_tensor
    _ = node_features.has_max_out_tensor

    _ = in_edge_features.uids
    _ = in_edge_features.srcs
    _ = in_edge_features.dsts
    _ = in_edge_features.dims
    _ = in_edge_features.layouts
    _ = in_edge_features.dtypes
    _ = in_edge_features.get_tensor_size(0)

    _ = out_edge_features.uids
    _ = out_edge_features.srcs
    _ = out_edge_features.dsts
    _ = out_edge_features.dims
    _ = out_edge_features.layouts
    _ = out_edge_features.dtypes

  @absltest.skipIf(("GITLAB_CI" in os.environ), "Running in gitlab ci")
  def test_basic(self) -> None:
    from altgraph import PyHloIr
    from random import randrange
    import numpy as np

    hlo_ir = PyHloIr(self.hlo_file, "gpu")

    hlo_ir.pre_fusion_optimizations()
    hlo_graph = hlo_ir.get_hlo_graph()
    num_alts = 1
    count = 1
    while num_alts > 0:
      logging.info("\n*****************************************")
      logging.info("Pass: %d" % count)
      logging.info("Running fusion dry run")
      hlo_ir.fusion_dry_run()
      hlo_graph = hlo_ir.get_hlo_graph()
      node_features = hlo_graph.get_node_features()
      num_operands = node_features.num_operands
      num_alts = len(hlo_graph.get_alternative_indices())

      if num_alts > 0:
        logging.info("Generating decisions...")
        decisions = []
        for alt_idx in hlo_graph.get_alternative_indices():
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
