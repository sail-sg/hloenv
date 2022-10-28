import os
import pathlib
import altgraph
import numpy as np
import tensorflow as tf
import ipdb
from altgraph import AltPipeline, HloEnv, HloPass, Pass, Pipeline
from general_fusion_pipeline import GeneralFusionPipeline
from typing import Tuple


def get_ragged_tensor_from_hlo(
  hlo_graph
) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
  """
  get the operands and users of each node from the hlo graph.
  operands: a dict in the form of raggedtensor, where the key is the node id,
  and the value is a list of operand indices.
  users: a dict in the form of raggedtensor, where the key is the node id,
  and the value is a list of user indices.
  input: hlo graph:
  output: a tuple of ragged tensor indicating the graph structure:
  (operands, users)
  """
  in_edge_features = hlo_graph.in_edge_features
  out_edge_features = hlo_graph.out_edge_features

  operands = tf.cast(
    tf.RaggedTensor.from_row_splits(
      values=in_edge_features.srcs, row_splits=hlo_graph.in_edge_offsets
    ), tf.int64
  )

  users = tf.cast(
    tf.RaggedTensor.from_row_splits(
      values=out_edge_features.dsts, row_splits=hlo_graph.out_edge_offsets
    ), tf.int64
  )

  return operands, users


def uniform_policy(hlo_graph) -> tf.RaggedTensor:
  """
  generate a uniform random score for each operand of each alternative.
  input: hlo_graph
  output: a tf.RaggedTensor with shape [num_alt_idx, num_operands].
  Each row is a list of probability to operand indices for the 
  corresponding alternative.
  """
  operands, users = get_ragged_tensor_from_hlo(hlo_graph)

  alternative_idx = tf.convert_to_tensor(hlo_graph.alternative_indices)

  alt_oprnd_idx: tf.RaggedTensor = tf.gather(operands, alternative_idx)

  # assign random score to each operand
  alt_oprnd_prob = tf.map_fn(
    lambda x: tf.random.uniform(shape=x.shape, minval=0, maxval=1),
    alt_oprnd_idx,
    fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32)
  )

  return alt_oprnd_prob


def argmax_sample(probability: tf.RaggedTensor, hlo_graph) -> tf.Tensor:
  """
  selecting the operand with the highest score for each alternative.
  input: 
    probability: a tf.RaggedTensor with shape [num_alt_idx, num_operands].
      Each row is a list of probability to operand indices for the 
      corresponding alternative.
    hlo_graph: the hlo graph
  output: a tf.Tensor with shape [num_alt_idx, 2], the 1st column is
  the alt_idx, the 2nd column is the operand_idx to be selected.
  """
  alternative_idx = tf.convert_to_tensor(
    hlo_graph.alternative_indices, dtype=tf.int64
  )

  alt_choice = tf.map_fn(
    lambda x: tf.argmax(x, axis=0),
    probability,
    fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int64)
  )

  return tf.stack([alternative_idx, alt_choice], axis=1)


if __name__ == "__main__":
  hlo_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "hlo_texts/jax-md/module_0013.jit__lambda_.7.before_optimizations.txt"
  )
  hlo_env = HloEnv(hlo_path, "gpu")
  general_fusion_pipeline = GeneralFusionPipeline(hlo_env)

  hlo_env.run(general_fusion_pipeline.pre_pass_optimizations)

  num_alts = 1
  while num_alts > 0:
    hlo_env.run(general_fusion_pipeline.pre_dry_pass_passes)
    hlo_env.run(general_fusion_pipeline.pass_dry_run)

    hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
    num_alts = len(hlo_graph.alternative_indices)

    if num_alts > 0:
      probablity = uniform_policy(hlo_graph)
      decisions = argmax_sample(probablity, hlo_graph)
      decisions = np.asarray(decisions)
      hlo_env.apply_alternatives(decisions)
      hlo_env.run(general_fusion_pipeline.post_dry_pass_passes)

  hlo_env.run(general_fusion_pipeline.post_pass_optimizations)
  hlo_env.prepare_hlo_module_for_ir_emitting()

  results = hlo_env.evaluate(100)
  timing = min(results.durations)
  print(f"Timing: {timing}")
