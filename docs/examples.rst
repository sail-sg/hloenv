.. _examples:

Examples Using HloEnv
========================

This documentation only covers the Python interface. We will walk through two simple examples that make use of several HloEnv features. We first show how to read in an HLO text file and turn it into HLO graph features which can be used for implementing a user-defined decision-making agent. We then present a very simple decision-making agent where the policy randomly choose from available actions.

Playing with HLO Graph Features
-------------------------------

First make sure your current working directory is correct.

.. code-block:: bash

    $ cd /path/to/hloenv/examples
    
The HloEnv module holds most functionality, so we usually import it first.

.. code-block:: python

    import os
    import pathlib
    import numpy as np
    from hloenv import HloEnv

Pick one hlo text file that we want to take a closer look.

.. code-block:: python

    hlo_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
      "hlo_texts/jax-md/module_0013.jit__lambda_.7.before_optimizations.txt"
    )
  
Now we are ready to create a basic hlo env object on GPU backend. We haven't worked on other ML hardwares so current GPU is the only backend HloEnv supports.

.. code-block:: python

    hlo_env = HloEnv(hlo_path, "gpu")
    
The hlo env can automatically extract features from hlo text files and organize them into another class HloGraph.

.. code-block:: python
    
    hlo_graph = hlo_env.get_hlo_graph()
    
To make the features more array programming friendly, all the graph features in HloGraph are organized in the form of `CSR <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_. There are three types of features: global graph feature, node feature, and in/out edge feature, all serve as the accessible members of HloGraph object. Details are in the following table:

.. list-table:: **Global Graph Features**
    :widths: 42 42
    :header-rows: 1
    
    * - Feature Name
      - Description
      
    * - out_edge_offsets
      - The offset index to the actual out edge node ID indices array
      
    * - out_edge_indices
      - The out edge node ID indices array
    
    * - in_edge_offsets
      - The offset index to the actual in edge node ID indices array
      
    * - in_edge_indices
      - The in edge node ID indices array
      
    * - alternative_indices
      - The indices to all the *kAlternative* nodes
      
    * - opcode_attr_counts
      - Number of attributes in HLO opcode
      
All edge features are vectors of the length of number of edges in the HLO graph. In/Out edge features share the same feature set as follows.

.. list-table:: **In and Out Edge Features**
    :widths: 42 42
    :header-rows: 1
    
    * - Feature Name
      - Description
      
    * - uids
      - Unique ID of the edge, a concatination of source and destination nodes uids
       
    * - srcs
      - Node index of source node
    
    * - dsts
      - Node index of destination node
      
    * - dims
      - Dimension of the tensor flows by this edge
    
    * - layout
      - Layout of the tensor flows by this edge
      
    * - lehmercodes
      - The `Lehmer code <https://en.wikipedia.org/wiki/Lehmer_code>`_ (a better embedding) of the tensor layout
      
    * - types
      - Edge type is one of the following: outside any fusion, inside fusion, and cross fusion
      
    * - dtypes
      - Data type of the tensor flows by this edge

All node features are vectors of the length of number of HloInstructions (nodes) in the HloModule (HloGraph).

.. list-table:: **Node Features**
    :widths: 42 42
    :header-rows: 1
    
    * - Feature Name
      - Description
      
    * - uids
      - Unique ID of an HloInstruction
      
    * - gids
      - Sub computation ID the HloInstruction belongs to, 0 means in main computation.
      
    * - normalized_num_group_inst
      - If an HloInstruction is inside a sub-computation, normalized_num_group_inst is the reciprocal of the total number of instructions in a sub-computation. This can serve as a weighting parameter for an instruction's impact
        
    * - num_users
      - Number of HloInstructions that uses the result of this HloInstruction
        
    * - num_operands
      - Number of HloInstructions whose results this HloInstruction uses
        
    * - opcodes
      - HLO opcode index, as defined `here <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/hlo/ir/hlo_opcode.h#L50>`__
        
    * - opcode_attrs
      - Unique attribute embeddings for each opcode
        
    * - num_opcode_attrs
      - List of pairs, each pair contains the number of integer attribute and the number of enum attribute in opcode_attrs
        
    * - is_alternative
      - List of boolean that shows if the HloInstruction is *kAlternative*
      
    * - is_in_fusion
      - List of boolean that shows if the HloInstruction is inside a fused computation
      
    * - in_tensor_sizes
      - The total input tensor size from all operands of this HloInstruction
        
    * - out_tensor_sizes
      - The output tensor size of this HloInstruction
        
    * - has_max_in_tensor
      - List of boolean that shows if one of the operands contains the max input tensor size
        
    * - has_max_out_tensor
      - List of boolean that shows if the output tensor size has the maximum size
        
    * - names
      - List of strings that shows the names of the HloInstruction
      
The full-size code can be found `here <https://github.com/sail-sg/hloenv/blob/altgraph-refactor-open/examples/hlo_play.py>`__. In our second example, we will show you how to use these features to create a simple decision-making agent and run XLA optimizations using it.

Defining a Custom Pipeline
--------------------------

All of the passes that were initially defined in the full XLA optimization pipeline for compilation for GPU have been included in :class:`~hloenv.HloEnv`. A list of all of these passes can be found in :ref:`xla_passes`.

A pass can be created by wrapping a :class:`~hloenv.HloPass` object within a :class:`~hloenv.Pass` object. This Pass can then be run within the :class:`~hloenv.HloEnv` to modify the :class:`~hloenv.HloModule` object loaded in the HloEnv.

.. code-block:: python

  hlo_env = HloEnv("path/to/hlo.txt", "gpu")
  fusion_pass = Pass(HloPass.GpuInstructionFusion(True))
  hlo_env.run(fusion_pass)

Passes can be organized into :class:`~hloenv.Pipeline` objects, which can also be run by the HloEnv.

.. code-block:: python

  op_splitter_pass = Pass(HloPass.VariadicOpSplitter())

  fusion_pipeline = Pipeline("fusion-pre")
  fusion_pipeline.add_pass(op_splitter_pass)

  # You can also add a HloPass directly without first wrapping it in Pass
  fusion_pipeline.add_pass(HloPass.GpuInstructionFusion(True))

  hlo_env.run(fusion_pipeline)

A Pipeline can contain other Pipelines and Passes recursively, for example:

.. code-block:: python

  fusion_pipeline_pre = Pipeline("fusion-pre")
  fusion_pipeline_pre.add_pass(HloPass.VariadicOpSplitter())

  self.fusion_pipeline_full = Pipeline("fusion")
  self.fusion_pipeline_full.add(fusion_pipeline_pre)
  self.fusion_pipeline_full.add(HloPass.GpuInstructionFusion(True))

To convert a pass into a `dry pass` we wrap it in an :class:`~hloenv.AltPipeline`. All rewrites performed by an AltPiplines are captures and converted into alternatives. If you wrap multiple passes or pipelines with this AltPipeline wrapper, `all` modifcations performed by each of those passes/pipelines will be in dry mode.

.. code-block:: python

  # Wrapping a single pass in an AltPipeline
  fusion_dry_pass = AltPipeline(
    Pass(
      HloPass.GpuInstructionFusion(True),  # may_duplicate
    )
  )

  fusion_pipeline_post = Pipeline("fusion-post")
  fusion_pipeline_post.add_pass(HloPass.FusionMerger())
  fusion_pipeline_post.add_pass(HloPass.GpuMultiOutputFusion())
  fusion_pipeline_post.add_pass(HloPass.HloCSE(True, True))

  # Wrapping multiple passes in an AltPipeline
  fusion_pipeline_post_dry = AltPipeline(fusion_pipeline_post)

  # Adding a pass directly to an AltPipeline
  fusion_pipeline_post_dry.add_pass(HloPass.HloDCE())

A sample General Fusion Pipeline can be found in `examples/general_fusion_pipeline.py` which contains the full XLA optimization pipeline, except we replace the vertical fusion pipeline with our custom General Fusion pass.

A Simple Decision-making Agent
------------------------------

We here present a very simple decision-making agent that randomly chooses from available actions in an optimization loop. 
The loop will isolate out the graph rewrite in an XLA pass, and layout the decisions to choose.
At a high level, the optimization loop follows these steps:

* run `pre_pass_optimizations`
* enter optimization loop
    * run `pre_dry_pass_passes`
    * open `pass_dry_run`
    * choose an action
    * apply the action
    * run `post_dry_pass_passes`
* run `post_pass_optimizations`

We can regard the `pre_pass_optimizations` as the `pre-processing` stage and `post_pass_optimizations` as the `post-processing` stage. 
So they are not included in the optimization loop. 

Every step of `pass_dry_run` will expose the alternatives (i.e. action space) to users. 
Note that it is also surrounded by `pre_dry_pass_passes` and `post_dry_pass_passes` for some pre/post processing. They are included in the optimization loop.

Here we are interested in `GeneralFusion` pipeline. All the above described steps are implemented and scheduled in the `GeneralFusionPipeline` class, which is a sample pipeline that we have provided in examples/general_fusion_pipeline.py

.. code-block:: python

  from general_fusion_pipeline import GeneralFusionPipeline
  from hloenv import AltPipeline, HloEnv, HloPass, Pass, Pipeline

  hlo_env = HloEnv(hlo_path, "gpu")
  general_fusion_pipeline = GeneralFusionPipeline(hlo_env)

The code of the optimization loop looks like this:

.. code-block:: python

  hlo_env.run(general_fusion_pipeline.pre_pass_optimizations)

  num_alts = 1
  while num_alts > 0:
    hlo_env.run(general_fusion_pipeline.pre_dry_pass_passes)
    # Open up the action space
    hlo_env.run(general_fusion_pipeline.pass_dry_run)

    # Get features from hlo_env
    hlo_graph = hlo_env.get_hlo_graph(do_hash_verification=False)
    num_alts = len(hlo_graph.alternative_indices)

    if num_alts > 0:
      # Obtain a probablity distribution over the action space
      probablity = uniform_policy(hlo_graph)
      # Sample an action
      decisions = argmax_sample(probablity, hlo_graph)
      decisions = np.asarray(decisions)
      # Apply action to the hlo_env
      hlo_env.apply_alternatives(decisions)
      hlo_env.run(general_fusion_pipeline.post_dry_pass_passes)

  hlo_env.run(general_fusion_pipeline.post_pass_optimizations)

The `hlo_graph` is the entry point of all available features. The `num_alts` is the number of alternatives (i.e. actions) available in the current state. When `num_alts` is 0, it means there is no more action to choose, and the optimization loop will terminate.

Next, we details how we implement the `uniform_policy` and `argmax_sample` functions.

The goal of `uniform_policy` is to output a probability distribution at each kAlternative node over all its operands (i.e. predecessors in HLO graph).
The probability distribution is a tf.RaggedTensor, where the outer dimension is the number of kAlternative nodes, and the inner dimension is the number of operands of each kAlternative node.

.. code-block:: python

  def uniform_policy(hlo_graph) -> tf.RaggedTensor:
    """Produce a uniform policy for the given hlo graph.

    Args:
      hlo_graph: the hlo graph
  
    Returns:
      a tf.RaggedTensor with shape [num_alt_idx, None]. The outer dimension
      is the alternative index, and the inner dimension is the operand index.
      Each row is a list of probability to operand indices for the 
      corresponding alternative.
    """
    # get graph structures
    operands, users = get_ragged_tensor_from_hlo(hlo_graph)
    # get the indices of kAlternative nodes
    alternative_idx = tf.convert_to_tensor(hlo_graph.alternative_indices)
    # get the indices of operands for each kAlternative node
    alt_oprnd_idx: tf.RaggedTensor = tf.gather(operands, alternative_idx)

    # assign random score to each operand
    alt_oprnd_prob = tf.map_fn(
      lambda x: tf.random.uniform(shape=x.shape, minval=0, maxval=1),
      alt_oprnd_idx,
      fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float32)
    )

  return alt_oprnd_prob

The action space is defined as a 2d-array of dimension [num_alt_idx, 2]. The first column is the index of the kAlternative node, and the second column is the index of the operand to choose.

To output an action, we implement the `argmax_sample` to choose the operand with the highest score for each kAlternative node.

.. code-block:: python

  def argmax_sample(probability: tf.RaggedTensor, hlo_graph) -> tf.Tensor:
    """Select the operand with the highest score for each alternative.

    Args:
      probability: a tf.RaggedTensor with shape [num_alt_idx, None].
        The outer dimension is the alternative index, and the inner 
        dimension is the operand index.
      
      hlo_graph: the hlo graph
    
    Returns:
      a tf.Tensor with shape [num_alt_idx, 2], the 1st column is
      the uids of alt_idx, the 2nd column is the operand_idx to be selected.
    """
    alt_uids = hlo_graph.node_features.uids[hlo_graph.alternative_indices]

    alt_uids = tf.convert_to_tensor(alt_uids, dtype=tf.int64)

    alt_choice = tf.map_fn(
      lambda x: tf.argmax(x, axis=0),
      probability,
      fn_output_signature=tf.TensorSpec(shape=[], dtype=tf.int64)
    )

    return tf.stack([alt_uids, alt_choice], axis=1)

The full-size code can be found `here <https://github.com/sail-sg/hloenv/blob/hloenv-refactor-open/examples/uniform_policy.py>`__.

Other Features
--------------

- Saving and Loading HLO module

At any stage of the optimization pipeline, we can export the current Hlo text to a string object for inspection.

.. code-block:: python

    init_hlo_str = hlo_env.export_hlo_to_str()
    
We can also save the snapshot of an HloEnv object at any stage and restore at a later stage.

.. code-block:: python

    saved_hlo_module = hlo_env.save_hlo()
    hlo_env.pre_fusion_optimizations()
    post_fusion_hlo_str = hlo_env.export_hlo_to_str()
    hlo_env.load_hlo(saved_hlo_module)
    
This can be useful when you want to explore different optimization actions from the same initial state.

- DAG Hash

The existing hash implementation in XLA is lacking in two ways which increase the number of hash collisions: 1) It simply hashes the instructions in the HLO graph in post-order, and does not recursively consider the structure and connections of each HLO instruction and computation in the HLO graph; 2) Instruction specific parameters (e.g. the size and stride of an HLO Convolution instruction) are not considered in the hash of each instruction as well. Our custom HloDAGHash function builds upon XLA’s hash implementation, but is designed to be a more powerful hash that additionally accounts for graph topology and the parameters unique to each instruction. This reduces the chance of a hash collision when determining if a graph has been seen before, or is identical to another graph.

.. code-block:: python

    hlo_hash = hlo_env.get_hlo_module_hash()
    
This is useful for de-duplicating the dataset or uniquely labeling the state when performing a search over the state space.

- Profiling an HLO Graph

To profile the runtime of an HLO graph we need to obtain both the executable and parameters. We obtain the executable by calling the standard compiler provided by XLA while setting *run_backend_only* to prevent the reinvocation of HLO passes. For parameters, we randomly generate N(0, 1) for floating-point parameters and fill const values for other types. A fixed random seed is used to keep the parameters consistent across the optimization process so that we can verify the correctness of optimizations. The only parameter for evaluate() is the repeated evaluation time.

.. code-block:: python
    
    num_eval_iterations = 100
    eval_result = hlo_env.evaluate(num_eval_iterations)
    
The above code will run the evaluation for 100 times and generate several metrics and output as shown below:

.. list-table:: **Evaluation Results**
    :widths: 42 42
    :header-rows: 1
    
    * - Name
      - Description
      
    * - durations
      - The default duration in nanoseconds. This returns the execution duration as measured within the Tensorflow evaluation code, starting from the point when the executable has been enqueued on the compute stream till the completion of the executable.
      
    * - compute_durations
      - The duration in nanoseconds of the computation, without data transfer, as measured on the device.
     
    * - full_durations
      - The full duration of the computation as measured within HloEnv.evaluate(). This captures the entire execution process including processes such as enqueueing the computation on the compute stream, and is hence more subject to timing noise.
      
    * - output
      - The output of the HloModule.
