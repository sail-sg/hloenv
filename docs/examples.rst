.. _examples:

Examples of using HloEnv
========================

This documentation only covers the Python interface. We will walk through two simple examples that make use of several HloEnv features. We first show how to read in an HLO text file and turn it into HLO graph features which can be used for implementing a user-defined decision-making agent. We then present a very simple decision-making agent where the policy randomly choose from available actions.

Playing with HLO graph features
-------------------------------

First make sure your current working directory is correct.

.. code-block:: bash

    $ cd /path/to/altgraph/examples
    
The HloEnv module holds most functionality, so we usually import it first.

.. code-block:: python

    import os
    import pathlib
    import numpy as np
    from altgraph import HloEnv

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
        
    * - num_users
      - Number of HloInstructions that uses the result of this HloInstruction
        
    * - num_operands
      - Number of HloInstructions whose results this HloInstruction uses
        
    * - opcodes
      - HLO opcode index, as defined `here <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/hlo/ir/hlo_opcode.h#L50>`_
        
    * - opcode_attrs
      - Unique attribute embeddings for each opcode
        
    * - num_opcode_attrs
      - List of pairs, each pair contains the number of integer attribute and the number of enum attribute in opcode_attrs
        
    * - is_alternative
      - List of boolean that shows if the HloInstruction is *kAlternative*
      
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
      
.. list-table:: **In/Out Edge Features**
    :widths: 42 42
    :header-rows: 1
    
    * - Feature Name
      - Description
      
    * - uids
      - Unique ID of the edge, a concatination of source and destination nodes' uids
      
    * - srcs
      - Node index of source node
    
    * - dsts
      - Node index of destination node
      
    * - dims
      - Dimension of the tensor flows by this edge
    
    * - layout
      - Layout of the tensor flows by this edge
      
    * - dtypes
      - Data type of the tensor flows by this edge
      
 The full-size code can be found `here <https://github.com/sail-sg/altgraph/blob/altgraph-refactor-open/examples/hlo_play.py>`_. In our second example, we will show you how to use these features to create a simple decision-making agent and run XLA optimizations using it.
      
A simple decision-making agent
------------------------------

TODO

Other Features
--------------

TODO (probably talk about DAGHash and evaluation if time permitted.)
