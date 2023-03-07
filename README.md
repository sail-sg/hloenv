# HloEnv

*HloEnv* is an environment based on Accelerated Linear Algebra
([XLA](https://www.tensorflow.org/xla/)) for deep learning compiler
optimization research. HloEnv transforms all graph rewrites into a
common representation, providing a flexible interface to control and
modify existing graph optimization passes. In this representation, an
XLA pass is converted into a set of sequential rewrite decisions.

HloEnv has the following major features:

  - **The alternative graph representation**

    HloEnv frames the graph optimization problem as a sequential
    rewrite decision problem, serving as a single-player game engine. 
    See [always_pick_first](examples/always_pick_first.py) and
    [uniform_policy](examples/uniform_policy.py) for examples of interacting with
    HloEnv and optimizing a HLO computation grpah.


  - **Python interface for full XLA optimization passes and pipelines**

    HloEnv provides a Python interface to control and modify existing
    graph optimization passes. See [general_fusion_pipeline](examples/general_fusion_pipeline.py) for
    an example of our customized XLA optimization pipeline.

The design of HloEnv points to a potential future where deep learning
compiler engineers only need to develop and maintain a simple set of
rewrite rules. The complicated heuristics are left to machine
learning-generated optimization strategies that generalize to both new
deep learning models and new deep learning hardware. 

## Citations

HloEnv is described in details in the following paper:

``` bibtex
@inproceedings{HloEnv2022,
  author = {Chin Yang Oh, Kunhao Zheng, Bingyi Kang, Xinyi Wan, Zhongwen Xu, Shuicheng Yan, Min Lin, Yangzihao Wang},
  title = {HloEnv: A Graph Rewrite Environment for Deep Learning Compiler Optimization Research},
  booktitle = {Workshop on ML for Systems at NeurIPS 2022},
  year = {2022},
  series = {NeurIPS '22},
  month = dec,
}
```

## Docs

The full documentation for HloEnv can be found at [this link](https://sail-sg.github.io/hloenv/).

## Quick Start

Download the pre-built HloEnv wheel from the [releases](https://github.com/sail-sg/hloenv/releases), and navigate to the folder that contains the .whl file and run:

```bash
python3 -m pip install hloenv-[x.y.z]-cp38-cp38-linux_x86_64.whl
```

To run examples, you need to install the following dependencies in your python package:

  - [TensorFlow==2.9.0](https://www.tensorflow.org/install)
  - NumPy

## Build HloEnv

To build HloEnv and evaluate HLO graph runtime, you need to install the following dependencies:

  - [Bazel 5.0.0](https://bazel.build/install?hl=en)
  - liblapack-dev and liblapacke-dev
  - CUDA 11.2 + cuDNN for CUDA 11.x (required by TensorFlow 2.9.0)
  - GCC 9.x (required by CUDA 11.x)

To clone HloEnv and install Tensorflow, run the following command:

```bash
git clone git@github.com:sail-sg/hloenv.git
cd hloenv
python -m venv hloenv-env
source hloenv-env/bin/activate
python3 -m pip install tensorflow==2.9.0
```

When met the minimum requirement, building HloEnv Python Wheels can be very simple. 

In the root directory of HloEnv, run the following command

```bash
make build
```
Will build the wheels file and put it under the folder *dist*. If encountered OOM issue while compiling, can add the following flag to Makefile's bazel run command:

```bash
--local_ram_resources={mem_size_in_MB}
```

```bash
make install
```
Will install the freshly built Python Wheels and all the dependencies.

## HLO Dataset

You can find the unoptimized HLO text files harvested from JAX implemented repositories on GitHub [here](https://drive.google.com/file/d/1XF7fz59HvsUIfUhgF-yXF7YE3y-lOBFZ/view?usp=sharing)

