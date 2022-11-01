# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//visibility:public"])

tf_cc_binary(
    name = "xla_compile",
    srcs = ["xla_compile.cc"],
    copts = [
        "-fexceptions",
    ],
    deps = [
        ":hlo_graph",
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_library(
    name = "hlo_graph",
    srcs = ["hlo_graph.cc"],
    hdrs = ["hlo_graph.h"],
    deps = [
        "//altgraph/utils:hlo_utils",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)


cc_library(
    name = "hlo_module",
    srcs = ["hlo_module.h"],
    deps = [
        "//altgraph/utils:hlo_utils",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
)

cc_library(
    name = "gpu_backend",
    hdrs = ["gpu_backend.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
)

cc_library(
    name = "hlo_pass_defs",
    hdrs = ["hlo_pass_defs.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
    ],
)

cc_library(
    name = "schedule",
    srcs = ["schedule.h"],
    deps = [
        ":hlo_module",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "hlo_env",
    srcs = ["hlo_env.cc"],
    hdrs = ["hlo_env.h"],
    deps = [
        ":gpu_backend",
        ":hlo_graph",
        ":hlo_module",
        ":schedule",
        "//altgraph/evaluation:evaluator",
        "@com_google_absl//absl/hash",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_comparison",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/python:types",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        # "@org_tensorflow//tensorflow/python:pybind11_lib",
    ],
)

genrule(
    name = "gen_extract_subgraphs",
    srcs = ["//altgraph/bin:extract_subgraphs"],
    outs = ["extract_subgraphs"],
    cmd = "cp $< $@ && chmod 755 $@",
)

py_library(
    name = "altgraph",
    srcs = ["__init__.py"],
    data = [
        "//altgraph/python:py_hlo_env.so",
        ":gen_extract_subgraphs",
    ],
)

